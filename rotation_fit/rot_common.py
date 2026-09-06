"""회전 로그 공통 로더 / 세그먼트 추출.

회전각 관측량에 대한 결론 (compare_signals.py / compare_pose_methods.py):
  - 현장 로그의 yaw_deg 는 전면 4모서리만 쓴 평면 PnP 다. 전면은 1.1 x 0.15 m
    (가로세로비 7:1) 짜리 얇은 면이라 yaw/pitch 가 매우 나쁘게 조건화되어,
    회전 방향조차 맞지 않는다 (12 세그먼트 중 7 개만 부호 일치, dyaw-dbearing
    상관 0.02). 회전량 관측값으로 쓸 수 없다.
  - 후면 키포인트까지 넣은 다중 키포인트 PnP 는 깊이 방향 지지를 얻어 yaw 가
    복구된다 (부호 일치 12/12, 상관 0.987). 이때부터 yaw 를 쓸 수 있다.
  - 전면중심 방위각 bearing = atan2(pos_x, pos_z) 는 두 방식 모두에서 안정적이다
    (정지구간 표준편차 0.02~0.03 deg). 단, bearing 은 차량 헤딩뿐 아니라 회전에
    따른 카메라 병진도 함께 반영한다: 회전중심이 카메라 뒤 d 만큼 있으면
    dbearing = dheading * (r + d) / r.  다중 키포인트 yaw 로 실측한 결과
    d ~ 1.0 m, 전역 비례상수 k = dbearing/dheading ~ 1.36 이다.

따라서 회전각 신호를 두 가지로 둔다 (ANGLE_SIGNAL):
  bearing = FACE / RECENTER 모드가 실제로 쓰는 오차신호 도메인
  yaw     = 순수 차량 헤딩 도메인 (WAYPOINT 모드, 물리 파라미터)
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REC_DIR = os.environ.get(
    "ROT_REC_DIR",
    r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec",
)

# pallet_state.t_mono 은 pose 결과 시각. inference_timing 의 실측 중앙값
# input_to_pose_result_ms = 31 ms 를 빼서 영상 획득 시각으로 되돌린다.
POSE_LATENCY_SEC = 0.031

# 어떤 pose 를 회전각 관측에 쓸지.
#   log        : 원본 pallet_state.csv (현장 4점 IPPE, 기록 시점 그대로)
#   recomp4    : 재추론 + 4점 IPPE 재계산 (전 프레임, 영상획득 시각)
#   recompmulti: 재추론 + 다중 키포인트 PnP (전 프레임, 영상획득 시각)
POSE_SOURCE = os.environ.get("ROT_POSE_SOURCE", "log")

# 회전각을 무엇으로 재는가.
#   bearing : 전면중심 방위각 변화량. 차량 헤딩 + 회전에 따른 카메라 병진이 섞인다
#             (= FACE/RECENTER 모드가 실제로 쓰는 오차신호).
#   yaw     : 팔레트 평면 yaw 변화량. 병진과 무관한 순수 차량 헤딩 변화량
#             (= WAYPOINT 모드가 쓰는 신호). 다중 키포인트 PnP 에서만 신뢰할 수 있다.
ANGLE_SIGNAL = os.environ.get("ROT_ANGLE_SIGNAL", "bearing")

ROT_CMDS = ("ROT_LEFT", "ROT_RIGHT")


def _load_pose_table(base: str, source: str):
    """(t_meas, bearing_deg, range_m, yaw_deg) 를 돌려준다. 실패하면 None."""
    if source == "log":
        path = base + "_pallet_state.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df = df[df.det_ok == 1].dropna(subset=["pos_x", "pos_z"])
        df = df[df.pos_z > 0.2]
        if len(df) < 20:
            return None
        return (df.t_mono.values - POSE_LATENCY_SEC,
                np.degrees(np.arctan2(df.pos_x.values, df.pos_z.values)),
                np.hypot(df.pos_x.values, df.pos_z.values),
                df.yaw_deg.values)

    path = base + "_pallet_state_recomputed.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 프레임 정렬 검증: raw.mp4 인덱스 + 고정 offset 으로 frame_i 를 복원하는데,
    # 중간에 프레임이 유실된 녹화(복구된 moovless 파일)에서는 고정 offset 이
    # 성립하지 않는다. 4점 재계산 결과를 원본 로그와 대조해 어긋나면 버린다.
    ref = base + "_pallet_state.csv"
    if os.path.exists(ref) and "x4" in df.columns:
        r = pd.read_csv(ref)
        r = r[r.det_ok == 1].dropna(subset=["pos_x", "pos_z"])
        j = df.merge(r[["frame_i", "pos_x", "pos_z"]], on="frame_i", how="inner")
        j = j.dropna(subset=["x4", "pos_x"])
        if len(j) < 30 or float(np.median(np.abs(j.x4 - j.pos_x))) > 0.005:
            return None
    xc, zc, yc = ("x4", "z4", "yaw4_deg") if source == "recomp4" \
        else ("xm", "zm", "yawm_deg")
    if xc not in df.columns:
        return None
    df = df[df.det_ok == 1].dropna(subset=[xc, zc])
    df = df[(df[zc] > 0.2) & np.isfinite(df.t_mono)]
    if len(df) < 20:
        return None
    return (df.t_mono.values,
            np.degrees(np.arctan2(df[xc].values, df[zc].values)),
            np.hypot(df[xc].values, df[zc].values),
            df[yc].values if yc in df.columns else np.full(len(df), np.nan))


@dataclass
class Segment:
    tag: str
    step: Optional[int]
    cmd: str
    sign: float                 # +1 = ROT_RIGHT, -1 = ROT_LEFT
    t_cmd: float                # 회전 명령 송신 t_mono
    t_stop: float               # STOP 명령 송신 t_mono
    t: np.ndarray               # 측정 시각 (영상 획득 기준)
    angle: np.ndarray           # 방향 정규화 누적 회전각 [deg], 명령 전 0
    bearing: np.ndarray
    rng: np.ndarray             # 팔레트까지 거리 |p| [m]
    ref_bearing: float
    ref_noise: float
    kind: Optional[str] = None          # v4_face_rotate / v4_waypoint_rotate ...
    diag: Dict = field(default_factory=dict)   # rotation_stop diagnostics
    end: Dict = field(default_factory=dict)    # end.result

    @property
    def cmd_duration(self) -> float:
        return self.t_stop - self.t_cmd


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_segments(rec_dir: str = REC_DIR,
                  pre_sec: float = 1.2,
                  post_sec: float = 5.0,
                  min_frames: int = 12,
                  source: Optional[str] = None,
                  signal: Optional[str] = None) -> List[Segment]:
    source = source or POSE_SOURCE
    signal = signal or ANGLE_SIGNAL
    segments: List[Segment] = []
    for cs in sorted(glob.glob(os.path.join(rec_dir, "*_control_seq.jsonl"))):
        base = cs[: -len("_control_seq.jsonl")]
        tag = os.path.basename(base).replace("forklift_v4_recording_", "")

        table = _load_pose_table(base, source)
        if table is None:
            continue
        t_meas, bearing, rng, yaw = table
        # 회전각 원신호. yaw 는 dyaw 가 dbearing 과 같은 부호이므로
        # 아래 정규화식(sign * (ref - value))을 그대로 쓸 수 있다.
        raw = bearing if signal == "bearing" else yaw
        if not np.all(np.isfinite(raw)):
            keep = np.isfinite(raw)
            if int(keep.sum()) < 20:
                continue
            t_meas, bearing, rng, yaw, raw = (t_meas[keep], bearing[keep],
                                              rng[keep], yaw[keep], raw[keep])

        records = _read_jsonl(cs)
        cmds = [r for r in records if r.get("phase") == "cmd"]
        stops = {r.get("step"): r for r in records
                 if r.get("phase") == "rotation_stop"}
        ends = {(r.get("step"), r.get("kind")): r for r in records
                if r.get("phase") == "end"}
        begins = {r.get("step"): r for r in records if r.get("phase") == "begin"}

        for i, rec in enumerate(cmds):
            if rec.get("cmd") not in ROT_CMDS:
                continue
            if i + 1 >= len(cmds):
                continue
            t0 = float(rec["t_mono"])
            t1 = float(cmds[i + 1]["t_mono"])
            if not (0.1 <= t1 - t0 <= 12.0):
                continue
            m = (t_meas >= t0 - pre_sec) & (t_meas <= t1 + post_sec)
            if int(m.sum()) < min_frames:
                continue
            ref_m = (t_meas >= t0 - pre_sec) & (t_meas <= t0 + 0.05)
            if int(ref_m.sum()) < 4:
                continue
            ref = float(np.median(raw[ref_m]))
            noise = float(np.std(raw[ref_m], ddof=1))
            sign = 1.0 if rec["cmd"] == "ROT_RIGHT" else -1.0
            angle = sign * (ref - raw[m])

            step = rec.get("step")
            begin = begins.get(step, {})
            kind = begin.get("kind")
            diag = (stops.get(step) or {}).get("diagnostics", {}) or {}
            end = {}
            for (s, k), r2 in ends.items():
                if s == step and str(k or "").endswith("rotate"):
                    end = r2.get("result", {}) or {}
            segments.append(Segment(
                tag=tag, step=step, cmd=rec["cmd"], sign=sign,
                t_cmd=t0, t_stop=t1, t=t_meas[m], angle=angle,
                bearing=bearing[m], rng=rng[m], ref_bearing=ref,
                ref_noise=noise, kind=kind, diag=diag, end=end,
            ))
    return segments


def interp_cross(t: np.ndarray, y: np.ndarray, level: float) -> Optional[float]:
    """y 가 level 을 처음 상향 통과하는 시각(선형보간). 없으면 None."""
    idx = np.where(y >= level)[0]
    if len(idx) == 0:
        return None
    j = int(idx[0])
    if j == 0:
        return float(t[0])
    y0, y1 = y[j - 1], y[j]
    if y1 == y0:
        return float(t[j])
    frac = (level - y0) / (y1 - y0)
    return float(t[j - 1] + frac * (t[j] - t[j - 1]))


def value_at(t: np.ndarray, y: np.ndarray, t_query: float) -> float:
    return float(np.interp(t_query, t, y))


def local_rate(t: np.ndarray, y: np.ndarray, t_end: float,
               window: float = 0.5, min_pts: int = 3) -> Optional[float]:
    """t_end 직전 window 초 구간의 최소자승 기울기 [deg/s]."""
    m = (t >= t_end - window) & (t <= t_end + 1e-9)
    if int(m.sum()) < min_pts:
        m = (t >= t_end - 2.0 * window) & (t <= t_end + 1e-9)
        if int(m.sum()) < min_pts:
            return None
    return float(np.polyfit(t[m], y[m], 1)[0])
