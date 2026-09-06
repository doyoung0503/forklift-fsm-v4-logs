# -*- coding: utf-8 -*-
"""4점 PnP vs 다중 키포인트 PnP — 회전 관측 품질 비교.

판단 지표
  1) 정지구간 yaw / bearing 표준편차          (작을수록 좋음)
  2) 회전 세그먼트에서 Δyaw 의 명령방향 부호 일치 개수
  3) Δyaw 와 Δbearing 의 상관 / 회귀 기울기   (1 에 가까울수록 yaw 가 쓸 만함)
  4) 회전 중 |p| 보존성                        (강체회전 가정 점검)
  5) 재투영 RMS
"""
from __future__ import annotations

import glob
import io
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
exec(io.open(os.path.join(HERE, "rot_common.py"), encoding="utf-8").read())

METHODS = {"4pt": ("x4", "z4", "yaw4_deg", "reproj4_px"),
           "multi": ("xm", "zm", "yawm_deg", "reprojm_px")}


def rot_windows(cs_path):
    """[(step, cmd, t_cmd, t_stop), ...]"""
    out = []
    cl = [r for r in _read_jsonl(cs_path) if r.get("phase") == "cmd"]
    for i, r in enumerate(cl):
        if r.get("cmd") not in ROT_CMDS or i + 1 >= len(cl):
            continue
        t0, t1 = float(r["t_mono"]), float(cl[i + 1]["t_mono"])
        if 0.1 <= t1 - t0 <= 12.0:
            out.append((r.get("step"), r["cmd"], t0, t1))
    return out


def main():
    rows, static_rows = [], []
    for cs in sorted(glob.glob(os.path.join(REC_DIR, "*_control_seq.jsonl"))):
        base = cs[: -len("_control_seq.jsonl")]
        rc = base + "_pallet_state_recomputed.csv"
        if not os.path.exists(rc):
            continue
        tag = os.path.basename(base).replace("forklift_v4_recording_", "")
        df = pd.read_csv(rc)
        df = df[(df.det_ok == 1) & np.isfinite(df.t_mono)]

        for step, cmd, t0, t1 in rot_windows(cs):
            pre = df[(df.t_mono >= t0 - 1.2) & (df.t_mono <= t0 + 0.05)]
            post = df[(df.t_mono >= t1 + 1.5) & (df.t_mono <= t1 + 4.0)]
            if len(pre) < 6 or len(post) < 6:
                continue
            row = dict(tag=tag, step=step, cmd=cmd,
                       sign=1.0 if cmd == "ROT_RIGHT" else -1.0)
            ok = True
            for name, (xc, zc, yc, rc_) in METHODS.items():
                if xc not in df.columns:
                    ok = False
                    break
                a, b = pre.dropna(subset=[xc, zc]), post.dropna(subset=[xc, zc])
                if len(a) < 6 or len(b) < 6:
                    ok = False
                    break
                bx = lambda d: np.degrees(np.arctan2(d[xc], d[zc]))
                rr = lambda d: np.hypot(d[xc], d[zc])
                row["dbear_" + name] = float(np.median(bx(b)) - np.median(bx(a)))
                row["dyaw_" + name] = float(np.median(b[yc]) - np.median(a[yc]))
                row["dr_" + name] = float(np.median(rr(b)) - np.median(rr(a)))
                # 정지구간 산포 (명령 전 창)
                static_rows.append(dict(
                    tag=tag, step=step, method=name,
                    yaw_sd=float(np.std(a[yc], ddof=1)),
                    bearing_sd=float(np.std(bx(a), ddof=1)),
                    z_sd=float(np.std(a[zc], ddof=1)),
                    reproj_med=float(np.nanmedian(a[rc_])) if rc_ in df else np.nan,
                ))
            if ok:
                rows.append(row)

    seg = pd.DataFrame(rows)
    st = pd.DataFrame(static_rows)
    seg.to_csv(os.path.join(OUT, "pose_method_segments.csv"), index=False,
               encoding="utf-8-sig")

    summary = []
    for name in METHODS:
        dy = seg["dyaw_" + name].values
        db = seg["dbear_" + name].values
        s = st[st.method == name]
        summary.append(dict(
            method=name,
            n_seg=len(seg),
            static_yaw_sd_deg=float(s.yaw_sd.median()),
            static_bearing_sd_deg=float(s.bearing_sd.median()),
            static_z_sd_m=float(s.z_sd.median()),
            reproj_rms_px=float(s.reproj_med.median()),
            yaw_sign_ok=int(np.sum(np.sign(dy) == np.sign(db))),
            corr_yaw_bear=float(np.corrcoef(dy, db)[0, 1]),
            slope_yaw_on_bear=float(np.polyfit(db, dy, 1)[0]),
            abs_dr_median_m=float(np.median(np.abs(seg["dr_" + name].values))),
        ))
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, "pose_method_summary.csv"), index=False,
              encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(sm.to_string(index=False, float_format=lambda v: "%8.3f" % v))
    print()
    print(seg[["tag", "step", "cmd", "dbear_4pt", "dbear_multi",
               "dyaw_4pt", "dyaw_multi", "dr_4pt", "dr_multi"]].to_string(
        index=False, float_format=lambda v: "%8.2f" % v))


if __name__ == "__main__":
    main()
