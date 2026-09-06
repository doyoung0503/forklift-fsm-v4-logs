# -*- coding: utf-8 -*-
"""덤프된 키포인트로 pose 를 다시 풀어 pallet_state 호환 CSV 를 만든다.

- 4점 IPPE (기존 방식) 와 다중 키포인트 PnP (신규) 를 같은 프레임에서 모두 계산해
  한 파일에 담는다. 열 이름은 pallet_state.csv 와 맞춘다.
- 시각은 inference_timing.csv 의 camera_input_host_mono_ms (영상 획득 시각, 초 단위)
  를 쓴다. control_seq.jsonl 의 t_mono 와 같은 monotonic 시계다.
- frame_i 정렬은 4점 재계산 결과를 로그값과 대조해 검증한다 (--verify).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.environ.get(
    "ROT_REC_DIR", r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec")
KPTS = os.path.join(HERE, "kpts")

FACE_KPTS = (0, 1, 2, 3)
PALLET_W = 1.100
PALLET_H = 0.150
PALLET_L = 1.100


def intrinsics(meta_path):
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    it = meta["intrinsics"]
    K = np.array([[it["fx"], 0.0, it["ppx"]],
                  [0.0, it["fy"], it["ppy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray(it.get("coeffs", [0, 0, 0, 0, 0]),
                      dtype=np.float64).reshape(-1, 1)
    return K, dist


def pose_face4(kpts, K, dist, face_w=PALLET_W, face_h=PALLET_H):
    """기존 calib/geometry.pose_from_kpts_pnp 와 동일한 4점 IPPE 경로."""
    img = np.asarray(kpts, dtype=np.float64)[list(FACE_KPTS), :2]
    if not np.all(np.isfinite(img)):
        return None
    hw, hh = face_w * 0.5, face_h * 0.5
    obj = np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0],
                    [hw, hh, 0.0], [-hw, hh, 0.0]], dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return None
    return angles_from(rvec, tvec.reshape(3), obj, img, K, dist)


def angles_from(rvec, center, obj, img, K, dist):
    R, _ = cv2.Rodrigues(rvec)
    n = R @ np.array([0.0, 0.0, 1.0])
    u = R @ np.array([1.0, 0.0, 0.0])
    yaw = float(np.degrees(np.arctan2(n[0], n[2])))
    pitch = float(np.degrees(np.arctan2(-n[1], n[2])))
    roll = float(np.degrees(np.arctan2(-u[1], u[0])))
    proj, _ = cv2.projectPoints(obj, rvec, np.asarray(center,
                                                      dtype=np.float64).reshape(3, 1),
                                K, dist)
    err = float(np.sqrt(np.mean(np.sum(
        (proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
    return dict(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                pos_x=float(center[0]), pos_y=float(center[1]),
                pos_z=float(center[2]), reproj_px=err, n_used=len(obj))


def load_multi_solver():
    """rotation_fit/pose_multi.py 의 다중 키포인트 PnP 를 이 파일의 규약으로 감싼다.

    pose_multi 는 dict(ok, yaw, pitch, roll, center(3,), rms, n_used, ...) 를
    돌려준다. 여기서 쓰는 이름(yaw_deg / pos_* / reproj_px)으로 변환한다.
    """
    path = os.path.join(HERE, "pose_multi.py")
    if not os.path.exists(path):
        return None
    ns = {}
    exec(open(path, encoding="utf-8").read(), ns)
    fn = None
    for name in ("pose_from_kpts_pnp_multi", "pose_from_kpts_multi",
                 "solve_pose_multi"):
        if name in ns:
            fn = ns[name]
            break
    if fn is None:
        return None

    def solve(kpts, K, dist):
        r = fn(kpts, K, dist)
        if not r or not r.get("ok"):
            return None
        c = np.asarray(r["center"], dtype=float).reshape(3)
        return dict(yaw_deg=float(r["yaw"]), pos_x=float(c[0]),
                    pos_y=float(c[1]), pos_z=float(c[2]),
                    reproj_px=float(r.get("rms") or np.nan),
                    n_used=int(r.get("n_used") or 0))

    return solve


def process(tag, multi_solver, verify=False):
    base = os.path.join(REC, tag)
    npz = np.load(os.path.join(KPTS, tag + "_kpts.npz"))
    kpts = npz["kpts"]
    frame_i = npz["frame_i"]
    K, dist = intrinsics(base + "_meta.json")
    timing = pd.read_csv(base + "_inference_timing.csv")
    tmap = dict(zip(timing.frame_i.values,
                    timing.camera_input_host_mono_ms.values / 1000.0))
    smap = dict(zip(timing.frame_i.values, timing.fsm_state.values))

    rows = []
    for i, fi in enumerate(frame_i):
        t = tmap.get(int(fi), np.nan)
        k = kpts[i]
        row = dict(frame_i=int(fi), t_mono=t,
                   fsm_state=smap.get(int(fi), ""), det_ok=0)
        if np.all(np.isfinite(k[list(FACE_KPTS), :2])):
            p4 = pose_face4(k, K, dist)
            if p4 is not None and p4["pos_z"] > 0.0:
                row.update(det_ok=1,
                           yaw4_deg=p4["yaw_deg"], x4=p4["pos_x"],
                           y4=p4["pos_y"], z4=p4["pos_z"],
                           reproj4_px=p4["reproj_px"])
            if multi_solver is not None:
                pm = multi_solver(k, K, dist)
                if pm is not None and pm.get("pos_z", 0.0) > 0.0:
                    row.update(yawm_deg=pm["yaw_deg"], xm=pm["pos_x"],
                               ym=pm.get("pos_y", np.nan), zm=pm["pos_z"],
                               reprojm_px=pm.get("reproj_px", np.nan),
                               n_used=pm.get("n_used", np.nan))
        rows.append(row)
    df = pd.DataFrame(rows)

    if verify:
        log = pd.read_csv(base + "_pallet_state.csv")
        log = log[log.det_ok == 1].dropna(subset=["pos_x", "pos_z"])
        j = df.merge(log[["frame_i", "yaw_deg", "pos_x", "pos_z"]],
                     on="frame_i", how="inner", suffixes=("", "_log"))
        j = j.dropna(subset=["x4", "pos_x"])
        if len(j):
            print("  verify n=%d  |dx|med=%.4f m  |dz|med=%.4f m  "
                  "|dyaw|med=%.3f deg" % (
                      len(j), float(np.median(np.abs(j.x4 - j.pos_x))),
                      float(np.median(np.abs(j.z4 - j.pos_z))),
                      float(np.median(np.abs(j.yaw4_deg - j.yaw_deg)))))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    solver = load_multi_solver()
    print("multi solver:", "loaded" if solver else "NOT FOUND (4-point only)")
    for f in sorted(glob.glob(os.path.join(KPTS, "*_kpts.npz"))):
        tag = os.path.basename(f)[: -len("_kpts.npz")]
        if args.only and args.only not in tag:
            continue
        print("[%s]" % tag, flush=True)
        df = process(tag, solver, verify=args.verify)
        dst = os.path.join(REC, tag + "_pallet_state_recomputed.csv")
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print("  saved", os.path.basename(dst), len(df), "rows", flush=True)


if __name__ == "__main__":
    main()
