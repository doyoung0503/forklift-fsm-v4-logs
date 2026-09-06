# -*- coding: utf-8 -*-
"""4점 PnP vs 9점(다중 키포인트) PnP 비교 평가.

지표
 1. 재투영 RMS [px]  - 전면 4점 기준 / 전체 9점 기준 (둘 다 보고)
 2. 정지구간 yaw sd [deg], pos_x/pos_z sd [m], bearing sd [deg]
 3. 회전구간 |p| = hypot(pos_x,pos_z) 보존성
 4. 회전구간 dyaw vs dbearing 기울기 (강체회전이면 1.0 이어야 함)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_multi as pm  # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
KPTS = os.path.join(HERE, "kpts")
REC = os.environ.get("ROT_REC_DIR",
                     r"c:\Users\dhshs\Downloads\로그수집\extracted\depth_cam\rec")

FX, FY, CX, CY = 605.906494140625, 605.9697875976562, 317.59619140625, 256.29229736328125
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros((5, 1))
POSE_LATENCY_SEC = 0.031
MOVE_CMDS = ("FWD", "BWD", "ROT_LEFT", "ROT_RIGHT")
METHODS = ("pnp4", "pnp9", "pnp9strict")


def read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def solve_all(kpts, obj9, method, face_h=0.150, use_weights=True):
    """(N,9,3) -> dict of arrays"""
    n = len(kpts)
    yaw = np.full(n, np.nan)
    pos = np.full((n, 3), np.nan)
    r4 = np.full(n, np.nan)
    r9 = np.full(n, np.nan)
    nused = np.zeros(n, dtype=int)
    for i in range(n):
        kp = kpts[i]
        if not np.isfinite(kp[:, :2]).all():
            continue
        if method == "pnp4":
            out = pm.pose_from_kpts_pnp4(kp[:4, :2], K, DIST,
                                         face_w=pm.PALLET_KPT_W, face_h=face_h)
            if not out.get("ok"):
                continue
            rvec, tvec = out["rvec"], out["tvec"]
            yaw[i] = out["yaw"]
            pos[i] = out["center"]
            nused[i] = 4
        else:
            out = pm.pose_from_kpts_pnp_multi(
                kp, K, DIST, obj=obj9, use_weights=use_weights,
                min_front=4 if method == "pnp9strict" else 2)
            if not out.get("ok"):
                continue
            rvec, tvec = out["rvec"], out["tvec"]
            yaw[i] = out["yaw"]
            pos[i] = out["center"]
            nused[i] = out["n_used"]
        d9 = pm._reproj(obj9, kp[:, :2], K, DIST, rvec, tvec)
        r4[i] = float(np.sqrt(np.mean(d9[:4] ** 2)))
        r9[i] = float(np.sqrt(np.mean(d9 ** 2)))
    return dict(yaw=yaw, pos=pos, r4=r4, r9=r9, nused=nused)


def stationary_windows(cs_path, settle=3.0, tail=0.3, min_len=2.0):
    """STOP 이후 다음 이동명령 전까지의 구간 (settle 초 경과 후)."""
    rows = read_jsonl(cs_path)
    cmds = [r for r in rows if r.get("phase") == "cmd"]
    wins = []
    for i, r in enumerate(cmds):
        if r.get("cmd") != "STOP":
            continue
        t0 = float(r["t_mono"]) + settle
        t1 = None
        for r2 in cmds[i + 1:]:
            t1 = float(r2["t_mono"]) - tail
            break
        if t1 is None or t1 - t0 < min_len:
            continue
        wins.append((t0, t1, r.get("step")))
    return wins


def rotation_windows(cs_path):
    rows = read_jsonl(cs_path)
    cmds = [r for r in rows if r.get("phase") == "cmd"]
    wins = []
    for i, r in enumerate(cmds):
        if r.get("cmd") not in ("ROT_LEFT", "ROT_RIGHT"):
            continue
        if i + 1 >= len(cmds):
            continue
        t0, t1 = float(r["t_mono"]), float(cmds[i + 1]["t_mono"])
        if 0.1 <= t1 - t0 <= 12.0:
            wins.append((t0, t1 + 2.0, r["cmd"], r.get("step")))
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_h", type=float, default=0.150,
                    help="4점 PnP 가 쓰는 전면높이 (현재 config 값 0.150)")
    ap.add_argument("--model", default="box", choices=["box", "free"])
    ap.add_argument("--no_weights", action="store_true")
    ap.add_argument("--common", action="store_true",
                    help="세 방법이 모두 성공한 프레임만 사용")
    args = ap.parse_args()

    obj9 = pm.pallet_object_points(args.model)
    print("object model = %s" % args.model)
    print("  " + np.array2string(obj9, precision=4, suppress_small=True).replace("\n", "\n  "))

    rows_r, rows_s, rows_rot = [], [], []
    for f in sorted(glob.glob(os.path.join(KPTS, "*_kpts.npz"))):
        tag = os.path.basename(f)[: -len("_kpts.npz")]
        base = os.path.join(REC, tag)
        ps = base + "_pallet_state.csv"
        cs = base + "_control_seq.jsonl"
        if not (os.path.exists(ps) and os.path.exists(cs)):
            continue
        d = np.load(f)
        kpts, conf, fi = d["kpts"], d["conf"], d["frame_i"]
        st = pd.read_csv(ps)
        t_of = dict(zip(st.frame_i.values, st.t_mono.values - POSE_LATENCY_SEC))
        t = np.array([t_of.get(int(x), np.nan) for x in fi])

        res = {}
        res["pnp4"] = solve_all(kpts, obj9, "pnp4", face_h=args.face_h)
        for m in METHODS[1:]:
            res[m] = solve_all(kpts, obj9, m, use_weights=not args.no_weights)

        short = tag[-6:]
        nfin = int(np.isfinite(kpts[:, :, :2]).all(axis=(1, 2)).sum())
        print("  [%s] finite %d | " % (short, nfin) + "  ".join(
            "%s ok %d" % (m, int(np.isfinite(res[m]["yaw"]).sum())) for m in METHODS))
        ok_both = np.ones(len(kpts), dtype=bool)
        for m in METHODS:
            ok_both &= np.isfinite(res[m]["yaw"])
        for m in METHODS:
            r = res[m]
            v = np.isfinite(r["yaw"]) & ok_both
            rows_r.append(dict(tag=short, method=m, n=int(v.sum()),
                               r4_med=float(np.median(r["r4"][v])),
                               r4_rms=float(np.sqrt(np.mean(r["r4"][v] ** 2))),
                               r9_med=float(np.median(r["r9"][v])),
                               r9_rms=float(np.sqrt(np.mean(r["r9"][v] ** 2)))))

        # ---- 정지구간 ----
        for (t0, t1, step) in stationary_windows(cs):
            m = np.isfinite(t) & (t >= t0) & (t <= t1)
            if args.common:
                m = m & ok_both
            if int(m.sum()) < 15:
                continue
            for meth in METHODS:
                r = res[meth]
                mm = m & np.isfinite(r["yaw"])
                if int(mm.sum()) < 15:
                    continue
                px, pz = r["pos"][mm, 0], r["pos"][mm, 2]
                bear = np.degrees(np.arctan2(px, pz))
                rows_s.append(dict(tag=short, step=step, method=meth, n=int(mm.sum()),
                                   yaw_sd=float(np.std(r["yaw"][mm], ddof=1)),
                                   x_sd=float(np.std(px, ddof=1)),
                                   z_sd=float(np.std(pz, ddof=1)),
                                   bear_sd=float(np.std(bear, ddof=1)),
                                   z_mean=float(np.mean(pz))))

        # ---- 회전구간 ----
        for (t0, t1, cmd, step) in rotation_windows(cs):
            m = np.isfinite(t) & (t >= t0 - 1.0) & (t <= t1)
            if args.common:
                m = m & ok_both
            if int(m.sum()) < 12:
                continue
            for meth in METHODS:
                r = res[meth]
                mm = m & np.isfinite(r["yaw"])
                if int(mm.sum()) < 12:
                    continue
                px, pz = r["pos"][mm, 0], r["pos"][mm, 2]
                rng = np.hypot(px, pz)
                bear = np.degrees(np.arctan2(px, pz))
                yw = r["yaw"][mm]
                db = bear - np.median(bear[:4])
                dy = yw - np.median(yw[:4])
                if np.std(db) < 1e-6:
                    continue
                slope = float(np.polyfit(db, dy, 1)[0])
                corr = float(np.corrcoef(db, dy)[0, 1])
                rows_rot.append(dict(tag=short, step=step, cmd=cmd, method=meth,
                                     n=int(mm.sum()), rng_mean=float(rng.mean()),
                                     rng_span=float(rng.max() - rng.min()),
                                     rng_sd=float(np.std(rng, ddof=1)),
                                     rng_rel=float(np.std(rng, ddof=1) / np.mean(rng)),
                                     dbear=float(db.max() - db.min()),
                                     slope=slope, corr=corr))

    R = pd.DataFrame(rows_r)
    S = pd.DataFrame(rows_s)
    T = pd.DataFrame(rows_rot)
    pd.set_option("display.width", 200)

    print("\n=== 1) 재투영 오차 [px] (동일 프레임 집합) ===")
    print(R.to_string(index=False))
    print("\n  전체 요약 (프레임 가중 평균):")
    for m in METHODS:
        g = R[R.method == m]
        w = g.n.values
        print("   %-10s front4 med %6.2f  rms %7.2f | all9 med %6.2f  rms %7.2f  (N=%d)" % (
            m, np.average(g.r4_med, weights=w), np.average(g.r4_rms, weights=w),
            np.average(g.r9_med, weights=w), np.average(g.r9_rms, weights=w), w.sum()))

    print("\n=== 2) 정지구간 안정성 ===")
    print(S.to_string(index=False))
    key = ["tag", "step"]
    sets = [set(map(tuple, S[S.method == m][key].values)) for m in METHODS]
    both = set.intersection(*sets)
    Sp = S[[tuple(v) in both for v in S[key].values]]
    print("\n  공통 창 %d개 - 중앙값 / 최악값:" % len(both))
    for m in METHODS:
        g = Sp[Sp.method == m]
        print("   %-10s yaw_sd med %6.3f max %6.3f deg | x_sd med %7.4f max %7.4f m"
              " | z_sd med %7.4f max %7.4f m | bear_sd med %6.3f max %6.3f deg"
              % (m, g.yaw_sd.median(), g.yaw_sd.max(), g.x_sd.median(), g.x_sd.max(),
                 g.z_sd.median(), g.z_sd.max(), g.bear_sd.median(), g.bear_sd.max()))

    print("\n=== 3-4) 회전구간 |p| 보존 + dyaw/dbearing ===")
    print(T.to_string(index=False))
    print("\n  중앙값 / 최악값:")
    for m in METHODS:
        g = T[T.method == m]
        print("   %-10s |p|span med %.4f max %.4f m | rel_sd med %.4f max %.4f"
              " | slope med %.3f min %.3f | corr med %.3f min %.3f (창 %d)"
              % (m, g.rng_span.median(), g.rng_span.max(), g.rng_rel.median(),
                 g.rng_rel.max(), g.slope.median(), g.slope.min(),
                 g["corr"].median(), g["corr"].min(), len(g)))

    print("\n  회전중심(ICR) 일관성: 강체회전 + 카메라 뒤 d 에 회전중심이면"
          " slope = |p|/(|p|+d)")
    for m in METHODS:
        g = T[(T.method == m) & (T["corr"] > 0.9)]
        if len(g) < 3:
            print("   %-10s corr>0.9 세그먼트 %d개 -> 검정 불가" % (m, len(g)))
            continue
        dd = g.rng_mean.values * (1.0 / g.slope.values - 1.0)
        pred = g.rng_mean.values / (g.rng_mean.values + np.median(dd))
        print("   %-10s N=%2d  d = %.2f +- %.2f m | slope 잔차rms %.3f"
              " (단일 slope 가정 %.3f)"
              % (m, len(g), float(np.median(dd)), float(np.std(dd, ddof=1)),
                 float(np.sqrt(np.mean((g.slope.values - pred) ** 2))),
                 float(np.std(g.slope.values, ddof=1))))

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    R.to_csv(os.path.join(HERE, "out", "eval_reproj.csv"), index=False)
    S.to_csv(os.path.join(HERE, "out", "eval_static.csv"), index=False)
    T.to_csv(os.path.join(HERE, "out", "eval_rotate.csv"), index=False)


if __name__ == "__main__":
    main()
