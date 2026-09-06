# -*- coding: utf-8 -*-
"""9개 키포인트의 3D 객체모델을 데이터로 추정한다 (bundle adjustment).

- 자유구조 모델 FREE: 9점 x 3좌표, gauge 7DOF 고정 -> 20 파라미터
    p0 = (0,0,0), p1 = (W,0,0) (W=1.100 로 스케일 고정), p3 = (a,b,0),
    나머지 6점(2,4,5,6,7,8) 자유
- 박스 모델 BOX: 직육면체 + 중심점 -> h, L, c(3) = 5 파라미터
  두 모델의 재투영 RMS 를 비교해 "팔레트가 박스인가" 를 검정한다.

객체 좌표계 규약 (기존 4점 PnP 와 동일):
    +X = kp0 -> kp1 (전면 상변, 화면 오른쪽)
    +Y = kp0 -> kp3 (아래)
    +Z = X x Y = 카메라 반대방향(팔레트 뒤쪽)  -> 정면일 때 R=I, n=R@(0,0,1)=(0,0,1)
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
KPTS = os.path.join(HERE, "kpts")
OUT = os.path.join(HERE, "out")

FX, FY, CX, CY = 605.906494140625, 605.9697875976562, 317.59619140625, 256.29229736328125
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros((5, 1))
IMW, IMH = 640, 480

W_GAUGE = 1.100    # 전면 상변 길이로 스케일 고정


def box_model(h, L, c=None):
    """W=W_GAUGE 고정 박스. p0 이 원점인 gauge 프레임."""
    W = W_GAUGE
    X = np.array([
        [0, 0, 0], [W, 0, 0], [W, h, 0], [0, h, 0],
        [0, 0, L], [W, 0, L], [W, h, L], [0, h, L],
        [W / 2, h / 2, L / 2],
    ], dtype=np.float64)
    if c is not None:
        X[8] = c
    return X


def load_frames(conf_min=0.60, vis_min=0.5, margin=6.0, only=None):
    recs = []
    for f in sorted(glob.glob(os.path.join(KPTS, "*_kpts.npz"))):
        tag = os.path.basename(f)[: -len("_kpts.npz")]
        if only and only not in tag:
            continue
        d = np.load(f)
        k, cf, fi = d["kpts"], d["conf"], d["frame_i"]
        good = np.isfinite(k).all(axis=(1, 2)) & (cf >= conf_min)
        good &= (k[:, :, 2] >= vis_min).all(axis=1)
        good &= (k[:, :, 0] > margin).all(axis=1) & (k[:, :, 0] < IMW - margin).all(axis=1)
        good &= (k[:, :, 1] > margin).all(axis=1) & (k[:, :, 1] < IMH - margin).all(axis=1)
        idx = np.where(good)[0]
        recs.append(dict(tag=tag, kpts=k[idx, :, :2].astype(np.float64),
                         frame_i=fi[idx], conf=cf[idx]))
    return recs


def initial_poses(uv, X0, reproj_max=8.0):
    """프레임별 SQPnP 초기 pose. 재투영 큰 프레임은 버린다."""
    keep, rvecs, tvecs, errs = [], [], [], []
    for i in range(len(uv)):
        ok, rv, tv = cv2.solvePnP(X0, uv[i], K, DIST, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        ok2, rv, tv = cv2.solvePnP(X0, uv[i], K, DIST, rvec=rv, tvec=tv,
                                   useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        pr, _ = cv2.projectPoints(X0, rv, tv, K, DIST)
        e = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - uv[i]) ** 2, axis=1))))
        if not np.isfinite(e) or e > reproj_max or tv[2, 0] <= 0:
            continue
        keep.append(i)
        rvecs.append(rv.ravel())
        tvecs.append(tv.ravel())
        errs.append(e)
    return np.array(keep, dtype=int), np.array(rvecs), np.array(tvecs), np.array(errs)


def project_all(params, nF, struct_fn):
    rv = params[:nF * 3].reshape(nF, 3)
    tv = params[nF * 3:nF * 6].reshape(nF, 3)
    X = struct_fn(params[nF * 6:])
    out = np.empty((nF, X.shape[0], 2))
    for f in range(nF):
        pr, _ = cv2.projectPoints(X, rv[f], tv[f], K, DIST)
        out[f] = pr.reshape(-1, 2)
    return out


def make_residual(uv, struct_fn):
    nF = uv.shape[0]

    def res(p):
        return (project_all(p, nF, struct_fn) - uv).ravel()
    return res


def sparsity(nF, nP, nS):
    m = lil_matrix((nF * nP * 2, nF * 6 + nS), dtype=int)
    for f in range(nF):
        r0 = f * nP * 2
        r1 = r0 + nP * 2
        for c in range(3):
            m[r0:r1, f * 3 + c] = 1
            m[r0:r1, nF * 3 + f * 3 + c] = 1
        m[r0:r1, nF * 6:] = 1
    return m


def struct_free(s):
    """20 파라미터 -> (9,3). gauge: p0 원점, p1=(W,0,0), p3=(a,b,0)"""
    X = np.zeros((9, 3))
    X[1] = [W_GAUGE, 0, 0]
    X[3] = [s[0], s[1], 0.0]
    X[[2, 4, 5, 6, 7, 8]] = s[2:20].reshape(6, 3)
    return X


def struct_box(s):
    """5 파라미터 -> (9,3): h, L, c(3)"""
    return box_model(s[0], s[1], s[2:5])


def canonicalize(X):
    """전면 4점 중심을 원점으로, +X=상변, +Y=아래, +Z=X x Y 인 프레임으로."""
    o = X[:4].mean(axis=0)
    ex = ((X[1] - X[0]) + (X[2] - X[3])) * 0.5
    ex = ex / np.linalg.norm(ex)
    ey = ((X[3] - X[0]) + (X[2] - X[1])) * 0.5
    ey = ey - np.dot(ey, ex) * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R = np.column_stack([ex, ey, ez])
    return (X - o) @ R


def report(name, X, res, uv):
    r = res.reshape(uv.shape)
    per_kpt = np.sqrt((r ** 2).sum(axis=2).mean(axis=0))
    rms = float(np.sqrt((r ** 2).sum(axis=2).mean()))
    print("\n[%s] reproj RMS = %.3f px   (frames=%d)" % (name, rms, uv.shape[0]))
    print("  per-kpt RMS px: " + " ".join("%d:%.2f" % (i, e) for i, e in enumerate(per_kpt)))
    Xc = canonicalize(X)
    print("  canonical object points (m):")
    for i, p in enumerate(Xc):
        print("    kp%d  %8.4f %8.4f %8.4f" % (i, p[0], p[1], p[2]))
    return rms, per_kpt, Xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_frames", type=int, default=480)
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h0", type=float, default=0.15)
    ap.add_argument("--L0", type=float, default=1.30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    recs = load_frames(only=args.only)
    uv_all = np.concatenate([r["kpts"] for r in recs], axis=0)
    print("files: " + str([(r["tag"][-6:], len(r["kpts"])) for r in recs]))
    print("usable frames: %d" % len(uv_all))

    X0 = box_model(args.h0, args.L0)
    keep, rv0, tv0, err0 = initial_poses(uv_all, X0)
    print("init PnP kept %d / %d  (median reproj %.2f px)" % (
        len(keep), len(uv_all), float(np.median(err0)) if len(err0) else -1))

    # pose 다양성 확보: bearing 기준 층화 서브샘플
    rng = np.random.default_rng(args.seed)
    bear = np.degrees(np.arctan2(tv0[:, 0], tv0[:, 2]))
    nb = 24
    span = max(1e-6, float(bear.max() - bear.min()))
    bins = np.clip(((bear - bear.min()) / span * nb).astype(int), 0, nb - 1)
    sel = []
    per = max(1, args.max_frames // nb)
    for b in range(nb):
        ii = np.where(bins == b)[0]
        if len(ii) == 0:
            continue
        sel.append(rng.choice(ii, size=min(per, len(ii)), replace=False))
    sel = np.sort(np.concatenate(sel))
    print("selected %d frames, bearing %.1f..%.1f deg, z %.2f..%.2f m" % (
        len(sel), bear[sel].min(), bear[sel].max(), tv0[sel, 2].min(), tv0[sel, 2].max()))

    uv = uv_all[keep][sel]
    rv, tv = rv0[sel], tv0[sel]
    nF = len(uv)
    p_pose = np.concatenate([rv.ravel(), tv.ravel()])

    results = {}
    for name, struct_fn, s0 in [
        ("BOX", struct_box,
         np.array([args.h0, args.L0, W_GAUGE / 2, args.h0 / 2, args.L0 / 2])),
        ("FREE", struct_free,
         np.concatenate([[0.0, args.h0], X0[[2, 4, 5, 6, 7, 8]].ravel()])),
    ]:
        p0 = np.concatenate([p_pose, s0])
        js = sparsity(nF, 9, len(s0))
        sol = least_squares(make_residual(uv, struct_fn), p0, jac_sparsity=js,
                            loss="huber", f_scale=2.0, method="trf",
                            xtol=1e-12, ftol=1e-12, max_nfev=300, verbose=0)
        Xf = struct_fn(sol.x[nF * 6:])
        rms, per_kpt, Xc = report(name, Xf, sol.fun, uv)
        results[name] = dict(X=Xf, Xc=Xc, rms=rms, per_kpt=per_kpt, sol=sol)

    sb = results["BOX"]["sol"].x[nF * 6:]
    print("\nBOX params: h = %.4f m, L = %.4f m, center = (%.4f, %.4f, %.4f)" %
          (sb[0], sb[1], sb[2], sb[3], sb[4]))
    print("  center in units of (W,h,L): (%.3f, %.3f, %.3f)" %
          (sb[2] / W_GAUGE, sb[3] / max(sb[0], 1e-9), sb[4] / max(sb[1], 1e-9)))

    Xf = results["FREE"]["Xc"]
    print("\nFREE 구조의 박스 이탈량 (m):")
    print("  전면 상변 |01| = %.4f, 하변 |32| = %.4f" %
          (np.linalg.norm(Xf[1] - Xf[0]), np.linalg.norm(Xf[2] - Xf[3])))
    print("  후면 상변 |45| = %.4f, 하변 |76| = %.4f" %
          (np.linalg.norm(Xf[5] - Xf[4]), np.linalg.norm(Xf[6] - Xf[7])))
    print("  높이 |03| = %.4f, |12| = %.4f, |47| = %.4f, |56| = %.4f" %
          (np.linalg.norm(Xf[3] - Xf[0]), np.linalg.norm(Xf[2] - Xf[1]),
           np.linalg.norm(Xf[7] - Xf[4]), np.linalg.norm(Xf[6] - Xf[5])))
    print("  깊이 |04| = %.4f, |15| = %.4f, |26| = %.4f, |37| = %.4f" %
          (np.linalg.norm(Xf[4] - Xf[0]), np.linalg.norm(Xf[5] - Xf[1]),
           np.linalg.norm(Xf[6] - Xf[2]), np.linalg.norm(Xf[7] - Xf[3])))
    print("  kp8 - mean(8corners) = " + str(np.round(Xf[8] - Xf[:8].mean(axis=0), 4)))
    print("  kp8 - mean(front4)   = " + str(np.round(Xf[8] - Xf[:4].mean(axis=0), 4)))

    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "object_model%s.npz" % (("_" + args.tag) if args.tag else ""))
    np.savez(dst,
             X_free=results["FREE"]["Xc"], X_box=canonicalize(results["BOX"]["X"]),
             box_params=sb, rms_free=results["FREE"]["rms"], rms_box=results["BOX"]["rms"],
             per_kpt_free=results["FREE"]["per_kpt"], per_kpt_box=results["BOX"]["per_kpt"])
    print("\nsaved " + dst)


if __name__ == "__main__":
    main()
