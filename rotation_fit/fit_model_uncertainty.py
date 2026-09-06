# -*- coding: utf-8 -*-
"""객체모델(h, L, kp8) 추정값의 불확실성 — 부트스트랩 + 녹화별 분해."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import fit_object_model as F


def fit_box(uv, rv, tv, h0=0.14, L0=1.15):
    nF = len(uv)
    s0 = np.array([h0, L0, F.W_GAUGE / 2, h0 / 2, L0 / 2])
    p0 = np.concatenate([rv.ravel(), tv.ravel(), s0])
    js = F.sparsity(nF, 9, len(s0))
    sol = least_squares(F.make_residual(uv, F.struct_box), p0, jac_sparsity=js,
                        loss="huber", f_scale=2.0, method="trf",
                        xtol=1e-10, ftol=1e-10, max_nfev=200)
    s = sol.x[nF * 6:]
    r = sol.fun.reshape(uv.shape)
    rms = float(np.sqrt((r ** 2).sum(axis=2).mean()))
    return s, rms


def main():
    recs = F.load_frames()
    print("per-recording BOX fit")
    print("%-8s %6s %8s %8s %8s %8s %8s %7s" %
          ("tag", "N", "h", "L", "c_x/W", "c_y/h", "c_z/L", "rms"))
    print("-" * 70)
    allsel = []
    for r in recs:
        uv_all = r["kpts"]
        if len(uv_all) < 60:
            print("%-8s %6d  (skip: too few)" % (r["tag"][-6:], len(uv_all)))
            continue
        X0 = F.box_model(0.14, 1.15)
        keep, rv0, tv0, e0 = F.initial_poses(uv_all, X0)
        rng = np.random.default_rng(0)
        sel = rng.choice(len(keep), size=min(200, len(keep)), replace=False)
        s, rms = fit_box(uv_all[keep][sel], rv0[sel], tv0[sel])
        print("%-8s %6d %8.4f %8.4f %8.3f %8.3f %8.3f %7.2f" %
              (r["tag"][-6:], len(sel), s[0], s[1], s[2] / F.W_GAUGE,
               s[3] / s[0], s[4] / s[1], rms))
        allsel.append((uv_all[keep], rv0, tv0))

    uv = np.concatenate([a[0] for a in allsel])
    rv = np.concatenate([a[1] for a in allsel])
    tv = np.concatenate([a[2] for a in allsel])
    print("\nbootstrap over %d frames (30 draws x 200 frames)" % len(uv))
    hs, Ls, cx, cy, cz = [], [], [], [], []
    for b in range(30):
        rng = np.random.default_rng(1000 + b)
        sel = rng.choice(len(uv), size=200, replace=False)
        s, rms = fit_box(uv[sel], rv[sel], tv[sel])
        hs.append(s[0]); Ls.append(s[1])
        cx.append(s[2] / F.W_GAUGE); cy.append(s[3] / s[0]); cz.append(s[4] / s[1])
    for nm, v in [("h [m]", hs), ("L [m]", Ls), ("kp8 x/W", cx),
                  ("kp8 y/h", cy), ("kp8 z/L", cz)]:
        v = np.asarray(v)
        print("  %-9s mean %8.4f  sd %8.4f  [%.4f, %.4f]" %
              (nm, v.mean(), v.std(ddof=1), v.min(), v.max()))


if __name__ == "__main__":
    main()
