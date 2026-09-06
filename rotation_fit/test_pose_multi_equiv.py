# -*- coding: utf-8 -*-
"""calib/geometry.py 와 rotation_fit/pose_multi.py 두 구현의 동치 검증.

geometry.py 는 pyrealsense2 / sklearn 을 import 하므로, 없는 환경에서는
더미 모듈을 주입해서 로드한다 (새 함수들은 둘 다 쓰지 않는다).

실행:  python test_pose_multi_equiv.py
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pose_multi as pm  # noqa: E402

DEPTH_CAM = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "extracted", "depth_cam"))


def _stub(name, attrs=None):
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m


def load_geometry():
    try:
        import pyrealsense2  # noqa: F401
    except Exception:
        _stub("pyrealsense2", {"rs2_deproject_pixel_to_point": lambda *a: None})
    try:
        import sklearn.linear_model  # noqa: F401
    except Exception:
        _stub("sklearn")
        _stub("sklearn.linear_model", {"RANSACRegressor": object})
        sys.modules["sklearn"].linear_model = sys.modules["sklearn.linear_model"]
    sys.path.insert(0, DEPTH_CAM)
    import importlib
    return importlib.import_module("calib.geometry")


class Intrin:
    fx = 605.906494140625
    fy = 605.9697875976562
    ppx = 317.59619140625
    ppy = 256.29229736328125
    coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]


def main():
    G = load_geometry()
    K = np.array([[Intrin.fx, 0, Intrin.ppx],
                  [0, Intrin.fy, Intrin.ppy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros((5, 1))

    # 1) 객체모델 동치
    a = G.pallet_object_points()
    b = pm.pallet_object_points("box")
    assert np.allclose(a, b, atol=1e-12), (a, b)
    print("[OK] pallet_object_points 동일  (max diff %.2e)" % np.abs(a - b).max())

    # 2) 실제 키포인트로 pose 동치
    import glob
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "kpts", "*_kpts.npz")))
    if not files:
        print("kpts npz 없음 - pose 비교 생략")
        return
    kp = []
    for f in files:
        k = np.load(f)["kpts"]
        k = k[np.isfinite(k[:, :, :2]).all(axis=(1, 2))]
        kp.append(k)
    kp = np.concatenate(kp)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(kp), size=min(600, len(kp)), replace=False)

    n_ok = n_fb = 0
    dmax = dict(yaw=0.0, pitch=0.0, roll=0.0, cx=0.0, cy=0.0, cz=0.0, rms=0.0)
    for i in sel:
        k = kp[i]
        okG, yG, pG, rG, cG, rvG, tvG, iG = G.pose_from_kpts_pnp_multi(
            k, Intrin, min_front=4, fallback4=False)
        oM = pm.pose_from_kpts_pnp_multi(k, K, D, min_front=4)
        assert bool(okG) == bool(oM["ok"]), (i, okG, oM["ok"])
        if not okG:
            n_fb += 1
            continue
        n_ok += 1
        dmax["yaw"] = max(dmax["yaw"], abs(yG - oM["yaw"]))
        dmax["pitch"] = max(dmax["pitch"], abs(pG - oM["pitch"]))
        dmax["roll"] = max(dmax["roll"], abs(rG - oM["roll"]))
        dmax["cx"] = max(dmax["cx"], abs(float(cG[0]) - float(oM["center"][0])))
        dmax["cy"] = max(dmax["cy"], abs(float(cG[1]) - float(oM["center"][1])))
        dmax["cz"] = max(dmax["cz"], abs(float(cG[2]) - float(oM["center"][2])))
        dmax["rms"] = max(dmax["rms"], abs(iG["rms"] - oM["rms"]))
        assert iG["n_used"] == oM["n_used"]
        assert np.array_equal(iG["inliers"], oM["inliers"])

    print("[OK] pose 비교 %d 프레임 (성공 %d, 실패일치 %d)" % (len(sel), n_ok, n_fb))
    for k2, v in dmax.items():
        print("     max |diff| %-5s = %.3e" % (k2, v))
    assert max(dmax["yaw"], dmax["pitch"], dmax["roll"]) < 1e-9
    assert max(dmax["cx"], dmax["cy"], dmax["cz"]) < 1e-6   # center 는 float32 반환
    print("[PASS] 두 구현 동치")


if __name__ == "__main__":
    main()
