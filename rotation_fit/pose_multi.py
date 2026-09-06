# -*- coding: utf-8 -*-
"""오프라인 재계산용 다중 키포인트 PnP (numpy + cv2 만 사용).

calib/geometry.py 의 pallet_object_points() / pose_from_kpts_pnp_multi() 와
동일한 수식의 독립 구현이다. (geometry.py 는 pyrealsense2/sklearn 을 import 하므로
오프라인에서는 쓸 수 없다.)  test_pose_multi_equiv.py 로 두 구현의 동치를 확인한다.

객체 좌표계 (기존 4점 PnP 와 동일 규약):
    원점 = 전면 4모서리의 중심
    +X = kp0 -> kp1 (전면 상변, 화면 오른쪽)
    +Y = 아래
    +Z = X x Y = 팔레트 뒤쪽(정면일 때 카메라에서 멀어지는 방향)
  -> 정면에서 R = I, n = R @ (0,0,1) = (0,0,1), yaw = 0.
"""
from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 객체 모델 — rotation_fit/fit_object_model.py 의 번들조정 결과
#   gauge: 전면 상변 |kp0-kp1| = 1.100 m (스케일 고정)
# ---------------------------------------------------------------------------
PALLET_KPT_W = 1.100      # 전면 폭 (스케일 gauge)
PALLET_KPT_H = 0.140      # 전면 높이 (BA 추정, 녹화간 sd 0.007)
PALLET_KPT_L = 1.150      # 팔레트 길이 (전->후, BA 추정, 녹화간 sd 0.04)

# BA 잔차 기반 키포인트별 표준편차 [px] (가중 PnP 용)
PALLET_KPT_SIGMA_PX = np.array(
    [1.26, 1.48, 1.75, 1.86, 2.12, 1.85, 1.90, 1.99, 1.31], dtype=np.float64)

# kp8(중심점)의 박스중심 대비 잔여 오프셋 (W,h,L 단위 비율로 측정: 0.491/0.529/0.530)
PALLET_KPT8_OFF = np.array([-0.0089, 0.0289, 0.0297], dtype=np.float64)

# 자유구조 번들조정 결과 (박스 가정 없이 9점을 그대로 추정한 것)
_X_FREE = np.array([
    [-0.5538, -0.0691,  0.0088],
    [ 0.5461, -0.0759, -0.0088],
    [ 0.5540,  0.0759,  0.0088],
    [-0.5463,  0.0691, -0.0088],
    [-0.5496,  0.0855,  1.2393],
    [ 0.5943,  0.0824,  1.2681],
    [ 0.5902,  0.2240,  1.2536],
    [-0.5410,  0.2224,  1.2292],
    [-0.0004,  0.0846,  0.6498],
], dtype=np.float64)


def pallet_object_points(model: str = "box", w: float = PALLET_KPT_W,
                         h: float = PALLET_KPT_H, l: float = PALLET_KPT_L) -> np.ndarray:
    """9개 키포인트의 3D 객체좌표 (9,3).

    model="box"  : 직육면체 가정. kp8 = 8모서리 중심.
    model="free" : 번들조정이 추정한 비강체 보정 포함 구조(고정 상수).
    kp 순서 0..3 = 전면 좌상/우상/우하/좌하, 4..7 = 후면 같은 순환순서, 8 = 팔레트 중심.
    """
    if model == "free":
        return _X_FREE.copy()
    hw, hh = 0.5 * float(w), 0.5 * float(h)
    L = float(l)
    return np.array([
        [-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0],
        [-hw, -hh, L], [hw, -hh, L], [hw, hh, L], [-hw, hh, L],
        [PALLET_KPT8_OFF[0] * w, PALLET_KPT8_OFF[1] * h, (0.5 + PALLET_KPT8_OFF[2]) * L],
    ], dtype=np.float64)


def _angles_from_R(R: np.ndarray):
    """기존 4점 PnP 와 동일한 각도 규약."""
    n = R @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.degrees(np.arctan2(n[0], n[2])))
    pitch = float(np.degrees(np.arctan2(-n[1], n[2])))
    u = R @ np.array([1.0, 0.0, 0.0])
    roll = float(np.degrees(np.arctan2(-u[1], u[0])))
    return yaw, pitch, roll


def _refine_weighted(obj, img, K, dist, rvec, tvec, w, iters=12, huber=2.5):
    """가중 + Huber IRLS Levenberg-Marquardt 정제.

    cv2.projectPoints 의 야코비안 앞 6열 = d(u,v)/d(rvec,tvec).
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
    lam = 1e-3
    prev = None
    for _ in range(int(iters)):
        pr, J = cv2.projectPoints(obj, rvec, tvec, K, dist)
        pr = pr.reshape(-1, 2)
        res = img - pr                                    # (N,2)
        d = np.linalg.norm(res, axis=1)
        # Huber: 큰 잔차는 선형 가중
        hw = np.ones_like(d)
        big = d > huber
        hw[big] = huber / np.maximum(d[big], 1e-9)
        wp = w * hw                                       # (N,)
        cost = float(np.sum(wp * d * d))
        if prev is not None and cost > prev:
            lam *= 10.0
            if lam > 1e6:
                break
        else:
            lam = max(lam * 0.5, 1e-9)
        prev = cost
        Wd = np.repeat(wp, 2)                             # (2N,)
        A = J[:, :6]                                      # (2N,6)
        r = res.reshape(-1)
        H = A.T @ (A * Wd[:, None])
        g = A.T @ (Wd * r)
        try:
            dx = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dx)):
            break
        rvec += dx[:3].reshape(3, 1)
        tvec += dx[3:].reshape(3, 1)
        if np.linalg.norm(dx) < 1e-10:
            break
    return rvec, tvec


def _reproj(obj, img, K, dist, rvec, tvec):
    pr, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.linalg.norm(img - pr.reshape(-1, 2), axis=1)


def pose_from_kpts_pnp_multi(kpts_all, K, dist=None, obj=None,
                             vis_thr: float = 0.30,
                             margin_px: float = 40.0,
                             img_wh=(640, 480),
                             min_pts: int = 6,
                             reproj_thr: float = 6.0,
                             rms_max: float = 8.0,
                             use_weights: bool = True,
                             min_front: int = 2):
    """보이는 모든 키포인트로 6D pose 를 푼다.

    kpts_all : (9,3) [x, y, vis] 또는 (9,2)
    K        : (3,3) 카메라 행렬,  dist: (5,1) 왜곡계수(None 이면 0)
    obj      : (9,3) 객체모델. None 이면 pallet_object_points("box").

    return dict:
        ok, yaw, pitch, roll, center(3,) = 전면 4모서리 중심의 3D 좌표,
        rvec, tvec, R, n_used, rms, inliers(bool 9,), centroid(3,)
    """
    fail = dict(ok=False, yaw=None, pitch=None, roll=None, center=None,
                rvec=None, tvec=None, R=None, n_used=0, rms=None,
                inliers=None, centroid=None)
    if kpts_all is None:
        return fail
    kp = np.asarray(kpts_all, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[0] < 9:
        return fail
    if obj is None:
        obj = pallet_object_points("box")
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)[:9]
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((5, 1)) if dist is None else np.asarray(dist, np.float64).reshape(-1, 1)

    uv = kp[:9, :2]
    vis = kp[:9, 2] if kp.shape[1] >= 3 else np.ones(9)

    W, H = float(img_wh[0]), float(img_wh[1])
    ok_m = np.isfinite(uv).all(axis=1) & (vis >= vis_thr)
    ok_m &= (uv[:, 0] > -margin_px) & (uv[:, 0] < W + margin_px)
    ok_m &= (uv[:, 1] > -margin_px) & (uv[:, 1] < H + margin_px)
    if int(ok_m.sum()) < min_pts or int(ok_m[:4].sum()) < min_front:
        return fail

    sig = PALLET_KPT_SIGMA_PX if use_weights else np.ones(9)
    wall = 1.0 / (sig ** 2)
    wall = wall / wall.mean()

    idx = np.where(ok_m)[0]
    for _ in range(3):                       # 이상치 배제 후 최대 2회 재계산
        o, i2 = obj[idx], uv[idx]
        try:
            ok, rvec, tvec = cv2.solvePnP(o, i2, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return fail
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            return fail
        rvec, tvec = _refine_weighted(o, i2, K, dist, rvec, tvec, wall[idx])
        if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
            return fail
        d = _reproj(o, i2, K, dist, rvec, tvec)
        bad = d > reproj_thr
        # 전면 4점은 재투영 이상치라도 우선 유지 (원점/각도 규약의 기준)
        keep = ~bad | (idx < 4)
        if bad.sum() == 0 or int(keep.sum()) < min_pts:
            break
        new_idx = idx[keep]
        if len(new_idx) == len(idx):
            break
        idx = new_idx

    o, i2 = obj[idx], uv[idx]
    d = _reproj(o, i2, K, dist, rvec, tvec)
    rms = float(np.sqrt(np.mean(d ** 2)))
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    if not np.isfinite(rms) or rms > rms_max:
        return fail

    c_front = obj[:4].mean(axis=0)
    center = (R @ c_front + t).astype(np.float64)
    centroid = (R @ obj[:8].mean(axis=0) + t).astype(np.float64)
    if not np.isfinite(center).all() or center[2] <= 0.0:
        return fail

    yaw, pitch, roll = _angles_from_R(R)
    if not (np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(roll)):
        return fail

    inl = np.zeros(9, dtype=bool)
    inl[idx] = True
    return dict(ok=True, yaw=yaw, pitch=pitch, roll=roll,
                center=center.astype(np.float32), rvec=rvec, tvec=tvec, R=R,
                n_used=int(len(idx)), rms=rms, inliers=inl,
                centroid=centroid.astype(np.float32))


def pose_from_kpts_pnp4(kpts4, K, dist=None, face_w=1.100, face_h=0.150):
    """기존 calib/geometry.pose_from_kpts_pnp 와 동일한 4점 IPPE 경로 (비교 기준)."""
    kp = np.asarray(kpts4, dtype=np.float64).reshape(-1, 2)[:4]
    if not np.all(np.isfinite(kp)):
        return dict(ok=False)
    hw, hh = face_w * 0.5, face_h * 0.5
    obj = np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]])
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((5, 1)) if dist is None else np.asarray(dist, np.float64).reshape(-1, 1)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, kp, K, dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return dict(ok=False)
    if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return dict(ok=False)
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _angles_from_R(R)
    center = tvec.reshape(3)
    if center[2] <= 0:
        return dict(ok=False)
    d = _reproj(obj, kp, K, dist, rvec, tvec)
    return dict(ok=True, yaw=yaw, pitch=pitch, roll=roll,
                center=center.astype(np.float32), rvec=rvec, tvec=tvec, R=R,
                rms=float(np.sqrt(np.mean(d ** 2))))
