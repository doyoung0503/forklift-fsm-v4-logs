# calib/geometry.py
# 파렛트 9개 키포인트 → 다점 PnP → yaw / pitch / roll / center (6D pose)

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import pyrealsense2 as rs
from typing import Tuple, Optional
from .config import (
    PALLET_FACE_W, PALLET_FACE_H, PALLET_BODY_D,
    POSE_FACE_KPTS, POSE_REAR_KPTS, POSE_CENTER_KPT, POSE_KPT_VIS_THR,
    PNP_RANSAC_REPROJ_ERROR_PX, PNP_RANSAC_ITERATIONS, PNP_RANSAC_CONFIDENCE,
    PLANE_INLIER_THRESH, PLANE_MAX_TRIALS, MIN_POINTS,
    SAMPLE_STRIDE, PLANE_Z_CLIP, POSE_POLY_SHRINK,
    POSE_WIDTH_MIN, POSE_WIDTH_MAX,
    FAST_DEPROJECT, RANSAC_MAX_POINTS,
)


def compute_yaw_deg_from_plane(a: float, b: float) -> float:
    # z = a x + b y + c → 법선(nx,ny,nz)=(-a,-b,1), 여기서 yaw는 x-z 투영을 사용
    nx, nz = -a, 1.0
    return float(np.degrees(np.arctan2(nx, nz)))


def compute_pitch_deg_from_plane(b: float) -> float:
    """
    z = a x + b y + c → 법선(nx,ny,nz)=(-a,-b,1). pitch 는 y-z 투영을 사용.
    카메라 y 축이 아래(+)이므로 부호를 뒤집어 '면 윗변이 뒤로 젖혀지면 +' 가 되게 한다.
    """
    ny, nz = -b, 1.0
    return float(np.degrees(np.arctan2(-ny, nz)))


def compute_roll_deg_from_corners(P4: np.ndarray) -> float:
    """
    전면부 상변(좌상→우상)이 카메라 수평축과 이루는 각.
    법선만으로는 면 내 회전(roll)을 알 수 없어 모서리 방향에서 구한다.
    화면 y 축이 아래(+)이므로 부호를 뒤집어 '반시계 회전이 +' 가 되게 한다.
    """
    u = P4[1] - P4[0]
    return float(np.degrees(np.arctan2(-float(u[1]), float(u[0]))))


def pallet_keypoints_3d(face_w: float = PALLET_FACE_W,
                        face_h: float = PALLET_FACE_H,
                        body_d: float = PALLET_BODY_D) -> np.ndarray:
    """Return the 3D pallet points corresponding to model keypoints 0..8.

    The origin stays at the front-face centre so the PnP translation remains
    compatible with the controller. Axes are x-right, y-down, z-front-to-rear.
    """
    hw, hh = face_w * 0.5, face_h * 0.5
    front = np.array([[-hw, -hh, 0.0],
                      [ hw, -hh, 0.0],
                      [ hw,  hh, 0.0],
                      [-hw,  hh, 0.0]], dtype=np.float64)
    rear = front.copy()
    rear[:, 2] = body_d

    max_index = max(max(POSE_FACE_KPTS), max(POSE_REAR_KPTS), POSE_CENTER_KPT)
    model = np.full((max_index + 1, 3), np.nan, dtype=np.float64)
    model[list(POSE_FACE_KPTS)] = front
    model[list(POSE_REAR_KPTS)] = rear
    model[POSE_CENTER_KPT] = (0.0, 0.0, body_d * 0.5)
    return model


def _pnp_correspondences(kpts: np.ndarray,
                         face_w: float,
                         face_h: float,
                         body_d: float,
                         vis_thr: float):
    """Select visible image points that have known pallet 3D coordinates."""
    if kpts is None:
        return None, None, None

    points = np.asarray(kpts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return None, None, None

    model = pallet_keypoints_3d(face_w, face_h, body_d)
    count = min(len(points), len(model))
    points = points[:count]
    model = model[:count]

    valid = np.all(np.isfinite(points[:, :2]), axis=1)
    valid &= np.all(np.isfinite(model), axis=1)
    if points.shape[1] >= 3:
        valid &= np.isfinite(points[:, 2]) & (points[:, 2] >= float(vis_thr))

    indices = np.flatnonzero(valid)
    if len(indices) < 4:
        # Preserve legacy behaviour when confidence is missing or unreliable.
        front_indices = np.asarray(POSE_FACE_KPTS, dtype=np.int32)
        if (len(points) > int(front_indices.max())
                and np.all(np.isfinite(points[front_indices, :2]))):
            indices = front_indices
        else:
            return None, None, None

    return model[indices], points[indices, :2], indices


def pose_from_kpts_pnp(kpts: np.ndarray, intrin,
                       face_w: float = PALLET_FACE_W,
                       face_h: float = PALLET_FACE_H,
                       body_d: float = PALLET_BODY_D,
                       vis_thr: float = POSE_KPT_VIS_THR):
    """Estimate pallet pose from every usable model keypoint.

    ``kpts`` may be ``(K, 2)`` or ``(K, 3)`` with confidence/visibility in
    column 3. Keypoints 0..3 are the front corners, 4..7 the rear corners,
    and 8 the body centre. Multi-point observations use robust RANSAC; when
    only the legacy front corners are usable, the proven planar IPPE solver
    is retained.
    """
    if kpts is None or intrin is None:
        return False, None, None, None, None, None, None

    obj, img, indices = _pnp_correspondences(
        kpts, face_w, face_h, body_d, vis_thr,
    )
    if obj is None:
        return False, None, None, None, None, None, None

    K = np.array([[intrin.fx, 0.0, intrin.ppx],
                  [0.0, intrin.fy, intrin.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray(getattr(intrin, "coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)

    try:
        front_only = (
            len(indices) == 4
            and np.array_equal(indices, np.asarray(POSE_FACE_KPTS, dtype=np.int64))
        )
        if front_only:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE,
            )
        else:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj, img, K, dist,
                iterationsCount=int(PNP_RANSAC_ITERATIONS),
                reprojectionError=float(PNP_RANSAC_REPROJ_ERROR_PX),
                confidence=float(PNP_RANSAC_CONFIDENCE),
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < 4:
                return False, None, None, None, None, None, None
            inlier_idx = inliers.reshape(-1)
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj[inlier_idx], img[inlier_idx], K, dist, rvec, tvec,
                )
    except cv2.error:
        return False, None, None, None, None, None, None
    if not ok:
        return False, None, None, None, None, None, None
    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return False, None, None, None, None, None, None

    R, _ = cv2.Rodrigues(rvec)
    n = R @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.degrees(np.arctan2(n[0], n[2])))
    pitch = float(np.degrees(np.arctan2(-n[1], n[2])))
    u = R @ np.array([1.0, 0.0, 0.0])
    roll = float(np.degrees(np.arctan2(-u[1], u[0])))

    center = tvec.reshape(3).astype(np.float32)
    if not (np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(roll)
            and np.all(np.isfinite(center)) and float(center[2]) > 0.0):
        return False, None, None, None, None, None, None
    return True, yaw, pitch, roll, center, rvec, tvec


def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W-1, int(x1)))
    y1 = max(0, min(H-1, int(y1)))
    x2 = max(0, min(W-1, int(x2)))
    y2 = max(0, min(H-1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ----------------------------------------------------------------------
# 1) 폴리곤 마스크 (평면적합용 depth 샘플 영역)
# ----------------------------------------------------------------------
def shrink_polygon(poly: np.ndarray, ratio: float) -> np.ndarray:
    """폴리곤을 중심 방향으로 ratio 만큼 축소. 모서리 바깥 배경 depth 유입 차단용."""
    c = poly.mean(axis=0, keepdims=True)
    return (c + (poly - c) * (1.0 - float(ratio))).astype(np.float32)


def polygon_mask(poly: np.ndarray, H: int, W: int, shrink: float = POSE_POLY_SHRINK) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = shrink_polygon(np.asarray(poly, dtype=np.float32), shrink)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


# ----------------------------------------------------------------------
# 2) depth 샘플 → 3D 포인트
# ----------------------------------------------------------------------
def _depths_vectorized(depth_frame, xs, ys) -> Optional[np.ndarray]:
    """depth 이미지를 통째로 받아 numpy 인덱싱. 실패하면 None → 루프 경로로 폴백."""
    try:
        img = np.asanyarray(depth_frame.get_data())
        scale = float(depth_frame.get_units())      # raw unit → meter
        return img[ys, xs].astype(np.float32) * scale
    except Exception:
        return None


def _deproject_vectorized(intr, xs, ys, d) -> np.ndarray:
    """왜곡계수가 0일 때만 유효한 핀홀 역투영."""
    X = (xs.astype(np.float32) - intr.ppx) / intr.fx * d
    Y = (ys.astype(np.float32) - intr.ppy) / intr.fy * d
    return np.stack([X, Y, d], axis=1).astype(np.float32)


def _no_distortion(intr) -> bool:
    try:
        return all(abs(float(c)) < 1e-12 for c in intr.coeffs)
    except Exception:
        return False


def robust_points_from_mask_or_roi(depth_frame, depth_intrin, mask_or_roi,
                                   stride=SAMPLE_STRIDE, z_inlier_thresh=PLANE_Z_CLIP,
                                   min_points=MIN_POINTS) -> Tuple[bool, Optional[np.ndarray]]:
    ys, xs = np.where(mask_or_roi > 0)
    if len(xs) == 0:
        return False, None
    xs = xs[::stride].astype(np.int32)
    ys = ys[::stride].astype(np.int32)

    # --- depth 읽기: 벡터화 우선, 실패 시 기존 루프 ---
    dists = None
    fast = FAST_DEPROJECT and _no_distortion(depth_intrin)
    if fast:
        dists = _depths_vectorized(depth_frame, xs, ys)
    if dists is None:
        fast = False
        dists = np.array([depth_frame.get_distance(int(u), int(v)) for u, v in zip(xs, ys)],
                         dtype=np.float32)

    valid = np.isfinite(dists) & (dists > 0)
    if not np.any(valid):
        return False, None
    xs_v, ys_v, d_v = xs[valid], ys[valid], dists[valid]
    if len(d_v) < min_points:
        return False, None

    # --- 역투영: 왜곡계수가 0이면 numpy, 아니면 rs 함수(왜곡 보정 포함) ---
    if fast:
        pts = _deproject_vectorized(depth_intrin, xs_v, ys_v, d_v)
    else:
        pts = np.array([
            rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], float(d))
            for u, v, d in zip(xs_v, ys_v, d_v)
        ], dtype=np.float32)

    z_med = np.median(pts[:, 2])
    inliers = np.abs(pts[:, 2] - z_med) <= float(z_inlier_thresh)
    pts_in = pts[inliers]
    if len(pts_in) < min_points:
        return False, None
    return True, pts_in


# ----------------------------------------------------------------------
# 3) 평면 적합 — intercept c 까지 반환 (광선∩평면에 필요)
# ----------------------------------------------------------------------
def fit_plane_from_points(pts_in: np.ndarray):
    """z = a*x + b*y + c 적합. return (ok, a, b, c)"""
    if pts_in is None or len(pts_in) < MIN_POINTS:
        return False, None, None, None
    # 점이 많으면 균등 서브샘플 — RANSAC 이 프레임당 최대 병목이다
    if RANSAC_MAX_POINTS and len(pts_in) > RANSAC_MAX_POINTS:
        step = int(np.ceil(len(pts_in) / RANSAC_MAX_POINTS))
        pts_in = pts_in[::step]
    X = pts_in[:, :2]
    Z = pts_in[:, 2]
    ransac = RANSACRegressor(residual_threshold=PLANE_INLIER_THRESH,
                             max_trials=PLANE_MAX_TRIALS, random_state=0)
    ransac.fit(X, Z)
    if ransac.inlier_mask_ is None or ransac.inlier_mask_.sum() < max(30, MIN_POINTS // 2):
        return False, None, None, None
    a, b = ransac.estimator_.coef_
    c = float(ransac.estimator_.intercept_)
    return True, float(a), float(b), c


# ----------------------------------------------------------------------
# 4) 광선 ∩ 평면 — 모서리 픽셀의 depth 값을 직접 쓰지 않는다
#    (모서리는 배경 경계라 depth 가 가장 불안정)
# ----------------------------------------------------------------------
def rays_from_pixels(depth_intrin, uv: np.ndarray) -> np.ndarray:
    """픽셀 → 단위깊이 방향벡터 (N,3). z 성분은 1."""
    return np.array([
        rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], 1.0)
        for u, v in uv
    ], dtype=np.float32)


def intersect_rays_with_plane(rays: np.ndarray, a: float, b: float, c: float) -> Optional[np.ndarray]:
    """P(t) = t*d 를 z = a*x + b*y + c 에 대입 → t = c / (dz - a*dx - b*dy)"""
    d = np.asarray(rays, dtype=np.float64)
    denom = d[:, 2] - a * d[:, 0] - b * d[:, 1]
    if np.any(np.abs(denom) < 1e-9):
        return None
    t = c / denom
    if np.any(t <= 0):
        return None
    return (d * t[:, None]).astype(np.float32)


# ----------------------------------------------------------------------
# 5) 3D 4모서리 → yaw / center / width
#    corners 순서: [좌상, 우상, 우하, 좌하]
# ----------------------------------------------------------------------
def pose_from_face_corners(P4: np.ndarray, a: float, b: float
                           ) -> Tuple[float, float, float, np.ndarray, float]:
    yaw = compute_yaw_deg_from_plane(a, b)
    pitch = compute_pitch_deg_from_plane(b)
    roll = compute_roll_deg_from_corners(P4)
    center = P4.mean(axis=0).astype(np.float32)
    # 상변(좌상-우상)과 하변(좌하-우하)의 3D 길이 평균 = 파렛트 전면부 실폭
    w_top = float(np.linalg.norm(P4[1] - P4[0]))
    w_bot = float(np.linalg.norm(P4[2] - P4[3]))
    width = 0.5 * (w_top + w_bot)
    return yaw, pitch, roll, center, width


# ----------------------------------------------------------------------
# 6) 엔트리 — 키포인트 4점에서 6D pose
# ----------------------------------------------------------------------
def _rvec_tvec_from_plane(P4: np.ndarray, a: float, b: float, center: np.ndarray):
    """
    평면 법선과 전면 상변으로 자세 회전행렬을 구성 → (rvec, tvec).
    화면에 자세 축을 그리기 위한 것. z 축 = 평면 법선, x 축 = 상변(좌상→우상).
    """
    try:
        n = np.array([-a, -b, 1.0], dtype=np.float64)
        n /= np.linalg.norm(n)
        x = (P4[1] - P4[0]).astype(np.float64)
        x = x - np.dot(x, n) * n          # 법선 성분 제거 → 평면 위 방향
        nx = np.linalg.norm(x)
        if nx < 1e-9:
            return None, None
        x /= nx
        y = np.cross(n, x)
        R = np.column_stack([x, y, n])
        rvec, _ = cv2.Rodrigues(R)
        tvec = np.asarray(center, dtype=np.float64).reshape(3, 1)
        return rvec, tvec
    except Exception:
        return None, None


def pose6d_from_kpts(depth_frame, depth_intrin, kpts4: np.ndarray, H: int, W: int):
    """
    kpts4: (4,2) 픽셀좌표 [좌상, 우상, 우하, 좌하]
    return: (ok, yaw_deg, pitch_deg, roll_deg, center3d(3,), width_m, pts_in, rvec, tvec)
            pts_in 은 평면적합에 쓴 3D 포인트(HUD 의 pts 표시용)
            rvec/tvec 은 화면에 자세 축을 그릴 때 쓴다
    """
    fail = (False, None, None, None, None, None, None, None, None)
    if kpts4 is None or len(kpts4) != 4:
        return fail

    mask = polygon_mask(kpts4, H, W)
    ok_pts, pts_in = robust_points_from_mask_or_roi(
        depth_frame=depth_frame, depth_intrin=depth_intrin, mask_or_roi=mask
    )
    if not ok_pts:
        return fail

    ok_plane, a, b, c = fit_plane_from_points(pts_in)
    if not ok_plane:
        return (False, None, None, None, None, None, pts_in, None, None)

    P4 = intersect_rays_with_plane(rays_from_pixels(depth_intrin, kpts4), a, b, c)
    if P4 is None:
        return (False, None, None, None, None, None, pts_in, None, None)

    yaw, pitch, roll, center, width = pose_from_face_corners(P4, a, b)

    # sanity: 3D 폭이 실제 파렛트 폭 범위를 벗어나면 오탐으로 본다.
    # ok=False 로 제어(FSM)에서는 무효화하되, 이미 산출된 yaw/pitch/roll 은 그대로 돌려준다
    # — 이 값들은 평면계수(a,b)와 P4 에서 나오므로 width 와 계산 경로가 다르다. HUD 참고 표시용.
    if not (POSE_WIDTH_MIN <= width <= POSE_WIDTH_MAX):
        print(f"[Geometry] width sanity fail: {width:.3f} m "
              f"(allowed {POSE_WIDTH_MIN}~{POSE_WIDTH_MAX})")
        rvec_f, tvec_f = _rvec_tvec_from_plane(P4, a, b, center)
        return (False, yaw, pitch, roll, center, width, pts_in, rvec_f, tvec_f)

    rvec, tvec = _rvec_tvec_from_plane(P4, a, b, center)
    return True, yaw, pitch, roll, center, width, pts_in, rvec, tvec
