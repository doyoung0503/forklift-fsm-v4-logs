# calib/geometry.py
# 파렛트 전면부 4모서리 키포인트 → 3D 평면 투영 → yaw / center / width (6D pose)

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # Camera-OFF UI tests do not use depth deprojection.
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


def pallet_keypoints_3d(face_w: float = PALLET_FACE_W,
                        face_h: float = PALLET_FACE_H,
                        body_d: float = PALLET_BODY_D) -> np.ndarray:
    """livegt 모델 키포인트 0..8에 대응하는 팔레트 3D 좌표."""
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


def largest_projected_vertical_face(R: np.ndarray, tvec: np.ndarray,
                                    K: np.ndarray, dist: np.ndarray,
                                    face_w: float = PALLET_FACE_W,
                                    face_h: float = PALLET_FACE_H,
                                    body_d: float = PALLET_BODY_D):
    """Choose the camera-facing vertical cuboid face with largest image area.

    The PnP keypoint numbering is used only to recover the cuboid pose.  This
    function then removes the semantic 0..3="front" assumption by considering
    all four vertical faces.  The returned pose has its origin at the selected
    face centre and local +Z pointing inward through the pallet body.
    """
    rotation = np.asarray(R, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    model = pallet_keypoints_3d(face_w, face_h, body_d)
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(model[:8], rvec, translation, K, dist)
    projected = projected.reshape(-1, 2)
    camera_points = (rotation @ model[:8].T).T + translation.reshape(1, 3)

    f0, f1, f2, f3 = map(int, POSE_FACE_KPTS)
    r0, r1, r2, r3 = map(int, POSE_REAR_KPTS)
    hw = float(face_w) * 0.5
    depth = float(body_d)

    # basis columns are the selected frame's +X, +Y, +Z expressed in the
    # original PnP object frame.  +Z always points from the selected surface
    # into the cuboid, preserving the existing front-normal yaw convention.
    specs = (
        ("z_min", (f0, f1, f2, f3), (0.0, 0.0, -1.0),
         (0.0, 0.0, 0.0), np.eye(3, dtype=np.float64), float(face_w)),
        ("z_max", (r0, r1, r2, r3), (0.0, 0.0, 1.0),
         (0.0, 0.0, depth),
         np.array([[-1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, -1.0]], dtype=np.float64), float(face_w)),
        ("x_min", (f0, f3, r3, r0), (-1.0, 0.0, 0.0),
         (-hw, 0.0, depth * 0.5),
         np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0]], dtype=np.float64), depth),
        ("x_max", (f1, r1, r2, f2), (1.0, 0.0, 0.0),
         (hw, 0.0, depth * 0.5),
         np.array([[0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]], dtype=np.float64), depth),
    )

    candidates = []
    area_by_face = {}
    facing_by_face = {}
    for name, indices, outward, centre, basis, width in specs:
        index_array = np.asarray(indices, dtype=np.int32)
        polygon = projected[index_array]
        area = (
            float(abs(cv2.contourArea(polygon.astype(np.float32))))
            if np.all(np.isfinite(polygon)) else 0.0
        )
        centre_obj = np.asarray(centre, dtype=np.float64)
        centre_cam = rotation @ centre_obj + translation
        outward_cam = rotation @ np.asarray(outward, dtype=np.float64)
        view_norm = max(1e-9, float(np.linalg.norm(centre_cam)))
        facing = float(np.dot(outward_cam, -centre_cam) / view_norm)
        positive_depth = bool(np.all(camera_points[index_array, 2] > 0.0))
        visible = positive_depth and facing > 0.0 and area > 0.0
        area_by_face[name] = area
        facing_by_face[name] = facing
        candidates.append(dict(
            name=name, indices=index_array, centre=centre_obj, basis=basis,
            width=width, area=area, facing=facing, visible=visible,
            polygon=polygon,
        ))

    visible_candidates = [item for item in candidates if item["visible"]]
    pool = visible_candidates or [item for item in candidates if item["area"] > 0.0]
    if not pool:
        return rotation, translation.astype(np.float32), dict(
            selected_front_face=None,
            selected_face_corners_px=None,
            selected_face_area_px2=None,
            selected_face_width_m=None,
            projected_face_areas_px2=area_by_face,
            projected_face_facing=facing_by_face,
        )

    selected = max(pool, key=lambda item: item["area"])
    selected_rotation = rotation @ selected["basis"]
    selected_translation = rotation @ selected["centre"] + translation
    return selected_rotation, selected_translation.astype(np.float32), dict(
        selected_front_face=selected["name"],
        selected_face_corners_px=selected["polygon"].astype(float).tolist(),
        selected_face_area_px2=float(selected["area"]),
        selected_face_width_m=float(selected["width"]),
        projected_face_areas_px2=area_by_face,
        projected_face_facing=facing_by_face,
    )


def visible_pnp_correspondences(kpts: np.ndarray,
                                face_w: float = PALLET_FACE_W,
                                face_h: float = PALLET_FACE_H,
                                body_d: float = PALLET_BODY_D,
                                vis_thr: float = POSE_KPT_VIS_THR):
    """3D 좌표가 있고 visibility 임계값을 넘는 관측점만 선택한다."""
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
        # 원본 livegt 프로젝트와 같은 호환 폴백: visibility 열이 없거나 신뢰할 수
        # 없을 때 전면 네 점의 유한 좌표가 있으면 기존 IPPE 경로를 유지한다.
        front_indices = np.asarray(POSE_FACE_KPTS, dtype=np.int32)
        if (len(points) > int(front_indices.max())
                and np.all(np.isfinite(points[front_indices, :2]))):
            indices = front_indices
        else:
            return None, None, None

    return model[indices], points[indices, :2], indices


def pose_from_visible_kpts_pnp(kpts: np.ndarray, intrin,
                               face_w: float = PALLET_FACE_W,
                               face_h: float = PALLET_FACE_H,
                               body_d: float = PALLET_BODY_D,
                               vis_thr: float = POSE_KPT_VIS_THR):
    """보이는 전면/후면/중심 키포인트를 모두 사용해 팔레트 pose를 푼다.

    4개의 전면점만 선택되면 기존 IPPE를 사용하고, 그 외 조합은 RANSAC EPNP와
    LM 정제를 사용한다. 마지막 반환값은 런타임 로그용 정보다.
    """
    info = dict(n_used=0, rms=None, inliers=None, centroid=None,
                indices=None, src=None)
    fail = (False, None, None, None, None, None, None, info)
    if kpts is None or intrin is None:
        return fail

    obj, img, indices = visible_pnp_correspondences(
        kpts, face_w, face_h, body_d, vis_thr,
    )
    if obj is None:
        return fail

    K = np.array([[intrin.fx, 0.0, intrin.ppx],
                  [0.0, intrin.fy, intrin.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray(
        getattr(intrin, "coeffs", [0, 0, 0, 0, 0]), dtype=np.float64,
    ).reshape(-1, 1)

    front_only = (
        len(indices) == len(POSE_FACE_KPTS)
        and np.array_equal(indices, np.asarray(POSE_FACE_KPTS, dtype=np.int64))
    )
    selected_inliers = np.arange(len(indices), dtype=np.int32)
    try:
        if front_only:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE,
            )
            src = "pnp4_ippe"
        else:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj, img, K, dist,
                iterationsCount=int(PNP_RANSAC_ITERATIONS),
                reprojectionError=float(PNP_RANSAC_REPROJ_ERROR_PX),
                confidence=float(PNP_RANSAC_CONFIDENCE),
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < 4:
                return fail
            selected_inliers = inliers.reshape(-1)
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj[selected_inliers], img[selected_inliers], K, dist,
                    rvec, tvec,
                )
            src = "pnp_visible_ransac"
    except cv2.error:
        return fail

    if (not ok or not np.all(np.isfinite(rvec))
            or not np.all(np.isfinite(tvec))):
        return fail

    # Reprojection quality belongs to the original labelled-corner PnP solve.
    # The pose returned to the FSM is canonicalised separately below.
    solve_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
    solve_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
    R, _ = cv2.Rodrigues(solve_rvec)
    R, selected_center, face_info = largest_projected_vertical_face(
        R, solve_tvec, K, dist, face_w, face_h, body_d,
    )
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.asarray(selected_center, dtype=np.float64).reshape(3, 1)
    n = R @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.degrees(np.arctan2(n[0], n[2])))
    pitch = float(np.degrees(np.arctan2(-n[1], n[2])))
    u = R @ np.array([1.0, 0.0, 0.0])
    roll = float(np.degrees(np.arctan2(-u[1], u[0])))
    center = tvec.reshape(3).astype(np.float32)
    if not (np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(roll)
            and np.all(np.isfinite(center)) and float(center[2]) > 0.0):
        return fail

    projected, _ = cv2.projectPoints(
        obj[selected_inliers], solve_rvec, solve_tvec, K, dist,
    )
    residual = projected.reshape(-1, 2) - img[selected_inliers]
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    absolute_inliers = np.asarray(indices)[selected_inliers]
    inlier_mask = np.zeros(len(pallet_keypoints_3d(face_w, face_h, body_d)),
                           dtype=bool)
    inlier_mask[absolute_inliers] = True
    info = dict(
        n_used=int(len(absolute_inliers)),
        rms=rms,
        inliers=inlier_mask,
        centroid=None,
        indices=np.asarray(indices, dtype=np.int32),
        src=src,
        **face_info,
    )
    return True, yaw, pitch, roll, center, rvec, tvec, info


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


def pose_from_kpts_pnp(kpts4: np.ndarray, intrin,
                       face_w: float = PALLET_FACE_W, face_h: float = PALLET_FACE_H):
    """
    depth 없이 전면 4모서리(픽셀)만으로 자세를 푼다 (solvePnP).
    kpts4: (4,2) [좌상, 우상, 우하, 좌하]
    return: (ok, yaw, pitch, roll, center3d(3,), rvec, tvec)
            rvec/tvec 은 화면에 자세 축을 그릴 때 쓴다. 각도 규약은 평면기반과 동일.
    """
    if kpts4 is None or len(kpts4) != 4 or intrin is None:
        return False, None, None, None, None, None, None

    # 모델 좌표계는 카메라/이미지와 같은 규약으로 둔다: x 오른쪽, y 아래, z 전방.
    # kpts4 순서(좌상,우상,우하,좌하)와 그대로 대응된다.
    hw, hh = face_w * 0.5, face_h * 0.5
    obj = np.array([[-hw, -hh, 0.0],
                    [ hw, -hh, 0.0],
                    [ hw,  hh, 0.0],
                    [-hw,  hh, 0.0]], dtype=np.float64)
    img = np.asarray(kpts4, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(img)):
        return False, None, None, None, None, None, None

    K = np.array([[intrin.fx, 0.0, intrin.ppx],
                  [0.0, intrin.fy, intrin.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray(getattr(intrin, "coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)

    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return False, None, None, None, None, None, None
    if not ok:
        return False, None, None, None, None, None, None
    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return False, None, None, None, None, None, None

    R, _ = cv2.Rodrigues(rvec)
    # 평면기반 yaw 와 같은 규약: 정면일 때 법선이 (0,0,1) 이 되도록 모델 +z 를 쓴다
    n = R @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.degrees(np.arctan2(n[0], n[2])))
    pitch = float(np.degrees(np.arctan2(-n[1], n[2])))
    # 상변 방향(모델 +x) 으로 면내 회전 — 화면상 반시계가 +
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


# ======================================================================
# 다중 키포인트(9점) PnP - 2026-09-03 추가
#
# 배경: 전면 4점만 쓰는 pose_from_kpts_pnp() 는 전면부가 1.10 x 0.14 m 의
#   가늘고 긴 평면이라 out-of-plane 회전이 거의 관측되지 않는다.  실제 로그에서
#   회전 중 yaw 가 반대 방향으로 움직이는 세그먼트가 12개 중 4개였다.
#   후면 4모서리(4~7)를 함께 쓰면 전후 baseline 1.15 m 가 생겨 yaw 가 관측된다.
#
# 객체모델은 녹화 5건 6211 프레임 번들조정으로 추정했다
#   (rotation_fit/fit_object_model.py, rotation_fit/fit_model_uncertainty.py).
#   전면 폭 1.100 m 를 스케일 gauge 로 고정했을 때
#     전면 높이 h = 0.140 m (녹화간 sd 0.007)
#     길이     L = 1.150 m (녹화간 sd 0.04)
#     kp8      = 8모서리 중심 (0.491 W, 0.529 h, 0.530 L)
#   박스 가정 재투영 RMS 1.75 px vs 자유구조 1.69 px -> 박스 모델로 충분.
#
# 객체 좌표계는 기존 4점 경로와 동일하다:
#   원점 = 전면 4모서리 중심, +X = kp0->kp1, +Y = 아래, +Z = 팔레트 뒤쪽.
#   따라서 tvec 이 곧 FSM 이 쓰는 "전면 중심" 이고 각도 규약도 그대로다.
# ======================================================================

PALLET_KPT_W = 1.100      # 전면 폭 (스케일 gauge)
PALLET_KPT_H = 0.140      # 전면 높이 (번들조정)
PALLET_KPT_L = 1.150      # 전->후 길이 (번들조정)
PALLET_KPT8_OFF = np.array([-0.0089, 0.0289, 0.0297], dtype=np.float64)
PALLET_KPT_SIGMA_PX = np.array(
    [1.26, 1.48, 1.75, 1.86, 2.12, 1.85, 1.90, 1.99, 1.31], dtype=np.float64)


def pallet_object_points(w: float = PALLET_KPT_W, h: float = PALLET_KPT_H,
                         l: float = PALLET_KPT_L) -> np.ndarray:
    """9개 키포인트의 3D 객체좌표 (9,3) float64.

    0~3 = 전면 좌상/우상/우하/좌하, 4~7 = 후면 같은 순환순서, 8 = 팔레트 중심.
    """
    hw, hh = 0.5 * float(w), 0.5 * float(h)
    L = float(l)
    return np.array([
        [-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0],
        [-hw, -hh, L], [hw, -hh, L], [hw, hh, L], [-hw, hh, L],
        [PALLET_KPT8_OFF[0] * w, PALLET_KPT8_OFF[1] * h,
         (0.5 + PALLET_KPT8_OFF[2]) * L],
    ], dtype=np.float64)


def _angles_from_R(R: np.ndarray):
    """pose_from_kpts_pnp() 와 완전히 동일한 각도 규약."""
    n = R @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.degrees(np.arctan2(n[0], n[2])))
    pitch = float(np.degrees(np.arctan2(-n[1], n[2])))
    u = R @ np.array([1.0, 0.0, 0.0])
    roll = float(np.degrees(np.arctan2(-u[1], u[0])))
    return yaw, pitch, roll


def _reproj_err(obj, img, K, dist, rvec, tvec) -> np.ndarray:
    pr, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.linalg.norm(img - pr.reshape(-1, 2), axis=1)


def _refine_weighted(obj, img, K, dist, rvec, tvec, w, iters=12, huber=2.5):
    """가중 + Huber IRLS Levenberg-Marquardt 정제.

    cv2.projectPoints 의 야코비안 앞 6열이 d(u,v)/d(rvec,tvec) 이다.
    cv2 의 solvePnPRefineLM 은 가중치를 받지 못하므로 직접 푼다.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
    lam = 1e-3
    prev = None
    for _ in range(int(iters)):
        pr, J = cv2.projectPoints(obj, rvec, tvec, K, dist)
        res = img - pr.reshape(-1, 2)
        d = np.linalg.norm(res, axis=1)
        hw = np.ones_like(d)
        big = d > huber
        hw[big] = huber / np.maximum(d[big], 1e-9)
        wp = w * hw
        cost = float(np.sum(wp * d * d))
        if prev is not None and cost > prev:
            lam *= 10.0
            if lam > 1e6:
                break
        else:
            lam = max(lam * 0.5, 1e-9)
        prev = cost
        Wd = np.repeat(wp, 2)
        A = J[:, :6]
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


def pose_from_kpts_pnp_multi(kpts_all, intrin, obj=None,
                             vis_thr: float = 0.30,
                             margin_px: float = 40.0,
                             img_wh=(640, 480),
                             min_pts: int = 6,
                             min_front: int = 4,
                             reproj_thr: float = 6.0,
                             rms_max: float = 8.0,
                             use_weights: bool = True,
                             fallback4: bool = True):
    """보이는 모든 키포인트로 6D pose 를 푼다 (SQPnP + 가중 LM + 이상치 배제).

    kpts_all : (9,3) [x, y, vis] 또는 (9,2). Perception.infer_front 의 kpts_all.
    intrin   : rs intrinsics (fx, fy, ppx, ppy, coeffs)
    min_front: 전면 4점 중 최소 몇 개가 유효해야 하는가.
        좌우 절단 프레임에서 kp0/kp3 이 x=0 에 클램프되고 vis 가 0.001 로
        떨어진다.  min_front=4 (기본) 면 그런 프레임을 버린다 - 회전구간
        |p| 보존이 rel_sd 0.0075 -> 0.0028 로 좋아진다(로그 5건 검증).
        커버리지를 우선하려면 2 로 낮춰 그 점만 제외한다.
    fallback4: 실패 시 기존 4점 IPPE 경로로 폴백한다.

    return: (ok, yaw, pitch, roll, center3d(3,), rvec, tvec, info)
        center3d = 전면 4모서리 중심의 3D 좌표 (기존 4점 경로의 tvec 과 같은 점)
        info = dict(n_used, rms, inliers(9,), centroid(3,), src)
            src 는 "pnp9", 폴백 시 "pnp4"
    """
    info = dict(n_used=0, rms=None, inliers=None, centroid=None, src=None)
    fail = (False, None, None, None, None, None, None, info)

    def _fallback():
        if not fallback4 or kpts_all is None:
            return fail
        try:
            k4 = np.asarray(kpts_all, dtype=np.float64)[:4, :2]
        except Exception:
            return fail
        ok4, y4, p4, r4, c4, rv4, tv4 = pose_from_kpts_pnp(k4, intrin)
        if not ok4:
            return fail
        i2 = dict(info)
        i2["src"] = "pnp4"
        i2["n_used"] = 4
        return (True, y4, p4, r4, c4, rv4, tv4, i2)

    if kpts_all is None or intrin is None:
        return _fallback()
    kp = np.asarray(kpts_all, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[0] < 9:
        return _fallback()

    if obj is None:
        obj = pallet_object_points()
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)[:9]
    K = np.array([[intrin.fx, 0.0, intrin.ppx],
                  [0.0, intrin.fy, intrin.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray(getattr(intrin, "coeffs", [0, 0, 0, 0, 0]),
                      dtype=np.float64).reshape(-1, 1)

    uv = kp[:9, :2]
    vis = kp[:9, 2] if kp.shape[1] >= 3 else np.ones(9)
    W, H = float(img_wh[0]), float(img_wh[1])
    ok_m = np.isfinite(uv).all(axis=1) & (vis >= vis_thr)
    ok_m &= (uv[:, 0] > -margin_px) & (uv[:, 0] < W + margin_px)
    ok_m &= (uv[:, 1] > -margin_px) & (uv[:, 1] < H + margin_px)
    if int(ok_m.sum()) < min_pts or int(ok_m[:4].sum()) < min_front:
        return _fallback()

    sig = PALLET_KPT_SIGMA_PX if use_weights else np.ones(9)
    wall = 1.0 / (sig ** 2)
    wall = wall / wall.mean()

    idx = np.where(ok_m)[0]
    rvec = tvec = None
    for _ in range(3):
        o, i2 = obj[idx], uv[idx]
        try:
            ok, rvec, tvec = cv2.solvePnP(o, i2, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return _fallback()
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            return _fallback()
        rvec, tvec = _refine_weighted(o, i2, K, dist, rvec, tvec, wall[idx])
        if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
            return _fallback()
        d = _reproj_err(o, i2, K, dist, rvec, tvec)
        bad = d > reproj_thr
        keep = ~bad | (idx < 4)      # 전면 4점은 원점/각도 기준이라 유지
        if bad.sum() == 0 or int(keep.sum()) < min_pts:
            break
        new_idx = idx[keep]
        if len(new_idx) == len(idx):
            break
        idx = new_idx

    o, i2 = obj[idx], uv[idx]
    d = _reproj_err(o, i2, K, dist, rvec, tvec)
    rms = float(np.sqrt(np.mean(d ** 2)))
    if not np.isfinite(rms) or rms > rms_max:
        return _fallback()

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    center = (R @ obj[:4].mean(axis=0) + t)
    centroid = (R @ obj[:8].mean(axis=0) + t)
    if not np.all(np.isfinite(center)) or float(center[2]) <= 0.0:
        return _fallback()
    yaw, pitch, roll = _angles_from_R(R)
    if not (np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(roll)):
        return _fallback()

    inl = np.zeros(9, dtype=bool)
    inl[idx] = True
    info = dict(n_used=int(len(idx)), rms=rms, inliers=inl,
                centroid=centroid.astype(np.float32), src="pnp9")
    return (True, yaw, pitch, roll, center.astype(np.float32), rvec, tvec, info)
