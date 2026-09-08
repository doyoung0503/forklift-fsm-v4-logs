"""Shared camera annotations for the live launcher and synthetic previews.

No camera, perception, PnP or control execution occurs here.
"""
import cv2
import numpy as np
from .config import COLOR_CNT, COLOR_BOX, COLOR_CENTER, POSE_CENTER_KPT


def draw_camera_overlay(vis, *, color_intrin, kpts_all=None,
                        selected_face_corners_px=None, selected_front_face=None,
                        rvec=None, tvec=None, yaw_deg=None, pitch_deg=None,
                        roll_deg=None, vision_independent=False):
    """Draw in place. Blind/timer-only runtime stages keep the RGB unannotated."""
    if vision_independent:
        return vis
    H, W = vis.shape[:2]
    if kpts_all is not None and len(kpts_all) >= 8:
        front = np.round(kpts_all[0:4, :2]).astype(np.int32)
        back  = np.round(kpts_all[4:8, :2]).astype(np.int32)
        cv2.polylines(vis, [front], isClosed=True, color=COLOR_CNT, thickness=2)
        cv2.polylines(vis, [back], isClosed=True, color=COLOR_BOX, thickness=1)
        for i in range(8):
            kx, ky = int(round(kpts_all[i, 0])), int(round(kpts_all[i, 1]))
            col = COLOR_CNT if i < 4 else COLOR_BOX
            cv2.circle(vis, (kx, ky), 4, col, -1)
            cv2.putText(vis, str(i), (kx + 5, ky - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if len(kpts_all) > POSE_CENTER_KPT:
            ck = kpts_all[POSE_CENTER_KPT]
            if np.all(np.isfinite(ck[:2])):
                kx, ky = int(round(ck[0])), int(round(ck[1]))
                cv2.drawMarker(vis, (kx, ky), COLOR_CENTER,
                               markerType=cv2.MARKER_CROSS,
                               markerSize=16, thickness=2)
                cv2.putText(vis, str(POSE_CENTER_KPT), (kx + 7, ky - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CENTER, 2)
    if selected_face_corners_px is not None:
        try:
            selected_poly = np.round(
                np.asarray(selected_face_corners_px, dtype=np.float64)
            ).astype(np.int32)
            if selected_poly.shape == (4, 2):
                cv2.polylines(
                    vis, [selected_poly], isClosed=True,
                    color=(255, 0, 255), thickness=3,
                )
                label_at = tuple(selected_poly[0])
                cv2.putText(
                    vis, f"FRONT(area): {selected_front_face}",
                    (int(label_at[0]) + 5, int(label_at[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                )
        except Exception:
            pass
    # 자세를 눈으로 보이게 — 파렛트 중심에 3D 좌표축을 투영해 그린다
    if rvec is not None:
        K = np.array([[color_intrin.fx, 0.0, color_intrin.ppx],
                      [0.0, color_intrin.fy, color_intrin.ppy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.asarray(color_intrin.coeffs, dtype=np.float64).reshape(-1, 1)
        L = 0.35
        # 표시 전용 축 방향 — 모델 좌표는 y 아래 / z 화면안쪽이라 그대로 그리면
        # Y·Z 가 뒤로 들어가 보인다. 보기 좋게 Y 는 위, Z 는 카메라 쪽으로 뒤집어 그린다.
        # (각도 계산 규약은 건드리지 않는다 — FSM 이 쓰는 값이다)
        axis_pts = np.array([[0, 0, 0], [L, 0, 0], [0, -L, 0], [0, 0, -L]], dtype=np.float64)
        proj, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        if np.all(np.isfinite(proj)):
            o = tuple(np.round(proj[0]).astype(int))
            for idx, (col, lab) in enumerate(
                    [((0, 0, 255), "X"), ((0, 255, 0), "Y"), ((255, 128, 0), "Z")], start=1):
                pt = tuple(np.round(proj[idx]).astype(int))
                cv2.arrowedLine(vis, o, pt, col, 2, tipLength=0.2)
                cv2.putText(vis, lab, (pt[0] + 4, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            # 각도 값을 파렛트 옆에 같이 띄운다
            tx, ty = o[0] + 12, o[1] + 18
            for j, (txt, col) in enumerate([
                    (f"yaw   {yaw_deg:+6.1f}", (200, 100, 255)),
                    (f"pitch {pitch_deg:+6.1f}", (0, 220, 255)),
                    (f"roll  {roll_deg:+6.1f}", (255, 200, 0))]):
                cv2.putText(vis, txt, (tx, ty + j * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    cv2.drawMarker(vis, (W // 2, H // 2), COLOR_CENTER, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    return vis
