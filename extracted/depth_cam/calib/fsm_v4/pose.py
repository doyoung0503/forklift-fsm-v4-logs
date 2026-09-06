"""Timestamped PnP filtering and pallet-frame transforms for FSM v4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from . import config as cfg


def wrap_180(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def rotation_center_in_pallet(
    yaw_deg: float, pallet_x_m: float, pallet_z_m: float,
) -> Tuple[float, float]:
    """Return forklift rotation centre in the fixed pallet coordinate frame."""

    yaw = math.radians(float(yaw_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    dx = cfg.CAMERA_TO_ROT_CENTER_X_M - float(pallet_x_m)
    dz = cfg.CAMERA_TO_ROT_CENTER_Z_M - float(pallet_z_m)
    return c * dx - s * dz, s * dx + c * dz


def camera_pose_to_vehicle(
    yaw_camera_deg: float, x_camera_m: float, z_camera_m: float,
) -> Tuple[float, float, float]:
    """Rotate camera-frame planar PnP pose into vehicle-forward axes."""

    delta = math.radians(cfg.CAMERA_YAW_IN_VEHICLE_DEG)
    c, s = math.cos(delta), math.sin(delta)
    x_vehicle = c * float(x_camera_m) + s * float(z_camera_m)
    z_vehicle = -s * float(x_camera_m) + c * float(z_camera_m)
    yaw_vehicle = wrap_180(float(yaw_camera_deg) + cfg.CAMERA_YAW_IN_VEHICLE_DEG)
    return yaw_vehicle, x_vehicle, z_vehicle


@dataclass(frozen=True)
class VisualPose:
    yaw_deg: float
    pallet_x_m: float
    pallet_z_m: float
    rot_x_pallet_m: float
    rot_z_pallet_m: float
    camera_vx_m_s: float
    camera_vz_m_s: float
    rot_vx_m_s: float
    rot_vz_m_s: float
    yaw_rate_deg_s: float
    measurement_mono: float

    @property
    def rotation_center_speed_m_s(self) -> float:
        return math.hypot(self.rot_vx_m_s, self.rot_vz_m_s)


class VisualPoseFilter:
    def __init__(self) -> None:
        self.pose: Optional[VisualPose] = None

    def reset(self) -> None:
        self.pose = None

    def seed(
        self, yaw_deg: float, x_m: float, z_m: float, measurement_mono: float,
    ) -> VisualPose:
        yaw_deg, x_m, z_m = camera_pose_to_vehicle(yaw_deg, x_m, z_m)
        return self.seed_vehicle(yaw_deg, x_m, z_m, measurement_mono)

    def seed_vehicle(
        self, yaw_deg: float, x_m: float, z_m: float, measurement_mono: float,
    ) -> VisualPose:
        """Seed from values already expressed in vehicle axes."""

        rx, rz = rotation_center_in_pallet(yaw_deg, x_m, z_m)
        self.pose = VisualPose(
            wrap_180(yaw_deg), float(x_m), float(z_m), rx, rz,
            0.0, 0.0, 0.0, 0.0, 0.0, float(measurement_mono),
        )
        return self.pose

    def update(
        self, yaw_deg: float, x_m: float, z_m: float, measurement_mono: float,
    ) -> Tuple[Optional[VisualPose], str]:
        values = (yaw_deg, x_m, z_m, measurement_mono)
        if not all(math.isfinite(float(value)) for value in values):
            return None, "non-finite PnP input"
        if float(z_m) <= 0.0:
            return None, "pallet behind camera"
        if self.pose is None:
            return self.seed(yaw_deg, x_m, z_m, measurement_mono), "seed"

        yaw_deg, x_m, z_m = camera_pose_to_vehicle(yaw_deg, x_m, z_m)
        old = self.pose
        dt = float(measurement_mono) - old.measurement_mono
        if dt <= 0.0:
            return old, "duplicate/out-of-order timestamp"
        dt = min(1.0, dt)
        raw_rx, raw_rz = rotation_center_in_pallet(yaw_deg, x_m, z_m)
        translation = math.hypot(
            raw_rx - old.rot_x_pallet_m, raw_rz - old.rot_z_pallet_m,
        )
        yaw_delta = abs(wrap_180(float(yaw_deg) - old.yaw_deg))
        max_translation = (
            cfg.POSE_GATE_POSITION_BASE_M
            + cfg.POSE_MAX_TRANSLATION_RATE_M_S * dt
        )
        max_yaw = cfg.POSE_GATE_YAW_BASE_DEG + cfg.POSE_MAX_YAW_RATE_DEG_S * dt
        if translation > max_translation:
            return None, f"translation innovation {translation:.3f}>{max_translation:.3f}m"
        if yaw_delta > max_yaw:
            return None, f"yaw innovation {yaw_delta:.2f}>{max_yaw:.2f}deg"

        # Accepted PnP measurements pass through unchanged. Robustness is
        # provided by the innovation gates above, not temporal EMA smoothing.
        yaw = wrap_180(float(yaw_deg))
        x = float(x_m)
        z = float(z_m)
        rx, rz = rotation_center_in_pallet(yaw, x, z)
        camera_vx_raw = (x - old.pallet_x_m) / dt
        camera_vz_raw = (z - old.pallet_z_m) / dt
        rot_vx_raw = (rx - old.rot_x_pallet_m) / dt
        rot_vz_raw = (rz - old.rot_z_pallet_m) / dt
        yaw_rate_raw = wrap_180(yaw - old.yaw_deg) / dt
        camera_vx = camera_vx_raw
        camera_vz = camera_vz_raw
        rot_vx = rot_vx_raw
        rot_vz = rot_vz_raw
        yaw_rate = yaw_rate_raw
        speed = math.hypot(rot_vx, rot_vz)
        if speed > cfg.POSE_MAX_TRANSLATION_RATE_M_S:
            scale = cfg.POSE_MAX_TRANSLATION_RATE_M_S / max(speed, 1e-9)
            rot_vx *= scale
            rot_vz *= scale
        yaw_rate = max(
            -cfg.POSE_MAX_YAW_RATE_DEG_S,
            min(cfg.POSE_MAX_YAW_RATE_DEG_S, yaw_rate),
        )
        self.pose = VisualPose(
            yaw, x, z, rx, rz, camera_vx, camera_vz, rot_vx, rot_vz,
            yaw_rate, float(measurement_mono),
        )
        return self.pose, "accepted"
