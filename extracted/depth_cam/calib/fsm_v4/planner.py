"""Visibility-constrained pallet-frame waypoint planner for FSM v4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import config as cfg
from .pose import VisualPose, wrap_180


@dataclass(frozen=True)
class Waypoint:
    turn_deg: float
    forward_m: float
    goal_x_m: float
    goal_z_m: float
    goal_distance_m: float
    predicted_center_bearing_deg: float
    visibility_lower_deg: float
    visibility_upper_deg: float
    visibility_min_margin_deg: float


@dataclass(frozen=True)
class PlanResult:
    waypoint: Optional[Waypoint]
    reason: str
    needs_recenter: bool = False
    recenter_turn_deg: Optional[float] = None


def configured_visible_half_angle_deg() -> float:
    """Usable half-FOV after reserving the configured image-edge margin."""

    half_fov = math.radians(cfg.CAMERA_HORIZONTAL_FOV_DEG * 0.5)
    usable_half_width = max(0.0, 1.0 - 2.0 * cfg.IMAGE_EDGE_MARGIN_NORM)
    return math.degrees(math.atan(usable_half_width * math.tan(half_fov)))


def _front_edge_points(pose: VisualPose) -> Tuple[Tuple[float, float], ...]:
    """Physical left/right front-face edges in vehicle-aligned camera axes."""

    yaw = math.radians(pose.yaw_deg)
    half_width = cfg.PALLET_FRONT_VISIBILITY_WIDTH_M * 0.5
    dx = half_width * math.cos(yaw)
    dz = -half_width * math.sin(yaw)
    return (
        (pose.pallet_x_m - dx, pose.pallet_z_m - dz),
        (pose.pallet_x_m + dx, pose.pallet_z_m + dz),
    )


def _point_after_action(
    point: Tuple[float, float], turn_deg: float, forward_m: float,
) -> Tuple[float, float]:
    """Transform a fixed world point after vehicle turn then forward motion."""

    turn = math.radians(float(turn_deg))
    c, s = math.cos(turn), math.sin(turn)
    dx = float(point[0]) - cfg.CAMERA_TO_ROT_CENTER_X_M
    dz = float(point[1]) - cfg.CAMERA_TO_ROT_CENTER_Z_M
    return (
        c * dx - s * dz + cfg.CAMERA_TO_ROT_CENTER_X_M,
        s * dx + c * dz + cfg.CAMERA_TO_ROT_CENTER_Z_M - float(forward_m),
    )


def _action_samples(turn_deg: float, forward_m: float):
    turn_steps = max(
        1,
        int(math.ceil(abs(float(turn_deg)) / cfg.VISIBILITY_PATH_SAMPLE_STEP_DEG)),
    )
    forward_steps = max(
        1,
        int(math.ceil(abs(float(forward_m)) / 0.05)),
    )
    for index in range(turn_steps + 1):
        yield float(turn_deg) * index / turn_steps, 0.0
    for index in range(1, forward_steps + 1):
        yield float(turn_deg), float(forward_m) * index / forward_steps


def action_visibility_margin_deg(
    pose: VisualPose, turn_deg: float, forward_m: float,
    half_angle_deg: Optional[float] = None,
) -> float:
    """Minimum horizontal FOV margin along a turn-then-forward action.

    The rotation is sampled because an offset camera follows an arc around the
    vehicle rotation centre.  Straight motion is sampled as well so this
    remains correct if the constraint is extended beyond the current pinhole
    horizontal model.
    """

    limit = (
        configured_visible_half_angle_deg()
        if half_angle_deg is None else float(half_angle_deg)
    )
    margin = float("inf")
    for sample_turn, sample_forward in _action_samples(turn_deg, forward_m):
        for point in _front_edge_points(pose):
            x_m, z_m = _point_after_action(point, sample_turn, sample_forward)
            if z_m <= cfg.VISIBILITY_MIN_CORNER_DEPTH_M:
                return float("-inf")
            bearing = math.degrees(math.atan2(x_m, z_m))
            margin = min(margin, limit - abs(bearing))
    return margin


def _pixel_visibility_axis_margins_norm(
    turn_deg: float, forward_m: float, vision_meta: Optional[Dict],
    edge_margin_norm: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Horizontal/vertical clearances; None when 3D metadata is unavailable."""

    if not vision_meta:
        return None
    try:
        corners = [tuple(map(float, point[:3])) for point in
                   vision_meta["face_corners_camera_m"]]
        if len(corners) != 4:
            return None
        fx = float(vision_meta["fx"])
        fy = float(vision_meta["fy"])
        ppx = float(vision_meta["ppx"])
        ppy = float(vision_meta["ppy"])
        width = float(vision_meta["frame_width"])
        height = float(vision_meta["frame_height"])
        if min(fx, fy, width, height) <= 0.0:
            return None
    except (KeyError, TypeError, ValueError, IndexError):
        return None

    camera_yaw = math.radians(cfg.CAMERA_YAW_IN_VEHICLE_DEG)
    c_cam, s_cam = math.cos(camera_yaw), math.sin(camera_yaw)
    edge_margin = (
        cfg.IMAGE_EDGE_MARGIN_NORM
        if edge_margin_norm is None else float(edge_margin_norm)
    )
    margin_x = edge_margin * width
    margin_y = edge_margin * height
    horizontal = float("inf")
    vertical = float("inf")
    for sample_turn, sample_forward in _action_samples(turn_deg, forward_m):
        for x_camera, y_camera, z_camera in corners:
            # Same camera->vehicle planar convention as camera_pose_to_vehicle.
            x_vehicle = c_cam * x_camera + s_cam * z_camera
            z_vehicle = -s_cam * x_camera + c_cam * z_camera
            x_vehicle, z_vehicle = _point_after_action(
                (x_vehicle, z_vehicle), sample_turn, sample_forward,
            )
            # Vehicle->camera inverse; yaw motion does not change camera Y.
            x_camera = c_cam * x_vehicle - s_cam * z_vehicle
            z_camera = s_cam * x_vehicle + c_cam * z_vehicle
            if z_camera <= cfg.VISIBILITY_MIN_CORNER_DEPTH_M:
                return float("-inf"), float("-inf")
            u = fx * x_camera / z_camera + ppx
            v = fy * y_camera / z_camera + ppy
            horizontal = min(
                horizontal,
                (u - margin_x) / width,
                (width - margin_x - u) / width,
            )
            vertical = min(
                vertical,
                (v - margin_y) / height,
                (height - margin_y - v) / height,
            )
    return horizontal, vertical


def _pixel_visibility_margin_norm(
    turn_deg: float, forward_m: float, vision_meta: Optional[Dict],
    edge_margin_norm: Optional[float] = None,
) -> Optional[float]:
    margins = _pixel_visibility_axis_margins_norm(
        turn_deg, forward_m, vision_meta, edge_margin_norm,
    )
    return None if margins is None else min(margins)


def _intrinsic_horizontal_margin_norm(
    pose: VisualPose, turn_deg: float, forward_m: float,
    vision_meta: Optional[Dict], edge_margin_norm: Optional[float] = None,
) -> Optional[float]:
    """Physical-edge horizontal clearance using stream intrinsics only."""

    if not vision_meta:
        return None
    try:
        fx = float(vision_meta["fx"])
        ppx = float(vision_meta["ppx"])
        width = float(vision_meta["frame_width"])
        if min(fx, width) <= 0.0:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    edge_margin = (
        cfg.IMAGE_EDGE_MARGIN_NORM
        if edge_margin_norm is None else float(edge_margin_norm)
    )
    left = edge_margin * width
    right = width - left
    camera_yaw = math.radians(cfg.CAMERA_YAW_IN_VEHICLE_DEG)
    c_cam, s_cam = math.cos(camera_yaw), math.sin(camera_yaw)
    clearance = float("inf")
    for sample_turn, sample_forward in _action_samples(turn_deg, forward_m):
        for point in _front_edge_points(pose):
            x_vehicle, z_vehicle = _point_after_action(
                point, sample_turn, sample_forward,
            )
            x_camera = c_cam * x_vehicle - s_cam * z_vehicle
            z_camera = s_cam * x_vehicle + c_cam * z_vehicle
            if z_camera <= cfg.VISIBILITY_MIN_CORNER_DEPTH_M:
                return float("-inf")
            u = fx * x_camera / z_camera + ppx
            clearance = min(
                clearance, (u - left) / width, (right - u) / width,
            )
    return clearance


def action_keeps_front_visible(
    pose: VisualPose, turn_deg: float, forward_m: float,
    vision_meta: Optional[Dict] = None,
    half_angle_deg: Optional[float] = None,
    edge_margin_norm: Optional[float] = None,
) -> bool:
    """True only when the physical front face stays inside the safe viewport."""

    if action_visibility_margin_deg(
        pose, turn_deg, forward_m, half_angle_deg=half_angle_deg,
    ) < 0.0:
        return False
    pixel_margin = _pixel_visibility_margin_norm(
        turn_deg, forward_m, vision_meta, edge_margin_norm,
    )
    if pixel_margin is not None:
        return pixel_margin >= 0.0
    intrinsic_margin = _intrinsic_horizontal_margin_norm(
        pose, turn_deg, forward_m, vision_meta, edge_margin_norm,
    )
    return intrinsic_margin is None or intrinsic_margin >= 0.0


def _maximum_visible_forward_m(
    pose: VisualPose, turn_deg: float, upper_m: float,
    vision_meta: Optional[Dict] = None,
) -> Tuple[float, float]:
    """Largest safe forward distance and its limiting angular margin."""

    upper = max(0.0, float(upper_m))
    rotation_margin = action_visibility_margin_deg(pose, turn_deg, 0.0)
    if rotation_margin < 0.0 or not action_keeps_front_visible(
        pose, turn_deg, 0.0, vision_meta,
    ):
        return 0.0, rotation_margin
    endpoint_margin = action_visibility_margin_deg(pose, turn_deg, upper)
    if endpoint_margin >= 0.0 and action_keeps_front_visible(
        pose, turn_deg, upper, vision_meta,
    ):
        return upper, endpoint_margin
    low, high = 0.0, upper
    for _ in range(cfg.VISIBILITY_FORWARD_SEARCH_ITERATIONS):
        middle = 0.5 * (low + high)
        if action_keeps_front_visible(pose, turn_deg, middle, vision_meta):
            low = middle
        else:
            high = middle
    return low, action_visibility_margin_deg(pose, turn_deg, low)


def _pose_after_action(
    pose: VisualPose, turn_deg: float, forward_m: float,
) -> VisualPose:
    """Noise-free pose used only for candidate endpoint constraints."""

    center = _point_after_action(
        (pose.pallet_x_m, pose.pallet_z_m), turn_deg, forward_m,
    )
    yaw = wrap_180(pose.yaw_deg - float(turn_deg))
    yaw_rad = math.radians(yaw)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    dx = cfg.CAMERA_TO_ROT_CENTER_X_M - center[0]
    dz = cfg.CAMERA_TO_ROT_CENTER_Z_M - center[1]
    rot_x = c * dx - s * dz
    rot_z = s * dx + c * dz
    return VisualPose(
        yaw, center[0], center[1], rot_x, rot_z,
        0.0, 0.0, 0.0, 0.0, 0.0, pose.measurement_mono,
    )


def _vision_meta_after_turn(vision_meta: Optional[Dict], turn_deg: float) -> Optional[Dict]:
    """Project measured camera-frame corners at the planned turn endpoint."""
    if not vision_meta or "face_corners_camera_m" not in vision_meta:
        return vision_meta
    result = dict(vision_meta)
    angle = math.radians(cfg.CAMERA_YAW_IN_VEHICLE_DEG)
    c, s = math.cos(angle), math.sin(angle)
    corners = []
    for point in vision_meta["face_corners_camera_m"]:
        x, y, z = map(float, point[:3])
        vx, vz = _point_after_action((c * x + s * z, -s * x + c * z), turn_deg, 0.0)
        corners.append((c * vx - s * vz, y, s * vx + c * vz))
    result["face_corners_camera_m"] = corners
    return result


def _maximum_staging_safe_forward_m(
    pose: VisualPose, turn_deg: float, upper_m: float,
) -> float:
    """Reserve manoeuvring room until lateral staging error is acceptable."""

    upper = max(0.0, float(upper_m))

    def acceptable(distance_m: float) -> bool:
        predicted = _pose_after_action(pose, turn_deg, distance_m)
        lateral, longitudinal = staging_position_errors(predicted)
        return (
            abs(lateral) <= cfg.FINAL_LATERAL_TOL_M
            or longitudinal <= -cfg.FINAL_DISTANCE_TOL_M
        )

    if acceptable(upper):
        return upper
    low, high = 0.0, upper
    for _ in range(cfg.VISIBILITY_FORWARD_SEARCH_ITERATIONS):
        middle = 0.5 * (low + high)
        if acceptable(middle):
            low = middle
        else:
            high = middle
    return low


def safe_straight_continuation_m(
    pose: VisualPose, requested_m: float, remaining_m: float,
    vision_meta: Optional[Dict] = None,
) -> float:
    """Revalidate the planned drive at the actual settled, not ideal, yaw."""
    gx, gz, goal_distance = goal_vector_vehicle(pose)
    _lateral, longitudinal = staging_position_errors(pose)
    if longitudinal >= 0.0:
        return 0.0
    upper = max(0.0, min(requested_m, remaining_m,
                         cfg.FWD_MACRO_MAX_DISTANCE_M, gz))
    visible, _margin = _maximum_visible_forward_m(pose, 0.0, upper, vision_meta)
    distance = min(visible, _maximum_staging_safe_forward_m(pose, 0.0, upper))
    if distance < cfg.FWD_RELIABLE_MIN_DISTANCE_M:
        return 0.0
    if math.hypot(gx, gz - distance) >= goal_distance:
        return 0.0
    return distance


def _candidate_turns(raw_turn_deg: float) -> Tuple[float, ...]:
    """Commandable turn candidates, including the exact desired direction."""

    maximum = cfg.ROT_MAX_WAYPOINT_TURN_DEG
    step = cfg.VISIBILITY_TURN_SEARCH_STEP_DEG
    minimum = cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG
    values = {0.0, max(-maximum, min(maximum, float(raw_turn_deg)))}
    count = int(math.floor(maximum / step))
    for index in range(1, count + 1):
        angle = index * step
        if angle + 1e-9 >= minimum:
            values.add(min(maximum, angle))
            values.add(max(-maximum, -angle))
    # A sub-minimum turn is not executed by the FSM; model it as zero instead.
    values = {
        0.0 if abs(value) < minimum else round(float(value), 9)
        for value in values
    }
    return tuple(sorted(values))


def safe_recenter_turn(
    pose: VisualPose, center_bearing_deg: float,
    vision_meta: Optional[Dict] = None,
) -> Optional[float]:
    """Largest physically-visible partial turn toward the optical axis."""

    desired = max(
        -cfg.ROT_MAX_WAYPOINT_TURN_DEG,
        min(cfg.ROT_MAX_WAYPOINT_TURN_DEG, float(center_bearing_deg)),
    )
    if abs(desired) < cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG:
        return None
    physical_half_fov = cfg.CAMERA_HORIZONTAL_FOV_DEG * 0.5
    candidates = [
        turn for turn in _candidate_turns(desired)
        if turn * desired > 0.0
    ]
    current_abs = abs(float(center_bearing_deg))
    ranked = []
    for turn in candidates:
        if action_keeps_front_visible(
            pose, turn, 0.0, vision_meta,
            half_angle_deg=physical_half_fov,
            edge_margin_norm=0.0,
        ):
            x_m, z_m = _point_after_action(
                (pose.pallet_x_m, pose.pallet_z_m), turn, 0.0,
            )
            predicted_abs = abs(math.degrees(math.atan2(x_m, z_m)))
            ranked.append((predicted_abs, abs(turn), turn))
    if not ranked:
        return None
    predicted_abs, _turn_size, best = min(ranked)
    return best if predicted_abs + 0.1 < current_abs else None


def goal_vector_vehicle(pose: VisualPose) -> Tuple[float, float, float]:
    """Vector from vehicle rotation centre to desired staging position."""

    yaw = math.radians(pose.yaw_deg)
    gx = (
        pose.pallet_x_m
        - cfg.STAGING_DISTANCE_M * math.sin(yaw)
        - cfg.CAMERA_TO_ROT_CENTER_X_M
    )
    gz = (
        pose.pallet_z_m
        - cfg.STAGING_DISTANCE_M * math.cos(yaw)
        - cfg.CAMERA_TO_ROT_CENTER_Z_M
    )
    return gx, gz, math.hypot(gx, gz)


def staging_position_errors(pose: VisualPose) -> Tuple[float, float]:
    """Return pallet-frame lateral and longitudinal staging errors.

    Longitudinal error is positive after the rotation centre has crossed the
    desired staging plane (overtravel), and negative while still approaching.
    """

    return (
        float(pose.rot_x_pallet_m),
        float(pose.rot_z_pallet_m + cfg.STAGING_DISTANCE_M),
    )


def staging_position_reached(pose: VisualPose) -> bool:
    """Legacy position target for planning, separate from insertion eligibility."""
    lateral_error, longitudinal_error = staging_position_errors(pose)
    return (abs(lateral_error) <= cfg.FINAL_LATERAL_TOL_M
            and abs(longitudinal_error) <= cfg.FINAL_DISTANCE_TOL_M)


def fork_opening_alignment(pose: VisualPose, *, enforce_entry_distance: bool = True):
    """Plan-view ray/face intersection; NOT a 3D collision clearance check.

    Rays start at the outer fork tips and travel along vehicle +Z. Face local
    X is across the pallet; face normal is (sin(yaw), cos(yaw)).
    """
    # Do not evaluate insertion alignment outside the camera-Z activation range.
    if (not math.isfinite(pose.pallet_z_m) or pose.pallet_z_m <= 0.0
            or (enforce_entry_distance
                and pose.pallet_z_m > cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M)):
        return False, ()
    yaw = math.radians(pose.yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    if not all(math.isfinite(v) for v in (c, s, pose.pallet_x_m, pose.pallet_z_m)) or c <= 1e-6:
        return False, ()
    hits = []
    ahead = True
    for offset in (-cfg.FORK_OUTER_SPAN_M/2, cfg.FORK_OUTER_SPAN_M/2):
        dx = cfg.CAMERA_TO_FORK_TIP_X_M + offset - pose.pallet_x_m
        z_hit = pose.pallet_z_m - s * dx / c
        ahead &= z_hit >= cfg.CAMERA_TO_FORK_TIP_Z_M
        hits.append(dx / c)
    half = cfg.INSERT_OPENING_SPAN_M / 2
    return bool(ahead and all(abs(x) <= half + 1e-9 for x in hits)), tuple(hits)


def insertion_alignment_turn(pose: VisualPose, vision_meta: Optional[Dict] = None,
                             *, alignment_entered: bool = False):
    """Smallest commandable in-place turn yielding valid fork rays and yaw.

    Uses the measured rotation-centre offset, not a yaw-only rotation about
    the camera. Return zero if already aligned; None if no safe candidate.
    """
    if (not math.isfinite(pose.pallet_z_m) or pose.pallet_z_m <= 0.0
            or (not alignment_entered
                and pose.pallet_z_m > cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M)):
        return None
    if (fork_opening_alignment(pose, enforce_entry_distance=False)[0]
            and abs(pose.yaw_deg) <= cfg.FINAL_YAW_TOL_DEG):
        return 0.0
    # The caller supplies a currently detected, valid PnP pose. If that
    # observation is already outside the conservative viewport, allow an
    # insertion alignment turn instead of rejecting every path at t=0.
    # Runtime vision-loss handling still applies while the turn executes.
    start_outside_safe_view = not action_keeps_front_visible(
        pose, 0.0, 0.0, vision_meta,
    )
    candidates = []
    turns = set(_candidate_turns(pose.yaw_deg))
    step = cfg.INSERT_FINE_MIN_TURN_DEG
    for i in range(1, int(math.ceil(cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG / step))):
        turns.update((i * step, -i * step))
    for turn in turns:
        if abs(turn) < cfg.INSERT_FINE_MIN_TURN_DEG:
            continue
        predicted = _pose_after_action(pose, turn, 0.0)
        if abs(predicted.yaw_deg) > cfg.FINAL_YAW_TOL_DEG:
            continue
        # Entry distance is checked above; camera Z may grow during rotation.
        fits, hits = fork_opening_alignment(predicted, enforce_entry_distance=False)
        if not fits:
            continue
        if not start_outside_safe_view and not action_keeps_front_visible(
            pose, turn, 0.0, vision_meta,
        ):
            continue
        clearance = min(cfg.INSERT_OPENING_SPAN_M/2 - abs(x) for x in hits)
        candidates.append((abs(turn), -clearance, turn))
    return min(candidates)[2] if candidates else None


def visibility_turn_interval(
    center_bearing_deg: float, vision_meta: Optional[Dict],
) -> Tuple[float, float]:
    """Conservative pure-rotation interval that keeps pallet corners visible."""

    lower = -cfg.ROT_MAX_WAYPOINT_TURN_DEG
    upper = cfg.ROT_MAX_WAYPOINT_TURN_DEG
    if vision_meta:  # Planning margins are independent of legacy live-guard flags.
        corners = vision_meta.get("face_corners_px")
        fx = vision_meta.get("fx")
        ppx = vision_meta.get("ppx")
        width = vision_meta.get("frame_width")
        try:
            us = [float(point[0]) for point in corners]
            fx, ppx, width = float(fx), float(ppx), float(width)
            margin_px = cfg.IMAGE_EDGE_MARGIN_NORM * width
            image_left = math.degrees(math.atan2(margin_px - ppx, fx))
            image_right = math.degrees(math.atan2(width - margin_px - ppx, fx))
            pallet_left = math.degrees(math.atan2(min(us) - ppx, fx))
            pallet_right = math.degrees(math.atan2(max(us) - ppx, fx))
            # Positive vehicle turn is ROT_RIGHT and shifts observed bearings left.
            lower = max(lower, pallet_right - image_right)
            upper = min(upper, pallet_left - image_left)
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            pass
    if cfg.PALLET_CENTER_VISIBILITY_GUARD_ENABLED:
        # Centre-bearing fallback/secondary bound.  Never claim a centre range
        # wider than the configured camera can display after margins.
        safe_center = min(
            cfg.PALLET_CENTER_SAFE_BEARING_DEG,
            configured_visible_half_angle_deg(),
        )
        lower = max(lower, center_bearing_deg - safe_center)
        upper = min(upper, center_bearing_deg + safe_center)
    return lower, upper


def plan_waypoint(
    pose: VisualPose, center_bearing_deg: float, bbox_margin_norm: float,
    remaining_forward_m: float, vision_meta: Optional[Dict],
) -> PlanResult:
    gx, gz, goal_distance = goal_vector_vehicle(pose)
    raw_turn = wrap_180(
        math.degrees(math.atan2(gx, gz)) - cfg.FORWARD_PATH_BIAS_DEG
    )
    if staging_position_reached(pose):
        return PlanResult(None, "staging position reached")
    lateral_error, longitudinal_error = staging_position_errors(pose)
    if longitudinal_error > cfg.FINAL_DISTANCE_TOL_M:
        return PlanResult(
            None,
            f"staging overtravel {longitudinal_error:.3f}m exceeds "
            f"{cfg.FINAL_DISTANCE_TOL_M:.3f}m limit",
        )
    if (
        longitudinal_error >= 0.0
        and abs(lateral_error) > cfg.FINAL_LATERAL_TOL_M
    ):
        return PlanResult(
            None,
            f"staging overtravel lateral {lateral_error:+.3f}m exceeds "
            f"{cfg.FINAL_LATERAL_TOL_M:.3f}m limit",
        )
    if remaining_forward_m < cfg.FWD_RELIABLE_MIN_DISTANCE_M:
        return PlanResult(None, "forward correction budget exhausted")

    if cfg.PREDICTIVE_VISIBILITY_PLANNER_ENABLED:
        # A visible start outside the reserved ROI may recover during the
        # planned turn. Never require an independent optical-axis recenter.
        starts_safe = action_keeps_front_visible(pose, 0.0, 0.0, vision_meta)
        candidates = []
        rotation_safe_turns = []
        for turn in _candidate_turns(raw_turn):
            if not action_keeps_front_visible(
                pose, turn, 0.0, vision_meta,
                half_angle_deg=None if starts_safe else cfg.CAMERA_HORIZONTAL_FOV_DEG * 0.5,
                edge_margin_norm=None if starts_safe else 0.0,
            ):
                continue
            turned_pose = _pose_after_action(pose, turn, 0.0)
            turned_meta = _vision_meta_after_turn(vision_meta, turn)
            # Forward motion must start and remain inside the reserved ROI.
            if not action_keeps_front_visible(turned_pose, 0.0, 0.0, turned_meta):
                continue
            rotation_safe_turns.append(turn)
            heading = math.radians(turn)
            goal_x_after_turn = gx * math.cos(heading) - gz * math.sin(heading)
            goal_z_after_turn = gx * math.sin(heading) + gz * math.cos(heading)
            requested_forward = min(
                goal_z_after_turn,
                cfg.FWD_MACRO_MAX_DISTANCE_M,
                remaining_forward_m,
            )
            if requested_forward < cfg.FWD_RELIABLE_MIN_DISTANCE_M:
                continue
            safe_forward, path_margin = _maximum_visible_forward_m(
                turned_pose, 0.0, requested_forward, turned_meta,
            )
            safe_forward = min(
                safe_forward,
                _maximum_staging_safe_forward_m(
                    pose, turn, requested_forward,
                ),
            )
            path_margin = action_visibility_margin_deg(
                turned_pose, 0.0, safe_forward,
            )
            if safe_forward + 1e-6 < cfg.FWD_RELIABLE_MIN_DISTANCE_M:
                continue
            residual_goal = math.hypot(
                goal_x_after_turn, goal_z_after_turn - safe_forward,
            )
            progress = goal_distance - residual_goal
            if progress <= 0.0:
                continue
            # Maximise staging progress first.  For effectively equal progress,
            # prefer the heading nearest the direct goal and then less turning.
            score = (
                round(progress, 6),
                round(safe_forward, 6),
                -abs(wrap_180(raw_turn - turn)),
                -abs(turn),
            )
            candidates.append((score, turn, safe_forward, path_margin))

        if not candidates:
            return PlanResult(
                None, "no rotation+forward action restores safe horizontal/vertical visibility",
            )

        _score, turn, forward, path_margin = max(
            candidates, key=lambda item: item[0],
        )
        center_after = _point_after_action(
            (pose.pallet_x_m, pose.pallet_z_m), turn, forward,
        )
        predicted_center = math.degrees(math.atan2(*center_after))
        lower = min(rotation_safe_turns) if rotation_safe_turns else turn
        upper = max(rotation_safe_turns) if rotation_safe_turns else turn
        return PlanResult(
            Waypoint(
                turn_deg=turn,
                forward_m=forward,
                goal_x_m=gx,
                goal_z_m=gz,
                goal_distance_m=goal_distance,
                predicted_center_bearing_deg=predicted_center,
                visibility_lower_deg=lower,
                visibility_upper_deg=upper,
                visibility_min_margin_deg=path_margin,
            ),
            "ok: predictive full-front visibility",
        )

    # Legacy approximation: compensate path bias, then shift the currently
    # observed corner bearings as though the camera rotated about its optical
    # centre.  Kept only as an explicit fallback for field comparison.
    lower, upper = visibility_turn_interval(center_bearing_deg, vision_meta)
    if lower > upper:
        return PlanResult(None, "no visibility-safe turn")
    turn = min(upper, max(lower, raw_turn))
    predicted_center = center_bearing_deg - turn

    heading = math.radians(turn)
    projected = gx * math.sin(heading) + gz * math.cos(heading)
    forward = min(projected, cfg.FWD_MACRO_MAX_DISTANCE_M, remaining_forward_m)
    if forward < cfg.FWD_RELIABLE_MIN_DISTANCE_M:
        return PlanResult(
            None,
            f"required forward step {forward:.3f}m below reliable minimum "
            f"{cfg.FWD_RELIABLE_MIN_DISTANCE_M:.3f}m",
        )
    return PlanResult(
        Waypoint(
            turn_deg=turn,
            forward_m=forward,
            goal_x_m=gx,
            goal_z_m=gz,
            goal_distance_m=goal_distance,
            predicted_center_bearing_deg=predicted_center,
            visibility_lower_deg=lower,
            visibility_upper_deg=upper,
            visibility_min_margin_deg=float("nan"),
        ),
        "ok",
    )
