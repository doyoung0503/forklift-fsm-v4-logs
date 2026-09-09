"""Single tuning surface for FSM v4.

Edit this file, restart ``main_rec_v4.py``, and the new values are applied to
both the v4 planner and the CAN movement templates.  Values marked PROVISIONAL
are safe placeholders for software integration, not measured vehicle data.
"""

from __future__ import annotations
import math

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..config import MODEL_PATH, CAMERA_ENABLED
from .. import config as _pose_config
from .rotation_artifact import select_rotation_response
from .rotation_model import RotationResponse
from .endpoint_response import load_selected_endpoint, validate_runtime_inference_contract


# ---------------------------------------------------------------------------
# 1) Physical external parameters (measure on the real forklift)
# Camera optical frame: +X right, +Y down, +Z forward.
# Rotation centre is expressed from the camera optical centre in vehicle X/Z.
# ---------------------------------------------------------------------------
COARSE_IMU_ENABLED = True
# Recalibrated with TX-timed rotation and near-range replanning (2026-09-09).
# D = -rot_z_pallet_m; trigger when abs(rot_x_pallet_m) exceeds the limit.
# 80% of the sampled correction-capacity lower envelope, rounded down.
COARSE_LATERAL_BASE_M = 0.10
COARSE_LATERAL_GAIN = 0.190
COARSE_LATERAL_CALIBRATION_MIN_DISTANCE_M = 3.0
COARSE_LATERAL_CALIBRATION_MAX_DISTANCE_M = 5.0
COARSE_IMU_SIGN = 1.0  # gyro Y: positive right, as in the packaged IMU diagnostic
COARSE_IMU_MAX_AGE_SEC = 0.25
COARSE_COMMAND_LEASE_SEC = 0.30
COARSE_ROTATION_TIMEOUT_SEC = 30.0
COARSE_TOTAL_TIMEOUT_SEC = 100.0
COARSE_SETTLE_SEC = 1.5
COARSE_IMU_POST_STOP_DELAY_SEC = 2.0  # exclude immediate post-stop samples
COARSE_STABLE_RATE_DEG_S = 0.5
COARSE_SETTLE_TIMEOUT_SEC = 5.0
COARSE_MAX_LATERAL_M = 5.0
COARSE_TRAVEL_LIMITS_ENABLED = False  # optional distance / fitted FWD duration caps
COARSE_MIN_LATERAL_M = 0.02
COARSE_DRIVE_HEADING_TOL_DEG = 5.0
COARSE_REACQUIRE_TIMEOUT_SEC = 10.0

CAMERA_TO_ROT_CENTER_X_M = 0.00       # PROVISIONAL; right-positive
CAMERA_TO_ROT_CENTER_Z_M = -0.68      # measured; centre 0.68 m behind camera
CAMERA_YAW_IN_VEHICLE_DEG = 0.00      # PROVISIONAL; camera forward, right-positive
# Conservative fallback for the active 640x480 colour stream.  Recorded
# RealSense intrinsics give 55.49-55.68 deg; live intrinsics still take
# precedence in the predictive visibility planner.
CAMERA_HORIZONTAL_FOV_DEG = 55.0
FORK_TIP_CLEARANCE_M = 0.40           # desired clearance before insertion
CAMERA_TO_FORK_TIP_X_M = 0.00         # PROVISIONAL; used for documentation/logs
CAMERA_TO_FORK_TIP_Z_M = 1.18         # measured; fork tip 1.18 m ahead of camera
EXTRINSICS_MEASURED = True
REQUIRE_MEASURED_EXTRINSICS = False   # True blocks automatic progression

# Derived from the two measured camera-frame Z positions.  Do not duplicate
# this physical length as another independently tuned constant.
ROT_CENTER_TO_FORK_TIP_M = round(
    CAMERA_TO_FORK_TIP_Z_M - CAMERA_TO_ROT_CENTER_Z_M, 2
)

# Staging target: rotation-centre-to-pallet front-plane distance along the
# pallet normal.  Usually change the two physical terms above, not this sum.
STAGING_DISTANCE_M = round(
    ROT_CENTER_TO_FORK_TIP_M + FORK_TIP_CLEARANCE_M, 2
)


# ---------------------------------------------------------------------------
# 2) CAN joystick strengths / actuator direction
# v4 applies these values at construction. v1-v3 remain unchanged until their
# own launcher is selected in a fresh process.
# ---------------------------------------------------------------------------
# Monitor-only tests should leave this False.  Set True only when the Kvaser
# interface and vehicle are connected and real movement is intended.
CAN_ENABLED = True
CAMERA_DISPLAY_SCALE = 1.5
# Fixed recording width. 550 px matches the previous normal compact layout
# while preventing frame-to-frame width changes from long HUD strings.
INTERFACE_PANEL_WIDTH = 550
FSM_PANEL_WIDTH = 620
# Pause at each high-level FSM checkpoint.  SPACE runs exactly one checkpoint
# stage while perception, HUD and recording continue in real time.
DEBUG_STEP_MODE = False
ROTATE_JOYSTICK_DEFLECTION = 30
FORWARD_JOYSTICK_DEFLECTION = 60
BACKWARD_JOYSTICK_DEFLECTION = 60
FORWARD_PATH_BIAS_DEG = 0.00            # PROVISIONAL; actual path right-positive


# ---------------------------------------------------------------------------
# 3) Rotation response and safety limits (torque/deflection 30)
# Startup defaults come from the 2026-08-28 PnP response collection.
# A full plateau/deceleration fit does not yet exist.
# ---------------------------------------------------------------------------
ROT_STARTUP_DELAY_SEC = 1.516  # six coarse traces, filtered-PnP median
ROT_STARTUP_TIMEOUT_SEC = 3.00
ROT_STARTUP_MIN_VISUAL_DELTA_DEG = 0.50
ROT_STARTUP_CONFIRM_FRAMES = 3
ROT_RATE_LIMIT_DEG_S = 60.0
# Median directed speed after filtered visual onset.  Only the first two bins
# have usable multi-segment support.  Beyond 0.4 s the controller deliberately
# falls back to live PnP rate instead of inventing a plateau.
ROT_LOG_SPEED_PROFILE_ENABLED = True
ROT_LOG_SPEED_PROFILE: Tuple[Tuple[float, float, float], ...] = (
    (0.00, 0.20, 5.883287),
    (0.20, 0.40, 11.702257),
)
ROT_STOP_LOOKAHEAD_SEC = 0.55
ROT_STOP_MARGIN_BASE_DEG = 2.50
ROT_STOP_MARGIN_MAX_DEG = 4.00
ROT_MIN_COMMANDABLE_ANGLE_DEG = 2.50
# Accept settled waypoint-turn error relative to its commanded target yaw.
# This is separate from actuator minimum angle and final-pose yaw tolerance.
WAYPOINT_SETTLED_YAW_TOL_DEG = 3.0
ROT_MAX_WAYPOINT_TURN_DEG = 20.0
ROT_MAX_COMMAND_SEC = 8.0
ROT_MAX_TOTAL_ALIGN_SEC = 120.0
# Learn a shorter command hold after a settled rotation exceeds its target.
# Only the time after the fitted 1.082 s startup/dead time is scaled.  The
# lower bound prevents one noisy pose jump from suppressing later commands.
ROT_ADAPTIVE_OVERSHOOT_ENABLED = False
# Session-only learning for the selected heading endpoint fit, independent
# of the legacy active-time scaling. No external logs or fitted-file writes.
ROT_ADAPTIVE_SLOPE_ENABLED = True
ROT_ADAPTIVE_SLOPE_ALPHA = 0.50
ROT_ADAPTIVE_SLOPE_MIN_MULTIPLIER = 0.5
ROT_ADAPTIVE_SLOPE_MAX_MULTIPLIER = 2.0
ROT_ADAPTIVE_SLOPE_MIN_ACTIVE_SEC = 0.10
ROT_ADAPTIVE_MIN_TIME_SCALE = 0.25
ROT_MAX_DIRECTION_CHANGES = 8

# ---------------------------------------------------------------------------
# 3b) Fitted rotation response (rotation_fit/, deflection 30)
# Continuous time-angle curve fitted to the recorded rotation segments.  It
# replaces the two-bin ROT_LOG_SPEED_PROFILE above: instead of a speed table it
# carries a start delay, an acceleration, a cruise rate, a STOP transport delay
# and a deceleration, so both the command time for a wanted angle and the
# post-STOP inertia are computed from one model.
#
# The selected endpoint regression is loaded once on process startup. Missing
# or mismatched selected artifacts BLOCK startup; the legacy constants below
# are retained only for explicit legacy-mode compatibility, not used by default.
# ---------------------------------------------------------------------------
ROT_USE_FITTED_RESPONSE = True
ROT_RESPONSE_MODE = "selected_delayed_linear"
if ROT_RESPONSE_MODE == "selected_delayed_linear":
    validate_runtime_inference_contract(_pose_config)
ROT_GENERATED_ARTIFACT_ENABLED = True
ROT_GENERATED_ARTIFACT_PATH = Path(__file__).with_name(
    "rotation_endpoint.selected.json"
)

# Fitted in the *heading* domain: degrees of vehicle heading, from pallet yaw
# solved with all visible keypoints.  Front-face-bearing modes convert with
# ROT_BEARING_CENTRE_OFFSET_M below.
ROT_RESPONSE_STARTUP_DELAY_SEC = 1.082     # ROT command -> motion onset
ROT_RESPONSE_STOP_DELAY_SEC = 0.352        # STOP command -> deceleration start
ROT_RESPONSE_ACCEL_DEG_S2 = 13.06
ROT_RESPONSE_DECEL_DEG_S2 = 300.0          # only an upper bound; see note
ROT_RESPONSE_MAX_RATE_DEG_S = 12.01
ROT_RESPONSE_STARTUP_DELAY_SD_SEC = 0.105  # dominant open-loop error source
ROT_RESPONSE_RESIDUAL_SD_DEG = 1.44        # total-angle prediction 1-sigma
ROT_RESPONSE_SETTLE_P90_SEC = 0.88         # measured STOP -> stable pose
# Latest settled heading rotations showed about 1.07 deg less post-STOP
# rotation than this response predicts. Subtract it from the coast estimate
# before solving for the ROT command hold time.
ROT_COAST_HEURISTIC_REDUCTION_DEG = 1.07
ROT_RESPONSE_FALLBACK_SOURCE = (
    "2026-09-03 rotation_fit: 9-keypoint PnP pallet yaw, 9 segments, "
    "deflection 30; delays pinned from the lower-noise bearing fit"
)
ROT_COMMAND_TIMEOUT_MARGIN_SEC = 1.00      # fitted hold + margin = hard timeout

# Longest a single ROT command may be held.  In one log the truck turned 6.5 deg
# and then stood still for 3 s with ROT_LEFT still on the bus, before moving
# again as STOP was written.  Reprojection error was 0.03-0.6 px there, so it
# was the vehicle, not perception.  Every rotation that behaved normally was
# commanded for under 2.5 s, so that is where the evidence ends; larger turns
# are split into several commands by the replan loop instead.
ROT_MAX_COMMAND_HOLD_SEC = 2.50
ROT_TX_TIMED_STOP_ENABLED = True  # CAN owner schedules STOP independently of vision.

# Rotation centre distance behind the camera, used to convert the fitted
# heading response into front-face-bearing degrees for FACE/RECENTER.
#   bearing change = heading change * (range + offset) / range
# It is the magnitude of the same measured Z extrinsic used by path planning,
# so it must not drift as a separately tuned physical constant.
ROT_BEARING_CENTRE_OFFSET_M = abs(CAMERA_TO_ROT_CENTER_Z_M)

ROTATION_RESPONSE_FALLBACK = RotationResponse(
    startup_delay_sec=ROT_RESPONSE_STARTUP_DELAY_SEC,
    stop_delay_sec=ROT_RESPONSE_STOP_DELAY_SEC,
    accel_deg_s2=ROT_RESPONSE_ACCEL_DEG_S2,
    decel_deg_s2=ROT_RESPONSE_DECEL_DEG_S2,
    max_rate_deg_s=ROT_RESPONSE_MAX_RATE_DEG_S,
    startup_delay_sd_sec=ROT_RESPONSE_STARTUP_DELAY_SD_SEC,
    residual_sd_deg=ROT_RESPONSE_RESIDUAL_SD_DEG,
    measured_settle_p90_sec=ROT_RESPONSE_SETTLE_P90_SEC,
    domain="heading",
    source=ROT_RESPONSE_FALLBACK_SOURCE,
)

# The publisher writes a sibling temporary file and atomically replaces this
# final path.  This selection is intentionally never refreshed while a process
# is running; a newly fitted response becomes active on the next clean start.
ROTATION_RESPONSE_SELECTION = (load_selected_endpoint(
    ROT_GENERATED_ARTIFACT_PATH, MODEL_PATH,
) if ROT_RESPONSE_MODE == "selected_delayed_linear" else select_rotation_response(
    ROT_GENERATED_ARTIFACT_PATH,
    ROTATION_RESPONSE_FALLBACK,
    inference_model_path=MODEL_PATH,
    expected_command_strength=ROTATE_JOYSTICK_DEFLECTION,
    max_rate_deg_s=ROT_RATE_LIMIT_DEG_S,
    startup_timeout_sec=ROT_STARTUP_TIMEOUT_SEC,
    max_command_hold_sec=ROT_MAX_COMMAND_HOLD_SEC,
    enabled=ROT_GENERATED_ARTIFACT_ENABLED,
))
ROTATION_RESPONSE = ROTATION_RESPONSE_SELECTION.response
ROT_GENERATED_ARTIFACT_ACTIVE = ROTATION_RESPONSE_SELECTION.artifact_active
ROT_GENERATED_ARTIFACT_SHA256 = (
    ROTATION_RESPONSE_SELECTION.artifact_sha256_id
)
ROT_GENERATED_ARTIFACT_MODEL_SHA256 = (
    ROTATION_RESPONSE_SELECTION.inference_model_sha256_id
)
ROT_GENERATED_ARTIFACT_FALLBACK_REASON = (
    ROTATION_RESPONSE_SELECTION.fallback_reason
)
ROT_RESPONSE_SOURCE = ROTATION_RESPONSE.source

# The generated response already contains the fitted STOP -> STOP+2 s inertia
# regression.  Applying the old coast correction on top would mix two models.
ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG = (
    0.0
    if ROT_GENERATED_ARTIFACT_ACTIVE
    else ROT_COAST_HEURISTIC_REDUCTION_DEG
)
# Never command below the range actually represented by a generated fit.  The
# fallback response keeps the configured field threshold unchanged.
if ROT_GENERATED_ARTIFACT_ACTIVE:
    ROT_MIN_COMMANDABLE_ANGLE_DEG = max(
        ROT_MIN_COMMANDABLE_ANGLE_DEG,
        ROTATION_RESPONSE.fitted_min_angle_deg,
    )


# ---------------------------------------------------------------------------
# 4) Forward/backward response model and macro-action bounds
# Shared forward/backward fit: delay -> acceleration -> cruise, inherited from
# the field logs.  Direction-specific values below are safety limits only.
# ---------------------------------------------------------------------------
FWD_T0_SEC = 0.5069
FWD_ACCEL_DURATION_SEC = 2.0357
FWD_ACCEL_M_S2 = 0.139644
FWD_DISTANCE_SCALE = 1.0
FWD_DISTANCE_BIAS_M = 0.0
FWD_COMMAND_MIN_SEC = 1.0
FWD_COMMAND_MAX_SEC = 15.0
FWD_TIMEOUT_MARGIN_SEC = 1.00          # fitted endpoint 이후 hard-timeout 여유
FWD_RELIABLE_MIN_DISTANCE_M = 0.10     # PROVISIONAL
FWD_MACRO_MAX_DISTANCE_M = 0.80        # PROVISIONAL
FWD_ALIGNED_MAX_DISTANCE_M = 1.50
FWD_ALIGNED_LATERAL_FULL_M = 0.10
FWD_ALIGNED_LATERAL_BASE_M = 0.20
FWD_MAX_TOTAL_CORRECTION_M = 5.00
FWD_STOP_LOOKAHEAD_SEC = 0.45           # PROVISIONAL
FWD_COAST_ALLOWANCE_M = 0.04            # PROVISIONAL
# Raw frame-to-frame PnP velocity can momentarily hit the 1.5 m/s pose gate
# even when the truck has moved only a few centimetres.  Predictive STOP uses
# the fitted speed envelope plus these conservative bounds instead of trusting
# that raw spike directly.
FWD_PREDICTIVE_SPEED_MULTIPLIER = 1.25
FWD_PREDICTIVE_MAX_ADVANCE_M = 0.12
FWD_PREDICTIVE_MIN_PROGRESS_RATIO = 0.50
FWD_MAX_COMMAND_SEC = 15.0

# Forward/backward use the same response fit; backward retains a tighter hard
# action limit because it moves away from the camera-facing target geometry.
BACK_TIMEOUT_MARGIN_SEC = 1.00            # PROVISIONAL
BACK_MAX_COMMAND_SEC = 6.0
BACK_MAX_DISTANCE_M = 0.50


# ---------------------------------------------------------------------------
# 5) Perception, timestamp and pose filtering
# ``SENSOR_PIPELINE_LATENCY_SEC`` supplements measured inference time because
# RealSense device time is not yet synchronized to host monotonic time.
# ---------------------------------------------------------------------------
SENSOR_PIPELINE_LATENCY_SEC = 0.05       # PROVISIONAL
FALLBACK_INFERENCE_LATENCY_SEC = 0.12
MAX_MEASUREMENT_AGE_SEC = 0.30
DETECTION_CONFIRM_FRAMES = 5
STABLE_POSE_FRAMES = 10
# Median-centred robust stability test.  A frame is an inlier only when yaw,
# X and Z are all inside these tolerances; 8/10 inliers passes, so up to two
# isolated model outliers do not reject an otherwise stationary window.
STABLE_YAW_MEDIAN_TOL_DEG = 4.0
STABLE_X_MEDIAN_TOL_M = 0.04
STABLE_Z_MEDIAN_TOL_M = 0.10
STABLE_MIN_INLIER_RATIO = 0.80
POSE_GATE_POSITION_BASE_M = 0.12
POSE_GATE_YAW_BASE_DEG = 12.0
POSE_MAX_TRANSLATION_RATE_M_S = 1.50
POSE_MAX_YAW_RATE_DEG_S = 120.0
# A single dropped frame does not stop an active command.  The last accepted
# pose is held for this continuous-loss grace period; a valid frame resets it.
VISION_LOSS_CONFIRM_SEC = 1.0
# After the confirmed loss has stopped the vehicle, wait this much longer for
# PnP to recover before declaring terminal failure.
MAX_PNP_LOSS_SEC = 1.0


# ---------------------------------------------------------------------------
# 6) Search, stopped observation and visibility
# ---------------------------------------------------------------------------
# SEARCH always rotates right; lack of detection for this duration is terminal.
SEARCH_TOTAL_TIMEOUT_SEC = 30.0
ACQUIRE_VERIFY_TIMEOUT_SEC = 4.0
# STOP is followed by a brake/coast guard.  Stability evaluation starts only
# after this guard and receives its own full timeout budget.
STOP_MIN_SETTLE_SEC = 2.0
STOP_MAX_SETTLE_SEC = 4.0
# Settle acceptance uses the median-centred STABLE_* window below.  It does
# not use a consecutive-frame (roughly 100 ms) pose derivative threshold.
PALLET_CENTER_FACE_TOL_DEG = 2.50
# While FACE is actively rotating, a live front-face centre inside this
# optical-axis tolerance overrides the response-model endpoint and STOPs now.
FACE_CENTER_IMMEDIATE_STOP_TOL_DEG = 0.50
PALLET_CENTER_RECENTER_TOL_DEG = 2.50
# Bearing-driven FACE is used only before staging, during the 4 m approach.
# Staging plans heading turns directly; there is no RECENTER fallback.
BEARING_BASED_ROTATION_ENABLED = True
# Disable centre-bearing-only visibility limits.
PALLET_CENTER_VISIBILITY_GUARD_ENABLED = False
PALLET_CENTER_SAFE_BEARING_DEG = 20.0
# Legacy trace flag: live ROI violations no longer interrupt motion.
# Predictive planning still enforces IMAGE_EDGE_MARGIN_NORM below.
IMAGE_EDGE_VISIBILITY_GUARD_ENABLED = False
BBOX_SAFE_MARGIN_NORM = 0.08
# 8% at HFOV 55 deg reserves about 3.88 deg at each physical edge.
IMAGE_EDGE_MARGIN_NORM = 0.08
# Predict the complete horizontal front-face outline through each candidate
# in-place turn and straight-forward macro action.  This is the physical outer
# width, deliberately wider than the 1.00 m labelled-keypoint PnP model.
PREDICTIVE_VISIBILITY_PLANNER_ENABLED = True
PALLET_FRONT_VISIBILITY_WIDTH_M = 1.10
VISIBILITY_TURN_SEARCH_STEP_DEG = 0.50
VISIBILITY_PATH_SAMPLE_STEP_DEG = 0.50
# 0.8 m / 2^16 is about 0.012 mm, already far below pose/model precision.
VISIBILITY_FORWARD_SEARCH_ITERATIONS = 16
VISIBILITY_MIN_CORNER_DEPTH_M = 0.10
# Do not stop an executing forward segment on instantaneous goal bearing.
# Pose-loss handling, distance/staging limits and timeouts remain active.
DRIVE_HEADING_GUARD_ENABLED = False
DRIVE_HEADING_TOL_DEG = 4.0


# ---------------------------------------------------------------------------
# 7) Distance, staging and final pose tolerances
# Approach to 4 m only from farther away; do not back away from closer starts.
# Starts at 2.2 m through the standoff band enter STAGING_PLAN directly.
# Below 2.2 m, check rotation-only insertion alignment before any translation.
# ---------------------------------------------------------------------------
SAFETY_STANDOFF_Z_M = 5.00
SAFETY_STANDOFF_BAND_M = 0.10
STANDOFF_MAX_CORRECTIONS = 4
# Staging acceptance is component-wise; do not add a tighter radial gate here.
FINAL_LATERAL_TOL_M = 0.10              # temporary field tolerance; validate before tightening
# Field-relaxed final yaw acceptance (previously +/-2.5 deg).
FINAL_YAW_TOL_DEG = 20.00
# Legacy display span only; approval uses the nine-block geometry below.
INSERT_OPENING_SPAN_M = 0.70
INSERT_PALLET_SIZE_M = 1.10
INSERT_BLOCK_SIZE_M = 0.20
INSERT_HOLE_WIDTH_M = 0.25
INSERT_WALL_MARGIN_M = 0.01  # Additional planar clearance, not a pose-error bound.
FORK_WIDTH_M = 0.115  # User-measured width of ONE fork (2026-09-08); None blocks insertion.
INSERT_ALIGNMENT_MAX_CAMERA_Z_M = 2.20  # PnP selected front-centre camera Z
INSERT_ALIGNMENT_MAX_CORRECTIONS = 3
INSERT_FINE_MIN_TURN_DEG = 0.50  # Final insertion alignment only; provisional micro-command floor
INSERT_TURN_REFINE_RESOLUTION_DEG = 0.05  # fallback interval width, not minimum executed angle
INSERT_FINE_TIMEOUT_SEC = 120.0  # Includes command, STOP settling and rechecks
INSERT_FINE_MIN_ACTIVE_SEC = 0.02  # Reject near-zero denominators in fine-turn adaptation
FORK_OUTER_SPAN_M = 0.60
# Longitudinal (forward-axis) staging tolerance and overtravel limit.
FINAL_DISTANCE_TOL_M = 0.20
MAX_CORRECTION_CYCLES = 10
MAX_NO_PROGRESS_CYCLES = 2
MIN_PROGRESS_M = 0.02
TOTAL_PIPELINE_TIMEOUT_SEC = 180.0


# ---------------------------------------------------------------------------
# 8) Insertion policy
# Approval requires the complete planned straight sweep to clear all nine blocks.
# Execution retains one fitted-time command without pose feedback.
# ---------------------------------------------------------------------------
AUTO_INSERT_ENABLED = True
INSERT_TOTAL_DISTANCE_M = 1.00          # Legacy only; runtime uses accepted camera Z minus remainder
INSERT_CAMERA_Z_REMAINDER_M = 0.30  # Freeze accepted front-centre Z minus this distance
INSERT_SEGMENT_DISTANCE_M = 1.00        # Legacy only; no runtime segmentation
INSERT_MAX_TOTAL_SEC = 20.0
# Legacy metadata/compatibility only; not applied during INSERT_DRIVE.
INSERT_YAW_ABORT_DEG = 4.0
INSERT_LATERAL_ABORT_M = 0.08


@dataclass(frozen=True)
class V4ConfigSnapshot:
    """Compact immutable snapshot written to each recording metadata file."""

    camera_to_rotation_center_x_m: float = CAMERA_TO_ROT_CENTER_X_M
    camera_to_rotation_center_z_m: float = CAMERA_TO_ROT_CENTER_Z_M
    camera_yaw_in_vehicle_deg: float = CAMERA_YAW_IN_VEHICLE_DEG
    camera_horizontal_fov_deg: float = CAMERA_HORIZONTAL_FOV_DEG
    rotation_center_to_fork_tip_m: float = ROT_CENTER_TO_FORK_TIP_M
    fork_tip_clearance_m: float = FORK_TIP_CLEARANCE_M
    camera_to_fork_tip_x_m: float = CAMERA_TO_FORK_TIP_X_M
    camera_to_fork_tip_z_m: float = CAMERA_TO_FORK_TIP_Z_M
    staging_distance_m: float = STAGING_DISTANCE_M
    extrinsics_measured: bool = EXTRINSICS_MEASURED
    require_measured_extrinsics: bool = REQUIRE_MEASURED_EXTRINSICS
    can_enabled: bool = CAN_ENABLED
    camera_enabled: bool = CAMERA_ENABLED
    camera_display_scale: float = CAMERA_DISPLAY_SCALE
    debug_step_mode: bool = DEBUG_STEP_MODE
    rotate_joystick_deflection: int = ROTATE_JOYSTICK_DEFLECTION
    forward_joystick_deflection: int = FORWARD_JOYSTICK_DEFLECTION
    backward_joystick_deflection: int = BACKWARD_JOYSTICK_DEFLECTION
    forward_path_bias_deg: float = FORWARD_PATH_BIAS_DEG
    bearing_based_rotation_enabled: bool = BEARING_BASED_ROTATION_ENABLED
    pallet_center_visibility_guard_enabled: bool = (
        PALLET_CENTER_VISIBILITY_GUARD_ENABLED
    )
    image_edge_visibility_guard_enabled: bool = (
        IMAGE_EDGE_VISIBILITY_GUARD_ENABLED
    )
    predictive_visibility_planner_enabled: bool = (
        PREDICTIVE_VISIBILITY_PLANNER_ENABLED
    )
    pallet_front_visibility_width_m: float = PALLET_FRONT_VISIBILITY_WIDTH_M
    visibility_turn_search_step_deg: float = VISIBILITY_TURN_SEARCH_STEP_DEG
    visibility_path_sample_step_deg: float = VISIBILITY_PATH_SAMPLE_STEP_DEG
    visibility_forward_search_iterations: int = (
        VISIBILITY_FORWARD_SEARCH_ITERATIONS
    )
    visibility_min_corner_depth_m: float = VISIBILITY_MIN_CORNER_DEPTH_M
    drive_heading_guard_enabled: bool = DRIVE_HEADING_GUARD_ENABLED
    drive_heading_tolerance_deg: float = DRIVE_HEADING_TOL_DEG
    rotation_startup_delay_sec: float = ROT_STARTUP_DELAY_SEC
    rotation_log_speed_profile_enabled: bool = ROT_LOG_SPEED_PROFILE_ENABLED
    rotation_log_speed_profile: Tuple[Tuple[float, float, float], ...] = (
        ROT_LOG_SPEED_PROFILE
    )
    rotation_stop_lookahead_sec: float = ROT_STOP_LOOKAHEAD_SEC
    rotation_max_command_sec: float = ROT_MAX_COMMAND_SEC
    rotation_max_waypoint_turn_deg: float = ROT_MAX_WAYPOINT_TURN_DEG
    waypoint_settled_yaw_tolerance_deg: float = WAYPOINT_SETTLED_YAW_TOL_DEG
    rotation_max_command_hold_sec: float = ROT_MAX_COMMAND_HOLD_SEC
    rotation_adaptive_overshoot_enabled: bool = ROT_ADAPTIVE_OVERSHOOT_ENABLED
    rotation_adaptive_slope_enabled: bool = ROT_ADAPTIVE_SLOPE_ENABLED
    rotation_adaptive_slope_alpha: float = ROT_ADAPTIVE_SLOPE_ALPHA
    rotation_adaptive_min_time_scale: float = ROT_ADAPTIVE_MIN_TIME_SCALE
    rotation_use_fitted_response: bool = ROT_USE_FITTED_RESPONSE
    rotation_response_kind: str = type(ROTATION_RESPONSE).__name__
    rotation_response_endpoint_only: bool = getattr(ROTATION_RESPONSE, 'endpoint_only', False)
    rotation_response_endpoint_slope_deg_s: float = getattr(ROTATION_RESPONSE, 'slope_deg_s', 0.)
    rotation_response_startup_delay_sec: float = (
        ROTATION_RESPONSE.startup_delay_sec
    )
    rotation_response_stop_delay_sec: float = ROTATION_RESPONSE.stop_delay_sec
    rotation_response_accel_duration_sec: float = getattr(
        ROTATION_RESPONSE, "accel_duration_sec",
        ROTATION_RESPONSE.max_rate_deg_s / max(1e-12, ROTATION_RESPONSE.accel_deg_s2),
    )
    rotation_response_accel_deg_s2: float = ROTATION_RESPONSE.accel_deg_s2
    rotation_response_decel_deg_s2: float = getattr(
        ROTATION_RESPONSE, "decel_deg_s2", 0.0,
    )
    rotation_response_max_rate_deg_s: float = ROTATION_RESPONSE.max_rate_deg_s
    rotation_response_startup_delay_sd_sec: float = (
        getattr(ROTATION_RESPONSE, "startup_delay_sd_sec", 0.0)
    )
    rotation_response_inertia_intercept_deg: float = getattr(
        ROTATION_RESPONSE, "inertia_intercept_deg", 0.0,
    )
    rotation_response_inertia_slope_sec: float = getattr(
        ROTATION_RESPONSE, "inertia_slope_sec", 0.0,
    )
    rotation_response_fitted_min_hold_sec: float = getattr(
        ROTATION_RESPONSE, "fitted_min_hold_sec", 0.0,
    )
    rotation_response_fitted_max_hold_sec: float = getattr(
        ROTATION_RESPONSE, "fitted_max_hold_sec", 0.0,
    )
    rotation_response_fitted_min_angle_deg: float = getattr(
        ROTATION_RESPONSE, "fitted_min_angle_deg", 0.0,
    )
    rotation_response_fitted_max_angle_deg: float = getattr(
        ROTATION_RESPONSE, "fitted_max_angle_deg", 0.0,
    )
    rotation_response_residual_sd_deg: float = ROTATION_RESPONSE.residual_sd_deg
    rotation_response_settle_p90_sec: float = (
        ROTATION_RESPONSE.measured_settle_p90_sec
    )
    rotation_coast_heuristic_reduction_deg: float = (
        ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG
    )
    rotation_bearing_centre_offset_m: float = ROT_BEARING_CENTRE_OFFSET_M
    rotation_response_source: str = ROT_RESPONSE_SOURCE
    rotation_generated_artifact_enabled: bool = ROT_GENERATED_ARTIFACT_ENABLED
    rotation_generated_artifact_active: bool = ROT_GENERATED_ARTIFACT_ACTIVE
    rotation_generated_artifact_path: str = str(ROT_GENERATED_ARTIFACT_PATH)
    rotation_generated_artifact_sha256: str = ROT_GENERATED_ARTIFACT_SHA256
    rotation_generated_model_sha256: str = ROT_GENERATED_ARTIFACT_MODEL_SHA256
    rotation_generated_fallback_reason: str = (
        ROT_GENERATED_ARTIFACT_FALLBACK_REASON
    )
    forward_t0_sec: float = FWD_T0_SEC
    forward_accel_duration_sec: float = FWD_ACCEL_DURATION_SEC
    forward_accel_m_s2: float = FWD_ACCEL_M_S2
    forward_timeout_margin_sec: float = FWD_TIMEOUT_MARGIN_SEC
    forward_reliable_min_distance_m: float = FWD_RELIABLE_MIN_DISTANCE_M
    forward_macro_max_distance_m: float = FWD_MACRO_MAX_DISTANCE_M
    forward_aligned_max_distance_m: float = FWD_ALIGNED_MAX_DISTANCE_M
    forward_aligned_lateral_full_m: float = FWD_ALIGNED_LATERAL_FULL_M
    forward_aligned_lateral_base_m: float = FWD_ALIGNED_LATERAL_BASE_M
    forward_predictive_speed_multiplier: float = FWD_PREDICTIVE_SPEED_MULTIPLIER
    forward_predictive_max_advance_m: float = FWD_PREDICTIVE_MAX_ADVANCE_M
    forward_predictive_min_progress_ratio: float = (
        FWD_PREDICTIVE_MIN_PROGRESS_RATIO
    )
    safety_standoff_z_m: float = SAFETY_STANDOFF_Z_M
    safety_standoff_band_m: float = SAFETY_STANDOFF_BAND_M
    vision_loss_confirm_sec: float = VISION_LOSS_CONFIRM_SEC
    max_pnp_loss_sec: float = MAX_PNP_LOSS_SEC
    stop_brake_guard_sec: float = STOP_MIN_SETTLE_SEC
    stop_stability_timeout_sec: float = STOP_MAX_SETTLE_SEC
    settle_uses_consecutive_frame_velocity: bool = False
    stable_pose_frames: int = STABLE_POSE_FRAMES
    stable_yaw_median_tolerance_deg: float = STABLE_YAW_MEDIAN_TOL_DEG
    stable_x_median_tolerance_m: float = STABLE_X_MEDIAN_TOL_M
    stable_z_median_tolerance_m: float = STABLE_Z_MEDIAN_TOL_M
    stable_min_inlier_ratio: float = STABLE_MIN_INLIER_RATIO
    face_center_immediate_stop_tol_deg: float = (
        FACE_CENTER_IMMEDIATE_STOP_TOL_DEG
    )
    final_lateral_tol_m: float = FINAL_LATERAL_TOL_M
    final_distance_tol_m: float = FINAL_DISTANCE_TOL_M
    final_yaw_tol_deg: float = FINAL_YAW_TOL_DEG
    insertion_pallet_size_m: float = INSERT_PALLET_SIZE_M
    insertion_block_size_m: float = INSERT_BLOCK_SIZE_M
    insertion_hole_width_m: float = INSERT_HOLE_WIDTH_M
    insertion_wall_margin_m: float = INSERT_WALL_MARGIN_M
    fork_width_m: Optional[float] = FORK_WIDTH_M
    fork_outer_span_m: float = FORK_OUTER_SPAN_M
    auto_insert_enabled: bool = AUTO_INSERT_ENABLED
    insertion_fine_min_turn_deg: float = INSERT_FINE_MIN_TURN_DEG
    insertion_turn_refine_resolution_deg: float = INSERT_TURN_REFINE_RESOLUTION_DEG
    insertion_fine_timeout_sec: float = INSERT_FINE_TIMEOUT_SEC


def metadata() -> Dict[str, object]:
    result = {f"v4_{key}": value for key, value in asdict(V4ConfigSnapshot()).items()}
    result.update({f"v4_{key.lower()}": value for key, value in globals().items()
                   if key.startswith("COARSE_")})
    return result


def validate() -> None:
    """Fail early on internally inconsistent edits."""

    errors = []
    if COARSE_IMU_SIGN not in (-1., 1.):
        errors.append("invalid coarse IMU sign")
    if (not all(math.isfinite(value) for value in (
            COARSE_LATERAL_BASE_M, COARSE_LATERAL_GAIN,
            COARSE_LATERAL_CALIBRATION_MIN_DISTANCE_M,
            COARSE_LATERAL_CALIBRATION_MAX_DISTANCE_M))
            or COARSE_LATERAL_BASE_M < COARSE_MIN_LATERAL_M
            or COARSE_LATERAL_GAIN <= 0.
            or not STAGING_DISTANCE_M < COARSE_LATERAL_CALIBRATION_MIN_DISTANCE_M
            <= COARSE_LATERAL_CALIBRATION_MAX_DISTANCE_M):
        errors.append("invalid coarse lateral threshold calibration")
    if not 0 < COARSE_IMU_MAX_AGE_SEC <= COARSE_COMMAND_LEASE_SEC <= 1.0:
        errors.append("invalid coarse IMU freshness / command lease")
    if not 0 < COARSE_MIN_LATERAL_M < COARSE_MAX_LATERAL_M:
        errors.append("invalid coarse lateral range")
    if not 0 < COARSE_SETTLE_SEC < COARSE_SETTLE_TIMEOUT_SEC < COARSE_TOTAL_TIMEOUT_SEC:
        errors.append("invalid coarse settle limits")
    if CAMERA_DISPLAY_SCALE <= 0.0:
        errors.append("CAMERA_DISPLAY_SCALE must be positive")
    if not 0.0 < CAMERA_HORIZONTAL_FOV_DEG < 180.0:
        errors.append("CAMERA_HORIZONTAL_FOV_DEG must be in (0, 180)")
    if REQUIRE_MEASURED_EXTRINSICS and not EXTRINSICS_MEASURED:
        errors.append("measured camera/rotation-centre extrinsics are required")
    if not 1 <= ROTATE_JOYSTICK_DEFLECTION <= 126:
        errors.append("ROTATE_JOYSTICK_DEFLECTION must be in [1, 126]")
    if not 1 <= FORWARD_JOYSTICK_DEFLECTION <= 126:
        errors.append("FORWARD_JOYSTICK_DEFLECTION must be in [1, 126]")
    if not 1 <= BACKWARD_JOYSTICK_DEFLECTION <= 126:
        errors.append("BACKWARD_JOYSTICK_DEFLECTION must be in [1, 126]")
    if ROT_STARTUP_DELAY_SEC >= ROT_MAX_COMMAND_SEC:
        errors.append("rotation startup delay must be below max command time")
    if not 0.0 < ROT_ADAPTIVE_MIN_TIME_SCALE <= 1.0:
        errors.append("rotation adaptive minimum time scale must be in (0, 1]")
    from .adaptive_slope import SessionSlope
    try:
        SessionSlope(ROT_ADAPTIVE_SLOPE_ALPHA, ROT_ADAPTIVE_SLOPE_MIN_MULTIPLIER,
                     ROT_ADAPTIVE_SLOPE_MAX_MULTIPLIER, ROT_ADAPTIVE_SLOPE_MIN_ACTIVE_SEC)
    except ValueError as exc:
        errors.append(str(exc))
    if ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG < 0.0:
        errors.append("rotation coast heuristic reduction must be non-negative")
    if getattr(ROTATION_RESPONSE, 'endpoint_only', False):
        if ROTATE_JOYSTICK_DEFLECTION != 30:
            errors.append('selected endpoint requires rotation strength 30')
        # The operational guard is user-configured independently of the
        # endpoint artifact's historical settle policy.
        if not math.isfinite(STOP_MIN_SETTLE_SEC) or STOP_MIN_SETTLE_SEC <= 0.0:
            errors.append('post-STOP guard must be finite and positive')
        if ROT_MAX_COMMAND_HOLD_SEC > ROTATION_RESPONSE.fitted_max_hold_sec:
            errors.append('selected endpoint hold exceeds observed time range')
    if ROT_USE_FITTED_RESPONSE:
        if (
            ROTATION_RESPONSE.startup_delay_sec < 0.0
            or ROTATION_RESPONSE.stop_delay_sec < 0.0
        ):
            errors.append("fitted rotation delays must be positive/non-negative")
        if not getattr(ROTATION_RESPONSE, 'endpoint_only', False) and ROTATION_RESPONSE.accel_deg_s2 <= 0.0:
            errors.append("fitted rotation acceleration must be positive")
        if (
            not getattr(ROTATION_RESPONSE, "uses_embedded_inertia", False)
            and ROTATION_RESPONSE.decel_deg_s2 <= 0.0
        ):
            errors.append("legacy fitted rotation deceleration must be positive")
        if ROTATION_RESPONSE.max_rate_deg_s <= 0.0:
            errors.append("fitted rotation cruise rate must be positive")
        if ROTATION_RESPONSE.max_rate_deg_s > ROT_RATE_LIMIT_DEG_S:
            errors.append("fitted cruise rate exceeds the rotation rate limit")
        # 단일 명령으로 낼 수 있는 최소 회전각보다 작은 목표는 명령할 수 없다.
        if ROT_MIN_COMMANDABLE_ANGLE_DEG < ROTATION_RESPONSE.min_total_deg:
            errors.append(
                "ROT_MIN_COMMANDABLE_ANGLE_DEG (%.2f) is below the smallest "
                "angle one command can produce (%.2f)" % (
                    ROT_MIN_COMMANDABLE_ANGLE_DEG,
                    ROTATION_RESPONSE.min_total_deg))
        if ROT_MAX_COMMAND_HOLD_SEC > ROT_MAX_COMMAND_SEC:
            errors.append("ROT_MAX_COMMAND_HOLD_SEC exceeds ROT_MAX_COMMAND_SEC")
    previous_end = 0.0
    if ROT_LOG_SPEED_PROFILE_ENABLED and not ROT_LOG_SPEED_PROFILE:
        errors.append("enabled rotation log speed profile must not be empty")
    for start, end, speed in ROT_LOG_SPEED_PROFILE:
        if start < previous_end or end <= start or speed <= 0.0:
            errors.append("rotation log speed bins must be ordered and positive")
            break
        previous_end = end
    if FWD_RELIABLE_MIN_DISTANCE_M > FWD_MACRO_MAX_DISTANCE_M:
        errors.append("forward reliable minimum exceeds macro maximum")
    if not FWD_MACRO_MAX_DISTANCE_M <= FWD_ALIGNED_MAX_DISTANCE_M < float("inf"):
        errors.append("aligned forward maximum must be finite and >= base maximum")
    if not 0.0 <= FWD_ALIGNED_LATERAL_FULL_M < FWD_ALIGNED_LATERAL_BASE_M < float("inf"):
        errors.append("aligned lateral thresholds must be finite and ordered")
    if PALLET_FRONT_VISIBILITY_WIDTH_M <= 0.0:
        errors.append("pallet visibility width must be positive")
    if VISIBILITY_TURN_SEARCH_STEP_DEG <= 0.0:
        errors.append("visibility turn search step must be positive")
    if VISIBILITY_PATH_SAMPLE_STEP_DEG <= 0.0:
        errors.append("visibility path sample step must be positive")
    if VISIBILITY_FORWARD_SEARCH_ITERATIONS < 1:
        errors.append("visibility forward search iterations must be positive")
    if VISIBILITY_MIN_CORNER_DEPTH_M <= 0.0:
        errors.append("visibility minimum corner depth must be positive")
    if FWD_PREDICTIVE_SPEED_MULTIPLIER <= 0.0:
        errors.append("forward predictive speed multiplier must be positive")
    if not 0.0 <= FWD_PREDICTIVE_MAX_ADVANCE_M <= FWD_MACRO_MAX_DISTANCE_M:
        errors.append("forward predictive maximum advance is out of range")
    if not 0.0 <= FWD_PREDICTIVE_MIN_PROGRESS_RATIO <= 1.0:
        errors.append("forward predictive minimum progress ratio must be in [0, 1]")
    if not (math.isfinite(INSERT_TURN_REFINE_RESOLUTION_DEG)
            and 0.0 < INSERT_TURN_REFINE_RESOLUTION_DEG
            < min(VISIBILITY_TURN_SEARCH_STEP_DEG, INSERT_FINE_MIN_TURN_DEG)):
        errors.append("invalid insertion turn refinement resolution")
    if not 0.0 < INSERT_FINE_MIN_TURN_DEG < ROT_MIN_COMMANDABLE_ANGLE_DEG:
        errors.append("fine insertion minimum must be positive and below normal rotation minimum")
    if not 0.0 < INSERT_FINE_TIMEOUT_SEC <= 120.0:
        errors.append("fine insertion timeout must be within 120 seconds")
    if not 0.0 < INSERT_FINE_MIN_ACTIVE_SEC <= ROT_ADAPTIVE_SLOPE_MIN_ACTIVE_SEC:
        errors.append("invalid fine insertion adaptive active-time minimum")
    if INSERT_SEGMENT_DISTANCE_M > INSERT_TOTAL_DISTANCE_M:
        errors.append("insertion segment exceeds total insertion distance")
    if AUTO_INSERT_ENABLED and not EXTRINSICS_MEASURED:
        errors.append("automatic insertion requires measured extrinsics")
    if STAGING_DISTANCE_M <= 0.0 or SAFETY_STANDOFF_Z_M <= 0.0:
        errors.append("staging and safety distances must be positive")
    if FINAL_LATERAL_TOL_M <= 0.0 or FINAL_DISTANCE_TOL_M <= 0.0:
        errors.append("final lateral/distance tolerances must be positive")
    if not 0.0 < FACE_CENTER_IMMEDIATE_STOP_TOL_DEG <= PALLET_CENTER_FACE_TOL_DEG:
        errors.append("FACE immediate STOP tolerance must be positive and no wider than FACE tolerance")
    if STABLE_POSE_FRAMES < 2 or DETECTION_CONFIRM_FRAMES < 2:
        errors.append("stable/detection confirmation requires at least 2 frames")
    if not 0.0 < STABLE_MIN_INLIER_RATIO <= 1.0:
        errors.append("stable minimum inlier ratio must be in (0, 1]")
    if VISION_LOSS_CONFIRM_SEC <= 0.0 or MAX_PNP_LOSS_SEC <= 0.0:
        errors.append("vision loss confirmation/recovery times must be positive")
    if FWD_TIMEOUT_MARGIN_SEC < 0.0 or BACK_TIMEOUT_MARGIN_SEC < 0.0:
        errors.append("translation timeout margins must be non-negative")
    if errors:
        raise ValueError("invalid FSM v4 config: " + "; ".join(errors))
