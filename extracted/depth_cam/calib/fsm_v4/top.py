"""FSM v4: stopped-observe, visible macro-action, settle, verify, replan."""

from __future__ import annotations

import math
import textwrap
import time
from statistics import median
from typing import Dict, List, Optional, Tuple

from calib.config import (
    COLOR_ALERT, COLOR_META, COLOR_STATUS_OK, COLOR_STATUS_TRK, COLOR_YAW,
)
from calib.control import (
    configure_drive_deflection,
    configure_rotate_in_place,
)
from calib.fsm.commands import CommandExecutor
from calib.fsm.status_helper import StatusHelper

from . import config as cfg
from .controllers import RotationController, adaptive_overshoot_scale
from .adaptive_slope import SessionSlope
from .motion import (
    backward_seconds, bounded_forward_lookahead,
    forward_predictive_stop_ready, forward_seconds, measurement_age,
)
from .planner import (
    Waypoint, action_keeps_front_visible, goal_vector_vehicle, plan_waypoint,
    safe_recenter_turn, staging_position_errors, staging_position_reached,
    safe_straight_continuation_m,
    fork_opening_alignment,
    insertion_alignment_turn,
)
from .pose import VisualPose, VisualPoseFilter, wrap_180
from .insertion_geometry import check_insertion_sweep, geometry_error
from .coarse import CoarseAlignmentMixin, COARSE_BLIND_STATES, COARSE_STATES


Line = Tuple[str, tuple]


class CalibrationFSMV4(CoarseAlignmentMixin):
    """Vision-primary bounded FSM using model/PnP observations."""

    accepts_vision_meta = True
    STOPPED_DECISION_STATES = frozenset({"STANDOFF_VERIFY", "STAGING_PLAN", "FINAL_POSE_LOCK"})
    inference_checkpoint_subs = frozenset()
    fsm_version = "v4"
    trace_metadata = {
        "fsm_v4_control": "stopped_observe_visible_macro_action_settle_replan",
        "v4_insertion_control": "accepted_distance_fitted_time_no_vision",
        "v4_insertion_geometry": "nine_blocks_continuous_planar_sweep_with_wall_margin",
        "v4_rotation_stop_speed_source": "raw consecutive PnP error rate",
        "v4_post_stop_rotation_sign": (
            "positive=same direction as stopped rotation; negative=rebound"
        ),
        "v4_config_file": "calib/fsm_v4/config.py",
        **cfg.metadata(),
    }

    def __init__(self, tracer=None):
        cfg.validate()
        self.tracer = tracer
        self.execu = CommandExecutor(tracer=tracer)
        self.status = StatusHelper()
        self._pose_filter = VisualPoseFilter()
        self._rotation = RotationController()

        applied_rotation = configure_rotate_in_place(cfg.ROTATE_JOYSTICK_DEFLECTION)
        applied_forward, applied_backward = configure_drive_deflection(
            cfg.FORWARD_JOYSTICK_DEFLECTION,
            cfg.BACKWARD_JOYSTICK_DEFLECTION,
        )
        print(
            "[FSM v4] joystick deflection: "
            f"rotate={applied_rotation}, forward={applied_forward}, "
            f"backward={applied_backward}"
        )
        if cfg.ROT_GENERATED_ARTIFACT_ACTIVE:
            print(
                "[FSM v4] generated rotation response active: "
                f"{cfg.ROT_GENERATED_ARTIFACT_SHA256}; "
                f"model={cfg.ROT_GENERATED_ARTIFACT_MODEL_SHA256}"
            )
        else:
            print(
                "[FSM v4 WARNING] generated rotation response unavailable; "
                "using built-in fallback: "
                f"{cfg.ROT_GENERATED_ARTIFACT_FALLBACK_REASON}"
            )
        if not cfg.EXTRINSICS_MEASURED:
            print(
                "[FSM v4 WARNING] provisional camera/rotation-centre "
                f"extrinsics X={cfg.CAMERA_TO_ROT_CENTER_X_M:+.3f}m, "
                f"Z={cfg.CAMERA_TO_ROT_CENTER_Z_M:+.3f}m; "
                "automatic insertion is disabled by default"
            )
        self.reset()

    @property
    def cmd_status(self):
        return self.status.cmd_status

    def get_command_status(self):
        return self.status.cmd_status

    @property
    def align_sub(self) -> str:
        return self.state

    @property
    def skip_detection(self) -> bool:
        return self.state in COARSE_BLIND_STATES or self.state in {"READY_TO_INSERT", "INSERT_DRIVE", "INSERT_SETTLE", "DONE", "FAILED"}

    @property
    def vision_independent(self) -> bool:
        """The runtime must keep ticking these states even without camera frames."""
        return self.skip_detection

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    @property
    def failure_state(self) -> Optional[str]:
        return self._failure_state

    @property
    def rotation_diagnostics(self) -> Dict[str, object]:
        """Latest model-derived rotation STOP/coast diagnostics."""
        return {
            "mode": self._rotation_stop_mode,
            "command": self._rotation_stop_command,
            "stop_reason": self._rotation_stop_reason,
            "stop_error_deg": self._rotation_stop_error_deg,
            "stop_measured_rate_deg_s": self._rotation_stop_measured_rate_deg_s,
            "stop_effective_rate_deg_s": self._rotation_stop_effective_rate_deg_s,
            "coast_deg": self._rotation_coast_deg,
            "coast_peak_abs_deg": self._rotation_coast_peak_abs_deg,
            "current_error_deg": self._rotation_coast_current_error_deg,
            "stop_measurement_mono": self._rotation_stop_measurement_mono,
        }

    @property
    def motion_target_status(self) -> Optional[Dict[str, object]]:
        """Live, UI-facing target and progress for the active macro action.

        ``CommandStatus`` describes the controller's stop condition, which is
        not always the motion requested by the planner (for example, rotation
        uses a predicted-error margin).  Keep this separate so the runtime
        diagram can always show the operator the requested angle or distance.
        """
        if self.state in {"COARSE_ROTATE", "COARSE_RETURN_ROTATE", "COARSE_DRIVE"}:
            turning = self.state != "COARSE_DRIVE"
            target = abs(self._coarse_target) if turning else self._coarse_plan[2]
            current = (math.copysign(1.0, self._coarse_target) * self._coarse_progress
                       if turning else max(0.0, target - (self._coarse_drive_end - time.monotonic())))
            return {
                "kind": "rotation" if turning else "timed_coarse",
                "direction": self._coarse_command if turning else "FORWARD (time estimate)",
                "fixed_target": True, "target_value": target,
                "current_value": current, "remaining_value": max(0.0, target-current),
                "progress_ratio": max(0.0, min(1.0, current/max(target, 1e-9))),
                "unit": "deg" if turning else "s",
            }
        if self.state == "SEARCH_SWEEP":
            return {
                "kind": "rotation",
                "direction": "RIGHT",
                "fixed_target": False,
                "description": "until pallet detected (no fixed angle)",
            }

        rotation_states = {
            "FACE_ROTATE", "RECENTER_ROTATE", "WAYPOINT_TURN", "FINAL_ROTATE",
        }
        if self.state in rotation_states and self._rotation.active:
            plan = self._rotation.plan
            signed_target = float(
                plan.target_deg if plan is not None
                else self._rotation.start_error_deg
            )
            target = abs(signed_target)
            current_error = self._rotation.last_error_deg
            completed = 0.0
            if current_error is not None:
                completed = max(
                    0.0,
                    wrap_180(
                        float(current_error) - self._rotation.start_error_deg
                    ) * self._rotation.expected_delta_sign,
                )
            remaining = max(0.0, target - completed)
            return {
                "kind": "rotation",
                "direction": (
                    "RIGHT" if self._rotation.command == "ROT_RIGHT" else "LEFT"
                ),
                "fixed_target": True,
                "target_value": target,
                "current_value": completed,
                "remaining_value": remaining,
                "progress_ratio": (
                    0.0 if target <= 0.0 else min(1.0, completed / target)
                ),
                "unit": "deg",
            }

        if self.state == "INSERT_DRIVE":
            elapsed = max(0.0, time.monotonic() - self._translation_started_mono)
            target = self._insert_hold_sec
            return {
                "kind": "timed_insertion", "direction": "FORWARD (MODEL TIME)",
                "fixed_target": True, "target_value": target,
                "current_value": min(elapsed, target),
                "remaining_value": max(0.0, target - elapsed),
                "progress_ratio": min(1.0, elapsed / max(target, 1e-9)), "unit": "s",
            }
        translation_states = {"STANDOFF_MOVE", "WAYPOINT_DRIVE"}
        if self.state in translation_states and self._translation_target_m is not None:
            target = abs(float(self._translation_target_m))
            pose = self._last_valid_pose
            completed = 0.0 if pose is None else self._translation_progress(pose)
            remaining = max(0.0, target - completed)
            command = self._translation_command or "FWD"
            return {
                "kind": "translation",
                "direction": "BACKWARD" if command == "BACK" else "FORWARD",
                "fixed_target": True,
                "target_value": target,
                "current_value": completed,
                "remaining_value": remaining,
                "progress_ratio": (
                    0.0 if target <= 0.0 else min(1.0, completed / target)
                ),
                "unit": "m",
            }

        return None

    def rotation_diagnostic_hud_lines(self) -> List[Line]:
        """Persistent HUD lines for empirical rotation-inertia collection."""
        diag = self.rotation_diagnostics
        rate = diag["stop_measured_rate_deg_s"]
        if rate is None:
            return [(
                "[ROT INERTIA] last STOP speed/coast: N/A",
                COLOR_META,
            )]
        command = diag["command"] or "ROT"
        coast = float(diag["coast_deg"] or 0.0)
        peak = float(diag["coast_peak_abs_deg"] or 0.0)
        return [
            (
                f"[ROT INERTIA] last STOP: {command} | "
                f"PnP speed={abs(float(rate)):.2f} deg/s",
                COLOR_YAW,
            ),
            (
                f"post-STOP rotation={coast:+.2f} deg "
                f"(same-dir + / rebound -), peak={peak:.2f} deg",
                COLOR_YAW,
            ),
        ]

    def reset(self) -> None:
        self._coarse_reset()
        self.state = "PRECHECK"
        self._pipeline_started_mono = time.monotonic()
        self._state_entered_mono = self._pipeline_started_mono
        self._state_deadline_mono: Optional[float] = None
        self._invalid_since_mono: Optional[float] = None
        self._invalid_frames = 0
        self._last_valid_pose: Optional[VisualPose] = None
        self._approved_stopped_observation = None
        self._last_valid_center: Optional[float] = None
        self._last_valid_margin: Optional[float] = None
        self._last_valid_vision_meta: Optional[Dict] = None
        self._samples: List[Tuple[VisualPose, float, float]] = []
        self._search_started_mono = self._pipeline_started_mono
        self._face_attempts = 0
        self._initial_visibility_complete = False
        self._initial_visibility_deadline = None
        self._initial_visible_snapshot = None
        self._initial_snapshot_used = False
        self._initial_visibility_stop_mono = None
        self._initial_rotation_reference = None
        self._standoff_corrections = 0
        self._correction_cycles = 0
        self._no_progress_cycles = 0
        self._last_goal_distance_m: Optional[float] = None
        self._forward_used_m = 0.0
        self._alignment_started_mono: Optional[float] = None
        self._direction_changes = 0
        self._last_rotation_command: Optional[str] = None
        self._rotation_mode: Optional[str] = None
        self._rotation_target_yaw_deg: Optional[float] = None
        self._rotation_stop_error_deg: Optional[float] = None
        self._rotation_stop_mode: Optional[str] = None
        self._rotation_stop_command: Optional[str] = None
        self._rotation_stop_reason: Optional[str] = None
        self._rotation_stop_measured_rate_deg_s: Optional[float] = None
        self._rotation_stop_effective_rate_deg_s: Optional[float] = None
        self._rotation_stop_measurement_mono: Optional[float] = None
        self._rotation_stop_expected_delta_sign: float = 0.0
        self._rotation_coast_deg: Optional[float] = None
        self._rotation_coast_peak_abs_deg: Optional[float] = None
        self._rotation_coast_current_error_deg: Optional[float] = None
        self._rotation_coast_tracking = False
        self._rotation_time_scale = 1.0
        self._session_slope = SessionSlope(
            cfg.ROT_ADAPTIVE_SLOPE_ALPHA, cfg.ROT_ADAPTIVE_SLOPE_MIN_MULTIPLIER,
            cfg.ROT_ADAPTIVE_SLOPE_MAX_MULTIPLIER, cfg.ROT_ADAPTIVE_SLOPE_MIN_ACTIVE_SEC,
        )
        self._rotation_actual_hold_sec = None
        self._rotation_active_time_scale = 1.0
        self._rotation_adaptive_result: Dict[str, object] = {}
        self._waypoint: Optional[Waypoint] = None
        self._translation_command: Optional[str] = None
        self._translation_started_mono: Optional[float] = None
        self._translation_deadline_mono: Optional[float] = None
        self._translation_start_z_m: Optional[float] = None
        self._translation_start_rot: Optional[Tuple[float, float]] = None
        self._translation_target_m: Optional[float] = None
        self._translation_motion_hits = 0
        self._translation_motion_confirmed = False
        self._translation_stop_info: Dict[str, object] = {}
        self._settle_started_mono: Optional[float] = None
        self._settle_evaluation_started_mono: Optional[float] = None
        self._insert_remaining_m = 0.0
        self._insert_accepted_z_m = None
        self._insert_hold_sec = None
        self._insertion_alignment_attempts = 0
        self._insertion_alignment_entered = False
        self._insertion_fine_started_mono = None
        self._insertion_started_mono: Optional[float] = None
        self._failure_reason: Optional[str] = None
        self._failure_state: Optional[str] = None
        self._visual_recovery_origin_state: Optional[str] = None
        self._pose_filter.reset()
        self._rotation.reset()
        self._exec("STOP")

    # ------------------------------------------------------------------ basics
    @staticmethod
    def _initial_front_visible(pose, vision_meta) -> bool:
        if pose is None:
            return False
        margin = None if vision_meta is None else vision_meta.get("bbox_margin_norm")
        if margin is not None and (
            not math.isfinite(float(margin)) or float(margin) < cfg.BBOX_SAFE_MARGIN_NORM
        ):
            return False
        return action_keeps_front_visible(pose, 0.0, 0.0, vision_meta)

    def _start_initial_visibility_sweep(self, now, lines) -> None:
        # No absolute heading sensor is available. This is a timed search
        # budget extrapolated from the endpoint model, not measured 360 deg.
        if self._initial_visibility_deadline is None:
            duration = (cfg.ROTATION_RESPONSE.startup_delay_sec
                        + 360.0 / cfg.ROTATION_RESPONSE.max_rate_deg_s)
            self._initial_visibility_deadline = now + duration
        remaining = max(0.0, self._initial_visibility_deadline - now)
        self._trace_begin(
            "v4_initial_visibility_sweep", "ROT_RIGHT",
            timeout_sec=remaining, nominal_search_angle_deg=360.0,
            angle_basis="time estimate; no absolute heading feedback",
        )
        self._set_state("INITIAL_VISIBILITY_SWEEP")
        self._initial_visibility_sweep_step(None, None, now, lines)

    def _start_initial_direction_correction(self, pose, vision_meta, now, lines) -> None:
        center, _margin = self._vision_values(vision_meta, pose)
        if not math.isfinite(center) or abs(center) < 1e-6:
            self._fail("initial cropped detection has no horizontal correction direction", lines)
            return
        if self._initial_visibility_deadline is None:
            self._initial_visibility_deadline = now + cfg.SEARCH_TOTAL_TIMEOUT_SEC
        if now >= self._initial_visibility_deadline:
            self._fail("initial visibility correction time budget exhausted", lines)
            return
        turn = math.copysign(min(cfg.ROT_MAX_WAYPOINT_TURN_DEG,
                                max(cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG, abs(center))), center)
        self._initial_rotation_reference = (pose, None if vision_meta is None else dict(vision_meta))
        if not self._begin_rotation(
            "INITIAL_POSE_ROTATE", "initial_direction", turn, pose,
            target_yaw_deg=wrap_180(pose.yaw_deg - turn),
        ):
            self._fail("initial directional correction is not commandable", lines)
            return
        lines.append((f"[INITIAL CORRECTION] detected bearing={center:+.2f}deg, turn={turn:+.2f}deg", COLOR_META))

    def _initial_visibility_sweep_step(self, pose, vision_meta, now, lines) -> None:
        remaining = self._initial_visibility_deadline - now
        if remaining <= 0.0:
            self._fail("initial visibility right sweep time budget exhausted", lines)
            return
        if self._initial_front_visible(pose, vision_meta):
            self._initial_visible_snapshot = (
                pose, None if vision_meta is None else dict(vision_meta),
            )
            self._initial_snapshot_used = False
            self._initial_visibility_stop_mono = now
            self._exec("STOP")
            self._trace_end(stop_reason="full_front_visible")
            self._pose_filter.reset()
            self._set_state("ACQUIRE_VERIFY", cfg.STOP_MIN_SETTLE_SEC + cfg.ACQUIRE_VERIFY_TIMEOUT_SEC)
            lines.append(("[INITIAL SEARCH] full front visible -> STOP/verify", COLOR_META))
            return
        if pose is not None:
            self._exec("STOP")
            self._trace_end(stop_reason="cropped_detection_direction_correction")
            self._start_initial_direction_correction(pose, vision_meta, now, lines)
            return
        self.execu.exec("ROT_RIGHT")
        self.status.start_timed("ROT_RIGHT", remaining)
        lines.append((f"[INITIAL SEARCH] right sweep, {remaining:.1f}s remaining", COLOR_STATUS_TRK))

    def _recover_initial_snapshot(self, now, lines) -> bool:
        snapshot = self._initial_visible_snapshot
        if snapshot is None or self._initial_snapshot_used:
            return False
        self._initial_snapshot_used = True
        pose, meta = snapshot
        if now - self._initial_visibility_stop_mono > (
            cfg.STOP_MIN_SETTLE_SEC + cfg.ACQUIRE_VERIFY_TIMEOUT_SEC
        ):
            return False
        center, _margin = self._vision_values(meta, pose)
        turn = safe_recenter_turn(pose, center, meta)
        self._initial_rotation_reference = snapshot
        if turn is None or not self._begin_rotation(
            "INITIAL_POSE_ROTATE", "initial_recover", turn, pose,
            target_yaw_deg=wrap_180(pose.yaw_deg - turn),
        ):
            return False
        lines.append((f"[INITIAL RECOVER] saved full-front pose, turn={turn:+.2f}deg", COLOR_META))
        return True

    def _exec(self, command: str) -> None:
        if command != "STOP":
            self._approved_stopped_observation = None
        self.execu.exec(command)
        self.status.start_timed(command, 0.0)

    def _set_state(self, state: str, deadline_sec: Optional[float] = None) -> None:
        # Only adjacent, stationary decisions may share an approved window.
        if state not in self.STOPPED_DECISION_STATES:
            self._approved_stopped_observation = None
        # Display context only: keep the interrupted phase through reacquisition.
        if state == "RECOVER_VISUAL" and self.state != "RECOVER_VISUAL":
            self._visual_recovery_origin_state = self.state
        elif state not in {"RECOVER_VISUAL", "ACQUIRE_VERIFY", "FAILED"}:
            self._visual_recovery_origin_state = None
        self.state = state
        self._state_entered_mono = time.monotonic()
        self._state_deadline_mono = (
            None if deadline_sec is None
            else self._state_entered_mono + max(0.0, float(deadline_sec))
        )
        self._samples.clear()

    def _begin_settle(self, state: str, now: Optional[float] = None) -> None:
        """Start brake guard; stability gets a fresh timer after the guard."""
        self._settle_started_mono = time.monotonic() if now is None else float(now)
        self._settle_evaluation_started_mono = None
        self._set_state(
            state, cfg.STOP_MIN_SETTLE_SEC + cfg.STOP_MAX_SETTLE_SEC,
        )

    def _trace_begin(self, kind: str, command: str, **params) -> None:
        if self.tracer is None:
            return
        if getattr(self.tracer, "_open_step", None) is not None:
            self.tracer.step_end(interrupted_by=kind)
        self.tracer.step_begin(kind, command, **params)

    def _trace_end(self, **result) -> None:
        if self.tracer is not None:
            self.tracer.step_end(**result)

    def _failure_hud_lines(self) -> List[Line]:
        stage = self._failure_state or "UNKNOWN"
        reason = self._failure_reason or "unspecified failure"
        wrapped_reason = textwrap.wrap(
            reason, width=68, break_long_words=True, break_on_hyphens=False,
        ) or [reason]
        result = [(f"[V4 FAILED] stage: {stage}", COLOR_ALERT)]
        result.extend((f"reason: {part}", COLOR_ALERT) for part in wrapped_reason)
        return result

    def _fail(self, reason: str, lines: List[Line]) -> None:
        failed_state = self.state
        failure_reason = str(reason).strip() or "unspecified failure"
        self._exec("STOP")
        self._trace_end(error=failure_reason, failed_state=failed_state)
        self._failure_reason = failure_reason
        self._failure_state = failed_state
        if self.tracer is not None:
            log_failure = getattr(self.tracer, "log_failure", None)
            if callable(log_failure):
                log_failure(failed_state, failure_reason)
        print(
            f"[FSM v4 FAILED] stage={failed_state} reason={failure_reason}",
            flush=True,
        )
        self._set_state("FAILED")
        lines.extend(self._failure_hud_lines())

    @staticmethod
    def _extract_raw_pose(
        yaw_smooth, offset_smooth, dist_z, vision_meta,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        raw_yaw = None if vision_meta is None else vision_meta.get("yaw_raw_deg")
        raw_x = None if vision_meta is None else vision_meta.get("pos_x_m")
        raw_z = None if vision_meta is None else vision_meta.get("pos_z_m")
        if raw_yaw is None:
            raw_yaw = yaw_smooth
        if raw_x is None and offset_smooth is not None:
            try:
                raw_x = offset_smooth[0]
            except (TypeError, IndexError):
                raw_x = None
        if raw_z is None:
            raw_z = dist_z
        try:
            return float(raw_yaw), float(raw_x), float(raw_z)
        except (TypeError, ValueError):
            return None, None, None

    def _observe_pose(
        self, det_ok, yaw_smooth, offset_smooth, dist_z, vision_meta,
        now_mono: float,
    ) -> Tuple[Optional[VisualPose], str]:
        if not det_ok:
            return None, "detection unavailable"
        yaw, x_m, z_m = self._extract_raw_pose(
            yaw_smooth, offset_smooth, dist_z, vision_meta,
        )
        if yaw is None or x_m is None or z_m is None:
            return None, "raw PnP pose unavailable"
        measured = now_mono
        if vision_meta is not None:
            try:
                measured = float(vision_meta.get("measurement_mono", now_mono))
            except (TypeError, ValueError):
                measured = now_mono
        age = now_mono - measured + cfg.SENSOR_PIPELINE_LATENCY_SEC
        if age > cfg.MAX_MEASUREMENT_AGE_SEC:
            return None, f"stale PnP age={age:.3f}s"
        return self._pose_filter.update(yaw, x_m, z_m, measured)

    @staticmethod
    def _vision_values(
        vision_meta: Optional[Dict], pose: VisualPose,
    ) -> Tuple[float, float]:
        center = None if vision_meta is None else vision_meta.get("center_bearing_deg")
        margin = None if vision_meta is None else vision_meta.get("bbox_margin_norm")
        if center is None:
            center = math.degrees(math.atan2(pose.pallet_x_m, pose.pallet_z_m))
        if margin is None:
            margin = 0.0
        return float(center), float(margin)

    def _stable_observation(
        self, pose: VisualPose, center: float, margin: float,
        required_frames: Optional[int] = None,
    ) -> Optional[Tuple[VisualPose, float, float]]:
        self._samples.append((pose, float(center), float(margin)))
        required = max(
            2,
            cfg.STABLE_POSE_FRAMES if required_frames is None else required_frames,
        )
        if len(self._samples) < required:
            return None
        window = self._samples[-required:]
        yaws = [item[0].yaw_deg for item in window]
        xs = [item[0].pallet_x_m for item in window]
        zs = [item[0].pallet_z_m for item in window]
        yaw_anchor = yaws[0]
        yaw_offsets = [wrap_180(yaw - yaw_anchor) for yaw in yaws]
        median_yaw = wrap_180(yaw_anchor + median(yaw_offsets))
        median_x = float(median(xs))
        median_z = float(median(zs))
        inliers = [
            item for item in window
            if (
                abs(wrap_180(item[0].yaw_deg - median_yaw))
                <= cfg.STABLE_YAW_MEDIAN_TOL_DEG
                and abs(item[0].pallet_x_m - median_x)
                <= cfg.STABLE_X_MEDIAN_TOL_M
                and abs(item[0].pallet_z_m - median_z)
                <= cfg.STABLE_Z_MEDIAN_TOL_M
            )
        ]
        min_inliers = math.ceil(required * cfg.STABLE_MIN_INLIER_RATIO)
        if len(inliers) < min_inliers:
            self._samples[:] = [self._samples[-1]]
            return None
        inlier_yaw_offsets = [
            wrap_180(item[0].yaw_deg - median_yaw) for item in inliers
        ]
        stable_yaw = wrap_180(median_yaw + median(inlier_yaw_offsets))
        stable_x = float(median(item[0].pallet_x_m for item in inliers))
        stable_z = float(median(item[0].pallet_z_m for item in inliers))
        stable_pose = self._pose_filter.seed_vehicle(
            stable_yaw, stable_x, stable_z, pose.measurement_mono,
        )
        stable_center = float(median(item[1] for item in inliers))
        stable_margin = float(min(item[2] for item in inliers))
        self._samples.clear()
        return stable_pose, stable_center, stable_margin

    def _recover_visual(self, reason: str, lines: List[Line]) -> None:
        self._exec("STOP")
        self._set_state("RECOVER_VISUAL", cfg.MAX_PNP_LOSS_SEC)
        lines.append((f"[V4] {reason} -> STOP/visual recovery", COLOR_ALERT))

    def _stopped_decision_observation(self, pose, center, margin, lines):
        """Reuse a fresh approved window only while new observations still agree."""
        approved = getattr(self, "_approved_stopped_observation", None)
        if approved is not None:
            previous, previous_center, previous_margin = approved
            now = time.monotonic()
            age = now - previous.measurement_mono
            current_age = now - pose.measurement_mono
            checks = (
                0.0 <= age <= cfg.MAX_MEASUREMENT_AGE_SEC,
                0.0 <= current_age <= cfg.MAX_MEASUREMENT_AGE_SEC,
                pose.measurement_mono >= previous.measurement_mono,
                abs(wrap_180(pose.yaw_deg - previous.yaw_deg)) <= cfg.STABLE_YAW_MEDIAN_TOL_DEG,
                abs(pose.pallet_x_m - previous.pallet_x_m) <= cfg.STABLE_X_MEDIAN_TOL_M,
                abs(pose.pallet_z_m - previous.pallet_z_m) <= cfg.STABLE_Z_MEDIAN_TOL_M,
                abs(wrap_180(center - previous_center)) <= cfg.STABLE_YAW_MEDIAN_TOL_DEG,
                math.isfinite(margin) and margin >= previous_margin,
            )
            if all(checks):
                lines.append((f"[OBSERVE REUSE] approved stopped pose age={age:.3f}s", COLOR_META))
                return approved
            self._approved_stopped_observation = None
        stable = self._stable_observation(pose, center, margin)
        if stable is not None:
            self._approved_stopped_observation = stable
        return stable

    # -------------------------------------------------------------- rotation
    def _rotation_error(self, pose: VisualPose, center: float) -> float:
        if self._rotation_mode in {"face", "recenter"}:
            return wrap_180(center)
        if self._rotation_target_yaw_deg is None:
            return wrap_180(pose.yaw_deg)
        return wrap_180(pose.yaw_deg - self._rotation_target_yaw_deg)

    def _continue_accepted_waypoint_turn(
        self, pose: VisualPose, settled_error: float,
        vision_meta: Optional[Dict], lines: List[Line],
    ) -> bool:
        """Consume a settled waypoint turn within tolerance without re-turning."""
        if (self.state != "WAYPOINT_TURN_SETTLE"
                or not math.isfinite(settled_error)
                or abs(settled_error) > cfg.WAYPOINT_SETTLED_YAW_TOL_DEG):
            return False
        self._rotation.reset()
        if 0.0 < pose.pallet_z_m <= cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M:
            self._set_state("FINAL_POSE_LOCK")
            return True
        if staging_position_reached(pose):
            self._set_state("FINAL_POSE_LOCK")
            return True
        distance = (safe_straight_continuation_m(
            pose, self._waypoint.forward_m,
            cfg.FWD_MAX_TOTAL_CORRECTION_M - self._forward_used_m, vision_meta,
        ) if self._waypoint is not None else 0.0)
        if distance <= 0.0:
            self._set_state("STAGING_PLAN")
            lines.append(("[TURN ACCEPT] no safe straight continuation -> replan", COLOR_META))
            return True
        self._last_goal_distance_m = goal_vector_vehicle(pose)[2]
        self._begin_translation(
            "WAYPOINT_DRIVE", "FWD", pose, distance, "v4_waypoint_forward",
        )
        lines.append((
            f"[TURN ACCEPT] error={settled_error:+.2f}deg within "
            f"+/-{cfg.WAYPOINT_SETTLED_YAW_TOL_DEG:.2f}deg -> "
            f"safe forward {distance:.3f}m", COLOR_META,
        ))
        return True

    def _insertion_fine_timed_out(self, now, lines) -> bool:
        started = getattr(self, "_insertion_fine_started_mono", None)
        if (started is not None and now - started >= cfg.INSERT_FINE_TIMEOUT_SEC
                and self.state not in {"READY_TO_INSERT", "INSERT_DRIVE", "INSERT_SETTLE", "DONE", "FAILED"}):
            self._exec("STOP")
            self._fail("insertion fine alignment exceeded 120 seconds", lines)
            return True
        return False

    def _accept_insertion(self, pose, lines) -> bool:
        check = check_insertion_sweep(
            pose, enforce_entry_distance=not getattr(self, "_insertion_alignment_entered", False))
        if not check.ok or abs(pose.yaw_deg) > cfg.FINAL_YAW_TOL_DEG:
            self._fail("insertion rejected: " + (check.reason if not check.ok else "yaw limit"), lines)
            return False
        z = float(pose.pallet_z_m)
        target = z - cfg.INSERT_CAMERA_Z_REMAINDER_M
        if (not math.isfinite(z) or z <= 0.0 or target <= 0.0
                or (not getattr(self, "_insertion_alignment_entered", False)
                    and z > cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M)):
            self._fail("invalid accepted camera-Z insertion distance", lines)
            return False
        self._insert_accepted_z_m = z
        self._insert_remaining_m = target
        self._set_state("READY_TO_INSERT")
        lines.append((f"[INSERT TARGET] accepted Z={z:.3f}m - "
                      f"{cfg.INSERT_CAMERA_Z_REMAINDER_M:.3f}m = {target:.3f}m", COLOR_META))
        return True

    def _route_start_distance(self, pose, vision_meta, now, lines) -> bool:
        """Plan from inside the standoff; use rotation-only alignment nearby."""
        if not math.isfinite(pose.pallet_z_m) or pose.pallet_z_m <= 0.0:
            return False
        if (pose.pallet_z_m < cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M
                or getattr(self, "_insertion_alignment_entered", False)):
            return self._near_insertion_step(pose, vision_meta, lines)
        if pose.pallet_z_m <= cfg.SAFETY_STANDOFF_Z_M + cfg.SAFETY_STANDOFF_BAND_M:
            self._set_state("STAGING_PLAN")
            if self._alignment_started_mono is None:
                self._alignment_started_mono = now
            self._last_goal_distance_m = None
            lines.append((f"[STANDOFF] z={pose.pallet_z_m:.3f}m -> staging", COLOR_META))
            return True
        return False

    def _near_insertion_step(self, pose, vision_meta, lines) -> bool:
        """Latch entry distance, then align using fork geometry after turns."""
        if not math.isfinite(pose.pallet_z_m) or pose.pallet_z_m <= 0.0:
            return False
        if 0.0 < pose.pallet_z_m <= cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M:
            self._insertion_alignment_entered = True
        if not getattr(self, "_insertion_alignment_entered", False):
            return False
        if self._insertion_fine_timed_out(time.monotonic(), lines):
            return True
        if geometry_error():
            self._fail("insertion geometry: " + geometry_error(), lines)
            return True
        turn = insertion_alignment_turn(pose, vision_meta, alignment_entered=True)
        if turn == 0.0:
            self._insertion_fine_started_mono = None
            if not self._accept_insertion(pose, lines):
                return True
            lines.append((f"[INSERT CHECK] full nine-block sweep clear, yaw={pose.yaw_deg:+.2f}deg", COLOR_STATUS_OK))
        elif turn is None:
            self._fail(
                f"no commandable visibility-safe insertion rotation "
                f"(minimum {cfg.INSERT_FINE_MIN_TURN_DEG:.2f}deg): "
                f"{check_insertion_sweep(pose, enforce_entry_distance=False).reason}", lines,
            )
        elif (abs(turn) >= cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG
              and self._insertion_fine_started_mono is None
              and self._insertion_alignment_attempts >= cfg.INSERT_ALIGNMENT_MAX_CORRECTIONS):
            self._fail("insertion alignment correction budget exceeded", lines)
        else:
            fine = abs(turn) < cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG
            if fine and self._insertion_fine_started_mono is None:
                self._insertion_fine_started_mono = time.monotonic()
            self._insertion_alignment_attempts += 1
            if not self._begin_rotation(
                "FINAL_ROTATE", "insert_align", turn, pose,
                target_yaw_deg=wrap_180(pose.yaw_deg - turn),
            ):
                self._fail("insertion alignment rotation is not commandable", lines)
            else:
                lines.append((f"[INSERT ALIGN] turn={turn:+.2f}deg; settle then recheck", COLOR_META))
        return True

    def _capture_rotation_stop(
        self, error_deg: float, update, pose: VisualPose, reason: str,
    ) -> None:
        """Freeze PnP speed at STOP and start measuring post-STOP rotation."""
        self._rotation_stop_error_deg = wrap_180(error_deg)
        # Monotonic controller elapsed at STOP, not plan duration or settle wait.
        self._rotation_actual_hold_sec = float(update.elapsed_sec)
        self._rotation_stop_mode = self._rotation_mode
        self._rotation_stop_command = self._rotation.command
        self._rotation_stop_reason = str(reason)
        self._rotation_stop_measured_rate_deg_s = float(
            update.measured_error_rate_deg_s
        )
        self._rotation_stop_effective_rate_deg_s = float(update.error_rate_deg_s)
        self._rotation_stop_measurement_mono = float(pose.measurement_mono)
        self._rotation_stop_expected_delta_sign = float(
            self._rotation.expected_delta_sign
        )
        self._rotation_coast_deg = 0.0
        self._rotation_coast_peak_abs_deg = 0.0
        self._rotation_coast_current_error_deg = self._rotation_stop_error_deg
        self._rotation_coast_tracking = True
        if self.tracer is not None:
            logger = getattr(self.tracer, "log_rotation_stop", None)
            if callable(logger):
                logger(**self.rotation_diagnostics)

    def _update_rotation_coast(self, pose: VisualPose, center: float) -> None:
        """Update signed coast angle from each valid post-STOP PnP pose."""
        if (
            not self._rotation_coast_tracking
            or self._rotation_stop_error_deg is None
            or self._rotation_stop_expected_delta_sign == 0.0
        ):
            return
        current_error = self._rotation_error(pose, center)
        coast = (
            wrap_180(current_error - self._rotation_stop_error_deg)
            * self._rotation_stop_expected_delta_sign
        )
        self._rotation_coast_current_error_deg = current_error
        self._rotation_coast_deg = float(coast)
        self._rotation_coast_peak_abs_deg = max(
            float(self._rotation_coast_peak_abs_deg or 0.0), abs(float(coast)),
        )

    def _rotation_trace_result(self) -> Dict[str, object]:
        return {
            "stop_measured_rate_deg_s": self._rotation_stop_measured_rate_deg_s,
            "stop_effective_rate_deg_s": self._rotation_stop_effective_rate_deg_s,
            "post_stop_rotation_deg": self._rotation_coast_deg,
            "post_stop_peak_abs_deg": self._rotation_coast_peak_abs_deg,
            "rotation_stop_reason": self._rotation_stop_reason,
            "adaptive_applied_time_scale": self._rotation_active_time_scale,
            **self._rotation_adaptive_result,
        }

    def _update_rotation_adaptation(self, settled_error_deg: float) -> None:
        """Learn the next active-time scale from this settled rotation."""

        plan = self._rotation.plan
        if (cfg.ROT_ADAPTIVE_SLOPE_ENABLED
                and getattr(self._rotation.response, 'endpoint_only', False)):
            self._rotation_adaptive_result = {}
            actual_hold = self._rotation_actual_hold_sec
            self._rotation_actual_hold_sec = None  # consume each action once
            if (plan is not None and self._rotation_mode in {"waypoint", "final", "insert_align"}
                    and actual_hold is not None
                    and self._rotation_stop_reason in {"fitted_hold_elapsed", "target_crossed"}):
                actual = (wrap_180(settled_error_deg - self._rotation.start_error_deg)
                          * self._rotation.expected_delta_sign)
                self._rotation_adaptive_result = self._session_slope.observe(
                    self._rotation.command, cfg.ROTATION_RESPONSE.slope_deg_s,
                    self._rotation.response.startup_delay_sec, actual_hold, actual,
                    min_active_sec=(cfg.INSERT_FINE_MIN_ACTIVE_SEC
                                    if self._rotation_mode == "insert_align" else None),
                )
                self._rotation_adaptive_result.update(
                    slope_hold_overrun_sec=actual_hold - plan.hold_sec,
                    slope_overshoot_deg=max(0.0, actual - abs(plan.target_deg)),
                    slope_overshoot_equivalent_sec=(
                        max(0.0, actual - abs(plan.target_deg))
                        / self._rotation.response.slope_deg_s
                    ),
                )
            return
        if not cfg.ROT_ADAPTIVE_OVERSHOOT_ENABLED or plan is None:
            self._rotation_adaptive_result = {}
            return
        requested = abs(float(plan.target_deg))
        actual = max(
            0.0,
            wrap_180(settled_error_deg - self._rotation.start_error_deg)
            * self._rotation.expected_delta_sign,
        )
        previous_scale = self._rotation_active_time_scale
        next_scale = adaptive_overshoot_scale(previous_scale, requested, actual)
        overshoot = max(0.0, actual - requested)
        self._rotation_time_scale = next_scale
        self._rotation_adaptive_result = {
            "adaptive_requested_deg": requested,
            "adaptive_actual_deg": actual,
            "adaptive_overshoot_deg": overshoot,
            "adaptive_next_time_scale": next_scale,
        }

    def _begin_rotation(
        self, state: str, mode: str, error_deg: float, pose: VisualPose,
        target_yaw_deg: Optional[float] = None,
    ) -> bool:
        error = wrap_180(error_deg)
        fine_allowed = mode == "insert_align"
        minimum = cfg.INSERT_FINE_MIN_TURN_DEG if fine_allowed else cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG
        if abs(error) < minimum:
            return False
        command = "ROT_RIGHT" if error > 0.0 else "ROT_LEFT"
        if self._last_rotation_command and command != self._last_rotation_command:
            self._direction_changes += 1
        self._last_rotation_command = command
        self._rotation_mode = mode
        self._rotation_target_yaw_deg = target_yaw_deg
        now = time.monotonic()
        # FACE/RECENTER servo on the front-face bearing, the others on pallet
        # yaw. Bearing also picks up the sideways camera swing, so the fitted
        # heading response has to be rescaled for those two modes.
        domain = "bearing" if mode in {"face", "recenter"} else "heading"
        self._rotation_actual_hold_sec = None
        self._rotation.start(
            error, command, now, pose.measurement_mono,
            domain=domain,
            range_m=math.hypot(pose.pallet_x_m, pose.pallet_z_m),
            adaptive_time_scale=self._rotation_time_scale,
            slope_multiplier=self._session_slope.multiplier(command),
            insertion_fine=fine_allowed,
        )
        self._rotation_active_time_scale = self._rotation.adaptive_time_scale
        self._rotation_adaptive_result = {}
        plan = self._rotation.plan
        # A generated response never extrapolates below its observed range.
        # Do not emit even a one-cycle ROT pulse when no reliable positive hold
        # exists for the requested angle.
        if plan is not None and not plan.feasible and plan.hold_sec <= 0.0:
            self._rotation.reset()
            return False
        plan_trace = {} if plan is None else {
            "fitted_hold_sec": plan.hold_sec,
            "fitted_hard_timeout_sec": plan.hard_timeout_sec,
            "fitted_stop_angle_deg": plan.predicted_stop_angle_deg,
            "fitted_stop_rate_deg_s": plan.predicted_stop_rate_deg_s,
            "fitted_coast_deg": plan.predicted_coast_deg,
            "fitted_total_deg": plan.predicted_total_deg,
            "fitted_settle_sec": plan.predicted_settle_sec,
            "fitted_angle_sd_deg": plan.angle_sd_deg,
            "fitted_feasible": plan.feasible,
            "fitted_note": plan.note,
            "fitted_domain": domain,
            "fitted_max_rate_deg_s": self._rotation.response.max_rate_deg_s,
            "adaptive_time_scale": self._rotation.adaptive_time_scale,
            "adaptive_base_hold_sec": self._rotation.base_plan_hold_sec,
            "adaptive_excluded_delay_sec": (
                self._rotation.adaptive_excluded_delay_sec
            ),
            "adaptive_base_active_sec": max(
                0.0,
                self._rotation.base_plan_hold_sec
                - self._rotation.adaptive_excluded_delay_sec,
            ),
            "adaptive_adjusted_active_sec": max(
                0.0,
                plan.hold_sec - self._rotation.adaptive_excluded_delay_sec,
            ),
        }
        if plan is not None and getattr(self._rotation.response, 'endpoint_only', False):
            plan_trace.update(
                fitted_prediction_kind='settled_endpoint_only',
                fitted_stop_angle_deg=None, fitted_stop_rate_deg_s=None,
                fitted_coast_deg=None, fitted_max_rate_deg_s=None,
                fitted_endpoint_slope_deg_s=self._rotation.response.slope_deg_s,
            )
        if mode in {"initial_recover", "initial_direction"}:
            plan_trace.update(
                reference_pose_source=("last fully visible pose before STOP"
                                       if mode == "initial_recover" else "current cropped detection"),
                reference_pose_measurement_mono=pose.measurement_mono,
                reference_pose_x_m=pose.pallet_x_m,
                reference_pose_z_m=pose.pallet_z_m,
                reference_pose_yaw_deg=pose.yaw_deg,
            )
        self._trace_begin(
            f"v4_{mode}_rotate", command,
            initial_error_deg=error,
            target_yaw_deg=target_yaw_deg,
            joystick_deflection=cfg.ROTATE_JOYSTICK_DEFLECTION,
            startup_delay_sec=self._rotation.response.startup_delay_sec,
            stop_lookahead_sec=cfg.ROT_STOP_LOOKAHEAD_SEC,
            **plan_trace,
        )
        budget = cfg.ROT_MAX_COMMAND_SEC if plan is None else min(
            cfg.ROT_MAX_COMMAND_SEC, plan.hard_timeout_sec,
        )
        self._set_state(state, budget)
        self.execu.exec(command)
        return True

    def _rotation_step(
        self, pose: VisualPose, center: float, vision_meta: Optional[Dict],
        settle_state: str, lines: List[Line],
    ) -> None:
        now = time.monotonic()
        error = self._rotation_error(pose, center)
        age = measurement_age(vision_meta, now)
        update = self._rotation.update(error, now, pose.measurement_mono, age)
        command = self._rotation.command or "STOP"
        face_centered_now = (
            self._rotation_mode == "face"
            and abs(center) <= cfg.FACE_CENTER_IMMEDIATE_STOP_TOL_DEG
        )
        if face_centered_now:
            self._exec("STOP")
            self._capture_rotation_stop(
                error, update, pose, "live_face_center",
            )
            self._begin_settle(settle_state, now)
            lines.append((
                f"[V4 FACE] immediate live-centre STOP, "
                f"front-centre={center:+.2f}deg",
                COLOR_META,
            ))
            return
        if update.timed_out:
            self._capture_rotation_stop(error, update, pose, "timeout")
            self._fail(
                f"{self._rotation_mode} rotation timeout; "
                f"motion={update.motion_confirmed}, error={error:+.2f}deg",
                lines,
            )
            return
        if (self._direction_changes > cfg.ROT_MAX_DIRECTION_CHANGES
                and self._rotation_mode != "insert_align"):
            self._capture_rotation_stop(
                error, update, pose, "direction_change_budget",
            )
            self._fail("rotation direction-change budget exceeded", lines)
            return
        crossed = self._rotation.start_error_deg * error <= 0.0
        if update.should_stop or (update.ready and crossed):
            self._exec("STOP")
            reason = update.stop_reason or (
                "predicted_margin" if update.should_stop else "target_crossed"
            )
            self._capture_rotation_stop(error, update, pose, reason)
            self._begin_settle(settle_state, now)
            if cfg.ROT_USE_FITTED_RESPONSE:
                lines.append((
                    f"[V4 ROT] STOP({reason}) error={error:+.2f}deg, "
                    f"coast margin={update.coast_margin_deg:.2f}deg, "
                    f"t={update.elapsed_sec:.2f}/{update.plan_hold_sec:.2f}s, "
                    f"live rate={update.measured_error_rate_deg_s:+.1f}deg/s",
                    COLOR_META,
                ))
            else:
                lines.append((
                    f"[V4 ROT] early STOP error={error:+.2f}deg, "
                    f"pred={update.predicted_error_deg:+.2f}deg, "
                    f"rate={update.error_rate_deg_s:+.1f}deg/s "
                    f"(live={update.measured_error_rate_deg_s:+.1f}, "
                    f"log={'on' if update.model_profile_active else 'off'})",
                    COLOR_META,
                ))
            return
        self.execu.exec(command)
        self.status.start_until(
            command, "predicted_rotation_error_deg",
            abs(update.predicted_error_deg), update.stop_margin_deg,
        )
        if cfg.ROT_USE_FITTED_RESPONSE:
            lines.append((
                f"[V4 ROT] {self._rotation_mode} {command}, "
                f"t={update.elapsed_sec:.2f}/{update.plan_hold_sec:.2f}s, "
                f"error={error:+.2f}deg, "
                f"coast margin={update.coast_margin_deg:.2f}deg, "
                f"live rate={update.measured_error_rate_deg_s:+.1f}deg/s",
                COLOR_STATUS_TRK,
            ))
        else:
            lines.append((
                f"[V4 ROT] {self._rotation_mode} {command}, "
                f"t={update.elapsed_sec:.2f}s, start="
                f"{update.startup_hits}/{cfg.ROT_STARTUP_CONFIRM_FRAMES}, "
                f"error={error:+.2f}deg, pred={update.predicted_error_deg:+.2f}deg, "
                f"rate={update.error_rate_deg_s:+.1f} "
                f"({'log+live' if update.model_profile_active else 'live'})",
                COLOR_STATUS_TRK,
            ))

    # ------------------------------------------------------------ translation
    def _begin_translation(
        self, state: str, command: str, pose: VisualPose, target_m: float,
        kind: str,
    ) -> None:
        now = time.monotonic()
        self._translation_command = command
        self._translation_started_mono = now
        self._translation_start_z_m = pose.pallet_z_m
        self._translation_start_rot = (
            pose.rot_x_pallet_m, pose.rot_z_pallet_m,
        )
        self._translation_target_m = abs(float(target_m))
        self._translation_motion_hits = 0
        self._translation_motion_confirmed = False
        self._translation_stop_info = {}
        if command == "FWD":
            fitted_duration = forward_seconds(target_m)
            duration = min(
                cfg.FWD_MAX_COMMAND_SEC,
                fitted_duration + cfg.FWD_TIMEOUT_MARGIN_SEC,
            )
        else:
            fitted_duration = backward_seconds(target_m)
            duration = min(
                cfg.BACK_MAX_COMMAND_SEC,
                fitted_duration + cfg.BACK_TIMEOUT_MARGIN_SEC,
            )
        self._translation_deadline_mono = now + duration
        self._trace_begin(
            kind, command, target_distance_m=target_m,
            fitted_duration_sec=fitted_duration,
            hard_timeout_sec=duration,
            forward_deflection=cfg.FORWARD_JOYSTICK_DEFLECTION,
            backward_deflection=cfg.BACKWARD_JOYSTICK_DEFLECTION,
        )
        self._set_state(state, duration)
        self.execu.exec(command)

    def _translation_progress(self, pose: VisualPose) -> float:
        if self._translation_start_rot is None:
            return 0.0
        return math.hypot(
            pose.rot_x_pallet_m - self._translation_start_rot[0],
            pose.rot_z_pallet_m - self._translation_start_rot[1],
        )

    def _translation_ready(self, pose: VisualPose, now: float) -> bool:
        if self._translation_started_mono is None or self._translation_start_z_m is None:
            return False
        command = self._translation_command
        directed = (
            self._translation_start_z_m - pose.pallet_z_m
            if command == "FWD"
            else pose.pallet_z_m - self._translation_start_z_m
        )
        if directed >= 0.02:
            self._translation_motion_hits += 1
        else:
            self._translation_motion_hits = 0
        if self._translation_motion_hits >= 2:
            self._translation_motion_confirmed = True
        delay = cfg.FWD_T0_SEC
        return self._translation_motion_confirmed and (
            now - self._translation_started_mono >= delay
        )

    def _stop_translation(
        self, settle_state: str, lines: List[Line], **info,
    ) -> None:
        self._exec("STOP")
        self._translation_stop_info = info
        self._begin_settle(settle_state)
        lines.append((f"[V4 MOVE] predictive STOP: {info}", COLOR_META))

    def _standoff_move_step(
        self, pose: VisualPose, vision_meta: Optional[Dict], lines: List[Line],
    ) -> None:
        now = time.monotonic()
        # The immediate endpoint uses this frame's raw PnP camera-Z.  The
        # predictive endpoint below keeps the filtered pose/velocity.
        live_z_m = pose.pallet_z_m
        if vision_meta is not None:
            try:
                candidate_z = float(vision_meta.get("pos_z_m", live_z_m))
                if math.isfinite(candidate_z) and candidate_z > 0.0:
                    live_z_m = candidate_z
            except (TypeError, ValueError):
                pass
        live_error = live_z_m - cfg.SAFETY_STANDOFF_Z_M
        if abs(live_error) <= cfg.SAFETY_STANDOFF_BAND_M:
            self._stop_translation(
                "STANDOFF_SETTLE", lines,
                live_target_immediate_stop=True,
                live_z_m=live_z_m,
                live_error_m=live_error,
            )
            return
        if self._translation_deadline_mono is None or now >= self._translation_deadline_mono:
            self._fail("standoff translation timeout", lines)
            return
        ready = self._translation_ready(pose, now)
        command = self._translation_command or "STOP"
        if not ready:
            self.execu.exec(command)
            # CAN may be disabled and visual motion may not have started yet,
            # but the HUD must still expose the logical FSM command.  Without
            # this update cmd_status stayed at the preceding STOP for the
            # entire startup/timeout interval.
            self.status.start_until(
                command, "motion_start_frames",
                self._translation_motion_hits, 2,
            )
            lines.append((
                f"[STANDOFF MOVE] startup {command}, z={pose.pallet_z_m:.3f}m, "
                f"hits={self._translation_motion_hits}/2",
                COLOR_STATUS_TRK,
            ))
            return
        age = measurement_age(vision_meta, now)
        horizon = age + cfg.FWD_STOP_LOOKAHEAD_SEC
        elapsed = now - (self._translation_started_mono or now)
        measured_closing_speed = (
            max(0.0, -pose.camera_vz_m_s)
            if command == "FWD"
            else max(0.0, pose.camera_vz_m_s)
        )
        predictive_advance, predictive_speed = bounded_forward_lookahead(
            elapsed, measured_closing_speed, horizon,
        )
        predicted_z = (
            pose.pallet_z_m - predictive_advance
            if command == "FWD"
            else pose.pallet_z_m + predictive_advance
        )
        error = pose.pallet_z_m - cfg.SAFETY_STANDOFF_Z_M
        predicted_error = predicted_z - cfg.SAFETY_STANDOFF_Z_M
        live_crossed = (
            self._translation_start_z_m is not None
            and (self._translation_start_z_m - cfg.SAFETY_STANDOFF_Z_M)
            * error <= 0.0
        )
        travelled = self._translation_progress(pose)
        predictive_ready = forward_predictive_stop_ready(
            travelled, self._translation_target_m or 0.0,
        )
        predicted_reached = (
            abs(predicted_error) <= cfg.SAFETY_STANDOFF_BAND_M
            or (
                self._translation_start_z_m is not None
                and (self._translation_start_z_m - cfg.SAFETY_STANDOFF_Z_M)
                * predicted_error <= 0.0
            )
        )
        if live_crossed or (predictive_ready and predicted_reached):
            self._stop_translation(
                "STANDOFF_SETTLE", lines,
                live_z_m=pose.pallet_z_m,
                predicted_z_m=predicted_z,
                error_m=error,
                travelled_m=travelled,
                predictive_advance_m=predictive_advance,
                predictive_speed_m_s=predictive_speed,
                predictive_gate_ready=predictive_ready,
            )
            return
        self.execu.exec(command)
        self.status.start_until(
            command, "predicted_standoff_error_m",
            abs(predicted_error), cfg.SAFETY_STANDOFF_BAND_M,
        )
        lines.append((
            f"[STANDOFF MOVE] {command} z={pose.pallet_z_m:.3f}m, "
            f"pred={predicted_z:.3f}m",
            COLOR_STATUS_TRK,
        ))

    def _forward_segment_step(
        self, pose: VisualPose, center: float, margin: float,
        vision_meta: Optional[Dict], settle_state: str, lines: List[Line],
        enforce_visibility: bool, enforce_staging_limit: bool = False,
    ) -> None:
        now = time.monotonic()
        travelled = self._translation_progress(pose)
        if enforce_staging_limit:
            lateral_error, longitudinal_error = staging_position_errors(pose)
            if longitudinal_error >= 0.0:
                self._stop_translation(
                    settle_state, lines,
                    live_staging_plane_stop=True,
                    lateral_error_m=lateral_error,
                    longitudinal_error_m=longitudinal_error,
                    travelled_m=travelled,
                )
                return
        if self._translation_deadline_mono is None or now >= self._translation_deadline_mono:
            self._fail(f"{self.state} forward timeout", lines)
            return
        ready = self._translation_ready(pose, now)
        target = self._translation_target_m or 0.0
        if enforce_visibility:
            gx, gz, _goal = goal_vector_vehicle(pose)
            bearing = math.degrees(math.atan2(gx, gz))
            if (
                cfg.DRIVE_HEADING_GUARD_ENABLED
                and abs(bearing) > cfg.DRIVE_HEADING_TOL_DEG
            ):
                self._stop_translation(
                    settle_state, lines,
                    heading_drift_deg=bearing, travelled_m=travelled,
                )
                return
        if not ready:
            self.execu.exec("FWD")
            # Keep the interface/diagram command independent from physical
            # CAN transmission and from visual motion confirmation.
            self.status.start_until(
                "FWD", "motion_start_frames",
                self._translation_motion_hits, 2,
            )
            lines.append((
                f"[{self.state}] startup, travel={travelled:.3f}/{target:.3f}m",
                COLOR_STATUS_TRK,
            ))
            return
        age = measurement_age(vision_meta, now)
        elapsed = now - (self._translation_started_mono or now)
        predictive_advance, predictive_speed = bounded_forward_lookahead(
            elapsed,
            pose.rotation_center_speed_m_s,
            age + cfg.FWD_STOP_LOOKAHEAD_SEC,
        )
        predicted = travelled + predictive_advance
        predictive_ready = forward_predictive_stop_ready(travelled, target)
        live_reached = travelled >= target
        if live_reached or (predictive_ready and predicted >= target):
            self._stop_translation(
                settle_state, lines,
                travelled_m=travelled, predicted_travel_m=predicted,
                target_m=target,
                predictive_advance_m=predictive_advance,
                predictive_speed_m_s=predictive_speed,
                predictive_gate_ready=predictive_ready,
            )
            return
        self.execu.exec("FWD")
        self.status.start_until("FWD", "predicted_travel_m", predicted, target)
        lines.append((
            f"[{self.state}] travel={travelled:.3f}m, "
            f"pred={predicted:.3f}/{target:.3f}m, "
            f"lookahead={predictive_advance:.3f}m, "
            f"gate={'on' if predictive_ready else 'off'}",
            COLOR_STATUS_TRK,
        ))

    # ---------------------------------------------------------------- settle
    def _settle_observation(
        self, pose: VisualPose, center: float, margin: float,
        lines: List[Line],
    ) -> Optional[Tuple[VisualPose, float, float]]:
        self._exec("STOP")
        now = time.monotonic()
        self._update_rotation_coast(pose, center)
        if self._settle_started_mono is None:
            self._settle_started_mono = now
            self._settle_evaluation_started_mono = None
        guard_elapsed = now - self._settle_started_mono
        if guard_elapsed < cfg.STOP_MIN_SETTLE_SEC:
            lines.append((
                f"[{self.state}] brake guard {guard_elapsed:.2f}/"
                f"{cfg.STOP_MIN_SETTLE_SEC:.2f}s",
                COLOR_STATUS_TRK,
            ))
            return None

        if self._settle_evaluation_started_mono is None:
            # Discard velocity inherited from the commanded motion.  The next
            # frames measure only post-guard vibration/pose movement.
            pose = self._pose_filter.seed_vehicle(
                pose.yaw_deg, pose.pallet_x_m, pose.pallet_z_m,
                pose.measurement_mono,
            )
            self._last_valid_pose = pose
            self._samples.clear()
            self._settle_evaluation_started_mono = now
            lines.append((
                f"[{self.state}] brake guard complete; velocity reset; "
                f"stability 0.00/{cfg.STOP_MAX_SETTLE_SEC:.2f}s",
                COLOR_META,
            ))
            return None

        evaluation_elapsed = now - self._settle_evaluation_started_mono
        # Do not classify stopped/moving from a consecutive-frame derivative.
        # At the current inference cadence a small PnP change becomes an
        # unrealistically large 100 ms velocity.  The robust 10-frame,
        # median-centred 8/10 pose window below is the sole settle decision.
        stable = self._stable_observation(pose, center, margin)
        if stable is not None:
            self._approved_stopped_observation = stable
            return stable
        if evaluation_elapsed >= cfg.STOP_MAX_SETTLE_SEC:
            self._fail(
                f"{self.state} did not collect {cfg.STABLE_POSE_FRAMES} "
                f"stable frames within {cfg.STOP_MAX_SETTLE_SEC:.2f}s",
                lines,
            )
        else:
            lines.append((
                f"[{self.state}] stable pose {len(self._samples)}/"
                f"{cfg.STABLE_POSE_FRAMES}; stability "
                f"{evaluation_elapsed:.2f}/{cfg.STOP_MAX_SETTLE_SEC:.2f}s",
                COLOR_STATUS_TRK,
            ))
        return None

    # -------------------------------------------------------- paused telemetry
    def observe_debug_telemetry(
        self, det_ok, detected_length, dist_z, yaw_smooth, offset_smooth,
        target_bearing_deg=None, vision_meta=None,
    ) -> None:
        """Update post-STOP diagnostics while manual FSM progression is paused."""
        del detected_length, target_bearing_deg
        if self.skip_detection or not self._rotation_coast_tracking:
            return
        now = time.monotonic()
        pose, _reason = self._observe_pose(
            det_ok, yaw_smooth, offset_smooth, dist_z, vision_meta, now,
        )
        if pose is None:
            return
        center, _margin = self._vision_values(vision_meta, pose)
        self._update_rotation_coast(pose, center)

    def _step_insertion_without_vision(self, now: float, lines: List[Line]) -> None:
        """One fitted-time command from the accepted distance; no pose feedback."""
        if self.state == "READY_TO_INSERT":
            self._exec("STOP")
            if not cfg.AUTO_INSERT_ENABLED:
                lines.append(("[READY TO INSERT] automatic insertion disabled", COLOR_STATUS_OK))
                return
            target = self._insert_remaining_m
            if (self._insert_accepted_z_m is None or not math.isfinite(target) or target <= 0.0):
                self._fail("insertion distance was not captured at acceptance", lines)
                return
            hold = forward_seconds(target)
            if (not math.isfinite(hold) or hold <= 0.0
                    or hold >= min(cfg.FWD_COMMAND_MAX_SEC, cfg.FWD_MAX_COMMAND_SEC)
                    or hold + cfg.STOP_MIN_SETTLE_SEC >= cfg.INSERT_MAX_TOTAL_SEC):
                self._fail("insertion duration exceeds fitted command limits", lines)
                return
            self._insert_hold_sec = hold
            self._insertion_started_mono = now
            self._translation_started_mono = now
            self._translation_deadline_mono = now + hold
            self._translation_target_m = target
            self._translation_command = "FWD"
            self._trace_begin(
                "v4_insert_timed", "FWD", target_distance_m=target,
                accepted_camera_z_m=self._insert_accepted_z_m,
                fitted_duration_sec=hold, control_basis="fitted_time_no_vision",
            )
            self._set_state("INSERT_DRIVE", hold)
            self.execu.exec("FWD")
            self.status.start_timed("FWD", hold)
            return
        if (self._insertion_started_mono is None
                or now - self._insertion_started_mono >= cfg.INSERT_MAX_TOTAL_SEC):
            self._fail("insertion total timeout", lines)
            return
        if self.state == "INSERT_DRIVE":
            if self._translation_deadline_mono is None:
                self._fail("insertion command deadline missing", lines)
                return
            remaining = self._translation_deadline_mono - now
            if remaining <= 0.0:
                self._exec("STOP")
                self._begin_settle("INSERT_SETTLE", now=now)
                lines.append(("[INSERT] fitted command complete -> STOP", COLOR_META))
            else:
                self.execu.exec("FWD")
                self.status.start_timed("FWD", remaining)
                lines.append((f"[INSERT] inference OFF; timed forward {remaining:.2f}s left", COLOR_META))
            return
        self._exec("STOP")
        if self._settle_started_mono is None:
            self._fail("insertion stop time missing", lines)
            return
        if now - self._settle_started_mono >= cfg.STOP_MIN_SETTLE_SEC:
            self._trace_end(control_basis="fitted_time_no_vision",
                            commanded_distance_m=self._translation_target_m,
                            command_duration_sec=self._insert_hold_sec,
                            actual_travel_m=None, stop_reason="fitted insertion time elapsed")
            self._insert_remaining_m = 0.0
            self._set_state("DONE")
        else:
            lines.append(("[INSERT] STOP guard; no PnP verification", COLOR_META))

    # ------------------------------------------------------------------ step
    def step(
        self, det_ok, detected_length, dist_z, yaw_smooth, offset_smooth,
        target_bearing_deg=None, vision_meta=None,
    ) -> List[Line]:
        del detected_length
        lines: List[Line] = []
        now = time.monotonic()
        if self._insertion_fine_timed_out(now, lines):
            return lines
        if (
            self.state not in {"DONE", "FAILED"}
            and now - self._pipeline_started_mono >= cfg.TOTAL_PIPELINE_TIMEOUT_SEC
        ):
            self._fail("total pipeline timeout", lines)
            return lines

        if self.state in COARSE_STATES:
            self._coarse_step(now, lines, det_ok, yaw_smooth, offset_smooth, dist_z, vision_meta)
            return lines

        if self.state in {"READY_TO_INSERT", "INSERT_DRIVE", "INSERT_SETTLE"}:
            self._step_insertion_without_vision(now, lines)
            return lines
        if self.state == "FAILED":
            self._exec("STOP")
            lines.extend(self._failure_hud_lines())
            return lines
        if self.state == "DONE":
            self._exec("STOP")
            lines.append(("[V4 DONE] timed insertion complete", COLOR_STATUS_OK))
            return lines

        pose, pose_reason = self._observe_pose(
            det_ok, yaw_smooth, offset_smooth, dist_z, vision_meta, now,
        )
        center = margin = None
        if pose is not None:
            center, margin = self._vision_values(vision_meta, pose)
            self._invalid_since_mono = None
            self._invalid_frames = 0
            self._last_valid_pose = pose
            self._last_valid_center = center
            self._last_valid_margin = margin
            self._last_valid_vision_meta = (
                None if vision_meta is None else dict(vision_meta)
            )
        if pose is None or center is None or margin is None:
            self._approved_stopped_observation = None

        if self.state == "PRECHECK":
            self._exec("STOP")
            self._set_state("SEARCH_SWEEP")
            self._search_started_mono = now
            lines.append(("[V4 PRECHECK] config valid -> SEARCH", COLOR_META))
            return lines

        if self.state == "INITIAL_VISIBILITY_SWEEP":
            # Searching must survive loss of PnP as the pallet leaves view.
            if det_ok and pose is None:
                self._exec("STOP")
                if now >= self._initial_visibility_deadline:
                    self._fail("detected pallet has no valid pose for initial correction", lines)
                return lines
            self._initial_visibility_sweep_step(pose, vision_meta, now, lines)
            return lines

        if self.state == "INITIAL_POSE_ROTATE":
            # One bounded recovery may use the saved observation while cropped.
            # Do not refresh its timestamp or use it to authorize translation.
            if not self._initial_front_visible(pose, vision_meta):
                pose, vision_meta = self._initial_rotation_reference
            center, _margin = self._vision_values(vision_meta, pose)
            self._rotation_step(pose, center, vision_meta, "INITIAL_POSE_SETTLE", lines)
            return lines

        if self.state == "INITIAL_POSE_SETTLE":
            self._exec("STOP")
            if now - self._settle_started_mono >= cfg.STOP_MIN_SETTLE_SEC:
                self._trace_end(stop_reason=self._rotation_stop_reason,
                                recovery_basis="saved full-front pose")
                self._rotation.reset()
                self._rotation_coast_tracking = False
                self._initial_visible_snapshot = None
                self._initial_rotation_reference = None
                self._initial_visibility_stop_mono = None
                self._pose_filter.reset()
                self._set_state("ACQUIRE_VERIFY", cfg.ACQUIRE_VERIFY_TIMEOUT_SEC)
            return lines

        if (self.state == "ACQUIRE_VERIFY"
                and getattr(self, "_initial_visible_snapshot", None) is not None):
            if now - self._initial_visibility_stop_mono < cfg.STOP_MIN_SETTLE_SEC:
                self._exec("STOP")
                self._samples.clear()
                return lines
            if not self._initial_front_visible(pose, vision_meta):
                if self._recover_initial_snapshot(now, lines):
                    return lines

        if self.state == "SEARCH_SWEEP":
            if pose is not None:
                if (not getattr(self, "_initial_visibility_complete", False)
                        and not self._initial_front_visible(pose, vision_meta)):
                    self._start_initial_direction_correction(pose, vision_meta, now, lines)
                    return lines
                self._exec("STOP")
                self._set_state("ACQUIRE_VERIFY", cfg.ACQUIRE_VERIFY_TIMEOUT_SEC)
                if not getattr(self, "_initial_visibility_complete", False):
                    self._initial_visible_snapshot = (
                        pose, None if vision_meta is None else dict(vision_meta),
                    )
                    self._initial_snapshot_used = False
                    self._initial_visibility_stop_mono = now
                    self._state_deadline_mono = now + cfg.STOP_MIN_SETTLE_SEC + cfg.ACQUIRE_VERIFY_TIMEOUT_SEC
                lines.append(("[SEARCH] candidate detected -> immediate STOP", COLOR_META))
                return lines
            if now - self._search_started_mono >= cfg.SEARCH_TOTAL_TIMEOUT_SEC:
                self._fail(
                    "no pallet detected during "
                    f"{cfg.SEARCH_TOTAL_TIMEOUT_SEC:g} s right-rotation search",
                    lines,
                )
                return lines
            if det_ok:
                self._exec("STOP")
                lines.append(("[SEARCH] detected pallet; waiting for correction pose", COLOR_META))
                return lines
            command = "ROT_RIGHT"
            self.execu.exec(command)
            self.status.start_timed(
                command,
                cfg.SEARCH_TOTAL_TIMEOUT_SEC - (now - self._search_started_mono),
            )
            lines.append(("[SEARCH] right rotation until detection (max 30 s)", COLOR_STATUS_TRK))
            return lines

        if self.state == "RECOVER_VISUAL":
            self._exec("STOP")
            if pose is not None:
                self._set_state("ACQUIRE_VERIFY", cfg.ACQUIRE_VERIFY_TIMEOUT_SEC)
                lines.append(("[RECOVER] PnP restored -> acquire verify", COLOR_META))
            elif self._state_deadline_mono is not None and now >= self._state_deadline_mono:
                self._fail("PnP recovery timeout", lines)
            else:
                lines.append((f"[RECOVER] waiting: {pose_reason}", COLOR_ALERT))
            return lines

        if pose is None or center is None or margin is None:
            if self.state == "ACQUIRE_VERIFY":
                self._exec("STOP")
                # Detection confirmation must be consecutive; one miss breaks
                # the current candidate window without ending monitor mode.
                self._samples.clear()
                if self._state_deadline_mono is not None and now >= self._state_deadline_mono:
                    self._pose_filter.reset()
                    self._set_state("SEARCH_SWEEP")
                    self._search_started_mono = now
                    lines.append(("[ACQUIRE] unstable candidate -> SEARCH", COLOR_ALERT))
                else:
                    lines.append((f"[ACQUIRE] candidate lost: {pose_reason}", COLOR_ALERT))
                return lines

            if self._invalid_since_mono is None:
                self._invalid_since_mono = now
                self._invalid_frames = 0
            self._invalid_frames += 1
            loss_sec = max(0.0, now - self._invalid_since_mono)

            if loss_sec >= cfg.VISION_LOSS_CONFIRM_SEC:
                self._recover_visual(
                    f"continuous PnP loss {loss_sec:.2f}s/"
                    f"{self._invalid_frames} frames ({pose_reason})",
                    lines,
                )
                return lines

            active_motion_states = {
                "FACE_ROTATE", "RECENTER_ROTATE", "WAYPOINT_TURN", "FINAL_ROTATE",
                "STANDOFF_MOVE", "WAYPOINT_DRIVE", "INSERT_DRIVE",
            }
            if (
                self.state in active_motion_states
                and self._last_valid_pose is not None
                and self._last_valid_center is not None
                and self._last_valid_margin is not None
            ):
                # Keep the already-issued bounded command alive during a short
                # dropout.  The old measurement timestamp is retained so the
                # normal predictive STOP logic sees the increasing data age.
                pose = self._last_valid_pose
                center = self._last_valid_center
                margin = self._last_valid_margin
                vision_meta = self._last_valid_vision_meta
                lines.append((
                    f"[VISION HOLD] miss={loss_sec:.2f}/"
                    f"{cfg.VISION_LOSS_CONFIRM_SEC:.2f}s, "
                    f"frames={self._invalid_frames}; last valid pose",
                    COLOR_ALERT,
                ))
            else:
                # No new planning or settle decision is made from a held pose.
                self._exec("STOP")
                lines.append((
                    f"[VISION WAIT] miss={loss_sec:.2f}/"
                    f"{cfg.VISION_LOSS_CONFIRM_SEC:.2f}s, "
                    f"frames={self._invalid_frames}; {pose_reason}",
                    COLOR_ALERT,
                ))
                return lines

        if self.state == "ACQUIRE_VERIFY":
            self._exec("STOP")
            if self._state_deadline_mono is not None and now >= self._state_deadline_mono:
                self._pose_filter.reset()
                self._set_state("SEARCH_SWEEP")
                self._search_started_mono = now
                lines.append(("[ACQUIRE] verification timeout -> SEARCH", COLOR_ALERT))
                return lines
            stable = self._stable_observation(
                pose, center, margin,
                required_frames=max(
                    cfg.DETECTION_CONFIRM_FRAMES, cfg.STABLE_POSE_FRAMES,
                ),
            )
            if stable is None:
                lines.append((
                    f"[ACQUIRE] stable detection {len(self._samples)}/"
                    f"{cfg.STABLE_POSE_FRAMES}", COLOR_STATUS_TRK,
                ))
                return lines
            pose, center, margin = stable
            # Once staging has begun, reacquisition must not restart FACE/standoff.
            if self._alignment_started_mono is not None:
                self._set_state("STAGING_PLAN")
                return lines
            if not getattr(self, "_initial_visibility_complete", False):
                if not self._initial_front_visible(pose, vision_meta):
                    self._start_initial_direction_correction(pose, vision_meta, now, lines)
                    return lines
                self._initial_visibility_complete = True
                self._initial_visible_snapshot = None
                self._initial_visibility_stop_mono = None
            if not self._coarse_required(pose) and self._route_start_distance(pose, vision_meta, now, lines):
                return lines
            if (
                cfg.BEARING_BASED_ROTATION_ENABLED
                and abs(center) > cfg.PALLET_CENTER_FACE_TOL_DEG
            ):
                self._face_attempts += 1
                safe_turn = safe_recenter_turn(pose, center, vision_meta)
                if safe_turn is None or not self._begin_rotation(
                    "FACE_ROTATE", "face", safe_turn, pose,
                ):
                    self._fail(
                        "no visibility-safe initial face rotation", lines,
                    )
            else:
                if self._maybe_begin_coarse(pose, now, lines):
                    return lines
                self._set_state("STANDOFF_VERIFY")
            lines.append((f"[ACQUIRE] pose locked, centre={center:+.2f}deg", COLOR_META))
            return lines

        if self.state in {"FACE_ROTATE", "RECENTER_ROTATE", "WAYPOINT_TURN", "FINAL_ROTATE"}:
            settle_map = {
                "FACE_ROTATE": "FACE_SETTLE",
                "RECENTER_ROTATE": "RECENTER_SETTLE",
                "WAYPOINT_TURN": "WAYPOINT_TURN_SETTLE",
                "FINAL_ROTATE": "FINAL_SETTLE",
            }
            self._rotation_step(
                pose, center, vision_meta, settle_map[self.state], lines,
            )
            return lines

        if self.state == "FACE_SETTLE":
            stable = self._settle_observation(pose, center, margin, lines)
            if stable is None:
                return lines
            pose, center, margin = stable
            self._update_rotation_adaptation(center)
            self._trace_end(
                stop_error_deg=self._rotation_stop_error_deg,
                settled_error_deg=center,
                **self._rotation_trace_result(),
            )
            self._rotation_coast_tracking = False
            if (
                abs(center) > cfg.PALLET_CENTER_FACE_TOL_DEG
                and self._face_attempts < cfg.ROT_MAX_DIRECTION_CHANGES
            ):
                self._face_attempts += 1
                safe_turn = safe_recenter_turn(pose, center, vision_meta)
                if safe_turn is None or not self._begin_rotation(
                    "FACE_ROTATE", "face", safe_turn, pose,
                ):
                    self._fail(
                        "no visibility-safe follow-up face rotation", lines,
                    )
            else:
                if self._maybe_begin_coarse(pose, now, lines):
                    return lines
                self._set_state("STANDOFF_VERIFY")
            return lines

        if self.state == "STANDOFF_VERIFY":
            self._exec("STOP")
            stable = self._stopped_decision_observation(pose, center, margin, lines)
            if stable is None:
                lines.append(("[STANDOFF] stable pose required", COLOR_STATUS_TRK))
                return lines
            pose, center, margin = stable
            error = pose.pallet_z_m - cfg.SAFETY_STANDOFF_Z_M
            if self._route_start_distance(pose, vision_meta, now, lines):
                return lines
            self._standoff_corrections += 1
            if self._standoff_corrections > cfg.STANDOFF_MAX_CORRECTIONS:
                self._fail("standoff correction budget exceeded", lines)
                return lines
            command = "FWD" if error > 0.0 else "BACK"
            distance = abs(error)
            if command == "BACK" and distance > cfg.BACK_MAX_DISTANCE_M:
                self._fail(f"required backward move {distance:.3f}m exceeds limit", lines)
                return lines
            self._begin_translation(
                "STANDOFF_MOVE", command, pose, distance, "v4_standoff_move",
            )
            return lines

        if self.state == "STANDOFF_MOVE":
            self._standoff_move_step(pose, vision_meta, lines)
            return lines

        if self.state == "STANDOFF_SETTLE":
            stable = self._settle_observation(pose, center, margin, lines)
            if stable is not None:
                stable_pose, _stable_center, _stable_margin = stable
                self._trace_end(
                    **self._translation_stop_info,
                    settled_z_m=stable_pose.pallet_z_m,
                )
                self._set_state("STANDOFF_VERIFY")
            return lines

        if self.state in {"STAGING_PLAN", "FINAL_POSE_LOCK"}:
            self._exec("STOP")
            if (
                self._alignment_started_mono is not None
                and self._insertion_fine_started_mono is None
                and now - self._alignment_started_mono
                >= cfg.ROT_MAX_TOTAL_ALIGN_SEC
            ):
                self._fail("alignment action budget timeout", lines)
                return lines
            stable = self._stopped_decision_observation(pose, center, margin, lines)
            if stable is None:
                lines.append((f"[{self.state}] stable stopped pose required", COLOR_STATUS_TRK))
                return lines
            pose, center, margin = stable
            gx, gz, goal_distance = goal_vector_vehicle(pose)
            if self._near_insertion_step(pose, vision_meta, lines):
                return lines
            if self.state == "FINAL_POSE_LOCK":
                lateral_error, longitudinal_error = staging_position_errors(pose)
                opening_ok, fork_hits = fork_opening_alignment(pose)
                yaw_ok = abs(pose.yaw_deg) <= cfg.FINAL_YAW_TOL_DEG
                if not opening_ok:
                    self._set_state("STAGING_PLAN")
                    lines.append((
                        f"[FINAL] position={goal_distance:.3f}m, "
                        f"fork hits={fork_hits} outside opening/ahead gate -> replan",
                        COLOR_ALERT,
                    ))
                    return lines
                if not yaw_ok:
                    target_yaw = 0.0
                    if not action_keeps_front_visible(
                        pose, pose.yaw_deg, 0.0, vision_meta,
                    ):
                        self._fail(
                            "final yaw rotation would leave the safe camera FOV",
                            lines,
                        )
                        return lines
                    if not self._begin_rotation(
                        "FINAL_ROTATE", "final", pose.yaw_deg, pose,
                        target_yaw_deg=target_yaw,
                    ):
                        self._fail("final yaw below commandable angle but outside tolerance", lines)
                    return lines
                if not self._accept_insertion(pose, lines):
                    return lines
                lines.append((
                    f"[FINAL LOCK] fork hits=({fork_hits[0]:+.3f},{fork_hits[1]:+.3f})m "
                    f"within +/-{cfg.INSERT_OPENING_SPAN_M/2:.3f}m, yaw={pose.yaw_deg:+.2f}deg",
                    COLOR_STATUS_OK,
                ))
                return lines

            if fork_opening_alignment(pose)[0] or staging_position_reached(pose):
                self._set_state("FINAL_POSE_LOCK")
                return lines
            if self._correction_cycles >= cfg.MAX_CORRECTION_CYCLES:
                self._fail("staging correction cycle budget exceeded", lines)
                return lines
            remaining = cfg.FWD_MAX_TOTAL_CORRECTION_M - self._forward_used_m
            result = plan_waypoint(pose, center, margin, remaining, vision_meta)
            if result.waypoint is None:
                if result.reason == "staging position reached":
                    self._set_state("FINAL_POSE_LOCK")
                else:
                    self._fail(f"waypoint planner: {result.reason}", lines)
                return lines
            self._waypoint = result.waypoint
            self._correction_cycles += 1
            self._last_goal_distance_m = result.waypoint.goal_distance_m
            if abs(result.waypoint.turn_deg) >= cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG:
                target_yaw = wrap_180(pose.yaw_deg - result.waypoint.turn_deg)
                self._begin_rotation(
                    "WAYPOINT_TURN", "waypoint", result.waypoint.turn_deg,
                    pose, target_yaw_deg=target_yaw,
                )
            else:
                self._begin_translation(
                    "WAYPOINT_DRIVE", "FWD", pose,
                    result.waypoint.forward_m, "v4_waypoint_forward",
                )
            lines.append((
                f"[PLAN] cycle={self._correction_cycles}, "
                f"goal=({gx:+.3f},{gz:+.3f})m, "
                f"turn={result.waypoint.turn_deg:+.2f}deg, "
                f"forward={result.waypoint.forward_m:.3f}m, "
                f"FOV margin={result.waypoint.visibility_min_margin_deg:.2f}deg",
                COLOR_META,
            ))
            return lines

        if self.state in {"RECENTER_SETTLE", "WAYPOINT_TURN_SETTLE", "FINAL_SETTLE"}:
            stable = self._settle_observation(pose, center, margin, lines)
            if stable is None:
                return lines
            stable_pose, stable_center, _stable_margin = stable
            settled_error = self._rotation_error(stable_pose, stable_center)
            self._update_rotation_adaptation(settled_error)
            self._trace_end(
                stop_error_deg=self._rotation_stop_error_deg,
                settled_error_deg=settled_error,
                **self._rotation_trace_result(),
            )
            self._rotation_coast_tracking = False
            next_state = (
                "FINAL_POSE_LOCK" if self.state == "FINAL_SETTLE"
                else "STAGING_PLAN"
            )
            if self._continue_accepted_waypoint_turn(
                stable_pose, settled_error, vision_meta, lines,
            ):
                return lines
            self._rotation.reset()
            self._set_state(next_state)
            return lines

        if self.state == "WAYPOINT_DRIVE":
            if 0.0 < pose.pallet_z_m <= cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M:
                self._stop_translation(
                    "WAYPOINT_DRIVE_SETTLE", lines, insertion_range_entered=True,
                    travelled_m=self._translation_progress(pose),
                )
                return lines
            self._forward_segment_step(
                pose, center, margin, vision_meta,
                "WAYPOINT_DRIVE_SETTLE", lines, enforce_visibility=True,
                enforce_staging_limit=True,
            )
            return lines

        if self.state == "WAYPOINT_DRIVE_SETTLE":
            stable = self._settle_observation(pose, center, margin, lines)
            if stable is None:
                return lines
            stable_pose, _stable_center, _stable_margin = stable
            travelled = self._translation_progress(stable_pose)
            self._forward_used_m += travelled
            if 0.0 < stable_pose.pallet_z_m <= cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M:
                self._trace_end(**self._translation_stop_info, settled_travel_m=travelled,
                                total_forward_used_m=self._forward_used_m)
                if self._forward_used_m > cfg.FWD_MAX_TOTAL_CORRECTION_M:
                    self._fail("forward correction hard limit exceeded", lines)
                else:
                    self._set_state("FINAL_POSE_LOCK")
                return lines
            _gx, _gz, goal_distance = goal_vector_vehicle(stable_pose)
            lateral_error, longitudinal_error = staging_position_errors(stable_pose)
            if self._last_goal_distance_m is not None:
                improvement = self._last_goal_distance_m - goal_distance
                if improvement < cfg.MIN_PROGRESS_M:
                    self._no_progress_cycles += 1
                else:
                    self._no_progress_cycles = 0
            if self._no_progress_cycles >= cfg.MAX_NO_PROGRESS_CYCLES:
                self._fail("two waypoint cycles made no visual progress", lines)
                return lines
            if longitudinal_error > cfg.FINAL_DISTANCE_TOL_M:
                self._fail(
                    f"staging overtravel {longitudinal_error:.3f}m exceeds "
                    f"{cfg.FINAL_DISTANCE_TOL_M:.3f}m limit",
                    lines,
                )
                return lines
            if (
                longitudinal_error >= 0.0
                and abs(lateral_error) > cfg.FINAL_LATERAL_TOL_M
            ):
                self._fail(
                    f"staging overtravel lateral {lateral_error:+.3f}m exceeds "
                    f"{cfg.FINAL_LATERAL_TOL_M:.3f}m limit",
                    lines,
                )
                return lines
            self._trace_end(
                **self._translation_stop_info,
                settled_travel_m=travelled,
                settled_goal_distance_m=goal_distance,
                settled_lateral_error_m=lateral_error,
                settled_longitudinal_error_m=longitudinal_error,
                total_forward_used_m=self._forward_used_m,
            )
            if self._forward_used_m > cfg.FWD_MAX_TOTAL_CORRECTION_M:
                self._fail("forward correction hard limit exceeded", lines)
            elif staging_position_reached(stable_pose):
                self._set_state("FINAL_POSE_LOCK")
            else:
                self._set_state("STAGING_PLAN")
            return lines

        self._fail(f"unknown v4 state: {self.state}", lines)
        return lines
