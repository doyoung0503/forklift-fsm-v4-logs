"""One-shot model yaw -> IMU turn -> fitted FWD -> opposite 90-degree turn."""
import math
from statistics import median

from calib.config import COLOR_META
from calib.control import configure_motion_deadline, motion_deadline_expired
from . import config as cfg
from .motion import forward_seconds
from .pose import wrap_180


COARSE_BLIND_STATES = frozenset({
    "COARSE_PREPARE", "COARSE_ROTATE", "COARSE_ROTATE_SETTLE",
    "COARSE_DRIVE", "COARSE_DRIVE_SETTLE", "COARSE_RETURN_ROTATE",
    "COARSE_RETURN_SETTLE",
})
COARSE_STATES = COARSE_BLIND_STATES | {"COARSE_REACQUIRE"}


def coarse_plan(pose):
    """Positive rotation is right; pallet X is right along the front face."""
    lateral = float(pose.rot_x_pallet_m)
    yaw = float(pose.yaw_deg)
    if not all(map(math.isfinite, (lateral, yaw, pose.rot_z_pallet_m))):
        raise ValueError("invalid coarse pose")
    if abs(yaw) >= 90.0:
        raise ValueError("coarse correction requires a front-facing pallet pose")
    if not cfg.COARSE_MIN_LATERAL_M <= abs(lateral) <= cfg.COARSE_MAX_LATERAL_M:
        raise ValueError("coarse lateral outside configured travel limits")
    if -pose.rot_z_pallet_m < cfg.STAGING_DISTANCE_M:
        raise ValueError("insufficient front-plane clearance for coarse rotation")
    side = math.copysign(1.0, lateral)
    turn = wrap_180(yaw - side * 90.0)
    duration = forward_seconds(abs(lateral))
    if not math.isfinite(duration) or duration >= min(cfg.FWD_COMMAND_MAX_SEC, cfg.FWD_MAX_COMMAND_SEC):
        raise ValueError("coarse distance exceeds fitted forward time range")
    return turn, abs(lateral), duration, side * 90.0


class CoarseAlignmentMixin:
    @property
    def requires_imu(self):
        return cfg.COARSE_IMU_ENABLED

    def _coarse_reset(self):
        self._coarse_done = False
        self._coarse_started = None
        self._coarse_bias = 0.0
        self._coarse_bias_samples = []
        self._coarse_bias_stamp = None
        self._coarse_last_quiet = None
        self._coarse_target = 0.0
        self._coarse_progress = 0.0
        # Source is bound by the runtime, and survives operator FSM resets.
        if not hasattr(self, "imu_source"):
            self.imu_source = None
        configure_motion_deadline(None)

    def _coarse_required(self, pose):
        return (cfg.COARSE_IMU_ENABLED and not getattr(self, "_coarse_done", False)
                and abs(pose.yaw_deg) > cfg.COARSE_YAW_TRIGGER_DEG)

    def _maybe_begin_coarse(self, pose, now, lines):
        if not self._coarse_required(pose):
            return False
        self._exec("STOP")
        try:
            self._coarse_plan = coarse_plan(pose)
        except ValueError as exc:
            self._fail(str(exc), lines)
            return True
        self._coarse_done = True
        self._coarse_started = now
        self._coarse_bias_samples = []
        self._coarse_bias_stamp = None
        self._set_state("COARSE_PREPARE")
        self._trace_begin("v4_coarse_alignment", "STOP", model_yaw_deg=pose.yaw_deg,
                          lateral_m=pose.rot_x_pallet_m,
                          first_turn_deg=self._coarse_plan[0],
                          forward_hold_sec=self._coarse_plan[2],
                          return_turn_deg=self._coarse_plan[3])
        lines.append(("[COARSE] stopped gyro bias measurement", COLOR_META))
        return True

    def _coarse_abort(self, reason, lines):
        self._exec("STOP")
        configure_motion_deadline(None)
        self._fail(reason, lines)

    def _coarse_yaw(self, sample):
        yaw, stamp, _rate, _error = sample
        return yaw - self._coarse_bias * (stamp - self._coarse_bias_origin)

    def _coarse_turn(self, target, sample, now, state):
        self._coarse_ref = self._coarse_yaw(sample)
        self._coarse_target = target
        self._coarse_progress = 0.0
        self._coarse_command = "ROT_RIGHT" if target > 0 else "ROT_LEFT"
        self._set_state(state, cfg.COARSE_ROTATION_TIMEOUT_SEC)
        configure_motion_deadline(now + cfg.COARSE_COMMAND_LEASE_SEC)
        self._exec(self._coarse_command)
        self._trace_begin("v4_coarse_turn", self._coarse_command,
                          target_deg=target, control_basis="imu_relative_yaw")

    def _coarse_settle(self, state, now):
        self._exec("STOP")
        configure_motion_deadline(None)
        self._coarse_last_quiet = None
        self._set_state(state, cfg.COARSE_SETTLE_TIMEOUT_SEC)

    def _coarse_step(self, now, lines, det_ok, yaw_smooth, offset_smooth, dist_z, vision_meta):
        if self._coarse_started is None or now - self._coarse_started >= cfg.COARSE_TOTAL_TIMEOUT_SEC:
            self._coarse_abort("coarse total timeout", lines)
            return
        if self.state == "COARSE_REACQUIRE":
            self._exec("STOP")
            if now >= self._state_deadline_mono:
                self._coarse_abort("coarse pallet reacquisition timeout", lines)
                return
            pose, _reason = self._observe_pose(det_ok, yaw_smooth, offset_smooth, dist_z, vision_meta, now)
            if pose is None:
                self._samples.clear()
                return
            center, margin = self._vision_values(vision_meta, pose)
            stable = self._stable_observation(pose, center, margin)
            if stable is not None:
                self._trace_end(control_basis="coarse_complete_visual_reacquired")
                self._invalid_since_mono = None
                self._alignment_started_mono = None
                self._set_state("ACQUIRE_VERIFY", cfg.ACQUIRE_VERIFY_TIMEOUT_SEC)
            return
        if motion_deadline_expired():
            self._coarse_abort("coarse command lease expired", lines)
            return
        sample = self.imu_source.snapshot() if self.imu_source is not None else None
        if sample is None or sample[1] is None:
            self._coarse_abort("IMU unavailable", lines)
            return
        yaw, stamp, rate, error = sample
        if (error or not all(map(math.isfinite, (yaw, stamp, rate)))
                or not 0 <= now - stamp <= cfg.COARSE_IMU_MAX_AGE_SEC):
            self._coarse_abort(f"IMU invalid/stale: {error or 'sample age'}", lines)
            return
        if self.state == "COARSE_PREPARE":
            self._exec("STOP")
            if stamp != self._coarse_bias_stamp:
                self._coarse_bias_samples.append(rate)
                self._coarse_bias_stamp = stamp
            if now - self._state_entered_mono < cfg.COARSE_SETTLE_SEC:
                return
            values = self._coarse_bias_samples
            if len(values) < 10 or max(values) - min(values) > cfg.COARSE_STABLE_RATE_DEG_S:
                self._coarse_abort("IMU not stationary for bias measurement", lines)
                return
            self._coarse_bias = median(values)
            self._coarse_bias_origin = stamp
            self._coarse_turn(self._coarse_plan[0], sample, now, "COARSE_ROTATE")
            return
        if self.state in {"COARSE_ROTATE", "COARSE_RETURN_ROTATE"}:
            self._coarse_progress = self._coarse_yaw(sample) - self._coarse_ref
            sign = math.copysign(1.0, self._coarse_target)
            if sign * self._coarse_progress >= abs(self._coarse_target):
                self._trace_end(actual_delta_deg=self._coarse_progress, stop_reason="imu target reached")
                self._coarse_settle("COARSE_ROTATE_SETTLE" if self.state == "COARSE_ROTATE"
                                    else "COARSE_RETURN_SETTLE", now)
            elif now >= self._state_deadline_mono or sign * self._coarse_progress < -5.0:
                self._coarse_abort("coarse turn timeout or reversed IMU direction", lines)
            else:
                configure_motion_deadline(now + cfg.COARSE_COMMAND_LEASE_SEC)
                lines.append((f"[COARSE IMU] {self._coarse_progress:+.1f}/{self._coarse_target:+.1f} deg", COLOR_META))
            return
        if self.state == "COARSE_DRIVE":
            if abs(self._coarse_yaw(sample) - self._coarse_drive_yaw) > cfg.COARSE_DRIVE_HEADING_TOL_DEG:
                self._coarse_abort("coarse straight-drive heading drift", lines)
            elif now >= self._coarse_drive_end:
                self._trace_end(command_duration_sec=self._coarse_plan[2], actual_travel_m=None)
                self._coarse_settle("COARSE_DRIVE_SETTLE", now)
            else:
                configure_motion_deadline(min(now + cfg.COARSE_COMMAND_LEASE_SEC,
                                              self._coarse_drive_end + cfg.COARSE_COMMAND_LEASE_SEC))
                lines.append((f"[COARSE FWD] {self._coarse_plan[1]:.3f} m / {self._coarse_drive_end-now:.2f}s left (time estimate)", COLOR_META))
            return
        self._exec("STOP")
        if abs(rate - self._coarse_bias) <= cfg.COARSE_STABLE_RATE_DEG_S:
            if self._coarse_last_quiet is None:
                self._coarse_last_quiet = now
        else:
            self._coarse_last_quiet = None
        if now >= self._state_deadline_mono:
            self._coarse_abort("coarse stop did not settle", lines)
            return
        if self._coarse_last_quiet is None or now - self._coarse_last_quiet < cfg.COARSE_SETTLE_SEC:
            return
        if self.state == "COARSE_ROTATE_SETTLE":
            settled_delta = self._coarse_yaw(sample) - self._coarse_ref
            if abs(settled_delta - self._coarse_target) > cfg.COARSE_DRIVE_HEADING_TOL_DEG:
                self._coarse_abort("coarse first-turn overshoot exceeds heading tolerance", lines)
                return
            self._coarse_drive_yaw = self._coarse_yaw(sample)
            self._coarse_drive_end = now + self._coarse_plan[2]
            self._set_state("COARSE_DRIVE")
            configure_motion_deadline(now + cfg.COARSE_COMMAND_LEASE_SEC)
            self._exec("FWD")
            self._trace_begin("v4_coarse_drive", "FWD", target_distance_m=self._coarse_plan[1],
                              fitted_duration_sec=self._coarse_plan[2], control_basis="fitted_time_no_vision")
        elif self.state == "COARSE_DRIVE_SETTLE":
            self._coarse_turn(self._coarse_plan[3], sample, now, "COARSE_RETURN_ROTATE")
        else:
            self._pose_filter.reset()
            self._set_state("COARSE_REACQUIRE", cfg.COARSE_REACQUIRE_TIMEOUT_SEC)
