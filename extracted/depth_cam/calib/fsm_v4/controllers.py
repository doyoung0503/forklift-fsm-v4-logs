"""Reusable bang-bang action controllers parameterized by v4 config."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from . import config as cfg
from .pose import wrap_180
from .rotation_model import RotationPlan


def adaptive_overshoot_scale(
    current_scale: float, requested_deg: float, actual_deg: float,
) -> float:
    """Reduce the active-time scale in proportion to settled overshoot.

    A 30 degree request that actually turns 40 degrees multiplies the current
    scale by 30/40.  Under-travel does not increase the scale because this
    first adaptive version is deliberately overshoot-only.
    """

    scale = min(1.0, max(cfg.ROT_ADAPTIVE_MIN_TIME_SCALE, float(current_scale)))
    requested = abs(float(requested_deg))
    actual = abs(float(actual_deg))
    if requested <= 0.0 or actual <= requested:
        return scale
    return max(
        cfg.ROT_ADAPTIVE_MIN_TIME_SCALE,
        min(1.0, scale * requested / actual),
    )


@dataclass(frozen=True)
class RotationUpdate:
    elapsed_sec: float
    directed_start_delta_deg: float
    startup_hits: int
    motion_confirmed: bool
    ready: bool
    timed_out: bool
    error_rate_deg_s: float
    measured_error_rate_deg_s: float
    model_error_rate_deg_s: float
    model_profile_active: bool
    response_age_sec: float
    measured_prediction_delta_deg: float
    model_prediction_delta_deg: float
    predicted_error_deg: float
    stop_margin_deg: float
    should_stop: bool
    # --- fitted-response fields (zero/None on the legacy path) ---
    plan_hold_sec: float = 0.0
    hold_reached: bool = False
    coast_margin_deg: float = 0.0
    stop_reason: str = ""


class RotationController:
    """Fixed-direction rotation with visual startup and predictive early STOP."""

    def __init__(self) -> None:
        self.active = False
        self.command: Optional[str] = None
        self.started_mono = 0.0
        self.start_error_deg = 0.0
        self.expected_delta_sign = 0.0
        self.startup_hits = 0
        self.motion_confirmed = False
        self.last_error_deg: Optional[float] = None
        self.last_measurement_mono: Optional[float] = None
        self.measured_rate_deg_s = 0.0
        self.plan: Optional[RotationPlan] = None
        self.base_plan_hold_sec = 0.0
        self.adaptive_time_scale = 1.0
        self.adaptive_excluded_delay_sec = 0.0
        self.response = cfg.ROTATION_RESPONSE

    def reset(self) -> None:
        self.__init__()

    def start(
        self, error_deg: float, command: str, now_mono: float,
        measurement_mono: float, domain: str = "bearing",
        range_m: Optional[float] = None, adaptive_time_scale: float = 1.0,
        slope_multiplier: float = 1.0,
        insertion_fine: bool = False,
    ) -> None:
        """Begin a rotation.

        ``domain`` says what the error is measured in: "bearing" for the
        FACE/RECENTER front-face bearing, "heading" for the WAYPOINT pallet-yaw
        error.  The fitted response is in heading degrees, so a bearing-domain
        rotation gets it rescaled for the current pallet range.
        """
        error = wrap_180(error_deg)
        self.active = True
        self.command = command
        self.started_mono = float(now_mono)
        self.start_error_deg = error
        self.expected_delta_sign = -1.0 if error > 0.0 else 1.0
        self.startup_hits = 0
        self.motion_confirmed = False
        self.last_error_deg = error
        self.last_measurement_mono = float(measurement_mono)
        self.measured_rate_deg_s = 0.0
        self.response = (
            cfg.ROTATION_RESPONSE.in_bearing_domain(
                range_m if range_m else cfg.STAGING_DISTANCE_M,
                cfg.ROT_BEARING_CENTRE_OFFSET_M,
            )
            if domain == "bearing" else cfg.ROTATION_RESPONSE
        )
        # Apply only to heading endpoint fits. Bearing geometry changes with
        # range, so FACE/RECENTER never share these yaw-domain observations.
        if (cfg.ROT_ADAPTIVE_SLOPE_ENABLED and domain == "heading"
                and getattr(self.response, 'endpoint_only', False)):
            multiplier = min(cfg.ROT_ADAPTIVE_SLOPE_MAX_MULTIPLIER,
                             max(cfg.ROT_ADAPTIVE_SLOPE_MIN_MULTIPLIER, slope_multiplier))
            self.response = replace(self.response,
                                    slope_deg_s=self.response.slope_deg_s * multiplier)
        self.plan = (
            self.response.plan(
                error,
                min_angle_deg=(cfg.INSERT_FINE_MIN_TURN_DEG if insertion_fine
                               else cfg.ROT_MIN_COMMANDABLE_ANGLE_DEG),
                max_angle_deg=cfg.ROT_MAX_WAYPOINT_TURN_DEG,
                timeout_margin_sec=cfg.ROT_COMMAND_TIMEOUT_MARGIN_SEC,
                max_hold_sec=cfg.ROT_MAX_COMMAND_HOLD_SEC,
                coast_reduction_deg=(
                    cfg.ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG
                ),
            )
            if cfg.ROT_USE_FITTED_RESPONSE else None
        )
        self.adaptive_time_scale = (
            min(1.0, max(cfg.ROT_ADAPTIVE_MIN_TIME_SCALE, float(adaptive_time_scale)))
            if cfg.ROT_ADAPTIVE_OVERSHOOT_ENABLED and not getattr(self.response, 'endpoint_only', False) else 1.0
        )
        if self.plan is not None:
            self.base_plan_hold_sec = self.plan.hold_sec
            self.adaptive_excluded_delay_sec = self.response.startup_delay_sec
            if self.adaptive_time_scale < 1.0:
                # Keep the fitted startup/dead time intact and scale only the
                # command time that can produce rotation.
                base_active_sec = max(
                    0.0, self.plan.hold_sec - self.adaptive_excluded_delay_sec,
                )
                hold_sec = (
                    self.adaptive_excluded_delay_sec
                    + base_active_sec * self.adaptive_time_scale
                )
                timeout_margin_sec = max(
                    0.0, self.plan.hard_timeout_sec - self.plan.hold_sec,
                )
                stop_angle = self.response.angle_at(hold_sec)
                stop_rate = self.response.rate_at(hold_sec)
                coast_deg = self.response.adjusted_coast_after_hold(
                    hold_sec,
                    cfg.ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG,
                )
                note = self.plan.note
                adaptive_note = "adaptive active-time x%.3f" % self.adaptive_time_scale
                note = f"{note}; {adaptive_note}" if note else adaptive_note
                self.plan = replace(
                    self.plan,
                    hold_sec=hold_sec,
                    hard_timeout_sec=hold_sec + timeout_margin_sec,
                    predicted_stop_angle_deg=stop_angle,
                    predicted_stop_rate_deg_s=stop_rate,
                    predicted_coast_deg=coast_deg,
                    predicted_total_deg=stop_angle + coast_deg,
                    predicted_settle_sec=self.response.settle_sec(hold_sec),
                    angle_sd_deg=self.response.angle_uncertainty_deg(hold_sec),
                    note=note,
                )

    @staticmethod
    def _logged_speed(response_age_sec: float) -> Optional[float]:
        """Return the supported log-median speed for time since onset."""

        if not cfg.ROT_LOG_SPEED_PROFILE_ENABLED or response_age_sec < 0.0:
            return None
        for start_sec, end_sec, speed_deg_s in cfg.ROT_LOG_SPEED_PROFILE:
            if start_sec <= response_age_sec < end_sec:
                return float(speed_deg_s)
        return None

    @staticmethod
    def _logged_angle(response_age_sec: float, horizon_sec: float) -> float:
        """Integrate only the measured portions of the log-speed profile."""

        if not cfg.ROT_LOG_SPEED_PROFILE_ENABLED or horizon_sec <= 0.0:
            return 0.0
        window_start = max(0.0, float(response_age_sec))
        window_end = window_start + float(horizon_sec)
        angle_deg = 0.0
        for start_sec, end_sec, speed_deg_s in cfg.ROT_LOG_SPEED_PROFILE:
            overlap = max(
                0.0,
                min(window_end, end_sec) - max(window_start, start_sec),
            )
            angle_deg += overlap * speed_deg_s
        return angle_deg

    def update(
        self, error_deg: float, now_mono: float, measurement_mono: float,
        measurement_age_sec: float,
    ) -> RotationUpdate:
        if not self.active:
            raise RuntimeError("rotation controller was not started")
        error = wrap_180(error_deg)
        elapsed = max(0.0, float(now_mono) - self.started_mono)
        directed_delta = (
            wrap_180(error - self.start_error_deg) * self.expected_delta_sign
        )
        if directed_delta >= cfg.ROT_STARTUP_MIN_VISUAL_DELTA_DEG:
            self.startup_hits += 1
        else:
            self.startup_hits = 0
        if self.startup_hits >= cfg.ROT_STARTUP_CONFIRM_FRAMES:
            self.motion_confirmed = True

        if (
            self.last_error_deg is not None
            and self.last_measurement_mono is not None
            and float(measurement_mono) > self.last_measurement_mono
        ):
            dt = float(measurement_mono) - self.last_measurement_mono
            if 0.01 <= dt <= 0.75:
                raw_rate = wrap_180(error - self.last_error_deg) / dt
                raw_rate = max(
                    -cfg.ROT_RATE_LIMIT_DEG_S,
                    min(cfg.ROT_RATE_LIMIT_DEG_S, raw_rate),
                )
                self.measured_rate_deg_s = raw_rate
        self.last_error_deg = error
        self.last_measurement_mono = float(measurement_mono)

        startup_delay_sec = (
            self.response.startup_delay_sec
            if self.plan is not None else cfg.ROT_STARTUP_DELAY_SEC
        )
        ready = self.motion_confirmed and elapsed >= startup_delay_sec
        startup_timed_out = (
            elapsed >= cfg.ROT_STARTUP_TIMEOUT_SEC and not self.motion_confirmed
        )
        action_timed_out = elapsed >= cfg.ROT_MAX_COMMAND_SEC
        measured_rate = self.measured_rate_deg_s
        response_age = elapsed - startup_delay_sec
        endpoint_only = getattr(self.response, 'endpoint_only', False)
        logged_speed = self._logged_speed(response_age) if ready and not endpoint_only else None
        model_rate = (
            self.expected_delta_sign * logged_speed
            if logged_speed is not None else 0.0
        )
        measured_approaching = error * measured_rate < 0.0
        if logged_speed is not None:
            # The offline measured profile supplies a conservative early-response
            # floor.  A faster live observation wins so STOP is never delayed
            # merely because the median profile was slower than this run.
            effective_speed = max(
                logged_speed,
                abs(measured_rate) if measured_approaching else 0.0,
            )
            rate = self.expected_delta_sign * effective_speed
        else:
            rate = measured_rate
        horizon = max(0.0, measurement_age_sec) + cfg.ROT_STOP_LOOKAHEAD_SEC
        measured_prediction = (
            measured_rate * horizon if measured_approaching else 0.0
        )
        model_prediction = (
            self.expected_delta_sign * self._logged_angle(response_age, horizon)
            if ready and not endpoint_only else 0.0
        )
        if abs(model_prediction) > abs(measured_prediction):
            prediction_delta = model_prediction
        else:
            prediction_delta = measured_prediction
        approaching = error * prediction_delta < 0.0
        predicted = wrap_180(
            error + (prediction_delta if approaching else 0.0)
        )
        margin = min(
            cfg.ROT_STOP_MARGIN_MAX_DEG,
            cfg.ROT_STOP_MARGIN_BASE_DEG + abs(prediction_delta),
        )
        crossed = approaching and error * predicted <= 0.0
        should_stop = bool(ready and (abs(predicted) <= margin or crossed))

        plan_hold = 0.0
        hold_reached = False
        coast_margin = 0.0
        stop_reason = ""
        if self.plan is not None:
            # Fitted response drives the STOP decision.
            #  1) planned hold reached  -> the model says the coast completes it
            #  2) live pose says the remaining error is already inside the
            #     predicted coast -> STOP now regardless of the planned time
            # The model rate is a floor, so a run that turns faster than the
            # fitted median stops early instead of late.
            response = self.response
            plan_hold = self.plan.hold_sec
            hold_reached = elapsed >= plan_hold
            live_rate = abs(measured_rate) if measured_approaching else None
            coast_margin = response.stop_now_margin_deg(
                elapsed, live_rate, max(0.0, measurement_age_sec),
                cfg.ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG,
            )
            live_stop = response.should_stop_now(
                error, elapsed, live_rate, max(0.0, measurement_age_sec),
                cfg.ROT_ACTIVE_COAST_HEURISTIC_REDUCTION_DEG,
            )
            # A real sign change of the live error, not the legacy lookahead
            # extrapolation -- the coast margin above already covers lookahead.
            overshot = error * self.start_error_deg <= 0.0
            if live_stop:
                stop_reason = "live_coast_margin"
            elif overshot:
                stop_reason = "target_crossed"
            elif hold_reached:
                stop_reason = "fitted_hold_elapsed"
            should_stop = bool(live_stop or overshot or hold_reached)

        return RotationUpdate(
            elapsed_sec=elapsed,
            directed_start_delta_deg=directed_delta,
            startup_hits=self.startup_hits,
            motion_confirmed=self.motion_confirmed,
            ready=ready,
            timed_out=startup_timed_out or action_timed_out,
            error_rate_deg_s=rate,
            measured_error_rate_deg_s=measured_rate,
            model_error_rate_deg_s=model_rate,
            model_profile_active=logged_speed is not None,
            response_age_sec=response_age,
            measured_prediction_delta_deg=measured_prediction,
            model_prediction_delta_deg=model_prediction,
            predicted_error_deg=predicted,
            stop_margin_deg=margin,
            should_stop=should_stop,
            plan_hold_sec=plan_hold,
            hold_reached=hold_reached,
            coast_margin_deg=coast_margin,
            stop_reason=stop_reason,
        )
