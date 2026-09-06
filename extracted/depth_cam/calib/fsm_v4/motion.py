"""Time conversions and model-based endpoint helpers for FSM v4."""

from __future__ import annotations

import math

from . import config as cfg


def drive_seconds(distance_m: float) -> float:
    """Shared forward/backward distance-to-time fit."""

    effective = max(
        0.0,
        cfg.FWD_DISTANCE_SCALE * abs(float(distance_m)) + cfg.FWD_DISTANCE_BIAS_M,
    )
    accel = max(1e-9, cfg.FWD_ACCEL_M_S2)
    accel_time = max(0.0, cfg.FWD_ACCEL_DURATION_SEC)
    accel_distance = 0.5 * accel * accel_time * accel_time
    max_speed = max(1e-6, accel * accel_time)
    if effective <= accel_distance:
        seconds = cfg.FWD_T0_SEC + math.sqrt(2.0 * effective / accel)
    else:
        seconds = (
            cfg.FWD_T0_SEC + accel_time
            + (effective - accel_distance) / max_speed
        )
    return max(cfg.FWD_COMMAND_MIN_SEC, min(cfg.FWD_COMMAND_MAX_SEC, seconds))


def forward_seconds(distance_m: float) -> float:
    return drive_seconds(distance_m)


def backward_seconds(distance_m: float) -> float:
    return min(cfg.BACK_MAX_COMMAND_SEC, drive_seconds(distance_m))


def fitted_forward_speed(elapsed_sec: float) -> float:
    """Expected forward speed at elapsed time after the FWD CAN write."""

    moving_sec = max(0.0, float(elapsed_sec) - cfg.FWD_T0_SEC)
    return min(
        cfg.FWD_ACCEL_M_S2 * moving_sec,
        cfg.FWD_ACCEL_M_S2 * cfg.FWD_ACCEL_DURATION_SEC,
    )


def bounded_forward_lookahead(
    elapsed_sec: float,
    measured_speed_m_s: float,
    horizon_sec: float,
) -> tuple[float, float]:
    """Return bounded predictive advance and the speed actually used.

    A one-frame PnP jump must not turn into 0.7--0.8 m of fictitious travel.
    The measured speed is capped by the fitted response envelope, and the
    complete lookahead (latency + coast) is capped independently.
    """

    fitted_limit = (
        fitted_forward_speed(elapsed_sec)
        * cfg.FWD_PREDICTIVE_SPEED_MULTIPLIER
    )
    used_speed = min(max(0.0, float(measured_speed_m_s)), fitted_limit)
    raw_advance = (
        used_speed * max(0.0, float(horizon_sec))
        + cfg.FWD_COAST_ALLOWANCE_M
    )
    advance = min(raw_advance, cfg.FWD_PREDICTIVE_MAX_ADVANCE_M)
    return advance, used_speed


def forward_predictive_stop_ready(travelled_m: float, target_m: float) -> bool:
    """Whether enough real displacement exists to allow predictive STOP."""

    target = max(0.0, float(target_m))
    if target <= 0.0:
        return True
    return max(0.0, float(travelled_m)) >= (
        target * cfg.FWD_PREDICTIVE_MIN_PROGRESS_RATIO
    )


def measurement_age(vision_meta: dict | None, now_mono: float) -> float:
    if vision_meta:
        try:
            measured = float(vision_meta["measurement_mono"])
            inference = max(0.0, now_mono - measured)
            return inference + cfg.SENSOR_PIPELINE_LATENCY_SEC
        except (KeyError, TypeError, ValueError):
            pass
    return cfg.FALLBACK_INFERENCE_LATENCY_SEC + cfg.SENSOR_PIPELINE_LATENCY_SEC
