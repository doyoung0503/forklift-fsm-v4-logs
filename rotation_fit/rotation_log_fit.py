"""Fit a four-phase rotation response from per-frame inference logs.

The fitted response is deliberately small and auditable:

1. command-to-visible-motion dead time,
2. constant angular acceleration,
3. constant maximum angular rate,
4. post-command rotation regressed linearly from the rate just before STOP.

Left and right runs are pooled after their observed angle changes are made
positive.  Only joystick/CAN deflection 30 is accepted by default.  The value
30 is a command deflection, not a calibrated torque in N*m.

The native input contract is the pair written by ``calib.tracelog``:

* ``*_control_seq.jsonl`` for real CAN write boundaries;
* ``*_inference_timing.csv`` for every camera frame and its host/sensor time.

A newly generated inference CSV can be joined by ``frame_i`` instead of
rewriting timing data.  The fit contract is deliberately fixed to the
``yaw_deg`` orientation signal in the vehicle-``heading`` domain; target
bearing is range-dependent because the camera is ahead of the rotation
centre.  See ``fit_piecewise_rotation_model.py --help``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

try:  # package import
    from .piecewise_rotation_model import LogRotationModel
except ImportError:  # direct script import
    from piecewise_rotation_model import LogRotationModel


ROTATE_MOVEMENTS = {"rotate_left_slow", "rotate_right_slow"}
ACTIVE_MOVEMENTS = ROTATE_MOVEMENTS | {
    "forward", "forward_slow", "backward", "turn_left", "turn_right",
    "forward_left", "forward_right", "backward_left", "backward_right",
}
LOGICAL_ROTATE_COMMANDS = {"ROT_LEFT", "ROT_RIGHT"}
FIT_ANGLE_COLUMN = "yaw_deg"
FIT_ANGLE_DOMAIN = "heading"


def validate_fit_angle_contract(
    angle_column: str,
    angle_domain: str = FIT_ANGLE_DOMAIN,
) -> None:
    """Reject signals that are not the canonical heading-orientation input.

    ``center_bearing_deg`` includes the camera's translational arc about the
    vehicle rotation centre and therefore changes with target range.  This
    fitter intentionally models vehicle heading from PnP yaw only; accepting a
    bearing signal here would make acceleration, maximum rate, and coast
    parameters range-dependent while producing an indistinguishable runtime
    artifact.
    """

    if angle_column != FIT_ANGLE_COLUMN or angle_domain != FIT_ANGLE_DOMAIN:
        raise ValueError(
            "rotation fitting is fixed to "
            f"angle_column={FIT_ANGLE_COLUMN!r}, "
            f"angle_domain={FIT_ANGLE_DOMAIN!r}; received "
            f"angle_column={angle_column!r}, angle_domain={angle_domain!r}"
        )


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite diagnostics to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def robust_sigma(values: Sequence[float]) -> float:
    """Gaussian-equivalent sigma estimated from median absolute deviation."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def unwrap_degrees(values: Sequence[float], period_deg: float) -> np.ndarray:
    """Unwrap an angular sequence with either a 180 or 360 degree period."""
    period = float(period_deg)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("angle period must be positive")
    x = np.asarray(values, dtype=float)
    phase = x * (2.0 * np.pi / period)
    return np.unwrap(phase) * (period / (2.0 * np.pi))


@dataclass(frozen=True)
class FitSettings:
    command_strength: int = 30
    angle_period_deg: float = 360.0
    pre_command_sec: float = 1.0
    inertia_horizon_sec: float = 2.0
    inertia_window_sec: float = 0.20
    onset_sustain_sec: float = 0.25
    onset_confirm_frames: int = 3
    onset_min_delta_deg: float = 0.20
    onset_noise_multiplier: float = 4.0
    onset_min_rate_deg_s: float = 0.30
    smoothing_window_sec: float = 0.30
    stop_rate_window_sec: float = 0.60
    max_frame_gap_sec: float = 0.35
    max_baseline_noise_deg: float = 2.0
    max_baseline_rate_deg_s: float = 1.0
    min_command_sec: float = 0.15
    min_drive_frames: int = 5
    min_cruise_observation_sec: float = 0.25
    min_cruise_segments: int = 2
    min_inertia_speed_span_deg_s: float = 1.0
    min_loo_predictions: int = 4
    max_loo_rmse_deg: float = 3.0
    fit_inertia_intercept: bool = False
    require_actual_can_timing: bool = True
    can_time_point: str = "return"

    def __post_init__(self) -> None:
        if self.command_strength != 30:
            raise ValueError(
                "this model contract is fixed to joystick deflection 30"
            )
        if self.can_time_point not in {"start", "return", "midpoint"}:
            raise ValueError("can_time_point must be start, return, or midpoint")
        for name in (
            "angle_period_deg", "pre_command_sec", "inertia_horizon_sec",
            "inertia_window_sec", "onset_sustain_sec", "smoothing_window_sec",
            "stop_rate_window_sec", "max_frame_gap_sec", "max_loo_rmse_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.min_loo_predictions) < 1:
            raise ValueError("min_loo_predictions must be positive")


@dataclass
class CanBoundary:
    movement: str
    step: Optional[int]
    write_start_s: float
    write_return_s: float
    t_iso: str = ""
    strength: Optional[int] = None
    payload: tuple[int, ...] = ()
    sync_done_s: Optional[float] = None
    source: str = "can_tx"

    def nominal_time(self, point: str) -> float:
        if point == "start":
            return self.write_start_s
        if point == "midpoint":
            return 0.5 * (self.write_start_s + self.write_return_s)
        return self.write_return_s


@dataclass(frozen=True)
class CommandWindow:
    recording: str
    step: Optional[int]
    command: str
    movement: str
    strength: Optional[int]
    start: CanBoundary
    stop: CanBoundary
    next_active_command_s: Optional[float]

    @property
    def direction(self) -> str:
        return "LEFT" if "left" in self.movement.lower() else "RIGHT"


@dataclass(frozen=True)
class FrameSeries:
    recording: str
    frame_i: np.ndarray
    host_mono_s: np.ndarray
    analysis_time_s: np.ndarray
    sensor_timestamp_ms: np.ndarray
    sensor_domain: tuple[str, ...]
    angle_deg: np.ndarray
    range_m: np.ndarray
    source_path: str
    frame_time_source: str
    sensor_to_host_scale: Optional[float] = None
    sensor_alignment_rmse_ms: Optional[float] = None
    model_ids: tuple[str, ...] = ()


@dataclass
class AnalysedSegment:
    recording: str
    step: Optional[int]
    command: str
    movement: str
    command_direction: str
    command_strength: Optional[int]
    command_timestamp_source: str
    cmd_write_start_s: float
    cmd_write_return_s: float
    cmd_sync_done_s: Optional[float]
    stop_write_start_s: float
    stop_write_return_s: float
    stop_sync_done_s: Optional[float]
    command_time_s: float
    stop_time_s: float
    command_duration_s: float
    frame_time_source: str = ""
    sensor_timestamp_domain: str = ""
    sensor_to_host_scale: Optional[float] = None
    sensor_alignment_rmse_ms: Optional[float] = None
    median_range_m: Optional[float] = None
    baseline_angle_deg: Optional[float] = None
    baseline_noise_deg: Optional[float] = None
    baseline_rate_deg_s: Optional[float] = None
    observed_angle_sign: Optional[float] = None
    last_still_frame_i: Optional[int] = None
    last_still_host_mono_s: Optional[float] = None
    last_still_analysis_time_s: Optional[float] = None
    last_still_sensor_timestamp_ms: Optional[float] = None
    first_motion_frame_i: Optional[int] = None
    motion_host_mono_s: Optional[float] = None
    motion_analysis_time_s: Optional[float] = None
    motion_sensor_timestamp_ms: Optional[float] = None
    motion_onset_estimate_analysis_time_s: Optional[float] = None
    onset_interval_low_s: Optional[float] = None
    onset_interval_high_s: Optional[float] = None
    dead_time_observed_s: Optional[float] = None
    dead_time_estimated_s: Optional[float] = None
    onset_threshold_deg: Optional[float] = None
    onset_still_threshold_deg: Optional[float] = None
    stop_observation_frame_i: Optional[int] = None
    stop_observation_host_mono_s: Optional[float] = None
    stop_observation_analysis_time_s: Optional[float] = None
    stop_observation_sensor_timestamp_ms: Optional[float] = None
    stop_rate_observed_deg_s: Optional[float] = None
    angle_at_stop_observation_deg: Optional[float] = None
    inertia_frame_i: Optional[int] = None
    inertia_host_mono_s: Optional[float] = None
    inertia_analysis_time_s: Optional[float] = None
    inertia_sensor_timestamp_ms: Optional[float] = None
    angle_at_inertia_horizon_deg: Optional[float] = None
    inertia_rotation_deg: Optional[float] = None
    observed_total_rotation_deg: Optional[float] = None
    accel_end_frame_i: Optional[int] = None
    accel_end_host_mono_s: Optional[float] = None
    accel_end_analysis_time_s: Optional[float] = None
    accel_end_sensor_timestamp_ms: Optional[float] = None
    accel_end_censored: bool = True
    drive_usable: bool = False
    inertia_usable: bool = False
    exclude_reasons: list[str] = field(default_factory=list)
    inertia_exclude_reasons: list[str] = field(default_factory=list)
    drive_time_from_onset_s: np.ndarray = field(
        default_factory=lambda: np.empty(0), repr=False
    )
    drive_angle_from_onset_deg: np.ndarray = field(
        default_factory=lambda: np.empty(0), repr=False
    )
    drive_frame_i: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int), repr=False
    )
    drive_host_mono_s: np.ndarray = field(
        default_factory=lambda: np.empty(0), repr=False
    )
    drive_sensor_timestamp_ms: np.ndarray = field(
        default_factory=lambda: np.empty(0), repr=False
    )

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("drive_time_from_onset_s", None)
        row.pop("drive_angle_from_onset_deg", None)
        row.pop("drive_frame_i", None)
        row.pop("drive_host_mono_s", None)
        row.pop("drive_sensor_timestamp_ms", None)
        row["exclude_reasons"] = "; ".join(self.exclude_reasons)
        row["inertia_exclude_reasons"] = "; ".join(
            self.inertia_exclude_reasons
        )
        return row


@dataclass(frozen=True)
class PhaseFit:
    accel_deg_s2: float
    accel_duration_sec: float
    max_rate_deg_s: float
    rmse_deg: float
    n_samples: int
    n_segments: int
    cruise_segments: int
    identifiable: bool
    note: str


@dataclass(frozen=True)
class InertiaFit:
    intercept_deg: float
    slope_sec: float
    rmse_deg: float
    n_segments: int
    speed_min_deg_s: float
    speed_max_deg_s: float
    identifiable: bool
    note: str


@dataclass(frozen=True)
class RotationFitResult:
    model: Optional[LogRotationModel]
    report: dict[str, Any]
    segments: tuple[AnalysedSegment, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path.name}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"expected a JSON object at {path.name}:{line_number}"
                )
            obj["_line_number"] = line_number
            records.append(obj)
    return records


def _payload_strength(movement: str, payload: Any) -> Optional[int]:
    if movement not in ROTATE_MOVEMENTS:
        return 0 if movement == "stop" else None
    try:
        values = [int(value) for value in payload]
    except (TypeError, ValueError):
        return None
    if len(values) < 2:
        return None
    return abs(values[1] - 127)


def _logical_windows(
    recording: str, records: Sequence[dict[str, Any]], settings: FitSettings,
) -> list[CommandWindow]:
    """Compatibility fallback for old logs without actual CAN events."""
    begins = {
        rec.get("step"): rec for rec in records if rec.get("phase") == "begin"
    }
    commands = sorted(
        (rec for rec in records if rec.get("phase") == "cmd"),
        key=lambda rec: float(rec.get("t_mono", math.inf)),
    )
    result: list[CommandWindow] = []
    for index, rec in enumerate(commands[:-1]):
        cmd = str(rec.get("cmd", ""))
        if cmd not in LOGICAL_ROTATE_COMMANDS:
            continue
        following = commands[index + 1]
        t0 = _finite(rec.get("t_mono"))
        t1 = _finite(following.get("t_mono"))
        if t0 is None or t1 is None or t1 <= t0:
            continue
        step = rec.get("step")
        begin = begins.get(step, {})
        strength_value = (begin.get("params") or {}).get(
            "joystick_deflection"
        )
        try:
            strength = int(strength_value)
        except (TypeError, ValueError):
            strength = None
        movement = (
            "rotate_left_slow" if cmd == "ROT_LEFT" else "rotate_right_slow"
        )
        start = CanBoundary(
            movement, step, t0, t0, str(rec.get("t_iso", "")), strength,
            source="logical_cmd_fallback",
        )
        stop = CanBoundary(
            "stop", following.get("step"), t1, t1,
            str(following.get("t_iso", "")), 0,
            source="logical_cmd_fallback",
        )
        next_active = None
        for later in commands[index + 2 :]:
            if str(later.get("cmd", "")) != "STOP":
                next_active = _finite(later.get("t_mono"))
                break
        result.append(CommandWindow(
            recording, step, cmd, movement, strength, start, stop, next_active,
        ))
    return result


def load_command_windows(
    control_path: Path, settings: FitSettings,
) -> list[CommandWindow]:
    """Read real command transitions, sorting by timestamps rather than JSON order."""
    records = _read_jsonl(control_path)
    recording = control_path.name[: -len("_control_seq.jsonl")]
    boundaries: list[CanBoundary] = []

    for rec in records:
        if rec.get("phase") != "can_tx":
            continue
        if rec.get("event") != "movement_write":
            continue
        if rec.get("source") != "command_burst":
            continue
        burst_index = rec.get("burst_index")
        if burst_index not in (None, 1, "1"):
            continue
        movement = str(rec.get("movement", ""))
        if not movement:
            continue
        start_ms = _finite(rec.get("write_start_host_mono_ms"))
        return_ms = _finite(rec.get("write_return_host_mono_ms"))
        fallback = _finite(rec.get("t_mono_ms"))
        if start_ms is None:
            start_ms = fallback
        if return_ms is None:
            return_ms = fallback
        if start_ms is None or return_ms is None:
            continue
        if return_ms < start_ms:
            start_ms, return_ms = return_ms, start_ms
        payload = tuple(int(x) for x in (rec.get("payload") or ()))
        boundaries.append(CanBoundary(
            movement=movement,
            step=rec.get("step"),
            write_start_s=start_ms / 1000.0,
            write_return_s=return_ms / 1000.0,
            t_iso=str(rec.get("t_iso", "")),
            strength=_payload_strength(movement, payload),
            payload=payload,
        ))

    boundaries.sort(key=lambda b: (b.write_start_s, b.write_return_s))
    if not boundaries:
        return _logical_windows(recording, records, settings)

    # Async trace writing means sync records need timestamp-based association.
    sync_records = sorted(
        (
            rec for rec in records
            if rec.get("phase") == "can_tx"
            and rec.get("event") == "command_sync_done"
        ),
        key=lambda rec: float(rec.get("sync_done_host_mono_ms", math.inf)),
    )
    for boundary in boundaries:
        candidates = []
        for rec in sync_records:
            if rec.get("movement") != boundary.movement:
                continue
            if rec.get("step") != boundary.step:
                continue
            done_ms = _finite(rec.get("sync_done_host_mono_ms"))
            if done_ms is None:
                continue
            done_s = done_ms / 1000.0
            if boundary.write_start_s <= done_s <= boundary.write_start_s + 1.0:
                candidates.append(done_s)
        if candidates:
            boundary.sync_done_s = min(candidates)

    windows: list[CommandWindow] = []
    for i, start in enumerate(boundaries):
        if start.movement not in ROTATE_MOVEMENTS:
            continue
        transition: Optional[CanBoundary] = None
        transition_index = -1
        for j in range(i + 1, len(boundaries)):
            if boundaries[j].movement != start.movement:
                transition = boundaries[j]
                transition_index = j
                break
        if transition is None:
            continue
        t_start = start.nominal_time(settings.can_time_point)
        t_stop = transition.nominal_time(settings.can_time_point)
        if t_stop <= t_start:
            continue
        next_active = None
        for later in boundaries[transition_index + 1 :]:
            if later.movement in ACTIVE_MOVEMENTS:
                next_active = later.nominal_time(settings.can_time_point)
                break
        cmd = "ROT_LEFT" if start.movement == "rotate_left_slow" else "ROT_RIGHT"
        windows.append(CommandWindow(
            recording=recording,
            step=start.step,
            command=cmd,
            movement=start.movement,
            strength=start.strength,
            start=start,
            stop=transition,
            next_active_command_s=next_active,
        ))
    return windows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _frame_analysis_timeline(
    host_s: np.ndarray,
    sensor_ms: np.ndarray,
    sensor_domains: Sequence[str],
) -> tuple[np.ndarray, str, Optional[float], Optional[float]]:
    """Map capture timestamps onto the command's host-monotonic coordinate.

    Absolute sensor and host timestamps generally have unrelated epochs.  Their
    *differences* are usable, so a robust affine clock mapping is estimated from
    all accepted frames.  This removes delivery-jitter from derivatives while
    retaining the median sensor-to-host latency in command/onset delays.  If the
    sensor clock is missing, changes domain, resets, or has an implausible rate,
    the original host arrival timestamps are used conservatively.
    """
    fallback = (
        np.asarray(host_s, dtype=float).copy(),
        "camera_input_host_mono_ms",
        None,
        None,
    )
    if len(host_s) < 3 or len(sensor_ms) != len(host_s):
        return fallback
    if not np.all(np.isfinite(sensor_ms)):
        return fallback
    domains = {str(value).strip() for value in sensor_domains if str(value).strip()}
    if len(domains) != 1:
        return fallback

    x = (np.asarray(sensor_ms, dtype=float) - float(sensor_ms[0])) / 1000.0
    y = np.asarray(host_s, dtype=float) - float(host_s[0])
    if np.any(np.diff(x) <= 0.0) or x[-1] <= 0.0 or y[-1] <= 0.0:
        return fallback
    span_scale = float(y[-1] / x[-1])
    if not 0.95 <= span_scale <= 1.05:
        return fallback

    design = np.column_stack([np.ones_like(x), x])
    initial, *_ = np.linalg.lstsq(design, y, rcond=None)
    initial[1] = min(1.05, max(0.95, float(initial[1])))
    initial_residual = design @ initial - y
    scale = max(0.002, robust_sigma(initial_residual))
    fitted = least_squares(
        lambda p: (p[0] + p[1] * x - y) / scale,
        initial,
        bounds=([-np.inf, 0.95], [np.inf, 1.05]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=2000,
    )
    offset, clock_scale = map(float, fitted.x)
    if not 0.951 <= clock_scale <= 1.049:
        return fallback
    mapped = float(host_s[0]) + offset + clock_scale * x
    if np.any(np.diff(mapped) <= 0.0):
        return fallback
    residual_s = mapped - np.asarray(host_s, dtype=float)
    rmse_ms = 1000.0 * float(np.sqrt(np.mean(residual_s ** 2)))
    return (
        mapped,
        "camera_sensor_timestamp_ms_affine_to_host_mono",
        clock_scale,
        rmse_ms,
    )


def load_frame_series(
    timing_path: Path,
    results_path: Path,
    *,
    angle_column: str,
    frame_id_column: str = "frame_i",
    host_time_column: str = "camera_input_host_mono_ms",
    sensor_time_column: str = "camera_sensor_timestamp_ms",
    sensor_domain_column: str = "camera_timestamp_domain",
    valid_columns: Sequence[str] = (),
    confidence_column: Optional[str] = None,
    min_confidence: Optional[float] = None,
    model_id_column: Optional[str] = None,
    angle_period_deg: float = 360.0,
) -> FrameSeries:
    """Join new inference results to native frame timing using ``frame_i``."""
    validate_fit_angle_contract(angle_column)
    timing_rows = _read_csv_rows(timing_path)
    result_rows = timing_rows if timing_path.resolve() == results_path.resolve() \
        else _read_csv_rows(results_path)
    if not result_rows:
        raise ValueError(f"no inference rows in {results_path}")
    if angle_column not in result_rows[0]:
        raise ValueError(
            f"angle column {angle_column!r} missing from {results_path.name}"
        )

    timing_by_frame: dict[int, dict[str, str]] = {}
    for row in timing_rows:
        value = _finite(row.get(frame_id_column))
        if value is not None:
            timing_by_frame[int(value)] = row

    auto_valid = ("inference_ran", "model_det_ok", "pnp_ok")
    samples: list[tuple[int, float, float, str, float, float]] = []
    model_ids: set[str] = set()
    for result in result_rows:
        frame_value = _finite(result.get(frame_id_column))
        if frame_value is None:
            continue
        frame_i = int(frame_value)
        timing = timing_by_frame.get(frame_i)
        if timing is None:
            continue
        angle = _finite(result.get(angle_column))
        host_ms = _finite(timing.get(host_time_column))
        if angle is None or host_ms is None:
            continue

        merged = dict(timing)
        merged.update({key: value for key, value in result.items() if value != ""})
        # When results come from a new model, do not accidentally gate them on
        # validity flags produced by the old model in the timing CSV.
        same_file = timing_path.resolve() == results_path.resolve()
        checks = tuple(valid_columns) if valid_columns else tuple(
            name for name in auto_valid
            if name in (merged if same_file else result)
        )
        result_contract = merged if same_file else result
        if any(not _truthy(result_contract.get(name)) for name in checks):
            continue
        if confidence_column and min_confidence is not None:
            confidence = _finite(result_contract.get(confidence_column))
            if confidence is None or confidence < float(min_confidence):
                continue
        if model_id_column:
            model_id = str(result_contract.get(model_id_column, "")).strip()
            if not model_id:
                raise ValueError(
                    f"empty model id in column {model_id_column!r} for "
                    f"frame {frame_i} of {results_path.name}"
                )
            model_ids.add(model_id)
        sensor_ms = _finite(timing.get(sensor_time_column))
        pos_x = _finite(merged.get("pos_x_m"))
        pos_z = _finite(merged.get("pos_z_m"))
        range_m = (
            math.hypot(pos_x, pos_z)
            if pos_x is not None and pos_z is not None else math.nan
        )
        samples.append((
            frame_i, host_ms / 1000.0,
            math.nan if sensor_ms is None else sensor_ms,
            str(timing.get(sensor_domain_column, "")), angle, range_m,
        ))

    if len(samples) < 3:
        raise ValueError(f"fewer than three valid joined frames in {results_path}")
    if len(model_ids) > 1:
        raise ValueError(
            f"mixed inference model ids in {results_path}: {sorted(model_ids)}"
        )
    samples.sort(key=lambda row: (row[1], row[0]))

    # Host timestamps can repeat on coarse clocks. Keep one row per monotonic
    # instant because derivatives cannot use dt=0.
    deduped: list[tuple[int, float, float, str, float, float]] = []
    for sample in samples:
        if deduped and sample[1] <= deduped[-1][1]:
            if sample[1] == deduped[-1][1]:
                deduped[-1] = sample
            continue
        deduped.append(sample)
    frame_i = np.asarray([row[0] for row in deduped], dtype=int)
    host_s = np.asarray([row[1] for row in deduped], dtype=float)
    sensor_ms = np.asarray([row[2] for row in deduped], dtype=float)
    sensor_domains = tuple(row[3] for row in deduped)
    angle = unwrap_degrees([row[4] for row in deduped], angle_period_deg)
    ranges = np.asarray([row[5] for row in deduped], dtype=float)
    analysis_s, time_source, clock_scale, alignment_rmse_ms = (
        _frame_analysis_timeline(host_s, sensor_ms, sensor_domains)
    )
    return FrameSeries(
        recording=timing_path.name[: -len("_inference_timing.csv")],
        frame_i=frame_i,
        host_mono_s=host_s,
        analysis_time_s=analysis_s,
        sensor_timestamp_ms=sensor_ms,
        sensor_domain=sensor_domains,
        angle_deg=angle,
        range_m=ranges,
        source_path=str(results_path),
        frame_time_source=time_source,
        sensor_to_host_scale=clock_scale,
        sensor_alignment_rmse_ms=alignment_rmse_ms,
        model_ids=tuple(sorted(model_ids)),
    )


def _local_polynomial(
    t: np.ndarray, y: np.ndarray, window_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local quadratic value/rate estimates for irregular camera intervals."""
    n = len(t)
    smooth = np.empty(n, dtype=float)
    rate = np.empty(n, dtype=float)
    half = 0.5 * float(window_sec)
    for i in range(n):
        indices = np.where(np.abs(t - t[i]) <= half)[0]
        if len(indices) < 3:
            nearest = np.argsort(np.abs(t - t[i]))[: min(5, n)]
            indices = np.sort(nearest)
        x = t[indices] - t[i]
        order = 2 if len(indices) >= 4 else 1
        design = np.column_stack([x ** power for power in range(order + 1)])
        if half > 0.0:
            u = np.minimum(1.0, np.abs(x) / max(half, 1e-6))
            weights = np.maximum(0.05, (1.0 - u ** 3) ** 3)
        else:
            weights = np.ones_like(x)
        root_w = np.sqrt(weights)
        coeff, *_ = np.linalg.lstsq(
            design * root_w[:, None], y[indices] * root_w, rcond=None
        )
        smooth[i] = coeff[0]
        rate[i] = coeff[1] if len(coeff) > 1 else 0.0
    return smooth, rate


def _robust_stop_state(
    t: np.ndarray,
    angle: np.ndarray,
    reference_time_s: float,
    window_sec: float,
    noise_deg: float,
) -> Optional[tuple[float, float]]:
    mask = (t <= reference_time_s + 1e-9) & (
        t >= reference_time_s - window_sec
    )
    x = t[mask] - reference_time_s
    y = angle[mask]
    if len(x) < 3 or np.ptp(x) <= 0.0:
        return None
    order = 2 if len(x) >= 5 else 1
    design = np.column_stack([x ** power for power in range(order + 1)])
    initial, *_ = np.linalg.lstsq(design, y, rcond=None)
    scale = max(0.05, float(noise_deg))
    result = least_squares(
        lambda coeff: (design @ coeff - y) / scale,
        initial,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=2000,
    )
    return float(result.x[0]), float(result.x[1])


def analyse_segment(
    window: CommandWindow, frames: FrameSeries, settings: FitSettings,
) -> AnalysedSegment:
    t_cmd = window.start.nominal_time(settings.can_time_point)
    t_stop = window.stop.nominal_time(settings.can_time_point)
    segment = AnalysedSegment(
        recording=window.recording,
        step=window.step,
        command=window.command,
        movement=window.movement,
        command_direction=window.direction,
        command_strength=window.strength,
        command_timestamp_source=window.start.source,
        cmd_write_start_s=window.start.write_start_s,
        cmd_write_return_s=window.start.write_return_s,
        cmd_sync_done_s=window.start.sync_done_s,
        stop_write_start_s=window.stop.write_start_s,
        stop_write_return_s=window.stop.write_return_s,
        stop_sync_done_s=window.stop.sync_done_s,
        command_time_s=t_cmd,
        stop_time_s=t_stop,
        command_duration_s=t_stop - t_cmd,
        frame_time_source=frames.frame_time_source,
        sensor_timestamp_domain=",".join(sorted({
            str(value).strip() for value in frames.sensor_domain
            if str(value).strip()
        })),
        sensor_to_host_scale=frames.sensor_to_host_scale,
        sensor_alignment_rmse_ms=frames.sensor_alignment_rmse_ms,
    )

    if window.strength != settings.command_strength:
        segment.exclude_reasons.append(
            f"command strength is {window.strength}, expected {settings.command_strength}"
        )
    if settings.require_actual_can_timing and window.start.source != "can_tx":
        segment.exclude_reasons.append("actual CAN write timing unavailable")
    if window.stop.movement != "stop":
        segment.exclude_reasons.append(
            f"rotation ended with {window.stop.movement}, not STOP"
        )
    if segment.command_duration_s < settings.min_command_sec:
        segment.exclude_reasons.append("command duration is too short")

    end_time = t_stop + settings.inertia_horizon_sec + settings.inertia_window_sec
    mask = (
        (frames.analysis_time_s >= t_cmd - settings.pre_command_sec)
        & (frames.analysis_time_s <= end_time + 1e-9)
    )
    if int(mask.sum()) < 6:
        segment.exclude_reasons.append("too few valid inference frames")
        segment.inertia_exclude_reasons.append("too few valid inference frames")
        return segment

    t = frames.analysis_time_s[mask]
    host_t = frames.host_mono_s[mask]
    fi = frames.frame_i[mask]
    sensor = frames.sensor_timestamp_ms[mask]
    angle = frames.angle_deg[mask]
    ranges = frames.range_m[mask]
    finite_ranges = ranges[np.isfinite(ranges)]
    if len(finite_ranges):
        segment.median_range_m = float(np.median(finite_ranges))
    baseline_mask = (t >= t_cmd - settings.pre_command_sec) & (t < t_cmd)
    if int(baseline_mask.sum()) < 4:
        segment.exclude_reasons.append("fewer than four pre-command frames")
        segment.inertia_exclude_reasons.append("baseline unavailable")
        return segment

    baseline = float(np.median(angle[baseline_mask]))
    noise = robust_sigma(angle[baseline_mask])
    if int(baseline_mask.sum()) >= 3:
        baseline_rate = float(np.polyfit(
            t[baseline_mask] - t_cmd, angle[baseline_mask], 1
        )[0])
    else:
        baseline_rate = math.nan
    segment.baseline_angle_deg = baseline
    segment.baseline_noise_deg = noise
    segment.baseline_rate_deg_s = baseline_rate
    if noise > settings.max_baseline_noise_deg:
        segment.exclude_reasons.append("pre-command angle noise is too large")
    if math.isfinite(baseline_rate) and abs(baseline_rate) > settings.max_baseline_rate_deg_s:
        segment.exclude_reasons.append("vehicle was not stationary before command")

    # Infer only the sign used for normalization.  Direction labels remain in
    # the output for a left/right residual audit.
    late_mask = (t >= max(t_cmd, t_stop - 0.30)) & (t <= t_stop + 1e-9)
    if int(late_mask.sum()) < 2:
        late_mask = (
            (t >= max(t_cmd, t_stop + settings.inertia_horizon_sec - settings.inertia_window_sec))
            & (t <= end_time)
        )
    late_delta = angle[late_mask] - baseline
    if len(late_delta):
        signed_delta = float(np.median(late_delta))
        if abs(signed_delta) <= max(0.05, noise):
            signed_delta = float(late_delta[np.argmax(np.abs(late_delta))])
    else:
        signed_delta = 0.0
    if abs(signed_delta) <= max(0.05, noise):
        segment.exclude_reasons.append("rotation direction cannot be inferred")
        segment.inertia_exclude_reasons.append("rotation direction cannot be inferred")
        return segment
    observed_sign = 1.0 if signed_delta > 0.0 else -1.0
    segment.observed_angle_sign = observed_sign
    directed = observed_sign * (angle - baseline)

    smooth, rate = _local_polynomial(
        t, directed, settings.smoothing_window_sec
    )
    threshold = max(
        settings.onset_min_delta_deg,
        settings.onset_noise_multiplier * noise,
    )
    segment.onset_threshold_deg = threshold
    command_indices = np.where((t >= t_cmd - 1e-9) & (t <= t_stop + 1e-9))[0]
    confirmation_index: Optional[int] = None
    for index in command_indices:
        horizon = np.where(
            (t >= t[index]) & (t <= t[index] + settings.onset_sustain_sec)
        )[0]
        if len(horizon) < settings.onset_confirm_frames:
            continue
        if t[horizon[-1]] - t[index] < 0.6 * settings.onset_sustain_sec:
            continue
        if smooth[index] < threshold:
            continue
        if float(np.max(smooth[horizon])) < 1.5 * threshold:
            continue
        if float(np.quantile(smooth[horizon], 0.25)) < 0.60 * threshold:
            continue
        if float(np.median(rate[horizon])) < settings.onset_min_rate_deg_s:
            continue
        if float(np.mean(rate[horizon] > 0.0)) < 0.60:
            continue
        confirmation_index = int(index)
        break

    if confirmation_index is None:
        segment.exclude_reasons.append("sustained motion onset not detected before STOP")
        segment.inertia_exclude_reasons.append("motion onset unavailable")
        return segment

    # The high threshold above confirms that motion is sustained; using its
    # crossing as the onset would systematically add sqrt(threshold/a) to the
    # dead time.  Backtrack through the confirmed run to the last frame still
    # inside a much smaller noise band.  A one-frame spike is ignored because
    # later still frames occur before the eventual sustained confirmation.
    still_threshold = max(0.01, min(0.5 * threshold, 2.0 * noise))
    segment.onset_still_threshold_deg = still_threshold
    prior_indices = np.arange(confirmation_index + 1, dtype=int)
    still_candidates = prior_indices[directed[prior_indices] <= still_threshold]
    last_still_index = (
        int(still_candidates[-1]) if len(still_candidates) else None
    )
    onset_index = confirmation_index
    if last_still_index is not None:
        changed_candidates = command_indices[
            (command_indices > last_still_index)
            & (directed[command_indices] > still_threshold)
        ]
        if len(changed_candidates):
            onset_index = int(changed_candidates[0])
    if last_still_index is not None:
        gap = t[onset_index] - t[last_still_index]
        if gap > settings.max_frame_gap_sec:
            segment.exclude_reasons.append(
                f"onset is interval-censored by a {gap:.3f}s frame gap"
            )
        segment.last_still_frame_i = int(fi[last_still_index])
        segment.last_still_host_mono_s = float(host_t[last_still_index])
        segment.last_still_analysis_time_s = float(t[last_still_index])
        if math.isfinite(sensor[last_still_index]):
            segment.last_still_sensor_timestamp_ms = float(sensor[last_still_index])
        segment.onset_interval_low_s = max(
            0.0, float(t[last_still_index] - t_cmd)
        )
    segment.first_motion_frame_i = int(fi[onset_index])
    segment.motion_host_mono_s = float(host_t[onset_index])
    segment.motion_analysis_time_s = float(t[onset_index])
    if math.isfinite(sensor[onset_index]):
        segment.motion_sensor_timestamp_ms = float(sensor[onset_index])
    segment.dead_time_observed_s = max(0.0, float(t[onset_index] - t_cmd))
    segment.onset_interval_high_s = segment.dead_time_observed_s

    # Motion is interval-censored: it began after the last unchanged image and
    # no later than the first sustained changed image.  Use the interval
    # midpoint for curve alignment while retaining the actual first-motion
    # frame/time above as the requested audit event.
    if last_still_index is not None:
        onset_estimate = t_cmd + 0.5 * (
            float(segment.onset_interval_low_s)
            + float(segment.onset_interval_high_s)
        )
    else:
        onset_estimate = max(t_cmd, float(t[onset_index]))
    segment.motion_onset_estimate_analysis_time_s = float(onset_estimate)
    segment.dead_time_estimated_s = float(onset_estimate - t_cmd)

    drive_mask = (t >= t[onset_index]) & (t <= t_stop + 1e-9)
    if int(drive_mask.sum()) < settings.min_drive_frames:
        segment.exclude_reasons.append("too few command-held frames after onset")
    else:
        segment.drive_time_from_onset_s = t[drive_mask] - onset_estimate
        segment.drive_angle_from_onset_deg = directed[drive_mask]
        segment.drive_frame_i = fi[drive_mask].copy()
        segment.drive_host_mono_s = host_t[drive_mask].copy()
        segment.drive_sensor_timestamp_ms = sensor[drive_mask].copy()

    pre_stop = np.where((t >= t[onset_index]) & (t <= t_stop + 1e-9))[0]
    if len(pre_stop):
        stop_index = int(pre_stop[-1])
        segment.stop_observation_frame_i = int(fi[stop_index])
        segment.stop_observation_host_mono_s = float(host_t[stop_index])
        segment.stop_observation_analysis_time_s = float(t[stop_index])
        if math.isfinite(sensor[stop_index]):
            segment.stop_observation_sensor_timestamp_ms = float(sensor[stop_index])
        stop_gap = t_stop - t[stop_index]
        if stop_gap > settings.max_frame_gap_sec:
            segment.inertia_exclude_reasons.append(
                f"last pre-STOP frame is {stop_gap:.3f}s old"
            )
        state = _robust_stop_state(
            t, directed, t[stop_index], settings.stop_rate_window_sec, noise
        )
        if state is None:
            segment.inertia_exclude_reasons.append("STOP rate fit has too few frames")
        else:
            stop_angle, stop_rate = state
            segment.angle_at_stop_observation_deg = stop_angle
            segment.stop_rate_observed_deg_s = stop_rate
            if stop_rate <= 0.0:
                segment.inertia_exclude_reasons.append(
                    "estimated pre-STOP rate is not positive"
                )

    target_time = t_stop + settings.inertia_horizon_sec
    post_indices = np.where(
        (t >= target_time - settings.inertia_window_sec)
        & (t <= target_time + settings.inertia_window_sec + 1e-9)
    )[0]
    if len(post_indices) < 2:
        segment.inertia_exclude_reasons.append(
            "fewer than two frames near STOP + inertia horizon"
        )
    else:
        nearest = int(post_indices[np.argmin(np.abs(t[post_indices] - target_time))])
        segment.inertia_frame_i = int(fi[nearest])
        segment.inertia_host_mono_s = float(host_t[nearest])
        segment.inertia_analysis_time_s = float(t[nearest])
        if math.isfinite(sensor[nearest]):
            segment.inertia_sensor_timestamp_ms = float(sensor[nearest])
        post_angle = float(np.median(directed[post_indices]))
        segment.angle_at_inertia_horizon_deg = post_angle
        segment.observed_total_rotation_deg = post_angle
        if post_angle <= max(0.05, noise):
            segment.inertia_exclude_reasons.append(
                "two-second total rotation is not above the baseline noise"
            )
        if segment.angle_at_stop_observation_deg is not None:
            coast = post_angle - segment.angle_at_stop_observation_deg
            segment.inertia_rotation_deg = coast
            if coast < -max(0.20, 3.0 * noise):
                segment.inertia_exclude_reasons.append(
                    "post-STOP angle reverses beyond the noise band"
                )
    if (
        window.next_active_command_s is not None
        and window.next_active_command_s <= target_time + settings.inertia_window_sec
    ):
        segment.inertia_exclude_reasons.append(
            "another active command contaminates the two-second horizon"
        )

    segment.drive_usable = not segment.exclude_reasons
    segment.inertia_usable = (
        segment.drive_usable
        and not segment.inertia_exclude_reasons
        and segment.stop_rate_observed_deg_s is not None
        and segment.inertia_rotation_deg is not None
    )
    return segment


def _apply_direction_sign_audit(
    segments: Sequence[AnalysedSegment],
) -> dict[str, Any]:
    """Reject sign-flipped outliers without fitting separate L/R magnitudes."""
    audit: dict[str, Any] = {"directions": {}}
    modal_signs: dict[str, int] = {}
    ambiguous: list[str] = []
    for direction in ("LEFT", "RIGHT"):
        candidates = [
            segment for segment in segments
            if segment.command_direction == direction
            and segment.drive_usable
            and segment.observed_angle_sign is not None
        ]
        counts = Counter(
            1 if float(segment.observed_angle_sign) > 0.0 else -1
            for segment in candidates
        )
        plus = int(counts.get(1, 0))
        minus = int(counts.get(-1, 0))
        modal: Optional[int] = None
        if plus > minus:
            modal = 1
        elif minus > plus:
            modal = -1
        elif plus + minus:
            ambiguous.append(direction)
        if modal is not None:
            modal_signs[direction] = modal
            for segment in candidates:
                sign = 1 if float(segment.observed_angle_sign) > 0.0 else -1
                if sign == modal:
                    continue
                reason = (
                    "observed rotation sign conflicts with the dominant "
                    f"{direction} sign"
                )
                segment.exclude_reasons.append(reason)
                segment.inertia_exclude_reasons.append(reason)
                segment.drive_usable = False
                segment.inertia_usable = False
        audit["directions"][direction] = {
            "positive": plus,
            "negative": minus,
            "modal_sign": modal,
        }

    opposite: Optional[bool] = None
    if "LEFT" in modal_signs and "RIGHT" in modal_signs:
        opposite = modal_signs["LEFT"] == -modal_signs["RIGHT"]
    audit["ambiguous_directions"] = ambiguous
    audit["left_right_modal_signs_are_opposite"] = opposite
    return audit


def discover_and_analyse(
    recordings_dir: Path,
    *,
    results_dir: Optional[Path] = None,
    results_suffix: str = "_inference_timing.csv",
    recording_names: Optional[Sequence[str]] = None,
    angle_column: str = FIT_ANGLE_COLUMN,
    angle_period_deg: float = 360.0,
    frame_id_column: str = "frame_i",
    valid_columns: Sequence[str] = (),
    confidence_column: Optional[str] = None,
    min_confidence: Optional[float] = None,
    model_id_column: Optional[str] = None,
    allow_mixed_model_ids: bool = False,
    settings: Optional[FitSettings] = None,
) -> tuple[list[AnalysedSegment], dict[str, Any]]:
    validate_fit_angle_contract(angle_column)
    settings = settings or FitSettings(angle_period_deg=angle_period_deg)
    results_dir = results_dir or recordings_dir
    if not recordings_dir.is_dir():
        raise FileNotFoundError(f"recordings directory not found: {recordings_dir}")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory not found: {results_dir}")
    control_paths = sorted(recordings_dir.glob("*_control_seq.jsonl"))
    available_by_name = {
        path.name[: -len("_control_seq.jsonl")]: path for path in control_paths
    }
    if recording_names is None:
        selected_names = sorted(available_by_name)
    else:
        requested_names = {str(name).strip() for name in recording_names}
        if "" in requested_names:
            raise ValueError("recording_names cannot contain an empty name")
        if not requested_names:
            raise ValueError("recording_names must select at least one recording")
        missing_names = sorted(requested_names - set(available_by_name))
        if missing_names:
            raise ValueError(
                "selected recording control log(s) not found: "
                + ", ".join(missing_names)
            )
        selected_names = sorted(requested_names)
    selected_paths = [available_by_name[name] for name in selected_names]
    segments: list[AnalysedSegment] = []
    file_report: dict[str, Any] = {
        "angle_column": FIT_ANGLE_COLUMN,
        "angle_domain": FIT_ANGLE_DOMAIN,
        "recordings_available": len(available_by_name),
        "recordings_selected": selected_names,
        "recordings_excluded_by_selection": sorted(
            set(available_by_name) - set(selected_names)
        ),
        "recordings_seen": 0,
        "recordings_loaded": 0,
        "recording_errors": {},
        "result_files": [],
        "model_ids": [],
        "recording_model_ids": {},
        "recording_timebases": {},
    }
    all_model_ids: set[str] = set()
    for control_path in selected_paths:
        file_report["recordings_seen"] += 1
        prefix = control_path.name[: -len("_control_seq.jsonl")]
        timing_path = recordings_dir / f"{prefix}_inference_timing.csv"
        results_path = results_dir / f"{prefix}{results_suffix}"
        if not timing_path.exists() or not results_path.exists():
            file_report["recording_errors"][prefix] = "timing or result CSV missing"
            continue
        try:
            frames = load_frame_series(
                timing_path,
                results_path,
                angle_column=angle_column,
                frame_id_column=frame_id_column,
                valid_columns=valid_columns,
                confidence_column=confidence_column,
                min_confidence=min_confidence,
                model_id_column=model_id_column,
                angle_period_deg=angle_period_deg,
            )
            windows = load_command_windows(control_path, settings)
        except Exception as exc:
            file_report["recording_errors"][prefix] = (
                f"{type(exc).__name__}: {exc}"
            )
            continue
        for window in windows:
            segments.append(analyse_segment(window, frames, settings))
        file_report["recordings_loaded"] += 1
        file_report["result_files"].append(str(results_path))
        file_report["recording_timebases"][prefix] = {
            "source": frames.frame_time_source,
            "sensor_to_host_scale": frames.sensor_to_host_scale,
            "sensor_alignment_rmse_ms": frames.sensor_alignment_rmse_ms,
        }
        all_model_ids.update(frames.model_ids)
        recording_ids = set(frames.model_ids)
        # Native timing rows have no weight hash.  Their companion metadata at
        # least records the model basename, which prevents known versions from
        # being pooled silently.  New result files should provide a hash via
        # model_id_column.
        if results_path.resolve() == timing_path.resolve() and not recording_ids:
            meta_path = recordings_dir / f"{prefix}_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    model_name = str(meta.get("model", "")).strip()
                    model_hash = str(meta.get("model_sha256", "")).strip()
                    meta_model = (
                        f"sha256:{model_hash}" if model_hash
                        else f"filename:{model_name}" if model_name else ""
                    )
                except (OSError, json.JSONDecodeError):
                    meta_model = ""
                if meta_model:
                    recording_ids.add(meta_model)
                    all_model_ids.add(meta_model)
        file_report["recording_model_ids"][prefix] = sorted(recording_ids)
    if len(all_model_ids) > 1 and not allow_mixed_model_ids:
        raise ValueError(
            "refusing to pool different inference model ids: "
            + ", ".join(sorted(all_model_ids))
        )
    file_report["model_ids"] = sorted(all_model_ids)
    file_report["allow_mixed_model_ids"] = bool(allow_mixed_model_ids)
    file_report["mixed_model_ids_detected"] = len(all_model_ids) > 1
    unidentified = [
        name for name, identifiers in file_report["recording_model_ids"].items()
        if not identifiers
    ]
    file_report["unidentified_model_recordings"] = unidentified
    file_report["direction_sign_audit"] = _apply_direction_sign_audit(segments)
    return segments, file_report


def _driven_angle(q: np.ndarray, accel: float, accel_duration: float) -> np.ndarray:
    q = np.maximum(0.0, np.asarray(q, dtype=float))
    vmax = accel * accel_duration
    return np.where(
        q <= accel_duration,
        0.5 * accel * q * q,
        vmax * (q - 0.5 * accel_duration),
    )


def fit_drive_phases(
    segments: Sequence[AnalysedSegment], settings: FitSettings,
) -> PhaseFit:
    usable = [
        segment for segment in segments
        if segment.drive_usable and len(segment.drive_time_from_onset_s) >= settings.min_drive_frames
    ]
    if not usable:
        return PhaseFit(math.nan, math.nan, math.nan, math.nan, 0, 0, 0,
                        False, "no usable command-held traces")
    max_time = max(float(np.max(s.drive_time_from_onset_s)) for s in usable)
    if max_time <= 0.10:
        return PhaseFit(math.nan, math.nan, math.nan, math.nan, 0, len(usable),
                        0, False, "post-onset command duration is too short")

    lower_tau = max(0.05, min(0.15, 0.25 * max_time))
    upper_tau = max(lower_tau + 0.05, 1.25 * max_time)

    def residual(params: np.ndarray) -> np.ndarray:
        accel, tau = params
        chunks = []
        for segment in usable:
            q = segment.drive_time_from_onset_s
            y = segment.drive_angle_from_onset_deg
            scale = max(0.08, float(segment.baseline_noise_deg or 0.0))
            # Each run contributes comparable total weight regardless of FPS.
            chunks.append(
                (_driven_angle(q, accel, tau) - y)
                / (scale * math.sqrt(max(1, len(q))))
            )
        return np.concatenate(chunks)

    starts = []
    for tau0 in np.linspace(lower_tau, upper_tau, 7):
        estimates = []
        for segment in usable:
            q = segment.drive_time_from_onset_s
            y = segment.drive_angle_from_onset_deg
            positive = q > 0.05
            if np.any(positive):
                estimates.extend((2.0 * y[positive] / (q[positive] ** 2)).tolist())
        a0 = float(np.median([x for x in estimates if 0.05 < x < 500.0])) \
            if any(0.05 < x < 500.0 for x in estimates) else 10.0
        starts.append(np.array([min(400.0, max(0.1, a0)), tau0]))

    best = None
    for initial in starts:
        candidate = least_squares(
            residual,
            initial,
            bounds=([0.05, lower_tau], [500.0, upper_tau]),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=10000,
        )
        if best is None or candidate.cost < best.cost:
            best = candidate
    assert best is not None
    accel, tau = (float(best.x[0]), float(best.x[1]))
    vmax = accel * tau
    raw_errors = np.concatenate([
        _driven_angle(s.drive_time_from_onset_s, accel, tau)
        - s.drive_angle_from_onset_deg
        for s in usable
    ])
    cruise_segments = sum(
        float(np.max(s.drive_time_from_onset_s))
        >= tau + settings.min_cruise_observation_sec
        for s in usable
    )
    near_upper_bound = tau >= 0.98 * upper_tau
    identifiable = (
        cruise_segments >= settings.min_cruise_segments and not near_upper_bound
    )
    note = ""
    if cruise_segments < settings.min_cruise_segments:
        note = (
            f"need {settings.min_cruise_segments} runs that observe the cruise "
            f"plateau for at least {settings.min_cruise_observation_sec:.2f}s"
        )
    elif near_upper_bound:
        note = "acceleration duration is at its data-dependent upper bound"
    return PhaseFit(
        accel_deg_s2=accel,
        accel_duration_sec=tau,
        max_rate_deg_s=vmax,
        rmse_deg=float(np.sqrt(np.mean(raw_errors ** 2))),
        n_samples=len(raw_errors),
        n_segments=len(usable),
        cruise_segments=int(cruise_segments),
        identifiable=identifiable,
        note=note,
    )


def fit_inertia(
    segments: Sequence[AnalysedSegment], settings: FitSettings,
) -> InertiaFit:
    usable = [segment for segment in segments if segment.inertia_usable]
    if not usable:
        return InertiaFit(math.nan, math.nan, math.nan, 0, math.nan, math.nan,
                          False, "no usable STOP + 2 s measurements")
    speed = np.asarray([s.stop_rate_observed_deg_s for s in usable], dtype=float)
    coast = np.asarray([s.inertia_rotation_deg for s in usable], dtype=float)
    scale = max(0.10, robust_sigma(coast))
    if settings.fit_inertia_intercept:
        initial_slope = max(0.0, float(np.polyfit(speed, coast, 1)[0])) \
            if len(speed) >= 2 and np.ptp(speed) > 0 else 0.0
        initial_intercept = max(0.0, float(np.median(coast - initial_slope * speed)))
        result = least_squares(
            lambda p: (p[0] + p[1] * speed - coast) / scale,
            [initial_intercept, initial_slope],
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            loss="soft_l1",
            f_scale=1.0,
        )
        intercept, slope = map(float, result.x)
        minimum_count = 4
    else:
        denom = float(np.dot(speed, speed))
        initial_slope = max(0.0, float(np.dot(speed, coast) / denom)) \
            if denom > 0.0 else 0.0
        result = least_squares(
            lambda p: (p[0] * speed - coast) / scale,
            [initial_slope],
            bounds=([0.0], [np.inf]),
            loss="soft_l1",
            f_scale=1.0,
        )
        intercept, slope = 0.0, float(result.x[0])
        minimum_count = 3
    predicted = intercept + slope * speed
    span = float(np.ptp(speed))
    identifiable = len(speed) >= minimum_count and span >= settings.min_inertia_speed_span_deg_s
    note = ""
    if len(speed) < minimum_count:
        note = f"need at least {minimum_count} independent STOP runs"
    elif span < settings.min_inertia_speed_span_deg_s:
        note = "STOP speeds do not span enough range for a slope"
    return InertiaFit(
        intercept_deg=intercept,
        slope_sec=slope,
        rmse_deg=float(np.sqrt(np.mean((predicted - coast) ** 2))),
        n_segments=len(usable),
        speed_min_deg_s=float(np.min(speed)),
        speed_max_deg_s=float(np.max(speed)),
        identifiable=identifiable,
        note=note,
    )


def _distribution(values: Sequence[float]) -> dict[str, Optional[float]]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "n": 0, "median": None, "mean": None, "sd": None,
            "mad_sigma": None, "p10": None, "p90": None,
        }
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "mad_sigma": robust_sigma(x),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def _metrics(errors: Sequence[float]) -> dict[str, Optional[float]]:
    x = np.asarray(errors, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "bias_deg": None, "mae_deg": None,
                "rmse_deg": None, "p90_abs_deg": None}
    return {
        "n": int(len(x)),
        "bias_deg": float(np.mean(x)),
        "mae_deg": float(np.mean(np.abs(x))),
        "rmse_deg": float(np.sqrt(np.mean(x ** 2))),
        "p90_abs_deg": float(np.percentile(np.abs(x), 90)),
    }


def _make_model(
    segments: Sequence[AnalysedSegment], dead_time: float,
    phase: PhaseFit, inertia: InertiaFit,
) -> LogRotationModel:
    usable = [s for s in segments if s.inertia_usable]
    holds = [s.command_duration_s for s in usable]
    angles = [s.observed_total_rotation_deg for s in usable]
    return LogRotationModel(
        command_strength=30,
        dead_time_sec=float(dead_time),
        accel_duration_sec=phase.accel_duration_sec,
        accel_deg_s2=phase.accel_deg_s2,
        max_rate_deg_s=phase.max_rate_deg_s,
        inertia_intercept_deg=inertia.intercept_deg,
        inertia_slope_sec=inertia.slope_sec,
        fitted_min_hold_sec=float(min(holds)),
        fitted_max_hold_sec=float(max(holds)),
        fitted_min_angle_deg=float(min(angles)),
        fitted_max_angle_deg=float(max(angles)),
    )


def _fit_once(
    segments: Sequence[AnalysedSegment], settings: FitSettings,
) -> tuple[float, PhaseFit, InertiaFit, Optional[LogRotationModel]]:
    delays = [
        s.dead_time_estimated_s for s in segments
        if s.drive_usable and s.dead_time_estimated_s is not None
    ]
    dead_time = float(np.median(delays)) if delays else math.nan
    phase = fit_drive_phases(segments, settings)
    inertia = fit_inertia(segments, settings)
    model = None
    if (
        math.isfinite(dead_time)
        and phase.identifiable
        and inertia.identifiable
        and any(s.inertia_usable for s in segments)
    ):
        model = _make_model(segments, dead_time, phase, inertia)
    return dead_time, phase, inertia, model


def fit_rotation_response(
    segments: Sequence[AnalysedSegment],
    *,
    settings: Optional[FitSettings] = None,
    angle_column: str = FIT_ANGLE_COLUMN,
    angle_domain: str = FIT_ANGLE_DOMAIN,
    source_report: Optional[dict[str, Any]] = None,
    bootstrap_runs: int = 0,
    random_seed: int = 20260904,
) -> RotationFitResult:
    settings = settings or FitSettings()
    validate_fit_angle_contract(angle_column, angle_domain)
    if source_report is not None:
        if source_report.get("angle_column") != FIT_ANGLE_COLUMN:
            raise ValueError(
                "source_report angle_column is missing or does not match the "
                "yaw_deg fit contract"
            )
        if source_report.get("angle_domain") != FIT_ANGLE_DOMAIN:
            raise ValueError(
                "source_report angle_domain is missing or does not match the "
                "heading fit contract"
            )
    dead_time, phase, inertia, model = _fit_once(segments, settings)
    delays = [
        s.dead_time_estimated_s for s in segments
        if s.drive_usable and s.dead_time_estimated_s is not None
    ]

    validation_rows = []
    if model is not None:
        for segment in segments:
            if not segment.inertia_usable:
                continue
            predicted = model.total_angle(segment.command_duration_s)
            observed = float(segment.observed_total_rotation_deg)
            validation_rows.append({
                "recording": segment.recording,
                "step": segment.step,
                "direction": segment.command_direction,
                "hold_sec": segment.command_duration_s,
                "observed_deg": observed,
                "predicted_deg": predicted,
                "error_deg": predicted - observed,
            })

    loo_rows = []
    loo_recordings_successful: set[str] = set()
    held_out_recordings = sorted({
        segment.recording for segment in segments if segment.inertia_usable
    })
    for held_out_recording in held_out_recordings:
        train = [
            s for s in segments
            if s.drive_usable and s.recording != held_out_recording
        ]
        if len(train) < 3:
            continue
        _d, _p, _i, loo_model = _fit_once(train, settings)
        if loo_model is None:
            continue
        held_out_segments = [
            segment for segment in segments
            if segment.inertia_usable
            and segment.recording == held_out_recording
        ]
        if not held_out_segments:
            continue
        loo_recordings_successful.add(held_out_recording)
        for held_out in held_out_segments:
            predicted = loo_model.total_angle(held_out.command_duration_s)
            observed = float(held_out.observed_total_rotation_deg)
            loo_rows.append({
                "recording": held_out.recording,
                "step": held_out.step,
                "direction": held_out.command_direction,
                "error_deg": predicted - observed,
            })

    bootstrap_values: dict[str, list[float]] = {
        "dead_time_sec": [], "accel_deg_s2": [], "accel_duration_sec": [],
        "max_rate_deg_s": [], "inertia_intercept_deg": [],
        "inertia_slope_sec": [],
    }
    rng = np.random.default_rng(random_seed)
    cluster_names = sorted({
        segment.recording for segment in segments if segment.drive_usable
    })
    clusters = {
        name: [
            segment for segment in segments
            if segment.drive_usable and segment.recording == name
        ]
        for name in cluster_names
    }
    for _ in range(max(0, int(bootstrap_runs))):
        if not cluster_names:
            break
        sampled: list[AnalysedSegment] = []
        for index in rng.integers(0, len(cluster_names), len(cluster_names)):
            sampled.extend(clusters[cluster_names[int(index)]])
        b_dead, b_phase, b_inertia, _ = _fit_once(sampled, settings)
        values = {
            "dead_time_sec": b_dead,
            "accel_deg_s2": b_phase.accel_deg_s2,
            "accel_duration_sec": b_phase.accel_duration_sec,
            "max_rate_deg_s": b_phase.max_rate_deg_s,
            "inertia_intercept_deg": b_inertia.intercept_deg,
            "inertia_slope_sec": b_inertia.slope_sec,
        }
        for key, value in values.items():
            if math.isfinite(value):
                bootstrap_values[key].append(float(value))
    bootstrap_ci = {}
    for key, values in bootstrap_values.items():
        bootstrap_ci[key] = {
            "n": len(values),
            "p2_5": float(np.percentile(values, 2.5)) if values else None,
            "p97_5": float(np.percentile(values, 97.5)) if values else None,
        }

    direction_audit = {}
    for direction in ("LEFT", "RIGHT"):
        errors = [
            row["error_deg"] for row in validation_rows
            if row["direction"] == direction
        ]
        direction_audit[direction] = _metrics(errors)

    drive_exclusions = Counter(
        reason for segment in segments for reason in segment.exclude_reasons
    )
    inertia_exclusions = Counter(
        reason
        for segment in segments
        for reason in segment.inertia_exclude_reasons
    )

    source = source_report or {}
    loo_metrics = _metrics([row["error_deg"] for row in loo_rows])
    loo_fold_rmse = {}
    for recording in sorted(loo_recordings_successful):
        errors = np.asarray([
            row["error_deg"] for row in loo_rows
            if row["recording"] == recording
        ], dtype=float)
        if len(errors):
            loo_fold_rmse[recording] = float(np.sqrt(np.mean(errors ** 2)))
    loo_equal_weight_rmse = (
        float(np.sqrt(np.mean(np.square(list(loo_fold_rmse.values())))))
        if loo_fold_rmse else None
    )
    deployment_blockers: list[str] = []
    if model is not None and not source.get("model_ids"):
        deployment_blockers.append("no inference model id/hash was recorded")
    if source.get("mixed_model_ids_detected"):
        deployment_blockers.append(
            "different inference model ids were pooled for diagnostics"
        )
    recording_errors = source.get("recording_errors") or {}
    if recording_errors:
        deployment_blockers.append(
            f"{len(recording_errors)} recording(s) were missing or failed to load"
        )
    unidentified = source.get("unidentified_model_recordings") or []
    if unidentified:
        deployment_blockers.append(
            f"{len(unidentified)} recording(s) lack model provenance"
        )
    sign_audit = source.get("direction_sign_audit") or {}
    ambiguous_directions = sign_audit.get("ambiguous_directions") or []
    if ambiguous_directions:
        deployment_blockers.append(
            "rotation sign is ambiguous for direction(s): "
            + ", ".join(str(value) for value in ambiguous_directions)
        )
    if sign_audit.get("left_right_modal_signs_are_opposite") is False:
        deployment_blockers.append(
            "LEFT and RIGHT have the same dominant observed angle sign"
        )
    if model is not None:
        if len(loo_recordings_successful) < len(held_out_recordings):
            deployment_blockers.append(
                f"only {len(loo_recordings_successful)} of "
                f"{len(held_out_recordings)} leave-one-recording-out folds "
                "could be fitted"
            )
        elif len(loo_recordings_successful) < settings.min_loo_predictions:
            deployment_blockers.append(
                f"only {len(loo_recordings_successful)} successful "
                "leave-one-recording-out folds; "
                f"need {settings.min_loo_predictions}"
            )
        elif (
            loo_equal_weight_rmse is None
            or loo_equal_weight_rmse > settings.max_loo_rmse_deg
        ):
            deployment_blockers.append(
                "leave-one-recording-out RMSE exceeds "
                f"{settings.max_loo_rmse_deg:.3f} deg"
            )

    safe_for_control = model is not None and not deployment_blockers
    if model is None:
        status = "NOT_IDENTIFIABLE"
    elif safe_for_control:
        status = "READY"
    else:
        status = "DIAGNOSTIC_ONLY"
    report: dict[str, Any] = {
        "status": status,
        "safe_for_control": safe_for_control,
        "deployment_blockers": deployment_blockers,
        "model_type": "deadtime_constant_acceleration_cruise_linear_inertia",
        "command_strength": settings.command_strength,
        "command_strength_units": "joystick_deflection_from_neutral_127",
        "angle_column": FIT_ANGLE_COLUMN,
        "angle_domain": FIT_ANGLE_DOMAIN,
        "angle_period_deg": settings.angle_period_deg,
        "timestamp_contract": {
            "command_boundary": f"first command-burst CAN write {settings.can_time_point}",
            "frame_join": "camera_input_host_mono_ms anchors sensor time to host monotonic",
            "frame_analysis": (
                "camera_sensor_timestamp_ms robust affine mapping to host monotonic "
                "when valid; otherwise camera_input_host_mono_ms"
            ),
            "capture_audit": (
                "raw camera_sensor_timestamp_ms and host arrival time retained"
            ),
            "limitation": (
                "the affine mapping removes frame-interval jitter but cannot identify "
                "absolute sensor-to-host delivery latency without an external clock event"
            ),
            "onset_is_interval_censored": True,
            "dead_time_estimator": (
                "median midpoint between the last-still frame and the first "
                "small change backtracked from a sustained-motion confirmation"
            ),
        },
        "settings": asdict(settings),
        "source": source,
        "counts": {
            "segments_total": len(segments),
            "segments_drive_usable": sum(s.drive_usable for s in segments),
            "segments_inertia_usable": sum(s.inertia_usable for s in segments),
            "drive_exclusions_by_reason": dict(drive_exclusions),
            "inertia_exclusions_by_reason": dict(inertia_exclusions),
        },
        "dead_time": _distribution(delays),
        "first_motion_frame_delay": _distribution([
            s.dead_time_observed_s for s in segments
            if s.drive_usable and s.dead_time_observed_s is not None
        ]),
        "dead_time_interval": {
            "last_still_relative_to_command_s": _distribution([
                s.onset_interval_low_s for s in segments
                if s.drive_usable and s.onset_interval_low_s is not None
            ]),
            "first_motion_relative_to_command_s": _distribution([
                s.onset_interval_high_s for s in segments
                if s.drive_usable and s.onset_interval_high_s is not None
            ]),
        },
        "can_write_bracket_ms": {
            "rotation_start": _distribution([
                1000.0 * (s.cmd_write_return_s - s.cmd_write_start_s)
                for s in segments
            ]),
            "stop": _distribution([
                1000.0 * (s.stop_write_return_s - s.stop_write_start_s)
                for s in segments
            ]),
        },
        "range_m": _distribution([
            s.median_range_m for s in segments
            if s.drive_usable and s.median_range_m is not None
        ]),
        "phase_fit": asdict(phase),
        "inertia_fit": asdict(inertia),
        "validation": {
            "in_sample": _metrics([row["error_deg"] for row in validation_rows]),
            "leave_one_recording_out": loo_metrics,
            "leave_one_recording_out_folds": {
                "attempted": len(held_out_recordings),
                "successful": len(loo_recordings_successful),
                "rmse_deg_by_recording": loo_fold_rmse,
                "equal_recording_weight_rmse_deg": loo_equal_weight_rmse,
            },
            "direction_audit": direction_audit,
            "rows": validation_rows,
            "loo_rows": loo_rows,
        },
        "bootstrap_95pct_cluster_by_recording": bootstrap_ci,
        "identifiability": {
            "dead_time": bool(delays),
            "acceleration_and_cruise": phase.identifiable,
            "linear_inertia": inertia.identifiable,
            "notes": [note for note in (phase.note, inertia.note) if note],
        },
    }
    # Fill the audit frame nearest the fitted acceleration/cruise breakpoint.
    if phase.identifiable:
        for segment in segments:
            if (
                not segment.drive_usable
                or segment.motion_onset_estimate_analysis_time_s is None
            ):
                continue
            target = (
                segment.motion_onset_estimate_analysis_time_s
                + phase.accel_duration_sec
            )
            if target > segment.stop_time_s:
                segment.accel_end_censored = True
                continue
            # We only have the selected drive samples here. Reconstruct the
            # nearest time; frame id was logged at onset and STOP, while the
            # exact phase timestamp remains the primary audit field.
            relative = segment.drive_time_from_onset_s
            if len(relative):
                nearest_relative = float(relative[np.argmin(
                    np.abs(relative - phase.accel_duration_sec)
                )])
                nearest_index = int(np.argmin(
                    np.abs(relative - phase.accel_duration_sec)
                ))
                segment.accel_end_analysis_time_s = (
                    segment.motion_onset_estimate_analysis_time_s + nearest_relative
                )
                if len(segment.drive_frame_i) > nearest_index:
                    segment.accel_end_frame_i = int(
                        segment.drive_frame_i[nearest_index]
                    )
                if len(segment.drive_host_mono_s) > nearest_index:
                    segment.accel_end_host_mono_s = float(
                        segment.drive_host_mono_s[nearest_index]
                    )
                if (
                    len(segment.drive_sensor_timestamp_ms) > nearest_index
                    and math.isfinite(
                        segment.drive_sensor_timestamp_ms[nearest_index]
                    )
                ):
                    segment.accel_end_sensor_timestamp_ms = float(
                        segment.drive_sensor_timestamp_ms[nearest_index]
                    )
                segment.accel_end_censored = False
    return RotationFitResult(model, _json_safe(report), tuple(segments))


def write_segments_csv(path: Path, segments: Iterable[AnalysedSegment]) -> None:
    rows = [segment.to_row() for segment in segments]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
