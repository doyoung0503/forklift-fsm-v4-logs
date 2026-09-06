"""Safe runtime loader for the generated log-based rotation response.

The offline fitter writes a ``LogRotationModel`` JSON artifact.  This module
validates that artifact independently of the fitting package and adapts its
piecewise response to the small interface used by the FSM v4 controller.

The active file is read once when :mod:`calib.fsm_v4.config` is imported.  A
publisher must write a sibling temporary file, flush/fsync it, and use
``os.replace``; opening and reading the final pathname once then observes
either the complete old file or the complete new file, never a hot reload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union

from .rotation_model import RotationPlan, RotationResponse, bearing_gain


__all__ = [
    "ArtifactValidationError",
    "PiecewiseRotationResponse",
    "RotationResponseSelection",
    "canonical_model_sha256",
    "load_validated_rotation_artifact",
    "select_rotation_response",
    "validate_rotation_artifact_bytes",
]


_SCHEMA_VERSION = 1
_MODEL_TYPE = "log_rotation_piecewise"
_FIT_MODEL_TYPE = "deadtime_constant_acceleration_cruise_linear_inertia"
_SHA256_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PARAMETER_KEYS = frozenset({
    "command_strength",
    "dead_time_sec",
    "accel_duration_sec",
    "accel_deg_s2",
    "max_rate_deg_s",
    "inertia_intercept_deg",
    "inertia_slope_sec",
})
_FIT_METADATA_KEYS = frozenset({
    "fitted_min_hold_sec",
    "fitted_max_hold_sec",
    "fitted_min_angle_deg",
    "fitted_max_angle_deg",
})
_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "model_type", "parameters", "fit_metadata", "fit_report",
})


class ArtifactValidationError(ValueError):
    """Raised when a generated response is not safe for this runtime."""


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ArtifactValidationError(f"{location} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactValidationError(f"{location} must be finite")
    return result


def _integer(value: object, location: str) -> int:
    if type(value) is not int:
        raise ArtifactValidationError(f"{location} must be a JSON integer")
    return value


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{location} must be a JSON object")
    return value


def _list(value: object, location: str) -> list[object]:
    if type(value) is not list:
        raise ArtifactValidationError(f"{location} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], location: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ArtifactValidationError(
            f"invalid {location}: " + "; ".join(details)
        )


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_json_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ArtifactValidationError("JSON contains a non-finite number")
    return value


def _reject_json_constant(value: str) -> None:
    raise ArtifactValidationError(f"invalid JSON numeric constant: {value}")


def canonical_model_sha256(path: Union[str, Path]) -> str:
    """Return the canonical model identity used by inference result CSVs."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class PiecewiseRotationResponse:
    """FSM-compatible view of a validated ``LogRotationModel`` artifact.

    Inertia is already the fitted post-STOP angle ``a + b * stop_rate``.  No
    legacy STOP-delay/deceleration coast or fixed coast correction is added.
    """

    dead_time_sec: float
    accel_duration_sec: float
    accel_deg_s2: float
    max_rate_deg_s: float
    inertia_intercept_deg: float
    inertia_slope_sec: float
    fitted_min_hold_sec: float
    fitted_max_hold_sec: float
    fitted_min_angle_deg: float
    fitted_max_angle_deg: float
    command_strength: int
    inertia_horizon_sec: float = 2.0
    residual_sd_deg: float = 0.0
    domain: str = "heading"
    source: str = ""

    # Lets the controller explicitly disable legacy coast corrections.
    uses_embedded_inertia: bool = True

    @property
    def startup_delay_sec(self) -> float:
        return self.dead_time_sec

    @property
    def stop_delay_sec(self) -> float:
        # The new fit represents STOP behaviour directly as an inertia angle.
        return 0.0

    @property
    def min_hold_sec(self) -> float:
        return self.fitted_min_hold_sec

    @property
    def min_total_deg(self) -> float:
        return self.fitted_min_angle_deg

    @property
    def measured_settle_p90_sec(self) -> float:
        # The fit observes post-STOP displacement at this horizon; it does not
        # claim a separately identifiable deceleration time.
        return self.inertia_horizon_sec

    def in_bearing_domain(
        self, range_m: float, rot_centre_offset_m: float,
    ) -> "PiecewiseRotationResponse":
        if self.domain == "bearing":
            return self
        gain = bearing_gain(range_m, rot_centre_offset_m)
        # If angle_b = gain * angle_h and rate_b = gain * rate_h, the inertia
        # intercept scales by gain while its slope (seconds) stays unchanged.
        return replace(
            self,
            accel_deg_s2=self.accel_deg_s2 * gain,
            max_rate_deg_s=self.max_rate_deg_s * gain,
            inertia_intercept_deg=self.inertia_intercept_deg * gain,
            fitted_min_angle_deg=self.fitted_min_angle_deg * gain,
            fitted_max_angle_deg=self.fitted_max_angle_deg * gain,
            residual_sd_deg=self.residual_sd_deg * gain,
            domain="bearing",
            source=self.source + " [bearing x%.3f]" % gain,
        )

    def driven_duration(self, hold_sec: float) -> float:
        return max(0.0, float(hold_sec) - self.dead_time_sec)

    def rate_at(self, elapsed_sec: float) -> float:
        driven = self.driven_duration(elapsed_sec)
        return min(self.max_rate_deg_s, self.accel_deg_s2 * driven)

    def angle_at(self, elapsed_sec: float) -> float:
        driven = self.driven_duration(elapsed_sec)
        ramp = min(driven, self.accel_duration_sec)
        cruise = max(0.0, driven - self.accel_duration_sec)
        return (
            0.5 * self.accel_deg_s2 * ramp * ramp
            + self.max_rate_deg_s * cruise
        )

    def inertia_angle(self, rate_deg_s: float) -> float:
        rate = abs(float(rate_deg_s))
        if rate == 0.0:
            return 0.0
        return max(
            0.0,
            self.inertia_intercept_deg + self.inertia_slope_sec * rate,
        )

    def coast(self, rate_deg_s: float) -> tuple[float, float]:
        return self.inertia_angle(rate_deg_s), self.inertia_horizon_sec

    def coast_after_hold(self, hold_sec: float) -> float:
        return self.inertia_angle(self.rate_at(hold_sec))

    def adjusted_coast_after_hold(
        self, hold_sec: float, coast_reduction_deg: float = 0.0,
    ) -> float:
        # ``coast_reduction_deg`` belongs to the old response.  Deliberately
        # ignore it: this artifact's regression already is the final inertia.
        return self.coast_after_hold(hold_sec)

    def total_angle(self, hold_sec: float) -> float:
        return self.angle_at(hold_sec) + self.coast_after_hold(hold_sec)

    def adjusted_total_angle(
        self, hold_sec: float, coast_reduction_deg: float = 0.0,
    ) -> float:
        return self.total_angle(hold_sec)

    def settle_sec(self, hold_sec: float) -> float:
        return self.inertia_horizon_sec

    def settle_wait_sec(self, hold_sec: float) -> float:
        return self.inertia_horizon_sec

    def angle_uncertainty_deg(self, hold_sec: float) -> float:
        return self.residual_sd_deg

    def _unbounded_hold(self, target_deg: float) -> Optional[float]:
        if target_deg <= 0.0:
            return 0.0
        if (
            self.inertia_intercept_deg > 0.0
            and target_deg <= self.inertia_intercept_deg
        ):
            return None
        accel_end = self.dead_time_sec + self.accel_duration_sec
        if target_deg <= self.total_angle(accel_end):
            lo = 0.0
            hi = self.accel_duration_sec
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if self.total_angle(self.dead_time_sec + mid) < target_deg:
                    lo = mid
                else:
                    hi = mid
            return self.dead_time_sec + 0.5 * (lo + hi)
        return accel_end + (
            target_deg - self.total_angle(accel_end)
        ) / self.max_rate_deg_s

    def _bounded_hold(self, target_deg: float, max_hold_sec: float) -> tuple[
        float, bool, str,
    ]:
        wanted = abs(float(target_deg))
        cap = min(float(max_hold_sec), self.fitted_max_hold_sec)
        if wanted < self.fitted_min_angle_deg:
            return 0.0, False, (
                "target below fitted minimum angle "
                f"{self.fitted_min_angle_deg:.3f} deg; no reliable command"
            )
        if wanted > self.fitted_max_angle_deg:
            return cap, False, (
                "target above fitted maximum angle "
                f"{self.fitted_max_angle_deg:.3f} deg; hold capped"
            )
        root = self._unbounded_hold(wanted)
        if root is None:
            return 0.0, False, "target is not attainable by fitted response"
        if root < self.fitted_min_hold_sec:
            return self.fitted_min_hold_sec, False, (
                "required hold is below fitted hold range"
            )
        if root > cap:
            return cap, False, "required hold exceeds supported hold cap"
        return root, True, ""

    def stop_now_margin_deg(
        self,
        elapsed_sec: float,
        measured_rate_deg_s: Optional[float] = None,
        measurement_age_sec: float = 0.0,
        coast_reduction_deg: float = 0.0,
    ) -> float:
        rate = self.rate_at(elapsed_sec)
        if measured_rate_deg_s is not None:
            rate = max(rate, abs(float(measured_rate_deg_s)))
        # Inertia is included exactly once.  The latency term covers motion
        # between the captured frame and the imminent STOP write.
        return self.inertia_angle(rate) + rate * max(
            0.0, float(measurement_age_sec)
        )

    def should_stop_now(
        self,
        remaining_deg: float,
        elapsed_sec: float,
        measured_rate_deg_s: Optional[float] = None,
        measurement_age_sec: float = 0.0,
        coast_reduction_deg: float = 0.0,
    ) -> bool:
        if elapsed_sec < self.dead_time_sec:
            return False
        return abs(float(remaining_deg)) <= self.stop_now_margin_deg(
            elapsed_sec, measured_rate_deg_s, measurement_age_sec,
        )

    def plan(
        self,
        error_deg: float,
        min_angle_deg: float = 2.5,
        max_angle_deg: float = 20.0,
        timeout_margin_sec: float = 1.0,
        max_hold_sec: float = 8.0,
        coast_reduction_deg: float = 0.0,
    ) -> RotationPlan:
        signed = float(error_deg)
        sign = 1.0 if signed > 0.0 else -1.0
        command = "ROT_RIGHT" if signed > 0.0 else "ROT_LEFT"
        wanted = abs(signed)
        target = min(wanted, float(max_angle_deg))
        notes = []
        if target < wanted:
            notes.append("target clipped to single-command limit")
        hold, feasible, model_note = self._bounded_hold(target, max_hold_sec)
        if wanted < float(min_angle_deg):
            feasible = False
            notes.append("below configured minimum command angle")
        if model_note:
            notes.append(model_note)
        stop_angle = self.angle_at(hold)
        stop_rate = self.rate_at(hold)
        coast_deg = self.coast_after_hold(hold)
        return RotationPlan(
            target_deg=sign * target,
            command=command,
            sign=sign,
            hold_sec=hold,
            hard_timeout_sec=hold + float(timeout_margin_sec),
            predicted_stop_angle_deg=stop_angle,
            predicted_stop_rate_deg_s=stop_rate,
            predicted_coast_deg=coast_deg,
            predicted_total_deg=stop_angle + coast_deg,
            predicted_settle_sec=self.inertia_horizon_sec,
            angle_sd_deg=self.residual_sd_deg,
            feasible=feasible,
            note="; ".join(notes),
        )


@dataclass(frozen=True)
class RotationResponseSelection:
    """Result of the one-shot startup selection."""

    response: Union[RotationResponse, PiecewiseRotationResponse]
    artifact_active: bool
    artifact_path: str
    artifact_sha256_id: str
    inference_model_sha256_id: str
    fallback_reason: str


def _validation_rmse(report: Mapping[str, object]) -> tuple[float, int, int, float]:
    settings = _mapping(report.get("settings"), "fit_report.settings")
    validation = _mapping(report.get("validation"), "fit_report.validation")
    folds = _mapping(
        validation.get("leave_one_recording_out_folds"),
        "fit_report.validation.leave_one_recording_out_folds",
    )
    threshold = _finite_number(
        settings.get("max_loo_rmse_deg"),
        "fit_report.settings.max_loo_rmse_deg",
    )
    minimum = _integer(
        settings.get("min_loo_predictions"),
        "fit_report.settings.min_loo_predictions",
    )
    attempted = _integer(
        folds.get("attempted"),
        "fit_report.validation.leave_one_recording_out_folds.attempted",
    )
    successful = _integer(
        folds.get("successful"),
        "fit_report.validation.leave_one_recording_out_folds.successful",
    )
    rmse = _finite_number(
        folds.get("equal_recording_weight_rmse_deg"),
        "fit_report.validation.leave_one_recording_out_folds."
        "equal_recording_weight_rmse_deg",
    )
    if threshold <= 0.0 or minimum < 1:
        raise ArtifactValidationError("invalid LOO validation thresholds")
    if attempted != successful or successful < minimum or rmse > threshold:
        raise ArtifactValidationError(
            "leave-one-recording-out validation does not pass its recorded gate"
        )
    return rmse, attempted, successful, threshold


def validate_rotation_artifact_bytes(
    payload: bytes,
    *,
    expected_model_sha256_id: str,
    expected_command_strength: int = 30,
    max_rate_deg_s: float = 60.0,
    startup_timeout_sec: float = 3.0,
    max_command_hold_sec: float = 8.0,
    source: str = "generated artifact",
) -> PiecewiseRotationResponse:
    """Strictly validate bytes and return the FSM response adapter."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not _SHA256_ID_RE.fullmatch(expected_model_sha256_id):
        raise ArtifactValidationError(
            "expected inference model id must be sha256:<64 lowercase hex>"
        )
    try:
        artifact = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid rotation artifact JSON: {exc}") from exc
    artifact = _mapping(artifact, "artifact")
    _exact_keys(artifact, _TOP_LEVEL_KEYS, "artifact")
    if _integer(artifact["schema_version"], "schema_version") != _SCHEMA_VERSION:
        raise ArtifactValidationError(f"schema_version must equal {_SCHEMA_VERSION}")
    if artifact["model_type"] != _MODEL_TYPE:
        raise ArtifactValidationError(f"model_type must equal {_MODEL_TYPE!r}")

    parameters = _mapping(artifact["parameters"], "parameters")
    metadata = _mapping(artifact["fit_metadata"], "fit_metadata")
    report = _mapping(artifact["fit_report"], "fit_report")
    _exact_keys(parameters, _PARAMETER_KEYS, "parameters")
    _exact_keys(metadata, _FIT_METADATA_KEYS, "fit_metadata")

    strength = _integer(parameters["command_strength"], "parameters.command_strength")
    if strength != expected_command_strength:
        raise ArtifactValidationError(
            "artifact command strength does not match runtime deflection"
        )
    if report.get("status") != "READY":
        raise ArtifactValidationError("fit_report.status must equal 'READY'")
    if report.get("safe_for_control") is not True:
        raise ArtifactValidationError("fit_report.safe_for_control must be true")
    if _list(
        report.get("deployment_blockers"), "fit_report.deployment_blockers",
    ) != []:
        raise ArtifactValidationError("fit_report.deployment_blockers must be empty")
    if report.get("model_type") != _FIT_MODEL_TYPE:
        raise ArtifactValidationError("fit_report.model_type is not supported")
    if _integer(
        report.get("command_strength"), "fit_report.command_strength",
    ) != strength:
        raise ArtifactValidationError("fit report and parameters disagree on strength")
    if report.get("angle_column") != "yaw_deg":
        raise ArtifactValidationError("fit_report.angle_column must equal 'yaw_deg'")
    if report.get("angle_domain") != "heading":
        raise ArtifactValidationError("fit_report.angle_domain must equal 'heading'")

    settings = _mapping(report.get("settings"), "fit_report.settings")
    if _integer(
        settings.get("command_strength"),
        "fit_report.settings.command_strength",
    ) != strength:
        raise ArtifactValidationError("fit settings use a different command strength")
    period = _finite_number(
        report.get("angle_period_deg"), "fit_report.angle_period_deg",
    )
    settings_period = _finite_number(
        settings.get("angle_period_deg"),
        "fit_report.settings.angle_period_deg",
    )
    if period != 360.0 or settings_period != 360.0:
        raise ArtifactValidationError("yaw heading fit must use a 360 degree period")
    if settings.get("require_actual_can_timing") is not True:
        raise ArtifactValidationError("fit must require actual CAN write timing")
    if settings.get("can_time_point") != "return":
        raise ArtifactValidationError("fit CAN timestamp must be the write return")
    horizon = _finite_number(
        settings.get("inertia_horizon_sec"),
        "fit_report.settings.inertia_horizon_sec",
    )
    if horizon <= 0.0:
        raise ArtifactValidationError("inertia horizon must be positive")

    source_report = _mapping(report.get("source"), "fit_report.source")
    if source_report.get("angle_column") != "yaw_deg" or source_report.get(
        "angle_domain"
    ) != "heading":
        raise ArtifactValidationError("source angle contract must be yaw_deg/heading")
    model_ids = _list(source_report.get("model_ids"), "fit_report.source.model_ids")
    if model_ids != [expected_model_sha256_id]:
        raise ArtifactValidationError(
            "artifact inference model SHA-256 does not match the loaded .pt model"
        )
    if source_report.get("mixed_model_ids_detected") is not False:
        raise ArtifactValidationError("mixed inference model ids are not allowed")
    if _mapping(
        source_report.get("recording_errors"), "fit_report.source.recording_errors",
    ):
        raise ArtifactValidationError("recording input errors are not allowed")
    if _list(
        source_report.get("unidentified_model_recordings"),
        "fit_report.source.unidentified_model_recordings",
    ):
        raise ArtifactValidationError("recordings without model identity are not allowed")
    sign_audit = _mapping(
        source_report.get("direction_sign_audit"),
        "fit_report.source.direction_sign_audit",
    )
    if _list(
        sign_audit.get("ambiguous_directions"),
        "fit_report.source.direction_sign_audit.ambiguous_directions",
    ):
        raise ArtifactValidationError("rotation direction signs are ambiguous")
    if sign_audit.get("left_right_modal_signs_are_opposite") is not True:
        raise ArtifactValidationError("LEFT and RIGHT signs must be opposite")

    identifiability = _mapping(
        report.get("identifiability"), "fit_report.identifiability",
    )
    for key in ("dead_time", "acceleration_and_cruise", "linear_inertia"):
        if identifiability.get(key) is not True:
            raise ArtifactValidationError(f"fit component {key!r} is not identifiable")
    loo_rmse, _attempted, _successful, _threshold = _validation_rmse(report)

    values = {
        key: _finite_number(parameters[key], f"parameters.{key}")
        for key in _PARAMETER_KEYS
        if key != "command_strength"
    }
    values.update({
        key: _finite_number(metadata[key], f"fit_metadata.{key}")
        for key in _FIT_METADATA_KEYS
    })
    response = PiecewiseRotationResponse(
        command_strength=strength,
        inertia_horizon_sec=horizon,
        residual_sd_deg=loo_rmse,
        domain="heading",
        source=source,
        **values,
    )

    if response.dead_time_sec < 0.0:
        raise ArtifactValidationError("dead_time_sec must be non-negative")
    if response.accel_duration_sec <= 0.0 or response.accel_deg_s2 <= 0.0:
        raise ArtifactValidationError("acceleration parameters must be positive")
    if response.max_rate_deg_s <= 0.0:
        raise ArtifactValidationError("max_rate_deg_s must be positive")
    if response.inertia_slope_sec < 0.0:
        raise ArtifactValidationError("inertia_slope_sec must be non-negative")
    if not math.isclose(
        response.accel_deg_s2 * response.accel_duration_sec,
        response.max_rate_deg_s,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ArtifactValidationError(
            "acceleration-to-cruise boundary is discontinuous"
        )
    if response.fitted_min_hold_sec < 0.0 or (
        response.fitted_max_hold_sec < response.fitted_min_hold_sec
    ):
        raise ArtifactValidationError("invalid fitted hold range")
    if response.fitted_min_angle_deg < 0.0 or (
        response.fitted_max_angle_deg < response.fitted_min_angle_deg
    ):
        raise ArtifactValidationError("invalid fitted angle range")
    if response.dead_time_sec >= float(startup_timeout_sec):
        raise ArtifactValidationError("fitted dead time reaches startup timeout")
    if response.max_rate_deg_s > float(max_rate_deg_s):
        raise ArtifactValidationError("fitted maximum rate exceeds runtime safety limit")
    if response.fitted_min_hold_sec > min(
        response.fitted_max_hold_sec, float(max_command_hold_sec),
    ):
        raise ArtifactValidationError("no fitted command hold is inside runtime limit")
    return response


def load_validated_rotation_artifact(
    path: Union[str, Path],
    *,
    expected_model_sha256_id: str,
    expected_command_strength: int = 30,
    max_rate_deg_s: float = 60.0,
    startup_timeout_sec: float = 3.0,
    max_command_hold_sec: float = 8.0,
    max_artifact_bytes: int = 16 * 1024 * 1024,
) -> tuple[PiecewiseRotationResponse, str]:
    """Read one immutable snapshot of ``path`` and validate it.

    Returns ``(response, artifact_sha256_id)``.  The bounded read also avoids
    accepting an accidentally published report of unbounded size.
    """

    artifact_path = Path(path)
    with artifact_path.open("rb") as handle:
        payload = handle.read(max_artifact_bytes + 1)
    if len(payload) > max_artifact_bytes:
        raise ArtifactValidationError(
            f"rotation artifact exceeds {max_artifact_bytes} bytes"
        )
    artifact_id = "sha256:" + hashlib.sha256(payload).hexdigest()
    response = validate_rotation_artifact_bytes(
        payload,
        expected_model_sha256_id=expected_model_sha256_id,
        expected_command_strength=expected_command_strength,
        max_rate_deg_s=max_rate_deg_s,
        startup_timeout_sec=startup_timeout_sec,
        max_command_hold_sec=max_command_hold_sec,
        source=f"{artifact_path.name} ({artifact_id})",
    )
    return response, artifact_id


def select_rotation_response(
    artifact_path: Union[str, Path],
    fallback: RotationResponse,
    *,
    inference_model_path: Union[str, Path],
    expected_command_strength: int = 30,
    max_rate_deg_s: float = 60.0,
    startup_timeout_sec: float = 3.0,
    max_command_hold_sec: float = 8.0,
    enabled: bool = True,
) -> RotationResponseSelection:
    """Load once at startup, falling back to the built-in response on error."""

    path = Path(artifact_path)
    if not enabled:
        return RotationResponseSelection(
            fallback, False, str(path), "", "", "artifact loading disabled",
        )
    model_id = ""
    try:
        model_id = canonical_model_sha256(inference_model_path)
        response, artifact_id = load_validated_rotation_artifact(
            path,
            expected_model_sha256_id=model_id,
            expected_command_strength=expected_command_strength,
            max_rate_deg_s=max_rate_deg_s,
            startup_timeout_sec=startup_timeout_sec,
            max_command_hold_sec=max_command_hold_sec,
        )
    except Exception as exc:
        return RotationResponseSelection(
            fallback,
            False,
            str(path),
            "",
            model_id,
            f"{type(exc).__name__}: {exc}",
        )
    return RotationResponseSelection(
        response, True, str(path), artifact_id, model_id, "",
    )
