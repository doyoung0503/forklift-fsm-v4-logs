"""Pure-Python runtime model for a log-fitted, fixed-strength rotation.

The model pools left and right observations after their angles have been
normalised into the commanded direction.  It therefore predicts magnitudes;
the sign of a requested angle is used only to select LEFT or RIGHT.

For a command held for ``h`` seconds, motion is modelled as::

    driven = max(0, h - dead_time)
    rate   = acceleration * driven              (acceleration phase)
             max_rate                           (cruise phase)
    angle  = integral(rate)
    coast  = max(0, intercept + slope * rate)   (when rate > 0)

``total_angle`` is the angle accumulated while the command is active plus the
post-command inertia angle.  All angles are degrees and all times are seconds.
The module intentionally uses only the Python standard library so the fitted
artifact can be loaded by the vehicle runtime without the fitting stack.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Optional


__all__ = ["LogRotationModel", "RotationCommandPlan"]


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


def _finite_float(value: object, name: str) -> float:
    """Return a finite float while rejecting booleans and non-numbers."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_number(value: object, name: str) -> float:
    """Validate a number originating in JSON (whose numeric types are fixed)."""

    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _check_keys(
    value: Mapping[str, object], expected: frozenset[str], location: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if extra:
        problems.append("unexpected " + ", ".join(extra))
    if problems:
        raise ValueError(f"invalid {location}: {'; '.join(problems)}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class RotationCommandPlan:
    """Result of inverting a :class:`LogRotationModel` for one target angle.

    Predicted angles and rates are magnitudes in the pooled response domain.
    ``sign`` and ``direction`` carry the requested physical direction.
    """

    target_deg: float
    sign: int
    direction: str
    command_strength: int
    hold_sec: float
    driven_sec: float
    command_end_rate_deg_s: float
    predicted_command_angle_deg: float
    predicted_inertia_angle_deg: float
    predicted_total_angle_deg: float
    feasible: bool
    extrapolated: bool
    note: str

    @property
    def command_duration_sec(self) -> float:
        """Alias that makes the meaning of ``hold_sec`` explicit."""

        return self.hold_sec

    @property
    def signed_predicted_total_deg(self) -> float:
        return self.sign * self.predicted_total_angle_deg


@dataclass(frozen=True, slots=True)
class LogRotationModel:
    """Piecewise rotation response fitted at joystick strength 30.

    The four ``fitted_*`` values describe the supported data range.  They are
    deliberately kept separate from the physical parameters: they gate safe
    inversion by default but do not change the response curve.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    MODEL_TYPE: ClassVar[str] = "log_rotation_piecewise"
    REQUIRED_COMMAND_STRENGTH: ClassVar[int] = 30

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
    command_strength: int = REQUIRED_COMMAND_STRENGTH

    def __post_init__(self) -> None:
        if type(self.command_strength) is not int:
            raise TypeError("command_strength must be an integer")
        if self.command_strength != self.REQUIRED_COMMAND_STRENGTH:
            raise ValueError(
                "this model is valid only for command_strength="
                f"{self.REQUIRED_COMMAND_STRENGTH}"
            )

        numeric_fields = (
            "dead_time_sec",
            "accel_duration_sec",
            "accel_deg_s2",
            "max_rate_deg_s",
            "inertia_intercept_deg",
            "inertia_slope_sec",
            "fitted_min_hold_sec",
            "fitted_max_hold_sec",
            "fitted_min_angle_deg",
            "fitted_max_angle_deg",
        )
        for name in numeric_fields:
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

        if self.dead_time_sec < 0.0:
            raise ValueError("dead_time_sec must be non-negative")
        if self.accel_duration_sec <= 0.0:
            raise ValueError("accel_duration_sec must be positive")
        if self.accel_deg_s2 <= 0.0:
            raise ValueError("accel_deg_s2 must be positive")
        if self.max_rate_deg_s <= 0.0:
            raise ValueError("max_rate_deg_s must be positive")
        if self.inertia_slope_sec < 0.0:
            raise ValueError("inertia_slope_sec must be non-negative")

        # These three quantities describe the same acceleration-to-cruise
        # boundary.  Rejecting an inconsistent artifact prevents a hidden rate
        # jump and keeps angle/rate derivatives continuous at that boundary.
        boundary_rate = self.accel_deg_s2 * self.accel_duration_sec
        if not math.isclose(
            boundary_rate, self.max_rate_deg_s, rel_tol=1e-9, abs_tol=1e-9,
        ):
            raise ValueError(
                "max_rate_deg_s must equal accel_deg_s2 * accel_duration_sec "
                "for a continuous piecewise response"
            )

        if self.fitted_min_hold_sec < 0.0:
            raise ValueError("fitted_min_hold_sec must be non-negative")
        if self.fitted_max_hold_sec < self.fitted_min_hold_sec:
            raise ValueError(
                "fitted_max_hold_sec must be >= fitted_min_hold_sec"
            )
        if self.fitted_min_angle_deg < 0.0:
            raise ValueError("fitted_min_angle_deg must be non-negative")
        if self.fitted_max_angle_deg < self.fitted_min_angle_deg:
            raise ValueError(
                "fitted_max_angle_deg must be >= fitted_min_angle_deg"
            )

    # ------------------------------------------------------------ forward model
    def driven_duration(self, hold_sec: float) -> float:
        """Time for which visible rotation has developed during a command."""

        hold = _finite_float(hold_sec, "hold_sec")
        if hold < 0.0:
            raise ValueError("hold_sec must be non-negative")
        return max(0.0, hold - self.dead_time_sec)

    def rate_at_command_end(self, hold_sec: float) -> float:
        """Expected angular-speed magnitude when the command is released."""

        driven = self.driven_duration(hold_sec)
        return min(self.max_rate_deg_s, self.accel_deg_s2 * driven)

    def angle_at_command_end(self, hold_sec: float) -> float:
        """Angle accumulated before command release, excluding inertia."""

        driven = self.driven_duration(hold_sec)
        ramp = min(driven, self.accel_duration_sec)
        ramp_angle = 0.5 * self.accel_deg_s2 * ramp * ramp
        cruise = max(0.0, driven - self.accel_duration_sec)
        return ramp_angle + self.max_rate_deg_s * cruise

    def inertia_angle(self, command_end_rate_deg_s: float) -> float:
        """Non-negative post-command rotation predicted from end speed.

        A signed live rate is accepted because left/right logs are pooled.  A
        zero rate always gives zero coast even when an unconstrained regression
        has a positive intercept; a command that never moved cannot coast.
        """

        rate = abs(_finite_float(command_end_rate_deg_s, "command_end_rate_deg_s"))
        if rate == 0.0:
            return 0.0
        return max(
            0.0,
            self.inertia_intercept_deg + self.inertia_slope_sec * rate,
        )

    def total_angle(self, hold_sec: float) -> float:
        """Total rotation magnitude after the vehicle has coasted to rest."""

        rate = self.rate_at_command_end(hold_sec)
        return self.angle_at_command_end(hold_sec) + self.inertia_angle(rate)

    # ------------------------------------------------------------- inverse model
    def _unbounded_duration_for(self, target_deg: float) -> Optional[float]:
        """Return the exact non-negative hold, or None for an angle in a gap."""

        if target_deg <= 0.0:
            return 0.0

        # A positive regression intercept creates a real minimum command: the
        # curve is zero through dead time and approaches the intercept as soon
        # as the end rate becomes positive.  Angles inside that jump cannot be
        # produced by this model.
        if self.inertia_intercept_deg > 0.0 and target_deg <= self.inertia_intercept_deg:
            return None

        accel_end_hold = self.dead_time_sec + self.accel_duration_sec
        accel_end_angle = self.total_angle(accel_end_hold)
        tolerance = max(1e-12, target_deg * 1e-12)
        if target_deg <= accel_end_angle + tolerance:
            lo = 0.0
            hi = self.accel_duration_sec
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                angle = self.total_angle(self.dead_time_sec + mid)
                if angle < target_deg:
                    lo = mid
                else:
                    hi = mid
            hold = self.dead_time_sec + 0.5 * (lo + hi)
            if not math.isclose(
                self.total_angle(hold), target_deg,
                rel_tol=1e-10, abs_tol=1e-10,
            ):
                return None
            return hold

        # Once the maximum rate is reached, inertia is constant and command
        # angle grows linearly at max_rate.
        return accel_end_hold + (
            target_deg - accel_end_angle
        ) / self.max_rate_deg_s

    def _plan(
        self,
        target_deg: float,
        sign: int,
        direction: str,
        hold_sec: float,
        *,
        feasible: bool,
        extrapolated: bool,
        note: str,
    ) -> RotationCommandPlan:
        hold = max(0.0, float(hold_sec))
        rate = self.rate_at_command_end(hold)
        command_angle = self.angle_at_command_end(hold)
        inertia = self.inertia_angle(rate)
        return RotationCommandPlan(
            target_deg=target_deg,
            sign=sign,
            direction=direction,
            command_strength=self.command_strength,
            hold_sec=hold,
            driven_sec=self.driven_duration(hold),
            command_end_rate_deg_s=rate,
            predicted_command_angle_deg=command_angle,
            predicted_inertia_angle_deg=inertia,
            predicted_total_angle_deg=command_angle + inertia,
            feasible=bool(feasible),
            extrapolated=bool(extrapolated),
            note=str(note),
        )

    def command_duration(
        self,
        target_deg: float,
        allow_extrapolation: bool = False,
        max_hold_sec: Optional[float] = None,
    ) -> RotationCommandPlan:
        """Plan a fixed-strength command for a signed target angle.

        By default, both the target angle and solved hold must lie inside the
        ranges represented by the fitting data.  Outside that range the plan is
        marked infeasible and either emits no command (below the fitted range)
        or returns the largest supported/capped hold (above it).

        Set ``allow_extrapolation`` to solve the same response curve outside the
        fitted range.  ``max_hold_sec`` remains a hard cap in either mode.
        """

        target = _finite_float(target_deg, "target_deg")
        if type(allow_extrapolation) is not bool:
            raise TypeError("allow_extrapolation must be a bool")
        cap: Optional[float] = None
        if max_hold_sec is not None:
            cap = _finite_float(max_hold_sec, "max_hold_sec")
            if cap < 0.0:
                raise ValueError("max_hold_sec must be non-negative")

        if target == 0.0:
            return self._plan(
                target, 0, "NONE", 0.0,
                feasible=True, extrapolated=False,
                note="zero target; no command",
            )

        sign = 1 if target > 0.0 else -1
        direction = "RIGHT" if sign > 0 else "LEFT"
        wanted = abs(target)
        root = self._unbounded_duration_for(wanted)
        in_angle_range = (
            self.fitted_min_angle_deg <= wanted <= self.fitted_max_angle_deg
        )
        in_hold_range = bool(
            root is not None
            and self.fitted_min_hold_sec <= root <= self.fitted_max_hold_sec
        )

        if allow_extrapolation:
            extrapolated = not (in_angle_range and in_hold_range)
            if root is None:
                return self._plan(
                    target, sign, direction, 0.0,
                    feasible=False, extrapolated=True,
                    note=(
                        "target lies below the model's minimum positive angle "
                        "created by the inertia intercept"
                    ),
                )
            if cap is not None and root > cap:
                return self._plan(
                    target, sign, direction, cap,
                    feasible=False,
                    extrapolated=(
                        extrapolated
                        or not self.fitted_min_hold_sec <= cap <= self.fitted_max_hold_sec
                    ),
                    note=f"required hold exceeds max_hold_sec={cap:.6g}",
                )
            return self._plan(
                target, sign, direction, root,
                feasible=True, extrapolated=extrapolated,
                note=("extrapolated outside fitted range" if extrapolated else ""),
            )

        if wanted < self.fitted_min_angle_deg:
            return self._plan(
                target, sign, direction, 0.0,
                feasible=False, extrapolated=False,
                note=(
                    f"target below fitted minimum angle "
                    f"{self.fitted_min_angle_deg:.6g} deg; no command"
                ),
            )

        upper_hold = self.fitted_max_hold_sec
        if cap is not None:
            upper_hold = min(upper_hold, cap)

        if wanted > self.fitted_max_angle_deg:
            return self._plan(
                target, sign, direction, upper_hold,
                feasible=False, extrapolated=False,
                note=(
                    f"target above fitted maximum angle "
                    f"{self.fitted_max_angle_deg:.6g} deg; hold capped"
                ),
            )

        if root is None:
            return self._plan(
                target, sign, direction, 0.0,
                feasible=False, extrapolated=False,
                note="target is not attainable by the fitted response",
            )

        if upper_hold < self.fitted_min_hold_sec:
            return self._plan(
                target, sign, direction, upper_hold,
                feasible=False, extrapolated=False,
                note="max_hold_sec is below the fitted minimum hold",
            )
        if root < self.fitted_min_hold_sec:
            return self._plan(
                target, sign, direction, self.fitted_min_hold_sec,
                feasible=False, extrapolated=False,
                note="required hold is below the fitted hold range",
            )
        if root > upper_hold:
            note = "required hold exceeds the fitted maximum hold"
            if cap is not None and cap < self.fitted_max_hold_sec:
                note = f"required hold exceeds max_hold_sec={cap:.6g}"
            return self._plan(
                target, sign, direction, upper_hold,
                feasible=False, extrapolated=False, note=note,
            )
        return self._plan(
            target, sign, direction, root,
            feasible=True, extrapolated=False, note="",
        )

    # -------------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON-artifact representation."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "model_type": self.MODEL_TYPE,
            "parameters": {
                "command_strength": self.command_strength,
                "dead_time_sec": self.dead_time_sec,
                "accel_duration_sec": self.accel_duration_sec,
                "accel_deg_s2": self.accel_deg_s2,
                "max_rate_deg_s": self.max_rate_deg_s,
                "inertia_intercept_deg": self.inertia_intercept_deg,
                "inertia_slope_sec": self.inertia_slope_sec,
            },
            "fit_metadata": {
                "fitted_min_hold_sec": self.fitted_min_hold_sec,
                "fitted_max_hold_sec": self.fitted_max_hold_sec,
                "fitted_min_angle_deg": self.fitted_min_angle_deg,
                "fitted_max_angle_deg": self.fitted_max_angle_deg,
            },
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize to strict JSON text (NaN and infinity are forbidden)."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, artifact: Mapping[str, object]) -> "LogRotationModel":
        """Build a model from a strictly validated artifact mapping.

        A top-level ``fit_report`` object is accepted and ignored so the
        offline fitter can append diagnostics without affecting runtime load.
        No other unknown keys are accepted.
        """

        if not isinstance(artifact, Mapping):
            raise ValueError("rotation model artifact must be a JSON object")
        required_top = {
            "schema_version", "model_type", "parameters", "fit_metadata",
        }
        allowed_top = required_top | {"fit_report"}
        missing = sorted(required_top - set(artifact))
        extra = sorted(set(artifact) - allowed_top)
        if missing or extra:
            problems = []
            if missing:
                problems.append("missing " + ", ".join(missing))
            if extra:
                problems.append("unexpected " + ", ".join(extra))
            raise ValueError("invalid artifact: " + "; ".join(problems))

        schema = artifact["schema_version"]
        if type(schema) is not int or schema != cls.SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be integer {cls.SCHEMA_VERSION}"
            )
        model_type = artifact["model_type"]
        if type(model_type) is not str or model_type != cls.MODEL_TYPE:
            raise ValueError(f"model_type must be {cls.MODEL_TYPE!r}")

        parameters = artifact["parameters"]
        metadata = artifact["fit_metadata"]
        if not isinstance(parameters, Mapping):
            raise ValueError("parameters must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise ValueError("fit_metadata must be a JSON object")
        _check_keys(parameters, _PARAMETER_KEYS, "parameters")
        _check_keys(metadata, _FIT_METADATA_KEYS, "fit_metadata")

        if "fit_report" in artifact and not isinstance(
            artifact["fit_report"], Mapping,
        ):
            raise ValueError("fit_report must be a JSON object when present")

        strength = parameters["command_strength"]
        if type(strength) is not int:
            raise ValueError("parameters.command_strength must be a JSON integer")
        if strength != cls.REQUIRED_COMMAND_STRENGTH:
            raise ValueError(
                "parameters.command_strength must equal "
                f"{cls.REQUIRED_COMMAND_STRENGTH}"
            )

        values = {
            name: _json_number(parameters[name], f"parameters.{name}")
            for name in _PARAMETER_KEYS
            if name != "command_strength"
        }
        values.update({
            name: _json_number(metadata[name], f"fit_metadata.{name}")
            for name in _FIT_METADATA_KEYS
        })
        return cls(command_strength=strength, **values)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "LogRotationModel":
        """Parse strict JSON text and validate it as a model artifact."""

        if not isinstance(payload, (str, bytes, bytearray)):
            raise TypeError("payload must be JSON text or UTF-8 bytes")

        def reject_constant(value: str) -> None:
            raise ValueError(f"invalid JSON numeric constant: {value}")

        try:
            parsed = json.loads(
                payload,
                parse_constant=reject_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid rotation model JSON: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("rotation model JSON must contain one object")
        return cls.from_dict(parsed)
