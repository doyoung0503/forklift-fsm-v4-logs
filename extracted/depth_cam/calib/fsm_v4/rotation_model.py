"""Log-fitted rotation response model for the fixed rotate deflection.

The model is a *bang-bang drive with transport delays* fitted to the recorded
rotation segments (see ``rotation_fit/`` for the fitting pipeline)::

    u(t) = 1  for  T_on <= t < hold + T_off,  else 0
    du/dt: +alpha while u = 1 (saturating at w_max), -beta while u = 0

``t`` is measured from the moment the ROT command frame is written to CAN and
``hold`` is how long the command is held before STOP is written.  Both edges
have their own transport delay: ``T_on`` before motion starts and ``T_off``
before the STOP takes effect.  ``T_off`` is why the post-STOP rotation does not
vanish as the STOP speed goes to zero -- the truck keeps accelerating for
``T_off`` seconds after STOP is commanded.

Left and right rotation are assumed to share one response; direction enters
only as a sign.  All angles are in degrees, all times in seconds, and the angle
is the *magnitude* of the rotation in the commanded direction.

This module imports nothing from the package so it can be used directly by the
offline fitting/validation scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "RotationResponse",
    "RotationPlan",
    "bearing_gain",
    "FITTED_2026_09_03",
]


def bearing_gain(range_m: float, rot_centre_offset_m: float) -> float:
    """How much larger a bearing change is than the heading change that caused it.

    Rotating the truck by ``theta`` about a centre ``d`` metres behind the
    camera swings the camera sideways as well as turning it, so a target at
    range ``r`` moves through ``theta * (r + d) / r`` of bearing.  FACE and
    RECENTER servo on bearing, WAYPOINT servos on pallet yaw (pure heading),
    so the two need different angular scales from the same response fit.
    """
    r = max(0.3, float(range_m))
    return (r + max(0.0, float(rot_centre_offset_m))) / r


@dataclass(frozen=True)
class RotationPlan:
    """Everything the controller needs to run one bounded rotation."""

    target_deg: float               # requested rotation magnitude
    command: str                    # "ROT_LEFT" / "ROT_RIGHT"
    sign: float                     # +1 right, -1 left (error-reduction sense)
    hold_sec: float                 # how long to hold the ROT command
    hard_timeout_sec: float         # command abort limit
    predicted_stop_angle_deg: float # rotation completed when STOP is written
    predicted_stop_rate_deg_s: float
    predicted_coast_deg: float      # rotation added after STOP
    predicted_total_deg: float      # stop angle + coast
    predicted_settle_sec: float     # STOP -> fully stopped
    angle_sd_deg: float             # 1-sigma spread from startup-delay jitter
    feasible: bool                  # target inside [min, max] single-shot range
    note: str = ""


@dataclass(frozen=True)
class RotationResponse:
    """Fitted rotation response at one fixed joystick deflection."""

    startup_delay_sec: float          # T_on
    stop_delay_sec: float             # T_off
    accel_deg_s2: float               # alpha
    decel_deg_s2: float               # beta
    max_rate_deg_s: float             # w_max
    startup_delay_sd_sec: float = 0.0 # 1-sigma spread of T_on across logs
    residual_sd_deg: float = 0.0      # 1-sigma total-angle prediction error
    # Measured STOP -> pose-stable time.  The fitted decel alone under-predicts
    # this: the truck rocks back a fraction of a degree after it stops turning,
    # and that rebound is not part of the drive model.  Use this, not
    # ``settle_sec``, to size a settle window.
    measured_settle_p90_sec: float = 0.0
    # Which angle the fitted rates refer to: "heading" (pallet yaw, pure vehicle
    # rotation) or "bearing" (front-face bearing, which also picks up the
    # sideways camera swing).  Use ``in_bearing_domain`` to convert.
    domain: str = "heading"
    source: str = ""

    def in_bearing_domain(self, range_m: float,
                          rot_centre_offset_m: float) -> "RotationResponse":
        """Same response expressed in front-face bearing degrees.

        Only the angular quantities scale; the two transport delays are
        properties of the drivetrain and do not.
        """
        if self.domain == "bearing":
            return self
        g = bearing_gain(range_m, rot_centre_offset_m)
        return RotationResponse(
            startup_delay_sec=self.startup_delay_sec,
            stop_delay_sec=self.stop_delay_sec,
            accel_deg_s2=self.accel_deg_s2 * g,
            decel_deg_s2=self.decel_deg_s2 * g,
            max_rate_deg_s=self.max_rate_deg_s * g,
            startup_delay_sd_sec=self.startup_delay_sd_sec,
            residual_sd_deg=self.residual_sd_deg * g,
            measured_settle_p90_sec=self.measured_settle_p90_sec,
            domain="bearing",
            source=self.source + " [bearing x%.3f]" % g,
        )

    # ------------------------------------------------------------ internals
    def _drive_seconds(self, hold_sec: float) -> float:
        """Seconds the drive is actually energised for this command hold."""
        return max(0.0, float(hold_sec) + self.stop_delay_sec
                   - self.startup_delay_sec)

    def _rate_after(self, drive_sec: float) -> float:
        return min(self.max_rate_deg_s, self.accel_deg_s2 * max(0.0, drive_sec))

    def _angle_over(self, drive_sec: float) -> float:
        """Angle swept while the drive is energised for ``drive_sec``."""
        d = max(0.0, float(drive_sec))
        t_sat = self.max_rate_deg_s / self.accel_deg_s2
        if d <= t_sat:
            return 0.5 * self.accel_deg_s2 * d * d
        return (0.5 * self.max_rate_deg_s * t_sat
                + self.max_rate_deg_s * (d - t_sat))

    # -------------------------------------------------------------- forward
    def rate_at(self, elapsed_sec: float) -> float:
        """Expected rotation rate at ``elapsed_sec`` after the ROT command,
        assuming the command is still held."""
        return self._rate_after(max(0.0, float(elapsed_sec)
                                    - self.startup_delay_sec))

    def angle_at(self, elapsed_sec: float) -> float:
        """Expected rotation completed ``elapsed_sec`` after the ROT command,
        assuming the command is still held."""
        return self._angle_over(max(0.0, float(elapsed_sec)
                                    - self.startup_delay_sec))

    def coast(self, rate_deg_s: float) -> Tuple[float, float]:
        """Rotation added, and time taken, if STOP is written while turning at
        ``rate_deg_s``.  Returns ``(coast_deg, coast_sec)``."""
        w0 = max(0.0, float(rate_deg_s))
        head = self.stop_delay_sec
        to_sat = max(0.0, (self.max_rate_deg_s - w0) / self.accel_deg_s2)
        if to_sat >= head:
            w1 = w0 + self.accel_deg_s2 * head
            angle = w0 * head + 0.5 * self.accel_deg_s2 * head * head
        else:
            w1 = self.max_rate_deg_s
            angle = (w0 * to_sat + 0.5 * self.accel_deg_s2 * to_sat * to_sat
                     + self.max_rate_deg_s * (head - to_sat))
        angle += w1 * w1 / (2.0 * self.decel_deg_s2)
        return angle, head + w1 / self.decel_deg_s2

    def coast_after_hold(self, hold_sec: float) -> float:
        """Rotation added after STOP for a command held ``hold_sec``."""
        return self.total_angle(hold_sec) - self.angle_at(hold_sec)

    def adjusted_coast_after_hold(
        self, hold_sec: float, coast_reduction_deg: float = 0.0,
    ) -> float:
        """Post-STOP rotation after applying a fixed field correction."""
        return max(
            0.0,
            self.coast_after_hold(hold_sec)
            - max(0.0, float(coast_reduction_deg)),
        )

    def adjusted_total_angle(
        self, hold_sec: float, coast_reduction_deg: float = 0.0,
    ) -> float:
        """Total rotation with the corrected post-STOP contribution."""
        return (
            self.angle_at(hold_sec)
            + self.adjusted_coast_after_hold(hold_sec, coast_reduction_deg)
        )

    def total_angle(self, hold_sec: float) -> float:
        """Total rotation from ROT command to full stop, for a given hold."""
        drive = self._drive_seconds(hold_sec)
        w_end = self._rate_after(drive)
        return self._angle_over(drive) + w_end * w_end / (2.0 * self.decel_deg_s2)

    def settle_sec(self, hold_sec: float) -> float:
        """STOP command -> rotation rate zero, from the drive model alone.

        This is the deceleration end, not the moment the pose reads stable;
        see ``settle_wait_sec``.
        """
        w_end = self._rate_after(self._drive_seconds(hold_sec))
        return self.stop_delay_sec + w_end / self.decel_deg_s2

    def settle_wait_sec(self, hold_sec: float) -> float:
        """How long to wait after STOP before trusting a settled pose."""
        return max(self.settle_sec(hold_sec), self.measured_settle_p90_sec)

    # -------------------------------------------------------------- inverse
    @property
    def min_hold_sec(self) -> float:
        """Shortest hold that produces any motion at all."""
        return max(0.0, self.startup_delay_sec - self.stop_delay_sec)

    @property
    def min_total_deg(self) -> float:
        """Smallest rotation from a sensible command.

        A command must be held at least until motion starts, otherwise the
        result is decided entirely by where the two transport delays happen to
        land.  Holding exactly to the onset is that shortest sensible command.
        """
        return self.total_angle(self.startup_delay_sec)

    def angle_uncertainty_deg(self, hold_sec: float) -> float:
        """1-sigma spread of the achieved angle caused by startup-delay jitter.

        The startup delay is the dominant error source: every second of delay
        error is a second of drive gained or lost at the current rate.
        """
        if self.startup_delay_sd_sec <= 0.0:
            return 0.0
        late = RotationResponse(
            self.startup_delay_sec + self.startup_delay_sd_sec,
            self.stop_delay_sec, self.accel_deg_s2, self.decel_deg_s2,
            self.max_rate_deg_s)
        early = RotationResponse(
            max(0.0, self.startup_delay_sec - self.startup_delay_sd_sec),
            self.stop_delay_sec, self.accel_deg_s2, self.decel_deg_s2,
            self.max_rate_deg_s)
        return 0.5 * abs(early.total_angle(hold_sec) - late.total_angle(hold_sec))

    def command_seconds(
        self, target_deg: float, max_hold_sec: float = 8.0,
        coast_reduction_deg: float = 0.0,
    ) -> float:
        """Hold time whose *total* rotation (drive + coast) equals ``target_deg``.

        ``total_angle`` is monotonically increasing in ``hold_sec`` above
        ``min_hold_sec``, so a bisection is exact to numerical precision.
        """
        target = abs(float(target_deg))
        lo = self.min_hold_sec
        hi = float(max_hold_sec)
        if self.adjusted_total_angle(hi, coast_reduction_deg) <= target:
            return hi
        if self.adjusted_total_angle(lo, coast_reduction_deg) >= target:
            return lo
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.adjusted_total_angle(mid, coast_reduction_deg) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ----------------------------------------------------------- live logic
    def stop_now_margin_deg(self, elapsed_sec: float,
                            measured_rate_deg_s: Optional[float] = None,
                            measurement_age_sec: float = 0.0,
                            coast_reduction_deg: float = 0.0) -> float:
        """Rotation still to come if STOP is written now.

        ``measured_rate_deg_s`` (from live pose) wins over the model rate when
        it is larger, so a run that is faster than the fitted median stops
        early rather than late.  ``measurement_age_sec`` covers the perception
        latency: the pose we are reacting to is already that old.
        """
        model_rate = self.rate_at(elapsed_sec)
        rate = model_rate
        if measured_rate_deg_s is not None:
            rate = max(model_rate, abs(float(measured_rate_deg_s)))
        coast_deg, _ = self.coast(rate)
        adjusted_coast = max(
            0.0,
            coast_deg - max(0.0, float(coast_reduction_deg)),
        )
        return adjusted_coast + rate * max(0.0, float(measurement_age_sec))

    def should_stop_now(self, remaining_deg: float, elapsed_sec: float,
                        measured_rate_deg_s: Optional[float] = None,
                        measurement_age_sec: float = 0.0,
                        coast_reduction_deg: float = 0.0) -> bool:
        """True when the predicted coast already covers the remaining error."""
        if elapsed_sec < self.startup_delay_sec - self.stop_delay_sec:
            return False
        return abs(float(remaining_deg)) <= self.stop_now_margin_deg(
            elapsed_sec, measured_rate_deg_s, measurement_age_sec,
            coast_reduction_deg)

    # --------------------------------------------------------------- planner
    def plan(self, error_deg: float, min_angle_deg: float = 2.5,
             max_angle_deg: float = 20.0, timeout_margin_sec: float = 1.0,
             max_hold_sec: float = 8.0,
             coast_reduction_deg: float = 0.0) -> RotationPlan:
        """Plan one bounded rotation that removes ``error_deg``.

        ``error_deg`` is the signed error to remove; a positive error means the
        target is to the right, which is corrected by ROT_RIGHT.
        """
        signed = float(error_deg)
        sign = 1.0 if signed > 0.0 else -1.0
        command = "ROT_RIGHT" if signed > 0.0 else "ROT_LEFT"
        want = abs(signed)
        note = ""
        feasible = True
        target = want
        if want > max_angle_deg:
            target = max_angle_deg
            note = "target clipped to single-command limit"
        floor = max(float(min_angle_deg), self.min_total_deg)
        if want < floor:
            feasible = False
            note = ("below the smallest reliable single command "
                    f"({floor:.2f} deg)")
        hold = self.command_seconds(
            target,
            max_hold_sec=max_hold_sec,
            coast_reduction_deg=coast_reduction_deg,
        )
        adjusted_total = self.adjusted_total_angle(hold, coast_reduction_deg)
        if adjusted_total < target - 1e-6:
            # The hold cap, not the angle cap, is what binds here.
            feasible = False
            note = ("hold capped at %.2f s -> only %.1f deg this command"
                    % (hold, adjusted_total))
        stop_angle = self.angle_at(hold)
        stop_rate = self.rate_at(hold)
        coast_deg = self.adjusted_coast_after_hold(
            hold, coast_reduction_deg,
        )
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
            predicted_settle_sec=self.settle_sec(hold),
            angle_sd_deg=self.angle_uncertainty_deg(hold),
            feasible=feasible,
            note=note,
        )


# ---------------------------------------------------------------------------
# Fitted parameters.  Regenerate with rotation_fit/fit_rotation_model.py and
# paste the numbers from out/rotation_model_fit.json.
# ---------------------------------------------------------------------------
FITTED_2026_09_03 = RotationResponse(
    startup_delay_sec=1.082,
    stop_delay_sec=0.352,
    accel_deg_s2=13.06,
    decel_deg_s2=300.0,
    max_rate_deg_s=12.01,
    startup_delay_sd_sec=0.105,
    residual_sd_deg=1.44,
    measured_settle_p90_sec=0.88,
    domain="heading",
    source=("2026-09-03 rotation_fit: 9-keypoint PnP pallet yaw, 9 segments, "
            "deflection 30; delays pinned from the lower-noise bearing fit"),
)
