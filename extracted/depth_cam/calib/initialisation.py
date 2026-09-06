"""Create a repeatable fork position before entering the driving FSM."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from . import config
from .control import (
    get_can_status,
    issue_command_fold,
    issue_command_folding_stop,
    issue_command_lift_down,
    issue_command_lift_stop,
    issue_command_lift_up,
    issue_command_reach_backward,
    issue_command_reach_forward,
    issue_command_reach_stop,
    issue_command_stop,
    issue_command_unfold,
)

__all__ = ["InitialisationError", "InitialisationSettings", "run_initialisation"]


class InitialisationError(RuntimeError):
    """Raised when fork initialisation cannot be completed safely."""


@dataclass(frozen=True)
class InitialisationSettings:
    enabled: bool
    upright_command: str
    upright_home_sec: float
    fold_sec: float
    lift_down_home_sec: float
    lift_up_sec: float
    reach_backward_home_sec: float
    reach_forward_sec: float
    settle_sec: float

    @classmethod
    def from_config(cls) -> "InitialisationSettings":
        return cls(
            enabled=config.INITIALISATION_ENABLED,
            upright_command=config.INITIALISATION_UPRIGHT_COMMAND,
            upright_home_sec=config.INITIALISATION_UPRIGHT_HOME_SEC,
            fold_sec=config.INITIALISATION_FOLD_SEC,
            lift_down_home_sec=config.INITIALISATION_LIFT_DOWN_HOME_SEC,
            lift_up_sec=config.INITIALISATION_LIFT_UP_SEC,
            reach_backward_home_sec=config.INITIALISATION_REACH_BACKWARD_HOME_SEC,
            reach_forward_sec=config.INITIALISATION_REACH_FORWARD_SEC,
            settle_sec=config.INITIALISATION_SETTLE_SEC,
        ).validated()

    def validated(self) -> "InitialisationSettings":
        if not isinstance(self.enabled, bool):
            raise InitialisationError("INITIALISATION_ENABLED must be bool")
        if self.upright_command not in {"fold", "unfold"}:
            raise InitialisationError(
                "INITIALISATION_UPRIGHT_COMMAND must be 'fold' or 'unfold'"
            )
        for name in (
            "upright_home_sec",
            "fold_sec",
            "lift_down_home_sec",
            "lift_up_sec",
            "reach_backward_home_sec",
            "reach_forward_sec",
            "settle_sec",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InitialisationError(f"{name} must be numeric: {value!r}")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise InitialisationError(
                    f"{name} must be finite and non-negative: {value!r}"
                )
        return self


@dataclass(frozen=True)
class _Step:
    name: str
    duration_sec: float
    start: Callable[[], None]
    stop: Callable[[], None]


def _assert_can_healthy(initial_error_count: int) -> None:
    status = get_can_status()
    # CAN OFF is an intentional dry-run mode. Commands still update the logical
    # state, while control._activate_command prevents every physical write.
    if not status.get("enabled", True):
        return
    if not status.get("ready") or not status.get("alive"):
        detail = status.get("last_error") or "CAN worker is not ready"
        raise InitialisationError(str(detail))
    current_errors = int(status.get("errors", 0))
    if current_errors > initial_error_count:
        detail = status.get("last_error") or f"CAN transmit errors: {current_errors}"
        raise InitialisationError(str(detail))


def _wait_while_healthy(duration_sec: float, initial_error_count: int) -> None:
    deadline = time.monotonic() + float(duration_sec)
    while True:
        _assert_can_healthy(initial_error_count)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.05))


def run_initialisation(
    settings: InitialisationSettings | None = None,
) -> None:
    """Run the configured homing/offset sequence and return to driving STOP."""
    settings = (settings or InitialisationSettings.from_config()).validated()
    if not settings.enabled:
        print("[INITIALISATION] disabled; skipping fork initialisation.")
        return

    status = get_can_status()
    can_enabled = bool(status.get("enabled", True))
    if can_enabled and (not status.get("ready") or not status.get("alive")):
        raise InitialisationError(
            status.get("last_error") or "CAN connection is not ready"
        )
    initial_error_count = int(status.get("errors", 0))

    upright_start = (
        issue_command_unfold
        if settings.upright_command == "unfold"
        else issue_command_fold
    )
    steps = (
        _Step(
            "upright home",
            settings.upright_home_sec,
            upright_start,
            issue_command_folding_stop,
        ),
        _Step(
            "target fold",
            settings.fold_sec,
            issue_command_fold,
            issue_command_folding_stop,
        ),
        _Step(
            "lift lower home",
            settings.lift_down_home_sec,
            issue_command_lift_down,
            issue_command_lift_stop,
        ),
        _Step(
            "lift target height",
            settings.lift_up_sec,
            issue_command_lift_up,
            issue_command_lift_stop,
        ),
        _Step(
            "reach rear home",
            settings.reach_backward_home_sec,
            issue_command_reach_backward,
            issue_command_reach_stop,
        ),
        _Step(
            "reach target forward",
            settings.reach_forward_sec,
            issue_command_reach_forward,
            issue_command_reach_stop,
        ),
    )

    mode = "CAN ON" if can_enabled else "CAN OFF dry-run"
    print(f"[INITIALISATION] starting fork initialisation ({mode}).")
    try:
        for index, step in enumerate(steps, start=1):
            duration = float(step.duration_sec)
            if duration == 0.0:
                print(
                    f"[INITIALISATION {index}/{len(steps)}] "
                    f"skip {step.name} (0.00s)"
                )
                continue
            print(
                f"[INITIALISATION {index}/{len(steps)}] "
                f"start {step.name} ({duration:.2f}s)"
            )
            step.start()
            try:
                _wait_while_healthy(duration, initial_error_count)
            finally:
                step.stop()
            _wait_while_healthy(float(settings.settle_sec), initial_error_count)
            print(
                f"[INITIALISATION {index}/{len(steps)}] complete {step.name}"
            )
    finally:
        issue_command_stop()
    print("[INITIALISATION] complete; driving mode + STOP restored.")
