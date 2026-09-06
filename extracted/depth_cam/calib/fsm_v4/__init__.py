"""FSM v4 public entry point.

The controller is imported lazily so deployment tooling can reuse the
standalone rotation-artifact validator without importing runtime config or
selecting an active response as a side effect.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .top import CalibrationFSMV4 as CalibrationFSMV4

__all__ = ["CalibrationFSMV4"]


def __getattr__(name: str):
    if name == "CalibrationFSMV4":
        from .top import CalibrationFSMV4

        return CalibrationFSMV4
    raise AttributeError(name)
