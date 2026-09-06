"""Compact runtime diagram for FSM v4."""

from __future__ import annotations

import textwrap
from typing import Tuple

import cv2
import numpy as np


V4_FLOW = [
    ("PRECHECK", {"PRECHECK"}),
    ("SEARCH RIGHT <= 30s", {"SEARCH_SWEEP"}),
    ("ACQUIRE / RECOVER", {"ACQUIRE_VERIFY", "RECOVER_VISUAL"}),
    ("COARSE FACE", {"FACE_ROTATE", "FACE_SETTLE"}),
    ("3m STANDOFF", {"STANDOFF_VERIFY", "STANDOFF_MOVE", "STANDOFF_SETTLE"}),
    ("STAGING PLAN", {"STAGING_PLAN"}),
    ("VISIBLE TURN", {"RECENTER_ROTATE", "RECENTER_SETTLE", "WAYPOINT_TURN", "WAYPOINT_TURN_SETTLE"}),
    ("FITTED FORWARD", {"WAYPOINT_DRIVE", "WAYPOINT_DRIVE_SETTLE"}),
    ("FINAL POSE LOCK", {"FINAL_POSE_LOCK", "FINAL_ROTATE", "FINAL_SETTLE"}),
    ("READY / INSERT", {"READY_TO_INSERT", "INSERT_DRIVE", "INSERT_SETTLE"}),
    ("DONE / FAILED", {"DONE", "FAILED"}),
]


def _motion_target_status(fsm):
    """Read the optional UI contract without coupling the diagram to the FSM."""
    try:
        status = getattr(fsm, "motion_target_status", None)
        status = status() if callable(status) else status
    except Exception:
        return None
    return status if isinstance(status, dict) else None


def _draw_motion_target(image, fsm, x: int, y: int, width: int) -> None:
    status = _motion_target_status(fsm)
    height = 66
    kind = None if status is None else status.get("kind")
    accent = (75, 190, 255) if kind == "rotation" else (90, 220, 145)
    if status is None:
        accent = (105, 110, 120)
    cv2.rectangle(image, (x, y), (x + width, y + height), (35, 39, 44), -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), accent, 2)

    if status is None:
        cv2.putText(
            image, "LIVE MOTION TARGET", (x + 10, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 208, 214), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image, "No angle/distance target in this state", (x + 10, y + 47),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (155, 160, 169), 1, cv2.LINE_AA,
        )
        return

    action = "ROTATION" if kind == "rotation" else "DISTANCE"
    direction = str(status.get("direction") or "-")
    cv2.putText(
        image, f"LIVE {action} TARGET  |  {direction}", (x + 10, y + 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, accent, 1, cv2.LINE_AA,
    )
    if not status.get("fixed_target", True):
        cv2.putText(
            image, str(status.get("description") or "no fixed target"),
            (x + 10, y + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (225, 227, 231), 1, cv2.LINE_AA,
        )
        return

    target = float(status.get("target_value", 0.0))
    current = float(status.get("current_value", 0.0))
    remaining = float(status.get("remaining_value", max(0.0, target - current)))
    unit = str(status.get("unit") or "")
    decimals = 1 if unit == "deg" else 3
    values = (
        f"goal {target:.{decimals}f}{unit}   "
        f"done {current:.{decimals}f}{unit}   "
        f"left {remaining:.{decimals}f}{unit}"
    )
    cv2.putText(
        image, values, (x + 10, y + 43), cv2.FONT_HERSHEY_SIMPLEX,
        0.43, (235, 237, 240), 1, cv2.LINE_AA,
    )
    bar_x, bar_y, bar_w, bar_h = x + 10, y + 51, width - 20, 7
    cv2.rectangle(
        image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
        (66, 70, 77), -1,
    )
    ratio = max(0.0, min(1.0, float(status.get("progress_ratio", 0.0))))
    fill = int(round(bar_w * ratio))
    if fill > 0:
        cv2.rectangle(
            image, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), accent, -1,
        )


def draw_fsm_v4_diagram_panel(fsm, panel_size: Tuple[int, int] = (480, 900)):
    height, width = map(int, panel_size)
    image = np.full((height, width, 3), (25, 28, 31), dtype=np.uint8)
    cv2.putText(
        image, "FSM v4 - stopped observe / visible macro action", (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA,
    )
    state = getattr(fsm, "state", "PRECHECK")
    failed = state == "FAILED"
    top_y, margin_x, gap = 46, 22, 5
    footer_reserve = 120 if failed else 112
    available = max(1, height - top_y - footer_reserve)
    box_h = max(24, min(31, (available - gap * (len(V4_FLOW) - 1)) // len(V4_FLOW)))
    box_w = width - 2 * margin_x
    for index, (label, states) in enumerate(V4_FLOW):
        y = top_y + index * (box_h + gap)
        active = state in states
        fill = (42, 103, 72) if active else (47, 50, 56)
        edge = (78, 224, 145) if active else (92, 96, 105)
        cv2.rectangle(image, (margin_x, y), (margin_x + box_w, y + box_h), fill, -1)
        cv2.rectangle(image, (margin_x, y), (margin_x + box_w, y + box_h), edge, 2)
        cv2.putText(
            image, label, (margin_x + 9, y + box_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (246, 246, 246), 1, cv2.LINE_AA,
        )
    command = getattr(getattr(fsm, "cmd_status", None), "code", "STOP")
    if failed:
        flow_bottom = top_y + (len(V4_FLOW) - 1) * (box_h + gap) + box_h
        panel_top = min(height - 108, flow_bottom + 7)
        panel_bottom = height - 7
        cv2.rectangle(
            image, (margin_x, panel_top), (margin_x + box_w, panel_bottom),
            (42, 35, 80), -1,
        )
        cv2.rectangle(
            image, (margin_x, panel_top), (margin_x + box_w, panel_bottom),
            (65, 85, 255), 2,
        )
        failure_state = (
            getattr(fsm, "failure_state", None)
            or getattr(fsm, "_failure_state", None)
            or "UNKNOWN"
        )
        failure_reason = (
            getattr(fsm, "failure_reason", None)
            or getattr(fsm, "_failure_reason", None)
            or "unspecified failure"
        )
        cv2.putText(
            image, f"FSM STOPPED - stage: {failure_state}",
            (margin_x + 10, panel_top + 23), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (105, 130, 255), 2, cv2.LINE_AA,
        )
        reason_lines = textwrap.wrap(
            str(failure_reason), width=88,
            break_long_words=True, break_on_hyphens=False,
        ) or [str(failure_reason)]
        for index, reason_line in enumerate(reason_lines[:3]):
            cv2.putText(
                image, ("reason: " if index == 0 else "        ") + reason_line,
                (margin_x + 10, panel_top + 47 + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 255),
                1, cv2.LINE_AA,
            )
        footer = f"state=FAILED  cmd={command}"
        cv2.putText(
            image, footer, (margin_x + 10, panel_bottom - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (196, 198, 205),
            1, cv2.LINE_AA,
        )
    else:
        flow_bottom = top_y + (len(V4_FLOW) - 1) * (box_h + gap) + box_h
        _draw_motion_target(
            image, fsm, margin_x, min(height - 110, flow_bottom + 9), box_w,
        )
        debug_enabled = bool(getattr(fsm, "debug_step_enabled", False))
        debug_running = bool(getattr(fsm, "debug_step_running", False))
        if debug_enabled:
            debug_text = (
                f"DEBUG STEP: {'RUNNING' if debug_running else 'PAUSED'}  "
                f"count={getattr(fsm, 'debug_step_count', 0)}"
            )
            cv2.putText(
                image, debug_text, (18, height - 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (90, 220, 255) if debug_running else (110, 240, 160),
                1, cv2.LINE_AA,
            )
        footer = f"state={state}  cmd={command}"
        cv2.putText(
            image, footer, (18, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (196, 198, 205), 1, cv2.LINE_AA,
        )
    return image
