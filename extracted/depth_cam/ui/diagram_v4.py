"""Compact runtime diagram for FSM v4."""

from __future__ import annotations

import textwrap
from typing import Tuple

import cv2
import numpy as np

from calib.fsm_v4 import config as cfg
from ui.text_rendering import draw_ui_text


V4_FLOW = [
    ("Acquire Pallet", {"SEARCH_SWEEP", "ACQUIRE_VERIFY", "FACE_ROTATE", "FACE_SETTLE",
                        "INITIAL_VISIBILITY_SWEEP", "INITIAL_POSE_ROTATE", "INITIAL_POSE_SETTLE"}),
    ("Set Standoff", {"STANDOFF_VERIFY", "STANDOFF_MOVE", "STANDOFF_SETTLE"}),
    ("Plan Approach", {"STAGING_PLAN"}),
    ("Execute Approach", {"RECENTER_ROTATE", "RECENTER_SETTLE", "WAYPOINT_TURN",
                           "WAYPOINT_TURN_SETTLE", "WAYPOINT_DRIVE", "WAYPOINT_DRIVE_SETTLE",
                           "FINAL_POSE_LOCK", "FINAL_ROTATE", "FINAL_SETTLE"}),
    ("Insert Forks", {"READY_TO_INSERT", "INSERT_DRIVE", "INSERT_SETTLE"}),
]


def _phase_index(fsm):
    state = getattr(fsm, "state", "PRECHECK")
    if state == "DONE":
        return 4
    if state == "FAILED":
        state = getattr(fsm, "failure_state", None) or getattr(fsm, "_failure_state", None)
    if state in {"RECOVER_VISUAL", "ACQUIRE_VERIFY"}:
        state = getattr(fsm, "_visual_recovery_origin_state", None) or state
    return next((i for i, (_, states) in enumerate(V4_FLOW) if state in states), None)


def _activity(state):
    if state == "RECOVER_VISUAL":
        return "Recovering vision"
    if state == "READY_TO_INSERT":
        return "Ready to insert"
    if state == "FINAL_POSE_LOCK":
        return "Checking final alignment"
    if state.endswith("SETTLE"):
        return "Settling and checking pose"
    if "ROTATE" in state or state == "WAYPOINT_TURN":
        return "Aligning for insertion" if state == "FINAL_ROTATE" else "Rotating"
    if state in {"SEARCH_SWEEP", "INITIAL_VISIBILITY_SWEEP"}:
        return "Searching for pallet"
    return {
        "PRECHECK": "Checking readiness", "ACQUIRE_VERIFY": "Verifying detection",
        "STANDOFF_VERIFY": "Checking distance", "STANDOFF_MOVE": "Adjusting distance",
        "STAGING_PLAN": "Planning next turn and drive", "WAYPOINT_DRIVE": "Driving to waypoint",
        "INSERT_DRIVE": "Inserting forks", "DONE": "Insertion complete", "FAILED": "Motion stopped",
    }.get(state, state.replace("_", " ").title())


def _text(image, value, x, y, width, scale=0.5, color=(235, 237, 240), thickness=1):
    """Fit labels to the available width, including the narrow runtime panel."""
    draw_ui_text(image, value, x, y, width, max(14, round(scale * 32)),
                 color, bold=thickness > 1)


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
        _text(image, "LIVE MOTION TARGET", x + 10, y + 21, width - 20,
              0.45, (205, 208, 214))
        _text(image, "No angle/distance target in this state", x + 10, y + 47,
              width - 20, 0.43, (155, 160, 169))
        return

    action = "ROTATION" if kind == "rotation" else "INSERTION TIME" if kind == "timed_insertion" else "DISTANCE"
    direction = str(status.get("direction") or "-")
    _text(image, f"LIVE {action} TARGET  |  {direction}", x + 10, y + 21,
          width - 20, 0.45, accent)
    if not status.get("fixed_target", True):
        _text(image, str(status.get("description") or "no fixed target"),
              x + 10, y + 47, width - 20, 0.43)
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
    _text(image, values, x + 10, y + 43, width - 20, 0.43)
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
    # Render small camera previews at a legible base size, then scale as a unit.
    if height < 480 or width < 420:
        canvas = draw_fsm_v4_diagram_panel(fsm, (max(480, height), max(420, width)))
        return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    image = np.full((height, width, 3), (25, 28, 31), dtype=np.uint8)
    state = getattr(fsm, "state", "PRECHECK")
    phase = _phase_index(fsm)
    failed, done = state == "FAILED", state == "DONE"
    recovering = state == "RECOVER_VISUAL" or (
        state == "ACQUIRE_VERIFY" and getattr(fsm, "_visual_recovery_origin_state", None)
    )
    accent = (80, 100, 255) if failed else (90, 220, 255) if recovering else (78, 224, 145)
    badge = ("FAILED" if failed else "DONE" if done else "PRECHECK" if state == "PRECHECK"
             else "RECOVERING" if recovering else "RUNNING")
    _text(image, "FSM v4 | Pallet approach", 18, 27, width - 155, 0.62, thickness=2)
    cv2.rectangle(image, (width - 129, 9), (width - 18, 34), (47, 50, 56), -1)
    _text(image, badge, width - 122, 27, 98, 0.43, accent)
    _text(image, _activity(state), 18, 54, width - 36, 0.48, accent)

    descriptions = [
        "Find pallet and center the view",
        f"Target: {cfg.SAFETY_STANDOFF_Z_M:.1f} m | skip when already closer",
        "Calculate the next turn and distance",
        "Turn, drive and check final alignment",
        "Advance forks by the accepted distance",
    ]
    top, gap, left, right = 68, 18, 22, width - 65
    footer_height = 134
    box_h = min(88, (height - top - footer_height - 4 * gap) // 5)
    centers = []
    for index, ((label, _states), description) in enumerate(zip(V4_FLOW, descriptions)):
        y = top + index * (box_h + gap)
        centers.append(y + box_h // 2)
        active = index == phase
        fill = (42, 35, 80) if active and failed else (42, 103, 72) if active else (47, 50, 56)
        edge = accent if active else (92, 96, 105)
        cv2.rectangle(image, (left, y), (right, y + box_h), fill, -1)
        cv2.rectangle(image, (left, y), (right, y + box_h), edge, 2)
        _text(image, f"{index + 1:02d}  {label}", left + 12, y + 21,
              right - left - 24, 0.62, thickness=2)
        _text(image, description, left + 12, y + box_h - 8,
              right - left - 24, 0.41, (185, 193, 204))
        if index < 4:
            middle = (left + right) // 2
            cv2.arrowedLine(image, (middle, y + box_h + 3),
                            (middle, y + box_h + gap - 3), (120, 130, 145), 1,
                            cv2.LINE_AA, tipLength=0.4)
    # The only return edge shown at this level is Execute -> Plan.
    loop_x = right + 20
    loop_color = (150, 165, 190)
    cv2.line(image, (right + 3, centers[3]), (loop_x, centers[3]), loop_color, 1, cv2.LINE_AA)
    cv2.line(image, (loop_x, centers[3]), (loop_x, centers[2]), loop_color, 1, cv2.LINE_AA)
    cv2.arrowedLine(image, (loop_x, centers[2]), (right + 3, centers[2]),
                    loop_color, 1, cv2.LINE_AA, tipLength=0.35)
    # Rotate a small label so the return edge does not reduce card width.
    label_image = np.full((19, 65, 3), (25, 28, 31), dtype=np.uint8)
    _text(label_image, "Replan", 2, 14, 61, 0.4, loop_color)
    label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    label_y = (centers[2] + centers[3] - 65) // 2
    image[label_y:label_y + 65, loop_x + 5:loop_x + 24] = label_image

    footer_y = top + 5 * box_h + 4 * gap + 12
    if failed:
        reason = str(getattr(fsm, "failure_reason", None)
                     or getattr(fsm, "_failure_reason", None) or "unspecified failure")
        origin = getattr(fsm, "failure_state", None) or getattr(fsm, "_failure_state", None) or "UNKNOWN"
        cv2.rectangle(image, (left, footer_y), (width - 22, footer_y + 77), (42, 35, 80), -1)
        _text(image, f"Stopped at: {origin}", left + 10, footer_y + 19, width - 64, 0.46, accent)
        lines = textwrap.wrap(reason, width=max(30, (width - 64) // 7))
        if len(lines) > 2:
            lines = lines[:2]
            lines[-1] = lines[-1].rstrip() + "..."
        for index, line in enumerate(lines):
            _text(image, line, left + 10, footer_y + 41 + index * 19, width - 64, 0.43)
    elif done:
        _text(image, "Insertion complete", left, footer_y + 26, width - 44, 0.6, accent, 2)
    else:
        _draw_motion_target(image, fsm, left, footer_y, width - 44)
    command = getattr(getattr(fsm, "cmd_status", None), "code", "STOP")
    if getattr(fsm, "debug_step_enabled", False):
        running = bool(getattr(fsm, "debug_step_running", False))
        debug = f"DEBUG STEP: {'RUNNING' if running else 'PAUSED'}  count={getattr(fsm, 'debug_step_count', 0)}"
        _text(image, debug, 18, height - 29, width - 36, 0.42, (90, 220, 255))
    _text(image, f"state={state}  cmd={command}", 18, height - 11, width - 36, 0.4, (196, 198, 205))
    return image
