"""Diagnose position drift caused by in-place 90-degree round trips.

The test deliberately runs model inference only while the forklift is stopped:

1. Capture a robust PnP baseline while the pallet is visible in front.
2. For the nominal 90-degree turn, stop the CAN rotation command when gyro.Y
   integration reaches 88 degrees (configurable with ``--imu-stop-deg``).
   After STOP, only measure the settled IMU value; do not apply correction
   pulses or any additional rotation.
3. Rotate 90 degrees back in the opposite direction without vision inference.
4. Capture another stopped PnP observation and compare it with the baseline.

Outputs are written below ``./rec/rotation_return_test_<timestamp>/``.
The sample CSV contains every inference attempt, including failed detections.
The JSON summary contains robust checkpoint statistics and before/after deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from calib.config import (
    ALIGN_ROTATE_TIMEOUT_SEC,
    POSE_CENTER_KPT,
    STOP_SEC,
    STREAM_FPS,
    STREAM_H,
    STREAM_W,
)
from calib.control import can_close, can_init, get_can_status
from calib.fsm.commands import CommandExecutor
from calib.geometry import pose_from_kpts_pnp
from calib.perception import Perception


WINDOW_NAME = "90deg rotation return diagnostic"


class UserAbort(RuntimeError):
    pass


def wrap_to_180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


class RelYawEstimator:
    """The same gyro.Y integration convention currently used by main_rec.py."""

    def __init__(self) -> None:
        self.first = True
        self.last_ts_ms: Optional[float] = None
        self.yaw_deg = 0.0
        self.init_yaw: Optional[float] = None
        self.last_rel = 0.0

    def update_from_frames(self, accel: Any, gyro: Any, ts_ms: float) -> float:
        if self.first:
            self.first = False
            self.last_ts_ms = float(ts_ms)
            self.init_yaw = 0.0
            self.last_rel = 0.0
            return 0.0

        dt = max(0.0, (float(ts_ms) - float(self.last_ts_ms)) / 1000.0)
        self.last_ts_ms = float(ts_ms)
        self.yaw_deg += math.degrees(float(gyro.y) * dt)
        if self.init_yaw is None:
            self.init_yaw = self.yaw_deg
        self.last_rel = wrap_to_180(self.yaw_deg - self.init_yaw)
        return self.last_rel


def color_fps_candidates(width: int, height: int) -> List[int]:
    supported: set[int] = set()
    try:
        for dev in rs.context().query_devices():
            for sensor in dev.query_sensors():
                for profile in sensor.get_stream_profiles():
                    try:
                        vp = profile.as_video_stream_profile()
                    except Exception:
                        continue
                    if (
                        profile.stream_type() == rs.stream.color
                        and profile.format() == rs.format.bgr8
                        and vp.width() == width
                        and vp.height() == height
                    ):
                        supported.add(int(profile.fps()))
    except Exception:
        pass
    return sorted(supported, reverse=True) or [30, 15, 6]


def start_color_imu_pipeline() -> Tuple[rs.pipeline, int]:
    """Start only color+IMU; PnP used by the FSM does not consume depth."""

    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() == 0:
        raise RuntimeError("RealSense device not found")

    pipeline = rs.pipeline()
    candidates = [int(STREAM_FPS)] if STREAM_FPS else color_fps_candidates(STREAM_W, STREAM_H)
    errors: List[str] = []
    for fps in candidates:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.accel)
        cfg.enable_stream(rs.stream.gyro)
        try:
            pipeline.start(cfg)
            print(f"[CAMERA] started {STREAM_W}x{STREAM_H} @ {fps}fps (color+IMU)")
            return pipeline, fps
        except Exception as exc:
            errors.append(f"{fps}fps: {exc}")
            print(f"[CAMERA] {fps}fps failed: {exc}")
    raise RuntimeError("Could not start color+IMU stream: " + " | ".join(errors))


@dataclass(frozen=True)
class TurnLeg:
    name: str
    command: str
    target_deg: float


@dataclass(frozen=True)
class RoundTrip:
    name: str
    outbound: TurnLeg
    returning: TurnLeg


SAMPLE_FIELDS = [
    "timestamp", "checkpoint", "attempt", "valid_index", "detected", "pnp_ok",
    "imu_yaw_deg", "x_m", "y_m", "z_m", "distance_m", "yaw_deg", "pitch_deg",
    "roll_deg", "center_u_px", "center_v_px", "center_bearing_deg",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


def robust_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    return {
        "median": median,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "mad": float(np.median(np.abs(arr - median))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def checkpoint_summary(label: str, samples: List[Dict[str, Any]], attempts: int) -> Dict[str, Any]:
    metrics = [
        "imu_yaw_deg", "x_m", "y_m", "z_m", "distance_m", "yaw_deg",
        "pitch_deg", "roll_deg", "center_u_px", "center_v_px", "center_bearing_deg",
    ]
    return {
        "label": label,
        "attempts": int(attempts),
        "valid_samples": len(samples),
        "pnp_success_rate": float(len(samples) / max(1, attempts)),
        "metrics": {
            name: robust_stats([float(sample[name]) for sample in samples])
            for name in metrics
        },
    }


def metric_median(summary: Dict[str, Any], name: str) -> float:
    return float(summary["metrics"][name]["median"])


def compare_checkpoints(
    reference: Dict[str, Any],
    returned: Dict[str, Any],
    warn_translation_m: float,
    warn_yaw_deg: float,
) -> Dict[str, Any]:
    dx = metric_median(returned, "x_m") - metric_median(reference, "x_m")
    dy = metric_median(returned, "y_m") - metric_median(reference, "y_m")
    dz = metric_median(returned, "z_m") - metric_median(reference, "z_m")
    center_shift_xz = math.hypot(dx, dz)
    vision_yaw_delta = wrap_to_180(
        metric_median(returned, "yaw_deg") - metric_median(reference, "yaw_deg")
    )
    imu_return_error = wrap_to_180(
        metric_median(returned, "imu_yaw_deg") - metric_median(reference, "imu_yaw_deg")
    )
    bearing_delta = wrap_to_180(
        metric_median(returned, "center_bearing_deg")
        - metric_median(reference, "center_bearing_deg")
    )

    position_warning = center_shift_xz > warn_translation_m
    yaw_warning = abs(imu_return_error) > warn_yaw_deg
    if not position_warning:
        assessment = "within_position_threshold"
    elif yaw_warning:
        assessment = "ambiguous_due_to_angular_return_error"
    else:
        assessment = "possible_in_place_translation"

    return {
        "reference": reference["label"],
        "returned": returned["label"],
        "delta_x_m": dx,
        "delta_y_m": dy,
        "delta_z_m": dz,
        "center_shift_xz_m": center_shift_xz,
        "vision_yaw_delta_deg": vision_yaw_delta,
        "center_bearing_delta_deg": bearing_delta,
        "imu_return_error_deg": imu_return_error,
        "warn_translation_m": warn_translation_m,
        "warn_yaw_deg": warn_yaw_deg,
        "position_warning": position_warning,
        "yaw_warning": yaw_warning,
        "assessment": assessment,
        "note": (
            "PnP coordinate change is camera-relative. A non-zero IMU return error can create "
            "an apparent x/z shift even without chassis translation."
        ),
    }


class RotationReturnDiagnostic:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / f"rotation_return_test_{stamp}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.samples_path = self.output_dir / "inference_samples.csv"
        self.events_path = self.output_dir / "events.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.comparisons_path = self.output_dir / "comparisons.csv"

        self.pipeline: Optional[rs.pipeline] = None
        self.stream_fps: Optional[int] = None
        self.perception: Optional[Perception] = None
        self.executor = CommandExecutor()
        self.imu = RelYawEstimator()
        self.rel_yaw = 0.0
        self.last_accel: Any = None
        self.last_color: Optional[np.ndarray] = None
        self.color_intrin: Any = None
        self.can_started = False
        self.samples_fp: Any = None
        self.samples_writer: Any = None
        self.events_fp: Any = None
        self.checkpoints: List[Dict[str, Any]] = []
        self.comparisons: List[Dict[str, Any]] = []
        self.turn_events: List[Dict[str, Any]] = []

    def log_event(self, event: str, **data: Any) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "monotonic_s": time.monotonic(),
            "event": event,
            **data,
        }
        self.events_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.events_fp.flush()

    def next_frame(self, timeout_ms: int = 5000) -> np.ndarray:
        assert self.pipeline is not None
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        accel_now = None
        for frame in frames:
            if not (hasattr(frame, "is_motion_frame") and frame.is_motion_frame()):
                continue
            stream_type = frame.get_profile().stream_type()
            motion = frame.as_motion_frame().get_motion_data()
            if stream_type == rs.stream.accel:
                accel_now = motion
                self.last_accel = motion
            elif stream_type == rs.stream.gyro:
                use_accel = accel_now if accel_now is not None else self.last_accel
                if use_accel is not None:
                    self.rel_yaw = self.imu.update_from_frames(
                        use_accel, motion, frame.get_timestamp()
                    )

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Color frame missing")
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        self.last_color = np.asanyarray(color_frame.get_data())
        return self.last_color

    def show(self, image: np.ndarray, lines: Sequence[str], wait_ms: int = 1) -> int:
        canvas = image.copy()
        overlay_h = 30 + 25 * len(lines)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], overlay_h), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            cv2.putText(
                canvas, str(line), (12, 24 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 27:
            raise UserAbort("ESC pressed")
        return key

    def infer(self, color: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        assert self.perception is not None
        det_ok, kpts4, bbox, kpts_all = self.perception.infer_front(color)
        record: Dict[str, Any] = {
            "detected": bool(det_ok),
            "pnp_ok": False,
            "imu_yaw_deg": float(self.rel_yaw),
        }
        vis = color.copy()

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            record.update(bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)

        if kpts_all is not None:
            for index, keypoint in enumerate(kpts_all):
                if len(keypoint) < 2 or not np.all(np.isfinite(keypoint[:2])):
                    continue
                u, v = int(round(float(keypoint[0]))), int(round(float(keypoint[1])))
                cv2.circle(vis, (u, v), 3, (255, 0, 255), -1)
                cv2.putText(vis, str(index), (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 0, 255), 1, cv2.LINE_AA)

        if not det_ok or kpts_all is None:
            return record, vis

        ok, yaw, pitch, roll, center, _rvec, _tvec = pose_from_kpts_pnp(
            kpts=kpts_all, intrin=self.color_intrin
        )
        if not ok:
            return record, vis

        x_m, y_m, z_m = map(float, center)
        if kpts_all is not None and len(kpts_all) > POSE_CENTER_KPT:
            center_u = float(kpts_all[POSE_CENTER_KPT][0])
            center_v = float(kpts_all[POSE_CENTER_KPT][1])
        else:
            center_u = float(np.mean(kpts4[:, 0]))
            center_v = float(np.mean(kpts4[:, 1]))
        bearing = math.degrees(
            math.atan2((center_u - float(self.color_intrin.ppx)) / float(self.color_intrin.fx), 1.0)
        )
        record.update(
            pnp_ok=True,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            distance_m=float(np.linalg.norm(center)),
            yaw_deg=float(yaw),
            pitch_deg=float(pitch),
            roll_deg=float(roll),
            center_u_px=center_u,
            center_v_px=center_v,
            center_bearing_deg=bearing,
        )
        cv2.drawMarker(vis, (int(round(center_u)), int(round(center_v))), (0, 255, 0),
                       cv2.MARKER_CROSS, 18, 2)
        return record, vis

    def write_sample(self, checkpoint: str, attempt: int, valid_index: int,
                     record: Dict[str, Any]) -> None:
        row = {field: "" for field in SAMPLE_FIELDS}
        row.update(record)
        row.update(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            checkpoint=checkpoint,
            attempt=attempt,
            valid_index=valid_index if valid_index > 0 else "",
        )
        self.samples_writer.writerow(row)
        self.samples_fp.flush()

    def wait_for_user_start(self) -> None:
        print("\n[READY] Place the pallet fully in front of the camera.")
        print("[READY] Press SPACE in the preview window to start. ESC aborts safely.")
        while True:
            color = self.next_frame()
            record, vis = self.infer(color)
            state = "PnP OK" if record["pnp_ok"] else "NO VALID PnP"
            key = self.show(vis, [
                f"PREVIEW: {state}",
                f"IMU yaw: {self.rel_yaw:+.2f} deg",
                "SPACE: start test   ESC: abort",
            ])
            if key == ord(" "):
                if record["pnp_ok"]:
                    self.log_event("user_start", imu_yaw_deg=self.rel_yaw)
                    return
                print("[READY] PnP is not valid yet; start rejected.")

    def hold_stop(self, seconds: float, phase: str) -> None:
        self.executor.exec("STOP")
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            color = self.next_frame()
            remaining = max(0.0, seconds - (time.monotonic() - started))
            self.show(color, [f"STOP / settle: {phase}", f"remaining {remaining:.1f}s",
                              f"IMU yaw {self.rel_yaw:+.2f} deg"])

    def settle_rotation(self, yaw_ref: float, phase: str, min_sec: float) -> Dict[str, Any]:
        """Wait for stopped IMU values to become stable and return a window median."""

        self.executor.exec("STOP")
        started = time.monotonic()
        window: List[Tuple[float, float]] = []
        stable = False
        spread = float("inf")
        median_delta = wrap_to_180(self.rel_yaw - yaw_ref)

        while time.monotonic() - started < self.args.rotation_settle_timeout_sec:
            color = self.next_frame()
            now = time.monotonic()
            elapsed = now - started
            delta = wrap_to_180(self.rel_yaw - yaw_ref)
            window.append((now, delta))
            cutoff = now - self.args.rotation_stable_window_sec
            window = [(ts, value) for ts, value in window if ts >= cutoff]
            values = [value for _ts, value in window]
            if values:
                median_delta = float(np.median(np.asarray(values, dtype=np.float64)))
                spread = float(max(values) - min(values))

            window_span = window[-1][0] - window[0][0] if len(window) >= 2 else 0.0
            enough_window = window_span >= self.args.rotation_stable_window_sec * 0.75
            stable = (
                elapsed >= min_sec
                and enough_window
                and spread <= self.args.rotation_stable_spread_deg
            )
            self.show(color, [
                f"STOP / IMU settle: {phase}",
                f"median delta {median_delta:+.2f}deg / spread {spread:.2f}deg",
                f"stable={stable}  elapsed={elapsed:.1f}s",
            ])
            if stable:
                break

        result = {
            "delta_deg": float(median_delta),
            "spread_deg": float(spread),
            "stable": bool(stable),
            "elapsed_s": float(time.monotonic() - started),
        }
        self.log_event("rotation_stop_settled", phase=phase, **result)
        if not stable:
            print(
                f"[WARN] IMU did not meet settled spread for {phase}: "
                f"median={median_delta:+.2f}deg, spread={spread:.2f}deg"
            )
        return result

    def capture_checkpoint(self, label: str) -> Dict[str, Any]:
        self.executor.exec("STOP")
        valid: List[Dict[str, Any]] = []
        attempts = 0
        started = time.monotonic()
        representative: Optional[np.ndarray] = None
        self.log_event("checkpoint_begin", checkpoint=label, requested_valid=self.args.samples)

        while len(valid) < self.args.samples:
            if time.monotonic() - started > self.args.capture_timeout:
                raise RuntimeError(
                    f"Checkpoint '{label}' timed out: {len(valid)}/{self.args.samples} valid PnP samples"
                )
            color = self.next_frame()
            attempts += 1
            record, vis = self.infer(color)
            valid_index = 0
            if record["pnp_ok"]:
                valid.append(record)
                valid_index = len(valid)
                representative = vis
            self.write_sample(label, attempts, valid_index, record)
            self.show(vis, [
                f"CHECKPOINT: {label}",
                f"valid PnP: {len(valid)}/{self.args.samples} (attempts {attempts})",
                f"IMU yaw: {self.rel_yaw:+.2f} deg",
            ])

        summary = checkpoint_summary(label, valid, attempts)
        self.checkpoints.append(summary)
        if representative is not None:
            med_x = metric_median(summary, "x_m")
            med_z = metric_median(summary, "z_m")
            med_yaw = metric_median(summary, "yaw_deg")
            annotated = representative.copy()
            cv2.putText(annotated, f"median x={med_x:+.3f}m z={med_z:.3f}m yaw={med_yaw:+.2f}deg",
                        (10, annotated.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(self.output_dir / f"{label}.png"), annotated)
        self.log_event("checkpoint_end", checkpoint=label, summary=summary)
        print(
            f"[CHECKPOINT] {label}: x={metric_median(summary, 'x_m'):+.3f}m, "
            f"z={metric_median(summary, 'z_m'):.3f}m, "
            f"yaw={metric_median(summary, 'yaw_deg'):+.2f}deg, "
            f"PnP={len(valid)}/{attempts}"
        )
        return summary

    def rotate(self, leg: TurnLeg) -> Dict[str, Any]:
        start_yaw = float(self.rel_yaw)
        nominal_target_deg = 90.0 if leg.target_deg > 0.0 else -90.0
        started = time.monotonic()
        self.executor.exec(leg.command)
        self.log_event("rotation_begin", name=leg.name, command=leg.command,
                       target_deg=leg.target_deg, imu_ref_deg=start_yaw)
        reached = False
        delta = 0.0
        while time.monotonic() - started < self.args.rotation_timeout:
            color = self.next_frame()
            delta = wrap_to_180(self.rel_yaw - start_yaw)
            reached = delta >= leg.target_deg if leg.target_deg > 0 else delta <= leg.target_deg
            self.show(color, [
                f"ROTATING: {leg.name} / {leg.command}",
                f"IMU delta {delta:+.2f} / target {leg.target_deg:+.2f} deg",
                "Inference OFF during rotation   ESC: emergency stop",
            ])
            if reached:
                break

        self.executor.exec("STOP")
        primary_elapsed = time.monotonic() - started
        threshold_delta = float(delta)
        self.log_event(
            "rotation_primary_stop",
            name=leg.name,
            command=leg.command,
            target_deg=leg.target_deg,
            threshold_delta_deg=threshold_delta,
            elapsed_s=primary_elapsed,
        )
        if not reached:
            result = {
                "name": leg.name,
                "command": leg.command,
                "target_deg": float(leg.target_deg),
                "threshold_delta_deg": threshold_delta,
                "actual_delta_deg": threshold_delta,
                "final_error_deg": wrap_to_180(leg.target_deg - threshold_delta),
                "primary_elapsed_s": float(primary_elapsed),
                "elapsed_s": float(primary_elapsed),
                "reached": False,
                "correction_enabled": bool(self.args.turn_correction),
                "correction_converged": False,
                "correction_attempts": [],
            }
            self.turn_events.append(result)
            self.log_event("rotation_end", **result)
            raise RuntimeError(f"Rotation timeout: {leg.name}, delta={delta:+.2f}deg")

        print(
            f"[ROTATE] {leg.name}: threshold {threshold_delta:+.2f}deg "
            f"in {primary_elapsed:.2f}s; settling before IMU correction"
        )
        settled_measurement = self.settle_rotation(
            start_yaw, f"primary stop after {leg.name}", self.args.settle_sec
        )

        correction_attempts: List[Dict[str, Any]] = []
        settled_delta = float(settled_measurement["delta_deg"])
        correction_error = wrap_to_180(leg.target_deg - settled_delta)
        post_primary_settled_delta = settled_delta
        post_primary_error = correction_error
        final_settle_measurement = settled_measurement
        self.log_event(
            "rotation_settled",
            name=leg.name,
            target_deg=leg.target_deg,
            nominal_target_deg=nominal_target_deg,
            settled_delta_deg=settled_delta,
            error_deg=correction_error,
            imu_stable=settled_measurement["stable"],
            imu_spread_deg=settled_measurement["spread_deg"],
        )

        if self.args.turn_correction:
            for attempt_index in range(1, self.args.correction_max_attempts + 1):
                correction_error = wrap_to_180(leg.target_deg - settled_delta)
                if abs(correction_error) <= self.args.correction_tolerance_deg:
                    break
                if abs(correction_error) > self.args.max_correctable_error_deg:
                    raise RuntimeError(
                        f"Settled rotation error is too large for pulse correction: "
                        f"{leg.name}, error={correction_error:+.2f}deg"
                    )

                correction_command = "ROT_RIGHT" if correction_error > 0.0 else "ROT_LEFT"
                pulse_sec = (
                    abs(correction_error)
                    / max(0.1, float(self.args.correction_rate_deg_s))
                    * float(self.args.correction_gain)
                )
                pulse_sec = max(
                    float(self.args.correction_min_pulse_sec),
                    min(float(self.args.correction_max_pulse_sec), pulse_sec),
                )
                before_delta = settled_delta
                before_error = correction_error
                pulse_started = time.monotonic()
                self.executor.exec(correction_command)
                self.log_event(
                    "rotation_correction_begin",
                    name=leg.name,
                    attempt=attempt_index,
                    command=correction_command,
                    target_deg=leg.target_deg,
                    before_delta_deg=before_delta,
                    before_error_deg=before_error,
                    planned_pulse_sec=pulse_sec,
                )

                # A correction is a bounded pulse. Stop early if IMU reaches/crosses
                # the target before the planned duration expires.
                while time.monotonic() - pulse_started < pulse_sec:
                    color = self.next_frame()
                    live_delta = wrap_to_180(self.rel_yaw - start_yaw)
                    live_error = wrap_to_180(leg.target_deg - live_delta)
                    crossed = (
                        (before_error > 0.0 and live_error <= 0.0)
                        or (before_error < 0.0 and live_error >= 0.0)
                    )
                    self.show(color, [
                        f"IMU CORRECTION {attempt_index}/{self.args.correction_max_attempts}",
                        f"{correction_command} pulse <= {pulse_sec:.2f}s",
                        f"delta {live_delta:+.2f}deg / error {live_error:+.2f}deg",
                        "Inference OFF   ESC: emergency stop",
                    ])
                    if abs(live_error) <= self.args.correction_tolerance_deg or crossed:
                        break

                self.executor.exec("STOP")
                actual_pulse_sec = time.monotonic() - pulse_started
                correction_measurement = self.settle_rotation(
                    start_yaw,
                    f"correction {attempt_index} after {leg.name}",
                    self.args.correction_settle_sec,
                )
                after_delta = float(correction_measurement["delta_deg"])
                after_error = wrap_to_180(leg.target_deg - after_delta)
                attempt_result = {
                    "attempt": attempt_index,
                    "command": correction_command,
                    "planned_pulse_sec": float(pulse_sec),
                    "actual_pulse_sec": float(actual_pulse_sec),
                    "before_delta_deg": float(before_delta),
                    "before_error_deg": float(before_error),
                    "after_delta_deg": float(after_delta),
                    "after_error_deg": float(after_error),
                    "imu_stable": bool(correction_measurement["stable"]),
                    "imu_spread_deg": float(correction_measurement["spread_deg"]),
                }
                correction_attempts.append(attempt_result)
                settled_delta = after_delta
                final_settle_measurement = correction_measurement
                self.log_event("rotation_correction_end", name=leg.name, **attempt_result)
                print(
                    f"[CORRECT] {leg.name} #{attempt_index}: {correction_command} "
                    f"{actual_pulse_sec:.2f}s, error {before_error:+.2f} -> "
                    f"{after_error:+.2f}deg"
                )

        final_delta = float(settled_delta)
        final_error = wrap_to_180(leg.target_deg - final_delta)
        nominal_90_error = wrap_to_180(nominal_target_deg - final_delta)
        correction_converged: Optional[bool] = None
        if self.args.turn_correction:
            correction_converged = abs(final_error) <= self.args.correction_tolerance_deg
        elapsed = time.monotonic() - started
        result = {
            "name": leg.name,
            "command": leg.command,
            "target_deg": float(leg.target_deg),
            "nominal_target_deg": float(nominal_target_deg),
            "threshold_delta_deg": threshold_delta,
            "post_primary_settled_delta_deg": float(post_primary_settled_delta),
            "post_primary_error_deg": float(post_primary_error),
            "post_primary_imu_stable": bool(settled_measurement["stable"]),
            "post_primary_imu_spread_deg": float(settled_measurement["spread_deg"]),
            "actual_delta_deg": float(final_delta),
            "final_error_deg": float(final_error),
            "final_error_to_nominal_90_deg": float(nominal_90_error),
            "final_imu_stable": bool(final_settle_measurement["stable"]),
            "final_imu_spread_deg": float(final_settle_measurement["spread_deg"]),
            "absolute_error_improvement_deg": float(
                abs(post_primary_error) - abs(final_error)
            ),
            "primary_elapsed_s": float(primary_elapsed),
            "elapsed_s": float(elapsed),
            "reached": True,
            "correction_enabled": bool(self.args.turn_correction),
            "correction_converged": correction_converged,
            "correction_attempts": correction_attempts,
        }
        self.turn_events.append(result)
        self.log_event("rotation_end", **result)
        print(
            f"[ROTATE] {leg.name}: settled {final_delta:+.2f}deg, "
            f"nominal 90deg error {nominal_90_error:+.2f}deg, "
            f"post-correction=OFF"
        )
        if self.args.turn_correction and not correction_converged:
            print(
                f"[WARN] {leg.name}: correction limit reached; continuing so the "
                "before/after inference result is still recorded."
            )
        return result

    def build_round_trips(self) -> List[RoundTrip]:
        angle = float(self.args.turn_deg)
        left_right = RoundTrip(
            "left_then_right",
            TurnLeg("left_outbound", "ROT_LEFT", -angle),
            TurnLeg("right_return", "ROT_RIGHT", +angle),
        )
        right_left = RoundTrip(
            "right_then_left",
            TurnLeg("right_outbound", "ROT_RIGHT", +angle),
            TurnLeg("left_return", "ROT_LEFT", -angle),
        )
        selected: List[RoundTrip] = []
        for _ in range(self.args.repeats):
            if self.args.sequence in ("left-right", "both"):
                selected.append(left_right)
            if self.args.sequence in ("right-left", "both"):
                selected.append(right_left)
        return selected

    def write_results(self, completed: bool, error: Optional[str] = None) -> None:
        summary = {
            "completed": completed,
            "error": error,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "configuration": {
                "samples_per_checkpoint": self.args.samples,
                "sequence": self.args.sequence,
                "repeats": self.args.repeats,
                "nominal_turn_deg": 90.0,
                "imu_stop_deg": self.args.turn_deg,
                "settle_sec": self.args.settle_sec,
                "rotation_timeout": self.args.rotation_timeout,
                "rotation_settle_timeout_sec": self.args.rotation_settle_timeout_sec,
                "rotation_stable_window_sec": self.args.rotation_stable_window_sec,
                "rotation_stable_spread_deg": self.args.rotation_stable_spread_deg,
                "turn_correction": self.args.turn_correction,
                "correction_tolerance_deg": self.args.correction_tolerance_deg,
                "correction_max_attempts": self.args.correction_max_attempts,
                "correction_settle_sec": self.args.correction_settle_sec,
                "correction_rate_deg_s": self.args.correction_rate_deg_s,
                "correction_gain": self.args.correction_gain,
                "correction_min_pulse_sec": self.args.correction_min_pulse_sec,
                "correction_max_pulse_sec": self.args.correction_max_pulse_sec,
                "max_correctable_error_deg": self.args.max_correctable_error_deg,
                "warn_translation_cm": self.args.warn_translation_cm,
                "warn_yaw_deg": self.args.warn_yaw_deg,
                "stream_width": STREAM_W,
                "stream_height": STREAM_H,
                "stream_fps": self.stream_fps,
            },
            "checkpoints": self.checkpoints,
            "turns": self.turn_events,
            "comparisons": self.comparisons,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        if self.comparisons:
            fields = [key for key in self.comparisons[0].keys() if key != "note"]
            with self.comparisons_path.open("w", newline="", encoding="utf-8-sig") as fp:
                writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.comparisons)

    def print_comparison(self, comparison: Dict[str, Any]) -> None:
        print(
            f"[COMPARE] {comparison['reference']} -> {comparison['returned']}: "
            f"dx={comparison['delta_x_m'] * 100:+.1f}cm, "
            f"dz={comparison['delta_z_m'] * 100:+.1f}cm, "
            f"xz shift={comparison['center_shift_xz_m'] * 100:.1f}cm, "
            f"IMU return={comparison['imu_return_error_deg']:+.2f}deg, "
            f"vision yaw={comparison['vision_yaw_delta_deg']:+.2f}deg, "
            f"assessment={comparison['assessment']}"
        )

    def run(self) -> int:
        self.samples_fp = self.samples_path.open("w", newline="", encoding="utf-8-sig")
        self.samples_writer = csv.DictWriter(self.samples_fp, fieldnames=SAMPLE_FIELDS,
                                             extrasaction="ignore")
        self.samples_writer.writeheader()
        self.events_fp = self.events_path.open("w", encoding="utf-8")
        error: Optional[str] = None
        completed = False

        try:
            if not can_init():
                status = get_can_status()
                raise RuntimeError(f"CAN initialization failed: {status.get('last_error')}")
            self.can_started = True
            self.executor.exec("STOP")
            status = get_can_status()
            if not status["ready"]:
                raise RuntimeError(f"CAN is not ready: {status}")

            self.pipeline, self.stream_fps = start_color_imu_pipeline()
            self.perception = Perception()
            self.wait_for_user_start()
            self.hold_stop(max(float(STOP_SEC), self.args.settle_sec), "initial settle")

            baseline = self.capture_checkpoint("baseline")
            previous = baseline
            baseline_ref = baseline
            for index, trip in enumerate(self.build_round_trips(), start=1):
                cycle_name = f"cycle_{index:02d}_{trip.name}"
                cycle_imu_start = metric_median(previous, "imu_yaw_deg")
                self.log_event("round_trip_begin", name=cycle_name, imu_start_deg=cycle_imu_start)
                self.rotate(trip.outbound)
                self.rotate(trip.returning)
                returned = self.capture_checkpoint(f"{cycle_name}_returned")

                incremental = compare_checkpoints(
                    previous, returned, self.args.warn_translation_cm / 100.0,
                    self.args.warn_yaw_deg,
                )
                incremental["comparison_type"] = "incremental"
                self.comparisons.append(incremental)
                self.print_comparison(incremental)

                if previous is not baseline_ref:
                    cumulative = compare_checkpoints(
                        baseline_ref, returned, self.args.warn_translation_cm / 100.0,
                        self.args.warn_yaw_deg,
                    )
                    cumulative["comparison_type"] = "from_baseline"
                    self.comparisons.append(cumulative)
                    self.print_comparison(cumulative)
                previous = returned
                self.log_event("round_trip_end", name=cycle_name,
                               returned_checkpoint=returned["label"])

            completed = True
            self.write_results(completed=True)
            print(f"\n[DONE] Results: {self.output_dir.resolve()}")
            print(f"[DONE] Summary: {self.summary_path.resolve()}")
            return 0
        except UserAbort as exc:
            error = str(exc)
            print(f"[ABORT] {error}")
            return 130
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {error}", file=sys.stderr)
            return 1
        finally:
            try:
                self.executor.exec("STOP")
            except Exception:
                pass
            if not completed:
                try:
                    self.write_results(completed=False, error=error)
                except Exception:
                    pass
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            if self.can_started:
                try:
                    can_close()
                except Exception:
                    pass
            if self.samples_fp is not None:
                self.samples_fp.close()
            if self.events_fp is not None:
                self.events_fp.close()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PnP position changes after in-place 90-degree round trips."
    )
    parser.add_argument("--sequence", choices=("left-right", "right-left", "both"),
                        default="both", help="Round-trip direction order (default: both).")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repetitions per selected order (default: 1).")
    parser.add_argument("--samples", type=int, default=15,
                        help="Valid stopped PnP samples per checkpoint (minimum 5).")
    parser.add_argument("--imu-stop-deg", "--turn-deg", dest="turn_deg",
                        type=float, default=88.0,
                        help="IMU angle that stops each nominal 90-degree turn (default: 88).")
    parser.add_argument("--settle-sec", type=float, default=max(1.5, float(STOP_SEC)))
    parser.add_argument("--capture-timeout", type=float, default=20.0)
    parser.add_argument("--rotation-timeout", type=float,
                        default=float(ALIGN_ROTATE_TIMEOUT_SEC))
    parser.add_argument("--rotation-settle-timeout-sec", type=float, default=4.0,
                        help="Maximum STOP wait for a stable IMU window.")
    parser.add_argument("--rotation-stable-window-sec", type=float, default=0.5,
                        help="Recent IMU window used for settled median/spread.")
    parser.add_argument("--rotation-stable-spread-deg", type=float, default=0.25,
                        help="Max-min limit for declaring stopped IMU stable.")
    # This experiment intentionally performs no post-stop correction. The
    # correction implementation remains in the file for logged A/B history,
    # but the executable path is fixed OFF for this 88-degree pre-stop test.
    parser.set_defaults(turn_correction=False)
    parser.add_argument("--correction-tolerance-deg", type=float, default=0.5,
                        help="Required settled 90-degree error (default: +/-0.5deg).")
    parser.add_argument("--correction-max-attempts", type=int, default=6)
    parser.add_argument("--correction-settle-sec", type=float, default=1.0,
                        help="STOP settling time after each correction pulse.")
    parser.add_argument("--correction-rate-deg-s", type=float, default=7.5,
                        help="Approximate slow in-place rotation rate used to size pulses.")
    parser.add_argument("--correction-gain", type=float, default=0.65,
                        help="Fraction of the estimated full correction applied per pulse.")
    parser.add_argument("--correction-min-pulse-sec", type=float, default=0.10)
    parser.add_argument("--correction-max-pulse-sec", type=float, default=0.35)
    parser.add_argument("--max-correctable-error-deg", type=float, default=10.0,
                        help="Abort instead of pulsing if settled error exceeds this safety bound.")
    parser.add_argument("--warn-translation-cm", type=float, default=3.0,
                        help="Flag camera-relative x/z center changes above this value.")
    parser.add_argument("--warn-yaw-deg", type=float, default=2.0,
                        help="Angular return-error warning threshold.")
    parser.add_argument("--output-dir", default="./rec")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.samples < 5:
        parser.error("--samples must be >= 5")
    if not (10.0 <= args.turn_deg <= 120.0):
        parser.error("--turn-deg must be between 10 and 120")
    if args.settle_sec < 0.0 or args.capture_timeout <= 0.0 or args.rotation_timeout <= 0.0:
        parser.error("time values must be positive")
    if args.rotation_settle_timeout_sec < max(args.settle_sec, args.correction_settle_sec):
        parser.error("--rotation-settle-timeout-sec must cover both settle minimums")
    if not (0.2 <= args.rotation_stable_window_sec <= args.rotation_settle_timeout_sec):
        parser.error("stable window must be between 0.2s and settle timeout")
    if not (0.05 <= args.rotation_stable_spread_deg <= 2.0):
        parser.error("--rotation-stable-spread-deg must be between 0.05 and 2.0")
    if not (0.1 <= args.correction_tolerance_deg <= 5.0):
        parser.error("--correction-tolerance-deg must be between 0.1 and 5.0")
    if args.correction_max_attempts < 1:
        parser.error("--correction-max-attempts must be >= 1")
    if args.correction_settle_sec < 0.2:
        parser.error("--correction-settle-sec must be >= 0.2")
    if args.correction_rate_deg_s <= 0.0 or not (0.05 <= args.correction_gain <= 1.0):
        parser.error("correction rate must be positive and gain must be in [0.05, 1.0]")
    if not (0.05 <= args.correction_min_pulse_sec <= args.correction_max_pulse_sec <= 1.0):
        parser.error("correction pulse range must satisfy 0.05 <= min <= max <= 1.0")
    if args.max_correctable_error_deg <= args.correction_tolerance_deg:
        parser.error("--max-correctable-error-deg must exceed correction tolerance")
    return args


def main() -> int:
    return RotationReturnDiagnostic(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
