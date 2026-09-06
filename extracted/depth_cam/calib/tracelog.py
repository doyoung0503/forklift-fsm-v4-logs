# calib/tracelog.py
# 정렬 시퀀스 로깅 — 추후 digital twin 에서 시퀀스를 재생/시각화하기 위한 기록.
#
#   {base}_meta.json          카메라 좌표계 / intrinsics / 설정 (1회)
#   {base}_pallet_state.csv   파렛트 상태(위치·방향) 를 TRACE_STATE_HZ 로 기록
#   {base}_control_seq.jsonl  제어 시퀀스. 한 줄이 한 이벤트, 순서가 곧 시퀀스
#
# control_seq.jsonl 의 phase:
#   "begin" 스텝 시작 (의도한 목표값 + 그 시점의 파렛트 상태)
#   "end"   스텝 종료 (실제 달성값 + 소요시간)
#   "cmd"   CAN 명령이 실제로 바뀐 순간 (CommandExecutor 중복억제 통과분)

from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any


CSV_HEADER = ("t_iso,t_mono,frame_i,fsm_state,align_sub,det_ok,"
              "pos_x,pos_y,pos_z,yaw_deg,yaw_model_raw_deg,width_m,fps,"
              "rot_stop_command,rot_stop_mode,rot_stop_measured_rate_deg_s,"
              "rot_stop_effective_rate_deg_s,rot_stop_error_deg,"
              "rot_post_stop_deg,rot_post_stop_peak_abs_deg\n")

INFERENCE_TIMING_HEADER = (
    "t_iso,frame_i,raw_video_frame_index,fsm_state,align_sub,camera_frame_number,"
    "camera_frame_delta,camera_sensor_timestamp_ms,camera_sensor_interval_ms,"
    "camera_timestamp_domain,camera_input_host_mono_ms,camera_input_interval_ms,"
    "inference_ran,inference_start_host_mono_ms,inference_end_host_mono_ms,"
    "model_inference_ms,input_to_inference_end_ms,pose_result_host_mono_ms,"
    "input_to_pose_result_ms,model_det_ok,pnp_ok,pose_src,pose_n_used,"
    "pose_rms_px,yaw_deg,pos_x_m,pos_z_m,"
    "center_bearing_deg\n"
)


def _f(v, nd=6):
    """None 안전 포맷."""
    return "" if v is None else f"{float(v):.{nd}f}"


class TraceLogger:
    def __init__(self, base_path: str, state_hz: float = 10.0, enabled: bool = True):
        self.enabled = bool(enabled)
        self.base = base_path
        self._period = 1.0 / max(0.1, float(state_hz))
        self._last_write = 0.0
        self._step = 0
        self._open_step: Optional[Dict[str, Any]] = None
        self._latest: Dict[str, Any] = {}
        self._emit_lock = threading.Lock()
        self._can_tx_queue: queue.Queue = queue.Queue()
        self._can_tx_writer_stop = threading.Event()
        self._can_tx_writer_thread: Optional[threading.Thread] = None
        self._csv = None
        self._timing = None
        self._seq = None
        self._timing_rows = 0
        self._last_camera_input_mono_ms: Optional[float] = None
        self._last_camera_sensor_timestamp_ms: Optional[float] = None
        self._last_camera_timestamp_domain: Optional[str] = None
        self._last_camera_frame_number: Optional[int] = None
        if not self.enabled:
            return
        d = os.path.dirname(base_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._csv = open(base_path + "_pallet_state.csv", "w", encoding="utf-8")
        self._csv.write(CSV_HEADER)
        self._timing = open(
            base_path + "_inference_timing.csv", "w", encoding="utf-8",
        )
        self._timing.write(INFERENCE_TIMING_HEADER)
        self._seq = open(base_path + "_control_seq.jsonl", "w", encoding="utf-8")
        self._can_tx_writer_thread = threading.Thread(
            target=self._can_tx_writer_loop,
            name="CanTxTraceWriter",
            daemon=True,
        )
        self._can_tx_writer_thread.start()

    # ------------------------------------------------------------------ meta
    def write_meta(self, intrin, depth_units: Optional[float], stream_fps, extra: Optional[dict] = None):
        """카메라 기준 좌표계 정보. depth 가 color 로 정렬돼 있으므로 color optical frame 기준."""
        if not self.enabled:
            return
        try:
            coeffs = [float(c) for c in intrin.coeffs]
        except Exception:
            coeffs = None
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "frame": "camera_optical (OpenCV): +X right, +Y down, +Z forward",
            "origin": "RealSense color optical center (depth aligned to color)",
            "units": {"length": "m", "angle": "deg", "time": "s"},
            "yaw_sign": "compute_yaw_deg_from_plane: atan2(-a, 1), 평면 z=ax+by+c 의 x-z 투영",
            "intrinsics": {
                "fx": getattr(intrin, "fx", None), "fy": getattr(intrin, "fy", None),
                "ppx": getattr(intrin, "ppx", None), "ppy": getattr(intrin, "ppy", None),
                "width": getattr(intrin, "width", None), "height": getattr(intrin, "height", None),
                "model": str(getattr(intrin, "model", "")), "coeffs": coeffs,
            },
            "depth_units_m": depth_units,
            "stream_fps": stream_fps,
            "inference_timing": {
                "file_suffix": "_inference_timing.csv",
                "units": "ms",
                "raw_video_frame_index": (
                    "zero-based index in *_raw.mp4; blank when this camera "
                    "frame was not written to the raw video"
                ),
                "camera_sensor_timestamp_ms": (
                    "RealSense frame timestamp in the per-row timestamp domain"
                ),
                "host_mono_ms": (
                    "Python monotonic clock; use differences, not wall time"
                ),
                "rotation_rate_timebase": (
                    "prefer camera_sensor_timestamp_ms for image-to-image intervals"
                ),
            },
            "can_tx_timing": {
                "file_suffix": "_control_seq.jsonl",
                "phase": "can_tx",
                "units": "ms",
                "movement_write": (
                    "Kvaser write() call/return boundary for every movement frame; "
                    "joystick_deflection is derived from the transmitted payload "
                    "relative to neutral 127"
                ),
                "command_sync_done": (
                    "Kvaser writeSync() completion after a changed-command burst"
                ),
            },
        }
        if extra:
            meta.update(extra)
        with open(self.base + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------- state
    def log_state(self, *, frame_i: int, fsm_state: str, align_sub: str, det_ok: bool,
                  center3d, yaw_deg, yaw_model, width_m, fps,
                  rotation_diagnostics: Optional[dict] = None) -> None:
        """매 프레임 호출. 최신 상태는 항상 갱신하고, 파일 기록만 주기적으로."""
        pos = (None, None, None)
        if center3d is not None:
            try:
                pos = (float(center3d[0]), float(center3d[1]), float(center3d[2]))
            except Exception:
                pos = (None, None, None)

        rot = rotation_diagnostics or {}
        self._latest = {
            "frame_i": frame_i, "fsm_state": fsm_state, "align_sub": align_sub,
            "det_ok": bool(det_ok), "pos": list(pos),
            "yaw_deg": yaw_deg, "yaw_model_raw_deg": yaw_model,
            "width_m": width_m,
            "rotation_diagnostics": dict(rot),
        }
        if not self.enabled or self._csv is None:
            return
        now = time.monotonic()
        if (now - self._last_write) < self._period:
            return
        self._last_write = now
        self._csv.write(
            f"{datetime.now().isoformat(timespec='milliseconds')},{now:.6f},{frame_i},"
            f"{fsm_state},{align_sub},{int(bool(det_ok))},"
            f"{_f(pos[0])},{_f(pos[1])},{_f(pos[2])},"
            f"{_f(yaw_deg,4)},{_f(yaw_model,4)},{_f(width_m,4)},"
            f"{_f(fps,2)},"
            f"{rot.get('command') or ''},{rot.get('mode') or ''},"
            f"{_f(rot.get('stop_measured_rate_deg_s'),4)},"
            f"{_f(rot.get('stop_effective_rate_deg_s'),4)},"
            f"{_f(rot.get('stop_error_deg'),4)},"
            f"{_f(rot.get('coast_deg'),4)},"
            f"{_f(rot.get('coast_peak_abs_deg'),4)}\n")
        self._csv.flush()

    # ------------------------------------------------------ inference timing
    def log_inference_timing(
        self, *, frame_i: int, fsm_state: str, align_sub: str,
        camera_frame_number, camera_sensor_timestamp_ms,
        camera_timestamp_domain, camera_input_host_mono_ms,
        inference_start_host_mono_ms, inference_end_host_mono_ms,
        pose_result_host_mono_ms, inference_ran: bool, model_det_ok: bool,
        pnp_ok: bool, yaw_deg, pos_x_m, pos_z_m, center_bearing_deg,
        pose_src=None, pose_n_used=None, pose_rms_px=None,
        raw_video_frame_index=None,
    ) -> None:
        """Write one timing/pose row for every camera frame, without downsampling."""
        if not self.enabled or self._timing is None:
            return

        try:
            frame_number = int(camera_frame_number)
        except (TypeError, ValueError):
            frame_number = None
        try:
            sensor_ms = float(camera_sensor_timestamp_ms)
        except (TypeError, ValueError):
            sensor_ms = None
        try:
            input_ms = float(camera_input_host_mono_ms)
        except (TypeError, ValueError):
            input_ms = None
        domain = "" if camera_timestamp_domain is None else str(camera_timestamp_domain)
        domain = domain.replace(",", ";")

        frame_delta = None
        if frame_number is not None and self._last_camera_frame_number is not None:
            frame_delta = frame_number - self._last_camera_frame_number
        input_interval_ms = None
        if input_ms is not None and self._last_camera_input_mono_ms is not None:
            input_interval_ms = input_ms - self._last_camera_input_mono_ms
        sensor_interval_ms = None
        if (
            sensor_ms is not None
            and self._last_camera_sensor_timestamp_ms is not None
            and domain == self._last_camera_timestamp_domain
        ):
            candidate = sensor_ms - self._last_camera_sensor_timestamp_ms
            if candidate >= 0.0:
                sensor_interval_ms = candidate

        start_ms = (
            None if inference_start_host_mono_ms is None
            else float(inference_start_host_mono_ms)
        )
        end_ms = (
            None if inference_end_host_mono_ms is None
            else float(inference_end_host_mono_ms)
        )
        pose_ms = (
            None if pose_result_host_mono_ms is None
            else float(pose_result_host_mono_ms)
        )
        model_ms = (
            None if start_ms is None or end_ms is None else end_ms - start_ms
        )
        input_to_inference_ms = (
            None if input_ms is None or end_ms is None else end_ms - input_ms
        )
        input_to_pose_ms = (
            None if input_ms is None or pose_ms is None else pose_ms - input_ms
        )

        try:
            raw_index = int(raw_video_frame_index)
            if raw_index < 0:
                raw_index = None
        except (TypeError, ValueError):
            raw_index = None

        self._timing.write(
            f"{datetime.now().isoformat(timespec='milliseconds')},{frame_i},"
            f"{'' if raw_index is None else raw_index},"
            f"{fsm_state},{align_sub},{'' if frame_number is None else frame_number},"
            f"{'' if frame_delta is None else frame_delta},{_f(sensor_ms,3)},"
            f"{_f(sensor_interval_ms,3)},{domain},{_f(input_ms,3)},"
            f"{_f(input_interval_ms,3)},{int(bool(inference_ran))},"
            f"{_f(start_ms,3)},{_f(end_ms,3)},{_f(model_ms,3)},"
            f"{_f(input_to_inference_ms,3)},{_f(pose_ms,3)},"
            f"{_f(input_to_pose_ms,3)},{int(bool(model_det_ok))},"
            f"{int(bool(pnp_ok))},{'' if pose_src is None else pose_src},"
            f"{'' if pose_n_used is None else int(pose_n_used)},"
            f"{_f(pose_rms_px,3)},{_f(yaw_deg,4)},{_f(pos_x_m,6)},"
            f"{_f(pos_z_m,6)},{_f(center_bearing_deg,4)}\n"
        )
        self._timing_rows += 1
        if self._timing_rows % 30 == 0:
            self._timing.flush()

        self._last_camera_frame_number = frame_number
        self._last_camera_sensor_timestamp_ms = sensor_ms
        self._last_camera_timestamp_domain = domain
        self._last_camera_input_mono_ms = input_ms

    # -------------------------------------------------------------- sequence
    def _emit(self, obj: dict, *, flush: bool = True) -> None:
        if not self.enabled or self._seq is None:
            return
        with self._emit_lock:
            self._seq.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if flush:
                self._seq.flush()

    def _can_tx_writer_loop(self) -> None:
        """Drain CAN events off the real-time owner thread."""
        while (
            not self._can_tx_writer_stop.is_set()
            or not self._can_tx_queue.empty()
        ):
            try:
                rec, flush = self._can_tx_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._emit(rec, flush=flush)
            finally:
                self._can_tx_queue.task_done()

    def step_begin(self, kind: str, cmd: str, **params) -> None:
        """제어 스텝 시작. kind = rotate_in_place | forward | backward | insert_forward"""
        self._step += 1
        rec = {
            "phase": "begin", "step": self._step, "kind": kind, "cmd": cmd,
            "t_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
            "params": params,
            "pallet_state_before": dict(self._latest),
        }
        self._open_step = rec
        self._emit(rec)

    def step_end(self, **result) -> None:
        """직전 step_begin 의 종료. 소요시간은 begin 대비로 계산."""
        if self._open_step is None:
            return
        t = time.monotonic()
        self._emit({
            "phase": "end", "step": self._open_step["step"],
            "kind": self._open_step["kind"], "cmd": self._open_step["cmd"],
            "t_iso": datetime.now().isoformat(timespec="milliseconds"), "t_mono": t,
            "duration_s": round(t - self._open_step["t_mono"], 4),
            "params": self._open_step["params"],
            "result": result,
            "pallet_state_after": dict(self._latest),
        })
        self._open_step = None

    def log_cmd(self, cmd: str) -> None:
        """CAN 명령이 실제로 바뀐 순간(중복 억제 통과분)."""
        self._emit({
            "phase": "cmd", "cmd": cmd,
            "t_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
            "step": self._step,
        })

    def log_can_tx(self, **details) -> None:
        """Record movement-frame writes from the CAN owner thread."""
        now_ns = time.monotonic_ns()
        rec = {
            "phase": "can_tx",
            "t_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": now_ns / 1_000_000_000.0,
            "t_mono_ms": now_ns / 1_000_000.0,
            "step": self._step,
            **details,
        }
        # The CAN thread only timestamps and enqueues. JSON serialization and
        # disk I/O happen on the dedicated trace-writer thread.
        self._can_tx_queue.put(
            (
                rec,
                details.get("event") in {
                    "command_sync_done", "startup_sync_done",
                    "shutdown_sync_done",
                },
            )
        )

    def log_rotation_stop(self, **diagnostics) -> None:
        """Record the exact STOP boundary used for coast/inertia estimation."""
        self._emit({
            "phase": "rotation_stop",
            "t_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
            "step": self._step,
            "diagnostics": diagnostics,
            "pallet_state": dict(self._latest),
        })

    def log_failure(self, state: str, reason: str) -> None:
        """Record a terminal FSM failure even when no control step is open."""
        self._emit({
            "phase": "failure",
            "state": str(state),
            "reason": str(reason),
            "t_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
            "step": self._step,
            "pallet_state": dict(self._latest),
        })

    def log_debug_step(self, action: str, state: str, **details) -> None:
        """Record manual stage-gate pause/resume actions."""
        self._emit({
            "phase": "debug_step",
            "action": str(action),
            "state": str(state),
            "t_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": time.monotonic(),
            "step": self._step,
            **details,
        })

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        self._can_tx_writer_stop.set()
        if self._can_tx_writer_thread is not None:
            self._can_tx_writer_thread.join()
            self._can_tx_writer_thread = None
        for f in (self._csv, self._timing, self._seq):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._csv = self._timing = self._seq = None
