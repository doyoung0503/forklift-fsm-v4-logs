from __future__ import annotations

import time
from statistics import median
from typing import List, Optional, Tuple

import calib.config as cfg
from calib.config import (
    ALIGN_BAND_M, ALIGN_DIST_M, ALIGN_ROTATE_TIMEOUT_SEC,
    COLOR_ALERT, COLOR_META, COLOR_STATUS_OK, COLOR_STATUS_TRK,
    DIST_STABLE_SPREAD_M, INFER_STABLE_FRAMES, LATERAL_STABLE_SPREAD_M,
    LATERAL_TURN_DEG, OFF_TOL_M, STOP_SEC,
    USE_PIECEWISE_FWD_FIT, YAW_STABLE_SPREAD_DEG, YAW_TOL_DEG,
)
from calib.motion_models import fwd_sec_from_offset_piecewise

from .commands import CommandExecutor
from .status_helper import StatusHelper
from .utils import SimpleTimer


# 카메라 추론이 필요한 정지 체크포인트와 live-PnP 회전 상태.
INFERENCE_CHECKPOINT_SUBS = frozenset({
    "DIST_CHECK", "POST_DISTANCE_INFER", "POST_LATERAL_INFER",
    "SNAPSHOT_YAW_ROTATE", "LATERAL_PRE_ROTATE", "LATERAL_FACE_ROTATE",
})

# 저장된 스냅샷/타이머만 쓰며 카메라 추론을 금지하는 상태.
OPEN_LOOP_MOTION_SUBS = frozenset({
    "DIST_MOVE", "LATERAL_FORWARD", "INSERT_FORWARD", "READY_TO_DONE",
})


def _wrap_to_180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _seconds_for_distance(distance_m: float) -> float:
    """저장된 이동 거리(m)를 CAN 전진/후진 명령 시간으로 환산한다."""
    distance_m = abs(float(distance_m))
    if USE_PIECEWISE_FWD_FIT:
        return max(0.1, float(fwd_sec_from_offset_piecewise(distance_m)))
    speed_mps = max(1e-3, float(getattr(cfg, "INSERT_FWD_MPS", 0.25)))
    return max(0.1, distance_m / speed_mps)


def _seconds_for_insertion(dist_z: Optional[float]) -> Optional[float]:
    if dist_z is None:
        return None
    pocket_m = float(getattr(cfg, "PALLET_POCKET_M", 0.0))
    speed_mps = max(1e-3, float(getattr(cfg, "INSERT_FWD_MPS", 0.25)))
    min_sec = float(getattr(cfg, "INS_FWD_MIN_SEC", 0.5))
    max_sec = float(getattr(cfg, "INS_FWD_MAX_SEC", 10.0))
    seconds = max(0.0, float(dist_z) + pocket_m) / speed_mps
    return max(min_sec, min(max_sec, seconds))


class AlignMachine:
    """정지 추론 스냅샷과 개방루프 동작을 분리한 ALIGN 서브 FSM.

    DIST_CHECK -> DIST_MOVE(추론 없음) -> POST_DISTANCE_INFER ->
    LATERAL_PRE_ROTATE -> LATERAL_FORWARD -> LATERAL_FACE_ROTATE ->
    POST_LATERAL_INFER(검증, 필요 시 정렬 반복) -> INSERT_FORWARD.
    lateral이 허용치 이내면 SNAPSHOT_YAW_ROTATE 또는 삽입으로 바로 진행한다.
    """

    # Subclasses can tune their initial stopped-distance target without
    # changing the behaviour of the existing v1 controller.
    distance_target_m = ALIGN_DIST_M
    distance_band_m = ALIGN_BAND_M

    def __init__(self, execu: CommandExecutor, status: StatusHelper, tracer=None):
        self.execu = execu
        self.status = status
        self.tracer = tracer
        self.sub = "DIST_CHECK"

        self._timer = SimpleTimer()
        self._interlock_active = False
        self._after_interlock_sub: Optional[str] = None

        self._motion_code: Optional[str] = None
        self._motion_sec = 0.0
        self._motion_deadline_ts: Optional[float] = None

        self._rotation_target_yaw_deg: Optional[float] = None
        self._rotation_turn_deg: Optional[float] = None
        self._rotation_deadline_ts: Optional[float] = None

        self._snapshot_dist_z: Optional[float] = None
        self._snapshot_yaw: Optional[float] = None
        self._snapshot_ox: Optional[float] = None
        self._latest_dist_z: Optional[float] = None

        self._lateral_side = 0  # +1=우측 이동, -1=좌측 이동
        self._lateral_m = 0.0
        self._lateral_sec = 0.0

        self._insert_sec: Optional[float] = None
        self._insert_deadline_ts: Optional[float] = None

        # 정지 상태에서 연속 추론값의 안정성을 검사하기 위한 버퍼.
        self._dist_samples: List[float] = []
        self._post_lateral_samples: List[Tuple[float, float]] = []

    @property
    def skip_detection(self) -> bool:
        """현재 모델 추론을 완전히 건너뛰어야 하는지 반환한다."""
        return self._interlock_active or self.sub in OPEN_LOOP_MOTION_SUBS

    def _exec(self, code: str) -> None:
        self.execu.exec(code)

    def _step_begin(self, kind: str, cmd: str, **params) -> None:
        if self.tracer is not None:
            self.tracer.step_begin(kind, cmd, **params)

    def _step_end(self, **result) -> None:
        if self.tracer is not None:
            self.tracer.step_end(**result)

    def reset(self, sub: str = "DIST_CHECK", clear_offset_done: bool = False) -> None:
        del clear_offset_done  # 기존 호출부 호환
        self.sub = sub
        self._interlock_active = False
        self._after_interlock_sub = None
        self._motion_code = None
        self._motion_sec = 0.0
        self._motion_deadline_ts = None
        self._rotation_target_yaw_deg = None
        self._rotation_turn_deg = None
        self._rotation_deadline_ts = None
        self._snapshot_dist_z = None
        self._snapshot_yaw = None
        self._snapshot_ox = None
        self._latest_dist_z = None
        self._lateral_side = 0
        self._lateral_m = 0.0
        self._lateral_sec = 0.0
        self._insert_sec = None
        self._insert_deadline_ts = None
        self._dist_samples.clear()
        self._post_lateral_samples.clear()

    def _clear_inference_samples(self, sub: Optional[str] = None) -> None:
        if sub in (None, "DIST_CHECK"):
            self._dist_samples.clear()
        if sub in (None, "POST_LATERAL_INFER"):
            self._post_lateral_samples.clear()

    def _stable_distance(self, value: float) -> Tuple[Optional[float], str]:
        """안정적인 거리 5개 이상이면 중앙값을, 아니면 대기 사유를 반환한다."""
        required = max(5, int(INFER_STABLE_FRAMES))
        self._dist_samples.append(float(value))
        if len(self._dist_samples) < required:
            return None, f"{len(self._dist_samples)}/{required}"

        window = self._dist_samples[-required:]
        spread = max(window) - min(window)
        if spread > DIST_STABLE_SPREAD_M:
            # 튄 값이 다음 판정에 계속 남지 않도록 현재값부터 다시 누적한다.
            self._dist_samples[:] = [float(value)]
            return None, f"불안정 spread={spread:.3f}m -> 1/{required} 재시작"

        result = float(median(window))
        self._dist_samples.clear()
        return result, f"stable {required}/{required}, spread={spread:.3f}m"

    def _stable_post_lateral(
        self, yaw: float, ox: float,
    ) -> Tuple[Optional[Tuple[float, float]], str]:
        """안정적인 yaw/lateral 5개 이상이면 각 중앙값을 반환한다."""
        required = max(5, int(INFER_STABLE_FRAMES))
        sample = (float(yaw), float(ox))
        self._post_lateral_samples.append(sample)
        if len(self._post_lateral_samples) < required:
            return None, f"{len(self._post_lateral_samples)}/{required}"

        window = self._post_lateral_samples[-required:]
        yaws = [item[0] for item in window]
        offsets = [item[1] for item in window]
        yaw_spread = max(yaws) - min(yaws)
        lateral_spread = max(offsets) - min(offsets)
        if yaw_spread > YAW_STABLE_SPREAD_DEG or lateral_spread > LATERAL_STABLE_SPREAD_M:
            self._post_lateral_samples[:] = [sample]
            return None, (
                f"불안정 yawΔ={yaw_spread:.2f}°, latΔ={lateral_spread:.3f}m "
                f"-> 1/{required} 재시작"
            )

        result = (float(median(yaws)), float(median(offsets)))
        self._post_lateral_samples.clear()
        return result, (
            f"stable {required}/{required}, yawΔ={yaw_spread:.2f}°, "
            f"latΔ={lateral_spread:.3f}m"
        )

    def _start_interlock_then(self, next_sub: str) -> None:
        self._interlock_active = True
        self._after_interlock_sub = next_sub
        self._exec("STOP")
        self._timer.start(STOP_SEC)
        self.status.start_timed("STOP", STOP_SEC)

    def _handle_interlock(self, lines: List[Tuple[str, tuple]]) -> bool:
        if not self._interlock_active:
            return False
        self._exec("STOP")
        lines.append((f"[ALIGN] STOP 인터록 -> {self._after_interlock_sub}", COLOR_META))
        if not self._timer.active():
            self._interlock_active = False
            if self._after_interlock_sub is not None:
                self.sub = self._after_interlock_sub
            self._after_interlock_sub = None
        # 다음 프레임부터 새 상태의 추론 정책을 적용한다.
        return True

    def _start_timed_motion(self, sub: str, code: str, seconds: float) -> None:
        self.sub = sub
        self._motion_code = code
        self._motion_sec = max(0.1, float(seconds))
        self._motion_deadline_ts = time.time() + self._motion_sec
        self._exec(code)
        self.status.start_timed(code, self._motion_sec)

    def _begin_rotation(self, sub: str, turn_deg: float, yaw_deg: float) -> None:
        turn_deg = _wrap_to_180(turn_deg)
        self.sub = sub
        self._rotation_turn_deg = turn_deg
        self._rotation_target_yaw_deg = _wrap_to_180(float(yaw_deg) - turn_deg)
        self._rotation_deadline_ts = time.time() + ALIGN_ROTATE_TIMEOUT_SEC
        code = "ROT_RIGHT" if turn_deg > 0.0 else "ROT_LEFT"
        self._exec(code)
        self.status.start_until(code, "|yaw_error|", abs(turn_deg), YAW_TOL_DEG)

    def _rotation_step(
        self, yaw_deg: Optional[float], next_sub: str,
        lines: List[Tuple[str, tuple]],
    ) -> None:
        target = self._rotation_target_yaw_deg
        if target is None or yaw_deg is None:
            self._step_end(error="pnp_yaw_unavailable")
            self._start_interlock_then("POST_LATERAL_INFER")
            lines.append((f"[{self.sub}] PnP yaw 유실 -> 정지 후 재추론", COLOR_ALERT))
            return

        error = _wrap_to_180(float(yaw_deg) - target)
        code = "ROT_RIGHT" if error > 0.0 else "ROT_LEFT"
        reached = abs(error) <= YAW_TOL_DEG
        self.status.start_until(code, "|yaw_error|", abs(error), YAW_TOL_DEG)

        if reached:
            finished_sub = self.sub
            self._step_end(final_yaw_error_deg=error, target_yaw_deg=target)
            self._rotation_target_yaw_deg = None
            self._rotation_turn_deg = None
            self._rotation_deadline_ts = None
            self._start_interlock_then(next_sub)
            lines.append((f"[{finished_sub}->{next_sub}] PnP yaw error={error:+.2f}°", COLOR_META))
            return

        if self._rotation_deadline_ts is not None and time.time() >= self._rotation_deadline_ts:
            finished_sub = self.sub
            self._step_end(timeout=True, final_yaw_error_deg=error, target_yaw_deg=target)
            self._rotation_target_yaw_deg = None
            self._rotation_turn_deg = None
            self._rotation_deadline_ts = None
            self._start_interlock_then("POST_LATERAL_INFER")
            lines.append((f"[{finished_sub}] 회전 timeout -> 정지 후 재추론", COLOR_ALERT))
            return

        self._exec(code)
        lines.append((f"[{self.sub}] PnP yaw error={error:+.2f}°", COLOR_STATUS_TRK))

    def _prepare_insertion(self, lines: List[Tuple[str, tuple]]) -> None:
        self._insert_sec = _seconds_for_insertion(self._latest_dist_z)
        if self._insert_sec is None or self._insert_sec <= 0.0:
            self.sub = "READY_TO_DONE"
            self._exec("STOP")
            lines.append(("[ALIGN->READY_TO_DONE] 삽입 거리 없음", COLOR_STATUS_OK))
            return
        self._start_interlock_then("INSERT_FORWARD")
        lines.append((f"[ALIGN->INSERT_FORWARD] FWD_SEC={self._insert_sec:.2f}s", COLOR_META))

    def _apply_alignment_snapshot(
        self, yaw: float, ox: float,
        lines: List[Tuple[str, tuple]],
    ) -> None:
        """한 번 받은 yaw/lateral 결과를 고정하고 이후 동작은 비전 없이 수행한다."""
        self._snapshot_yaw, self._snapshot_ox = float(yaw), float(ox)

        if abs(ox) > OFF_TOL_M:
            self._lateral_side = +1 if ox > 0.0 else -1
            self._lateral_m = abs(float(ox))
            self._lateral_sec = _seconds_for_distance(self._lateral_m)
            # 팔레트 yaw를 먼저 보상하고 lateral 이동 방향으로 ±90° 더 회전한다.
            target = _wrap_to_180(float(yaw) + self._lateral_side * LATERAL_TURN_DEG)
            self._step_begin(
                "lateral_pre_rotate", "ROT_RIGHT" if target > 0.0 else "ROT_LEFT",
                driver="live_pnp_yaw", snapshot_yaw_deg=yaw,
                offset_x_m=ox, requested_turn_deg=target,
            )
            if abs(target) <= YAW_TOL_DEG:
                self._step_end(skipped=True, requested_turn_deg=target)
                self._start_interlock_then("LATERAL_FORWARD")
            else:
                self._begin_rotation("LATERAL_PRE_ROTATE", target, yaw)
            lines.append((
                f"[정렬 snapshot] yaw={yaw:+.2f}°, lateral={ox:+.3f}m "
                f"-> 측면회전 {target:+.2f}°", COLOR_META,
            ))
            return

        if abs(yaw) > YAW_TOL_DEG:
            target = _wrap_to_180(yaw)
            self._step_begin(
                "snapshot_yaw_rotate", "ROT_RIGHT" if target > 0.0 else "ROT_LEFT",
                driver="live_pnp_yaw", snapshot_yaw_deg=yaw,
                requested_turn_deg=target,
            )
            self._begin_rotation("SNAPSHOT_YAW_ROTATE", target, yaw)
            lines.append((f"[정렬 snapshot] lateral 허용, yaw 회전 {target:+.2f}°", COLOR_META))
            return

        lines.append((
            f"[정렬 snapshot] OK: yaw={yaw:+.2f}°, lateral={ox:+.3f}m "
            f"(tol={OFF_TOL_M:.2f}m)", COLOR_STATUS_OK,
        ))
        self._prepare_insertion(lines)

    def step(
        self, det_ok: bool, detected_length: Optional[float],
        dist_z: Optional[float], yaw: Optional[float], ox: Optional[float],
    ) -> List[Tuple[str, tuple]]:
        del detected_length
        lines: List[Tuple[str, tuple]] = []

        if self._handle_interlock(lines):
            return lines

        # 추론 체크포인트에서는 유효한 결과가 올 때까지 반드시 STOP한다.
        if self.sub in INFERENCE_CHECKPOINT_SUBS and not det_ok:
            self._clear_inference_samples(self.sub)
            self._exec("STOP")
            self.status.start_timed("STOP", 0.0)
            lines.append((f"[{self.sub}] 유효 추론이 끊겨 안정화 누적 초기화 (STOP)", COLOR_ALERT))
            return lines

        # 이동 직전 거리값이 연속 5회 이상 안정적일 때 중앙값을 확정한다.
        if self.sub == "DIST_CHECK":
            self._exec("STOP")
            if dist_z is None:
                self._clear_inference_samples("DIST_CHECK")
                lines.append(("[DIST_CHECK] distance N/A (STOP)", COLOR_ALERT))
                return lines

            stable_dist_z, stability = self._stable_distance(float(dist_z))
            if stable_dist_z is None:
                lines.append((f"[DIST_CHECK] 거리 안정화 대기: {stability} (STOP)", COLOR_STATUS_TRK))
                return lines

            self._snapshot_dist_z = self._latest_dist_z = stable_dist_z
            target_dist_m = float(self.distance_target_m)
            distance_band_m = float(self.distance_band_m)
            distance_error = stable_dist_z - target_dist_m
            if abs(distance_error) <= distance_band_m:
                self._start_interlock_then("POST_DISTANCE_INFER")
                lines.append((
                    f"[DIST_CHECK] z median={stable_dist_z:.3f}m ({stability}), "
                    f"밴드 OK -> 정지 후 추론",
                    COLOR_META,
                ))
                return lines

            code = "FWD" if distance_error > 0.0 else "BACK"
            seconds = _seconds_for_distance(abs(distance_error))
            self._step_begin(
                "distance_move", code, driver="distance_snapshot_timer",
                snapshot_dist_z_m=stable_dist_z, target_dist_m=target_dist_m,
                distance_error_m=distance_error, duration_sec=seconds,
            )
            self._start_timed_motion("DIST_MOVE", code, seconds)
            lines.append((
                f"[DIST_CHECK->DIST_MOVE] z median={stable_dist_z:.3f}m ({stability}), "
                f"{code} {seconds:.2f}s; 추론 OFF",
                COLOR_META,
            ))
            return lines

        # 현재 프레임 dist_z를 절대 참조하지 않고 저장값+타이머만 쓴다.
        if self.sub == "DIST_MOVE":
            if self._motion_deadline_ts is None or self._motion_code is None:
                self._start_interlock_then("POST_DISTANCE_INFER")
                lines.append(("[DIST_MOVE] 저장된 이동 정보 없음 -> 재추론", COLOR_ALERT))
                return lines
            if time.time() < self._motion_deadline_ts:
                self._exec(self._motion_code)
                remain = max(0.0, self._motion_deadline_ts - time.time())
                lines.append((
                    f"[DIST_MOVE] {self._motion_code}, 잔여 {remain:.2f}s (실시간 추론 없음)",
                    COLOR_STATUS_TRK,
                ))
                return lines
            self._step_end(snapshot_dist_z_m=self._snapshot_dist_z, duration_sec=self._motion_sec)
            self._motion_deadline_ts = None
            self._motion_code = None
            self._start_interlock_then("POST_DISTANCE_INFER")
            lines.append(("[DIST_MOVE->POST_DISTANCE_INFER] STOP 후 1회 추론", COLOR_META))
            return lines

        # 거리 이동 후 정확히 한 유효 추론 프레임으로 yaw/lateral 체인을 계획한다.
        if self.sub == "POST_DISTANCE_INFER":
            self._exec("STOP")
            if yaw is None or ox is None:
                lines.append(("[POST_DISTANCE_INFER] yaw/lateral 결과 대기", COLOR_ALERT))
                return lines
            if dist_z is not None:
                self._latest_dist_z = float(dist_z)
            self._apply_alignment_snapshot(float(yaw), float(ox), lines)
            return lines

        if self.sub == "SNAPSHOT_YAW_ROTATE":
            self._rotation_step(yaw, "POST_LATERAL_INFER", lines)
            return lines

        if self.sub == "LATERAL_PRE_ROTATE":
            self._rotation_step(yaw, "LATERAL_FORWARD", lines)
            return lines

        if self.sub == "LATERAL_FORWARD":
            if self._motion_deadline_ts is None:
                self._step_begin(
                    "lateral_forward", "FWD", driver="lateral_snapshot_timer",
                    lateral_m=self._lateral_m, duration_sec=self._lateral_sec,
                )
                self._start_timed_motion("LATERAL_FORWARD", "FWD", self._lateral_sec)
                lines.append((
                    f"[LATERAL_FORWARD] lateral={self._lateral_m:.3f}m, "
                    f"FWD {self._lateral_sec:.2f}s", COLOR_META,
                ))
                return lines
            if time.time() < self._motion_deadline_ts:
                self._exec("FWD")
                remain = max(0.0, self._motion_deadline_ts - time.time())
                lines.append((f"[LATERAL_FORWARD] 잔여 {remain:.2f}s (추론 없음)", COLOR_STATUS_TRK))
                return lines
            self._step_end(lateral_m=self._lateral_m, duration_sec=self._lateral_sec)
            self._motion_deadline_ts = None
            self._motion_code = None
            self._rotation_turn_deg = -self._lateral_side * LATERAL_TURN_DEG
            self._rotation_target_yaw_deg = None
            self._rotation_deadline_ts = None
            self._start_interlock_then("LATERAL_FACE_ROTATE")
            lines.append(("[LATERAL_FORWARD->LATERAL_FACE_ROTATE] 전면 방향 90° 회전", COLOR_META))
            return lines

        if self.sub == "LATERAL_FACE_ROTATE":
            if yaw is None:
                self._exec("STOP")
                lines.append(("[LATERAL_FACE_ROTATE] PnP yaw 대기", COLOR_ALERT))
                return lines
            if self._rotation_target_yaw_deg is None:
                target = -self._lateral_side * LATERAL_TURN_DEG
                self._step_begin(
                    "lateral_face_rotate", "ROT_RIGHT" if target > 0.0 else "ROT_LEFT",
                    driver="live_pnp_yaw", requested_turn_deg=target,
                )
                self._begin_rotation("LATERAL_FACE_ROTATE", target, float(yaw))
            self._rotation_step(yaw, "POST_LATERAL_INFER", lines)
            return lines

        # 전면 복귀 후 yaw/lateral이 연속 5회 이상 안정적일 때 중앙값으로 검증한다.
        if self.sub == "POST_LATERAL_INFER":
            self._exec("STOP")
            if yaw is None or ox is None:
                self._clear_inference_samples("POST_LATERAL_INFER")
                lines.append(("[POST_LATERAL_INFER] 값 유실, 안정화 누적 초기화 (STOP)", COLOR_ALERT))
                return lines
            if dist_z is not None:
                self._latest_dist_z = float(dist_z)

            stable_pose, stability = self._stable_post_lateral(float(yaw), float(ox))
            if stable_pose is None:
                lines.append((f"[POST_LATERAL_INFER] 자세 안정화 대기: {stability} (STOP)", COLOR_STATUS_TRK))
                return lines

            stable_yaw, stable_ox = stable_pose
            lines.append((
                f"[POST_LATERAL_INFER] median yaw={stable_yaw:+.2f}°, "
                f"lateral={stable_ox:+.3f}m ({stability})",
                COLOR_META,
            ))
            self._apply_alignment_snapshot(stable_yaw, stable_ox, lines)
            return lines

        if self.sub == "INSERT_FORWARD":
            if self._insert_sec is None:
                self.sub = "READY_TO_DONE"
                self._exec("STOP")
                lines.append(("[INSERT_FORWARD->READY_TO_DONE] 삽입 시간 없음", COLOR_ALERT))
                return lines
            if self._insert_deadline_ts is None:
                self._insert_deadline_ts = time.time() + self._insert_sec
                self._step_begin(
                    "insert_forward", "FWD", driver="snapshot_distance_timer",
                    duration_sec=self._insert_sec,
                )
                self._exec("FWD")
                self.status.start_timed("FWD", self._insert_sec)
                lines.append((f"[INSERT_FORWARD] FWD {self._insert_sec:.2f}s (추론 없음)", COLOR_META))
                return lines
            if time.time() < self._insert_deadline_ts:
                self._exec("FWD")
                remain = max(0.0, self._insert_deadline_ts - time.time())
                lines.append((f"[INSERT_FORWARD] 잔여 {remain:.2f}s", COLOR_STATUS_TRK))
                return lines
            self._step_end(duration_sec=self._insert_sec)
            self._insert_deadline_ts = None
            self._start_interlock_then("READY_TO_DONE")
            lines.append(("[INSERT_FORWARD->READY_TO_DONE] 삽입 종료", COLOR_META))
            return lines

        if self.sub == "READY_TO_DONE":
            self._exec("STOP")
            lines.append(("[READY_TO_DONE] 정렬 완료", COLOR_STATUS_OK))
            return lines

        self._exec("STOP")
        lines.append((f"[ALIGN] 알 수 없는 상태 {self.sub} (STOP)", COLOR_ALERT))
        return lines
