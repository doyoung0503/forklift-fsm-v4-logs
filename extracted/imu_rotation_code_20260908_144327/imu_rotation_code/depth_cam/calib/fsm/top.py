# calib/fsm/top.py
# -----------------------------------------------------------------------------
# Top-level FSM for pallet-front alignment using RGB-D perception.
# 상태 요약:
#   SEARCH → DETECTED → INITIAL_FACE_STOP → INITIAL_FACE_ROTATE
#          → INITIAL_FACE_SETTLE → ALIGN(DIST_CHECK) → DONE
# - SEARCH: PnP 성공까지 제자리 회전하며 실시간 탐지, 성공 즉시 STOP
# - DETECTED: 최초 탐지의 팔레트 중심 방위각을 고정하고 비전 추론을 중지
# - INITIAL_FACE_*: 정지 후 IMU 누적각만으로 팔레트 중심을 바라보도록 회전
# - ALIGN: 세부 정렬 하위 상태기(거리/야우/오프셋 보정)
# - RECOVER: 시야 확보/전면 마스크 폭 확장 동작 후 HOLD
# - CHECK: 최종 허용 오차(yaw/offset) 안정화 검사
# - DONE: 정렬 완료 상태 유지(정지)
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional, List, Tuple
import time

from calib.config import (
    COLOR_META, COLOR_ALERT, COLOR_STATUS_OK, COLOR_STATUS_TRK,  # HUD 색상
    YAW_TOL_DEG, OFF_TOL_M,              # 최종 허용 오차
    ALIGN_DIST_M, ALIGN_BAND_M,          # 거리 밴드(목표 거리 ± 밴드)
    CMD_STABLE_THR,                      # 안정화에 필요한 연속 프레임 수
    INITIAL_FACE_STOP_SEC,
    INITIAL_FACE_SETTLE_SEC,
    INITIAL_FACE_ROTATE_TIMEOUT_SEC,
)
from .commands import CommandExecutor      # CAN 제어 명령 실행기
from .status_helper import StatusHelper    # HUD용 현재 명령/진행도 관리
from .utils import Stabilizer, within_band # 안정화 카운터, 밴드 체크
from .align import AlignMachine            # 정렬 하위 상태기
from .recover import RecoverMachine        # 리커버(시야확보) 하위 상태기


class CalibrationFSM:
    """
    Top-level:
      SEARCH → DETECTED → INITIAL_FACE_STOP → INITIAL_FACE_ROTATE
             → INITIAL_FACE_SETTLE → ALIGN(DIST_CHECK) → DONE
    - SEARCH: PnP 성공까지 제자리 회전 탐색, 성공 즉시 정지
    - DETECTED: 최초 팔레트 중심 방위각 스냅샷 확정
    - INITIAL_FACE_*: 추가 비전 없이 IMU 누적각으로 팔레트 중심 방향 정렬
    - ALIGN: 거리/야우/오프셋을 순차적으로 정렬
    - RECOVER: 전면 폭 부족 시 회전/대기 등으로 시야 확보 후 HOLD
    - CHECK: yaw/offset 최종 안정화 검사(CMD_STABLE_THR 프레임)
    - DONE: 정렬 완료(정지 유지)
    """
    def __init__(self, tracer=None):
        # FSM 현재 상태
        self.state: str = "SEARCH"
        self.tracer = tracer

        # 하위 모듈: 제어, HUD, 서브 상태기
        self.execu = CommandExecutor(tracer=tracer)
        self.status = StatusHelper()
        self.align = AlignMachine(self.execu, self.status, tracer=tracer)
        self.recover = RecoverMachine(self.execu, self.status)

        # CHECK 안정화용 카운터
        self._stb = Stabilizer(CMD_STABLE_THR)

        # STOP 연속 송신 억제를 위한 마지막 전송 기록(0.2s 스로틀에 사용)
        self._last_cmd: Optional[str] = None
        self._last_ts: float = 0.0

        # 최초 탐지 스냅샷 기반 IMU 정면 정렬 상태
        self._initial_face_yaw_deg: Optional[float] = None
        self._initial_face_offset_x_m: Optional[float] = None
        self._initial_face_dist_z_m: Optional[float] = None
        self._initial_face_target_deg: Optional[float] = None
        self._initial_face_rel_ref: Optional[float] = None
        self._initial_face_deadline: float = 0.0
        self._initial_face_rotate_deadline: float = 0.0

    # -------------------------------------------------------------------------
    # 내부 유틸: 검사/대기 프레임 동안 STOP을 명시적으로 송신
    # - 같은 프레임/아주 짧은 시간 간격으로 STOP이 과도 송신되지 않도록 0.2s 스로틀
    # - HUD도 함께 STOP으로 업데이트하여 표시-제어 일치 보장
    # -------------------------------------------------------------------------
    def _ensure_stop(self) -> None:
        now = time.time()
        if (self.execu.last_cmd != "STOP" or self._last_cmd != "STOP"
                or (now - self._last_ts) > 0.2):
            self.execu.exec("STOP")
            self.status.start_timed("STOP", 0.0)
            self._last_cmd, self._last_ts = "STOP", now

    @staticmethod
    def _wrap_to_180(deg: float) -> float:
        return (float(deg) + 180.0) % 360.0 - 180.0

    def _clear_initial_face(self) -> None:
        self._initial_face_yaw_deg = None
        self._initial_face_offset_x_m = None
        self._initial_face_dist_z_m = None
        self._initial_face_target_deg = None
        self._initial_face_rel_ref = None
        self._initial_face_deadline = 0.0
        self._initial_face_rotate_deadline = 0.0

    # 외부(HUD) 조회용: 현재 명령 상태(라벨/진행도) 객체 — 함수형 API(하위 호환)
    def get_command_status(self):
        return self.status.cmd_status

    # 속성형 API — main_rec.py에서 fsm.cmd_status로 직접 접근 가능
    @property
    def cmd_status(self):
        return self.status.cmd_status

    @property
    def align_sub(self) -> str:
        """로깅/HUD 용 현재 정렬 하위 상태."""
        return self.align.sub

    @property
    def skip_detection(self) -> bool:
        """초기 정렬 및 ALIGN의 모든 개방루프 동작 중 추론을 끈다.

        ALIGN에서는 DIST_CHECK/POST_DISTANCE_INFER/POST_LATERAL_INFER처럼
        정지 상태의 스냅샷 체크포인트에서만 추론이 켜진다.
        """
        initial_face_states = {
            "DETECTED", "INITIAL_FACE_STOP",
            "INITIAL_FACE_ROTATE", "INITIAL_FACE_SETTLE",
        }
        return self.state == "DONE" or self.state in initial_face_states or (
            self.state == "ALIGN" and self.align.skip_detection
        )

    # -------------------------------------------------------------------------
    # 주 상태 전이 함수
    # - det_ok            : front 탐지 여부
    # - detected_length   : 전면(마스크/박스)의 가로 길이(시야 확보 판단)
    # - dist_z            : 목표물과의 Z 거리
    # - yaw_smooth        : 평활화된 yaw(법선 기반, 팔레트 면 기준 정렬용)
    # - offset_smooth     : (offset_x, offset_y, ...) 튜플(여기선 x만 사용)
    # - rel_yaw           : ★ IMU 기반 상대 yaw(gyro-Y 적분, deg) — 회전 종료 조건/진행률
    # - target_bearing_deg: 모델의 팔레트 중심 키포인트가 이루는 수평 방위각
    # 반환: HUD에 표시할 로그 텍스트/색상 리스트
    # -------------------------------------------------------------------------
    def step(self,
             det_ok: bool,
             detected_length: Optional[float],
             dist_z: Optional[float],
             yaw_smooth: Optional[float],
             offset_smooth: Optional[tuple],
             rel_yaw: Optional[float] = None,
             target_bearing_deg: Optional[float] = None,
             ) -> List[Tuple[str, tuple]]:

        lines: List[Tuple[str, tuple]] = []

        # offset_smooth가 (ox, oy, ...) 형태일 수 있으므로 안전하게 추출
        ox: Optional[float] = None
        if offset_smooth is not None:
            try:
                ox = float(offset_smooth[0])
            except Exception:
                ox = None

        # 허용 오차 판정(최종 체크/결정에 사용)
        yaw_ok = (yaw_smooth is not None) and (abs(yaw_smooth) <= YAW_TOL_DEG)
        off_ok = (ox is not None) and (abs(ox) <= OFF_TOL_M)
        band_ok = within_band(dist_z, ALIGN_DIST_M, ALIGN_BAND_M)

        # HUD의 until 진행도(예: |yaw|/tol, |offset|/tol)를 최신으로 갱신
        self.status.update_until_metric(yaw_smooth, ox)

        # ------------------------------ SEARCH ------------------------------
        if self.state == "SEARCH":
            # 서브 상태기 초기화(탐지 유무와 관계 없이 SEARCH 진입 시 리셋)
            # ★ 여기서만 좌우(offset) 보정 1회 권한을 되돌린다 = '정렬 프레임워크 최초 실행'
            self.align.reset("DIST_CHECK", clear_offset_done=True)
            self.recover.reset()

            if (det_ok and target_bearing_deg is not None and rel_yaw is not None):
                # 팔레트 면이나 전면 사각형 중심이 아니라 모델의 중심 키포인트(8번)가
                # 만드는 수평 방위각을 고정한다. 음수=좌회전, 양수=우회전이다.
                bearing_deg = float(target_bearing_deg)
                self._initial_face_yaw_deg = (
                    None if yaw_smooth is None else float(yaw_smooth)
                )
                self._initial_face_offset_x_m = float(ox)
                self._initial_face_dist_z_m = float(dist_z)
                self._initial_face_target_deg = self._wrap_to_180(bearing_deg)
                self._initial_face_rel_ref = None  # 실제 회전 시작 직전에 다시 잡는다.
                self._initial_face_deadline = time.time() + INITIAL_FACE_STOP_SEC
                self.state = "DETECTED"
                self._ensure_stop()
                yaw_text = "N/A" if yaw_smooth is None else f"{yaw_smooth:+.2f}°"
                lines.append((
                    f"[SEARCH→DETECTED] pallet bearing={bearing_deg:+.2f}° "
                    f"(x={ox:+.3f}m, z={dist_z:.3f}m, surface yaw={yaw_text})",
                    COLOR_META,
                ))
            elif det_ok:
                self._ensure_stop()
                lines.append(("[SEARCH] 탐지는 됐지만 중심 키포인트/IMU 값 대기", COLOR_ALERT))
            else:
                # PnP가 성공할 때까지 같은 방향으로 제자리 회전하며 실시간 탐지한다.
                code = "ROT_LEFT" if self.execu.last_dir > 0 else "ROT_RIGHT"
                self.execu.exec(code)
                self.status.start_timed(code, 0.0)
                self._last_cmd, self._last_ts = code, time.time()
                lines.append((f"[SEARCH] PnP 탐지 대기: {code} 제자리 회전", COLOR_STATUS_TRK))
            return lines

        # ----------------------------- DETECTED -----------------------------
        if self.state == "DETECTED":
            # 현재 프레임 탐지는 무시하고 최초 중심 방위각 스냅샷만 사용한다.
            self._ensure_stop()
            if self._initial_face_target_deg is None:
                self.state = "SEARCH"
                self._clear_initial_face()
                lines.append(("[DETECTED→SEARCH] 초기 yaw 스냅샷 없음", COLOR_ALERT))
                return lines
            self.state = "INITIAL_FACE_STOP"
            self.status.start_timed("STOP", max(0.0, self._initial_face_deadline - time.time()))
            lines.append(("[DETECTED→INITIAL_FACE_STOP] 추가 탐지 중지", COLOR_META))
            return lines

        # ------------------------ INITIAL_FACE_STOP ------------------------
        if self.state == "INITIAL_FACE_STOP":
            self._ensure_stop()
            remain = max(0.0, self._initial_face_deadline - time.time())
            self.status.start_timed("STOP", remain)
            if remain > 0.0:
                lines.append((f"[INITIAL_FACE_STOP] 정지 안정화 ({remain:.1f}s)", COLOR_STATUS_TRK))
                return lines

            target = self._initial_face_target_deg
            if target is None or rel_yaw is None:
                self.state = "SEARCH"
                self._clear_initial_face()
                lines.append(("[INITIAL_FACE_STOP→SEARCH] 초기 yaw/IMU 값 없음", COLOR_ALERT))
                return lines

            if abs(target) <= YAW_TOL_DEG:
                self.state = "INITIAL_FACE_SETTLE"
                self._initial_face_deadline = time.time() + INITIAL_FACE_SETTLE_SEC
                lines.append(("[INITIAL_FACE_STOP→INITIAL_FACE_SETTLE] 초기 yaw 허용범위", COLOR_META))
                return lines

            self._initial_face_rel_ref = float(rel_yaw)
            self._initial_face_rotate_deadline = time.time() + INITIAL_FACE_ROTATE_TIMEOUT_SEC
            self.state = "INITIAL_FACE_ROTATE"
            code = "ROT_RIGHT" if target > 0.0 else "ROT_LEFT"
            self.execu.exec(code)
            self.status.start_until(code, "rel_yaw", 0.0, target)
            if self.tracer is not None:
                self.tracer.step_begin(
                    "initial_face_rotate", code,
                    driver="initial_pallet_bearing_then_imu",
                    surface_yaw_deg=self._initial_face_yaw_deg,
                    offset_x_m=self._initial_face_offset_x_m,
                    dist_z_m=self._initial_face_dist_z_m,
                    target_rel_yaw_deg=target,
                )
            lines.append((
                f"[INITIAL_FACE_STOP→INITIAL_FACE_ROTATE] "
                f"pallet-center target={target:+.2f}°",
                COLOR_META,
            ))
            return lines

        # ----------------------- INITIAL_FACE_ROTATE -----------------------
        if self.state == "INITIAL_FACE_ROTATE":
            target = self._initial_face_target_deg
            ref = self._initial_face_rel_ref
            if target is None or ref is None or rel_yaw is None:
                self.execu.exec("STOP")
                self.state = "SEARCH"
                self._clear_initial_face()
                lines.append(("[INITIAL_FACE_ROTATE→SEARCH] IMU 기준값 유실", COLOR_ALERT))
                return lines

            delta = self._wrap_to_180(float(rel_yaw) - ref)
            code = "ROT_RIGHT" if target > 0.0 else "ROT_LEFT"
            reached = (delta >= target) if target > 0.0 else (delta <= target)

            if reached:
                self.execu.exec("STOP")
                self.status.start_timed("STOP", INITIAL_FACE_SETTLE_SEC)
                if self.tracer is not None:
                    self.tracer.step_end(actual_rel_yaw_deg=delta, target_rel_yaw_deg=target)
                self.state = "INITIAL_FACE_SETTLE"
                self._initial_face_deadline = time.time() + INITIAL_FACE_SETTLE_SEC
                lines.append((f"[INITIAL_FACE_ROTATE→INITIAL_FACE_SETTLE] {delta:+.2f}°", COLOR_META))
                return lines

            if time.time() >= self._initial_face_rotate_deadline:
                self.execu.exec("STOP")
                if self.tracer is not None:
                    self.tracer.step_end(timeout=True, actual_rel_yaw_deg=delta)
                self.state = "SEARCH"
                self._clear_initial_face()
                lines.append((f"[INITIAL_FACE_ROTATE→SEARCH] timeout, delta={delta:+.2f}°", COLOR_ALERT))
                return lines

            self.execu.exec(code)
            self.status.start_until(code, "rel_yaw", delta, target)
            lines.append((f"[INITIAL_FACE_ROTATE] IMU {delta:+.2f}/{target:+.2f}°", COLOR_STATUS_TRK))
            return lines

        # ----------------------- INITIAL_FACE_SETTLE -----------------------
        if self.state == "INITIAL_FACE_SETTLE":
            self._ensure_stop()
            remain = max(0.0, self._initial_face_deadline - time.time())
            self.status.start_timed("STOP", remain)
            if remain > 0.0:
                lines.append((f"[INITIAL_FACE_SETTLE] 회전 후 정지 ({remain:.1f}s)", COLOR_STATUS_TRK))
                return lines

            self.state = "ALIGN"
            self.align.reset("DIST_CHECK")
            self._clear_initial_face()
            lines.append(("[INITIAL_FACE_SETTLE→ALIGN.DIST_CHECK] 탐지 재개", COLOR_META))
            return lines

        # ------------------------------ RECOVER -----------------------------
        if self.state == "RECOVER":
            rec_lines = self.recover.step(det_ok, detected_length, ox)
            lines.extend(rec_lines)

            if self.recover.sub == "HOLD":
                self.state = "CHECK"
                self._stb.reset()
                lines.append(("[RECOVER→CHECK]", COLOR_META))
            return lines

        # ------------------------------- CHECK ------------------------------
        if self.state == "CHECK":
            self._ensure_stop()

            both_ok = ((yaw_smooth is not None and abs(yaw_smooth) <= YAW_TOL_DEG)
                       and (ox is not None and abs(ox) <= OFF_TOL_M))
            tag = "CHECK_OK" if both_ok else "CHECK_NOK"

            if self._stb.stable(tag):
                if both_ok:
                    self.state = "DONE"
                    lines.append(("[CHECK→DONE] 정렬 완료", COLOR_STATUS_OK))
                else:
                    # 새 정지 스냅샷을 받아 yaw/lateral 정렬을 다시 계획한다.
                    self.state = "ALIGN"
                    self.align.reset("POST_LATERAL_INFER")
                    lines.append(("[CHECK→ALIGN] 정지 스냅샷 재정렬", COLOR_STATUS_TRK))
            else:
                lines.append((f"[CHECK] 검수 중 [{self._stb.k}/{CMD_STABLE_THR}]", COLOR_STATUS_TRK))
            return lines

        # ------------------------------- ALIGN ------------------------------
        if self.state == "ALIGN":
            # 정렬 서브머신 실행 — ★ rel_yaw 전달 (90° 체인 종료/진행률에 사용)
            aln_lines = self.align.step(
                det_ok, detected_length, dist_z,
                yaw_smooth,  # 정지 체크포인트에서 저장할 팔레트 면 기준 yaw
                ox,          # offset_x
                rel_yaw      # ★ IMU 기반 상대 yaw(gyro-Y)
            )
            lines.extend(aln_lines)

            if self.align.sub == "READY_TO_DONE":
                self.state = "DONE"
                self._ensure_stop()
                lines.append(("[ALIGN→DONE] 정렬 완료", COLOR_STATUS_OK))
            return lines

        # -------------------------------- DONE ------------------------------
        if self.state == "DONE":
            self._ensure_stop()
            lines.append(("[DONE] 정렬 완료 상태 유지", COLOR_STATUS_OK))
            return lines

        # ------------------------------ Fallback ----------------------------
        self._ensure_stop()
        lines.append((f"[{self.state}] 대기", COLOR_META))
        return lines
