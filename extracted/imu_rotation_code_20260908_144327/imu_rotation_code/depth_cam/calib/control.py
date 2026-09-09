# calib/control.py
# =============================================================================
# CAN 통신 제어부 (fsm.py와의 연동 전용)
#  - 근거: control_forklift_v2.py (DirectFrameForkliftController)
#  - Kvaser CANlib + Frame 사용
#  - movement/control 템플릿, ID/플래그, Heartbeat 규격을 동일하게 유지
#  - fsm.py 퍼블릭 API 시그니처(확정):
#       issue_command_forward_and_turn(turn_dir), issue_command_rotate_in_place(turn_dir),
#       issue_command_backward(), issue_command_forward(), issue_command_stop()
#       (+) issue_command_backward_and_turn(turn_dir)
# =============================================================================
from __future__ import annotations
from typing import Optional, Dict
from dataclasses import dataclass, field
import threading
import time

from canlib import canlib, Frame  # Kvaser CANlib

__all__ = [
    "can_init", "can_close", "start_heartbeat", "stop_heartbeat", "send_heartbeat",
    "configure_drive_deflection", "configure_forward_slow", "configure_rotate_in_place",
    "get_can_status",
    "issue_command_forward", "issue_command_backward",
    "issue_command_forward_slow",
    "issue_command_forward_and_turn", "issue_command_backward_and_turn",
    "issue_command_rotate_in_place", "issue_command_stop",
    "issue_command_fold", "issue_command_unfold",
    "issue_command_lift_up", "issue_command_lift_down",
    "issue_command_reach_forward", "issue_command_reach_backward",
    "issue_command_folding_stop", "issue_command_lift_stop",
    "issue_command_reach_stop",
]

# =============================================================================
# 공용 상수/플래그
# =============================================================================
class MessageFlag:
    STD = 0x0000  # 표준 11-bit ID
    EXT = 0x0004  # 확장 29-bit ID

# ---- 설정 (필요시 외부에서 수정 가능) ----
CAN_CHANNEL = 0
CAN_BITRATE = 500_000
USE_EXTENDED_IDS = False  # False=표준11bit, True=확장29bit

# IDs (동일)
CAN_MOVEMENT_ID = 0x01E3
CAN_CONTROL_ID  = 0x02E3

# Heartbeat
HEARTBEAT_ID = 0x764
HEARTBEAT_PERIOD = 0.200
HEARTBEAT_DATA = [0x00]

# control_forklift_v2.py와 동일한 주기 송신 규격
MOV_PERIOD = 0.010
CTRL_PERIOD = 0.005
DRIVE_ENTRY_BURST_N = 5
DRIVE_ENTRY_BURST_DT = 0.005

# =============================================================================
# 조이스틱 강도/템플릿 (v2와 동일 계산)
# =============================================================================
AN_NEUTRAL = 127
JOYSTICK_FORWARD = 60
JOYSTICK_BACKWARD = 60
JOYSTICK_LEFT = 60
JOYSTICK_RIGHT = 60
JOYSTICK_ROTATE_LEFT = JOYSTICK_LEFT // 3    # FSM 제자리 좌회전: 60 -> 20
JOYSTICK_ROTATE_RIGHT = JOYSTICK_RIGHT // 3  # FSM 제자리 우회전: 60 -> 20
JOYSTICK_FORWARD_SLOW = 30  # FSM v3 visual-servo forward; tune above dead-band

AN_FORWARD = min(255, AN_NEUTRAL - JOYSTICK_FORWARD)   # 67
AN_BACKWARD = max(0,   AN_NEUTRAL + JOYSTICK_BACKWARD) # 187
AN_LEFT =     min(255, AN_NEUTRAL + JOYSTICK_LEFT)     # 187
AN_RIGHT =    max(0,   AN_NEUTRAL - JOYSTICK_RIGHT)    # 67
AN_ROTATE_LEFT =  min(255, AN_NEUTRAL + JOYSTICK_ROTATE_LEFT)   # 147
AN_ROTATE_RIGHT = max(0,   AN_NEUTRAL - JOYSTICK_ROTATE_RIGHT)  # 107
AN_FORWARD_SLOW = min(255, AN_NEUTRAL - JOYSTICK_FORWARD_SLOW)  # 97

AN_N = AN_NEUTRAL

# Byte 의미: [1]=좌/우, [2]=전/후진. 나머지는 이 FSM에서 중립값을 유지한다.
MOVEMENT_TEMPLATES: Dict[str, list[int]] = {
    "stop":            [AN_N, AN_N, AN_N,        AN_N, AN_N, AN_N,         AN_N, AN_N],
    "forward":         [AN_N, AN_N, AN_FORWARD,  AN_N, AN_N, AN_N,         AN_N, AN_N],
    "forward_slow":    [AN_N, AN_N, AN_FORWARD_SLOW, AN_N, AN_N, AN_N,     AN_N, AN_N],
    "backward":        [AN_N, AN_N, AN_BACKWARD, AN_N, AN_N, AN_N,         AN_N, AN_N],
    "turn_left":       [AN_N, AN_LEFT, AN_N,     AN_N, AN_N, AN_N,         AN_N, AN_N],
    "turn_right":      [AN_N, AN_RIGHT,AN_N,     AN_N, AN_N, AN_N,         AN_N, AN_N],

    # FSM 제자리 회전: 검증된 A/D 바이트 위치를 유지하고 강도만 1/3로 낮춤
    "rotate_left_slow":  [AN_N, AN_ROTATE_LEFT,  AN_N, AN_N, AN_N, AN_N,   AN_N, AN_N],
    "rotate_right_slow": [AN_N, AN_ROTATE_RIGHT, AN_N, AN_N, AN_N, AN_N,   AN_N, AN_N],

    # 전/후진 + 조향
    "forward_left":    [AN_N, AN_LEFT, AN_FORWARD,  AN_N, AN_N, AN_N,      AN_N, AN_N],
    "forward_right":   [AN_N, AN_RIGHT,AN_FORWARD,  AN_N, AN_N, AN_N,      AN_N, AN_N],
    "backward_left":   [AN_N, AN_LEFT,  AN_BACKWARD, AN_N, AN_N, AN_N,     AN_N, AN_N],
    "backward_right":  [AN_N, AN_RIGHT, AN_BACKWARD, AN_N, AN_N, AN_N,     AN_N, AN_N],
}

# control 템플릿 (5번째 바이트 sync 카운터는 runtime 갱신)
CONTROL_TEMPLATES: Dict[str, list[int]] = {
    "driving_mode":    [0x42, 0x00, 0x00, 0x0A, 0x00, 0x40, 0x69, 0x93],
    "lift_mode":       [0x42, 0x00, 0x00, 0x05, 0x00, 0x40, 0x69, 0x93],
    "folding_mode":    [0x42, 0x00, 0x00, 0x06, 0x00, 0x40, 0x69, 0x93],
    "reach_mode":      [0x42, 0x00, 0x00, 0x09, 0x00, 0x40, 0x69, 0x93],
    # control_forklift_v2.py에서 실제 장비용으로 사용한 포크 제어 매핑
    "lift_up":         [0x42, 0x00, 0x00, 0x15, 0x00, 0x40, 0x69, 0x93],
    "lift_down":       [0x42, 0x00, 0x00, 0x25, 0x00, 0x40, 0x69, 0x93],
    "fold":            [0x42, 0x00, 0x00, 0x26, 0x00, 0x40, 0x69, 0x93],
    "unfold":          [0x42, 0x00, 0x00, 0x16, 0x00, 0x40, 0x69, 0x93],
    "reach_forward":   [0x42, 0x00, 0x00, 0x19, 0x00, 0x40, 0x69, 0x93],
    "reach_backward":  [0x42, 0x00, 0x00, 0x29, 0x00, 0x40, 0x69, 0x93],
    "emergency":       [0x80, 0x00, 0x00, 0x00, 0x01, 0x40, 0x69, 0x93],  # sync 미적용
}

# =============================================================================
# 내부 상태/도우미
# =============================================================================
@dataclass
class _BusCtx:
    ch: Optional[canlib.Channel] = None
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    sync_counter: int = 0x0A
    is_extended: bool = False
    current_movement: str = "stop"
    current_control: str = "driving_mode"
    tx_counts: Dict[str, int] = field(default_factory=lambda: {
        "movement": 0, "control": 0, "heartbeat": 0,
    })
    tx_errors: int = 0
    last_error: Optional[str] = None
    bus_status: Optional[str] = None
    bus_error_counters: Optional[str] = None

_CTX = _BusCtx()
_CAN_THREAD = None
_CAN_STOP = threading.Event()
_CAN_WAKE = threading.Event()
_CMD_PENDING = threading.Event()
_HB_NOW = threading.Event()
_CAN_READY = threading.Event()
_CAN_INIT_OK = False

def _flags():
    return MessageFlag.EXT if _CTX.is_extended else MessageFlag.STD

def _next_sync() -> int:
    # control frame 생성은 CAN 작업 스레드 한 곳에서만 수행한다.
    _CTX.sync_counter = (_CTX.sync_counter + 1) & 0x0F
    return _CTX.sync_counter

def _write(frame: Frame, kind: str) -> None:
    if _CTX.ch is None:
        raise RuntimeError("CAN channel is not open")
    _CTX.ch.write(frame)
    _CTX.tx_counts[kind] += 1

def _mk_control(ctrl_type: str) -> Frame:
    data = CONTROL_TEMPLATES.get(ctrl_type, CONTROL_TEMPLATES["driving_mode"]).copy()
    if ctrl_type != "emergency":
        data[4] = _next_sync()
    return Frame(id_=CAN_CONTROL_ID, data=data, flags=_flags())

def _mk_movement(name: str) -> Frame:
    data = MOVEMENT_TEMPLATES.get(name, MOVEMENT_TEMPLATES["stop"])
    return Frame(id_=CAN_MOVEMENT_ID, data=data, flags=_flags())

def _mk_heartbeat() -> Frame:
    return Frame(id_=HEARTBEAT_ID, data=HEARTBEAT_DATA, flags=_flags())

def _set_tx_state(movement: str, control: str = "driving_mode") -> None:
    """주기 송신 루프가 유지할 현재 명령을 원자적으로 갱신한다."""
    if movement not in MOVEMENT_TEMPLATES:
        movement = "stop"
    if control not in CONTROL_TEMPLATES:
        control = "driving_mode"
    with _CTX.state_lock:
        _CTX.current_movement = movement
        _CTX.current_control = control

def _get_tx_state() -> tuple[str, str]:
    with _CTX.state_lock:
        return _CTX.current_movement, _CTX.current_control

def _write_command_once(movement: str, control: str) -> None:
    """명령 변경 직후 v2와 같은 control → heartbeat → movement 묶음을 송신한다."""
    _write(_mk_control(control), "control")
    _write(_mk_heartbeat(), "heartbeat")
    _write(_mk_movement(movement), "movement")

def _refresh_bus_diagnostics() -> None:
    """Kvaser 컨트롤러의 실제 버스 상태와 CAN 오류 카운터를 갱신한다."""
    if _CTX.ch is None:
        return
    try:
        _CTX.bus_status = str(_CTX.ch.readStatus())
        _CTX.bus_error_counters = str(_CTX.ch.read_error_counters())
    except Exception as e:
        _CTX.bus_status = f"diagnostic unavailable: {type(e).__name__}: {e}"

def _can_tx_worker(channel: int, bitrate: int) -> None:
    """CAN 핸들의 생성부터 모든 송신과 종료까지 단일 스레드에서 처리한다."""
    global _CAN_INIT_OK
    try:
        _CTX.ch = canlib.openChannel(channel)
        br_map = {
            1_000_000: canlib.Bitrate.BITRATE_1M,
            500_000:   canlib.Bitrate.BITRATE_500K,
            250_000:   canlib.Bitrate.BITRATE_250K,
            125_000:   canlib.Bitrate.BITRATE_125K,
        }
        _CTX.ch.setBusParams(br_map.get(bitrate, canlib.Bitrate.BITRATE_500K))
        _CTX.ch.busOn()

        # v2 시작 버스트. writeSync로 드라이버 큐에서 실제 전송 완료까지 확인한다.
        for _ in range(5):
            _write_command_once("stop", "driving_mode")
            time.sleep(0.005)
        _CTX.ch.writeSync(200)
        _refresh_bus_diagnostics()

        _CAN_INIT_OK = True
        _CAN_READY.set()
        print("[CAN TX] 단일 작업 스레드 시작 "
              f"(movement={MOV_PERIOD*1000:.0f}ms, control={CTRL_PERIOD*1000:.0f}ms, "
              f"heartbeat={HEARTBEAT_PERIOD*1000:.0f}ms)")

        now = time.monotonic()
        next_mov = now
        next_ctrl = now
        next_hb = now + HEARTBEAT_PERIOD
        next_diag = now + 5.0
        last_sent_movement = "stop"

        while not _CAN_STOP.is_set():
            _CAN_WAKE.clear()
            if _CMD_PENDING.is_set():
                _CMD_PENDING.clear()
                movement, control = _get_tx_state()
                entry_burst = movement != "stop" and last_sent_movement == "stop"
                burst_n = DRIVE_ENTRY_BURST_N if entry_burst else 1
                for _ in range(burst_n):
                    _write_command_once(movement, control)
                    if burst_n > 1:
                        time.sleep(DRIVE_ENTRY_BURST_DT)
                _CTX.ch.writeSync(100)
                _refresh_bus_diagnostics()
                data = MOVEMENT_TEMPLATES[movement]
                print(f"[CAN TX CMD] {movement}: id=0x{CAN_MOVEMENT_ID:03X}, "
                      f"data={[f'0x{b:02X}' for b in data]}, burst={burst_n}, "
                      f"status={_CTX.bus_status}, counters={_CTX.bus_error_counters}")
                last_sent_movement = movement
                now = time.monotonic()
                next_mov = now + MOV_PERIOD
                next_ctrl = now + CTRL_PERIOD

            if _HB_NOW.is_set():
                _HB_NOW.clear()
                _write(_mk_heartbeat(), "heartbeat")

            now = time.monotonic()
            movement, control = _get_tx_state()
            if now >= next_ctrl:
                _write(_mk_control(control), "control")
                next_ctrl = now + CTRL_PERIOD
            if now >= next_mov:
                _write(_mk_movement(movement), "movement")
                next_mov = now + MOV_PERIOD
            if now >= next_hb:
                _write(_mk_heartbeat(), "heartbeat")
                next_hb = now + HEARTBEAT_PERIOD
            if now >= next_diag:
                if movement != "stop":
                    _refresh_bus_diagnostics()
                    print(f"[CAN TX ALIVE] movement={movement}, tx={_CTX.tx_counts}, "
                          f"errors={_CTX.tx_errors}, status={_CTX.bus_status}, "
                          f"counters={_CTX.bus_error_counters}")
                next_diag = now + 5.0

            wait_s = max(0.0, min(next_ctrl, next_mov, next_hb) - time.monotonic())
            _CAN_WAKE.wait(min(wait_s, 0.050))

    except Exception as e:
        _CAN_INIT_OK = False
        _CTX.tx_errors += 1
        _CTX.last_error = f"{type(e).__name__}: {e}"
        print(f"[CAN TX ERROR] {_CTX.last_error}")
    finally:
        _CAN_READY.set()
        if _CTX.ch is not None:
            try:
                # 종료 시 마지막 동작 프레임 뒤에 반드시 중립 프레임이 남도록 한다.
                for _ in range(3):
                    _write_command_once("stop", "driving_mode")
                    time.sleep(0.010)
                _CTX.ch.writeSync(200)
            except Exception as e:
                _CTX.tx_errors += 1
                _CTX.last_error = f"{type(e).__name__}: {e}"
            try:
                _CTX.ch.busOff()
                _CTX.ch.close()
            except Exception:
                pass
            _CTX.ch = None

# =============================================================================
# 초기화/종료/하트비트
# =============================================================================
def can_init(channel: int = CAN_CHANNEL, bitrate: int = CAN_BITRATE, is_extended_id: bool = USE_EXTENDED_IDS) -> bool:
    """
    Kvaser CAN 초기화 (성공 시 True)
    - bus on 이후 안정화 시퀀스(드라이빙 모드/정지/하트비트) 송신
    - 단일 CAN 작업 스레드가 heartbeat/movement/control을 모두 송신
    """
    global _CAN_THREAD, _CAN_INIT_OK
    if _CAN_THREAD is not None and _CAN_THREAD.is_alive():
        return _CAN_INIT_OK

    _CTX.is_extended = bool(is_extended_id)
    _CTX.sync_counter = 0x0A
    _CTX.tx_counts = {"movement": 0, "control": 0, "heartbeat": 0}
    _CTX.tx_errors = 0
    _CTX.last_error = None
    _CTX.bus_status = None
    _CTX.bus_error_counters = None
    _set_tx_state("stop", "driving_mode")
    _CAN_INIT_OK = False
    _CAN_STOP.clear()
    _CAN_WAKE.clear()
    _CMD_PENDING.clear()
    _HB_NOW.clear()
    _CAN_READY.clear()
    _CAN_THREAD = threading.Thread(
        target=_can_tx_worker, args=(channel, bitrate), name="CANBusOwner", daemon=True,
    )
    _CAN_THREAD.start()
    if not _CAN_READY.wait(timeout=3.0):
        _CTX.last_error = "CAN worker initialization timeout"
        print(f"[CAN INIT ERROR] {_CTX.last_error}")
        _CAN_STOP.set()
        _CAN_WAKE.set()
        return False
    return _CAN_INIT_OK

def can_close():
    """STOP 송신 후 버스 해제"""
    global _CAN_THREAD, _CAN_INIT_OK
    _set_tx_state("stop", "driving_mode")
    _CAN_STOP.set()
    _CAN_WAKE.set()
    if _CAN_THREAD is not None and _CAN_THREAD.is_alive():
        _CAN_THREAD.join(timeout=2.0)
    _CAN_THREAD = None
    _CAN_INIT_OK = False
    print(f"[CAN CLOSED] tx={_CTX.tx_counts}, errors={_CTX.tx_errors}, "
          f"last_error={_CTX.last_error}")

def send_heartbeat():
    """Heartbeat 1회 송신"""
    _HB_NOW.set()
    _CAN_WAKE.set()

def get_can_status() -> dict:
    """HUD/진단용 CAN 작업 스레드 및 누적 송신 상태."""
    alive = _CAN_THREAD is not None and _CAN_THREAD.is_alive()
    movement, control = _get_tx_state()
    return {
        "alive": alive,
        "ready": bool(_CAN_INIT_OK and alive),
        "movement": movement,
        "control": control,
        "tx_counts": dict(_CTX.tx_counts),
        "errors": _CTX.tx_errors,
        "last_error": _CTX.last_error,
        "bus_status": _CTX.bus_status,
        "bus_error_counters": _CTX.bus_error_counters,
    }

def start_heartbeat():
    """하트비트는 CAN 작업 스레드가 자동 송신한다(호환용 API)."""
    return

def stop_heartbeat():
    """CAN 작업 스레드 수명과 함께 종료된다(호환용 API)."""
    return

# =============================================================================
# fsm.py에서 호출하는 퍼블릭 API
#  - 명령 변경 시 즉시 송신하고, 백그라운드에서 v2와 같은 주기로 유지 송신
# =============================================================================
def _activate_command(movement: str, control: str = "driving_mode") -> None:
    """현재 명령을 갱신하고 CAN 작업 스레드를 즉시 깨운다."""
    _set_tx_state(movement, control)
    if _CAN_THREAD is None or not _CAN_THREAD.is_alive():
        print(f"[MOCK CMD] {movement}: CAN 작업 스레드가 실행 중이 아닙니다.")
        return
    _CMD_PENDING.set()
    _CAN_WAKE.set()

def issue_command_forward_and_turn(turn_dir: int) -> None:
    """
    전진 + 회전 (turn_dir: +1=좌, -1=우)
    v2 매핑: forward_left / forward_right + driving_mode control 동반
    """
    name = "forward_left" if turn_dir > 0 else "forward_right"
    _activate_command(name)

def issue_command_backward_and_turn(turn_dir: int) -> None:
    """
    후진 + 회전 (turn_dir: +1=좌, -1=우)
    - 좌회전: 왼쪽 바퀴 약화(= 좌로 조향) → backward_left
    - 우회전: 오른쪽 바퀴 약화(= 우로 조향) → backward_right
    """
    name = "backward_left" if turn_dir > 0 else "backward_right"
    _activate_command(name)

def issue_command_rotate_in_place(turn_dir: int) -> None:
    """
    제자리 회전 (turn_dir: +1=좌, -1=우).
    실제 장비에서 검증된 v2의 A/D 바이트 위치를 유지하면서 강도는 1/3을 사용한다.
    """
    name = "rotate_left_slow" if turn_dir > 0 else "rotate_right_slow"
    _activate_command(name)

def configure_rotate_in_place(joystick_deflection: int) -> int:
    """Update the in-place rotation templates; return applied strength."""
    strength = max(1, min(126, int(joystick_deflection)))
    left_analog = min(255, AN_NEUTRAL + strength)
    right_analog = max(0, AN_NEUTRAL - strength)
    MOVEMENT_TEMPLATES["rotate_left_slow"] = [
        AN_N, left_analog, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N,
    ]
    MOVEMENT_TEMPLATES["rotate_right_slow"] = [
        AN_N, right_analog, AN_N, AN_N, AN_N, AN_N, AN_N, AN_N,
    ]
    return strength

def configure_drive_deflection(
    forward_deflection: int, backward_deflection: Optional[int] = None,
) -> tuple[int, int]:
    """직진 전진/후진 템플릿의 조이스틱 세기를 갱신한다.

    기존 FSM은 기본 템플릿을 그대로 사용하고, FSM v4만 생성 시점에
    전용 config 값을 이 함수에 전달한다.
    """
    forward = max(1, min(126, int(forward_deflection)))
    backward = forward if backward_deflection is None else max(
        1, min(126, int(backward_deflection))
    )
    forward_analog = max(0, AN_NEUTRAL - forward)
    backward_analog = min(255, AN_NEUTRAL + backward)
    MOVEMENT_TEMPLATES["forward"] = [
        AN_N, AN_N, forward_analog, AN_N, AN_N, AN_N, AN_N, AN_N,
    ]
    MOVEMENT_TEMPLATES["backward"] = [
        AN_N, AN_N, backward_analog, AN_N, AN_N, AN_N, AN_N, AN_N,
    ]
    return forward, backward

def issue_command_backward() -> None:
    """후진 직진"""
    _activate_command("backward")

def issue_command_forward() -> None:
    """전진 직진"""
    _activate_command("forward")

def issue_command_forward_slow() -> None:
    """FSM v3 vision-servo 저속 전진 직진."""
    _activate_command("forward_slow")

def configure_forward_slow(joystick_deflection: int) -> int:
    """Update only the v3 low-speed forward template; return applied strength."""
    strength = max(1, min(126, int(joystick_deflection)))
    analog = min(255, AN_NEUTRAL - strength)
    MOVEMENT_TEMPLATES["forward_slow"] = [
        AN_N, AN_N, analog, AN_N, AN_N, AN_N, AN_N, AN_N,
    ]
    return strength

def issue_command_stop() -> None:
    """정지 (driving_mode + stop + heartbeat)"""
    _activate_command("stop")

def issue_command_fold() -> None:
    """포크/마스트 접기 동작. 주행 movement는 항상 중립으로 유지한다."""
    _activate_command("stop", "fold")

def issue_command_unfold() -> None:
    """포크/마스트 세우기 동작. 주행 movement는 항상 중립으로 유지한다."""
    _activate_command("stop", "unfold")

def issue_command_folding_stop() -> None:
    """폴딩 축을 중립 모드로 복귀한다."""
    _activate_command("stop", "folding_mode")

def issue_command_lift_up() -> None:
    """포크를 올린다. 주행 movement는 항상 중립으로 유지한다."""
    _activate_command("stop", "lift_up")

def issue_command_lift_down() -> None:
    """포크를 내린다. 주행 movement는 항상 중립으로 유지한다."""
    _activate_command("stop", "lift_down")

def issue_command_lift_stop() -> None:
    """리프트 축을 중립 모드로 복귀한다."""
    _activate_command("stop", "lift_mode")

def issue_command_reach_forward() -> None:
    """포크 리치를 앞으로 이동한다. 주행 movement는 항상 중립이다."""
    _activate_command("stop", "reach_forward")

def issue_command_reach_backward() -> None:
    """포크 리치를 뒤로 이동한다. 주행 movement는 항상 중립이다."""
    _activate_command("stop", "reach_backward")

def issue_command_reach_stop() -> None:
    """리치 축을 중립 모드로 복귀한다."""
    _activate_command("stop", "reach_mode")
