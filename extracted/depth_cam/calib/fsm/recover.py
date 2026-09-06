# calib/fsm/recover.py
from __future__ import annotations
from typing import Optional, List, Tuple

from calib.config import (
    OFF_TOL_M, WIDTH_MIN_FULL,
    COLOR_ALERT, COLOR_STATUS_TRK, COLOR_META
)
from .commands import CommandExecutor
from .status_helper import StatusHelper
from .utils import Stabilizer

class RecoverMachine:
    """전면(폭) 확보 단계: SEARCH 이후 or DETECTED에서 폭 부족 시 사용."""
    def __init__(self, execu: CommandExecutor, status: StatusHelper):
        self.sub: str = "DECIDE_TURN"
        self._stb = Stabilizer()
        self.execu = execu
        self.status = status

    def reset(self):
        self.sub = "DECIDE_TURN"
        self._stb.reset()

    def step(self, det_ok: bool, detected_length: Optional[float], ox: Optional[float]) -> List[Tuple[str, tuple]]:
        lines: List[Tuple[str, tuple]] = []

        if not det_ok:
            # 탐지될 때까지 최근 방향으로 회전
            if self.sub.startswith("RECOVER_ROTATE_"):
                lines.append(("[RECOVER] 회전 유지", COLOR_ALERT))
                return lines
            self.execu.exec("ROT_LEFT" if self.execu.last_dir > 0 else "ROT_RIGHT")
            lines.append(("[RECOVER] 탐지 대기", COLOR_STATUS_TRK))
            return lines

        if self.sub == "DECIDE_TURN":
            if (ox is None) or (abs(ox) <= OFF_TOL_M):
                tag = "DECIDE_CENTER"
                if self._stb.stable(tag):
                    self.sub = "RECOVER_ROTATE_LEFT" if self.execu.last_dir > 0 else "RECOVER_ROTATE_RIGHT"
                    code = "ROT_LEFT" if self.execu.last_dir > 0 else "ROT_RIGHT"
                    self.execu.exec(code)
                    self.status.start_timed(code, 0.0)
                    lines.append(("[RECOVER] 중심 → 최근방향 회전", COLOR_STATUS_TRK))
                else:
                    lines.append((f"[대기] RECOVER.DECIDE(center) [{self._stb.k}/*]", COLOR_STATUS_TRK))
                return lines
            else:
                if ox > 0:
                    tag = "DECIDE_RIGHT"
                    if self._stb.stable(tag):
                        self.sub = "RECOVER_ROTATE_RIGHT"
                        self.execu.exec("ROT_RIGHT")
                        self.status.start_timed("ROT_RIGHT", 0.0)
                        lines.append(("[RECOVER] offset>0 → 우회전", COLOR_STATUS_TRK))
                    else:
                        lines.append((f"[대기] RECOVER.DECIDE(right) [{self._stb.k}/*]", COLOR_STATUS_TRK))
                    return lines
                else:
                    tag = "DECIDE_LEFT"
                    if self._stb.stable(tag):
                        self.sub = "RECOVER_ROTATE_LEFT"
                        self.execu.exec("ROT_LEFT")
                        self.status.start_timed("ROT_LEFT", 0.0)
                        lines.append(("[RECOVER] offset<0 → 좌회전", COLOR_STATUS_TRK))
                    else:
                        lines.append((f"[대기] RECOVER.DECIDE(left) [{self._stb.k}/*]", COLOR_STATUS_TRK))
                    return lines

        if self.sub in ("RECOVER_ROTATE_LEFT", "RECOVER_ROTATE_RIGHT"):
            if (detected_length is not None) and (detected_length >= WIDTH_MIN_FULL):
                self.sub = "HOLD"
                lines.append(("[RECOVER] 전면 확보 → HOLD", COLOR_STATUS_TRK))
            else:
                code = "ROT_LEFT" if self.sub.endswith("LEFT") else "ROT_RIGHT"
                self.execu.exec(code)
                self.status.start_timed(code, 0.0)
                lines.append(("[RECOVER] 전면 확보 회전 중", COLOR_ALERT))
            return lines

        if self.sub == "HOLD":
            lines.append(("[RECOVER→CHECK]", COLOR_META))
            return lines

        return lines
