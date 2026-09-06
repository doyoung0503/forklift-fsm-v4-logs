# calib/fsm/utils.py
from __future__ import annotations
from typing import Optional
import time

from calib.config import CMD_STABLE_THR


def within_band(val: Optional[float], target: float, band: float) -> bool:
    """값이 target±band 범위에 있는지 확인."""
    return (val is not None) and (abs(val - target) <= band)


class Stabilizer:
    """
    같은 '판정 태그(tag)'가 연속 thr 프레임 유지될 때 True를 반환하는 안정화기.
    - thr 미지정 시 config의 CMD_STABLE_THR을 사용.
    - stable(tag): 같은 tag가 연속 thr회 관측되면 True
    - reset(): 내부 카운터 초기화
    - k: 현재 연속 관측 프레임 수(읽기 전용)
    """
    def __init__(self, thr: Optional[int] = None):
        self._thr: int = int(thr) if thr is not None else int(CMD_STABLE_THR)
        self._tag: Optional[str] = None
        self._k: int = 0

    def reset(self) -> None:
        self._tag, self._k = None, 0

    @property
    def k(self) -> int:
        return self._k

    @property
    def thr(self) -> int:
        return self._thr

    def set_thr(self, thr: int) -> None:
        """런타임에 임계 프레임 수를 변경."""
        self._thr = int(thr)
        # 보수적으로 카운터 리셋 (임계 변경 시 과거 관측 무효화)
        self.reset()

    def stable(self, tag: str) -> bool:
        """같은 tag가 연속 thr회 들어오면 True."""
        if self._tag == tag:
            self._k += 1
        else:
            self._tag, self._k = tag, 1
        return self._k >= self._thr


class SimpleTimer:
    """간단한 카운트다운 타이머 (초 기반)."""
    def __init__(self):
        self._until: float = 0.0

    def start(self, sec: float) -> None:
        self._until = time.time() + float(sec)

    def active(self) -> bool:
        return time.time() < self._until

    def remain(self) -> float:
        return max(0.0, self._until - time.time())
