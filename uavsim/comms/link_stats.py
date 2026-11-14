"""TX/RX history and bandwidth bookkeeping shared by every link backend.

Both `GnuRadioChannel` (the real PHY) and `LoopbackChannel` (the fast
approximate one) keep the same three logs, which is all the HUD needs:

* ``_tx_log``      -- every payload handed to the transmitter (airtime used)
* ``_rx_log``      -- every payload that was actually recovered
* ``_rx_raw_log``  -- a short slice of the received waveform per burst
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


class LinkStatsMixin:
    def _init_stats(self, signal_history_seconds: float) -> None:
        self._signal_history_seconds = signal_history_seconds
        self._tx_log: Deque[Tuple[float, bytes]] = deque()
        self._rx_log: Deque[Tuple[float, bytes]] = deque()
        self._rx_raw_log: Deque[Tuple[float, np.ndarray]] = deque()

    def _clear_stats(self) -> None:
        self._tx_log.clear()
        self._rx_log.clear()
        self._rx_raw_log.clear()

    def _prune(self, log: Deque, now: float) -> None:
        cutoff = now - self._signal_history_seconds
        while log and log[0][0] < cutoff:
            log.popleft()

    def tx_bandwidth_bps(self, window: float = 1.0) -> float:
        return self._bandwidth(self._tx_log, window)

    def rx_bandwidth_bps(self, window: float = 1.0) -> float:
        return self._bandwidth(self._rx_log, window)

    def tx_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, bytes]]:
        return self._recent(self._tx_log, window)

    def rx_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, bytes]]:
        return self._recent(self._rx_log, window)

    def rx_raw_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, np.ndarray]]:
        return self._recent(self._rx_raw_log, window)

    def _bandwidth(self, log: Deque[Tuple[float, bytes]], window: float) -> float:
        now = time.monotonic()
        self._prune(log, now)
        cutoff = now - window
        total_bytes = sum(len(payload) for t, payload in log if t >= cutoff)
        return total_bytes / window if window > 0 else 0.0

    def _recent(self, log: Deque, window: Optional[float]) -> List:
        now = time.monotonic()
        self._prune(log, now)
        if window is None:
            return list(log)
        cutoff = now - window
        return [(t, p) for t, p in log if t >= cutoff]
