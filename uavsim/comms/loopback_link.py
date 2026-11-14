"""Fast, approximate stand-in for `GnuRadioChannel` (pure numpy).

GNU Radio is the reference physical layer of this simulator, but it is
hard to install on Windows and too slow for batch experiments. This
backend keeps the *same interface and the same frame structure* and
replaces the DSP with a statistical model:

* A chip/bit error probability derived from ``noise_voltage`` with a
  BPSK-like curve, ``p = 0.5 * erfc(sqrt(K) / noise_voltage)``. ``K`` is
  calibrated so that a jammer centre (noise 0.71) gives ~10% chip errors,
  where the real GMSK link stops working. It is an approximation, not a
  replacement for GNU Radio results.
* The 32-bit access code must match with at most 4 bit errors. Like the
  real implementation, the header is *not* spread by DSSS.
* Payload + CRC32 must arrive with zero bit errors. With DSSS enabled
  each bit is a majority vote over N hard chips.

Use it with ``python main.py --link loopback``.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from uavsim.comms.link_stats import LinkStatsMixin

ACCESS_CODE_BITS = 32
ACCESS_CODE_THRESHOLD = 4
CRC_BITS = 32
CHIP_ERROR_SCALE = 0.41     # K in p = 0.5*erfc(sqrt(K)/noise_voltage)
DELIVERY_LATENCY = 0.012    # s, roughly one GNU Radio burst
_SAMPLES_PER_SYMBOL = 4


def chip_error_probability(noise_voltage: float) -> float:
    if noise_voltage <= 1e-9:
        return 0.0
    return 0.5 * math.erfc(math.sqrt(CHIP_ERROR_SCALE) / noise_voltage)


def header_success_probability(p: float) -> float:
    n, t = ACCESS_CODE_BITS, ACCESS_CODE_THRESHOLD
    return sum(math.comb(n, k) * p**k * (1.0 - p) ** (n - k) for k in range(t + 1))


def despread_bit_error_probability(p: float, n_chips: int) -> float:
    """Majority vote over `n_chips` hard chips (ties broken 50/50)."""
    if n_chips <= 1:
        return p
    pe = sum(
        math.comb(n_chips, k) * p**k * (1.0 - p) ** (n_chips - k)
        for k in range(n_chips // 2 + 1, n_chips + 1)
    )
    if n_chips % 2 == 0:
        half = n_chips // 2
        pe += 0.5 * math.comb(n_chips, half) * p**half * (1.0 - p) ** half
    return pe


def packet_success_probability(payload_len: int, noise_voltage: float, dssc_n: int = 0) -> float:
    p = chip_error_probability(noise_voltage)
    bit_error = despread_bit_error_probability(p, dssc_n) if dssc_n > 0 else p
    data_bits = payload_len * 8 + CRC_BITS
    return header_success_probability(p) * (1.0 - bit_error) ** data_bits


class LoopbackChannel(LinkStatsMixin):
    def __init__(
        self,
        noise_voltage: float = 0.005,
        signal_history_seconds: float = 3.0,
        dssc_N: int = 0,
        seed: Optional[int] = None,
        latency: float = DELIVERY_LATENCY,
    ) -> None:
        self._init_stats(signal_history_seconds)
        self._noise_voltage = noise_voltage
        self._dssc_N = dssc_N
        self._latency = latency
        self._rng = np.random.default_rng(seed)
        self._pending: Deque[Tuple[float, Optional[bytes], np.ndarray, float]] = deque()
        self._rx_queue: Deque[bytes] = deque()
        self._last_correlation = 0.0
        self._worker_crashed_info = ""

    # -- same surface as GnuRadioChannel ---------------------------------
    @property
    def worker_alive(self) -> bool:
        return True

    @property
    def noise_voltage(self) -> float:
        return self._noise_voltage

    @noise_voltage.setter
    def noise_voltage(self, value: float) -> None:
        self._noise_voltage = value

    @property
    def dssc_N(self) -> int:
        return self._dssc_N

    @dssc_N.setter
    def dssc_N(self, value: int) -> None:
        self._dssc_N = value

    @property
    def last_correlation(self) -> float:
        return self._last_correlation

    def _transmit(self, payload: bytes) -> None:
        now = time.monotonic()
        self._tx_log.append((now, payload))
        self._prune(self._tx_log, now)

        nv = self._noise_voltage
        p = chip_error_probability(nv)
        ok = self._rng.random() < packet_success_probability(len(payload), nv, self._dssc_N)
        if self._dssc_N > 0:
            correlation = max(0.0, 1.0 - 2.0 * p)
        else:
            correlation = 1.0 if ok else 0.0
        raw = self._synth_waveform(payload, nv)
        self._pending.append((now + self._latency, payload if ok else None, raw, correlation))

    def _receive(self) -> Optional[bytes]:
        self._drain()
        return self._rx_queue.popleft() if self._rx_queue else None

    def clear(self, reset_noise: float = 0.0) -> None:
        self._clear_stats()
        self._pending.clear()
        self._rx_queue.clear()
        self._noise_voltage = reset_noise

    # ------------------------------------------------------------------
    def _drain(self) -> None:
        now = time.monotonic()
        while self._pending and self._pending[0][0] <= now:
            t, payload, raw, correlation = self._pending.popleft()
            self._last_correlation = correlation
            self._rx_raw_log.append((t, raw))
            self._prune(self._rx_raw_log, t)
            if payload is not None:
                self._rx_log.append((t, payload))
                self._prune(self._rx_log, t)
                self._rx_queue.append(payload)

    def _synth_waveform(self, payload: bytes, noise_voltage: float) -> np.ndarray:
        """Short smoothed-NRZ waveform of the first symbols, for the HUD scope."""
        n_symbols = self._dssc_N if self._dssc_N > 0 else 8
        bits = np.unpackbits(np.frombuffer(payload[:1] or b"\x00", dtype=np.uint8))
        symbols = np.resize(bits, n_symbols).astype(float) * 2.0 - 1.0
        wave = np.repeat(symbols, _SAMPLES_PER_SYMBOL)
        wave = np.convolve(wave, np.ones(3) / 3.0, mode="same")
        wave = wave + noise_voltage * self._rng.standard_normal(wave.size)
        peak = float(np.max(np.abs(wave))) or 1.0
        return wave / peak
