"""Fast numpy stand-in for `GnuRadioChannel`, with selectable mitigation.

GNU Radio is the reference physical layer of this simulator, but it is
hard to install on Windows and too slow for batch experiments. This
backend keeps the *same interface and frame structure* and runs every
packet through the bit-true modem in `uavsim.comms.phy.modem`: real ±1
chips, real Gaussian noise, an optional CW tone jammer, sync-word check,
despreading, Viterbi and CRC.

The noise level is calibrated (implementation-loss factor K = 0.41) so
that with no mitigation it reproduces where the GMSK link stops working
(~10% chip errors at noise voltage 0.71). The closed-form functions
below are that same calibration, kept for quick estimates and tests.

Mitigation profiles (``profile``): see `uavsim.comms.phy.modem.PROFILES`.

Use it with ``python main.py --link loopback``.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from uavsim.comms.link_stats import LinkStatsMixin
from uavsim.comms.phy.modem import IMPLEMENTATION_LOSS_K, PROFILES, NumpyPhy, PhyProfile

ACCESS_CODE_BITS = 32
ACCESS_CODE_THRESHOLD = 4
CRC_BITS = 32
CHIP_ERROR_SCALE = IMPLEMENTATION_LOSS_K   # K in p = 0.5*erfc(sqrt(K)/noise_voltage)
DELIVERY_LATENCY = 0.012                   # s, roughly one GNU Radio burst


# -- closed-form calibration (no mitigation, hard decisions) ---------------
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


# -- channel -----------------------------------------------------------------
class LoopbackChannel(LinkStatsMixin):
    def __init__(
        self,
        noise_voltage: float = 0.005,
        signal_history_seconds: float = 3.0,
        dssc_N: int = 0,
        seed: Optional[int] = None,
        latency: float = DELIVERY_LATENCY,
        profile: str = "none",
    ) -> None:
        self._init_stats(signal_history_seconds)
        self._noise_voltage = noise_voltage
        self._tone_amplitude = 0.0
        self._latency = latency
        self._rng = np.random.default_rng(seed)
        self._pending: Deque[Tuple[float, Optional[bytes], np.ndarray, float]] = deque()
        self._rx_queue: Deque[bytes] = deque()
        self._last_correlation = 0.0
        self._worker_crashed_info = ""
        self._seq = 0
        self.profile = "dsss-lorenz" if dssc_N > 0 and profile == "none" else profile

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
    def tone_amplitude(self) -> float:
        """CW jammer amplitude relative to the signal (GNU Radio has no equivalent)."""
        return self._tone_amplitude

    @tone_amplitude.setter
    def tone_amplitude(self, value: float) -> None:
        self._tone_amplitude = value

    @property
    def profile(self) -> str:
        return self._profile.name

    @profile.setter
    def profile(self, name: str) -> None:
        self._profile: PhyProfile = PROFILES[name]
        self._phy = NumpyPhy(self._profile, self._rng)

    @property
    def profile_label(self) -> str:
        return self._profile.label

    @property
    def dssc_N(self) -> int:
        return self._profile.dsss_n

    @dssc_N.setter
    def dssc_N(self, value: int) -> None:
        self.profile = "dsss-lorenz" if value > 0 else "none"

    @property
    def last_correlation(self) -> float:
        return self._last_correlation

    def _transmit(self, payload: bytes) -> None:
        now = time.monotonic()
        self._tx_log.append((now, payload))
        self._prune(self._tx_log, now)
        result = self._phy.burst(payload, self._seq, self._noise_voltage, self._tone_amplitude)
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        self._pending.append((now + self._latency, result.payload, result.preview,
                              result.correlation))

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
