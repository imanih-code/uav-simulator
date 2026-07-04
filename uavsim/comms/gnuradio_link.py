"""The actual physical layer for UAV <-> Operator communication, built on
real GNU Radio signal processing -- not a simulated stand-in.

Every payload handed to `GnuRadioChannel._transmit()` is genuinely:
  1. Framed with a preamble + access code + CRC32 + flush padding.
  2. GMSK-modulated into complex baseband samples (`digital.gmsk_mod`).
  3. Pushed through a simulated RF channel with AWGN noise
     (`channels.channel_model`) -- this is where a future Jammer would
     simply crank up the noise/frequency-offset, with zero changes
     anywhere else in the app.
  4. GMSK-demodulated, bit-synchronized against the access code
     (`digital.correlate_access_code_tag_bb`), and CRC-checked.

Packets that fail to sync, get truncated, or fail their CRC are dropped --
exactly like a real radio link -- rather than being handed to the app as
garbage. This is what makes the drone's control link and telemetry link
"real" instead of a bare in-memory queue.

Each channel (uplink commands, downlink telemetry) runs its own dedicated
background thread. GNU Radio's own C++ blocks dominate the per-burst cost
(building/tearing down a tiny flowgraph, not the DSP itself -- measured at
roughly ~10ms/burst regardless of payload size), so a single worker thread
per channel comfortably keeps up with this app's message rates without
blocking the render loop.
"""
from __future__ import annotations

import queue
import random
import threading
import time
import zlib
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from gnuradio import blocks, channels, digital, gr

# -- RF link constants, validated empirically against this exact GNU Radio
# build before being used anywhere else in the app. -------------------------
_PREAMBLE = b"\x55\x55"                 # alternating bits: lets clock recovery settle
_ACCESS_CODE_BYTES = b"\xac\xdd\xa4\xe2"  # arbitrary but fixed sync word
_ACCESS_CODE_BITS = "".join(f"{byte:08b}" for byte in _ACCESS_CODE_BYTES)
_TAIL_PADDING = b"\x55" * 8              # lets the filter pipeline fully flush
_ACCESS_CODE_THRESHOLD = 2               # bit errors tolerated in the sync word
_CRC_SIZE_BYTES = 4

DEFAULT_SAMPLES_PER_SYMBOL = 4
DEFAULT_NOISE_VOLTAGE = 0.05             # AWGN level; a future Jammer raises this


def _run_burst(
    payload: bytes,
    samples_per_symbol: int,
    noise_voltage: float,
    frequency_offset: float,
) -> Optional[bytes]:
    """Modulate `payload`, push it through a simulated noisy channel, and
    demodulate it back. Returns the recovered payload, or None if the
    burst was lost (no sync found, truncated, or failed its CRC)."""
    crc = zlib.crc32(payload).to_bytes(_CRC_SIZE_BYTES, "big")
    frame = _PREAMBLE + _ACCESS_CODE_BYTES + payload + crc + _TAIL_PADDING

    source = blocks.vector_source_b(list(frame), False)
    modulator = digital.gmsk_mod(samples_per_symbol=samples_per_symbol, bt=0.35)
    channel = channels.channel_model(
        noise_voltage=noise_voltage,
        frequency_offset=frequency_offset,
        noise_seed=random.randint(1, 2_000_000_000),
    )
    demodulator = digital.gmsk_demod(samples_per_symbol=samples_per_symbol)
    correlator = digital.correlate_access_code_tag_bb(
        _ACCESS_CODE_BITS, _ACCESS_CODE_THRESHOLD, "sync"
    )
    sink = blocks.vector_sink_b(1)

    top_block = gr.top_block()
    top_block.connect(source, modulator, channel, demodulator, correlator, sink)
    top_block.run()

    bits = np.array(sink.data(), dtype=np.uint8)
    sync_offsets = [tag.offset for tag in sink.tags() if str(tag.key) == "sync"]
    if not sync_offsets:
        return None  # no sync found -- packet lost to noise

    needed_bits = (len(payload) + _CRC_SIZE_BYTES) * 8
    recovered_bits = bits[sync_offsets[0]: sync_offsets[0] + needed_bits]
    if len(recovered_bits) < needed_bits:
        return None  # truncated -- not enough signal survived to the sink

    recovered = bytes(np.packbits(recovered_bits))
    recovered_payload, recovered_crc = recovered[: len(payload)], recovered[len(payload):]
    if zlib.crc32(recovered_payload).to_bytes(_CRC_SIZE_BYTES, "big") != recovered_crc:
        return None  # corrupted -- CRC mismatch, drop it rather than deliver garbage

    return recovered_payload


class GnuRadioChannel:
    """One real, simulated RF link. Transmissions go in on `_transmit`;
    whatever survives the channel comes out through `_receive`.

    Keeps two separate histories: `tx_transmissions` (everything actually
    sent -- airtime used, whether or not it arrived) and `rx_transmissions`
    (only what was successfully recovered). This is what lets the TX and RX
    HUD panels show genuinely different things when the channel is noisy.
    """

    def __init__(
        self,
        noise_voltage: float = DEFAULT_NOISE_VOLTAGE,
        samples_per_symbol: int = DEFAULT_SAMPLES_PER_SYMBOL,
        frequency_offset: float = 0.0,
        signal_history_seconds: float = 3.0,
        max_pending: int = 256,
    ) -> None:
        self.noise_voltage = noise_voltage
        self.samples_per_symbol = samples_per_symbol
        self.frequency_offset = frequency_offset
        self._signal_history_seconds = signal_history_seconds

        self._tx_log: Deque[Tuple[float, bytes]] = deque()
        self._rx_log: Deque[Tuple[float, bytes]] = deque()

        self._job_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max_pending)
        self._rx_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max_pending)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # -- transmit side -------------------------------------------------------
    def _transmit(self, payload: bytes) -> None:
        now = time.monotonic()
        self._tx_log.append((now, payload))
        self._prune(self._tx_log, now)
        try:
            self._job_queue.put_nowait(payload)
        except queue.Full:
            pass  # the link is saturated -- a real radio would drop it too

    def _worker_loop(self) -> None:
        while True:
            payload = self._job_queue.get()
            recovered = _run_burst(
                payload, self.samples_per_symbol, self.noise_voltage, self.frequency_offset
            )
            if recovered is None:
                continue
            now = time.monotonic()
            self._rx_log.append((now, recovered))
            self._prune(self._rx_log, now)
            try:
                self._rx_queue.put_nowait(recovered)
            except queue.Full:
                pass

    # -- receive side ---------------------------------------------------------
    def _receive(self) -> Optional[bytes]:
        try:
            return self._rx_queue.get_nowait()
        except queue.Empty:
            return None

    # -- stats used by the HUD -------------------------------------------------
    def _prune(self, log: Deque[Tuple[float, bytes]], now: float) -> None:
        cutoff = now - self._signal_history_seconds
        while log and log[0][0] < cutoff:
            log.popleft()

    def tx_bandwidth_bps(self, window: float = 1.0) -> float:
        return self._bandwidth(self._tx_log, window)

    def rx_bandwidth_bps(self, window: float = 1.0) -> float:
        return self._bandwidth(self._rx_log, window)

    def _bandwidth(self, log: Deque[Tuple[float, bytes]], window: float) -> float:
        now = time.monotonic()
        self._prune(log, now)
        cutoff = now - window
        total_bytes = sum(len(payload) for t, payload in log if t >= cutoff)
        return total_bytes / window if window > 0 else 0.0

    def tx_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, bytes]]:
        return self._recent(self._tx_log, window)

    def rx_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, bytes]]:
        return self._recent(self._rx_log, window)

    def _recent(
        self, log: Deque[Tuple[float, bytes]], window: Optional[float]
    ) -> List[Tuple[float, bytes]]:
        now = time.monotonic()
        self._prune(log, now)
        if window is None:
            return list(log)
        cutoff = now - window
        return [(t, p) for t, p in log if t >= cutoff]
