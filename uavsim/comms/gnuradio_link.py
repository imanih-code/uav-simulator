"""The actual physical layer for UAV <-> Operator communication, built on
real GNU Radio signal processing -- not a simulated stand-in.

Every payload handed to `GnuRadioChannel._transmit()` is genuinely:
  1. Framed with a preamble + access code + CRC32 + flush padding.
  2. GMSK-modulated into complex baseband samples (`digital.gmsk_mod`).
  3.    Pushed through a simulated RF channel with AWGN noise
     (`channels.channel_model`). The noise voltage is updated in
     real time by jammers via the `noise_voltage` property.
  4. GMSK-demodulated, bit-synchronized against the access code
     (`digital.correlate_access_code_tag_bb`), and CRC-checked.

Packets that fail to sync, get truncated, or fail their CRC are dropped --
exactly like a real radio link -- rather than being handed to the app as
garbage. This is what makes the drone's control link and telemetry link
"real" instead of a bare in-memory queue.

Each channel (uplink commands, downlink telemetry) runs its own dedicated
**subprocess**. GNU Radio's C++ blocks dominate the per-burst cost
(building/tearing down a tiny flowgraph, not the DSP itself -- measured at
roughly ~10ms/burst regardless of payload size), so a single worker
per channel comfortably keeps up with this app's message rates without
blocking the render loop.

GNU Radio runs in isolated subprocesses via `multiprocessing` so its
internal TPB (thread-per-block) scheduler cannot conflict with the main
process's Pygame/OpenGL context (which would trigger GLXBadContextState).
"""
from __future__ import annotations

import multiprocessing as mp
import multiprocessing.queues
import os
import pickle
import queue
import random
import time
import zlib
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

os.environ.pop("GR_SCHEDULER", None)  # only TPB exists in GR 3.10 — suppress spurious warning
import traceback
from gnuradio import blocks, channels, digital, gr

# -- RF link constants, validated empirically against this exact GNU Radio
# build before being used anywhere else in the app. -------------------------
_PREAMBLE = b"\x55\x55"                 # alternating bits: lets clock recovery settle
_ACCESS_CODE_BYTES = b"\xac\xdd\xa4\xe2"  # arbitrary but fixed sync word
_ACCESS_CODE_BITS = "".join(f"{byte:08b}" for byte in _ACCESS_CODE_BYTES)
_TAIL_PADDING = b"\x55" * 8              # lets the filter pipeline fully flush
_ACCESS_CODE_THRESHOLD = 4               # bit errors tolerated in the sync word
_CRC_SIZE_BYTES = 4

DEFAULT_SAMPLES_PER_SYMBOL = 4
DEFAULT_NOISE_VOLTAGE = 0.005            # AWGN level; jammers raise this in real time


def _run_burst(
    payload: bytes,
    samples_per_symbol: int,
    noise_voltage: float,
    frequency_offset: float,
) -> Tuple[Optional[bytes], np.ndarray]:
    """
    Returns (recovered_payload, raw_real_samples).
    Packets with CRC mismatch are dropped (None).
    """
    crc = zlib.crc32(payload).to_bytes(_CRC_SIZE_BYTES, "big")
    frame = _PREAMBLE + _ACCESS_CODE_BYTES + payload + crc + _TAIL_PADDING

    # gmsk_mod(do_unpack=True) by default internally unpacks bytes to bits
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
    raw_sink = blocks.vector_sink_c(1)

    top_block = gr.top_block()
    top_block.connect(source, modulator, channel)
    top_block.connect(channel, demodulator, correlator, sink)
    top_block.connect(channel, raw_sink)
    top_block.run()

    # Raw signal: real part of the first payload byte (after preamble + access code)
    prefix_bytes = (len(_PREAMBLE) + len(_ACCESS_CODE_BYTES))
    byte0_bytes = 1
    prefix_symbols = prefix_bytes * 8
    byte0_symbols = 8
    start_sample = prefix_symbols * samples_per_symbol
    end_sample = (prefix_symbols + byte0_symbols) * samples_per_symbol
    raw_complex = np.array(raw_sink.data(), dtype=np.complex64)
    clipped = raw_complex[start_sample:end_sample]
    raw_real = np.real(clipped)
    max_abs = np.max(np.abs(raw_real)) if len(raw_real) > 0 else 1.0
    if max_abs > 0:
        raw_real = raw_real / max_abs

    # Demodulated bits — 1 byte per bit (0 or 1)
    bits = np.array(sink.data(), dtype=np.uint8)
    sync_offsets = [tag.offset for tag in sink.tags() if str(tag.key) == "sync"]
    if not sync_offsets:
        return None, raw_real  # no sync

    # In this GR 3.10 build, correlate_access_code_tag_bb tags the first bit
    # AFTER the access code (END).  No skip needed — payload starts here.
    needed_bits = (len(payload) + _CRC_SIZE_BYTES) * 8
    start = sync_offsets[0]
    recovered_bits = bits[start: start + needed_bits]
    if len(recovered_bits) < needed_bits:
        return None, raw_real  # truncated

    recovered = bytes(np.packbits(recovered_bits))
    recovered_payload = recovered[: len(payload)]
    recovered_crc = recovered[len(payload): len(payload) + _CRC_SIZE_BYTES]
    if zlib.crc32(recovered_payload).to_bytes(_CRC_SIZE_BYTES, "big") != recovered_crc:
        return None, raw_real

    return recovered_payload, raw_real


def _worker_process(
    job_queue: "mp.Queue",
    result_queue: "mp.Queue",
    samples_per_symbol: int,
    noise_voltage: float,
    frequency_offset: float,
) -> None:
    """Entrypoint for the GNU Radio subprocess.

    Everything in here runs in an isolated process, so the TPB scheduler's
    internal threads cannot interfere with the main process's GLX context.

    Messages on `job_queue` are tagged tuples:
    - ``("payload", bytes)`` for actual transmissions
    - ``("noise", float)`` to update the noise level
    - ``("reset",)`` to discard any in-flight job and return to idle
    """
    random.seed()
    try:
        result_queue.put(("worker_started",))
    except Exception:
        pass

    while True:
        msg = job_queue.get()
        if isinstance(msg, tuple):
            tag = msg[0]
            if tag == "noise":
                noise_voltage = msg[1]
                continue
            if tag == "reset":
                continue  # discard any in-flight job and wait for next msg
        payload = msg if isinstance(msg, bytes) else msg[1]
        try:
            recovered, raw_samples = _run_burst(
                payload, samples_per_symbol, noise_voltage, frequency_offset
            )
        except Exception as e:
            tb = traceback.format_exc()
            try:
                result_queue.put(("worker_error", repr(e), tb))
            except Exception:
                pass
            continue
        serialised = pickle.dumps(raw_samples)
        result_queue.put((recovered, serialised, time.monotonic()))


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
        self._signal_history_seconds = signal_history_seconds
        self._noise_voltage = noise_voltage

        self._tx_log: Deque[Tuple[float, bytes]] = deque()
        self._rx_log: Deque[Tuple[float, bytes]] = deque()
        self._rx_raw_log: Deque[Tuple[float, np.ndarray]] = deque()

        self._rx_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max_pending)
        self._worker_crashed: bool = False
        self._worker_crashed_info: str = ""

        pending = max(max_pending, 1024)
        ctx = mp.get_context("spawn")
        self._job_queue: "mp.Queue" = ctx.Queue(maxsize=pending)
        self._result_queue: "mp.Queue" = ctx.Queue(maxsize=pending)
        self._worker = ctx.Process(
            target=_worker_process,
            args=(
                self._job_queue,
                self._result_queue,
                samples_per_symbol,
                noise_voltage,
                frequency_offset,
            ),
            daemon=True,
        )
        self._worker.start()
        # Brief polling to confirm the subprocess entered the main loop
        for _ in range(50):
            try:
                item = self._result_queue.get_nowait()
                self._handle_control_message(item)
            except queue.Empty:
                pass
            time.sleep(0.002)

    @property
    def worker_alive(self) -> bool:
        return not self._worker_crashed and self._worker.is_alive()

    @property
    def noise_voltage(self) -> float:
        """Current noise level applied by the GNU Radio subprocess."""
        return self._noise_voltage

    @noise_voltage.setter
    def noise_voltage(self, value: float) -> None:
        """Update the noise level in the subprocess in real time."""
        self._noise_voltage = value
        try:
            self._job_queue.put_nowait(("noise", value))
        except queue.Full:
            pass

    # -- transmit side -------------------------------------------------------
    def _transmit(self, payload: bytes) -> None:
        now = time.monotonic()
        self._tx_log.append((now, payload))
        self._prune(self._tx_log, now)
        try:
            self._job_queue.put_nowait(("payload", payload))
        except queue.Full:
            pass  # the link is saturated -- a real radio would drop it too

    # -- receive side ---------------------------------------------------------
    def _handle_control_message(self, item: tuple) -> None:
        tag = item[0]
        if tag == "worker_error":
            self._worker_crashed = True
            self._worker_crashed_info = f"[{tag}] {item[1]}\n{item[2]}"
            print(self._worker_crashed_info, flush=True)
        # "worker_started" is silently consumed

    def _drain(self) -> None:
        """Drain the result queue, populating logs and the rx queue."""
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item and isinstance(item[0], str):
                self._handle_control_message(item)
                continue
            recovered, serialised, now = item
            raw_samples = pickle.loads(serialised)
            self._rx_raw_log.append((now, raw_samples))
            self._prune(self._rx_raw_log, now)
            if recovered is not None:
                self._rx_log.append((now, recovered))
                self._prune(self._rx_log, now)
                try:
                    self._rx_queue.put_nowait(recovered)
                except queue.Full:
                    pass

    def _receive(self) -> Optional[bytes]:
        self._drain()
        try:
            return self._rx_queue.get_nowait()
        except queue.Empty:
            return None

    # -- reset / clear ---------------------------------------------------------
    def clear(self, reset_noise: float = 0.0) -> None:
        self._tx_log.clear()
        self._rx_log.clear()
        self._rx_raw_log.clear()
        self._discard_all()
        # Reset noise to a clean level so the worker doesn't carry stale
        # high noise from a previous jammer encounter.
        self.noise_voltage = reset_noise
        try:
            self._job_queue.put_nowait(("reset",))
        except queue.Full:
            pass
        time.sleep(0.02)
        self._discard_all()
        # Tell the worker to discard any in-flight job
        try:
            self._job_queue.put_nowait(("reset",))
        except queue.Full:
            pass
        # The worker might finish its current burst AFTER the first drain
        # but BEFORE receiving ("reset",).  Drain again to catch that stray.
        time.sleep(0.02)  # generous for a ~10ms flowgraph
        self._discard_all()

    def _discard_all(self) -> None:
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item and isinstance(item[0], str):
                self._handle_control_message(item)
        while True:
            try:
                self._rx_queue.get_nowait()
            except queue.Empty:
                break

    # -- stats used by the HUD -------------------------------------------------
    def _prune(self, log: Deque[Tuple[float, bytes]], now: float) -> None:
        cutoff = now - self._signal_history_seconds
        while log and log[0][0] < cutoff:
            log.popleft()

    def tx_bandwidth_bps(self, window: float = 1.0) -> float:
        return self._bandwidth(self._tx_log, window)

    def rx_bandwidth_bps(self, window: float = 1.0) -> float:
        return self._bandwidth(self._rx_log, window)

    def rx_raw_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, np.ndarray]]:
        return self._recent(self._rx_raw_log, window)

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
