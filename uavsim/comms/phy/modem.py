"""Bit-true baseband modem: every chip of every packet is simulated.

Frame on air (same structure as the GNU Radio link):

    [32-bit sync word][payload + CRC32 (optionally convolutionally coded)]
     each bit optionally spread into N chips (DSSS)

Channel per chip ``n``:

    r[n] = s[n] + sigma * w[n] + A * cos(2*pi*f*n + phi)

* ``s`` = ±1 chips, ``w`` = white Gaussian noise. ``sigma`` comes from the
  link budget's noise voltage with the same implementation-loss factor
  used to calibrate against GNU Radio's GMSK modem (``K = 0.41``), so
  without mitigation this reproduces the old statistical model.
* ``A cos(...)`` = a narrowband (CW tone) jammer, if any.

Receiver: sync-word check (<= 4 bit errors), despreading (soft = correlate
real values; hard = majority vote on chip signs, which is what the GNU
Radio implementation does), optional soft Viterbi, CRC32.

Cost model: spreading by N and coding at rate 1/2 multiply the number of
chips -- i.e. airtime, or equivalently 1/throughput at a fixed chip rate.
``BurstResult.chips`` reports it so comparisons can be made at equal
airtime.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uavsim.comms.phy import fec
from uavsim.comms.phy.sequences import KeyedChips, LorenzChips, header_chips

SYNC_WORD = b"\xac\xdd\xa4\xe2"            # same access code as the GNU Radio link
SYNC_BITS = np.unpackbits(np.frombuffer(SYNC_WORD, dtype=np.uint8))
SYNC_THRESHOLD = 4                          # bit errors tolerated in the sync word
IMPLEMENTATION_LOSS_K = 0.41                # p_chip = 0.5*erfc(sqrt(K)/noise_voltage)
DEFAULT_TONE_FREQ = 0.0021                  # cycles per chip (near-carrier CW jammer)


@dataclass(frozen=True)
class PhyProfile:
    name: str
    dsss_n: int = 0                 # chips per (coded) bit; 0 = no spreading
    sequence: str = "keyed"         # "keyed" or "lorenz"
    soft: bool = True               # soft (correlation) vs hard (majority) despreading
    spread_header: bool = False     # spread the sync word too
    fec: bool = False               # K=7 rate-1/2 convolutional code
    label: str = ""
    # SINR improvement for 50% delivery of an 11-byte RC frame against
    # wideband noise, measured with `python -m uavsim.experiments.link_curves`
    # (docs/results/link_summary.md). Used to shift the HUD's link colours.
    measured_gain_db: float = 0.0

    def airtime_factor(self, payload_bytes: int) -> float:
        """Chips on air relative to the unprotected frame of the same payload."""
        data = (payload_bytes + 4) * 8
        base = len(SYNC_BITS) + data
        coded = 2 * (data + fec.TAIL_BITS) if self.fec else data
        n = max(self.dsss_n, 1)
        header = len(SYNC_BITS) * (n if self.spread_header else 1)
        return (header + coded * n) / base


PROFILES = {
    p.name: p for p in (
        PhyProfile("none", label="NONE"),
        PhyProfile("dsss-lorenz", dsss_n=8, sequence="lorenz", soft=False,
                   spread_header=False, label="DSSS8 LORENZ", measured_gain_db=6.2),
        PhyProfile("dsss-keyed", dsss_n=8, sequence="keyed", soft=True,
                   spread_header=True, label="DSSS8 KEYED", measured_gain_db=9.0),
        PhyProfile("fec", fec=True, label="FEC R1/2", measured_gain_db=6.8),
        PhyProfile("dsss+fec", dsss_n=4, sequence="keyed", soft=True,
                   spread_header=True, fec=True, label="DSSS4+FEC", measured_gain_db=12.8),
    )
}
PROFILE_ORDER = tuple(PROFILES)


@dataclass(frozen=True)
class BurstResult:
    payload: Optional[bytes]   # None if sync or CRC failed
    synced: bool
    correlation: float         # mean |despread value| in [0, 1]; 1 = clean
    chips: int                 # chips on air (airtime)
    preview: np.ndarray        # first received samples, normalised, for the HUD scope


def chip_sigma(noise_voltage: float) -> float:
    return noise_voltage / math.sqrt(2.0 * IMPLEMENTATION_LOSS_K)


class NumpyPhy:
    def __init__(self, profile: PhyProfile, rng: Optional[np.random.Generator] = None,
                 key: int = 0xC0FFEE) -> None:
        self.profile = profile
        self.rng = rng or np.random.default_rng()
        self._chips = LorenzChips() if profile.sequence == "lorenz" else KeyedChips(key)

    def burst(self, payload: bytes, seq: int, noise_voltage: float,
              tone_amplitude: float = 0.0, tone_freq: float = DEFAULT_TONE_FREQ) -> BurstResult:
        p = self.profile
        crc = zlib.crc32(payload).to_bytes(4, "big")
        data_bits = np.unpackbits(np.frombuffer(payload + crc, dtype=np.uint8))
        coded = fec.encode(data_bits) if p.fec else data_bits
        n = p.dsss_n

        # -- transmitter -------------------------------------------------
        sync_sym = SYNC_BITS.astype(np.float64) * 2.0 - 1.0
        data_sym = coded.astype(np.float64) * 2.0 - 1.0
        if n > 0:
            hdr_seq = header_chips(len(SYNC_BITS), n) if p.spread_header else None
            tx_header = (sync_sym[:, None] * hdr_seq).ravel() if p.spread_header else sync_sym
            tx_data = (data_sym[:, None] * self._chips.tx(len(coded), n, seq)).ravel()
            rx_seq = self._chips.rx(len(coded), n, seq)  # always advance (Lorenz lockstep)
        else:
            tx_header, tx_data, hdr_seq, rx_seq = sync_sym, data_sym, None, None
        tx = np.concatenate([tx_header, tx_data])

        # -- channel ---------------------------------------------------------
        r = tx + chip_sigma(noise_voltage) * self.rng.standard_normal(tx.size)
        if tone_amplitude > 0.0:
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            r = r + tone_amplitude * np.cos(2.0 * np.pi * tone_freq * np.arange(tx.size) + phase)
        preview = r[:32] / (np.max(np.abs(r[:32])) or 1.0)

        # -- receiver ----------------------------------------------------------
        r_header, r_data = r[: tx_header.size], r[tx_header.size:]
        if n > 0 and p.spread_header:
            header_soft = np.sum(r_header.reshape(-1, n) * hdr_seq, axis=1)
        else:
            header_soft = r_header
        errors = int(np.sum((header_soft > 0) != SYNC_BITS.astype(bool)))
        if errors > SYNC_THRESHOLD:
            return BurstResult(None, False, 0.0, tx.size, preview)

        if n > 0:
            blocks = r_data.reshape(-1, n)
            values = blocks if p.soft else np.sign(blocks)
            soft = np.sum(values * rx_seq, axis=1) / n
        else:
            soft = r_data
        correlation = float(min(1.0, np.mean(np.abs(soft))))

        if p.fec:
            bits = fec.decode(soft if p.soft else np.sign(soft))
        else:
            bits = (soft > 0).astype(np.uint8)
        recovered = np.packbits(bits).tobytes()
        body, rx_crc = recovered[: len(payload)], recovered[len(payload): len(payload) + 4]
        ok = zlib.crc32(body).to_bytes(4, "big") == rx_crc
        return BurstResult(body if ok else None, True, correlation, tx.size, preview)
