"""Spreading-chip generators for DSSS.

Two families, so their difference can be measured instead of argued:

* ``LorenzChips`` -- what the simulator used so far: the x component of a
  Lorenz attractor (Euler, dt=0.01) thresholded at its median. The
  attractor evolves slowly compared with the step, so consecutive chips
  are almost always equal (mean run ~150 chips): with N=8 nearly every
  bit is spread by a constant +1 or -1. Against white noise that is still
  a repetition code (same gain), but it does not spread a narrowband
  jammer at all, and anyone who knows the (public) initial conditions can
  reproduce it.

* ``KeyedChips`` -- chips from a counter-based PRNG (Philox) keyed by a
  shared secret and the packet sequence number. Every packet gets a fresh,
  balanced, white-looking sequence; the receiver regenerates it from the
  sequence number alone, so a lost packet never desynchronises the two
  ends (the Lorenz scheme relies on both ends advancing in lockstep).
  Philox is not a cryptographic cipher; a fielded system would use e.g.
  AES-CTR or ChaCha20, which have the same interface.
"""
from __future__ import annotations

import numpy as np

from uavsim.comms.comm_chaos_adapter import ChaoticLayer

# Public sequence used to spread the sync word when the header is spread.
# It must be known in advance by any receiver, so it is not secret.
_HEADER_KEY = 0x5EED_0F_5E


class LorenzChips:
    """Legacy chaotic chips (continuous attractor state across packets)."""

    name = "lorenz"

    def __init__(self) -> None:
        self._tx = ChaoticLayer()
        self._rx = ChaoticLayer()

    def tx(self, n_bits: int, n_chips: int, seq: int) -> np.ndarray:
        return _binarise(self._tx.generate(n_bits * n_chips)).reshape(n_bits, n_chips)

    def rx(self, n_bits: int, n_chips: int, seq: int) -> np.ndarray:
        return _binarise(self._rx.generate(n_bits * n_chips)).reshape(n_bits, n_chips)


class KeyedChips:
    """Stateless keyed chips: f(key, packet seq) -> ±1 sequence."""

    name = "keyed"

    def __init__(self, key: int = 0xC0FFEE) -> None:
        self.key = key

    def tx(self, n_bits: int, n_chips: int, seq: int) -> np.ndarray:
        return keyed_chips(self.key, seq, n_bits * n_chips).reshape(n_bits, n_chips)

    rx = tx  # the receiver computes exactly the same thing from the seq number


def keyed_chips(key: int, counter: int, length: int) -> np.ndarray:
    gen = np.random.Generator(np.random.Philox(key=key, counter=counter))
    return gen.integers(0, 2, size=length, dtype=np.int8).astype(np.float64) * 2.0 - 1.0


def header_chips(n_bits: int, n_chips: int) -> np.ndarray:
    return keyed_chips(_HEADER_KEY, 0, n_bits * n_chips).reshape(n_bits, n_chips)


def _binarise(raw: np.ndarray) -> np.ndarray:
    return np.where(raw > np.median(raw), 1.0, -1.0)


def mean_run_length(chips: np.ndarray) -> float:
    flat = np.asarray(chips).ravel()
    changes = np.flatnonzero(np.diff(flat) != 0)
    if len(changes) < 2:
        return float(len(flat))
    return float(np.mean(np.diff(changes)))
