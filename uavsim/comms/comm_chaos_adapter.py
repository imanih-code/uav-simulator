from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class SpreadingSequence:
    """Generates ±1 spreading chips from a Lorenz attractor.

    Both sides share identical parameters so their sequences are
    identical — the "shared language" that rejects jammers.
    """

    def __init__(
        self,
        lorenz: ChaoticLayer,
        N: int = 256,
    ) -> None:
        self._lorenz = lorenz
        self.N = N

    def generate(self, n_bits: int) -> np.ndarray:
        """Produce ``n_bits × N`` chips, reshaped to ``(n_bits, N)``, values ±1."""
        raw = self._lorenz.generate(n_bits * self.N)
        threshold = np.median(raw)
        return np.where(raw > threshold, 1.0, -1.0).reshape(n_bits, self.N)


class DSSSChaotic:
    """DSSS modem that spreads/despreads bits using chaotic sequences.

    Each bit is multiplied by N chaotic chips (the "shared secret").
    At the receiver, correlation collapses the signal while AWGN averages
    to zero — the matched filter is optimal for AWGN.

    Parameters
    ----------
    sequence : SpreadingSequence
        Source of ±1 chaotic chips (wraps a ChaoticLayer).
    """

    def __init__(self, sequence: SpreadingSequence) -> None:
        self._seq = sequence

    @property
    def N(self) -> int:
        return self._seq.N

    def spread(self, bits: np.ndarray) -> np.ndarray:
        """Spread each bit into ``N`` chips.

        Parameters
        ----------
        bits : np.ndarray
            uint8 array of 0/1 values, shape ``(n_bits,)``.

        Returns
        -------
        np.ndarray
            uint8 chips, shape ``(n_bits * N,)``, ready for GMSK modulation.
        """
        n_bits = len(bits)
        seq = self._seq.generate(n_bits)               # (n_bits, N), ±1
        bits_f = 2.0 * bits.astype(np.float64) - 1.0   # (n_bits,), ±1
        chips_f = bits_f[:, np.newaxis] * seq           # (n_bits, N), ±1
        chips_f = chips_f.ravel()                       # (n_bits * N,)
        return ((chips_f + 1.0) * 0.5).astype(np.uint8)

    def despread(self, chips: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recover bits from noisy chips via correlation.

        Parameters
        ----------
        chips : np.ndarray
            uint8 array of 0/1 values, shape ``(n_chips,)``.
            Length should be a multiple of ``N``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (bits, raw_correlation)
            bits : uint8 array, shape ``(n_chips // N,)``
            raw_correlation : float64 array, per-bit correlation values
        """
        n_chips = len(chips)
        n_bits = n_chips // self.N
        seq = self._seq.generate(n_bits)               # (n_bits, N), ±1
        chips_f = 2.0 * chips[:n_bits * self.N].astype(np.float64) - 1.0
        blocks = chips_f.reshape(n_bits, self.N)       # (n_bits, N)
        corr = np.sum(blocks * seq, axis=1)             # (n_bits,)
        return (corr > 0).astype(np.uint8), corr


class ChaoticLayer:
    """Lorenz attractor as a shared secret bridge between Operator and UAV.

    Both sides hold identical parameters (sigma, rho, beta) and the same
    initial conditions.  Integrating forward produces identical chaotic
    sequences — the "secret language" that only they share.

    The attractor state (x, y, z) persists across calls so the sequence
    is continuous, not reset per message.

    Lorenz equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
    """

    def __init__(
        self,
        x0: float = 1.0,
        y0: float = 1.0,
        z0: float = 1.0,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
    ) -> None:
        self._sigma = sigma
        self._rho = rho
        self._beta = beta
        self._dt = dt
        self._state = np.array([x0, y0, z0], dtype=np.float64)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @state.setter
    def state(self, s: np.ndarray) -> None:
        self._state[:] = s

    def step(self) -> np.ndarray:
        x, y, z = self._state
        dx = self._sigma * (y - x)
        dy = x * (self._rho - z) - y
        dz = x * y - self._beta * z
        self._state += self._dt * np.array([dx, dy, dz])
        return self._state.copy()

    def generate(self, length: int) -> np.ndarray:
        out = np.empty(length, dtype=np.float64)
        for i in range(length):
            self.step()
            out[i] = self._state[0]
        return out

    def reset(self, x0: float = 1.0, y0: float = 1.0, z0: float = 1.0) -> None:
        self._state[:] = [x0, y0, z0]


class CommChaosAdapter:
    """Cross-correlates raw signal chunks against known patterns.

    Combined with ChaoticLayer this becomes a simple "secret bridge":
    both sides evolve the same Lorenz attractor and use its output to
    generate/expect particular raw-signal shapes.
    """

    def __init__(self) -> None:
        self._patterns: Dict[str, np.ndarray] = {}
        self.lorenz = ChaoticLayer()

    def learn(self, label: str, samples: np.ndarray) -> None:
        """Store a signal pattern for a command label."""
        norm = samples - np.mean(samples)
        norm = norm / (np.linalg.norm(norm) + 1e-10)
        self._patterns[label] = norm

    def match(self, samples: np.ndarray) -> Tuple[Optional[str], float]:
        """Cross-correlate `samples` against all known patterns.

        Returns (label, confidence) of the best match, or
        (None, 0.0) if no patterns exist or confidence is too low.
        """
        if not self._patterns:
            return None, 0.0

        query = samples - np.mean(samples)
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-10:
            return None, 0.0
        query = query / q_norm

        best_label: Optional[str] = None
        best_conf = 0.0

        for label, pat in self._patterns.items():
            corr = np.correlate(query, pat, mode="valid")
            peak = float(np.max(np.abs(corr)))
            if peak > best_conf:
                best_conf = peak
                best_label = label

        return (best_label, best_conf) if best_conf > 0.3 else (None, 0.0)

    def forget(self, label: str) -> None:
        self._patterns.pop(label, None)

    def clear(self) -> None:
        self._patterns.clear()
