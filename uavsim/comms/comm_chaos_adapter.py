from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


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
