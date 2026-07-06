"""Communication chaos adapter: detects which command was attempted by
cross-correlating the raw signal against known patterns, even when the
CRC fails.

Usage:
    adapter = CommChaosAdapter()
    adapter.learn("THR+1", raw_samples)   # register pattern
    label, conf = adapter.match(samples)   # (None, 0.0) if no match
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class CommChaosAdapter:
    """Cross-correlates raw signal chunks against known patterns."""

    def __init__(self) -> None:
        self._patterns: Dict[str, np.ndarray] = {}

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
        """Remove a stored pattern."""
        self._patterns.pop(label, None)

    def clear(self) -> None:
        """Clear all stored patterns."""
        self._patterns.clear()
