"""Adaptador de caos de comunicación: detecta qué comando se intentó
transmitir correlacionando la señal raw contra patrones conocidos,
incluso cuando el CRC falla.

Uso:
    adapter = CommChaosAdapter()
    adapter.learn("THR+1", raw_samples)   # registrar patrón
    label, conf = adapter.match(samples)   # (None, 0.0) si no hay match
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class CommChaosAdapter:
    """Correlacionador de señales raw contra patrones conocidos."""

    def __init__(self) -> None:
        self._patterns: Dict[str, np.ndarray] = {}

    def learn(self, label: str, samples: np.ndarray) -> None:
        """Almacena un patrón de señal para un comando."""
        norm = samples - np.mean(samples)
        norm = norm / (np.linalg.norm(norm) + 1e-10)
        self._patterns[label] = norm

    def match(self, samples: np.ndarray) -> Tuple[Optional[str], float]:
        """Correlaciona `samples` contra todos los patrones.

        Returns (label, confidence) de la mejor coincidencia, o
        (None, 0.0) si no hay patrones o la confianza es muy baja.
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
        """Elimina un patrón."""
        self._patterns.pop(label, None)

    def clear(self) -> None:
        """Borra todos los patrones."""
        self._patterns.clear()
