from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Jammer:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 50.0
    noise_level: float = 0.5
    cylinder_height: float = 0.5
    cylinder_radius: float = 0.12

    def noise_at(self, pos: np.ndarray) -> float:
        dist = np.linalg.norm(pos[:2] - self.position[:2])
        if dist >= self.radius:
            return 0.0
        return self.noise_level * (1.0 - dist / self.radius)

    def is_in_range(self, pos: np.ndarray) -> bool:
        return np.linalg.norm(pos[:2] - self.position[:2]) <= self.radius
