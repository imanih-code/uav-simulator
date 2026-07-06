from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Jammer:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 80.0
    center_noise: float = 0.8
    edge_noise: float = 0.15
    cylinder_height: float = 0.5
    cylinder_radius: float = 0.12

    def noise_at(self, pos: np.ndarray) -> float:
        if not self.is_in_range(pos):
            return 0.0
        t = np.linalg.norm(pos - self.position) / self.radius  # 0 at center, 1 at edge
        # Linear interpolation: center_noise at t=0, edge_noise at t=1
        return self.center_noise + (self.edge_noise - self.center_noise) * t

    def is_in_range(self, pos: np.ndarray) -> bool:
        return np.linalg.norm(pos - self.position) <= self.radius
