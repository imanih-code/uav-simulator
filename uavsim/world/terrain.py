"""Rolling terrain: a smooth, deterministic heightfield.

Built from a handful of Gaussian hills plus a gentle long-wavelength
ripple, then flattened around the take-off point so the ground station
and the home pad sit on level ground. Heights stay below ``max_height``
so the default RTH altitude (15 m) always clears the hills.

`height_at` / `normal_at` accept scalars or numpy arrays, so the same
object serves the physics (one point per tick), the RF model and the
renderer (whole grids).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FlatGroundPlane:
    ground_z: float = 0.0

    def height_at(self, x, y):
        return np.zeros_like(np.asarray(x, dtype=float)) + self.ground_z

    def normal_at(self, x, y) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Terrain:
    seed: int = 11
    max_height: float = 9.0
    flat_radius: float = 30.0      # m around the origin kept perfectly flat
    blend_distance: float = 30.0   # m over which the hills fade in
    hills: int = 16
    ground_z: float = 0.0          # height of the flat home area
    _params: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        centres = rng.uniform(-210.0, 210.0, size=(self.hills, 2))
        sigmas = rng.uniform(18.0, 55.0, size=self.hills)
        heights = rng.uniform(2.0, self.max_height, size=self.hills)
        ripple_phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
        object.__setattr__(self, "_params", (centres, sigmas, heights, ripple_phase))

    def height_at(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        centres, sigmas, heights, phase = self._params
        h = np.zeros(np.broadcast(x, y).shape)
        for (cx, cy), s, a in zip(centres, sigmas, heights):
            h = h + a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * s * s))
        h = h + 0.8 * (1.0 + np.sin(x / 37.0 + phase[0]) * np.cos(y / 43.0 + phase[1]))
        h = np.minimum(h, self.max_height)
        r = np.hypot(x, y)
        t = np.clip((r - self.flat_radius) / self.blend_distance, 0.0, 1.0)
        mask = t * t * (3.0 - 2.0 * t)  # smoothstep: 0 inside the flat area
        return self.ground_z + h * mask

    def normal_at(self, x, y) -> np.ndarray:
        eps = 0.25
        dhdx = (self.height_at(x + eps, y) - self.height_at(x - eps, y)) / (2.0 * eps)
        dhdy = (self.height_at(x, y + eps) - self.height_at(x, y - eps)) / (2.0 * eps)
        n = np.array([-float(dhdx), -float(dhdy), 1.0])
        return n / np.linalg.norm(n)
