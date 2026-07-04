"""Simple battery model for the UAV.

Deliberately independent from motors/rigid body: it only tracks a charge
percentage that drains faster the harder the motors are working. This is
what feeds the "battery" reading in telemetry/HUD.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Battery:
    charge_percent: float = 100.0
    idle_drain_rate: float = 0.05     # %/sec drained even at zero throttle
    load_drain_rate: float = 1.2      # %/sec drained at full combined throttle

    def update(self, dt: float, total_throttle: float, max_total_throttle: float) -> None:
        """Drain the battery based on how hard the motors are working.

        `total_throttle` is the sum of every motor's throttle (0..1 each);
        `max_total_throttle` is that sum's maximum (number of motors).
        """
        load_fraction = 0.0 if max_total_throttle <= 0 else total_throttle / max_total_throttle
        drain = (self.idle_drain_rate + self.load_drain_rate * load_fraction) * dt
        self.charge_percent = max(0.0, self.charge_percent - drain)
