"""Physical model of a single motor + propeller unit.

This class knows nothing about keyboards, radios or the rest of the
airframe. It receives a *command* in [0, 1] (what an ESC gets from the
flight controller) and produces thrust and a yaw reaction torque.

Two things make it behave like real hardware instead of an ideal force
source:

* **Spin-up lag.** The rotor cannot change speed instantly; its speed
  follows the command as a first-order system with time constant
  ``time_constant`` (~20-50 ms for small brushless motors).
* **Thrust grows with the square of rotor speed.** Command 0.5 gives
  roughly a quarter of max thrust, not half.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Propeller drag torque per newton of thrust (k_Q / k_T, metres). Real
# props sit around 0.01-0.03 m.
YAW_TORQUE_PER_NEWTON = 0.016


@dataclass
class Motor:
    position_body: np.ndarray   # fixed offset from the UAV centre of mass
    max_thrust: float           # newtons at full speed
    spin_direction: int         # +1 clockwise, -1 counter-clockwise (seen from above)
    time_constant: float = 0.035
    command: float = 0.0        # last command from the flight controller, [0, 1]
    speed: float = 0.0          # normalised rotor speed, [0, 1]

    def update(self, dt: float) -> None:
        """Advance the rotor speed toward the command (exact first-order step)."""
        alpha = 1.0 - math.exp(-dt / self.time_constant)
        self.speed += (self.command - self.speed) * alpha

    def reset(self) -> None:
        self.command = 0.0
        self.speed = 0.0

    def thrust(self, efficiency: float = 1.0) -> float:
        return self.max_thrust * efficiency * self.speed * self.speed

    def reaction_torque_z(self, thrust: float) -> float:
        """Yaw torque on the airframe about body Z from spinning the prop."""
        return -self.spin_direction * YAW_TORQUE_PER_NEWTON * thrust
