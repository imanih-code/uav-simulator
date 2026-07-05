"""Physical model of a single UAV motor + propeller unit.

This class knows nothing about keyboards, commands, or the rest of the
airframe. It only tracks its own throttle and turns that into a thrust
vector and a small yaw reaction torque. The UAV feeds these into a
RigidBody; the resulting motion (translation, pitch, roll, yaw) is never
scripted directly, it simply falls out of the physics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Simplified reaction-torque model: real propellers resist the motor's
# spin with a torque roughly proportional to thrust. We keep this as a
# tunable constant rather than a full aerodynamic model.
_YAW_REACTION_COEFFICIENT = 0.02


@dataclass
class Motor:
    """A motor mounted at a fixed position on the UAV chassis (body frame).

    Throttle changes in fixed steps per received command rather than a
    rate scaled by frame time. That's because commands now travel over a
    real (GNU Radio) radio link and arrive asynchronously, at whatever
    rate the operator's transmitter manages -- not once per render frame.
    Holding a key just means "keep transmitting THROTTLE_UP", and each
    delivered command nudges the throttle by one fixed step, exactly like
    a real digital RC link.
    """

    position_body: np.ndarray   # fixed offset from the UAV center of mass
    max_thrust: float            # newtons, thrust delivered at throttle == 1.0
    spin_direction: int          # +1 clockwise, -1 counter-clockwise
    throttle_step_up: float = 0.03     # throttle units gained per THROTTLE_UP command
    throttle_step_down: float = 0.03   # throttle units lost per THROTTLE_DOWN command
    throttle: float = 0.0              # current throttle, clamped to [0, 1]

    def increase_throttle(self) -> None:
        """Numeric-key control: one command = one step up, capped at max."""
        self.throttle = min(1.0, self.throttle + self.throttle_step_up)

    def decrease_throttle(self) -> None:
        """Letter-key control: one command = one step down, floored at 0."""
        self.throttle = max(0.0, self.throttle - self.throttle_step_down)

    @property
    def thrust_magnitude(self) -> float:
        return self.throttle * self.max_thrust

    def thrust_force_body(self) -> np.ndarray:
        """Thrust force vector in the body frame.

        Every motor pushes along the airframe's local +Z axis. As the
        airframe rotates, this vector rotates along with it (handled by
        the RigidBody, not here) -- that rotation is precisely what lets
        differential thrust turn into lateral motion.
        """
        return np.array([0.0, 0.0, self.thrust_magnitude])

    def reaction_torque_z(self) -> float:
        """Yaw reaction torque about the body Z axis from spinning the prop."""
        return -self.spin_direction * _YAW_REACTION_COEFFICIENT * self.thrust_magnitude
