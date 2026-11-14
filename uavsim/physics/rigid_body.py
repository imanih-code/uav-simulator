"""Generic 6-degrees-of-freedom rigid body dynamics.

Deliberately independent from anything UAV-specific: it only knows how to
integrate forces/torques applied at arbitrary points, expressed in the
body frame, into linear and angular motion. Orientation math is delegated
to `scipy.spatial.transform.Rotation` and vector math to `numpy` instead of
hand-rolling quaternion/rotation-matrix code, per the "use a reliable
library for physics math" rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

GRAVITY = np.array([0.0, 0.0, -9.81])  # m/s^2, world frame (Z is up)


@dataclass
class RigidBody:
    mass: float
    inertia_diag: np.ndarray  # principal moments of inertia [Ixx, Iyy, Izz]

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: Rotation = field(default_factory=Rotation.identity)
    angular_velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Passive aerodynamic drag: air resists existing velocity and spin.
    # Linear term ~ skin/induced drag at low speed, quadratic term ~
    # 0.5 * rho * Cd * A for a small multirotor. Attitude stability comes
    # from the flight controller, not from drag, so angular drag is small.
    linear_drag_coefficient: float = 0.10     # N per (m/s)
    quadratic_drag_coefficient: float = 0.03  # N per (m/s)^2
    angular_drag_coefficient: float = 0.002   # N*m per (rad/s)

    def apply_body_forces(
        self,
        dt: float,
        forces_body: Sequence[np.ndarray],
        application_points_body: Sequence[np.ndarray],
        extra_torque_body: Optional[np.ndarray] = None,
    ) -> None:
        """Integrate one physics step from forces expressed in body frame.

        `forces_body[i]` is applied at `application_points_body[i]` (both
        relative to the body frame / center of mass). A UAV simply hands
        over its per-motor thrust vectors and mount positions; this class
        has no idea motors even exist.
        """
        forces_body = list(forces_body)
        application_points_body = list(application_points_body)
        if extra_torque_body is None:
            extra_torque_body = np.zeros(3)

        net_force_body = np.sum(forces_body, axis=0) if forces_body else np.zeros(3)
        net_torque_body = extra_torque_body.copy()
        for force, point in zip(forces_body, application_points_body):
            net_torque_body += np.cross(point, force)

        self._integrate_linear_motion(dt, net_force_body)
        self._integrate_angular_motion(dt, net_torque_body)

    # -- internal steps ---------------------------------------------------
    def _integrate_linear_motion(self, dt: float, net_force_body: np.ndarray) -> None:
        net_force_world = self.orientation.apply(net_force_body)
        net_force_world = net_force_world + self.mass * GRAVITY
        speed = float(np.linalg.norm(self.velocity))
        drag_force_world = -(
            self.linear_drag_coefficient + self.quadratic_drag_coefficient * speed
        ) * self.velocity
        linear_acceleration = (net_force_world + drag_force_world) / self.mass

        self.velocity = self.velocity + linear_acceleration * dt
        self.position = self.position + self.velocity * dt

    def _integrate_angular_motion(self, dt: float, net_torque_body: np.ndarray) -> None:
        inertia = self.inertia_diag
        omega = self.angular_velocity_body

        # Euler's rigid body equation: I*omega_dot + omega x (I*omega) = torque
        gyroscopic_term = np.cross(omega, inertia * omega)
        drag_torque = -self.angular_drag_coefficient * omega
        angular_acceleration = (net_torque_body + drag_torque - gyroscopic_term) / inertia

        self.angular_velocity_body = omega + angular_acceleration * dt

        delta_rotation = Rotation.from_rotvec(self.angular_velocity_body * dt)
        self.orientation = self.orientation * delta_rotation

    def attitude_rpy(self) -> np.ndarray:
        """Roll, pitch, yaw in radians -- convenient for HUD/telemetry."""
        return self.orientation.as_euler("xyz")
