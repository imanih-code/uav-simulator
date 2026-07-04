"""UAV entity: a quadcopter airframe with 4 motors in an X configuration.

The UAV does not know about keyboards or rendering. It only:
  1. Reads bit-encoded Commands from its CommGatewayInput.
  2. Adjusts the corresponding motor's throttle.
  3. Lets RigidBody physics turn motor thrust into real motion -- no
     movement direction is ever scripted explicitly.
  4. Stays resting on the ground until its own thrust can lift it (via
     `apply_ground_contact`, a world-level concern, not a UAV one).
  5. Periodically reports its own state -- including battery and total
     mass -- as telemetry through its CommGatewayOutput.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from uavsim.comms.command import Command, CommandOpcode
from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.telemetry import TelemetryPacket
from uavsim.physics.battery import Battery
from uavsim.physics.motor import Motor
from uavsim.physics.rigid_body import RigidBody
from uavsim.world.environment import (
    FlatGroundPlane,
    WORLD_EXTENT_HALF,
    apply_ground_contact,
    apply_world_bounds,
)


class MotorId:
    """Named indices matching the X airframe layout below."""

    FRONT_RIGHT = 0
    BACK_RIGHT = 1
    BACK_LEFT = 2
    FRONT_LEFT = 3


@dataclass
class UAVConfig:
    body_mass: float = 1.0                  # kg, airframe + electronics
    motor_mass: float = 0.05                # kg, each of the 4 motors
    inertia_diag: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.04])
    )
    arm_length: float = 0.25                # meters, center of mass to motor
    max_thrust_per_motor: float = 6.0       # newtons
    telemetry_interval: float = 0.1         # seconds between telemetry packets

    @property
    def total_mass(self) -> float:
        """Body + all 4 motors, lumped into a single mass for the physics.

        Each motor does have its own (smaller) mass, but for the rigid
        body equations we treat the airframe as one uniform point mass at
        the center of gravity -- we're not modeling each motor's own
        inertial contribution separately.
        """
        return self.body_mass + 4 * self.motor_mass


def _default_motor_layout(arm_length: float, max_thrust: float) -> List[Motor]:
    """Build the 4 motors in a standard X airframe layout.

    Positions are in the body frame, relative to the drone's center of
    mass (the "red point" of the simplified 3D model). +X is forward,
    +Y is left, +Z is up. Motors are placed diagonally, X-style, and
    alternate spin direction so their reaction torques mostly cancel.
    """
    offset = arm_length / np.sqrt(2)
    return [
        Motor(np.array([offset, -offset, 0.0]), max_thrust, spin_direction=+1),   # front-right
        Motor(np.array([-offset, -offset, 0.0]), max_thrust, spin_direction=-1),  # back-right
        Motor(np.array([-offset, offset, 0.0]), max_thrust, spin_direction=+1),   # back-left
        Motor(np.array([offset, offset, 0.0]), max_thrust, spin_direction=-1),    # front-left
    ]


class UAV:
    def __init__(
        self,
        config: UAVConfig,
        ground: FlatGroundPlane,
        command_input: CommGatewayInput,
        telemetry_output: CommGatewayOutput,
    ) -> None:
        self.config = config
        self.ground = ground
        self.motors = _default_motor_layout(config.arm_length, config.max_thrust_per_motor)
        self.body = RigidBody(mass=config.total_mass, inertia_diag=config.inertia_diag)
        self.battery = Battery()

        # The UAV starts parked on the ground, at rest, exactly like a real
        # drone before takeoff -- not mid-air, not already falling.
        self.body.position = np.array([0.0, 0.0, ground.ground_z])

        self._command_input = command_input
        self._telemetry_output = telemetry_output
        self._time_since_last_telemetry = 0.0
        self._armed = True
        self.is_grounded = True

    # -- command handling --------------------------------------------------
    def _process_incoming_commands(self) -> None:
        for raw_packet in self._command_input.receive_all():
            try:
                command = Command.decode(raw_packet)
            except ValueError:
                continue  # corrupted-but-CRC-valid-by-luck packet; drop it
            self._apply_command(command)

    def _apply_command(self, command: Command) -> None:
        if command.opcode == CommandOpcode.THROTTLE_UP:
            self.motors[command.motor_id].increase_throttle()
        elif command.opcode == CommandOpcode.THROTTLE_DOWN:
            self.motors[command.motor_id].decrease_throttle()
        elif command.opcode == CommandOpcode.THROTTLE_UP_ALL:
            for motor in self.motors:
                motor.increase_throttle()
        elif command.opcode == CommandOpcode.THROTTLE_DOWN_ALL:
            for motor in self.motors:
                motor.decrease_throttle()
        elif command.opcode == CommandOpcode.ARM:
            self._armed = True
        elif command.opcode == CommandOpcode.DISARM:
            self._armed = False
        elif command.opcode == CommandOpcode.EMERGENCY_CUT:
            self._armed = False
            for motor in self.motors:
                motor.throttle = 0.0

    # -- physics -------------------------------------------------------------
    def _integrate_physics(self, dt: float) -> None:
        if not self._armed:
            for motor in self.motors:
                motor.throttle = 0.0

        forces_body = [motor.thrust_force_body() for motor in self.motors]
        points_body = [motor.position_body for motor in self.motors]
        yaw_reaction_torque = np.array(
            [0.0, 0.0, sum(motor.reaction_torque_z() for motor in self.motors)]
        )
        self.body.apply_body_forces(dt, forces_body, points_body, yaw_reaction_torque)
        self.is_grounded = apply_ground_contact(self.body, self.ground, dt)
        apply_world_bounds(self.body, WORLD_EXTENT_HALF)

    # -- battery ---------------------------------------------------------------
    def _update_battery(self, dt: float) -> None:
        total_throttle = sum(motor.throttle for motor in self.motors)
        self.battery.update(dt, total_throttle, max_total_throttle=len(self.motors))

    # -- telemetry ------------------------------------------------------------
    def _send_telemetry_if_due(self, dt: float) -> None:
        self._time_since_last_telemetry += dt
        if self._time_since_last_telemetry < self.config.telemetry_interval:
            return
        self._time_since_last_telemetry = 0.0

        packet = TelemetryPacket(
            timestamp=time.time(),
            position=self.body.position.copy(),
            velocity=self.body.velocity.copy(),
            attitude_rpy=self.body.attitude_rpy(),
            angular_velocity=self.body.angular_velocity_body.copy(),
            motor_throttle=np.array([m.throttle for m in self.motors]),
            battery_percent=self.battery.charge_percent,
            mass=self.config.total_mass,
        )
        self._telemetry_output.send(packet.encode())

    # -- public API ------------------------------------------------------------
    def update(self, dt: float) -> None:
        """Advance the UAV by one simulation tick of length `dt` seconds."""
        if dt <= 0.0:
            return
        self._process_incoming_commands()
        self._integrate_physics(dt)
        self._update_battery(dt)
        self._send_telemetry_if_due(dt)

    def motor_world_positions(self) -> List[np.ndarray]:
        """Motor mount points in world space -- used only by the renderer."""
        return [
            self.body.position + self.body.orientation.apply(motor.position_body)
            for motor in self.motors
        ]
