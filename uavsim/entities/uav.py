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
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

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
    body_mass: float = 0.22                 # kg, airframe + electronics
    motor_mass: float = 0.012               # kg, each of the 4 motors
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
        self._raw_throttle: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._armed = True
        self.is_grounded = True
        self._command_log: deque = deque(maxlen=10)
        self._was_airborne = False
        self._health = 100.0
        self._dead = False

    # -- command handling --------------------------------------------------
    def _process_incoming_commands(self) -> None:
        if self._dead:
            return
        for raw_packet in self._command_input.receive_all():
            try:
                command = Command.decode(raw_packet)
            except ValueError:
                self._command_log.append(
                    (CommandOpcode.EMERGENCY_CUT, 0, False)
                )
                continue
            self._apply_command(command)

    def _apply_command(self, command: Command) -> None:
        if self._dead:
            return
        self._command_log.append((int(command.opcode), command.motor_id, True))
        if command.opcode == CommandOpcode.THROTTLE_UP:
            self._raw_throttle[command.motor_id] = min(
                1.0, self._raw_throttle[command.motor_id] + self.motors[0].throttle_step_up
            )
        elif command.opcode == CommandOpcode.THROTTLE_DOWN:
            self._raw_throttle[command.motor_id] = max(
                0.0, self._raw_throttle[command.motor_id] - self.motors[0].throttle_step_down
            )
        elif command.opcode == CommandOpcode.THROTTLE_UP_ALL:
            for i in range(4):
                self._raw_throttle[i] = min(1.0, self._raw_throttle[i] + self.motors[0].throttle_step_up)
        elif command.opcode == CommandOpcode.THROTTLE_DOWN_ALL:
            for i in range(4):
                self._raw_throttle[i] = max(0.0, self._raw_throttle[i] - self.motors[0].throttle_step_down)
        elif command.opcode == CommandOpcode.ARM:
            self._armed = True
        elif command.opcode == CommandOpcode.DISARM:
            self._armed = False
        elif command.opcode == CommandOpcode.EMERGENCY_CUT:
            self._armed = False
            for i in range(4):
                self._raw_throttle[i] = 0.0

    # -- hover controller ----------------------------------------------------

    # -- hover controller ----------------------------------------------------
    # Gain that opposes vertical velocity to produce a natural altitude hold.
    # Higher values = stiffer hold; too high can oscillate.
    _HOVER_GAIN = 0.4

    # -- physics -------------------------------------------------------------
    def _integrate_physics(self, dt: float) -> None:
        if self._dead:
            for motor in self.motors:
                motor.throttle = 0.0
            self.body.apply_body_forces(dt, [], [], None)
            self.is_grounded = apply_ground_contact(self.body, self.ground, dt)
            apply_world_bounds(self.body, WORLD_EXTENT_HALF)
            return

        if self._armed:
            for i in range(4):
                self.motors[i].throttle = self._raw_throttle[i]
        else:
            for motor in self.motors:
                motor.throttle = 0.0

        # Pre-contact velocity for crash detection
        pre_vel_z = self.body.velocity[2]

        # Hover correction: damps upward velocity so the drone doesn't
        # runaway when throttle is above hover level. Does NOT oppose
        # downward motion — when throttle is cut the drone falls freely.
        hover_correction = -max(self.body.velocity[2], 0.0) * self._HOVER_GAIN
        effective_throttles = [
            max(0.0, min(1.0, m.throttle + hover_correction))
            for m in self.motors
        ]

        # Damage reduces effective max thrust
        dmg_factor = 1.0 - (100.0 - self._health) * 0.005  # 1.0 at 100% hp, 0.5 at 0% hp
        effective_max = self.config.max_thrust_per_motor * dmg_factor

        points_body = [motor.position_body for motor in self.motors]
        forces_body = [
            np.array([0.0, 0.0, t * effective_max])
            for t, _ in zip(effective_throttles, self.motors)
        ]
        yaw_reaction_torque = np.array([
            0.0, 0.0,
            sum(-m.spin_direction * 0.02 * t for t, m in zip(effective_throttles, self.motors)),
        ])
        self.body.apply_body_forces(dt, forces_body, points_body, yaw_reaction_torque)

        # Crash detection
        was_grounded = self.is_grounded
        self.is_grounded = apply_ground_contact(self.body, self.ground, dt)
        if not was_grounded and self.is_grounded and pre_vel_z < -3.0:
            impact = abs(pre_vel_z) - 3.0
            damage = impact * 6.0
            self._health = max(0.0, self._health - damage)
            if self._health <= 0.0:
                self._dead = True

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
            command_log=tuple(self._command_log),
            health_percent=self._health,
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

    def reset(self) -> None:
        """Reset the UAV to its initial parked-on-ground state."""
        self.body.position = np.array([0.0, 0.0, self.ground.ground_z])
        self.body.velocity = np.zeros(3)
        self.body.orientation = Rotation.identity()
        self.body.angular_velocity_body = np.zeros(3)
        for motor in self.motors:
            motor.throttle = 0.0
        self._raw_throttle = [0.0, 0.0, 0.0, 0.0]
        self._last_pair_cmd_time = [0.0, 0.0]
        self.battery.charge_percent = 100.0
        self._health = 100.0
        self._dead = False
        self.is_grounded = True
        self._command_log.clear()
        self._was_airborne = False
        # Drain any pending command packets
        self._command_input.receive_all()

    @property
    def command_input(self) -> CommGatewayInput:
        return self._command_input

    def motor_world_positions(self) -> List[np.ndarray]:
        """Motor mount points in world space -- used only by the renderer."""
        return [
            self.body.position + self.body.orientation.apply(motor.position_body)
            for motor in self.motors
        ]
