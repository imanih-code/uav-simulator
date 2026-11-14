"""UAV entity: a quadcopter airframe with 4 motors in an X configuration.

The UAV does not know about keyboards or rendering. Every fixed physics
tick it:

  1. Drains RC frames from its CommGatewayInput (absolute stick positions,
     see `uavsim.control.rc`) and feeds the link monitor.
  2. Handles arming / disarming from the pilot's arm switch.
  3. Lets the failsafe decide who is in control: the pilot's last sticks,
     a self-levelling hold, or the autopilot (HOLD / LAND / RTH / DROP).
  4. Runs the onboard flight controller, which turns that into 4 motor
     commands -- the pilot never drives motors directly.
  5. Integrates motor spin-up, thrust and torque on the rigid body; the
     motion still falls out of the physics, nothing is scripted.
  6. Periodically reports its state as telemetry through its
     CommGatewayOutput.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.telemetry import TelemetryPacket
from uavsim.control.failsafe import Failsafe, FailsafeParams, FailsafeStage, LinkMonitor
from uavsim.control.flight_controller import (
    ControlInput,
    ControlKind,
    FCParams,
    FlightController,
    Mixer,
    VehicleState,
)
from uavsim.control.rc import FailsafeAction, FlightMode, RCPacket, Sticks
from uavsim.physics.battery import Battery
from uavsim.physics.motor import YAW_TORQUE_PER_NEWTON, Motor
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
    """A ~450 mm class quadcopter (the size of a typical camera/research drone)."""

    body_mass: float = 0.80                 # kg, frame + battery + electronics
    motor_mass: float = 0.05                # kg, each of the 4 motors
    inertia_diag: np.ndarray = field(
        default_factory=lambda: np.array([0.0082, 0.0082, 0.0149])
    )
    arm_length: float = 0.225               # m, centre of mass to motor
    max_thrust_per_motor: float = 6.0       # N -> thrust/weight ~2.4, hover ~41% thrust
    motor_time_constant: float = 0.035      # s, rotor spin-up lag
    telemetry_interval: float = 0.1         # s between telemetry packets
    rc_rate_hz: float = 50.0                # expected RC frame rate (for link quality)
    crash_speed: float = 3.0                # m/s vertical impact before damage starts

    @property
    def total_mass(self) -> float:
        """Body + all 4 motors, lumped into a single rigid body."""
        return self.body_mass + 4 * self.motor_mass


def _default_motor_layout(config: UAVConfig) -> List[Motor]:
    """Standard X layout. Body frame: +X forward, +Y left, +Z up.

    Diagonal motors share a spin direction so reaction torques cancel at
    equal thrust; yaw comes from speeding up one diagonal pair.
    """
    offset = config.arm_length / np.sqrt(2)
    tc = config.motor_time_constant
    thrust = config.max_thrust_per_motor
    return [
        Motor(np.array([offset, -offset, 0.0]), thrust, spin_direction=+1, time_constant=tc),   # FR
        Motor(np.array([-offset, -offset, 0.0]), thrust, spin_direction=-1, time_constant=tc),  # BR
        Motor(np.array([-offset, offset, 0.0]), thrust, spin_direction=+1, time_constant=tc),   # BL
        Motor(np.array([offset, offset, 0.0]), thrust, spin_direction=-1, time_constant=tc),    # FL
    ]


class UAV:
    def __init__(
        self,
        config: UAVConfig,
        ground: FlatGroundPlane,
        command_input: CommGatewayInput,
        telemetry_output: CommGatewayOutput,
        fc_params: Optional[FCParams] = None,
        failsafe_params: Optional[FailsafeParams] = None,
    ) -> None:
        self.config = config
        self.ground = ground
        self.motors = _default_motor_layout(config)
        self.body = RigidBody(mass=config.total_mass, inertia_diag=config.inertia_diag)
        self.battery = Battery()

        mixer = Mixer(
            [m.position_body for m in self.motors],
            [m.spin_direction for m in self.motors],
            YAW_TORQUE_PER_NEWTON,
            config.max_thrust_per_motor,
        )
        self.flight_controller = FlightController(
            config.total_mass, config.inertia_diag, mixer, fc_params
        )
        self.link = LinkMonitor(config.rc_rate_hz)
        self.failsafe = Failsafe(failsafe_params)

        self._command_input = command_input
        self._telemetry_output = telemetry_output
        self.reset()

    # -- public API ------------------------------------------------------------
    def reset(self) -> None:
        """Back to the initial state: parked on the ground, disarmed."""
        self.body.position = np.array([0.0, 0.0, float(self.ground.height_at(0.0, 0.0))])
        self.body.velocity = np.zeros(3)
        self.body.orientation = Rotation.identity()
        self.body.angular_velocity_body = np.zeros(3)
        for motor in self.motors:
            motor.reset()
        self.battery.charge_percent = 100.0
        self.flight_controller.reset()
        self.failsafe.reset()
        self.link.reset()

        self.armed = False
        self.home = self.body.position.copy()
        self.is_grounded = True
        self.sim_time = 0.0
        self.rejected_frames = 0
        self._health = 100.0
        self._dead = False
        self._last_rc: Optional[RCPacket] = None
        # Like a real FC: the arm switch must be seen OFF before arming is allowed.
        self._arm_blocked = True
        self._applied = ControlInput(ControlKind.MANUAL, FlightMode.ANGLE, Sticks())
        self._time_since_last_telemetry = 0.0
        self._telemetry_seq = 0
        self._command_input.receive_all()

    def step(self, dt: float, now: float) -> None:
        """Advance one fixed physics/control tick of `dt` seconds at sim time `now`."""
        if dt <= 0.0:
            return
        self.sim_time = now
        self._process_incoming_frames(now)
        self._update_arming(now)
        self._run_control(dt, now)
        self._integrate_physics(dt)
        self._update_battery(dt)
        self._send_telemetry_if_due(dt)

    @property
    def command_input(self) -> CommGatewayInput:
        return self._command_input

    @property
    def health(self) -> float:
        return self._health

    @property
    def dead(self) -> bool:
        return self._dead

    @property
    def flight_mode(self) -> FlightMode:
        return self._last_rc.mode if self._last_rc else FlightMode.ANGLE

    def link_quality(self) -> float:
        return self.link.link_quality(self.sim_time)

    def motor_world_positions(self) -> List[np.ndarray]:
        """Motor mount points in world space -- used only by the renderer."""
        return [
            self.body.position + self.body.orientation.apply(motor.position_body)
            for motor in self.motors
        ]

    # -- uplink ------------------------------------------------------------------
    def _process_incoming_frames(self, now: float) -> None:
        for raw in self._command_input.receive_all():
            try:
                packet = RCPacket.decode(raw)
            except ValueError:
                self.rejected_frames += 1
                continue
            self.link.packet_received(now, packet.seq)
            self._last_rc = packet

    def _update_arming(self, now: float) -> None:
        rc = self._last_rc
        link_ok = self.link.time_since_last(now) <= self.failsafe.p.loss_timeout
        if self.armed:
            # Pilot disarm only counts if we can actually hear the pilot.
            if link_ok and rc is not None and not rc.arm_switch:
                self._disarm()
            return
        if rc is None or not link_ok:
            return
        if not rc.arm_switch:
            self._arm_blocked = False
            return
        if self._arm_blocked or self._dead or not self.is_grounded:
            return
        manual_throttle = rc.mode in (FlightMode.ACRO, FlightMode.ANGLE)
        if rc.sticks.throttle > (0.10 if manual_throttle else 0.60):
            return  # refuse to arm with the throttle up, like every real FC
        self.armed = True
        self.home = self.body.position.copy()
        self.flight_controller.reset(self._vehicle_state())
        self.failsafe.reset()

    def _disarm(self) -> None:
        self.armed = False
        self._arm_blocked = True
        self.failsafe.reset()
        for motor in self.motors:
            motor.command = 0.0

    # -- control -----------------------------------------------------------------
    def _vehicle_state(self) -> VehicleState:
        return VehicleState(
            position=self.body.position,
            velocity=self.body.velocity,
            orientation=self.body.orientation,
            angular_velocity_body=self.body.angular_velocity_body,
            landed=self.is_grounded,
        )

    def _run_control(self, dt: float, now: float) -> None:
        if not self.armed or self._dead:
            for motor in self.motors:
                motor.command = 0.0
            return
        rc = self._last_rc
        decision = self.failsafe.update(
            now=now,
            time_since_packet=self.link.time_since_last(now),
            position=self.body.position,
            home=self.home,
            last_sticks=rc.sticks if rc else Sticks(),
            last_mode=rc.mode if rc else FlightMode.ANGLE,
            configured_action=rc.failsafe_action if rc else FailsafeAction.LAND,
            touchdown=self.flight_controller.touchdown,
        )
        if decision.force_disarm:
            self._disarm()
            return
        self._applied = decision.control
        commands = self.flight_controller.update(dt, self._vehicle_state(), decision.control)
        for motor, command in zip(self.motors, commands):
            motor.command = float(command)

    # -- physics -------------------------------------------------------------------
    def _integrate_physics(self, dt: float) -> None:
        # Damage reduces effective thrust: 100% at full health, 50% at 0 HP.
        efficiency = 1.0 - (100.0 - self._health) * 0.005
        thrusts = []
        for motor in self.motors:
            motor.update(dt)
            thrusts.append(motor.thrust(efficiency))

        points_body = [motor.position_body for motor in self.motors]
        forces_body = [np.array([0.0, 0.0, t]) for t in thrusts]
        yaw_torque = sum(m.reaction_torque_z(t) for m, t in zip(self.motors, thrusts))
        self.body.apply_body_forces(dt, forces_body, points_body, np.array([0.0, 0.0, yaw_torque]))

        was_grounded = self.is_grounded
        contact = apply_ground_contact(self.body, self.ground, dt)
        self.is_grounded = contact.grounded
        # Speed into the surface: a hard landing or flying into a hillside.
        if not was_grounded and contact.grounded and contact.impact_speed > self.config.crash_speed:
            excess = contact.impact_speed - self.config.crash_speed
            self._health = max(0.0, self._health - excess * 6.0)
            if self._health <= 0.0:
                self._die()
        apply_world_bounds(self.body, WORLD_EXTENT_HALF)

    def _die(self) -> None:
        self._dead = True
        self._disarm()

    # -- battery -----------------------------------------------------------------
    def _update_battery(self, dt: float) -> None:
        total = sum(m.speed for m in self.motors)
        self.battery.update(dt, total, max_total_throttle=len(self.motors))
        if self.battery.charge_percent <= 0.0 and not self._dead:
            self._die()

    # -- telemetry ---------------------------------------------------------------
    def applied_sticks(self) -> Tuple[float, float, float, float]:
        """Sticks the controller is acting on right now (centred while on autopilot)."""
        if not self.armed and self._last_rc is not None:
            s = self._last_rc.sticks
            return (s.throttle, s.roll, s.pitch, s.yaw)
        if self._applied.kind is ControlKind.NAV:
            return (0.5, 0.0, 0.0, 0.0)
        s = self._applied.sticks
        return (s.throttle, s.roll, s.pitch, s.yaw)

    def _send_telemetry_if_due(self, dt: float) -> None:
        self._time_since_last_telemetry += dt
        if self._time_since_last_telemetry < self.config.telemetry_interval:
            return
        self._time_since_last_telemetry = 0.0
        self._telemetry_seq = (self._telemetry_seq + 1) & 0xFFFF
        stage = self.failsafe.stage if self.armed else FailsafeStage.OK
        packet = TelemetryPacket(
            seq=self._telemetry_seq,
            timestamp=self.sim_time,
            position=self.body.position.copy(),
            velocity=self.body.velocity.copy(),
            attitude_rpy=self.body.attitude_rpy(),
            angular_velocity=self.body.angular_velocity_body.copy(),
            motor_throttle=np.array([m.command for m in self.motors]),
            battery_percent=self.battery.charge_percent,
            mass=self.config.total_mass,
            health_percent=self._health,
            applied_sticks=self.applied_sticks(),
            uplink_lq=self.link_quality(),
            flight_mode=int(self.flight_mode),
            failsafe_stage=int(stage),
            failsafe_action=int(self.failsafe.action),
            armed=self.armed,
            landed=self.is_grounded,
        )
        self._telemetry_output.send(packet.encode())
