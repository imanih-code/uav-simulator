"""Onboard flight controller: what a real quadcopter runs on its own board.

The pilot never drives individual motors. Sticks (or the failsafe
autopilot) produce *setpoints*; this controller turns them into four motor
commands every physics tick through a classic cascade:

    position --P--> velocity --PI--> acceleration --> tilt + collective thrust
                                                          |
                         attitude error --P--> body-rate setpoint
                                                          |
                         body-rate error --PI--> angular acceleration --> torque
                                                          |
                                     mixer (allocation matrix inverse)
                                                          |
                                     per-motor thrust -> motor command (sqrt)

Which stages run depends on the flight mode:

* ACRO     -- sticks are body-rate setpoints. Nothing self-levels.
* ANGLE    -- sticks are tilt angles; throttle is raw collective thrust.
* ALT_HOLD -- ANGLE, plus throttle stick = climb rate (centre holds altitude).
* POS_HOLD -- ALT_HOLD, plus roll/pitch = horizontal velocity (centre holds position).
* NAV      -- used only by the failsafe autopilot: fly to a 3D point, or
              descend at a fixed rate (landing).

The controller reads the vehicle's true state (perfect IMU / barometer /
GNSS). Sensor noise and GNSS jamming are deliberately left for a later
phase so the effect of the *radio* link can be studied in isolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from uavsim.control.pid import PID
from uavsim.control.rc import FlightMode, Sticks

GRAVITY = 9.81


class ControlKind(Enum):
    MANUAL = auto()  # pilot sticks interpreted according to `mode`
    NAV = auto()     # autopilot target (failsafe)


@dataclass
class NavTarget:
    """Autopilot setpoint used by the failsafe (HOLD / RTH / LAND)."""

    position: np.ndarray                  # world xyz to fly to / hold
    max_speed_xy: float = 6.0             # m/s
    descend_rate: Optional[float] = None  # if set: ignore target z, sink at this rate (m/s)


@dataclass
class ControlInput:
    kind: ControlKind
    mode: FlightMode = FlightMode.ANGLE
    sticks: Sticks = field(default_factory=Sticks)
    nav: Optional[NavTarget] = None


@dataclass
class VehicleState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: Rotation
    angular_velocity_body: np.ndarray
    landed: bool


@dataclass
class FCParams:
    # Pilot-facing limits
    max_angle_deg: float = 35.0
    acro_rates_deg: Sequence[float] = (400.0, 400.0, 200.0)  # roll, pitch, yaw at full stick
    angle_yaw_rate_deg: float = 120.0
    max_climb: float = 3.0          # m/s, ALT/POS hold, full throttle stick
    max_descent: float = 2.0        # m/s, ALT/POS hold, zero throttle stick
    max_speed_xy: float = 8.0       # m/s, POS hold, full stick
    stick_deadband: float = 0.06
    idle_throttle: float = 0.06     # motor command while armed and idle

    # Attitude loop (rad/s of rate setpoint per rad of error)
    att_kp: Sequence[float] = (7.0, 7.0, 3.5)
    max_rate_deg: Sequence[float] = (400.0, 400.0, 200.0)

    # Rate loop (rad/s^2 of angular acceleration per rad/s of rate error)
    rate_kp: Sequence[float] = (16.0, 16.0, 10.0)
    rate_ki: Sequence[float] = (12.0, 12.0, 6.0)
    rate_i_limit: Sequence[float] = (30.0, 30.0, 15.0)

    # Altitude
    pos_z_kp: float = 1.4
    vel_z_kp: float = 4.0
    vel_z_ki: float = 2.5
    vel_z_i_limit: float = 10.0     # m/s^2 -- enough to cover 50% thrust loss

    # Horizontal position
    pos_xy_kp: float = 1.0
    vel_xy_kp: float = 2.0
    vel_xy_ki: float = 0.4
    vel_xy_i_limit: float = 2.0


class Mixer:
    """Quad-X allocation: collective thrust + 3 torques <-> 4 motor thrusts.

    Built from the actual motor positions and spin directions, so it stays
    correct if the airframe geometry changes. Saturation strategy
    ("airmode"-like): roll/pitch authority first, then yaw, then collective
    thrust is shifted (not clipped per motor) so attitude control survives
    full and zero throttle.
    """

    def __init__(
        self,
        positions_body: Sequence[np.ndarray],
        spin_directions: Sequence[int],
        yaw_torque_per_newton: float,
        max_thrust_per_motor: float,
    ) -> None:
        a = np.zeros((4, 4))
        for i, (p, s) in enumerate(zip(positions_body, spin_directions)):
            a[0, i] = 1.0                           # collective thrust
            a[1, i] = p[1]                          # roll torque  = y * f
            a[2, i] = -p[0]                         # pitch torque = -x * f
            a[3, i] = -s * yaw_torque_per_newton    # yaw reaction torque
        self.allocation = a
        self.inverse = np.linalg.inv(a)
        self.max_thrust = max_thrust_per_motor

    def mix(self, collective: float, torque: np.ndarray, min_thrust: float) -> np.ndarray:
        """Return per-motor thrust in newtons, within [min_thrust, max_thrust]."""
        f_rp = self.inverse[:, 1:3] @ torque[:2]
        f_yaw = self.inverse[:, 3] * torque[2]
        available = self.max_thrust - min_thrust

        span_rp = float(np.ptp(f_rp))
        if span_rp > available:
            f_att = f_rp * (available / span_rp)
        else:
            span_total = float(np.ptp(f_rp + f_yaw))
            if span_total > available:
                k = (available - span_rp) / max(float(np.ptp(f_yaw)), 1e-9)
                f_att = f_rp + max(0.0, min(1.0, k)) * f_yaw
            else:
                f_att = f_rp + f_yaw

        per_motor = collective / 4.0
        low = min_thrust - float(f_att.min())
        high = self.max_thrust - float(f_att.max())
        per_motor = min(max(per_motor, low), high)
        return np.clip(per_motor + f_att, min_thrust, self.max_thrust)


class FlightController:
    def __init__(
        self,
        mass: float,
        inertia_diag: np.ndarray,
        mixer: Mixer,
        params: Optional[FCParams] = None,
    ) -> None:
        self.mass = mass
        self.inertia = np.asarray(inertia_diag, dtype=float)
        self.mixer = mixer
        self.p = params or FCParams()
        self._max_total_thrust = 4.0 * mixer.max_thrust

        self._rate_pid = PID(self.p.rate_kp, self.p.rate_ki, 0.0, self.p.rate_i_limit, size=3)
        self._vz_pid = PID(self.p.vel_z_kp, self.p.vel_z_ki, 0.0, self.p.vel_z_i_limit)
        self._vxy_pid = PID(self.p.vel_xy_kp, self.p.vel_xy_ki, 0.0, self.p.vel_xy_i_limit, size=2)

        self._yaw_sp = 0.0
        self._z_sp = 0.0
        self._z_latched = False
        self._xy_sp = np.zeros(2)
        self._xy_latched = False
        self.touchdown = False  # NAV descent reached the ground (UAV disarms)
        self._last_control_id = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, state: Optional[VehicleState] = None) -> None:
        self._rate_pid.reset()
        self._vz_pid.reset()
        self._vxy_pid.reset()
        self._z_latched = False
        self._xy_latched = False
        self.touchdown = False
        if state is not None:
            self._yaw_sp = _yaw_of(state.orientation)
            self._z_sp = float(state.position[2])
            self._xy_sp = state.position[:2].copy()

    def disarmed_output(self) -> np.ndarray:
        return np.zeros(4)

    # ------------------------------------------------------------------
    def update(self, dt: float, state: VehicleState, cmd: ControlInput) -> np.ndarray:
        """One control tick. Returns per-motor commands in [0, 1]."""
        p = self.p
        yaw = _yaw_of(state.orientation)
        self.touchdown = False

        # A change of mode or of who is flying (pilot <-> autopilot) must not
        # reuse an altitude/position latched long ago: brake and re-latch.
        control_id = (cmd.kind, cmd.mode)
        if control_id != self._last_control_id:
            self._z_latched = False
            self._xy_latched = False
            self._last_control_id = control_id

        if cmd.kind is ControlKind.NAV:
            return self._update_nav(dt, state, cmd.nav, yaw)

        mode = cmd.mode
        s = cmd.sticks

        if mode in (FlightMode.ACRO, FlightMode.ANGLE):
            if state.landed and s.throttle < 0.05:
                return self._idle(state)
            collective = s.throttle * self._max_total_thrust
            if mode is FlightMode.ACRO:
                rates = np.radians(np.asarray(p.acro_rates_deg))
                rate_sp = np.array([s.roll * rates[0], s.pitch * rates[1], -s.yaw * rates[2]])
                self._yaw_sp = yaw
                return self._rate_and_mix(dt, state, rate_sp, collective)
            max_angle = math.radians(p.max_angle_deg)
            self._advance_yaw_sp(dt, s.yaw, yaw)
            return self._attitude_and_mix(
                dt, state, s.roll * max_angle, s.pitch * max_angle, collective
            )

        # ALT_HOLD / POS_HOLD: throttle stick is a climb-rate request.
        climb = _deadband(2.0 * s.throttle - 1.0, p.stick_deadband)
        if state.landed and climb <= 0.0:
            return self._idle(state)
        vz_sp = self._altitude_velocity_sp(state, climb)

        if mode is FlightMode.POS_HOLD:
            stick_fwd = _deadband(s.pitch, p.stick_deadband)
            stick_right = _deadband(s.roll, p.stick_deadband)
            v_xy_sp = self._horizontal_velocity_sp(state, stick_fwd, stick_right, yaw)
            roll_sp, pitch_sp = self._tilt_for_velocity(dt, state, v_xy_sp, yaw)
        else:
            max_angle = math.radians(p.max_angle_deg)
            roll_sp, pitch_sp = s.roll * max_angle, s.pitch * max_angle
            self._xy_latched = False

        self._advance_yaw_sp(dt, s.yaw, yaw)
        collective = self._collective_for_vz(dt, state, vz_sp, roll_sp, pitch_sp)
        return self._attitude_and_mix(dt, state, roll_sp, pitch_sp, collective)

    # -- NAV (failsafe autopilot) --------------------------------------
    def _update_nav(self, dt: float, state: VehicleState, nav: NavTarget, yaw: float) -> np.ndarray:
        p = self.p
        if nav.descend_rate is not None:
            if state.landed:
                output = self._idle(state)
                self.touchdown = True
                return output
            vz_sp = -abs(nav.descend_rate)
        else:
            vz_sp = float(np.clip(
                p.pos_z_kp * (nav.position[2] - state.position[2]), -p.max_descent, p.max_climb
            ))
            if state.landed and vz_sp <= 0.0:
                return self._idle(state)

        err_xy = nav.position[:2] - state.position[:2]
        v_xy_sp = p.pos_xy_kp * err_xy
        speed = float(np.linalg.norm(v_xy_sp))
        if speed > nav.max_speed_xy:
            v_xy_sp *= nav.max_speed_xy / speed

        roll_sp, pitch_sp = self._tilt_for_velocity(dt, state, v_xy_sp, yaw)
        self._yaw_sp = yaw  # keep current heading
        collective = self._collective_for_vz(dt, state, vz_sp, roll_sp, pitch_sp)
        return self._attitude_and_mix(dt, state, roll_sp, pitch_sp, collective)

    # -- outer loops ----------------------------------------------------
    def _altitude_velocity_sp(self, state: VehicleState, climb: float) -> float:
        p = self.p
        z = float(state.position[2])
        vz = float(state.velocity[2])
        if climb != 0.0:
            self._z_latched = False
            return climb * (p.max_climb if climb > 0 else p.max_descent)
        if not self._z_latched:
            # Stick just centred: brake first, then latch the altitude.
            if abs(vz) < 0.25:
                self._z_sp = z
                self._z_latched = True
            else:
                return 0.0
        return float(np.clip(p.pos_z_kp * (self._z_sp - z), -p.max_descent, p.max_climb))

    def _horizontal_velocity_sp(
        self, state: VehicleState, stick_fwd: float, stick_right: float, yaw: float
    ) -> np.ndarray:
        p = self.p
        if stick_fwd != 0.0 or stick_right != 0.0:
            self._xy_latched = False
            v_fwd = stick_fwd * p.max_speed_xy
            v_left = -stick_right * p.max_speed_xy
            c, s = math.cos(yaw), math.sin(yaw)
            return np.array([c * v_fwd - s * v_left, s * v_fwd + c * v_left])
        if not self._xy_latched:
            if np.linalg.norm(state.velocity[:2]) < 0.4:
                self._xy_sp = state.position[:2].copy()
                self._xy_latched = True
            else:
                return np.zeros(2)
        v = p.pos_xy_kp * (self._xy_sp - state.position[:2])
        speed = float(np.linalg.norm(v))
        if speed > p.max_speed_xy:
            v *= p.max_speed_xy / speed
        return v

    def _tilt_for_velocity(
        self, dt: float, state: VehicleState, v_xy_sp: np.ndarray, yaw: float
    ) -> tuple:
        """Velocity error -> horizontal acceleration -> roll/pitch setpoints."""
        max_angle = math.radians(self.p.max_angle_deg)
        accel = self._vxy_pid.update(v_xy_sp - state.velocity[:2], dt)
        max_accel = GRAVITY * math.tan(max_angle)
        norm = float(np.linalg.norm(accel))
        if norm > max_accel:
            accel *= max_accel / norm
        c, s = math.cos(yaw), math.sin(yaw)
        a_fwd = c * accel[0] + s * accel[1]
        a_left = -s * accel[0] + c * accel[1]
        pitch_sp = math.atan2(a_fwd, GRAVITY)                    # + pitch = nose down = forward
        roll_sp = math.atan2(-a_left * math.cos(pitch_sp), GRAVITY)  # + roll = right side down
        return (
            float(np.clip(roll_sp, -max_angle, max_angle)),
            float(np.clip(pitch_sp, -max_angle, max_angle)),
        )

    def _collective_for_vz(
        self, dt: float, state: VehicleState, vz_sp: float, roll_sp: float, pitch_sp: float
    ) -> float:
        az = float(self._vz_pid.update(vz_sp - state.velocity[2], dt)[0])
        tilt = max(math.cos(roll_sp) * math.cos(pitch_sp), 0.5)
        thrust = self.mass * (GRAVITY + az) / tilt
        return float(np.clip(thrust, 0.0, self._max_total_thrust))

    def _advance_yaw_sp(self, dt: float, yaw_stick: float, yaw: float) -> None:
        rate = -_deadband(yaw_stick, self.p.stick_deadband) * math.radians(self.p.angle_yaw_rate_deg)
        self._yaw_sp = yaw + float(np.clip(_wrap(self._yaw_sp + rate * dt - yaw), -0.6, 0.6))

    # -- inner loops ----------------------------------------------------
    def _attitude_and_mix(
        self, dt: float, state: VehicleState, roll_sp: float, pitch_sp: float, collective: float
    ) -> np.ndarray:
        desired = Rotation.from_euler("xyz", [roll_sp, pitch_sp, self._yaw_sp])
        error_body = (state.orientation.inv() * desired).as_rotvec()
        rate_sp = np.asarray(self.p.att_kp) * error_body
        max_rate = np.radians(np.asarray(self.p.max_rate_deg))
        rate_sp = np.clip(rate_sp, -max_rate, max_rate)
        return self._rate_and_mix(dt, state, rate_sp, collective)

    def _rate_and_mix(
        self, dt: float, state: VehicleState, rate_sp: np.ndarray, collective: float
    ) -> np.ndarray:
        omega = state.angular_velocity_body
        alpha = self._rate_pid.update(rate_sp - omega, dt)
        torque = self.inertia * alpha + np.cross(omega, self.inertia * omega)
        min_thrust = self.p.idle_throttle ** 2 * self.mixer.max_thrust
        thrusts = self.mixer.mix(collective, torque, min_thrust)
        return np.sqrt(thrusts / self.mixer.max_thrust)

    def _idle(self, state: VehicleState) -> np.ndarray:
        """Armed but sitting on the ground: props spin at idle, loops reset."""
        self.reset(state)
        return np.full(4, self.p.idle_throttle)


def _yaw_of(orientation: Rotation) -> float:
    return float(orientation.as_euler("xyz")[2])


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _deadband(value: float, band: float) -> float:
    if abs(value) <= band:
        return 0.0
    return math.copysign((abs(value) - band) / (1.0 - band), value)
