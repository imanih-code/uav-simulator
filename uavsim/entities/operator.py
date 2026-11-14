"""UAVOperator: the pilot's transmitter (and ground station receiver).

Works like a real RC transmitter: it reads the pilot's sticks and
switches, and sends one RC frame with their *absolute* positions at a
fixed rate (50 Hz), whether or not anything changed. Holding a stick does
not "repeat a command"; a lost frame is simply replaced by the next one.

Keyboard layout (Mode 2, like most RC transmitters):

    left stick           right stick
    W / S  throttle      I / K  pitch forward / back
    A / D  yaw L / R     J / L  roll left / right

Roll, pitch and yaw spring back to centre when released, like real
sticks. Throttle depends on the flight mode: in ALT_HOLD / POS_HOLD it is
a spring stick too (centre = hold altitude), in ANGLE / ACRO it stays
where you leave it (W/S move it up/down).

A gamepad, when connected, drives the sticks directly (see `PilotInput`).

The operator also receives telemetry and measures downlink quality.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Set, Tuple

from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.telemetry import TelemetryPacket
from uavsim.control.failsafe import windowed_quality
from uavsim.control.rc import FailsafeAction, FlightMode, RCPacket, Sticks

_MODE_CYCLE = (FlightMode.POS_HOLD, FlightMode.ALT_HOLD, FlightMode.ANGLE, FlightMode.ACRO)
_FAILSAFE_CYCLE = (FailsafeAction.RTH, FailsafeAction.HOLD, FailsafeAction.LAND, FailsafeAction.DROP)

_STICK_RATE = 3.0          # stick travel per second while a key is held (full deflection in 1/3 s)
_STICK_RETURN_RATE = 6.0   # spring-back speed when released
_THROTTLE_KEY_RATE = 0.35  # ANGLE/ACRO: throttle change per second while W/S held
_GAMEPAD_DEADZONE = 0.08


@dataclass
class PilotInput:
    """Everything the pilot did this frame (filled by the window layer)."""

    keys: Set[str] = field(default_factory=set)  # held keys among w s a d i k j l
    # (left_x, left_y, right_x, right_y) in [-1, 1], pygame convention (up = -1); None if no pad
    gamepad_axes: Optional[Tuple[float, float, float, float]] = None
    toggle_arm: bool = False
    cycle_mode: bool = False
    cycle_failsafe: bool = False
    toggle_tx: bool = False


class UAVOperator:
    def __init__(
        self,
        command_output: CommGatewayOutput,
        telemetry_input: CommGatewayInput,
        rc_rate_hz: float = 50.0,
        telemetry_rate_hz: float = 10.0,
    ) -> None:
        self.command_output = command_output
        self.telemetry_input = telemetry_input
        self.rc_rate_hz = rc_rate_hz
        self.telemetry_rate_hz = telemetry_rate_hz
        self.mode = FlightMode.ALT_HOLD
        self.failsafe_action = FailsafeAction.RTH
        self.reset()

    def reset(self) -> None:
        self.arm_switch = False
        self.tx_enabled = True
        self.sticks = Sticks(throttle=self._neutral_throttle())
        self.latest_telemetry: Optional[TelemetryPacket] = None
        self.last_telemetry_time: Optional[float] = None
        self.frames_sent = 0
        self._seq = 0
        self._next_send_time: Optional[float] = None
        self._telemetry_arrivals: Deque[float] = deque()
        self._first_telemetry_time: Optional[float] = None

    # -- pilot input (real time) ---------------------------------------------
    def handle_input(self, pilot: PilotInput, dt: float) -> None:
        if pilot.toggle_arm:
            self.arm_switch = not self.arm_switch
        if pilot.cycle_mode:
            self.mode = _next_in(_MODE_CYCLE, self.mode)
            if self.mode in (FlightMode.ALT_HOLD, FlightMode.POS_HOLD):
                self.sticks = Sticks(0.5, self.sticks.roll, self.sticks.pitch, self.sticks.yaw)
        if pilot.cycle_failsafe:
            self.failsafe_action = _next_in(_FAILSAFE_CYCLE, self.failsafe_action)
        if pilot.toggle_tx:
            self.tx_enabled = not self.tx_enabled

        if pilot.gamepad_axes is not None:
            self.sticks = self._sticks_from_gamepad(pilot.gamepad_axes, pilot.keys, dt)
        else:
            self.sticks = self._sticks_from_keyboard(pilot.keys, dt)

    def _sticks_from_keyboard(self, keys: Set[str], dt: float) -> Sticks:
        k = {key.lower() for key in keys}
        roll = _approach(self.sticks.roll, _axis(k, "l", "j"), dt)
        pitch = _approach(self.sticks.pitch, _axis(k, "i", "k"), dt)
        yaw = _approach(self.sticks.yaw, _axis(k, "d", "a"), dt)
        if self._throttle_springs():
            throttle = _approach(self.sticks.throttle, 0.5 + 0.5 * _axis(k, "w", "s"), dt,
                                 centre=0.5)
        else:
            throttle = self.sticks.throttle + _axis(k, "w", "s") * _THROTTLE_KEY_RATE * dt
        return Sticks(throttle, roll, pitch, yaw).clamped()

    def _sticks_from_gamepad(
        self, axes: Tuple[float, float, float, float], keys: Set[str], dt: float
    ) -> Sticks:
        lx, ly, rx, ry = (_pad_deadzone(a) for a in axes)
        keyboard = self._sticks_from_keyboard(keys, dt)
        return Sticks(
            throttle=(1.0 - ly) * 0.5,
            roll=rx + keyboard.roll,
            pitch=-ry + keyboard.pitch,
            yaw=lx + keyboard.yaw,
        ).clamped()

    def _throttle_springs(self) -> bool:
        return self.mode in (FlightMode.ALT_HOLD, FlightMode.POS_HOLD)

    def _neutral_throttle(self) -> float:
        return 0.5 if self._throttle_springs() else 0.0

    # -- transmitter (simulation time) ---------------------------------------
    def tick(self, now: float) -> None:
        """Send an RC frame if one is due at simulation time `now`."""
        period = 1.0 / self.rc_rate_hz
        if self._next_send_time is None or self._next_send_time < now - period:
            self._next_send_time = now
        if now + 1e-9 < self._next_send_time:
            return
        self._next_send_time += period
        if not self.tx_enabled:
            return
        frame = RCPacket(
            seq=self._seq,
            sticks=self.sticks,
            arm_switch=self.arm_switch,
            mode=self.mode,
            failsafe_action=self.failsafe_action,
        )
        self._seq = (self._seq + 1) & 0xFF
        self.frames_sent += 1
        self.command_output.send(frame.encode())

    # -- ground station receiver -----------------------------------------------
    def poll_telemetry(self, now: float) -> Optional[TelemetryPacket]:
        """Drain pending telemetry, keeping the latest valid packet."""
        for raw_packet in self.telemetry_input.receive_all():
            try:
                self.latest_telemetry = TelemetryPacket.decode(raw_packet)
            except ValueError:
                continue
            if self._first_telemetry_time is None:
                self._first_telemetry_time = now
            self.last_telemetry_time = now
            self._telemetry_arrivals.append(now)
        return self.latest_telemetry

    def downlink_quality(self, now: float, window: float = 1.0) -> float:
        return windowed_quality(self._telemetry_arrivals, self._first_telemetry_time, now,
                                self.telemetry_rate_hz, window)

    def telemetry_age(self, now: float) -> float:
        if self.last_telemetry_time is None:
            return float("inf")
        return now - self.last_telemetry_time


def _axis(keys: Set[str], positive: str, negative: str) -> float:
    return (1.0 if positive in keys else 0.0) - (1.0 if negative in keys else 0.0)


def _approach(current: float, target: float, dt: float, centre: float = 0.0) -> float:
    rate = _STICK_RETURN_RATE if target == centre else _STICK_RATE
    step = rate * dt
    if abs(target - current) <= step:
        return target
    return current + step if target > current else current - step


def _pad_deadzone(value: float) -> float:
    if abs(value) < _GAMEPAD_DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - _GAMEPAD_DEADZONE) / (1.0 - _GAMEPAD_DEADZONE)


def _next_in(cycle: tuple, current):
    index = cycle.index(current) if current in cycle else -1
    return cycle[(index + 1) % len(cycle)]
