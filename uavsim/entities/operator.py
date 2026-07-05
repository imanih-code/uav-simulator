"""UAVOperator: the human-in-the-loop controller.

Reads which control keys are currently held down, translates them into
bit-encoded Commands, and transmits them through its CommGatewayOutput --
which is now a real (simulated) radio link. It also owns a
CommGatewayInput used to receive telemetry broadcast by the UAV; the HUD
reads that buffered telemetry directly from the operator.

Commands are rate-limited per motor+direction rather than sent once per
render frame: a real digital transmitter has its own update rate,
independent of however fast the game is rendering, and GNU Radio's actual
per-burst cost makes flooding the link at 60/sec both unrealistic and
unnecessary.

Holding a key longer progressively accelerates the send rate so the
throttle steps feel fine with short taps but grow coarser when held.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from uavsim.comms.command import Command, CommandOpcode
from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.telemetry import TelemetryPacket

# Keyboard layout: a numeric key accelerates a motor, the letter directly
# above it on a QWERTY keyboard decelerates that same motor.
#   1 / Q -> front-right motor
#   2 / W -> back-right motor
#   3 / E -> back-left motor
#   4 / R -> front-left motor
#   5 / T -> all four motors at once
THROTTLE_UP_KEYS = {"1": 0, "2": 1, "3": 2, "4": 3, "5": -1}
THROTTLE_DOWN_KEYS = {"q": 0, "w": 1, "e": 2, "r": 3, "t": -1}

# Base interval between consecutive transmissions of the same command.
# The actual interval shrinks as a key is held (see hold-duration
# acceleration below), but 20 Hz is the floor.
_COMMAND_SEND_INTERVAL = 0.05

# Hold-duration thresholds and their corresponding send intervals.
_HOLD_ACCELERATION: Tuple[Tuple[float, float], ...] = (
    (0.0,    0.05),    # tap → 20 Hz, fine steps
    (0.3,    0.025),   # ⅓ s → 40 Hz
    (0.7,    0.012),   # ⅔ s → ~80 Hz
    (1.5,    0.008),   # 1½ s → ~120 Hz
)


@dataclass
class UAVOperator:
    command_output: CommGatewayOutput
    telemetry_input: CommGatewayInput
    latest_telemetry: Optional[TelemetryPacket] = field(default=None, init=False)
    _last_sent_at: Dict[Tuple[CommandOpcode, int], float] = field(
        default_factory=dict, init=False
    )
    _key_hold_start: Dict[str, float] = field(default_factory=dict, init=False)

    @staticmethod
    def _send_interval(hold_duration: float) -> float:
        """Return the send interval for a key held for `hold_duration` seconds.

        Short taps keep the base interval for fine adjustment; longer
        holds progressively shorten it so the throttle moves faster.
        """
        for threshold, interval in reversed(_HOLD_ACCELERATION):
            if hold_duration >= threshold:
                return interval
        return _COMMAND_SEND_INTERVAL

    def handle_pressed_keys(self, pressed_keys: Set[str]) -> None:
        """Translate currently-held keys into commands sent to the UAV.

        Called once per simulation tick with the set of keys currently
        held down. The send rate per (opcode, motor) pair accelerates as
        the key is held, then resets when released.
        """
        now = time.monotonic()

        # Start tracking newly pressed keys; remove released ones.
        for key in pressed_keys:
            if key not in self._key_hold_start:
                self._key_hold_start[key] = now
        for key in list(self._key_hold_start.keys()):
            if key not in pressed_keys:
                del self._key_hold_start[key]

        for key in pressed_keys:
            command = self._command_for_key(key)
            if command is None:
                continue
            send_key = (command.opcode, command.motor_id)
            hold_duration = now - self._key_hold_start.get(key, now)
            interval = self._send_interval(hold_duration)

            last_sent = self._last_sent_at.get(send_key, 0.0)
            if now - last_sent < interval:
                continue
            self._last_sent_at[send_key] = now
            self.command_output.send(command.encode())

    @staticmethod
    def _command_for_key(key: str) -> Optional[Command]:
        key = key.lower()
        if key in THROTTLE_UP_KEYS:
            motor_id = THROTTLE_UP_KEYS[key]
            if motor_id == -1:
                return Command(CommandOpcode.THROTTLE_UP_ALL)
            return Command(CommandOpcode.THROTTLE_UP, motor_id=motor_id)
        if key in THROTTLE_DOWN_KEYS:
            motor_id = THROTTLE_DOWN_KEYS[key]
            if motor_id == -1:
                return Command(CommandOpcode.THROTTLE_DOWN_ALL)
            return Command(CommandOpcode.THROTTLE_DOWN, motor_id=motor_id)
        return None

    def poll_telemetry(self) -> Optional[TelemetryPacket]:
        """Drain any pending telemetry packets, keeping only the latest.

        Corrupted packets that somehow pass the radio link's own CRC check
        (astronomically unlikely, but never assume never) are simply
        skipped rather than crashing the HUD.
        """
        for raw_packet in self.telemetry_input.receive_all():
            try:
                self.latest_telemetry = TelemetryPacket.decode(raw_packet)
            except ValueError:
                continue
        return self.latest_telemetry
