"""Telemetry protocol used by the UAV to report its own state back to the
UAVOperator over a CommGatewayOutput/Input pair. The HUD never talks to the
UAV directly: it only ever reads whatever telemetry the Operator has
buffered from these packets.

Encoding relies on the standard library `struct` module instead of a
hand-rolled binary format, since that's the reliable, well tested tool for
this exact job.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# timestamp(1) + position(3) + velocity(3) + attitude_rpy(3)
# + angular_velocity(3) + motor_throttle(4) + battery_percent(1) + mass(1)
# -> 19 doubles, network byte order.
# Appended: command_log (4 ints) — each int encodes (opcode<<16)|(motor_id<<8)|valid_flag.
_STRUCT_FORMAT = "!19d4i"
PACKET_SIZE = struct.calcsize(_STRUCT_FORMAT)


@dataclass(frozen=True)
class TelemetryPacket:
    """A snapshot of the UAV's state at a given simulation time."""

    timestamp: float
    position: np.ndarray          # world frame, meters (x, y, z)
    velocity: np.ndarray          # world frame, m/s
    attitude_rpy: np.ndarray      # roll, pitch, yaw, radians
    angular_velocity: np.ndarray  # body frame, rad/s
    motor_throttle: np.ndarray    # 4 values in [0, 1], one per motor
    battery_percent: float        # 0..100
    mass: float                   # kg, body + all motors combined
    command_log: Tuple[Tuple[int, int, bool], ...] = ()  # (opcode, motor, valid)

    _LOG_SLOTS = 4

    def encode(self) -> bytes:
        log_ints = []
        for op, mid, ok in self.command_log:
            if ok:
                log_ints.append((op << 16) | ((mid + 1) << 8) | 1)
            else:
                log_ints.append((7 << 16) | 0)  # opcode 7 = BAD marker
        log_ints += [0] * (self._LOG_SLOTS - len(log_ints))
        return struct.pack(
            _STRUCT_FORMAT,
            self.timestamp,
            *self.position.tolist(),
            *self.velocity.tolist(),
            *self.attitude_rpy.tolist(),
            *self.angular_velocity.tolist(),
            *self.motor_throttle.tolist(),
            self.battery_percent,
            self.mass,
            *log_ints,
        )

    @staticmethod
    def decode(data: bytes) -> "TelemetryPacket":
        if len(data) != PACKET_SIZE:
            raise ValueError(f"Expected {PACKET_SIZE} bytes, got {len(data)}")

        values = struct.unpack(_STRUCT_FORMAT, data)
        log_ints = values[19:23]
        command_log = []
        for v in log_ints:
            if v == 0:
                continue
            valid = bool(v & 1)
            if not valid:
                command_log.append((7, 0, False))
                continue
            opcode = (v >> 16) & 0xFF
            motor_id_raw = (v >> 8) & 0xFF
            motor_id = motor_id_raw - 1 if motor_id_raw > 0 else 0
            command_log.append((opcode, motor_id, True))
        return TelemetryPacket(
            timestamp=values[0],
            position=np.array(values[1:4]),
            velocity=np.array(values[4:7]),
            attitude_rpy=np.array(values[7:10]),
            angular_velocity=np.array(values[10:13]),
            motor_throttle=np.array(values[13:17]),
            battery_percent=values[17],
            mass=values[18],
            command_log=tuple(command_log),
        )
