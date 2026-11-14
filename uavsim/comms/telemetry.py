"""Telemetry protocol used by the UAV to report its own state back to the
UAVOperator over a CommGatewayOutput/Input pair. The HUD never talks to the
UAV directly: it only ever reads whatever telemetry the Operator has
buffered from these packets -- so when the downlink is jammed, the pilot
genuinely loses this information.

Besides the flight state, every packet carries what the *receiver* knows
about the uplink: the sticks it is actually applying (which differ from
the pilot's sticks during a failsafe), the uplink link quality it
measured, the flight mode, and the failsafe stage/action.

The packet is deliberately compact (56 bytes, was 174 with doubles):
inside a jammer every extra bit is another chance for the CRC to fail, so
values are sent as scaled integers at the resolution a ground station
actually needs (cm/s, 0.006 deg, 0.4% throttle...).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# seq(H) + timestamp(f, s) + position(3f, m)
# + velocity(3h, cm/s) + attitude(3h, 1e-4 rad) + angular velocity(3h, 1e-3 rad/s)
# + motor commands(4B, /255) + battery(B, 0.5%) + health(B, 0.5%) + mass(H, g)
# + applied sticks(4h: throttle*1000, roll/pitch/yaw*500)
# + uplink LQ %(B) + flight mode(B) + failsafe stage<<4 | action(B) + flags(B)
_STRUCT_FORMAT = "!Hf3f3h3h3h4BBBH4h4B"
PACKET_SIZE = struct.calcsize(_STRUCT_FORMAT)

_VEL_SCALE = 100.0
_ATT_SCALE = 1e4
_RATE_SCALE = 1e3

FLAG_ARMED = 0x01
FLAG_LANDED = 0x02


@dataclass(frozen=True)
class TelemetryPacket:
    """A snapshot of the UAV's state at a given simulation time."""

    timestamp: float
    position: np.ndarray          # world frame, meters (x, y, z)
    velocity: np.ndarray          # world frame, m/s
    attitude_rpy: np.ndarray      # roll, pitch, yaw, radians
    angular_velocity: np.ndarray  # body frame, rad/s
    motor_throttle: np.ndarray    # 4 motor commands in [0, 1]
    battery_percent: float        # 0..100
    mass: float                   # kg, body + all motors combined
    health_percent: float = 100.0
    seq: int = 0
    applied_sticks: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    uplink_lq: float = 0.0        # % of expected RC frames received, measured on the UAV
    flight_mode: int = 0          # rc.FlightMode value
    failsafe_stage: int = 0       # failsafe.FailsafeStage value
    failsafe_action: int = 0      # rc.FailsafeAction value
    armed: bool = False
    landed: bool = True

    def encode(self) -> bytes:
        thr, roll, pitch, yaw = self.applied_sticks
        flags = (FLAG_ARMED if self.armed else 0) | (FLAG_LANDED if self.landed else 0)
        return struct.pack(
            _STRUCT_FORMAT,
            self.seq & 0xFFFF,
            self.timestamp,
            *self.position.tolist(),
            *_i16(self.velocity, _VEL_SCALE),
            *_i16(self.attitude_rpy, _ATT_SCALE),
            *_i16(self.angular_velocity, _RATE_SCALE),
            *(int(round(_clamp(m, 0.0, 1.0) * 255)) for m in self.motor_throttle),
            int(round(_clamp(self.battery_percent, 0.0, 100.0) * 2)),
            int(round(_clamp(self.health_percent, 0.0, 100.0) * 2)),
            int(round(_clamp(self.mass * 1000.0, 0.0, 65535.0))),
            int(round(_clamp(thr, 0.0, 1.0) * 1000)),
            int(round(_clamp(roll, -1.0, 1.0) * 500)),
            int(round(_clamp(pitch, -1.0, 1.0) * 500)),
            int(round(_clamp(yaw, -1.0, 1.0) * 500)),
            int(round(_clamp(self.uplink_lq, 0.0, 100.0))),
            int(self.flight_mode) & 0xFF,
            ((int(self.failsafe_stage) & 0x0F) << 4) | (int(self.failsafe_action) & 0x0F),
            flags,
        )

    @staticmethod
    def decode(data: bytes) -> "TelemetryPacket":
        if len(data) != PACKET_SIZE:
            raise ValueError(f"Expected {PACKET_SIZE} bytes, got {len(data)}")
        v = struct.unpack(_STRUCT_FORMAT, data)
        seq, timestamp = v[0], v[1]
        position = np.array(v[2:5], dtype=float)
        velocity = np.array(v[5:8], dtype=float) / _VEL_SCALE
        attitude = np.array(v[8:11], dtype=float) / _ATT_SCALE
        rates = np.array(v[11:14], dtype=float) / _RATE_SCALE
        motors = np.array(v[14:18], dtype=float) / 255.0
        battery, health, mass_g = v[18] / 2.0, v[19] / 2.0, v[20]
        thr, roll, pitch, yaw = v[21:25]
        lq, mode, fs, flags = v[25:29]
        return TelemetryPacket(
            seq=seq,
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            attitude_rpy=attitude,
            angular_velocity=rates,
            motor_throttle=motors,
            battery_percent=battery,
            mass=mass_g / 1000.0,
            health_percent=health,
            applied_sticks=(thr / 1000.0, roll / 500.0, pitch / 500.0, yaw / 500.0),
            uplink_lq=float(lq),
            flight_mode=mode,
            failsafe_stage=fs >> 4,
            failsafe_action=fs & 0x0F,
            armed=bool(flags & FLAG_ARMED),
            landed=bool(flags & FLAG_LANDED),
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _i16(values, scale: float):
    return [int(round(_clamp(float(x) * scale, -32767.0, 32767.0))) for x in values]
