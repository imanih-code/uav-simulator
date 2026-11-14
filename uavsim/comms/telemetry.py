"""Telemetry protocol used by the UAV to report its own state back to the
UAVOperator over a CommGatewayOutput/Input pair. The HUD never talks to the
UAV directly: it only ever reads whatever telemetry the Operator has
buffered from these packets -- so when the downlink is jammed, the pilot
genuinely loses this information.

Besides the flight state, every packet carries what the *receiver* knows
about the uplink: the sticks it is actually applying (which differ from
the pilot's sticks during a failsafe), the uplink link quality it
measured, the flight mode, and the failsafe stage/action.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# seq(H) + timestamp(d) + position(3d) + velocity(3d) + attitude_rpy(3d)
# + angular_velocity(3d) + motor_command(4d) + battery(d) + mass(d) + health(d)
# + applied sticks (4h: throttle*1000, roll/pitch/yaw*500)
# + uplink LQ % (B) + flight mode (B) + failsafe stage<<4 | action (B) + flags (B)
_STRUCT_FORMAT = "!H20d4h4B"
PACKET_SIZE = struct.calcsize(_STRUCT_FORMAT)

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
            *self.velocity.tolist(),
            *self.attitude_rpy.tolist(),
            *self.angular_velocity.tolist(),
            *self.motor_throttle.tolist(),
            self.battery_percent,
            self.mass,
            self.health_percent,
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
        seq = v[0]
        d = v[1:21]
        thr, roll, pitch, yaw = v[21:25]
        lq, mode, fs, flags = v[25:29]
        return TelemetryPacket(
            seq=seq,
            timestamp=d[0],
            position=np.array(d[1:4]),
            velocity=np.array(d[4:7]),
            attitude_rpy=np.array(d[7:10]),
            angular_velocity=np.array(d[10:13]),
            motor_throttle=np.array(d[13:17]),
            battery_percent=d[17],
            mass=d[18],
            health_percent=d[19],
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
