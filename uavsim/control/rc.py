"""RC uplink protocol: absolute stick positions, sent at a fixed rate.

This replaces the old incremental per-motor protocol (``THROTTLE_UP`` for
motor 2, ...). Real RC links (SBUS, CRSF/ELRS, Crossfire) do not send
"nudge" commands: every frame carries the *absolute* position of every
stick and switch. That property matters a lot under jamming:

* Losing a packet only means the receiver keeps the previous frame for a
  few milliseconds. Nothing is "lost forever" the way a dropped
  ``THROTTLE_UP`` increment was.
* The receiver can measure link quality by counting sequence numbers, and
  can decide on its own when the link is gone (failsafe).

Wire format (network byte order, 11 bytes, CRC32 added by the link):

    B  packet type (always ``RC_PACKET_TYPE``)
    B  sequence number (wraps at 256)
    H  throttle   0..1000      -> 0.0 .. 1.0
    h  roll    -500..500       -> -1.0 .. +1.0   (+ = roll right)
    h  pitch   -500..500       -> -1.0 .. +1.0   (+ = pitch forward / nose down)
    h  yaw     -500..500       -> -1.0 .. +1.0   (+ = yaw right / clockwise from above)
    B  flags: bit0 arm switch, bits1-2 flight mode, bits3-4 failsafe action
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

RC_PACKET_TYPE = 0xC1
RC_FORMAT = "!BBHhhhB"
RC_PACKET_SIZE = struct.calcsize(RC_FORMAT)

_THROTTLE_SCALE = 1000
_AXIS_SCALE = 500


class FlightMode(IntEnum):
    """Pilot-selectable flight modes, from hardest to easiest to fly."""

    ACRO = 0      # sticks command body rates; nothing self-levels
    ANGLE = 1     # sticks command tilt angle; throttle is raw collective thrust
    ALT_HOLD = 2  # ANGLE + throttle stick commands climb rate (centre = hold altitude)
    POS_HOLD = 3  # ALT_HOLD + roll/pitch command horizontal velocity (centre = hold position)


class FailsafeAction(IntEnum):
    """What the UAV does on its own once the uplink has been gone too long."""

    HOLD = 0  # hover in place (needs an onboard position estimate, i.e. GNSS)
    LAND = 1  # level out and descend slowly where it is, then disarm
    RTH = 2   # return to the arming point at a safe altitude, then land
    DROP = 3  # cut the motors (Betaflight-style "drop")


@dataclass(frozen=True)
class Sticks:
    throttle: float = 0.0  # 0..1
    roll: float = 0.0      # -1..1
    pitch: float = 0.0     # -1..1
    yaw: float = 0.0       # -1..1

    def clamped(self) -> "Sticks":
        return Sticks(
            throttle=_clamp(self.throttle, 0.0, 1.0),
            roll=_clamp(self.roll, -1.0, 1.0),
            pitch=_clamp(self.pitch, -1.0, 1.0),
            yaw=_clamp(self.yaw, -1.0, 1.0),
        )

    def quantized(self) -> "Sticks":
        """Same values the receiver will see after a round trip on the wire."""
        return Sticks(
            throttle=round(_clamp(self.throttle, 0.0, 1.0) * _THROTTLE_SCALE) / _THROTTLE_SCALE,
            roll=round(_clamp(self.roll, -1.0, 1.0) * _AXIS_SCALE) / _AXIS_SCALE,
            pitch=round(_clamp(self.pitch, -1.0, 1.0) * _AXIS_SCALE) / _AXIS_SCALE,
            yaw=round(_clamp(self.yaw, -1.0, 1.0) * _AXIS_SCALE) / _AXIS_SCALE,
        )


@dataclass(frozen=True)
class RCPacket:
    seq: int
    sticks: Sticks
    arm_switch: bool
    mode: FlightMode
    failsafe_action: FailsafeAction

    def encode(self) -> bytes:
        s = self.sticks.clamped()
        flags = (
            (1 if self.arm_switch else 0)
            | ((int(self.mode) & 0b11) << 1)
            | ((int(self.failsafe_action) & 0b11) << 3)
        )
        return struct.pack(
            RC_FORMAT,
            RC_PACKET_TYPE,
            self.seq & 0xFF,
            int(round(s.throttle * _THROTTLE_SCALE)),
            int(round(s.roll * _AXIS_SCALE)),
            int(round(s.pitch * _AXIS_SCALE)),
            int(round(s.yaw * _AXIS_SCALE)),
            flags,
        )

    @staticmethod
    def decode(data: bytes) -> "RCPacket":
        if len(data) != RC_PACKET_SIZE:
            raise ValueError(f"RC packet must be {RC_PACKET_SIZE} bytes, got {len(data)}")
        ptype, seq, thr, roll, pitch, yaw, flags = struct.unpack(RC_FORMAT, data)
        if ptype != RC_PACKET_TYPE:
            raise ValueError(f"Not an RC packet (type 0x{ptype:02x})")
        if thr > _THROTTLE_SCALE or max(abs(roll), abs(pitch), abs(yaw)) > _AXIS_SCALE:
            raise ValueError("RC channel out of range")
        if flags >> 5:
            raise ValueError("Unknown RC flag bits set")
        return RCPacket(
            seq=seq,
            sticks=Sticks(
                throttle=thr / _THROTTLE_SCALE,
                roll=roll / _AXIS_SCALE,
                pitch=pitch / _AXIS_SCALE,
                yaw=yaw / _AXIS_SCALE,
            ),
            arm_switch=bool(flags & 1),
            mode=FlightMode((flags >> 1) & 0b11),
            failsafe_action=FailsafeAction((flags >> 3) & 0b11),
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
