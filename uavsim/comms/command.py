"""Bit-level command protocol shared between UAVOperator and UAV.

Commands are packed into a single byte so they can be transmitted over a
CommGateway exactly like a real digital radio link would transmit opcodes.
The receiver (UAV) owns the decoding algorithm and translates the bits back
into a `Command` it then applies to a specific motor.

Byte layout (MSB -> LSB):

    [7 6 5] opcode      (3 bits, up to 8 opcodes)
    [4 3]   motor_id    (2 bits, motors 0-3)
    [2 1 0] reserved    (3 bits, free for future use, e.g. jammer/noise flags)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CommandOpcode(IntEnum):
    """High level action requested by the operator."""

    THROTTLE_UP = 0b000
    THROTTLE_DOWN = 0b001
    ARM = 0b010
    DISARM = 0b011
    EMERGENCY_CUT = 0b100


_OPCODE_SHIFT = 5
_MOTOR_ID_SHIFT = 3
_OPCODE_MASK = 0b111
_MOTOR_ID_MASK = 0b11
_RESERVED_MASK = 0b111


@dataclass(frozen=True)
class Command:
    """A single decoded instruction understood by the UAV."""

    opcode: CommandOpcode
    motor_id: int = 0
    reserved: int = 0

    def encode(self) -> bytes:
        """Pack this command into a single byte for transmission."""
        if not 0 <= self.motor_id <= _MOTOR_ID_MASK:
            raise ValueError(f"motor_id must be in [0, {_MOTOR_ID_MASK}]")
        if not 0 <= self.reserved <= _RESERVED_MASK:
            raise ValueError(f"reserved must be in [0, {_RESERVED_MASK}]")

        packed = (
            ((int(self.opcode) & _OPCODE_MASK) << _OPCODE_SHIFT)
            | ((self.motor_id & _MOTOR_ID_MASK) << _MOTOR_ID_SHIFT)
            | (self.reserved & _RESERVED_MASK)
        )
        return bytes([packed])

    @staticmethod
    def decode(data: bytes) -> "Command":
        """Unpack a single byte received over a CommGateway into a Command.

        This is the "algorithm in the receiver" that translates the raw
        signal into an instruction the UAV can act on.
        """
        if len(data) != 1:
            raise ValueError("Command.decode expects exactly one byte")

        packed = data[0]
        opcode_bits = (packed >> _OPCODE_SHIFT) & _OPCODE_MASK
        motor_id = (packed >> _MOTOR_ID_SHIFT) & _MOTOR_ID_MASK
        reserved = packed & _RESERVED_MASK

        try:
            opcode = CommandOpcode(opcode_bits)
        except ValueError as exc:
            raise ValueError(f"Unknown opcode bits: {opcode_bits:03b}") from exc

        return Command(opcode=opcode, motor_id=motor_id, reserved=reserved)
