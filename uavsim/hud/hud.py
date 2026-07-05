"""HUD: reads everything it shows either from telemetry the UAVOperator has
buffered, or straight from the gateways (bandwidth, raw signal history).
The TX panel reads from the UAV's command input (Rx side of the command
link) so it shows what actually survived the noisy channel. The RX panel
reads from the Operator's telemetry input (Rx side of the telemetry link).
Actual drawing lives in the rendering package; this class only decides
"what to show".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from uavsim.comms.gateway import CommGatewayInput
from uavsim.entities.operator import UAVOperator

# How much signal history each oscilloscope-style HUD graph keeps on screen.
SIGNAL_WINDOW_SECONDS = 1.5


@dataclass(frozen=True)
class HUDSnapshot:
    has_telemetry: bool
    position: Optional[Tuple[float, float, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None
    attitude_deg: Optional[Tuple[float, float, float]] = None
    motor_throttle: Optional[Tuple[float, float, float, float]] = None
    battery_percent: Optional[float] = None
    health_percent: Optional[float] = None
    mass_kg: Optional[float] = None

    uplink_bandwidth_bps: float = 0.0
    downlink_bandwidth_bps: float = 0.0
    uplink_signal: Tuple[Tuple[float, bytes], ...] = field(default_factory=tuple)
    downlink_signal: Tuple[Tuple[float, bytes], ...] = field(default_factory=tuple)
    uplink_raw: Tuple[Tuple[float, np.ndarray], ...] = field(default_factory=tuple)
    downlink_raw: Tuple[Tuple[float, np.ndarray], ...] = field(default_factory=tuple)
    now: float = 0.0

    command_log: Tuple[Tuple[str, bool], ...] = ()  # (label, is_valid)

_OPCODE_LABELS = {
    0: "THR+", 1: "THR-", 2: "ARM", 3: "DSRM", 4: "CUT",
    5: "THR+A", 6: "THR-A",
}


class HUD:
    def __init__(self, operator: UAVOperator, uav_command_input: CommGatewayInput) -> None:
        self._operator = operator
        self._uav_command_input = uav_command_input

    def refresh(self) -> HUDSnapshot:
        """Pull the freshest telemetry plus live channel stats."""
        packet = self._operator.poll_telemetry()
        now = time.monotonic()

        # TX panel shows what the UAV actually received (after noise/demod/CRC),
        # not what the Operator transmitted — so we read from the Rx side.
        uplink_bandwidth = self._uav_command_input.bandwidth_bps()
        downlink_bandwidth = self._operator.telemetry_input.bandwidth_bps()
        # Request all available history (up to the channel's 3s prune) so
        # multi-byte telemetry packets don't vanish at the left edge when
        # their timestamp passes the display window cutoff — later bytes
        # of the same packet may still be visible.
        uplink_signal = tuple(
            self._uav_command_input.recent_transmissions()
        )
        downlink_signal = tuple(
            self._operator.telemetry_input.recent_transmissions()
        )
        uplink_raw = tuple(
            self._uav_command_input.recent_raw_transmissions()
        )
        downlink_raw = tuple(
            self._operator.telemetry_input.recent_raw_transmissions()
        )

        command_log = []
        if packet is not None:
            for opcode, motor_id, valid in packet.command_log:
                if not valid:
                    command_log.append(("BAD", False))
                else:
                    label = _OPCODE_LABELS.get(opcode, f"OP{opcode}")
                    if motor_id < 4:
                        label += str(motor_id + 1)
                    command_log.append((label, True))

        if packet is None:
            return HUDSnapshot(
                has_telemetry=False,
                uplink_bandwidth_bps=uplink_bandwidth,
                downlink_bandwidth_bps=downlink_bandwidth,
                uplink_signal=uplink_signal,
                downlink_signal=downlink_signal,
                uplink_raw=uplink_raw,
                downlink_raw=downlink_raw,
                now=now,
                command_log=tuple(command_log),
            )

        attitude_deg = tuple(np.degrees(packet.attitude_rpy).tolist())
        return HUDSnapshot(
            has_telemetry=True,
            position=tuple(packet.position.tolist()),
            velocity=tuple(packet.velocity.tolist()),
            attitude_deg=attitude_deg,
            motor_throttle=tuple(packet.motor_throttle.tolist()),
            battery_percent=packet.battery_percent,
            health_percent=packet.health_percent,
            mass_kg=packet.mass,
            uplink_bandwidth_bps=uplink_bandwidth,
            downlink_bandwidth_bps=downlink_bandwidth,
            uplink_signal=uplink_signal,
            downlink_signal=downlink_signal,
            uplink_raw=uplink_raw,
            downlink_raw=downlink_raw,
            now=now,
            command_log=tuple(command_log),
        )
