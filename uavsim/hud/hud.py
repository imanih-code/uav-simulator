"""HUD: reads everything it shows either from telemetry the UAVOperator has
buffered, from the operator's own transmitter state, or straight from the
gateways (bandwidth, raw signal history). It never reads the UAV directly:
when the downlink is jammed the flight data freezes / disappears, exactly
as it would on a real ground station.

The TX panel reads from the UAV's command input (Rx side of the command
link) so it shows what actually survived the noisy channel. The RX panel
reads from the Operator's telemetry input. Drawing lives in the rendering
package; this class only decides "what to show".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from uavsim.comms.gateway import CommGatewayInput
from uavsim.comms.telemetry import TelemetryPacket
from uavsim.entities.operator import UAVOperator

# How much signal history each oscilloscope-style HUD graph keeps on screen.
SIGNAL_WINDOW_SECONDS = 1.5
# Telemetry older than this is shown as stale ("TLM LOST").
TELEMETRY_STALE_SECONDS = 0.5

Stick4 = Tuple[float, float, float, float]  # throttle, roll, pitch, yaw


@dataclass(frozen=True)
class HUDSnapshot:
    has_telemetry: bool
    position: Optional[Tuple[float, float, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None
    attitude_deg: Optional[Tuple[float, float, float]] = None
    motor_throttle: Optional[Stick4] = None
    battery_percent: Optional[float] = None
    health_percent: Optional[float] = None
    mass_kg: Optional[float] = None

    # Uplink as seen by the UAV (from telemetry)
    rx_sticks: Optional[Stick4] = None
    uplink_lq: Optional[float] = None
    flight_mode: Optional[int] = None
    failsafe_stage: int = 0
    failsafe_action: int = 0
    armed: bool = False
    landed: bool = True
    telemetry_stale: bool = True

    # Transmitter state (operator side, always available)
    tx_sticks: Stick4 = (0.0, 0.0, 0.0, 0.0)
    tx_mode: int = 0
    tx_failsafe_action: int = 0
    tx_arm_switch: bool = False
    tx_enabled: bool = True
    downlink_lq: float = 0.0

    uplink_bandwidth_bps: float = 0.0
    downlink_bandwidth_bps: float = 0.0
    uplink_signal: Tuple[Tuple[float, bytes], ...] = field(default_factory=tuple)
    downlink_signal: Tuple[Tuple[float, bytes], ...] = field(default_factory=tuple)
    uplink_raw: Tuple[Tuple[float, np.ndarray], ...] = field(default_factory=tuple)
    downlink_raw: Tuple[Tuple[float, np.ndarray], ...] = field(default_factory=tuple)
    now: float = 0.0

    uplink_correlation: float = 0.0
    downlink_correlation: float = 0.0
    worker_error: str = ""


# Physical plausibility bounds for telemetry fields.
_WORLD_BOUND = 500.0
_VELOCITY_MAX = 500.0
_ANGLE_MAX = 100.0
_ANGULAR_VEL_MAX = 1000.0


def _is_plausible(packet: TelemetryPacket) -> bool:
    checks = (
        all(-_WORLD_BOUND <= v <= _WORLD_BOUND for v in packet.position),
        all(-_VELOCITY_MAX <= v <= _VELOCITY_MAX for v in packet.velocity),
        all(-_ANGLE_MAX <= v <= _ANGLE_MAX for v in packet.attitude_rpy),
        all(-_ANGULAR_VEL_MAX <= v <= _ANGULAR_VEL_MAX for v in packet.angular_velocity),
        all(0.0 <= v <= 1.0 for v in packet.motor_throttle),
        0.0 <= packet.battery_percent <= 100.0,
        0.0 <= packet.health_percent <= 100.0,
        0.1 <= packet.mass <= 10.0,
    )
    return all(checks)


class HUD:
    def __init__(self, operator: UAVOperator, uav_command_input: CommGatewayInput) -> None:
        self._operator = operator
        self._uav_command_input = uav_command_input

    def refresh(
        self,
        sim_time: float,
        worker_error: str = "",
        uplink_correlation: float = 0.0,
        downlink_correlation: float = 0.0,
    ) -> HUDSnapshot:
        op = self._operator
        packet = op.poll_telemetry(sim_time)
        s = op.sticks

        common = dict(
            tx_sticks=(s.throttle, s.roll, s.pitch, s.yaw),
            tx_mode=int(op.mode),
            tx_failsafe_action=int(op.failsafe_action),
            tx_arm_switch=op.arm_switch,
            tx_enabled=op.tx_enabled,
            downlink_lq=op.downlink_quality(sim_time),
            uplink_bandwidth_bps=self._uav_command_input.bandwidth_bps(),
            downlink_bandwidth_bps=op.telemetry_input.bandwidth_bps(),
            uplink_signal=tuple(self._uav_command_input.recent_transmissions()),
            downlink_signal=tuple(op.telemetry_input.recent_transmissions()),
            uplink_raw=tuple(self._uav_command_input.recent_raw_transmissions()),
            downlink_raw=tuple(op.telemetry_input.recent_raw_transmissions()),
            now=time.monotonic(),
            uplink_correlation=uplink_correlation,
            downlink_correlation=downlink_correlation,
            worker_error=worker_error,
        )

        if packet is None or not _is_plausible(packet):
            return HUDSnapshot(has_telemetry=False, **common)

        return HUDSnapshot(
            has_telemetry=True,
            position=tuple(packet.position.tolist()),
            velocity=tuple(packet.velocity.tolist()),
            attitude_deg=tuple(np.degrees(packet.attitude_rpy).tolist()),
            motor_throttle=tuple(packet.motor_throttle.tolist()),
            battery_percent=packet.battery_percent,
            health_percent=packet.health_percent,
            mass_kg=packet.mass,
            rx_sticks=packet.applied_sticks,
            uplink_lq=packet.uplink_lq,
            flight_mode=packet.flight_mode,
            failsafe_stage=packet.failsafe_stage,
            failsafe_action=packet.failsafe_action,
            armed=packet.armed,
            landed=packet.landed,
            telemetry_stale=op.telemetry_age(sim_time) > TELEMETRY_STALE_SECONDS,
            **common,
        )
