"""Link-loss detection and failsafe, the part that decides what a jammed
drone actually does.

Modelled on how real flight stacks behave (Betaflight / INAV / ArduPilot
all share the same broad shape):

    link OK ──(no valid RC frame for loss_timeout)──> HOLD_LAST
    HOLD_LAST: keep flying on the last valid sticks (short glitches are normal)
    STAGE1   : after hold_last_time -> sticks centred, aircraft self-levels
               (ACRO is forced to ANGLE, ALT/POS hold altitude/position)
    STAGE2   : after stage2_time -> the configured action takes over:
               HOLD (hover in place), LAND, RTH (return to home) or DROP

    HOLD_LAST / STAGE1 recover as soon as a valid frame arrives.
    STAGE2 only hands control back when BOTH hold:
      * the link has been continuously good for recovery_time -- a jammer
        that lets one frame through now and then must not yank the
        aircraft back and forth;
      * the pilot moves a stick (roll/pitch/yaw beyond override_threshold),
        like INAV / ArduPilot. Flying out of a jammer's range during RTH
        must not drop a hands-off pilot back into manual mid-air.

All times are simulation time in seconds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Deque, Optional

import numpy as np

from uavsim.control.flight_controller import ControlInput, ControlKind, NavTarget
from uavsim.control.rc import FailsafeAction, FlightMode, Sticks


class FailsafeStage(IntEnum):
    OK = 0
    HOLD_LAST = 1
    STAGE1 = 2
    STAGE2 = 3


class RTHPhase(IntEnum):
    NONE = 0
    CLIMB = 1
    RETURN = 2
    DESCEND = 3


@dataclass
class FailsafeParams:
    loss_timeout: float = 0.10      # s without a valid frame -> "RX loss" (5 frames at 50 Hz)
    hold_last_time: float = 0.30    # s of loss during which the last sticks are kept
    stage2_time: float = 1.00       # s of loss before the failsafe action starts
    recovery_time: float = 0.50     # s of continuous good link to leave STAGE2
    rth_altitude: float = 15.0      # m, minimum altitude for the return leg
    nav_speed: float = 6.0          # m/s, RTH / HOLD horizontal speed limit
    land_rate: float = 1.0          # m/s, descent rate for LAND and the RTH final leg
    override_threshold: float = 0.3  # stick deflection that takes control back from STAGE2


class LinkMonitor:
    """Tracks valid RC frames: time since the last one and link quality.

    Link quality (LQ) is the percentage of expected frames that actually
    arrived over the last ``window`` seconds -- the same number ELRS and
    Crossfire show on the pilot's OSD.
    """

    def __init__(self, expected_rate_hz: float, window: float = 1.0) -> None:
        self.expected_rate_hz = expected_rate_hz
        self.window = window
        self._arrivals: Deque[float] = deque()
        self.first_packet_time: Optional[float] = None
        self.last_packet_time: Optional[float] = None
        self.last_seq: Optional[int] = None

    def reset(self) -> None:
        self._arrivals.clear()
        self.first_packet_time = None
        self.last_packet_time = None
        self.last_seq = None

    def packet_received(self, now: float, seq: int) -> None:
        if self.first_packet_time is None:
            self.first_packet_time = now
        self._arrivals.append(now)
        self.last_packet_time = now
        self.last_seq = seq

    def time_since_last(self, now: float) -> float:
        if self.last_packet_time is None:
            return float("inf")
        return now - self.last_packet_time

    def link_quality(self, now: float) -> float:
        return windowed_quality(self._arrivals, self.first_packet_time, now,
                                self.expected_rate_hz, self.window)


def windowed_quality(arrivals: Deque[float], first_time: Optional[float], now: float,
                     rate_hz: float, window: float) -> float:
    """% of expected packets received in the last `window` s (or since the first one)."""
    if first_time is None:
        return 0.0
    while arrivals and arrivals[0] < now - window:
        arrivals.popleft()
    span = min(window, now - first_time + 1.0 / rate_hz)
    return min(100.0, 100.0 * len(arrivals) / max(rate_hz * span, 1.0))


@dataclass
class FailsafeOutput:
    control: ControlInput
    force_disarm: bool = False


class Failsafe:
    def __init__(self, params: Optional[FailsafeParams] = None) -> None:
        self.p = params or FailsafeParams()
        self.reset()

    def reset(self) -> None:
        self.stage = FailsafeStage.OK
        self.action = FailsafeAction.HOLD
        self.rth_phase = RTHPhase.NONE
        self._good_since: Optional[float] = None
        self._anchor = np.zeros(3)   # where STAGE2 started
        self._rth_z = 0.0

    @property
    def active(self) -> bool:
        return self.stage != FailsafeStage.OK

    def update(
        self,
        now: float,
        time_since_packet: float,
        position: np.ndarray,
        home: np.ndarray,
        last_sticks: Sticks,
        last_mode: FlightMode,
        configured_action: FailsafeAction,
        touchdown: bool,
    ) -> FailsafeOutput:
        link_lost = time_since_packet > self.p.loss_timeout
        pilot_override = max(abs(last_sticks.roll), abs(last_sticks.pitch),
                             abs(last_sticks.yaw)) > self.p.override_threshold
        self._advance_stage(now, time_since_packet, link_lost, position, configured_action,
                            pilot_override)

        if self.stage in (FailsafeStage.OK, FailsafeStage.HOLD_LAST):
            return FailsafeOutput(ControlInput(ControlKind.MANUAL, last_mode, last_sticks))

        if self.stage is FailsafeStage.STAGE1:
            if last_mode in (FlightMode.ALT_HOLD, FlightMode.POS_HOLD):
                return FailsafeOutput(ControlInput(ControlKind.MANUAL, last_mode, Sticks(throttle=0.5)))
            # ACRO/ANGLE: level out, keep the last throttle.
            return FailsafeOutput(
                ControlInput(ControlKind.MANUAL, FlightMode.ANGLE, Sticks(throttle=last_sticks.throttle))
            )

        # STAGE2
        if self.action is FailsafeAction.DROP:
            return FailsafeOutput(ControlInput(ControlKind.MANUAL, last_mode, Sticks()), force_disarm=True)
        if touchdown and self.action in (FailsafeAction.LAND, FailsafeAction.RTH):
            return FailsafeOutput(ControlInput(ControlKind.MANUAL, last_mode, Sticks()), force_disarm=True)
        return FailsafeOutput(ControlInput(ControlKind.NAV, nav=self._nav_target(position, home)))

    # ------------------------------------------------------------------
    def _advance_stage(
        self,
        now: float,
        time_since_packet: float,
        link_lost: bool,
        position: np.ndarray,
        configured_action: FailsafeAction,
        pilot_override: bool,
    ) -> None:
        if self.stage is FailsafeStage.STAGE2:
            if link_lost:
                self._good_since = None
            elif self._good_since is None:
                self._good_since = now
            elif now - self._good_since >= self.p.recovery_time and pilot_override:
                self.stage = FailsafeStage.OK
                self.rth_phase = RTHPhase.NONE
                self._good_since = None
            return

        if not link_lost:
            self.stage = FailsafeStage.OK
            return

        if time_since_packet >= self.p.stage2_time:
            self.stage = FailsafeStage.STAGE2
            self.action = configured_action
            self._anchor = np.asarray(position, dtype=float).copy()
            self._rth_z = max(float(position[2]), self.p.rth_altitude)
            self.rth_phase = RTHPhase.CLIMB if configured_action is FailsafeAction.RTH else RTHPhase.NONE
            self._good_since = None
        elif time_since_packet >= self.p.hold_last_time:
            self.stage = FailsafeStage.STAGE1
        else:
            self.stage = FailsafeStage.HOLD_LAST

    def _nav_target(self, position: np.ndarray, home: np.ndarray) -> NavTarget:
        p = self.p
        if self.action is FailsafeAction.HOLD:
            return NavTarget(self._anchor.copy(), p.nav_speed)
        if self.action is FailsafeAction.LAND:
            return NavTarget(self._anchor.copy(), p.nav_speed, descend_rate=p.land_rate)

        # RTH: climb in place -> fly home at altitude -> descend.
        if self.rth_phase is RTHPhase.CLIMB:
            if position[2] >= self._rth_z - 1.0:
                self.rth_phase = RTHPhase.RETURN
            else:
                return NavTarget(np.array([*self._anchor[:2], self._rth_z]), p.nav_speed)
        if self.rth_phase is RTHPhase.RETURN:
            if np.linalg.norm(position[:2] - home[:2]) <= 1.5:
                self.rth_phase = RTHPhase.DESCEND
            else:
                return NavTarget(np.array([home[0], home[1], self._rth_z]), p.nav_speed)
        return NavTarget(np.array([home[0], home[1], position[2]]), p.nav_speed,
                         descend_rate=p.land_rate)
