"""Jammers: ground-based transmitters that raise the interference floor.

A jammer no longer has a hard "range circle". It radiates a power (dBm)
from an antenna a couple of metres above the ground, and the RF model
(`uavsim.world.rf`) works out how much of it reaches each receiver --
the UAV for the uplink, the ground station for the downlink -- and adds
it to the noise there. Whether that kills a link depends on geometry:
the jammer's distance to the receiver *versus* the wanted transmitter's.

Kinds:

* BARRAGE -- wideband noise, always on.
* PULSED  -- wideband noise, on for ``duty`` of every ``period`` seconds.
* SWEPT   -- sweeps across a wide band and only sits on our channel for
             ``duty * period`` seconds once per sweep.
* TONE    -- a continuous narrowband carrier (CW) right next to ours. The
             numpy radio simulates it as an actual tone added to the chips,
             so spreading with a good sequence can reject it; GNU Radio's
             channel cannot, so there it is counted as extra noise.

The temporal ones interact with the failsafe timers: a jammer whose "on"
bursts are shorter than the RX-loss timeout (0.1 s) degrades link quality
without ever triggering a failsafe.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class JammerKind(Enum):
    BARRAGE = "BARRAGE"
    PULSED = "PULSED"
    SWEPT = "SWEPT"
    TONE = "TONE"


# Default timing per kind: (period s, duty fraction)
_DEFAULT_TIMING = {
    JammerKind.BARRAGE: (1.0, 1.0),
    JammerKind.PULSED: (0.25, 0.3),
    JammerKind.SWEPT: (0.6, 0.12),
    JammerKind.TONE: (1.0, 1.0),
}


@dataclass
class Jammer:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # ground point
    kind: JammerKind = JammerKind.BARRAGE
    power_dbm: float = 20.0         # radiated power while on (100 mW)
    antenna_gain_dbi: float = 2.0
    antenna_height: float = 2.0     # m above `position`
    period: float = 1.0             # s
    duty: float = 1.0               # fraction of the period spent on our channel
    phase: float = 0.0              # s, offset so jammers are not synchronised

    @staticmethod
    def make(position: np.ndarray, kind: JammerKind, power_dbm: float = 20.0,
             phase: float = None) -> "Jammer":
        period, duty = _DEFAULT_TIMING[kind]
        if phase is None:
            phase = random.uniform(0.0, period)
        return Jammer(position=np.asarray(position, dtype=float), kind=kind,
                      power_dbm=power_dbm, period=period, duty=duty, phase=phase)

    @property
    def antenna_position(self) -> np.ndarray:
        return self.position + np.array([0.0, 0.0, self.antenna_height])

    @property
    def eirp_dbm(self) -> float:
        return self.power_dbm + self.antenna_gain_dbi

    def is_on(self, t: float) -> bool:
        if self.duty >= 1.0:
            return True
        return ((t + self.phase) % self.period) < self.duty * self.period
