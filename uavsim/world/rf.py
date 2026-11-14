"""Physical RF model: link budgets and SINR for the uplink and downlink.

Replaces the old "noise ramps linearly inside an 80 m circle" rule with
the quantities a radio engineer would compute:

    received power  P_r = P_t + G_t + G_r - PL(d)                [dBm]
    interference    I   = sum over jammers that are on (in mW)
    SINR            = P_r - 10*log10(N + I)                      [dB]

for each receiver separately:

* **Uplink** (RC frames): transmitter = ground station, receiver = UAV.
* **Downlink** (telemetry): transmitter = UAV, receiver = ground station.

So a jammer near the UAV can kill the control link while the ground
station still hears telemetry fine, and vice versa.

Path loss is a log-distance model anchored at free space (1 m, 2.44 GHz,
~40 dB). The exponent depends on how high the *higher* antenna of the
pair is: a link with an elevated end is close to line of sight (n ~ 2),
a link with both ends near the ground suffers ground reflection and
clutter (n ~ 2.9). This is a deliberately simple air-to-ground model;
fading/shadowing is not included.

Mapping to the simulated channel: the radio channel adds complex AWGN of
standard deviation ``noise_voltage`` to a unit-power GMSK signal, so the
per-sample SNR is ``1 / noise_voltage**2``. ``sinr_to_noise_voltage``
inverts that, which ties the old calibration points together: SINR 3 dB
~ noise 0.71 (old jammer centre, link dead), 16.5 dB ~ 0.15 (old edge),
46 dB ~ 0.005 (old clean floor).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from uavsim.entities.jammer import Jammer

SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class RFParams:
    frequency_hz: float = 2.44e9          # 2.4 GHz ISM, where most RC links live
    noise_floor_dbm: float = -108.0       # kTB over ~1 MHz + 6 dB noise figure
    ground_station_tx_dbm: float = 27.0   # 500 mW RC transmitter (ELRS / Crossfire class)
    uav_tx_dbm: float = 20.0              # 100 mW telemetry transmitter
    ground_station_gain_dbi: float = 2.0
    uav_gain_dbi: float = 2.0
    exponent_los: float = 2.0             # path-loss exponent with an elevated antenna
    exponent_ground: float = 2.9          # both antennas near the ground
    los_height: float = 30.0              # m: higher antenna above this -> exponent_los
    min_distance: float = 1.0             # m, keeps the log model finite
    uav_antenna_height: float = 0.1       # m above the UAV's reference point

    @property
    def reference_loss_db(self) -> float:
        """Free-space loss at 1 m for this frequency."""
        return 20.0 * math.log10(4.0 * math.pi * self.frequency_hz / SPEED_OF_LIGHT)


@dataclass(frozen=True)
class LinkBudget:
    uplink_signal_dbm: float
    uplink_interference_dbm: float     # jammers only (-inf if none on)
    uplink_sinr_db: float
    downlink_signal_dbm: float
    downlink_interference_dbm: float
    downlink_sinr_db: float


def path_loss_db(distance, height_a, height_b, params: RFParams):
    """Log-distance path loss with a height-dependent exponent (vectorised)."""
    d = np.maximum(np.asarray(distance, dtype=float), params.min_distance)
    h = np.maximum(np.asarray(height_a, dtype=float), np.asarray(height_b, dtype=float))
    blend = np.clip(1.0 - h / params.los_height, 0.0, 1.0)
    n = params.exponent_los + (params.exponent_ground - params.exponent_los) * blend
    return params.reference_loss_db + 10.0 * n * np.log10(d)


def sinr_to_noise_voltage(sinr_db: float) -> float:
    """Noise voltage that gives this per-sample SINR on a unit-power signal."""
    return float(10.0 ** (-np.clip(sinr_db, -30.0, 120.0) / 20.0))


def apply_to_links(budget: LinkBudget, uplink, downlink) -> None:
    """Push the budget's SINRs into two radio channels (GNU Radio or loopback).

    Skips tiny changes so the GNU Radio worker queue is not flooded.
    """
    for link, sinr in ((uplink, budget.uplink_sinr_db), (downlink, budget.downlink_sinr_db)):
        nv = sinr_to_noise_voltage(sinr)
        if abs(nv - link.noise_voltage) > 0.01 * max(nv, 1e-6):
            link.noise_voltage = nv


def _dbm_to_mw(dbm):
    return 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


def _mw_to_dbm(mw):
    mw = np.asarray(mw, dtype=float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(mw)


class RFEnvironment:
    def __init__(self, params: RFParams = None) -> None:
        self.p = params or RFParams()

    # -- single evaluation (every frame) ---------------------------------
    def evaluate(
        self,
        uav_position: np.ndarray,
        ground_station: np.ndarray,
        jammers: Sequence[Jammer],
        t: float,
    ) -> LinkBudget:
        p = self.p
        uav = np.asarray(uav_position, dtype=float) + np.array([0.0, 0.0, p.uav_antenna_height])
        gs = np.asarray(ground_station, dtype=float)
        pl_link = float(path_loss_db(np.linalg.norm(uav - gs), uav[2], gs[2], p))

        up_signal = p.ground_station_tx_dbm + p.ground_station_gain_dbi + p.uav_gain_dbi - pl_link
        down_signal = p.uav_tx_dbm + p.uav_gain_dbi + p.ground_station_gain_dbi - pl_link

        active = [j for j in jammers if j.is_on(t)]
        up_i = self._interference_mw(uav, p.uav_gain_dbi, active)
        down_i = self._interference_mw(gs, p.ground_station_gain_dbi, active)
        noise = _dbm_to_mw(p.noise_floor_dbm)
        return LinkBudget(
            uplink_signal_dbm=up_signal,
            uplink_interference_dbm=float(_mw_to_dbm(up_i)),
            uplink_sinr_db=float(up_signal - _mw_to_dbm(noise + up_i)),
            downlink_signal_dbm=down_signal,
            downlink_interference_dbm=float(_mw_to_dbm(down_i)),
            downlink_sinr_db=float(down_signal - _mw_to_dbm(noise + down_i)),
        )

    def _interference_mw(self, receiver: np.ndarray, rx_gain: float,
                         jammers: Iterable[Jammer]) -> float:
        total = 0.0
        for j in jammers:
            src = j.antenna_position
            pl = float(path_loss_db(np.linalg.norm(receiver - src), receiver[2], src[2], self.p))
            total += float(_dbm_to_mw(j.eirp_dbm + rx_gain - pl))
        return total

    # -- map (for the heatmap) -------------------------------------------
    def uplink_sinr_grid(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        altitude: float,
        ground_station: np.ndarray,
        jammers: Sequence[Jammer],
    ) -> np.ndarray:
        """Uplink SINR (dB) a UAV would see at each (x, y) at `altitude`,
        with every jammer on (worst case). Shape ``(len(ys), len(xs))``."""
        p = self.p
        gx, gy = np.meshgrid(xs, ys)
        z = altitude + p.uav_antenna_height
        gs = np.asarray(ground_station, dtype=float)
        d_gs = np.sqrt((gx - gs[0]) ** 2 + (gy - gs[1]) ** 2 + (z - gs[2]) ** 2)
        signal = (p.ground_station_tx_dbm + p.ground_station_gain_dbi + p.uav_gain_dbi
                  - path_loss_db(d_gs, z, gs[2], p))
        interference = np.zeros_like(gx)
        for j in jammers:
            src = j.antenna_position
            d = np.sqrt((gx - src[0]) ** 2 + (gy - src[1]) ** 2 + (z - src[2]) ** 2)
            interference += _dbm_to_mw(j.eirp_dbm + p.uav_gain_dbi - path_loss_db(d, z, src[2], p))
        return signal - _mw_to_dbm(_dbm_to_mw(p.noise_floor_dbm) + interference)
