import math

import numpy as np
import pytest
from conftest import Sim

from uavsim.control.failsafe import FailsafeStage
from uavsim.control.rc import FailsafeAction, FlightMode
from uavsim.entities.jammer import Jammer, JammerKind
from uavsim.world.rf import RFEnvironment, RFParams, path_loss_db, sinr_to_noise_voltage

GS = np.array([-4.0, 0.0, 1.5])


def barrage(x, y, power=20.0):
    return Jammer.make(np.array([x, y, 0.0]), JammerKind.BARRAGE, power, phase=0.0)


def test_free_space_reference_and_exponents():
    p = RFParams()
    assert p.reference_loss_db == pytest.approx(40.2, abs=0.1)
    # Elevated end -> free-space exponent 2: +20 dB per decade.
    assert path_loss_db(100.0, 50.0, 1.5, p) == pytest.approx(p.reference_loss_db + 40.0)
    # Both ends near the ground lose more over the same distance.
    assert path_loss_db(100.0, 2.0, 1.5, p) > path_loss_db(100.0, 40.0, 1.5, p) + 7.0


def test_sinr_to_noise_voltage_matches_old_calibration_points():
    assert sinr_to_noise_voltage(3.0) == pytest.approx(0.708, abs=0.01)
    assert sinr_to_noise_voltage(16.5) == pytest.approx(0.15, abs=0.005)
    assert sinr_to_noise_voltage(46.0) == pytest.approx(0.005, abs=0.0005)


def test_clean_sky_keeps_both_links_healthy_across_the_world():
    rf = RFEnvironment()
    for x, y in ((0, 0), (150, 0), (-190, 190)):
        b = rf.evaluate(np.array([x, y, 10.0]), GS, [], 0.0)
        assert b.uplink_sinr_db > 35.0 and b.downlink_sinr_db > 25.0
        assert math.isinf(b.uplink_interference_dbm)


def test_jammer_powers_add_up():
    rf = RFEnvironment()
    uav = np.array([50.0, 0.0, 10.0])
    one = rf.evaluate(uav, GS, [barrage(50, 60)], 0.0)
    two = rf.evaluate(uav, GS, [barrage(50, 60), barrage(50, -60)], 0.0)
    assert two.uplink_interference_dbm - one.uplink_interference_dbm == pytest.approx(3.01, abs=0.02)


def test_uplink_and_downlink_are_jammed_independently():
    rf = RFEnvironment()
    # Jammer next to the UAV: the UAV's receiver (uplink) suffers far more
    # than the ground station's (downlink), which is ~110 m from the jammer.
    jammer = barrage(110, 0)
    near_jammer = rf.evaluate(np.array([100.0, 0.0, 10.0]), GS, [jammer], 0.0)
    assert near_jammer.uplink_sinr_db < 0.0, "control link dead next to the jammer"
    assert near_jammer.downlink_sinr_db > near_jammer.uplink_sinr_db + 15.0

    jammer_at_home = barrage(-4, 15)
    far_uav = rf.evaluate(np.array([120.0, 0.0, 30.0]), GS, [jammer_at_home], 0.0)
    assert far_uav.downlink_sinr_db < far_uav.uplink_sinr_db - 5.0


def test_jammer_timing():
    ts = np.arange(0.0, 10.0, 0.001)
    pulsed = Jammer.make(np.zeros(3), JammerKind.PULSED, phase=0.0)
    swept = Jammer.make(np.zeros(3), JammerKind.SWEPT, phase=0.0)
    assert np.mean([pulsed.is_on(t) for t in ts]) == pytest.approx(0.3, abs=0.01)
    assert np.mean([swept.is_on(t) for t in ts]) == pytest.approx(0.12, abs=0.01)
    assert swept.duty * swept.period < 0.1, "a sweep dwell shorter than the RX-loss timeout"
    assert all(barrage(0, 0).is_on(t) for t in ts[::97])


def test_heatmap_grid_matches_point_evaluation():
    rf = RFEnvironment()
    jammers = [barrage(60, 40), Jammer.make(np.array([-80.0, 10.0, 0.0]), JammerKind.PULSED,
                                            phase=0.0)]
    xs = np.array([-50.0, 0.0, 70.0])
    ys = np.array([-20.0, 35.0])
    grid = rf.uplink_sinr_grid(xs, ys, 12.0, GS, jammers)
    always_on = [Jammer(position=j.position, kind=JammerKind.BARRAGE, power_dbm=j.power_dbm)
                 for j in jammers]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            point = rf.evaluate(np.array([x, y, 12.0]), GS, always_on, 0.0)
            assert grid[iy, ix] == pytest.approx(point.uplink_sinr_db, abs=1e-9)


def test_flying_into_a_barrage_jammer_triggers_rth_home():
    sim = Sim(mode=FlightMode.POS_HOLD, jammers=[barrage(120, 0)])
    sim.op.failsafe_action = FailsafeAction.RTH
    sim.take_off(6.0)
    while sim.uav.failsafe.stage is not FailsafeStage.STAGE2:
        sim.run(0.1, keys={"i"})
        assert sim.t < 40.0, "never lost the link flying at the jammer"
    lost_at = sim.uav.body.position[0]
    assert 30.0 < lost_at < 110.0
    while sim.uav.armed:
        sim.run(0.5)
        assert sim.t < 120.0, "RTH did not finish"
    assert np.linalg.norm(sim.uav.body.position[:2] - sim.uav.home[:2]) < 1.5
    assert sim.uav.health == 100.0


def test_short_pulses_degrade_lq_without_failsafe():
    # 10 W pulses next to the take-off point: link dead while on, clean while off.
    pulsed = Jammer.make(np.array([8.0, 0.0, 0.0]), JammerKind.PULSED, 40.0, phase=0.0)
    sim = Sim(mode=FlightMode.ALT_HOLD, jammers=[pulsed])
    sim.arm()
    worst = FailsafeStage.OK
    lq = []
    for _ in range(40):
        sim.run(0.1)
        worst = max(worst, sim.uav.failsafe.stage)
        lq.append(sim.uav.link_quality())
    assert worst < FailsafeStage.STAGE1
    assert 55.0 < np.mean(lq[10:]) < 85.0
    assert sim.uav.armed
