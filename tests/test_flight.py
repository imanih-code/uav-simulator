import math

import numpy as np
from conftest import Sim

from uavsim.control.failsafe import FailsafeStage
from uavsim.control.flight_controller import Mixer
from uavsim.control.rc import FailsafeAction, FlightMode, Sticks
from uavsim.physics.motor import YAW_TORQUE_PER_NEWTON


def test_mixer_reproduces_requested_wrench():
    off = 0.225 / math.sqrt(2)
    positions = [np.array(p) for p in ([off, -off, 0], [-off, -off, 0], [-off, off, 0], [off, off, 0])]
    mixer = Mixer(positions, [1, -1, 1, -1], YAW_TORQUE_PER_NEWTON, 6.0)
    wanted = np.array([10.0, 0.05, -0.08, 0.01])
    thrusts = mixer.mix(wanted[0], wanted[1:], min_thrust=0.0)
    np.testing.assert_allclose(mixer.allocation @ thrusts, wanted, atol=1e-9)


def test_mixer_keeps_attitude_authority_at_full_throttle():
    off = 0.225 / math.sqrt(2)
    positions = [np.array(p) for p in ([off, -off, 0], [-off, -off, 0], [-off, off, 0], [off, off, 0])]
    mixer = Mixer(positions, [1, -1, 1, -1], YAW_TORQUE_PER_NEWTON, 6.0)
    thrusts = mixer.mix(24.0, np.array([0.3, 0.0, 0.0]), min_thrust=0.0)
    roll_torque = (mixer.allocation @ thrusts)[1]
    assert thrusts.max() <= 6.0
    assert abs(roll_torque - 0.3) < 1e-9  # collective was lowered, not the roll torque


def test_arming_requires_switch_cycle_and_low_throttle():
    sim = Sim(mode=FlightMode.ANGLE)
    sim.op.arm_switch = True
    sim.run(0.2, sticks=Sticks(throttle=0.0))
    assert not sim.uav.armed, "arm switch was never seen OFF"
    sim.op.arm_switch = False
    sim.run(0.1, sticks=Sticks(throttle=0.5))
    sim.op.arm_switch = True
    sim.run(0.2, sticks=Sticks(throttle=0.5))
    assert not sim.uav.armed, "armed with throttle up"
    sim.run(0.2, sticks=Sticks(throttle=0.0))
    assert sim.uav.armed


def test_alt_hold_takeoff_and_hover_is_stable(sim):
    sim.take_off(5.0)
    z0 = sim.uav.body.position[2]
    track = []
    sim.run(5.0, record=track)
    z = np.array([p[2] for _, p, _ in track])
    assert abs(z.max() - z.min()) < 0.15
    assert abs(sim.uav.body.velocity[2]) < 0.1
    assert np.degrees(np.abs(sim.uav.body.attitude_rpy()[:2])).max() < 1.0
    assert 4.5 < z0 < 7.0


def test_hover_needs_about_41_percent_thrust(sim):
    sim.take_off(4.0)
    speeds = np.array([m.speed for m in sim.uav.motors])
    thrust_fraction = float(np.mean(speeds ** 2))
    assert 0.38 < thrust_fraction < 0.45


def test_pos_hold_flies_forward_then_stops():
    sim = Sim(mode=FlightMode.POS_HOLD)
    sim.take_off(5.0)
    x0 = sim.uav.body.position[0]
    sim.run(2.0, keys={"i"})
    assert sim.uav.body.velocity[0] > 3.0, "pitch forward should fly along +X at yaw 0"
    sim.run(4.0)
    stopped_at = sim.uav.body.position.copy()
    sim.run(3.0)
    assert np.linalg.norm(sim.uav.body.velocity) < 0.2
    assert np.linalg.norm(sim.uav.body.position[:2] - stopped_at[:2]) < 0.3
    assert sim.uav.body.position[0] - x0 > 5.0


def test_angle_mode_tracks_stick_angle():
    sim = Sim(mode=FlightMode.ANGLE)
    sim.arm()
    sim.run(2.0, sticks=Sticks(throttle=0.6))
    sim.run(1.0, sticks=Sticks(throttle=0.45, roll=0.5))
    roll_deg = math.degrees(sim.uav.body.attitude_rpy()[0])
    assert abs(roll_deg - 17.5) < 2.0
    sim.run(1.0, sticks=Sticks(throttle=0.45))
    assert abs(math.degrees(sim.uav.body.attitude_rpy()[0])) < 1.0


def test_acro_mode_tracks_body_rate():
    sim = Sim(mode=FlightMode.ACRO)
    sim.arm()
    sim.run(1.5, sticks=Sticks(throttle=0.6))
    sim.run(0.3, sticks=Sticks(throttle=0.45, pitch=0.25))
    rate = math.degrees(sim.uav.body.angular_velocity_body[1])
    assert abs(rate - 100.0) < 8.0


def test_failsafe_stages_then_rth_lands_home_and_disarms():
    sim = Sim(mode=FlightMode.POS_HOLD)
    sim.op.failsafe_action = FailsafeAction.RTH
    sim.take_off(6.0)
    sim.run(3.0, keys={"i"})
    sim.run(3.0)
    assert np.linalg.norm(sim.uav.body.position[:2]) > 15.0

    sim.op.tx_enabled = False
    track = []
    sim.run(45.0, record=track)
    stages = [stage for _, _, stage in track]
    first = {s: next(t for t, _, st in track if st == s) - track[0][0] for s in set(stages)}
    # Stages are timed from the last frame received, up to one RC period
    # (20 ms) before the transmitter was switched off.
    assert 0.08 < first[FailsafeStage.HOLD_LAST] < 0.12
    assert 0.27 < first[FailsafeStage.STAGE1] < 0.31
    assert 0.97 < first[FailsafeStage.STAGE2] < 1.01
    assert max(p[2] for _, p, _ in track) > 14.0, "RTH should climb to the safe altitude"
    assert not sim.uav.armed
    assert sim.uav.is_grounded
    assert np.linalg.norm(sim.uav.body.position[:2] - sim.uav.home[:2]) < 1.5
    assert sim.uav.health == 100.0, "RTH landing must be gentle"


def test_short_dropout_recovers_immediately():
    sim = Sim(mode=FlightMode.POS_HOLD)
    sim.take_off(5.0)
    sim.op.tx_enabled = False
    sim.run(0.6)
    assert sim.uav.failsafe.stage is FailsafeStage.STAGE1
    sim.op.tx_enabled = True
    sim.run(0.05)
    assert sim.uav.failsafe.stage is FailsafeStage.OK


def test_stage2_needs_sustained_link_to_recover():
    sim = Sim(mode=FlightMode.POS_HOLD)
    sim.op.failsafe_action = FailsafeAction.HOLD
    sim.take_off(5.0)
    hold_point = sim.uav.body.position.copy()
    sim.op.tx_enabled = False
    sim.run(3.0)
    assert sim.uav.failsafe.stage is FailsafeStage.STAGE2
    assert np.linalg.norm(sim.uav.body.position - hold_point) < 0.5
    sim.op.tx_enabled = True
    sim.run(2.0)
    assert sim.uav.failsafe.stage is FailsafeStage.STAGE2, "hands-off pilot: keep the action"
    sim.run(0.2, keys={"l"})
    assert sim.uav.failsafe.stage is FailsafeStage.OK
    assert sim.uav.armed


def test_stage2_ignores_stick_motion_until_link_is_stable():
    sim = Sim(mode=FlightMode.POS_HOLD)
    sim.op.failsafe_action = FailsafeAction.HOLD
    sim.take_off(5.0)
    sim.op.tx_enabled = False
    sim.run(2.0)
    sim.op.tx_enabled = True
    sim.run(0.3, sticks=Sticks(throttle=0.5, roll=0.8))
    assert sim.uav.failsafe.stage is FailsafeStage.STAGE2
    sim.run(0.3, sticks=Sticks(throttle=0.5, roll=0.8))
    assert sim.uav.failsafe.stage is FailsafeStage.OK


def test_mode_change_after_failsafe_does_not_reuse_stale_altitude():
    # Hover at 5 m latches the ALT_HOLD target; RTH then climbs to 15 m.
    # Taking over at 15 m must hold 15 m, not dive back to the old 5 m latch.
    sim = Sim(mode=FlightMode.ALT_HOLD)
    sim.op.failsafe_action = FailsafeAction.RTH
    sim.take_off(5.0)
    sim.run(2.0, keys={"i"})
    sim.op.tx_enabled = False
    while sim.uav.body.position[2] < 14.5:
        sim.run(0.1)
        assert sim.t < 60.0
    sim.op.tx_enabled = True
    sim.run(0.7, keys={"l"})
    assert sim.uav.failsafe.stage is FailsafeStage.OK
    z_takeover = sim.uav.body.position[2]
    sim.run(4.0)
    assert abs(sim.uav.body.position[2] - z_takeover) < 1.0


def test_drop_failsafe_cuts_motors():
    sim = Sim(mode=FlightMode.ALT_HOLD)
    sim.op.failsafe_action = FailsafeAction.DROP
    sim.take_off(5.0)
    sim.op.tx_enabled = False
    sim.run(1.2)
    assert not sim.uav.armed
    assert sim.uav.body.velocity[2] < 0.0


def test_pilot_disarm_in_flight_falls():
    sim = Sim(mode=FlightMode.ALT_HOLD)
    sim.take_off(5.0)
    sim.op.arm_switch = False
    sim.run(0.5)
    assert not sim.uav.armed
    assert sim.uav.body.velocity[2] < -2.0


def test_uplink_lq_is_reported_in_telemetry():
    sim = Sim(mode=FlightMode.ALT_HOLD)
    sim.arm()
    sim.run(1.5)
    tlm = sim.op.latest_telemetry
    assert tlm is not None and tlm.armed
    assert tlm.uplink_lq >= 96.0
    assert sim.op.downlink_quality(sim.t) >= 90.0
