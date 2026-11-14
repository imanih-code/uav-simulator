import numpy as np
import pytest

from uavsim.comms.comm_chaos_adapter import ChaoticLayer
from uavsim.comms.loopback_link import (
    LoopbackChannel,
    chip_error_probability,
    packet_success_probability,
)
from uavsim.comms.telemetry import TelemetryPacket
from uavsim.control.rc import (
    RC_PACKET_SIZE,
    FailsafeAction,
    FlightMode,
    RCPacket,
    Sticks,
)


def test_rc_packet_roundtrip():
    packet = RCPacket(
        seq=255,
        sticks=Sticks(throttle=0.734, roll=-0.5, pitch=1.0, yaw=0.002),
        arm_switch=True,
        mode=FlightMode.POS_HOLD,
        failsafe_action=FailsafeAction.RTH,
    )
    raw = packet.encode()
    assert len(raw) == RC_PACKET_SIZE == 11
    decoded = RCPacket.decode(raw)
    assert decoded.seq == 255
    assert decoded.sticks == packet.sticks.quantized()
    assert decoded.arm_switch and decoded.mode is FlightMode.POS_HOLD
    assert decoded.failsafe_action is FailsafeAction.RTH


def test_rc_packet_clamps_and_rejects_garbage():
    raw = RCPacket(1, Sticks(throttle=3.0, roll=-9.0), False, FlightMode.ACRO,
                   FailsafeAction.DROP).encode()
    decoded = RCPacket.decode(raw)
    assert decoded.sticks.throttle == 1.0 and decoded.sticks.roll == -1.0
    with pytest.raises(ValueError):
        RCPacket.decode(raw[:-1])
    with pytest.raises(ValueError):
        RCPacket.decode(b"\x00" + raw[1:])


def test_telemetry_roundtrip():
    packet = TelemetryPacket(
        timestamp=12.5,
        position=np.array([1.0, -2.0, 3.0]),
        velocity=np.array([0.1, 0.2, -0.3]),
        attitude_rpy=np.array([0.01, -0.02, 1.5]),
        angular_velocity=np.array([0.0, 0.1, -0.1]),
        motor_throttle=np.array([0.6, 0.62, 0.64, 0.66]),
        battery_percent=88.0,
        mass=1.0,
        health_percent=75.0,
        seq=4242,
        applied_sticks=(0.5, -0.25, 0.5, 0.0),
        uplink_lq=96.0,
        flight_mode=int(FlightMode.ALT_HOLD),
        failsafe_stage=3,
        failsafe_action=int(FailsafeAction.LAND),
        armed=True,
        landed=False,
    )
    raw = packet.encode()
    assert len(raw) == 56
    decoded = TelemetryPacket.decode(raw)
    assert decoded.seq == 4242
    np.testing.assert_allclose(decoded.position, packet.position, atol=1e-5)
    np.testing.assert_allclose(decoded.velocity, packet.velocity, atol=0.005)
    np.testing.assert_allclose(decoded.attitude_rpy, packet.attitude_rpy, atol=1e-4)
    np.testing.assert_allclose(decoded.angular_velocity, packet.angular_velocity, atol=1e-3)
    np.testing.assert_allclose(decoded.motor_throttle, packet.motor_throttle, atol=0.002)
    assert decoded.battery_percent == 88.0 and decoded.health_percent == 75.0
    assert decoded.mass == 1.0
    assert decoded.applied_sticks == packet.applied_sticks
    assert decoded.uplink_lq == 96.0
    assert decoded.failsafe_stage == 3 and decoded.failsafe_action == int(FailsafeAction.LAND)
    assert decoded.armed and not decoded.landed


def test_fast_lorenz_generate_is_bit_identical_to_step():
    fast, slow = ChaoticLayer(), ChaoticLayer()
    out = fast.generate(5000)
    ref = np.array([slow.step()[0] for _ in range(5000)])
    assert np.array_equal(out, ref)
    assert np.array_equal(fast.state, slow.state)


def test_loopback_loss_grows_with_noise_and_dsss_helps():
    assert chip_error_probability(0.005) < 1e-12
    p_edge = packet_success_probability(11, 0.15)
    p_mid = packet_success_probability(11, 0.45)
    p_centre = packet_success_probability(11, 0.71)
    assert p_edge > 0.99 > p_mid > p_centre
    assert packet_success_probability(11, 0.45, dssc_n=8) > p_mid


def test_loopback_channel_delivers_and_drops():
    clean = LoopbackChannel(noise_voltage=0.0, latency=0.0, seed=3)
    for i in range(20):
        clean._transmit(bytes([i]))
    received = []
    while (p := clean._receive()) is not None:
        received.append(p)
    assert received == [bytes([i]) for i in range(20)]

    jammed = LoopbackChannel(noise_voltage=0.71, latency=0.0, seed=3)
    for i in range(200):
        jammed._transmit(b"x" * 11)
    count = 0
    while jammed._receive() is not None:
        count += 1
    assert count < 20
