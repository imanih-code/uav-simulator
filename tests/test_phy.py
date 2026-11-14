import math

import numpy as np
import pytest

from uavsim.comms.loopback_link import LoopbackChannel, packet_success_probability
from uavsim.comms.phy import fec
from uavsim.comms.phy.modem import PROFILES, NumpyPhy
from uavsim.comms.phy.sequences import KeyedChips, LorenzChips, keyed_chips, mean_run_length
from uavsim.entities.jammer import Jammer, JammerKind
from uavsim.world.rf import RFEnvironment, apply_to_links, sinr_to_noise_voltage

PAYLOAD = bytes(range(11))


def delivery(profile: str, sinr_db: float, trials: int = 300, tone_sir_db: float = None,
             seed: int = 5) -> float:
    phy = NumpyPhy(PROFILES[profile], np.random.default_rng(seed))
    nv = sinr_to_noise_voltage(sinr_db)
    amp = 0.0 if tone_sir_db is None else math.sqrt(2.0 * 10.0 ** (-tone_sir_db / 10.0))
    return sum(phy.burst(PAYLOAD, i, nv, amp).payload == PAYLOAD for i in range(trials)) / trials


def test_viterbi_roundtrip_and_corrects_errors():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 300).astype(np.uint8)
    coded = fec.encode(bits)
    assert len(coded) == 2 * (300 + fec.TAIL_BITS)
    soft = coded.astype(float) * 2.0 - 1.0
    assert np.array_equal(fec.decode(soft), bits)
    flipped = soft.copy()
    flipped[[10, 60, 200, 333, 500]] *= -1  # isolated hard errors
    assert np.array_equal(fec.decode(flipped), bits)


def test_viterbi_beats_uncoded_at_equal_energy_per_bit():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    sigma = 0.6  # per coded symbol; uncoded gets 2x the energy per bit (sigma/sqrt2)
    uncoded = (bits * 2.0 - 1.0) + sigma / math.sqrt(2) * rng.standard_normal(bits.size)
    coded = fec.encode(bits) * 2.0 - 1.0
    decoded = fec.decode(coded + sigma * rng.standard_normal(coded.size))
    assert np.mean(decoded != bits) < 0.2 * np.mean((uncoded > 0) != bits)


def test_lorenz_chips_are_long_runs_keyed_chips_are_white():
    lorenz = LorenzChips().tx(500, 8, 0)
    keyed = KeyedChips().tx(500, 8, 0)
    assert mean_run_length(lorenz) > 50
    assert 1.7 < mean_run_length(keyed) < 2.3
    assert abs(keyed.mean()) < 0.05


def test_keyed_chips_are_stateless_per_packet():
    assert np.array_equal(keyed_chips(1, 42, 64), keyed_chips(1, 42, 64))
    assert not np.array_equal(keyed_chips(1, 42, 64), keyed_chips(1, 43, 64))
    assert not np.array_equal(keyed_chips(1, 42, 64), keyed_chips(2, 42, 64))


@pytest.mark.parametrize("sinr", [8.0, 9.0, 10.0])
def test_bit_true_baseline_matches_calibrated_model(sinr):
    expected = packet_success_probability(len(PAYLOAD), sinr_to_noise_voltage(sinr))
    assert delivery("none", sinr, trials=600) == pytest.approx(expected, abs=0.07)


def test_mitigation_ordering_against_wideband_noise():
    at_3db = {name: delivery(name, 3.0) for name in PROFILES}
    assert at_3db["none"] < 0.05
    assert at_3db["dsss-keyed"] > at_3db["dsss-lorenz"] + 0.2   # soft + spread header
    assert at_3db["dsss+fec"] > 0.95
    assert delivery("dsss+fec", -2.0) > 0.5                      # >10 dB better than none


def test_lorenz_spreading_does_not_reject_a_tone_but_keyed_does():
    clean = 30.0
    lorenz = delivery("dsss-lorenz", clean, tone_sir_db=3.0)
    none = delivery("none", clean, tone_sir_db=3.0)
    keyed = delivery("dsss-keyed", clean, tone_sir_db=3.0)
    assert lorenz <= none + 0.05, "constant chips leave the tone untouched"
    assert keyed > 0.9
    assert delivery("dsss+fec", clean, tone_sir_db=-3.0) > 0.7


@pytest.mark.parametrize("name", list(PROFILES))
def test_recorded_gains_match_the_modem(name):
    # measured_gain_db drives the HUD colours; it must stay true to the modem.
    threshold_none = 9.0  # 50% point of an 11-byte frame without mitigation
    at_threshold = delivery(name, threshold_none - PROFILES[name].measured_gain_db, trials=300)
    assert 0.3 < at_threshold < 0.7


def test_airtime_factors():
    size = len(PAYLOAD)
    assert PROFILES["none"].airtime_factor(size) == 1.0
    assert PROFILES["dsss-keyed"].airtime_factor(size) == pytest.approx(8.0)
    assert 1.8 < PROFILES["fec"].airtime_factor(size) < 2.2


def test_tone_jammer_is_kept_apart_in_the_link_budget():
    rf = RFEnvironment()
    gs = np.array([-4.0, 0.0, 1.5])
    uav = np.array([60.0, 0.0, 10.0])
    clean = rf.evaluate(uav, gs, [], 0.0)
    tone = Jammer.make(np.array([80.0, 0.0, 0.0]), JammerKind.TONE, 10.0, phase=0.0)
    jammed = rf.evaluate(uav, gs, [tone], 0.0)
    assert jammed.uplink_sinr_db < clean.uplink_sinr_db - 20.0
    assert jammed.wideband_sinr_db(True) == pytest.approx(clean.uplink_sinr_db, abs=1e-6)
    sir = jammed.uplink_signal_dbm - jammed.uplink_tone_dbm
    assert jammed.tone_amplitude(True) == pytest.approx(math.sqrt(2 * 10 ** (-sir / 10)))

    up, down = LoopbackChannel(), LoopbackChannel()
    apply_to_links(jammed, up, down)
    assert up.tone_amplitude == pytest.approx(jammed.tone_amplitude(True))
    assert up.noise_voltage == pytest.approx(sinr_to_noise_voltage(clean.uplink_sinr_db))


def test_loopback_profile_switching():
    link = LoopbackChannel(profile="dsss+fec")
    assert link.profile_label == "DSSS4+FEC" and link.dssc_N == 4
    link.dssc_N = 8
    assert link.profile == "dsss-lorenz"
    link.dssc_N = 0
    assert link.profile == "none"
