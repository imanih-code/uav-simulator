import numpy as np
import pytest
from conftest import Sim
from scipy.spatial.transform import Rotation

from uavsim.control.rc import FlightMode, Sticks
from uavsim.entities.jammer import Jammer, JammerKind
from uavsim.physics.rigid_body import RigidBody
from uavsim.world.environment import World, apply_ground_contact
from uavsim.world.rf import RFEnvironment
from uavsim.world.terrain import FlatGroundPlane, Terrain


def test_terrain_is_flat_at_home_and_bounded():
    t = Terrain()
    xs = np.linspace(-200, 200, 161)
    gx, gy = np.meshgrid(xs, xs)
    h = t.height_at(gx, gy)
    assert h.max() <= t.max_height + 1e-9 and h.min() >= 0.0
    assert h.max() > 5.0, "there should be real hills"
    near = np.hypot(gx, gy) <= t.flat_radius
    assert np.allclose(h[near], 0.0)
    np.testing.assert_allclose(t.normal_at(0.0, 0.0), [0.0, 0.0, 1.0], atol=1e-9)
    assert float(t.height_at(12.3, -4.5)) == pytest.approx(float(t.height_at(np.array([12.3]),
                                                                               np.array([-4.5]))[0]))


def test_world_puts_ground_station_on_the_terrain():
    world = World()
    gs = world.ground_station
    assert gs[2] == pytest.approx(float(world.ground.height_at(gs[0], gs[1])) + 1.5)


def _body_at(position, velocity):
    body = RigidBody(mass=1.0, inertia_diag=np.array([0.01, 0.01, 0.02]))
    body.position = np.array(position, dtype=float)
    body.velocity = np.array(velocity, dtype=float)
    body.orientation = Rotation.from_euler("xyz", [0.2, -0.1, 0.5])
    return body


def test_contact_on_flat_ground_matches_old_behaviour():
    body = _body_at([0.0, 0.0, -0.01], [2.0, 0.0, -4.0])
    result = apply_ground_contact(body, FlatGroundPlane(), 0.004)
    assert result.grounded and result.impact_speed == pytest.approx(4.0)
    assert body.position[2] == 0.0 and body.velocity[2] == 0.0
    assert body.velocity[0] == pytest.approx(2.0 * (1.0 - 4.0 * 0.004))
    assert body.attitude_rpy()[:2] == pytest.approx([0.0, 0.0])


def test_horizontal_flight_into_a_slope_counts_as_impact():
    t = Terrain()
    # Find a steep spot along +x.
    xs = np.arange(30.0, 120.0, 0.5)
    slopes = np.gradient(t.height_at(xs, 0.0), xs)
    x = float(xs[np.argmax(slopes)])
    surface = float(t.height_at(x, 0.0))
    body = _body_at([x, 0.0, surface - 0.05], [12.0, 0.0, 0.0])
    result = apply_ground_contact(body, t, 0.004)
    normal = t.normal_at(x, 0.0)
    assert result.grounded
    assert result.impact_speed == pytest.approx(-12.0 * normal[0], rel=1e-6)
    assert np.dot(body.velocity, normal) >= -1e-9, "no velocity left going into the hill"


def test_flying_low_into_a_hill_damages_the_uav():
    sim = Sim(mode=FlightMode.ALT_HOLD)
    sim.uav.ground = Terrain()
    sim.take_off(5.0)
    sim.run(6.0, sticks=Sticks(throttle=0.5, pitch=1.0))   # full forward, ~13 m/s at 5 m
    assert sim.uav.health < 100.0


def test_rf_with_flat_terrain_equals_no_terrain():
    gs = np.array([-4.0, 0.0, 1.5])
    jam = [Jammer.make(np.array([70.0, 20.0, 0.0]), JammerKind.BARRAGE, 12.0, phase=0.0)]
    a = RFEnvironment().evaluate(np.array([40.0, 5.0, 8.0]), gs, jam, 0.0)
    b = RFEnvironment(terrain=FlatGroundPlane()).evaluate(np.array([40.0, 5.0, 8.0]), gs, jam, 0.0)
    assert a == b


def test_rf_uses_height_above_the_terrain():
    t = Terrain()
    xs = np.arange(40.0, 190.0, 1.0)
    x = float(xs[np.argmax(t.height_at(xs, 0.0))])
    hill = float(t.height_at(x, 0.0))
    gs = np.array([-4.0, 0.0, 1.5])
    over_hill = RFEnvironment(terrain=t).evaluate(np.array([x, 0.0, hill + 2.0]), gs, [], 0.0)
    flat_low = RFEnvironment().evaluate(np.array([x, 0.0, 2.0]), gs, [], 0.0)
    # 2 m above the ground in both cases -> same ground-hugging exponent;
    # the hill version is only slightly closer to line of sight via 3D distance.
    assert abs(over_hill.uplink_signal_dbm - flat_low.uplink_signal_dbm) < 1.0
    elevated = RFEnvironment().evaluate(np.array([x, 0.0, hill + 2.0]), gs, [], 0.0)
    assert elevated.uplink_signal_dbm > over_hill.uplink_signal_dbm + 2.0
