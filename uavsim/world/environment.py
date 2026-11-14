"""Simulated world: terrain, ground station, bounds and ground contact.

`apply_ground_contact` is a small, generic function (not a UAV concept)
that keeps any `RigidBody` from sinking into the terrain: it pushes the
body back onto the surface, removes the velocity component going into the
surface (reporting how fast that was, so crashes into hillsides count, not
just hard vertical landings), and applies friction while resting.
`RigidBody` itself stays completely unaware that terrain exists -- this
function is the only bridge between "world" and "physics".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from uavsim.physics.rigid_body import RigidBody
from uavsim.world.terrain import FlatGroundPlane, Terrain

__all__ = [
    "WORLD_EXTENT_HALF", "FlatGroundPlane", "Terrain", "World", "ContactResult",
    "apply_ground_contact", "apply_world_bounds",
]

# Half the side length of the square world. The UAV cannot fly beyond
# [-WORLD_EXTENT_HALF, WORLD_EXTENT_HALF] in X or Y.
WORLD_EXTENT_HALF = 200

# Friction applied to sliding while resting on the ground, per second.
_GROUND_FRICTION_PER_SECOND = 4.0

Ground = Union[FlatGroundPlane, Terrain]


@dataclass
class World:
    ground: Ground = field(default_factory=Terrain)
    extent_half: float = WORLD_EXTENT_HALF
    # The pilot's ground station (RC transmitter + telemetry receiver):
    # a few metres behind the take-off point, antenna at head height.
    ground_station: np.ndarray = None

    def __post_init__(self) -> None:
        if self.ground_station is None:
            x, y = -4.0, 0.0
            self.ground_station = np.array([x, y, float(self.ground.height_at(x, y)) + 1.5])


@dataclass(frozen=True)
class ContactResult:
    grounded: bool
    impact_speed: float = 0.0   # m/s into the surface just before contact


def apply_world_bounds(body: RigidBody, extent_half: float) -> None:
    """Clamp `body` inside the square world and kill velocity pushing outward."""
    for axis in (0, 1):
        if body.position[axis] < -extent_half:
            body.position[axis] = -extent_half
            if body.velocity[axis] < 0.0:
                body.velocity[axis] = 0.0
        elif body.position[axis] > extent_half:
            body.position[axis] = extent_half
            if body.velocity[axis] > 0.0:
                body.velocity[axis] = 0.0


def apply_ground_contact(body: RigidBody, ground: Ground, dt: float) -> ContactResult:
    """Keep `body` on or above the terrain surface.

    Passive contact model: position is projected back onto the surface and
    the velocity component into it is removed (no bounce). Enough for a
    UAV that rests on its gear and takes off under its own thrust.
    """
    x, y = float(body.position[0]), float(body.position[1])
    surface = float(ground.height_at(x, y))
    if body.position[2] > surface:
        return ContactResult(False)

    normal = ground.normal_at(x, y)
    body.position[2] = surface
    friction = max(0.0, 1.0 - _GROUND_FRICTION_PER_SECOND * dt)
    v_normal = float(np.dot(body.velocity, normal))
    v_tangent = body.velocity - v_normal * normal
    impact = max(0.0, -v_normal)
    # Drop the part going into the ground; slide along it with friction.
    body.velocity = max(v_normal, 0.0) * normal + friction * v_tangent

    # Landing gear sits on the ground: it cannot roll or pitch while
    # resting, only keep its heading (and slowly stop spinning on it).
    yaw = body.orientation.as_euler("xyz")[2]
    body.orientation = Rotation.from_euler("z", yaw)
    body.angular_velocity_body = np.array([0.0, 0.0, body.angular_velocity_body[2] * friction])
    return ContactResult(True, impact)
