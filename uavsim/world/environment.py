"""Minimal simulated world.

For now this is a single flat, infinite ground plane at z = 0 and nothing
else. Jammers and other environmental effects will attach to `World` later
without requiring any change to UAV or UAVOperator.

`apply_ground_contact` is a small, generic function (not a UAV concept) that
keeps any `RigidBody` from sinking through the ground plane: it clamps the
body to ground level, cancels downward velocity, and applies simple
friction to horizontal sliding while resting. `RigidBody` itself stays
completely unaware that terrain exists -- this function is the only bridge
between "world" and "physics".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from uavsim.physics.rigid_body import RigidBody

# Simple friction coefficient applied to horizontal velocity while resting
# on the ground (landing gear grip), per second of contact.
_GROUND_FRICTION_PER_SECOND = 4.0


@dataclass(frozen=True)
class FlatGroundPlane:
    ground_z: float = 0.0

    def height_at(self, x: float, y: float) -> float:
        del x, y  # flat plane: height is the same everywhere, for now
        return self.ground_z


@dataclass
class World:
    ground: FlatGroundPlane = field(default_factory=FlatGroundPlane)


def apply_ground_contact(body: RigidBody, ground: FlatGroundPlane, dt: float) -> bool:
    """Keep `body` from falling through `ground`.

    Returns True if the body is currently resting on the ground. This is a
    passive contact model: it does not simulate a real normal-force impulse,
    it simply clamps position/velocity every tick, which is enough for a
    UAV that starts parked on the ground and only leaves it once its own
    thrust exceeds gravity.
    """
    grounded = body.position[2] <= ground.ground_z
    if not grounded:
        return False

    body.position[2] = ground.ground_z
    if body.velocity[2] < 0.0:
        body.velocity[2] = 0.0

    friction = max(0.0, 1.0 - _GROUND_FRICTION_PER_SECOND * dt)
    body.velocity[0] *= friction
    body.velocity[1] *= friction
    return True
