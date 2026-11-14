"""Headless simulation harness: operator + UAV + radio + RF scene, no window.

Steps everything in simulation time at the same 250 Hz as `main.py`, so
experiments and tests exercise exactly the code the interactive simulator
runs -- just faster than real time and without pygame/OpenGL.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.loopback_link import LoopbackChannel
from uavsim.control.rc import FlightMode, Sticks
from uavsim.entities.jammer import Jammer
from uavsim.entities.operator import PilotInput, UAVOperator
from uavsim.entities.uav import UAV, UAVConfig
from uavsim.world.environment import FlatGroundPlane, World
from uavsim.world.rf import RFEnvironment, apply_to_links

DT = 1.0 / 250.0


class Sim:
    """Operator + UAV over the numpy radio, optionally inside an RF scene.

    With ``jammers`` given, link noise (and tone) follows the physical link
    budget every step, like `main.py` does every frame. ``profile`` selects
    the radio's mitigation profile on both links.
    """

    def __init__(self, mode: FlightMode = FlightMode.ALT_HOLD, seed: int = 1,
                 jammers: Optional[Sequence[Jammer]] = None, profile: str = "none") -> None:
        self.jammers: List[Jammer] = list(jammers) if jammers is not None else []
        self.rf = RFEnvironment() if jammers is not None else None
        self.ground_station = World().ground_station
        self.budget = None
        self.uplink = LoopbackChannel(noise_voltage=0.0, latency=0.0, seed=seed, profile=profile)
        self.downlink = LoopbackChannel(noise_voltage=0.0, latency=0.0, seed=seed + 1,
                                        profile=profile)
        self.uav = UAV(
            UAVConfig(),
            FlatGroundPlane(),
            CommGatewayInput(self.uplink),
            CommGatewayOutput(self.downlink),
        )
        self.op = UAVOperator(CommGatewayOutput(self.uplink), CommGatewayInput(self.downlink))
        self.op.mode = mode
        self.op.sticks = Sticks(throttle=0.5 if mode >= FlightMode.ALT_HOLD else 0.0)
        self.t = 0.0

    def run(self, seconds: float, keys=(), sticks: Sticks = None, record=None) -> None:
        """Advance the simulation. Either hold keyboard `keys` or force `sticks`."""
        for _ in range(int(round(seconds / DT))):
            if sticks is not None:
                self.op.sticks = sticks
            else:
                self.op.handle_input(PilotInput(keys=set(keys)), DT)
            if self.rf is not None:
                self.budget = self.rf.evaluate(self.uav.body.position, self.ground_station,
                                               self.jammers, self.t)
                apply_to_links(self.budget, self.uplink, self.downlink)
            self.op.tick(self.t)
            self.uav.step(DT, self.t)
            self.op.poll_telemetry(self.t)
            if record is not None:
                record.append((self.t, self.uav.body.position.copy(), self.uav.failsafe.stage))
            self.t += DT

    def arm(self) -> None:
        self.op.arm_switch = False
        self.run(0.1, sticks=self.op.sticks)
        self.op.arm_switch = True
        self.run(0.1, sticks=self.op.sticks)
        if not self.uav.armed:
            raise RuntimeError("UAV refused to arm")

    def take_off(self, altitude: float = 5.0) -> None:
        self.arm()
        while self.uav.body.position[2] < altitude:
            self.run(0.1, keys={"w"})
            if self.t > 30.0:
                raise RuntimeError("never reached take-off altitude")
        self.run(3.0)  # sticks centred: brake and hold
