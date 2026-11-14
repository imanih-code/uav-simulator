import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput  # noqa: E402
from uavsim.comms.loopback_link import LoopbackChannel  # noqa: E402
from uavsim.control.rc import FlightMode, Sticks  # noqa: E402
from uavsim.entities.operator import PilotInput, UAVOperator  # noqa: E402
from uavsim.entities.uav import UAV, UAVConfig  # noqa: E402
from uavsim.world.environment import FlatGroundPlane  # noqa: E402

DT = 1.0 / 250.0


class Sim:
    """Headless operator + UAV over a clean loopback radio, in sim time."""

    def __init__(self, mode: FlightMode = FlightMode.ALT_HOLD, seed: int = 1) -> None:
        self.uplink = LoopbackChannel(noise_voltage=0.0, latency=0.0, seed=seed)
        self.downlink = LoopbackChannel(noise_voltage=0.0, latency=0.0, seed=seed + 1)
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
        assert self.uav.armed

    def take_off(self, altitude: float = 5.0) -> None:
        self.arm()
        while self.uav.body.position[2] < altitude:
            self.run(0.1, keys={"w"})
            assert self.t < 30.0, "never reached take-off altitude"
        self.run(3.0)  # sticks centred: brake and hold


@pytest.fixture
def sim() -> Sim:
    return Sim()
