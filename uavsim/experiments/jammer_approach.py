"""Closed-loop scenario: fly straight at a jammer and see where control is lost.

For every mitigation profile and jammer type, the UAV takes off at the
ground station, climbs to 10 m and flies in POS_HOLD at ~3 m/s toward a
jammer 150 m away. Recorded, as distance to the jammer:

* ``lq90``  -- uplink link quality first drops below 90%
* ``rxloss`` -- first RX LOSS (no valid RC frame for 0.1 s)
* ``failsafe`` -- failsafe STAGE2 (link gone for 1 s; the UAV then holds)

Smaller distance = the drone stays under pilot control closer to the
jammer. This is the end-to-end number the link curves translate into.

    python -m uavsim.experiments.jammer_approach --out results --seeds 3
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from typing import Dict, List

import numpy as np

from uavsim.comms.phy.modem import PROFILE_ORDER, PROFILES
from uavsim.control.failsafe import FailsafeStage
from uavsim.control.rc import FailsafeAction, FlightMode, Sticks
from uavsim.entities.jammer import Jammer, JammerKind
from uavsim.experiments.harness import Sim

JAMMER_X = 150.0
JAMMER_POWER_DBM = 10.0
CRUISE = Sticks(throttle=0.5, roll=0.0, pitch=0.4, yaw=0.0)   # ~3.2 m/s forward
KINDS = (JammerKind.BARRAGE, JammerKind.TONE)


def approach(profile: str, kind: JammerKind, seed: int) -> Dict[str, float]:
    jammer = Jammer.make(np.array([JAMMER_X, 0.0, 0.0]), kind, JAMMER_POWER_DBM, phase=0.0)
    sim = Sim(mode=FlightMode.POS_HOLD, seed=seed, jammers=[jammer], profile=profile)
    sim.op.failsafe_action = FailsafeAction.HOLD
    sim.take_off(10.0)
    marks: Dict[str, float] = {"lq90": math.nan, "rxloss": math.nan, "failsafe": math.nan}
    while sim.t < 120.0:
        sim.run(0.1, sticks=CRUISE)
        distance = JAMMER_X - float(sim.uav.body.position[0])
        stage = sim.uav.failsafe.stage
        if math.isnan(marks["lq90"]) and sim.uav.link_quality() < 90.0:
            marks["lq90"] = distance
        if math.isnan(marks["rxloss"]) and stage >= FailsafeStage.HOLD_LAST:
            marks["rxloss"] = distance
        if stage is FailsafeStage.STAGE2:
            marks["failsafe"] = distance
            break
        if distance < 5.0:
            break
    return marks


def run(seeds: int) -> List[dict]:
    rows = []
    for kind in KINDS:
        for profile in PROFILE_ORDER:
            for seed in range(seeds):
                marks = approach(profile, kind, 100 + seed)
                rows.append(dict(jammer=kind.value, profile=profile, seed=seed, **marks))
                print(f"{kind.value:8} {profile:12} seed {seed}: " +
                      ", ".join(f"{k}={v:.1f} m" for k, v in marks.items()), flush=True)
    return rows


def summary_markdown(rows: List[dict]) -> str:
    lines = [
        f"Jammer {JAMMER_POWER_DBM:.0f} dBm at {JAMMER_X:.0f} m from the ground station; "
        "UAV at 10 m altitude flying straight at it. Distances to the jammer (median over seeds); "
        "smaller is better.\n",
        "| Jammer | Profile | LQ < 90% at | First RX loss at | Failsafe at |",
        "|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for profile in PROFILE_ORDER:
            sel = [r for r in rows if r["jammer"] == kind.value and r["profile"] == profile]
            cells = []
            for key in ("lq90", "rxloss", "failsafe"):
                values = [r[key] for r in sel if not math.isnan(r[key])]
                cells.append(f"{statistics.median(values):.0f} m" if values else "never")
            lines.append(f"| {kind.value} | {PROFILES[profile].label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    start = time.perf_counter()
    rows = run(args.seeds)
    with open(os.path.join(args.out, "jammer_approach.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = summary_markdown(rows)
    with open(os.path.join(args.out, "jammer_approach.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"done in {time.perf_counter() - start:.0f} s -> {args.out}/")


if __name__ == "__main__":
    main()
