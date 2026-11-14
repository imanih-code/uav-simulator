"""Packet delivery ratio (PDR) curves for every mitigation profile.

Two sweeps, for an 11-byte RC frame and the 56-byte telemetry packet:

* ``noise``: wideband interference (barrage/pulsed/swept jammers, thermal
  noise) -- PDR versus SINR.
* ``tone``:  a narrowband CW jammer on top of a clean link (SNR 30 dB) --
  PDR versus SIR (signal power / tone power).

For each curve it reports the SINR/SIR needed for 50% delivery, the gain
over no mitigation, the airtime cost, and the gain left after paying for
that airtime (10*log10(airtime)). That last number separates real
processing/coding gain from "just transmitting longer".

    python -m uavsim.experiments.link_curves --out results --trials 300
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import List, Tuple

import numpy as np

from uavsim.comms.phy.modem import PROFILE_ORDER, PROFILES, NumpyPhy
from uavsim.comms.telemetry import PACKET_SIZE as TELEMETRY_SIZE
from uavsim.control.rc import RC_PACKET_SIZE
from uavsim.world.rf import sinr_to_noise_voltage

PACKETS = {"rc": RC_PACKET_SIZE, "telemetry": TELEMETRY_SIZE}
NOISE_SWEEP = np.arange(-8.0, 14.5, 1.0)
TONE_SWEEP = np.arange(-14.0, 10.5, 1.0)
TONE_SNR_DB = 30.0


def pdr(profile: str, size: int, noise_voltage: float, tone_amplitude: float,
        trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    phy = NumpyPhy(PROFILES[profile], rng)
    ok = 0
    for i in range(trials):
        payload = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        ok += phy.burst(payload, i, noise_voltage, tone_amplitude).payload == payload
    return ok / trials


def threshold_50(xs: np.ndarray, ys: List[float]) -> float:
    """Lowest x where the curve crosses 50% (linear interpolation)."""
    ys = np.asarray(ys)
    above = np.flatnonzero(ys >= 0.5)
    if len(above) == 0:
        return math.nan
    i = above[0]
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def run(trials: int, seed: int) -> Tuple[List[dict], List[dict]]:
    rows, summary = [], []
    for kind, sweep in (("noise", NOISE_SWEEP), ("tone", TONE_SWEEP)):
        for packet, size in PACKETS.items():
            base = None
            for profile in PROFILE_ORDER:
                curve = []
                for x in sweep:
                    if kind == "noise":
                        nv, amp = sinr_to_noise_voltage(x), 0.0
                    else:
                        nv = sinr_to_noise_voltage(TONE_SNR_DB)
                        amp = math.sqrt(2.0 * 10.0 ** (-x / 10.0))
                    value = pdr(profile, size, nv, amp, trials, seed + int(x * 10) + 1000)
                    curve.append(value)
                    rows.append(dict(sweep=kind, packet=packet, profile=profile,
                                     x_db=float(x), pdr=value))
                th = threshold_50(sweep, curve)
                airtime = PROFILES[profile].airtime_factor(size)
                if profile == "none":
                    base = th
                gain = base - th if not (math.isnan(base) or math.isnan(th)) else math.nan
                summary.append(dict(
                    sweep=kind, packet=packet, profile=profile, threshold_db=th,
                    gain_db=gain, airtime=airtime,
                    gain_per_airtime_db=gain - 10.0 * math.log10(airtime),
                ))
    return rows, summary


def write_csv(path: str, rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_markdown(summary: List[dict]) -> str:
    lines = []
    for kind, title, unit in (("noise", "Wideband noise (PDR 50% vs SINR)", "SINR"),
                              ("tone", f"CW tone jammer at SNR {TONE_SNR_DB:.0f} dB "
                                       "(PDR 50% vs SIR)", "SIR")):
        lines.append(f"### {title}\n")
        lines.append(f"| Packet | Profile | {unit} for 50% | Gain | Airtime | Gain − airtime cost |")
        lines.append("|---|---|---|---|---|---|")
        for s in (s for s in summary if s["sweep"] == kind):
            lines.append(
                f"| {s['packet']} | {PROFILES[s['profile']].label} | {_fmt(s['threshold_db'])} dB "
                f"| {_fmt(s['gain_db'], sign=True)} dB | ×{s['airtime']:.1f} "
                f"| {_fmt(s['gain_per_airtime_db'], sign=True)} dB |")
        lines.append("")
    return "\n".join(lines)


def _fmt(value: float, sign: bool = False) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:+.1f}" if sign else f"{value:.1f}"


def plot(rows: List[dict], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    for col, kind in enumerate(("noise", "tone")):
        for row, packet in enumerate(PACKETS):
            ax = axes[row][col]
            for profile in PROFILE_ORDER:
                pts = [r for r in rows if r["sweep"] == kind and r["packet"] == packet
                       and r["profile"] == profile]
                ax.plot([r["x_db"] for r in pts], [r["pdr"] for r in pts], marker="o",
                        markersize=3, label=PROFILES[profile].label)
            ax.set_title(f"{packet} ({PACKETS[packet]} B) — "
                         f"{'wideband noise' if kind == 'noise' else 'CW tone, SNR 30 dB'}")
            ax.set_xlabel("SINR (dB)" if kind == "noise" else "SIR (dB)")
            ax.set_ylabel("packet delivery ratio")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    start = time.perf_counter()
    rows, summary = run(args.trials, args.seed)
    write_csv(os.path.join(args.out, "link_curves.csv"), rows)
    write_csv(os.path.join(args.out, "link_summary.csv"), summary)
    md = summary_markdown(summary)
    with open(os.path.join(args.out, "link_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    plotted = plot(rows, os.path.join(args.out, "link_curves.png"))
    print(md)
    print(f"done in {time.perf_counter() - start:.0f} s -> {args.out}/"
          + ("" if plotted else " (install matplotlib for link_curves.png)"))


if __name__ == "__main__":
    main()
