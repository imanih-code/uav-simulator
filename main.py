"""Entry point: wires comms, physics, flight control, camera, HUD and
rendering together, then runs the real-time simulation loop.

Flight (Mode 2 transmitter layout, see README for the full table)
------
W / S          throttle (ALT/POS hold: climb / descend; ANGLE/ACRO: raise / lower)
A / D          yaw left / right
I / K          pitch forward / back
J / L          roll left / right
SPACE          arm switch (arming needs the UAV on the ground and throttle low)
M              cycle flight mode: POS_HOLD -> ALT_HOLD -> ANGLE -> ACRO
G              cycle failsafe action: RTH -> HOLD -> LAND -> DROP
X              transmitter on/off (lose the link on purpose to test failsafe)
A gamepad, if connected, drives the sticks (A = arm, B = mode, Y = failsafe).

Simulation
----------
Arrow keys / mouse  camera;  V  FOLLOW / FREE camera
F              cycle radio mitigation: NONE -> DSSS8 LORENZ -> DSSS8 KEYED -> FEC
               -> DSSS4+FEC (GNU Radio backend: Lorenz DSSS N=8 on/off)
H              uplink interference map on the ground on/off
P              pause;  BACKSPACE  reset UAV + reposition jammers
F12            screenshot;  ESC  quit

Physics and the flight controller run at a fixed rate (PHYSICS_HZ) in
simulation time, independent of the render frame rate.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

import numpy as np
import pygame
from OpenGL.GL import GL_BACK, GL_RGB, GL_UNSIGNED_BYTE, glReadPixels, glReadBuffer

from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.phy.modem import PROFILE_ORDER, PROFILES
from uavsim.entities.jammer import Jammer, JammerKind
from uavsim.entities.operator import PilotInput, UAVOperator
from uavsim.entities.uav import UAV, UAVConfig
from uavsim.hud.hud import HUD
from uavsim.rendering.camera import Camera
from uavsim.rendering.renderer import Renderer
from uavsim.rendering.window import Window
from uavsim.world.environment import World
from uavsim.world.rf import LinkBudget, RFEnvironment, apply_to_links

TARGET_FPS = 60
PHYSICS_HZ = 250
PHYSICS_DT = 1.0 / PHYSICS_HZ
MAX_STEPS_PER_FRAME = 25  # never try to catch up more than 0.1 s of sim time
_BASE_NOISE = 0.005
_JAMMER_COUNT = 6
_JAMMER_MIN_DISTANCE_FROM_HOME = 90.0   # m
_JAMMER_MIN_SEPARATION = 60.0           # m
# Radiated power drawn per jammer (4-25 mW). With the 500 mW ground station
# this leaves danger zones around each jammer instead of jamming everything.
_JAMMER_POWER_RANGE_DBM = (6.0, 14.0)
_JAMMER_KIND_WEIGHTS = {
    JammerKind.BARRAGE: 3, JammerKind.PULSED: 2, JammerKind.SWEPT: 2, JammerKind.TONE: 2,
}


def make_link(backend: str, noise: float):
    if backend == "gnuradio":
        from uavsim.comms.gnuradio_link import GnuRadioChannel
        return GnuRadioChannel(noise_voltage=noise)
    from uavsim.comms.loopback_link import LoopbackChannel
    return LoopbackChannel(noise_voltage=noise)


def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import gnuradio  # noqa: F401
        return "gnuradio"
    except ImportError:
        print("GNU Radio not found: using the approximate numpy link (--link loopback).")
        return "loopback"


def build_simulation(world: World, backend: str):
    """Create the UAV and Operator, connected by two independent radio
    links: commands (Operator -> UAV) and telemetry (UAV -> Operator)."""
    command_link = make_link(backend, _BASE_NOISE)
    telemetry_link = make_link(backend, _BASE_NOISE)

    config = UAVConfig()
    uav = UAV(
        config=config,
        ground=world.ground,
        command_input=CommGatewayInput(command_link),
        telemetry_output=CommGatewayOutput(telemetry_link),
    )
    operator = UAVOperator(
        command_output=CommGatewayOutput(command_link),
        telemetry_input=CommGatewayInput(telemetry_link),
        rc_rate_hz=config.rc_rate_hz,
        telemetry_rate_hz=1.0 / config.telemetry_interval,
    )
    hud = HUD(operator, uav.command_input)
    return command_link, telemetry_link, uav, operator, hud


def update_radio(rf: RFEnvironment, world: World, jammers: List[Jammer], uav: UAV,
                 command_link, telemetry_link, sim_time: float) -> LinkBudget:
    """Recompute both link budgets for the current geometry and push them
    into the radio channels as noise levels."""
    budget = rf.evaluate(uav.body.position, world.ground_station, jammers, sim_time)
    apply_to_links(budget, command_link, telemetry_link)
    return budget


def cycle_mitigation(command_link, telemetry_link) -> None:
    """F key. Numpy radio: step through every mitigation profile. GNU Radio:
    toggle its (Lorenz) DSSS N=8, the only scheme implemented there."""
    for link in (command_link, telemetry_link):
        link.clear(link.noise_voltage)
        if hasattr(link, "profile"):
            order = list(PROFILE_ORDER)
            link.profile = order[(order.index(link.profile) + 1) % len(order)]
        else:
            link.dssc_N = 0 if link.dssc_N > 0 else 8


def mitigation_label(link) -> str:
    if hasattr(link, "profile_label"):
        return link.profile_label
    return f"DSSS{link.dssc_N} LORENZ" if link.dssc_N > 0 else "NONE"


def mitigation_gain_db(link) -> float:
    if hasattr(link, "profile"):
        return PROFILES[link.profile].measured_gain_db
    return PROFILES["dsss-lorenz"].measured_gain_db if link.dssc_N > 0 else 0.0


def place_random_jammers(jammers: List[Jammer], count: int, extent_half: float,
                         home: np.ndarray, ground) -> None:
    kinds = list(_JAMMER_KIND_WEIGHTS)
    weights = list(_JAMMER_KIND_WEIGHTS.values())
    margin = 5.0
    for _ in range(count * 50):
        if len(jammers) >= count:
            break
        pos = np.array([
            random.uniform(-extent_half + margin, extent_half - margin),
            random.uniform(-extent_half + margin, extent_half - margin),
            0.0,
        ])
        pos[2] = float(ground.height_at(pos[0], pos[1]))  # stand on the terrain
        if np.linalg.norm(pos[:2] - home[:2]) < _JAMMER_MIN_DISTANCE_FROM_HOME:
            continue
        if any(np.linalg.norm(pos[:2] - j.position[:2]) < _JAMMER_MIN_SEPARATION for j in jammers):
            continue
        jammers.append(Jammer.make(pos, random.choices(kinds, weights)[0],
                                   random.uniform(*_JAMMER_POWER_RANGE_DBM)))


def reset_jammers(jammers: List[Jammer], extent_half: float, home: np.ndarray,
                  ground) -> None:
    jammers.clear()
    place_random_jammers(jammers, _JAMMER_COUNT, extent_half, home, ground)


def keep_above_ground(eye: np.ndarray, ground, clearance: float = 0.6) -> np.ndarray:
    """Stop the camera from sinking into a hill."""
    floor = float(ground.height_at(eye[0], eye[1])) + clearance
    return eye if eye[2] >= floor else np.array([eye[0], eye[1], floor])


def save_screenshot(window: Window, index: int) -> None:
    glReadBuffer(GL_BACK)
    w, h = window.width, window.height
    buf = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    surf = pygame.transform.flip(pygame.image.fromstring(buf, (w, h), "RGB"), False, True)
    pygame.image.save(surf, f"assets/images/screenshot_{index:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="UAV jamming simulator")
    parser.add_argument("--link", choices=("auto", "gnuradio", "loopback"), default="auto",
                        help="radio backend: real GNU Radio PHY or the approximate numpy model")
    args = parser.parse_args()
    backend = pick_backend(args.link)

    world = World()
    command_link, telemetry_link, uav, operator, hud = build_simulation(world, backend)
    camera = Camera()
    rf = RFEnvironment(terrain=world.ground)
    window = Window()
    renderer = Renderer(world, uav, window.width, window.height)

    jammers: List[Jammer] = []
    reset_jammers(jammers, world.extent_half, uav.body.position, world.ground)

    paused = False
    show_heatmap = True
    screenshot_count = 0
    os.makedirs("assets/images", exist_ok=True)

    sim_time = 0.0
    accumulator = 0.0
    budget = update_radio(rf, world, jammers, uav, command_link, telemetry_link, sim_time)
    previous_time = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            frame_dt = min(now - previous_time, 0.25)
            previous_time = now

            input_state = window.poll()
            if input_state.quit:
                break

            if input_state.toggle_pause:
                paused = not paused
                pygame.mouse.set_visible(paused)
                pygame.event.set_grab(not paused)
                pygame.mouse.get_rel()

            if input_state.toggle_camera_mode:
                camera.toggle_mode(uav.body.position)
            if input_state.reset:
                uav.reset()
                operator.reset()
                command_link.clear(_BASE_NOISE)
                telemetry_link.clear(_BASE_NOISE)
                reset_jammers(jammers, world.extent_half, uav.body.position, world.ground)
                sim_time = 0.0
                accumulator = 0.0
            if input_state.toggle_heatmap:
                show_heatmap = not show_heatmap
            if input_state.toggle_dssc:
                cycle_mitigation(command_link, telemetry_link)

            worker_error = ""
            if not command_link.worker_alive:
                worker_error = f"CMD LINK: {command_link._worker_crashed_info}"
            elif not telemetry_link.worker_alive:
                worker_error = f"TLM LINK: {telemetry_link._worker_crashed_info}"

            if not paused:
                operator.handle_input(
                    PilotInput(
                        keys=input_state.flight_keys,
                        gamepad_axes=input_state.gamepad_axes,
                        toggle_arm=input_state.toggle_arm,
                        cycle_mode=input_state.cycle_mode,
                        cycle_failsafe=input_state.cycle_failsafe,
                        toggle_tx=input_state.toggle_tx,
                    ),
                    frame_dt,
                )
                budget = update_radio(rf, world, jammers, uav, command_link, telemetry_link,
                                      sim_time)
                accumulator += frame_dt
                steps = 0
                while accumulator >= PHYSICS_DT and steps < MAX_STEPS_PER_FRAME:
                    operator.tick(sim_time)
                    uav.step(PHYSICS_DT, sim_time)
                    sim_time += PHYSICS_DT
                    accumulator -= PHYSICS_DT
                    steps += 1
                if steps == MAX_STEPS_PER_FRAME:
                    accumulator = 0.0  # running slower than real time: drop the backlog

            snapshot = hud.refresh(
                sim_time,
                worker_error=worker_error,
                uplink_correlation=command_link.last_correlation,
                downlink_correlation=telemetry_link.last_correlation,
            )
            camera.update(frame_dt, uav.body.position, input_state.arrow_keys,
                          input_state.mouse_delta)

            window.begin_frame()
            eye, target = camera.eye_and_target(uav.body.position)
            eye = keep_above_ground(eye, world.ground)
            renderer.draw_scene(eye, target, jammers, sim_time, budget,
                                heatmap_rf=rf if show_heatmap else None,
                                link_margin_db=mitigation_gain_db(command_link))
            renderer.draw_hud(snapshot, paused, jammers, budget,
                              dssc_N=command_link.dssc_N, link_backend=backend,
                              gamepad=window.gamepad_name,
                              phy_label=mitigation_label(command_link))
            if input_state.screenshot:
                screenshot_count += 1
                save_screenshot(window, screenshot_count)
            window.end_frame()

            frame_time = time.perf_counter() - now
            time.sleep(max(0.0, 1.0 / TARGET_FPS - frame_time))
    finally:
        Window.quit()


if __name__ == "__main__":
    main()
