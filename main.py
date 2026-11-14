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
F              chaotic DSSS anti-jamming (N=8) on/off
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
from uavsim.entities.jammer import Jammer
from uavsim.entities.operator import PilotInput, UAVOperator
from uavsim.entities.uav import UAV, UAVConfig
from uavsim.hud.hud import HUD
from uavsim.rendering.camera import Camera
from uavsim.rendering.renderer import Renderer
from uavsim.rendering.window import Window
from uavsim.world.environment import World

TARGET_FPS = 60
PHYSICS_HZ = 250
PHYSICS_DT = 1.0 / PHYSICS_HZ
MAX_STEPS_PER_FRAME = 25  # never try to catch up more than 0.1 s of sim time
_BASE_NOISE = 0.005
_JAMMER_COUNT = 6


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


def update_jammer_noise(jammers: List[Jammer], uav: UAV, command_link, telemetry_link) -> None:
    noise = _BASE_NOISE
    for jammer in jammers:
        noise = max(noise, jammer.noise_at(uav.body.position))
    if noise != command_link.noise_voltage:
        command_link.noise_voltage = noise
        telemetry_link.noise_voltage = noise


def place_random_jammers(
    jammers: List[Jammer], count: int, extent_half: float, uav_pos: np.ndarray, ground_z: float
) -> None:
    jammer_radius = Jammer().radius
    exclusion = jammer_radius * 1.5
    margin = 5.0
    placed = 0
    for _ in range(count * 20):
        if placed >= count:
            break
        x = random.uniform(-extent_half + margin, extent_half - margin)
        y = random.uniform(-extent_half + margin, extent_half - margin)
        pos = np.array([x, y, ground_z])
        if np.linalg.norm(pos[:2] - uav_pos[:2]) < exclusion:
            continue
        if any(np.linalg.norm(pos[:2] - j.position[:2]) < jammer_radius * 2 for j in jammers):
            continue
        jammers.append(Jammer(position=pos))
        placed += 1


def reset_jammers(jammers: List[Jammer], extent_half: float, uav_pos: np.ndarray,
                  ground_z: float) -> None:
    jammers.clear()
    place_random_jammers(jammers, _JAMMER_COUNT, extent_half, uav_pos, ground_z)


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
    window = Window()
    renderer = Renderer(world, uav, window.width, window.height)

    jammers: List[Jammer] = []
    reset_jammers(jammers, world.extent_half, uav.body.position, world.ground.ground_z)

    paused = False
    screenshot_count = 0
    os.makedirs("assets/images", exist_ok=True)

    sim_time = 0.0
    accumulator = 0.0
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
                reset_jammers(jammers, world.extent_half, uav.body.position, world.ground.ground_z)
                sim_time = 0.0
                accumulator = 0.0
            if input_state.toggle_dssc:
                new_n = 0 if command_link.dssc_N > 0 else 8
                for link in (command_link, telemetry_link):
                    link.clear(link.noise_voltage)
                    link.dssc_N = new_n

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
                update_jammer_noise(jammers, uav, command_link, telemetry_link)
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
            renderer.draw_scene(eye, target, jammers)
            renderer.draw_hud(snapshot, paused, jammers,
                              command_link.noise_voltage, telemetry_link.noise_voltage,
                              dssc_N=command_link.dssc_N, link_backend=backend,
                              gamepad=window.gamepad_name)
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
