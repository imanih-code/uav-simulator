"""Entry point: wires comms, physics, camera, HUD and rendering together,
then runs the real-time simulation loop.

Controls
--------
1 / 2 / 3 / 4  -> increase throttle on front-right / back-right / back-left
                   / front-left motor (holds up to full throttle)
Q / W / E / R  -> decrease throttle on the same four motors, gradually
                   down to zero
Arrow keys     -> FREE mode: fly the spectator camera around
                   FOLLOW mode: orbit around the UAV / zoom in-out
Mouse          -> look around (orbit in FOLLOW mode, free-look in FREE mode)
V              -> toggle between FOLLOW (chase cam) and FREE (spectator cam)
P              -> pause / unpause simulation
Backspace      -> reset UAV and reposition jammers randomly
ESC            -> quit

The UAV starts parked on the ground, at rest -- it only leaves the ground
once the operator gives it enough thrust to actually take off.
Jammers (orange-red cylinders with red range circles) are placed randomly
at startup and repositioned on reset.  They raise the noise floor on both
radio links whenever the UAV's centre is inside their effect radius.
"""
from __future__ import annotations

import random
import time
from typing import List, Tuple

import numpy as np
import pygame
from OpenGL.GL import GL_BACK, GL_RGB, GL_UNSIGNED_BYTE, glReadPixels, glReadBuffer

from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.gnuradio_link import GnuRadioChannel
from uavsim.entities.jammer import Jammer
from uavsim.entities.operator import UAVOperator
from uavsim.entities.uav import UAV, UAVConfig
from uavsim.hud.hud import HUD
from uavsim.rendering.camera import Camera
from uavsim.rendering.renderer import Renderer
from uavsim.rendering.window import Window
from uavsim.world.environment import World

TARGET_FPS = 60
_BASE_NOISE = 0.005
_JAMMER_COUNT = 6


def build_simulation(world: World) -> Tuple[GnuRadioChannel, GnuRadioChannel, UAV, UAVOperator, HUD]:
    """Create the UAV and Operator, and connect them with two real radio
    links (each a `GnuRadioChannel`): one carrying commands
    (Operator -> UAV) and one carrying telemetry (UAV -> Operator). Both
    are genuinely GMSK-modulated, pushed through a simulated noisy
    channel, and demodulated back -- not an in-memory shortcut. The TX
    panel in the HUD reads from the UAV's command input (Rx side of the
    command link) so it shows what actually survived the noisy channel.
    """
    command_link = GnuRadioChannel(noise_voltage=_BASE_NOISE)
    telemetry_link = GnuRadioChannel(noise_voltage=_BASE_NOISE)
    # Activar DSSS caótico: añadir dssc_N=256 (o el valor deseado) a ambos canales

    uav = UAV(
        config=UAVConfig(),
        ground=world.ground,
        command_input=CommGatewayInput(command_link),
        telemetry_output=CommGatewayOutput(telemetry_link),
    )
    operator = UAVOperator(
        command_output=CommGatewayOutput(command_link),
        telemetry_input=CommGatewayInput(telemetry_link),
    )
    hud = HUD(operator, uav.command_input)
    return command_link, telemetry_link, uav, operator, hud


def update_jammer_noise(
    jammers: List[Jammer],
    uav: UAV,
    command_link: GnuRadioChannel,
    telemetry_link: GnuRadioChannel,
) -> None:
    noise = _BASE_NOISE
    uav_pos = uav.body.position
    for jammer in jammers:
        n = jammer.noise_at(uav_pos)
        if n > noise:
            noise = n
    command_link.noise_voltage = noise
    telemetry_link.noise_voltage = noise


def _jammer_radius() -> float:
    return Jammer().radius


def place_random_jammers(
    jammers: List[Jammer],
    count: int,
    extent_half: float,
    uav_pos: np.ndarray,
    ground_z: float,
) -> None:
    jammer_radius = _jammer_radius()
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
        too_close = False
        for j in jammers:
            if np.linalg.norm(pos[:2] - j.position[:2]) < jammer_radius * 2:
                too_close = True
                break
        if too_close:
            continue
        jammers.append(Jammer(position=pos))
        placed += 1


def reset_jammers(jammers: List[Jammer], extent_half: float,
                  uav_pos: np.ndarray, ground_z: float) -> None:
    jammers.clear()
    place_random_jammers(jammers, _JAMMER_COUNT, extent_half, uav_pos, ground_z)


def main() -> None:
    world = World()
    command_link, telemetry_link, uav, operator, hud = build_simulation(world)
    camera = Camera()

    window = Window()
    renderer = Renderer(world, uav, window.width, window.height)

    jammers: List[Jammer] = []
    reset_jammers(jammers, world.extent_half, uav.body.position, world.ground.ground_z)

    paused = False
    screenshot_count = 0
    os.makedirs("assets/images", exist_ok=True)

    previous_time = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt = now - previous_time
            previous_time = now

            input_state = window.poll()
            if input_state.quit:
                break

            # Pause toggle
            if input_state.toggle_pause:
                paused = not paused
                if paused:
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
                else:
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                    pygame.mouse.get_rel()

            # Camera always works (even when paused)
            if input_state.toggle_camera_mode:
                camera.toggle_mode(uav.body.position)
            if input_state.reset:
                uav.reset()
                command_link.clear()
                telemetry_link.clear()
                reset_jammers(jammers, world.extent_half, uav.body.position,
                              world.ground.ground_z)

            if input_state.toggle_dssc:
                new_val = 0 if command_link.dssc_N > 0 else 8
                command_link.clear()
                telemetry_link.clear()
                command_link.dssc_N = new_val
                telemetry_link.dssc_N = new_val

            worker_error = ""
            if not command_link.worker_alive:
                worker_error = f"CMD LINK: {command_link._worker_crashed_info}"
            elif not telemetry_link.worker_alive:
                worker_error = f"TLM LINK: {telemetry_link._worker_crashed_info}"

            if paused:
                snapshot = hud.refresh(worker_error=worker_error,
                                       uplink_correlation=command_link.last_correlation,
                                       downlink_correlation=telemetry_link.last_correlation)
            else:
                # Update noise before sending anything so the first post-reset
                # command doesn't use stale high noise from a previous jammer.
                update_jammer_noise(jammers, uav, command_link, telemetry_link)
                if input_state.motor_keys and not uav.radio_enabled:
                    uav.radio_enabled = True
                operator.handle_pressed_keys(input_state.motor_keys)
                uav.update(dt)
                snapshot = hud.refresh(worker_error=worker_error,
                                       uplink_correlation=command_link.last_correlation,
                                       downlink_correlation=telemetry_link.last_correlation)

            camera.update(dt, uav.body.position, input_state.arrow_keys, input_state.mouse_delta)

            window.begin_frame()
            eye, target = camera.eye_and_target(uav.body.position)
            renderer.draw_scene(eye, target, jammers)
            renderer.draw_hud(snapshot, paused, jammers,
                               command_link.noise_voltage,
                               telemetry_link.noise_voltage,
                               dssc_N=command_link.dssc_N)
            if input_state.screenshot:
                screenshot_count += 1
                glReadBuffer(GL_BACK)
                w, h = window.width, window.height
                buf = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
                surf = pygame.image.fromstring(buf, (w, h), "RGB")
                surf = pygame.transform.flip(surf, False, True)
                path = f"assets/images/screenshot_{screenshot_count:04d}.png"
                pygame.image.save(surf, path)

            window.end_frame()

            frame_time = time.perf_counter() - now
            time.sleep(max(0.0, 1.0 / TARGET_FPS - frame_time))
    finally:
        Window.quit()


if __name__ == "__main__":
    main()
