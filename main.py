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
ESC            -> quit

The UAV starts parked on the ground, at rest -- it only leaves the ground
once the operator gives it enough thrust to actually take off.
"""
from __future__ import annotations

import time
from typing import Tuple

from uavsim.comms.gateway import CommGatewayInput, CommGatewayOutput
from uavsim.comms.gnuradio_link import GnuRadioChannel
from uavsim.entities.operator import UAVOperator
from uavsim.entities.uav import UAV, UAVConfig
from uavsim.hud.hud import HUD
from uavsim.rendering.camera import Camera
from uavsim.rendering.renderer import Renderer
from uavsim.rendering.window import Window
from uavsim.world.environment import World

TARGET_FPS = 60


def build_simulation(world: World) -> Tuple[UAV, UAVOperator, HUD]:
    """Create the UAV and Operator, and connect them with two real radio
    links (each a `GnuRadioChannel`): one carrying commands
    (Operator -> UAV) and one carrying telemetry (UAV -> Operator). Both
    are genuinely GMSK-modulated, pushed through a simulated noisy
    channel, and demodulated back -- not an in-memory shortcut. The HUD is
    built on top of the Operator's telemetry buffer, never on the UAV
    itself.
    """
    command_link = GnuRadioChannel()
    telemetry_link = GnuRadioChannel()

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
    hud = HUD(operator)
    return uav, operator, hud


def main() -> None:
    world = World()
    uav, operator, hud = build_simulation(world)
    camera = Camera()

    window = Window()
    renderer = Renderer(world, uav, window.width, window.height)

    previous_time = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt = now - previous_time
            previous_time = now

            input_state = window.poll()
            if input_state.quit:
                break
            if input_state.toggle_camera_mode:
                camera.toggle_mode(uav.body.position)

            operator.handle_pressed_keys(input_state.motor_keys)
            uav.update(dt)
            camera.update(dt, uav.body.position, input_state.arrow_keys, input_state.mouse_delta)
            snapshot = hud.refresh()

            window.begin_frame()
            eye, target = camera.eye_and_target(uav.body.position)
            renderer.draw_scene(eye, target)
            renderer.draw_hud(snapshot)
            window.end_frame()

            frame_time = time.perf_counter() - now
            time.sleep(max(0.0, 1.0 / TARGET_FPS - frame_time))
    finally:
        Window.quit()


if __name__ == "__main__":
    main()
