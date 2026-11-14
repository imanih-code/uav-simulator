"""OpenGL window + input polling, isolated from simulation logic.

`pygame` is used purely as a cross-platform window/context/input layer.
All actual drawing goes through PyOpenGL, so no textures or high level 2D
APIs are involved -- only raw points, lines and vertices, as requested.

Every pygame event is drained exactly once per frame, in `poll()`, and
handed back as a small `InputState`. This avoids the classic bug of
calling `pygame.event.get()` more than once per frame (the second call
would just see an empty list, silently dropping key-down events like the
camera-mode toggle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

import pygame
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
    glClear,
    glClearColor,
    glEnable,
    glLoadIdentity,
    glMatrixMode,
)
from OpenGL.GLU import gluPerspective

# Flight sticks, Mode 2 (see uavsim/entities/operator.py):
#   W/S throttle, A/D yaw (left stick) -- I/K pitch, J/L roll (right stick)
_FLIGHT_KEY_CODES = {
    "w": pygame.K_w,
    "s": pygame.K_s,
    "a": pygame.K_a,
    "d": pygame.K_d,
    "i": pygame.K_i,
    "k": pygame.K_k,
    "j": pygame.K_j,
    "l": pygame.K_l,
}

# Gamepad: axis indices for (left_x, left_y, right_x, right_y). These match
# XInput / SDL GameController pads (Xbox, most PlayStation pads under SDL2).
# Change them if your controller reports sticks on other axes.
GAMEPAD_AXES = (0, 1, 2, 3)
GAMEPAD_BUTTON_ARM = 0    # A / Cross
GAMEPAD_BUTTON_MODE = 1   # B / Circle
GAMEPAD_BUTTON_FAILSAFE = 3  # Y / Triangle

_ARROW_KEY_CODES = {
    "up": pygame.K_UP,
    "down": pygame.K_DOWN,
    "left": pygame.K_LEFT,
    "right": pygame.K_RIGHT,
}


@dataclass
class InputState:
    quit: bool = False
    toggle_camera_mode: bool = False
    reset: bool = False
    toggle_pause: bool = False
    toggle_dssc: bool = False
    screenshot: bool = False
    toggle_arm: bool = False
    cycle_mode: bool = False
    cycle_failsafe: bool = False
    toggle_tx: bool = False
    toggle_heatmap: bool = False

    flight_keys: Set[str] = field(default_factory=set)
    gamepad_axes: Optional[Tuple[float, float, float, float]] = None
    arrow_keys: Set[str] = field(default_factory=set)
    mouse_delta: Tuple[float, float] = (0.0, 0.0)


class Window:
    """Owns the OS window, the GL context, and all raw input polling."""

    def __init__(self, width: int = 1024, height: int = 768, title: str = "UAVSIM") -> None:
        pygame.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)
        self.width = width
        self.height = height
        self._configure_gl()
        self._configure_mouse()
        pygame.joystick.init()
        self._gamepad = None
        self._open_gamepad()

    def _open_gamepad(self) -> None:
        if pygame.joystick.get_count() > 0:
            self._gamepad = pygame.joystick.Joystick(0)
            self._gamepad.init()
        else:
            self._gamepad = None

    @property
    def gamepad_name(self) -> str:
        return self._gamepad.get_name() if self._gamepad is not None else ""

    def _configure_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.67, 0.76, 0.85, 1.0)  # horizon colour; the sky gradient covers it
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, self.width / self.height, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)

    def _configure_mouse(self) -> None:
        # Grabbed + hidden so mouse movement reads as pure look-around
        # motion (free camera / orbit), never hitting the screen edges.
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        pygame.mouse.get_rel()  # discard the first, possibly-huge delta

    def poll(self) -> InputState:
        """Drain this frame's events and current key/mouse state exactly
        once, returning everything the rest of the app needs."""
        state = InputState()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    state.quit = True
                elif event.key == pygame.K_v:
                    state.toggle_camera_mode = True
                elif event.key == pygame.K_BACKSPACE:
                    state.reset = True
                elif event.key == pygame.K_p:
                    state.toggle_pause = True
                elif event.key == pygame.K_h:
                    state.toggle_heatmap = True
                elif event.key == pygame.K_f:
                    state.toggle_dssc = True
                elif event.key == pygame.K_F12:
                    state.screenshot = True
                elif event.key == pygame.K_SPACE:
                    state.toggle_arm = True
                elif event.key == pygame.K_m:
                    state.cycle_mode = True
                elif event.key == pygame.K_g:
                    state.cycle_failsafe = True
                elif event.key == pygame.K_x:
                    state.toggle_tx = True
            elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                self._open_gamepad()
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == GAMEPAD_BUTTON_ARM:
                    state.toggle_arm = True
                elif event.button == GAMEPAD_BUTTON_MODE:
                    state.cycle_mode = True
                elif event.button == GAMEPAD_BUTTON_FAILSAFE:
                    state.cycle_failsafe = True

        pressed = pygame.key.get_pressed()
        state.flight_keys = {name for name, code in _FLIGHT_KEY_CODES.items() if pressed[code]}
        if self._gamepad is not None and self._gamepad.get_numaxes() > max(GAMEPAD_AXES):
            state.gamepad_axes = tuple(self._gamepad.get_axis(i) for i in GAMEPAD_AXES)
        state.arrow_keys = {name for name, code in _ARROW_KEY_CODES.items() if pressed[code]}
        state.mouse_delta = pygame.mouse.get_rel()
        return state

    @staticmethod
    def begin_frame() -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    @staticmethod
    def end_frame() -> None:
        pygame.display.flip()

    @staticmethod
    def quit() -> None:
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        pygame.quit()
