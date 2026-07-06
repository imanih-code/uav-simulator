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
from typing import Set, Tuple

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

_MOTOR_KEY_CODES = {
    "1": pygame.K_1,
    "2": pygame.K_2,
    "3": pygame.K_3,
    "4": pygame.K_4,
    "5": pygame.K_5,
    "q": pygame.K_q,
    "w": pygame.K_w,
    "e": pygame.K_e,
    "r": pygame.K_r,
    "t": pygame.K_t,
}

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
    toggle_crc: bool = False
    place_jammers: bool = False
    jammer_count_up: bool = False
    jammer_count_down: bool = False
    motor_keys: Set[str] = field(default_factory=set)
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

    def _configure_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.05, 0.05, 0.08, 1.0)
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
                elif event.key == pygame.K_c:
                    state.toggle_crc = True
                elif event.key == pygame.K_j:
                    state.place_jammers = True
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    state.jammer_count_up = True
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    state.jammer_count_down = True

        pressed = pygame.key.get_pressed()
        state.motor_keys = {name for name, code in _MOTOR_KEY_CODES.items() if pressed[code]}
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
