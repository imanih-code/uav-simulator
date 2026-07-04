"""World camera: an external observer that watches the UAV move through the
world. It is never attached to the UAV's own orientation -- there is no
first-person/cockpit mode, by design.

Two modes:
  * FOLLOW -- orbits around the UAV at a fixed distance, like a GTA-style
    chase camera. The mouse and arrow keys rotate the orbit angle.
  * FREE   -- a fully independent spectator camera. Arrow keys move it
    through space, the mouse looks around, completely decoupled from the
    UAV's position.

`V` toggles between the two (handled by whoever owns input, this class
just exposes `toggle_mode`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

import numpy as np

_MIN_ELEVATION = math.radians(-80.0)
_MAX_ELEVATION = math.radians(80.0)
_MIN_DISTANCE = 2.0
_MAX_DISTANCE = 40.0


class CameraMode(Enum):
    FOLLOW = auto()
    FREE = auto()


@dataclass
class Camera:
    mouse_sensitivity: float = 0.0025
    orbit_key_rate: float = 1.2       # rad/sec when using arrow keys to orbit
    zoom_speed: float = 6.0           # units/sec
    free_move_speed: float = 6.0      # units/sec

    mode: CameraMode = field(default=CameraMode.FOLLOW)

    # FOLLOW mode state: spherical coordinates around the UAV.
    follow_azimuth: float = math.radians(135.0)
    follow_elevation: float = math.radians(25.0)
    follow_distance: float = 6.0

    # FREE mode state: fully independent position + look direction.
    free_position: np.ndarray = field(
        default_factory=lambda: np.array([6.0, 6.0, 4.0])
    )
    free_yaw: float = math.radians(-135.0)
    free_pitch: float = math.radians(-20.0)

    def toggle_mode(self, current_target: np.ndarray) -> None:
        """Switch modes, keeping the viewpoint roughly where it was so the
        cut isn't jarring."""
        if self.mode is CameraMode.FOLLOW:
            eye, _ = self._follow_eye_and_target(current_target)
            self.free_position = eye.copy()
            self.free_yaw, self.free_pitch = _look_direction_to_yaw_pitch(
                current_target - eye
            )
            self.mode = CameraMode.FREE
        else:
            offset = self.free_position - current_target
            distance = float(np.linalg.norm(offset))
            self.follow_distance = min(max(distance, _MIN_DISTANCE), _MAX_DISTANCE)
            self.follow_azimuth = math.atan2(offset[1], offset[0])
            horizontal = math.hypot(offset[0], offset[1])
            self.follow_elevation = math.atan2(offset[2], horizontal)
            self.mode = CameraMode.FOLLOW

    def update(
        self,
        dt: float,
        target_position: np.ndarray,
        arrow_keys: set,
        mouse_delta: Tuple[float, float],
    ) -> None:
        if self.mode is CameraMode.FOLLOW:
            self._update_follow(dt, arrow_keys, mouse_delta)
        else:
            self._update_free(dt, arrow_keys, mouse_delta)

    def eye_and_target(self, target_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode is CameraMode.FOLLOW:
            return self._follow_eye_and_target(target_position)
        return self._free_eye_and_target()

    # -- FOLLOW mode ---------------------------------------------------------
    def _update_follow(self, dt: float, arrow_keys: set, mouse_delta: Tuple[float, float]) -> None:
        dx, dy = mouse_delta
        self.follow_azimuth += dx * self.mouse_sensitivity
        self.follow_elevation -= dy * self.mouse_sensitivity

        if "left" in arrow_keys:
            self.follow_azimuth -= self.orbit_key_rate * dt
        if "right" in arrow_keys:
            self.follow_azimuth += self.orbit_key_rate * dt
        if "up" in arrow_keys:
            self.follow_distance -= self.zoom_speed * dt
        if "down" in arrow_keys:
            self.follow_distance += self.zoom_speed * dt

        self.follow_elevation = _clamp(self.follow_elevation, _MIN_ELEVATION, _MAX_ELEVATION)
        self.follow_distance = _clamp(self.follow_distance, _MIN_DISTANCE, _MAX_DISTANCE)

    def _follow_eye_and_target(self, target_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cos_el = math.cos(self.follow_elevation)
        offset = np.array(
            [
                self.follow_distance * cos_el * math.cos(self.follow_azimuth),
                self.follow_distance * cos_el * math.sin(self.follow_azimuth),
                self.follow_distance * math.sin(self.follow_elevation),
            ]
        )
        eye = target_position + offset
        return eye, target_position

    # -- FREE mode -------------------------------------------------------------
    def _update_free(self, dt: float, arrow_keys: set, mouse_delta: Tuple[float, float]) -> None:
        dx, dy = mouse_delta
        self.free_yaw += dx * self.mouse_sensitivity
        self.free_pitch -= dy * self.mouse_sensitivity
        self.free_pitch = _clamp(self.free_pitch, _MIN_ELEVATION, _MAX_ELEVATION)

        forward = _yaw_pitch_to_forward(self.free_yaw, self.free_pitch)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        right = right / right_norm if right_norm > 1e-9 else np.array([1.0, 0.0, 0.0])

        move = np.zeros(3)
        if "up" in arrow_keys:
            move += forward
        if "down" in arrow_keys:
            move -= forward
        if "right" in arrow_keys:
            move += right
        if "left" in arrow_keys:
            move -= right

        norm = np.linalg.norm(move)
        if norm > 1e-9:
            self.free_position = self.free_position + (move / norm) * self.free_move_speed * dt

    def _free_eye_and_target(self) -> Tuple[np.ndarray, np.ndarray]:
        forward = _yaw_pitch_to_forward(self.free_yaw, self.free_pitch)
        return self.free_position, self.free_position + forward


def _yaw_pitch_to_forward(yaw: float, pitch: float) -> np.ndarray:
    cos_pitch = math.cos(pitch)
    return np.array(
        [
            cos_pitch * math.cos(yaw),
            cos_pitch * math.sin(yaw),
            math.sin(pitch),
        ]
    )


def _look_direction_to_yaw_pitch(direction: np.ndarray) -> Tuple[float, float]:
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return 0.0, 0.0
    direction = direction / norm
    yaw = math.atan2(direction[1], direction[0])
    pitch = math.asin(_clamp(direction[2], -1.0, 1.0))
    return yaw, pitch


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
