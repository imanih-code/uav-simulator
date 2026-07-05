"""Draws the flat world, the UAV, and the HUD (numeric read-outs + two
oscilloscope-style signal panels). Only `GL_POINTS` and `GL_LINES` are
used -- no textures, no meshes -- matching the "solo puntos y vertices"
requirement. Text is drawn with the hand-rolled stroke font in
`vector_font.py`, which is itself nothing but line segments.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from OpenGL.GL import (
    GL_DEPTH_TEST,
    GL_LINE_LOOP,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_MODELVIEW_MATRIX,
    GL_POINTS,
    GL_PROJECTION,
    GL_PROJECTION_MATRIX,
    GL_VIEWPORT,
    glBegin,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glGetDoublev,
    glGetIntegerv,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPointSize,
    glPopMatrix,
    glPushMatrix,
    glVertex2f,
    glVertex3f,
)
from OpenGL.GLU import gluLookAt, gluProject

from uavsim.entities.uav import UAV
from uavsim.hud.hud import SIGNAL_WINDOW_SECONDS, HUDSnapshot
from uavsim.rendering.vector_font import draw_text
from uavsim.world.environment import WORLD_EXTENT_HALF, World

GROUND_COLOR = (0.2, 0.25, 0.3)
GROUND_MAJOR_COLOR = (0.3, 0.4, 0.5)
GROUND_AXIS_COLOR = (0.6, 0.3, 0.3)
UAV_CENTER_COLOR = (1.0, 0.0, 0.0)
UAV_MOTOR_COLORS = [
    (1.0, 1.0, 0.0),   # 0 FR → amarillo
    (0.0, 1.0, 1.0),   # 1 BR → cyan
    (1.0, 0.0, 1.0),   # 2 BL → magenta
    (0.5, 1.0, 0.0),   # 3 FL → verde lima
]
UAV_ARM_COLOR = (0.6, 0.6, 0.6)

HUD_TEXT_COLOR = (0.75, 1.0, 0.8)
HUD_LOG_OK_COLOR = (0.3, 1.0, 0.4)     # verde
HUD_LOG_BAD_COLOR = (1.0, 0.3, 0.3)     # rojo
HUD_PANEL_BORDER_COLOR = (0.4, 0.6, 0.5)
HUD_WAVEFORM_COLOR = (0.3, 1.0, 0.5)
HUD_BAR_COLOR = (0.9, 0.9, 0.9)

GROUND_GRID_SPAN = WORLD_EXTENT_HALF
GROUND_MAJOR_STEP = 5

_SIGNAL_BITS_SAMPLED = 8
_SIGNAL_BIT_DURATION = 0.01  # seconds per drawn bit


class Renderer:
    """Draws one scene per frame. Owns no camera or simulation state."""

    def __init__(self, world: World, uav: UAV, hud_width: int, hud_height: int) -> None:
        self.world = world
        self.uav = uav
        self.hud_width = hud_width
        self.hud_height = hud_height
        self._motor_labels: List[Tuple[float, float, int]] = []

    def draw_scene(self, camera_eye: np.ndarray, camera_target: np.ndarray) -> None:
        gluLookAt(
            camera_eye[0], camera_eye[1], camera_eye[2],
            camera_target[0], camera_target[1], camera_target[2],
            0.0, 0.0, 1.0,
        )
        self._draw_ground()
        motor_points = self._draw_uav()
        self._project_motor_labels(motor_points)

    def draw_hud(self, snapshot: HUDSnapshot) -> None:
        self._begin_hud_overlay()
        self._draw_motor_labels()
        self._draw_telemetry_readout(snapshot)
        self._draw_command_log(snapshot)
        self._draw_signal_panels(snapshot)
        if snapshot.has_telemetry and snapshot.motor_throttle is not None:
            self._draw_throttle_bars(snapshot.motor_throttle)
        self._end_hud_overlay()

    # -- world ---------------------------------------------------------------
    def _draw_ground(self) -> None:
        ground_z = self.world.ground.ground_z
        glBegin(GL_LINES)

        # Minor grid lines (every 1 unit)
        glColor3f(*GROUND_COLOR)
        for i in range(-GROUND_GRID_SPAN, GROUND_GRID_SPAN + 1):
            if i % GROUND_MAJOR_STEP == 0:
                continue
            glVertex3f(i, -GROUND_GRID_SPAN, ground_z)
            glVertex3f(i, GROUND_GRID_SPAN, ground_z)
            glVertex3f(-GROUND_GRID_SPAN, i, ground_z)
            glVertex3f(GROUND_GRID_SPAN, i, ground_z)

        # Major grid lines (every GROUND_MAJOR_STEP units)
        glColor3f(*GROUND_MAJOR_COLOR)
        for i in range(-GROUND_GRID_SPAN, GROUND_GRID_SPAN + 1, GROUND_MAJOR_STEP):
            glVertex3f(i, -GROUND_GRID_SPAN, ground_z)
            glVertex3f(i, GROUND_GRID_SPAN, ground_z)
            glVertex3f(-GROUND_GRID_SPAN, i, ground_z)
            glVertex3f(GROUND_GRID_SPAN, i, ground_z)

        # X axis (red tint)
        glColor3f(*GROUND_AXIS_COLOR)
        glVertex3f(-GROUND_GRID_SPAN, 0.0, ground_z)
        glVertex3f(GROUND_GRID_SPAN, 0.0, ground_z)
        glVertex3f(0.0, -GROUND_GRID_SPAN, ground_z)
        glVertex3f(0.0, GROUND_GRID_SPAN, ground_z)

        glEnd()

    def _draw_uav(self) -> List[np.ndarray]:
        center = self.uav.body.position
        motor_points = self.uav.motor_world_positions()

        glColor3f(*UAV_ARM_COLOR)
        glBegin(GL_LINES)
        for point in motor_points:
            glVertex3f(center[0], center[1], center[2])
            glVertex3f(point[0], point[1], point[2])
        # Draw a cross shape for better visibility
        arm = 0.15
        glColor3f(0.8, 0.8, 0.8)
        glVertex3f(center[0] - arm, center[1], center[2])
        glVertex3f(center[0] + arm, center[1], center[2])
        glVertex3f(center[0], center[1] - arm, center[2])
        glVertex3f(center[0], center[1] + arm, center[2])
        glEnd()

        glPointSize(10.0)
        glColor3f(*UAV_CENTER_COLOR)
        glBegin(GL_POINTS)
        glVertex3f(center[0], center[1], center[2])
        glEnd()

        glPointSize(8.0)
        glBegin(GL_POINTS)
        for i, point in enumerate(motor_points):
            glColor3f(*UAV_MOTOR_COLORS[i])
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        return motor_points

    # -- motor labels (projected to screen coordinates) -------------------------
    def _project_motor_labels(self, motor_points: List[np.ndarray]) -> None:
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        labels = []
        for i, pt in enumerate(motor_points):
            win = gluProject(float(pt[0]), float(pt[1]), float(pt[2]),
                             modelview, projection, viewport)
            if win is not None and 0.0 <= win[2] <= 1.0:
                labels.append((int(win[0]), int(win[1]), i))
        self._motor_labels = labels

    def _draw_motor_labels(self) -> None:
        if not self._motor_labels:
            return
        glBegin(GL_LINES)
        for sx, sy, idx in self._motor_labels:
            glColor3f(*UAV_MOTOR_COLORS[idx])
            leader_h = 14.0
            glVertex2f(sx, sy + leader_h)
            glVertex2f(sx, sy)
            label = str(idx + 1)
            draw_text(label, sx - 4, sy + leader_h + 2, 7.0, 10.0, 1.5, self._segment_drawer())
        glEnd()

    # -- HUD overlay setup -----------------------------------------------------
    def _begin_hud_overlay(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.hud_width, 0, self.hud_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

    def _end_hud_overlay(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # -- HUD: numeric telemetry readout -----------------------------------------
    def _draw_telemetry_readout(self, snapshot: HUDSnapshot) -> None:
        margin = 14
        char_w, char_h, spacing, line_gap = 7.0, 10.0, 2.0, 6.0
        line_height = char_h + line_gap

        lines = self._telemetry_lines(snapshot)

        glColor3f(*HUD_TEXT_COLOR)
        glBegin(GL_LINES)
        y = self.hud_height - margin - char_h
        for line in lines:
            draw_text(line, margin, y, char_w, char_h, spacing, self._segment_drawer())
            y -= line_height
        glEnd()

    @staticmethod
    def _telemetry_lines(snapshot: HUDSnapshot) -> List[str]:
        if not snapshot.has_telemetry:
            return ["BAT:--", "ALT:--", "MASS:--", "POSX:--", "POSY:--",
                    "ROL:--", "PIT:--", "YAW:--"]

        x, y, z = snapshot.position
        roll, pitch, yaw = snapshot.attitude_deg
        return [
            f"BAT:{snapshot.battery_percent:.0f}%",
            f"ALT:{z:.1f}",
            f"MASS:{snapshot.mass_kg:.2f}",
            f"POSX:{x:.1f}",
            f"POSY:{y:.1f}",
            f"ROL:{roll:.1f}",
            f"PIT:{pitch:.1f}",
            f"YAW:{yaw:.1f}",
        ]

    # -- HUD: command log -------------------------------------------------------
    def _draw_command_log(self, snapshot: HUDSnapshot) -> None:
        if not snapshot.command_log:
            return
        margin = 14
        char_w, char_h, spacing = 6.0, 9.0, 1.5
        line_h = char_h + 4
        telemetry_height = 8 * (10 + 6)
        # Start below the telemetry readout so no overlap
        y = self.hud_height - margin - 10 - telemetry_height - 12 - (len(snapshot.command_log)) * line_h

        glBegin(GL_LINES)
        for label, valid in snapshot.command_log:
            y += line_h
            color = HUD_LOG_OK_COLOR if valid else HUD_LOG_BAD_COLOR
            glColor3f(*color)
            draw_text(label, margin, y, char_w, char_h, spacing, self._segment_drawer())
        glEnd()

    # -- HUD: TX/RX oscilloscope-style signal panels ----------------------------
    def _draw_signal_panels(self, snapshot: HUDSnapshot) -> None:
        panel_width, panel_height = 220.0, 46.0
        gap = 10.0
        x = self.hud_width - panel_width - 14.0
        tx_y = self.hud_height - panel_height - 14.0
        rx_y = tx_y - panel_height - gap

        self._draw_signal_panel(
            "TX", snapshot.uplink_signal, snapshot.uplink_bandwidth_bps, snapshot.now,
            x, tx_y, panel_width, panel_height,
        )
        self._draw_signal_panel(
            "RX", snapshot.downlink_signal, snapshot.downlink_bandwidth_bps, snapshot.now,
            x, rx_y, panel_width, panel_height,
        )

    def _draw_signal_panel(
        self,
        label: str,
        transmissions: Tuple[Tuple[float, bytes], ...],
        bandwidth_bps: float,
        now: float,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        glColor3f(*HUD_PANEL_BORDER_COLOR)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()

        char_w, char_h, spacing = 6.0, 9.0, 1.5
        glColor3f(*HUD_TEXT_COLOR)
        glBegin(GL_LINES)
        label_text = f"{label} {bandwidth_bps:.0f}B/S"
        draw_text(label_text, x + 4, y + height - char_h - 3, char_w, char_h, spacing,
                   self._segment_drawer())
        glEnd()

        points = _waveform_points(transmissions, now, SIGNAL_WINDOW_SECONDS)
        low_y = y + height * 0.18
        high_y = y + height * 0.55
        glColor3f(*HUD_WAVEFORM_COLOR)
        glBegin(GL_LINE_STRIP)
        for t_offset, level in points:
            t_clamped = min(max(t_offset, 0.0), SIGNAL_WINDOW_SECONDS)
            px = x + (t_clamped / SIGNAL_WINDOW_SECONDS) * width
            py = high_y if level else low_y
            glVertex2f(px, py)
        glEnd()

    # -- HUD: per-motor throttle bars --------------------------------------------
    def _draw_throttle_bars(self, throttle: Tuple[float, float, float, float]) -> None:
        bar_width, gap, max_height, margin = 20.0, 12.0, 110.0, 16.0
        base_y = margin + 14.0

        glBegin(GL_LINES)
        for i, value in enumerate(throttle):
            x = margin + i * (bar_width + gap)
            glColor3f(*UAV_MOTOR_COLORS[i])
            glVertex2f(x, base_y)
            glVertex2f(x, base_y + max_height * value)
            glVertex2f(x + bar_width, base_y)
            glVertex2f(x + bar_width, base_y + max_height * value)
            glVertex2f(x, base_y)
            glVertex2f(x + bar_width, base_y)
        glEnd()

        glColor3f(*HUD_TEXT_COLOR)
        glBegin(GL_LINES)
        for i in range(len(throttle)):
            x = margin + i * (bar_width + gap)
            draw_text(str(i + 1), x + bar_width * 0.3, margin - 2, 6.0, 9.0, 1.5,
                       self._segment_drawer())
        glEnd()

    @staticmethod
    def _segment_drawer():
        def _draw(p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
            glVertex2f(p1[0], p1[1])
            glVertex2f(p2[0], p2[1])

        return _draw


def _waveform_points(
    transmissions: Tuple[Tuple[float, bytes], ...],
    now: float,
    window_seconds: float,
) -> List[Tuple[float, int]]:
    """Turn a burst of raw transmissions into an NRZ square-wave trace.

    Each transmitted message becomes a short burst of high/low pulses (the
    first `_SIGNAL_BITS_SAMPLED` bits of its first byte), starting exactly
    when it was sent. This is literally the bit pattern the CommGateway
    encoded, not a decorative animation.
    """
    window_start = now - window_seconds
    points: List[Tuple[float, int]] = [(0.0, 0)]

    for timestamp, payload in transmissions:
        if not payload:
            continue
        t0 = timestamp - window_start
        first_byte = payload[0]
        bits = [(first_byte >> i) & 1 for i in range(_SIGNAL_BITS_SAMPLED - 1, -1, -1)]

        points.append((t0, 0))
        for i, bit in enumerate(bits):
            seg_start = t0 + i * _SIGNAL_BIT_DURATION
            seg_end = seg_start + _SIGNAL_BIT_DURATION
            points.append((seg_start, bit))
            points.append((seg_end, bit))
        points.append((t0 + len(bits) * _SIGNAL_BIT_DURATION, 0))

    points.append((window_seconds, 0))
    points.sort(key=lambda p: p[0])
    return points
