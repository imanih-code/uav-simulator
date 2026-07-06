"""Draws the flat world, the UAV, and the HUD (numeric read-outs + two
oscilloscope-style signal panels). Only `GL_POINTS` and `GL_LINES` are
used -- no textures, no meshes -- matching the "solo puntos y vertices"
requirement. Text is drawn with the hand-rolled stroke font in
`vector_font.py`, which is itself nothing but line segments.
"""
from __future__ import annotations

import time
from collections import deque
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
    GL_SCISSOR_TEST,
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
    glScissor,
    glVertex2f,
    glVertex3f,
)
from OpenGL.GLU import gluLookAt, gluProject

from uavsim.entities.jammer import Jammer
from uavsim.entities.uav import UAV
from uavsim.hud.hud import SIGNAL_WINDOW_SECONDS, HUDSnapshot
from uavsim.rendering.vector_font import draw_text
from uavsim.world.environment import WORLD_EXTENT_HALF, World

GROUND_COLOR = (0.2, 0.25, 0.3)
GROUND_MAJOR_COLOR = (0.3, 0.4, 0.5)
GROUND_AXIS_COLOR = (0.6, 0.3, 0.3)
UAV_CENTER_COLOR = (1.0, 0.0, 0.0)
UAV_BILLBOARD_CIRCLE_COLOR = (1.0, 1.0, 1.0)  # white, camera-facing circle
UAV_CHASSIS_CIRCLE_COLOR = (1.0, 1.0, 1.0)    # white, chassis-tilted circle
UAV_MOTOR_COLORS = [
    (1.0, 1.0, 0.0),   # 0 FR → amarillo
    (0.0, 1.0, 1.0),   # 1 BR → cyan
    (1.0, 0.0, 1.0),   # 2 BL → magenta
    (0.5, 1.0, 0.0),   # 3 FL → verde lima
]
UAV_ARM_COLOR = (0.6, 0.6, 0.6)

HUD_TEXT_COLOR = (0.75, 1.0, 0.8)
HUD_LOG_OK_COLOR = (0.3, 1.0, 0.4)     # verde — recibido correctamente
HUD_LOG_BAD_COLOR = (1.0, 0.3, 0.3)     # rojo — CRC corrupto
HUD_LOG_SENT_COLOR = (1.0, 0.65, 0.0)   # naranja — enviado pero no recibido
HUD_RAW_SIGNAL_COLOR = (0.2, 0.5, 0.8)  # azul tenue, señal analógica cruda
HUD_PANEL_BORDER_COLOR = (0.4, 0.6, 0.5)
HUD_WAVEFORM_COLOR = (0.3, 1.0, 0.5)
HUD_BAR_COLOR = (0.9, 0.9, 0.9)

JAMMER_BODY_COLOR = (1.0, 0.3, 0.1)     # naranja-rojizo
JAMMER_CIRCLE_COLOR = (1.0, 0.1, 0.1)   # rojo para círculos de radio
JAMMER_BLINK_COLOR_ON = (1.0, 0.0, 0.0)  # rojo encendido
JAMMER_BLINK_COLOR_OFF = (0.2, 0.0, 0.0) # rojo apagado (tenue)

MINIMAP_SIZE = 180
MINIMAP_MARGIN = 14
MINIMAP_EXTENT = 100.0
MINIMAP_BG_COLOR = (0.08, 0.10, 0.14)
MINIMAP_BORDER_COLOR = (0.4, 0.6, 0.5)
MINIMAP_UAV_COLOR = (1.0, 0.0, 0.0)
MINIMAP_MOTOR_RADIUS_PX = 8.0
MINIMAP_MOTOR_DOT_RADIUS = 2.5
MINIMAP_X_ARM = 5.0

NOISE_PANEL_HEIGHT = 55

GROUND_GRID_SPAN = WORLD_EXTENT_HALF
GROUND_MAJOR_STEP = 5

_SIGNAL_BIT_DURATION = 0.01  # seconds per drawn bit


class Renderer:
    """Draws one scene per frame. Owns no camera or simulation state."""

    def __init__(self, world: World, uav: UAV, hud_width: int, hud_height: int) -> None:
        self.world = world
        self.uav = uav
        self.hud_width = hud_width
        self.hud_height = hud_height
        self._motor_labels: List[Tuple[float, float, int]] = []
        self._noise_history: deque = deque(maxlen=300)

    def draw_scene(self, camera_eye: np.ndarray, camera_target: np.ndarray,
                   jammers: List[Jammer] = []) -> None:
        gluLookAt(
            camera_eye[0], camera_eye[1], camera_eye[2],
            camera_target[0], camera_target[1], camera_target[2],
            0.0, 0.0, 1.0,
        )
        self._draw_ground()
        motor_points = self._draw_uav(camera_eye)
        self._project_motor_labels(motor_points)
        for jammer in jammers:
            self._draw_jammer(jammer, camera_eye)

    def draw_hud(self, snapshot: HUDSnapshot, paused: bool = False,
                 jammers: List[Jammer] = [], uplink_noise: float = 0.0,
                 downlink_noise: float = 0.0) -> None:
        self._begin_hud_overlay()
        self._draw_motor_labels()
        self._draw_telemetry_readout(snapshot)
        self._draw_command_log(snapshot)
        self._draw_signal_panels(snapshot)
        if snapshot.has_telemetry and snapshot.motor_throttle is not None:
            self._draw_throttle_bars(snapshot.motor_throttle)
        self._draw_noise_timeline(uplink_noise, downlink_noise)
        self._draw_minimap(jammers)
        if paused:
            self._draw_pause_overlay(len(jammers))
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

    def _draw_uav(self, camera_eye: np.ndarray) -> List[np.ndarray]:
        center = self.uav.body.position
        motor_points = self.uav.motor_world_positions()
        num_segments = 24
        radius = 0.10

        # -- Camera-facing billboard circle (always faces the viewer) --
        view_dir = camera_eye - center
        view_dist = np.linalg.norm(view_dir)
        if view_dist < 1e-8:
            view_dir = np.array([0.0, 0.0, 1.0])
        else:
            view_dir /= view_dist
        ref_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(view_dir, ref_up)) > 0.99:
            ref_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, ref_up)
        right /= np.linalg.norm(right)
        up_perp = np.cross(right, view_dir)
        up_perp /= np.linalg.norm(up_perp)

        glColor3f(*UAV_BILLBOARD_CIRCLE_COLOR)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            theta = 2.0 * np.pi * i / num_segments
            p = center + radius * (np.cos(theta) * right + np.sin(theta) * up_perp)
            glVertex3f(p[0], p[1], p[2])
        glEnd()

        # -- Chassis-aligned horizontal circle (tilts with the UAV) --
        chassis_radius = 0.12
        glColor3f(*UAV_CHASSIS_CIRCLE_COLOR)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            theta = 2.0 * np.pi * i / num_segments
            local_p = np.array([chassis_radius * np.cos(theta), chassis_radius * np.sin(theta), 0.0])
            world_p = center + self.uav.body.orientation.apply(local_p)
            glVertex3f(world_p[0], world_p[1], world_p[2])
        glEnd()

        # -- Up arrow (UAV local +Z direction, shows which side is top) --
        arrow_len = 0.28
        arrow_head = 0.06
        up_dir = self.uav.body.orientation.apply(np.array([0.0, 0.0, 1.0]))
        tip = center + up_dir * arrow_len
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        # Dashed shaft
        dash_len = 0.03
        gap_len = 0.02
        total = 0.0
        while total + dash_len < arrow_len - arrow_head:
            a = center + up_dir * total
            b = center + up_dir * (total + dash_len)
            glVertex3f(a[0], a[1], a[2])
            glVertex3f(b[0], b[1], b[2])
            total += dash_len + gap_len
        # Arrowhead: two lines at the tip
        for sign in (-1, 1):
            head_local = np.array([sign * arrow_head, 0.0, -arrow_head])
            head_world = tip + self.uav.body.orientation.apply(head_local)
            glVertex3f(tip[0], tip[1], tip[2])
            glVertex3f(head_world[0], head_world[1], head_world[2])
        glEnd()

        # -- Motor circles (one per motor, tilt with the UAV) --
        motor_radius = 0.08
        for i, mpos in enumerate(motor_points):
            glColor3f(*UAV_MOTOR_COLORS[i])
            glBegin(GL_LINE_LOOP)
            for j in range(num_segments):
                theta = 2.0 * np.pi * j / num_segments
                local_p = np.array([motor_radius * np.cos(theta), motor_radius * np.sin(theta), 0.0])
                world_p = mpos + self.uav.body.orientation.apply(local_p)
                glVertex3f(world_p[0], world_p[1], world_p[2])
            glEnd()

        # -- Arms connecting center to motors --
        glColor3f(*UAV_ARM_COLOR)
        glBegin(GL_LINES)
        for point in motor_points:
            glVertex3f(center[0], center[1], center[2])
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        # -- Cross at the center --
        arm = 0.15
        glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_LINES)
        glVertex3f(center[0] - arm, center[1], center[2])
        glVertex3f(center[0] + arm, center[1], center[2])
        glVertex3f(center[0], center[1] - arm, center[2])
        glVertex3f(center[0], center[1] + arm, center[2])
        glEnd()

        # -- Center point --
        glPointSize(6.0)
        glColor3f(*UAV_CENTER_COLOR)
        glBegin(GL_POINTS)
        glVertex3f(center[0], center[1], center[2])
        glEnd()

        # -- Motor points --
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for i, point in enumerate(motor_points):
            glColor3f(*UAV_MOTOR_COLORS[i])
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        return motor_points

    def _draw_jammer(self, jammer: Jammer, camera_eye: np.ndarray) -> None:
        pos = jammer.position
        top = pos + np.array([0.0, 0.0, jammer.cylinder_height])
        radius = jammer.radius
        num_seg = 48

        # -- Ground circle (horizontal, dotted) --
        glColor3f(*JAMMER_CIRCLE_COLOR)
        glBegin(GL_LINES)
        for i in range(0, num_seg, 2):
            a = 2.0 * np.pi * i / num_seg
            b = 2.0 * np.pi * (i + 1) / num_seg
            glVertex3f(pos[0] + radius * np.cos(a), pos[1] + radius * np.sin(a), pos[2])
            glVertex3f(pos[0] + radius * np.cos(b), pos[1] + radius * np.sin(b), pos[2])
        glEnd()

        # -- Billboard circle (camera-facing, vertical, dotted) --
        view_dir = camera_eye - pos
        view_dist = np.linalg.norm(view_dir)
        if view_dist > 1e-8:
            view_dir /= view_dist
        ref_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(view_dir, ref_up)) > 0.99:
            ref_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, ref_up)
        right /= np.linalg.norm(right)
        up_perp = np.cross(right, view_dir)
        up_perp /= np.linalg.norm(up_perp)

        glColor3f(*JAMMER_CIRCLE_COLOR)
        glBegin(GL_LINES)
        for i in range(0, num_seg, 2):
            theta = 2.0 * np.pi * i / num_seg
            theta2 = 2.0 * np.pi * (i + 1) / num_seg
            p1 = pos + radius * (np.cos(theta) * right + np.sin(theta) * up_perp)
            p2 = pos + radius * (np.cos(theta2) * right + np.sin(theta2) * up_perp)
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
        glEnd()

        # -- Cylinder body --
        cyl_seg = 12
        cyl_r = jammer.cylinder_radius
        cyl_h = jammer.cylinder_height
        glColor3f(*JAMMER_BODY_COLOR)
        glBegin(GL_LINES)
        for i in range(cyl_seg):
            theta = 2.0 * np.pi * i / cyl_seg
            bx = cyl_r * np.cos(theta)
            by = cyl_r * np.sin(theta)
            # Vertical line
            glVertex3f(pos[0] + bx, pos[1] + by, pos[2])
            glVertex3f(pos[0] + bx, pos[1] + by, pos[2] + cyl_h)
            # Bottom circle segment
            b2 = 2.0 * np.pi * ((i + 1) % cyl_seg) / cyl_seg
            glVertex3f(pos[0] + cyl_r * np.cos(theta), pos[1] + cyl_r * np.sin(theta), pos[2])
            glVertex3f(pos[0] + cyl_r * np.cos(b2), pos[1] + cyl_r * np.sin(b2), pos[2])
            # Top circle segment
            glVertex3f(pos[0] + cyl_r * np.cos(theta), pos[1] + cyl_r * np.sin(theta), pos[2] + cyl_h)
            glVertex3f(pos[0] + cyl_r * np.cos(b2), pos[1] + cyl_r * np.sin(b2), pos[2] + cyl_h)
        glEnd()

        # -- Blinking dot on top --
        blink_on = (int(time.time() * 2) % 2) == 0
        if blink_on:
            glColor3f(*JAMMER_BLINK_COLOR_ON)
        else:
            glColor3f(*JAMMER_BLINK_COLOR_OFF)
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glVertex3f(top[0], top[1], top[2])
        glEnd()
        glPointSize(1.0)

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
            return ["BAT:--", "ALT:--", "MASS:--", "HP:--",
                    "POSX:--", "POSY:--",
                    "ROL:--", "PIT:--", "YAW:--"]

        x, y, z = snapshot.position
        roll, pitch, yaw = snapshot.attitude_deg
        return [
            f"BAT:{snapshot.battery_percent:.0f}%",
            f"ALT:{z:.1f}",
            f"MASS:{snapshot.mass_kg:.2f}",
            f"HP:{snapshot.health_percent:.0f}%" if snapshot.health_percent is not None else "HP:--",
            f"POSX:{x:.1f}",
            f"POSY:{y:.1f}",
            f"ROL:{roll:.1f}",
            f"PIT:{pitch:.1f}",
            f"YAW:{yaw:.1f}",
        ]

    # -- HUD: command log (SENT vs RCVD) ----------------------------------------
    def _draw_command_log(self, snapshot: HUDSnapshot) -> None:
        if not snapshot.sent_log and not snapshot.command_log:
            return
        margin = 14
        char_w, char_h, spacing = 6.0, 9.0, 1.5
        line_h = char_h + 4
        telemetry_height = 9 * (10 + 6)
        # Two columns: SENT left, RCVD right
        sent_x = margin
        rcvd_x = margin + 85

        sent_items = list(reversed(snapshot.sent_log))
        rcvd_items = list(reversed(snapshot.command_log))  # [(label, valid), ...]

        n = max(len(sent_items), len(rcvd_items))
        y = self.hud_height - margin - 10 - telemetry_height - 12 - n * line_h

        # Column headers
        glBegin(GL_LINES)
        col_header_y = y - line_h + 2
        glColor3f(0.5, 0.5, 0.5)
        draw_text("SENT", sent_x, col_header_y, char_w, char_h, spacing, self._segment_drawer())
        draw_text("RCVD", rcvd_x, col_header_y, char_w, char_h, spacing, self._segment_drawer())
        glEnd()

        glBegin(GL_LINES)
        for i in range(n):
            y += line_h
            sent_label = sent_items[i] if i < len(sent_items) else ""
            rcvd_label, rcvd_valid = rcvd_items[i] if i < len(rcvd_items) else ("", True)

            if sent_label:
                match = sent_label == rcvd_label and rcvd_valid
                s_color = HUD_LOG_OK_COLOR if match else HUD_LOG_SENT_COLOR
                glColor3f(*s_color)
                draw_text(sent_label, sent_x, y, char_w, char_h, spacing, self._segment_drawer())

            if rcvd_label:
                r_color = HUD_LOG_OK_COLOR if rcvd_valid else HUD_LOG_BAD_COLOR
                glColor3f(*r_color)
                draw_text(rcvd_label, rcvd_x, y, char_w, char_h, spacing, self._segment_drawer())
        glEnd()

    # -- HUD: TX/RX oscilloscope-style signal panels ----------------------------
    def _draw_signal_panels(self, snapshot: HUDSnapshot) -> None:
        panel_width, panel_height = 300.0, 85.0
        gap = 8.0
        x = self.hud_width - panel_width - 14.0
        tx_y = self.hud_height - panel_height - 14.0
        rx_y = tx_y - panel_height - gap

        self._draw_signal_panel(
            "TX", snapshot.uplink_signal, snapshot.uplink_raw,
            snapshot.uplink_bandwidth_bps, snapshot.now,
            x, tx_y, panel_width, panel_height,
        )
        self._draw_signal_panel(
            "RX", snapshot.downlink_signal, snapshot.downlink_raw,
            snapshot.downlink_bandwidth_bps, snapshot.now,
            x, rx_y, panel_width, panel_height,
        )

    def _draw_signal_panel(
        self,
        label: str,
        transmissions: Tuple[Tuple[float, bytes], ...],
        raw_transmissions: Tuple[Tuple[float, np.ndarray], ...],
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

        window = SIGNAL_WINDOW_SECONDS
        bit_dur = _SIGNAL_BIT_DURATION

        # -- Raw analog waveform (upper portion) --
        raw_low_y = y + height * 0.46
        raw_high_y = y + height * 0.90
        raw_mid_y = (raw_low_y + raw_high_y) / 2.0
        raw_amp = (raw_high_y - raw_low_y) / 2.0
        glColor3f(*HUD_RAW_SIGNAL_COLOR)
        glBegin(GL_LINE_STRIP)
        for ts, samples in raw_transmissions:
            t0 = ts - (now - window)
            byte0_end_raw = t0 + 8.0 * bit_dur
            if byte0_end_raw < 0.0:
                continue
            n = len(samples)
            for i in range(n):
                t = t0 + (i / n) * 8.0 * bit_dur
                if t < -bit_dur:
                    continue
                if t > window + bit_dur:
                    break
                t_clamped = max(0.0, min(t, window))
                px = x + (t_clamped / window) * width
                py = raw_mid_y + samples[i] * raw_amp
                glVertex2f(px, py)
        glEnd()

        # -- Demodulated digital bits (lower portion) --
        low_y = y + height * 0.06
        high_y = y + height * 0.38
        glColor3f(*HUD_WAVEFORM_COLOR)
        glBegin(GL_LINES)
        for seg in _waveform_segments(transmissions, now, window, label):
            t0, l0, t1, l1 = seg
            px0 = x + (t0 / window) * width
            py0 = high_y if l0 else low_y
            px1 = x + (t1 / window) * width
            py1 = high_y if l1 else low_y
            glVertex2f(px0, py0)
            glVertex2f(px1, py1)
        glEnd()

        # Y-axis level labels
        label_x = x + 2
        glColor3f(*HUD_TEXT_COLOR)
        glBegin(GL_LINES)
        draw_text("1", label_x, high_y - char_h - 1, 5.0, 8.0, 1.5, self._segment_drawer())
        draw_text("0", label_x, low_y - 1, 5.0, 8.0, 1.5, self._segment_drawer())
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

    def _draw_noise_timeline(self, uplink_noise: float, downlink_noise: float) -> None:
        self._noise_history.append((uplink_noise, downlink_noise))
        hist = list(self._noise_history)
        if len(hist) < 2:
            return

        panel_width = 300.0
        panel_height = 85.0
        gap = 8.0
        x = self.hud_width - panel_width - 14.0
        tx_y = self.hud_height - panel_height - 14.0
        rx_y = tx_y - panel_height - gap
        y = rx_y - NOISE_PANEL_HEIGHT - gap

        max_noise = max(max(u, d) for u, d in hist) or 1.0

        graph_left = x + 4
        graph_right = x + panel_width - 4
        graph_width = graph_right - graph_left
        graph_bottom = y + 4
        graph_top = y + NOISE_PANEL_HEIGHT - 12
        graph_height = graph_top - graph_bottom

        # Border
        glColor3f(*HUD_PANEL_BORDER_COLOR)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + panel_width, y)
        glVertex2f(x + panel_width, y + NOISE_PANEL_HEIGHT)
        glVertex2f(x, y + NOISE_PANEL_HEIGHT)
        glEnd()

        # Label + current values
        glColor3f(*HUD_TEXT_COLOR)
        glBegin(GL_LINES)
        draw_text(f"N {uplink_noise:.2f}/{downlink_noise:.2f}", x + 4,
                   y + NOISE_PANEL_HEIGHT - 9 - 3, 5.5, 8.0, 1.5,
                   self._segment_drawer())
        glEnd()

        # Uplink noise line (TX)
        glColor3f(*HUD_RAW_SIGNAL_COLOR)
        glBegin(GL_LINE_STRIP)
        for i, (u, _) in enumerate(hist):
            px = graph_left + (i / (len(hist) - 1)) * graph_width
            py = graph_bottom + (u / max_noise) * graph_height
            glVertex2f(px, py)
        glEnd()

        # Downlink noise line (RX)
        glColor3f(*HUD_WAVEFORM_COLOR)
        glBegin(GL_LINE_STRIP)
        for i, (_, d) in enumerate(hist):
            px = graph_left + (i / (len(hist) - 1)) * graph_width
            py = graph_bottom + (d / max_noise) * graph_height
            glVertex2f(px, py)
        glEnd()

    def _draw_minimap(self, jammers: List[Jammer]) -> None:
        uav_pos = self.uav.body.position
        rpy = self.uav.body.attitude_rpy()
        uav_yaw = rpy[2]

        map_size = MINIMAP_SIZE
        margin = MINIMAP_MARGIN
        extent = MINIMAP_EXTENT
        scale = map_size / (2.0 * extent)

        mx = self.hud_width - map_size - margin
        my = margin
        cx = mx + map_size / 2.0
        cy = my + map_size / 2.0

        # Enable scissor to clip content to minimap bounds
        glEnable(GL_SCISSOR_TEST)
        glScissor(mx, my, map_size, map_size)

        # Background fill
        glColor3f(*MINIMAP_BG_COLOR)
        glBegin(GL_LINES)
        step = 2
        for yy in range(int(my) + 1, int(my + map_size), step):
            glVertex2f(mx, yy)
            glVertex2f(mx + map_size, yy)
        glEnd()

        cos_yaw = np.cos(uav_yaw)
        sin_yaw = np.sin(uav_yaw)

        # Jammers
        for jammer in jammers:
            dx = jammer.position[0] - uav_pos[0]
            dy = jammer.position[1] - uav_pos[1]
            jx = cx + dx * scale
            jy = cy + dy * scale

            # Range circle
            r_px = jammer.radius * scale
            num_seg = 36
            glColor3f(*JAMMER_CIRCLE_COLOR)
            glBegin(GL_LINES)
            for i in range(0, num_seg, 2):
                a = 2.0 * np.pi * i / num_seg
                b = 2.0 * np.pi * (i + 1) / num_seg
                glVertex2f(jx + r_px * np.cos(a), jy + r_px * np.sin(a))
                glVertex2f(jx + r_px * np.cos(b), jy + r_px * np.sin(b))
            glEnd()

            # Jammer dot
            glColor3f(*JAMMER_BODY_COLOR)
            glPointSize(4.0)
            glBegin(GL_POINTS)
            glVertex2f(jx, jy)
            glEnd()

        glPointSize(1.0)

        # UAV icon: motor circles at fixed icon radius, rotated by yaw
        for i, motor in enumerate(self.uav.motors):
            mb = motor.position_body
            angle = np.arctan2(mb[1], mb[0])
            total_angle = angle + uav_yaw
            px = cx + MINIMAP_MOTOR_RADIUS_PX * np.cos(total_angle)
            py = cy + MINIMAP_MOTOR_RADIUS_PX * np.sin(total_angle)

            glColor3f(*UAV_MOTOR_COLORS[i])
            glBegin(GL_LINE_LOOP)
            for j in range(8):
                a = 2.0 * np.pi * j / 8
                glVertex2f(px + MINIMAP_MOTOR_DOT_RADIUS * np.cos(a),
                           py + MINIMAP_MOTOR_DOT_RADIUS * np.sin(a))
            glEnd()

        # X at center (body axes, rotated by yaw)
        arm = MINIMAP_X_ARM
        glColor3f(*MINIMAP_UAV_COLOR)
        glBegin(GL_LINES)
        glVertex2f(cx - arm * cos_yaw, cy - arm * sin_yaw)
        glVertex2f(cx + arm * cos_yaw, cy + arm * sin_yaw)
        glVertex2f(cx + arm * sin_yaw, cy - arm * cos_yaw)
        glVertex2f(cx - arm * sin_yaw, cy + arm * cos_yaw)
        glEnd()

        # Disable scissor, draw border on top (always complete)
        glDisable(GL_SCISSOR_TEST)
        glColor3f(*MINIMAP_BORDER_COLOR)
        glBegin(GL_LINE_LOOP)
        glVertex2f(mx, my)
        glVertex2f(mx + map_size, my)
        glVertex2f(mx + map_size, my + map_size)
        glVertex2f(mx, my + map_size)
        glEnd()

    def _draw_pause_overlay(self, jammer_count: int) -> None:
        cx = self.hud_width / 2.0
        cy = self.hud_height / 2.0
        cw, ch, sp = 10.0, 14.0, 3.0
        text = "PAUSED"
        glColor3f(1.0, 1.0, 0.5)
        glBegin(GL_LINES)
        draw_text(text, cx - len(text) * (cw + sp) / 2.0, cy - ch / 2.0, cw, ch, sp,
                   self._segment_drawer())
        glEnd()

    @staticmethod
    def _segment_drawer():
        def _draw(p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
            glVertex2f(p1[0], p1[1])
            glVertex2f(p2[0], p2[1])

        return _draw


def _waveform_segments(
    transmissions: Tuple[Tuple[float, bytes], ...],
    now: float,
    window_seconds: float,
    key: str = "",
) -> List[Tuple[float, int, float, int]]:
    """Turn transmissions into NRZ square-wave segments.

    Each returned tuple is (t_start, level_start, t_end, level_end)
    representing a single segment.  All segments are independent so they
    can be drawn with GL_LINES — no LINE_STRIP diagonals between unrelated
    points.

    Tracks the last signal level per *key* across calls via a function
    attribute so each panel (TX/RX) keeps its own state and the waveform
    doesn't jump to 0 at the left edge every frame.

    Processes ALL bytes of each payload, not just the first byte.
    After a gap of 3+ bit_durations the line returns to level 0 (idle).
    """
    window_start = now - window_seconds
    state: dict = getattr(_waveform_segments, '_state', {})
    segments: List[Tuple[float, int, float, int]] = []
    prev_t = 0.0
    prev_l = state.get(key, 0)
    idle_timeout = 3.0 * _SIGNAL_BIT_DURATION
    bit_dur = _SIGNAL_BIT_DURATION

    for timestamp, payload in transmissions:
        if not payload:
            continue
        t0 = timestamp - window_start
        all_bytes_end = t0 + len(payload) * 8 * bit_dur
        if all_bytes_end < 0.0:
            continue

        for byte_idx, byte in enumerate(payload):
            byte_start = t0 + byte_idx * 8 * bit_dur
            if byte_start > window_seconds:
                break
            byte_end = byte_start + 8 * bit_dur
            if byte_end < 0.0:
                continue

            # Gap from previous position → horizontal at current level,
            # clamped. If the gap exceeds idle_timeout, return to 0.
            gap_end = max(0.0, byte_start)
            if gap_end > prev_t:
                gap = gap_end - prev_t
                if gap > idle_timeout:
                    hold_end = prev_t + idle_timeout
                    segments.append((prev_t, prev_l, hold_end, prev_l))
                    segments.append((hold_end, prev_l, hold_end, 0))
                    if hold_end < gap_end:
                        segments.append((hold_end, 0, gap_end, 0))
                    prev_l = 0
                else:
                    segments.append((prev_t, prev_l, gap_end, prev_l))

            bits = [(byte >> i) & 1 for i in range(7, -1, -1)]
            for i, bit in enumerate(bits):
                t_start = byte_start + i * bit_dur
                t_end = t_start + bit_dur
                if t_start > window_seconds:
                    break
                if t_end < 0.0:
                    continue
                t_start_clamped = max(0.0, t_start)
                t_end_clamped = min(window_seconds, t_end)
                if t_start_clamped > t_end_clamped:
                    continue
                if bit != prev_l:
                    segments.append((t_start_clamped, prev_l, t_start_clamped, bit))
                segments.append((t_start_clamped, bit, t_end_clamped, bit))
                prev_l = bit
                prev_t = t_end_clamped

    # Fill remaining time to the end of the window.
    # Return to level 0 if the idle gap exceeds idle_timeout.
    if prev_t < window_seconds:
        remaining = window_seconds - prev_t
        if remaining > idle_timeout:
            hold_end = prev_t + idle_timeout
            segments.append((prev_t, prev_l, hold_end, prev_l))
            segments.append((hold_end, prev_l, hold_end, 0))
            segments.append((hold_end, 0, window_seconds, 0))
            prev_l = 0
        else:
            segments.append((prev_t, prev_l, window_seconds, prev_l))

    state[key] = prev_l
    _waveform_segments._state = state
    return segments
