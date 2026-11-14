"""Draws the flat world, the UAV, and the HUD (numeric read-outs + two
oscilloscope-style signal panels). Only `GL_POINTS` and `GL_LINES` are
used -- no textures, no meshes. Text is rendered via TTF texture atlas
(`ttf_font.py`).
"""
from __future__ import annotations

import time
from collections import deque
from typing import List, Tuple

import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_LINE_LOOP,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_MODELVIEW_MATRIX,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROJECTION,
    GL_PROJECTION_MATRIX,
    GL_SCISSOR_TEST,
    GL_SRC_ALPHA,
    GL_TRIANGLE_STRIP,
    GL_VIEWPORT,
    glBegin,
    glBlendFunc,
    glColor3f,
    glColor4f,
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
from uavsim.rendering.ttf_font import TTFFont
from uavsim.world.environment import WORLD_EXTENT_HALF, World

GROUND_COLOR = (0.2, 0.25, 0.3)
GROUND_MAJOR_COLOR = (0.3, 0.4, 0.5)
GROUND_AXIS_COLOR = (0.6, 0.3, 0.3)
UAV_CENTER_COLOR = (1.0, 0.0, 0.0)
UAV_BILLBOARD_CIRCLE_COLOR = (1.0, 1.0, 1.0)  # white, camera-facing circle
UAV_CHASSIS_CIRCLE_COLOR = (1.0, 1.0, 1.0)    # white, chassis-tilted circle
UAV_MOTOR_COLORS = [
    (1.0, 1.0, 0.0),   # 0 FR → yellow
    (0.0, 1.0, 1.0),   # 1 BR → cyan
    (1.0, 0.0, 1.0),   # 2 BL → magenta
    (0.5, 1.0, 0.0),   # 3 FL → lime green
]
UAV_ARM_COLOR = (0.6, 0.6, 0.6)

HUD_TEXT_COLOR = (0.75, 1.0, 0.8)
HUD_LOG_OK_COLOR = (0.3, 1.0, 0.4)     # green — received correctly
HUD_RAW_SIGNAL_COLOR = (0.2, 0.5, 0.8)  # dim blue, raw analog signal
HUD_PANEL_BORDER_COLOR = (0.4, 0.6, 0.5)
HUD_WAVEFORM_COLOR = (0.3, 1.0, 0.5)
HUD_BAR_COLOR = (0.9, 0.9, 0.9)

HOME_COLOR = (0.3, 0.9, 1.0)
SHADOW_COLOR = (0.35, 0.35, 0.4)
STICK_TX_COLOR = (0.3, 1.0, 0.4)        # what the pilot sends
STICK_RX_COLOR = (0.3, 0.8, 1.0)        # what the UAV is applying
BANNER_ALERT_COLOR = (1.0, 0.25, 0.2)
BANNER_WARN_COLOR = (1.0, 0.75, 0.1)
BANNER_INFO_COLOR = (0.6, 0.6, 0.6)

MODE_NAMES = {0: "ACRO", 1: "ANGLE", 2: "ALT", 3: "POS"}
FAILSAFE_ACTION_NAMES = {0: "HOLD", 1: "LAND", 2: "RTH", 3: "DROP"}

JAMMER_BODY_COLOR = (1.0, 0.3, 0.1)     # orange-red
JAMMER_CIRCLE_COLOR = (1.0, 0.1, 0.1)   # red for radius circles
JAMMER_BLINK_COLOR_ON = (1.0, 0.0, 0.0)  # red on
JAMMER_BLINK_COLOR_OFF = (0.2, 0.0, 0.0) # red dim

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
        self._font = TTFFont("assets/fonts/7-segment.ttf", size=14)
        self._pause_font = TTFFont("assets/fonts/7-segment.ttf", size=28)

    def draw_scene(self, camera_eye: np.ndarray, camera_target: np.ndarray,
                   jammers: List[Jammer] = []) -> None:
        gluLookAt(
            camera_eye[0], camera_eye[1], camera_eye[2],
            camera_target[0], camera_target[1], camera_target[2],
            0.0, 0.0, 1.0,
        )
        self._draw_ground()
        self._draw_home_marker()
        self._draw_ground_shadow()
        motor_points = self._draw_uav(camera_eye)
        self._project_motor_labels(motor_points)
        for jammer in jammers:
            self._draw_jammer(jammer, camera_eye)

    def draw_hud(self, snapshot: HUDSnapshot, paused: bool = False,
                 jammers: List[Jammer] = [], uplink_noise: float = 0.0,
                 downlink_noise: float = 0.0, dssc_N: int = 0,
                 link_backend: str = "", gamepad: str = "") -> None:
        self._begin_hud_overlay()
        self._draw_motor_labels()
        self._draw_telemetry_readout(snapshot)
        self._draw_dssc_indicator(dssc_N,
                                   uplink_corr=snapshot.uplink_correlation,
                                   downlink_corr=snapshot.downlink_correlation)
        self._draw_flight_status(snapshot, link_backend, gamepad)
        self._draw_banner(snapshot)
        self._draw_signal_panels(snapshot)
        if snapshot.has_telemetry and snapshot.motor_throttle is not None:
            self._draw_throttle_bars(snapshot.motor_throttle, snapshot.now)
        self._draw_noise_timeline(uplink_noise, downlink_noise)
        if dssc_N > 0:
            ds_x = self.hud_width - MINIMAP_SIZE - MINIMAP_MARGIN
            ds_y = MINIMAP_MARGIN + MINIMAP_SIZE + 6
            self._font.draw(f"DSSS:{dssc_N}", ds_x, ds_y, colour=(0.3, 0.8, 1.0))
        self._draw_minimap(jammers)
        if paused:
            self._draw_pause_overlay(len(jammers))
        if snapshot.worker_error:
            self._draw_worker_error(snapshot.worker_error)
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

    def _draw_home_marker(self) -> None:
        """'H' on the ground where the UAV was armed (RTH destination)."""
        hx, hy = float(self.uav.home[0]), float(self.uav.home[1])
        z = self.world.ground.ground_z + 0.01
        s = 0.6
        glColor3f(*HOME_COLOR)
        glBegin(GL_LINES)
        for x0, y0, x1, y1 in ((-s, -s, -s, s), (s, -s, s, s), (-s, 0.0, s, 0.0)):
            glVertex3f(hx + x0, hy + y0, z)
            glVertex3f(hx + x1, hy + y1, z)
        glEnd()
        glBegin(GL_LINE_LOOP)
        for i in range(24):
            a = 2.0 * np.pi * i / 24
            glVertex3f(hx + 1.2 * np.cos(a), hy + 1.2 * np.sin(a), z)
        glEnd()

    def _draw_ground_shadow(self) -> None:
        """Drop line + footprint under the UAV: the main depth cue for a pilot."""
        c = self.uav.body.position
        z = self.world.ground.ground_z + 0.01
        if c[2] - z < 0.05:
            return
        glColor3f(*SHADOW_COLOR)
        glBegin(GL_LINES)
        dash, gap = 0.25, 0.25
        h = z
        while h < c[2]:
            glVertex3f(c[0], c[1], h)
            glVertex3f(c[0], c[1], min(h + dash, c[2]))
            h += dash + gap
        for p in self.uav.motor_world_positions():
            glVertex3f(c[0], c[1], z)
            glVertex3f(p[0], p[1], z)
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
        leader_h = 14.0
        glBegin(GL_LINES)
        for sx, sy, idx in self._motor_labels:
            glColor3f(*UAV_MOTOR_COLORS[idx])
            glVertex2f(sx, sy + leader_h)
            glVertex2f(sx, sy)
        glEnd()
        for sx, sy, idx in self._motor_labels:
            self._font.draw(str(idx + 1), sx - 4, sy + leader_h + 2,
                            colour=UAV_MOTOR_COLORS[idx])

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
        line_gap = 6
        line_height = self._font._size + line_gap

        lines = self._telemetry_lines(snapshot)
        # Frozen telemetry (downlink jammed) is greyed out: those are old values.
        colour = BANNER_INFO_COLOR if snapshot.telemetry_stale else HUD_TEXT_COLOR

        y = self.hud_height - margin - self._font._size
        for line in lines:
            self._font.draw(line, margin, y, colour=colour)
            y -= line_height

    @staticmethod
    def _telemetry_lines(snapshot: HUDSnapshot) -> List[str]:
        if not snapshot.has_telemetry:
            return ["BAT:--", "ALT:--", "VZ:--", "SPD:--", "HP:--",
                    "POSX:--", "POSY:--", "ROL:--", "PIT:--", "YAW:--"]
        x, y, z = snapshot.position
        vx, vy, vz = snapshot.velocity
        roll, pitch, yaw = snapshot.attitude_deg
        return [
            f"BAT:{snapshot.battery_percent:.0f}%",
            f"ALT:{z:.1f}",
            f"VZ:{vz:+.1f}",
            f"SPD:{np.hypot(vx, vy):.1f}",
            f"HP:{snapshot.health_percent:.0f}%",
            f"POSX:{x:.1f}",
            f"POSY:{y:.1f}",
            f"ROL:{roll:.1f}",
            f"PIT:{pitch:.1f}",
            f"YAW:{yaw:.1f}",
        ]

    def _draw_dssc_indicator(self, dssc_N: int, uplink_corr: float = 0.0,
                             downlink_corr: float = 0.0) -> None:
        if dssc_N <= 0:
            return
        panel_width = 300.0
        gap = 8.0
        x = self.hud_width - panel_width - 14.0
        tx_y = self.hud_height - 85.0 - 14.0
        rx_y = tx_y - 85.0 - gap
        y = rx_y - 55.0 - gap - self._font._size - 4 - 14

        bar_y = y
        bar_h = 10
        bar_w = panel_width - 8

        glBegin(GL_LINES)
        # border
        glColor3f(*HUD_PANEL_BORDER_COLOR)
        glVertex2f(x + 4, bar_y)
        glVertex2f(x + 4 + bar_w, bar_y)
        glVertex2f(x + 4, bar_y + bar_h)
        glVertex2f(x + 4 + bar_w, bar_y + bar_h)
        glVertex2f(x + 4, bar_y)
        glVertex2f(x + 4, bar_y + bar_h)
        glVertex2f(x + 4 + bar_w, bar_y)
        glVertex2f(x + 4 + bar_w, bar_y + bar_h)

        # uplink fill (left half)
        ul_fill = int(bar_w * uplink_corr)
        ul_color = (0.3 + 0.7 * (1.0 - uplink_corr),
                    0.3 + 0.7 * uplink_corr,
                    0.0)
        glColor3f(*ul_color)
        glVertex2f(x + 5, bar_y + 1)
        glVertex2f(x + 5 + ul_fill, bar_y + 1)
        glVertex2f(x + 5, bar_y + 1)
        glVertex2f(x + 5, bar_y + bar_h - 1)
        glVertex2f(x + 5 + ul_fill, bar_y + 1)
        glVertex2f(x + 5 + ul_fill, bar_y + bar_h - 1)
        glVertex2f(x + 5, bar_y + bar_h - 1)
        glVertex2f(x + 5 + ul_fill, bar_y + bar_h - 1)

        # downlink fill (right half)
        dl_fill = int(bar_w * downlink_corr)
        dl_color = (0.3 + 0.7 * (1.0 - downlink_corr),
                    0.3 + 0.7 * downlink_corr,
                    0.0)
        bar_mid = x + 4 + bar_w // 2
        glColor3f(*dl_color)
        glVertex2f(bar_mid, bar_y + 1)
        glVertex2f(bar_mid + dl_fill // 2, bar_y + 1)
        glVertex2f(bar_mid, bar_y + 1)
        glVertex2f(bar_mid, bar_y + bar_h - 1)
        glVertex2f(bar_mid + dl_fill // 2, bar_y + 1)
        glVertex2f(bar_mid + dl_fill // 2, bar_y + bar_h - 1)
        glVertex2f(bar_mid, bar_y + bar_h - 1)
        glVertex2f(bar_mid + dl_fill // 2, bar_y + bar_h - 1)

        # divider line between uplink/downlink halves
        glColor3f(*HUD_PANEL_BORDER_COLOR)
        glVertex2f(bar_mid, bar_y)
        glVertex2f(bar_mid, bar_y + bar_h)
        glEnd()

        # labels
        label_y = bar_y - 18
        self._font.draw(f"UL {uplink_corr:.2f}", x + 5, label_y,
                        colour=HUD_TEXT_COLOR)
        self._font.draw(f"DL {downlink_corr:.2f}",
                        bar_mid + 2, label_y,
                        colour=HUD_TEXT_COLOR)


    # -- HUD: flight status, link quality and stick monitor ---------------------
    def _draw_flight_status(self, snapshot: HUDSnapshot, link_backend: str, gamepad: str) -> None:
        margin = 14
        line_h = self._font._size + 6
        y = self.hud_height - margin - self._font._size - 10 * line_h - 10

        mode = MODE_NAMES.get(snapshot.tx_mode, "?")
        fs = FAILSAFE_ACTION_NAMES.get(snapshot.tx_failsafe_action, "?")
        arm_text = "ARM:ON" if snapshot.tx_arm_switch else "ARM:OFF"
        arm_colour = BANNER_WARN_COLOR if snapshot.tx_arm_switch else HUD_TEXT_COLOR
        if snapshot.uplink_lq is None or snapshot.telemetry_stale:
            ul = "--"
        else:
            ul = f"{snapshot.uplink_lq:.0f}"
        link_name = {"gnuradio": "GNURADIO", "loopback": "NUMPY"}.get(
            link_backend, link_backend.upper())

        rows = [
            (f"MODE:{mode}", HUD_TEXT_COLOR),
            (arm_text, arm_colour),
            (f"FS:{fs}", HUD_TEXT_COLOR),
            (f"LQ UP:{ul} DN:{snapshot.downlink_lq:.0f}", self._lq_colour(snapshot)),
            (f"LINK:{link_name}" + (" PAD" if gamepad else ""), BANNER_INFO_COLOR),
        ]
        for text, colour in rows:
            self._font.draw(text, margin, y, colour=colour)
            y -= line_h

        box = 46.0
        gap = 8.0
        y -= box - self._font._size + 4
        self._draw_stick_pair("TX", snapshot.tx_sticks, margin, y, box, gap, STICK_TX_COLOR)
        rx = None
        if snapshot.has_telemetry and not snapshot.telemetry_stale:
            rx = snapshot.rx_sticks
        self._draw_stick_pair("RX", rx, margin, y - box - 22, box, gap, STICK_RX_COLOR)

    @staticmethod
    def _lq_colour(snapshot: HUDSnapshot) -> Tuple[float, float, float]:
        values = [snapshot.downlink_lq]
        if snapshot.uplink_lq is not None and not snapshot.telemetry_stale:
            values.append(snapshot.uplink_lq)
        worst = min(values)
        if worst >= 80:
            return HUD_LOG_OK_COLOR
        if worst >= 40:
            return BANNER_WARN_COLOR
        return BANNER_ALERT_COLOR

    def _draw_stick_pair(self, label: str, sticks, x: float, y: float, box: float,
                         gap: float, colour: Tuple[float, float, float]) -> None:
        """Two stick gimbals: left = yaw/throttle, right = roll/pitch (Mode 2)."""
        left_x = x + 24
        right_x = left_x + box + gap
        self._font.draw(label, x, y + box / 2.0 - self._font._size / 2.0, colour=colour)
        for bx in (left_x, right_x):
            glColor3f(*HUD_PANEL_BORDER_COLOR)
            glBegin(GL_LINE_LOOP)
            glVertex2f(bx, y)
            glVertex2f(bx + box, y)
            glVertex2f(bx + box, y + box)
            glVertex2f(bx, y + box)
            glEnd()
            glBegin(GL_LINES)
            glVertex2f(bx + box / 2.0, y + 3)
            glVertex2f(bx + box / 2.0, y + box - 3)
            glVertex2f(bx + 3, y + box / 2.0)
            glVertex2f(bx + box - 3, y + box / 2.0)
            glEnd()
        if sticks is None:
            self._font.draw("--", right_x + box + 6, y + box / 2.0 - self._font._size / 2.0,
                            colour=BANNER_INFO_COLOR)
            return
        throttle, roll, pitch, yaw = sticks
        half = box / 2.0 - 4.0
        points = (
            (left_x + box / 2.0 + yaw * half, y + 4.0 + throttle * (box - 8.0)),
            (right_x + box / 2.0 + roll * half, y + box / 2.0 + pitch * half),
        )
        glPointSize(7.0)
        glColor3f(*colour)
        glBegin(GL_POINTS)
        for px, py in points:
            glVertex2f(px, py)
        glEnd()
        glPointSize(1.0)

    def _draw_banner(self, snapshot: HUDSnapshot) -> None:
        """Top-centre alert: what a real OSD would flash at the pilot."""
        blink = int(snapshot.now * 3) % 2 == 0
        text, colour, font = "", HUD_TEXT_COLOR, self._pause_font
        if not snapshot.tx_enabled:
            text, colour = "TX OFF", BANNER_WARN_COLOR
        elif snapshot.has_telemetry and snapshot.telemetry_stale:
            text, colour = "TLM LOST", BANNER_WARN_COLOR
        elif snapshot.has_telemetry and snapshot.armed and snapshot.failsafe_stage >= 3:
            action = FAILSAFE_ACTION_NAMES.get(snapshot.failsafe_action, "?")
            text, colour = f"FAILSAFE {action}", BANNER_ALERT_COLOR
        elif snapshot.has_telemetry and snapshot.armed and snapshot.failsafe_stage >= 1:
            text, colour = "RX LOSS", BANNER_ALERT_COLOR
        elif snapshot.has_telemetry and not snapshot.armed:
            text, colour, font = "DISARMED  SPACE TO ARM", BANNER_INFO_COLOR, self._font
            blink = True
        if not text or not blink:
            return
        tw = font.text_width(text)
        font.draw(text, self.hud_width / 2.0 - tw / 2.0, self.hud_height - 60, colour=colour)

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

        label_text = f"{label} {bandwidth_bps:.0f}B/S"
        self._font.draw(label_text, x + 4, y + height - self._font._size - 3,
                        colour=HUD_TEXT_COLOR)

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
        self._font.draw("1", label_x, high_y - self._font._size - 1, colour=HUD_TEXT_COLOR)
        self._font.draw("0", label_x, low_y - 1, colour=HUD_TEXT_COLOR)

    # -- HUD: per-motor throttle bars --------------------------------------------
    def _draw_throttle_bars(self, throttle: Tuple[float, float, float, float],
                            now: float) -> None:
        bar_width, gap, max_height, margin = 20.0, 12.0, 110.0, 16.0
        base_y = margin + 14.0

        glBegin(GL_LINES)
        for i, value in enumerate(throttle):
            x = margin + i * (bar_width + gap)
            top = base_y + max_height * value
            glColor3f(*UAV_MOTOR_COLORS[i])
            glVertex2f(x, base_y)
            glVertex2f(x, top)
            glVertex2f(x + bar_width, base_y)
            glVertex2f(x + bar_width, top)
            glVertex2f(x, base_y)
            glVertex2f(x + bar_width, base_y)
        glEnd()

        # Max-throttle indicator line + blinking MAX label
        glBegin(GL_LINES)
        for i, value in enumerate(throttle):
            x = margin + i * (bar_width + gap)
            # Horizontal line at full throttle
            max_y = base_y + max_height
            glColor3f(*UAV_MOTOR_COLORS[i])
            glVertex2f(x + 2, max_y)
            glVertex2f(x + bar_width - 2, max_y)
        glEnd()

        for i, value in enumerate(throttle):
            if value >= 1.0 and int(now * 2) % 2 == 0:
                x = margin + i * (bar_width + gap)
                max_y = base_y + max_height
                tw = self._font.text_width("MAX")
                self._font.draw("MAX", x + (bar_width - tw) / 2.0, max_y + 3,
                                colour=(1.0, 0.8, 0.0))

        for i in range(len(throttle)):
            x = margin + i * (bar_width + gap)
            self._font.draw(str(i + 1), x + bar_width * 0.3, margin - 2,
                            colour=HUD_TEXT_COLOR)

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
        self._font.draw(f"N {uplink_noise:.2f}/{downlink_noise:.2f}", x + 4,
                        y + NOISE_PANEL_HEIGHT - self._font._size - 3,
                        colour=HUD_TEXT_COLOR)

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

        # Home point
        hx = cx + (self.uav.home[0] - uav_pos[0]) * scale
        hy = cy + (self.uav.home[1] - uav_pos[1]) * scale
        glColor3f(*HOME_COLOR)
        glBegin(GL_LINE_LOOP)
        for j in range(4):
            a = np.pi / 4 + j * np.pi / 2
            glVertex2f(hx + 4.0 * np.cos(a), hy + 4.0 * np.sin(a))
        glEnd()

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

        # X at center with arms pointing to the 4 motor diagonals
        arm = MINIMAP_X_ARM
        c45 = np.cos(uav_yaw + np.pi / 4)
        s45 = np.sin(uav_yaw + np.pi / 4)
        c_m45 = np.cos(uav_yaw - np.pi / 4)
        s_m45 = np.sin(uav_yaw - np.pi / 4)
        glColor3f(*MINIMAP_UAV_COLOR)
        glBegin(GL_LINES)
        glVertex2f(cx - arm * c45, cy - arm * s45)
        glVertex2f(cx + arm * c45, cy + arm * s45)
        glVertex2f(cx - arm * c_m45, cy - arm * s_m45)
        glVertex2f(cx + arm * c_m45, cy + arm * s_m45)
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

    def _draw_worker_error(self, error: str) -> None:
        lines = error.split("\n")
        line_h = self._font._size + 4
        margin = 8
        x = margin
        y = margin + line_h * len(lines)
        for line in reversed(lines):
            self._font.draw(line, x, y, colour=(1.0, 0.3, 0.3))
            y -= line_h

    def _draw_pause_overlay(self, jammer_count: int) -> None:
        cx = self.hud_width / 2.0
        cy = self.hud_height / 2.0
        text = "PAUSED"
        f = self._pause_font
        tw = f.text_width(text)
        pad = 14
        bx = cx - tw / 2.0 - pad
        by = cy - f._size / 2.0 - pad
        bw = tw + pad * 2
        bh = f._size + pad * 2

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_TRIANGLE_STRIP)
        glVertex2f(bx, by)
        glVertex2f(bx + bw, by)
        glVertex2f(bx, by + bh)
        glVertex2f(bx + bw, by + bh)
        glEnd()
        glDisable(GL_BLEND)

        f.draw(text, cx - tw / 2.0, cy - f._size / 2.0, colour=(1.0, 1.0, 0.5))

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
