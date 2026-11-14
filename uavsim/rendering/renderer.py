"""Draws the flat world, the UAV, and the HUD (numeric read-outs + two
oscilloscope-style signal panels). Only `GL_POINTS` and `GL_LINES` are
used -- no textures, no meshes. Text is rendered via TTF texture atlas
(`ttf_font.py`).
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_ARRAY,
    GL_FALSE,
    GL_FLOAT,
    GL_QUADS,
    GL_TRUE,
    GL_VERTEX_ARRAY,
    glColorPointer,
    glDepthMask,
    glDisableClientState,
    glDrawArrays,
    glEnableClientState,
    glVertexPointer,
    GL_DEPTH_TEST,
    GL_FOG,
    GL_LIGHTING,
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

from uavsim.entities.jammer import Jammer, JammerKind
from uavsim.entities.uav import UAV
from uavsim.hud.hud import SIGNAL_WINDOW_SECONDS, HUDSnapshot
from uavsim.rendering.scene3d import (
    TerrainMesh,
    UavModel,
    draw_shadow,
    draw_sky,
    draw_tower,
    enable_fog,
    set_sun,
)
from uavsim.rendering.ttf_font import TTFFont
from uavsim.world.environment import WORLD_EXTENT_HALF, World
from uavsim.world.rf import LinkBudget, RFEnvironment

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

JAMMER_KIND_COLORS = {
    JammerKind.BARRAGE: (1.0, 0.2, 0.1),   # red: always on
    JammerKind.PULSED: (1.0, 0.3, 0.85),   # magenta: on/off bursts
    JammerKind.SWEPT: (1.0, 0.65, 0.1),    # orange: sweeps through our channel
    JammerKind.TONE: (0.7, 1.0, 0.2),      # lime: narrowband carrier
}
GROUND_STATION_COLOR = (0.85, 0.95, 1.0)
HEATMAP_CELL = 5.0            # m
HEATMAP_REFRESH = 0.5         # s of sim time
# Thresholds from the calibrated link model (loopback_link): 11-byte RC
# frames arrive ~90% of the time at 11 dB and ~10% at 7 dB.
HEATMAP_GOOD_SINR = 14.0      # dB: comfortable margin -> map transparent
HEATMAP_MARGINAL_SINR = 11.0  # dB: below this RC frames start getting lost
HEATMAP_DEAD_SINR = 7.0       # dB: below this the control link is effectively gone

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
        self._sinr_history: deque = deque(maxlen=300)
        self._heatmap_key = None
        self._heatmap_time = -1e9
        self._heatmap_verts = np.zeros((0, 3), dtype=np.float32)
        self._heatmap_cols = np.zeros((0, 4), dtype=np.float32)
        self._heatmap_cells = None  # (centres_xy, rgba) for the minimap
        self._heatmap_visible = False
        self._margin = 0.0
        self._terrain_mesh = TerrainMesh(world.ground, world.extent_half)
        self._uav_model = UavModel(uav)
        self._last_sim_time = None
        self._font = TTFFont("assets/fonts/7-segment.ttf", size=14)
        self._pause_font = TTFFont("assets/fonts/7-segment.ttf", size=28)

    def draw_scene(self, camera_eye: np.ndarray, camera_target: np.ndarray,
                   jammers: List[Jammer] = (), sim_time: float = 0.0,
                   budget: LinkBudget = None, heatmap_rf: RFEnvironment = None,
                   link_margin_db: float = 0.0) -> None:
        """`link_margin_db`: how much lower SINR the active mitigation tolerates
        (measured by uavsim.experiments.link_curves); shifts every colour
        threshold so the map shows where *this* radio loses the link."""
        self._margin = link_margin_db
        dt = 0.0 if self._last_sim_time is None else min(max(sim_time - self._last_sim_time, 0.0), 0.1)
        self._last_sim_time = sim_time

        draw_sky()
        gluLookAt(
            camera_eye[0], camera_eye[1], camera_eye[2],
            camera_target[0], camera_target[1], camera_target[2],
            0.0, 0.0, 1.0,
        )
        set_sun()
        enable_fog()
        self._terrain_mesh.draw()
        self._heatmap_visible = heatmap_rf is not None
        if heatmap_rf is not None:
            self._draw_heatmap(heatmap_rf, jammers, sim_time)
        self._draw_home_marker()
        self._draw_ground_station()
        if budget is not None:
            self._draw_link_line(budget)
        self._draw_ground_shadow()
        motor_points = self._draw_uav(dt)
        self._project_motor_labels(motor_points)
        for jammer in jammers:
            self._draw_jammer(jammer, sim_time)

    def _draw_hud_backdrops(self) -> None:
        """Translucent dark panels so HUD text stays readable over a bright sky."""
        w, h = self.hud_width, self.hud_height
        rects = (
            (6, h - 452, 186, h - 6),                    # flight data, status, sticks
            (w - 320, h - 312, w - 6, h - 6),            # scopes, SINR, DSSS bar
            (6, 6, 150, 168),                            # motor bars
            (w - MINIMAP_SIZE - MINIMAP_MARGIN - 4, MINIMAP_MARGIN + MINIMAP_SIZE + 2,
             w - MINIMAP_MARGIN + 4, MINIMAP_MARGIN + MINIMAP_SIZE + 46),  # minimap labels
        )
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.02, 0.03, 0.05, 0.55)
        for x0, y0, x1, y1 in rects:
            glBegin(GL_TRIANGLE_STRIP)
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x0, y1)
            glVertex2f(x1, y1)
            glEnd()
        glDisable(GL_BLEND)

    def draw_hud(self, snapshot: HUDSnapshot, paused: bool = False,
                 jammers: List[Jammer] = (), budget: LinkBudget = None,
                 dssc_N: int = 0, link_backend: str = "", gamepad: str = "",
                 phy_label: str = "") -> None:
        self._begin_hud_overlay()
        self._draw_hud_backdrops()
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
        self._draw_sinr_timeline(budget)
        label_x = self.hud_width - MINIMAP_SIZE - MINIMAP_MARGIN
        label_y = MINIMAP_MARGIN + MINIMAP_SIZE + 6
        if phy_label:
            colour = (0.3, 0.8, 1.0) if phy_label != "NONE" else BANNER_INFO_COLOR
            self._font.draw(f"PHY:{phy_label}", label_x, label_y, colour=colour)
        if self._heatmap_visible:
            self._font.draw("MAP:WORST UP", label_x, label_y + self._font._size + 6,
                            colour=(1.0, 0.75, 0.1))
        self._draw_minimap(jammers)
        if paused:
            self._draw_pause_overlay(len(jammers))
        if snapshot.worker_error:
            self._draw_worker_error(snapshot.worker_error)
        self._end_hud_overlay()

    # -- world ---------------------------------------------------------------
    def _draw_heatmap(self, rf: RFEnvironment, jammers: List[Jammer], t: float) -> None:
        """Uplink SINR a UAV would get at its current altitude, painted on the
        ground (worst case: every jammer on). Transparent where the link is
        healthy, yellow -> red where it degrades / dies."""
        altitude = max(float(self.uav.body.position[2]), 2.0)
        key = (round(altitude), tuple(id(j) for j in jammers), round(self._margin, 1))
        if key != self._heatmap_key or t - self._heatmap_time > HEATMAP_REFRESH:
            self._rebuild_heatmap(rf, jammers, altitude)
            self._heatmap_key = key
            self._heatmap_time = t
        n = len(self._heatmap_verts)
        if n == 0:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self._heatmap_verts)
        glColorPointer(4, GL_FLOAT, 0, self._heatmap_cols)
        glDrawArrays(GL_QUADS, 0, n)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def _rebuild_heatmap(self, rf: RFEnvironment, jammers: List[Jammer], altitude: float) -> None:
        half = self.world.extent_half
        c = HEATMAP_CELL
        xs = np.arange(-half + c / 2.0, half, c)
        ys = np.arange(-half + c / 2.0, half, c)
        sinr = rf.uplink_sinr_grid(xs, ys, altitude, self.world.ground_station, jammers)
        gx, gy = np.meshgrid(xs, ys)
        good = HEATMAP_GOOD_SINR - self._margin
        dead = HEATMAP_DEAD_SINR - self._margin
        mask = sinr < good
        sx, sy, sv = gx[mask], gy[mask], sinr[mask]
        severity = np.clip((good - sv) / (good - dead), 0, 1)
        yellow = np.array([1.0, 0.85, 0.1])
        red = np.array([1.0, 0.1, 0.05])
        rgb = yellow + (red - yellow) * severity[:, None]
        alpha = 0.05 + 0.25 * severity
        rgba = np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32)
        self._heatmap_cells = (np.stack([sx, sy], axis=1), rgba)

        h = c / 2.0
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        verts = np.empty((len(sx), 4, 3), dtype=np.float32)
        verts[:, :, 0] = sx[:, None] + corners[None, :, 0]
        verts[:, :, 1] = sy[:, None] + corners[None, :, 1]
        verts[:, :, 2] = self.world.ground.height_at(verts[:, :, 0], verts[:, :, 1]) + 0.12
        self._heatmap_verts = np.ascontiguousarray(verts.reshape(-1, 3))
        self._heatmap_cols = np.ascontiguousarray(np.repeat(rgba, 4, axis=0))

    def _draw_ground_station(self) -> None:
        """Pilot's ground station: tripod + antenna."""
        gs = self.world.ground_station
        z0 = float(self.world.ground.height_at(gs[0], gs[1]))
        top = np.array([gs[0], gs[1], gs[2] - 0.3])
        glColor3f(*GROUND_STATION_COLOR)
        glBegin(GL_LINES)
        for a in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
            glVertex3f(gs[0] + 0.5 * np.cos(a), gs[1] + 0.5 * np.sin(a), z0)
            glVertex3f(*top)
        glVertex3f(*top)
        glVertex3f(gs[0], gs[1], gs[2] + 0.25)
        glEnd()
        glPointSize(6.0)
        glBegin(GL_POINTS)
        glVertex3f(gs[0], gs[1], gs[2] + 0.25)
        glEnd()
        glPointSize(1.0)

    def _draw_link_line(self, budget: LinkBudget) -> None:
        """Ground station -> UAV, coloured by uplink SINR; dashed when it is dead."""
        a = np.asarray(self.world.ground_station, dtype=float)
        b = self.uav.body.position
        sinr = budget.uplink_sinr_db
        glColor3f(*_sinr_colour(sinr + self._margin))
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            return
        direction = (b - a) / length
        glBegin(GL_LINES)
        if sinr + self._margin >= HEATMAP_DEAD_SINR:
            glVertex3f(*a)
            glVertex3f(*b)
        else:
            d = 0.0
            while d < length:
                p0 = a + direction * d
                p1 = a + direction * min(d + 1.0, length)
                glVertex3f(*p0)
                glVertex3f(*p1)
                d += 2.0
        glEnd()

    def _draw_home_marker(self) -> None:
        """'H' on the ground where the UAV was armed (RTH destination)."""
        hx, hy = float(self.uav.home[0]), float(self.uav.home[1])
        z = float(self.world.ground.height_at(hx, hy)) + 0.05
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
        """Shadow blob + dashed drop line: the main depth cues for a pilot."""
        c = self.uav.body.position
        z = float(self.world.ground.height_at(c[0], c[1]))
        height = float(c[2]) - z
        draw_shadow(float(c[0]), float(c[1]), z, height)
        if height < 0.3:
            return
        glColor3f(*SHADOW_COLOR)
        glBegin(GL_LINES)
        dash, gap = 0.25, 0.25
        h = z + 0.05
        while h < c[2]:
            glVertex3f(c[0], c[1], h)
            glVertex3f(c[0], c[1], min(h + dash, c[2]))
            h += dash + gap
        glEnd()

    def _draw_uav(self, dt: float) -> List[np.ndarray]:
        self._uav_model.draw(dt, UAV_MOTOR_COLORS)
        return self.uav.motor_world_positions()

    def _draw_jammer(self, jammer: Jammer, t: float) -> None:
        """Lattice tower, antenna light (lit while transmitting) and a ring
        that pulses outward while the jammer is on."""
        top = jammer.antenna_position
        colour = JAMMER_KIND_COLORS.get(jammer.kind, JAMMER_BODY_COLOR)
        on = jammer.is_on(t)
        draw_tower(jammer.position, jammer.antenna_height, colour)

        glPointSize(9.0)
        glColor3f(*(colour if on else tuple(0.25 * c for c in colour)))
        glBegin(GL_POINTS)
        glVertex3f(*top)
        glEnd()
        glPointSize(1.0)

        if on:
            phase = (t * 1.5) % 1.0
            radius = 1.0 + 12.0 * phase
            fade = 1.0 - phase
            glColor3f(*(fade * c + (1.0 - fade) * 0.6 for c in colour))
            glBegin(GL_LINE_LOOP)
            for i in range(32):
                a = 2.0 * np.pi * i / 32
                glVertex3f(top[0] + radius * np.cos(a), top[1] + radius * np.sin(a), top[2])
            glEnd()

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
        glDisable(GL_FOG)
        glDisable(GL_LIGHTING)

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

    def _draw_sinr_timeline(self, budget: LinkBudget) -> None:
        """Uplink / downlink SINR history (simulation instrument, not OSD)."""
        if budget is None:
            return
        self._sinr_history.append((budget.uplink_sinr_db, budget.downlink_sinr_db))
        hist = list(self._sinr_history)
        if len(hist) < 2:
            return

        panel_width = 300.0
        panel_height = 85.0
        gap = 8.0
        x = self.hud_width - panel_width - 14.0
        tx_y = self.hud_height - panel_height - 14.0
        rx_y = tx_y - panel_height - gap
        y = rx_y - NOISE_PANEL_HEIGHT - gap

        lo, hi = -10.0, 50.0
        graph_left = x + 4
        graph_width = panel_width - 8
        graph_bottom = y + 4
        graph_height = NOISE_PANEL_HEIGHT - 16

        def to_y(v: float) -> float:
            return graph_bottom + (min(max(v, lo), hi) - lo) / (hi - lo) * graph_height

        glColor3f(*HUD_PANEL_BORDER_COLOR)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + panel_width, y)
        glVertex2f(x + panel_width, y + NOISE_PANEL_HEIGHT)
        glVertex2f(x, y + NOISE_PANEL_HEIGHT)
        glEnd()

        # Dead-link threshold
        glColor3f(0.45, 0.12, 0.1)
        glBegin(GL_LINES)
        glVertex2f(graph_left, to_y(HEATMAP_DEAD_SINR - self._margin))
        glVertex2f(graph_left + graph_width, to_y(HEATMAP_DEAD_SINR - self._margin))
        glEnd()

        up, down = hist[-1]
        self._font.draw(f"SINR UP {up:.0f} DN {down:.0f} DB", x + 4,
                        y + NOISE_PANEL_HEIGHT - self._font._size - 3, colour=HUD_TEXT_COLOR)

        for index, colour in ((0, HUD_RAW_SIGNAL_COLOR), (1, HUD_WAVEFORM_COLOR)):
            glColor3f(*colour)
            glBegin(GL_LINE_STRIP)
            for i, values in enumerate(hist):
                glVertex2f(graph_left + (i / (len(hist) - 1)) * graph_width, to_y(values[index]))
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

        # Interference map (same cells as the ground heatmap)
        if self._heatmap_visible and self._heatmap_cells is not None:
            centres, rgba = self._heatmap_cells
            rel = centres - uav_pos[:2]
            inside = np.all(np.abs(rel) <= extent + HEATMAP_CELL, axis=1)
            if np.any(inside):
                pts = np.ascontiguousarray(
                    np.stack([cx + rel[inside, 0] * scale, cy + rel[inside, 1] * scale], axis=1),
                    dtype=np.float32)
                cols = np.ascontiguousarray(rgba[inside] * np.array([1, 1, 1, 1.6], dtype=np.float32))
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glPointSize(max(1.0, HEATMAP_CELL * scale))
                glEnableClientState(GL_VERTEX_ARRAY)
                glEnableClientState(GL_COLOR_ARRAY)
                glVertexPointer(2, GL_FLOAT, 0, pts)
                glColorPointer(4, GL_FLOAT, 0, cols)
                glDrawArrays(GL_POINTS, 0, len(pts))
                glDisableClientState(GL_COLOR_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisable(GL_BLEND)

        # Jammers (colour = kind)
        glPointSize(6.0)
        glBegin(GL_POINTS)
        for jammer in jammers:
            glColor3f(*JAMMER_KIND_COLORS.get(jammer.kind, JAMMER_BODY_COLOR))
            glVertex2f(cx + (jammer.position[0] - uav_pos[0]) * scale,
                       cy + (jammer.position[1] - uav_pos[1]) * scale)
        glEnd()

        # Ground station
        gx = cx + (self.world.ground_station[0] - uav_pos[0]) * scale
        gy = cy + (self.world.ground_station[1] - uav_pos[1]) * scale
        glColor3f(*GROUND_STATION_COLOR)
        glBegin(GL_LINE_LOOP)
        glVertex2f(gx, gy + 5.0)
        glVertex2f(gx - 4.0, gy - 3.0)
        glVertex2f(gx + 4.0, gy - 3.0)
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

def _sinr_colour(sinr_db: float) -> Tuple[float, float, float]:
    if sinr_db >= HEATMAP_GOOD_SINR:
        return (0.3, 1.0, 0.4)
    if sinr_db >= HEATMAP_MARGINAL_SINR:
        return (1.0, 0.85, 0.1)
    if sinr_db >= HEATMAP_DEAD_SINR:
        return (1.0, 0.5, 0.1)
    return (1.0, 0.15, 0.1)


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
