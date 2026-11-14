"""3D scene pieces: sky, fog, shaded terrain, the quadcopter model, its
shadow and the jammer towers.

Everything here is fixed-function OpenGL (what PyOpenGL + pygame give us
without shaders). Static geometry (terrain) is built once into numpy
arrays and drawn with vertex arrays in a single call; the small dynamic
models use immediate mode.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_ARRAY,
    GL_COLOR_MATERIAL,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FALSE,
    GL_FLOAT,
    GL_FOG,
    GL_FOG_COLOR,
    GL_FOG_END,
    GL_FOG_MODE,
    GL_FOG_START,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_LOOP,
    GL_LINEAR,
    GL_LINES,
    GL_MODELVIEW,
    GL_NORMALIZE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POLYGON_OFFSET_FILL,
    GL_POSITION,
    GL_PROJECTION,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TRIANGLE_FAN,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_INT,
    GL_VERTEX_ARRAY,
    glBegin,
    glBlendFunc,
    glColor3f,
    glColor4f,
    glColorMaterial,
    glColorPointer,
    glDepthMask,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glDrawElements,
    glEnable,
    glEnableClientState,
    glEnd,
    glFogf,
    glFogfv,
    glFogi,
    glLightfv,
    glLoadIdentity,
    glMatrixMode,
    glMultMatrixd,
    glNormal3f,
    glOrtho,
    glPolygonOffset,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glTranslatef,
    glVertex2f,
    glVertex3f,
    glVertexPointer,
)

SKY_TOP = (0.18, 0.33, 0.56)
SKY_HORIZON = (0.67, 0.76, 0.85)
LIGHT_DIR = np.array([0.62, 0.38, 0.62]) / np.linalg.norm([0.62, 0.38, 0.62])  # afternoon sun
FOG_START, FOG_END = 110.0, 460.0

Colour = Tuple[float, float, float]


# -- sky / fog / light --------------------------------------------------------
def draw_sky() -> None:
    """Vertical gradient behind everything (drawn before the 3D scene)."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_FOG)
    glBegin(GL_QUADS)
    glColor3f(*SKY_HORIZON)
    glVertex2f(0, 0)
    glVertex2f(1, 0)
    glColor3f(*SKY_TOP)
    glVertex2f(1, 1)
    glVertex2f(0, 1)
    glEnd()
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def enable_fog() -> None:
    glEnable(GL_FOG)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogfv(GL_FOG_COLOR, (*SKY_HORIZON, 1.0))
    glFogf(GL_FOG_START, FOG_START)
    glFogf(GL_FOG_END, FOG_END)


def set_sun() -> None:
    """Call right after the view transform so the light is fixed in the world."""
    glLightfv(GL_LIGHT0, GL_POSITION, (*LIGHT_DIR, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.85, 0.84, 0.78, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.38, 0.40, 0.46, 1.0))


def _lit(on: bool) -> None:
    if on:
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
    else:
        glDisable(GL_LIGHTING)


# -- terrain ------------------------------------------------------------------
class TerrainMesh:
    """Height-shaded, sun-lit terrain + a faint 10 m grid draped over it."""

    def __init__(self, ground, extent_half: float, step: float = 5.0) -> None:
        coords = np.arange(-extent_half, extent_half + step / 2.0, step)
        gx, gy = np.meshgrid(coords, coords)
        h = np.asarray(ground.height_at(gx, gy), dtype=float)
        dhdy, dhdx = np.gradient(h, step, step)
        normals = np.stack([-dhdx, -dhdy, np.ones_like(h)], axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        shade = 0.42 + 0.68 * np.clip(normals @ LIGHT_DIR, 0.0, 1.0)

        top = max(float(h.max()), 1.0)
        t = np.clip((h - float(h.min())) / top, 0.0, 1.0)[..., None]
        low = np.array([0.30, 0.45, 0.22])
        high = np.array([0.50, 0.46, 0.31])
        base = low + (high - low) * t
        rng = np.random.default_rng(3)
        base = base * (0.94 + 0.06 * rng.random(h.shape)[..., None])
        pad = (np.abs(gx) <= 3.0) & (np.abs(gy) <= 3.0)
        base[pad] = (0.55, 0.55, 0.52)
        colours = np.clip(base * shade[..., None], 0.0, 1.0)

        n = len(coords)
        self.vertices = np.ascontiguousarray(
            np.stack([gx, gy, h], axis=-1).reshape(-1, 3), dtype=np.float32)
        self.colours = np.ascontiguousarray(colours.reshape(-1, 3), dtype=np.float32)
        idx = np.arange(n * n).reshape(n, n)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        self.indices = np.ascontiguousarray(
            np.stack([a, b, d, a, d, c], axis=-1).reshape(-1), dtype=np.uint32)

        # 10 m grid lines following the surface
        every = max(1, int(round(10.0 / step)))
        lifted = np.stack([gx, gy, h + 0.04], axis=-1)
        segments = []
        for i in range(0, n, every):
            row, col = lifted[i], lifted[:, i]
            segments.append(np.stack([row[:-1], row[1:]], axis=1))
            segments.append(np.stack([col[:-1], col[1:]], axis=1))
        lines = np.concatenate(segments).reshape(-1, 3)
        self.grid = np.ascontiguousarray(lines, dtype=np.float32)
        axis = (np.abs(lines[:, 0]) < 1e-6) | (np.abs(lines[:, 1]) < 1e-6)
        grid_col = np.tile(np.array([0.06, 0.09, 0.05, 0.38], dtype=np.float32), (len(lines), 1))
        grid_col[axis] = (0.55, 0.25, 0.2, 0.45)
        self.grid_colours = np.ascontiguousarray(grid_col)

    def draw(self) -> None:
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 2.0)  # push the surface back so overlays win the depth test
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glColorPointer(3, GL_FLOAT, 0, self.colours)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)
        glDisable(GL_POLYGON_OFFSET_FILL)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glVertexPointer(3, GL_FLOAT, 0, self.grid)
        glColorPointer(4, GL_FLOAT, 0, self.grid_colours)
        glDrawArrays(GL_LINES, 0, len(self.grid))
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


# -- primitives -----------------------------------------------------------------
_FACES = (
    ((1, 0, 0), ((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5))),
    ((-1, 0, 0), ((-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5))),
    ((0, 1, 0), ((-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5))),
    ((0, -1, 0), ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5))),
    ((0, 0, 1), ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5))),
    ((0, 0, -1), ((-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5))),
)


def draw_box(centre: Sequence[float], size: Sequence[float], colour: Colour) -> None:
    glColor3f(*colour)
    glPushMatrix()
    glTranslatef(*centre)
    glScalef(*size)
    glBegin(GL_QUADS)
    for normal, corners in _FACES:
        glNormal3f(*normal)
        for c in corners:
            glVertex3f(*c)
    glEnd()
    glPopMatrix()


def _disc(cx: float, cy: float, z: float, radius: float, segments: int = 24) -> None:
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(cx, cy, z)
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        glVertex3f(cx + radius * math.cos(a), cy + radius * math.sin(a), z)
    glEnd()


# -- quadcopter -------------------------------------------------------------------
class UavModel:
    """A ~450 mm quad: body, red front arms / grey rear arms, motors,
    skids and four props that spin with each motor's real speed."""

    LIFT = 0.09                 # model drawn above the physics point so the skids touch down
    PROP_RADIUS = 0.115
    VISUAL_REV_PER_S = 9.0      # deliberately slow so blades stay readable on screen

    def __init__(self, uav) -> None:
        self.uav = uav
        self.prop_angle = np.zeros(len(uav.motors))

    def draw(self, dt: float, motor_colours: Sequence[Colour]) -> None:
        body = self.uav.body
        m = np.eye(4)
        m[:3, :3] = body.orientation.as_matrix()
        m[:3, 3] = body.position
        glPushMatrix()
        glMultMatrixd(np.ascontiguousarray(m.T).ravel())
        glTranslatef(0.0, 0.0, self.LIFT)

        _lit(True)
        draw_box((0.0, 0.0, 0.0), (0.20, 0.13, 0.06), (0.17, 0.18, 0.20))
        draw_box((0.01, 0.0, 0.045), (0.13, 0.09, 0.03), (0.86, 0.87, 0.89))
        draw_box((0.115, 0.0, -0.005), (0.03, 0.05, 0.035), (0.08, 0.08, 0.09))   # camera
        for motor in self.uav.motors:
            p = motor.position_body
            length = float(np.hypot(p[0], p[1]))
            front = p[0] > 0.0
            glPushMatrix()
            glRotatef(math.degrees(math.atan2(p[1], p[0])), 0.0, 0.0, 1.0)
            draw_box((length / 2.0, 0.0, 0.0), (length, 0.028, 0.02),
                     (0.85, 0.20, 0.16) if front else (0.74, 0.75, 0.78))
            glPopMatrix()
            draw_box((p[0], p[1], 0.012), (0.045, 0.045, 0.04), (0.12, 0.12, 0.13))
        for side in (-1.0, 1.0):                                             # skids
            y = 0.065 * side
            draw_box((0.05, y, -0.055), (0.012, 0.012, 0.07), (0.2, 0.2, 0.22))
            draw_box((-0.05, y, -0.055), (0.012, 0.012, 0.07), (0.2, 0.2, 0.22))
            draw_box((0.0, y, -0.09), (0.24, 0.014, 0.012), (0.2, 0.2, 0.22))
        _lit(False)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        r = self.PROP_RADIUS
        for i, motor in enumerate(self.uav.motors):
            p = motor.position_body
            speed = float(motor.speed)
            self.prop_angle[i] += motor.spin_direction * speed * self.VISUAL_REV_PER_S * 360.0 * dt
            colour = motor_colours[i % len(motor_colours)]
            z = 0.04
            glColor4f(*colour, 0.08 + 0.22 * speed)          # blur disc grows with speed
            _disc(p[0], p[1], z, r)
            glColor4f(*colour, 0.9)
            glBegin(GL_LINE_LOOP)
            for k in range(24):
                a = 2.0 * math.pi * k / 24
                glVertex3f(p[0] + r * math.cos(a), p[1] + r * math.sin(a), z)
            glEnd()
            glPushMatrix()
            glTranslatef(p[0], p[1], z + 0.002)
            glRotatef(self.prop_angle[i] % 360.0, 0.0, 0.0, 1.0)
            glColor4f(0.05, 0.05, 0.06, 0.95 - 0.6 * speed)
            glBegin(GL_QUADS)
            for blade in (1.0, -1.0):
                glVertex3f(0.0, -0.008, 0.0)
                glVertex3f(blade * r, -0.012, 0.0)
                glVertex3f(blade * r, 0.012, 0.0)
                glVertex3f(0.0, 0.008, 0.0)
            glEnd()
            glPopMatrix()
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glPopMatrix()


def draw_shadow(x: float, y: float, ground_z: float, height_above: float) -> None:
    """Soft dark blob on the ground under the UAV; fades and grows with height."""
    alpha = 0.5 * math.exp(-max(height_above, 0.0) / 18.0)
    if alpha < 0.02:
        return
    radius = 0.32 + 0.02 * height_above
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDepthMask(GL_FALSE)
    for scale, a in ((1.0, alpha * 0.5), (0.65, alpha)):
        glColor4f(0.0, 0.0, 0.0, a)
        _disc(x, y, ground_z + 0.06, radius * scale, 20)
    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)


def draw_tower(base: np.ndarray, height: float, colour: Colour) -> None:
    """Lattice mast from `base` (on the ground) up to `height` metres."""
    b, t = 0.45, 0.12
    levels = 4
    corners = ((1, 1), (-1, 1), (-1, -1), (1, -1))

    def ring(z: float) -> list:
        w = b + (t - b) * (z / height)
        return [(base[0] + cx * w, base[1] + cy * w, base[2] + z) for cx, cy in corners]

    rings = [ring(height * k / levels) for k in range(levels + 1)]
    glColor3f(*(0.55 * c + 0.2 for c in colour))
    glBegin(GL_LINES)
    for k in range(levels):
        lo, hi = rings[k], rings[k + 1]
        for i in range(4):
            j = (i + 1) % 4
            glVertex3f(*lo[i]); glVertex3f(*hi[i])        # leg  # noqa: E702
            glVertex3f(*lo[i]); glVertex3f(*hi[j])        # brace  # noqa: E702
            glVertex3f(*hi[i]); glVertex3f(*hi[j])        # ring  # noqa: E702
    glVertex3f(base[0], base[1], base[2] + height)
    glVertex3f(base[0], base[1], base[2] + height + 0.6)  # antenna rod
    glEnd()
