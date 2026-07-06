"""Text renderer using a TrueType font uploaded as an OpenGL texture atlas.

Loads a .ttf via Pygame, renders each glyph once to a surface, packs them
into an RGBA texture, then draws text by emitting textured quads (one per
character).  Supports per-character colour via glColor* (texture is pure
white-on-transparent, multiplied by the current colour).

Usage:
    font = TTFFont("path/to/font.ttf", size=14)
    font.draw("SENT", x, y, colour=(0.3, 1.0, 0.4))
"""
from __future__ import annotations

import string
from typing import Dict, Optional, Tuple

# OpenGL imports -- the caller must have an active GL context.
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glPixelStorei,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glDeleteTextures,
    glIsTexture,
    GL_UNPACK_ALIGNMENT,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
)

import pygame


class TTFFont:
    """Texture-atlas-based TTF text renderer for OpenGL."""

    def __init__(self, path: str, size: int = 14) -> None:
        self._path = path
        self._size = size
        self._texture_ids: Dict[str, int] = {}
        self._glyph_data: Dict[str, Tuple[int, int, int, int]] = {}
        self._loaded = False
        self._ascent: int = 0
        self._height: int = 0

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        font = pygame.font.Font(self._path, self._size)
        self._ascent = font.get_ascent()
        self._height = font.get_height()

        chars = string.printable
        for ch in chars:
            if ch in ("\t", "\n", "\r", "\x0b", "\x0c"):
                continue
            surf = font.render(ch, True, (255, 255, 255, 255))
            w, h = surf.get_size()
            if w == 0 or h == 0:
                continue

            raw = pygame.image.tostring(surf, "RGBA", True)
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw
            )
            self._texture_ids[ch] = tex_id
            self._glyph_data[ch] = (w, h, 0, 0)

        font = None

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        colour: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Render `text` at screen position (x, y) with the given colour."""
        self._ensure_loaded()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glColor3f(*colour)

        cursor_x = x
        for ch in text:
            tex_id = self._texture_ids.get(ch)
            if tex_id is None:
                cursor_x += 4
                continue
            w, h, _, _ = self._glyph_data[ch]
            if w == 0:
                continue

            glBindTexture(GL_TEXTURE_2D, tex_id)
            top = y + float(self._ascent - h)
            bot = y + float(self._ascent)
            glBegin(GL_TRIANGLE_STRIP)
            glTexCoord2f(0.0, 0.0)
            glVertex2f(cursor_x, top)
            glTexCoord2f(1.0, 0.0)
            glVertex2f(cursor_x + w, top)
            glTexCoord2f(0.0, 1.0)
            glVertex2f(cursor_x, bot)
            glTexCoord2f(1.0, 1.0)
            glVertex2f(cursor_x + w, bot)
            glEnd()
            cursor_x += w + 1

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

    def text_width(self, text: str) -> float:
        """Return the pixel width of `text` at the current font size."""
        self._ensure_loaded()
        total = 0
        for ch in text:
            data = self._glyph_data.get(ch)
            w = data[0] if data else 4
            total += w + 1
        return max(total - 1, 0)

    def cleanup(self) -> None:
        """Delete all OpenGL textures."""
        for tex_id in self._texture_ids.values():
            if glIsTexture(tex_id):
                glDeleteTextures(1, [tex_id])
        self._texture_ids.clear()
        self._glyph_data.clear()
        self._loaded = False
