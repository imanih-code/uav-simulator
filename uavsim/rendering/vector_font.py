"""A tiny stroke/vector font, drawn entirely with line segments.

No textures, no font files -- every character is just a handful of
(point, point) line segments in a 0..1 x 0..1 box, in the spirit of an old
oscilloscope or avionics character display. This is intentionally not a
beautiful font: it only needs to be legible at HUD sizes for digits, a
handful of letters used in short labels (BAT, ALT, ROL, PIT, YAW, POS,
MASS, TX, RX...), and a few punctuation marks.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

# Named reference points on a 0..1 x 0..1 grid.
_TL, _TM, _TR = (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)
_ML, _MM, _MR = (0.0, 0.5), (0.5, 0.5), (1.0, 0.5)
_BL, _BM, _BR = (0.0, 0.0), (0.5, 0.0), (1.0, 0.0)

# The classic 7-segment layout, reused as the base for digits and several
# letters that happen to render legibly on a 7-segment-style display.
_TOP = (_TL, _TR)
_TOP_LEFT = (_TL, _ML)
_TOP_RIGHT = (_TR, _MR)
_MID = (_ML, _MR)
_BOT_LEFT = (_ML, _BL)
_BOT_RIGHT = (_MR, _BR)
_BOTTOM = (_BL, _BR)

GLYPHS: Dict[str, List[Segment]] = {
    "0": [_TOP, _TOP_LEFT, _TOP_RIGHT, _BOT_LEFT, _BOT_RIGHT, _BOTTOM, (_BL, _TR)],
    "1": [_TOP_RIGHT, _BOT_RIGHT],
    "2": [_TOP, _TOP_RIGHT, _MID, _BOT_LEFT, _BOTTOM],
    "3": [_TOP, _TOP_RIGHT, _MID, _BOT_RIGHT, _BOTTOM],
    "4": [_TOP_LEFT, _MID, _TOP_RIGHT, _BOT_RIGHT],
    "5": [_TOP, _TOP_LEFT, _MID, _BOT_RIGHT, _BOTTOM],
    "6": [_TOP, _TOP_LEFT, _MID, _BOT_LEFT, _BOT_RIGHT, _BOTTOM],
    "7": [_TOP, _TOP_RIGHT, _BOT_RIGHT],
    "8": [_TOP, _TOP_LEFT, _TOP_RIGHT, _MID, _BOT_LEFT, _BOT_RIGHT, _BOTTOM],
    "9": [_TOP, _TOP_LEFT, _TOP_RIGHT, _MID, _BOT_RIGHT, _BOTTOM],
    "A": [_TOP, _TOP_LEFT, _TOP_RIGHT, _MID, _BOT_LEFT, _BOT_RIGHT],
    "B": [_TOP_LEFT, _MID, _BOT_LEFT, _BOT_RIGHT, _BOTTOM],
    "G": [_TOP, _TOP_LEFT, _MID, _BOT_LEFT, _BOT_RIGHT, _BOTTOM],
    "H": [_TOP_LEFT, _TOP_RIGHT, _MID, _BOT_LEFT, _BOT_RIGHT],
    "I": [(_TM, _BM)],
    "K": [_TOP_LEFT, _BOT_LEFT, (_ML, _TR), (_ML, _BR)],
    "L": [_TOP_LEFT, _BOT_LEFT, _BOTTOM],
    "M": [_TOP_LEFT, _BOT_LEFT, _TOP_RIGHT, _BOT_RIGHT, (_TL, _MM), (_TR, _MM)],
    "N": [_TOP_LEFT, _BOT_LEFT, _TOP_RIGHT, _BOT_RIGHT, (_TL, _BR)],
    "O": [_TOP, _TOP_LEFT, _TOP_RIGHT, _BOT_LEFT, _BOT_RIGHT, _BOTTOM],
    "P": [_TOP, _TOP_LEFT, _TOP_RIGHT, _MID, _BOT_LEFT],
    "R": [_TOP, _TOP_LEFT, _TOP_RIGHT, _MID, _BOT_LEFT, (_MM, _BR)],
    "S": [_TOP, _TOP_LEFT, _MID, _BOT_RIGHT, _BOTTOM],
    "T": [_TOP, (_TM, _BM)],
    "W": [_TOP_LEFT, _BOT_LEFT, _TOP_RIGHT, _BOT_RIGHT, (_BL, _MM), (_BR, _MM)],
    "X": [(_TL, _BR), (_TR, _BL)],
    "Y": [(_TL, _MM), (_TR, _MM), (_MM, _BM)],
    "Z": [_TOP, (_TR, _BL), _BOTTOM],
    ".": [((0.4, 0.0), (0.5, 0.1))],
    "-": [_MID],
    ":": [((0.45, 0.28), (0.55, 0.28)), ((0.45, 0.72), (0.55, 0.72))],
    "%": [(_BL, _TR), ((0.02, 0.82), (0.18, 0.98)), ((0.82, 0.02), (0.98, 0.18))],
    "/": [(_BL, _TR)],
    " ": [],
    "=": [_MID, ((0.0, 0.3), (1.0, 0.3))],
    "U": [_TOP_LEFT, _BOT_LEFT, _BOTTOM, _BOT_RIGHT, _TOP_RIGHT],
    "V": [(_TL, _BM), (_TR, _BM)],
    "E": [_TOP, _TOP_LEFT, _MID, _BOT_LEFT, _BOTTOM],
    "D": [_TOP, _TOP_LEFT, _TOP_RIGHT, _BOT_RIGHT, _BOTTOM, _BOT_LEFT],
    "J": [_TOP_RIGHT, _BOT_RIGHT, _BOTTOM],
    "C": [_TOP, _TOP_LEFT, _BOT_LEFT, _BOTTOM],
    "+": [_MID, ((0.5, 0.2), (0.5, 0.8))],
}


def draw_text(
    text: str,
    x: float,
    y: float,
    char_width: float,
    char_height: float,
    spacing: float,
    draw_segment,
) -> float:
    """Lay out `text` starting at (x, y) using the given glyph cell size.

    `draw_segment(p1, p2)` is called once per line to actually paint it
    (kept generic here so this module has zero OpenGL dependency of its
    own). Returns the x position right after the last character, so
    callers can chain labels together.
    """
    cursor_x = x
    for char in text.upper():
        segments = GLYPHS.get(char, [])
        for (sx1, sy1), (sx2, sy2) in segments:
            p1 = (cursor_x + sx1 * char_width, y + sy1 * char_height)
            p2 = (cursor_x + sx2 * char_width, y + sy2 * char_height)
            draw_segment(p1, p2)
        cursor_x += char_width + spacing
    return cursor_x


def text_width(text: str, char_width: float, spacing: float) -> float:
    if not text:
        return 0.0
    return len(text) * char_width + (len(text) - 1) * spacing
