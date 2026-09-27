from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TextBlock:
    """One text region on the page (usually one speech bubble)."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0
    text: str = ""            # OCR result
    translation: str = ""     # translated text
    vertical: bool = False    # source text written top-to-bottom
    font_px: float = 0.0      # estimated source glyph size in px
    fg: tuple = (0, 0, 0)     # source text colour (RGB)
    bg: tuple = (255, 255, 255)  # background colour around the text
    in_bubble: bool = False   # sits on a flat-colour bubble
    stroke: Optional[tuple] = None  # outline colour if the source text had one
    render_box: Optional[tuple] = None  # area available for the new text
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # full-page bool mask of this block's glyphs
    bubble: Optional[tuple] = field(default=None, repr=False)  # (x, y, bool mask) convex bubble interior

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def box(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2
