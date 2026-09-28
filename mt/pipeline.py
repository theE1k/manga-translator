from __future__ import annotations

import time

import numpy as np

from .analyze import analyze
from .types import TextBlock
from .typeset import render


def reading_order(blocks: list[TextBlock], rtl: bool) -> list[TextBlock]:
    if not blocks:
        return blocks
    band = float(np.median([b.h for b in blocks])) * 0.6 or 1
    return sorted(blocks, key=lambda b: (int(b.y1 / band), -b.cx if rtl else b.cx))


class Pipeline:
    """detect -> analyse -> OCR -> translate -> erase -> re-letter. Components load on first use."""

    def __init__(self, detector=None, ocr=None, translator=None, inpainter=None):
        self._det, self._ocr, self._tr, self._inp = detector, ocr, translator, inpainter

    @property
    def detector(self):
        if self._det is None:
            from .detect import TextDetector
            self._det = TextDetector()
        return self._det

    @property
    def ocr(self):
        if self._ocr is None:
            from .ocr import OCR
            self._ocr = OCR()
        return self._ocr

    @property
    def translator(self):
        if self._tr is None:
            from .translate import Translator
            self._tr = Translator()
        return self._tr

    @property
    def inpainter(self):
        if self._inp is None:
            from .inpaint import Inpainter
            self._inp = Inpainter()
        return self._inp

    def read(self, rgb: np.ndarray, src: str = "ja") -> list[TextBlock]:
        """detect -> analyse -> reading order -> OCR. Blocks keep page-sized glyph masks."""
        blocks, _ = self.detector.detect(rgb)
        if not blocks:
            return []
        analyze(rgb, blocks, src)
        blocks = reading_order(blocks, rtl=(src == "ja"))
        self.ocr(rgb, blocks, src)
        return blocks

    def finish(self, rgb: np.ndarray, blocks: list[TextBlock], src: str, tgt: str,
               history: list[tuple[str, str]] | None = None, ahead: list[str] | None = None,
               erase_only: list[TextBlock] | None = None):
        """translate -> erase -> re-letter. history: earlier (source, translation) pairs of the chapter;
        ahead: source lines that come right after this page (context only)."""
        t = {}
        t0 = time.time()
        self.translator(blocks, src, tgt, history, ahead)
        t["translate"] = time.time() - t0
        done = [b for b in blocks if b.translation]
        t0 = time.time()
        clean = self.inpainter(rgb, done + list(erase_only or []))
        t["inpaint"] = time.time() - t0
        t0 = time.time()
        out = render(clean, done, tgt)
        t["typeset"] = time.time() - t0
        return out, t

    def run(self, rgb: np.ndarray, src: str = "ja", tgt: str = "zh", verbose=True,
            history: list[tuple[str, str]] | None = None, ahead: list[str] | None = None):
        """Whole page in one go (CLI / tests)."""
        t0 = time.time()
        blocks = self.read(rgb, src)
        t = {"read": time.time() - t0}
        if not blocks:
            return rgb, [], t
        out, t2 = self.finish(rgb, blocks, src, tgt, history, ahead)
        t.update(t2)
        if verbose:
            log_blocks(blocks, t)
        return out, blocks, t


def log_blocks(blocks, t):
    for b in blocks:
        print(f"  [{b.x1},{b.y1} {'V' if b.vertical else 'H'} {b.font_px:.0f}px] {b.text!r} -> {b.translation!r}")
    print("  timing: " + ", ".join(f"{k} {v:.2f}s" for k, v in t.items()))
