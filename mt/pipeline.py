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

    def run(self, rgb: np.ndarray, src: str = "ja", tgt: str = "zh", verbose=True,
            history: list[tuple[str, str]] | None = None):
        """history: earlier (source, translation) pairs of the same chapter, used as translation context."""
        t = {}
        t0 = time.time()
        blocks, _ = self.detector.detect(rgb)
        t["detect"] = time.time() - t0
        if not blocks:
            return rgb, [], t

        t0 = time.time()
        analyze(rgb, blocks, src)
        blocks = reading_order(blocks, rtl=(src == "ja"))
        t["analyze"] = time.time() - t0

        t0 = time.time()
        self.ocr(rgb, blocks, src)
        t["ocr"] = time.time() - t0

        t0 = time.time()
        self.translator(blocks, src, tgt, history)
        t["translate"] = time.time() - t0

        done = [b for b in blocks if b.translation]
        t0 = time.time()
        clean = self.inpainter(rgb, done)
        t["inpaint"] = time.time() - t0

        t0 = time.time()
        out = render(clean, done, tgt)
        t["typeset"] = time.time() - t0

        if verbose:
            for b in blocks:
                print(f"  [{b.x1},{b.y1} {'V' if b.vertical else 'H'} {b.font_px:.0f}px] {b.text!r} -> {b.translation!r}")
            print("  timing: " + ", ".join(f"{k} {v:.2f}s" for k, v in t.items()))
        return out, blocks, t
