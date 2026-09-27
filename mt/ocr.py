"""OCR.

* Japanese: manga-ocr (kha-white/manga-ocr-base) - reads whole vertical/horizontal bubbles, handles furigana-free manga text well.
* Korean / Chinese / English: EasyOCR.
Both are loaded lazily, so you only download what you use.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .types import TextBlock

_EASY_LANGS = {"ko": ["ko", "en"], "zh": ["ch_sim", "en"], "zh-TW": ["ch_tra", "en"], "en": ["en"]}


class OCR:
    def __init__(self):
        self._manga = None
        self._easy = {}

    def manga(self):
        if self._manga is None:
            from manga_ocr import MangaOcr
            self._manga = MangaOcr()
        return self._manga

    def easy(self, lang):
        if lang not in self._easy:
            import easyocr
            self._easy[lang] = easyocr.Reader(_EASY_LANGS[lang], gpu=False, verbose=False)
        return self._easy[lang]

    def __call__(self, rgb: np.ndarray, blocks: list[TextBlock], lang: str) -> None:
        H, W = rgb.shape[:2]
        for b in blocks:
            p = int(max(2, b.font_px * 0.15))
            crop = rgb[max(0, b.y1 - p):min(H, b.y2 + p), max(0, b.x1 - p):min(W, b.x2 + p)]
            if crop.size == 0:
                continue
            if lang == "ja":
                b.text = self.manga()(Image.fromarray(crop)).strip()
            else:
                res = self.easy(lang).readtext(crop, detail=0, paragraph=True)
                sep = "" if lang.startswith("zh") else " "
                b.text = sep.join(res).strip()
