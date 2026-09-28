"""OCR.

* Japanese: manga-ocr (kha-white/manga-ocr-base) - reads whole vertical/horizontal bubbles, handles furigana-free manga text well.
* Korean / Chinese / English: PaddleOCR PP-OCRv5 (ONNX, via RapidOCR). Run on the whole page (tiled for tall
  webtoon strips) so its own line detector sees enough context, then each line is assigned to the
  comic-text-detector block that contains it. Much more accurate than EasyOCR on Korean syllables (였/했/줘).
Both are loaded lazily, so you only download what you use.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .detect import _tiles
from .types import TextBlock

_PADDLE_LANGS = {"ko": "KOREAN", "zh": "CH", "zh-TW": "CHINESE_CHT", "en": "EN"}


class OCR:
    def __init__(self):
        self._manga = None
        self._paddle = {}

    def manga(self):
        if self._manga is None:
            from manga_ocr import MangaOcr
            self._manga = MangaOcr()
        return self._manga

    def paddle(self, lang):
        if lang not in self._paddle:
            from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
            base = {"Det.lang_type": LangDet.CH, "Det.ocr_version": OCRVersion.PPOCRV5,
                    "Det.model_type": ModelType.MOBILE, "Rec.lang_type": getattr(LangRec, _PADDLE_LANGS[lang]),
                    "Rec.model_type": ModelType.MOBILE, "Global.use_cls": False, "Global.log_level": "error"}
            try:
                eng = RapidOCR(params={**base, "Rec.ocr_version": OCRVersion.PPOCRV5})
            except ValueError:  # language has no v5 model -> v4
                eng = RapidOCR(params={**base, "Rec.ocr_version": OCRVersion.PPOCRV4})
            self._paddle[lang] = eng
        return self._paddle[lang]

    def __call__(self, rgb: np.ndarray, blocks: list[TextBlock], lang: str) -> None:
        if lang == "ja":
            self._manga_blocks(rgb, blocks)
        else:
            self._paddle_page(rgb, blocks, lang)

    def _manga_blocks(self, rgb, blocks):
        H, W = rgb.shape[:2]
        for b in blocks:
            p = int(max(2, b.font_px * 0.15))
            crop = rgb[max(0, b.y1 - p):min(H, b.y2 + p), max(0, b.x1 - p):min(W, b.x2 + p)]
            if crop.size:
                b.text = self.manga()(Image.fromarray(crop)).strip()

    def _paddle_page(self, rgb, blocks, lang):
        eng = self.paddle(lang)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        lines = []  # (x1, y1, x2, y2, text)
        for ty1, ty2, tx1, tx2 in _tiles(*rgb.shape[:2]):
            res = eng(bgr[ty1:ty2, tx1:tx2])
            if res.txts is None:
                continue
            for box, txt in zip(res.boxes, res.txts):
                xs, ys = [p[0] + tx1 for p in box], [p[1] + ty1 for p in box]
                lines.append((min(xs), min(ys), max(xs), max(ys), txt.strip()))

        sep = "" if lang.startswith("zh") else " "
        for b in blocks:
            pad = max(4, b.font_px * 0.5)
            mine = [l for l in lines if l[4]
                    and b.x1 - pad <= (l[0] + l[2]) / 2 <= b.x2 + pad
                    and b.y1 - pad <= (l[1] + l[3]) / 2 <= b.y2 + pad]
            b.text = sep.join(l[4] for l in _reading_sort(_dedupe(mine)))


def _dedupe(lines):
    """Tiles overlap, so a line can be read twice; keep the longer reading of overlapping boxes."""
    kept = []
    for l in sorted(lines, key=lambda l: -len(l[4])):
        area = max(1e-6, (l[2] - l[0]) * (l[3] - l[1]))
        if all(max(0, min(l[2], k[2]) - max(l[0], k[0])) * max(0, min(l[3], k[3]) - max(l[1], k[1])) / area < 0.5
               for k in kept):
            kept.append(l)
    return kept


def _reading_sort(lines):
    """Rows top-to-bottom (lines whose vertical centres are within half a line height), then left-to-right."""
    if not lines:
        return lines
    h = float(np.median([l[3] - l[1] for l in lines])) or 1
    rows = []
    for l in sorted(lines, key=lambda l: (l[1] + l[3]) / 2):
        cy = (l[1] + l[3]) / 2
        if rows and abs(cy - rows[-1][0]) < h * 0.5:
            rows[-1][1].append(l)
        else:
            rows.append([cy, [l]])
    return [l for _, row in rows for l in sorted(row, key=lambda l: l[0])]
