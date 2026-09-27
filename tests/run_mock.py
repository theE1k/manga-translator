"""Run the full pipeline on the synthetic page with mock detector/OCR/translator.

Exercises the real analyse -> erase -> re-letter code; only the ML models are mocked.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mt.detect import attach_masks  # noqa: E402
from mt.pipeline import Pipeline  # noqa: E402
from mt.types import TextBlock  # noqa: E402


class MockDet:
    def __init__(self, items):
        self.items = items

    def detect(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blocks = [TextBlock(*it["box"]) for it in self.items]
        seg = (gray < 50).astype(np.float32)
        attach_masks(blocks, seg > 0.5)
        return blocks, seg


class MockOCR:
    def __init__(self, items):
        self.items = items

    def __call__(self, rgb, blocks, lang):
        for b in blocks:
            it = min(self.items, key=lambda i: abs(i["box"][0] - b.x1) + abs(i["box"][1] - b.y1))
            b.text = it["src"]
            b._tr = it["tr"]


class MockTr:
    def __call__(self, blocks, src, tgt):
        for b in blocks:
            b.translation = b._tr


def main(d):
    d = Path(d)
    items = json.load(open(d / "test_page.json"))
    rgb = np.asarray(Image.open(d / "test_page.png").convert("RGB"))
    pipe = Pipeline(detector=MockDet(items), ocr=MockOCR(items), translator=MockTr())
    out, blocks, _ = pipe.run(rgb, "ja", "zh")
    Image.fromarray(out).save(d / "test_out.png")
    clean = pipe.inpainter(rgb, blocks)
    Image.fromarray(clean).save(d / "test_clean.png")
    sbs = np.concatenate([rgb, np.full((rgb.shape[0], 16, 3), 255, np.uint8), out], 1)
    Image.fromarray(sbs).save(d / "test_side_by_side.png")
    for b in blocks:
        print(b.box, "render", b.render_box, "bubble", b.in_bubble, "stroke", b.stroke, "fg", b.fg)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
