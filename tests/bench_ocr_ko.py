"""Compare Korean OCR engines on the synthetic webtoon (tests/make_webtoon.py) using the real detector.

  python tests/bench_ocr_ko.py out/webtoon_ko.png
"""
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from mt.analyze import analyze
from mt.detect import TextDetector
from mt.pipeline import reading_order

TRUTH = [
    "오늘은 정말 긴 하루였어...",
    "너 지금 어디야? 빨리 와!",
    "그날 이후로 모든 것이 달라지기 시작했다.",
    "잠깐만, 이거 네가 한 거야?",
    "기다려 줘서 고마워",
    "다음에 또 만나자, 약속이야.",
]


def norm(s):
    return "".join(s.split())


def cer(a, b):
    a, b = norm(a), norm(b)
    return 1 - SequenceMatcher(None, a, b).ratio()


def crops(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    blocks, _ = TextDetector().detect(rgb)
    analyze(rgb, blocks, "ko")
    blocks = reading_order(blocks, rtl=False)
    H, W = rgb.shape[:2]
    out = []
    for b in blocks:
        p = int(max(2, b.font_px * 0.15))
        out.append(rgb[max(0, b.y1 - p):min(H, b.y2 + p), max(0, b.x1 - p):min(W, b.x2 + p)])
    return out


def easy():
    import easyocr
    r = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return lambda im: " ".join(r.readtext(im, detail=0, paragraph=True))


def paddle(version):
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
    eng = RapidOCR(params={
        "Det.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": LangDet.CH,
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
        "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.lang_type": LangRec.KOREAN,
        "Rec.ocr_version": version, "Rec.model_type": ModelType.MOBILE,
        "Global.use_cls": False, "Global.log_level": "error",
    })

    def run(im):
        import cv2
        pad = max(16, im.shape[0] // 3)
        im = cv2.copyMakeBorder(im[:, :, ::-1], pad, pad, pad, pad, cv2.BORDER_REPLICATE)  # RGB->BGR + margin
        res = eng(im)
        if res.txts is None:
            return ""
        # top-to-bottom line order
        order = sorted(range(len(res.txts)), key=lambda i: res.boxes[i][0][1])
        return " ".join(res.txts[i] for i in order)
    return run


if __name__ == "__main__":
    cs = crops(sys.argv[1] if len(sys.argv) > 1 else "out/webtoon_ko.png")
    print(f"{len(cs)} blocks detected")
    from rapidocr import OCRVersion
    engines = {"easyocr": easy, "ppocrv5-ko": lambda: paddle(OCRVersion.PPOCRV5)}
    for name, make in engines.items():
        try:
            f = make()
        except Exception as e:  # engine/model not available
            print(f"\n== {name}: unavailable ({e})")
            continue
        t0 = time.time()
        outs = [f(c) for c in cs]
        dt = time.time() - t0
        exact = sum(norm(o) == norm(t) for o, t in zip(outs, TRUTH))
        mean_cer = np.mean([cer(o, t) for o, t in zip(outs, TRUTH)])
        print(f"\n== {name}: exact {exact}/{len(TRUTH)}, char error {mean_cer:.1%}, {dt:.1f}s")
        for o, t in zip(outs, TRUTH):
            print(("  ✓ " if norm(o) == norm(t) else "  ✗ ") + o + ("" if norm(o) == norm(t) else f"   (truth: {t})"))
