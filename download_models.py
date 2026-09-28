"""Download the open-source models and the font. Safe to re-run (skips finished files)."""
import sys
from pathlib import Path

import requests

from mt import config

FILES = [
    ("comic-text-detector (text detection, ~95 MB)",
     "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
     config.DETECTOR_ONNX),
    ("LaMa (text removal on artwork, ~200 MB)",
     "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
     config.LAMA_ONNX),
    ("Noto Sans CJK SC Bold (lettering font, ~16 MB)",
     "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
     config.FONTS / "NotoSansCJKsc-Bold.otf"),
]


def fetch(name, url, dst: Path):
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"✓ {name}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"↓ {name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                    sys.stdout.flush()
    print()
    tmp.rename(dst)


def warm_ocr(langs):
    """manga-ocr and PaddleOCR download their own weights on first use; do it now."""
    if "ja" in langs:
        print("↓ manga-ocr (Japanese OCR, ~450 MB)")
        from manga_ocr import MangaOcr
        MangaOcr()
    from mt.ocr import OCR
    ocr = OCR()
    for l in langs:
        if l != "ja":
            print(f"↓ PaddleOCR PP-OCRv5 ({l})")
            ocr.paddle(l)


if __name__ == "__main__":
    langs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ja"]
    for item in FILES:
        fetch(*item)
    warm_ocr(langs)
    print("done.")
