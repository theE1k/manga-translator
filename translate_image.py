"""Translate manga pages from the command line.

  python translate_image.py page.jpg                      # ja -> zh, writes page.zh.png
  python translate_image.py chapter_dir/ --src ko --tgt zh
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mt import Pipeline

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--src", default="ja", choices=["ja", "ko", "zh", "zh-TW", "en"])
    ap.add_argument("--tgt", default="zh", choices=["zh", "zh-TW", "en", "ja", "ko"])
    ap.add_argument("-o", "--out", help="output folder (default: next to input)")
    a = ap.parse_args()

    files = []
    for p in map(Path, a.inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p]

    pipe = Pipeline()
    for f in files:
        print(f"== {f}")
        rgb = np.asarray(Image.open(f).convert("RGB"))
        out, _, _ = pipe.run(rgb, a.src, a.tgt)
        dst = Path(a.out) if a.out else f.parent
        dst.mkdir(parents=True, exist_ok=True)
        path = dst / f"{f.stem}.{a.tgt}.png"
        Image.fromarray(out).save(path)
        print(f"   -> {path}")


if __name__ == "__main__":
    main()
