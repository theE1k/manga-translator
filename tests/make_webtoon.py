"""Draw a synthetic Korean webtoon strip (original art, no real comic) for end-to-end tests.

  python tests/make_webtoon.py out/webtoon_ko.png
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_BOLD_INDEX = 6  # Bold face inside the collection
W, H = 800, 4200


def gradient(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    a = (1 - t) * np.array(top) + t * np.array(bottom)
    return Image.fromarray(np.repeat(a, w, axis=1).astype(np.uint8))


def font(size):
    return ImageFont.truetype(FONT, size, index=FONT_BOLD_INDEX)


def htext(d, cx, cy, lines, size, fill=(20, 20, 20), stroke=None, sw=0):
    f = font(size)
    lh = size * 1.3
    top = cy - lh * len(lines) / 2
    for i, l in enumerate(lines):
        kw = dict(stroke_width=sw, stroke_fill=stroke) if sw else {}
        d.text((cx, top + i * lh + lh / 2), l, font=f, fill=fill, anchor="mm", **kw)


def bubble(d, cx, cy, rw, rh, tail=None):
    if tail:
        d.polygon([(cx - 20, cy + rh - 10), (cx + 20, cy + rh - 10), tail], fill="white", outline=(30, 30, 30))
    d.ellipse((cx - rw, cy - rh, cx + rw, cy + rh), fill="white", outline=(30, 30, 30), width=3)


def main(path):
    img = Image.new("RGB", (W, H), (250, 250, 250))
    # scene 1: evening sky street
    img.paste(gradient(W, 1100, (255, 170, 120), (90, 70, 140)), (0, 80))
    # scene 2: indoor warm wall
    img.paste(gradient(W, 1000, (240, 225, 200), (200, 175, 150)), (0, 1500))
    # scene 3: dark night with blur glow
    night = gradient(W, 1100, (20, 25, 60), (60, 40, 90))
    g = ImageDraw.Draw(night)
    for i in range(25):
        x, y = (i * 137) % W, (i * 311) % 1100
        g.ellipse((x - 18, y - 18, x + 18, y + 18), fill=(255, 220, 150))
    night = night.filter(ImageFilter.GaussianBlur(6))
    img.paste(night, (0, 2900))

    d = ImageDraw.Draw(img)
    # simple silhouettes
    d.rectangle((0, 900, W, 1180), fill=(60, 50, 80))
    d.ellipse((300, 500, 500, 700), fill=(40, 35, 55))
    d.rectangle((330, 690, 470, 1000), fill=(40, 35, 55))
    d.rectangle((0, 2250, W, 2500), fill=(150, 110, 80))

    bubble(d, 250, 300, 190, 110, tail=(330, 500))
    htext(d, 250, 300, ["오늘은 정말", "긴 하루였어..."], 34)
    bubble(d, 560, 1300, 200, 100, tail=(480, 1150))
    htext(d, 560, 1300, ["너 지금 어디야?", "빨리 와!"], 34)
    # narration box
    d.rectangle((60, 1560, 520, 1680), fill="white", outline=(30, 30, 30), width=3)
    htext(d, 290, 1620, ["그날 이후로 모든 것이", "달라지기 시작했다."], 30)
    bubble(d, 520, 1950, 210, 120)
    htext(d, 520, 1950, ["잠깐만, 이거", "네가 한 거야?"], 36)
    # text directly on artwork (white with dark outline)
    htext(d, 400, 3150, ["기다려 줘서 고마워"], 40, fill=(255, 255, 255), stroke=(20, 20, 40), sw=4)
    bubble(d, 300, 3700, 220, 120, tail=(420, 3900))
    htext(d, 300, 3700, ["다음에 또", "만나자, 약속이야."], 34)
    img.save(path)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "webtoon_ko.png")
