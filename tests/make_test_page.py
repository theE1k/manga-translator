"""Draw a synthetic manga page (original art, no real manga) with known text positions.

Used to test erasing + re-lettering without the detector/OCR/LLM models.
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H = 900, 1280
rng = np.random.default_rng(3)


def screentone(w, h, pitch=7, r=1.6, shade=90):
    a = np.full((h, w), 255, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx % pitch - pitch / 2) ** 2 + (yy % pitch - pitch / 2) ** 2) ** 0.5
    a[d < r] = shade
    return a


def vtext(draw, x_right, y_top, lines, size, fill=(0, 0, 0), stroke=None, sw=0):
    f = ImageFont.truetype(FONT, size, index=0)
    boxes = []
    for ci, line in enumerate(lines):
        cx = x_right - size / 2 - ci * size * 1.3
        for ri, ch in enumerate(line):
            cy = y_top + ri * size * 1.05 + size / 2
            ch = {"、": "︑", "。": "︒", "…": "︙", "ー": "丨"}.get(ch, ch)
            kw = dict(stroke_width=sw, stroke_fill=stroke) if sw else {}
            draw.text((cx, cy), ch, font=f, fill=fill, anchor="mm", **kw)
    w = size * 1.3 * (len(lines) - 1) + size
    h = max(len(l) for l in lines) * size * 1.05
    return [int(x_right - w) - 2, int(y_top) - 2, int(x_right) + 2, int(y_top + h) + 2]


def htext(draw, cx, cy, lines, size, fill=(0, 0, 0)):
    f = ImageFont.truetype(FONT, size, index=0)
    lh = size * 1.2
    top = cy - lh * len(lines) / 2
    ws = []
    for i, l in enumerate(lines):
        draw.text((cx, top + i * lh + lh / 2), l, font=f, fill=fill, anchor="mm")
        ws.append(f.getlength(l))
    w = max(ws)
    return [int(cx - w / 2) - 2, int(top) - 2, int(cx + w / 2) + 2, int(top + lh * len(lines)) + 2]


def bubble(draw, box, lw=3):
    draw.ellipse(box, fill=(255, 255, 255), outline=(0, 0, 0), width=lw)


def main(out_dir):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    # panels
    panels = [(20, 20, 880, 420), (20, 440, 440, 860), (460, 440, 880, 860), (20, 880, 880, 1260)]
    # panel 1: sky gradient + hills
    g = np.linspace(170, 245, 400).astype(np.uint8)
    p1 = np.repeat(g[:, None], 860, 1)
    img.paste(Image.fromarray(np.stack([p1 - 20, p1 - 5, p1], -1)), (20, 20))
    d.polygon([(20, 420), (20, 300), (200, 230), (420, 320), (640, 250), (880, 330), (880, 420)], fill=(90, 110, 90))
    # panel 2: screentone background with a figure silhouette
    img.paste(Image.fromarray(screentone(420, 420)).convert("RGB"), (20, 440))
    d.ellipse((150, 560, 330, 760), fill=(60, 60, 60))
    d.rectangle((170, 740, 310, 860), fill=(60, 60, 60))
    # panel 3: speed lines
    for i in range(70):
        y = 440 + i * 6
        d.line((460, y, 880, 440 + 210 + (y - 650) * 0.2), fill=(150, 150, 150), width=1)
    # panel 4: textured floor
    tex = (rng.normal(200, 18, (380, 860))).clip(0, 255).astype(np.uint8)
    img.paste(Image.fromarray(tex).filter(ImageFilter.GaussianBlur(1.2)).convert("RGB"), (20, 880))
    for p in panels:
        d.rectangle(p, outline=(0, 0, 0), width=4)

    items = []
    # bubble A: vertical, 3 columns (panel 1)
    bubble(d, (560, 40, 860, 380))
    items.append(dict(box=vtext(d, 760, 90, ["なんだよ、", "急に呼び出して", "…どうした？"], 30),
                      src="なんだよ、急に呼び出して…どうした？", tr="干嘛啊，突然把我叫出来……怎么了？"))
    # bubble B: vertical, 2 columns (panel 1)
    bubble(d, (60, 60, 300, 330))
    items.append(dict(box=vtext(d, 215, 110, ["見せたい", "ものがあるの"], 28),
                      src="見せたいものがあるの", tr="有样东西想给你看"))
    # bubble C: on screentone panel, vertical (panel 2)
    bubble(d, (240, 460, 430, 700))
    items.append(dict(box=vtext(d, 368, 500, ["ほんとに", "来たんだ"], 26),
                      src="ほんとに来たんだ", tr="你真的来了啊"))
    # narration: white-outlined text straight on screentone (panel 2) -> LaMa/inpaint path
    items.append(dict(box=vtext(d, 110, 470, ["その日", "から"], 24, fill=(0, 0, 0), stroke=(255, 255, 255), sw=3),
                      src="その日から", tr="从那天起"))
    # bubble D: horizontal (panel 3)
    bubble(d, (500, 600, 860, 800))
    items.append(dict(box=htext(d, 680, 700, ["ちょっと待って！", "話を聞いて！"], 28),
                      src="ちょっと待って！話を聞いて！", tr="等一下！听我说！"))
    # bubble E: big shout, vertical (panel 4)
    bubble(d, (560, 900, 860, 1240), lw=4)
    items.append(dict(box=vtext(d, 752, 950, ["絶対に", "許さない", "からな！"], 40),
                      src="絶対に許さないからな！", tr="我绝对不会原谅你的！"))
    # bubble F: small, vertical (panel 4)
    bubble(d, (80, 920, 330, 1150))
    items.append(dict(box=vtext(d, 232, 965, ["えっ…？", "なんで…"], 26),
                      src="えっ…？なんで…", tr="诶……？为什么……"))

    img.save(f"{out_dir}/test_page.png")
    json.dump(items, open(f"{out_dir}/test_page.json", "w"), ensure_ascii=False, indent=1)
    print("saved", len(items), "items")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
