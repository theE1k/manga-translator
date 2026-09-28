"""Re-lettering: put the translation back into the bubble so it looks original.

* keeps the original direction (vertical CJK stays vertical) when the target language allows it
* matches the original glyph size, colour and outline
* balanced line/column lengths (bubble-shaped, not ragged), CJK line-break rules
* vertical punctuation forms (，。「」…) and full-width Latin in vertical text
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .types import TextBlock

VERTICAL_TGT = {"zh", "zh-TW", "ja"}

NO_START = set("，。、．：；！？）」』】》〉’”…ー―～·,.!?:;)]}>%ぁぃぅぇぉっゃゅょァィゥェォッャュョ々〜")
NO_END = set("（「『【《〈‘“([{<")

VERT_MAP = {
    "，": "︐", "、": "︑", "。": "︒", "：": "︓", "；": "︔", "！": "︕", "？": "︖",
    "「": "﹁", "」": "﹂", "『": "﹃", "』": "﹄", "（": "︵", "）": "︶", "【": "︻", "】": "︼",
    "《": "︽", "》": "︾", "〈": "︿", "〉": "﹀", "…": "︙", "‥": "︰", "—": "︱", "―": "︱",
    "～": "≀", "〜": "≀", "ー": "丨", "-": "丨", "“": "﹁", "”": "﹂", "‘": "﹃", "’": "﹄",
}
PAIRS = {"！？": "⁉", "？！": "⁈", "！！": "‼", "？？": "⁇"}


@lru_cache(maxsize=64)
def _font(size: int, path: str = str(config.FONT_PATH)):
    try:
        return ImageFont.truetype(path, size, index=config.FONT_INDEX)
    except OSError:
        for p in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default(size)


def _to_fullwidth(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch == " ":
            out.append("　")
        elif 0x21 <= o <= 0x7E:
            out.append(chr(o + 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def _is_cjk_text(s: str) -> bool:
    n = sum(1 for c in s if unicodedata.east_asian_width(c) in "WF")
    return n >= max(1, len(s) * 0.3)


# ---------------------------------------------------------------- tokenising / wrapping
# numbers / Latin words that must never be split across lines: 98.6%  1,196  2011.3.18  JBSN
_UNBREAKABLE = re.compile(r"[0-9０-９A-Za-z][0-9０-９A-Za-z.,．，:：/%％\-]*[0-9０-９A-Za-z%％]|[0-9０-９A-Za-z]")
_KANA = re.compile(r"[\u3040-\u30ff]")


def _words(text: str) -> list[str]:
    """Chinese text -> words (jieba) so lines break between words; other CJK -> characters.
    Number/Latin runs are always kept whole."""
    out = []
    pos = 0
    for m in _UNBREAKABLE.finditer(text):
        out += _cjk_pieces(text[pos:m.start()])
        out.append(m.group())
        pos = m.end()
    return out + _cjk_pieces(text[pos:])


def _cjk_pieces(seg: str) -> list[str]:
    if not seg:
        return []
    if _KANA.search(seg) is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            return [w for w in jieba.lcut(seg, HMM=True) if w]
        except ImportError:
            pass
    return list(seg)


def _tokens(text: str, cjk: bool, words: bool = True) -> list[str]:
    if not cjk:
        return [w for w in text.split(" ") if w]
    toks = []
    pieces = _words(text) if words else _chars_keep_runs(text)
    for piece in pieces:
        if piece[0] in NO_START and toks:
            toks[-1] += piece  # punctuation sticks to the previous token
        elif toks and toks[-1] and toks[-1][-1] in NO_END:
            toks[-1] += piece
        elif toks and len(piece) == 1 and _UNBREAKABLE.fullmatch(toks[-1]) and _is_cjk_text(piece):
            toks[-1] += piece  # keep a counter with its number: 2011年, 1196名
        elif piece.strip():
            toks.append(piece)
    return toks


def _chars_keep_runs(text: str) -> list[str]:
    """Per-character pieces, except number/Latin runs stay whole."""
    out, pos = [], 0
    for m in _UNBREAKABLE.finditer(text):
        out += list(text[pos:m.start()])
        out.append(m.group())
        pos = m.end()
    return out + list(text[pos:])


def _phrases(tokens: list[str]) -> list[str]:
    """Group CJK char tokens into phrases that end at punctuation (how letterers break lines)."""
    out, cur = [], ""
    for t in tokens:
        cur += t
        if t[-1] in NO_START or t[-1] in "　 ":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def _wrap(tokens, measure, limit, joiner):
    lines, cur = [], ""
    expanded = []
    for t in tokens:  # too long for a line: phrase -> words -> characters
        if not joiner and len(t) > 1 and measure(t) > limit:
            for wd in _tokens(t, True):
                if len(wd) > 1 and measure(wd) > limit:
                    expanded.extend(_tokens(wd, True, words=False))
                else:
                    expanded.append(wd)
        else:
            expanded.append(t)
    for t in expanded:
        cand = cur + joiner + t if cur else t
        if cur and measure(cand) > limit:
            lines.append(cur)
            cur = t
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _balanced_wrap(tokens, measure, limit, joiner):
    """Wrap to the fewest lines, then shrink the width so the lines come out even."""
    lines = _wrap(tokens, measure, limit, joiner)
    if len(lines) <= 1:
        return lines
    lo, hi, best = 1.0, float(limit), lines
    for _ in range(12):
        mid = (lo + hi) / 2
        cand = _wrap(tokens, measure, mid, joiner)
        if len(cand) <= len(lines) and max(measure(l) for l in cand) <= mid + 0.5:
            best, hi = cand, mid
        else:
            lo = mid
    return best


# ---------------------------------------------------------------- layouts
def _layout_h(text, size, box_w, cjk, phrase=False):
    f = _font(size)
    joiner = "" if cjk else " "
    toks = _tokens(text, cjk)
    if cjk and phrase:
        toks = _phrases(toks)
    lines = _balanced_wrap(toks, f.getlength, box_w, joiner)
    lh = size * (1.18 if cjk else 1.12)
    w = max((f.getlength(l) for l in lines), default=0)
    return lines, w, lh * len(lines) - (lh - size), lh


def _vert_text(text):
    t = _to_fullwidth(text.replace("...", "…"))
    for a, b in PAIRS.items():
        t = t.replace(a, b)
    t = t.replace("　", "")
    return t


def _layout_v(text, size, box_h, phrase=False):
    chars = list(_vert_text(text))
    toks = _tokens("".join(chars), True)
    if phrase:
        toks = _phrases(toks)
    adv = size * 1.02
    cols = _balanced_wrap(toks, lambda s: len(s) * adv, box_h, "")
    col_w = size * 1.22
    w = col_w * len(cols) - (col_w - size)
    h = max((len(c) for c in cols), default=0) * adv
    return cols, w, h, adv, col_w


def _fit(fn, box_w, box_h, cap, floor=9):
    lo, hi, best = floor, max(floor, int(cap)), None
    while lo <= hi:
        mid = (lo + hi) // 2
        res = fn(mid)
        if res[1] <= box_w and res[2] <= box_h:
            best = (mid, res)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        best = (floor, fn(floor))
    return best


def _best(fit_fn):
    """Prefer breaking at punctuation unless that costs more than ~12% of the font size."""
    plain = fit_fn(False)
    phrased = fit_fn(True)
    return phrased if phrased[0] >= plain[0] * 0.88 else plain


# ---------------------------------------------------------------- drawing
def render(img: np.ndarray, blocks: list[TextBlock], tgt: str) -> np.ndarray:
    pil = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for b in blocks:
        text = (b.translation or "").strip()
        if not text:
            continue
        x1, y1, x2, y2 = b.render_box or b.box
        bw, bh = max(8, x2 - x1), max(8, y2 - y1)
        cap = b.font_px * 1.05
        cjk = _is_cjk_text(text)
        vertical = b.vertical and tgt in VERTICAL_TGT and cjk
        sw = 0
        if b.stroke is not None:
            sw = max(1, int(round(b.font_px * 0.1)))
        fg = tuple(b.fg)

        if vertical:
            size, (cols, w, h, adv, col_w) = _best(
                lambda ph: _fit(lambda s: _layout_v(text, s, bh, ph), bw - 2 * sw, bh - 2 * sw, cap))
            sw = max(1, int(round(size * 0.1))) if b.stroke is not None else 0
            f = _font(size)
            left = x1 + (bw - w) / 2
            top = y1 + (bh - h) / 2
            glyphs = []
            for ci, col in enumerate(cols):
                cx = left + w - size / 2 - ci * col_w  # right-to-left
                for ri, ch in enumerate(col):
                    ch = VERT_MAP.get(ch, ch)
                    glyphs.append((cx, top + ri * adv + adv / 2, ch))
            for pass_ in ("stroke", "fill"):
                for cx, cy, ch in glyphs:
                    _draw_char(draw, f, cx, cy, ch, fg, b.stroke, sw, pass_)
        else:
            size, (lines, w, h, lh) = _best(
                lambda ph: _fit(lambda s: _layout_h(text, s, bw - 2 * sw, cjk, ph), bw - 2 * sw, bh - 2 * sw, cap))
            sw = max(1, int(round(size * 0.1))) if b.stroke is not None else 0
            f = _font(size)
            top = y1 + (bh - h) / 2
            for pass_ in ("stroke", "fill"):
                for i, line in enumerate(lines):
                    cx = x1 + bw / 2
                    cy = top + i * lh + size / 2
                    if pass_ == "stroke" and sw:
                        draw.text((cx, cy), line, font=f, fill=b.stroke, anchor="mm",
                                  stroke_width=sw, stroke_fill=b.stroke)
                    elif pass_ == "fill":
                        draw.text((cx, cy), line, font=f, fill=fg, anchor="mm")
    return np.asarray(pil)


def _draw_char(draw, f, cx, cy, ch, fg, stroke, sw, pass_):
    rotate = unicodedata.east_asian_width(ch) not in "WFA" and ch not in "⁉⁈‼⁇≀丨"
    if rotate:  # stray half-width symbol: rotate 90° like proper vertical typesetting
        size = f.size
        tmp = Image.new("L", (size * 2, size * 2), 0)
        ImageDraw.Draw(tmp).text((size, size), ch, font=f, fill=255, anchor="mm")
        tmp = tmp.rotate(-90)
        color = fg if pass_ == "fill" else stroke
        if pass_ == "stroke" and not sw:
            return
        draw.bitmap((cx - size, cy - size), tmp, fill=color)
        return
    if pass_ == "stroke":
        if sw:
            draw.text((cx, cy), ch, font=f, fill=stroke, anchor="mm", stroke_width=sw, stroke_fill=stroke)
    else:
        draw.text((cx, cy), ch, font=f, fill=fg, anchor="mm")
