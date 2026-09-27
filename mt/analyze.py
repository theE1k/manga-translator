"""Per-block analysis: writing direction, glyph size, colours, speech-bubble shape.

This is what lets the re-lettered text match the original: same size, same colour,
same direction, centred in the real bubble instead of the tight OCR box.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import TextBlock

VERTICAL_SRC = {"ja", "zh", "zh-TW"}


def _runs(profile: np.ndarray, rel=0.08):
    """Lengths of 'on' runs in a projection profile, ignoring tiny gaps."""
    if profile.max() <= 0:
        return []
    on = profile > profile.max() * rel
    runs, n, gap = [], 0, 0
    for v in on:
        if v:
            n += 1 + gap if n else 1
            gap = 0
        elif n:
            gap += 1
            if gap > max(2, n * 0.15):
                runs.append(n)
                n, gap = 0, 0
    if n:
        runs.append(n)
    return [r for r in runs if r >= 3]


def analyze(rgb: np.ndarray, blocks: list[TextBlock], src_lang: str) -> None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    for b in blocks:
        m = b.mask[b.y1:b.y2, b.x1:b.x2] if b.mask is not None else None
        if m is None or m.sum() < 10:
            m = np.zeros((b.h, b.w), bool)
            crop = gray[b.y1:b.y2, b.x1:b.x2]
            m = crop < min(128, np.median(crop) - 40)
            if b.mask is None:
                b.mask = np.zeros((H, W), bool)
            b.mask[b.y1:b.y2, b.x1:b.x2] |= m

        # ---- direction + glyph size ----
        col_runs = _runs(m.sum(axis=0).astype(float))
        row_runs = _runs(m.sum(axis=1).astype(float))
        if src_lang in VERTICAL_SRC:
            if b.h > b.w * 1.25:
                b.vertical = True
            elif b.w > b.h * 1.6:
                b.vertical = False
            else:  # squarish: more distinct columns than rows -> vertical
                b.vertical = len(col_runs) >= len(row_runs)
        else:
            b.vertical = False
        runs = col_runs if b.vertical else row_runs
        if runs:
            b.font_px = float(np.median(runs)) / 0.88  # ink width -> em size
        else:
            b.font_px = float(min(b.w, b.h))
        b.font_px = float(np.clip(b.font_px, 8, max(8, min(b.w, b.h))))

        # ---- colours ----
        pad = int(max(4, b.font_px * 0.9))
        X1, Y1 = max(0, b.x1 - pad), max(0, b.y1 - pad)
        X2, Y2 = min(W, b.x2 + pad), min(H, b.y2 + pad)
        full = b.mask[Y1:Y2, X1:X2].astype(np.uint8)
        k1 = max(3, int(b.font_px * 0.25)) | 1
        k2 = max(k1 + 4, int(b.font_px * 0.8)) | 1
        inner = cv2.dilate(full, np.ones((k1, k1), np.uint8)) > 0
        outer = cv2.dilate(full, np.ones((k2, k2), np.uint8)) > 0
        region = rgb[Y1:Y2, X1:X2].reshape(-1, 3).astype(np.float32)
        # background = pixels in a band around the glyphs (between the columns + a margin)
        bg_px = region[(outer & ~inner).reshape(-1)]
        if len(bg_px) < 20:
            bg_px = region
        bg = np.median(bg_px, axis=0)
        flat_share = float(np.mean(np.linalg.norm(bg_px - bg, axis=1) < 30))
        txt_px = region[full.reshape(-1) > 0]
        if len(txt_px):
            dist = np.linalg.norm(txt_px - bg, axis=1)
            core = txt_px[dist >= np.percentile(dist, 60)]
            fg = np.median(core, axis=0)
        else:
            fg = np.array([0, 0, 0]) if bg.mean() > 127 else np.array([255, 255, 255])
        b.bg = tuple(int(v) for v in bg)
        b.fg = tuple(int(v) for v in fg)

        # outline: thin ring right outside the glyphs that differs from both fg and bg
        ring = (cv2.dilate(full, np.ones((5, 5), np.uint8)) > 0) & ~(full > 0)
        ring_px = region[ring.reshape(-1)]
        b.stroke = None
        b.in_bubble = flat_share > 0.9
        if not b.in_bubble and len(ring_px) > 20:
            rc = np.median(ring_px, axis=0)
            if np.linalg.norm(rc - fg) > 120:
                b.stroke = tuple(int(v) for v in rc)
            else:  # busy background and no clear outline: add a contrasting one for legibility
                b.stroke = (255, 255, 255) if fg.mean() < 128 else (0, 0, 0)

        # ---- bubble extent ----
        b.render_box = _bubble_box(rgb, b)


def _bubble_box(rgb: np.ndarray, b: TextBlock):
    """Find the flat-coloured area around the text (the bubble) and return a box inside it."""
    H, W = rgb.shape[:2]
    grow_x = int(max(b.w * 0.6, b.font_px * 2))
    grow_y = int(max(b.h * 0.6, b.font_px * 2))
    X1, Y1 = max(0, b.x1 - grow_x), max(0, b.y1 - grow_y)
    X2, Y2 = min(W, b.x2 + grow_x), min(H, b.y2 + grow_y)
    if not b.in_bubble:
        return (b.x1, b.y1, b.x2, b.y2)

    crop = rgb[Y1:Y2, X1:X2]
    mk = b.mask[Y1:Y2, X1:X2].astype(np.uint8)
    k = max(3, int(b.font_px * 0.3)) | 1
    near = (cv2.dilate(mk, np.ones((k, k), np.uint8)) > 0) & (mk == 0)
    diff = np.abs(crop.astype(np.int16) - np.array(b.bg, np.int16)).max(axis=2)
    flat = (diff < 30).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # cut thin leaks through gaps
    # the flat area right around the glyphs is the bubble; its convex hull covers the text itself
    n, lab = cv2.connectedComponents(flat, connectivity=4)
    ids = lab[near]
    ids = ids[ids > 0]
    if not len(ids):
        return (b.x1, b.y1, b.x2, b.y2)
    bub = lab == np.bincount(ids).argmax()
    # convex hull of the bubble = where it is safe to paint (never the outline or art outside)
    pts = cv2.findNonZero(bub.astype(np.uint8))
    hull = np.zeros(bub.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    hull = cv2.erode(hull, np.ones((3, 3), np.uint8)) > 0
    b.bubble = (X1, Y1, hull)

    # largest rectangle centred on the text centre that fits inside the bubble (try a few aspect ratios)
    ch, cw = hull.shape
    cx, cy = b.cx - X1, b.cy - Y1
    best, best_area = (int(cx), int(cy), int(cx) + 1, int(cy) + 1), 0
    text_aspect = max(0.2, b.w / max(1, b.h))
    for aspect in (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0):  # width / height
        lo, hi = 0.0, float(max(ch, cw))
        for _ in range(14):  # binary search the half-height
            mid = (lo + hi) / 2
            hx, hy = mid * aspect, mid
            rx1, rx2 = int(cx - hx), int(np.ceil(cx + hx))
            ry1, ry2 = int(cy - hy), int(np.ceil(cy + hy))
            if rx1 < 0 or ry1 < 0 or rx2 > cw or ry2 > ch:
                hi = mid
                continue
            # corners + edge midpoints are enough for convex shapes
            pts = [(ry1, rx1), (ry1, rx2 - 1), (ry2 - 1, rx1), (ry2 - 1, rx2 - 1),
                   (ry1, (rx1 + rx2) // 2), (ry2 - 1, (rx1 + rx2) // 2),
                   ((ry1 + ry2) // 2, rx1), ((ry1 + ry2) // 2, rx2 - 1)]
            if all(hull[p] for p in pts):
                lo = mid
            else:
                hi = mid
        hx, hy = lo * aspect, lo
        # prefer a shape close to the original text block: the translation replaces it 1:1
        area = hx * hy * (1 - 0.25 * min(1.0, abs(np.log(aspect / text_aspect))))
        if area > best_area:
            best_area = area
            best = (int(cx - hx), int(cy - hy), int(cx + hx), int(cy + hy))
    x1, y1, x2, y2 = best
    # small margin to the outline
    mx, my = int((x2 - x1) * 0.05), int((y2 - y1) * 0.04)
    x1, y1, x2, y2 = x1 + mx, y1 + my, x2 - mx, y2 - my
    if (x2 - x1) * (y2 - y1) < 0.5 * b.w * b.h:  # hull failed (odd shape) -> trust the text box
        x1, y1, x2, y2 = b.x1 - X1, b.y1 - Y1, b.x2 - X1, b.y2 - Y1
    # cap growth (the "bubble" could be a huge white page background)
    max_w, max_h = b.w * 1.6 + b.font_px, b.h * 1.4 + b.font_px
    if x2 - x1 > max_w:
        c = (x1 + x2) / 2; x1, x2 = c - max_w / 2, c + max_w / 2
    if y2 - y1 > max_h:
        c = (y1 + y2) / 2; y1, y2 = c - max_h / 2, c + max_h / 2
    return (int(x1 + X1), int(y1 + Y1), int(x2 + X1), int(y2 + Y1))
