"""Remove the original lettering.

* Text on a flat bubble: refine the glyph mask, paint the bubble colour -> pixel-perfect.
* Text on a flat colour or smooth gradient outside a bubble: fill with the fitted colour plane (instant).
* Text on art / screentone: LaMa on a crop around the text.
  Preferred: TorchScript big-lama on the Apple GPU (MPS), crop size 256-512 by region size.
  Fallback: ONNX LaMa on the CPU (https://huggingface.co/Carve/LaMa-ONNX, fixed 512x512), then OpenCV Telea.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .types import TextBlock

LAMA_BUCKETS = (256, 384, 512)


def refine_mask(rgb: np.ndarray, b: TextBlock) -> np.ndarray:
    """Grow the detector mask to whole glyphs (incl. anti-aliased edges) without touching bubble outlines."""
    H, W = rgb.shape[:2]
    pad = int(max(3, b.font_px * 0.3))
    X1, Y1 = max(0, b.x1 - pad), max(0, b.y1 - pad)
    X2, Y2 = min(W, b.x2 + pad), min(H, b.y2 + pad)
    crop = rgb[Y1:Y2, X1:X2].astype(np.int16)
    base = b.mask[Y1:Y2, X1:X2].astype(np.uint8)

    dist = np.abs(crop - np.array(b.bg, np.int16)).max(axis=2)
    cand = (dist > 35).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    near = cv2.dilate(base, np.ones((3, 3), np.uint8)) > 0
    keep = np.zeros(n, bool)
    lim = b.font_px * 1.6
    hit = np.unique(lab[near & (cand > 0)])
    for i in hit:
        if i == 0:
            continue
        x, y, w, h, area = stats[i]
        if w <= lim and h <= lim or (b.stroke is not None):  # glyph-sized piece, not a bubble outline
            keep[i] = True
    # never erase big dark shapes (bubble outlines, panel borders) even if the detector mask touched them
    big = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        big[i] = (w > lim * 2.5 or h > lim * 2.5) and b.stroke is None
    glyph = keep[lab] | ((base > 0) & ~big[lab])
    k = max(3, int(b.font_px * (0.10 if b.in_bubble else 0.35 if b.stroke is not None else 0.18))) | 1
    glyph = cv2.dilate(glyph.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    out = np.zeros((H, W), bool)
    out[Y1:Y2, X1:X2] = glyph
    return out


class Inpainter:
    def __init__(self, model_path=config.LAMA_ONNX, torch_path=config.LAMA_TORCH):
        self.sess = None
        self.torch_model = None
        if torch_path and torch_path.exists():
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.torch = torch
                    self.torch_model = torch.jit.load(str(torch_path), map_location="cpu").eval().to("mps")
            except Exception as e:  # pragma: no cover
                print(f"[inpaint] big-lama on MPS unavailable ({e}); using ONNX on CPU")
                self.torch_model = None
        if self.torch_model is None and model_path and model_path.exists():
            import onnxruntime as ort
            try:
                self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                ins = self.sess.get_inputs()
                self.in_img = next((i.name for i in ins if "mask" not in i.name.lower()), ins[0].name)
                self.in_mask = next((i.name for i in ins if "mask" in i.name.lower()), ins[1].name)
            except Exception as e:  # pragma: no cover
                print(f"[inpaint] LaMa unavailable ({e}); using OpenCV fallback")
                self.sess = None

    def warmup(self):
        """The GPU compiles a graph per crop size (~1 s each): do all sizes before the first real page."""
        for side in LAMA_BUCKETS:
            crop = np.full((side, side, 3), 128, np.uint8)
            crop[::7] = 0  # textured, so the plane fill would not short-cut it
            cm = np.zeros((side, side), np.uint8)
            cm[side // 3:side // 2, side // 5:side * 4 // 5] = 255
            self._fill(crop, cm)

    def __call__(self, rgb: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
        out = rgb.copy()
        heavy = []
        for b in blocks:
            m = refine_mask(rgb, b)
            if b.in_bubble:
                if b.bubble is not None:  # only paint inside the bubble
                    bx, by, hull = b.bubble
                    safe = np.zeros_like(m)
                    safe[by:by + hull.shape[0], bx:bx + hull.shape[1]] = hull
                    m &= safe
                out[m] = b.bg
            else:
                heavy.append(m)
        if heavy:
            mask = np.zeros(rgb.shape[:2], np.uint8)
            for m in heavy:
                mask[m] = 255
            out = self._inpaint_regions(out, mask)
        return out

    def _inpaint_regions(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((15, 15), np.uint8)))
        H, W = mask.shape
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            side = int(max(w, h) * 1.8) + 32
            cx, cy = x + w // 2, y + h // 2
            X1, Y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            X2, Y2 = min(W, X1 + side), min(H, Y1 + side)
            X1, Y1 = max(0, X2 - side), max(0, Y2 - side)
            crop, cm = img[Y1:Y2, X1:X2], mask[Y1:Y2, X1:X2]
            if not cm.any():
                continue
            filled = _plane_fill(crop, cm)
            if filled is None:
                filled = self._fill(crop, cm)
            # feathered composite so the patch edge is invisible
            alpha = cv2.GaussianBlur(cv2.dilate(cm, np.ones((3, 3), np.uint8)).astype(np.float32) / 255, (5, 5), 0)
            alpha = np.maximum(alpha, cm.astype(np.float32) / 255)[..., None]
            img[Y1:Y2, X1:X2] = (filled * alpha + crop * (1 - alpha)).astype(np.uint8)
        return img

    def _fill(self, crop: np.ndarray, cm: np.ndarray) -> np.ndarray:
        if self.torch_model is not None:
            return self._lama_mps(crop, cm)
        if self.sess is not None:
            return self._lama(crop, cm)
        return cv2.inpaint(crop, cm, 5, cv2.INPAINT_TELEA)

    def _lama_mps(self, crop: np.ndarray, cm: np.ndarray) -> np.ndarray:
        h, w = cm.shape
        # three fixed sizes: the GPU compiles a graph per input shape, so arbitrary sizes would recompile
        S = next(b for b in LAMA_BUCKETS if max(h, w) <= b or b == LAMA_BUCKETS[-1])
        im = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA if max(h, w) > S else cv2.INTER_CUBIC)
        mk = (cv2.resize(cm, (S, S), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(im.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to("mps")
            m = t.from_numpy(mk[None, None]).to("mps")
            y = self.torch_model(x, m)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        return cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)

    def _lama(self, crop: np.ndarray, cm: np.ndarray) -> np.ndarray:
        h, w = cm.shape
        S = 512
        im = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA if max(h, w) > S else cv2.INTER_CUBIC)
        mk = (cv2.resize(cm, (S, S), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        x = im.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        y = self.sess.run(None, {self.in_img: x, self.in_mask: mk[None, None]})[0][0]
        y = y.transpose(1, 2, 0)
        if y.max() <= 1.5:
            y = y * 255
        y = np.clip(y, 0, 255).astype(np.uint8)
        return cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)


def _plane_fill(crop: np.ndarray, cm: np.ndarray, max_resid: float = 6.0):
    """If the background around the text is a flat colour or a smooth linear gradient, return the
    crop with the masked pixels painted from that fitted plane; None when the background has texture."""
    inner = cv2.dilate(cm, np.ones((5, 5), np.uint8))
    ring = (cv2.dilate(cm, np.ones((17, 17), np.uint8)) > 0) & (inner == 0)
    ys, xs = np.nonzero(ring)
    if len(xs) < 40:
        return None
    A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float32)
    vals = crop[ys, xs].astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    if (vals - A @ coef).std(axis=0).max() > max_resid:
        return None
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    plane = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float32) @ coef
    out = crop.copy()
    sel = inner > 0
    out[sel] = np.clip(plane[sel], 0, 255).astype(np.uint8)
    return out
