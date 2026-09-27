"""Text detection with comic-text-detector (ONNX).

Model: https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx
It returns (a) YOLO boxes for text blocks and (b) a pixel-level text segmentation mask.
Tall webtoon strips are processed in overlapping tiles so small text is not shrunk away.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .types import TextBlock


class TextDetector:
    def __init__(self, model_path=config.DETECTOR_ONNX, size=config.DET_SIZE):
        import onnxruntime as ort

        providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        try:
            self.sess = ort.InferenceSession(str(model_path), providers=providers)
        except Exception:  # CoreML can reject some graphs; CPU always works
            self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size

    # ---------- single square-ish image ----------
    def _run(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        s = self.size
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.zeros((s, s, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        outs = self.sess.run(None, {self.input_name: blob})

        blks, seg = None, None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] >= 6:
                blks = o[0]
            elif o.ndim == 4 and o.shape[1] == 1:
                seg = o[0, 0]
        if blks is None or seg is None:
            raise RuntimeError(f"Unexpected detector outputs: {[o.shape for o in outs]}")

        if seg.max() > 1.0 or seg.min() < 0.0:  # logits -> prob
            seg = 1 / (1 + np.exp(-seg))
        seg = seg[:nh, :nw]
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)

        # YOLO rows: cx, cy, bw, bh, obj, cls...
        obj = blks[:, 4]
        cls = blks[:, 5:].max(axis=1) if blks.shape[1] > 5 else np.ones_like(obj)
        conf = obj * cls
        keep = conf > config.DET_CONF
        b, conf = blks[keep, :4], conf[keep]
        boxes = []
        if len(b):
            x1 = (b[:, 0] - b[:, 2] / 2) / r
            y1 = (b[:, 1] - b[:, 3] / 2) / r
            bw, bh = b[:, 2] / r, b[:, 3] / r
            rects = [[float(a), float(c), float(d), float(e)] for a, c, d, e in zip(x1, y1, bw, bh)]
            idx = cv2.dnn.NMSBoxes(rects, conf.tolist(), config.DET_CONF, config.DET_NMS)
            for i in np.array(idx).flatten():
                x, y, ww, hh = rects[i]
                boxes.append([max(0, x), max(0, y), min(w, x + ww), min(h, y + hh), float(conf[i])])
        return boxes, seg

    # ---------- public ----------
    def detect(self, rgb: np.ndarray) -> tuple[list[TextBlock], np.ndarray]:
        """Returns text blocks and a full-page float text mask (0..1)."""
        h, w = rgb.shape[:2]
        tiles = _tiles(h, w)
        seg_full = np.zeros((h, w), np.float32)
        boxes = []
        for (ty1, ty2, tx1, tx2) in tiles:
            bxs, seg = self._run(rgb[ty1:ty2, tx1:tx2])
            seg_full[ty1:ty2, tx1:tx2] = np.maximum(seg_full[ty1:ty2, tx1:tx2], seg)
            for x1, y1, x2, y2, c in bxs:
                boxes.append([x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1, c])
        boxes = _dedupe(boxes)
        mask_bin = seg_full > config.MASK_THRESH
        blocks = [TextBlock(int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2)), score=c)
                  for x1, y1, x2, y2, c in boxes]
        blocks = [b for b in blocks if b.w >= 6 and b.h >= 6]
        attach_masks(blocks, mask_bin)
        return blocks, seg_full


def attach_masks(blocks: list[TextBlock], mask_bin: np.ndarray) -> None:
    """Give every block its own glyph mask (page-sized bool array), limited to a padded box."""
    h, w = mask_bin.shape
    for b in blocks:
        pad = max(2, int(0.06 * min(b.w, b.h)))
        x1, y1 = max(0, b.x1 - pad), max(0, b.y1 - pad)
        x2, y2 = min(w, b.x2 + pad), min(h, b.y2 + pad)
        m = np.zeros((h, w), bool)
        m[y1:y2, x1:x2] = mask_bin[y1:y2, x1:x2]
        b.mask = m


def _tiles(h, w):
    """Split very tall (webtoon) or very wide pages into overlapping square-ish tiles."""
    if h <= w * 2 and w <= h * 2:
        return [(0, h, 0, w)]
    out = []
    if h > w:
        th = int(w * 1.5)
        step = int(th * 0.8)
        y = 0
        while True:
            y2 = min(h, y + th)
            out.append((max(0, y2 - th), y2, 0, w))
            if y2 >= h:
                break
            y += step
    else:
        tw = int(h * 1.5)
        step = int(tw * 0.8)
        x = 0
        while True:
            x2 = min(w, x + tw)
            out.append((0, h, max(0, x2 - tw), x2))
            if x2 >= w:
                break
            x += step
    return out


def _dedupe(boxes):
    """Drop duplicates from overlapping tiles: a box mostly inside a bigger box is removed."""
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in boxes:
        area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for k in kept:
            ix = max(0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            if ix * iy / area > 0.6:
                # extend the kept box if the smaller one pokes out (text cut at a tile seam)
                k[0], k[1] = min(k[0], b[0]), min(k[1], b[1])
                k[2], k[3] = max(k[2], b[2]), max(k[3], b[3])
                dup = True
                break
        if not dup:
            kept.append(list(b))
    return kept
