"""Per-chapter memory for the extension: which slice says what, in page order, persisted to disk.

A chapter is identified by the reading page URL (``ctx``); a slice by its vertical position on that
page (``y``, CSS px). Ordering by position rather than by request time keeps the context right even
when slices are translated out of order (scrolling back, jumping ahead).

Also stores "carry-over" images: when a speech bubble is cut by the seam between two slices, the
server letters it on the two slices stitched together and keeps the lower slice's already-lettered
pixels here until that slice is requested.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

ROOT = config.CACHE / "chapters"
CARRY = config.CACHE / "carry"
MAX_AGE_DAYS = 30


def _key(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


class ChapterStore:
    def __init__(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        CARRY.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._prune()

    # ---------------------------------------------------------------- chapters
    def _load(self, ctx: str) -> dict:
        k = _key(ctx)
        if k not in self._mem:
            f = ROOT / f"{k}.json"
            try:
                self._mem[k] = json.loads(f.read_text()) if f.exists() else {"ctx": ctx, "slices": {}}
            except (OSError, ValueError):
                self._mem[k] = {"ctx": ctx, "slices": {}}
        return self._mem[k]

    def record(self, ctx: str, y: float, lines: list[tuple[str, str]]) -> None:
        """Remember the (source, translation) lines of the slice at position y (translation may be '')."""
        if not ctx:
            return
        with self._lock:
            ch = self._load(ctx)
            ch["slices"][str(round(y))] = [list(p) for p in lines]
            ch["updated"] = time.time()
            (ROOT / f"{_key(ctx)}.json").write_text(json.dumps(ch, ensure_ascii=False))

    def translated(self, ctx: str, y: float) -> bool:
        with self._lock:
            lines = self._load(ctx)["slices"].get(str(round(y))) if ctx else None
        return bool(lines) and any(t for _, t in lines)

    def history(self, ctx: str, y: float, n: int = 20) -> list[tuple[str, str]]:
        """The last n lines of the slices above position y, in page order."""
        if not ctx:
            return []
        with self._lock:
            sl = self._load(ctx)["slices"]
            out = [tuple(p) for k in sorted(sl, key=float) if float(k) < y - 1 for p in sl[k]]
        return out[-n:]

    # ---------------------------------------------------------------- carry-over images
    def put_carry(self, img_hash: str, rgb: np.ndarray, handled: list[tuple]) -> None:
        Image.fromarray(rgb).save(CARRY / f"{img_hash}.png")
        (CARRY / f"{img_hash}.json").write_text(json.dumps({"handled": handled}))

    def get_carry(self, img_hash: str):
        f = CARRY / f"{img_hash}.png"
        if not f.exists():
            return None, []
        meta = json.loads((CARRY / f"{img_hash}.json").read_text())
        return np.asarray(Image.open(f).convert("RGB")), [tuple(b) for b in meta["handled"]]

    def _prune(self):
        cutoff = time.time() - MAX_AGE_DAYS * 86400
        for d in (ROOT, CARRY):
            for f in d.iterdir():
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
