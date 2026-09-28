"""Local translation server used by the Safari / Chrome extension.

  python server.py                 # http://0.0.0.0:8765  (reachable from your iPhone on the same Wi-Fi)
  MT_TOKEN=secret python server.py # require the same token in the extension settings

Open http://localhost:8765 in a browser for a drag-and-drop test page.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import threading
from contextlib import contextmanager
import time
from collections import OrderedDict

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from mt import Pipeline, config
from mt.chapter import ChapterStore
from mt.pipeline import log_blocks
from mt.typeset import render

app = FastAPI(title="manga-translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipe = Pipeline()
class PriorityLock:
    """One user of the local models at a time; waiting on-screen slices (prio 1) go before prefetch (0).
    Time spent waiting is added to t["wait"] so the log separates queueing from work."""

    def __init__(self):
        self._cv = threading.Condition()
        self._busy = False
        self._waiting = [0, 0]

    @contextmanager
    def hold(self, prio: int = 0, t: dict | None = None):
        prio = 1 if prio else 0
        t0 = time.time()
        with self._cv:
            self._waiting[prio] += 1
            while self._busy or (prio == 0 and self._waiting[1] > 0):
                self._cv.wait()
            self._waiting[prio] -= 1
            self._busy = True
        if t is not None:
            t["wait"] = t.get("wait", 0) + time.time() - t0
        try:
            yield
        finally:
            with self._cv:
                self._busy = False
                self._cv.notify_all()


cpu = PriorityLock()  # local models (detect / OCR / inpaint / letter); the API call runs outside it
TOKEN = os.environ.get("MT_TOKEN", "")
config.CACHE.mkdir(parents=True, exist_ok=True)

chapters = ChapterStore()
_reads: OrderedDict[str, list] = OrderedDict()  # image sha1 (+src) -> detected + OCR'd blocks
_inflight: dict[str, threading.Event] = {}     # slice sha1 -> set once the slice above it is done with the seam
_inflight_lock = threading.Lock()
SEAM_BAND, EDGE = 700, 6


def _decode(data: bytes | None):
    if not data:
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        return None


def _read(rgb, src: str, key: str) -> list:
    """Detect + OCR one image once; neighbours reuse it as look-ahead and later as the current slice.
    Call with the cpu lock held."""
    if key in _reads:
        _reads.move_to_end(key)
        return _reads[key]
    blocks = pipe.read(rgb, src)
    _reads[key] = blocks
    while len(_reads) > 24:
        _reads.popitem(last=False)
    return blocks


def _embed(b, height: int, dy: int):
    """Copy of a block moved down by dy into a page `height` px tall (coordinates and masks)."""
    nb = copy.copy(b)
    nb.y1, nb.y2 = b.y1 + dy, b.y2 + dy
    if b.render_box:
        x1, y1, x2, y2 = b.render_box
        nb.render_box = (x1, y1 + dy, x2, y2 + dy)
    if b.bubble is not None:
        bx, by, hull = b.bubble
        nb.bubble = (bx, by + dy, hull)
    if b.mask is not None:
        m = np.zeros((height, b.mask.shape[1]), bool)
        h = min(b.mask.shape[0], height - dy)
        m[dy:dy + h] = b.mask[:h]
        nb.mask = m
    return nb


def _overlaps(b, box, frac=0.3) -> bool:
    ix = max(0, min(b.x2, box[2]) - max(b.x1, box[0]))
    iy = max(0, min(b.y2, box[3]) - max(b.y1, box[1]))
    return ix * iy > frac * max(1, b.w * b.h)


def _check(token: str | None):
    if TOKEN and token != TOKEN:
        raise HTTPException(401, "bad token")


@app.get("/health")
def health(x_mt_token: str | None = Header(default=None)):
    _check(x_mt_token)
    return {"ok": True, "backend": config.BACKEND, "model": config.MODEL}


@app.post("/translate")
def translate(image: UploadFile = File(...), src: str = Form("ja"), tgt: str = Form("zh"),
              ctx: str = Form(""), y: float = Form(-1),
              next1: UploadFile | None = File(None), next1_y: float = Form(-1),
              next2: UploadFile | None = File(None), prio: int = Form(0),
              x_mt_token: str | None = Header(default=None)):
    """Translate one slice.

    ctx/y: chapter (page URL) and the slice's vertical position -> earlier lines become context and
    this slice's lines are remembered. next1/next2: the next two slices -> their text is look-ahead
    context, and a bubble cut by the seam to next1 is lettered across both slices.
    Several requests run at once: local models take turns (cpu lock), API calls overlap.
    """
    _check(x_mt_token)
    data = image.file.read()
    if not data:
        raise HTTPException(400, "empty image")
    n1 = next1.file.read() if next1 else None
    n2 = next2.file.read() if next2 else None
    h = hashlib.sha1(data).hexdigest()
    n1h = hashlib.sha1(n1).hexdigest() if n1 else ""
    n2h = hashlib.sha1(n2).hexdigest() if n2 else ""
    key = hashlib.sha1(f"{h}|{n1h}|{src}|{tgt}|{config.MODEL}".encode()).hexdigest()
    cached, meta = config.CACHE / f"{key}.jpg", config.CACHE / f"{key}.json"
    if cached.exists():
        if meta.exists() and y >= 0:
            chapters.record(ctx, y, json.loads(meta.read_text())["lines"])
        return Response(cached.read_bytes(), media_type="image/jpeg", headers={"X-MT-Cache": "hit"})
    try:
        im = Image.open(io.BytesIO(data))
        has_alpha = im.mode in ("RGBA", "LA", "P")
        rgb = np.asarray(im.convert("RGB"))
    except Exception:
        raise HTTPException(400, "not an image")
    if min(rgb.shape[:2]) < 64:
        raise HTTPException(400, "image too small")

    # the slice above may be lettering a bubble into our top edge: wait for it. Concurrent requests
    # can arrive out of order, so if our own text touches the top edge, give the slice above a moment
    # to show up before going it alone.
    t = {}
    ev = _inflight.get(h)
    if ev is None:
        with cpu.hold(prio, t):
            t0 = time.time()
            carried, _ = chapters.get_carry(h)
            top_cut = carried is None and any(b.y1 <= EDGE for b in _read(rgb, src, f"{h}|{src}|"))
            t["read"] = time.time() - t0
        deadline = time.time() + (3 if top_cut else 0)
        while ev is None and time.time() < deadline:
            time.sleep(0.1)
            ev = _inflight.get(h)
    if ev is not None:
        ev.wait(90)
    mine_done = threading.Event()
    if n1h:
        with _inflight_lock:
            _inflight[n1h] = mine_done
    try:
        with cpu.hold(prio, t):
            t0 = time.time()
            base, handled = chapters.get_carry(h)
            if base is None or base.shape != rgb.shape:
                base, handled = rgb, []
            H, W = base.shape[:2]
            own = _read(base, src, f"{h}|{src}|{'carry' if handled else ''}")
            mine = [b for b in own if not any(_overlaps(b, box) for box in handled)]
            nxt = _decode(n1)
            n1_blocks = _read(nxt, src, f"{n1h}|{src}|") if nxt is not None else []
            n2_rgb = _decode(n2)
            n2_blocks = _read(n2_rgb, src, f"{n2h}|{src}|") if n2_rgb is not None else []

            # a bubble cut by the seam: only then read the band around the seam and work on both slices
            page, crossing = base, []
            if (nxt is not None and nxt.shape[1] == W
                    and (any(b.y2 >= H - EDGE for b in mine) or any(b.y1 <= EDGE for b in n1_blocks))):
                B = min(SEAM_BAND, H, nxt.shape[0])
                band = pipe.read(np.vstack([base[H - B:], nxt[:B]]), src)
                cut = [b for b in band if b.y1 < B - 2 and b.y2 > B + 2]
                if cut:
                    page = np.vstack([base, nxt])
                    crossing = [_embed(b, page.shape[0], H - B) for b in cut]
                    mine = [_embed(b, page.shape[0], 0) for b in mine
                            if not any(_overlaps(b, c.box) for c in crossing)]
            # the slice below was translated first (reader jumped ahead): it already carries the
            # bubble's translation, so here we only erase our half
            below_done = bool(crossing) and next1_y >= 0 and chapters.translated(ctx, next1_y)
            erase_only = crossing if below_done else []
            if not below_done:
                mine += crossing
            ahead = [b.text for b in n1_blocks if b.text and not (crossing and b.y1 <= EDGE)]
            ahead += [b.text for b in n2_blocks if b.text]
            if y >= 0 and not chapters.translated(ctx, y):  # source lines -> context for slices in flight
                chapters.record(ctx, y, [(b.text, "") for b in mine if b.text])
            t["read"] = t.get("read", 0) + time.time() - t0
        if not crossing or below_done:
            mine_done.set()  # nothing to hand down: the slice below need not wait

        history = chapters.history(ctx, y) if y >= 0 else []
        t0 = time.time()
        pipe.translator(mine, src, tgt, history or None, ahead or None)
        t["translate"] = time.time() - t0

        with cpu.hold(prio, t):
            t0 = time.time()
            done = [b for b in mine if b.translation]
            out_page = render(pipe.inpainter(page, done + erase_only), done, tgt)
            t["letter"] = time.time() - t0
        out = out_page[:H]
        if crossing and not below_done and n1h:
            chapters.put_carry(n1h, out_page[H:], [(c.x1, c.y1 - H, c.x2, c.y2 - H) for c in crossing])
        lines = [(b.text, b.translation) for b in mine if b.text]
        if y >= 0:
            chapters.record(ctx, y, lines)
        log_blocks(mine, t)
        if crossing:
            print(f"  {len(crossing)} bubble(s) cut by the seam" + (" (erased only)" if below_done else ""))
    finally:
        mine_done.set()
        with _inflight_lock:
            if _inflight.get(n1h) is mine_done:
                del _inflight[n1h]

    buf = io.BytesIO()
    if has_alpha:
        Image.fromarray(out).save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")
    Image.fromarray(out).save(buf, "JPEG", quality=93, subsampling=0)
    cached.write_bytes(buf.getvalue())
    meta.write_text(json.dumps({"lines": lines}, ensure_ascii=False))
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"X-MT-Blocks": str(len(mine)), "X-MT-Time": f"{sum(t.values()):.1f}", "X-MT-Wait": f"{t.get('wait', 0):.1f}",
                             "X-MT-Crossing": str(len(crossing)), "X-MT-History": str(len(history)),
                             "X-MT-Ahead": str(len(ahead))})


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Manga Translator</title>
<style>body{font:15px -apple-system,sans-serif;margin:16px;background:#111;color:#eee}
.row{display:flex;gap:12px;flex-wrap:wrap}img{max-width:min(48vw,700px);height:auto}
select,input,button{font-size:15px;margin-right:8px}</style>
<h3>Manga Translator</h3>
<input type=file id=f accept="image/*"> from <select id=s><option>ja<option>ko<option>zh<option>en</select>
to <select id=t><option>zh<option>zh-TW<option>en<option>ja<option>ko</select>
<button onclick=go()>Translate</button> <span id=st></span>
<div class=row><img id=a><img id=b></div>
<script>
async function go(){const file=f.files[0];if(!file)return;a.src=URL.createObjectURL(file);st.textContent='working...';
const fd=new FormData();fd.append('image',file);fd.append('src',s.value);fd.append('tgt',t.value);
const t0=Date.now();const r=await fetch('/translate',{method:'POST',body:fd});
if(!r.ok){st.textContent='error '+r.status;return}
b.src=URL.createObjectURL(await r.blob());st.textContent=((Date.now()-t0)/1000).toFixed(1)+'s'}
</script>"""


def warmup():
    """Load and exercise every local model before the first page arrives (first request would
    otherwise pay ~10 s: detector compile, OCR, inpainting on the GPU, jieba, font, API metadata)."""
    t0 = time.time()
    with cpu.hold(1):
        blank = np.full((1200, 690, 3), 255, np.uint8)
        pipe.detector.detect(blank)
        # a line of real text, so the OCR recogniser (not just its line finder) is loaded too
        from PIL import ImageDraw, ImageFont
        sample = Image.new("RGB", (690, 200), "white")
        ImageDraw.Draw(sample).text((40, 70), "오늘은 정말 긴 하루였어", fill="black",
                                    font=ImageFont.truetype(str(config.FONT_PATH), 40))
        sample = np.asarray(sample)
        for lang in os.environ.get("MT_WARM_LANGS", "ko").split(","):
            if lang and lang != "ja":
                pipe.ocr.paddle(lang)(np.ascontiguousarray(sample[:, :, ::-1]))
        pipe.inpainter.warmup()
        import jieba
        jieba.setLogLevel(60)
        jieba.initialize()
        render(blank[:200].copy(), [], "zh")
        from mt.typeset import _font
        _font(30)
    from mt.translate import _supports
    tr = pipe.translator
    _supports(tr.url, tr.model, "reasoning")
    print(f"warm-up done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    threading.Thread(target=warmup, daemon=True).start()
    port = int(os.environ.get("MT_PORT", 8765))
    print(f"manga-translator on http://0.0.0.0:{port}  (translation: {config.BACKEND} {config.MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=port)
