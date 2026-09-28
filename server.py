"""Local translation server used by the Safari / Chrome extension.

  python server.py                 # http://0.0.0.0:8765  (reachable from your iPhone on the same Wi-Fi)
  MT_TOKEN=secret python server.py # require the same token in the extension settings

Open http://localhost:8765 in a browser for a drag-and-drop test page.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
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

app = FastAPI(title="manga-translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipe = Pipeline()
lock = threading.Lock()  # models are not thread-safe; pages are processed one at a time
TOKEN = os.environ.get("MT_TOKEN", "")
config.CACHE.mkdir(parents=True, exist_ok=True)

chapters = ChapterStore()
_read_cache: OrderedDict[str, list[str]] = OrderedDict()  # image sha1 -> OCR lines (look-ahead only)


def _decode(data: bytes | None):
    if not data:
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        return None


def _lines_of(data: bytes | None, src: str) -> list[str]:
    """OCR text of a look-ahead slice (cached, never translated here)."""
    rgb = _decode(data)
    if rgb is None:
        return []
    h = hashlib.sha1(data).hexdigest()
    if h not in _read_cache:
        _read_cache[h] = [b.text for b in pipe.read(rgb, src) if b.text]
        while len(_read_cache) > 64:
            _read_cache.popitem(last=False)
    return _read_cache[h]


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
              next2: UploadFile | None = File(None),
              x_mt_token: str | None = Header(default=None)):
    """Translate one slice.

    ctx/y: chapter (page URL) and the slice's vertical position -> earlier lines become context and
    this slice's lines are remembered. next1/next2: the next two slices -> their text is given to the
    model as look-ahead context, and a bubble cut by the seam to next1 is lettered across both slices.
    """
    _check(x_mt_token)
    data = image.file.read()
    if not data:
        raise HTTPException(400, "empty image")
    n1 = next1.file.read() if next1 else None
    n2 = next2.file.read() if next2 else None
    h = hashlib.sha1(data).hexdigest()
    n1h = hashlib.sha1(n1).hexdigest() if n1 else ""
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

    with lock:
        t0 = time.time()
        # this slice may already carry the lower half of a bubble lettered together with the slice above
        base, handled = chapters.get_carry(h)
        if base is None or base.shape != rgb.shape:
            base, handled = rgb, []
        H = base.shape[0]
        nxt = _decode(n1)
        stitch = (nxt is not None and nxt.shape[1] == base.shape[1]
                  and not (next1_y >= 0 and chapters.translated(ctx, next1_y)))
        page = np.vstack([base, nxt]) if stitch else base

        blocks = pipe.read(page, src)
        mine = [b for b in blocks if b.y1 < H - 2 and not any(_overlaps(b, box) for box in handled)]
        crossing = [b for b in mine if b.y2 > H + 2]
        ahead = [b.text for b in blocks if b.y1 >= H - 2 and b.text]
        if not stitch:
            ahead = _lines_of(n1, src)
        ahead += _lines_of(n2, src)
        history = chapters.history(ctx, y) if y >= 0 else []
        t = {"read": time.time() - t0}

        out_page, t2 = pipe.finish(page, mine, src, tgt, history or None, ahead or None)
        t.update(t2)
        out = out_page[:H]
        if stitch and crossing and n1h:
            chapters.put_carry(n1h, out_page[H:], [(b.x1, b.y1 - H, b.x2, b.y2 - H) for b in crossing])
        lines = [(b.text, b.translation) for b in mine if b.text]
        if y >= 0:
            chapters.record(ctx, y, lines)
        log_blocks(mine, t)
        if crossing:
            print(f"  {len(crossing)} bubble(s) cut by the seam lettered across both slices")

    buf = io.BytesIO()
    if has_alpha:
        Image.fromarray(out).save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")
    Image.fromarray(out).save(buf, "JPEG", quality=93, subsampling=0)
    cached.write_bytes(buf.getvalue())
    meta.write_text(json.dumps({"lines": lines}, ensure_ascii=False))
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"X-MT-Blocks": str(len(mine)), "X-MT-Time": f"{sum(t.values()):.1f}",
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


if __name__ == "__main__":
    port = int(os.environ.get("MT_PORT", 8765))
    print(f"manga-translator on http://0.0.0.0:{port}  (translation: {config.BACKEND} {config.MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=port)
