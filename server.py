"""Local translation server used by the Safari / Chrome extension.

  python server.py                 # http://0.0.0.0:8765  (reachable from your iPhone on the same Wi-Fi)
  MT_TOKEN=secret python server.py # require the same token in the extension settings

Open http://localhost:8765 in a browser for a drag-and-drop test page.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
from collections import OrderedDict

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from mt import Pipeline, config

app = FastAPI(title="manga-translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipe = Pipeline()
lock = threading.Lock()  # models are not thread-safe; pages are processed one at a time
TOKEN = os.environ.get("MT_TOKEN", "")
config.CACHE.mkdir(parents=True, exist_ok=True)

# Per-chapter translation memory: the extension sends the reading page URL as `ctx`, so each new
# slice is translated knowing the dialogue before it (names, tone, running jokes stay consistent).
HISTORY: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
HISTORY_LINES, HISTORY_CHAPTERS = 20, 50


def _history(ctx: str) -> list[tuple[str, str]]:
    if not ctx:
        return []
    HISTORY.move_to_end(ctx) if ctx in HISTORY else HISTORY.setdefault(ctx, [])
    while len(HISTORY) > HISTORY_CHAPTERS:
        HISTORY.popitem(last=False)
    return HISTORY[ctx]


def _check(token: str | None):
    if TOKEN and token != TOKEN:
        raise HTTPException(401, "bad token")


@app.get("/health")
def health(x_mt_token: str | None = Header(default=None)):
    _check(x_mt_token)
    return {"ok": True, "backend": config.BACKEND, "model": config.MODEL}


@app.post("/translate")
def translate(image: UploadFile = File(...), src: str = Form("ja"), tgt: str = Form("zh"),
              ctx: str = Form(""), x_mt_token: str | None = Header(default=None)):
    _check(x_mt_token)
    data = image.file.read()
    if not data:
        raise HTTPException(400, "empty image")
    key = hashlib.sha1(data + f"|{src}|{tgt}|{config.MODEL}".encode()).hexdigest()
    cached = config.CACHE / f"{key}.jpg"
    if cached.exists():
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
        hist = _history(ctx)
        out, blocks, t = pipe.run(rgb, src, tgt, history=hist[-HISTORY_LINES:] or None)
        hist.extend((b.text, b.translation) for b in blocks if b.text and b.translation)
        del hist[:-HISTORY_LINES]
    buf = io.BytesIO()
    if has_alpha:
        Image.fromarray(out).save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")
    Image.fromarray(out).save(buf, "JPEG", quality=93, subsampling=0)
    cached.write_bytes(buf.getvalue())
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"X-MT-Blocks": str(len(blocks)),
                             "X-MT-Time": f"{sum(t.values()):.1f}"})


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
