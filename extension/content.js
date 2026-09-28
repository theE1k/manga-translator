// Content script: finds manga page images, sends them for translation, swaps the translated page in place.
(() => {
  if (window.__mtLoaded) return;
  window.__mtLoaded = true;
  const api = globalThis.browser ?? globalThis.chrome;
  const DEFAULTS = { src: "ko", tgt: "zh", auto: false, button: true, server: "http://127.0.0.1:8765", token: "" };
  const MIN_SIDE = 300;          // ignore icons, avatars, ads
  let cfg = { ...DEFAULTS };
  let showOriginal = false;
  const state = new WeakMap();   // img -> {status, orig, translated}
  const queue = [];
  let busy = false, done = 0, total = 0;

  // ---------------------------------------------------------------- UI
  const btn = document.createElement("div");
  btn.id = "mt-fab";
  btn.innerHTML = '<span class="mt-label">译</span><span class="mt-count"></span>';
  const count = btn.querySelector(".mt-count");
  let pressTimer = null, longPressed = false;
  btn.addEventListener("pointerdown", () => {
    longPressed = false;
    pressTimer = setTimeout(() => { longPressed = true; toggleOriginal(); }, 550);
  });
  btn.addEventListener("pointerup", () => clearTimeout(pressTimer));
  btn.addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    if (longPressed) return;
    const imgs = candidates();
    if (imgs.length && imgs.every((i) => state.get(i)?.status === "done")) return toggleOriginal();
    translatePage();
  });

  function setStatus() {
    btn.classList.toggle("mt-busy", busy);
    btn.classList.toggle("mt-orig", showOriginal);
    count.textContent = total ? `${done}/${total}` : "";
  }

  function toast(text) {
    let t = document.getElementById("mt-toast");
    if (!t) { t = document.createElement("div"); t.id = "mt-toast"; document.documentElement.appendChild(t); }
    t.textContent = text; t.classList.add("mt-show");
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("mt-show"), 3500);
  }

  // ---------------------------------------------------------------- images
  function candidates() {
    return [...document.images].filter((img) => {
      const w = img.naturalWidth, h = img.naturalHeight;
      const r = img.getBoundingClientRect();
      return img.complete && w >= MIN_SIDE && h >= MIN_SIDE && r.width >= 150 && r.height >= 150;
    });
  }

  function nearViewport(img, margin = 1.5) {
    const r = img.getBoundingClientRect(), vh = innerHeight;
    return r.bottom > -vh * 0.2 && r.top < vh * (1 + margin);
  }

  async function imageData(img) {
    const src = img.currentSrc || img.src;
    // 1) blob:/data:/same-origin or CORS-enabled: read directly from the page
    if (src.startsWith("data:")) return { dataUrl: src };
    try {
      const r = await fetch(src, { credentials: "include" });
      if (r.ok) return { dataUrl: await blobToDataUrl(await r.blob()) };
    } catch (_) { /* CORS -> fall through */ }
    // 2) already-decoded pixels (works when the image is not cross-origin tainted)
    try {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext("2d").drawImage(img, 0, 0);
      return { dataUrl: c.toDataURL("image/jpeg", 0.95) };
    } catch (_) { /* tainted */ }
    // 3) let the background fetch it with extension host permission
    return { url: src, referer: location.href };
  }

  function blobToDataUrl(blob) {
    return new Promise((res, rej) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.onerror = rej; fr.readAsDataURL(blob); });
  }

  function swap(img, url) {
    // <picture><source srcset> would override img.src, so neutralise them
    const pic = img.parentElement && img.parentElement.tagName === "PICTURE" ? img.parentElement : null;
    if (pic) pic.querySelectorAll("source").forEach((s) => { s.dataset.mtSrcset ??= s.srcset; s.removeAttribute("srcset"); });
    if (img.hasAttribute("srcset")) { img.dataset.mtSrcset ??= img.srcset; img.removeAttribute("srcset"); }
    img.src = url;
  }

  async function process(img) {
    const st = { status: "working", orig: img.currentSrc || img.src };
    state.set(img, st);
    img.classList.add("mt-working");
    try {
      const payload = await imageData(img);
      const res = await api.runtime.sendMessage({ type: "translate", ...payload, src: cfg.src, tgt: cfg.tgt });
      if (!res || res.error) throw new Error(res ? res.error : "no response from background");
      st.translated = res.dataUrl;
      st.status = "done";
      if (!showOriginal) swap(img, st.translated);
    } catch (e) {
      st.status = "error";
      console.warn("[manga-translator]", e);
      toast("翻译失败: " + (e.message || e));
    } finally {
      img.classList.remove("mt-working");
    }
  }

  async function pump() {
    if (busy) return;
    busy = true; setStatus();
    while (queue.length) {
      // closest-to-viewport first
      queue.sort((a, b) => Math.abs(a.getBoundingClientRect().top) - Math.abs(b.getBoundingClientRect().top));
      const img = queue.shift();
      await process(img);
      done++; setStatus();
    }
    busy = false; setStatus();
  }

  function enqueue(imgs) {
    const fresh = imgs.filter((i) => !state.has(i) && !queue.includes(i));
    if (!fresh.length) return;
    queue.push(...fresh);
    total += fresh.length;
    setStatus();
    pump();
  }

  function translatePage() {
    showOriginal = false;
    const imgs = candidates();
    if (!imgs.length) return toast("这个页面没找到漫画图片");
    enqueue(imgs.filter((i) => nearViewport(i, 3)));
    cfg.auto = true; // keep following as the reader scrolls / loads more pages
  }

  function toggleOriginal() {
    showOriginal = !showOriginal;
    for (const img of document.images) {
      const st = state.get(img);
      if (st && st.status === "done") swap(img, showOriginal ? st.orig : st.translated);
    }
    setStatus();
    toast(showOriginal ? "显示原图（再长按切回译文）" : "显示译文");
  }

  // auto mode: translate pages as they approach the viewport
  let scrollT = null;
  function autoScan() {
    if (!cfg.auto || showOriginal) return;
    enqueue(candidates().filter((i) => nearViewport(i)));
  }
  addEventListener("scroll", () => { clearTimeout(scrollT); scrollT = setTimeout(autoScan, 250); }, { passive: true });
  new MutationObserver(() => { clearTimeout(scrollT); scrollT = setTimeout(autoScan, 400); })
    .observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("load", (e) => { if (e.target.tagName === "IMG") { clearTimeout(scrollT); scrollT = setTimeout(autoScan, 250); } }, true);

  // readers that reuse one <img> and change its src for the next page
  new MutationObserver((muts) => {
    for (const m of muts) {
      const img = m.target, st = state.get(img);
      if (!st || img.src.startsWith("data:") || img.src === st.orig) continue;
      if (st.translated && img.src === st.translated) continue;
      state.delete(img); // new page in the same element
    }
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["src"], subtree: true });

  // ---------------------------------------------------------------- settings
  async function loadCfg() {
    cfg = Object.assign({}, DEFAULTS, await api.storage.local.get(Object.keys(DEFAULTS)));
    const show = cfg.button && candidates().length > 0;
    if (show && !btn.isConnected) document.documentElement.appendChild(btn);
    if (!show && btn.isConnected && !cfg.button) btn.remove();
    autoScan();
  }
  api.storage.onChanged.addListener(loadCfg);
  api.runtime.onMessage.addListener((msg) => {
    if (msg.type === "translate-page") translatePage();
    if (msg.type === "toggle-original") toggleOriginal();
  });
  loadCfg();
  // images often arrive late on reader sites
  setTimeout(loadCfg, 1500);
  setTimeout(loadCfg, 5000);
})();
