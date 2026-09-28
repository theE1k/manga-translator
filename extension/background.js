// Background: fetches page images (bypasses page CORS with host permission) and talks to the local server.
const api = globalThis.browser ?? globalThis.chrome;
const DEFAULTS = { server: "http://127.0.0.1:8765", token: "", src: "ko", tgt: "zh" };

function b64ToBlob(dataUrl) {
  const [head, b64] = dataUrl.split(",");
  const mime = (head.match(/data:(.*?);/) || [])[1] || "image/jpeg";
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

async function blobToDataUrl(blob) {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let s = "";
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return `data:${blob.type || "image/jpeg"};base64,${btoa(s)}`;
}

// Image hosts that check Referer (hotlink protection) reject requests from the extension, which
// carry no page Referer. A session rule adds the reading page's URL to the extension's own requests.
const REFERER_RULE = 1;
async function setReferer(pageUrl) {
  const dnr = api.declarativeNetRequest;
  if (!dnr || !pageUrl) return;
  try {
    await dnr.updateSessionRules({
      removeRuleIds: [REFERER_RULE],
      addRules: [{
        id: REFERER_RULE,
        priority: 1,
        action: { type: "modifyHeaders", requestHeaders: [{ header: "referer", operation: "set", value: pageUrl }] },
        condition: { tabIds: [-1], resourceTypes: ["xmlhttprequest", "other", "image"] }
      }]
    });
  } catch (e) { console.warn("referer rule failed", e); }
}

async function toBlob(m) {
  if (m.dataUrl) return b64ToBlob(m.dataUrl);
  await setReferer(m.referer);
  const host = (() => { try { return new URL(m.url).host; } catch (_) { return m.url; } })();
  let r;
  try {
    r = await fetch(m.url, { credentials: "include" });
  } catch (e) {
    throw new Error(`读取原图失败（${host}，网络错误）: ${e.message}`);
  }
  if (!r.ok) throw new Error(`读取原图失败（${host}，HTTP ${r.status}）`);
  return r.blob();
}

// The server may be restarting for a few seconds: retry connection errors before giving up.
async function postWithRetry(url, init, tries = 3) {
  for (let i = 1; ; i++) {
    try {
      return await fetch(url, init);
    } catch (e) {
      if (i >= tries) throw new Error(`连不上翻译服务 ${url.replace(/\/translate$/, "")}（服务没开？在项目目录运行 ./run.sh）`);
      await new Promise((res) => setTimeout(res, 2000));
    }
  }
}

async function translate(msg) {
  const cfg = Object.assign({}, DEFAULTS, await api.storage.local.get(Object.keys(DEFAULTS)));
  const blob = await toBlob(msg);
  const fd = new FormData();
  fd.append("image", blob, "page");
  if (msg.y != null) fd.append("y", String(msg.y));
  if (msg.prio) fd.append("prio", "1");
  // next slices: look-ahead context + bubbles cut by the seam. Optional - failures are ignored.
  for (const [i, n] of (msg.next || []).entries()) {
    try {
      fd.append(`next${i + 1}`, await toBlob(n), "next");
      if (i === 0 && n.y != null) fd.append("next1_y", String(n.y));
    } catch (e) { console.warn("next slice skipped", e); break; }
  }
  fd.append("src", msg.src || cfg.src);
  fd.append("tgt", msg.tgt || cfg.tgt);
  if (msg.ctx) fd.append("ctx", msg.ctx);  // chapter URL -> server keeps earlier lines as context
  const headers = cfg.token ? { "X-MT-Token": cfg.token } : {};
  const r = await postWithRetry(cfg.server.replace(/\/$/, "") + "/translate", { method: "POST", body: fd, headers });
  if (!r.ok) throw new Error(`翻译服务出错 ${r.status}: ${(await r.text()).slice(0, 120)}`);
  return { dataUrl: await blobToDataUrl(await r.blob()) };
}

async function health() {
  const cfg = Object.assign({}, DEFAULTS, await api.storage.local.get(Object.keys(DEFAULTS)));
  const headers = cfg.token ? { "X-MT-Token": cfg.token } : {};
  const r = await fetch(cfg.server.replace(/\/$/, "") + "/health", { headers });
  if (!r.ok) throw new Error(`server ${r.status}`);
  return r.json();
}

api.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const job = msg.type === "translate" ? translate(msg) : msg.type === "health" ? health() : null;
  if (!job) return false;
  job.then(sendResponse, (e) => sendResponse({ error: String(e && e.message || e) }));
  return true; // async response
});
