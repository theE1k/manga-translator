// Background: fetches page images (bypasses page CORS with host permission) and talks to the local server.
const api = globalThis.browser ?? globalThis.chrome;
const DEFAULTS = { server: "http://127.0.0.1:8765", token: "", src: "ja", tgt: "zh" };

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

async function translate(msg) {
  const cfg = Object.assign({}, DEFAULTS, await api.storage.local.get(Object.keys(DEFAULTS)));
  let blob;
  if (msg.dataUrl) blob = b64ToBlob(msg.dataUrl);
  else {
    const r = await fetch(msg.url, { credentials: "include" });
    if (!r.ok) throw new Error(`image fetch ${r.status}`);
    blob = await r.blob();
  }
  const fd = new FormData();
  fd.append("image", blob, "page");
  fd.append("src", msg.src || cfg.src);
  fd.append("tgt", msg.tgt || cfg.tgt);
  const headers = cfg.token ? { "X-MT-Token": cfg.token } : {};
  const r = await fetch(cfg.server.replace(/\/$/, "") + "/translate", { method: "POST", body: fd, headers });
  if (!r.ok) throw new Error(`server ${r.status}: ${(await r.text()).slice(0, 120)}`);
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
