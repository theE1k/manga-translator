const api = globalThis.browser ?? globalThis.chrome;
const $ = (id) => document.getElementById(id);
const FIELDS = ["server", "token", "src", "tgt", "auto", "button"];

async function load() {
  const cfg = Object.assign({}, MT_DEFAULTS, await api.storage.local.get(FIELDS));
  for (const k of FIELDS) {
    const el = $(k);
    if (el.type === "checkbox") el.checked = !!cfg[k]; else el.value = cfg[k];
  }
  ping();
}

async function save() {
  const v = {};
  for (const k of FIELDS) { const el = $(k); v[k] = el.type === "checkbox" ? el.checked : el.value.trim(); }
  await api.storage.local.set(v);
}

async function ping() {
  $("status").textContent = "连接服务器…";
  const r = await api.runtime.sendMessage({ type: "health" }).catch((e) => ({ error: String(e) }));
  $("status").textContent = r && !r.error ? `✓ 已连接 · 翻译模型 ${r.model}` : `✗ 连不上服务器：${r ? r.error : ""}`;
}

async function toTab(type) {
  const [tab] = await api.tabs.query({ active: true, currentWindow: true });
  if (tab) api.tabs.sendMessage(tab.id, { type }).catch(() => {});
  window.close();
}

for (const k of FIELDS) $(k).addEventListener("change", async () => { await save(); if (k === "server" || k === "token") ping(); });
$("go").addEventListener("click", async () => { await save(); toTab("translate-page"); });
$("orig").addEventListener("click", () => toTab("toggle-original"));
load();
