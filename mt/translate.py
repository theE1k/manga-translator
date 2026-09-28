"""Translation with a local open-source LLM through Ollama.

Whole page is sent at once (numbered, in reading order) so the model sees the
conversation and keeps names / tone consistent.

Suggested models:
  qwen2.5:7b          default, ~4.7 GB, all language pairs
  qwen2.5:3b          smaller/faster, ~2 GB
  sakura (SakuraLLM)  Japanese -> Chinese specialist (import GGUF into Ollama)
"""
from __future__ import annotations

import json
import re

import requests

from . import config
from .types import TextBlock

SYSTEM = (
    "You are a professional manga/webtoon translator and letterer. "
    "Translate speech-bubble text into natural, colloquial {tgt} the way an official "
    "localisation would read: keep character voice, exclamations and sound effects, "
    "keep it short enough to fit the bubble, never add explanations or notes. "
    "Lines are in reading order and belong to the same page, so use them as context."
)


def _is_sakura(model: str) -> bool:
    return "sakura" in model.lower()


class Translator:
    def __init__(self, model=config.OLLAMA_MODEL, url=config.OLLAMA_URL, timeout=180):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.http = requests.Session()
        self.http.trust_env = False  # never send localhost traffic through a system proxy (Clash etc.)

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["format"] = "json"
        r = self.http.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def translate_lines(self, lines: list[str], src: str, tgt: str) -> list[str]:
        if not lines:
            return []
        tgt_name = config.LANG_NAMES.get(tgt, tgt)
        src_name = config.LANG_NAMES.get(src, src)

        if _is_sakura(self.model):  # Sakura expects plain line-by-line ja->zh
            sys_p = "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
            out = self._chat(sys_p, "将下面的日文文本翻译成中文：\n" + "\n".join(lines), False)
            res = [l.strip() for l in out.strip().split("\n") if l.strip()]
            if len(res) == len(lines):
                return res
        else:
            payload = {str(i + 1): t for i, t in enumerate(lines)}
            user = (f"Translate each {src_name} value into {tgt_name}. Reply with a JSON object "
                    f"with exactly the same keys and only the translations as values.\n"
                    + json.dumps(payload, ensure_ascii=False))
            try:
                out = self._chat(SYSTEM.format(tgt=tgt_name), user, True)
                data = json.loads(out)
                if isinstance(data, dict) and "translations" in data and isinstance(data["translations"], dict):
                    data = data["translations"]
                res = [str(data.get(str(i + 1), "")).strip() for i in range(len(lines))]
                if all(res):
                    return res
            except (json.JSONDecodeError, requests.RequestException, KeyError):
                pass

        # fallback: one line at a time
        res = []
        for t in lines:
            out = self._chat(SYSTEM.format(tgt=tgt_name),
                             f"Translate into {tgt_name}. Output only the translation.\n{t}", False)
            res.append(_clean(out))
        return res

    def __call__(self, blocks: list[TextBlock], src: str, tgt: str) -> None:
        todo = [b for b in blocks if b.text]
        for b, t in zip(todo, self.translate_lines([b.text for b in todo], src, tgt)):
            b.translation = _clean(t)


def _clean(s: str) -> str:
    s = s.strip().strip('"').strip()
    s = re.sub(r"^(translation|译文|翻译)\s*[:：]\s*", "", s, flags=re.I)
    return s
