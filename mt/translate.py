"""Translation with a local open-source LLM through Ollama.

Whole page is sent at once (numbered, in reading order) so the model sees the
conversation and keeps names / tone consistent.

Suggested models:
  qwen3:4b-instruct   default, ~2.5 GB; on ko->zh webtoon lines as good as qwen2.5:7b, 40% faster
  qwen2.5:7b          ~4.7 GB, slightly more natural slang, needs ~4.5 GB RAM
  (qwen2.5:3b / 1.5b and hymt:1.8b were tested and mistranslate colloquial lines; see tests/bench_translate.py)
  sakura (SakuraLLM)  Japanese -> Chinese specialist (import GGUF into Ollama)
  hymt:1.8b           Tencent HY-MT1.5-1.8B, dedicated 33-language translation model, ~1.1 GB
                      (import with: ollama create hymt:1.8b -f Modelfile.hy-mt)
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
    "{extra}"
)

# Source-specific guidance: small models otherwise soften slang, read internet shorthand literally,
# or translate idioms word by word.
SRC_NOTES = {
    "ko": (
        " The source is a Korean webtoon. Keep the register: swearing stays swearing at the same "
        "strength (씨발/ㅆ발 -> 操/妈的, 새끼/ㅅ끼 -> 小子/崽子/混蛋, ㅂ신/병신 -> 傻逼/废物, 개- as an intensifier -> 超/真他妈), "
        "school slang stays slang (왕따/전따 -> 被孤立的/全校公敌, 찐따 -> 废物/窝囊废, 일진 -> 不良/混混头子, "
        "노답 -> 没救了, ㅈ되다 -> 完蛋). ㅋㅋ -> 哈哈. 선배 -> 学长/学姐, 샘/선생님 -> 老师. "
        "Translate the meaning of idioms, not the words (e.g. 멱따는 소리 = shrieking like a pig being slaughtered, "
        "~할 줄 알다 = to know how to ~). OCR may contain small typos; infer the intended word from context. "
        "Keep Korean names as consistent Chinese transliterations."
    ),
}


def _is_sakura(model: str) -> bool:
    return "sakura" in model.lower()


def _is_hymt(model: str) -> bool:
    m = model.lower()
    return "hymt" in m or "hy-mt" in m


# HY-MT's own target-language names; its official prompts (see model card) differ for Chinese pairs.
_HYMT_NAMES = {"zh": "中文", "zh-TW": "繁体中文", "en": "英语", "ja": "日语", "ko": "韩语"}


class Translator:
    def __init__(self, model=config.OLLAMA_MODEL, url=config.OLLAMA_URL, timeout=180):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.http = requests.Session()
        self.http.trust_env = False  # never send localhost traffic through a system proxy (Clash etc.)

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        # HY-MT ships its own sampling defaults in the Modelfile; don't override them
        opts = {"num_ctx": 4096} if _is_hymt(self.model) else {"temperature": 0.2, "num_ctx": 4096}
        body = {"model": self.model, "stream": False, "options": opts, "messages": msgs}
        if self.model.startswith("qwen3") and "instruct" not in self.model:
            body["think"] = False  # hybrid Qwen3: skip the reasoning trace, it only adds latency here
        if json_mode:
            body["format"] = "json"
        r = self.http.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def translate_lines(self, lines: list[str], src: str, tgt: str,
                        history: list[tuple[str, str]] | None = None) -> list[str]:
        if not lines:
            return []
        tgt_name = config.LANG_NAMES.get(tgt, tgt)
        src_name = config.LANG_NAMES.get(src, src)

        if _is_hymt(self.model):
            return self._hymt(lines, src, tgt)

        if _is_sakura(self.model):  # Sakura expects plain line-by-line ja->zh
            sys_p = "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
            out = self._chat(sys_p, "将下面的日文文本翻译成中文：\n" + "\n".join(lines), False)
            res = [l.strip() for l in out.strip().split("\n") if l.strip()]
            if len(res) == len(lines):
                return res
        else:
            payload = {str(i + 1): t for i, t in enumerate(lines)}
            ctx = ""
            if history:
                ctx = ("Earlier lines of this chapter, already translated (context only, do not output them):\n"
                       + "\n".join(f"{a} => {b}" for a, b in history) + "\n\n")
            user = (ctx + f"Translate each {src_name} value into {tgt_name}. Reply with a JSON object "
                    f"with exactly the same keys and only the translations as values.\n"
                    + json.dumps(payload, ensure_ascii=False))
            try:
                out = self._chat(self._system(src, tgt), user, True)
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
            out = self._chat(self._system(src, tgt),
                             f"Translate into {tgt_name}. Output only the translation.\n{t}", False)
            res.append(_clean(out))
        return res

    def _system(self, src: str, tgt: str) -> str:
        return SYSTEM.format(tgt=config.LANG_NAMES.get(tgt, tgt), extra=SRC_NOTES.get(src, ""))

    def _hymt(self, lines: list[str], src: str, tgt: str, ctx_lines: int = 3) -> list[str]:
        """One bubble per call with the previous bubbles as context (HY-MT's contextual-translation prompt)."""
        name = _HYMT_NAMES.get(tgt, config.LANG_NAMES.get(tgt, tgt))
        out = []
        for i, t in enumerate(lines):
            ctx = "\n".join(lines[max(0, i - ctx_lines):i])
            if ctx:
                prompt = f"{ctx}\n参考上面的信息，把下面的文本翻译成{name}，注意不需要翻译上文，也不要额外解释：\n{t}"
            elif tgt.startswith("zh") or src.startswith("zh"):
                prompt = f"将以下文本翻译为{name}，注意只需要输出翻译后的结果，不要额外解释：\n\n{t}"
            else:
                prompt = f"Translate the following segment into {config.LANG_NAMES.get(tgt, tgt)}, without additional explanation.\n\n{t}"
            out.append(_clean(self._chat("", prompt, False)))
        return out

    def __call__(self, blocks: list[TextBlock], src: str, tgt: str,
                 history: list[tuple[str, str]] | None = None) -> None:
        todo = [b for b in blocks if b.text]
        for b, t in zip(todo, self.translate_lines([b.text for b in todo], src, tgt, history)):
            b.translation = _clean(t)


def _clean(s: str) -> str:
    s = s.strip().strip('"').strip()
    s = re.sub(r"^(translation|译文|翻译)\s*[:：]\s*", "", s, flags=re.I)
    return s
