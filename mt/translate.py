"""Translation with an LLM: a cloud OpenAI-compatible API (OpenRouter etc.) or a local model through Ollama.

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
import os
import re
import time

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


_MODEL_PARAMS: dict[str, set] = {}


def _supports(url: str, model: str, param: str) -> bool:
    """OpenRouter lists each model's supported request parameters; fetch once and cache."""
    if "openrouter.ai" not in url:
        return False
    if not _MODEL_PARAMS:
        try:
            data = requests.get(f"{url}/models", timeout=30).json()["data"]
            _MODEL_PARAMS.update({m["id"]: set(m.get("supported_parameters") or []) for m in data})
        except (requests.RequestException, KeyError, ValueError):
            return False
    return param in _MODEL_PARAMS.get(model, set())


class Translator:
    def __init__(self, model=None, url=None, timeout=180, backend=None):
        self.backend = backend or config.BACKEND
        if self.backend == "api":
            self.model, self.url = model or config.API_MODEL, (url or config.API_URL).rstrip("/")
        else:
            self.model, self.url = model or config.OLLAMA_MODEL, (url or config.OLLAMA_URL).rstrip("/")
        self.timeout = timeout
        self.http = requests.Session()
        self.http.trust_env = False  # never send localhost traffic through a system proxy (Clash etc.)

    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        if self.backend == "api":
            return self._chat_api(msgs, json_mode)
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

    def _chat_api(self, msgs, json_mode: bool) -> str:
        # Explicit output cap: some providers otherwise reserve their whole context window for the
        # reply and then reject the request as "exceeds maximum context length" (HTTP 400).
        body = {"model": self.model, "messages": msgs, "temperature": 0.3, "max_tokens": 1024}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if _supports(self.url, self.model, "reasoning"):
            # Reasoning models (DeepSeek V4, GLM, Nemotron) think by default: ~30x the output tokens and
            # 3x the latency for a bubble of dialogue. Off unless MT_API_REASONING=on.
            body["reasoning"] = {"enabled": os.environ.get("MT_API_REASONING", "off") == "on"}
        if "openrouter.ai" in self.url:
            # Route around first-party hosts that moderate adult content, and (unless the model is a
            # free one, which requires it) around providers that keep or train on prompts.
            prov = {"ignore": [p for p in os.environ.get("MT_API_IGNORE", "DeepSeek,Alibaba").split(",") if p],
                    "sort": "latency"}  # fastest-responding host first: the reader is waiting
            if not self.model.endswith(":free"):
                prov["data_collection"] = "deny"
            body["provider"] = prov
        for wait in (2, 5, 15, None):  # rate limits (429) and flaky upstreams (5xx) are common on cheap/free routes
            r = self.http.post(f"{self.url}/chat/completions", json=body, timeout=self.timeout,
                               headers={"Authorization": f"Bearer {config.API_KEY}", "X-Title": "manga-translator"})
            if r.status_code not in (429, 500, 502, 503) or wait is None:
                break
            time.sleep(wait)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}", response=r)
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            raise requests.HTTPError("refused by content filter")
        return choice["message"]["content"] or ""

    def translate_lines(self, lines: list[str], src: str, tgt: str,
                        history: list[tuple[str, str]] | None = None,
                        ahead: list[str] | None = None) -> list[str]:
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
                       + "\n".join(f"{a} => {b}" if b else a for a, b in history) + "\n\n")
            if ahead:
                ctx += ("Lines that come right after this page (context only, do not output them):\n"
                        + "\n".join(ahead) + "\n\n")
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
            try:
                out = self._chat(self._system(src, tgt),
                                 f"Translate into {tgt_name}. Output only the translation.\n{t}", False)
            except requests.RequestException as e:  # one bad line must not sink the whole page
                print(f"  translate failed ({e}); leaving original: {t!r}")
                out = ""
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
                 history: list[tuple[str, str]] | None = None, ahead: list[str] | None = None) -> None:
        # only blocks with actual words: a bubble tail or a stray mark read as "↑" / "+" must not be
        # erased and redrawn
        todo = [b for b in blocks if any(c.isalpha() for c in b.text)]
        for b, t in zip(todo, self.translate_lines([b.text for b in todo], src, tgt, history, ahead)):
            t = _clean(t)
            if tgt.startswith("zh") and src != "ja" and re.search(r"[\u3040-\u30ff]", t):
                t = ""  # answered in Japanese (happens on logos / watermarks): leave the original
            if "".join(t.split()) == "".join(b.text.split()):
                t = ""  # unchanged (logos, names in Latin letters): keep the original pixels
            b.translation = t


def _clean(s: str) -> str:
    s = s.strip().strip('"').strip()
    s = re.sub(r"^(translation|译文|翻译)\s*[:：]\s*", "", s, flags=re.I)
    return s
