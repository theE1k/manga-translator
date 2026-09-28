"""Compare translation models on typical webtoon dialogue (ko -> zh).

  python tests/bench_translate.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt.translate import Translator

LINES = [
    "오늘은 정말 긴 하루였어...",
    "너 지금 어디야? 빨리 와!",
    "그날 이후로 모든 것이 달라지기 시작했다.",
    "잠깐만, 이거 네가 한 거야?",
    "기다려 줘서 고마워",
    "다음에 또 만나자, 약속이야.",
    "뭐야, 왜 이렇게 늦었어?",
    "야, 너 진짜 미쳤냐?",
    "선배, 저 오늘 좀 일찍 가도 돼요?",
    "헐... 대박.",
    "그런 눈으로 보지 마. 부끄럽잖아.",
    "괜찮아, 내가 옆에 있을게.",
]


class NoCtx(Translator):
    """HY-MT without neighbouring lines as context."""
    def _hymt(self, lines, src, tgt, ctx_lines=0):
        return super()._hymt(lines, src, tgt, ctx_lines=0)


RUNS = [
    ("qwen2.5:7b", lambda: Translator(model="qwen2.5:7b")),
    ("qwen3:4b-instruct", lambda: Translator(model="qwen3:4b-instruct")),
    ("qwen2.5:3b", lambda: Translator(model="qwen2.5:3b")),
    ("qwen2.5:1.5b", lambda: Translator(model="qwen2.5:1.5b")),
    ("hymt:1.8b (context)", lambda: Translator(model="hymt:1.8b")),
    ("hymt:1.8b (no context)", lambda: NoCtx(model="hymt:1.8b")),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    results = {}
    for name, make in RUNS:
        if only and not any(o in name for o in only):
            continue
        tr = make()
        tr.translate_lines(LINES[:1], "ko", "zh")  # warm-up: load model into memory
        t0 = time.time()
        results[name] = (tr.translate_lines(LINES, "ko", "zh"), time.time() - t0)
    for i, src in enumerate(LINES):
        print(f"\n{src}")
        for name, (out, _) in results.items():
            print(f"  {name:24s} {out[i]}")
    print()
    for name, (_, dt) in results.items():
        print(f"{name:24s} {dt:5.1f}s for {len(LINES)} lines")
