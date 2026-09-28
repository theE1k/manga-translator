"""Translate a real webtoon chapter slice by slice (like the extension does) and dump the lines that
have reference translations, for side-by-side grading.

  python tests/bench_context.py out/real/ep1_lines.json  qwen3:4b-instruct baseline
  strategies: baseline (generic prompt, no context) | prompt (Korean notes) | ctx (notes + chapter history)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt import translate as T


def guard(min_free_gb=10, max_swap_gb=8):
    """Stop before the Mac runs out of room: swap lives on the internal disk."""
    import re
    import shutil
    import subprocess
    free = shutil.disk_usage("/System/Volumes/Data").free / 2**30
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = float(re.search(r"used = ([\d.]+)M", sw).group(1)) / 1024
    if free < min_free_gb or used > max_swap_gb:
        sys.exit(f"ABORT: internal disk free {free:.1f} GB, swap used {used:.1f} GB")

lines_path, model, strategy = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None  # only the first N slices (smoke test)
rows = json.loads(Path(lines_path).read_text())
# Reference translations: [[slice, source_line, reference], ...]. Not shipped (the source lines are
# copyrighted comic dialogue) - build your own set from a chapter you own and point MT_BENCH_REFS at it.
refs = json.loads(Path(__import__("os").environ.get(
    "MT_BENCH_REFS", Path(__file__).parent / "data/ko_zh_webtoon_ref.json")).read_text())
want = {(s, t) for s, t, _ in refs}

if strategy == "baseline":
    T.SRC_NOTES = {}
tr = T.Translator(model=model, backend="api" if "/" in model else "ollama")  # "vendor/model" = cloud API
tr.translate_lines(["안녕"], "ko", "zh")  # warm-up

slices = {}
for r in rows:
    slices.setdefault(r["slice"], []).append(r["text"])

history, got, t0 = [], {}, time.time()
for sl in sorted(slices)[:limit]:
    guard()
    src = slices[sl]
    out = tr.translate_lines(src, "ko", "zh", history if strategy == "ctx" else None)
    for a, b in zip(src, out):
        history.append((a, b))
        if (sl, a) in want:
            got[f"{sl}|{a}"] = b
    history = history[-20:]
dt = time.time() - t0
name = f"{model.replace('/', '_').replace(':', '_')}__{strategy}"
if "/" in model:
    name += "__think-" + __import__("os").environ.get("MT_API_REASONING", "off")
Path("out/bench").mkdir(parents=True, exist_ok=True)
Path(f"out/bench/{name}.json").write_text(json.dumps({"seconds": dt, "out": got}, ensure_ascii=False, indent=0))
print(f"{name}: {len(got)}/{len(refs)} ref lines, {dt:.0f}s for {len(slices)} slices")
