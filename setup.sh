#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon). Usage:
#   ./setup.sh            # Japanese source only
#   ./setup.sh ja,ko,zh   # also Korean + Chinese OCR (installs EasyOCR)
set -euo pipefail
cd "$(dirname "$0")"
LANGS="${1:-ja}"
MODEL="${MT_OLLAMA_MODEL:-qwen3:4b-instruct}"

need_mb=2500                                   # Python packages ~1.5 GB + models ~0.8 GB
[[ "$LANGS" != "ja" ]] && need_mb=$((need_mb + 500))
have_model=0
if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$MODEL"; then
  have_model=1
else
  need_mb=$((need_mb + 2500))
fi
avail_mb=$(df -m . | awk 'NR==2{print $4}')
echo "Free disk: ${avail_mb} MB, needs ~${need_mb} MB"
if [ "${avail_mb}" -lt "${need_mb}" ]; then
  echo "⚠️  Not enough free space. Free up some disk first."; exit 1
fi

if command -v uv >/dev/null 2>&1; then          # preferred: uv fetches a clean Python 3.12
  [ -d .venv ] || uv venv .venv --python 3.12
  PIP="uv pip install --no-cache --python .venv/bin/python"
else
  PY=""
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  echo "Using $($PY --version)"
  [ -d .venv ] || "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  PIP=".venv/bin/pip install --no-cache-dir"
fi
source .venv/bin/activate
$PIP -r requirements.txt
[[ "$LANGS" != "ja" ]] && $PIP "easyocr>=1.7"

python download_models.py "$LANGS"

if [ "$have_model" = 1 ]; then
  echo "✓ Ollama model $MODEL already installed"
elif command -v ollama >/dev/null 2>&1; then
  echo "↓ Ollama model $MODEL"
  (pgrep -x ollama >/dev/null || (ollama serve >/dev/null 2>&1 &) ; sleep 2)
  ollama pull "$MODEL"
else
  echo "⚠️  Ollama not found: install from https://ollama.com then run: ollama pull $MODEL"
fi

echo
echo "✅ Setup done. Start the server with:  ./run.sh"
