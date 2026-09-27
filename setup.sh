#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon). Usage:
#   ./setup.sh            # Japanese source only
#   ./setup.sh ja,ko,zh   # also Korean + Chinese OCR
set -euo pipefail
cd "$(dirname "$0")"
LANGS="${1:-ja}"

avail_gb=$(df -g . | awk 'NR==2{print $4}')
echo "Free disk: ${avail_gb} GB (needs ~4 GB: Python packages ~1.5 GB, models ~0.8 GB, Qwen2.5-3B ~2 GB)"
if [ "${avail_gb}" -lt 4 ]; then
  echo "⚠️  Not enough free space. Free up some disk first."; exit 1
fi

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
echo "Using $($PY --version)"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -r requirements.txt

python download_models.py "$LANGS"

if command -v ollama >/dev/null 2>&1; then
  MODEL="${MT_OLLAMA_MODEL:-qwen2.5:3b}"
  echo "↓ Ollama model $MODEL"
  (pgrep -x ollama >/dev/null || (ollama serve >/dev/null 2>&1 &) ; sleep 2)
  ollama pull "$MODEL"
else
  echo "⚠️  Ollama not found: install from https://ollama.com then run: ollama pull qwen2.5:3b"
fi

echo
echo "✅ Setup done. Start the server with:  ./run.sh"
