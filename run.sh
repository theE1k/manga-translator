#!/usr/bin/env bash
# Start the local translation server (and Ollama if it is not running).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
if ! grep -qs "^MT_API_KEY=." .env && command -v ollama >/dev/null 2>&1 && ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then  # local model only when no API key
  (ollama serve >/dev/null 2>&1 &)
  sleep 2
fi
IP=$(ipconfig getifaddr en0 2>/dev/null || echo "your-mac-ip")
echo "iPhone / extension server URL:  http://${IP}:${MT_PORT:-8765}"
exec python server.py
