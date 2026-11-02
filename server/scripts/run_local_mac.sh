#!/usr/bin/env bash
# Run the local Handy AI API with the existing GGUF Whisper and Ollama Qwen.
# Never kills an existing process or changes a user's model/data directory.
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$SERVER_DIR/.venv/bin/python"
MODEL="$SERVER_DIR/models/whisper-large-v3-Q5_K_M.gguf"
CLI="$SERVER_DIR/tools/transcribe.cpp/build/bin/transcribe-cli"
KEY_FILE="$HOME/.config/handy/local-api-key"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python venv at $PYTHON; activate/install the server environment first." >&2
  exit 1
fi
if [[ ! -s "$KEY_FILE" ]]; then
  echo "API key missing from $KEY_FILE. Keep your existing key or generate a new one with openssl." >&2
  exit 1
fi
if [[ ! -s "$MODEL" || ! -x "$CLI" ]]; then
  echo "GGUF model link or transcribe-cli is missing." >&2
  echo "Run: bash $SERVER_DIR/scripts/setup_local_mac.sh" >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && \
   lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8765 is already in use. Stop the existing Handy/Uvicorn server first." >&2
  lsof -nP -iTCP:8765 -sTCP:LISTEN >&2
  exit 1
fi

cd "$SERVER_DIR"
mkdir -p data
export HANDY_API_KEY="$(cat "$KEY_FILE")"
export HANDY_DB_PATH="$SERVER_DIR/data/handy.sqlite3"

export HANDY_ASR_ENABLED=true
# Native Handy uses exactly the desktop TranscriptionManager (the same
# language selection, long-form chunking, custom words and text normalization).
# The API starts before Cargo compiles, so the executable is checked per job.
export HANDY_ASR_BACKEND=native_handy
export HANDY_NATIVE_ASR_EXECUTABLE="${HANDY_NATIVE_ASR_EXECUTABLE:-$SERVER_DIR/../src-tauri/target/debug/handy}"
export HANDY_GGUF_MODEL_DIR="${HANDY_MEETING_MODEL_DIR:-$SERVER_DIR/data/asr-models}"
export HANDY_GGUF_MODEL_PATH="$MODEL"
export HANDY_TRANSCRIBE_CLI_PATH="$CLI"
export HANDY_GGUF_COMPUTE_BACKEND=auto

export HANDY_LLM_ENABLED=true
export HANDY_QWEN_PROVIDER=ollama
export HANDY_QWEN_BASE_URL="http://127.0.0.1:11434/v1"
export HANDY_QWEN_MODEL="qwen3.5:9b"
export HANDY_QWEN_API_KEY="ollama"

echo "Starting Handy API: desktop-native ASR + Ollama Qwen on 127.0.0.1:8765"
exec "$PYTHON" -m uvicorn handy_server.main:app \
  --host 127.0.0.1 --port 8765 --workers 1
