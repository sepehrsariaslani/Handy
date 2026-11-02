#!/usr/bin/env bash
# Start the local AI API with Handy's native dev window on macOS.
# Reuse an authorized existing server; never kill or adopt unrelated processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/link-dev-whisper-model.sh

# Only recycle a Vite process proven to be from THIS repository.
# This handles orphan dev servers such as node .../Handy/node_modules/.bin/vite
# without killing arbitrary processes listening on port 1420.
VITE_PIDS="$(lsof -tiTCP:1420 -sTCP:LISTEN 2>/dev/null | sort -u || true)"
if [[ -n "$VITE_PIDS" ]]; then
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$command_line" != *"$ROOT/node_modules/.bin/vite"* ]]; then
      echo "Port 1420 belongs to an unrelated process; refusing to stop it:" >&2
      ps -p "$pid" -o pid,command >&2 || true
      exit 1
    fi
  done <<< "$VITE_PIDS"

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    echo "[Handy Dev] Stopping leftover Vite from this project (PID $pid)."
    kill "$pid" 2>/dev/null || true
  done <<< "$VITE_PIDS"

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! lsof -nP -iTCP:1420 -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if lsof -nP -iTCP:1420 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port 1420 is still occupied. No additional process was terminated." >&2
    lsof -nP -iTCP:1420 -sTCP:LISTEN >&2
    exit 1
  fi
fi

KEY_FILE="$HOME/.config/handy/local-api-key"
if [[ ! -s "$KEY_FILE" ]]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  (umask 077; openssl rand -hex 32 > "$KEY_FILE")
  echo "[Handy Dev] Created the local API key automatically."
fi
chmod 600 "$KEY_FILE"

BACKEND_PID=""
cleanup() {
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "[Handy Dev] Stopping the API launched by this session."
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  if ! curl -fsS --max-time 8 \
    -H "Authorization: Bearer $(cat "$KEY_FILE")" \
    http://127.0.0.1:8765/v1/capabilities >/dev/null; then
    echo "An API is already listening on 8765 but does not accept Handy's local key." >&2
    echo "Stop that server or repair its HANDY_API_KEY; it was not terminated." >&2
    exit 1
  fi
  echo "[Handy Dev] Reusing the authorized local AI backend on port 8765."
else
  mkdir -p server/data
  bash server/scripts/run_local_mac.sh >server/data/handy-dev-backend.log 2>&1 &
  BACKEND_PID=$!
  echo "[Handy Dev] Starting AI backend automatically (PID $BACKEND_PID)."
  echo "[Handy Dev] Backend log: $ROOT/server/data/handy-dev-backend.log"
fi


# A GGUF startup probe can load the model before Uvicorn begins listening.
# Do not open the desktop until the authenticated API has finished startup.
if [[ -n "$BACKEND_PID" ]]; then
  echo "[Handy Dev] Checking AI backend startup..."
  READY=false
  for _ in {1..150}; do
    if curl -fsS --max-time 3 \
      -H "Authorization: Bearer $(cat "$KEY_FILE")" \
      http://127.0.0.1:8765/v1/capabilities >/dev/null 2>&1; then
      READY=true
      break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      echo "[Handy Dev] Local backend stopped unexpectedly." >&2
      tail -n 25 server/data/handy-dev-backend.log >&2 || true
      exit 1
    fi
    sleep 1
  done
  if [[ "$READY" != true ]]; then
    echo "[Handy Dev] AI backend did not become ready." >&2
    tail -n 25 server/data/handy-dev-backend.log >&2 || true
    exit 1
  fi
  echo "[Handy Dev] Local AI API is ready."
fi

bun tauri dev --config src-tauri/tauri.dev.conf.json
