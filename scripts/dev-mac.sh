#!/usr/bin/env bash
# Start the local AI API with Handy's native dev window on macOS.
# Reuse an authorized existing server; never kill or adopt unrelated processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/link-dev-whisper-model.sh

# Never kill an existing development server: it may belong to an active
# Tauri session. Reuse Vite only if its command points at this checkout.
REUSE_VITE=false
VITE_PIDS="$(lsof -tiTCP:1420 -sTCP:LISTEN 2>/dev/null | sort -u || true)"
if [[ -n "$VITE_PIDS" ]]; then
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$command_line" != *"$ROOT/node_modules/.bin/vite"* ]]; then
      echo "Port 1420 belongs to a different process; refusing to interfere:" >&2
      ps -p "$pid" -o pid,command >&2 || true
      exit 1
    fi
  done <<< "$VITE_PIDS"

  if ! curl -fsS --max-time 3 http://localhost:1420/ >/dev/null 2>&1; then
    echo "The existing Handy Vite server is not responding. Stop its terminal first." >&2
    exit 1
  fi
  REUSE_VITE=true
  echo "[Handy Dev] Reusing existing Handy Vite on port 1420 (no process killed)."
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

if [[ "$REUSE_VITE" == true ]]; then
  # Tauri's config overlays are merged in order. Disabling the default
  # beforeDevCommand avoids starting a second Vite on the same port.
  bun tauri dev --config src-tauri/tauri.dev.conf.json \
    --config '{"build":{"beforeDevCommand":null}}'
else
  bun tauri dev --config src-tauri/tauri.dev.conf.json
fi
