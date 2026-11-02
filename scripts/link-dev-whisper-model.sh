#!/usr/bin/env bash
# Reuse the already-downloaded Handy Whisper model in the isolated dev app.
# This creates a symlink only: no model download, copy or destructive writes.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="whisper-large-v3-Q5_K_M.gguf"
DEST="$HOME/Library/Application Support/com.pais.handy.dev/models/$NAME"

# Prefer the local server's existing symlink, which the user already verified
# when setting up GGUF meeting transcription.
MODEL="$ROOT/server/models/$NAME"

if [[ ! -s "$MODEL" ]]; then
  CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--handy-computer--whisper-large-v3-gguf/snapshots"
  if [[ -d "$CACHE" ]]; then
    MODEL="$(find -L "$CACHE" -type f -name "$NAME" -print -quit)"
  fi
fi

# No model to reuse. Preserve the normal onboarding model chooser.
if [[ -z "$MODEL" || ! -s "$MODEL" ]]; then
  echo "[Handy Dev] No pre-existing Whisper Large V3 GGUF found; skipping link."
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
if [[ ! -e "$DEST" && ! -L "$DEST" ]]; then
  ln -s "$MODEL" "$DEST"
  echo "[Handy Dev] Existing Whisper Large V3 linked without downloading."
elif [[ -s "$DEST" ]]; then
  echo "[Handy Dev] Whisper Large V3 already available in the dev app."
else
  echo "[Handy Dev] Existing model link is broken: $DEST" >&2
  exit 1
fi

# Show only file size and path, never print model data.
ls -lhL "$DEST"
