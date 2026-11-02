#!/usr/bin/env bash
# Reuse a GGUF Whisper model already downloaded by the Handy desktop.
# This script links a local file and builds a CLI; it never downloads model weights.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is intended for macOS." >&2
  exit 1
fi

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$SERVER_DIR/models" "$SERVER_DIR/tools"

MODEL_SOURCE="${HANDY_GGUF_SOURCE:-}"
if [[ -z "$MODEL_SOURCE" ]]; then
  CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--handy-computer--whisper-large-v3-gguf/snapshots"
  if [[ ! -d "$CACHE" ]]; then
    echo "Whisper GGUF cache not found at $CACHE" >&2
    echo "Set HANDY_GGUF_SOURCE to the existing model file and rerun setup." >&2
    exit 1
  fi
  MODEL_SOURCE="$(find -L "$CACHE" -type f -name 'whisper-large-v3-Q5_K_M.gguf' -print -quit)"
fi

if [[ ! -s "$MODEL_SOURCE" ]]; then
  echo "Existing Whisper GGUF file not found or empty: $MODEL_SOURCE" >&2
  exit 1
fi

DEST="$SERVER_DIR/models/whisper-large-v3-Q5_K_M.gguf"
if [[ ! -e "$DEST" && ! -L "$DEST" ]]; then
  ln -s "$MODEL_SOURCE" "$DEST"
elif [[ ! -s "$DEST" ]]; then
  echo "Model link already exists but is broken: $DEST" >&2
  exit 1
fi
echo "Using local Whisper model (no weight download):"
ls -lhL "$DEST"

SOURCE="$SERVER_DIR/tools/transcribe.cpp"
BINARY="$SOURCE/build/bin/transcribe-cli"
if [[ ! -x "$BINARY" ]]; then
  if ! command -v cmake >/dev/null 2>&1; then
    echo "CMake is required. Install it with: brew install cmake" >&2
    exit 1
  fi
  if [[ ! -d "$SOURCE/.git" ]]; then
    if [[ -e "$SOURCE" ]]; then
      echo "Source directory already exists but is not a Git checkout: $SOURCE" >&2
      exit 1
    fi
    git clone --depth 1 https://github.com/handy-computer/transcribe.cpp.git "$SOURCE"
  fi
  cmake -S "$SOURCE" -B "$SOURCE/build" -DCMAKE_BUILD_TYPE=Release \
    -DTRANSCRIBE_METAL=ON -DTRANSCRIBE_BUILD_TESTS=OFF
  cmake --build "$SOURCE/build" --target transcribe-cli -j 4
fi

test -x "$BINARY"
echo "GGUF transcription CLI is ready at: $BINARY"
echo "Next: bash $SERVER_DIR/scripts/run_local_mac.sh"
