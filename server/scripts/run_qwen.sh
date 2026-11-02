#!/usr/bin/env bash
# Requires a compatible version of vLLM (Qwen3.5 supported) and an NVIDIA GPU.
set -euo pipefail
exec vllm serve Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000 \
  --max-model-len "${HANDY_QWEN_MAX_CONTEXT:-8192}" \
  --gpu-memory-utilization "${HANDY_QWEN_GPU_UTILIZATION:-0.75}" \
  --trust-remote-code
