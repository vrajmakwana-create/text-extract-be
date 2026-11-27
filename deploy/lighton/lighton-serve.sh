#!/usr/bin/env bash
# Serve LightOnOCR-3-0.8B with vLLM on 127.0.0.1:8000 (OpenAI-compatible API).
# Sized for a g6f.2xlarge: a 6 GB slice of an L4, of which 5.78 GiB is usable.
#   --gpu-memory-utilization 0.85 fails: only ~4.66 GiB is free at startup.
#   VLLM_USE_BREAKABLE_CUDAGRAPH=0: the default graph capture runs out of memory on the vGPU.
#   --max-num-seqs 8: smaller CUDA graphs; the backend sends at most 8 requests at once.
# On a full GPU (e.g. g6.xlarge, 24 GB) you can raise the memory and --max-num-seqs.
cd "$HOME/workspace"
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
exec .venv-vllm/bin/vllm serve lightonai/LightOnOCR-3-0.8B \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization "${VLLM_GPU_MEM:-0.75}" \
  --max-model-len 8192 \
  --max-num-seqs "${VLLM_MAX_SEQS:-8}" \
  --limit-mm-per-prompt '{"image": 1}'
