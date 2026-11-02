#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/mnt/HDD1/unsloth_and_hugging_face_models/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/mnt/HDD1/unsloth_and_hugging_face_models/vllm_cache
export TRITON_CACHE_DIR=/mnt/HDD1/unsloth_and_hugging_face_models/triton_cache
export XDG_CACHE_HOME=/mnt/HDD1/unsloth_and_hugging_face_models/xdg_cache
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

exec /mnt/HDD1/Project_Code/VLM_benchmark/cosmos_Bench/.venv/bin/vllm serve nvidia/Cosmos3-Nano \
  --host 0.0.0.0 \
  --port 20265 \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --gpu-memory-utilization 0.92
