#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/HDD1/unsloth_and_hugging_face_models/huggingface}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/.venv/bin/hf" ]]; then
  HF_CLI="${HF_CLI:-$SCRIPT_DIR/.venv/bin/hf}"
else
  HF_CLI="${HF_CLI:-$(command -v hf)}"
fi
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-32}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

download_full() {
  printf 'Downloading Transformers checkpoint: %s\n' "$1"
  "$HF_CLI" download "$1" --max-workers "$HF_DOWNLOAD_WORKERS"
}

download_config() {
  printf 'Downloading model-card config only: %s\n' "$1"
  "$HF_CLI" download "$1" config.json --max-workers "$HF_DOWNLOAD_WORKERS"
}

download_ud() {
  local repo="$1"
  shift
  local base="${repo##*/}"
  base="${base%-GGUF}"
  local files=()
  local quant
  for quant in "$@"; do
    files+=("${base}-${quant}.gguf")
  done
  printf 'Downloading UD GGUF files: %s (%s)\n' "$repo" "$*"
  "$HF_CLI" download "$repo" "${files[@]}" mmproj-F16.gguf --max-workers "$HF_DOWNLOAD_WORKERS"
}

# Official Qwen checkpoints use their Transformers model-card loading path.
download_full Qwen/Qwen2-VL-2B-Instruct
download_full Qwen/Qwen2-VL-7B-Instruct
download_full Qwen/Qwen2.5-VL-3B-Instruct
download_full Qwen/Qwen3-VL-4B-Instruct
download_full Qwen/Qwen3-VL-4B-Thinking
download_config Qwen/Qwen2.5-VL-32B-Instruct
download_config Qwen/Qwen2.5-VL-72B-Instruct
download_config Qwen/Qwen2-VL-72B-Instruct

# Keep architecture metadata for dense 32B GGUF memory/context estimates.
# The full precision weights are intentionally omitted because they exceed the
# 30-GiB VRAM budget; the runnable UD GGUFs are downloaded below.
download_config Qwen/Qwen3-VL-32B-Instruct
download_config Qwen/Qwen3-VL-32B-Thinking

# Core UD Dynamic 2.0 quantizations for newly discovered runnable sizes.
download_ud unsloth/Qwen3-VL-4B-Instruct-GGUF \
  UD-Q4_K_XL UD-Q8_K_XL
download_ud unsloth/Qwen3-VL-4B-Thinking-GGUF \
  UD-Q4_K_XL UD-Q8_K_XL
download_ud unsloth/Qwen3-VL-8B-Instruct-GGUF \
  UD-Q4_K_XL UD-Q8_K_XL
download_ud unsloth/Qwen3-VL-8B-Thinking-GGUF \
  UD-Q4_K_XL UD-Q8_K_XL
download_ud unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF \
  UD-Q4_K_XL
download_ud unsloth/Qwen3-VL-32B-Instruct-GGUF \
  UD-Q4_K_XL
download_ud unsloth/Qwen3-VL-32B-Thinking-GGUF \
  UD-Q4_K_XL

# Long-context variants at every size whose GGUF can still be useful under
# the 30-GiB cap. The 2B Q8 files let us compare quantization at 1M context.
download_ud unsloth/Qwen3-VL-2B-Instruct-1M-GGUF \
  UD-Q4_K_XL UD-Q8_K_XL
download_ud unsloth/Qwen3-VL-2B-Thinking-1M-GGUF \
  UD-Q4_K_XL UD-Q8_K_XL
download_ud unsloth/Qwen3-VL-4B-Instruct-1M-GGUF \
  UD-Q4_K_XL
download_ud unsloth/Qwen3-VL-4B-Thinking-1M-GGUF \
  UD-Q4_K_XL
download_ud unsloth/Qwen3-VL-8B-Instruct-1M-GGUF \
  UD-Q4_K_XL
download_ud unsloth/Qwen3-VL-8B-Thinking-1M-GGUF \
  UD-Q4_K_XL
download_ud unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF \
  UD-Q4_K_XL
download_ud unsloth/Qwen3-VL-30B-A3B-Thinking-1M-GGUF \
  UD-Q4_K_XL

# Qwen2.5-VL's larger official sizes do not fit in full precision. Include
# high-quality UD GGUFs that can plausibly fit the 30-GiB GPU budget.
download_ud unsloth/Qwen2.5-VL-32B-Instruct-GGUF \
  UD-Q4_K_XL UD-Q5_K_XL
download_ud unsloth/Qwen2.5-VL-72B-Instruct-GGUF \
  UD-IQ2_XXS

# The original Qwen2-VL 72B only has quants small enough to test near the
# 30-GiB ceiling. Follow bartowski's llama.cpp model-card file layout.
printf 'Downloading Qwen2-VL 72B GGUF and projector from model-card repo\n'
"$HF_CLI" download bartowski/Qwen2-VL-72B-Instruct-GGUF \
  Qwen2-VL-72B-Instruct-IQ2_XXS.gguf \
  mmproj-Qwen2-VL-72B-Instruct-f16.gguf \
  --max-workers "$HF_DOWNLOAD_WORKERS"
