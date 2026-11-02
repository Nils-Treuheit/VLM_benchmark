#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/mnt/HDD1/unsloth_and_hugging_face_models/huggingface}"
exec "$HERE/.venv/bin/python" "$@"
