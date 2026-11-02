#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$HERE/.venv"
UV=/home/nils/.local/bin/uv

"$UV" venv --python 3.13 --seed --managed-python "$ENV_DIR"
"$UV" pip install --python "$ENV_DIR/bin/python" --torch-backend=cu130 \
  "vllm==0.21.0" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git@55f0554d0bf167603432041c794a8babbac759b9#subdirectory=packages/vllm-cosmos3" \
  openai

"$ENV_DIR/bin/python" -c 'import importlib.metadata as m; print("vLLM", m.version("vllm")); print("vllm-cosmos3", m.version("vllm-cosmos3"))'
