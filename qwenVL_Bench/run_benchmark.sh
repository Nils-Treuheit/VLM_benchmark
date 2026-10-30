#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-/home/nils/Project_Code/codex_opencode_pipeline}"
cd "$PROJECT_DIR"
if [[ ! -f "$PIPELINE_DIR/src/executor.py" ]]; then
    echo "Model-control executor not found at $PIPELINE_DIR/src/executor.py." >&2
    echo "Set PIPELINE_DIR to the codex_opencode_pipeline checkout." >&2
    exit 2
fi
python3 "$PIPELINE_DIR/src/executor.py" release-local
uv run python benchmark_runners.py "$@"
