#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PIPELINE_DIR="${PIPELINE_DIR:-/home/nils/Project_Code/codex_opencode_pipeline}"
PYTHON="${WORKSPACE_DIR}/.venv/bin/python"
LOCK_FILE="${TMPDIR:-/tmp}/phi_Bench.benchmark.lock"
cd "$PROJECT_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another Phi benchmark run is already active." >&2
    exit 3
fi
if [[ ! -f "$PIPELINE_DIR/src/executor.py" ]]; then
    echo "Model-control executor not found at $PIPELINE_DIR/src/executor.py." >&2
    echo "Set PIPELINE_DIR to the codex_opencode_pipeline checkout." >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Workspace Python not found at $PYTHON." >&2
    exit 2
fi
python3 "$PIPELINE_DIR/src/executor.py" release-local
uv pip install --python "$PYTHON" -r "$PROJECT_DIR/requirements-phi.txt"
"$PYTHON" benchmark.py "$@"
"$PYTHON" "$PROJECT_DIR/generate_data.py"
