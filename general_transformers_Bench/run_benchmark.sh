#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-/home/nils/Project_Code/codex_opencode_pipeline}"
EXECUTOR="$PIPELINE_DIR/src/executor.py"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$PROJECT_DIR/results/runs/$RUN_ID"

if [[ ! -f "$EXECUTOR" ]]; then
    echo "Model-control executor not found at $EXECUTOR." >&2
    exit 2
fi

run_model() {
    local model_id="$1"
    local output_dir="$2"
    local python_bin="$3"

    if ! python3 "$EXECUTOR" release-local; then
        echo "release-local failed; skipping inference for $model_id." >&2
        return 4
    fi
    "$python_bin" "$PROJECT_DIR/benchmark.py" \
        --model "$model_id" --output-dir "$output_dir"
}

mkdir -p "$RUN_DIR"

set +e
run_model "google/paligemma2-3b-mix-224" \
    "$RUN_DIR/paligemma2_mix_224" \
    "$PROJECT_DIR/../.venv/bin/python"
paligemma_status=$?
set -e

# Florence has a separate Transformers runtime. The next release check guards
# the handoff and prevents overlap if the first process left any model active.
set +e
run_model "microsoft/Florence-2-large-ft" \
    "$RUN_DIR/florence2_large_ft" \
    "$PROJECT_DIR/environments/florence/.venv/bin/python"
florence_status=$?
set -e

aggregate_status=0
site_status=0
if [[ "$paligemma_status" -eq 0 && "$florence_status" -eq 0 \
      && -f "$RUN_DIR/paligemma2_mix_224/results.json" \
      && -f "$RUN_DIR/florence2_large_ft/results.json" ]]; then
    set +e
    "$PROJECT_DIR/../.venv/bin/python" "$PROJECT_DIR/aggregate_results.py" --run-dir "$RUN_DIR"
    aggregate_status=$?
    if [[ "$aggregate_status" -eq 0 ]]; then
    "$PROJECT_DIR/../.venv/bin/python" "$PROJECT_DIR/generate_data.py"
    site_status=$?
    else
        site_status=$aggregate_status
    fi
    set -e
else
    aggregate_status=1
    echo "Current model gates did not both pass. Existing aggregate report and site were left untouched." >&2
fi

if [[ "$paligemma_status" -ne 0 || "$florence_status" -ne 0 \
      || "$aggregate_status" -ne 0 || "$site_status" -ne 0 ]]; then
    echo "One or both model runs failed; inspect each model's report.md and results.json." >&2
    exit 1
fi
