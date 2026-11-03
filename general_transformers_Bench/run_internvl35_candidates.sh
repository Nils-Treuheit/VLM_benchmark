#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-/home/nils/Project_Code/codex_opencode_pipeline}"
EXECUTOR="$PIPELINE_DIR/src/executor.py"
PYTHON_BIN="$PROJECT_DIR/../.venv/bin/python"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-internvl35-candidates}"
RUN_DIR="$PROJECT_DIR/results/runs/$RUN_ID"

if [[ ! -f "$EXECUTOR" ]]; then
    echo "Model-control executor not found at $EXECUTOR." >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Benchmark Python not found at $PYTHON_BIN." >&2
    exit 2
fi

mkdir -p "$RUN_DIR"
overall_status=0

run_model() {
    local model_id="$1"
    local output_dir="$2"
    if ! python3 "$EXECUTOR" release-local; then
        echo "release-local failed; not loading $model_id." >&2
        return 4
    fi
    "$PYTHON_BIN" "$PROJECT_DIR/benchmark.py" --model "$model_id" --output-dir "$output_dir"
}

run_one() {
    local model_id="$1"
    local slug="$2"
    local status=0
    if [[ -f "$RUN_DIR/$slug/results.json" ]] && "$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); m=r["model"]; tasks=m.get("tasks",{}); names=set(r.get("task_names",[])); ok=m.get("status")=="complete" and set(tasks)==names and bool(names) and all(t.get("completed_n")==t.get("requested_n") and t.get("failed_n")==0 for t in tasks.values()); raise SystemExit(0 if ok else 1)' "$RUN_DIR/$slug/results.json"; then
        echo "Candidate $model_id already completed all eight tasks; retaining its results."
        return 0
    fi
    run_model "$model_id" "$RUN_DIR/$slug" || status=$?
    if [[ "$status" -ne 0 ]]; then
        overall_status=1
        echo "Candidate $model_id ended with status $status; continuing serially." >&2
    elif ! "$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); m=r["model"]; tasks=m.get("tasks",{}); names=set(r.get("task_names",[])); ok=m.get("status")=="complete" and set(tasks)==names and bool(names) and all(t.get("completed_n")==t.get("requested_n") and t.get("failed_n")==0 for t in tasks.values()); raise SystemExit(0 if ok else 1)' \
        "$RUN_DIR/$slug/results.json"; then
        overall_status=1
        echo "Candidate $model_id did not complete all eight tasks; continuing serially." >&2
    fi
}

run_one "OpenGVLab/InternVL3_5-8B-Instruct" "internvl35_8b_instruct"
run_one "OpenGVLab/InternVL3_5-8B-MPO" "internvl35_8b_mpo"
run_one "OpenGVLab/InternVL3_5-8B" "internvl35_8b_cascaderl"
run_one "bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M" "internvl35_30b_a3b_q4_k_m_gguf"
run_one "bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M" "internvl35_14b_q4_k_m_gguf"
run_one "bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q8_0" "internvl35_14b_q8_0_gguf"

BASELINE_RUN_ID="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_id"])' \
    "$PROJECT_DIR/results/general_transformers/results.json")"
if [[ ! "$BASELINE_RUN_ID" =~ ^[[:alnum:]-]+$ ]]; then
    echo "Could not resolve the existing baseline run ID; candidate outputs were preserved." >&2
    exit 1
fi
BASELINE_DIR="$PROJECT_DIR/results/runs/$BASELINE_RUN_ID"
for slug in molmo2_8b paligemma2_mix_224 florence2_large_ft; do
    if [[ ! -f "$BASELINE_DIR/$slug/results.json" ]]; then
        echo "Completed baseline output is missing: $BASELINE_DIR/$slug/results.json" >&2
        exit 1
    fi
    ln -sfn "$BASELINE_DIR/$slug" "$RUN_DIR/$slug"
done

if [[ -f "$RUN_DIR/internvl35_8b_instruct/results.json" \
   && -f "$RUN_DIR/internvl35_8b_mpo/results.json" \
   && -f "$RUN_DIR/internvl35_8b_cascaderl/results.json" \
   && -f "$RUN_DIR/internvl35_30b_a3b_q4_k_m_gguf/results.json" \
   && -f "$RUN_DIR/internvl35_14b_q4_k_m_gguf/results.json" \
   && -f "$RUN_DIR/internvl35_14b_q8_0_gguf/results.json" ]]; then
    if "$PYTHON_BIN" "$PROJECT_DIR/aggregate_results.py" \
        --run-dir "$RUN_DIR" --include-internvl35-candidates; then
        "$PYTHON_BIN" "$PROJECT_DIR/generate_data.py" || overall_status=1
    else
        overall_status=1
    fi
else
    overall_status=1
    echo "At least one candidate has no result artifact; aggregate report was not changed." >&2
fi

echo "Candidate run artifacts: $RUN_DIR"
exit "$overall_status"
