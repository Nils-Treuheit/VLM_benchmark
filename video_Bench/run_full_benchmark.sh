#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE="/home/nils/Project_Code/codex_opencode_pipeline"
PYTHON="/mnt/HDD1/Project_Code/robotics_VLM/.venv/bin/python"
cd "$ROOT"

while read -r model track; do
  release_output="$(cd "$PIPELINE" && python3 src/executor.py release-local)"
  printf '%s\n' "$release_output"
  if [[ "$release_output" == *cleanup_failed* ]]; then
    echo "Local model release was incomplete; refusing to start another model." >&2
    exit 3
  fi
  if ! "$PYTHON" video_Bench/run_video_benchmark.py --model "$model" --track "$track"; then
    printf 'Run stopped or failed: %s / %s\n' "$model" "$track" >&2
  fi
done <<'RUNS'
internvideo3_8b_instruct mvbench
internvideo3_8b_instruct vue_tr
timelens2_8b vue_tr
llava_next_video_7b mvbench
llava_next_video_7b vue_tr
videochat3_4b mvbench
videochat3_4b vue_tr
RUNS

"$PYTHON" video_Bench/aggregate_results.py
