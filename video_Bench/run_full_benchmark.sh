#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/HDD1/Project_Code/VLM_benchmark"
PIPELINE="/home/nils/Project_Code/codex_opencode_pipeline"
PYTHON="/mnt/HDD1/Project_Code/robotics_VLM/.venv/bin/python"
cd "$ROOT"

while read -r model track; do
  python3 "$PIPELINE/src/executor.py" release-local
  "$PYTHON" video_Bench/run_video_benchmark.py --model "$model" --track "$track"
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
