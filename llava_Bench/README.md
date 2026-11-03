# LLaVA family benchmark

This benchmark compares cached LLaVA-family checkpoints on the same fixed image tasks as the [Qwen-VL benchmark](../qwenVL_Bench/README.md). It covers diagram questions, text reading, five additional VQA categories, and graph analysis.

The completed results use fixed samples and report task-level accuracy, runtime, and model coverage. See the [results report](results/llava_family/report.md) or open the [interactive report](index.html).

## Run

Prepare shared data once, then run selected local checkpoints in host context:

```bash
uv sync
uv run python ../qwenVL_Bench/prepare_vqa_panel.py
uv run python ../qwenVL_Bench/prepare_graph_panel.py
./run_benchmark.sh
```

Use `./run_benchmark.sh --list-models` to inspect complete local snapshots. Use `--models` to select exact model IDs, `--tasks` to select tasks, and `--resume` to continue matching runs.
