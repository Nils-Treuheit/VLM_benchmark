# General Transformers VLM benchmark

This suite compares locally available Transformers VLMs on the shared image tasks from Qwen-VL. Model-specific loading and prompt wrappers are in `model_adapters.py`.

## Current results

Molmo2, PaliGemma 2, and Florence-2 completed all eight shared task groups. InternVL 3.5 attempts failed before completing preflight, and Llama 3.2 Vision remains unavailable because its local checkpoint is incomplete. See the [interactive report](index.html) for scores and coverage.

## Run

Run in host context with prepared shared data and locally cached checkpoints:

```bash
./run_benchmark.sh
```

The runner releases managed model memory and evaluates one model at a time. Results go under `results/runs/`; the aggregate report and page are generated from completed run artifacts.

Per-model setup and candidate details are in the local `README.agent.md` and `environments/`.
