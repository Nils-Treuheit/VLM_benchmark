# LLaVA family benchmark

This benchmark compares cached LLaVA-family Transformers checkpoints on the same fixed image tasks as `qwenVL_Bench`. It reuses that benchmark's prepared VQA panel data, prompts, sample selection, and scorers. The runner is offline and runs one checkpoint and one image request at a time.

## Run

Run from the project checkout in host context so the launcher can release local model memory before inference:

```bash
uv sync
uv run python ../qwenVL_Bench/prepare_vqa_panel.py
uv run python ../qwenVL_Bench/prepare_graph_panel.py
./run_benchmark.sh
```

The preparation commands create the shared panel files under `qwenVL_Bench/data/`. They may download source datasets and images if those are not cached. Inference reads the prepared files and cached checkpoints without downloading.

The full run uses 100 AI2D examples, 100 TextVQA examples, 25 examples for each additional VQA category, and all 500 GRAB-Lite examples. Every task uses seed 3407. Results are written to `results/llava_family/results.json` and `results/llava_family/report.md`.

Use `--list-models` to inspect exact cached snapshots and weight completeness without starting inference. Select exact model IDs with `--models`. Use `--tasks` to choose task IDs, `--samples` to apply one count to each selected task, and `--resume` to continue matching runs.

```bash
./run_benchmark.sh --list-models
./run_benchmark.sh --models llava-hf/LLaVA-NeXT-Video-7B-hf --tasks ai2d textvqa --samples 25
```

The 34B checkpoint is skipped when its cached weights alone exceed the 30 GiB ceiling. The current cached weights occupy about 64.7 GiB, before runtime memory. LLaVA-OneVision-2 uses its checkpoint-provided Transformers code and requires Transformers 5.7 or newer. The report records the Transformers version, and resume requires a version match. Partial and failed model runs remain in the JSON but do not qualify as complete-panel results.

## Tasks and measures

- **AI2D:** cached test split, exact A/B/C/D accuracy.
- **TextVQA / OCR:** cached validation split, existing normalized consensus score.
- **VQA panel:** truth, counting, description, affordance, and negation; the TextVQA result supplies OCR. Scores use the shared category scorers.
- **GRAB-Lite:** all 500 fixed graph questions, exact matching, and five task-group scores.

The processor applies each checkpoint's own single-image preprocessing and chat template. Shared question text and answer constraints stay unchanged. LLaVA-NeXT-Video receives still images through its image input path; no frames or timestamps are synthesized.

Results describe a fixed local sample, not leaderboard scores. Per-item predictions, outputs, errors, timing, token counts, peak allocated VRAM, checkpoint identity, and effective context are retained in the JSON.

## Report site

Generate the static report data and serve the site locally:

```bash
python report-site/generate_data.py
cd report-site/dist && python3 -m http.server 8174 --bind 127.0.0.1
```

Open <http://127.0.0.1:8174/>. The report shows the eight-task macro score, per-task accuracy, runtime metrics, and skipped-model status. It includes aggregate results only.
