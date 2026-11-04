# Benchmark plan index

Completed suites are tracked in their reports and the repository [README](../README.md). This index keeps only active benchmark work and shared reporting guidance.

## Shared benchmark contract

Every model that supports the required image and text inputs should run the same fixed tasks:

1. AI2D test, 100 examples, seed 3407, exact A/B/C/D accuracy.
2. TextVQA validation, 100 examples, seed 3407, existing normalized consensus scorer.
3. The six-category VQA panel: truth, counting, description, OCR, affordance, and negation. Reuse its prepared fixed examples and existing category scorers.
4. GRAB-Lite graph analysis, 500 examples, seed 3407, exact match and five group scores.

Keep the same prepared examples, question text, answer constraints, image resize policy, one image per request, no batching, and at most one active model. Use the same 30 GiB target where the selected backend supports that limit. Report unsupported tasks or models as unsupported or failed; do not silently omit them from a complete score or substitute a different task. Preserve per-item output, errors, latency, output tokens, peak VRAM, model identity, backend, and effective context in results.

The shared task prompts and score definitions are in [qwenVL_Bench/README.md](../qwenVL_Bench/README.md). The current runner and report format are in [benchmark.py](../qwenVL_Bench/benchmark.py) and [benchmark_runners.py](../qwenVL_Bench/benchmark_runners.py). Dataset preparation lives in [prepare_vqa_panel.py](../qwenVL_Bench/prepare_vqa_panel.py) and [prepare_graph_panel.py](../qwenVL_Bench/prepare_graph_panel.py). Existing prepared data under `qwenVL_Bench/data/` should be reused.

## Plans

| Group | Plan |
|---|---|
| Grounding specialists | [grounding-specialists.md](grounding-specialists.md) |
| Selective Edge Limited refinement | [selective-edge-refinement.md](selective-edge-refinement.md) |

The completed General Transformers, local-foundation, Qwen-VL reference, and video comparisons are documented in their benchmark reports. Candidates excluded by cache, compatibility, or hardware gates remain listed in the relevant report inventory; they are not pending runs.

Shared report-site guidance: [general VLM report-site guideline](general-vlm-report-site-guideline.md).

## Local model inventory

The inventory was read from `HF_HOME=/mnt/HDD1/unsloth_and_hugging_face_models/huggingface`. These entries exist in the cache. Before implementation, resolve each exact snapshot and required weight files without downloading anything. Cache presence alone does not establish completeness, loadability, image support, or license acceptance.

The benchmark implementation should discover only locally complete checkpoints, record the cache snapshot and backend configuration, and remain offline. Run GPU validation in host context after `release-local`, serially, as required by the repository instructions.

## Report site delivery

After each benchmark group completes its planned runs and writes `report.md`, place the complete static report page at `<benchmark-folder>/index.html`, with scripts, styles, data, and assets beside it. Do not use a nested `report-site/` or `dist/` directory, or a redirect page. Keep a repository-root `index.html` linking to every available benchmark page. Follow the [general VLM report site design guideline](general-vlm-report-site-guideline.md). Keep site data and recommendations traceable to the report and recorded results; preserve group-specific metrics and report incomplete or unsupported coverage clearly. The Pages artifact must preserve folder paths, so `/qwenVL_Bench/` resolves to `qwenVL_Bench/index.html`.
