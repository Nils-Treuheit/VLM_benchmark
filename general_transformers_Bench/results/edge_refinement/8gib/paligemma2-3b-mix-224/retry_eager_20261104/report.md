# google/paligemma2-3b-mix-224 shared-suite run

Generated: 2026-11-04 21:25:01 +0100
Run status: context_probe_failed
Backend: Transformers (local cached checkpoint); Transformers 5.22.1; PyTorch 2.14.1+cu130.
Seed: 3407; memory ceiling: 8.0 GiB.
All eight shared tasks run only after AI2D format and TextVQA real-image preflights pass.
The preflight AI2D and TextVQA requests are reused in the scored task results.
Florence's `<VQA>` route is experimental because the cached processor does not register that task; it is promoted for scoring only after both real-image checks pass.
Full-suite macro is the unweighted mean of all eight task scores and is withheld unless every requested example completes.

## Preflight

| Task | Passed | Answer format | Parse error | Output tokens |
|---|---:|---|---:|---:|

## Scores

| Task | Completed / requested | Score |
|---|---:|---:|
| AI2D | 0/100 | — |
| TextVQA / OCR | 0/100 | — |
| Truth | 0/25 | — |
| Counting | 0/25 | — |
| Description | 0/25 | — |
| Affordance | 0/25 | — |
| Negation | 0/25 | — |
| GRAB-Lite | 0/500 | — |

Full-suite macro: —.

## Runtime and resources

Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Peak reserved GiB | Sampled process peak GiB | Parse errors |
|---|---:|---:|---:|---:|---:|---:|
| AI2D | — | — | — | — | — | 0 |
| TextVQA / OCR | — | — | — | — | — | 0 |
| Truth | — | — | — | — | — | 0 |
| Counting | — | — | — | — | — | 0 |
| Description | — | — | — | — | — | 0 |
| Affordance | — | — | — | — | — | 0 |
| Negation | — | — | — | — | — | 0 |
| GRAB-Lite | — | — | — | — | — | 0 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78`.
Cache status: ready; checkpoint weight files: 5.65 GiB.
Raw per-item outputs and errors are in `results.json`.

## Edge cap verification

Whole-device cap: 8.0 GiB; idle baseline: 0.831 GiB.
Whole-device peak immediately after model load: 7.082 GiB.
Whole-device peak from model load through scoring: 8.039 GiB.
Scoring peak across completed tasks: None GiB.
Cap verified: `False`; verification: `{'whole_device_cap_gib': 8.0, 'load_peak_whole_device_gib': 7.082, 'load_peak_within_cap': True, 'context_probe_passed': False, 'complete_shared_task_suite': False, 'scored_items_within_probed_context': True, 'scoring_peak_within_cap': False, 'whole_device_sampling_verified': True, 'whole_device_sampler_error': None, 'task_sampling_verified': False, 'whole_device_peak_within_cap': False}`.

### Real-image context probe

Passed: `False`; AI2D index: 2426; image: [415, 376] px; measured context: 2291 tokens; input/output: 2290/1 tokens; probe peak: 8.0390625 GiB.
Image selection: maximum measured total input-token proxy across fixed AI2D examples; visual-token counts unavailable for the full sample; configured limit reached: `False`.
