# google/paligemma2-3b-mix-224 shared-suite run

Generated: 2026-11-04 21:28:00 +0100
Run status: complete
Backend: Transformers (local cached checkpoint); Transformers 5.22.1; PyTorch 2.14.1+cu130.
Seed: 3407; memory ceiling: 8.0 GiB.
All eight shared tasks run only after AI2D format and TextVQA real-image preflights pass.
The preflight AI2D and TextVQA requests are reused in the scored task results.
Florence's `<VQA>` route is experimental because the cached processor does not register that task; it is promoted for scoring only after both real-image checks pass.
Full-suite macro is the unweighted mean of all eight task scores and is withheld unless every requested example completes.

## Preflight

| Task | Passed | Answer format | Parse error | Output tokens |
|---|---:|---|---:|---:|
| AI2D | yes | valid | no | 2 |
| TextVQA / OCR | yes | valid | no | 2 |

## Scores

| Task | Completed / requested | Score |
|---|---:|---:|
| AI2D | 100/100 | 71.0% |
| TextVQA / OCR | 100/100 | 58.0% |
| Truth | 25/25 | 97.2% |
| Counting | 25/25 | 78.0% |
| Description | 25/25 | 69.2% |
| Affordance | 25/25 | 8.0% |
| Negation | 25/25 | 52.0% |
| GRAB-Lite | 500/500 | 2.8% |

Full-suite macro: 54.5%.

## Runtime and resources

Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Peak reserved GiB | Sampled process peak GiB | Parse errors |
|---|---:|---:|---:|---:|---:|---:|
| AI2D | 0.04 | 46.60 | 5.74 | 6.50 | — | 0 |
| TextVQA / OCR | 0.07 | 55.78 | 5.74 | 6.50 | — | 0 |
| Truth | 0.04 | 47.45 | 5.73 | 5.76 | — | 0 |
| Counting | 0.04 | 47.65 | 5.73 | 5.76 | — | 0 |
| Description | 0.04 | 50.08 | 5.73 | 5.76 | — | 0 |
| Affordance | 0.05 | 48.00 | 5.73 | 5.76 | — | 0 |
| Negation | 0.04 | 46.30 | 5.74 | 5.77 | — | 0 |
| GRAB-Lite | 0.12 | 40.78 | 5.76 | 5.82 | — | 16 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78`.
Cache status: ready; checkpoint weight files: 5.65 GiB.
Raw per-item outputs and errors are in `results.json`.

## Edge cap verification

Whole-device cap: 8.0 GiB; idle baseline: 0.831 GiB.
Whole-device peak immediately after model load: 7.082 GiB.
Whole-device peak from model load through scoring: 7.955 GiB.
Scoring peak across completed tasks: 7.271 GiB.
Cap verified: `True`; verification: `{'whole_device_cap_gib': 8.0, 'load_peak_whole_device_gib': 7.082, 'load_peak_within_cap': True, 'context_probe_passed': True, 'complete_shared_task_suite': True, 'scored_items_within_probed_context': True, 'scoring_peak_within_cap': True, 'whole_device_sampling_verified': True, 'whole_device_sampler_error': None, 'task_sampling_verified': True, 'whole_device_peak_within_cap': True}`.

### Real-image context probe

Passed: `True`; AI2D index: 2426; image: [415, 376] px; measured context: 1800 tokens; input/output: 1799/1 tokens; probe peak: 7.951171875 GiB.
Image selection: maximum measured total input-token proxy across fixed AI2D examples; visual-token counts unavailable for the full sample; configured limit reached: `True`.
