# google/paligemma2-3b-mix-224 shared-suite run

Generated: 2026-11-03 02:01:42 +0100
Run status: complete
Backend: Transformers (local cached checkpoint); Transformers 4.57.1; PyTorch 2.14.1.
Seed: 3407; memory ceiling: 30.0 GiB.
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
| TextVQA / OCR | 100/100 | 57.6% |
| Truth | 25/25 | 97.2% |
| Counting | 25/25 | 78.0% |
| Description | 25/25 | 69.2% |
| Affordance | 25/25 | 8.0% |
| Negation | 25/25 | 52.0% |
| GRAB-Lite | 500/500 | 2.6% |

Full-suite macro: 54.4%.

## Runtime and resources

Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Parse errors |
|---|---:|---:|---:|---:|
| AI2D | 0.08 | 26.59 | 5.74 | 0 |
| TextVQA / OCR | 0.07 | 56.52 | 5.73 | 0 |
| Truth | 0.04 | 49.88 | 5.73 | 0 |
| Counting | 0.04 | 51.10 | 5.73 | 0 |
| Description | 0.04 | 51.92 | 5.73 | 0 |
| Affordance | 0.05 | 48.35 | 5.73 | 0 |
| Negation | 0.04 | 47.08 | 5.74 | 0 |
| GRAB-Lite | 0.13 | 37.48 | 5.76 | 17 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78`.
Cache status: ready; checkpoint weight files: 5.65 GiB.
Raw per-item outputs and errors are in `results.json`.
