# microsoft/Florence-2-large-ft shared-suite run

Generated: 2026-11-03 02:08:36 +0100
Run status: complete
Backend: Transformers (local cached checkpoint); Transformers 4.44.2; PyTorch 2.15.0.
Seed: 3407; memory ceiling: 30.0 GiB.
All eight shared tasks run only after AI2D format and TextVQA real-image preflights pass.
The preflight AI2D and TextVQA requests are reused in the scored task results.
Florence's `<VQA>` route is experimental because the cached processor does not register that task; it is promoted for scoring only after both real-image checks pass.
Full-suite macro is the unweighted mean of all eight task scores and is withheld unless every requested example completes.

## Preflight

| Task | Passed | Answer format | Parse error | Output tokens |
|---|---:|---|---:|---:|
| AI2D | yes | valid | no | 9 |
| TextVQA / OCR | yes | valid | no | 8 |

## Scores

| Task | Completed / requested | Score |
|---|---:|---:|
| AI2D | 100/100 | 2.0% |
| TextVQA / OCR | 100/100 | 45.4% |
| Truth | 25/25 | 48.8% |
| Counting | 25/25 | 53.6% |
| Description | 25/25 | 45.6% |
| Affordance | 25/25 | 0.0% |
| Negation | 25/25 | 12.0% |
| GRAB-Lite | 500/500 | 0.0% |

Full-suite macro: 25.9%.

## Runtime and resources

Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Parse errors |
|---|---:|---:|---:|---:|
| AI2D | 0.07 | 105.45 | 1.74 | 82 |
| TextVQA / OCR | 0.09 | 150.00 | 1.74 | 0 |
| Truth | 0.07 | 119.78 | 1.74 | 0 |
| Counting | 0.07 | 129.23 | 1.74 | 0 |
| Description | 0.07 | 127.33 | 1.74 | 0 |
| Affordance | 0.23 | 211.08 | 1.74 | 0 |
| Negation | 0.08 | 130.16 | 1.74 | 16 |
| GRAB-Lite | 0.14 | 48.70 | 1.74 | 467 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Florence-2-large-ft/snapshots/4a12a2b54b7016a48a22037fbd62da90cd566f2a`.
Cache status: ready; checkpoint weight files: 1.44 GiB.
For AI2D and negation choice questions, an exact generated option-text match is mapped to its displayed label before shared scoring. The raw generated text is retained in each item's post-processing record.
Raw per-item outputs and errors are in `results.json`.
