# allenai/Molmo2-8B shared-suite run

Generated: 2026-11-03 02:16:01 +0100
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
| TextVQA / OCR | yes | valid | no | 4 |

## Scores

| Task | Completed / requested | Score |
|---|---:|---:|
| AI2D | 100/100 | 87.0% |
| TextVQA / OCR | 100/100 | 74.2% |
| Truth | 25/25 | 8.0% |
| Counting | 25/25 | 12.0% |
| Description | 25/25 | 21.2% |
| Affordance | 25/25 | 4.0% |
| Negation | 25/25 | 80.0% |
| GRAB-Lite | 500/500 | 13.8% |

Full-suite macro: 37.5%.

## Runtime and resources

Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Parse errors |
|---|---:|---:|---:|---:|
| AI2D | 0.26 | 7.59 | 17.52 | 0 |
| TextVQA / OCR | 0.25 | 18.34 | 17.52 | 0 |
| Truth | 0.23 | 31.60 | 17.45 | 0 |
| Counting | 0.16 | 21.40 | 17.45 | 0 |
| Description | 0.18 | 25.06 | 17.45 | 0 |
| Affordance | 0.23 | 16.04 | 17.52 | 0 |
| Negation | 0.16 | 12.27 | 17.52 | 0 |
| GRAB-Lite | 0.26 | 13.76 | 17.53 | 0 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b`.
Cache status: ready; checkpoint weight files: 32.27 GiB.
Molmo2 uses its processor chat template with the unchanged shared question and decodes only the generated completion. The cached weights are loaded in bfloat16.
Raw per-item outputs and errors are in `results.json`.
