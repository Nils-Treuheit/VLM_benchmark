# unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit shared-suite run

Generated: 2026-11-03 16:22:46 +0100
Run status: complete
Backend: Transformers (Llama 3.2 Vision BNB 4-bit); Transformers 4.57.1; PyTorch 2.14.1+cu130.
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
| AI2D | 100/100 | 74.0% |
| TextVQA / OCR | 100/100 | 66.8% |
| Truth | 25/25 | 53.2% |
| Counting | 25/25 | 24.0% |
| Description | 25/25 | 29.6% |
| Affordance | 25/25 | 0.0% |
| Negation | 25/25 | 24.0% |
| GRAB-Lite | 500/500 | 7.0% |

Full-suite macro: 34.8%.

## Runtime and resources

Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Peak reserved GiB | Sampled process peak GiB | Parse errors |
|---|---:|---:|---:|---:|---:|---:|
| AI2D | 0.32 | 7.58 | 7.61 | 9.74 | — | 0 |
| TextVQA / OCR | 0.37 | 14.99 | 7.61 | 9.74 | — | 0 |
| Truth | 0.37 | 16.18 | 7.61 | 9.74 | — | 0 |
| Counting | 0.40 | 18.65 | 7.61 | 9.74 | — | 0 |
| Description | 0.39 | 17.66 | 7.61 | 9.74 | — | 0 |
| Affordance | 1.02 | 38.49 | 7.61 | 9.78 | — | 0 |
| Negation | 0.76 | 34.42 | 7.61 | 9.78 | — | 2 |
| GRAB-Lite | 1.29 | 39.68 | 7.61 | 9.80 | — | 3 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/llama3_2_vision_candidates/Llama-3.2-11B-Vision-Instruct-bnb-4bit`.
Cache status: ready; checkpoint weight files: 6.69 GiB.
Raw per-item outputs and errors are in `results.json`.
