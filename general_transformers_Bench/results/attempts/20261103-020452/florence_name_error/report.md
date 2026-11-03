# microsoft/Florence-2-large-ft shared-suite run

Generated: 2026-11-03 02:07:01 +0100
Run status: failed
Backend: Transformers (local cached checkpoint); Transformers 4.44.2; PyTorch 2.15.0.
Seed: 3407; memory ceiling: 30.0 GiB.
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

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Parse errors |
|---|---:|---:|---:|---:|
| AI2D | — | — | — | 0 |
| TextVQA / OCR | — | — | — | 0 |
| Truth | — | — | — | 0 |
| Counting | — | — | — | 0 |
| Description | — | — | — | 0 |
| Affordance | — | — | — | 0 |
| Negation | — | — | — | 0 |
| GRAB-Lite | — | — | — | 0 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Florence-2-large-ft/snapshots/4a12a2b54b7016a48a22037fbd62da90cd566f2a`.
Cache status: ready; checkpoint weight files: 1.44 GiB.
Raw per-item outputs and errors are in `results.json`.
