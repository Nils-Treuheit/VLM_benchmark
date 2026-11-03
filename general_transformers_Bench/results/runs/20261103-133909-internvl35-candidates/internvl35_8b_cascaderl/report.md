# OpenGVLab/InternVL3_5-8B shared-suite run

Generated: 2026-11-03 14:00:28 +0100
Run status: complete
Backend: Transformers (official InternVL3.5 HF-format checkpoint); Transformers 4.57.1; PyTorch 2.14.1+cu130.
Seed: 3407; memory ceiling: 30.0 GiB.
All eight shared tasks run only after AI2D format and TextVQA real-image preflights pass.
The preflight AI2D and TextVQA requests are reused in the scored task results.
Florence's `<VQA>` route is experimental because the cached processor does not register that task; it is promoted for scoring only after both real-image checks pass.
Full-suite macro is the unweighted mean of all eight task scores and is withheld unless every requested example completes.

## Preflight

| Task | Passed | Answer format | Parse error | Output tokens |
|---|---:|---|---:|---:|
| AI2D | yes | valid | no | 1 |
| TextVQA / OCR | yes | valid | no | 2 |

## Scores

| Task | Completed / requested | Score |
|---|---:|---:|
| AI2D | 100/100 | 78.0% |
| TextVQA / OCR | 100/100 | 78.0% |
| Truth | 25/25 | 92.8% |
| Counting | 25/25 | 73.2% |
| Description | 25/25 | 64.8% |
| Affordance | 25/25 | 8.0% |
| Negation | 25/25 | 68.0% |
| GRAB-Lite | 500/500 | 14.6% |

Full-suite macro: 59.7%.

## Runtime and resources
Load peak allocated/reserved: 15.886 / 15.924 GiB; sampled process peak: 17.65 GiB (nvidia-smi query-compute-apps sampled every 0.5 seconds).

The 26 GiB target applies to torch.cuda.max_memory_allocated. The 30 GiB hard cap uses sampled per-process nvidia-smi memory.
Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Peak reserved GiB | Sampled process peak GiB | Parse errors |
|---|---:|---:|---:|---:|---:|---:|
| AI2D | 0.34 | 5.34 | 16.77 | 17.04 | 17.65 | 0 |
| TextVQA / OCR | 0.37 | 8.88 | 16.77 | 17.04 | 17.65 | 0 |
| Truth | 0.31 | 3.25 | 16.77 | 16.94 | 17.65 | 0 |
| Counting | 0.30 | 3.62 | 16.77 | 16.94 | 17.65 | 0 |
| Description | 0.27 | 6.16 | 16.77 | 16.94 | 17.65 | 0 |
| Affordance | 0.35 | 4.62 | 16.77 | 17.04 | 17.65 | 0 |
| Negation | 0.31 | 3.18 | 16.77 | 17.04 | 17.65 | 0 |
| GRAB-Lite | 0.44 | 5.88 | 16.77 | 17.04 | 17.65 | 0 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/internvl3_5_candidates/InternVL3_5-8B-CascadeRL`.
Cache status: ready; checkpoint weight files: 15.89 GiB.
InternVL uses pinned revision `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79`, none; official BF16 checkpoint weights, official custom chat, and 448 px dynamic tiling with max_num=12 plus thumbnail. Per-item tile counts are stored in `results.json`.
Raw per-item outputs and errors are in `results.json`.
