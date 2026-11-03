# bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q4_K_M shared-suite run

Generated: 2026-11-03 14:41:22 +0100
Run status: complete
Backend: llama.cpp CUDA (GGUF plus matching mmproj); Transformers 4.57.1; PyTorch 2.14.1+cu130.
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
| TextVQA / OCR | 100/100 | 71.4% |
| Truth | 25/25 | 49.2% |
| Counting | 25/25 | 55.2% |
| Description | 25/25 | 56.4% |
| Affordance | 25/25 | 4.0% |
| Negation | 25/25 | 76.0% |
| GRAB-Lite | 500/500 | 16.2% |

Full-suite macro: 50.3%.

## Runtime and resources
Load peak allocated/reserved: None / None GiB; sampled process peak: 19.48 GiB (nvidia-smi query-compute-apps sampled llama-server every 0.5 seconds).

The 26 GiB target applies to torch.cuda.max_memory_allocated. The 30 GiB hard cap uses sampled per-process nvidia-smi memory.
Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Peak reserved GiB | Sampled process peak GiB | Parse errors |
|---|---:|---:|---:|---:|---:|---:|
| AI2D | 0.49 | 4.12 | 19.48 | — | 19.48 | 2 |
| TextVQA / OCR | 1.18 | 3.98 | 19.48 | — | 19.48 | 0 |
| Truth | 0.54 | 11.07 | 19.48 | — | 19.48 | 0 |
| Counting | 0.48 | 4.84 | 19.48 | — | 19.48 | 0 |
| Description | 0.45 | 7.93 | 19.48 | — | 19.48 | 0 |
| Affordance | 1.15 | 2.57 | 19.48 | — | 19.48 | 0 |
| Negation | 0.85 | 27.12 | 19.48 | — | 19.48 | 1 |
| GRAB-Lite | 2.18 | 7.29 | 19.48 | — | 19.48 | 0 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/internvl3_5_candidates/InternVL3_5-30B-A3B-GGUF`.
Cache status: ready; checkpoint weight files: 17.94 GiB.
InternVL uses pinned revision `a6824f88ffe59c72fc9864a87d40753556cba5d9`, Q4_K_M, official custom chat, and 448 px dynamic tiling with max_num=12 plus thumbnail. Per-item tile counts are stored in `results.json`.
Raw per-item outputs and errors are in `results.json`.
