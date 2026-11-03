# bartowski/OpenGVLab_InternVL3_5-14B-GGUF:Q4_K_M shared-suite run

Generated: 2026-11-03 15:04:29 +0100
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
| AI2D | 100/100 | 79.0% |
| TextVQA / OCR | 100/100 | 65.9% |
| Truth | 25/25 | 57.2% |
| Counting | 25/25 | 22.4% |
| Description | 25/25 | 53.2% |
| Affordance | 25/25 | 4.0% |
| Negation | 25/25 | 72.0% |
| GRAB-Lite | 500/500 | 14.8% |

Full-suite macro: 46.1%.

## Runtime and resources
Load peak allocated/reserved: None / None GiB; sampled process peak: 10.74 GiB (nvidia-smi query-compute-apps sampled llama-server every 0.5 seconds).

The 26 GiB target applies to torch.cuda.max_memory_allocated. The 30 GiB hard cap uses sampled per-process nvidia-smi memory.
Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.

| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Peak reserved GiB | Sampled process peak GiB | Parse errors |
|---|---:|---:|---:|---:|---:|---:|
| AI2D | 0.47 | 15.44 | 10.74 | — | 10.74 | 1 |
| TextVQA / OCR | 1.09 | 4.62 | 10.74 | — | 10.74 | 0 |
| Truth | 0.50 | 9.82 | 10.74 | — | 10.74 | 0 |
| Counting | 0.44 | 7.47 | 10.74 | — | 10.74 | 0 |
| Description | 0.41 | 8.02 | 10.74 | — | 10.74 | 0 |
| Affordance | 1.04 | 2.95 | 10.74 | — | 10.74 | 0 |
| Negation | 0.83 | 14.26 | 10.74 | — | 10.74 | 0 |
| GRAB-Lite | 3.42 | 51.71 | 10.74 | — | 10.74 | 3 |

## Method and provenance

Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.
See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/internvl3_5_candidates/InternVL3_5-14B-GGUF`.
Cache status: ready; checkpoint weight files: 9.04 GiB.
InternVL uses pinned revision `55921358acc81c09965257918cfa4e6f6bd9a1c0`, Q4_K_M, official custom chat, and 448 px dynamic tiling with max_num=12 plus thumbnail. Per-item tile counts are stored in `results.json`.
Raw per-item outputs and errors are in `results.json`.
