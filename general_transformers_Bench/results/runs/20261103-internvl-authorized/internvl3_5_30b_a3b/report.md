# OpenGVLab/InternVL3_5-30B-A3B shared-suite run

Generated: 2026-11-03 10:00:17 +0100
Run status: failed
Backend: Transformers (local cached checkpoint); Transformers 5.19.0; PyTorch 2.13.0+cu130.
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
Load peak allocated/reserved: None / None GiB; sampled process peak: 0.486 GiB (None).

The 26 GiB target applies to torch.cuda.max_memory_allocated. The 30 GiB hard cap uses sampled per-process nvidia-smi memory.
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
Cache snapshot: `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--OpenGVLab--InternVL3_5-30B-A3B/snapshots/ba1762d830e2aeddb3272b66ba9e0abf7e69f626`.
Cache status: ready; checkpoint weight files: 57.46 GiB.
InternVL uses pinned revision `None`, NF4 double quantization with BF16 compute, official custom chat, and 448 px dynamic tiling with max_num=12 plus thumbnail. Per-item tile counts are stored in `results.json`.
Raw per-item outputs and errors are in `results.json`.

## Loader diagnosis

Transformers 5.19 BNB NF4 quantizes Linear4bit modules, but the pinned Qwen3-MoE fused expert tensors remain BF16. A CPU meta-skeleton audit found 30,848,730,112 parameters and only 1,220,542,464 eligible for NF4 in 290 modules (3.96% of parameters). The loader therefore requested 30.15 GiB during allocator warmup with 29.93 GiB free; no inference started.

The diagnostic used a CPU meta skeleton only; it did not load checkpoint weights or execute inference. The current loader is not retried without a reviewed route that quantizes the fused expert weights while preserving the pinned checkpoint and memory limits.
