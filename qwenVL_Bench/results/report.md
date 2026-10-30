# Qwen2/3-VL local benchmark

Generated: 2026-10-28 16:36:22 +0100
Sample size: 100 examples per task; seed 3407.
Hard limits: 30.0 GiB VRAM and 262,144 total tokens; 8-bit hqq KV cache.

## Task performance

Scores are AI2D exact option accuracy and TextVQA consensus accuracy.
Deltas are percentage points from the best completed model on that task.

| Model | AI2D score (n) | Δ pp | TextVQA score (n) | Δ pp | Status |
|---|---:|---:|---:|---:|---|
| Qwen/Qwen2.5-VL-7B-Instruct | — (0/0) | — | — (0/0) | — | skipped |
| Qwen/Qwen3-VL-2B-Thinking | 29.0% (100/100) | -54.0 | 0.0% (100/100) | -83.7 | complete |
| Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 | — (0/0) | — | — (0/0) | — | skipped |
| Qwen/Qwen3-VL-8B-Instruct | 83.0% (100/100) | +0.0 | 83.7% (100/100) | +0.0 | complete |
| Qwen/Qwen3-VL-8B-Thinking | — (0/0) | — | — (0/0) | — | skipped |
| unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit | — (0/0) | — | — (0/0) | — | skipped |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | — (0/0) | — | — (0/0) | — | skipped |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | — (0/100) | — | — (0/100) | — | inference_failed |
| zmli/G-Substrate-Qwen3-VL-2B | 3.0% (100/100) | -80.0 | 38.3% (100/100) | -45.4 | complete |

## VRAM and speed by task

VRAM is this process's peak CUDA allocated/reserved memory after model load. Wall time includes image preprocessing and generation; generation tok/s includes the full generate call. Counts show successful / requested examples; failures are excluded from the score denominator.

| Model | Task | Completed / requested | Peak allocated GiB | Peak reserved GiB | Mean wall s/item | Median wall s/item | Generation tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-VL-7B-Instruct | ai2d | 0/0 | — | — | — | — | — |
| Qwen/Qwen2.5-VL-7B-Instruct | textvqa | 0/0 | — | — | — | — | — |
| Qwen/Qwen3-VL-2B-Thinking | ai2d | 100/100 | 4.15 | 4.22 | 3.01 | 3.27 | 77.56 |
| Qwen/Qwen3-VL-2B-Thinking | textvqa | 100/100 | 4.15 | 4.26 | 1.41 | 0.94 | 73.57 |
| Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 | ai2d | 0/0 | — | — | — | — | — |
| Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 | textvqa | 0/0 | — | — | — | — | — |
| Qwen/Qwen3-VL-8B-Instruct | ai2d | 100/100 | 16.59 | 16.72 | 0.34 | 0.21 | 22.09 |
| Qwen/Qwen3-VL-8B-Instruct | textvqa | 100/100 | 16.58 | 16.77 | 0.22 | 0.21 | 19.49 |
| Qwen/Qwen3-VL-8B-Thinking | ai2d | 0/0 | — | — | — | — | — |
| Qwen/Qwen3-VL-8B-Thinking | textvqa | 0/0 | — | — | — | — | — |
| unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit | ai2d | 0/0 | — | — | — | — | — |
| unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit | textvqa | 0/0 | — | — | — | — | — |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | ai2d | 0/0 | — | — | — | — | — |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | textvqa | 0/0 | — | — | — | — | — |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | ai2d | 0/100 | 0.00 | 0.00 | — | — | — |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | textvqa | 0/100 | 0.00 | 0.00 | — | — | — |
| zmli/G-Substrate-Qwen3-VL-2B | ai2d | 100/100 | 7.11 | 7.67 | 0.18 | 0.17 | 34.68 |
| zmli/G-Substrate-Qwen3-VL-2B | textvqa | 100/100 | 7.11 | 7.68 | 0.59 | 0.14 | 36.90 |

## Method

Models are discovered from the local Hugging Face cache. GGUF repositories are excluded because this harness keeps one Transformers backend for comparable measurements.
AI2D uses the cached test split; TextVQA uses the cached validation split. Arrow files are read directly from disk without executing dataset loading scripts. Both use seeded random examples and a common 896-pixel maximum image side.
Qwen3-VL uses the Unsloth guide's Instruct or Thinking sampling values, with a 256-token cap. Other checkpoints use greedy decoding. Results describe this local sample, not leaderboard scores.

Raw per-item outputs and failures are in `results.json`.

## Context and MTP configuration

Context VRAM is estimated from measured model-load allocation, checkpoint KV dimensions, HQQ Q8 storage after a 128-token FP16 residual, and a 1.5 GiB safety reserve. Each VRAM cell includes its context size in parentheses. If a target is unreachable, the cell shows the estimated last context that fits. These are configuration estimates, not long-context generation runs. Image and video tokens count toward context. Native MTP weights were not found in ready checkpoints.

| Model | Checkpoint context | Min VRAM / context | Default VRAM / context | Max VRAM / context | MTP head | Max MTP drafts |
|---|---:|---:|---:|---:|---|---:|
| Qwen/Qwen2.5-VL-7B-Instruct | — | — | — | — | — | 0 |
| Qwen/Qwen3-VL-2B-Thinking | 262144.00 | 4.46 GiB (8K) | 11.84 GiB (128K) | 19.72 GiB (256K) | no | 0 |
| Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 | — | — | — | — | — | 0 |
| Qwen/Qwen3-VL-8B-Instruct | 262144.00 | 17.00 GiB (8K) | 26.49 GiB (128K) | 28.47 GiB (153K) | no | 0 |
| Qwen/Qwen3-VL-8B-Thinking | — | — | — | — | — | 0 |
| unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit | — | — | — | — | — | 0 |
| unsloth/Qwen3-VL-8B-Thinking-bnb-4bit | — | — | — | — | — | 0 |
| unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit | 262144.00 | 8.47 GiB (8K) | 17.96 GiB (128K) | 28.09 GiB (256K) | no | 0 |
| zmli/G-Substrate-Qwen3-VL-2B | 262144.00 | 7.46 GiB (8K) | 14.84 GiB (128K) | 22.72 GiB (256K) | no | 0 |
