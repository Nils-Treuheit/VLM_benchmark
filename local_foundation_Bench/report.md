# Local foundation and GGUF VLM benchmark

Updated: 2026-11-05 04:17 CET
Run timestamps: 2026-11-03 18:08:32 +0100, 2026-11-04 12:49:50 +0100, 2026-11-04 12:56:16 +0100, 2026-11-04 13:52:12 +0100, 2026-11-04 14:55:50 +0100, 2026-11-04 19:17:55 +0100, 20:17:47 +0100, 20:24:26 +0100 (8 GiB Edge refinement), 21:41:32 and 22:36:16 +0100 (10 GiB Edge refinement), and 04:09:40 and 04:17:15 +0100 (12 GiB Edge refinement).
Completion: 5 of 5 latest model runs completed all 825 requests.
Execution: serial, one image per request, fixed examples, no batching.

## Summary

Measured candidates include the configured Qwen3.8 Ollama route and four Liquid VLM GGUF variants. The corrected LFM2.5-VL 1.6B run is the current result for that model ID; its earlier run remains in the attempt history.
The full-suite macro is the unweighted mean of eight task scores. It is reported only for models that completed every requested example.

## Edge refinement

| Whole-device cap | Model | Status | Full-suite macro | Measured context | Probe peak | Scoring peak | Whole-run peak |
|---:|---|---|---:|---:|---:|---:|---:|
| 8 GiB | LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | verified | 58.54% | 128,000 tokens | 5.258 GiB | 6.793 GiB | 6.793 GiB |
| 8 GiB | unsloth/LFM2.5-VL-3B-GGUF | verified | 58.44% | 32,768 tokens | 4.566 GiB | 4.573 GiB | 4.573 GiB |
| 8 GiB | LiquidAI/LFM3-VL-2B-GGUF | verified | 55.73% | 128,000 tokens | 5.020 GiB | 5.430 GiB | 5.430 GiB |
| 10 GiB | LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | verified | 58.54% | 128,000 tokens | 5.204 GiB | 5.607 GiB | 5.693 GiB |
| 10 GiB | unsloth/LFM2.5-VL-3B-GGUF | verified | 58.44% | 32,768 tokens | 4.547 GiB | 4.554 GiB | 4.554 GiB |
| 12 GiB | LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | verified | 58.54% | 128,000 tokens | 5.164 GiB | 5.554 GiB | 5.554 GiB |
| 12 GiB | unsloth/LFM2.5-VL-3B-GGUF | verified | 58.44% | 32,768 tokens | 4.547 GiB | 4.554 GiB | 4.554 GiB |

All seven listed runs completed all eight shared tasks under their measured whole-device limits. The 10 GiB [LFM3 NVFP4 report](results/runs/20261104-214132/edge_10gib/lfm3_2b_nvfp4/report.md) and [raw results](results/runs/20261104-214132/edge_10gib/lfm3_2b_nvfp4/results.json), and [LFM2.5 3B report](results/runs/20261104-223616/edge_10gib/lfm25_3b/report.md) and [raw results](results/runs/20261104-223616/edge_10gib/lfm25_3b/results.json) were independently reviewed. The [8 GiB LFM3 NVFP4 report](results/runs/20261104-191755/edge_8gib/lfm3_2b_nvfp4/report.md) and [raw results](results/runs/20261104-191755/edge_8gib/lfm3_2b_nvfp4/results.json), the [LFM2.5 3B report](results/runs/20261104-201747/edge_8gib/lfm25_3b/report.md) and [raw results](results/runs/20261104-201747/edge_8gib/lfm25_3b/results.json), and the [LFM3 Q4 report](results/runs/20261104-202426/edge_8gib/lfm3_2b/report.md) and [raw results](results/runs/20261104-202426/edge_8gib/lfm3_2b/results.json) preserve the 8 GiB evidence. The [12 GiB LFM3 NVFP4 report](results/runs/edge_refinement/12gib/evidence_retry_20261105/lfm3_2b_nvfp4/report.md) and [raw results](results/runs/edge_refinement/12gib/evidence_retry_20261105/lfm3_2b_nvfp4/results.json) were independently reviewed. Two earlier LFM3 12 GiB attempts remain preserved under `results/runs/results/runs/edge_refinement/12gib/` and `results/runs/edge_refinement/12gib/lfm3_2b_nvfp4/`; their evidence gaps are resolved by the verified retry. The [12 GiB LFM2.5 3B report](results/runs/edge_refinement/12gib/lfm25_3b/report.md) and [raw results](results/runs/edge_refinement/12gib/lfm25_3b/results.json) were also independently reviewed. Other selected candidates and 14/16 GiB cap stages remain pending.

## Quality results

| Model and route | Status | Samples | Full-suite macro | VQA panel macro | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k (qwen3.8:27b-mtp4-ud_q4_K_XL-128k; MTP4; UD Q4_K_XL) | complete | 825/825 | 63.5% | 65.8% | 88.0% | 81.3% | 90.4% | 81.6% | 69.2% | 8.0% | 64.0% | 25.8% |
| unsloth/LFM2.5-VL-3B-GGUF (llama.cpp b12097 CUDA 13.2 OpenAI-compatible server) | complete | 825/825 | 58.4% | 63.5% | 73.0% | 84.3% | 93.2% | 64.0% | 67.6% | 8.0% | 64.0% | 13.4% |
| LiquidAI/LFM3-VL-2B-GGUF (llama.cpp b12097 CUDA 13.2 OpenAI-compatible server) | complete | 825/825 | 55.7% | 55.4% | 86.0% | 74.2% | 61.2% | 72.4% | 60.4% | 8.0% | 56.0% | 27.6% |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 (llama.cpp b12097 CUDA 13.2 OpenAI-compatible server) | complete | 825/825 | 58.5% | 57.1% | 92.0% | 78.5% | 54.4% | 75.6% | 66.0% | 4.0% | 64.0% | 33.8% |
| unsloth/LFM2.5-VL-1.6B-GGUF (llama.cpp b12097 CUDA 13.2 OpenAI-compatible server) | complete | 825/825 | 37.5% | 39.8% | 56.0% | 62.0% | 68.8% | 13.2% | 58.8% | 4.0% | 32.0% | 4.8% |

## Selection guide

Best complete full-suite macro: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 63.5%.
Best complete result for AI2D: `LiquidAI/LFM3-VL-2B-GGUF:NVFP4` at 92.0%.
Best complete result for TextVQA / OCR: `unsloth/LFM2.5-VL-3B-GGUF` at 84.3%.
Best complete result for Truth: `unsloth/LFM2.5-VL-3B-GGUF` at 93.2%.
Best complete result for Counting: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 81.6%.
Best complete result for Description: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 69.2%.
Best complete result for Affordance: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 8.0%.
Best complete result for Negation: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 64.0%.
Best complete result for GRAB-Lite: `LiquidAI/LFM3-VL-2B-GGUF:NVFP4` at 33.8%.
Picks use only full-suite-complete models; ties retain the first listed model.

## Real-image preflight

| Model | AI2D format | TextVQA nonempty answer |
|---|---:|---:|
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | pass | pass |
| unsloth/LFM2.5-VL-3B-GGUF | pass | pass |
| LiquidAI/LFM3-VL-2B-GGUF | pass | pass |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | pass | pass |
| unsloth/LFM2.5-VL-1.6B-GGUF | pass | pass |

## Runtime and resources

| Model | Context | Peak whole-device VRAM GiB | Memory scope |
|---|---:|---:|---|
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | 131072 | 25.43 | nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline |
| unsloth/LFM2.5-VL-3B-GGUF | 8192 | 4.16 | nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline |
| LiquidAI/LFM3-VL-2B-GGUF | 8192 | 4.37 | nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | 8192 | 4.90 | nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline |
| unsloth/LFM2.5-VL-1.6B-GGUF | 8192 | 5.13 | nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline |

### Per-task latency and throughput

| Model | Task | Completed | Mean generation seconds | Output tokens/second |
|---|---|---:|---:|---:|
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | AI2D | 100/100 | 0.47 | 4.25 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | TextVQA / OCR | 100/100 | 0.67 | 6.55 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Truth | 25/25 | 0.44 | 4.60 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Counting | 25/25 | 0.43 | 4.80 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Description | 25/25 | 0.44 | 6.24 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Affordance | 25/25 | 0.67 | 4.05 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Negation | 25/25 | 0.64 | 3.14 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | GRAB-Lite | 500/500 | 2.82 | 2.39 |
| unsloth/LFM2.5-VL-3B-GGUF | AI2D | 100/100 | 0.08 | 25.39 |
| unsloth/LFM2.5-VL-3B-GGUF | TextVQA / OCR | 100/100 | 0.24 | 17.55 |
| unsloth/LFM2.5-VL-3B-GGUF | Truth | 25/25 | 0.05 | 44.13 |
| unsloth/LFM2.5-VL-3B-GGUF | Counting | 25/25 | 0.04 | 45.97 |
| unsloth/LFM2.5-VL-3B-GGUF | Description | 25/25 | 0.04 | 65.86 |
| unsloth/LFM2.5-VL-3B-GGUF | Affordance | 25/25 | 0.24 | 13.35 |
| unsloth/LFM2.5-VL-3B-GGUF | Negation | 25/25 | 0.10 | 20.81 |
| unsloth/LFM2.5-VL-3B-GGUF | GRAB-Lite | 500/500 | 0.39 | 7.62 |
| LiquidAI/LFM3-VL-2B-GGUF | AI2D | 100/100 | 1.33 | 521.25 |
| LiquidAI/LFM3-VL-2B-GGUF | TextVQA / OCR | 100/100 | 1.48 | 514.52 |
| LiquidAI/LFM3-VL-2B-GGUF | Truth | 25/25 | 0.76 | 509.80 |
| LiquidAI/LFM3-VL-2B-GGUF | Counting | 25/25 | 2.85 | 532.75 |
| LiquidAI/LFM3-VL-2B-GGUF | Description | 25/25 | 0.79 | 516.02 |
| LiquidAI/LFM3-VL-2B-GGUF | Affordance | 25/25 | 2.23 | 518.06 |
| LiquidAI/LFM3-VL-2B-GGUF | Negation | 25/25 | 2.97 | 527.41 |
| LiquidAI/LFM3-VL-2B-GGUF | GRAB-Lite | 500/500 | 5.19 | 490.33 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | AI2D | 100/100 | 1.54 | 470.61 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | TextVQA / OCR | 100/100 | 1.39 | 441.93 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | Truth | 25/25 | 1.15 | 327.46 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | Counting | 25/25 | 3.49 | 414.09 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | Description | 25/25 | 0.94 | 258.58 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | Affordance | 25/25 | 2.63 | 365.38 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | Negation | 25/25 | 4.80 | 370.69 |
| LiquidAI/LFM3-VL-2B-GGUF:NVFP4 | GRAB-Lite | 500/500 | 5.47 | 419.21 |
| unsloth/LFM2.5-VL-1.6B-GGUF | AI2D | 100/100 | 0.15 | 33.85 |
| unsloth/LFM2.5-VL-1.6B-GGUF | TextVQA / OCR | 100/100 | 0.19 | 30.47 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Truth | 25/25 | 0.17 | 15.84 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Counting | 25/25 | 0.16 | 16.84 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Description | 25/25 | 0.17 | 21.48 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Affordance | 25/25 | 0.19 | 25.26 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Negation | 25/25 | 0.18 | 58.80 |
| unsloth/LFM2.5-VL-1.6B-GGUF | GRAB-Lite | 500/500 | 0.36 | 42.23 |

## Candidate inventory

| Checkpoint | License metadata | Status | Disposition |
|---|---|---|---|
| `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` ([model card](https://huggingface.co/Qwen/Qwen3.8-27B)) | Apache-2.0 | scored | MTP4 UD Q4_K_XL at 128K context; complete shared-suite run is recorded in the results table. |
| `unsloth/LFM2.5-VL-1.6B-GGUF` ([model card](https://huggingface.co/unsloth/LFM2.5-VL-1.6B-GGUF/blob/6d323fde0c0a094eeeb2a2452c590fa17a3ff76b/README.md)) | LFM 1.0 | scored | BF16 GGUF with matching F16 projector; current full-suite result uses model-card sampling and the Unsloth tutorial image cap of 64 tokens through llama.cpp b12097. |
| `unsloth/LFM2.5-VL-3B-GGUF:UD-Q4_K_XL` ([model card](https://huggingface.co/unsloth/LFM2.5-VL-3B-GGUF)) | LFM Open License v1.0 | scored | UD-Q4_K_XL with matching F16 projector; full 825-example run uses temperature 0.2, top-k 50, repetition penalty 1.0, and llama.cpp b12097. |
| `LiquidAI/LFM3-VL-2B-GGUF:Q4_K_M` ([model card](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF)) | LFM Open License v1.0 | scored | Q4_K_M with matching F16 projector; full 825-example run uses model-card sampling, no reasoning preservation, and a 4,096-token output allowance. |
| `LiquidAI/LFM3-VL-2B-GGUF:NVFP4` ([model card](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF)) | LFM Open License v1.0 | scored | Full 825-example run on RTX 5090 with NVFP4 feed-forward weights, Q8_0 remaining weights, matching F16 projector, and llama.cpp b12097. |
| `Qwen/Qwen3.8-27B` ([model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md)) | Apache-2.0 | not_run | The cached full-precision snapshot is 51.77 GiB, above the 30 GiB device budget. |
| `turboderp/Qwen3.8-27B-exl3` ([model card](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/blob/a35e75a73baee51da709329d19294245cbeeb5d8/README.md)) | See linked model card | not_run | The benchmark used the configured MTP4 Ollama route; this separate EXL3 artifact has no runner in the shared suite. |
| `HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3` ([model card](https://huggingface.co/HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3/blob/411edf8fac168a7f6ba9b5fb3fca439ff9dc98cb/README.md)) | See linked model card | incomplete | The cached snapshot contains MTP head files only and no base checkpoint. |
| `unsloth/Qwen3.8-27B-NVFP4` ([model card](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/blob/f0b7c9e722f5565102fff8481c99e4d86ae099c7/README.md)) | Apache-2.0 | not_run | The scored Qwen run used the configured MTP4 UD Q4_K_XL Ollama profile; the NVFP4 snapshot is a separate serving route. |
| `SVPG-Labs/diffusiongemma-26b-djev-v10-step160-NVFP4` ([model card](https://huggingface.co/SVPG-Labs/diffusiongemma-26b-djev-v10-step160-NVFP4/blob/65526826901a65e92e82540bd152d85f591e42f3/README.md)) | Apache-2.0 plus Gemma terms | not_run | The cached revision documents D-Jev structured decisions (noul/choice/score) and says ordinary chat/free-text generation was not benchmarked; a direct shared-answer route was not established. |
| `unsloth/diffusiongemma-26B-A4B-it-GGUF` ([model card](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF/blob/aab0a2972da0e41310fbcca5ea63fc47eb932a71/README.md)) | See linked model card | incomplete | The cached snapshot contains a GGUF model file but no multimodal projector, so the local image-question route cannot be assembled. |
| `unsloth/MiMo-VL-7B-RL-GGUF` ([model card](https://huggingface.co/unsloth/MiMo-VL-7B-RL-GGUF/blob/9a1d60e80d877e6ee044d2c3b92e95c1c35b04ab/README.md)) | MIT | incomplete | The cached snapshot is empty; model and projector files are absent. |

## Methodology

All measured models use the shared fixed task definitions, examples, prompts, and scorers in the [Qwen-VL benchmark README](../qwenVL_Bench/README.md). The sample seed is 3407. Each request carries one image and runs serially. Liquid GGUF variants use a host-native llama.cpp b12097 CUDA 13.2 build.
Primary task sources: [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d), [TextVQA](https://huggingface.co/datasets/facebook/textvqa), [VQAv2](https://visualqa.org/download.html), [PercepTaxBench](https://huggingface.co/datasets/RyanWW/PercepTaxBench), [NegVQA](https://arxiv.org/abs/2505.22946), and [GRAB-Lite](https://huggingface.co/datasets/jonathan-roberts1/GRAB-lite).
AI2D uses exact four-choice letter accuracy. TextVQA and the truth, count, description, and OCR categories use the shared normalized human-answer consensus. Affordance uses normalized reference matching. Negation uses exact A/B accuracy. GRAB-Lite uses exact match, including its five task-group scores in the JSON artifacts.
The six-category VQA panel macro weights truth, counting, description, OCR, affordance, and negation equally. The full-suite macro weights all eight task scores equally and is local comparison evidence, not an official benchmark score.

## Backend settings and provenance

### `qwen3.8:27b-mtp4-ud_q4_K_XL-128k`

Backend: Ollama OpenAI-compatible multimodal chat API; context: 131072 tokens; status: complete.
Quantization: UD Q4_K_XL; MTP4.
Generation settings: `{"num_predict": 512, "seed": 3407, "temperature": 0, "think": false, "top_p": 1}`.
Runtime versions: `{"ollama": "local managed service; version unavailable via benchmark API"}`.
Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/20261103-180832/qwen3_8_mtp4_ud_q4_k_xl_128k/results.json).

### `unsloth/LFM2.5-VL-3B-GGUF`

Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; context: 8192 tokens; status: complete.
Base model card: [https://huggingface.co/LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B).
Cached revision: `22f063714556bd4daa81c40dc1ff15ca37ae92ea`.
Advertised context: 32768 tokens; configured run context: 8192 tokens.
Quantization: UD-Q4_K_XL + F16 mmproj.
Image processing: `{"image_format": "JPEG quality 92", "image_max_tokens": "llama.cpp default; not explicitly capped", "images_per_request": 1}`.
Generation settings: `{"max_new_tokens": 512, "min_p": 0.0, "repeat_penalty": 1.0, "seed": 3407, "temperature": 0.2, "top_k": 50, "top_p": 1.0}`.
Runtime versions: `{"CUDA": "13.2", "llama.cpp": "b12097"}`.
Cached model and projector size: 2.39 GiB.
Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/20261104-124950/lfm25_3b/results.json).

### `LiquidAI/LFM3-VL-2B-GGUF`

Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; context: 8192 tokens; status: complete.
Base model card: [https://huggingface.co/LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B).
Cached revision: `1add221a5ea6c047fcf1eaaf2d247939d7cf8849`.
Advertised context: 131072 tokens; configured run context: 8192 tokens.
Quantization: Q4_K_M + F16 mmproj.
Image processing: `{"image_format": "JPEG quality 92", "image_max_tokens": "llama.cpp default; not explicitly capped", "images_per_request": 1}`.
Generation settings: `{"max_new_tokens": 4096, "min_p": 0.0, "repeat_penalty": 1.1, "seed": 3407, "temperature": 0.1, "top_k": 50, "top_p": 1.0}`.
Runtime versions: `{"CUDA": "13.2", "llama.cpp": "b12097"}`.
Cached model and projector size: 1.38 GiB.
Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/20261104-125616/lfm3_2b/results.json).

### `LiquidAI/LFM3-VL-2B-GGUF:NVFP4`

Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; context: 8192 tokens; status: complete.
Base model card: [https://huggingface.co/LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B).
Cached revision: `1add221a5ea6c047fcf1eaaf2d247939d7cf8849`.
Advertised context: 131072 tokens; configured run context: 8192 tokens.
Quantization: NVFP4 feed-forward weights, Q8_0 other weights + F16 mmproj.
Image processing: `{"image_format": "JPEG quality 92", "image_max_tokens": "llama.cpp default; not explicitly capped", "images_per_request": 1}`.
Generation settings: `{"max_new_tokens": 4096, "min_p": 0.0, "repeat_penalty": 1.1, "seed": 3407, "temperature": 0.1, "top_k": 50, "top_p": 1.0}`.
Runtime versions: `{"CUDA": "13.2", "llama.cpp": "b12097"}`.
Cached model and projector size: 1.55 GiB.
Recovery: 1 failed request(s) were retried and scored; the original error remains recorded in `repair_attempts` in the results JSON.
Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/20261104-135212/lfm3_2b_nvfp4/results.json).

### `unsloth/LFM2.5-VL-1.6B-GGUF`

Backend: llama.cpp b12097 CUDA 13.2 OpenAI-compatible server; context: 8192 tokens; status: complete.
Base model card: [https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B).
Cached revision: `6d323fde0c0a094eeeb2a2452c590fa17a3ff76b`.
Advertised context: 32768 tokens; configured run context: 8192 tokens.
Quantization: BF16 GGUF + F16 mmproj.
Image processing: `{"image_format": "JPEG quality 92", "image_max_tokens": 64, "images_per_request": 1}`.
Generation settings: `{"max_new_tokens": 512, "min_p": 0.15, "repeat_penalty": 1.05, "seed": 3407, "temperature": 0.1, "top_k": 50, "top_p": 1.0}`.
Runtime versions: `{"CUDA": "13.2", "llama.cpp": "b12097"}`.
Cached model and projector size: 2.98 GiB.
Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/20261104-145550/lfm25_1_6b/results.json).

## Limitations

These fixed samples are selection evidence, not leaderboard scores. The Liquid GGUF runner sends JPEG quality 92. LFM2.5-VL 1.6B uses a 64 image-token cap; the other Liquid variants leave image-token limits at the llama.cpp default. The original model-card processor settings are not fully enforced by the GGUF routes. Whole-device memory samples include the desktop baseline; they are not process-local estimates. Advertised maximum context is distinct from the configured context.
License labels are taken from the linked model cards at their cached revisions where available. This report records model provenance and does not grant redistribution or use rights.
The Qwen MTP4 route uses a 128K configured context with thinking disabled. Liquid GGUF variants use an 8K configured context. The current LFM2.5-VL 1.6B result uses temperature 0.1, min-p 0.15, repetition penalty 1.05, top-k 50, a fixed seed, and the Unsloth tutorial's 64 image-token cap. LFM2.5 3B and both LFM3 quantizations use their model-card sampling settings. LFM3 uses a 4,096-token output allowance because 512 tokens ended before its first final answer. The NVFP4 checkpoint uses an upstream-supported llama.cpp build on the RTX 5090. Unsupported and incomplete artifacts were not ranked.
The runner records whole-device peak VRAM for each model run. Its sampling does not reset at task boundaries, so per-task peak VRAM is not reported.

## Attempts

- `20261103-180832/qwen3_8_mtp4_ud_q4_k_xl_128k`: complete.
- `20261104-124950/lfm25_3b`: complete.
- `20261104-125616/lfm3_2b`: complete.
- `20261104-135212/lfm3_2b_nvfp4`: complete.
- `20261104-145550/lfm25_1_6b`: complete.
- `20261103-174832/lfm25_vl_1_6b_bf16_gguf`: failed; AttributeError: MONOTONIC.
- `20261103-174926/lfm25_vl_1_6b_bf16_gguf`: failed; AttributeError: MONOTONIC.
- `20261103-175015/lfm25_vl_1_6b_bf16_gguf`: complete; incomplete run retained.
- `20261103-180331/qwen3_8_mtp4_ud_q4_k_xl_128k`: preflight_failed; AI2D format and TextVQA nonempty answer checks did not both pass.
- `20261104-124442/lfm25_1_6b`: complete; incomplete run retained.
- `20261104-125503/lfm3_2b`: preflight_failed; AI2D format and TextVQA nonempty-answer preflight did not both pass.

Raw item outputs are kept in run artifacts and are not copied into the static report page.
