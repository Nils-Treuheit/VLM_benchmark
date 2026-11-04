# Local foundation and GGUF VLM benchmark

Updated: 2026-11-03 18:39 CET
Run timestamps: 2026-11-03 17:50:15 +0100, 2026-11-03 18:08:32 +0100.
Completion: 2 of 2 latest model runs completed all 825 requests.
Execution: serial, one image per request, fixed examples, no batching.

## Summary

Two eligible routes were measured: Qwen3.8 through the configured Ollama MTP4 UD Q4_K_XL 128K profile, and LFM2.5-VL through the robotics adapter's llama.cpp MTMD route.
The full-suite macro is the unweighted mean of eight task scores. It is reported only for models that completed every requested example.

## Quality results

| Model and route | Status | Samples | Full-suite macro | VQA panel macro | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unsloth/LFM2.5-VL-1.6B-GGUF (llama-cpp-python 0.3.31 MTMDChatHandler CUDA) | complete | 825/825 | 13.3% | 8.2% | 53.0% | 7.9% | 0.0% | 0.0% | 5.2% | 0.0% | 36.0% | 4.0% |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k (qwen3.8:27b-mtp4-ud_q4_K_XL-128k; MTP4; UD Q4_K_XL) | complete | 825/825 | 63.5% | 65.8% | 88.0% | 81.3% | 90.4% | 81.6% | 69.2% | 8.0% | 64.0% | 25.8% |

## Selection guide

Best complete full-suite macro: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 63.5%.
Best complete result for AI2D: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 88.0%.
Best complete result for TextVQA / OCR: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 81.3%.
Best complete result for Truth: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 90.4%.
Best complete result for Counting: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 81.6%.
Best complete result for Description: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 69.2%.
Best complete result for Affordance: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 8.0%.
Best complete result for Negation: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 64.0%.
Best complete result for GRAB-Lite: `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` at 25.8%.
Picks use only full-suite-complete models; ties retain the first listed model.

## Real-image preflight

| Model | AI2D format | TextVQA nonempty answer |
|---|---:|---:|
| unsloth/LFM2.5-VL-1.6B-GGUF | pass | pass |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | pass | pass |

## Runtime and resources

| Model | Context | Peak whole-device VRAM GiB | Memory scope |
|---|---:|---:|---|
| unsloth/LFM2.5-VL-1.6B-GGUF | 8192 | 5.63 | nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | 131072 | 25.43 | nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline |

### Per-task latency and throughput

| Model | Task | Completed | Mean generation seconds | Output tokens/second |
|---|---|---:|---:|---:|
| unsloth/LFM2.5-VL-1.6B-GGUF | AI2D | 100/100 | 0.38 | 84.57 |
| unsloth/LFM2.5-VL-1.6B-GGUF | TextVQA / OCR | 100/100 | 0.09 | 225.38 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Truth | 25/25 | 0.07 | 183.11 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Counting | 25/25 | 0.10 | 268.87 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Description | 25/25 | 0.07 | 173.37 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Affordance | 25/25 | 0.11 | 255.53 |
| unsloth/LFM2.5-VL-1.6B-GGUF | Negation | 25/25 | 0.19 | 272.38 |
| unsloth/LFM2.5-VL-1.6B-GGUF | GRAB-Lite | 500/500 | 1.01 | 280.93 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | AI2D | 100/100 | 0.47 | 4.25 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | TextVQA / OCR | 100/100 | 0.67 | 6.55 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Truth | 25/25 | 0.44 | 4.60 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Counting | 25/25 | 0.43 | 4.80 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Description | 25/25 | 0.44 | 6.24 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Affordance | 25/25 | 0.67 | 4.05 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | Negation | 25/25 | 0.64 | 3.14 |
| qwen3.8:27b-mtp4-ud_q4_K_XL-128k | GRAB-Lite | 500/500 | 2.82 | 2.39 |

## Candidate inventory

| Checkpoint | License metadata | Status | Disposition |
|---|---|---|---|
| `qwen3.8:27b-mtp4-ud_q4_K_XL-128k` ([model card](https://huggingface.co/Qwen/Qwen3.8-27B)) | Apache-2.0 | scored | MTP4 UD Q4_K_XL at 128K context; complete shared-suite run is recorded in the results table. |
| `unsloth/LFM2.5-VL-1.6B-GGUF` ([model card](https://huggingface.co/unsloth/LFM2.5-VL-1.6B-GGUF/blob/6d323fde0c0a094eeeb2a2452c590fa17a3ff76b/README.md)) | LFM 1.0 | scored | BF16 GGUF with the matching F16 projector; complete shared-suite run uses the robotics MTMD adapter. |
| `Qwen/Qwen3.8-27B` ([model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md)) | Apache-2.0 | not_run | The cached full-precision snapshot is 51.77 GiB, above the 30 GiB device budget. |
| `turboderp/Qwen3.8-27B-exl3` ([model card](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/blob/a35e75a73baee51da709329d19294245cbeeb5d8/README.md)) | See linked model card | not_run | The benchmark used the configured MTP4 Ollama route; this separate EXL3 artifact has no runner in the shared suite. |
| `HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3` ([model card](https://huggingface.co/HermiHg/Qwen3.8-27B-Optimized-MTP-Heads-exl3/blob/411edf8fac168a7f6ba9b5fb3fca439ff9dc98cb/README.md)) | See linked model card | incomplete | The cached snapshot contains MTP head files only and no base checkpoint. |
| `unsloth/Qwen3.8-27B-NVFP4` ([model card](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/blob/f0b7c9e722f5565102fff8481c99e4d86ae099c7/README.md)) | Apache-2.0 | not_run | The scored Qwen run used the configured MTP4 UD Q4_K_XL Ollama profile; the NVFP4 snapshot is a separate serving route. |
| `SVPG-Labs/diffusiongemma-26b-djev-v10-step160-NVFP4` ([model card](https://huggingface.co/SVPG-Labs/diffusiongemma-26b-djev-v10-step160-NVFP4/blob/65526826901a65e92e82540bd152d85f591e42f3/README.md)) | Apache-2.0 plus Gemma terms | not_run | The cached revision documents D-Jev structured decisions (noul/choice/score) and says ordinary chat/free-text generation was not benchmarked; a direct shared-answer route was not established. |
| `unsloth/diffusiongemma-26B-A4B-it-GGUF` ([model card](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF/blob/aab0a2972da0e41310fbcca5ea63fc47eb932a71/README.md)) | See linked model card | incomplete | The cached snapshot contains a GGUF model file but no multimodal projector, so the local image-question route cannot be assembled. |
| `unsloth/MiMo-VL-7B-RL-GGUF` ([model card](https://huggingface.co/unsloth/MiMo-VL-7B-RL-GGUF/blob/9a1d60e80d877e6ee044d2c3b92e95c1c35b04ab/README.md)) | MIT | incomplete | The cached snapshot is empty; model and projector files are absent. |

## Methodology

All measured models use the shared fixed task definitions, examples, prompts, and scorers in the [Qwen-VL benchmark README](../qwenVL_Bench/README.md). The sample seed is 3407. Each request carries one image and runs serially.
Primary task sources: [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d), [TextVQA](https://huggingface.co/datasets/facebook/textvqa), [VQAv2](https://visualqa.org/download.html), [PercepTaxBench](https://huggingface.co/datasets/RyanWW/PercepTaxBench), [NegVQA](https://arxiv.org/abs/2505.22946), and [GRAB-Lite](https://huggingface.co/datasets/jonathan-roberts1/GRAB-lite).
AI2D uses exact four-choice letter accuracy. TextVQA and the truth, count, description, and OCR categories use the shared normalized human-answer consensus. Affordance uses normalized reference matching. Negation uses exact A/B accuracy. GRAB-Lite uses exact match, including its five task-group scores in the JSON artifacts.
The six-category VQA panel macro weights truth, counting, description, OCR, affordance, and negation equally. The full-suite macro weights all eight task scores equally and is local comparison evidence, not an official benchmark score.

## Backend settings and provenance

### `unsloth/LFM2.5-VL-1.6B-GGUF`

Backend: llama-cpp-python 0.3.31 MTMDChatHandler CUDA; context: 8192 tokens; status: complete.
Cached revision: `6d323fde0c0a094eeeb2a2452c590fa17a3ff76b`.
Quantization: BF16 GGUF; F16 multimodal projector.
Generation settings: `{"max_new_tokens": 512, "min_p": 0.05, "repeat_penalty": 1.05, "seed": null, "temperature": 0.1, "top_p": 0.95}`.
Runtime versions: `{"datasets": "5.0.1", "llama-cpp-python": "0.3.31", "numpy": "2.2.6", "torch": "2.13.0"}`.
Cached model and projector size: 2.98 GiB.
Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/20261103-175015/lfm25_vl_1_6b_bf16_gguf/results.json).

### `qwen3.8:27b-mtp4-ud_q4_K_XL-128k`

Backend: Ollama OpenAI-compatible multimodal chat API; context: 131072 tokens; status: complete.
Quantization: UD Q4_K_XL; MTP4.
Generation settings: `{"num_predict": 512, "seed": 3407, "temperature": 0, "think": false, "top_p": 1}`.
Runtime versions: `{"ollama": "local managed service; version unavailable via benchmark API"}`.
Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/20261103-180832/qwen3_8_mtp4_ud_q4_k_xl_128k/results.json).

## Limitations

These fixed samples are selection evidence, not leaderboard scores. The backends use model-specific image preprocessing. Whole-device memory samples include the desktop baseline; they are not process-local estimates. Advertised maximum context is distinct from the configured context.
License labels are taken from the linked model cards at their cached revisions where available. This report records model provenance and does not grant redistribution or use rights.
The Qwen MTP4 route uses a 128K configured context with thinking disabled. LFM2.5-VL uses its documented BF16 GGUF plus F16 projector at an 8K configured context. Unsupported and incomplete artifacts were not ranked.
The runner records whole-device peak VRAM for each model run. Its sampling does not reset at task boundaries, so per-task peak VRAM is not reported.

## Attempts

- `20261103-175015/lfm25_vl_1_6b_bf16_gguf`: complete.
- `20261103-180832/qwen3_8_mtp4_ud_q4_k_xl_128k`: complete.
- `20261103-174832/lfm25_vl_1_6b_bf16_gguf`: failed; AttributeError: MONOTONIC.
- `20261103-174926/lfm25_vl_1_6b_bf16_gguf`: failed; AttributeError: MONOTONIC.
- `20261103-180331/qwen3_8_mtp4_ud_q4_k_xl_128k`: preflight_failed; AI2D format and TextVQA nonempty answer checks did not both pass.

Raw item outputs are kept in run artifacts and are not copied into the static report page.
