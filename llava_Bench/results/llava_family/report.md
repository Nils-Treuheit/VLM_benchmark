# LLaVA family benchmark

Generated: 2026-10-30 17:22:20 +0100
Seed: 3407; GPU memory ceiling: 30.0 GiB.
Transformers: 5.22.1.
Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.
The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.

## Scores

| Model | Status | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-hf/LLaVA-NeXT-Video-7B-hf | complete | 58.0% | 58.1% | 95.6% | 58.0% | 71.2% | 8.0% | 32.0% | 4.8% | 8/8 | 48.2% |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | complete | 81.0% | 81.6% | 95.6% | 68.0% | 80.8% | 4.0% | 44.0% | 7.6% | 8/8 | 57.8% |
| llava-hf/llava-v1.6-mistral-7b-hf | complete | 64.0% | 77.8% | 98.4% | 59.2% | 75.6% | 8.0% | 40.0% | 6.8% | 8/8 | 53.7% |
| xtuner/llava-phi-3-mini-hf | complete | 59.0% | 49.2% | 86.4% | 41.2% | 69.2% | 4.0% | 32.0% | 8.2% | 8/8 | 43.6% |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | skipped | — | — | — | — | — | — | — | — | 0/8 | — |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | complete | 85.0% | 73.7% | 97.2% | 40.8% | 71.2% | 4.0% | 84.0% | 15.0% | 8/8 | 58.9% |

## Runtime details

Times cover generation after preprocessing. Peak allocated VRAM is process-local.
Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.

| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| llava-hf/LLaVA-NeXT-Video-7B-hf | ai2d | 100/100 | 0.20 | 9.82 | 15.13 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | textvqa | 100/100 | 0.25 | 17.36 | 15.11 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_truth | 25/25 | 0.21 | 9.74 | 15.10 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_count | 25/25 | 0.22 | 14.27 | 14.93 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_description | 25/25 | 0.22 | 13.40 | 15.10 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_affordance | 25/25 | 0.23 | 13.51 | 15.11 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_negation | 25/25 | 0.22 | 11.74 | 15.12 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | graph_grab_lite | 500/500 | 0.29 | 17.12 | 15.13 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | ai2d | 100/100 | 0.30 | 6.71 | 17.26 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | textvqa | 100/100 | 0.49 | 8.13 | 16.93 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_truth | 25/25 | 0.25 | 8.11 | 15.97 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_count | 25/25 | 0.24 | 8.98 | 15.90 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_description | 25/25 | 0.25 | 11.19 | 15.97 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_affordance | 25/25 | 0.46 | 5.88 | 16.93 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_negation | 25/25 | 0.36 | 5.62 | 17.87 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | graph_grab_lite | 500/500 | 0.80 | 4.78 | 18.17 |
| llava-hf/llava-v1.6-mistral-7b-hf | ai2d | 100/100 | 0.22 | 13.74 | 15.02 |
| llava-hf/llava-v1.6-mistral-7b-hf | textvqa | 100/100 | 0.26 | 20.52 | 15.02 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_truth | 25/25 | 0.23 | 13.27 | 15.01 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_count | 25/25 | 0.24 | 17.63 | 14.94 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_description | 25/25 | 0.23 | 16.53 | 15.01 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_affordance | 25/25 | 0.24 | 16.70 | 15.02 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_negation | 25/25 | 0.23 | 13.12 | 15.02 |
| llava-hf/llava-v1.6-mistral-7b-hf | graph_grab_lite | 500/500 | 0.32 | 23.85 | 15.02 |
| xtuner/llava-phi-3-mini-hf | ai2d | 100/100 | 2.39 | 106.94 | 8.12 |
| xtuner/llava-phi-3-mini-hf | textvqa | 100/100 | 2.43 | 105.47 | 8.11 |
| xtuner/llava-phi-3-mini-hf | vqa_truth | 25/25 | 2.43 | 105.44 | 8.10 |
| xtuner/llava-phi-3-mini-hf | vqa_count | 25/25 | 2.43 | 105.55 | 8.10 |
| xtuner/llava-phi-3-mini-hf | vqa_description | 25/25 | 2.41 | 106.16 | 8.10 |
| xtuner/llava-phi-3-mini-hf | vqa_affordance | 25/25 | 2.42 | 105.91 | 8.11 |
| xtuner/llava-phi-3-mini-hf | vqa_negation | 25/25 | 2.43 | 105.15 | 8.12 |
| xtuner/llava-phi-3-mini-hf | graph_grab_lite | 500/500 | 2.42 | 105.62 | 8.16 |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | ai2d | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | textvqa | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_truth | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_count | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_description | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_affordance | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_negation | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | graph_grab_lite | 0/0 | — | — | — |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | ai2d | 100/100 | 0.09 | 26.04 | 16.88 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | textvqa | 100/100 | 0.20 | 34.20 | 16.38 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | vqa_truth | 25/25 | 0.07 | 30.07 | 16.09 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | vqa_count | 25/25 | 0.07 | 34.68 | 16.07 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | vqa_description | 25/25 | 0.08 | 36.70 | 16.09 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | vqa_affordance | 25/25 | 0.14 | 18.22 | 16.38 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | vqa_negation | 25/25 | 0.13 | 15.40 | 17.39 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | graph_grab_lite | 500/500 | 0.78 | 5.33 | 17.59 |

## Method

Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.
Each request contains one still image and one shared task prompt. The processor applies its model-specific image preprocessing and chat template.
Requests run serially with greedy decoding and at most one model loaded at a time.
The 34B checkpoint is skipped before loading when its cached weights alone exceed the 30 GiB budget.
Results record the Transformers version; resume requires the same version for comparable outputs.
Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.

Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.
