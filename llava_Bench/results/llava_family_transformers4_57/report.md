# LLaVA family benchmark

Generated: 2026-10-30 12:23:45 +0100
Seed: 3407; GPU memory ceiling: 30.0 GiB.
Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.
The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.

## Scores

| Model | Status | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-hf/LLaVA-NeXT-Video-7B-hf | complete | 57.0% | 58.1% | 95.6% | 58.0% | 71.2% | 8.0% | 32.0% | 4.8% | 8/8 | 48.1% |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | complete | 82.0% | 81.6% | 95.6% | 68.0% | 80.8% | 4.0% | 44.0% | 8.0% | 8/8 | 58.0% |
| llava-hf/llava-v1.6-mistral-7b-hf | complete | 64.0% | 77.5% | 98.4% | 59.2% | 75.6% | 8.0% | 40.0% | 6.4% | 8/8 | 53.6% |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | skipped | — | — | — | — | — | — | — | — | 0/8 | — |
| xtuner/llava-phi-3-mini-hf | complete | 58.0% | 50.2% | 86.4% | 37.2% | 69.2% | 4.0% | 32.0% | 8.2% | 8/8 | 43.1% |

## Runtime details

Times cover generation after preprocessing. Peak allocated VRAM is process-local.
Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.

| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| llava-hf/LLaVA-NeXT-Video-7B-hf | ai2d | 100/100 | 0.20 | 10.05 | 14.97 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | textvqa | 100/100 | 0.24 | 17.75 | 14.95 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_truth | 25/25 | 0.21 | 9.73 | 14.94 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_count | 25/25 | 0.22 | 14.35 | 14.77 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_description | 25/25 | 0.22 | 13.51 | 14.94 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_affordance | 25/25 | 0.23 | 13.62 | 14.95 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | vqa_negation | 25/25 | 0.22 | 11.85 | 14.96 |
| llava-hf/LLaVA-NeXT-Video-7B-hf | graph_grab_lite | 500/500 | 0.28 | 17.48 | 14.97 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | ai2d | 100/100 | 0.29 | 6.94 | 16.58 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | textvqa | 100/100 | 0.49 | 8.16 | 16.45 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_truth | 25/25 | 0.25 | 7.96 | 15.73 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_count | 25/25 | 0.24 | 8.84 | 15.66 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_description | 25/25 | 0.26 | 10.94 | 15.73 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_affordance | 25/25 | 0.46 | 5.85 | 16.45 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | vqa_negation | 25/25 | 0.36 | 5.53 | 16.87 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | graph_grab_lite | 500/500 | 0.80 | 4.77 | 17.22 |
| llava-hf/llava-v1.6-mistral-7b-hf | ai2d | 100/100 | 0.22 | 13.76 | 15.60 |
| llava-hf/llava-v1.6-mistral-7b-hf | textvqa | 100/100 | 0.27 | 20.36 | 15.60 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_truth | 25/25 | 0.23 | 13.19 | 15.60 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_count | 25/25 | 0.24 | 17.47 | 15.52 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_description | 25/25 | 0.24 | 16.36 | 15.60 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_affordance | 25/25 | 0.25 | 16.53 | 15.60 |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_negation | 25/25 | 0.23 | 13.04 | 15.60 |
| llava-hf/llava-v1.6-mistral-7b-hf | graph_grab_lite | 500/500 | 0.32 | 23.69 | 15.60 |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | ai2d | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | textvqa | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_truth | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_count | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_description | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_affordance | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | vqa_negation | 0/0 | — | — | — |
| llava-hf/LLaVA-NeXT-Video-34B-DPO-hf | graph_grab_lite | 0/0 | — | — | — |
| xtuner/llava-phi-3-mini-hf | ai2d | 100/100 | 2.49 | 102.61 | 8.09 |
| xtuner/llava-phi-3-mini-hf | textvqa | 100/100 | 2.53 | 101.11 | 8.07 |
| xtuner/llava-phi-3-mini-hf | vqa_truth | 25/25 | 2.52 | 101.74 | 8.07 |
| xtuner/llava-phi-3-mini-hf | vqa_count | 25/25 | 2.56 | 100.16 | 8.07 |
| xtuner/llava-phi-3-mini-hf | vqa_description | 25/25 | 2.59 | 98.69 | 8.07 |
| xtuner/llava-phi-3-mini-hf | vqa_affordance | 25/25 | 2.60 | 98.39 | 8.07 |
| xtuner/llava-phi-3-mini-hf | vqa_negation | 25/25 | 2.53 | 101.37 | 8.09 |
| xtuner/llava-phi-3-mini-hf | graph_grab_lite | 500/500 | 2.52 | 101.51 | 8.12 |

## Method

Checkpoints are resolved from the local Hugging Face cache. No downloads are performed.
Each request contains one still image and one shared task prompt. The processor applies its model-specific image preprocessing and chat template.
Requests run serially with greedy decoding and at most one model loaded at a time.
The 34B checkpoint is skipped before loading when its cached weights alone exceed the 30 GiB budget.
Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.

Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.
