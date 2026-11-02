# Phi vision family benchmark

Generated: 2026-11-02 14:36:38 +0100
Seed: 3407; GPU memory ceiling: 30.0 GiB.
Transformers: 5.22.1.
Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.
The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.

## Scores

| Model | Status | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| microsoft/Phi-3.5-vision-instruct | failed | — | — | — | — | — | — | — | — | 0/8 | — |
| microsoft/Phi-3-vision-128k-instruct | skipped | — | — | — | — | — | — | — | — | 0/8 | — |
| microsoft/Phi-4-multimodal-instruct | failed | — | — | — | — | — | — | — | — | 0/8 | — |
| sovitrath/Phi-3.5-vision-instruct | skipped | — | — | — | — | — | — | — | — | 0/8 | — |

## Runtime details

Times cover generation after preprocessing. Peak allocated VRAM is process-local.
Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.

| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| microsoft/Phi-3.5-vision-instruct | ai2d | 0/0 | — | — | — |
| microsoft/Phi-3.5-vision-instruct | textvqa | 0/0 | — | — | — |
| microsoft/Phi-3.5-vision-instruct | vqa_truth | 0/0 | — | — | — |
| microsoft/Phi-3.5-vision-instruct | vqa_count | 0/0 | — | — | — |
| microsoft/Phi-3.5-vision-instruct | vqa_description | 0/0 | — | — | — |
| microsoft/Phi-3.5-vision-instruct | vqa_affordance | 0/0 | — | — | — |
| microsoft/Phi-3.5-vision-instruct | vqa_negation | 0/0 | — | — | — |
| microsoft/Phi-3.5-vision-instruct | graph_grab_lite | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | ai2d | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | textvqa | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | vqa_truth | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | vqa_count | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | vqa_description | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | vqa_affordance | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | vqa_negation | 0/0 | — | — | — |
| microsoft/Phi-3-vision-128k-instruct | graph_grab_lite | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | ai2d | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | textvqa | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_truth | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_count | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_description | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_affordance | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_negation | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | graph_grab_lite | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | ai2d | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | textvqa | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | vqa_truth | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | vqa_count | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | vqa_description | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | vqa_affordance | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | vqa_negation | 0/0 | — | — | — |
| sovitrath/Phi-3.5-vision-instruct | graph_grab_lite | 0/0 | — | — | — |

## Method

Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.
Each request contains one still image and one shared task prompt. The processor applies its model-specific image preprocessing and chat template.
Requests run serially with greedy decoding and at most one model loaded at a time.
Incomplete checkpoints are retained with their cache errors; over-budget checkpoints are skipped before loading.
Results record the Transformers version; resume requires the same version for comparable outputs.
Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.

Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.


## Checkpoint identity and compatibility

| Model | Snapshot | Cached weights (GiB) | Configuration identity | Cache and comparison status |
|---|---|---:|---|---|
| microsoft/Phi-3.5-vision-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Phi-3.5-vision-instruct/snapshots/12b77fb40b63a2c73c68243d3f767aab688a1b2a` | 7.72 | `phi3_v`; context `131072` | RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 1 but got size 834 for tensor number 1 in the list. |
| microsoft/Phi-3-vision-128k-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Phi-3-vision-128k-instruct/snapshots/ed2772fabe9dc9acd0caad54b62761d92520cc44` | 0.00 | `phi3_v`; context `131072` | model weight files are missing |
| microsoft/Phi-4-multimodal-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81` | 10.38 | `phi4mm`; context `131072` | ModuleNotFoundError: No module named 'backoff' |
| sovitrath/Phi-3.5-vision-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--sovitrath--Phi-3.5-vision-instruct/snapshots/c9d9284022ce4c7a00ace7fc7e9ea64445720d5d` | 0.00 | `phi3_v`; context `131072` | incomplete checkpoint; missing 2 weight shard(s) |

Configuration similarity alone does not establish weight identity. The converted Phi-3.5 checkpoint is treated as separate only if both weight sets are complete and can be compared.
