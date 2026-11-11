# LLaVA family benchmark

Generated: 2026-11-06 20:10:21 +0100
Seed: 3407; GPU memory ceiling: 16.0 GiB.
Transformers: 5.22.1.
Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.
The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.

## Scores

| Model | Status | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-hf/llava-v1.6-mistral-7b-hf | context_probe_failed | — | — | — | — | — | — | — | — | 0/8 | — |

## Runtime details

Times cover generation after preprocessing. Peak allocated VRAM is process-local.
Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.

| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| llava-hf/llava-v1.6-mistral-7b-hf | ai2d | 0/0 | — | — | — |
| llava-hf/llava-v1.6-mistral-7b-hf | textvqa | 0/0 | — | — | — |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_truth | 0/0 | — | — | — |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_count | 0/0 | — | — | — |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_description | 0/0 | — | — | — |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_affordance | 0/0 | — | — | — |
| llava-hf/llava-v1.6-mistral-7b-hf | vqa_negation | 0/0 | — | — | — |
| llava-hf/llava-v1.6-mistral-7b-hf | graph_grab_lite | 0/0 | — | — | — |

## Edge cap verification

Allocator allowance: selected cap minus measured idle baseline minus 0.64 GiB reserve.
### llava-hf/llava-v1.6-mistral-7b-hf

Cap verified: `False`; evidence: `{'whole_device_cap_gib': 16.0, 'context_probe_passed': False, 'complete_shared_task_suite': False, 'scored_items_within_probed_context': False, 'scoring_peak_within_cap': False, 'load_peak_whole_device_gib': 15.279, 'load_peak_within_cap': True, 'whole_device_sampling_verified': True, 'whole_device_sampler_error': None, 'task_sampling_verified': False, 'whole_device_peak_gib': 15.975, 'whole_device_peak_within_cap': True, 'whole_run_peak_within_cap': True}`.
Whole-device peak: 15.975 GiB; load peak: 15.279 GiB; scoring peak: None GiB.
Idle baseline: 0.685 GiB; allocator reserve: 0.64 GiB; allocator/load limit: 14.6754296875 GiB.
Context probe: None tokens; image index 1717; input measure `total_input_token_proxy`; probe peak 15.974609375 GiB.


## Method

Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.
Each request contains one still image and one shared task prompt. The processor applies its model-specific image preprocessing and chat template.
Requests run serially with greedy decoding and at most one model loaded at a time.
The 34B checkpoint is skipped before loading when its cached weights alone exceed the 16.0 GiB budget.
Results record the Transformers version; resume requires the same version for comparable outputs.
Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.

Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.
