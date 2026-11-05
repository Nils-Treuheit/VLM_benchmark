# LLaVA family benchmark

Generated: 2026-11-05 06:23:02 +0100
Seed: 3407; GPU memory ceiling: 14.0 GiB.
Transformers: 5.22.1.
Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.
The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.

## Scores

| Model | Status | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xtuner/llava-phi-3-mini-hf | complete | 59.0% | 49.2% | 86.4% | 41.2% | 69.2% | 4.0% | 32.0% | 8.2% | 8/8 | 43.6% |

## Runtime details

Times cover generation after preprocessing. Peak allocated VRAM is process-local.
Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.

| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| xtuner/llava-phi-3-mini-hf | ai2d | 100/100 | 2.41 | 106.03 | 8.12 |
| xtuner/llava-phi-3-mini-hf | textvqa | 100/100 | 2.42 | 105.66 | 8.11 |
| xtuner/llava-phi-3-mini-hf | vqa_truth | 25/25 | 2.42 | 105.84 | 8.10 |
| xtuner/llava-phi-3-mini-hf | vqa_count | 25/25 | 2.42 | 105.95 | 8.10 |
| xtuner/llava-phi-3-mini-hf | vqa_description | 25/25 | 2.42 | 105.57 | 8.10 |
| xtuner/llava-phi-3-mini-hf | vqa_affordance | 25/25 | 2.43 | 105.31 | 8.11 |
| xtuner/llava-phi-3-mini-hf | vqa_negation | 25/25 | 2.42 | 105.84 | 8.12 |
| xtuner/llava-phi-3-mini-hf | graph_grab_lite | 500/500 | 2.42 | 105.77 | 8.16 |

## Edge cap verification

Allocator allowance: selected cap minus measured idle baseline minus 0.75 GiB reserve.
### xtuner/llava-phi-3-mini-hf

Cap verified: `True`; evidence: `{'whole_device_cap_gib': 14.0, 'context_probe_passed': True, 'complete_shared_task_suite': True, 'scored_items_within_probed_context': True, 'scoring_peak_within_cap': True, 'load_peak_whole_device_gib': 9.012, 'load_peak_within_cap': True, 'whole_device_sampling_verified': True, 'whole_device_sampler_error': None, 'task_sampling_verified': True, 'whole_device_peak_gib': 12.264, 'whole_device_peak_within_cap': True, 'whole_run_peak_within_cap': True}`.
Whole-device peak: 12.264 GiB; load peak: 9.012 GiB; scoring peak: 12.264 GiB.
Idle baseline: 0.811 GiB; allocator reserve: 0.75 GiB; allocator/load limit: 12.439453125 GiB.
Context probe: 4096 tokens; image index 2426; input measure `total_input_token_proxy`; probe peak 12.263671875 GiB.


## Method

Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.
Each request contains one still image and one shared task prompt. The processor applies its model-specific image preprocessing and chat template.
Requests run serially with greedy decoding and at most one model loaded at a time.
The 34B checkpoint is skipped before loading when its cached weights alone exceed the 14.0 GiB budget.
Results record the Transformers version; resume requires the same version for comparable outputs.
Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.

Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.
