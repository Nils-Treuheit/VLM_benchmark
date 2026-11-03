# Phi vision family benchmark

Generated: 2026-11-02 17:56:50 +0100
Seed: 3407; GPU memory ceiling: 30.0 GiB.
Runtime: Python 3.13.13; PyTorch 2.14.1+cu130; CUDA 13.0; Transformers 4.57.1.
Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.
The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.

## Tasks and scoring

| Task | Examples | Dataset and split | Score |
|---|---:|---|---|
| AI2D | 100 | AI2D test | Exact A/B/C/D accuracy |
| TextVQA / OCR | 100 | TextVQA validation | Normalized human-answer consensus |
| Truth, counting, description | 25 each | VQAv2 validation | Normalized consensus per category |
| Affordance | 25 | PercepTaxBench real/test | Normalized answer match |
| Negation | 25 | NegVQA dev | Exact A/B choice accuracy |
| GRAB-Lite | 500 | GRAB-Lite test | Exact match overall and across five groups |

The same fixed examples, image resize policy, one-image prompt wording, and output constraints were used for every model. Requests ran serially without batching. The macro is a local comparison score, not a benchmark leaderboard score.

## Scores

| Model | Status | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| microsoft/Phi-3.5-vision-instruct | complete | 74.0% | 19.8% | 24.0% | 4.8% | 9.2% | 4.0% | 44.0% | 6.2% | 8/8 | 23.2% |
| microsoft/Phi-4-multimodal-instruct | complete | 81.0% | 77.0% | 88.8% | 68.4% | 70.4% | 8.0% | 60.0% | 11.6% | 8/8 | 58.1% |
| microsoft/Phi-3-vision-128k-instruct | complete | 28.0% | 3.0% | 0.0% | 0.0% | 0.0% | 0.0% | 8.0% | 7.4% | 8/8 | 5.8% |

## Runtime details

Times cover generation after preprocessing. Peak allocated VRAM is process-local.
Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.

| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| microsoft/Phi-3.5-vision-instruct | ai2d | 100/100 | 4.91 | 52.11 | 8.33 |
| microsoft/Phi-3.5-vision-instruct | textvqa | 100/100 | 4.88 | 52.48 | 8.29 |
| microsoft/Phi-3.5-vision-instruct | vqa_truth | 25/25 | 4.84 | 52.90 | 8.28 |
| microsoft/Phi-3.5-vision-instruct | vqa_count | 25/25 | 4.85 | 52.79 | 8.28 |
| microsoft/Phi-3.5-vision-instruct | vqa_description | 25/25 | 4.84 | 52.89 | 8.28 |
| microsoft/Phi-3.5-vision-instruct | vqa_affordance | 25/25 | 4.84 | 52.86 | 8.29 |
| microsoft/Phi-3.5-vision-instruct | vqa_negation | 25/25 | 4.86 | 52.66 | 8.32 |
| microsoft/Phi-3.5-vision-instruct | graph_grab_lite | 500/500 | 4.87 | 52.59 | 8.41 |
| microsoft/Phi-4-multimodal-instruct | ai2d | 100/100 | 0.21 | 9.74 | 15.85 |
| microsoft/Phi-4-multimodal-instruct | textvqa | 100/100 | 0.35 | 10.56 | 14.27 |
| microsoft/Phi-4-multimodal-instruct | vqa_truth | 25/25 | 0.19 | 10.78 | 13.13 |
| microsoft/Phi-4-multimodal-instruct | vqa_count | 25/25 | 0.18 | 10.95 | 13.13 |
| microsoft/Phi-4-multimodal-instruct | vqa_description | 25/25 | 0.19 | 13.51 | 13.13 |
| microsoft/Phi-4-multimodal-instruct | vqa_affordance | 25/25 | 0.32 | 8.12 | 14.25 |
| microsoft/Phi-4-multimodal-instruct | vqa_negation | 25/25 | 0.27 | 7.38 | 16.80 |
| microsoft/Phi-4-multimodal-instruct | graph_grab_lite | 500/500 | 1.17 | 2.79 | 20.48 |
| microsoft/Phi-3-vision-128k-instruct | ai2d | 100/100 | 5.39 | 47.51 | 10.88 |
| microsoft/Phi-3-vision-128k-instruct | textvqa | 100/100 | 5.37 | 47.64 | 10.79 |
| microsoft/Phi-3-vision-128k-instruct | vqa_truth | 25/25 | 5.37 | 47.71 | 10.75 |
| microsoft/Phi-3-vision-128k-instruct | vqa_count | 25/25 | 5.38 | 47.61 | 10.76 |
| microsoft/Phi-3-vision-128k-instruct | vqa_description | 25/25 | 5.36 | 47.73 | 10.76 |
| microsoft/Phi-3-vision-128k-instruct | vqa_affordance | 25/25 | 5.36 | 47.76 | 10.79 |
| microsoft/Phi-3-vision-128k-instruct | vqa_negation | 25/25 | 5.38 | 47.62 | 10.87 |
| microsoft/Phi-3-vision-128k-instruct | graph_grab_lite | 500/500 | 5.59 | 45.79 | 11.04 |

## GRAB-Lite group scores

| Model | function | properties | real | series | transform |
|---|---:|---:|---:|---:|---:|
| microsoft/Phi-3.5-vision-instruct | 2.0% | 9.0% | 9.0% | 7.0% | 4.0% |
| microsoft/Phi-4-multimodal-instruct | 7.0% | 9.0% | 17.0% | 19.0% | 6.0% |
| microsoft/Phi-3-vision-128k-instruct | 3.0% | 11.0% | 10.0% | 11.0% | 2.0% |

## Method

Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.
Each request contains one still image and one shared task prompt. Model-specific local processors apply Phi image preprocessing and the documented Phi image-token format.
Requests run serially with greedy decoding and at most one model loaded at a time.
Incomplete checkpoints are retained with their cache errors; over-budget checkpoints are skipped before loading.
All runnable models used Transformers 4.57.1. Resume requires the same Transformers version for comparable outputs.
Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.

Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.


## Checkpoint identity and compatibility

| Model | Snapshot | Cached weights (GiB) | Configuration identity | Backend | Remote code | Attention | Compatibility patches | Cache and comparison status |
|---|---|---:|---|---|---|---|---|---|
| microsoft/Phi-3.5-vision-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Phi-3.5-vision-instruct/snapshots/12b77fb40b63a2c73c68243d3f767aab688a1b2a` | 7.72 | `phi3_v`; context `131072` | Transformers (cached checkpoint) | True | eager | DynamicCache legacy aliases required by cached Phi-3 remote code, eager attention required by the cached Phi-3.5 model | complete |
| microsoft/Phi-4-multimodal-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81` | 10.38 | `phi4mm`; context `131072` | Transformers (cached checkpoint) | True | sdpa | Phi4MMModel generation helper bridge required by the cached PEFT wrapper, DynamicCache legacy aliases required by cached Phi-4 attention code, num_logits_to_keep=0 required by cached Phi-4 forward code | complete |
| microsoft/Phi-3-vision-128k-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Phi-3-vision-128k-instruct/snapshots/ed2772fabe9dc9acd0caad54b62761d92520cc44` | 7.72 | `phi3_v`; context `131072` | Transformers (cached checkpoint) | True | eager | none | complete |
