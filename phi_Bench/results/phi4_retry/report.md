# Phi vision family benchmark

Generated: 2026-11-02 15:50:56 +0100
Seed: 3407; GPU memory ceiling: 30.0 GiB.
Runtime: Python unrecorded; PyTorch unrecorded; CUDA unrecorded; Transformers 4.57.1.
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
| microsoft/Phi-4-multimodal-instruct | failed | — | — | — | — | — | — | — | — | 0/8 | — |

## Runtime details

Times cover generation after preprocessing. Peak allocated VRAM is process-local.
Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.

| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
| microsoft/Phi-4-multimodal-instruct | ai2d | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | textvqa | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_truth | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_count | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_description | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_affordance | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | vqa_negation | 0/0 | — | — | — |
| microsoft/Phi-4-multimodal-instruct | graph_grab_lite | 0/0 | — | — | — |

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
| microsoft/Phi-4-multimodal-instruct | `/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/93f923e1a7727d1c4f446756212d9d3e8fcc5d81` | 10.38 | `phi4mm`; context `131072` | AttributeError: 'Phi4MMModel' object has no attribute 'prepare_inputs_for_generation' |

Configuration similarity alone does not establish weight identity. The converted Phi-3.5 checkpoint is treated as separate only if both weight sets are complete and can be compared.
