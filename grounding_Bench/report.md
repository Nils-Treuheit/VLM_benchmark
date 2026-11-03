# Grounding Specialist Benchmark Report

**Evaluation status:** not run

**Generated:** 2026-11-03T16:33:21.537730+00:00

## Scope

This benchmark compares two pinned grounding checkpoints on four separate localization tasks. Results are not part of the generic image-question-answering ranking.

| Task | Dataset and split | Sample | Primary measures |
|---|---|---:|---|
| Referring expression | RefCOCOg test | 100 | Mean IoU and Acc@IoU 0.5 |
| Open-vocabulary detection | COCO val2017 | 100 | COCOeval AP on sampled images with uniform detection scores; local F1@IoU 0.5 |
| GUI grounding | ScreenSpot-Pro test | 100 | Point-in-target accuracy and Acc@IoU 0.5 |
| Text-region detection | HierText validation | 100 | Box-based F1@IoU 0.5, precision, and recall |

Sample count: 100 per task; seed 3407.

## Model coverage

| Model | Revision | Backend | Precision | Mode | Status | Complete tasks |
|---|---|---|---|---|---|---:|
| `nvidia/LocateAnything-3B` | `not run` | not run | — | — | not run | 0/400 |
| `GroundingPI/GroundAnything-VLM` | `not run` | not run | — | — | not run | 0/400 |

Runner/library versions by model:

- `nvidia/LocateAnything-3B`: not recorded
- `GroundingPI/GroundAnything-VLM`: not recorded

## Task results

### Referring expression

Dataset: RefCOCOg test; target sample: 100 examples. Measures: Mean IoU and Acc@IoU 0.5.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | mean IoU —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | mean IoU —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |

### Open-vocabulary detection

Dataset: COCO val2017; target sample: 100 examples. Measures: COCOeval AP on sampled images with uniform detection scores; local F1@IoU 0.5.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | local F1@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | local F1@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |

### GUI grounding

Dataset: ScreenSpot-Pro test; target sample: 100 examples. Measures: Point-in-target accuracy and Acc@IoU 0.5.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | point-in-target —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | point-in-target —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |

### Text-region detection

Dataset: HierText validation; target sample: 100 examples. Measures: Box-based F1@IoU 0.5, precision, and recall.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | F1 —; P —; R — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | F1 —; P —; R — | — s | — tok/s | — GiB | 0 malformed, 0 truncated, 0 empty, 0 duplicate |

## Selection guide

No complete model is eligible for a quality recommendation yet.
The local task macro weights the four task scores equally. RefCOCOg uses Acc@IoU 0.5, COCO uses sampled-image COCO AP when available and otherwise local micro F1@IoU 0.5, GUI uses point-in-target accuracy, and HierText uses detection F1. It is a local comparison metric.

| Task | Best complete model | Primary score |
|---|---|---:|
| Referring expression | No complete result | — |
| Open-vocabulary detection | No complete result | — |
| GUI grounding | No complete result | — |
| Text-region detection | No complete result | — |

## Method

The two pinned model snapshots use their own processors, chat templates, and task prompts. Sampling uses seed 3407 and the same 100 manifest rows for both models. Requests run serially, with one image per call and deterministic decoding. LocateAnything uses hybrid generation, BF16, and up to 8,192 output tokens. GroundAnything uses its model-specific BF16/eager SGLang release profile and up to 4,096 output tokens.

RefCOCOg reports mean IoU and Acc@IoU 0.5. COCO reports COCOeval AP on the selected image IDs when local COCO annotations and category mappings are available, plus local micro F1 at IoU 0.5. COCO predictions use a uniform confidence score of 1.0 because these models do not emit box confidence. GUI grounding reports point-in-target accuracy and box Acc@IoU 0.5. HierText reports corpus-level one-to-one detection precision, recall, and F1 at IoU 0.5. Empty, malformed, truncated, and duplicate outputs are counted. Whole-device VRAM includes the desktop baseline.

## Data availability and limitations

No dataset annotations or image assets were found in the inspected local Hugging Face dataset cache at implementation time. No assets were downloaded or copied. The benchmark runner therefore has no measured results yet. Supply legally obtained local dataset exports and image paths using the schema in `README.md`, create shared manifests with `prepare_data.py`, then run both models.
RefCOCOg uses its public test split. COCO uses val2017. ScreenSpot-Pro uses test. HierText uses validation, matching the planned protocol. Dataset images and item-level predictions remain local.

## Runtime and artifacts

Per-item results preserve the prompt, raw output, canonical predictions, parser state, error, latency, token counts, image dimensions, and scores in `results/runs/`. LocateAnything model-load peak and per-task peaks are recorded separately; the GroundAnything server load happens before the client run. This folder is excluded from the site. The site includes aggregate metrics only. Whole-device VRAM is sampled every 0.5 seconds with `nvidia-smi`; the reported maximum includes all processes and desktop use. Package versions describe the client runner; the GroundAnything service inventory is saved from its `/models` endpoint, while the custom server's full package set is not exposed by that endpoint.

## Sources

- [LocateAnything-3B model card](https://huggingface.co/nvidia/LocateAnything-3B)
- [LocateAnything evaluation](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [GroundAnything-VLM model card](https://huggingface.co/GroundingPI/GroundAnything-VLM)
- [RefCOCOg](https://github.com/lichengunc/refer)
- [COCO](https://cocodataset.org/#download)
- [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)
- [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981)
- [HierText](https://github.com/google-research-datasets/hiertext)
- [GroundAnything evaluation](https://github.com/groundingpi/GroundAnything)
