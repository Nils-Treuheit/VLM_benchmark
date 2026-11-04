# Grounding Specialist Benchmark Report

**Evaluation status:** partial

**Generated:** 2026-11-04T11:02:23.066494+00:00

## Scope

This report combines grounding-specialist localization and answer tasks with large vision-model baselines. Shared domain runs use the same image IDs and category labels across prompt-capable models. Detection, segmentation, grounding, OCR, and VQA remain separate measurements.

| Task | Dataset and split | Sample | Primary measures |
|---|---|---:|---|
| Referring expression | RefCOCOg test | 100 | Mean IoU and Acc@IoU 0.5 |
| Open-vocabulary detection | COCO val2017 | 100 | COCOeval AP on sampled images with uniform detection scores; local F1@IoU 0.5 |
| GUI grounding | ScreenSpot-Pro test | 100 | Point-in-target accuracy and Acc@IoU 0.5 |
| Text-region detection | HierText validation | 100 | Word-box F1@IoU 0.5 plus matched-word OCR exact accuracy and normalized edit similarity |

Sample targets: 100 per grounding task, 25 per VQAv2 panel, 100 TextVQA questions, and 100 COCO images; seed 3407.

## Model coverage

| Model | Revision | Backend | Precision | Mode | Status | Complete tasks |
|---|---|---|---|---|---|---:|
| `nvidia/LocateAnything-3B` | `c32291ca5e996f5a7a485845b4f57a233936bba0` | Transformers | bfloat16 | hybrid | complete | 0/400 |
| `GroundingPI/GroundAnything-VLM` | `not run` | not run | — | — | not run | 0/400 |

Runner/library versions by model:

- `nvidia/LocateAnything-3B`: torch 2.11.0, transformers 4.57.1, Pillow 12.2.0, safetensors 0.8.0, openai 2.44.0
- `GroundingPI/GroundAnything-VLM`: not recorded

## Task results

### Referring expression

Dataset: RefCOCOg test; target sample: 100 examples. Measures: Mean IoU and Acc@IoU 0.5.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | mean IoU —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | mean IoU —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

### Open-vocabulary detection

Dataset: COCO val2017; target sample: 100 examples. Measures: COCOeval AP on sampled images with uniform detection scores; local F1@IoU 0.5.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | local F1@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | local F1@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

### GUI grounding

Dataset: ScreenSpot-Pro test; target sample: 100 examples. Measures: Point-in-target accuracy and Acc@IoU 0.5.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | point-in-target —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | point-in-target —; Acc@0.5 — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

### Text-region detection

Dataset: HierText validation; target sample: 100 examples. Measures: Word-box F1@IoU 0.5 plus matched-word OCR exact accuracy and normalized edit similarity.

| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | not run | 0/100 | Box F1 —; OCR exact —; NED — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | not run | 0/100 | Box F1 —; OCR exact —; NED — | — s | — tok/s | — GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

## OCR, VQA, and counting

These answer-generation tasks are reported separately from localization and object detection. TextVQA measures question answering that depends on reading scene text. HierText adds word recognition to the existing text-box task by matching recognized words to word boxes at IoU 0.5.

| Model | Task | Status | Coverage | VQAv2 consensus / exact | Mean latency | Output rate | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | Counting VQA | complete | 25/25 | 0.0% / count exact 0.0% | 0.457 s | 136.360 tok/s | 20.610 GiB |
| `nvidia/LocateAnything-3B` | Binary VQA | complete | 25/25 | 0.0% / canonical exact 0.0% | 0.480 s | 119.749 tok/s | 20.612 GiB |
| `nvidia/LocateAnything-3B` | Open VQA | complete | 25/25 | 0.0% / canonical exact 0.0% | 0.457 s | 137.548 tok/s | 20.659 GiB |
| `nvidia/LocateAnything-3B` | OCR-dependent VQA | complete | 100/100 | 0.0% / canonical exact — | 0.504 s | 69.931 tok/s | 20.621 GiB |
| `GroundingPI/GroundAnything-VLM` | Counting VQA | not run | 0/25 | — / count exact — | — s | — tok/s | — GiB |
| `GroundingPI/GroundAnything-VLM` | Binary VQA | not run | 0/25 | — / canonical exact — | — s | — tok/s | — GiB |
| `GroundingPI/GroundAnything-VLM` | Open VQA | not run | 0/25 | — / canonical exact — | — s | — tok/s | — GiB |
| `GroundingPI/GroundAnything-VLM` | OCR-dependent VQA | not run | 0/100 | — / canonical exact — | — s | — tok/s | — GiB |

The VQAv2 panel reuses the exact 25-question sample from the general VLM skill panel. TextVQA uses a deterministic 100-question sample from the cached validation split. Scores use the 10-human-answer consensus metric.

## Object detection, classification, and segmentation

This COCO val2017 scorecard uses open-vocabulary YOLOE-26-L as the primary detector/segmenter, with closed-vocabulary YOLO26-L detection and segmentation plus SAM 3.1 as additional references. Detection and segmentation use COCO AP. Classification is label accuracy among predictions matched to any ground-truth object at box IoU 0.5; it is not whole-image classification accuracy.

| Model | Status | Images | Box AP / AP50 | Mask AP / AP50 | Label accuracy / joint / class F1 | RefCOCOg Acc@0.5 / latency | Mean latency | Images/s | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yoloe26l-seg` | not run | 0 | — / — | — / — | — / — / — | — / — s | — s | — | — GiB |
| `yolo26l` | not run | 0 | — / — | — / — | — / — / — | — / — s | — s | — | — GiB |
| `yolo26l-seg` | not run | 0 | — / — | — / — | — / — / — | — / — s | — s | — | — GiB |
| `sam3.1` | not run | 0 | — / — | — / — | — / — / — | — / — s | — s | — | — GiB |
| `nvidia/LocateAnything-3B` | not run | 0 | — / — | — / — | — / — / — | — / — s | — s | — | — GiB |
| `GroundingPI/GroundAnything-VLM` | not run | 0 | — / — | — / — | — / — / — | — / — s | — s | — | — GiB |

YOLOE-26 and SAM 3.1 receive the same full COCO category-name list per image. SAM latency includes one text-prompt query per category; YOLOE latency includes prompt setup and one image pass. YOLO26 uses its fixed COCO vocabulary. COCO detection and instance-segmentation AP use the official evaluator. Optional RefCOCOg text grounding reports mean IoU and Acc@IoU 0.5 for YOLOE and SAM. Throughput is single-image serial inference; model load time is excluded. The grounding specialists do not provide confidence scores, so their COCO AP uses uniform scores and is less informative than class-aware F1 for direct comparison. No overall rank combines these metrics.

## Selection guide

No complete model is eligible for a quality recommendation yet.
The local task macro weights the four task scores equally. RefCOCOg uses Acc@IoU 0.5, COCO uses sampled-image COCO AP when available and otherwise local micro F1@IoU 0.5, GUI uses point-in-target accuracy, and HierText uses detection F1. It is a local comparison metric.

| Task | Best complete model | Primary score |
|---|---|---:|
| Referring expression | No complete result | — |
| Open-vocabulary detection | No complete result | — |
| GUI grounding | No complete result | — |
| Text-region detection | No complete result | — |

## Additional dataset tracks

COCO remains the shared natural-image anchor. These tracks add different evaluation pressure; their metrics are kept separate from COCO and from each other.

| Dataset | Status | Task | Fit and limitation |
|---|---|---|---|
| DOTA v1.0 validation | 100-image shared HBB manifest prepared; GPU run pending | Shared aerial detection plus native OBB for YOLO26-L | 100 of 458 validation images are sampled for horizontal-box COCO AP across YOLOE, SAM 3.1, and both grounding specialists. YOLO26-L-OBB uses the full validation split for the separate native OBB reference. DOTA terms restrict use to academic purposes and prohibit commercial use. |
| BOP-H3 HOPE v2 validation | 50-image shared manifest; SAM 3.1 and LocateAnything runs complete; other baselines pending | Object-centric 2D detection; instance masks where supported | All 50 local validation images are shared by YOLOE, SAM 3.1, and both grounding specialists. Box AP is common; mask AP is reported only for models that return masks. This is the 2D HOPE v2 subset, not the full BOP-H3 pose/onboarding protocol. |
| KITTI object detection | download image_2 training archive | Automotive 2D detection | Labels and calibration are local, but camera images are missing. Download the KITTI image_2 archive before using a fixed train/validation split. |
| LVIS v1 validation | candidate | Long-tail detection and instance segmentation | Useful for rare-category metrics, but it reuses COCO imagery. Download only validation annotations first; equal full-vocabulary SAM and VLM runs across 1,203 labels would be costly and need a fixed prompt protocol. |
| RF100-VL | candidate; Roboflow API key not present | Multi-domain open-vocabulary detection | Highest-value new download for domain shift. Start with aerial, document, laboratory, flora/fauna, and industrial subsets, not all 100 datasets. The downloader requires a Roboflow API key. |
| BOP-H3 HOT3D/HANDAL | separate follow-up | Unseen-object detection, onboarding, and 6D pose | Keep separate from class-name box AP. Use the official BOP protocol for pose tasks and review each dataset's terms before expanding beyond local HOPE v2. |

### Shared non-COCO domain runs

Every row uses the prepared image IDs and category list for its domain. Box AP and class-aware F1 are shared across vision models and grounding specialists; mask AP is shown when available.

| Dataset | Model | Status | Images | Input size | Box AP / AP50 | Class-aware F1 | Mask AP | Mean latency | Images/s |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| BOP-H3 HOPE v2 | LocateAnything-3B | complete | 50/50 | native model processor | 13.0% / 17.0% | 30.5% | — | 5.572 s | 0.179 |
| BOP-H3 HOPE v2 | SAM 3.1 | complete | 50/50 | native image dimensions; YOLO imgsz=640; batch=1 | 3.8% / 4.6% | 14.3% | 4.4% | 0.709 s | 1.411 |
| BOP-H3 HOPE v2 | YOLOE-26-L | complete | 50/50 | native image dimensions; YOLO imgsz=640; batch=1 | 2.6% / 3.6% | 3.3% | 2.1% | 0.033 s | 30.463 |

### DOTA v1.0 result

Status: not run. YOLO26-L-OBB validation AP: —; AP50: —; inference latency: — ms/image; whole-device peak VRAM: — GiB.
This is an in-domain OBB reference, not a zero-shot YOLOE comparison. Validation speed excludes model load; peak whole-device VRAM sampling includes model load.

## Method

The two pinned model snapshots use their own processors, chat templates, and task prompts. Sampling uses seed 3407; both models receive the same fixed task-specific manifest rows. Requests run serially, with one image per call and deterministic decoding. LocateAnything uses hybrid generation, BF16, and up to 8,192 output tokens. GroundAnything uses its model-specific BF16/eager SGLang release profile and up to 4,096 output tokens.

RefCOCOg reports mean IoU and Acc@IoU 0.5. COCO reports COCOeval AP on the selected image IDs when local COCO annotations and category mappings are available, plus local micro F1 at IoU 0.5. COCO predictions use a uniform confidence score of 1.0 because these models do not emit box confidence. GUI grounding reports point-in-target accuracy and box Acc@IoU 0.5. HierText reports corpus-level one-to-one detection precision, recall, and F1 at IoU 0.5. Empty, malformed, truncated, and duplicate outputs are counted. Whole-device VRAM includes the desktop baseline.

## Data availability and limitations

Local manifests were used for these runs. Dataset images and item-level annotations were not copied into the site. 1 model(s) have at least one completed example; partial and failed coverage remains visible in the scorecards.
RefCOCOg uses its public test split. COCO uses val2017. ScreenSpot-Pro uses test. HierText uses validation, matching the planned protocol. VQAv2 uses a 25-question existing panel; TextVQA uses a deterministic sample of 100 validation questions. These specialists do not document general VQA as a native task, so their answer scores measure exploratory transfer. TextVQA is OCR-dependent VQA, while HierText provides the direct word-box and transcription score. Dataset images and item-level predictions remain local.

## Runtime and artifacts

Per-item results preserve the prompt, raw output, canonical predictions, parser state, error, latency, token counts, image dimensions, and scores in `results/runs/`. LocateAnything model-load peak and per-task peaks are recorded separately; the GroundAnything server load happens before the client run. This folder is excluded from the site. The site includes aggregate metrics only. Whole-device VRAM is sampled every 0.5 seconds with `nvidia-smi`; the reported maximum includes all processes and desktop use. Package versions describe the client runner; the GroundAnything service inventory is saved from its `/models` endpoint, while the custom server's full package set is not exposed by that endpoint.

## Sources

- [LocateAnything-3B model card](https://huggingface.co/nvidia/LocateAnything-3B)
- [LocateAnything evaluation](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [LocateAnything evaluation data](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData)
- [GroundAnything-VLM model card](https://huggingface.co/GroundingPI/GroundAnything-VLM)
- [RefCOCOg](https://github.com/lichengunc/refer)
- [COCO](https://cocodataset.org/#download)
- [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)
- [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981)
- [HierText](https://github.com/google-research-datasets/hiertext)
- [GroundAnything evaluation](https://github.com/groundingpi/GroundAnything)
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics YOLOE-26](https://docs.ultralytics.com/models/yoloe)
- [Meta SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [VQAv2 dataset and terms](https://visualqa.org/terms.html)
- [TextVQA dataset](https://huggingface.co/datasets/facebook/textvqa)
- [VQAv2 evaluator](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py)
- [DOTA v1.0 dataset](https://captain-whu.github.io/DOTA/dataset)
- [Ultralytics YOLO26 OBB and DOTA](https://docs.ultralytics.com/tasks/obb)
- [LVIS dataset](https://www.lvisdataset.org/)
- [Roboflow 100-VL benchmark](https://github.com/roboflow/rf100-vl)
- [KITTI object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php)
- [BOP datasets and BOP-H3](https://bop.felk.cvut.cz/datasets/)
- [BOP challenge tracks](https://bop.felk.cvut.cz/challenges/)
