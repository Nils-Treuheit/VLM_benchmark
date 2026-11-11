# Grounding Specialist Benchmark Report

**Evaluation status:** complete

**Generated:** 2026-11-11T17:00:57.099713+00:00

## Scope

This report combines grounding-specialist localization and answer tasks with large vision-model baselines. Shared domain runs use the same image IDs and category labels across prompt-capable models. Detection, segmentation, grounding, OCR, and VQA remain separate measurements.

| Task | Dataset and split | Sample | Primary measures |
|---|---|---:|---|
| Referring expression | RefCOCOg lmms-lab test derivative | 100 | Mean IoU and Acc@IoU 0.5 |
| Open-vocabulary detection | COCO val2017 | 100 | COCOeval AP on sampled images with uniform detection scores; local F1@IoU 0.5 |
| GUI grounding | ScreenSpot-Pro test | 100 | Point-in-target accuracy and Acc@IoU 0.5 |
| Text-region detection | HierText validation | 100 | Word-box F1@IoU 0.5 plus matched-word OCR exact accuracy and normalized edit similarity |

Sample targets: 100 per grounding task, 25 per VQAv2 panel, 100 TextVQA questions, and 100 COCO images; seed 3407.

## Model coverage

| Model | Revision | Backend | Precision | Mode | Status | Complete tasks |
|---|---|---|---|---|---|---:|
| `nvidia/LocateAnything-3B` | `c32291ca5e996f5a7a485845b4f57a233936bba0` | Transformers | bfloat16 | hybrid | complete | 400/400 |
| `GroundingPI/GroundAnything-VLM` | `0f3cf22027493b4f97f342dae1fb298c38b68d27` | GroundAnything custom SGLang | bfloat16 | autoregressive; release BF16/eager SGLang profile | complete | 400/400 |

Runner/library versions by model:

- `nvidia/LocateAnything-3B`: torch 2.11.0, transformers 4.57.1, Pillow 12.2.0, safetensors 0.8.0, openai 2.44.0
- `GroundingPI/GroundAnything-VLM`: torch 2.11.0, transformers 5.19.0, Pillow 12.2.0, safetensors 0.8.0, openai 2.44.0

## Task results

### Referring expression

Dataset: RefCOCOg lmms-lab test derivative; target sample: 100 examples. Measures: Mean IoU and Acc@IoU 0.5.

| Model | Status | Coverage | Scores | Mean / median / p95 / max query (s) | Model load (s) | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | complete | 100/100 | mean IoU 81.0%; Acc@0.5 85.0% | 2.334 / 0.167 / 0.667 / 62.191 | 17.424 | 138.130 tok/s | 11.103 GiB | 0 malformed, 0 invalid, 5 truncated, 0 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | complete | 100/100 | mean IoU 85.5%; Acc@0.5 90.0% | 0.507 / 0.314 / 0.423 / 19.303 | 624.000 | 58.055 tok/s | 17.366 GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

LocateAnything had 5 truncated RefCOCOg outputs at its 8,192-token cap; their query times ranged from 33.401 to 62.191 s. These tail requests exceed the nearest-rank p95 and are visible in the max query column.
GroundAnything's slowest RefCOCOg query took 19.303 s; the mean includes it, while p95 excludes the slowest five percent.

### Open-vocabulary detection

Dataset: COCO val2017; target sample: 100 examples. Measures: COCOeval AP on sampled images with uniform detection scores; local F1@IoU 0.5.

| Model | Status | Coverage | Scores | Mean / median / p95 / max query (s) | Model load (s) | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | complete | 100/100 | AP 13.2%; AP50 15.2% | 12.218 / 13.234 / 14.219 / 25.778 | — | 292.430 tok/s | 11.439 GiB | 0 malformed, 17 invalid, 88 truncated, 10 empty, 0 duplicate |
| `GroundingPI/GroundAnything-VLM` | complete | 100/100 | AP 4.8%; AP50 7.2% | 13.571 / 13.033 / 15.602 / 24.264 | — | 72.827 tok/s | 24.354 GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

### GUI grounding

Dataset: ScreenSpot-Pro test; target sample: 100 examples. Measures: Point-in-target accuracy and Acc@IoU 0.5.

| Model | Status | Coverage | Scores | Mean / median / p95 / max query (s) | Model load (s) | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | complete | 100/100 | point-in-target 38.0%; Acc@0.5 0.0% | 2.647 / 0.534 / 35.253 / 35.623 | 4.667 | 45.446 tok/s | 20.803 GiB | 0 malformed, 4 invalid, 6 truncated, 3 empty, 7985 duplicate |
| `GroundingPI/GroundAnything-VLM` | complete | 100/100 | point-in-target 65.0%; Acc@0.5 0.0% | 0.563 / 0.515 / 0.833 / 2.001 | 10.000 | 22.748 tok/s | 19.608 GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

The fixed ScreenSpot-Pro sample contains 100 examples selected with seed 3407 from all 1,581 official test annotations; the full 1,588-image repository is stored under the public-datasets tree. One annotation extended one pixel above its screenshot and was clipped to the visible image boundary before normalization. LocateAnything's successful run used Transformers 4.57.1 and a 4,096-patch image cap; its earlier Transformers 5.19.0 attempt failed before inference and remains preserved separately.

### Text-region detection

Dataset: HierText validation; target sample: 100 examples. Measures: Word-box F1@IoU 0.5 plus matched-word OCR exact accuracy and normalized edit similarity.

| Model | Status | Coverage | Scores | Mean / median / p95 / max query (s) | Model load (s) | Output rate | Peak whole-device VRAM | Parse issues |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | complete | 100/100 | Box F1 12.8%; OCR exact 59.4%; NED 80.7% | 15.070 / 2.785 / 46.816 / 49.960 | 4.479 | 216.800 tok/s | 21.252 GiB | 0 malformed, 40 invalid, 22 truncated, 0 empty, 19130 duplicate |
| `GroundingPI/GroundAnything-VLM` | complete | 100/100 | Box F1 27.0%; OCR exact 70.7%; NED 89.9% | 6.005 / 5.280 / 14.812 / 23.768 | 624.000 | 68.730 tok/s | 24.282 GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

LocateAnything's complete HierText run capped image patches at 4,096 and allowed up to 8,192 output tokens. A native 25,600-patch attempt is preserved separately at 99/100 after one CUDA OOM.

## OCR, VQA, and counting

These answer-generation tasks are reported separately from localization and object detection. TextVQA measures question answering that depends on reading scene text. HierText adds word recognition to the existing text-box task by matching recognized words to word boxes at IoU 0.5.

| Model | Task | Status | Coverage | VQAv2 consensus / exact | Mean / median query (s) | p95 query (s) | Output rate | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `nvidia/LocateAnything-3B` | Counting VQA | complete | 25/25 | 0.0% / count exact 0.0% | 0.457 / 0.425 | 0.757 | 136.360 tok/s | 8.638 GiB |
| `nvidia/LocateAnything-3B` | Binary VQA | complete | 25/25 | 0.0% / canonical exact 0.0% | 0.480 / 0.539 | 0.752 | 119.749 tok/s | 8.638 GiB |
| `nvidia/LocateAnything-3B` | Open VQA | complete | 25/25 | 0.0% / canonical exact 0.0% | 0.457 / 0.462 | 0.770 | 137.548 tok/s | 8.638 GiB |
| `nvidia/LocateAnything-3B` | OCR-dependent VQA | complete | 100/100 | 0.0% / canonical exact — | 0.504 / 0.456 | 0.799 | 69.931 tok/s | 8.638 GiB |
| `GroundingPI/GroundAnything-VLM` | Counting VQA | complete | 25/25 | 0.0% / count exact 0.0% | 0.387 / 0.286 | 0.917 | 60.833 tok/s | 23.936 GiB |
| `GroundingPI/GroundAnything-VLM` | Binary VQA | complete | 25/25 | 0.0% / canonical exact 0.0% | 0.281 / 0.280 | 0.332 | 58.006 tok/s | 23.936 GiB |
| `GroundingPI/GroundAnything-VLM` | Open VQA | complete | 25/25 | 0.0% / canonical exact 0.0% | 0.266 / 0.261 | 0.348 | 56.775 tok/s | 23.940 GiB |
| `GroundingPI/GroundAnything-VLM` | OCR-dependent VQA | complete | 100/100 | 0.0% / canonical exact — | 0.564 / 0.431 | 1.126 | 36.053 tok/s | 23.937 GiB |

The VQAv2 panel reuses the exact 25-question sample from the general VLM skill panel. TextVQA uses a deterministic 100-question sample from the cached validation split. Scores use the 10-human-answer consensus metric.

## Object detection, classification, and segmentation

This COCO val2017 scorecard uses open-vocabulary YOLOE-26-L as the primary detector/segmenter, with closed-vocabulary YOLO26-L detection and segmentation plus SAM 3.1 as additional references. Detection and segmentation use COCO AP. Classification is label accuracy among predictions matched to any ground-truth object at box IoU 0.5; it is not whole-image classification accuracy.

| Model | Status | Images | Box AP / AP50 | Mask AP / AP50 | Label accuracy / joint / class F1 | RefCOCOg Acc@0.5 / mean, median, p95, max query s / model load s | Mean / median COCO query (s) | Images/s | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yoloe26l-seg` | complete | 100 | 55.2% / 71.3% | 34.1% / 58.4% | 93.5% / 86.8% / 10.2% | 100.0% / 0.307, 0.299, 0.329, 0.399 / 1.994 | 0.026 / 0.022 | 38.093 | 3.051 GiB |
| `yolo26l` | complete | 100 | 66.4% / 81.4% | — / — | 95.6% / 91.2% / 11.8% | — | 0.010 / 0.008 | 96.856 | 2.022 GiB |
| `yolo26l-seg` | complete | 100 | 66.8% / 81.8% | 43.8% / 71.6% | 95.3% / 91.3% / 13.0% | — | 0.012 / 0.010 | 84.075 | 2.920 GiB |
| `sam3.1` | complete | 100 | 60.7% / 75.4% | 52.7% / 73.5% | 97.9% / 78.3% / 63.0% | 39.0% / 0.080, 0.080, 0.094, 0.099 / 58.098 | 2.129 / 2.126 | 0.470 | 5.906 GiB |
| `nvidia/LocateAnything-3B` | complete | 100 | 13.2% / 15.2% | — / — | — / — / 24.1% | — | 12.218 / 13.234 | 0.082 | 11.439 GiB |
| `GroundingPI/GroundAnything-VLM` | complete | 100 | 4.8% / 7.2% | — / — | — / — / 8.5% | — | 13.571 / 13.033 | 0.074 | 24.354 GiB |

### Accuracy and query-time comparison

On the shared 100-image COCO panel, YOLOE-26-L reached 55.2% box AP at 0.026 s/query. LocateAnything reached 13.2% AP at 12.218 s/query (465.406× the YOLOE query time), and GroundAnything reached 4.8% AP at 13.571 s/query (516.944×). Neither grounding VLM improved COCO AP on this panel. Closed-vocabulary YOLO26-L reached 66.4% AP at 0.010 s/query. These COCO AP values use uniform VLM box scores, and the per-model query timing boundaries differ as described below.
On the adapted RefCOCOg panel, YOLOE-26-L reached 100.0% Acc@IoU 0.5 at 0.307 s/query; LocateAnything reached 85.0% at 2.334 s/query, and GroundAnything reached 90.0% at 0.507 s/query. The VLMs did not exceed the YOLOE score on this panel. Cold-start load is separate: it was captured for these RefCOCOg runs, while the older VLM COCO runs have no recoverable model-load duration.

YOLOE-26 and SAM 3.1 receive the same full COCO category-name list per image. SAM latency includes one text-prompt query per category; YOLOE latency includes prompt setup and one image pass. YOLO26 uses its fixed COCO vocabulary. COCO detection and instance-segmentation AP use the official evaluator. Optional RefCOCOg text grounding reports mean IoU and Acc@IoU 0.5 for YOLOE and SAM. Mean, median, p95, maximum query latency, and initial model or service load duration are reported separately. The GroundAnything load value for this run measures server process start through API readiness, including imports, checkpoint loading, and KV-cache allocation. Throughput is single-image serial inference; model load time is excluded from query latency. The grounding specialists do not provide confidence scores, so their COCO AP uses uniform scores and is less informative than class-aware F1 for direct comparison. No overall rank combines these metrics.

## Selection guide

Best complete grounding specialist: `GroundingPI/GroundAnything-VLM` with local task macro 46.7%.
The local task macro weights the four task scores equally. RefCOCOg uses Acc@IoU 0.5, COCO uses sampled-image COCO AP when available and otherwise local micro F1@IoU 0.5, GUI uses point-in-target accuracy, and HierText uses detection F1. It is a local comparison metric.

| Task | Best grounding specialist | Primary score |
|---|---|---:|
| Referring expression | `GroundingPI/GroundAnything-VLM` | acc_iou_05: 90.0% |
| Open-vocabulary detection | `nvidia/LocateAnything-3B` | COCO AP: 13.2% |
| GUI grounding | `GroundingPI/GroundAnything-VLM` | point_in_target: 65.0% |
| Text-region detection | `GroundingPI/GroundAnything-VLM` | f1: 27.0% |

## Additional dataset tracks

COCO remains the shared natural-image anchor. These tracks add different evaluation pressure; their metrics are kept separate from COCO and from each other.

| Dataset | Status | Task | Fit and limitation |
|---|---|---|---|
| COCO val2017 | 100-image shared manifest; YOLOE, YOLO26, SAM 3.1, LocateAnything, and GroundAnything runs complete | Shared detection, classification, instance segmentation, and speed | Every available vision baseline uses the same 100 annotated images and all 80 COCO classes. Grounding specialists are scored on boxes; models without confidence scores use uniform scores for COCO AP. |
| DOTA v1.0 validation | Shared HBB runs complete for YOLOE, YOLO26-L, SAM 3.1, and GroundAnything; LocateAnything native-input attempt partial (51/100), 4,096-patch recovery 100/100 with 41 parser-valid; full-split YOLO26-L-OBB complete | Shared aerial detection plus native OBB for YOLO26-L | 100 of 456 labeled validation images are sampled for horizontal-box COCO AP across YOLOE, SAM 3.1, and available grounding specialists. GroundAnything completed all 100 items with five outputs truncated at 2,048 tokens; official COCO AP was unavailable because three predicted labels did not map to annotation categories. LocateAnything's native-input 2,048-token run completed 51/100 before 49 CUDA OOM failures; its 51 completed items include 33 parser-valid outputs and yield COCOeval AP 0.046%. A separate run caps the image processor at 4,096 patches and completes 100/100, with 41 parser-valid outputs; 62 responses were truncated at 2,048 tokens. COCOeval AP is unavailable because one predicted label did not map to a category. Keep the two LocateAnything settings separate. YOLO26-L-OBB uses the full 458-file validation split, including two background-only images, for the separate native OBB reference. Its clean rerun recorded 8.6 ms/image and 2.351 GiB peak whole-device VRAM. DOTA terms restrict use to academic purposes and prohibit commercial use. |
| BOP-H3 HOPE v2 validation | 50-image shared manifest; LocateAnything, GroundAnything, SAM 3.1, and YOLOE complete; closed-vocabulary YOLO26 omitted because its COCO label set does not cover HOPE classes | Object-centric 2D detection; instance masks where supported | All 50 local validation images are shared by YOLOE, SAM 3.1, and both grounding specialists. GroundAnything's official COCO AP is unavailable because one predicted label did not map to the annotation categories; its class-aware F1 remains reportable. Mask AP is shown only for models that return masks. This is the 2D HOPE v2 subset, not the full BOP-H3 pose/onboarding protocol. |
| KITTI object detection | Deprioritized; local training-panel inference complete | Automotive 2D detection | The fixed 100-image panel is sampled from a deterministic 80/20 partition of labeled training images and uses Car, Pedestrian, and Cyclist boxes. It excludes Van, Person_sitting, Truck, Tram, Misc, and DontCare labels; its COCO-shaped annotations support local comparative AP/F1, not official KITTI AP_R40. LocateAnything used a 2,048-token response cap; 98 outputs reached the cap, and one unmapped label prevented official COCOeval AP. GroundAnything query latency excludes the approximately 10-second shared service startup. Public test labels are unavailable. Per scope, KITTI remains a local exploratory panel and has no remaining benchmark work or official AP_R40 completion gate. |
| LVIS v1 validation | GroundAnything, LocateAnything, SAM 3.1, and YOLOE-26-L complete at 100/100; YOLO26-L and YOLO26-L-seg have no supported detections for the selected categories | Long-tail detection and instance segmentation | This exploratory panel uses 10 rare, 10 common, and 10 frequent categories across 100 fixed LVIS val images sourced from COCO 2017 train/val. It uses official LVIS AP and frequency-group metrics, not the full 1,203-category leaderboard; COCO-trained baselines may have seen some train images. YOLO26-L and YOLO26-L-seg were run on the same panel, but their fixed COCO vocabulary produced no detections for these selected long-tail categories, so LVIS AP is unsupported. |
| RF100-VL | All-subset runs complete across 100 panels | Multi-domain open-vocabulary detection | The local test export contains 100 subsets. Per-subset panels use up to 20 shared seeded test images. nvidia/LocateAnything-3B: 100/100, GroundingPI/GroundAnything-VLM: 100/100, yoloe26l-seg: 100/100. All local test exports contain bounding boxes but no instance masks, including subsets whose names mention segmentation; mask AP is unsupported. Closed-vocabulary YOLO26 results remain limited to panels with matching labels. |
| BOP-H3 HOT3D/HANDAL | separate follow-up | Unseen-object detection, onboarding, and 6D pose | Keep separate from class-name box AP. Use the official BOP protocol for pose tasks and review each dataset's terms before expanding beyond local HOPE v2. |

### Shared non-COCO domain runs

Every row uses the prepared image IDs and category list for its domain. Query latency excludes model load. YOLO timing includes image loading, preprocessing, and prompt setup; GroundAnything timing in the saved runs starts after local image reading, base64 encoding, and request serialization, then covers the HTTP call through the full response. Historical LocateAnything timings cover only model.generate(), excluding image reading, processor encoding, decoding, and parsing. The ratios compare each model with the fastest complete YOLO detector or segmenter on that domain, but the saved timing scopes differ.

| Dataset | Model | Status | Images | Model load s | Input size | Box AP / AP50 / AP75 | AR@300 | Class-aware F1 | Mask AP | AP rare / common / frequent | Mean query s | Median query s | p95 query s | Parse invalid / empty / truncated | Images/s | Latency / fastest YOLO |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| COCO val2017 | GroundAnything-VLM | complete | 100/100 | — | native model processor | 4.8% / 7.2% / 5.1% | — | 8.5% | — | — / — / — | 13.571 | 13.033 | 15.602 | 0 / 0 / 0 | 0.074 | 1314.390× |
| COCO val2017 | LocateAnything-3B | complete | 100/100 | — | native model processor | 13.2% / 15.2% / 14.1% | — | 24.1% | — | — / — / — | 12.218 | 13.234 | 14.219 | 17 / 10 / 88 | 0.082 | 1183.351× |
| COCO val2017 | YOLOE-26-L | complete | 100/100 | 1.994 | native image dimensions; YOLO imgsz=640; batch=1 | 55.2% / 71.3% / — | — | 10.2% | 34.1% | — / — / — | 0.026 | 0.022 | 0.038 | — | 38.093 | 2.543× |
| COCO val2017 | YOLO26-L | complete | 100/100 | — | native image dimensions; YOLO imgsz=640; batch=1 | 66.4% / 81.4% / — | — | 11.8% | — | — / — / — | 0.010 | 0.008 | 0.022 | — | 96.856 | 1.000× |
| COCO val2017 | YOLO26-L-seg | complete | 100/100 | — | native image dimensions; YOLO imgsz=640; batch=1 | 66.8% / 81.8% / — | — | 13.0% | 43.8% | — / — / — | 0.012 | 0.010 | 0.026 | — | 84.075 | 1.152× |
| COCO val2017 | SAM 3.1 | complete | 100/100 | 58.098 | native image dimensions; YOLO imgsz=640; batch=1 | 60.7% / 75.4% / — | — | 63.0% | 52.7% | — / — / — | 2.129 | 2.126 | 2.175 | — | 0.470 | 206.200× |
| DOTA v1.0 | GroundAnything-VLM | complete | 100/100 | — | native model processor | — / — / — | — | 1.3% | — | — / — / — | 8.070 | 5.325 | 25.576 | 0 / 0 / 5 | 0.124 | 108.038× |
| DOTA v1.0 | LocateAnything-3B | complete | 100/100 | 4.422 | native processor; max image patches=4096 | — / — / — | — | 0.1% | — | — / — / — | 4.563 | 6.427 | 7.384 | 59 / 31 / 62 | 0.219 | 61.089× |
| DOTA v1.0 | SAM 3.1 | complete | 100/100 | — | native image dimensions; YOLO imgsz=1024; batch=1 | 0.6% / 1.6% / 0.5% | — | 2.7% | — | — / — / — | 0.488 | 0.429 | 0.776 | — | 2.049 | 6.534× |
| DOTA v1.0 | YOLO26-L | complete | 100/100 | — | native image dimensions; YOLO imgsz=1024; batch=1 | 0.1% / 0.3% / 0.0% | — | 0.4% | — | — / — / — | 0.075 | 0.045 | 0.176 | — | 13.388 | 1.000× |
| DOTA v1.0 | YOLOE-26-L | complete | 100/100 | — | native image dimensions; YOLO imgsz=1024; batch=1 | 0.1% / 0.5% / 0.1% | — | 0.9% | — | — / — / — | 0.126 | 0.085 | 0.255 | — | 7.935 | 1.687× |
| BOP-H3 HOPE v2 | GroundAnything-VLM | complete | 50/50 | — | native model processor | — / — / — | — | 37.6% | — | — / — / — | 5.310 | 5.247 | 5.813 | 0 / 0 / 0 | 0.188 | 161.747× |
| BOP-H3 HOPE v2 | LocateAnything-3B | complete | 50/50 | — | native model processor | 13.0% / 17.0% / 13.4% | — | 30.5% | — | — / — / — | 5.572 | 5.985 | 6.820 | 48 / 0 / 40 | 0.179 | 169.728× |
| BOP-H3 HOPE v2 | SAM 3.1 | complete | 50/50 | — | native image dimensions; YOLO imgsz=640; batch=1 | 3.8% / 4.6% / 4.2% | — | 14.3% | 4.4% | — / — / — | 0.709 | 0.708 | 0.724 | — | 1.411 | 21.586× |
| BOP-H3 HOPE v2 | YOLOE-26-L | complete | 50/50 | — | native image dimensions; YOLO imgsz=640; batch=1 | 2.6% / 3.6% / 2.6% | — | 3.3% | 2.1% | — / — / — | 0.033 | 0.033 | 0.035 | — | 30.463 | 1.000× |
| LVIS v1 validation (30-category stratified panel) | GroundAnything-VLM | complete | 100/100 | — | native model processor | 39.4% / 51.5% / 42.2% | 49.4% | 7.4% | — | 50.1% / 40.6% / 27.4% | 6.246 | 5.578 | 9.083 | — | 0.160 | 549.151× |
| LVIS v1 validation (30-category stratified panel) | LocateAnything-3B | complete | 100/100 | 4.832 | native model processor | 12.5% / 13.5% / 13.1% | 12.7% | 8.2% | — | 19.0% / 10.0% / 8.5% | 3.696 | 5.656 | 6.290 | — | 0.271 | 324.935× |
| LVIS v1 validation (30-category stratified panel) | SAM 3.1 | complete | 100/100 | 20.575 | native image dimensions; YOLO imgsz=640; batch=1 | 36.4% / 40.2% / 38.5% | 37.0% | 31.0% | 32.3% | 57.9% / 27.0% / 24.4% | 0.813 | 0.812 | 0.824 | — | 1.230 | 71.473× |
| LVIS v1 validation (30-category stratified panel) | YOLO26-L | unsupported | 100/100 | 1.485 | native image dimensions; YOLO imgsz=640; batch=1 | — / — / — | — | 0.0% | — | — / — / — | 0.020 | 0.018 | 0.034 | — | 49.651 | —× |
| LVIS v1 validation (30-category stratified panel) | YOLO26-L-seg | unsupported | 100/100 | 2.291 | native image dimensions; YOLO imgsz=640; batch=1 | — / — / — | — | 0.0% | — | — / — / — | 0.022 | 0.020 | 0.041 | — | 45.161 | —× |
| LVIS v1 validation (30-category stratified panel) | YOLOE-26-L | complete | 100/100 | 1.842 | native image dimensions; YOLO imgsz=640; batch=1 | 43.3% / 53.6% / 45.8% | 50.6% | 8.4% | 30.7% | 56.9% / 38.3% / 34.6% | 0.011 | 0.010 | 0.026 | — | 87.921 | 1.000× |
| KITTI object training (local 2D panel) | GroundAnything-VLM | complete | 100/100 | 10.000 | native model processor | 11.9% / 25.0% / 11.3% | — | 56.2% | — | — / — / — | 1.156 | 1.104 | 1.848 | 0 / 0 / 0 | 0.865 | 99.213× |
| KITTI object training (local 2D panel) | LocateAnything-3B | complete | 100/100 | 4.610 | native model processor | — / — / — | — | 25.4% | — | — / — / — | 5.769 | 5.769 | 6.072 | 1 / 0 / 98 | 0.173 | 495.171× |
| KITTI object training (local 2D panel) | SAM 3.1 | complete | 100/100 | 7.644 | native image dimensions; YOLO imgsz=640; batch=1 | 26.2% / 52.4% / 25.6% | — | 63.6% | — | — / — / — | 0.123 | 0.122 | 0.127 | — | 8.099 | 10.597× |
| KITTI object training (local 2D panel) | YOLO26-L | complete | 100/100 | 1.714 | native image dimensions; YOLO imgsz=640; batch=1 | 23.8% / 45.6% / 22.9% | — | 8.8% | — | — / — / — | 0.012 | 0.012 | 0.012 | — | 85.830 | 1.000× |
| KITTI object training (local 2D panel) | YOLO26-L-seg | complete | 100/100 | 1.779 | native image dimensions; YOLO imgsz=640; batch=1 | 24.5% / 46.8% / 22.8% | — | 9.8% | — | — / — / — | 0.014 | 0.014 | 0.015 | — | 70.229 | 1.222× |
| KITTI object training (local 2D panel) | YOLOE-26-L | complete | 100/100 | 1.518 | native image dimensions; YOLO imgsz=640; batch=1 | 20.9% / 43.2% / 20.0% | — | 15.6% | — | — / — / — | 0.030 | 0.031 | 0.035 | — | 33.104 | 2.593× |
| RF100-VL: 13 Lkc01 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.6% / 2.9% / 0.2% | — | 10.0% | — | — / — / — | 0.562 | 0.497 | 0.625 | 0 / 0 / 0 | 1.781 | 2.993× |
| RF100-VL: 13 Lkc01 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.4% / 2.8% / 0.0% | — | 10.0% | — | — / — / — | 6.366 | 6.164 | 6.771 | 0 / 0 / 20 | 0.157 | 33.923× |
| RF100-VL: 13 Lkc01 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.7% / 4.2% / 0.2% | — | 7.5% | — | — / — / — | 0.188 | 0.160 | 0.240 | — | 5.329 | 1.000× |
| RF100-VL: 2024 Frc | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 4.8% / 7.8% / 5.1% | — | 7.6% | — | — / — / — | 2.271 | 2.029 | 3.150 | 0 / 0 / 0 | 0.440 | 24.268× |
| RF100-VL: 2024 Frc | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 1.9% | — | — / — / — | 4.560 | 5.535 | 6.588 | 6 / 1 / 15 | 0.219 | 48.739× |
| RF100-VL: 2024 Frc | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.4% / 2.9% / 1.8% | — | 3.9% | — | — / — / — | 0.094 | 0.046 | 0.377 | — | 10.688 | 1.000× |
| RF100-VL: Actions | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 2.1% / 4.9% / 0.5% | — | 5.3% | — | — / — / — | 2.082 | 2.028 | 2.549 | 0 / 0 / 0 | 0.480 | 27.088× |
| RF100-VL: Actions | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 6.281 | 6.097 | 7.289 | 0 / 0 / 20 | 0.159 | 81.701× |
| RF100-VL: Actions | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 5.4% / 8.5% / 7.1% | — | 10.0% | — | — / — / — | 0.077 | 0.055 | 0.079 | — | 13.009 | 1.000× |
| RF100-VL: Activity diagrams | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.9% / 3.0% / 0.6% | — | 12.0% | — | — / — / — | 3.407 | 3.128 | 4.750 | 1 / 0 / 0 | 0.293 | 47.168× |
| RF100-VL: Activity diagrams | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.802 | 5.795 | 7.917 | 10 / 0 / 19 | 0.172 | 80.321× |
| RF100-VL: Activity diagrams | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.6% / 1.1% / 0.5% | — | 2.9% | — | — / — / — | 0.072 | 0.057 | 0.096 | — | 13.843 | 1.000× |
| RF100-VL: Aerial airport aircraft | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 12.0% / 35.8% / 4.1% | — | 55.1% | — | — / — / — | 0.929 | 0.748 | 2.075 | 0 / 0 / 0 | 1.077 | 121.234× |
| RF100-VL: Aerial airport aircraft | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 2.3% / 5.0% / 2.3% | — | 10.0% | — | — / — / — | 5.068 | 5.524 | 6.136 | 2 / 2 / 18 | 0.197 | 661.439× |
| RF100-VL: Aerial airport aircraft | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 9.8% / 20.3% / 8.8% | — | 16.2% | — | — / — / — | 0.054 | 0.034 | 0.066 | — | 18.464 | 7.068× |
| RF100-VL: Aerial Cows | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 16.9% / 42.0% / 11.6% | — | 61.9% | — | — / — / — | 1.136 | 1.137 | 1.792 | 0 / 0 / 0 | 0.880 | 13.060× |
| RF100-VL: Aerial Cows | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 1.2% / 3.2% / 1.2% | — | 7.5% | — | — / — / — | 5.538 | 6.123 | 6.278 | 3 / 2 / 18 | 0.181 | 63.642× |
| RF100-VL: Aerial Cows | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 5.1% / 14.8% / 1.1% | — | 27.5% | — | — / — / — | 0.087 | 0.072 | 0.105 | — | 11.493 | 1.000× |
| RF100-VL: Aerial Pool | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 9.2% / 19.4% / 6.0% | — | 38.3% | — | — / — / — | 2.234 | 1.877 | 4.286 | 0 / 0 / 0 | 0.448 | 30.999× |
| RF100-VL: Aerial Pool | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 7.5% | — | — / — / — | 6.760 | 6.531 | 9.345 | 3 / 0 / 20 | 0.148 | 93.823× |
| RF100-VL: Aerial Pool | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 3.8% / 8.5% / 2.7% | — | 7.3% | — | — / — / — | 0.072 | 0.056 | 0.103 | — | 13.879 | 1.000× |
| RF100-VL: Aerial Sheep | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 39.5% / 85.4% / 30.9% | — | 91.9% | — | — / — / — | 2.708 | 2.327 | 6.265 | 0 / 0 / 0 | 0.369 | 15.514× |
| RF100-VL: Aerial Sheep | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 1.4% / 2.7% / 1.3% | — | 4.4% | — | — / — / — | 6.071 | 6.276 | 6.873 | 1 / 1 / 19 | 0.165 | 34.780× |
| RF100-VL: Aerial Sheep | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 12.8% / 30.3% / 10.0% | — | 43.6% | — | — / — / — | 0.175 | 0.141 | 0.394 | — | 5.728 | 1.000× |
| RF100-VL: Aircraft Turnaround Dataset | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 10.5% / 16.9% / 11.2% | — | 41.0% | — | — / — / — | 1.320 | 1.290 | 1.704 | 0 / 0 / 0 | 0.758 | 20.138× |
| RF100-VL: Aircraft Turnaround Dataset | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 5.7% / 7.5% / 6.0% | — | 33.3% | — | — / — / — | 6.327 | 6.148 | 7.270 | 0 / 0 / 20 | 0.158 | 96.542× |
| RF100-VL: Aircraft Turnaround Dataset | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 12.9% / 20.7% / 14.3% | — | 6.1% | — | — / — / — | 0.066 | 0.051 | 0.082 | — | 15.258 | 1.000× |
| RF100-VL: All Elements | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 13.2% / 23.1% / 13.0% | — | 32.6% | — | — / — / — | 2.781 | 2.196 | 4.606 | 0 / 0 / 0 | 0.360 | 33.479× |
| RF100-VL: All Elements | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 7.2% | — | — / — / — | 5.318 | 6.238 | 7.507 | 6 / 2 / 16 | 0.188 | 64.021× |
| RF100-VL: All Elements | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 2.2% / 3.1% / 2.6% | — | 4.4% | — | — / — / — | 0.083 | 0.039 | 0.358 | — | 12.038 | 1.000× |
| RF100-VL: Apoce Aerial Photographs For Object Detection Of Construction Equipment | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 4.7% / 8.9% / 4.1% | — | 17.9% | — | — / — / — | 1.513 | 1.438 | 1.915 | 0 / 0 / 0 | 0.661 | 15.316× |
| RF100-VL: Apoce Aerial Photographs For Object Detection Of Construction Equipment | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 9.1% / 13.1% / 11.9% | — | 24.5% | — | — / — / — | 5.830 | 5.939 | 8.106 | 3 / 0 / 17 | 0.172 | 59.032× |
| RF100-VL: Apoce Aerial Photographs For Object Detection Of Construction Equipment | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 10.1% / 14.8% / 11.6% | — | 3.3% | — | — / — / — | 0.099 | 0.053 | 0.229 | — | 10.126 | 1.000× |
| RF100-VL: Aquarium Combined | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 14.1% / 26.5% / 11.5% | — | 44.4% | — | — / — / — | 2.129 | 1.938 | 3.339 | 0 / 0 / 0 | 0.470 | 22.874× |
| RF100-VL: Aquarium Combined | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 8.1% / 8.9% / 8.7% | — | 13.3% | — | — / — / — | 5.708 | 6.163 | 6.877 | 4 / 1 / 18 | 0.175 | 61.332× |
| RF100-VL: Aquarium Combined | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 24.9% / 41.8% / 25.5% | — | 9.1% | — | — / — / — | 0.093 | 0.080 | 0.133 | — | 10.745 | 1.000× |
| RF100-VL: Asphaltdistressdetection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.4% / 0.7% / 0.4% | — | 3.5% | — | — / — / — | 1.169 | 1.095 | 1.569 | 0 / 0 / 0 | 0.855 | 17.402× |
| RF100-VL: Asphaltdistressdetection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 4.472 | 5.486 | 5.724 | 10 / 3 / 16 | 0.224 | 66.572× |
| RF100-VL: Asphaltdistressdetection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.2% / 0.7% / 0.2% | — | 1.0% | — | — / — / — | 0.067 | 0.047 | 0.168 | — | 14.887 | 1.000× |
| RF100-VL: Ball | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | — / — / — | — | 17.0% | — | — / — / — | 3.494 | 3.534 | 3.740 | 0 / 0 / 0 | 0.286 | 46.127× |
| RF100-VL: Ball | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.9% / 1.0% / 1.0% | — | 3.1% | — | — / — / — | 5.112 | 6.335 | 6.900 | 20 / 0 / 15 | 0.196 | 67.484× |
| RF100-VL: Ball | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.4% / 0.6% / 0.4% | — | 2.2% | — | — / — / — | 0.076 | 0.060 | 0.088 | — | 13.200 | 1.000× |
| RF100-VL: Bees | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 4.7% / 14.4% / 1.7% | — | 34.6% | — | — / — / — | 0.588 | 0.417 | 1.364 | 0 / 0 / 0 | 1.700 | 9.349× |
| RF100-VL: Bees | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 11.1% / 21.8% / 10.1% | — | 35.0% | — | — / — / — | 2.384 | 0.194 | 6.792 | 13 / 13 / 7 | 0.420 | 37.885× |
| RF100-VL: Bees | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.3% / 2.9% / 0.2% | — | 2.6% | — | — / — / — | 0.063 | 0.047 | 0.077 | — | 15.893 | 1.000× |
| RF100-VL: Bibdetection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 22.2% / 56.6% / 8.8% | — | 72.0% | — | — / — / — | 0.475 | 0.450 | 0.583 | 0 / 0 / 0 | 2.104 | 6.606× |
| RF100-VL: Bibdetection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 6.2% / 24.0% / 1.0% | — | 47.6% | — | — / — / — | 6.586 | 6.296 | 8.100 | 0 / 0 / 20 | 0.152 | 91.527× |
| RF100-VL: Bibdetection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 14.5% / 58.7% / 7.0% | — | 28.8% | — | — / — / — | 0.072 | 0.054 | 0.102 | — | 13.897 | 1.000× |
| RF100-VL: Buoy Onboarding | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 6.8% / 17.8% / 3.4% | — | 33.0% | — | — / — / — | 1.728 | 1.687 | 2.159 | 0 / 0 / 0 | 0.579 | 34.548× |
| RF100-VL: Buoy Onboarding | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 15.8% | — | — / — / — | 0.968 | 0.403 | 5.444 | 19 / 8 / 2 | 1.033 | 19.349× |
| RF100-VL: Buoy Onboarding | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 23.1% / 50.8% / 21.6% | — | 29.1% | — | — / — / — | 0.050 | 0.035 | 0.058 | — | 19.989 | 1.000× |
| RF100-VL: Cable damage | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 3.3% / 12.2% / 1.4% | — | 25.7% | — | — / — / — | 0.514 | 0.405 | 0.722 | 0 / 0 / 0 | 1.947 | 9.827× |
| RF100-VL: Cable damage | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 2.9% / 6.4% / 1.2% | — | 15.4% | — | — / — / — | 2.773 | 0.266 | 6.270 | 9 / 8 / 9 | 0.361 | 53.044× |
| RF100-VL: Cable damage | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.7% / 0.0% | — | 3.6% | — | — / — / — | 0.052 | 0.035 | 0.060 | — | 19.128 | 1.000× |
| RF100-VL: Canalstenosis | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 1.125 | 0.972 | 1.848 | 0 / 0 / 0 | 0.889 | 26.147× |
| RF100-VL: Canalstenosis | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 0.0% | — | — / — / — | 5.471 | 5.452 | 5.660 | 0 / 0 / 20 | 0.183 | 127.123× |
| RF100-VL: Canalstenosis | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.043 | 0.027 | 0.048 | — | 23.237 | 1.000× |
| RF100-VL: Car Logo Detection | GroundAnything-VLM | complete | 17/17 | 71.045 | native model processor | 4.2% / 4.6% / 4.6% | — | 10.9% | — | — / — / — | 0.667 | 0.638 | 0.905 | 0 / 0 / 0 | 1.498 | 6.047× |
| RF100-VL: Car Logo Detection | LocateAnything-3B | complete | 17/17 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 4.660 | 6.244 | 7.805 | 1 / 1 / 12 | 0.215 | 42.225× |
| RF100-VL: Car Logo Detection | YOLOE-26-L | complete | 17/17 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.7% / 0.9% / 0.9% | — | 2.8% | — | — / — / — | 0.110 | 0.080 | 0.437 | — | 9.061 | 1.000× |
| RF100-VL: Circuit Voltages | GroundAnything-VLM | complete | 14/14 | 71.045 | native model processor | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.832 | 0.799 | 1.354 | 0 / 0 / 0 | 1.202 | 16.183× |
| RF100-VL: Circuit Voltages | LocateAnything-3B | complete | 14/14 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 2.344 | 0.265 | 6.519 | 9 / 8 / 5 | 0.427 | 45.582× |
| RF100-VL: Circuit Voltages | YOLOE-26-L | complete | 14/14 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.3% / 0.0% | — | 1.0% | — | — / — / — | 0.051 | 0.028 | 0.350 | — | 19.447 | 1.000× |
| RF100-VL: Clashroyalechardetector | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 1.7% / 3.1% / 1.4% | — | 1.0% | — | — / — / — | 6.561 | 6.459 | 6.854 | 0 / 0 / 0 | 0.152 | 122.685× |
| RF100-VL: Clashroyalechardetector | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 1.984 | 1.115 | 6.100 | 19 / 14 / 3 | 0.504 | 37.107× |
| RF100-VL: Clashroyalechardetector | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.0% / 0.0% | — | 0.2% | — | — / — / — | 0.053 | 0.036 | 0.062 | — | 18.700 | 1.000× |
| RF100-VL: Cod Mw Warzone | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 24.2% / 52.2% / 20.2% | — | 62.6% | — | — / — / — | 0.401 | 0.309 | 0.685 | 0 / 0 / 0 | 2.496 | 8.409× |
| RF100-VL: Cod Mw Warzone | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 17.3% / 26.6% / 20.2% | — | 45.5% | — | — / — / — | 5.806 | 5.474 | 6.632 | 0 / 0 / 20 | 0.172 | 121.881× |
| RF100-VL: Cod Mw Warzone | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 22.9% / 36.9% / 22.4% | — | 14.0% | — | — / — / — | 0.048 | 0.031 | 0.048 | — | 20.993 | 1.000× |
| RF100-VL: Conveyor T Shirts | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 4.9% / 11.2% / 3.1% | — | 28.3% | — | — / — / — | 0.354 | 0.277 | 0.658 | 0 / 0 / 0 | 2.825 | 8.946× |
| RF100-VL: Conveyor T Shirts | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.4% / 1.0% / 0.3% | — | 5.8% | — | — / — / — | 5.714 | 5.646 | 6.031 | 0 / 0 / 20 | 0.175 | 144.384× |
| RF100-VL: Conveyor T Shirts | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.3% / 0.1% | — | 3.3% | — | — / — / — | 0.040 | 0.016 | 0.047 | — | 25.270 | 1.000× |
| RF100-VL: Countingpills | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 64.1% / 90.4% / 80.1% | — | 91.5% | — | — / — / — | 0.847 | 0.851 | 1.391 | 0 / 0 / 0 | 1.181 | 14.818× |
| RF100-VL: Countingpills | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 10.3% / 14.2% / 14.2% | — | 25.3% | — | — / — / — | 6.184 | 6.205 | 6.617 | 0 / 0 / 20 | 0.162 | 108.187× |
| RF100-VL: Countingpills | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 40.7% / 65.7% / 45.1% | — | 64.3% | — | — / — / — | 0.057 | 0.038 | 0.098 | — | 17.496 | 1.000× |
| RF100-VL: Crystal Clean Brain Tumors Mri Dataset | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 17.6% / 33.4% / 16.9% | — | 47.1% | — | — / — / — | 0.711 | 0.699 | 0.771 | 0 / 0 / 0 | 1.407 | 17.783× |
| RF100-VL: Crystal Clean Brain Tumors Mri Dataset | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 19.4% / 34.5% / 15.9% | — | 54.3% | — | — / — / — | 1.314 | 0.250 | 5.623 | 5 / 0 / 4 | 0.761 | 32.886× |
| RF100-VL: Crystal Clean Brain Tumors Mri Dataset | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.8% / 1.1% / 0.9% | — | 2.3% | — | — / — / — | 0.040 | 0.023 | 0.049 | — | 25.021 | 1.000× |
| RF100-VL: Dataconvert | GroundAnything-VLM | complete | 19/19 | 71.045 | native model processor | 24.5% / 30.2% / 28.2% | — | 49.7% | — | — / — / — | 0.300 | 0.257 | 0.664 | 0 / 0 / 0 | 3.338 | 6.489× |
| RF100-VL: Dataconvert | LocateAnything-3B | complete | 19/19 | 12.532 | native processor; max image patches=4096 | 3.7% / 6.5% / 3.4% | — | 13.6% | — | — / — / — | 3.302 | 5.464 | 6.076 | 8 / 8 / 11 | 0.303 | 71.516× |
| RF100-VL: Dataconvert | YOLOE-26-L | complete | 19/19 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 42.5% / 55.7% / 48.4% | — | 49.1% | — | — / — / — | 0.046 | 0.030 | 0.358 | — | 21.659 | 1.000× |
| RF100-VL: Deepfruits | GroundAnything-VLM | complete | 18/18 | 71.045 | native model processor | 32.1% / 64.6% / 30.1% | — | 77.7% | — | — / — / — | 0.744 | 0.601 | 2.163 | 0 / 0 / 0 | 1.344 | 15.809× |
| RF100-VL: Deepfruits | LocateAnything-3B | complete | 18/18 | 12.532 | native processor; max image patches=4096 | 6.7% / 9.9% / 6.8% | — | 16.7% | — | — / — / — | 5.802 | 5.593 | 8.096 | 0 / 0 / 18 | 0.172 | 123.241× |
| RF100-VL: Deepfruits | YOLOE-26-L | complete | 18/18 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 35.5% / 59.7% / 35.5% | — | 61.6% | — | — / — / — | 0.047 | 0.026 | 0.362 | — | 21.243 | 1.000× |
| RF100-VL: Deeppcb | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 2.215 | 2.006 | 3.173 | 0 / 0 / 0 | 0.451 | 41.647× |
| RF100-VL: Deeppcb | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 4.454 | 5.650 | 6.345 | 17 / 4 / 15 | 0.225 | 83.729× |
| RF100-VL: Deeppcb | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.1% / 0.1% | — | 0.7% | — | — / — / — | 0.053 | 0.034 | 0.068 | — | 18.798 | 1.000× |
| RF100-VL: Defect Detection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 4.7% / 9.9% / 3.5% | — | 23.4% | — | — / — / — | 1.657 | 1.436 | 2.531 | 0 / 0 / 0 | 0.604 | 15.663× |
| RF100-VL: Defect Detection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 6.238 | 6.110 | 6.798 | 1 / 0 / 20 | 0.160 | 58.982× |
| RF100-VL: Defect Detection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.3% / 0.0% | — | 2.5% | — | — / — / — | 0.106 | 0.089 | 0.147 | — | 9.455 | 1.000× |
| RF100-VL: Dentalai | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.1% / 0.2% / 0.0% | — | 1.2% | — | — / — / — | 2.265 | 2.214 | 4.143 | 0 / 0 / 0 | 0.442 | 52.606× |
| RF100-VL: Dentalai | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.2% / 0.0% | — | 1.2% | — | — / — / — | 5.917 | 5.624 | 6.831 | 1 / 0 / 20 | 0.169 | 137.435× |
| RF100-VL: Dentalai | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.3% / 2.3% / 1.1% | — | 2.9% | — | — / — / — | 0.043 | 0.026 | 0.057 | — | 23.227 | 1.000× |
| RF100-VL: Electric Pylon Detection In Rsi | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 2.7% / 4.8% / 3.3% | — | 20.6% | — | — / — / — | 0.423 | 0.421 | 0.696 | 0 / 0 / 0 | 2.366 | 8.503× |
| RF100-VL: Electric Pylon Detection In Rsi | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.1% / 1.5% / 0.0% | — | 3.6% | — | — / — / — | 3.492 | 5.465 | 6.731 | 9 / 9 / 11 | 0.286 | 70.257× |
| RF100-VL: Electric Pylon Detection In Rsi | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.1% / 0.1% | — | 2.6% | — | — / — / — | 0.050 | 0.037 | 0.058 | — | 20.122 | 1.000× |
| RF100-VL: Everdaynew | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.2% / 0.3% / 0.2% | — | 2.1% | — | — / — / — | 3.133 | 2.541 | 6.739 | 0 / 0 / 0 | 0.319 | 48.108× |
| RF100-VL: Everdaynew | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 0.9% | — | — / — / — | 5.868 | 6.460 | 7.999 | 0 / 0 / 17 | 0.170 | 90.105× |
| RF100-VL: Everdaynew | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.6% / 1.0% / 0.7% | — | 2.6% | — | — / — / — | 0.065 | 0.049 | 0.132 | — | 15.355 | 1.000× |
| RF100-VL: Exploratorium Daphnia | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 3.9% / 7.0% / 3.5% | — | 7.3% | — | — / — / — | 3.405 | 2.977 | 5.313 | 0 / 0 / 0 | 0.294 | 58.296× |
| RF100-VL: Exploratorium Daphnia | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.6% / 0.7% / 0.7% | — | 0.7% | — | — / — / — | 2.955 | 0.619 | 6.627 | 20 / 15 / 8 | 0.338 | 50.596× |
| RF100-VL: Exploratorium Daphnia | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.9% / 1.4% / 1.2% | — | 1.0% | — | — / — / — | 0.058 | 0.042 | 0.119 | — | 17.120 | 1.000× |
| RF100-VL: Flir Camera Objects | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 19.3% / 35.1% / 17.8% | — | 61.2% | — | — / — / — | 0.957 | 0.993 | 1.422 | 0 / 0 / 0 | 1.045 | 17.914× |
| RF100-VL: Flir Camera Objects | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 14.6% / 23.0% / 20.6% | — | 22.2% | — | — / — / — | 5.589 | 5.512 | 6.034 | 0 / 0 / 20 | 0.179 | 104.638× |
| RF100-VL: Flir Camera Objects | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 29.7% / 54.1% / 24.7% | — | 14.9% | — | — / — / — | 0.053 | 0.040 | 0.068 | — | 18.721 | 1.000× |
| RF100-VL: Floating Waste | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 11.6% / 16.9% / 10.0% | — | 25.2% | — | — / — / — | 1.116 | 1.059 | 1.445 | 0 / 0 / 0 | 0.896 | 20.354× |
| RF100-VL: Floating Waste | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 11.7% / 15.2% / 11.4% | — | 36.1% | — | — / — / — | 4.319 | 5.926 | 6.334 | 7 / 6 / 14 | 0.232 | 78.786× |
| RF100-VL: Floating Waste | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 9.4% / 10.9% / 10.5% | — | 14.8% | — | — / — / — | 0.055 | 0.044 | 0.067 | — | 18.240 | 1.000× |
| RF100-VL: Football Player Detection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 35.8% / 70.1% / 30.3% | — | 87.6% | — | — / — / — | 1.937 | 1.716 | 2.533 | 0 / 0 / 0 | 0.516 | 41.887× |
| RF100-VL: Football Player Detection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 13.6% / 32.1% / 11.2% | — | 8.2% | — | — / — / — | 6.131 | 6.065 | 6.517 | 0 / 0 / 20 | 0.163 | 132.560× |
| RF100-VL: Football Player Detection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 37.0% / 63.2% / 37.6% | — | 76.6% | — | — / — / — | 0.046 | 0.031 | 0.056 | — | 21.622 | 1.000× |
| RF100-VL: Fruitjes | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 12.5% / 13.7% / 12.9% | — | 16.3% | — | — / — / — | 1.892 | 1.605 | 1.710 | 0 / 0 / 0 | 0.529 | 29.844× |
| RF100-VL: Fruitjes | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 8.8% / 9.4% / 9.4% | — | 32.9% | — | — / — / — | 1.759 | 0.555 | 6.282 | 9 / 0 / 4 | 0.569 | 27.739× |
| RF100-VL: Fruitjes | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 8.8% / 11.6% / 8.5% | — | 4.0% | — | — / — / — | 0.063 | 0.047 | 0.068 | — | 15.773 | 1.000× |
| RF100-VL: Grapes 5 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 15.3% / 21.8% / 16.2% | — | 13.2% | — | — / — / — | 3.131 | 2.004 | 5.469 | 0 / 0 / 0 | 0.319 | 73.236× |
| RF100-VL: Grapes 5 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 3.6% / 4.0% / 4.0% | — | 12.2% | — | — / — / — | 5.907 | 5.704 | 6.307 | 0 / 0 / 20 | 0.169 | 138.165× |
| RF100-VL: Grapes 5 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 25.9% / 33.9% / 25.6% | — | 17.1% | — | — / — / — | 0.043 | 0.023 | 0.058 | — | 23.390 | 1.000× |
| RF100-VL: Grass Weeds | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 20.7% / 45.5% / 14.5% | — | 62.9% | — | — / — / — | 0.367 | 0.280 | 0.897 | 0 / 0 / 0 | 2.727 | 10.119× |
| RF100-VL: Grass Weeds | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.861 | 0.064 | 5.373 | 17 / 17 / 3 | 1.162 | 23.749× |
| RF100-VL: Grass Weeds | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 13.7% / 23.3% / 14.5% | — | 33.8% | — | — / — / — | 0.036 | 0.018 | 0.039 | — | 27.595 | 1.000× |
| RF100-VL:  Grccs | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.1% / 0.2% / 0.1% | — | 2.5% | — | — / — / — | 2.423 | 2.054 | 2.703 | 0 / 0 / 0 | 0.413 | 44.957× |
| RF100-VL:  Grccs | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 1.4% | — | — / — / — | 4.685 | 6.196 | 6.949 | 16 / 5 / 14 | 0.213 | 86.938× |
| RF100-VL:  Grccs | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 3.2% / 6.3% / 3.5% | — | 4.8% | — | — / — / — | 0.054 | 0.050 | 0.083 | — | 18.555 | 1.000× |
| RF100-VL: Gwhd2021 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 29.1% / 66.8% / 21.6% | — | 79.8% | — | — / — / — | 3.250 | 2.819 | 6.091 | 0 / 0 / 0 | 0.308 | 62.368× |
| RF100-VL: Gwhd2021 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.9% / 1.6% / 1.2% | — | 3.4% | — | — / — / — | 6.734 | 6.628 | 7.207 | 0 / 0 / 20 | 0.148 | 129.227× |
| RF100-VL: Gwhd2021 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 2.6% / 5.7% / 2.1% | — | 17.2% | — | — / — / — | 0.052 | 0.039 | 0.050 | — | 19.189 | 1.000× |
| RF100-VL: Halo Infinite Angel Videogame | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 15.5% / 33.0% / 13.6% | — | 47.1% | — | — / — / — | 0.665 | 0.633 | 0.771 | 0 / 0 / 0 | 1.503 | 19.464× |
| RF100-VL: Halo Infinite Angel Videogame | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 15.6% / 26.3% / 13.0% | — | 51.1% | — | — / — / — | 6.126 | 5.742 | 8.137 | 0 / 0 / 20 | 0.163 | 179.179× |
| RF100-VL: Halo Infinite Angel Videogame | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 7.7% / 10.7% / 7.1% | — | 7.2% | — | — / — / — | 0.034 | 0.015 | 0.038 | — | 29.251 | 1.000× |
| RF100-VL: Human Detection In Floods | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 17.8% / 45.1% / 11.4% | — | 42.3% | — | — / — / — | 2.695 | 3.416 | 3.758 | 0 / 0 / 0 | 0.371 | 28.036× |
| RF100-VL: Human Detection In Floods | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.2% / 0.5% / 0.0% | — | 1.8% | — | — / — / — | 5.987 | 6.243 | 6.842 | 1 / 1 / 19 | 0.167 | 62.283× |
| RF100-VL: Human Detection In Floods | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 20.8% / 48.2% / 12.8% | — | 21.9% | — | — / — / — | 0.096 | 0.073 | 0.208 | — | 10.403 | 1.000× |
| RF100-VL: Inbreast | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.6% / 2.3% / 0.0% | — | 14.0% | — | — / — / — | 0.463 | 0.409 | 0.438 | 0 / 0 / 0 | 2.162 | 8.257× |
| RF100-VL: Inbreast | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 6.031 | 5.992 | 6.253 | 0 / 0 / 20 | 0.166 | 107.681× |
| RF100-VL: Inbreast | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 5.4% / 12.3% / 6.6% | — | 10.8% | — | — / — / — | 0.056 | 0.036 | 0.062 | — | 17.854 | 1.000× |
| RF100-VL: Infraredimageofpowerequipment | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.1% / 0.3% / 0.0% | — | 1.8% | — | — / — / — | 1.963 | 1.718 | 3.185 | 0 / 0 / 0 | 0.509 | 49.775× |
| RF100-VL: Infraredimageofpowerequipment | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.135 | 5.519 | 6.242 | 3 / 2 / 18 | 0.195 | 130.212× |
| RF100-VL: Infraredimageofpowerequipment | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.3% / 0.0% | — | 1.0% | — | — / — / — | 0.039 | 0.019 | 0.049 | — | 25.357 | 1.000× |
| RF100-VL: Into The Vale | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 26.5% / 30.2% / 27.9% | — | 32.3% | — | — / — / — | 1.311 | 1.284 | 1.416 | 0 / 0 / 0 | 0.763 | 16.859× |
| RF100-VL: Into The Vale | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 26.7% / 30.0% / 29.5% | — | 30.8% | — | — / — / — | 4.891 | 6.111 | 6.902 | 8 / 1 / 15 | 0.204 | 62.881× |
| RF100-VL: Into The Vale | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 21.9% / 28.7% / 19.7% | — | 10.4% | — | — / — / — | 0.078 | 0.060 | 0.124 | — | 12.857 | 1.000× |
| RF100-VL: Invoice Processing | GroundAnything-VLM | complete | 12/12 | 71.045 | native model processor | 7.3% / 22.5% / 3.3% | — | 34.3% | — | — / — / — | 1.405 | 1.346 | 1.678 | 0 / 0 / 0 | 0.712 | 23.920× |
| RF100-VL: Invoice Processing | LocateAnything-3B | complete | 12/12 | 12.532 | native processor; max image patches=4096 | 0.7% / 2.3% / 0.0% | — | 5.6% | — | — / — / — | 5.945 | 5.852 | 6.835 | 0 / 0 / 12 | 0.168 | 101.221× |
| RF100-VL: Invoice Processing | YOLOE-26-L | complete | 12/12 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.2% / 0.4% / 0.1% | — | 1.7% | — | — / — / — | 0.059 | 0.030 | 0.392 | — | 17.025 | 1.000× |
| RF100-VL: Ism Band Packet Detection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 4.1% / 13.2% / 1.6% | — | 20.7% | — | — / — / — | 0.545 | 0.511 | 0.656 | 0 / 0 / 0 | 1.835 | 12.749× |
| RF100-VL: Ism Band Packet Detection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 0.0% | — | — / — / — | 5.106 | 5.291 | 5.716 | 20 / 1 / 19 | 0.196 | 119.413× |
| RF100-VL: Ism Band Packet Detection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.1% / 2.9% / 0.4% | — | 13.4% | — | — / — / — | 0.043 | 0.025 | 0.051 | — | 23.388 | 1.000× |
| RF100-VL: Jellyfish | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 2.4% / 13.1% / 0.2% | — | 30.6% | — | — / — / — | 0.835 | 0.784 | 1.261 | 0 / 0 / 0 | 1.198 | 16.496× |
| RF100-VL: Jellyfish | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.6% / 1.7% / 0.2% | — | 3.1% | — | — / — / — | 2.304 | 0.217 | 6.326 | 13 / 13 / 7 | 0.434 | 45.516× |
| RF100-VL: Jellyfish | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.3% / 7.0% / 0.1% | — | 15.3% | — | — / — / — | 0.051 | 0.034 | 0.062 | — | 19.754 | 1.000× |
| RF100-VL: L10Ul502 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.5% / 2.4% / 0.0% | — | 0.7% | — | — / — / — | 4.819 | 2.047 | 3.475 | 0 / 0 / 1 | 0.208 | 61.303× |
| RF100-VL: L10Ul502 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 0.0% | — | — / — / — | 5.860 | 6.028 | 6.576 | 2 / 0 / 19 | 0.171 | 74.546× |
| RF100-VL: L10Ul502 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.2% / 1.1% / 0.0% | — | 0.3% | — | — / — / — | 0.079 | 0.061 | 0.129 | — | 12.721 | 1.000× |
| RF100-VL: Label Printing Defect Version 2 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 8.0% / 8.0% / 8.0% | — | 10.6% | — | — / — / — | 0.477 | 0.369 | 1.078 | 0 / 0 / 0 | 2.097 | 8.940× |
| RF100-VL: Label Printing Defect Version 2 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 34.9% / 34.9% / 34.9% | — | 37.9% | — | — / — / — | 5.451 | 5.356 | 5.894 | 0 / 0 / 20 | 0.183 | 102.190× |
| RF100-VL: Label Printing Defect Version 2 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 45.5% / 46.6% / 46.6% | — | 4.6% | — | — / — / — | 0.053 | 0.038 | 0.069 | — | 18.746 | 1.000× |
| RF100-VL: Lacrosse Object Detection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 16.6% / 21.2% / 18.9% | — | 13.0% | — | — / — / — | 1.068 | 0.953 | 1.523 | 0 / 0 / 0 | 0.936 | 20.015× |
| RF100-VL: Lacrosse Object Detection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 3.9% / 4.6% / 4.1% | — | 5.7% | — | — / — / — | 6.824 | 6.619 | 8.091 | 0 / 0 / 20 | 0.147 | 127.843× |
| RF100-VL: Lacrosse Object Detection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 8.9% / 11.5% / 10.2% | — | 5.0% | — | — / — / — | 0.053 | 0.039 | 0.067 | — | 18.734 | 1.000× |
| RF100-VL: Liver Disease | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 1.2% / 2.1% / 1.0% | — | 2.0% | — | — / — / — | 2.005 | 1.279 | 4.584 | 0 / 0 / 0 | 0.499 | 45.022× |
| RF100-VL: Liver Disease | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 4.496 | 5.236 | 5.361 | 5 / 0 / 17 | 0.222 | 100.948× |
| RF100-VL: Liver Disease | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.1% / 0.1% | — | 1.0% | — | — / — / — | 0.045 | 0.028 | 0.062 | — | 22.454 | 1.000× |
| RF100-VL: Macro Segmentation | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 4.5% / 12.8% / 2.3% | — | 14.5% | — | — / — / — | 5.169 | 5.278 | 8.090 | 0 / 0 / 0 | 0.193 | 60.835× |
| RF100-VL: Macro Segmentation | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 3.4% | — | — / — / — | 5.126 | 6.311 | 7.222 | 13 / 0 / 15 | 0.195 | 60.330× |
| RF100-VL: Macro Segmentation | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 5.3% / 8.0% / 7.2% | — | 0.9% | — | — / — / — | 0.085 | 0.070 | 0.148 | — | 11.770 | 1.000× |
| RF100-VL: Mahjong | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.1% / 0.2% / 0.2% | — | 0.9% | — | — / — / — | 6.532 | 6.512 | 7.969 | 0 / 0 / 0 | 0.153 | 62.884× |
| RF100-VL: Mahjong | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 0.0% | — | — / — / — | 3.659 | 2.127 | 7.854 | 15 / 5 / 6 | 0.273 | 35.226× |
| RF100-VL: Mahjong | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.1% / 0.1% | — | 1.0% | — | — / — / — | 0.104 | 0.085 | 0.169 | — | 9.627 | 1.000× |
| RF100-VL: Marine sharks | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 15.6% / 26.8% / 10.3% | — | 22.4% | — | — / — / — | 1.210 | 0.993 | 2.220 | 0 / 0 / 0 | 0.826 | 36.245× |
| RF100-VL: Marine sharks | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 39.7% / 44.6% / 42.5% | — | 27.7% | — | — / — / — | 3.785 | 5.993 | 6.287 | 8 / 5 / 12 | 0.264 | 113.384× |
| RF100-VL: Marine sharks | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 11.4% / 13.6% / 13.0% | — | 9.2% | — | — / — / — | 0.081 | 0.062 | 0.101 | — | 12.418 | 2.412× |
| RF100-VL: Needle Base Tip Min Max | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 17.4% / 31.5% / 12.4% | — | 33.2% | — | — / — / — | 1.010 | 1.028 | 1.065 | 0 / 0 / 0 | 0.990 | 15.281× |
| RF100-VL: Needle Base Tip Min Max | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 36.5% | — | — / — / — | 5.359 | 6.115 | 6.738 | 1 / 0 / 17 | 0.187 | 81.053× |
| RF100-VL: Needle Base Tip Min Max | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 11.1% / 17.9% / 8.0% | — | 8.3% | — | — / — / — | 0.066 | 0.047 | 0.108 | — | 15.124 | 1.000× |
| RF100-VL: New Defects In Wood | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 12.3% / 23.0% / 12.2% | — | 27.9% | — | — / — / — | 1.229 | 1.170 | 1.581 | 0 / 0 / 0 | 0.813 | 21.107× |
| RF100-VL: New Defects In Wood | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 3.1% / 5.4% / 2.2% | — | 17.2% | — | — / — / — | 5.925 | 6.130 | 6.564 | 6 / 0 / 19 | 0.169 | 101.728× |
| RF100-VL: New Defects In Wood | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.7% / 3.7% / 1.1% | — | 4.6% | — | — / — / — | 0.058 | 0.041 | 0.114 | — | 17.169 | 1.000× |
| RF100-VL: Nih Xray | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.0% / 0.2% / 0.0% | — | 1.0% | — | — / — / — | 1.798 | 1.730 | 2.295 | 0 / 0 / 0 | 0.556 | 39.295× |
| RF100-VL: Nih Xray | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.2% / 0.5% / 0.0% | — | 3.1% | — | — / — / — | 5.694 | 6.261 | 6.398 | 11 / 0 / 18 | 0.176 | 124.468× |
| RF100-VL: Nih Xray | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.1% / 0.0% | — | 0.3% | — | — / — / — | 0.046 | 0.033 | 0.055 | — | 21.860 | 1.000× |
| RF100-VL: Orgharvest | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 7.0% / 20.5% / 3.5% | — | 35.8% | — | — / — / — | 2.857 | 2.324 | 8.558 | 0 / 0 / 0 | 0.350 | 55.392× |
| RF100-VL: Orgharvest | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.8% / 1.2% / 1.2% | — | 1.0% | — | — / — / — | 6.015 | 6.006 | 6.647 | 0 / 0 / 20 | 0.166 | 116.606× |
| RF100-VL: Orgharvest | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.7% / 1.2% / 0.7% | — | 2.6% | — | — / — / — | 0.052 | 0.033 | 0.170 | — | 19.385 | 1.000× |
| RF100-VL: Orionproducts | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 5.1% / 8.2% / 5.5% | — | 15.8% | — | — / — / — | 2.704 | 1.992 | 6.359 | 0 / 0 / 0 | 0.370 | 62.614× |
| RF100-VL: Orionproducts | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.9% / 1.0% / 1.0% | — | 1.9% | — | — / — / — | 2.706 | 0.488 | 6.340 | 5 / 1 / 8 | 0.370 | 62.646× |
| RF100-VL: Orionproducts | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.3% / 0.5% / 0.3% | — | 1.4% | — | — / — / — | 0.043 | 0.031 | 0.056 | — | 23.155 | 1.000× |
| RF100-VL: Paper Parts | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 8.0% / 15.8% / 7.2% | — | 20.9% | — | — / — / — | 3.798 | 3.522 | 5.275 | 1 / 0 / 0 | 0.263 | 52.667× |
| RF100-VL: Paper Parts | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 16.4% / 23.8% / 14.1% | — | 25.8% | — | — / — / — | 4.661 | 6.102 | 6.885 | 7 / 1 / 14 | 0.215 | 64.623× |
| RF100-VL: Paper Parts | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.6% / 0.0% | — | 0.3% | — | — / — / — | 0.072 | 0.062 | 0.083 | — | 13.866 | 1.000× |
| RF100-VL: Peixos Fish | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 8.5% / 17.0% / 6.0% | — | 44.2% | — | — / — / — | 0.841 | 0.665 | 1.553 | 0 / 0 / 0 | 1.189 | 11.560× |
| RF100-VL: Peixos Fish | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 2.6% / 4.6% / 2.5% | — | 16.9% | — | — / — / — | 2.083 | 0.260 | 6.444 | 8 / 3 / 6 | 0.480 | 28.643× |
| RF100-VL: Peixos Fish | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 2.3% / 3.7% / 2.1% | — | 8.4% | — | — / — / — | 0.073 | 0.059 | 0.113 | — | 13.749 | 1.000× |
| RF100-VL: Penguin Finder Seg | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 45.5% / 58.3% / 52.8% | — | 74.7% | — | — / — / — | 0.590 | 0.490 | 1.203 | 0 / 0 / 0 | 1.695 | 11.123× |
| RF100-VL: Penguin Finder Seg | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 15.4% / 16.5% / 16.5% | — | 30.4% | — | — / — / — | 6.453 | 6.259 | 7.730 | 0 / 0 / 20 | 0.155 | 121.670× |
| RF100-VL: Penguin Finder Seg | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 65.1% / 81.3% / 69.3% | — | 31.9% | — | — / — / — | 0.053 | 0.039 | 0.070 | — | 18.855 | 1.000× |
| RF100-VL: Pig Detection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 6.5% / 12.9% / 5.8% | — | 24.7% | — | — / — / — | 0.977 | 0.953 | 1.035 | 0 / 0 / 0 | 1.023 | 15.198× |
| RF100-VL: Pig Detection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 3.9% / 6.4% / 3.7% | — | 13.1% | — | — / — / — | 6.081 | 6.646 | 11.367 | 5 / 0 / 16 | 0.164 | 94.563× |
| RF100-VL: Pig Detection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.7% / 3.3% / 1.5% | — | 7.5% | — | — / — / — | 0.064 | 0.046 | 0.065 | — | 15.551 | 1.000× |
| RF100-VL: Pill | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 8.3% / 12.5% / 10.0% | — | 17.2% | — | — / — / — | 2.153 | 1.977 | 3.752 | 0 / 0 / 0 | 0.464 | 41.843× |
| RF100-VL: Pill | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 6.0% | — | — / — / — | 5.195 | 6.542 | 8.236 | 1 / 1 / 14 | 0.192 | 100.975× |
| RF100-VL: Pill | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 8.1% / 14.3% / 8.0% | — | 21.4% | — | — / — / — | 0.051 | 0.035 | 0.067 | — | 19.435 | 1.000× |
| RF100-VL: Recode Waste | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 21.7% / 25.4% / 22.7% | — | 53.6% | — | — / — / — | 2.880 | 2.718 | 4.055 | 0 / 0 / 0 | 0.347 | 51.738× |
| RF100-VL: Recode Waste | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 2.5% | — | — / — / — | 6.636 | 6.645 | 7.554 | 0 / 0 / 20 | 0.151 | 119.198× |
| RF100-VL: Recode Waste | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 17.2% / 20.4% / 18.2% | — | 24.8% | — | — / — / — | 0.056 | 0.040 | 0.063 | — | 17.961 | 1.000× |
| RF100-VL: Roboflow Trained Dataset | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 17.8% / 29.7% / 22.3% | — | 39.0% | — | — / — / — | 0.974 | 0.945 | 1.240 | 0 / 0 / 0 | 1.026 | 16.126× |
| RF100-VL: Roboflow Trained Dataset | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 33.4% / 52.8% / 37.6% | — | 66.7% | — | — / — / — | 2.931 | 0.314 | 6.476 | 5 / 3 / 9 | 0.341 | 48.515× |
| RF100-VL: Roboflow Trained Dataset | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 17.8% / 26.6% / 19.6% | — | 21.2% | — | — / — / — | 0.060 | 0.046 | 0.075 | — | 16.551 | 1.000× |
| RF100-VL: Screwdetectclassification | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.8% / 3.3% / 0.1% | — | 3.3% | — | — / — / — | 2.479 | 1.728 | 6.308 | 0 / 0 / 0 | 0.403 | 54.132× |
| RF100-VL: Screwdetectclassification | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.891 | 5.840 | 6.202 | 0 / 0 / 20 | 0.170 | 128.640× |
| RF100-VL: Screwdetectclassification | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.9% / 1.2% / 1.1% | — | 3.8% | — | — / — / — | 0.046 | 0.025 | 0.050 | — | 21.838 | 1.000× |
| RF100-VL: Sea Cucumbers New Tiles | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.273 | 0.262 | 0.332 | 0 / 0 / 0 | 3.661 | 4.934× |
| RF100-VL: Sea Cucumbers New Tiles | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 1.2% / 1.5% / 1.5% | — | 4.2% | — | — / — / — | 1.281 | 0.097 | 6.156 | 16 / 16 / 4 | 0.780 | 23.143× |
| RF100-VL: Sea Cucumbers New Tiles | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 12.0% / 17.1% / 15.9% | — | 18.6% | — | — / — / — | 0.055 | 0.037 | 0.066 | — | 18.060 | 1.000× |
| RF100-VL: Signatures | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 62.0% / 96.9% / 72.0% | — | 90.2% | — | — / — / — | 0.356 | 0.380 | 0.509 | 0 / 0 / 0 | 2.805 | 6.878× |
| RF100-VL: Signatures | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 39.6% / 61.3% / 41.2% | — | 73.2% | — | — / — / — | 5.570 | 5.918 | 6.490 | 2 / 1 / 19 | 0.180 | 107.481× |
| RF100-VL: Signatures | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 43.2% / 62.1% / 46.4% | — | 13.3% | — | — / — / — | 0.052 | 0.039 | 0.061 | — | 19.296 | 1.000× |
| RF100-VL: Smd Components | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 3.5% / 7.3% / 2.8% | — | 10.9% | — | — / — / — | 2.772 | 2.214 | 4.765 | 0 / 0 / 0 | 0.361 | 44.781× |
| RF100-VL: Smd Components | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 3.7% | — | — / — / — | 2.167 | 0.821 | 7.044 | 3 / 0 / 4 | 0.461 | 35.003× |
| RF100-VL: Smd Components | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.3% / 0.3% / 0.3% | — | 2.2% | — | — / — / — | 0.062 | 0.047 | 0.085 | — | 16.153 | 1.000× |
| RF100-VL: Soda Bottles | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 8.2% / 24.2% / 4.3% | — | 46.4% | — | — / — / — | 2.693 | 2.534 | 3.944 | 0 / 0 / 0 | 0.371 | 32.589× |
| RF100-VL: Soda Bottles | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 1.2% / 2.1% / 1.5% | — | 5.1% | — | — / — / — | 6.611 | 6.550 | 7.504 | 0 / 0 / 20 | 0.151 | 79.993× |
| RF100-VL: Soda Bottles | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 6.7% / 16.2% / 4.9% | — | 13.3% | — | — / — / — | 0.083 | 0.069 | 0.086 | — | 12.100 | 1.000× |
| RF100-VL: Speech Bubbles Detection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 5.1% / 11.1% / 4.9% | — | 22.1% | — | — / — / — | 2.389 | 2.314 | 3.239 | 0 / 0 / 0 | 0.419 | 39.457× |
| RF100-VL: Speech Bubbles Detection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.5% / 1.0% / 0.3% | — | 9.2% | — | — / — / — | 5.765 | 6.202 | 8.764 | 5 / 1 / 17 | 0.173 | 95.212× |
| RF100-VL: Speech Bubbles Detection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 8.7% / 13.5% / 11.6% | — | 8.2% | — | — / — / — | 0.061 | 0.035 | 0.215 | — | 16.516 | 1.000× |
| RF100-VL: Spinefrxnormalvindr | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 1.2% / 2.0% / 1.3% | — | 4.3% | — | — / — / — | 1.211 | 1.067 | 2.117 | 0 / 0 / 0 | 0.826 | 24.497× |
| RF100-VL: Spinefrxnormalvindr | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 6.174 | 6.058 | 6.807 | 7 / 0 / 20 | 0.162 | 124.939× |
| RF100-VL: Spinefrxnormalvindr | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.1% / 0.0% | — | 1.7% | — | — / — / — | 0.049 | 0.032 | 0.071 | — | 20.235 | 1.000× |
| RF100-VL: Sssod | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 5.6% / 9.8% / 6.0% | — | 26.2% | — | — / — / — | 0.317 | 0.308 | 0.347 | 0 / 0 / 0 | 3.158 | 10.165× |
| RF100-VL: Sssod | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 3.1% / 9.1% / 1.0% | — | 21.1% | — | — / — / — | 3.252 | 5.184 | 5.501 | 6 / 5 / 12 | 0.307 | 104.395× |
| RF100-VL: Sssod | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 3.4% / 8.8% / 2.8% | — | 10.5% | — | — / — / — | 0.031 | 0.010 | 0.047 | — | 32.100 | 1.000× |
| RF100-VL: Stomata Cells | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 2.6% / 5.2% / 2.1% | — | 18.3% | — | — / — / — | 0.479 | 0.452 | 0.671 | 0 / 0 / 0 | 2.087 | 11.823× |
| RF100-VL: Stomata Cells | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.038 | 5.519 | 5.948 | 2 / 1 / 18 | 0.199 | 124.329× |
| RF100-VL: Stomata Cells | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 5.1% / 9.5% / 4.4% | — | 21.9% | — | — / — / — | 0.041 | 0.023 | 0.048 | — | 24.680 | 1.000× |
| RF100-VL: TACO trash | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 8.5% / 10.8% / 8.8% | — | 6.2% | — | — / — / — | 5.039 | 4.266 | 7.235 | 0 / 0 / 0 | 0.198 | 66.567× |
| RF100-VL: TACO trash | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 15.3% / 15.8% / 15.8% | — | 11.9% | — | — / — / — | 2.819 | 0.919 | 6.640 | 13 / 8 / 7 | 0.355 | 37.240× |
| RF100-VL: TACO trash | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 9.7% / 10.8% / 9.3% | — | 4.0% | — | — / — / — | 0.076 | 0.055 | 0.106 | — | 13.210 | 1.000× |
| RF100-VL: The Dreidel Project | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 17.6% / 22.4% / 17.3% | — | 31.5% | — | — / — / — | 1.376 | 1.284 | 1.626 | 0 / 0 / 0 | 0.727 | 34.329× |
| RF100-VL: The Dreidel Project | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 22.5% | — | — / — / — | 4.339 | 5.995 | 6.681 | 7 / 3 / 13 | 0.230 | 108.234× |
| RF100-VL: The Dreidel Project | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 10.9% / 12.8% / 11.8% | — | 8.1% | — | — / — / — | 0.040 | 0.022 | 0.063 | — | 24.944 | 1.000× |
| RF100-VL: Thermal Cheetah | GroundAnything-VLM | complete | 13/13 | 71.045 | native model processor | 33.5% / 47.5% / 39.5% | — | 68.6% | — | — / — / — | 0.279 | 0.265 | 0.334 | 0 / 0 / 0 | 3.581 | 8.311× |
| RF100-VL: Thermal Cheetah | LocateAnything-3B | complete | 13/13 | 12.532 | native processor; max image patches=4096 | 5.9% / 5.9% / 5.9% | — | 10.0% | — | — / — / — | 0.515 | 0.095 | 5.545 | 12 / 12 / 1 | 1.943 | 15.320× |
| RF100-VL: Thermal Cheetah | YOLOE-26-L | complete | 13/13 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 1.0% / 1.6% / 0.5% | — | 10.5% | — | — / — / — | 0.034 | 0.009 | 0.311 | — | 29.765 | 1.000× |
| RF100-VL: Tomatoes 2 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 51.6% / 64.6% / 57.9% | — | 74.9% | — | — / — / — | 0.847 | 0.768 | 1.288 | 0 / 0 / 0 | 1.181 | 16.203× |
| RF100-VL: Tomatoes 2 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 22.5% / 25.5% / 24.4% | — | 41.7% | — | — / — / — | 5.089 | 6.176 | 7.620 | 0 / 0 / 15 | 0.197 | 97.365× |
| RF100-VL: Tomatoes 2 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 54.9% / 69.1% / 59.8% | — | 30.9% | — | — / — / — | 0.052 | 0.044 | 0.051 | — | 19.132 | 1.000× |
| RF100-VL: Trail Camera | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 26.2% / 50.2% / 23.0% | — | 57.8% | — | — / — / — | 0.508 | 0.371 | 0.782 | 0 / 0 / 0 | 1.969 | 7.755× |
| RF100-VL: Trail Camera | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 34.1% / 52.3% / 39.3% | — | 56.6% | — | — / — / — | 3.549 | 5.357 | 6.362 | 3 / 2 / 12 | 0.282 | 54.187× |
| RF100-VL: Trail Camera | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 32.3% / 53.2% / 32.0% | — | 31.0% | — | — / — / — | 0.065 | 0.038 | 0.176 | — | 15.268 | 1.000× |
| RF100-VL: Train | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.2% / 0.9% / 0.0% | — | 2.3% | — | — / — / — | 0.880 | 0.554 | 1.922 | 0 / 0 / 0 | 1.136 | 27.855× |
| RF100-VL: Train | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.212 | 5.486 | 6.336 | 1 / 0 / 18 | 0.192 | 164.986× |
| RF100-VL: Train | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.7% / 2.5% / 0.1% | — | 5.3% | — | — / — / — | 0.032 | 0.010 | 0.042 | — | 31.653 | 1.000× |
| RF100-VL: Truck Movement | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | — / — / — | — | 12.3% | — | — / — / — | 1.857 | 1.333 | 4.654 | 0 / 0 / 0 | 0.538 | 14.224× |
| RF100-VL: Truck Movement | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 0.0% | — | — / — / — | 6.322 | 6.289 | 6.423 | 0 / 0 / 20 | 0.158 | 48.420× |
| RF100-VL: Truck Movement | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 4.0% / 7.8% / 3.5% | — | 1.8% | — | — / — / — | 0.131 | 0.118 | 0.155 | — | 7.659 | 1.000× |
| RF100-VL: Tube | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 20.4% / 24.8% / 20.7% | — | 40.2% | — | — / — / — | 1.861 | 1.487 | 3.473 | 0 / 0 / 0 | 0.537 | 56.029× |
| RF100-VL: Tube | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 14.8% | — | — / — / — | 6.410 | 6.796 | 7.911 | 0 / 0 / 18 | 0.156 | 193.011× |
| RF100-VL: Tube | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 4.8% / 7.2% / 4.4% | — | 12.1% | — | — / — / — | 0.033 | 0.011 | 0.040 | — | 30.113 | 1.000× |
| RF100-VL: Uavdet Small | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 11.3% / 18.1% / 12.3% | — | 82.3% | — | — / — / — | 5.781 | 5.714 | 7.303 | 0 / 0 / 0 | 0.173 | 127.207× |
| RF100-VL: Uavdet Small | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 5.5% / 6.6% / 6.5% | — | 1.8% | — | — / — / — | 6.346 | 6.350 | 6.683 | 15 / 0 / 20 | 0.158 | 139.641× |
| RF100-VL: Uavdet Small | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 6.7% / 12.8% / 6.0% | — | 33.2% | — | — / — / — | 0.045 | 0.036 | 0.048 | — | 22.003 | 1.000× |
| RF100-VL: Ufba 425 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | — / — / — | — | 1.6% | — | — / — / — | 4.613 | 4.503 | 5.333 | 0 / 0 / 0 | 0.217 | 129.844× |
| RF100-VL: Ufba 425 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 1.2% | — | — / — / — | 1.940 | 1.045 | 6.450 | 12 / 7 / 3 | 0.515 | 54.608× |
| RF100-VL: Ufba 425 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.036 | 0.011 | 0.041 | — | 28.149 | 1.000× |
| RF100-VL: Underwater Objects | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 1.1% / 1.7% / 1.3% | — | 9.6% | — | — / — / — | 1.340 | 0.979 | 2.313 | 0 / 0 / 0 | 0.746 | 27.429× |
| RF100-VL: Underwater Objects | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.7% / 1.1% / 0.7% | — | 4.8% | — | — / — / — | 3.303 | 5.500 | 5.909 | 14 / 1 / 11 | 0.303 | 67.602× |
| RF100-VL: Underwater Objects | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 2.2% / 3.3% / 2.9% | — | 2.7% | — | — / — / — | 0.049 | 0.038 | 0.054 | — | 20.467 | 1.000× |
| RF100-VL: Urine Analysis1 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 9.7% / 19.0% / 9.2% | — | 28.5% | — | — / — / — | 4.533 | 1.858 | 10.269 | 0 / 0 / 0 | 0.221 | 146.322× |
| RF100-VL: Urine Analysis1 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.853 | 0.227 | 6.097 | 20 / 6 / 2 | 1.172 | 27.531× |
| RF100-VL: Urine Analysis1 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.1% / 0.0% | — | 0.8% | — | — / — / — | 0.031 | 0.010 | 0.051 | — | 32.278 | 1.000× |
| RF100-VL: Varroa Mites Detection  Test Set | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 1.9% / 6.3% / 0.0% | — | 8.0% | — | — / — / — | 0.229 | 0.210 | 0.278 | 0 / 0 / 0 | 4.357 | 9.013× |
| RF100-VL: Varroa Mites Detection  Test Set | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.273 | 5.210 | 5.489 | 0 / 0 / 20 | 0.190 | 207.105× |
| RF100-VL: Varroa Mites Detection  Test Set | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.4% / 0.0% | — | 1.3% | — | — / — / — | 0.025 | 0.008 | 0.033 | — | 39.274 | 1.000× |
| RF100-VL: Water Meter | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 11.7% / 32.2% / 5.9% | — | 47.5% | — | — / — / — | 1.847 | 1.693 | 2.282 | 0 / 0 / 0 | 0.542 | 48.681× |
| RF100-VL: Water Meter | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 8.4% / 17.1% / 7.6% | — | 35.4% | — | — / — / — | 3.671 | 5.569 | 5.939 | 1 / 0 / 12 | 0.272 | 96.781× |
| RF100-VL: Water Meter | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 2.1% / 5.8% / 0.6% | — | 5.3% | — | — / — / — | 0.038 | 0.011 | 0.043 | — | 26.364 | 1.000× |
| RF100-VL: Wb Prova | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 33.8% / 47.4% / 40.5% | — | 56.4% | — | — / — / — | 0.786 | 0.756 | 0.893 | 0 / 0 / 0 | 1.273 | 14.428× |
| RF100-VL: Wb Prova | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 8.7% / 10.8% / 10.8% | — | 26.9% | — | — / — / — | 6.134 | 6.188 | 6.488 | 12 / 0 / 20 | 0.163 | 112.661× |
| RF100-VL: Wb Prova | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 18.0% / 32.9% / 17.2% | — | 32.6% | — | — / — / — | 0.054 | 0.033 | 0.124 | — | 18.368 | 1.000× |
| RF100-VL: Weeds4 | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 7.2% / 11.9% / 6.0% | — | 12.5% | — | — / — / — | 1.454 | 0.991 | 2.219 | 0 / 0 / 0 | 0.688 | 27.233× |
| RF100-VL: Weeds4 | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.4% / 1.2% / 0.0% | — | 2.9% | — | — / — / — | 3.711 | 5.398 | 6.130 | 12 / 5 / 13 | 0.269 | 69.492× |
| RF100-VL: Weeds4 | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 4.4% / 4.8% / 4.6% | — | 1.1% | — | — / — / — | 0.053 | 0.038 | 0.049 | — | 18.724 | 1.000× |
| RF100-VL: Wheel Defect Detection | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 16.2% / 19.2% / 15.3% | — | 24.1% | — | — / — / — | 3.602 | 0.788 | 1.151 | 1 / 1 / 1 | 0.278 | 63.914× |
| RF100-VL: Wheel Defect Detection | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 7.2% / 7.7% / 7.7% | — | 16.0% | — | — / — / — | 5.643 | 5.550 | 5.847 | 4 / 0 / 20 | 0.177 | 100.140× |
| RF100-VL: Wheel Defect Detection | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.056 | 0.040 | 0.089 | — | 17.746 | 1.000× |
| RF100-VL: Wildfire Smoke | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 2.8% / 7.7% / 1.0% | — | 22.7% | — | — / — / — | 0.228 | 0.214 | 0.285 | 0 / 0 / 0 | 4.385 | 5.559× |
| RF100-VL: Wildfire Smoke | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 3.9% / 10.9% / 0.0% | — | 13.3% | — | — / — / — | 3.010 | 2.776 | 6.062 | 10 / 10 / 10 | 0.332 | 73.371× |
| RF100-VL: Wildfire Smoke | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 7.6% / 12.8% / 7.9% | — | 6.1% | — | — / — / — | 0.041 | 0.025 | 0.046 | — | 24.378 | 1.000× |
| RF100-VL: Wine Labels | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 17.2% / 25.6% / 20.4% | — | 35.8% | — | — / — / — | 2.327 | 2.300 | 2.454 | 0 / 0 / 0 | 0.430 | 47.811× |
| RF100-VL: Wine Labels | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | — / — / — | — | 9.6% | — | — / — / — | 1.417 | 0.586 | 6.205 | 14 / 5 / 2 | 0.706 | 29.116× |
| RF100-VL: Wine Labels | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.8% / 1.6% / 0.4% | — | 1.0% | — | — / — / — | 0.049 | 0.035 | 0.040 | — | 20.543 | 1.000× |
| RF100-VL: X Ray Id | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.1% / 0.2% / 0.0% | — | 0.7% | — | — / — / — | 1.392 | 1.329 | 1.763 | 0 / 0 / 0 | 0.718 | 38.137× |
| RF100-VL: X Ray Id | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.767 | 5.557 | 6.545 | 2 / 0 / 20 | 0.173 | 157.966× |
| RF100-VL: X Ray Id | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.1% / 0.0% | — | 0.8% | — | — / — / — | 0.037 | 0.029 | 0.036 | — | 27.390 | 1.000× |
| RF100-VL: Xray | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.189 | 0.188 | 0.191 | 0 / 0 / 0 | 5.295 | 6.746× |
| RF100-VL: Xray | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.301 | 5.491 | 5.967 | 1 / 1 / 19 | 0.189 | 189.372× |
| RF100-VL: Xray | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 0.028 | 0.009 | 0.038 | — | 35.721 | 1.000× |
| RF100-VL: Zebrasatasturias | GroundAnything-VLM | complete | 20/20 | 71.045 | native model processor | 9.4% / 26.5% / 4.3% | — | 44.9% | — | — / — / — | 0.643 | 0.398 | 0.817 | 0 / 0 / 0 | 1.555 | 22.418× |
| RF100-VL: Zebrasatasturias | LocateAnything-3B | complete | 20/20 | 12.532 | native processor; max image patches=4096 | 0.0% / 0.2% / 0.0% | — | 1.3% | — | — / — / — | 1.801 | 0.122 | 5.859 | 15 / 14 / 6 | 0.555 | 62.773× |
| RF100-VL: Zebrasatasturias | YOLOE-26-L | complete | 20/20 | 7.798 | native image dimensions; YOLO imgsz=640; batch=1 | 0.2% / 0.4% / 0.4% | — | 1.8% | — | — / — / — | 0.029 | 0.010 | 0.036 | — | 34.863 | 1.000× |

#### RF100-VL macro comparison

Each macro metric averages reportable values from complete per-subset panels, each containing up to 20 shared images. Coverage is shown against all 100 local subsets; a metric that is unsupported for one panel omits that panel. Mean query time averages per-domain means equally and excludes model initialization. Latency ratios compare each model with the fastest complete YOLO detector or segmenter on the same panel. LocateAnything's all-subset run uses the 4,096-patch image cap and 2,048-token output cap; earlier native-input OOM attempts remain preserved. AP from panels with zero successful inference is unavailable.

| Model | Complete panels | AP panels | Macro box AP | F1 panels | Macro class-aware F1 | Mean query s | Mean model load s | Mean ratio to fastest YOLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GroundAnything-VLM | 100/100 | 97/100 | 11.8% | 100/100 | 29.4% | 1.793 | 71.045 | 33.068× |
| LocateAnything-3B | 100/100 | 80/100 | 6.5% | 100/100 | 13.0% | 4.661 | 12.532 | 91.998× |
| YOLOE-26-L | 100/100 | 100/100 | 9.0% | 100/100 | 11.2% | 0.060 | 7.798 | 1.075× |
For LVIS official AP, both grounding specialists receive a uniform confidence score of 1.0 because neither emits per-box confidence; YOLOE and SAM use their model scores. Treat specialist AP comparisons as quality references rather than confidence-calibrated rankings.
Model-load warning for SAM 3.1 on COCO val2017: SAM 3.1 model builder reported: loaded /mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--facebook--sam3.1/blobs/0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6 and found missing and/or unexpected keys: missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight', 'backbone.vision_backbone.convs.3.conv_1x1.bias', 'backbone.vision_backbone.convs.3.conv_3x3.weight', 'backbone.vision_backbone.convs.3.conv_3x3.bias']
Model-load warning for SAM 3.1 on LVIS v1 validation (30-category stratified panel): SAM 3.1 model builder reported four missing checkpoint parameters: backbone.vision_backbone.convs.3.conv_1x1.weight, backbone.vision_backbone.convs.3.conv_1x1.bias, backbone.vision_backbone.convs.3.conv_3x3.weight, backbone.vision_backbone.convs.3.conv_3x3.bias.
Model-load warning for SAM 3.1 on KITTI object training (local 2D panel): SAM 3.1 model builder reported: loaded /mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--facebook--sam3.1/blobs/0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6 and found missing and/or unexpected keys: missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight', 'backbone.vision_backbone.convs.3.conv_1x1.bias', 'backbone.vision_backbone.convs.3.conv_3x3.weight', 'backbone.vision_backbone.convs.3.conv_3x3.bias']

### DOTA v1.0 result

Status: complete. YOLO26-L-OBB validation AP: 33.8%; AP50: 61.8%; inference latency: 8.627 ms/image; whole-device peak VRAM: 2.351 GiB.
This is an in-domain OBB reference, not a zero-shot YOLOE comparison. Validation speed excludes model load; peak whole-device VRAM includes model loading and host baseline use.

## Method

The two pinned model snapshots use their own processors, chat templates, and task prompts. Sampling uses seed 3407; both models receive the same fixed task-specific manifest rows. Requests run serially, with one image per call and deterministic decoding. LocateAnything uses hybrid generation, BF16, and up to 8,192 output tokens. GroundAnything uses its model-specific BF16/eager SGLang release profile and up to 4,096 output tokens.

RefCOCOg reports mean IoU and Acc@IoU 0.5. COCO reports COCOeval AP on the selected image IDs when local COCO annotations and category mappings are available, plus local micro F1 at IoU 0.5. COCO predictions use a uniform confidence score of 1.0 because these models do not emit box confidence. For category-list prompts, a labeled None is parser-valid only when that category has no ground-truth box in the image. GUI grounding reports point-in-target accuracy and box Acc@IoU 0.5. HierText reports corpus-level one-to-one detection precision, recall, and F1 at IoU 0.5. Empty, malformed, truncated, and duplicate outputs are counted. Whole-device VRAM includes the desktop baseline.

## Data availability and limitations

Prepared manifests exist for refcocog, coco, screenspot_pro, hiertext, vqa_count, vqa_truth, vqa_description, textvqa. Core task manifests without local preparation include none. Dataset images and item-level annotations were not copied into the site. 2 model(s) have at least one completed example; partial and failed coverage remains visible in the scorecards.
The intended protocol uses RefCOCOg test, COCO val2017, ScreenSpot-Pro test, and HierText validation. A dataset split is considered run-ready only when its local manifest and assets are present. ScreenSpot-Pro was obtained from its official public repository and is stored at /mnt/HDD1/Project_Data/public_datasets/screenspot_pro/; the sample uses 100 of 1,581 test annotations with seed 3407. One box extending one pixel above its screenshot was clipped to the visible image boundary before normalization. The RefCOCOg panel uses the lmms-lab formatted test derivative, which contains two captions for a circled region rather than the canonical direct referring-expression export. The first caption is used as the query phrase, both references are retained, and pixel xywh boxes are converted to normalized xyxy. Treat these results as an adapted caption-to-region protocol, not as directly comparable to standard RefCOCOg scores. Its local README provides citation information but no separate license statement; the original RefCOCOg and COCO terms apply. HierText uses validation, matching the planned protocol. VQAv2 uses a 25-question existing panel; TextVQA uses a deterministic sample of 100 validation questions. These specialists do not document general VQA as a native task, so their answer scores measure exploratory transfer. TextVQA is OCR-dependent VQA, while HierText provides the direct word-box and transcription score. Dataset images and item-level predictions remain local.

## Runtime and artifacts

Per-item results preserve the prompt, raw output, canonical predictions, parser state, error, query latency, token counts, image dimensions, and scores in `results/runs/`. LocateAnything model-load peak and per-task peaks are recorded separately. Model initialization is excluded from query timing; report its duration separately when captured. Existing LocateAnything results time only `model.generate()`, excluding local image reading, processor encoding, output decoding, and parsing. The updated runner records the full adapter query from local image reading and processor encoding through decoded and parsed output; it also saves pure generation latency separately for token-rate calculations. GroundAnything timing in the saved runs starts after local image reading, base64 encoding, and request serialization, then spans the HTTP response; YOLO timing includes single-image loading, preprocessing, prompt setup, and inference, with warmup images excluded. These model-specific scopes still differ, so latency ratios are indicative rather than fully matched end-to-end measurements. The combined GroundAnything run measured one 624 s service startup shared by its RefCOCOg, HierText, VQAv2, and TextVQA tasks; the repeated task-table value refers to that single load. GroundAnything load duration is recorded for the ScreenSpot-Pro run at approximately 10 s from service startup through API readiness; older preloaded-service COCO, DOTA, HOPE, and LVIS runs lack a captured startup interval, which remains unavailable. This folder is excluded from the site. The site includes aggregate metrics only. Whole-device VRAM is sampled every 0.5 seconds with `nvidia-smi`; the reported maximum includes all processes and desktop use. Package versions describe the client runner; the GroundAnything service inventory is saved from its `/models` endpoint, while the custom server's full package set is not exposed by that endpoint.

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
- [LVIS official API](https://github.com/lvis-dataset/lvis-api)
- [Roboflow 100-VL benchmark](https://github.com/roboflow/rf100-vl)
- [KITTI object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php)
- [BOP datasets and BOP-H3](https://bop.felk.cvut.cz/datasets/)
- [BOP challenge tracks](https://bop.felk.cvut.cz/challenges/)
