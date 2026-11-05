# Grounding Specialist Benchmark Report

**Evaluation status:** complete

**Generated:** 2026-11-05T13:21:21.507116+00:00

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
| `nvidia/LocateAnything-3B` | complete | 100/100 | AP 13.2%; AP50 15.2% | 12.218 / 13.234 / 14.219 / 25.778 | — | 292.430 tok/s | 11.439 GiB | 0 malformed, 56 invalid, 88 truncated, 10 empty, 0 duplicate |
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
| `GroundingPI/GroundAnything-VLM` | complete | 100/100 | Box F1 27.0%; OCR exact 70.7%; NED 89.9% | 6.005 / 5.280 / 14.812 / 23.768 | — | 68.730 tok/s | 24.282 GiB | 0 malformed, 0 invalid, 0 truncated, 0 empty, 0 duplicate |

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

YOLOE-26 and SAM 3.1 receive the same full COCO category-name list per image. SAM latency includes one text-prompt query per category; YOLOE latency includes prompt setup and one image pass. YOLO26 uses its fixed COCO vocabulary. COCO detection and instance-segmentation AP use the official evaluator. Optional RefCOCOg text grounding reports mean IoU and Acc@IoU 0.5 for YOLOE and SAM. Mean, median, p95, maximum query latency, and initial model or service load duration are reported separately. The GroundAnything load value for this run measures server process start through API readiness, including imports, checkpoint loading, and KV-cache allocation. Throughput is single-image serial inference; model load time is excluded from query latency. The grounding specialists do not provide confidence scores, so their COCO AP uses uniform scores and is less informative than class-aware F1 for direct comparison. No overall rank combines these metrics.

## Selection guide

Best complete model: `GroundingPI/GroundAnything-VLM` with local task macro 46.7%.
The local task macro weights the four task scores equally. RefCOCOg uses Acc@IoU 0.5, COCO uses sampled-image COCO AP when available and otherwise local micro F1@IoU 0.5, GUI uses point-in-target accuracy, and HierText uses detection F1. It is a local comparison metric.

| Task | Best complete model | Primary score |
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
| DOTA v1.0 validation | Shared HBB runs complete for YOLOE, YOLO26-L, SAM 3.1, and GroundAnything; LocateAnything native-input attempt partial (51/100), 4,096-patch recovery 100/100 but parser-invalid; full-split YOLO26-L-OBB complete | Shared aerial detection plus native OBB for YOLO26-L | 100 of 456 labeled validation images are sampled for horizontal-box COCO AP across YOLOE, SAM 3.1, and available grounding specialists. GroundAnything completed all 100 items with five outputs truncated at 2,048 tokens; official COCO AP was unavailable because three predicted labels did not map to annotation categories. LocateAnything's native-input 2,048-token run completed 51/100 before 49 CUDA OOM failures. A separate run caps the image processor at 4,096 patches and completes 100/100, but all 100 outputs include invalid records; COCOeval AP is unavailable. Keep the two LocateAnything settings separate. YOLO26-L-OBB uses the full 458-file validation split, including two background-only images, for the separate native OBB reference. Its clean rerun recorded 8.6 ms/image and 2.351 GiB peak whole-device VRAM. DOTA terms restrict use to academic purposes and prohibit commercial use. |
| BOP-H3 HOPE v2 validation | 50-image shared manifest; LocateAnything, GroundAnything, SAM 3.1, and YOLOE complete; closed-vocabulary YOLO26 omitted because its COCO label set does not cover HOPE classes | Object-centric 2D detection; instance masks where supported | All 50 local validation images are shared by YOLOE, SAM 3.1, and both grounding specialists. GroundAnything's official COCO AP is unavailable because one predicted label did not map to the annotation categories; its class-aware F1 remains reportable. Mask AP is shown only for models that return masks. This is the 2D HOPE v2 subset, not the full BOP-H3 pose/onboarding protocol. |
| KITTI object detection | Local inference complete for YOLOE, YOLO26, SAM 3.1, LocateAnything, and GroundAnything; official AP_R40 pending | Automotive 2D detection | The fixed 100-image panel is sampled from a deterministic 80/20 partition of labeled training images and uses Car, Pedestrian, and Cyclist boxes. It excludes Van, Person_sitting, Truck, Tram, Misc, and DontCare labels; its COCO-shaped annotations support local comparative AP/F1, not official KITTI AP_R40. LocateAnything used a 2,048-token response cap; 98 outputs reached the cap, and one unmapped label prevented official COCOeval AP. GroundAnything query latency excludes the approximately 10-second shared service startup. Public test labels are unavailable, and the official KITTI object devkit or a verified equivalent is still needed for AP_R40. |
| LVIS v1 validation | GroundAnything, LocateAnything, SAM 3.1, and YOLOE-26-L complete at 100/100; YOLO26-L and YOLO26-L-seg have no supported detections for the selected categories | Long-tail detection and instance segmentation | This exploratory panel uses 10 rare, 10 common, and 10 frequent categories across 100 fixed LVIS val images sourced from COCO 2017 train/val. It uses official LVIS AP and frequency-group metrics, not the full 1,203-category leaderboard; COCO-trained baselines may have seen some train images. YOLO26-L and YOLO26-L-seg were run on the same panel, but their fixed COCO vocabulary produced no detections for these selected long-tail categories, so LVIS AP is unsupported. |
| RF100-VL | YOLOE-26-L, SAM 3.1, GroundAnything, and LocateAnything complete on all five 20-image panels; YOLO26-L and YOLO26-L-seg complete on the airplane, boat, and person panels | Multi-domain open-vocabulary detection | The five local test subsets are aerial-airport, activity-diagrams, cable-damage, marine-sharks, and TACO trash. YOLOE-26-L, SAM 3.1, GroundAnything-VLM, and LocateAnything use the same seeded images on every panel. Closed-vocabulary YOLO26-L and YOLO26-L-seg were added on the aerial-airport and marine-sharks panels, where airplane, boat, and person labels overlap their COCO vocabulary; results are not extrapolated to panels without matching labels. The LocateAnything recovery caps image patches at 4,096 and output tokens at 2,048; it completed all five 20-image panels, while native-input attempts remain preserved with OOMs. Its detection quality remains low and parser-invalid outputs are shown in the detailed table. GroundAnything's shared SGLang service startup took about 10 seconds and is excluded from per-query latency. SAM checkpoint warnings are retained. |
| BOP-H3 HOT3D/HANDAL | separate follow-up | Unseen-object detection, onboarding, and 6D pose | Keep separate from class-name box AP. Use the official BOP protocol for pose tasks and review each dataset's terms before expanding beyond local HOPE v2. |

### Shared non-COCO domain runs

Every row uses the prepared image IDs and category list for its domain. Query latency excludes model load. YOLO timing includes image loading, preprocessing, and prompt setup; GroundAnything timing in the saved runs starts after local image reading, base64 encoding, and request serialization, then covers the HTTP call through the full response. Historical LocateAnything timings cover only model.generate(), excluding image reading, processor encoding, decoding, and parsing. The ratios compare each model with the fastest complete YOLO detector or segmenter on that domain, but the saved timing scopes differ.

| Dataset | Model | Status | Images | Model load s | Input size | Box AP / AP50 / AP75 | AR@300 | Class-aware F1 | Mask AP | AP rare / common / frequent | Mean query s | Median query s | p95 query s | Parse invalid / empty / truncated | Images/s | Latency / fastest YOLO |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| COCO val2017 | GroundAnything-VLM | complete | 100/100 | — | native model processor | 4.8% / 7.2% / 5.1% | — | 8.5% | — | — / — / — | 13.571 | 13.033 | 15.602 | 0 / 0 / 0 | 0.074 | 1314.390× |
| COCO val2017 | LocateAnything-3B | complete | 100/100 | — | native model processor | 13.2% / 15.2% / 14.1% | — | 24.1% | — | — / — / — | 12.218 | 13.234 | 14.219 | 56 / 10 / 88 | 0.082 | 1183.351× |
| COCO val2017 | YOLOE-26-L | complete | 100/100 | 1.994 | native image dimensions; YOLO imgsz=640; batch=1 | 55.2% / 71.3% / — | — | 10.2% | 34.1% | — / — / — | 0.026 | 0.022 | 0.038 | — | 38.093 | 2.543× |
| COCO val2017 | YOLO26-L | complete | 100/100 | — | native image dimensions; YOLO imgsz=640; batch=1 | 66.4% / 81.4% / — | — | 11.8% | — | — / — / — | 0.010 | 0.008 | 0.022 | — | 96.856 | 1.000× |
| COCO val2017 | YOLO26-L-seg | complete | 100/100 | — | native image dimensions; YOLO imgsz=640; batch=1 | 66.8% / 81.8% / — | — | 13.0% | 43.8% | — / — / — | 0.012 | 0.010 | 0.026 | — | 84.075 | 1.152× |
| COCO val2017 | SAM 3.1 | complete | 100/100 | 58.098 | native image dimensions; YOLO imgsz=640; batch=1 | 60.7% / 75.4% / — | — | 63.0% | 52.7% | — / — / — | 2.129 | 2.126 | 2.175 | — | 0.470 | 206.200× |
| DOTA v1.0 | GroundAnything-VLM | complete | 100/100 | — | native model processor | — / — / — | — | 1.3% | — | — / — / — | 8.070 | 5.325 | 25.576 | 0 / 0 / 5 | 0.124 | 108.038× |
| DOTA v1.0 | LocateAnything-3B | complete | 100/100 | 4.422 | native processor; max image patches=4096 | — / — / — | — | 0.1% | — | — / — / — | 4.563 | 6.427 | 7.384 | 100 / 31 / 62 | 0.219 | 61.089× |
| DOTA v1.0 | SAM 3.1 | complete | 100/100 | — | native image dimensions; YOLO imgsz=1024; batch=1 | 0.6% / 1.6% / 0.5% | — | 2.7% | — | — / — / — | 0.488 | 0.429 | 0.776 | — | 2.049 | 6.534× |
| DOTA v1.0 | YOLO26-L | complete | 100/100 | — | native image dimensions; YOLO imgsz=1024; batch=1 | 0.1% / 0.3% / 0.0% | — | 0.4% | — | — / — / — | 0.075 | 0.045 | 0.176 | — | 13.388 | 1.000× |
| DOTA v1.0 | YOLOE-26-L | complete | 100/100 | — | native image dimensions; YOLO imgsz=1024; batch=1 | 0.1% / 0.5% / 0.1% | — | 0.9% | — | — / — / — | 0.126 | 0.085 | 0.255 | — | 7.935 | 1.687× |
| BOP-H3 HOPE v2 | GroundAnything-VLM | complete | 50/50 | — | native model processor | — / — / — | — | 37.6% | — | — / — / — | 5.310 | 5.247 | 5.813 | 0 / 0 / 0 | 0.188 | 161.747× |
| BOP-H3 HOPE v2 | LocateAnything-3B | complete | 50/50 | — | native model processor | 13.0% / 17.0% / 13.4% | — | 30.5% | — | — / — / — | 5.572 | 5.985 | 6.820 | 50 / 0 / 40 | 0.179 | 169.728× |
| BOP-H3 HOPE v2 | SAM 3.1 | complete | 50/50 | — | native image dimensions; YOLO imgsz=640; batch=1 | 3.8% / 4.6% / 4.2% | — | 14.3% | 4.4% | — / — / — | 0.709 | 0.708 | 0.724 | — | 1.411 | 21.586× |
| BOP-H3 HOPE v2 | YOLOE-26-L | complete | 50/50 | — | native image dimensions; YOLO imgsz=640; batch=1 | 2.6% / 3.6% / 2.6% | — | 3.3% | 2.1% | — / — / — | 0.033 | 0.033 | 0.035 | — | 30.463 | 1.000× |
| LVIS v1 validation (30-category stratified panel) | GroundAnything-VLM | complete | 100/100 | — | native model processor | 39.4% / 51.5% / 42.2% | 49.4% | 7.4% | — | 50.1% / 40.6% / 27.4% | 6.246 | 5.578 | 9.083 | — | 0.160 | 549.151× |
| LVIS v1 validation (30-category stratified panel) | LocateAnything-3B | complete | 100/100 | 4.832 | native model processor | 12.5% / 13.5% / 13.1% | 12.7% | 8.2% | — | 19.0% / 10.0% / 8.5% | 3.696 | 5.656 | 6.290 | — | 0.271 | 324.935× |
| LVIS v1 validation (30-category stratified panel) | SAM 3.1 | complete | 100/100 | 20.575 | native image dimensions; YOLO imgsz=640; batch=1 | 36.4% / 40.2% / 38.5% | 37.0% | 31.0% | 32.3% | 57.9% / 27.0% / 24.4% | 0.813 | 0.812 | 0.824 | — | 1.230 | 71.473× |
| LVIS v1 validation (30-category stratified panel) | YOLO26-L | unsupported | 100/100 | 1.485 | native image dimensions; YOLO imgsz=640; batch=1 | — / — / — | — | 0.0% | — | — / — / — | 0.020 | 0.018 | 0.034 | — | 49.651 | —× |
| LVIS v1 validation (30-category stratified panel) | YOLO26-L-seg | unsupported | 100/100 | 2.291 | native image dimensions; YOLO imgsz=640; batch=1 | — / — / — | — | 0.0% | — | — / — / — | 0.022 | 0.020 | 0.041 | — | 45.161 | —× |
| LVIS v1 validation (30-category stratified panel) | YOLOE-26-L | complete | 100/100 | 1.842 | native image dimensions; YOLO imgsz=640; batch=1 | 43.3% / 53.6% / 45.8% | 50.6% | 8.4% | 30.7% | 56.9% / 38.3% / 34.6% | 0.011 | 0.010 | 0.026 | — | 87.921 | 1.000× |
| KITTI object training (local 2D panel) | GroundAnything-VLM | complete | 100/100 | 10.000 | native model processor | 11.9% / 25.0% / 11.3% | — | 56.2% | — | — / — / — | 1.156 | 1.104 | 1.848 | 0 / 0 / 0 | 0.865 | 99.213× |
| KITTI object training (local 2D panel) | LocateAnything-3B | complete | 100/100 | 4.610 | native model processor | — / — / — | — | 25.4% | — | — / — / — | 5.769 | 5.769 | 6.072 | 3 / 0 / 98 | 0.173 | 495.171× |
| KITTI object training (local 2D panel) | SAM 3.1 | complete | 100/100 | 7.644 | native image dimensions; YOLO imgsz=640; batch=1 | 26.2% / 52.4% / 25.6% | — | 63.6% | — | — / — / — | 0.123 | 0.122 | 0.127 | — | 8.099 | 10.597× |
| KITTI object training (local 2D panel) | YOLO26-L | complete | 100/100 | 1.714 | native image dimensions; YOLO imgsz=640; batch=1 | 23.8% / 45.6% / 22.9% | — | 8.8% | — | — / — / — | 0.012 | 0.012 | 0.012 | — | 85.830 | 1.000× |
| KITTI object training (local 2D panel) | YOLO26-L-seg | complete | 100/100 | 1.779 | native image dimensions; YOLO imgsz=640; batch=1 | 24.5% / 46.8% / 22.8% | — | 9.8% | — | — / — / — | 0.014 | 0.014 | 0.015 | — | 70.229 | 1.222× |
| KITTI object training (local 2D panel) | YOLOE-26-L | complete | 100/100 | 1.518 | native image dimensions; YOLO imgsz=640; batch=1 | 20.9% / 43.2% / 20.0% | — | 15.6% | — | — / — / — | 0.030 | 0.031 | 0.035 | — | 33.104 | 2.593× |
| RF100-VL: Activity diagrams | GroundAnything-VLM | complete | 20/20 | 10.000 | native model processor | 0.9% / 3.0% / 0.6% | — | 12.0% | — | — / — / — | 3.417 | 3.182 | 4.792 | 1 / 0 / 0 | 0.293 | 80.415× |
| RF100-VL: Activity diagrams | LocateAnything-3B | complete | 20/20 | 4.436 | native processor; max image patches=4096 | 0.0% / 0.0% / 0.0% | — | 0.0% | — | — / — / — | 5.755 | 5.859 | 7.288 | 11 / 0 / 19 | 0.174 | 135.414× |
| RF100-VL: Activity diagrams | SAM 3.1 | complete | 20/20 | 8.907 | native image dimensions; YOLO imgsz=640; batch=1 | 2.2% / 3.1% / 2.2% | — | 2.4% | — | — / — / — | 0.368 | 0.364 | 0.386 | — | 2.717 | 8.660× |
| RF100-VL: Activity diagrams | YOLOE-26-L | complete | 20/20 | 1.518 | native image dimensions; YOLO imgsz=640; batch=1 | 0.6% / 1.1% / 0.5% | — | 2.9% | — | — / — / — | 0.042 | 0.048 | 0.057 | — | 23.532 | 1.000× |
| RF100-VL: Aerial airport aircraft | GroundAnything-VLM | complete | 20/20 | 10.000 | native model processor | 12.0% / 35.8% / 4.1% | — | 55.1% | — | — / — / — | 0.940 | 0.754 | 2.085 | 0 / 0 / 0 | 1.064 | 122.669× |
| RF100-VL: Aerial airport aircraft | LocateAnything-3B | complete | 20/20 | 4.598 | native processor; max image patches=4096 | 2.5% / 5.0% / 2.4% | — | 10.0% | — | — / — / — | 5.285 | 5.758 | 6.308 | 2 / 2 / 18 | 0.189 | 689.831× |
| RF100-VL: Aerial airport aircraft | SAM 3.1 | complete | 20/20 | 8.993 | native image dimensions; YOLO imgsz=640; batch=1 | 43.8% / 85.5% / 41.5% | — | 84.0% | — | — / — / — | 0.089 | 0.084 | 0.107 | — | 11.224 | 11.628× |
| RF100-VL: Aerial airport aircraft | YOLO26-L | complete | 20/20 | 1.450 | native image dimensions; YOLO imgsz=640; batch=1 | 6.3% / 9.8% / 7.3% | — | 23.5% | — | — / — / — | 0.008 | 0.008 | 0.009 | — | 130.514 | 1.000× |
| RF100-VL: Aerial airport aircraft | YOLO26-L-seg | complete | 20/20 | 1.514 | native image dimensions; YOLO imgsz=640; batch=1 | 5.8% / 9.0% / 6.8% | — | 19.9% | — | — / — / — | 0.011 | 0.011 | 0.013 | — | 88.993 | 1.467× |
| RF100-VL: Aerial airport aircraft | YOLOE-26-L | complete | 20/20 | 1.554 | native image dimensions; YOLO imgsz=640; batch=1 | 9.8% / 20.3% / 8.8% | — | 16.2% | — | — / — / — | 0.040 | 0.042 | 0.054 | — | 24.814 | 5.260× |
| RF100-VL: Cable damage | GroundAnything-VLM | complete | 20/20 | 10.000 | native model processor | 3.3% / 12.2% / 1.4% | — | 25.7% | — | — / — / — | 0.497 | 0.413 | 0.725 | 0 / 0 / 0 | 2.013 | 13.412× |
| RF100-VL: Cable damage | LocateAnything-3B | complete | 20/20 | 4.753 | native processor; max image patches=4096 | 1.5% / 3.4% / 0.7% | — | 15.4% | — | — / — / — | 2.989 | 0.526 | 6.583 | 12 / 9 / 9 | 0.335 | 80.694× |
| RF100-VL: Cable damage | SAM 3.1 | complete | 20/20 | 9.385 | native image dimensions; YOLO imgsz=640; batch=1 | — / — / — | — | 0.0% | — | — / — / — | 0.121 | 0.119 | 0.144 | — | 8.295 | 3.254× |
| RF100-VL: Cable damage | YOLOE-26-L | complete | 20/20 | 1.494 | native image dimensions; YOLO imgsz=640; batch=1 | 0.1% / 0.7% / 0.0% | — | 3.6% | — | — / — / — | 0.037 | 0.039 | 0.050 | — | 26.995 | 1.000× |
| RF100-VL: Marine sharks | GroundAnything-VLM | complete | 20/20 | 10.000 | native model processor | 15.6% / 26.8% / 10.3% | — | 22.4% | — | — / — / — | 1.196 | 1.006 | 2.231 | 0 / 0 / 0 | 0.836 | 35.826× |
| RF100-VL: Marine sharks | LocateAnything-3B | complete | 20/20 | 4.650 | native processor; max image patches=4096 | 39.7% / 44.6% / 42.5% | — | 27.7% | — | — / — / — | 4.260 | 6.335 | 6.836 | 16 / 5 / 12 | 0.235 | 127.614× |
| RF100-VL: Marine sharks | SAM 3.1 | complete | 20/20 | 9.749 | native image dimensions; YOLO imgsz=640; batch=1 | 43.2% / 60.1% / 47.5% | — | 53.7% | — | — / — / — | 0.200 | 0.195 | 0.222 | — | 5.008 | 5.981× |
| RF100-VL: Marine sharks | YOLO26-L | complete | 20/20 | 1.473 | native image dimensions; YOLO imgsz=640; batch=1 | 32.4% / 38.0% / 35.0% | — | 9.1% | — | — / — / — | 0.033 | 0.027 | 0.078 | — | 29.955 | 1.000× |
| RF100-VL: Marine sharks | YOLO26-L-seg | complete | 20/20 | 1.481 | native image dimensions; YOLO imgsz=640; batch=1 | 31.7% / 36.5% / 34.5% | — | 8.7% | — | — / — / — | 0.034 | 0.023 | 0.074 | — | 29.780 | 1.006× |
| RF100-VL: Marine sharks | YOLOE-26-L | complete | 20/20 | 1.509 | native image dimensions; YOLO imgsz=640; batch=1 | 11.4% / 13.6% / 13.0% | — | 9.2% | — | — / — / — | 0.042 | 0.044 | 0.052 | — | 23.893 | 1.254× |
| RF100-VL: TACO trash | GroundAnything-VLM | complete | 20/20 | 10.000 | native model processor | 8.5% / 10.8% / 8.8% | — | 6.2% | — | — / — / — | 5.057 | 4.289 | 7.275 | 0 / 0 / 0 | 0.198 | 76.437× |
| RF100-VL: TACO trash | LocateAnything-3B | complete | 20/20 | 4.634 | native processor; max image patches=4096 | 19.5% / 21.1% / 21.1% | — | 14.0% | — | — / — / — | 3.657 | 1.201 | 7.991 | 18 / 8 / 8 | 0.273 | 55.279× |
| RF100-VL: TACO trash | SAM 3.1 | complete | 20/20 | 9.024 | native image dimensions; YOLO imgsz=640; batch=1 | 32.1% / 34.5% / 31.9% | — | 23.5% | — | — / — / — | 0.671 | 0.671 | 0.692 | — | 1.491 | 10.138× |
| RF100-VL: TACO trash | YOLOE-26-L | complete | 20/20 | 1.515 | native image dimensions; YOLO imgsz=640; batch=1 | 9.7% / 10.8% / 9.3% | — | 4.0% | — | — / — / — | 0.066 | 0.067 | 0.092 | — | 15.115 | 1.000× |

#### RF100-VL macro comparison

Each macro metric averages reportable values from complete 20/20 panels; a metric that is unsupported for one panel omits that panel. Mean query time averages per-domain means equally and excludes model initialization. Latency ratios compare each model with the fastest complete YOLO detector or segmenter on that same panel. LocateAnything's five-panel macro uses the uniform 4,096-patch image cap and 2,048-token output cap; earlier native-input OOM attempts remain preserved. AP from panels with zero successful inference is shown as unavailable.

| Model | Complete panels | AP panels | Macro box AP | F1 panels | Macro class-aware F1 | Mean query s | Mean model load s | Mean ratio to fastest YOLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GroundAnything-VLM | 5/5 | 5/5 | 8.0% | 5/5 | 24.3% | 2.221 | 10.000 | 65.752× |
| LocateAnything-3B | 5/5 | 5/5 | 12.6% | 5/5 | 13.4% | 4.389 | 4.614 | 217.766× |
| SAM 3.1 | 5/5 | 4/5 | 30.3% | 5/5 | 32.7% | 0.290 | 9.212 | 7.932× |
| YOLO26-L | 2/5 | 2/5 | 19.4% | 2/5 | 16.3% | 0.021 | 1.462 | 1.000× |
| YOLO26-L-seg | 2/5 | 2/5 | 18.8% | 2/5 | 14.3% | 0.022 | 1.498 | 1.236× |
| YOLOE-26-L | 5/5 | 5/5 | 6.3% | 5/5 | 7.2% | 0.046 | 1.518 | 1.903× |
For LVIS official AP, both grounding specialists receive a uniform confidence score of 1.0 because neither emits per-box confidence; YOLOE and SAM use their model scores. Treat specialist AP comparisons as quality references rather than confidence-calibrated rankings.
Model-load warning for SAM 3.1 on COCO val2017: SAM 3.1 model builder reported: loaded /mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--facebook--sam3.1/blobs/0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6 and found missing and/or unexpected keys: missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight', 'backbone.vision_backbone.convs.3.conv_1x1.bias', 'backbone.vision_backbone.convs.3.conv_3x3.weight', 'backbone.vision_backbone.convs.3.conv_3x3.bias']
Model-load warning for SAM 3.1 on LVIS v1 validation (30-category stratified panel): SAM 3.1 model builder reported four missing checkpoint parameters: backbone.vision_backbone.convs.3.conv_1x1.weight, backbone.vision_backbone.convs.3.conv_1x1.bias, backbone.vision_backbone.convs.3.conv_3x3.weight, backbone.vision_backbone.convs.3.conv_3x3.bias.
Model-load warning for SAM 3.1 on KITTI object training (local 2D panel): SAM 3.1 model builder reported: loaded /mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--facebook--sam3.1/blobs/0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6 and found missing and/or unexpected keys: missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight', 'backbone.vision_backbone.convs.3.conv_1x1.bias', 'backbone.vision_backbone.convs.3.conv_3x3.weight', 'backbone.vision_backbone.convs.3.conv_3x3.bias']
Model-load warning for SAM 3.1 on RF100-VL: Activity diagrams: SAM 3.1 model builder reported four missing checkpoint parameters: backbone.vision_backbone.convs.3.conv_1x1.weight, backbone.vision_backbone.convs.3.conv_1x1.bias, backbone.vision_backbone.convs.3.conv_3x3.weight, backbone.vision_backbone.convs.3.conv_3x3.bias.
Model-load warning for SAM 3.1 on RF100-VL: Aerial airport aircraft: SAM 3.1 model builder reported four missing checkpoint parameters: backbone.vision_backbone.convs.3.conv_1x1.weight, backbone.vision_backbone.convs.3.conv_1x1.bias, backbone.vision_backbone.convs.3.conv_3x3.weight, backbone.vision_backbone.convs.3.conv_3x3.bias.
Model-load warning for SAM 3.1 on RF100-VL: Cable damage: SAM 3.1 model builder reported four missing checkpoint parameters: backbone.vision_backbone.convs.3.conv_1x1.weight, backbone.vision_backbone.convs.3.conv_1x1.bias, backbone.vision_backbone.convs.3.conv_3x3.weight, backbone.vision_backbone.convs.3.conv_3x3.bias.
Model-load warning for SAM 3.1 on RF100-VL: Marine sharks: SAM 3.1 model builder reported four missing checkpoint parameters: backbone.vision_backbone.convs.3.conv_1x1.weight, backbone.vision_backbone.convs.3.conv_1x1.bias, backbone.vision_backbone.convs.3.conv_3x3.weight, backbone.vision_backbone.convs.3.conv_3x3.bias.
Model-load warning for SAM 3.1 on RF100-VL: TACO trash: SAM 3.1 model builder reported four missing checkpoint parameters: backbone.vision_backbone.convs.3.conv_1x1.weight, backbone.vision_backbone.convs.3.conv_1x1.bias, backbone.vision_backbone.convs.3.conv_3x3.weight, backbone.vision_backbone.convs.3.conv_3x3.bias.

### DOTA v1.0 result

Status: complete. YOLO26-L-OBB validation AP: 33.8%; AP50: 61.8%; inference latency: 8.627 ms/image; whole-device peak VRAM: 2.351 GiB.
This is an in-domain OBB reference, not a zero-shot YOLOE comparison. Validation speed excludes model load; peak whole-device VRAM includes model loading and host baseline use.

## Method

The two pinned model snapshots use their own processors, chat templates, and task prompts. Sampling uses seed 3407; both models receive the same fixed task-specific manifest rows. Requests run serially, with one image per call and deterministic decoding. LocateAnything uses hybrid generation, BF16, and up to 8,192 output tokens. GroundAnything uses its model-specific BF16/eager SGLang release profile and up to 4,096 output tokens.

RefCOCOg reports mean IoU and Acc@IoU 0.5. COCO reports COCOeval AP on the selected image IDs when local COCO annotations and category mappings are available, plus local micro F1 at IoU 0.5. COCO predictions use a uniform confidence score of 1.0 because these models do not emit box confidence. GUI grounding reports point-in-target accuracy and box Acc@IoU 0.5. HierText reports corpus-level one-to-one detection precision, recall, and F1 at IoU 0.5. Empty, malformed, truncated, and duplicate outputs are counted. Whole-device VRAM includes the desktop baseline.

## Data availability and limitations

Prepared manifests exist for refcocog, coco, screenspot_pro, hiertext, vqa_count, vqa_truth, vqa_description, textvqa. Core task manifests without local preparation include none. Dataset images and item-level annotations were not copied into the site. 2 model(s) have at least one completed example; partial and failed coverage remains visible in the scorecards.
The intended protocol uses RefCOCOg test, COCO val2017, ScreenSpot-Pro test, and HierText validation. A dataset split is considered run-ready only when its local manifest and assets are present. ScreenSpot-Pro was obtained from its official public repository and is stored at /mnt/HDD1/Project_Data/public_datasets/screenspot_pro/; the sample uses 100 of 1,581 test annotations with seed 3407. One box extending one pixel above its screenshot was clipped to the visible image boundary before normalization. The RefCOCOg panel uses the lmms-lab formatted test derivative, which contains two captions for a circled region rather than the canonical direct referring-expression export. The first caption is used as the query phrase, both references are retained, and pixel xywh boxes are converted to normalized xyxy. Treat these results as an adapted caption-to-region protocol, not as directly comparable to standard RefCOCOg scores. Its local README provides citation information but no separate license statement; the original RefCOCOg and COCO terms apply. HierText uses validation, matching the planned protocol. VQAv2 uses a 25-question existing panel; TextVQA uses a deterministic sample of 100 validation questions. These specialists do not document general VQA as a native task, so their answer scores measure exploratory transfer. TextVQA is OCR-dependent VQA, while HierText provides the direct word-box and transcription score. Dataset images and item-level predictions remain local.

## Runtime and artifacts

Per-item results preserve the prompt, raw output, canonical predictions, parser state, error, query latency, token counts, image dimensions, and scores in `results/runs/`. LocateAnything model-load peak and per-task peaks are recorded separately. Model initialization is excluded from query timing; report its duration separately when captured. Existing LocateAnything results time only `model.generate()`, excluding local image reading, processor encoding, output decoding, and parsing. The updated runner records the full adapter query from local image reading and processor encoding through decoded and parsed output; it also saves pure generation latency separately for token-rate calculations. GroundAnything timing in the saved runs starts after local image reading, base64 encoding, and request serialization, then spans the HTTP response; YOLO timing includes single-image loading, preprocessing, prompt setup, and inference, with warmup images excluded. These model-specific scopes still differ, so latency ratios are indicative rather than fully matched end-to-end measurements. GroundAnything load duration is recorded for the ScreenSpot-Pro run at approximately 10 s from service startup through API readiness; older service-preloaded runs without a captured startup interval leave this value unavailable. This folder is excluded from the site. The site includes aggregate metrics only. Whole-device VRAM is sampled every 0.5 seconds with `nvidia-smi`; the reported maximum includes all processes and desktop use. Package versions describe the client runner; the GroundAnything service inventory is saved from its `/models` endpoint, while the custom server's full package set is not exposed by that endpoint.

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
