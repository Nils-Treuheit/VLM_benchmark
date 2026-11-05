# Grounding specialist benchmark

This suite compares pinned LocateAnything-3B and GroundAnything-VLM checkpoints on localization, OCR, VQA, and counting. It keeps those results separate from the generic image-QA ranking.

## Current status

The model snapshots and all indexed weight shards are cached locally. Local VQAv2 panels and a deterministic 100-question TextVQA sample can be prepared from existing assets. Grounding task assets require their own local exports. TextVQA measures OCR-dependent VQA rather than direct transcription; HierText supplies direct word detection and OCR scores. VQA and counting are exploratory transfer measurements because the specialist model cards do not document general VQA as a native prompt task.

Prepare the VQAv2 and TextVQA manifests from the existing local panels and dataset cache:

```bash
python3 grounding_Bench/prepare_aux_data.py --textvqa-samples 100 --seed 3407
```

This writes ignored manifests and sampled image copies under `grounding_Bench/data/`. It records the source hash and dataset terms. VQAv2 has 25 questions each for counting, yes/no, and open-answer; TextVQA has 100 deterministic validation examples. Scores include the ten-answer VQA consensus measure, with canonical exact match shown as a supplementary measure.

## Prepare local data

Acquire each dataset under its own terms, then convert the selected split into one JSONL source per task under `data/source/`. Keep all images outside Git. Each row must use normalized `[x1,y1,x2,y2]` coordinates in the range 0–1 and point to a local image path. Image paths can be absolute or relative to this benchmark folder.

`import_datasets.py` converts already available official exports into this schema. It reads the LocateAnything/Rex-Omni `RefCOCOg_test.jsonl` and `COCO.jsonl`, raw ScreenSpot-Pro annotations, the official HierText validation file, and the corresponding local images. It performs no downloads. The standard source layout is documented in the [LocateAnything evaluation guide](https://github.com/NVlabs/Eagle/blob/main/Embodied/evaluation/README.md), and ScreenSpot-Pro publishes its annotation files beside its screenshots.

For the local lmms-lab parquet copy, prepare the fixed RefCOCOg panel directly from the embedded test images:

```bash
python3 grounding_Bench/prepare_refcocog_lmms.py \
  --dataset-dir /mnt/HDD1/Project_Data/public_datasets/refcocog_dataset
```

This reads the full 5,023-row test split, samples 100 rows deterministically, and extracts only those images under `/mnt/HDD1/Project_Data/public_datasets/derived/`. It uses the first of two caption answers as the query phrase, preserves both references, and converts the pixel `xywh` box to normalized `xyxy`. This is the lmms-lab formatted RefCOCOg test derivative, not the canonical direct referring-expression export. Report its scores as an adapted caption-to-region protocol, not as directly comparable to standard RefCOCOg results. The local dataset README provides the RefCOCOg citation but does not specify a separate license; the original RefCOCOg and COCO terms still apply.

For only the direct OCR task, acquire the [official HierText validation annotations and images](https://github.com/google-research-datasets/hiertext#dataset-downloading--processing) under the repository and image-source terms. The image archive extracts to `data/hiertext/validation/validation/`; place the official `validation.jsonl.gz` at `data/hiertext/gt/validation.jsonl.gz`. Then create the word-level sample:

```bash
python3 grounding_Bench/prepare_hiertext.py --samples 100 --seed 3407
```

The manifest uses the official 1,724-image validation split, but scores an explicitly seeded 100-image subset. HierText annotations are CC BY-SA 4.0; the Open Images-derived image terms remain separate and must be reviewed for local use.

Write a local license/access review file after checking each dataset's terms:

```json
{
  "refcocog": "reviewed terms and intended evaluation use",
  "coco": "reviewed terms and intended evaluation use",
  "screenspot_pro": "reviewed terms and intended evaluation use",
  "hiertext": "reviewed terms and intended evaluation use"
}
```

Then import the local exports. This example assumes the folders contain the official files; it does not create them:

```bash
python3 import_datasets.py \
  --licenses-file /path/to/checked-licenses.json \
  --refcocog-jsonl /path/to/EvalData/_annotations/box_eval/RefCOCOg_test.jsonl \
  --eval-data-root /path/to/EvalData \
  --coco-jsonl /path/to/EvalData/_annotations/box_eval/COCO.jsonl \
  --coco-image-root /path/to/EvalData \
  --coco-instances /path/to/EvalData/coco/instances_val2017.json \
  --screenspot-annotations /path/to/ScreenSpot-Pro/annotations \
  --screenspot-images /path/to/ScreenSpot-Pro/images \
  --hiertext-validation /path/to/HierText/gt/validation.jsonl.gz \
  --hiertext-images /path/to/HierText/validation
```

The importer uses word-level HierText regions and axis-aligned boxes. It uses image dimensions from the official annotations or local image headers, checks the COCO category mapping, and records the verified terms in each source sidecar.

Required row fields:

| File | Required fields | Notes |
|---|---|---|
| `refcocog.jsonl` | `id`, `image`, `phrase`, `target_box` | One test expression and its target box. |
| `coco.jsonl` | `id`, `image`, `image_id`, `categories`, `targets_by_category` | `categories` is a list of `{ "id": 1, "name": "person" }`; the target map uses category names as keys and normalized boxes as values. |
| `screenspot_pro.jsonl` | `id`, `image`, `instruction`, `target_box` | `target_point` is optional; the annotated box supplies the point-in-target criterion. |
| `hiertext.jsonl` | `id`, `image`, `target_boxes` | Convert each annotated text region to a normalized axis-aligned box. |

Add a `<task>.meta.json` sidecar for every source file. It must contain the expected dataset and split, plus a `license` field that records the terms checked for that local copy. Expected pairs are RefCOCOg/test, COCO/val2017, ScreenSpot-Pro/test, and HierText/validation. The manifest records these fields and a SHA-256 digest of the source annotation export.

For text detection and direct OCR, the local scorer matches predicted word boxes to HierText word boxes one-to-one at IoU 0.5. It reports word-box precision/recall/F1, case-insensitive exact transcription accuracy, and normalized edit similarity on matched words. This is a local axis-aligned-box metric, not the official polygon benchmark. COCOeval uses the supplied official `instances_val2017.json` when `pycocotools` and category IDs are available; its AP is computed on the fixed sampled image IDs.

Create fixed manifests once and reuse them for both model runs:

```bash
python3 prepare_data.py --source-dir data/source --output-dir data/manifests --samples 100 --seed 3407
```

The script writes image paths and annotation records only. It does not copy image bytes. Keep generated manifests out of the report site and do not commit restricted item-level annotations.

## Run

Run one model at a time. The default task list includes the four localization tasks and the prepared VQA/counting panels. If only the answer tasks are ready, specify `--tasks vqa_count vqa_truth vqa_description textvqa`. For LocateAnything, the runner releases managed local engines before loading the pinned snapshot directly with Transformers. Its local image processor defaults to 25,600 patches; pass `--max-image-tokens 4096` to cap high-resolution inputs when GPU memory is insufficient. The selected cap is saved with the run because it changes image detail and inference time.

```bash
python3 benchmark.py --models nvidia/LocateAnything-3B --tasks vqa_count vqa_truth vqa_description textvqa --seed 3407 --output-dir results/runs
```

Before GroundAnything inference, release local engines, then start its dedicated server from the local GroundAnything source checkout using the documented VLM SGLang configuration. The VLM endpoint must be available at `http://127.0.0.1:8102/v1`; do not start the sibling diffusion service.

```bash
cd /path/to/GroundAnything
python3 run.py serve --config configs/release/vlm_sglang.yaml
python3 benchmark.py --models GroundingPI/GroundAnything-VLM --tasks vqa_count vqa_truth vqa_description textvqa --seed 3407 --output-dir results/runs
```

Add `--resume` to continue a partial run. It preserves completed items and retries saved failures. Run the models in separate invocations and stop the first model before starting the second. Every example is sent serially with one image per request. The runner writes local per-item raw outputs, parse states, scores, errors, latency, token counts, image dimensions, and whole-device peak VRAM.

Refresh the report and its aggregate-only page after a run:

```bash
python3 generate_report.py
```

The static report page, script, data, styles, and mark are all stored beside `index.html`. The page does not include images, prompts, annotations, or item-level predictions.

## Protocol

Both models use their own prompt templates and output parsers. LocateAnything uses its BF16 hybrid Transformers path and up to 8,192 output tokens. GroundAnything uses its dedicated BF16/eager SGLang serving profile and up to 4,096 tokens. The parser retains raw special-token output and converts the model-specific 0–1000 or 0–999 coordinate grids to normalized values.

RefCOCOg reports mean IoU and Acc@IoU 0.5. COCO reports sampled-image COCOeval AP and local micro F1 at IoU 0.5 where category labels map. ScreenSpot-Pro reports point-in-target accuracy and Acc@IoU 0.5. HierText reports local box-based detection precision, recall, and F1 at IoU 0.5. Task scores are not merged into VQA results. A local equal-task macro appears only when every task is complete.

Whole-device VRAM is sampled every 0.5 seconds with `nvidia-smi` and includes desktop baseline use. Generation latency includes inference inside the model or API request. Token throughput appears only when output token counts are available. Runs preserve model revision, backend, precision, generation mode, and available package versions.

## COCO vision baselines

`vision_benchmark.py` uses the large open-vocabulary YOLOE-26-L segmentation checkpoint as the primary COCO baseline. It compares YOLO26-L closed-vocabulary detection and segmentation checkpoints, plus SAM 3.1. Every YOLO checkpoint is an `L` variant; weights must be supplied as local paths.

Run one checkpoint at a time. The runner releases managed local engines before inference and stops if release fails:

```bash
python3 grounding_Bench/vision_benchmark.py \
  --manifest grounding_Bench/data/manifests/coco.json \
  --coco-annotations /path/to/instances_val2017.json \
  --refcocog-manifest grounding_Bench/data/manifests/refcocog.json \
  --model yoloe26l-seg --weights yoloe26l-seg=/local/weights/yoloe-26l-seg.pt
```

Use `yolo26l` for the closed-vocabulary detector, `yolo26l-seg` for its instance segmenter, and `sam3.1` for the text-prompt segmenter. The runner accepts one model per invocation and writes local results to `results/vision_runs/`. Then regenerate the site with `python3 grounding_Bench/generate_report.py`.

Box and mask AP use COCOeval on the identical image IDs. The category prompt list is the complete COCO vocabulary for every image. YOLO26 uses its fixed COCO labels; YOLOE and SAM 3.1 receive all category names as text prompts. SAM runs one text query per category, so its latency includes all 80 prompts. YOLOE includes text-prompt setup and one image pass. All inference is batch one at 640 pixels for YOLO, and native input dimensions for SAM.

When the optional RefCOCOg manifest is supplied, YOLOE and SAM 3.1 also receive each referring phrase and report mean box IoU and Acc@IoU 0.5. Closed-set YOLO26 is marked unsupported for text grounding.

The classification score is label accuracy among predictions matched to any ground-truth object at box IoU 0.5. The report also shows joint detection-and-classification accuracy over all ground-truth objects, so missed detections count as errors. This is an object-label metric, not ImageNet top-1 image classification. Models without masks show segmentation as unsupported. Keep all these metrics separate; the report does not merge them into a single score.

The SAM 3.1 package and checkpoint have been used for local runs. Its loader reports four missing backbone convolution parameters; see the run artifacts and report. YOLO checkpoints are stored in `results/checkpoints/`, not the standard Ultralytics cache.

## Shared domain benchmarks

Prepare deterministic shared manifests from the local DOTA v1 and BOP-H3 HOPE v2 validation assets:

```bash
python3 grounding_Bench/prepare_domain_benchmarks.py
```

Each domain writes the same COCO-shaped manifest and annotation file for the vision and grounding-specialist runners. Run each model separately so only one model uses the GPU at a time:

```bash
python3 grounding_Bench/benchmark.py \
  --data-dir grounding_Bench/data/domain_manifests/dota_v1 \
  --tasks coco --coco-annotations grounding_Bench/data/domain_manifests/dota_v1/coco_annotations.json \
  --models nvidia/LocateAnything-3B \
  --output-dir grounding_Bench/results/domain_runs/dota_v1/specialists
python3 grounding_Bench/vision_benchmark.py \
  --manifest grounding_Bench/data/domain_manifests/dota_v1/coco.json \
  --coco-annotations grounding_Bench/data/domain_manifests/dota_v1/coco_annotations.json \
  --model yoloe26l-seg --weights yoloe26l-seg=/local/weights/yoloe-26l-seg.pt \
  --output-dir grounding_Bench/results/domain_runs/dota_v1/vision
python3 grounding_Bench/dota_obb_benchmark.py \
  --data grounding_Bench/data/domain_manifests/dota_v1/DOTAv1.yaml \
  --weights /path/to/yolo26l-obb.pt
python3 grounding_Bench/generate_report.py
```

Repeat both shared runners for `bop_h3_hope_v2` and each additional model. DOTA uses 100 of 458 validation images for the shared horizontal-box comparison. Its YOLO26-L-OBB result uses the full validation split and is an in-domain reference. DOTA's terms restrict use to academic purposes and prohibit commercial use; review its [official terms](https://captain-whu.github.io/DOTA/dataset).

HOPE v2 has 50 local validation images with object boxes and masks. Box AP and class-aware F1 are shared across YOLOE-26-L, SAM 3.1, and both grounding specialists. Mask AP is reported for models that return masks. This does not claim to reproduce the full BOP-H3 onboarding or 6D pose protocol.

Prepare the fixed KITTI image-only 2D panel from its labeled training split:

```bash
python3 grounding_Bench/prepare_kitti_benchmark.py --samples 100 --seed 3407
```

The local files now pair across all 7,481 image, label, calibration, and Velodyne entries. The manifest creates a deterministic 80/20 partition of labeled training images, then samples 100 validation images containing Car, Pedestrian, or Cyclist objects. It hashes the selected labels and records the full split. It excludes Van, Person_sitting, Truck, Tram, Misc, and DontCare. Run a local COCOeval comparison with:

```bash
python3 grounding_Bench/vision_benchmark.py \
  --manifest grounding_Bench/data/domain_manifests/kitti_training_2d/coco.json \
  --coco-annotations grounding_Bench/data/domain_manifests/kitti_training_2d/coco_annotations.json \
  --model yoloe26l-seg \
  --weights yoloe26l-seg=grounding_Bench/results/checkpoints/yoloe-26l-seg.pt \
  --output-dir grounding_Bench/results/domain_runs/kitti_training_2d/vision
```

This local COCOeval comparison is not official KITTI AP_R40. The official KITTI object devkit or a verified equivalent is still required for AP_R40. Public test labels are unavailable.

Prepare the five fixed 20-image RF100-VL test panels from local, MIT-licensed dataset exports:

```bash
python3 grounding_Bench/prepare_rf100_benchmark.py --samples 20 --seed 3407
```

This uses `aerial-airport`, `activity-diagrams`, `cable-damage`, `marine-sharks`, and `taco-trash-annotations-in-context`. It writes selected COCO annotations and shared image manifests under `data/domain_manifests/rf100_vl/`; image files stay in the public data tree. Run `vision_benchmark.py` and `benchmark.py` once per panel and model, using each panel's `coco.json` and `coco_annotations.json`. YOLOE-26-L and SAM 3.1 have complete results; LocateAnything has repeated high-resolution OOMs on two panels. RF100 exports contain no usable instance masks, so report mask AP as unsupported. The static report shows per-domain results and a macro over complete panels only.

LVIS has a completed exploratory 30-category panel; it reuses COCO imagery and does not represent the full 1,203-category leaderboard.

## OCR, VQA, and counting

The specialist runner uses short answer-only prompts for VQAv2 counting, yes/no, open-answer, and TextVQA. TextVQA evaluates reading text within natural images through answers. It does not provide a direct transcription metric. HierText provides direct word-box detection and word transcription metrics when its validation images and annotations are prepared through the standard importer above.

GroundAnything's released prompt templates document OCR and layout detection. LocateAnything documents text-region detection. Neither checkpoint documents generic VQA as a supported inference task. The VQAv2 and TextVQA scores therefore measure transfer under a common answer-only prompt, not native-task performance. VQAv2 panels reuse the same question IDs as the existing Qwen panel. Per-task latency, output rate, and whole-device VRAM are recorded alongside answer quality.

## References

- [LocateAnything model and evaluation guide](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [LocateAnything evaluation data package](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData)
- [GroundAnything-VLM and evaluation tools](https://github.com/groundingpi/GroundAnything)
- [RefCOCO repository](https://github.com/lichengunc/refer)
- [COCO dataset](https://cocodataset.org/#download)
- [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981)
- [HierText dataset](https://github.com/google-research-datasets/hiertext)
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics YOLOE-26](https://docs.ultralytics.com/models/yoloe)
- [Meta SAM 3.1 release and code](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [VQAv2 terms](https://visualqa.org/terms.html)
- [TextVQA dataset and terms](https://huggingface.co/datasets/facebook/textvqa)
