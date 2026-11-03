# Grounding specialist benchmark

This suite compares the pinned LocateAnything-3B and GroundAnything-VLM checkpoints on four localization tasks. It keeps localization results separate from the generic image-QA ranking.

## Current status

The model snapshots and all indexed weight shards are cached locally. The required task annotations and images were not found in the inspected Hugging Face dataset cache. This benchmark does not download datasets or model weights. The report page and runner are ready; scorecards remain empty until local data manifests and runs exist.

## Prepare local data

Acquire each dataset under its own terms, then convert the selected split into one JSONL source per task under `data/source/`. Keep all images outside Git. Each row must use normalized `[x1,y1,x2,y2]` coordinates in the range 0–1 and point to a local image path. Image paths can be absolute or relative to this benchmark folder.

`import_datasets.py` converts already available official exports into this schema. It reads the LocateAnything/Rex-Omni `RefCOCOg_test.jsonl` and `COCO.jsonl`, raw ScreenSpot-Pro annotations, the official HierText validation file, and the corresponding local images. It performs no downloads. The standard source layout is documented in the [LocateAnything evaluation guide](https://github.com/NVlabs/Eagle/blob/main/Embodied/evaluation/README.md), and ScreenSpot-Pro publishes its annotation files beside its screenshots.

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

For text detection, the local scorer uses one-to-one box matching at IoU 0.5. It is a local box-based comparison score, not the official polygon benchmark score. COCOeval uses the supplied official `instances_val2017.json` when `pycocotools` and category IDs are available; its AP is computed on the fixed sampled image IDs.

Create fixed manifests once and reuse them for both model runs:

```bash
python3 prepare_data.py --source-dir data/source --output-dir data/manifests --samples 100 --seed 3407
```

The script writes image paths and annotation records only. It does not copy image bytes. Keep generated manifests out of the report site and do not commit restricted item-level annotations.

## Run

Run one model at a time. For LocateAnything, the runner releases managed local engines before loading the pinned snapshot directly with Transformers.

```bash
python3 benchmark.py --models nvidia/LocateAnything-3B --tasks refcocog coco screenspot_pro hiertext --samples 100 --seed 3407 --coco-annotations /path/to/instances_val2017.json --output-dir results/runs
```

Before GroundAnything inference, release local engines, then start its dedicated server from the local GroundAnything source checkout using the documented VLM SGLang configuration. The VLM endpoint must be available at `http://127.0.0.1:8102/v1`; do not start the sibling diffusion service.

```bash
cd /path/to/GroundAnything
python3 run.py serve --config configs/release/vlm_sglang.yaml
python3 benchmark.py --models GroundingPI/GroundAnything-VLM --tasks refcocog coco screenspot_pro hiertext --samples 100 --seed 3407 --coco-annotations /path/to/instances_val2017.json --output-dir results/runs
```

Add `--resume` to continue a partial run. Run the models in separate invocations and stop the first model before starting the second. Every example is sent serially with one image per request. The runner writes local per-item raw outputs, parse states, scores, errors, latency, token counts, image dimensions, and whole-device peak VRAM.

Refresh the report and its aggregate-only page after a run:

```bash
python3 generate_report.py
```

The static report page, script, data, styles, and mark are all stored beside `index.html`. The page does not include images, prompts, annotations, or item-level predictions.

## Protocol

Both models use their own prompt templates and output parsers. LocateAnything uses its BF16 hybrid Transformers path and up to 8,192 output tokens. GroundAnything uses its dedicated BF16/eager SGLang serving profile and up to 4,096 tokens. The parser retains raw special-token output and converts the model-specific 0–1000 or 0–999 coordinate grids to normalized values.

RefCOCOg reports mean IoU and Acc@IoU 0.5. COCO reports sampled-image COCOeval AP and local micro F1 at IoU 0.5 where category labels map. ScreenSpot-Pro reports point-in-target accuracy and Acc@IoU 0.5. HierText reports local box-based detection precision, recall, and F1 at IoU 0.5. Task scores are not merged into VQA results. A local equal-task macro appears only when every task is complete.

Whole-device VRAM is sampled every 0.5 seconds with `nvidia-smi` and includes desktop baseline use. Generation latency includes inference inside the model or API request. Token throughput appears only when output token counts are available. Runs preserve model revision, backend, precision, generation mode, and available package versions.

## References

- [LocateAnything model and evaluation guide](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [LocateAnything evaluation data package](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData)
- [GroundAnything-VLM and evaluation tools](https://github.com/groundingpi/GroundAnything)
- [RefCOCO repository](https://github.com/lichengunc/refer)
- [COCO dataset](https://cocodataset.org/#download)
- [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981)
- [HierText dataset](https://github.com/google-research-datasets/hiertext)
