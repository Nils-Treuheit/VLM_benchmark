#!/usr/bin/env python3
"""Score a saved LVIS panel run with the official LVIS evaluation API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_run(run_path: Path, manifest_path: Path, annotations_path: Path) -> dict[str, Any]:
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    from lvis import LVIS, LVISResults, LVISEval

    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = [int(value) for value in manifest["selected_image_ids"]]
    category_ids = [int(category["id"]) for category in manifest["examples"][0]["categories"]]
    lvis_gt = LVIS(str(annotations_path))
    model_id = str(run.get("model_id", ""))
    uniform_confidence = model_id in {"GroundingPI/GroundAnything-VLM", "nvidia/LocateAnything-3B"}
    source_items = run.get("items")
    specialist_items = False
    if source_items is None:
        task = run.get("tasks", {}).get("coco", {})
        source_items = task.get("items", [])
        specialist_items = True
    examples = {str(row["id"]): row for row in manifest["examples"]}
    items: dict[int, dict[str, Any]] = {}
    for item in source_items:
        image_id = item.get("image_id")
        if image_id is None and specialist_items:
            image_id = item.get("input_record", {}).get("image_id")
        if image_id is None:
            continue
        normalized = item
        if specialist_items:
            row = examples.get(str(item.get("id")), {})
            width, height = item.get("image_width"), item.get("image_height")
            aliases = {str(k).casefold(): str(v).casefold()
                       for k, v in row.get("prediction_aliases", {}).items()}
            category_by_name = {str(category["name"]).casefold(): int(category["id"])
                                for category in row.get("categories", [])}
            predictions = []
            if item.get("status") == "complete" and width and height:
                for record in item.get("parse", {}).get("records", []):
                    if record.get("kind") != "box" or not record.get("valid"):
                        continue
                    label = str(record.get("label") or "").casefold()
                    category_id = category_by_name.get(aliases.get(label, label))
                    box = record.get("box_xyxy_norm")
                    if category_id is None or not box or len(box) != 4:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in box)
                    predictions.append({"box": [x1 * width, y1 * height, x2 * width, y2 * height],
                                        "category_id": category_id, "score": 1.0})
            normalized = {"image_id": int(image_id), "predictions": predictions,
                          "classification": {}, "latency_s": item.get("latency_s"),
                          "status": item.get("status")}
        items[int(image_id)] = normalized
    if set(items) != set(expected_ids):
        raise ValueError(f"run image IDs differ from manifest: {len(items)} saved, {len(expected_ids)} expected")

    detections = []
    segmentations = []
    completed = 0
    for image_id in expected_ids:
        item = items[image_id]
        if item.get("status", "complete") != "complete" or "predictions" not in item or "classification" not in item:
            continue
        completed += 1
        for prediction in item.get("predictions", []):
            category_id = int(prediction["category_id"])
            if category_id not in category_ids:
                raise ValueError(f"prediction category {category_id} is outside the LVIS panel")
            x1, y1, x2, y2 = (float(value) for value in prediction["box"])
            score = 1.0 if uniform_confidence else float(prediction.get("score", 1.0))
            detection = {"image_id": image_id, "category_id": category_id,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": score}
            detections.append(detection)
            if prediction.get("segmentation"):
                segmentations.append({"image_id": image_id, "category_id": category_id,
                                      "segmentation": prediction["segmentation"],
                                      "score": score})

    output: dict[str, Any] = {
        "api": "lvis 0.5.3", "images_expected": len(expected_ids),
        "images_completed": completed, "category_count": len(category_ids),
        "detection_confidence_policy": ("uniform score 1.0; model does not emit per-box confidence"
                                         if uniform_confidence else "model-provided detection confidence"),
    }
    for task, records in (("object_detection", detections), ("instance_segmentation", segmentations)):
        if not records:
            output[task] = {"status": "unsupported" if task == "instance_segmentation" else "no_detections"}
            continue
        lvis_dt = LVISResults(lvis_gt, records, max_dets=300)
        evaluator = LVISEval(lvis_gt, lvis_dt, iou_type="bbox" if task == "object_detection" else "segm")
        evaluator.params.img_ids = expected_ids
        evaluator.params.cat_ids = category_ids
        evaluator.run()
        metrics = evaluator.get_results()
        output[task] = {
            "status": "complete" if completed == len(expected_ids) else "partial",
            "metric": "official LVIS v1 AP",
            "AP": metrics.get("AP"), "AP50": metrics.get("AP50"), "AP75": metrics.get("AP75"),
            "AP_rare": metrics.get("APr"), "AP_common": metrics.get("APc"),
            "AP_frequent": metrics.get("APf"), "AR_300": metrics.get("AR@300"),
            "detections": len(records),
        }
    run["lvis_official_evaluation"] = output
    if specialist_items:
        task = run.setdefault("tasks", {}).setdefault("coco", {})
        summary = task.get("summary", {})
        metrics = summary.get("metrics", {})
        run["sample_count"] = completed
        run["input_size"] = "native model processor"
        run["items"] = [items[image_id] for image_id in expected_ids]
        run["runtime"] = {
            "latency_mean_s": summary.get("mean_latency_s"),
            "latency_median_s": summary.get("median_latency_s"),
            "images_per_s": (1 / summary["mean_latency_s"]
                             if summary.get("mean_latency_s") else None),
            "timing_scope": run.get("query_timing_scope"),
        }
        run["tasks"]["object_detection"] = {
            **output.get("object_detection", {}),
            "class_aware_F1_IoU50": metrics.get("micro_f1"),
        }
        run["tasks"]["instance_segmentation"] = output.get("instance_segmentation", {"status": "unsupported"})
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Prepared LVIS-format annotation subset, not the COCO-shaped inference annotations")
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run, args.manifest, args.annotations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
