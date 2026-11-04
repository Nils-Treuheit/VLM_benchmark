#!/usr/bin/env python3
"""Score a saved LVIS panel run with the official LVIS evaluation API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_run(run_path: Path, manifest_path: Path, annotations_path: Path) -> dict[str, Any]:
    from lvis import LVIS, LVISResults, LVISEval

    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = [int(value) for value in manifest["selected_image_ids"]]
    category_ids = [int(category["id"]) for category in manifest["examples"][0]["categories"]]
    lvis_gt = LVIS(str(annotations_path))
    items = {int(item["image_id"]): item for item in
             run.get("items", []) if item.get("image_id") is not None}
    if set(items) != set(expected_ids):
        raise ValueError(f"run image IDs differ from manifest: {len(items)} saved, {len(expected_ids)} expected")

    detections = []
    segmentations = []
    completed = 0
    for image_id in expected_ids:
        item = items[image_id]
        if "predictions" not in item or "classification" not in item:
            continue
        completed += 1
        for prediction in item.get("predictions", []):
            category_id = int(prediction["category_id"])
            if category_id not in category_ids:
                raise ValueError(f"prediction category {category_id} is outside the LVIS panel")
            x1, y1, x2, y2 = (float(value) for value in prediction["box"])
            detection = {"image_id": image_id, "category_id": category_id,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(prediction.get("score", 1.0))}
            detections.append(detection)
            if prediction.get("segmentation"):
                segmentations.append({"image_id": image_id, "category_id": category_id,
                                      "segmentation": prediction["segmentation"],
                                      "score": float(prediction.get("score", 1.0))})

    output: dict[str, Any] = {
        "api": "lvis 0.5.3", "images_expected": len(expected_ids),
        "images_completed": completed, "category_count": len(category_ids),
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
