#!/usr/bin/env python3
"""Split combined RF100 vision-model outputs into per-dataset scorecards."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _classification(predictions: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
    used: set[int] = set()
    matched = correct = 0
    for pred in sorted(predictions, key=lambda row: float(row["score"]), reverse=True):
        options = [(_iou(pred["box"], target["box"]), i, target)
                   for i, target in enumerate(gt) if i not in used]
        overlap, index, target = max(options, default=(0.0, -1, None))
        if overlap < 0.5:
            continue
        used.add(index)
        matched += 1
        correct += int(pred["category_id"] == target["category_id"])
    return {
        "matched_objects_iou50": matched,
        "label_correct": correct,
        "classification_accuracy_on_matched": correct / matched if matched else None,
        "joint_localization_and_classification_accuracy": correct / len(gt) if gt else None,
        "ground_truth_objects": len(gt),
    }


def _class_counts(predictions: list[dict[str, Any]], gt: list[dict[str, Any]],
                  category_ids: set[int]) -> dict[str, int]:
    tp = fp = fn = 0
    for category_id in category_ids:
        targets = [row for row in gt if row["category_id"] == category_id]
        used: set[int] = set()
        candidates = sorted((row for row in predictions if row["category_id"] == category_id),
                            key=lambda row: float(row["score"]), reverse=True)
        for prediction in candidates:
            overlaps = [(_iou(prediction["box"], target["box"]), i)
                        for i, target in enumerate(targets) if i not in used]
            overlap, index = max(overlaps, default=(0.0, -1))
            if overlap >= 0.5:
                tp += 1
                used.add(index)
            else:
                fp += 1
        fn += len(targets) - len(used)
    return {"tp": tp, "fp": fp, "fn": fn}


def _evaluate_bbox(annotation_path: Path, image_ids: list[int],
                   predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {"status": "unsupported", "reason": "model produced no detections"}
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(str(annotation_path))
    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = [float(value) if np.isfinite(value) else None for value in evaluator.stats]
    return {"status": "complete", "AP": values[0], "AP50": values[1], "AP75": values[2],
            "AP_small": values[3], "AP_medium": values[4], "AP_large": values[5],
            "AR_1": values[6], "AR_10": values[7], "AR_100": values[8]}


def split_run(run_path: Path, manifest_path: Path, output_root: Path,
              group: str | None = None) -> list[Path]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combined_annotations = json.loads(Path(manifest["annotation_file"]).read_text(encoding="utf-8"))
    combined_names = {int(row["id"]): str(row["name"]) for row in combined_annotations["categories"]}
    results_by_id = {str(row["id"]): row for row in run.get("items", [])}
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in manifest.get("examples", []):
        rows_by_dataset.setdefault(str(row["source_dataset"]), []).append(row)

    written = []
    model_id = str(run.get("model_id", "vision_model"))
    for source_dataset, rows in sorted(rows_by_dataset.items()):
        slug = None
        for path in (HERE / "data" / "domain_manifests" / "rf100_vl").glob("*/coco.json"):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if candidate.get("source_dataset") == source_dataset:
                slug = path.parent.name
                domain_manifest = candidate
                break
        if slug is None:
            raise FileNotFoundError(f"per-domain manifest not found for {source_dataset}")
        annotation_path = Path(domain_manifest["annotation_file"])
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        target_names = {str(row["name"]): int(row["id"]) for row in annotations["categories"]}
        if len(target_names) != len(annotations["categories"]):
            raise ValueError(f"{source_dataset}: duplicate category names prevent safe id remapping")
        rows_by_id = {str(row["id"]): row for row in rows}
        image_id_by_source = {int(image["id"]): int(image["id"]) for image in annotations["images"]}
        gts: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_id_by_source.values()}
        for annotation in annotations["annotations"]:
            x, y, width, height = map(float, annotation["bbox"])
            gts[int(annotation["image_id"])].append({
                "box": [x, y, x + width, y + height],
                "category_id": int(annotation["category_id"]),
            })

        items = []
        bbox_rows = []
        segm_rows = []
        tp = fp = fn = matched = correct = gt_count = 0
        latencies = []
        for combined_row in rows:
            result = results_by_id.get(str(combined_row["id"]))
            if result is None:
                continue
            source_image_id = int(combined_row["source_image_id"])
            image_id = image_id_by_source.get(source_image_id)
            if image_id is None:
                raise ValueError(f"{source_dataset}: selected source image {source_image_id} missing from annotations")
            categories = {str(row["name"]): int(row["id"]) for row in combined_row["source_categories"]}
            predictions = []
            for raw in result.get("predictions", []):
                name = combined_names.get(int(raw["category_id"]), str(raw.get("name", "")))
                target_id = target_names.get(name)
                if target_id is None or name not in categories:
                    continue
                x1, y1, x2, y2 = map(float, raw["box"])
                prediction = {"box": [x1, y1, x2, y2], "score": float(raw["score"]),
                              "category_id": target_id, "name": name}
                if raw.get("segmentation") is not None:
                    prediction["segmentation"] = raw["segmentation"]
                predictions.append(prediction)
                bbox_rows.append({"image_id": image_id, "category_id": target_id,
                                  "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(raw["score"])})
                if raw.get("segmentation") is not None:
                    segm_rows.append({"image_id": image_id, "category_id": target_id,
                                      "segmentation": raw["segmentation"], "score": float(raw["score"])})
            gt = gts[image_id]
            class_stats = _classification(predictions, gt)
            counts = _class_counts(predictions, gt, set(target_names.values()))
            tp += counts["tp"]
            fp += counts["fp"]
            fn += counts["fn"]
            matched += class_stats["matched_objects_iou50"]
            correct += class_stats["label_correct"]
            gt_count += class_stats["ground_truth_objects"]
            latency = result.get("latency_s")
            if latency is not None:
                latencies.append(float(latency))
            items.append({**result, "image_id": image_id, "predictions": predictions,
                          "classification": class_stats, "prediction_count": len(predictions),
                          "gt_instance_count": len(gt)})

        detection = _evaluate_bbox(annotation_path, list(gts), bbox_rows)
        total = 2 * tp + fp + fn
        detection.update({"class_aware_F1_IoU50": 2 * tp / total if total else None,
                          "tp": tp, "fp": fp, "fn": fn})
        has_masks = any(bool(row.get("segmentation")) for row in annotations["annotations"])
        segmentation = ({"status": "unsupported", "reason": "model produced no instance masks"}
                        if not segm_rows else {"status": "unsupported", "reason": "mask metric adapter not enabled"})
        if not has_masks:
            segmentation = {"status": "unsupported", "reason": "source annotations contain no instance masks"}
        summary = {
            "accuracy_on_matched_objects": correct / matched if matched else None,
            "matched_object_coverage": matched / gt_count if gt_count else None,
            "joint_detection_and_classification_accuracy": correct / gt_count if gt_count else None,
            "matched_object_count": matched, "ground_truth_object_count": gt_count,
            "correct_label_count": correct,
        }
        if matched:
            summary["status"] = "complete"
        else:
            summary["status"] = "not run"
        per_domain = {
            key: run.get(key) for key in ("model_id", "checkpoint", "backend", "input_size",
                                           "warmup_images_excluded_from_timing", "model_load_warnings")
        }
        per_domain.update({
            "dataset": domain_manifest.get("dataset", f"RF100-VL: {source_dataset}"),
            "split": domain_manifest.get("split", "test"), "sample_count": len(items),
            "image_ids": [item["image_id"] for item in items],
            "category_prompt_set": sorted(target_names),
            "tasks": {"object_detection": detection, "instance_segmentation": segmentation,
                      "classification": summary},
            "runtime": {
                **run.get("runtime", {}),
                "latency_mean_s": statistics.mean(latencies) if latencies else None,
                "latency_median_s": statistics.median(latencies) if latencies else None,
                "images_per_s": len(latencies) / sum(latencies) if latencies and sum(latencies) else None,
                "timing_scope": run.get("runtime", {}).get("timing_scope"),
            },
            "items": items,
        })
        target = output_root / f"rf100_{slug}" / "vision" / f"{model_id}.json"
        if group:
            target = output_root / f"rf100_{slug}" / group / f"{model_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(per_domain, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Combined vision_benchmark JSON output")
    parser.add_argument("--manifest", type=Path,
                        default=HERE / "data" / "domain_manifests" / "manifests" / "rf100_vl_all" / "coco.json")
    parser.add_argument("--output-root", type=Path, default=HERE / "results" / "domain_runs")
    parser.add_argument("--group", help="Optional isolated report group directory")
    args = parser.parse_args()
    outputs = split_run(args.run, args.manifest, args.output_root, args.group)
    print(json.dumps({"domains": len(outputs), "files": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
