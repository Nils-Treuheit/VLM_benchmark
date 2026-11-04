#!/usr/bin/env python3
"""Serial COCO detection and instance segmentation benchmark for local baselines.

The script intentionally requires local checkpoint paths. Ultralytics and SAM
must never fetch model files implicitly during a benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SAM31 = Path(
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/"
    "models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt"
)
MODELS = ("yoloe26l-seg", "yolo26l", "yolo26l-seg", "sam3.1")


class DeviceMemory:
    """Sample whole-device memory, including other desktop processes."""

    def __init__(self) -> None:
        self.peak_mib: int | None = None
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, name="vision-gpu-memory", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def sample(self) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=5,
            )
            values = [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]
            if values:
                self.peak_mib = max(self.peak_mib or 0, sum(values))
                self.samples += 1
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(0.5)

    def gib(self) -> float | None:
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


def release_local() -> None:
    workflow = ROOT / ".agentic_workflow"
    executor = workflow / "src" / "executor.py"
    if not executor.is_file():
        raise FileNotFoundError(f"required local-engine release helper is missing: {executor}")
    subprocess.run([sys.executable, "src/executor.py", "release-local"], cwd=workflow, check=True)


def parse_weights(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key not in MODELS or key in result:
            raise ValueError(f"weight must be unique MODEL=LOCAL_PATH, supported models: {', '.join(MODELS)}")
        weight = Path(path).expanduser().resolve()
        if not weight.is_file():
            raise FileNotFoundError(f"local model weights are missing: {weight}")
        result[key] = weight
    return result


def checkpoint_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_manifest(path: Path, task: str = "coco") -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task:
        raise ValueError(f"manifest task must be {task!r}")
    if task == "refcocog" and (payload.get("dataset"), payload.get("split")) != ("RefCOCOg", "test"):
        raise ValueError("text-grounding manifest must be the shared RefCOCOg test manifest")
    if task == "coco" and (not payload.get("dataset") or not payload.get("split")):
        raise ValueError("detection manifest must identify its dataset and split")
    rows = payload.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest has no examples")
    for row in rows:
        image = Path(row["image"])
        if not image.is_absolute():
            image = (path.parent / image).resolve()
            row["image"] = str(image)
        if not image.is_file():
            raise FileNotFoundError(f"local image is missing: {image}")
        if task == "coco" and not isinstance(row.get("image_id"), int):
            raise ValueError("every COCO example requires its numeric image_id")
        if task == "refcocog" and (not row.get("phrase") or not row.get("target_box")):
            raise ValueError("every RefCOCOg example requires a phrase and normalized target_box")
    return rows


def mask_rle(mask: Any, height: int, width: int, mask_utils: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(mask)
    if array.shape != (height, width):
        from PIL import Image
        array = np.asarray(Image.fromarray(array.astype("uint8")).resize((width, height), Image.Resampling.NEAREST))
    encoded = mask_utils.encode(np.asfortranarray(array.astype("uint8")))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def load_yolo(model_id: str, weight: Path) -> tuple[Any, str]:
    from ultralytics import YOLO, YOLOE

    if model_id == "yoloe26l-seg":
        return YOLOE(str(weight)), "Ultralytics YOLOE"
    return YOLO(str(weight)), "Ultralytics YOLO"


def load_sam31(weight: Path) -> tuple[Any, Any, str]:
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=str(weight), load_from_HF=False, device="cuda", eval_mode=True,
    )
    return model, Sam3Processor(model), f"SAM 3.1 BF16 autocast / PyTorch {torch.__version__}"


def yolo_predictions(model: Any, model_id: str, row: dict[str, Any], names: list[str],
                     imgsz: int = 640) -> tuple[list[dict[str, Any]], float]:
    """Return xyxy pixel boxes and optional masks; prompt setup is timed."""
    started = time.perf_counter()
    if model_id == "yoloe26l-seg":
        model.set_classes(names)
    result = model.predict(
        source=row["image"], imgsz=imgsz, device=0, batch=1, verbose=False,
        conf=0.001, max_det=300,
    )[0]
    elapsed = time.perf_counter() - started
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], elapsed
    xyxy = boxes.xyxy.detach().float().cpu().tolist()
    scores = boxes.conf.detach().float().cpu().tolist()
    class_ids = boxes.cls.detach().int().cpu().tolist()
    masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else None
    records = []
    for i, (box, score, class_id) in enumerate(zip(xyxy, scores, class_ids)):
        name = str(result.names[int(class_id)])
        if model_id == "yoloe26l-seg":
            # YOLOE's prediction class indices follow set_classes(names).
            name = names[int(class_id)] if 0 <= int(class_id) < len(names) else name
        records.append({"box": [float(v) for v in box], "score": float(score), "name": name,
                        "mask": masks[i] if masks is not None else None})
    return records, elapsed


def sam_predictions(processor: Any, row: dict[str, Any], names: list[str]) -> tuple[list[dict[str, Any]], float]:
    from PIL import Image
    import torch

    started = time.perf_counter()
    image = Image.open(row["image"]).convert("RGB")
    records = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for name in names:
            output = processor.set_text_prompt(state=state, prompt=name)
            boxes = output.get("boxes")
            scores = output.get("scores")
            masks = output.get("masks")
            if boxes is None:
                continue
            boxes = boxes.detach().float().cpu().tolist()
            scores = scores.detach().float().cpu().tolist() if scores is not None else [1.0] * len(boxes)
            masks = masks.detach().cpu().numpy() if masks is not None else None
            for i, box in enumerate(boxes):
                if len(box) != 4:
                    raise ValueError("SAM 3.1 returned a non-xyxy box")
                records.append({"box": [float(v) for v in box], "score": float(scores[i]), "name": name,
                                "mask": masks[i].squeeze() if masks is not None else None})
    return records, time.perf_counter() - started


def category_accuracy(predictions: list[dict[str, Any]], gt: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    """Report label accuracy on geometrically matched objects and joint accuracy."""
    ranked = sorted(predictions, key=lambda item: item["score"], reverse=True)
    used: set[int] = set()
    matched = correct = joint_correct = 0
    for prediction in ranked:
        overlaps = [(_pixel_iou(prediction["box"], target["box"]), idx, target)
                    for idx, target in enumerate(gt) if idx not in used]
        if not overlaps:
            continue
        overlap, idx, target = max(overlaps, key=lambda item: item[0])
        if overlap < iou_threshold:
            continue
        used.add(idx)
        matched += 1
        is_correct = prediction["category_id"] == target["category_id"]
        correct += int(is_correct)
        joint_correct += int(is_correct and overlap >= iou_threshold)
    return {"matched_objects_iou50": matched, "label_correct": correct,
            "classification_accuracy_on_matched": correct / matched if matched else None,
            "joint_localization_and_classification_accuracy": joint_correct / len(gt) if gt else None,
            "ground_truth_objects": len(gt)}


def _pixel_iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_counts(predictions: list[dict[str, Any]], gt: list[dict[str, Any]], category_ids: list[int]) -> dict[str, int]:
    tp = fp = fn = 0
    for category_id in category_ids:
        targets = [item for item in gt if item["category_id"] == category_id]
        candidates = sorted((item for item in predictions if item["category_id"] == category_id),
                            key=lambda item: item["score"], reverse=True)
        used: set[int] = set()
        for prediction in candidates:
            overlaps = [(_pixel_iou(prediction["box"], target["box"]), index)
                        for index, target in enumerate(targets) if index not in used]
            best = max(overlaps, default=(0.0, -1))
            if best[0] >= 0.5:
                tp += 1
                used.add(best[1])
            else:
                fp += 1
        fn += len(targets) - len(used)
    return {"tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True,
                        help="Official instances_val2017.json with segmentation polygons/RLE")
    parser.add_argument("--refcocog-manifest", type=Path,
                        help="Optional RefCOCOg test manifest for text grounding; supported by YOLOE and SAM 3.1")
    parser.add_argument("--model", choices=MODELS, required=True, help="Run exactly one model per invocation")
    parser.add_argument("--weights", required=True, help="Explicit local MODEL=PATH checkpoint; no auto-download")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "vision_runs")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--imgsz", type=int,
                        help="YOLO inference size; defaults to the shared manifest's input_size")
    args = parser.parse_args()
    try:
        weights = parse_weights([args.weights])
        if args.model not in weights:
            raise ValueError("--weights model key must match --model")
        manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        rows = read_manifest(args.manifest, "coco")[:args.max_images]
        imgsz = args.imgsz or int(manifest_payload.get("input_size", 640))
        if imgsz < 1:
            raise ValueError("--imgsz and manifest input_size must be positive")
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import numpy as np

        coco = COCO(str(args.coco_annotations))
        image_ids = [row["image_id"] for row in rows]
        categories = coco.loadCats(coco.getCatIds())
        category_names = [str(category["name"]) for category in categories]
        category_by_name = {name.casefold(): int(category["id"]) for name, category in
                            ((c["name"], c) for c in categories)}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_ids, catIds=list(category_by_name.values()))):
            if ann.get("iscrowd"):
                continue
            annotations_by_image[int(ann["image_id"])].append(ann)
        release_local()
        device_memory = DeviceMemory()
        device_memory.start()
        model_load_started = time.perf_counter()
        if args.model == "sam3.1":
            model, processor, backend = load_sam31(weights[args.model])
        else:
            model, backend = load_yolo(args.model, weights[args.model])
            processor = None
        model_load_duration_s = time.perf_counter() - model_load_started
        warmup_count = min(max(args.warmup_images, 0), len(rows))
        for warmup_row in rows[:warmup_count]:
            if args.model == "sam3.1":
                sam_predictions(processor, warmup_row, category_names)
            else:
                yolo_predictions(model, args.model, warmup_row, category_names, imgsz)
        grounding_items = []
        if args.refcocog_manifest:
            if args.model in {"yoloe26l-seg", "sam3.1"}:
                ref_rows = read_manifest(args.refcocog_manifest, "refcocog")[:args.max_images]
                for ref_row in ref_rows:
                    if args.model == "sam3.1":
                        predictions, elapsed = sam_predictions(processor, ref_row, [ref_row["phrase"]])
                    else:
                        predictions, elapsed = yolo_predictions(model, args.model, ref_row, [ref_row["phrase"]], imgsz)
                    from PIL import Image
                    with Image.open(ref_row["image"]) as image:
                        width, height = image.size
                    target = [float(ref_row["target_box"][0]) * width,
                              float(ref_row["target_box"][1]) * height,
                              float(ref_row["target_box"][2]) * width,
                              float(ref_row["target_box"][3]) * height]
                    best_iou = max((_pixel_iou(prediction["box"], target) for prediction in predictions), default=0.0)
                    grounding_items.append({"id": str(ref_row.get("id")), "phrase": ref_row["phrase"],
                                            "latency_s": elapsed, "best_iou": best_iou,
                                            "acc_iou_05": float(best_iou >= 0.5)})
            else:
                grounding_items = []
        item_results = []
        bbox_results: list[dict[str, Any]] = []
        segm_results: list[dict[str, Any]] = []
        class_counts = {"tp": 0, "fp": 0, "fn": 0}
        per_image_times = []
        for row in rows:
            image_id = int(row["image_id"])
            gt = []
            for ann in annotations_by_image[image_id]:
                category = coco.loadCats([ann["category_id"]])[0]
                gt.append({"box": [float(ann["bbox"][0]), float(ann["bbox"][1]),
                                     float(ann["bbox"][0] + ann["bbox"][2]),
                                     float(ann["bbox"][1] + ann["bbox"][3])],
                           "category_id": int(ann["category_id"]), "name": category["name"], "ann": ann})
            if args.model == "sam3.1":
                predictions, elapsed = sam_predictions(processor, row, category_names)
            else:
                predictions, elapsed = yolo_predictions(model, args.model, row, category_names, imgsz)
            aliases = row.get("prediction_aliases", {})
            if aliases:
                alias_lookup = {str(alias).casefold(): str(name) for alias, name in aliases.items()}
                predictions = [{**prediction,
                                "name": alias_lookup.get(prediction["name"].casefold(), prediction["name"])}
                               for prediction in predictions]
            per_image_times.append(elapsed)
            height, width = int(coco.imgs[image_id]["height"]), int(coco.imgs[image_id]["width"])
            canonical = []
            for prediction in predictions:
                category_id = category_by_name.get(prediction["name"].casefold())
                if category_id is None:
                    continue
                x1, y1, x2, y2 = prediction["box"]
                x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
                y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                record = {"box": [x1, y1, x2, y2], "score": prediction["score"],
                          "category_id": category_id, "name": prediction["name"]}
                bbox_results.append({"image_id": image_id, "category_id": category_id,
                                     "bbox": [x1, y1, x2 - x1, y2 - y1], "score": prediction["score"]})
                if prediction["mask"] is not None:
                    rle = mask_rle(prediction["mask"], height, width, mask_utils)
                    record["segmentation"] = rle
                    segm_results.append({"image_id": image_id, "category_id": category_id,
                                         "segmentation": rle, "score": prediction["score"]})
                canonical.append(record)
            item_results.append({"id": str(row.get("id", image_id)), "image_id": image_id,
                                 "latency_s": elapsed, "predictions": canonical,
                                 "classification": category_accuracy(canonical, gt),
                                 "prediction_count": len(canonical), "gt_instance_count": len(gt)})
            image_counts = class_aware_counts(canonical, gt, list(category_by_name.values()))
            for key in class_counts:
                class_counts[key] += image_counts[key]
        device_memory.stop()

        def evaluate(result_rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
            has_instance_masks = any(annotation.get("segmentation")
                                     for image_annotations in annotations_by_image.values()
                                     for annotation in image_annotations)
            if kind == "segm" and not has_instance_masks:
                return {"status": "unsupported", "reason": "the selected dataset annotations contain no instance masks"}
            if not result_rows:
                return {"status": "unsupported", "reason": "model produced no instance masks" if kind == "segm"
                        else "model produced no detections"}
            detections = coco.loadRes(result_rows)
            evaluation = COCOeval(coco, detections, kind)
            evaluation.params.imgIds = image_ids
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
            stats = [float(value) if np.isfinite(value) else None for value in evaluation.stats]
            return {"status": "complete", "AP": stats[0], "AP50": stats[1], "AP75": stats[2],
                    "AP_small": stats[3], "AP_medium": stats[4], "AP_large": stats[5],
                    "AR_1": stats[6], "AR_10": stats[7], "AR_100": stats[8]}

        matched_count = sum(item["classification"]["matched_objects_iou50"] for item in item_results)
        correct_labels = sum(item["classification"]["label_correct"] for item in item_results)
        gt_object_count = sum(item["classification"]["ground_truth_objects"] for item in item_results)
        class_total = sum(class_counts.values())
        class_f1 = (2 * class_counts["tp"] / (2 * class_counts["tp"] + class_counts["fp"] + class_counts["fn"])
                    if class_total and (2 * class_counts["tp"] + class_counts["fp"] + class_counts["fn"]) else None)
        detection_result = evaluate(bbox_results, "bbox")
        detection_result.update({"class_aware_F1_IoU50": class_f1, **class_counts})
        output = {
            "model_id": args.model, "checkpoint": checkpoint_identity(weights[args.model]), "backend": backend,
            "dataset": manifest_payload.get("dataset"),
            "split": manifest_payload.get("split"),
            "prediction_aliases": manifest_payload.get("examples", [{}])[0].get("prediction_aliases", {}),
            "sample_count": len(rows), "image_ids": image_ids,
            "input_size": f"native image dimensions; YOLO imgsz={imgsz}; batch=1",
            "warmup_images_excluded_from_timing": warmup_count,
            "category_prompt_set": category_names,
            "tasks": {"object_detection": detection_result,
                      "instance_segmentation": evaluate(segm_results, "segm"),
                      "classification": {"status": "complete" if matched_count else "not run",
                         "definition": "COCO label accuracy among predictions matched to any ground-truth object at IoU >= 0.5; separate from detection AP",
                         "accuracy_on_matched_objects": correct_labels / matched_count if matched_count else None,
                         "matched_object_coverage": matched_count / gt_object_count if gt_object_count else None,
                         "joint_detection_and_classification_accuracy": correct_labels / gt_object_count if gt_object_count else None,
                         "matched_object_count": matched_count,
                         "ground_truth_object_count": gt_object_count,
                         "correct_label_count": correct_labels}},
            "runtime": {"latency_mean_s": statistics.mean(per_image_times) if per_image_times else None,
                        "latency_median_s": statistics.median(per_image_times) if per_image_times else None,
                        "images_per_s": len(per_image_times) / sum(per_image_times) if sum(per_image_times) else None,
                        "model_load_duration_s": model_load_duration_s,
                        "peak_whole_device_vram_gib": device_memory.gib(),
                        "gpu_memory_samples": device_memory.samples,
                        "gpu_memory_error": device_memory.error,
                        "timing_scope": "single image; preprocessing and prompt setup included; model load excluded"},
            "items": item_results,
        }
        if args.refcocog_manifest:
            if args.model not in {"yoloe26l-seg", "sam3.1"}:
                output["tasks"]["text_grounding"] = {
                    "status": "unsupported", "reason": "closed-set YOLO26 does not accept text prompts"}
            elif not grounding_items:
                output["tasks"]["text_grounding"] = {"status": "not run", "reason": "no grounding examples were evaluated"}
            else:
                output["tasks"]["text_grounding"] = {
                    "status": "complete", "dataset": "RefCOCOg test", "sample_count": len(grounding_items),
                    "mean_iou": statistics.mean(item["best_iou"] for item in grounding_items),
                    "acc_iou_05": statistics.mean(item["acc_iou_05"] for item in grounding_items),
                    "mean_latency_s": statistics.mean(item["latency_s"] for item in grounding_items),
                    "items": grounding_items,
                }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / f"{args.model}.json"
        target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"{args.model}: {len(rows)} images -> {target}")
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
