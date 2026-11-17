#!/usr/bin/env python3
"""Run real Florence-2 `<OD>` panels and score saved outputs offline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

MODEL_ID = "microsoft/Florence-2-large-ft"
REVISION = "4a12a2b54b7016a48a22037fbd62da90cd566f2a"
SNAPSHOT = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--microsoft--Florence-2-large-ft/snapshots") / REVISION
PANEL_DEFAULTS = {
    "coco_val2017": {
        "manifest": "grounding_Bench/data/domain_manifests/coco_val2017/coco.json",
        "annotations": "/mnt/HDD1/Project_Data/public_datasets/coco/annotations/instances_val2017.json",
        "result": "grounding_Bench/results/domain_runs/coco_val2017/florence2/microsoft--Florence-2-large-ft.json",
        "count": 100,
    },
    "dota_v1": {
        "manifest": "grounding_Bench/data/domain_manifests/dota_v1/coco.json",
        "annotations": "grounding_Bench/data/domain_manifests/dota_v1/coco_annotations.json",
        "result": "grounding_Bench/results/domain_runs/dota_v1/florence2/microsoft--Florence-2-large-ft.json",
        "count": 100,
    },
    "bop_h3_hope_v2": {
        "manifest": "grounding_Bench/data/domain_manifests/bop_h3_hope_v2/coco.json",
        "annotations": "grounding_Bench/data/domain_manifests/bop_h3_hope_v2/coco_annotations.json",
        "result": "grounding_Bench/results/domain_runs/bop_h3_hope_v2/florence2/microsoft--Florence-2-large-ft.json",
        "count": 50,
    },
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "grounding_Bench" / "vision_benchmark.py").is_file():
            return parent
    raise RuntimeError("cannot locate repository root from runner path")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_path(root: Path, path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def panel_paths(root: Path, name: str, manifest_arg: str | None,
                annotations_arg: str | None) -> tuple[Path, Path, dict[str, Any]]:
    if name not in PANEL_DEFAULTS:
        raise ValueError(f"unknown panel {name!r}; choose from {', '.join(PANEL_DEFAULTS)}")
    defaults = PANEL_DEFAULTS[name]
    manifest_path = resolve_path(root, manifest_arg or defaults["manifest"])
    annotations_path = resolve_path(root, annotations_arg or defaults["annotations"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "coco" or not manifest.get("dataset") or not manifest.get("split"):
        raise ValueError(f"{manifest_path} is not a dataset-identified COCO-shaped manifest")
    rows = manifest.get("examples")
    if not isinstance(rows, list) or len(rows) != defaults["count"]:
        raise ValueError(f"{name} must contain exactly {defaults['count']} fixed examples")
    for row in rows:
        image = Path(row["image"])
        if not image.is_absolute():
            image = (manifest_path.parent / image).resolve()
            row["image"] = str(image)
        if not image.is_file():
            raise FileNotFoundError(f"local image is missing: {image}")
        if isinstance(row.get("image_id"), bool) or not isinstance(row.get("image_id"), int):
            raise ValueError("every panel item must have a numeric image_id")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"local COCO annotations are missing: {annotations_path}")
    ids = [int(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest repeats an image_id")
    return manifest_path, annotations_path, manifest


def release_local(root: Path) -> None:
    workflow = root / ".agentic_workflow"
    result = subprocess.run(
        ["python3", "src/executor.py", "release-local"], cwd=workflow,
        check=False, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"release-local failed ({result.returncode}): {result.stdout.strip()} {result.stderr.strip()}")
    try:
        status = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"release-local returned unrecognized output: {result.stdout.strip()}") from exc
    if status.get("deep_cleanup") != "released" or status.get("fast_cleanup") != "released":
        raise RuntimeError(f"release-local did not confirm both engine pools released: {status}")


class DeviceMemory:
    """Sample whole-device VRAM, including the desktop baseline."""
    def __init__(self) -> None:
        self.peak_mib: int | None = None
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, name="florence-gpu-memory", daemon=True)
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=3)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                )
                values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
                if values:
                    self.peak_mib = max(self.peak_mib or 0, sum(values))
                    self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(0.5)

    def gib(self) -> float | None:
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


def package_versions(torch_module: Any) -> dict[str, str | None]:
    names = ("transformers", "torchvision", "accelerate", "timm", "Pillow", "safetensors")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    versions["torch"] = str(torch_module.__version__)
    versions["torch_cuda"] = str(torch_module.version.cuda) if torch_module.version.cuda else None
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def verify_florence_runtime(versions: dict[str, str | None]) -> None:
    expected = {"python": "3.11.15", "transformers": "4.44.2",
                "torch": "2.15.0+cu132", "torch_cuda": "13.2"}
    for package, version in expected.items():
        actual = versions.get(package)
        if actual != version:
            raise RuntimeError(f"Florence runtime requires {package} {version}, got {actual!r}")


def parse_predictions(parsed: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        raise ValueError(f"<OD> processor returned {type(parsed).__name__}, expected a mapping")
    task_result = parsed.get("<OD>", parsed)
    if not isinstance(task_result, dict):
        raise ValueError("<OD> task result is not a mapping")
    boxes = task_result.get("bboxes")
    labels = task_result.get("labels")
    if not isinstance(boxes, list) or not isinstance(labels, list) or len(boxes) != len(labels):
        raise ValueError("<OD> result must contain equally sized bboxes and labels lists")
    predictions = []
    for index, (box, label) in enumerate(zip(boxes, labels)):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"<OD> box {index} must contain four xyxy coordinates")
        values = [float(value) for value in box]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"<OD> box {index} contains a non-finite coordinate")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"<OD> label {index} must be a nonempty string")
        x1, y1, x2, y2 = values
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append({"box": [x1, y1, x2, y2], "name": label.strip(), "score": 1.0})
    return predictions


def run_inference(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest_path, _, manifest = panel_paths(root, args.panel, args.manifest, args.annotations)
    rows = manifest["examples"]
    output_path = resolve_path(root, args.predictions_output)
    if output_path.exists():
        raise FileExistsError(f"attempt output already exists: {output_path}")

    # This must precede importing torch, checking CUDA, starting memory sampling, or loading Florence.
    release_local(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/mnt/HDD1/unsloth_and_huggingface_models/huggingface"
    sys.path.insert(0, str(root / "grounding_Bench"))
    sys.path.insert(0, str(root / "general_transformers_Bench"))
    import torch
    from PIL import Image
    from model_adapters import load_florence

    versions = package_versions(torch)
    verify_florence_runtime(versions)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Florence panel is blocked")
    for filename in ("configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py", "config.json"):
        if not (SNAPSHOT / filename).is_file():
            raise FileNotFoundError(f"pinned Florence snapshot is missing {filename}: {SNAPSHOT}")
    config_path = SNAPSHOT / "config.json"

    items: list[dict[str, Any]] = []
    warmup_attempts: list[dict[str, Any]] = []
    failure: str | None = None
    failure_phase: str | None = None
    load_elapsed: float | None = None
    warmup_count = min(5, len(rows))
    model = processor = None
    memory = DeviceMemory()
    try:
        memory.start()
        load_started = time.perf_counter()
        try:
            model, processor = load_florence(SNAPSHOT)
        except Exception as exc:
            load_elapsed = time.perf_counter() - load_started
            failure_phase = "model_load"
            failure = f"{type(exc).__name__}: {exc}"
        else:
            load_elapsed = time.perf_counter() - load_started

        if failure is None:
            for index, row in enumerate(rows[:warmup_count], start=1):
                attempt: dict[str, Any] = {"index": index, "image_id": int(row["image_id"]),
                                           "status": "failed", "raw_output": None}
                try:
                    with Image.open(row["image"]) as source:
                        image = source.convert("RGB")
                    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(model.device, torch.float16)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                            max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=3,
                        )
                    raw = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
                    attempt["raw_output"] = raw
                    parsed = processor.post_process_generation(raw, task="<OD>", image_size=image.size)
                    predictions = parse_predictions(parsed, image.width, image.height)
                    attempt.update({"status": "complete", "parse_status": "valid",
                                    "prediction_count": len(predictions)})
                    print(f"warmup {index}/{warmup_count}: image_id={row['image_id']}", flush=True)
                except Exception as exc:
                    attempt.update({"status": "failed", "parse_status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}"})
                    failure_phase = "warmup"
                    failure = attempt["error"]
                warmup_attempts.append(attempt)
                if failure is not None:
                    break

        if failure is None:
            for index, row in enumerate(rows, start=1):
                started = time.perf_counter()
                item: dict[str, Any] = {
                    "id": str(row.get("id", row["image_id"])), "image_id": int(row["image_id"]),
                    "status": "failed", "latency_s": None, "prediction_count": 0,
                    "predictions": [], "parse_status": "not_started", "raw_output": None,
                }
                try:
                    with Image.open(row["image"]) as source:
                        image = source.convert("RGB")
                    item["image_width"], item["image_height"] = image.size
                    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(model.device, torch.float16)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                            max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=3,
                        )
                    raw = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
                    item["raw_output"] = raw
                    parsed = processor.post_process_generation(raw, task="<OD>", image_size=image.size)
                    predictions = parse_predictions(parsed, image.width, image.height)
                    item.update({"status": "complete", "parse_status": "valid", "predictions": predictions,
                                 "prediction_count": len(predictions)})
                except Exception as exc:
                    item.update({"status": "failed", "parse_status": "failed",
                                 "error": f"{type(exc).__name__}: {exc}"})
                    failure = failure or item["error"]
                    failure_phase = failure_phase or "scored_inference"
                item["latency_s"] = time.perf_counter() - started
                items.append(item)
                print(f"inference {index}/{len(rows)} image_id={row['image_id']} status={item['status']} "
                      f"predictions={item['prediction_count']} latency_s={item['latency_s']:.3f}", flush=True)
    except Exception as exc:
        failure = failure or f"{type(exc).__name__}: {exc}"
        failure_phase = failure_phase or "inference_setup"
    finally:
        memory.stop()
        if model is not None:
            del model
        if processor is not None:
            del processor
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    complete = (failure is None and len(items) == len(rows)
                and all(item["status"] == "complete" for item in items))
    payload = {
        "model_id": MODEL_ID, "revision": REVISION, "snapshot": str(SNAPSHOT),
        "config_sha256": sha256_file(config_path), "backend": "Transformers Florence2ForConditionalGeneration",
        "precision": "float16", "generation": {"task_token": "<OD>", "max_new_tokens": args.max_new_tokens,
                      "do_sample": False, "num_beams": 3},
        "runtime_versions": versions, "dataset": manifest.get("dataset"),
        "split": manifest.get("split"), "panel": args.panel,
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "expected_count": len(rows), "image_ids": [int(row["image_id"]) for row in rows],
        "warmup_images_excluded_from_timing": warmup_count,
        "warmup_attempts": warmup_attempts,
        "model_load_duration_s": load_elapsed,
        "peak_whole_device_vram_gib": memory.gib(), "gpu_memory_samples": memory.samples,
        "gpu_memory_error": memory.error, "items": items, "panel_error": failure,
        "failure_phase": failure_phase,
        "status": "complete" if complete else "partial" if items else "failed",
    }
    atomic_json(output_path, payload)
    print(f"saved inference artifact: {output_path} ({payload['status']}, {len(items)}/{len(rows)})")
    return 0 if payload["status"] == "complete" else 1


def iou_xyxy(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def score_class_aware(items: list[dict[str, Any]], annotations_by_image: dict[int, list[dict[str, Any]]],
                      category_by_name: dict[str, int], aliases: dict[str, str]) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    tp = fp = fn = unknown = 0
    canonical_by_image: dict[int, list[dict[str, Any]]] = {}
    targets_by_category: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for image_id, annotations in annotations_by_image.items():
        for annotation in annotations:
            targets_by_category.setdefault(int(annotation["category_id"]), []).append((image_id, annotation))

    candidates: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for item in items:
        image_id = int(item["image_id"])
        canonical: list[dict[str, Any]] = []
        for prediction in item["predictions"]:
            raw_name = str(prediction["name"]).strip()
            mapped_name = aliases.get(raw_name.casefold(), raw_name)
            category_id = category_by_name.get(mapped_name.casefold())
            record = {**prediction, "name": mapped_name, "category_id": category_id}
            if category_id is None:
                unknown += 1
                fp += 1
                continue
            canonical.append(record)
            candidates.setdefault(category_id, []).append((image_id, record))
        canonical_by_image[image_id] = canonical

    matched_by_image: dict[int, set[int]] = {image_id: set() for image_id in annotations_by_image}
    for category_id in category_by_name.values():
        preds = sorted(candidates.get(category_id, []), key=lambda row: float(row[1].get("score", 1.0)), reverse=True)
        used: dict[int, set[int]] = {}
        for image_id, prediction in preds:
            targets = annotations_by_image.get(image_id, [])
            overlaps = [(iou_xyxy(prediction["box"], [ann["bbox"][0], ann["bbox"][1],
                                                        ann["bbox"][0] + ann["bbox"][2],
                                                        ann["bbox"][1] + ann["bbox"][3]]), index)
                        for index, ann in enumerate(targets)
                        if int(ann["category_id"]) == category_id and index not in used.get(image_id, set())]
            overlap, index = max(overlaps, default=(0.0, -1))
            if overlap >= 0.5:
                tp += 1
                used.setdefault(image_id, set()).add(index)
                matched_by_image[image_id].add(index)
            else:
                fp += 1
    total_targets = sum(len(anns) for anns in annotations_by_image.values())
    fn = total_targets - tp
    denominator = 2 * tp + fp + fn
    metrics = {
        "status": "complete", "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "class_aware_F1_IoU50": 2 * tp / denominator if denominator else None,
        "unmapped_label_predictions": unknown,
        "unmapped_label_policy": "unmapped labels count as false positives for class-aware F1; unmapped labels are excluded from COCOeval AP",
    }
    return metrics, canonical_by_image


def score_panel(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest_path, annotation_path, manifest = panel_paths(root, args.panel, args.manifest, args.annotations)
    inference_path = resolve_path(root, args.predictions_input)
    result_path = resolve_path(root, args.result_output or PANEL_DEFAULTS[args.panel]["result"])
    if result_path.exists() and not args.replace:
        raise FileExistsError(f"complete result already exists; use --replace only after reviewing its provenance: {result_path}")
    run = json.loads(inference_path.read_text(encoding="utf-8"))
    rows = manifest["examples"]
    expected_ids = [int(row["image_id"]) for row in rows]
    if (run.get("panel") != args.panel or run.get("model_id") != MODEL_ID
            or run.get("status") != "complete" or run.get("expected_count") != len(rows)
            or run.get("image_ids") != expected_ids or len(run.get("items", [])) != len(rows)
            or any(item.get("status") != "complete" or item.get("parse_status") != "valid" for item in run["items"])):
        raise ValueError("inference artifact is incomplete or does not match the exact manifest coverage")

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    sys.path.insert(0, str(root / "grounding_Bench"))
    from vision_benchmark import category_accuracy

    coco = COCO(str(annotation_path))
    image_ids = expected_ids
    annotation_categories = {int(category["id"]): str(category["name"])
                             for category in coco.loadCats(coco.getCatIds())}
    category_by_name = {name.casefold().strip(): category_id
                        for category_id, name in annotation_categories.items()}
    for row in rows:
        for category in row.get("categories", []):
            category_id = int(category["id"])
            if category_id not in annotation_categories:
                raise ValueError(f"manifest category {category_id} is absent from annotations")
            if annotation_categories[category_id].casefold().strip() != str(category["name"]).casefold().strip():
                raise ValueError(f"manifest category name mismatch for category {category_id}")
    category_ids = sorted(annotation_categories)
    categories = coco.loadCats(category_ids)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_ids, catIds=category_ids)):
        if not ann.get("iscrowd"):
            annotations_by_image[int(ann["image_id"])].append(ann)
    aliases: dict[str, str] = {}
    for row in rows:
        for alias, canonical in (row.get("prediction_aliases") or {}).items():
            aliases[str(alias).casefold().strip()] = str(canonical).strip()

    metrics, canonical_by_image = score_class_aware(run["items"], annotations_by_image, category_by_name, aliases)
    bbox_results = []
    for image_id, predictions in canonical_by_image.items():
        for prediction in predictions:
            x1, y1, x2, y2 = prediction["box"]
            bbox_results.append({"image_id": image_id, "category_id": prediction["category_id"],
                                 "bbox": [x1, y1, x2 - x1, y2 - y1], "score": 1.0})

    def empty_detection_results() -> Any:
        empty = COCO()
        empty.dataset = {"images": [coco.imgs[image_id] for image_id in image_ids],
                         "categories": categories, "annotations": []}
        empty.createIndex()
        return empty

    if bbox_results:
        detections = coco.loadRes(bbox_results)
    else:
        detections = empty_detection_results()
    evaluation = COCOeval(coco, detections, "bbox")
    evaluation.params.imgIds = image_ids
    evaluation.params.catIds = category_ids
    evaluation.params.maxDets = [1, 10, 100]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    raw_stats = []
    for value in evaluation.stats:
        score = float(value)
        raw_stats.append(score if math.isfinite(score) and score >= 0 else None)
    detection = {
        "status": "complete", "AP": raw_stats[0], "AP50": raw_stats[1], "AP75": raw_stats[2],
        "AP_small": raw_stats[3], "AP_medium": raw_stats[4], "AP_large": raw_stats[5],
        "AR_1": raw_stats[6], "AR_10": raw_stats[7], "AR_100": raw_stats[8],
        **metrics,
        "coco_eval_score_policy": "uniform score 1.0; AP is tie-sensitive and non-calibrated",
        "coco_eval_unmapped_label_policy": "unmapped labels are excluded from COCOeval AP and counted as false positives in class-aware F1",
    }
    per_item = []
    for item in run["items"]:
        image_id = int(item["image_id"])
        targets = [{"box": [float(ann["bbox"][0]), float(ann["bbox"][1]),
                             float(ann["bbox"][0] + ann["bbox"][2]),
                             float(ann["bbox"][1] + ann["bbox"][3])],
                    "category_id": int(ann["category_id"])} for ann in annotations_by_image[image_id]]
        predictions = [{"box": record["box"], "score": 1.0, "category_id": record["category_id"]}
                       for record in canonical_by_image[image_id]]
        classification = category_accuracy(predictions, targets)
        mapped_count = len(canonical_by_image[image_id])
        unmapped_count = len(item["predictions"]) - mapped_count
        per_item.append({key: item[key] for key in ("id", "image_id", "status", "latency_s", "prediction_count", "parse_status", "raw_output", "image_width", "image_height") if key in item}
                        | {"predictions": item["predictions"], "unmapped_label_prediction_count": unmapped_count,
                           "classification": classification})

    runtime_versions = run.get("runtime_versions", {})
    output = {
        "model_id": MODEL_ID, "revision": REVISION,
        "checkpoint": {"path": run["snapshot"], "revision": REVISION,
                      "config_sha256": run["config_sha256"], "bytes": None},
        "backend": run.get("backend"), "precision": run.get("precision"),
        "generation_mode": "Florence-2 native <OD> object detection; fixed task, no class prompts",
        "client_package_versions": runtime_versions,
        "dataset": manifest.get("dataset"), "split": manifest.get("split"),
        "sample_count": len(rows), "image_ids": image_ids, "input_size": "native image dimensions",
        "category_prompt_protocol": "fixed <OD> task token; no category prompts; unlike prompted YOLOE and grounding specialists",
        "manifest_sha256": sha256_file(manifest_path), "annotation_file": str(annotation_path),
        "warmup_images_excluded_from_timing": run.get("warmup_images_excluded_from_timing"),
        "tasks": {
            "object_detection": detection,
            "instance_segmentation": {"status": "unsupported", "reason": "Florence-2 <OD> returns boxes and labels only"},
            "classification": {
                "status": "complete" if any(item.get("classification", {}).get("matched_objects_iou50", 0) for item in per_item) else "not run",
                "definition": "label accuracy among predictions matched to any ground-truth box at IoU >= 0.5",
                "accuracy_on_matched_objects": _ratio_sum(per_item, "label_correct", "matched_objects_iou50"),
                "joint_detection_and_classification_accuracy": _ratio_sum(per_item, "label_correct", "ground_truth_objects"),
            },
        },
        "runtime": {
            "latency_mean_s": statistics.mean(item["latency_s"] for item in per_item),
            "latency_median_s": statistics.median(item["latency_s"] for item in per_item),
            "images_per_s": len(per_item) / sum(item["latency_s"] for item in per_item) if sum(item["latency_s"] for item in per_item) else None,
            "model_load_duration_s": run.get("model_load_duration_s"),
            "peak_whole_device_vram_gib": run.get("peak_whole_device_vram_gib"),
            "gpu_memory_samples": run.get("gpu_memory_samples"), "gpu_memory_error": run.get("gpu_memory_error"),
            "timing_scope": "image reading, preprocessing, generation, decoding, and parsing; model load excluded",
        },
        "items": per_item,
    }
    atomic_json(result_path, output)
    print(f"saved complete scored result: {result_path} ({len(image_ids)}/{len(rows)})")
    return 0


def _ratio_sum(items: list[dict[str, Any]], numerator: str, denominator: str) -> float | None:
    top = sum(int(item.get("classification", {}).get(numerator, 0) or 0) for item in items)
    bottom = sum(int(item.get("classification", {}).get(denominator, 0) or 0) for item in items)
    return top / bottom if bottom else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="run Florence on one fixed panel; GPU use requires a fresh release-local pass")
    infer.add_argument("--panel", choices=tuple(PANEL_DEFAULTS), required=True)
    infer.add_argument("--manifest")
    infer.add_argument("--annotations")
    infer.add_argument("--predictions-output", required=True, help="unique local attempt artifact path")
    infer.add_argument("--max-new-tokens", type=int, default=1024)
    score = subparsers.add_parser("score", help="score a complete inference artifact using pycocotools")
    score.add_argument("--panel", choices=tuple(PANEL_DEFAULTS), required=True)
    score.add_argument("--manifest")
    score.add_argument("--annotations")
    score.add_argument("--predictions-input", required=True)
    score.add_argument("--result-output", help="defaults to the panel's grounding_Bench result path")
    score.add_argument("--replace", action="store_true", help="replace an existing result only after provenance review")
    args = parser.parse_args()
    if args.command == "infer":
        return run_inference(args)
    return score_panel(args)


if __name__ == "__main__":
    raise SystemExit(main())
