#!/usr/bin/env python3
"""Serial local evaluation of LocateAnything-3B and GroundAnything-VLM."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import model_adapters as adapters  # noqa: E402
import scoring  # noqa: E402

TASKS = ("refcocog", "coco", "screenspot_pro", "hiertext")
MODEL_IDS = (adapters.LOCATE_ID, adapters.GROUND_ID)
SEED = 3407
SAMPLE_COUNT = 100
MAX_TOKENS = {adapters.LOCATE_ID: 8192, adapters.GROUND_ID: 4096}


class WholeDeviceMemory:
    """Poll whole-device GPU memory; values include desktop baseline use."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="grounding-gpu-memory", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=3,
                )
                readings = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
                if readings:
                    value = sum(readings)
                    self.peak_mib = max(self.peak_mib or 0, value)
                    self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval_s)

    @property
    def peak_gib(self) -> float | None:
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


def load_manifest(path: Path, task: str, count: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task:
        raise ValueError(f"{path} declares task={payload.get('task')!r}, expected {task!r}")
    rows = payload.get("examples")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"{path} must contain exactly {count} sampled examples")
    if payload.get("sample_count") != count:
        raise ValueError(f"{path} sample_count does not match its examples")
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = (HERE / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"local image is missing: {image_path}")
        row["_image_path"] = str(image_path)
    return rows


def score_item(task: str, parsed: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if task == "refcocog":
        return scoring.score_referring(parsed, row["target_box"])
    if task == "screenspot_pro":
        return scoring.score_gui(parsed, row["target_box"], row.get("target_point"))
    if task == "hiertext":
        predicted = [r["box_xyxy_norm"] for r in parsed["records"]
                    if r["kind"] == "box" and r["valid"]]
        counts = scoring.match_boxes(predicted, row["target_boxes"])
        return {**counts, **scoring.f1_counts(**counts)}
    if task == "coco":
        tp = fp = fn = 0
        by_category = {}
        for category in row["categories"]:
            name = category["name"]
            targets = row["targets_by_category"].get(str(name), [])
            predicted = [r["box_xyxy_norm"] for r in parsed["records"]
                         if r["kind"] == "box" and r["valid"] and (r.get("label") or "").casefold() == str(name).casefold()]
            counts = scoring.match_boxes(predicted, targets)
            by_category[str(name)] = {**counts, **scoring.f1_counts(**counts)}
            tp += counts["tp"]
            fp += counts["fp"]
            fn += counts["fn"]
        expected_labels = {str(category["name"]).casefold() for category in row["categories"]}
        unknown_labels = [r.get("label") for r in parsed["records"] if r["kind"] == "box"
                          and (r.get("label") or "").casefold() not in expected_labels]
        fp += len(unknown_labels)
        return {**{f"micro_{k}": v for k, v in scoring.f1_counts(tp, fp, fn).items()},
                "tp": tp, "fp": fp, "fn": fn, "by_category": by_category,
                "category_mapping_supported": not unknown_labels,
                "unknown_label_predictions": len(unknown_labels)}
    raise ValueError(task)


def duplicate_count(parsed: dict[str, Any]) -> int:
    seen = set()
    duplicates = 0
    for record in parsed.get("records", []):
        geometry = record.get("box_xyxy_norm") or record.get("point_xy_norm")
        key = (record.get("label"), record.get("kind"), tuple(geometry or ()))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def enforce_no_target_policy(task: str, parsed: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    no_target_records = [record for record in parsed.get("records", []) if record.get("kind") == "none"]
    if not no_target_records:
        return parsed
    has_target = {
        "refcocog": True,
        "screenspot_pro": True,
        "coco": any(row.get("targets_by_category", {}).values()),
        "hiertext": bool(row.get("target_boxes")),
    }[task]
    parsed["no_target_allowed"] = not has_target
    if has_target:
        for record in no_target_records:
            record["valid"] = False
        parsed["valid"] = False
    return parsed


def aggregate(task: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [item.get("scores", {}) for item in items if item.get("status") == "complete"]
    fields = {
        "refcocog": ("iou", "acc_iou_05"),
        "screenspot_pro": ("point_in_target", "acc_iou_05", "best_box_iou"),
        "hiertext": ("precision", "recall", "f1"),
        "coco": ("micro_precision", "micro_recall", "micro_f1"),
    }[task]
    if task in {"hiertext", "coco"}:
        count_keys = ("tp", "fp", "fn")
        totals = {key: sum(int(s.get(key, 0)) for s in scores) for key in count_keys}
        counts = scoring.f1_counts(**totals)
        metrics = ({name: counts[name] for name in fields} if task == "hiertext" else
                   {f"micro_{name}": counts[name] for name in ("precision", "recall", "f1")})
    else:
        metrics = {name: statistics.mean(s[name] for s in scores if s.get(name) is not None)
                   if any(s.get(name) is not None for s in scores) else None for name in fields}
    latencies = [item["latency_s"] for item in items if item.get("status") == "complete" and item.get("latency_s") is not None]
    return {
        "n_examples": len(items), "completed": len(scores), "metrics": metrics,
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "malformed": sum(bool(i.get("parse", {}).get("malformed")) for i in items),
        "invalid_outputs": sum(i.get("parse", {}).get("valid") is False for i in items),
        "empty_outputs": sum(bool(i.get("parse", {}).get("empty")) for i in items),
        "truncated": sum(i.get("finish_reason") == "length" for i in items),
        "duplicate_predictions": sum(i.get("duplicate_count", 0) for i in items),
    }


def coco_official_ap(annotation_file: Path | None, rows: list[dict[str, Any]],
                     items: list[dict[str, Any]]) -> dict[str, Any]:
    """Run pycocotools COCOeval on the fixed image subset when IDs map cleanly."""
    if annotation_file is None:
        return {"status": "unavailable", "reason": "pass --coco-annotations to enable COCOeval"}
    if not annotation_file.is_file():
        return {"status": "unavailable", "reason": f"annotation file not found: {annotation_file}"}
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {"status": "unavailable", "reason": "pycocotools is not installed in this runtime"}
    item_by_id = {str(item.get("id")): item for item in items}
    image_ids: list[int] = []
    requested_pairs: set[tuple[int, int]] = set()
    requested_category_ids: set[int] = set()
    predictions = []
    unmapped_labels: set[str] = set()
    for row in rows:
        image_id = int(row["image_id"])
        image_ids.append(image_id)
        row_categories = {}
        for category in row["categories"]:
            category_id = int(category["id"])
            row_categories[str(category["name"]).casefold()] = category_id
            requested_pairs.add((image_id, category_id))
            requested_category_ids.add(category_id)
        item = item_by_id.get(str(row["id"]))
        if not item or item.get("status") != "complete":
            continue
        parsed = item.get("parse", {})
        if not parsed.get("valid"):
            continue
        width, height = item.get("image_width"), item.get("image_height")
        if not width or not height:
            continue
        for record in parsed.get("records", []):
            label = (record.get("label") or "").casefold()
            category_id = row_categories.get(label)
            box = record.get("box_xyxy_norm")
            if record.get("kind") != "box" or not record.get("valid") or not box:
                continue
            if category_id is None:
                unmapped_labels.add(label or "(missing label)")
                continue
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": image_id, "category_id": category_id,
                "bbox": [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height],
                "score": 1.0,
            })
    if not rows or any("image_id" not in row for row in rows):
        return {"status": "unsupported", "reason": "manifest lacks COCO image IDs"}
    if unmapped_labels:
        return {"status": "unsupported", "reason": "predicted labels did not map to COCO category IDs",
                "unmapped_labels": sorted(unmapped_labels)}
    try:
        ground_truth = COCO(str(annotation_file))
        # The prompt asks for image-specific categories. Exclude unrequested
        # image/category pairs from GT so they are not scored as false negatives.
        subset_gt = COCO()
        subset_gt.dataset = copy.deepcopy(ground_truth.dataset)
        subset_gt.dataset["images"] = [image for image in subset_gt.dataset["images"]
                                       if int(image["id"]) in set(image_ids)]
        subset_gt.dataset["categories"] = [category for category in subset_gt.dataset["categories"]
                                           if int(category["id"]) in requested_category_ids]
        subset_gt.dataset["annotations"] = [annotation for annotation in subset_gt.dataset["annotations"]
                                             if (int(annotation["image_id"]), int(annotation["category_id"])) in requested_pairs]
        subset_gt.createIndex()
        if predictions:
            detections = subset_gt.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {"images": subset_gt.dataset["images"],
                                  "categories": subset_gt.dataset["categories"], "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(subset_gt, detections, "bbox")
        evaluator.params.imgIds = sorted(set(image_ids))
        evaluator.params.catIds = sorted(requested_category_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        if float(evaluator.stats[0]) < 0:
            return {"status": "unsupported", "reason": "sampled image/category pairs contain no evaluable COCO positives",
                    "image_count": len(set(image_ids))}
        return {"status": "complete", "metric": "COCOeval bbox AP on sampled image/category pairs",
                "AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
                "AP75": float(evaluator.stats[2]), "detections": len(predictions),
                "image_count": len(set(image_ids)),
                "confidence_policy": "uniform 1.0; models emit no per-box confidence"}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def release_local() -> None:
    executor = ROOT / ".agentic_workflow" / "src" / "executor.py"
    if not executor.is_file():
        raise FileNotFoundError(f"required local engine release helper is missing: {executor}")
    subprocess.run([sys.executable, str(executor), "release-local"], cwd=ROOT, check=True)


def run_model(model_id: str, rows_by_task: dict[str, list[dict[str, Any]]], tasks: list[str],
              output_path: Path, resume: bool, ground_base_url: str,
              coco_annotations: Path | None, data_provenance: dict[str, Any]) -> dict[str, Any]:
    snapshot, revision = adapters.snapshot_for(model_id)
    prior = {}
    if resume and output_path.is_file():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if prior.get("sampling") != data_provenance:
            raise ValueError(f"cannot resume {output_path}: manifest provenance differs from the existing run")
    if model_id == adapters.LOCATE_ID:
        release_local()
        load_memory = WholeDeviceMemory()
        load_memory.start()
        model = adapters.LocateAnythingAdapter(snapshot)
        load_memory.stop()
        load_peak = load_memory.peak_gib
        service_inventory = None
    else:
        model = adapters.GroundAnythingAdapter(ground_base_url)
        # Check that the configured model-specific endpoint is alive before any examples run.
        service_inventory = model.model_inventory()
        load_peak = None
    result = {
        "model_id": model_id, "revision": revision, "snapshot": str(snapshot),
        "backend": model.backend, "precision": model.precision, "generation_mode": model.mode,
        "client_package_versions": adapters.package_versions(), "service_inventory": service_inventory,
        "model_load_peak_whole_device_vram_gib": load_peak,
        "sampling": data_provenance,
        "tasks": prior.get("tasks", {}), "status": "running",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        rows = rows_by_task[task]
        task_items = result["tasks"].get(task, {}).get("items", [])
        saved = {str(item.get("id")): item for item in task_items}
        memory = WholeDeviceMemory()
        memory.start()
        for row in rows:
            item_id = str(row["id"])
            if item_id in saved:
                continue
            prompt = adapters.prompt_for(model_id, task, row)
            record = {"id": item_id, "status": "failed", "prompt": prompt,
                      "input_record": {key: value for key, value in row.items() if not key.startswith("_")},
                      "image_width": None, "image_height": None}
            try:
                from PIL import Image
                with Image.open(row["_image_path"]) as image:
                    record["image_width"], record["image_height"] = image.size
                response = model.generate(Path(row["_image_path"]), prompt, MAX_TOKENS[model_id])
                parsed = enforce_no_target_policy(task, response["parse"], row)
                record.update({key: value for key, value in response.items() if key != "parse"})
                record["parse"] = parsed
                record["duplicate_count"] = duplicate_count(parsed)
                record["scores"] = score_item(task, parsed, row)
                if record.get("output_tokens") and record.get("latency_s"):
                    record["output_tokens_per_s"] = record["output_tokens"] / record["latency_s"]
                record["status"] = "complete"
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
            saved[item_id] = record
            task_items = [saved[str(row_["id"])] for row_ in rows if str(row_["id"]) in saved]
            result["tasks"][task] = {"items": task_items, "summary": aggregate(task, task_items)}
            result["status"] = "running"
            temporary = output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(output_path)
        memory.stop()
        result["tasks"][task]["summary"]["peak_whole_device_vram_gib"] = memory.peak_gib
        result["tasks"][task]["summary"]["gpu_memory_sample_count"] = memory.samples
        result["tasks"][task]["summary"]["gpu_memory_error"] = memory.error
        if task == "coco":
            result["tasks"][task]["summary"]["coco_official_ap"] = coco_official_ap(
                coco_annotations, rows, task_items)
    result["status"] = "complete" if all(
        result.get("tasks", {}).get(task, {}).get("summary", {}).get("completed") == len(rows_by_task[task])
        for task in tasks
    ) else "partial"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data" / "manifests")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "runs")
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=[adapters.LOCATE_ID])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--groundanything-base-url", default=adapters.GROUND_BASE_URL)
    parser.add_argument("--coco-annotations", type=Path,
                        help="local instances_val2017.json for COCOeval on the selected images")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        for model_id in MODEL_IDS:
            snapshot, revision = adapters.snapshot_for(model_id)
            print(f"{model_id}\t{revision}\t{snapshot}")
        return 0
    if len(args.models) != 1:
        parser.error("run one model per invocation; finish and release it before starting the other model")
    rows_by_task = {}
    data_provenance = {"seed": args.seed, "samples_per_task": args.samples, "tasks": {}}
    for task in args.tasks:
        manifest = args.data_dir / f"{task}.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing sampled manifest {manifest}; run prepare_data.py after preparing local data")
        rows_by_task[task] = load_manifest(manifest, task, args.samples)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("seed") != args.seed:
            raise ValueError(f"{manifest} uses seed {payload.get('seed')}, expected {args.seed}")
        data_provenance["tasks"][task] = {
            key: payload.get(key) for key in ("dataset", "split", "source_sha256", "license_or_access_terms", "sample_count")
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        safe = model_id.replace("/", "--")
        run_model(model_id, rows_by_task, args.tasks,
                  args.output_dir / f"{safe}.json", args.resume, args.groundanything_base_url,
                  args.coco_annotations, data_provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
