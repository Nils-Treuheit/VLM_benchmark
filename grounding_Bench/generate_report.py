#!/usr/bin/env python3
"""Build the grounding report and aggregate-only static-site data from runs."""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
MODEL_ORDER = ("nvidia/LocateAnything-3B", "GroundingPI/GroundAnything-VLM")
TASKS = {
    "refcocog": {"label": "Referring expression", "split": "RefCOCOg lmms-lab test derivative", "n": 100,
                "measure": "Mean IoU and Acc@IoU 0.5", "primary": "acc_iou_05"},
    "coco": {"label": "Open-vocabulary detection", "split": "COCO val2017", "n": 100,
            "measure": "COCOeval AP on sampled images with uniform detection scores; local F1@IoU 0.5", "primary": "AP"},
    "screenspot_pro": {"label": "GUI grounding", "split": "ScreenSpot-Pro test", "n": 100,
                      "measure": "Point-in-target accuracy and Acc@IoU 0.5", "primary": "point_in_target"},
    "hiertext": {"label": "Text-region detection", "split": "HierText validation", "n": 100,
                 "measure": "Word-box F1@IoU 0.5 plus matched-word OCR exact accuracy and normalized edit similarity", "primary": "f1"},
}
AUX_TASKS = {
    "vqa_count": {"label": "Counting VQA", "split": "VQAv2 validation count panel", "n": 25,
                  "measure": "VQAv2 consensus accuracy and exact canonical count", "primary": "vqa_score"},
    "vqa_truth": {"label": "Binary VQA", "split": "VQAv2 validation yes/no panel", "n": 25,
                  "measure": "VQAv2 consensus accuracy and exact canonical answer", "primary": "vqa_score"},
    "vqa_description": {"label": "Open VQA", "split": "VQAv2 validation open-answer panel", "n": 25,
                         "measure": "VQAv2 consensus accuracy and exact canonical answer", "primary": "vqa_score"},
    "textvqa": {"label": "OCR-dependent VQA", "split": "TextVQA validation", "n": 100,
                "measure": "VQAv2 consensus accuracy; requires reading text in natural images", "primary": "vqa_score"},
}
VISION_MODELS = {
    "yoloe26l-seg": "YOLOE-26-L open-vocabulary detector/segmenter",
    "yolo26l": "YOLO26-L detector",
    "yolo26l-seg": "YOLO26-L instance segmenter",
    "sam3.1": "SAM 3.1 text-prompt segmenter",
}
REFERENCES = {
    "LocateAnything-3B model card": "https://huggingface.co/nvidia/LocateAnything-3B",
    "LocateAnything evaluation": "https://github.com/NVlabs/Eagle/tree/main/Embodied",
    "LocateAnything evaluation data": "https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData",
    "GroundAnything-VLM model card": "https://huggingface.co/GroundingPI/GroundAnything-VLM",
    "RefCOCOg": "https://github.com/lichengunc/refer",
    "COCO": "https://cocodataset.org/#download",
    "ScreenSpot-Pro": "https://huggingface.co/datasets/likaixin/ScreenSpot-Pro",
    "ScreenSpot-Pro paper": "https://arxiv.org/abs/2504.07981",
    "HierText": "https://github.com/google-research-datasets/hiertext",
    "GroundAnything evaluation": "https://github.com/groundingpi/GroundAnything",
    "Ultralytics YOLO26": "https://docs.ultralytics.com/models/yolo26",
    "Ultralytics YOLOE-26": "https://docs.ultralytics.com/models/yoloe",
    "Meta SAM 3.1 release": "https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md",
    "VQAv2 dataset and terms": "https://visualqa.org/terms.html",
    "TextVQA dataset": "https://huggingface.co/datasets/facebook/textvqa",
    "VQAv2 evaluator": "https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py",
    "DOTA v1.0 dataset": "https://captain-whu.github.io/DOTA/dataset",
    "Ultralytics YOLO26 OBB and DOTA": "https://docs.ultralytics.com/tasks/obb",
    "LVIS dataset": "https://www.lvisdataset.org/",
    "LVIS official API": "https://github.com/lvis-dataset/lvis-api",
    "Roboflow 100-VL benchmark": "https://github.com/roboflow/rf100-vl",
    "KITTI object benchmark": "https://www.cvlibs.net/datasets/kitti/eval_object.php",
    "BOP datasets and BOP-H3": "https://bop.felk.cvut.cz/datasets/",
    "BOP challenge tracks": "https://bop.felk.cvut.cz/challenges/",
}


def _recorded_model_load(run: dict[str, Any]) -> tuple[float | None, str | None]:
    duration = run.get("model_load_duration_s")
    timing = run.get("model_load_timing")
    if duration is None and run.get("groundanything_service_startup_duration_s") is not None:
        duration = run["groundanything_service_startup_duration_s"]
        timing = run.get("groundanything_service_startup_timing") or timing
    return duration, timing


def load_runs() -> dict[str, dict[str, Any]]:
    result = {}
    for model_id in MODEL_ORDER:
        domain_task_metadata = {}
        path = RUNS / (model_id.replace("/", "--") + ".json")
        if path.is_file():
            result[model_id] = json.loads(path.read_text(encoding="utf-8"))
            load_duration, load_timing = _recorded_model_load(result[model_id])
            result[model_id]["model_load_duration_s"] = load_duration
            result[model_id]["model_load_timing"] = load_timing
        for run_group in ("locateany_tf457", "specialists"):
            coco_path = (HERE / "results" / "domain_runs" / "coco_val2017" /
                         run_group / (model_id.replace("/", "--") + ".json"))
            if coco_path.is_file():
                coco_run = json.loads(coco_path.read_text(encoding="utf-8"))
                result.setdefault(model_id, {"model_id": model_id, "tasks": {}})
                result[model_id].setdefault("tasks", {}).update(coco_run.get("tasks", {}))
                coco_load_duration, coco_load_timing = _recorded_model_load(coco_run)
                for task in coco_run.get("tasks", {}):
                    domain_task_metadata[task] = {
                        "model_load_duration_s": coco_load_duration,
                        "model_load_timing": coco_load_timing,
                        "max_new_tokens": coco_run.get("max_new_tokens_by_task", {}).get(task),
                        "image_token_limit": coco_run.get("image_token_limit_by_task", {}).get(task),
                        "input_size": coco_run.get("input_size"),
                        "source_run": coco_path.parent.name,
                    }
        model = result.get(model_id)
        if not model:
            continue
        task_metadata = model.setdefault("task_run_metadata", {})
        task_metadata.update(domain_task_metadata)
        model_path = path
        for task, task_run in model.get("tasks", {}).items():
            task_metadata.setdefault(task, {
                "model_load_duration_s": model.get("model_load_duration_s"),
                "max_new_tokens": model.get("max_new_tokens_by_task", {}).get(task),
                "image_token_limit": model.get("image_token_limit_by_task", {}).get(task),
                "input_size": model.get("input_size"),
                "source_run": model_path.parent.name,
            })
        for candidate_path in sorted(RUNS.glob(f"*/{model_id.replace('/', '--')}.json")):
            if candidate_path == path:
                continue
            try:
                candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for task, candidate_task in candidate.get("tasks", {}).items():
                candidate_count = int(candidate_task.get("summary", {}).get("completed", 0) or 0)
                current = model.setdefault("tasks", {}).get(task, {})
                current_count = int(current.get("summary", {}).get("completed", 0) or 0)
                if candidate_count <= current_count:
                    continue
                candidate_load_duration, candidate_load_timing = _recorded_model_load(candidate)
                model["tasks"][task] = candidate_task
                model["task_run_metadata"][task] = {
                    "model_load_duration_s": candidate_load_duration,
                    "model_load_timing": candidate_load_timing,
                    "max_new_tokens": candidate.get("max_new_tokens_by_task", {}).get(task),
                    "image_token_limit": candidate.get("image_token_limit_by_task", {}).get(task),
                    "input_size": candidate.get("input_size"),
                    "source_run": candidate_path.parent.name,
                }
    return result


def load_vision_runs() -> dict[str, dict[str, Any]]:
    result = {}
    for model_id in VISION_MODELS:
        path = HERE / "results" / "vision_runs" / f"{model_id}.json"
        if path.is_file():
            result[model_id] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_dota_run() -> dict[str, Any]:
    path = HERE / "results" / "vision_runs" / "dota-yolo26l-obb.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_domain_runs() -> dict[str, dict[str, dict[str, Any]]]:
    root = HERE / "results" / "domain_runs"
    result: dict[str, dict[str, dict[str, Any]]] = {}
    coverage: dict[tuple[str, str], tuple[int, float]] = {}
    if not root.is_dir():
        return result
    for path in sorted(root.glob("*/*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_id = str(payload.get("model_id") or path.stem.replace("--", "/"))
        domain = path.parent.parent.name
        completed = int(payload.get("sample_count", 0) or 0)
        for task in payload.get("tasks", {}).values():
            summary = task.get("summary", {})
            completed = max(completed, int(summary.get("completed", 0) or 0))
            completed = max(completed, sum(item.get("status", "complete") == "complete"
                                           for item in task.get("items", [])))
        key = (domain, model_id)
        rank = (completed, path.stat().st_mtime)
        if rank >= coverage.get(key, (-1, -1.0)):
            payload["_source_path"] = str(path.relative_to(HERE))
            result.setdefault(domain, {})[model_id] = payload
            coverage[key] = rank
    return result


def query_latency_stats(items: list[dict[str, Any]]) -> dict[str, float | None]:
    values = sorted(float(item["latency_s"]) for item in items
                    if item.get("status", "complete") == "complete" and item.get("latency_s") is not None)
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": sum(values) / len(values),
            "median": statistics.median(values),
            "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)],
            "max": max(values)}


def recorded_timing_scope(model_id: str, run: dict[str, Any]) -> str | None:
    scope = run.get("query_timing_scope")
    if model_id == "nvidia/LocateAnything-3B" and run and scope in (
            None, "model input encoding through output decoding; model load excluded"):
        return ("Historical results time only model.generate(); model load, image reading, processor encoding, "
                "output decoding, and parsing are excluded")
    return scope


def domain_summaries(runs: dict[str, dict[str, dict[str, Any]]],
                     vision_runs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    labels = {
        "yoloe26l-seg": "YOLOE-26-L",
        "yolo26l": "YOLO26-L",
        "yolo26l-seg": "YOLO26-L-seg",
        "sam3.1": "SAM 3.1",
        "nvidia/LocateAnything-3B": "LocateAnything-3B",
        "GroundingPI/GroundAnything-VLM": "GroundAnything-VLM",
    }
    summaries = []
    domain_paths = [(name, HERE / "data" / "domain_manifests" / name)
                    for name in ("coco_val2017", "dota_v1", "bop_h3_hope_v2", "lvis_v1")]
    kitti_root = HERE / "data" / "domain_manifests" / "kitti_training_2d"
    if (kitti_root / "coco.json").is_file():
        domain_paths.append(("kitti_training_2d", kitti_root))
    rf100_root = HERE / "data" / "domain_manifests" / "rf100_vl"
    if rf100_root.is_dir():
        domain_paths.extend((f"rf100_{path.name}", path) for path in sorted(rf100_root.iterdir())
                            if path.is_dir() and (path / "coco.json").is_file())
    for domain, manifest_dir in domain_paths:
        manifest_name = "manifest.json" if domain == "lvis_v1" else "coco.json"
        manifest_path = manifest_dir / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        domain_label = manifest.get("dataset", domain)
        expected = int(manifest.get("sample_count", 0))
        for model_id, run in sorted(runs.get(domain, {}).items()):
            tasks = run.get("tasks", {})
            if "object_detection" in tasks:
                detection = tasks.get("object_detection", {})
                class_aware_f1 = detection.get("class_aware_F1_IoU50")
                segmentation = tasks.get("instance_segmentation", {})
                official_lvis = run.get("lvis_official_evaluation", {})
                if official_lvis:
                    detection = official_lvis.get("object_detection", detection)
                    segmentation = official_lvis.get("instance_segmentation", segmentation)
                runtime = run.get("runtime", {})
                sample_count = int(run.get("sample_count", 0))
                timing = query_latency_stats(run.get("items", []))
                official_status = official_lvis.get("object_detection", {}).get("status")
                if domain == "lvis_v1":
                    if detection.get("status") == "unsupported":
                        status = "unsupported"
                    else:
                        status = ("complete" if sample_count >= expected and official_status == "complete"
                                  else "partial" if sample_count else "not run")
                else:
                    status = "complete" if sample_count >= expected and expected else "partial" if sample_count else "not run"
                summaries.append({
                    "dataset": domain_label, "model_id": model_id, "model": labels.get(model_id, model_id),
                    "status": status,
                    "sample_count": sample_count, "expected_count": expected,
                    "input_size": run.get("input_size"),
                    "AP": detection.get("AP"), "AP50": detection.get("AP50"),
                    "AP75": detection.get("AP75"), "AR_300": detection.get("AR_300"),
                    "class_aware_F1_IoU50": detection.get("class_aware_F1_IoU50", class_aware_f1),
                    "mask_AP": segmentation.get("AP"), "latency_s": runtime.get("latency_mean_s"),
                    "latency_median_s": runtime.get("latency_median_s", timing["median"]),
                    "latency_p95_s": timing["p95"],
                    "model_load_duration_s": runtime.get("model_load_duration_s", run.get("model_load_duration_s")),
                    "model_load_warnings": run.get("model_load_warnings", []),
                    "parse_issues": "—",
                    "query_timing_scope": runtime.get("timing_scope"),
                    "images_per_s": runtime.get("images_per_s"),
                    "AP_rare": detection.get("AP_rare"), "AP_common": detection.get("AP_common"),
                    "AP_frequent": detection.get("AP_frequent"),
                })
            else:
                task = tasks.get("coco", {})
                summary = task.get("summary", {})
                official = summary.get("coco_official_ap", {})
                sample_count = int(summary.get("completed", 0))
                timing = query_latency_stats(task.get("items", []))
                latency = timing["mean"] or summary.get("mean_latency_s")
                reportable_official = official if sample_count else {}
                summaries.append({
                    "dataset": domain_label, "model_id": model_id, "model": labels.get(model_id, model_id),
                    "status": ("complete" if sample_count >= expected and expected else
                               "partial" if int(summary.get("n_examples", sample_count)) else "not run"),
                    "sample_count": sample_count, "expected_count": expected,
                    "input_size": run.get("input_size", "native model processor"),
                    "AP": reportable_official.get("AP"), "AP50": reportable_official.get("AP50"),
                    "AP75": reportable_official.get("AP75"), "AR_300": reportable_official.get("AR_300"),
                    "class_aware_F1_IoU50": (summary.get("metrics", {}).get("micro_f1")
                                              if sample_count else None),
                    "mask_AP": None, "latency_s": latency,
                    "latency_median_s": timing["median"], "latency_p95_s": timing["p95"],
                    "model_load_duration_s": run.get("model_load_duration_s"),
                    "parse_issues": (f"{summary.get('invalid_outputs', 0)} / "
                                     f"{summary.get('empty_outputs', 0)} / "
                                     f"{summary.get('truncated', 0)}" if sample_count else "—"),
                    "query_timing_scope": recorded_timing_scope(model_id, run) or
                        "HTTP request and response; model load excluded",
                    "images_per_s": 1.0 / latency if latency else None,
                })
        if domain == "coco_val2017":
            for model_id, run in (vision_runs or {}).items():
                if any(row["dataset"] == domain_label and row["model_id"] == model_id for row in summaries):
                    continue
                tasks = run.get("tasks", {})
                runtime = run.get("runtime", {})
                timing = query_latency_stats(run.get("items", []))
                sample_count = int(run.get("sample_count", 0))
                detection = tasks.get("object_detection", {})
                segmentation = tasks.get("instance_segmentation", {})
                summaries.append({
                    "dataset": domain_label, "model_id": model_id, "model": labels.get(model_id, model_id),
                    "status": "complete" if sample_count >= expected and expected else "partial" if sample_count else "not run",
                    "sample_count": sample_count, "expected_count": expected,
                    "input_size": run.get("input_size"),
                    "AP": detection.get("AP"), "AP50": detection.get("AP50"),
                    "class_aware_F1_IoU50": detection.get("class_aware_F1_IoU50"),
                    "mask_AP": segmentation.get("AP"),
                    "latency_s": runtime.get("latency_mean_s", timing["mean"]),
                    "latency_median_s": runtime.get("latency_median_s", timing["median"]),
                    "latency_p95_s": timing["p95"],
                    "model_load_duration_s": runtime.get("model_load_duration_s"),
                    "model_load_warnings": run.get("model_load_warnings", []),
                    "query_timing_scope": runtime.get("timing_scope"),
                    "images_per_s": runtime.get("images_per_s"),
                })
    yolo_ids = {"yoloe26l-seg", "yolo26l", "yolo26l-seg"}
    fastest_yolo: dict[str, dict[str, Any]] = {}
    for result in summaries:
        if result["model_id"] in yolo_ids and result["status"] == "complete" and result.get("latency_s"):
            current = fastest_yolo.get(result["dataset"])
            if current is None or result["latency_s"] < current["latency_s"]:
                fastest_yolo[result["dataset"]] = result
    for result in summaries:
        reference = fastest_yolo.get(result["dataset"])
        result["fastest_yolo_model"] = reference["model"] if reference else None
        result["latency_ratio_to_fastest_yolo"] = (
            result["latency_s"] / reference["latency_s"]
            if result["status"] == "complete" and reference and result.get("latency_s")
            and reference.get("latency_s") else None)
    return summaries


def dota_track_summary(run: dict[str, Any]) -> dict[str, Any]:
    task = run.get("tasks", {}).get("oriented_detection", {})
    return {"status": task.get("status", "not run"), "metrics": task,
            "runtime": run.get("runtime", {}), "dataset": run.get("dataset", "DOTA v1.0 validation")}


def vision_summary(runs: dict[str, dict[str, Any]], specialists: dict[str, dict[str, Any]],
                   expected_images: int = 100) -> list[dict[str, Any]]:
    models = []
    for model_id, label in VISION_MODELS.items():
        run = runs.get(model_id, {})
        tasks = run.get("tasks", {})
        sample_count = int(run.get("sample_count", 0))
        models.append({
            "model_id": model_id, "label": label,
            "status": "complete" if sample_count >= expected_images else "partial" if sample_count else "not run",
            "sample_count": sample_count,
            "detection": tasks.get("object_detection", {}),
            "segmentation": tasks.get("instance_segmentation", {}),
            "classification": tasks.get("classification", {}),
            "grounding": tasks.get("text_grounding", {}),
            "runtime": run.get("runtime", {}),
        })
        grounding = models[-1]["grounding"]
        grounding_timing = query_latency_stats(grounding.get("items", []))
        if grounding_timing["mean"] is not None:
            grounding.update({
                "mean_latency_s": grounding_timing["mean"],
                "median_latency_s": grounding_timing["median"],
                "p95_latency_s": grounding_timing["p95"],
                "max_latency_s": grounding_timing["max"],
            })
    for model_id, label in (("nvidia/LocateAnything-3B", "LocateAnything-3B grounding specialist"),
                            ("GroundingPI/GroundAnything-VLM", "GroundAnything-VLM grounding specialist")):
        run = specialists.get(model_id, {})
        task = run.get("tasks", {}).get("coco", {})
        summary = task.get("summary", {})
        metrics = summary.get("metrics", {})
        ap = summary.get("coco_official_ap", {})
        complete = int(summary.get("completed", 0))
        model_status = "complete" if complete == 100 else "partial" if complete else "not run"
        latencies = [item.get("latency_s") for item in task.get("items", [])
                     if item.get("status") == "complete" and item.get("latency_s")]
        mean_latency = sum(latencies) / len(latencies) if latencies else None
        models.append({
            "model_id": model_id, "label": label, "status": model_status,
            "sample_count": complete,
            "detection": {"AP": ap.get("AP"), "AP50": ap.get("AP50"),
                          "class_aware_F1_IoU50": metrics.get("micro_f1")},
            "segmentation": {"status": "unsupported", "reason": "the grounding output contains boxes, not masks"},
            "classification": {}, "grounding": {},
            "runtime": {"latency_mean_s": mean_latency,
                        "latency_median_s": query_latency_stats(task.get("items", []))["median"],
                        "latency_p95_s": query_latency_stats(task.get("items", []))["p95"],
                        "images_per_s": 1 / mean_latency if mean_latency else None,
                        "peak_whole_device_vram_gib": summary.get("peak_whole_device_vram_gib")},
            "model_load_duration_s": run.get("model_load_duration_s"),
            "query_timing_scope": run.get("query_timing_scope"),
        })
    return models


def selected_metric(model: dict[str, Any], task: str) -> tuple[str | None, float | None]:
    task_result = model.get("tasks", {}).get(task, {})
    summary = task_result.get("summary", {})
    metrics = summary.get("metrics", {})
    if task == "coco":
        ap = summary.get("coco_official_ap", {})
        if ap.get("status") == "complete":
            return "COCO AP", ap.get("AP")
        return "local micro F1@IoU 0.5", metrics.get("micro_f1")
    key = TASKS[task]["primary"]
    return key, metrics.get(key)


def assemble(runs: dict[str, dict[str, Any]], vision_runs: dict[str, dict[str, Any]] | None = None,
             dota_run: dict[str, Any] | None = None,
             domain_runs: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    models = []
    for model_id in MODEL_ORDER:
        run = runs.get(model_id)
        task_status = {}
        all_task_info = {**TASKS, **AUX_TASKS}
        for task, task_info in all_task_info.items():
            task_result = run.get("tasks", {}).get(task, {}) if run else {}
            summary = task_result.get("summary", {})
            task_metadata = (run or {}).get("task_run_metadata", {}).get(task, {})
            completed = summary.get("completed", 0)
            if task in TASKS:
                score_name, primary_score = selected_metric(run, task) if run else (None, None)
            else:
                score_name = task_info["primary"]
                primary_score = summary.get("metrics", {}).get(score_name)
            task_status[task] = {
                "status": ("complete" if completed == task_info["n"] else
                           "partial" if completed else "not run"),
                "n_examples": summary.get("n_examples", task_info["n"] if run else 0),
                "completed": completed,
                "metrics": summary.get("metrics", {}),
                "primary_metric": score_name,
                "primary_score": primary_score,
                "coco_official_ap": summary.get("coco_official_ap"),
                "mean_latency_s": summary.get("mean_latency_s"),
                "median_latency_s": summary.get("median_latency_s") or query_latency_stats(task_result.get("items", []))["median"],
                "p95_latency_s": summary.get("p95_latency_s") or query_latency_stats(task_result.get("items", []))["p95"],
                "max_latency_s": query_latency_stats(task_result.get("items", []))["max"],
                "model_load_duration_s": (summary.get("model_load_duration_s") or
                                          task_metadata.get("model_load_duration_s")),
                "image_token_limit": task_metadata.get("image_token_limit"),
                "max_new_tokens": task_metadata.get("max_new_tokens"),
                "task_input_size": task_metadata.get("input_size"),
                "source_run": task_metadata.get("source_run"),
                "truncated_query_latency_range_s": (
                    [min(item["latency_s"] for item in task_result.get("items", [])
                         if item.get("finish_reason") == "length" or item.get("truncated") is True),
                     max(item["latency_s"] for item in task_result.get("items", [])
                         if item.get("finish_reason") == "length" or item.get("truncated") is True)]
                    if any(item.get("finish_reason") == "length" or item.get("truncated") is True
                           for item in task_result.get("items", [])) else None
                ),
                "mean_output_tokens_per_s": _mean_output_rate(task_result.get("items", [])),
                "peak_whole_device_vram_gib": summary.get("peak_whole_device_vram_gib"),
                "malformed": summary.get("malformed", 0),
                "invalid_outputs": summary.get("invalid_outputs", 0),
                "empty_outputs": summary.get("empty_outputs", 0),
                "truncated": summary.get("truncated", 0),
                "duplicate_predictions": summary.get("duplicate_predictions", 0),
            }
        complete = all(task_status[task]["status"] == "complete" for task in TASKS)
        component_scores = [task_status[task]["primary_score"] for task in TASKS]
        local_macro = sum(component_scores) / len(component_scores) if complete and all(v is not None for v in component_scores) else None
        models.append({
            "model_id": model_id,
            "revision": run.get("revision") if run else None,
            "backend": run.get("backend") if run else None,
            "precision": run.get("precision") if run else None,
            "generation_mode": run.get("generation_mode") if run else None,
            "model_load_peak_whole_device_vram_gib": run.get("model_load_peak_whole_device_vram_gib") if run else None,
            "model_load_duration_s": run.get("model_load_duration_s") if run else None,
            "model_load_timing": run.get("model_load_timing") if run else None,
            "query_timing_scope": (recorded_timing_scope(model_id, run) if run else None) or
                ("HTTP request through complete response; model load excluded" if model_id == "GroundingPI/GroundAnything-VLM"
                 else "full per-query timing is recorded after the next run" if run else None),
            "package_versions": run.get("client_package_versions") if run else {},
            "status": "complete" if complete else
                "partial" if run and any(value["completed"] for value in task_status.values()) else "not run",
            "complete_suite": complete,
            "local_macro": local_macro,
            "tasks": task_status,
        })
    complete_models = [m for m in models if m["complete_suite"] and m["local_macro"] is not None]
    winner = max(complete_models, key=lambda m: m["local_macro"], default=None)
    task_winners = {}
    for task in TASKS:
        candidates = [m for m in models if m["tasks"][task]["status"] == "complete"
                      and m["tasks"][task]["primary_score"] is not None]
        task_winners[task] = max(candidates, key=lambda m: m["tasks"][task]["primary_score"], default=None)
    tested_model_count = sum(1 for m in models if any(t["completed"] for t in m["tasks"].values()))
    local_manifests = HERE / "data" / "manifests"
    shared_coco_manifest = HERE / "data" / "domain_manifests" / "coco_val2017" / "coco.json"
    prepared_tasks = [task for task in (*TASKS, *AUX_TASKS)
                      if (local_manifests / f"{task}.json").is_file()
                      or (task == "coco" and shared_coco_manifest.is_file())]
    prepared_aux_manifests = all((local_manifests / f"{task}.json").is_file() for task in AUX_TASKS)
    prepared_any_manifests = any((local_manifests / f"{task}.json").is_file()
                                 for task in (*TASKS, *AUX_TASKS))
    dota_summary = dota_track_summary(dota_run or {})
    dataset_tracks = [
        {"dataset": "COCO val2017", "status": "100-image shared manifest; YOLOE, YOLO26, SAM 3.1, LocateAnything, and GroundAnything runs complete",
         "task": "Shared detection, classification, instance segmentation, and speed",
         "note": "Every available vision baseline uses the same 100 annotated images and all 80 COCO classes. Grounding specialists are scored on boxes; models without confidence scores use uniform scores for COCO AP."},
        {"dataset": "DOTA v1.0 validation", "status": "Shared HBB runs complete for YOLOE, YOLO26-L, SAM 3.1, and GroundAnything; LocateAnything native-input attempt partial (51/100), 4,096-patch recovery 100/100 with 41 parser-valid; full-split YOLO26-L-OBB complete",
         "task": "Shared aerial detection plus native OBB for YOLO26-L",
         "note": "100 of 456 labeled validation images are sampled for horizontal-box COCO AP across YOLOE, SAM 3.1, and available grounding specialists. GroundAnything completed all 100 items with five outputs truncated at 2,048 tokens; official COCO AP was unavailable because three predicted labels did not map to annotation categories. LocateAnything's native-input 2,048-token run completed 51/100 before 49 CUDA OOM failures; its 51 completed items include 33 parser-valid outputs and yield COCOeval AP 0.046%. A separate run caps the image processor at 4,096 patches and completes 100/100, with 41 parser-valid outputs; 62 responses were truncated at 2,048 tokens. COCOeval AP is unavailable because one predicted label did not map to a category. Keep the two LocateAnything settings separate. YOLO26-L-OBB uses the full 458-file validation split, including two background-only images, for the separate native OBB reference. Its clean rerun recorded 8.6 ms/image and 2.351 GiB peak whole-device VRAM. DOTA terms restrict use to academic purposes and prohibit commercial use."},
        {"dataset": "BOP-H3 HOPE v2 validation", "status": "50-image shared manifest; LocateAnything, GroundAnything, SAM 3.1, and YOLOE complete; closed-vocabulary YOLO26 omitted because its COCO label set does not cover HOPE classes",
         "task": "Object-centric 2D detection; instance masks where supported",
         "note": "All 50 local validation images are shared by YOLOE, SAM 3.1, and both grounding specialists. GroundAnything's official COCO AP is unavailable because one predicted label did not map to the annotation categories; its class-aware F1 remains reportable. Mask AP is shown only for models that return masks. This is the 2D HOPE v2 subset, not the full BOP-H3 pose/onboarding protocol."},
        {"dataset": "KITTI object detection", "status": "Deprioritized; local training-panel inference complete",
         "task": "Automotive 2D detection",
         "note": "The fixed 100-image panel is sampled from a deterministic 80/20 partition of labeled training images and uses Car, Pedestrian, and Cyclist boxes. It excludes Van, Person_sitting, Truck, Tram, Misc, and DontCare labels; its COCO-shaped annotations support local comparative AP/F1, not official KITTI AP_R40. LocateAnything used a 2,048-token response cap; 98 outputs reached the cap, and one unmapped label prevented official COCOeval AP. GroundAnything query latency excludes the approximately 10-second shared service startup. Public test labels are unavailable. Per scope, KITTI remains a local exploratory panel and has no remaining benchmark work or official AP_R40 completion gate."},
        {"dataset": "LVIS v1 validation", "status": "GroundAnything, LocateAnything, SAM 3.1, and YOLOE-26-L complete at 100/100; YOLO26-L and YOLO26-L-seg have no supported detections for the selected categories",
         "task": "Long-tail detection and instance segmentation",
         "note": "This exploratory panel uses 10 rare, 10 common, and 10 frequent categories across 100 fixed LVIS val images sourced from COCO 2017 train/val. It uses official LVIS AP and frequency-group metrics, not the full 1,203-category leaderboard; COCO-trained baselines may have seen some train images. YOLO26-L and YOLO26-L-seg were run on the same panel, but their fixed COCO vocabulary produced no detections for these selected long-tail categories, so LVIS AP is unsupported."},
        {"dataset": "RF100-VL", "status": "Current all-subset comparison is built from the highest-coverage saved run for each model and subset",
         "task": "Multi-domain open-vocabulary detection",
         "note": "Per-subset panels use up to 20 shared seeded test images. Only models with complete coverage across all local subsets appear in the current macro comparison. The local test exports contain boxes but no masks; mask AP is unsupported. Closed-vocabulary results are limited to panels with matching labels."},
        {"dataset": "BOP-H3 HOT3D/HANDAL", "status": "separate follow-up",
         "task": "Unseen-object detection, onboarding, and 6D pose",
         "note": "Keep separate from class-name box AP. Use the official BOP protocol for pose tasks and review each dataset's terms before expanding beyond local HOPE v2."},
    ]
    domain_results = domain_summaries(domain_runs or {}, vision_runs or {})
    rf_rows = [row for row in domain_results if row["dataset"].startswith("RF100-VL:")]
    rf_panels = sorted({row["dataset"] for row in rf_rows})
    rf_panel_total = sum(1 for path in (HERE / "data" / "domain_manifests" / "rf100_vl").glob("*/coco.json"))
    all_rf_complete = False
    rf_summary: dict[str, Any] = {"panel_count": len(rf_panels), "panel_total": rf_panel_total, "models": []}
    current_rf_models: set[str] = set()
    if rf_panels:
        required_rf_models = ("nvidia/LocateAnything-3B", "GroundingPI/GroundAnything-VLM", "yoloe26l-seg")
        completed = {model: sum(row["status"] == "complete" for row in rf_rows if row["model_id"] == model)
                     for model in required_rf_models}
        rf_track = next(row for row in dataset_tracks if row["dataset"] == "RF100-VL")
        all_rf_complete = rf_panel_total == 100 and all(count == rf_panel_total for count in completed.values())
        current_rf_models = {model for model, count in completed.items() if count == rf_panel_total and rf_panel_total}
        coverage = ", ".join(f"{model}: {completed[model]}/{rf_panel_total}" for model in required_rf_models)
        rf_track["status"] = (f"All-subset runs complete across {rf_panel_total} panels" if all_rf_complete
                              else f"Expanded local panel coverage recorded; required-model completion: {coverage}")
        rf_track["note"] = (
            f"The local test export contains {rf_panel_total} subsets. Per-subset panels use up to 20 shared "
            "seeded test images. " + coverage + ". All local test exports contain bounding boxes but no "
            "instance masks, including subsets whose names mention segmentation; mask AP is unsupported. "
            "Closed-vocabulary YOLO26 results remain limited to panels with matching labels."
            + (" Live all-subset run status is tracked in PLAN.md." if not all_rf_complete else "")
        )
        # Keep the published RF100 table focused on the current, comparable all-subset run set.
        if current_rf_models:
            rf_rows = [row for row in rf_rows if row["model_id"] in current_rf_models]
            domain_results = [row for row in domain_results
                              if not row["dataset"].startswith("RF100-VL:") or row["model_id"] in current_rf_models]

        def macro(model_rows: list[dict[str, Any]], key: str) -> float | None:
            values = [float(row[key]) for row in model_rows
                      if row["status"] == "complete" and row.get(key) is not None]
            return sum(values) / len(values) if values else None

        rf_summary["models"] = []
        for model_id in sorted(current_rf_models):
            model_rows = [row for row in rf_rows if row["model_id"] == model_id]
            complete_rows = [row for row in model_rows if row["status"] == "complete"]
            rf_summary["models"].append({
                "model_id": model_id,
                "model": model_rows[0]["model"],
                "complete_panels": len(complete_rows),
                "ap_panels": sum(row.get("AP") is not None for row in complete_rows),
                "macro_AP": macro(model_rows, "AP"),
                "f1_panels": sum(row.get("class_aware_F1_IoU50") is not None for row in complete_rows),
                "macro_F1": macro(model_rows, "class_aware_F1_IoU50"),
            })
        date_values = []
        for model_id in current_rf_models:
            for payload in (domain_runs or {}).values():
                run = payload.get(model_id, {})
                if "rf100_" not in run.get("_source_path", ""):
                    continue
                match = re.search(r"20\d{6}", run["_source_path"])
                if match:
                    date_values.append(match.group(0))
        if date_values:
            latest = max(date_values)
            rf_summary["latest_result_date"] = f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if winner and all_rf_complete else "not run" if not runs else "partial",
        "tested_model_count": tested_model_count,
        "data_status": ("core tasks complete; expanded RF100-VL coverage pending"
                        if models and all(m["complete_suite"] for m in models) and not all_rf_complete else
                        "all core and expanded RF100-VL tasks have recorded results"
                        if models and all(m["complete_suite"] for m in models) and all_rf_complete else
                        "some prepared tasks have recorded runs" if runs else
                       "local auxiliary manifests prepared" if prepared_aux_manifests else
                       "some local manifests prepared" if prepared_any_manifests else
                       "not present in inspected cache"),
        "prepared_tasks": prepared_tasks,
        "candidate_model_count": len(MODEL_ORDER),
        "sample_plan": {"grounding_each": 100, "vqa_each": 25, "textvqa": 100, "vision_each": 100},
        "seed": 3407,
        "models": models,
        "best_complete": winner,
        "task_winners": {task: (task_winner["model_id"] if task_winner else None)
                         for task, task_winner in task_winners.items()},
        "runtime_summary": "; ".join(sorted({
            ", ".join(value for value in (m.get("backend"), m.get("precision"), m.get("generation_mode")) if value)
            for m in models if m.get("backend")
        })) or "Not run",
        "tasks": TASKS,
        "aux_tasks": AUX_TASKS,
        "vision": vision_summary(vision_runs or {}, runs, expected_images=100),
        "dota_track": dota_summary,
        "dataset_tracks": dataset_tracks,
        "domain_benchmarks": domain_results,
        "rf100_summary": rf_summary,
        "latest_rf100_result_date": rf_summary.get("latest_result_date"),
        "references": REFERENCES,
    }


def _mean_output_rate(items: list[dict[str, Any]]) -> float | None:
    values = [float(item["output_tokens_per_s"]) for item in items
              if item.get("status") == "complete" and item.get("output_tokens_per_s") is not None]
    return sum(values) / len(values) if values else None


def fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if percent else f"{value:.3f}"


def write_report(data: dict[str, Any]) -> None:
    lines = [
        "# Grounding Specialist Benchmark Report", "",
        f"**Evaluation status:** {data['status']}", "",
        f"**Generated:** {data['generated_at']}", "",
        "## Scope", "",
        "This report combines grounding-specialist localization and answer tasks with large vision-model baselines. "
        "Shared domain runs use the same image IDs and category labels across prompt-capable models. "
        "Detection, segmentation, grounding, OCR, and VQA remain separate measurements.", "",
        "| Task | Dataset and split | Sample | Primary measures |",
        "|---|---|---:|---|",
    ]
    for task in TASKS.values():
        lines.append(f"| {task['label']} | {task['split']} | {task['n']} | {task['measure']} |")
    plan = data["sample_plan"]
    lines.extend(["", f"Sample targets: {plan['grounding_each']} per grounding task, {plan['vqa_each']} per VQAv2 panel, "
                  f"{plan['textvqa']} TextVQA questions, and {plan['vision_each']} COCO images; seed {data['seed']}.",
                  "", "## Model coverage", "",
                  "| Model | Revision | Backend | Precision | Mode | Status | Complete tasks |",
                  "|---|---|---|---|---|---|---:|"])
    for model in data["models"]:
        done = sum(model["tasks"][task]["completed"] for task in TASKS)
        lines.append(f"| `{model['model_id']}` | `{model['revision'] or 'not run'}` | "
                     f"{model['backend'] or 'not run'} | {model['precision'] or '—'} | "
                     f"{model['generation_mode'] or '—'} | {model['status']} | {done}/400 |")
    lines.extend(["", "Runner/library versions by model:", ""])
    for model in data["models"]:
        versions = ", ".join(f"{name} {value}" for name, value in model["package_versions"].items() if value)
        lines.append(f"- `{model['model_id']}`: {versions or 'not recorded'}")
    lines.extend(["", "## Task results", ""])
    for task_id, task in TASKS.items():
        lines.extend([f"### {task['label']}", "",
                      f"Dataset: {task['split']}; target sample: {task['n']} examples. "
                      f"Measures: {task['measure']}.", "",
                      "| Model | Status | Coverage | Scores | Mean / median / p95 / max query (s) | Model load (s) | Output rate | Peak whole-device VRAM | Parse issues |",
                      "|---|---|---:|---|---:|---:|---:|---:|---:|"])
        for model in data["models"]:
            task_result = model["tasks"][task_id]
            metrics = task_result["metrics"]
            if task_id == "refcocog":
                score_text = f"mean IoU {fmt(metrics.get('iou'), True)}; Acc@0.5 {fmt(metrics.get('acc_iou_05'), True)}"
            elif task_id == "coco":
                ap = task_result.get("coco_official_ap") or {}
                score_text = (f"AP {fmt(ap.get('AP'), True)}; AP50 {fmt(ap.get('AP50'), True)}"
                              if ap.get("status") == "complete" else
                              f"local F1@0.5 {fmt(metrics.get('micro_f1'), True)}")
            elif task_id == "screenspot_pro":
                score_text = f"point-in-target {fmt(metrics.get('point_in_target'), True)}; Acc@0.5 {fmt(metrics.get('acc_iou_05'), True)}"
            else:
                score_text = (f"Box F1 {fmt(metrics.get('f1'), True)}; OCR exact {fmt(metrics.get('ocr_exact_accuracy'), True)}; "
                              f"NED {fmt(metrics.get('ocr_normalized_edit_similarity'), True)}")
            lines.append(f"| `{model['model_id']}` | {task_result['status']} | {task_result['completed']}/{task['n']} | "
                         f"{score_text} | {fmt(task_result['mean_latency_s'])} / {fmt(task_result['median_latency_s'])} / "
                         f"{fmt(task_result['p95_latency_s'])} / {fmt(task_result['max_latency_s'])} | "
                         f"{fmt(task_result['model_load_duration_s'])} | "
                         f"{fmt(task_result['mean_output_tokens_per_s'])} tok/s | "
                         f"{fmt(task_result['peak_whole_device_vram_gib'])} GiB | "
                      f"{task_result['malformed']} malformed, {task_result['invalid_outputs']} invalid, {task_result['truncated']} truncated, "
                         f"{task_result['empty_outputs']} empty, {task_result['duplicate_predictions']} duplicate |")
        lines.append("")
        if task_id == "refcocog":
            for model in data["models"]:
                result = model["tasks"][task_id]
                slow_range = result.get("truncated_query_latency_range_s")
                if slow_range:
                    lines.append(
                        f"LocateAnything had {result['truncated']} truncated RefCOCOg outputs at its 8,192-token cap; "
                        f"their query times ranged from {fmt(slow_range[0])} to {fmt(slow_range[1])} s. "
                        "These tail requests exceed the nearest-rank p95 and are visible in the max query column."
                    )
                elif model["model_id"] == "GroundingPI/GroundAnything-VLM" and result.get("max_latency_s", 0) > 10:
                    lines.append(
                        f"GroundAnything's slowest RefCOCOg query took {fmt(result['max_latency_s'])} s; "
                        "the mean includes it, while p95 excludes the slowest five percent."
                    )
            lines.append("")
        if task_id == "screenspot_pro":
            lines.append(
                "The fixed ScreenSpot-Pro sample contains 100 examples selected with seed 3407 from all 1,581 official test annotations; the full 1,588-image repository is stored under the public-datasets tree. One annotation extended one pixel above its screenshot and was clipped to the visible image boundary before normalization. LocateAnything's successful run used Transformers 4.57.1 and a 4,096-patch image cap; its earlier Transformers 5.19.0 attempt failed before inference and remains preserved separately."
            )
            lines.append("")
        if task_id == "hiertext":
            for model in data["models"]:
                result = model["tasks"][task_id]
                if result.get("image_token_limit"):
                    lines.append(
                        f"LocateAnything's complete HierText run capped image patches at "
                        f"{result['image_token_limit']:,} and allowed up to "
                        f"{result.get('max_new_tokens') or 8_192:,} output tokens. "
                        "A native 25,600-patch attempt is preserved separately at 99/100 after one CUDA OOM."
                    )
            if any(model["tasks"][task_id].get("image_token_limit") for model in data["models"]):
                lines.append("")
    lines.extend([
        "## OCR, VQA, and counting", "",
        "These answer-generation tasks are reported separately from localization and object detection. "
        "TextVQA measures question answering that depends on reading scene text. HierText adds word recognition "
        "to the existing text-box task by matching recognized words to word boxes at IoU 0.5.", "",
        "| Model | Task | Status | Coverage | VQAv2 consensus / exact | Mean / median query (s) | p95 query (s) | Output rate | Peak VRAM |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model in data["models"]:
        for task_id, task in AUX_TASKS.items():
            result = model["tasks"][task_id]
            metrics = result["metrics"]
            score = fmt(metrics.get("vqa_score"), True)
            if task_id == "vqa_count":
                score += f" / count exact {fmt(metrics.get('count_exact_match'), True)}"
            else:
                score += f" / canonical exact {fmt(metrics.get('exact_match'), True)}"
            lines.append(f"| `{model['model_id']}` | {task['label']} | {result['status']} | "
                         f"{result['completed']}/{task['n']} | {score} | "
                         f"{fmt(result['mean_latency_s'])} / {fmt(result['median_latency_s'])} | "
                         f"{fmt(result['p95_latency_s'])} | {fmt(result['mean_output_tokens_per_s'])} tok/s | "
                         f"{fmt(result['peak_whole_device_vram_gib'])} GiB |")
    lines.extend([
        "", "The VQAv2 panel reuses the exact 25-question sample from the general VLM skill panel. TextVQA uses a deterministic 100-question "
        "sample from the cached validation split. Scores use the 10-human-answer consensus metric.",
        "", "## Object detection, classification, and segmentation", "",
    ])
    lines.extend([
        "This COCO val2017 scorecard uses open-vocabulary YOLOE-26-L as the primary detector/segmenter, "
        "with closed-vocabulary YOLO26-L detection and segmentation plus SAM 3.1 as additional references. "
        "Detection and segmentation use COCO AP. Classification is label accuracy among predictions matched "
        "to any ground-truth object at box IoU 0.5; it is not whole-image classification accuracy.", "",
        "| Model | Status | Images | Box AP / AP50 | Mask AP / AP50 | Label accuracy / joint / class F1 | RefCOCOg Acc@0.5 / mean, median, p95, max query s / model load s | Mean / median COCO query (s) | Images/s | Peak VRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model in data["vision"]:
        detection = model["detection"] or {}
        segmentation = model["segmentation"] or {}
        classification = model["classification"] or {}
        grounding = model["grounding"] or {}
        runtime = model["runtime"] or {}
        grounding_cell = (
            f"{fmt(grounding.get('acc_iou_05'), True)} / {fmt(grounding.get('mean_latency_s'))}, "
            f"{fmt(grounding.get('median_latency_s'))}, {fmt(grounding.get('p95_latency_s'))}, "
            f"{fmt(grounding.get('max_latency_s'))} / "
            f"{fmt(runtime.get('model_load_duration_s', model.get('model_load_duration_s')))}"
            if grounding.get("sample_count") else "—"
        )
        lines.append(f"| `{model['model_id']}` | {model['status']} | {model['sample_count']} | "
                     f"{fmt(detection.get('AP'), True)} / {fmt(detection.get('AP50'), True)} | "
                     f"{fmt(segmentation.get('AP'), True)} / {fmt(segmentation.get('AP50'), True)} | "
                     f"{fmt(classification.get('accuracy_on_matched_objects'), True)} / "
                     f"{fmt(classification.get('joint_detection_and_classification_accuracy'), True)} / "
                     f"{fmt(detection.get('class_aware_F1_IoU50'), True)} | "
                     f"{grounding_cell} | "
                     f"{fmt(runtime.get('latency_mean_s'))} / {fmt(runtime.get('latency_median_s'))} | {fmt(runtime.get('images_per_s'))} | "
                     f"{fmt(runtime.get('peak_whole_device_vram_gib'))} GiB |")
    vision_by_id = {model["model_id"]: model for model in data["vision"]}
    locate = vision_by_id.get("nvidia/LocateAnything-3B", {})
    ground = vision_by_id.get("GroundingPI/GroundAnything-VLM", {})
    yoloe = vision_by_id.get("yoloe26l-seg", {})
    yolo = vision_by_id.get("yolo26l", {})
    comparison_models = (locate, ground, yoloe, yolo)
    if all(model.get("status") == "complete" for model in comparison_models):
        yoloe_time = yoloe.get("runtime", {}).get("latency_mean_s")
        comparison = []
        for model in (locate, ground):
            latency = model.get("runtime", {}).get("latency_mean_s")
            ratio = latency / yoloe_time if latency is not None and yoloe_time else None
            comparison.append((model, latency, ratio))
        locate_grounding = next((model for model in data["models"]
                                 if model["model_id"] == "nvidia/LocateAnything-3B"), {}).get("tasks", {}).get("refcocog", {})
        ground_grounding = next((model for model in data["models"]
                                 if model["model_id"] == "GroundingPI/GroundAnything-VLM"), {}).get("tasks", {}).get("refcocog", {})
        vision_grounding = {model["model_id"]: model for model in data["vision"]}
        yoloe_grounding = vision_grounding.get("yoloe26l-seg", {}).get("grounding", {})
        if yoloe_grounding.get("sample_count") == data["sample_plan"]["grounding_each"]:
            loc_iou = locate_grounding.get("metrics", {}).get("acc_iou_05")
            gr_iou = ground_grounding.get("metrics", {}).get("acc_iou_05")
            y_iou = yoloe_grounding.get("acc_iou_05")
            loc_time = locate_grounding.get("mean_latency_s")
            gr_time = ground_grounding.get("mean_latency_s")
            y_time = yoloe_grounding.get("mean_latency_s")
            lines.extend([
                "", "### Accuracy and query-time comparison", "",
                f"On the shared 100-image COCO panel, YOLOE-26-L reached {fmt(yoloe.get('detection', {}).get('AP'), True)} box AP "
                f"at {fmt(yoloe_time)} s/query. LocateAnything reached {fmt(locate.get('detection', {}).get('AP'), True)} AP "
                f"at {fmt(comparison[0][1])} s/query ({fmt(comparison[0][2])}× the YOLOE query time), and GroundAnything "
                f"reached {fmt(ground.get('detection', {}).get('AP'), True)} AP at {fmt(comparison[1][1])} s/query "
                f"({fmt(comparison[1][2])}×). Neither grounding VLM improved COCO AP on this panel. Closed-vocabulary "
                f"YOLO26-L reached {fmt(yolo.get('detection', {}).get('AP'), True)} AP at "
                f"{fmt(yolo.get('runtime', {}).get('latency_mean_s'))} s/query. These COCO AP values use uniform VLM box scores, "
                "and the per-model query timing boundaries differ as described below.",
            ])
            if all(value is not None for value in (loc_iou, gr_iou, y_iou, loc_time, gr_time, y_time)):
                lines.append(
                    f"On the adapted RefCOCOg panel, YOLOE-26-L reached {fmt(y_iou, True)} Acc@IoU 0.5 at "
                    f"{fmt(y_time)} s/query; LocateAnything reached {fmt(loc_iou, True)} at {fmt(loc_time)} s/query, "
                    f"and GroundAnything reached {fmt(gr_iou, True)} at {fmt(gr_time)} s/query. The VLMs did not "
                    "exceed the YOLOE score on this panel. Cold-start load is separate: it was captured for these "
                    "RefCOCOg runs, while the older VLM COCO runs have no recoverable model-load duration."
                )
    lines.extend([
        "", "YOLOE-26 and SAM 3.1 receive the same full COCO category-name list per image. SAM latency includes "
        "one text-prompt query per category; YOLOE latency includes prompt setup and one image pass. YOLO26 "
        "uses its fixed COCO vocabulary. COCO detection and instance-segmentation AP use the official evaluator. "
        "Optional RefCOCOg text grounding reports mean IoU and Acc@IoU 0.5 for YOLOE and SAM. "
        "Mean, median, p95, maximum query latency, and initial model or service load duration are reported separately. "
        "The GroundAnything load value for this run measures server process start through API readiness, including imports, checkpoint loading, and KV-cache allocation. "
        "Throughput is single-image serial inference; model load time is excluded from query latency. The grounding specialists "
        "do not provide confidence scores, so their COCO AP uses uniform scores and is less informative than "
        "class-aware F1 for direct comparison. No overall rank combines these metrics.",
        "", "## Selection guide", "",
        (f"Best complete grounding specialist: `{data['best_complete']['model_id']}` with local task macro "
         f"{fmt(data['best_complete']['local_macro'], True)}." if data["best_complete"] else
         "No complete model is eligible for a quality recommendation yet."),
        "The local task macro weights the four task scores equally. RefCOCOg uses Acc@IoU 0.5, "
        "COCO uses sampled-image COCO AP when available and otherwise local micro F1@IoU 0.5, "
        "GUI uses point-in-target accuracy, and HierText uses detection F1. It is a local comparison metric.",
        "", "| Task | Best grounding specialist | Primary score |",
        "|---|---|---:|",
    ])
    for task_id, task in TASKS.items():
        winner_id = data["task_winners"].get(task_id)
        winner_model = next((model for model in data["models"] if model["model_id"] == winner_id), None)
        score = winner_model["tasks"][task_id]["primary_score"] if winner_model else None
        label = winner_model["tasks"][task_id]["primary_metric"] if winner_model else "no complete result"
        lines.append(f"| {task['label']} | `{winner_id}` | {label}: {fmt(score, True)} |" if winner_id else
                     f"| {task['label']} | No complete result | — |")
    lines.extend([
        "", "## Additional dataset tracks", "",
        "COCO remains the shared natural-image anchor. These tracks add different evaluation pressure; "
        "their metrics are kept separate from COCO and from each other.", "",
        "| Dataset | Status | Task | Fit and limitation |",
        "|---|---|---|---|",
    ])
    for track in data["dataset_tracks"]:
        lines.append(f"| {track['dataset']} | {track['status']} | {track['task']} | {track['note']} |")
    lines.extend([
        "", "### Shared non-COCO domain runs", "",
        "Every row uses the prepared image IDs and category list for its domain. Query latency excludes model load. YOLO timing includes image loading, preprocessing, and prompt setup; GroundAnything timing in the saved runs starts after local image reading, base64 encoding, and request serialization, then covers the HTTP call through the full response. Historical LocateAnything timings cover only model.generate(), excluding image reading, processor encoding, decoding, and parsing. The ratios compare each model with the fastest complete YOLO detector or segmenter on that domain, but the saved timing scopes differ.",
        "", "| Dataset | Model | Status | Images | Model load s | Input size | Box AP / AP50 / AP75 | AR@300 | Class-aware F1 | Mask AP | AP rare / common / frequent | Mean query s | Median query s | p95 query s | Parse invalid / empty / truncated | Images/s | Latency / fastest YOLO |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|",
    ])
    for result in data["domain_benchmarks"]:
        lines.append(
            f"| {result['dataset']} | {result['model']} | {result['status']} | "
            f"{result['sample_count']}/{result['expected_count']} | {fmt(result.get('model_load_duration_s'))} | "
            f"{result.get('input_size') or '—'} | "
            f"{fmt(result.get('AP'), True)} / {fmt(result.get('AP50'), True)} / {fmt(result.get('AP75'), True)} | "
            f"{fmt(result.get('AR_300'), True)} | "
            f"{fmt(result.get('class_aware_F1_IoU50'), True)} | {fmt(result.get('mask_AP'), True)} | "
            f"{fmt(result.get('AP_rare'), True)} / {fmt(result.get('AP_common'), True)} / "
            f"{fmt(result.get('AP_frequent'), True)} | "
            f"{fmt(result.get('latency_s'))} | {fmt(result.get('latency_median_s'))} | "
            f"{fmt(result.get('latency_p95_s'))} | {result.get('parse_issues', '—')} | "
            f"{fmt(result.get('images_per_s'))} | "
            f"{fmt(result.get('latency_ratio_to_fastest_yolo'))}× |"
        )
    rf_rows = [row for row in data["domain_benchmarks"] if row["dataset"].startswith("RF100-VL:")]
    rf_models = sorted({row["model_id"] for row in rf_rows})
    rf_panel_total = sum(1 for path in (HERE / "data" / "domain_manifests" / "rf100_vl").glob("*/coco.json"))
    if rf_rows:
        lines.extend([
            "", "#### RF100-VL macro comparison", "",
            f"Each macro metric averages reportable values from complete per-subset panels, each containing up to 20 shared images. Coverage is shown against all {rf_panel_total} local subsets; a metric that is unsupported for one panel omits that panel. Mean query time averages per-domain means equally and excludes model initialization. Latency ratios compare each model with the fastest complete YOLO detector or segmenter on the same panel. LocateAnything's all-subset run uses the 4,096-patch image cap and 2,048-token output cap; earlier native-input OOM attempts remain preserved. AP from panels with zero successful inference is unavailable.",
            "", "| Model | Complete panels | AP panels | Macro box AP | F1 panels | Macro class-aware F1 | Mean query s | Mean model load s | Mean ratio to fastest YOLO |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for model_id in rf_models:
            group = [row for row in rf_rows if row["model_id"] == model_id]
            complete = [row for row in group if row["status"] == "complete"]
            def metric_values(key: str) -> list[float]:
                return [float(row[key]) for row in complete if row.get(key) is not None]
            def mean_metric(key: str) -> float | None:
                values = metric_values(key)
                return sum(values) / len(values) if values else None
            lines.append(
                f"| {group[0]['model']} | {len(complete)}/{rf_panel_total} | {len(metric_values('AP'))}/{rf_panel_total} | "
                f"{fmt(mean_metric('AP'), True)} | {len(metric_values('class_aware_F1_IoU50'))}/{rf_panel_total} | "
                f"{fmt(mean_metric('class_aware_F1_IoU50'), True)} | {fmt(mean_metric('latency_s'))} | "
                f"{fmt(mean_metric('model_load_duration_s'))} | "
                f"{fmt(mean_metric('latency_ratio_to_fastest_yolo'))}× |"
            )
    if not data["domain_benchmarks"]:
        lines.append("| DOTA v1.0 and BOP-H3 HOPE v2 | All shared models | not run | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.append("For LVIS official AP, both grounding specialists receive a uniform confidence score of 1.0 because neither emits per-box confidence; YOLOE and SAM use their model scores. Treat specialist AP comparisons as quality references rather than confidence-calibrated rankings.")
    for result in data["domain_benchmarks"]:
        for warning in result.get("model_load_warnings", []):
            lines.append(f"Model-load warning for {result['model']} on {result['dataset']}: {warning}")
    dota = data["dota_track"]
    dota_metrics = dota.get("metrics", {})
    dota_runtime = dota.get("runtime", {})
    dota_latency = ("pending clean serial rerun" if not dota_runtime.get("inference_ms_per_image")
                    else f"{fmt(dota_runtime.get('inference_ms_per_image'))} ms/image")
    lines.extend([
        "", "### DOTA v1.0 result", "",
        f"Status: {dota.get('status', 'not run')}. YOLO26-L-OBB validation AP: "
        f"{fmt(dota_metrics.get('AP'), True)}; AP50: {fmt(dota_metrics.get('AP50'), True)}; "
        f"inference latency: {dota_latency}; "
        f"whole-device peak VRAM: {fmt(dota_runtime.get('peak_whole_device_vram_gib'))} GiB.",
        "This is an in-domain OBB reference, not a zero-shot YOLOE comparison. Validation speed excludes model load; peak whole-device VRAM includes model loading and host baseline use.",
    ])
    lines.extend([
        "", "## Method", "",
        "The two pinned model snapshots use their own processors, chat templates, and task prompts. "
        "Sampling uses seed 3407; both models receive the same fixed task-specific manifest rows. Requests run serially, "
        "with one image per call and deterministic decoding. LocateAnything uses hybrid generation, BF16, "
        "and up to 8,192 output tokens. GroundAnything uses its model-specific BF16/eager SGLang release "
        "profile and up to 4,096 output tokens.",
        "",
        "RefCOCOg reports mean IoU and Acc@IoU 0.5. COCO reports COCOeval AP on the selected image IDs "
        "when local COCO annotations and category mappings are available, plus local micro F1 at IoU 0.5. "
        "COCO predictions use a uniform confidence score of 1.0 because these models do not emit box confidence. "
        "For category-list prompts, a labeled None is parser-valid only when that category has no ground-truth box in the image. "
        "GUI grounding reports point-in-target accuracy and box Acc@IoU 0.5. HierText reports corpus-level "
        "one-to-one detection precision, recall, and F1 at IoU 0.5. Empty, malformed, truncated, and duplicate "
        "outputs are counted. Whole-device VRAM includes the desktop baseline.",
        "",
        "## Data availability and limitations", "",
        ("No model inference measurements are present yet. Local manifests are prepared for "
         f"{', '.join(data['prepared_tasks']) or 'no tasks'}. Any remaining grounding tasks and COCO vision "
         "assets still require local manifests. Follow `README.md` to prepare the remaining data and run each model." if not data["tested_model_count"] else
         f"Prepared manifests exist for {', '.join(data['prepared_tasks']) or 'no task manifests'}. "
         f"Core task manifests without local preparation include {', '.join(task for task in (*TASKS, *AUX_TASKS) if task not in data['prepared_tasks']) or 'none'}. "
         f"Dataset images and item-level annotations were not copied "
         f"into the site. {data['tested_model_count']} model(s) have at least one completed example; partial "
         "and failed coverage remains visible in the scorecards."),
        "The intended protocol uses RefCOCOg test, COCO val2017, ScreenSpot-Pro test, and HierText validation. "
        "A dataset split is considered run-ready only when its local manifest and assets are present. "
        "ScreenSpot-Pro was obtained from its official public repository and is stored at /mnt/HDD1/Project_Data/public_datasets/screenspot_pro/; the sample uses 100 of 1,581 test annotations with seed 3407. One box extending one pixel above its screenshot was clipped to the visible image boundary before normalization. "
        "The RefCOCOg panel uses the lmms-lab formatted test derivative, which contains two captions for a circled "
        "region rather than the canonical direct referring-expression export. The first caption is used as the "
        "query phrase, both references are retained, and pixel xywh boxes are converted to normalized xyxy. "
        "Treat these results as an adapted caption-to-region protocol, not as directly comparable to standard "
        "RefCOCOg scores. Its local README provides citation information but no separate license statement; "
        "the original RefCOCOg and COCO terms apply. "
        "HierText uses "
        "validation, matching the planned protocol. VQAv2 uses a 25-question existing panel; TextVQA uses "
        "a deterministic sample of 100 validation questions. These specialists do not document general VQA "
        "as a native task, so their answer scores measure exploratory transfer. TextVQA is OCR-dependent VQA, "
        "while HierText provides the direct word-box and transcription score. Dataset images and item-level "
        "predictions remain local.",
        "",
        "## Runtime and artifacts", "",
        "Per-item results preserve the prompt, raw output, canonical predictions, parser state, error, query latency, "
        "token counts, image dimensions, and scores in `results/runs/`. LocateAnything model-load peak and "
        "per-task peaks are recorded separately. Model initialization is excluded from query timing; report its duration separately when captured. Existing LocateAnything results time only `model.generate()`, excluding local image reading, processor encoding, output decoding, and parsing. The updated runner records the full adapter query from local image reading and processor encoding through decoded and parsed output; it also saves pure generation latency separately for token-rate calculations. GroundAnything timing in the saved runs starts after local image reading, base64 encoding, and request serialization, then spans the HTTP response; YOLO timing includes single-image loading, preprocessing, prompt setup, and inference, with warmup images excluded. These model-specific scopes still differ, so latency ratios are indicative rather than fully matched end-to-end measurements. The combined GroundAnything run measured one 624 s service startup shared by its RefCOCOg, HierText, VQAv2, and TextVQA tasks; the repeated task-table value refers to that single load. GroundAnything load duration is recorded for the ScreenSpot-Pro run at approximately 10 s from service startup through API readiness; older preloaded-service COCO, DOTA, HOPE, and LVIS runs lack a captured startup interval, which remains unavailable. "
        "This folder is excluded from the site. "
        "The site includes aggregate metrics only. Whole-device VRAM is sampled every 0.5 seconds with "
        "`nvidia-smi`; the reported maximum includes all processes and desktop use. Package versions describe "
        "the client runner; the GroundAnything service inventory is saved from its `/models` endpoint, while "
        "the custom server's full package set is not exposed by that endpoint.",
        "",
        "## Sources", "",
    ])
    for label, url in REFERENCES.items():
        lines.append(f"- [{label}]({url})")
    lines.append("")
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = assemble(load_runs(), load_vision_runs(), load_dota_run(), load_domain_runs())
    js_data = "window.GROUNDING_REPORT = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (HERE / "data.js").write_text(js_data, encoding="utf-8")
    write_report(data)


if __name__ == "__main__":
    main()
