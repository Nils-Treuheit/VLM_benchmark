#!/usr/bin/env python3
"""Build the grounding report and aggregate-only static-site data from runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
MODEL_ORDER = ("nvidia/LocateAnything-3B", "GroundingPI/GroundAnything-VLM")
TASKS = {
    "refcocog": {"label": "Referring expression", "split": "RefCOCOg test", "n": 100,
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


def load_runs() -> dict[str, dict[str, Any]]:
    result = {}
    for model_id in MODEL_ORDER:
        path = RUNS / (model_id.replace("/", "--") + ".json")
        if path.is_file():
            result[model_id] = json.loads(path.read_text(encoding="utf-8"))
        coco_path = (HERE / "results" / "domain_runs" / "coco_val2017" /
                     "locateany_tf457" / (model_id.replace("/", "--") + ".json"))
        if coco_path.is_file():
            coco_run = json.loads(coco_path.read_text(encoding="utf-8"))
            result.setdefault(model_id, {"model_id": model_id, "tasks": {}})
            result[model_id].setdefault("tasks", {}).update(coco_run.get("tasks", {}))
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
    if not root.is_dir():
        return result
    for path in sorted(root.glob("*/*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_id = str(payload.get("model_id") or path.stem.replace("--", "/"))
        result.setdefault(path.parent.parent.name, {})[model_id] = payload
    return result


def domain_summaries(runs: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    labels = {
        "yoloe26l-seg": "YOLOE-26-L",
        "yolo26l": "YOLO26-L",
        "yolo26l-seg": "YOLO26-L-seg",
        "sam3.1": "SAM 3.1",
        "nvidia/LocateAnything-3B": "LocateAnything-3B",
        "GroundingPI/GroundAnything-VLM": "GroundAnything-VLM",
    }
    summaries = []
    for domain in ("coco_val2017", "dota_v1", "bop_h3_hope_v2", "lvis_v1"):
        manifest_name = "manifest.json" if domain == "lvis_v1" else "coco.json"
        manifest_path = HERE / "data" / "domain_manifests" / domain / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        domain_label = manifest.get("dataset", domain)
        expected = int(manifest.get("sample_count", 0))
        for model_id, run in sorted(runs.get(domain, {}).items()):
            tasks = run.get("tasks", {})
            if "object_detection" in tasks:
                detection = tasks.get("object_detection", {})
                segmentation = tasks.get("instance_segmentation", {})
                official_lvis = run.get("lvis_official_evaluation", {})
                if official_lvis:
                    detection = official_lvis.get("object_detection", detection)
                    segmentation = official_lvis.get("instance_segmentation", segmentation)
                runtime = run.get("runtime", {})
                sample_count = int(run.get("sample_count", 0))
                official_status = official_lvis.get("object_detection", {}).get("status")
                if domain == "lvis_v1":
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
                    "class_aware_F1_IoU50": detection.get("class_aware_F1_IoU50"),
                    "mask_AP": segmentation.get("AP"), "latency_s": runtime.get("latency_mean_s"),
                    "images_per_s": runtime.get("images_per_s"),
                    "AP_rare": detection.get("AP_rare"), "AP_common": detection.get("AP_common"),
                    "AP_frequent": detection.get("AP_frequent"),
                })
            else:
                task = tasks.get("coco", {})
                summary = task.get("summary", {})
                official = summary.get("coco_official_ap", {})
                sample_count = int(summary.get("completed", 0))
                latency = summary.get("mean_latency_s")
                summaries.append({
                    "dataset": domain_label, "model_id": model_id, "model": labels.get(model_id, model_id),
                    "status": "complete" if sample_count >= expected and expected else "partial" if sample_count else "not run",
                    "sample_count": sample_count, "expected_count": expected,
                    "input_size": "native model processor",
                    "AP": official.get("AP"), "AP50": official.get("AP50"),
                    "class_aware_F1_IoU50": summary.get("metrics", {}).get("micro_f1"),
                    "mask_AP": None, "latency_s": latency,
                    "images_per_s": 1.0 / latency if latency else None,
                })
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
                        "images_per_s": 1 / mean_latency if mean_latency else None,
                        "peak_whole_device_vram_gib": summary.get("peak_whole_device_vram_gib")},
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
            "package_versions": run.get("client_package_versions") if run else {},
            "status": run.get("status", "not run") if run else "not run",
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
    manifests_present = any(bool(run.get("sampling", {}).get("tasks")) for run in runs.values())
    local_manifests = HERE / "data" / "manifests"
    prepared_tasks = [task for task in (*TASKS, *AUX_TASKS) if (local_manifests / f"{task}.json").is_file()]
    prepared_aux_manifests = all((local_manifests / f"{task}.json").is_file() for task in AUX_TASKS)
    prepared_any_manifests = any((local_manifests / f"{task}.json").is_file()
                                 for task in (*TASKS, *AUX_TASKS))
    dota_summary = dota_track_summary(dota_run or {})
    dataset_tracks = [
        {"dataset": "COCO val2017", "status": "100-image shared manifest; YOLOE, YOLO26, SAM 3.1, and LocateAnything runs complete; GroundAnything runtime unavailable",
         "task": "Shared detection, classification, instance segmentation, and speed",
         "note": "Every available vision baseline uses the same 100 annotated images and all 80 COCO classes. Grounding specialists are scored on boxes; models without confidence scores use uniform scores for COCO AP."},
        {"dataset": "DOTA v1.0 validation", "status": "100-image shared HBB runs complete for YOLOE, YOLO26-L, and SAM 3.1; LocateAnything rerun pending GPU release; full-split YOLO26-L-OBB AP complete, speed rerun pending",
         "task": "Shared aerial detection plus native OBB for YOLO26-L",
         "note": "100 of 456 labeled validation images are sampled for horizontal-box COCO AP across YOLOE, SAM 3.1, and available grounding specialists. LocateAnything's first attempt overlapped another GPU job and is excluded; a clean rerun awaits the required GPU release check. YOLO26-L-OBB uses the full 458-file validation split, including two background-only images, for the separate native OBB reference. DOTA terms restrict use to academic purposes and prohibit commercial use."},
        {"dataset": "BOP-H3 HOPE v2 validation", "status": "50-image shared manifest; LocateAnything, SAM 3.1, and YOLOE complete; closed-vocabulary YOLO26 omitted because its COCO label set does not cover HOPE classes",
         "task": "Object-centric 2D detection; instance masks where supported",
         "note": "All 50 local validation images are shared by YOLOE, SAM 3.1, and both grounding specialists. Box AP is common; mask AP is reported only for models that return masks. This is the 2D HOPE v2 subset, not the full BOP-H3 pose/onboarding protocol."},
        {"dataset": "KITTI object detection", "status": "download image_2 training archive",
         "task": "Automotive 2D detection",
         "note": "Labels and calibration are local, but camera images are missing. Download the KITTI image_2 archive before using a fixed train/validation split."},
        {"dataset": "LVIS v1 validation", "status": "100-image panel prepared; GPU runs pending",
         "task": "Long-tail detection and instance segmentation",
         "note": "This exploratory panel uses 10 rare, 10 common, and 10 frequent categories across 100 fixed LVIS val images sourced from COCO 2017 train/val. It uses official LVIS AP and frequency-group metrics, not the full 1,203-category leaderboard; COCO-trained baselines may have seen some train images."},
        {"dataset": "RF100-VL", "status": "candidate; Roboflow API key not present",
         "task": "Multi-domain open-vocabulary detection",
         "note": "Highest-value new download for domain shift. Start with aerial, document, laboratory, flora/fauna, and industrial subsets, not all 100 datasets. The downloader requires a Roboflow API key."},
        {"dataset": "BOP-H3 HOT3D/HANDAL", "status": "separate follow-up",
         "task": "Unseen-object detection, onboarding, and 6D pose",
         "note": "Keep separate from class-name box AP. Use the official BOP protocol for pose tasks and review each dataset's terms before expanding beyond local HOPE v2."},
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if winner else "not run" if not runs else "partial",
        "tested_model_count": tested_model_count,
        "data_status": "local manifests used" if manifests_present else
                       "local auxiliary manifests prepared" if prepared_aux_manifests else
                       "some local manifests prepared" if prepared_any_manifests else
                       "not present in inspected cache",
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
        "domain_benchmarks": domain_summaries(domain_runs or {}),
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
                      "| Model | Status | Coverage | Scores | Mean latency | Output rate | Peak whole-device VRAM | Parse issues |",
                      "|---|---|---:|---|---:|---:|---:|---:|"])
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
                         f"{score_text} | {fmt(task_result['mean_latency_s'])} s | "
                         f"{fmt(task_result['mean_output_tokens_per_s'])} tok/s | "
                         f"{fmt(task_result['peak_whole_device_vram_gib'])} GiB | "
                      f"{task_result['malformed']} malformed, {task_result['invalid_outputs']} invalid, {task_result['truncated']} truncated, "
                         f"{task_result['empty_outputs']} empty, {task_result['duplicate_predictions']} duplicate |")
        lines.append("")
    lines.extend([
        "## OCR, VQA, and counting", "",
        "These answer-generation tasks are reported separately from localization and object detection. "
        "TextVQA measures question answering that depends on reading scene text. HierText adds word recognition "
        "to the existing text-box task by matching recognized words to word boxes at IoU 0.5.", "",
        "| Model | Task | Status | Coverage | VQAv2 consensus / exact | Mean latency | Output rate | Peak VRAM |",
        "|---|---|---|---:|---:|---:|---:|---:|",
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
                         f"{fmt(result['mean_latency_s'])} s | {fmt(result['mean_output_tokens_per_s'])} tok/s | "
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
        "| Model | Status | Images | Box AP / AP50 | Mask AP / AP50 | Label accuracy / joint / class F1 | RefCOCOg Acc@0.5 / latency | Mean latency | Images/s | Peak VRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model in data["vision"]:
        detection = model["detection"] or {}
        segmentation = model["segmentation"] or {}
        classification = model["classification"] or {}
        grounding = model["grounding"] or {}
        runtime = model["runtime"] or {}
        lines.append(f"| `{model['model_id']}` | {model['status']} | {model['sample_count']} | "
                     f"{fmt(detection.get('AP'), True)} / {fmt(detection.get('AP50'), True)} | "
                     f"{fmt(segmentation.get('AP'), True)} / {fmt(segmentation.get('AP50'), True)} | "
                     f"{fmt(classification.get('accuracy_on_matched_objects'), True)} / "
                     f"{fmt(classification.get('joint_detection_and_classification_accuracy'), True)} / "
                     f"{fmt(detection.get('class_aware_F1_IoU50'), True)} | "
                     f"{fmt(grounding.get('acc_iou_05'), True)} / {fmt(grounding.get('mean_latency_s'))} s | "
                     f"{fmt(runtime.get('latency_mean_s'))} s | {fmt(runtime.get('images_per_s'))} | "
                     f"{fmt(runtime.get('peak_whole_device_vram_gib'))} GiB |")
    lines.extend([
        "", "YOLOE-26 and SAM 3.1 receive the same full COCO category-name list per image. SAM latency includes "
        "one text-prompt query per category; YOLOE latency includes prompt setup and one image pass. YOLO26 "
        "uses its fixed COCO vocabulary. COCO detection and instance-segmentation AP use the official evaluator. "
        "Optional RefCOCOg text grounding reports mean IoU and Acc@IoU 0.5 for YOLOE and SAM. "
        "Throughput is single-image serial inference; model load time is excluded. The grounding specialists "
        "do not provide confidence scores, so their COCO AP uses uniform scores and is less informative than "
        "class-aware F1 for direct comparison. No overall rank combines these metrics.",
        "", "## Selection guide", "",
        (f"Best complete model: `{data['best_complete']['model_id']}` with local task macro "
         f"{fmt(data['best_complete']['local_macro'], True)}." if data["best_complete"] else
         "No complete model is eligible for a quality recommendation yet."),
        "The local task macro weights the four task scores equally. RefCOCOg uses Acc@IoU 0.5, "
        "COCO uses sampled-image COCO AP when available and otherwise local micro F1@IoU 0.5, "
        "GUI uses point-in-target accuracy, and HierText uses detection F1. It is a local comparison metric.",
        "", "| Task | Best complete model | Primary score |",
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
        "Every row uses the prepared image IDs and category list for its domain. Box AP and class-aware F1 are shared across vision models and grounding specialists; mask AP is shown when available.",
        "", "| Dataset | Model | Status | Images | Input size | Box AP / AP50 | Class-aware F1 | Mask AP | AP rare / common / frequent | Mean latency | Images/s |",
        "|---|---|---|---:|---|---:|---:|---:|---|---:|---:|",
    ])
    for result in data["domain_benchmarks"]:
        lines.append(
            f"| {result['dataset']} | {result['model']} | {result['status']} | "
            f"{result['sample_count']}/{result['expected_count']} | {result.get('input_size') or '—'} | "
            f"{fmt(result.get('AP'), True)} / {fmt(result.get('AP50'), True)} | "
            f"{fmt(result.get('class_aware_F1_IoU50'), True)} | {fmt(result.get('mask_AP'), True)} | "
            f"{fmt(result.get('AP_rare'), True)} / {fmt(result.get('AP_common'), True)} / "
            f"{fmt(result.get('AP_frequent'), True)} | "
            f"{fmt(result.get('latency_s'))} s | {fmt(result.get('images_per_s'))} |"
        )
    if not data["domain_benchmarks"]:
        lines.append("| DOTA v1.0 and BOP-H3 HOPE v2 | All shared models | not run | — | — | — | — | — | — | — |")
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
        "This is an in-domain OBB reference, not a zero-shot YOLOE comparison. Validation speed excludes model load; the displayed speed and peak VRAM await a clean serial rerun.",
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
        "GUI grounding reports point-in-target accuracy and box Acc@IoU 0.5. HierText reports corpus-level "
        "one-to-one detection precision, recall, and F1 at IoU 0.5. Empty, malformed, truncated, and duplicate "
        "outputs are counted. Whole-device VRAM includes the desktop baseline.",
        "",
        "## Data availability and limitations", "",
        ("No model inference measurements are present yet. Local manifests are prepared for "
         f"{', '.join(data['prepared_tasks']) or 'no tasks'}. Any remaining grounding tasks and COCO vision "
         "assets still require local manifests. Follow `README.md` to prepare the remaining data and run each model." if not data["tested_model_count"] else
         f"Local manifests were used for these runs. Dataset images and item-level annotations were not copied "
         f"into the site. {data['tested_model_count']} model(s) have at least one completed example; partial "
         "and failed coverage remains visible in the scorecards."),
        "RefCOCOg uses its public test split. COCO uses val2017. ScreenSpot-Pro uses test. HierText uses "
        "validation, matching the planned protocol. VQAv2 uses a 25-question existing panel; TextVQA uses "
        "a deterministic sample of 100 validation questions. These specialists do not document general VQA "
        "as a native task, so their answer scores measure exploratory transfer. TextVQA is OCR-dependent VQA, "
        "while HierText provides the direct word-box and transcription score. Dataset images and item-level "
        "predictions remain local.",
        "",
        "## Runtime and artifacts", "",
        "Per-item results preserve the prompt, raw output, canonical predictions, parser state, error, latency, "
        "token counts, image dimensions, and scores in `results/runs/`. LocateAnything model-load peak and "
        "per-task peaks are recorded separately; the GroundAnything server load happens before the client run. "
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
