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
                 "measure": "Box-based F1@IoU 0.5, precision, and recall", "primary": "f1"},
}
REFERENCES = {
    "LocateAnything-3B model card": "https://huggingface.co/nvidia/LocateAnything-3B",
    "LocateAnything evaluation": "https://github.com/NVlabs/Eagle/tree/main/Embodied",
    "GroundAnything-VLM model card": "https://huggingface.co/GroundingPI/GroundAnything-VLM",
    "RefCOCOg": "https://github.com/lichengunc/refer",
    "COCO": "https://cocodataset.org/#download",
    "ScreenSpot-Pro": "https://huggingface.co/datasets/likaixin/ScreenSpot-Pro",
    "ScreenSpot-Pro paper": "https://arxiv.org/abs/2504.07981",
    "HierText": "https://github.com/google-research-datasets/hiertext",
    "GroundAnything evaluation": "https://github.com/groundingpi/GroundAnything",
}


def load_runs() -> dict[str, dict[str, Any]]:
    result = {}
    for model_id in MODEL_ORDER:
        path = RUNS / (model_id.replace("/", "--") + ".json")
        if path.is_file():
            result[model_id] = json.loads(path.read_text(encoding="utf-8"))
    return result


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


def assemble(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    models = []
    for model_id in MODEL_ORDER:
        run = runs.get(model_id)
        task_status = {}
        for task, task_info in TASKS.items():
            task_result = run.get("tasks", {}).get(task, {}) if run else {}
            summary = task_result.get("summary", {})
            completed = summary.get("completed", 0)
            score_name, primary_score = selected_metric(run, task) if run else (None, None)
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
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if winner else "not run" if not runs else "partial",
        "tested_model_count": sum(1 for m in models if any(t["completed"] for t in m["tasks"].values())),
        "candidate_model_count": len(MODEL_ORDER),
        "sample_count_per_task": 100,
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
        "This benchmark compares two pinned grounding checkpoints on four separate localization tasks. "
        "Results are not part of the generic image-question-answering ranking.", "",
        "| Task | Dataset and split | Sample | Primary measures |",
        "|---|---|---:|---|",
    ]
    for task in TASKS.values():
        lines.append(f"| {task['label']} | {task['split']} | {task['n']} | {task['measure']} |")
    lines.extend(["", f"Sample count: {data['sample_count_per_task']} per task; seed {data['seed']}.",
                  "", "## Model coverage", "",
                  "| Model | Revision | Backend | Precision | Mode | Status | Complete tasks |",
                  "|---|---|---|---|---|---|---:|"])
    for model in data["models"]:
        done = sum(t["completed"] for t in model["tasks"].values())
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
                score_text = f"F1 {fmt(metrics.get('f1'), True)}; P {fmt(metrics.get('precision'), True)}; R {fmt(metrics.get('recall'), True)}"
            lines.append(f"| `{model['model_id']}` | {task_result['status']} | {task_result['completed']}/{task['n']} | "
                         f"{score_text} | {fmt(task_result['mean_latency_s'])} s | "
                         f"{fmt(task_result['mean_output_tokens_per_s'])} tok/s | "
                         f"{fmt(task_result['peak_whole_device_vram_gib'])} GiB | "
                         f"{task_result['malformed']} malformed, {task_result['truncated']} truncated, "
                         f"{task_result['empty_outputs']} empty, {task_result['duplicate_predictions']} duplicate |")
        lines.append("")
    lines.extend([
        "## Selection guide", "",
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
        "", "## Method", "",
        "The two pinned model snapshots use their own processors, chat templates, and task prompts. "
        "Sampling uses seed 3407 and the same 100 manifest rows for both models. Requests run serially, "
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
        "No dataset annotations or image assets were found in the inspected local Hugging Face dataset cache "
        "at implementation time. No assets were downloaded or copied. The benchmark runner therefore has no "
        "measured results yet. Supply legally obtained local dataset exports and image paths using the schema "
        "in `README.md`, create shared manifests with `prepare_data.py`, then run both models.",
        "RefCOCOg uses its public test split. COCO uses val2017. ScreenSpot-Pro uses test. HierText uses "
        "validation, matching the planned protocol. Dataset images and item-level predictions remain local.",
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
    data = assemble(load_runs())
    js_data = "window.GROUNDING_REPORT = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (HERE / "data.js").write_text(js_data, encoding="utf-8")
    write_report(data)


if __name__ == "__main__":
    main()
