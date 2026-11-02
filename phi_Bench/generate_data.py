#!/usr/bin/env python3
"""Build an aggregate-only Phi report site bundle from benchmark results."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "phi_family" / "results.json"
DESTINATION = ROOT / "data.json"
REPORT_DESTINATION = ROOT / "report.md"
TASKS = [
    ("ai2d", "AI2D", 100),
    ("textvqa", "TextVQA / OCR", 100),
    ("vqa_truth", "Truth", 25),
    ("vqa_count", "Counting", 25),
    ("vqa_description", "Description", 25),
    ("vqa_affordance", "Affordance", 25),
    ("vqa_negation", "Negation", 25),
    ("graph_grab_lite", "GRAB-Lite", 500),
]


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    shutil.copyfile(SOURCE.with_name("report.md"), REPORT_DESTINATION)
    task_ids = [task_id for task_id, _, _ in TASKS]
    models = []
    for original in source.get("models", []):
        task_data = {}
        for task_id, label, requested in TASKS:
            metrics = original.get("tasks", {}).get(task_id, {})
            items = metrics.get("items", [])
            task_data[task_id] = {
                "label": label,
                "requested_n": metrics.get("requested_n", requested),
                "completed_n": metrics.get("completed_n", 0),
                "failed_n": metrics.get("failed_n", 0),
                "score": metrics.get("score"),
                "mean_generation_s": metrics.get("mean_generation_s"),
                "generated_tokens_per_s": metrics.get("generated_tokens_per_s"),
                "peak_allocated_gib": metrics.get("peak_allocated_gib"),
                "parse_errors": sum(bool(item.get("parse_error")) for item in items),
                "error": metrics.get("error"),
                "group_scores": metrics.get("group_scores") if task_id == "graph_grab_lite" else None,
            }
        complete = all(
            task_data[task_id]["completed_n"] == requested
            and task_data[task_id]["score"] is not None
            for task_id, _, requested in TASKS
        )
        complete_tasks = sum(
            task_data[task_id]["completed_n"] == requested
            for task_id, _, requested in TASKS
        )
        models.append({
            "model_id": original["model_id"],
            "status": "complete" if complete else original.get("status", "partial"),
            "backend": original.get("backend"),
            "snapshot": original.get("snapshot"),
            "config_identity": original.get("config_identity"),
            "comparison": original.get("comparison"),
            "checkpoint_context": original.get("checkpoint_context") or (original.get("config_identity") or {}).get("max_position_embeddings"),
            "weight_gib": (original.get("weight_bytes") or 0) / (1024 ** 3),
            "error": original.get("error"),
            "trust_remote_code": original.get("trust_remote_code"),
            "attention_implementation": original.get("attention_implementation"),
            "prompt_template": original.get("prompt_template"),
            "compatibility_patches": original.get("compatibility_patches", []),
            "complete_tasks": complete_tasks,
            "full_benchmark_macro": sum(task_data[task_id]["score"] for task_id in task_ids) / len(task_ids) if complete else None,
            "tasks": task_data,
        })

    complete_models = [model for model in models if model["status"] == "complete"]
    leaderboard = sorted(complete_models, key=lambda model: model["full_benchmark_macro"], reverse=True)
    data = {
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "benchmark_generated_at": source.get("generated_at"),
        "seed": source.get("seed"),
        "memory_cap_gib": source.get("memory_cap_gib"),
        "runtime_versions": source.get("runtime_versions", {}),
        "task_order": task_ids,
        "task_labels": {task_id: label for task_id, label, _ in TASKS},
        "task_samples": {task_id: requested for task_id, _, requested in TASKS},
        "models": models,
        "leaders": {
            "overall": leaderboard[0]["model_id"] if leaderboard else None,
            **{
                task_id: max(complete_models, key=lambda model: model["tasks"][task_id]["score"], default={}).get("model_id")
                for task_id, _, _ in TASKS
            },
        },
    }
    DESTINATION.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {DESTINATION} ({DESTINATION.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
