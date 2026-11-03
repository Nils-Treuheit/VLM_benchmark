#!/usr/bin/env python3
"""Create the aggregate-only static site bundle from the final report data."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT
RESULTS = ROOT / "results" / "general_transformers" / "results.json"
REPORT = ROOT / "results" / "general_transformers" / "report.md"


def main() -> int:
    if not RESULTS.is_file() or not REPORT.is_file():
        raise SystemExit("Generate the aggregate report and results.json before site data.")
    source = json.loads(RESULTS.read_text(encoding="utf-8"))
    report_text = REPORT.read_text(encoding="utf-8")
    if source.get("generated_at") not in report_text:
        raise SystemExit("The aggregate report timestamp does not match results.json.")
    if not source.get("run_id") or f"Run ID: {source['run_id']}" not in report_text:
        raise SystemExit("The aggregate report run ID does not match results.json.")
    missing_models = [
        model.get("model_id") for model in source.get("models", [])
        if model.get("model_id") not in report_text
    ]
    if missing_models:
        raise SystemExit("The aggregate report omits model(s): " + ", ".join(missing_models))
    complete_count = sum(bool(model.get("full_benchmark_complete")) for model in source.get("models", []))
    unrun_count = len(source.get("planned_candidates_not_run", []))
    unrun_noun = "candidate" if unrun_count == 1 else "candidates"
    models = []
    for model in source.get("models", []):
        preflight = model.get("preflight", {})
        checks = {
            task: {
                "passed": item.get("passed"),
                "answer_format_ok": item.get("answer_format_ok"),
                "nonempty_answer": item.get("nonempty_answer"),
                "parse_error": item.get("parse_error"),
                "output_tokens": item.get("output_tokens"),
            }
            for task, item in preflight.get("checks", {}).items()
        }
        task_summary = {}
        for task, record in model.get("tasks", {}).items():
            task_summary[task] = {
                key: record.get(key)
                for key in (
                    "requested_n", "completed_n", "failed_n", "score", "parse_errors",
                    "mean_generation_s", "generated_tokens_per_s",
                    "peak_allocated_gib", "group_scores",
                )
            }
        models.append({
            "model_id": model.get("model_id"),
            "status": model.get("status"),
            "backend": model.get("backend"),
            "checkpoint_revision": model.get("revision") or (
                Path(model["snapshot"]).name if model.get("snapshot") else None
            ),
            "cache_status": model.get("cache_status"),
            "weight_gib": model.get("weight_gib"),
            "model_weight_bytes": model.get("model_weight_bytes"),
            "auxiliary_weight_bytes": model.get("auxiliary_weight_bytes"),
            "estimated_loaded_weight_gib": model.get("estimated_loaded_weight_gib"),
            "checkpoint_context": model.get("checkpoint_context"),
            "image_resolution_px": model.get("image_resolution_px"),
            "runtime_versions": model.get("runtime_versions"),
            "precision": model.get("precision"),
            "attention_backend": model.get("attention_backend"),
            "preflight": {"passed": preflight.get("passed"), "checks": checks},
            "scoring_route": model.get("scoring_route"),
            "error": model.get("error"),
            "failure_diagnosis": model.get("failure_diagnosis"),
            "complete_tasks": model.get("complete_tasks", 0),
            "full_benchmark_complete": model.get("full_benchmark_complete", False),
            "full_benchmark_macro": model.get("full_benchmark_macro"),
            "peak_allocated_gib": model.get("peak_allocated_gib"),
            "peak_memory_task": model.get("peak_memory_task"),
            "load_peak_allocated_gib": model.get("load_peak_allocated_gib"),
            "load_peak_reserved_gib": model.get("load_peak_reserved_gib"),
            "sampled_process_peak_gib": model.get("sampled_process_peak_gib"),
            "memory_metric": model.get("memory_metric"),
            "sampled_process_sample_count": model.get("sampled_process_sample_count"),
            "sampled_process_peak_method": model.get("sampled_process_peak_method"),
            "tasks": task_summary,
        })
    data = {
        "generated_at": source.get("generated_at"),
        "run_id": source.get("run_id"),
        "status": source.get("status"),
        "status_message": (
            "Complete full-suite runs are ready for comparison."
            if source.get("status") == "complete"
            else (
                f"{complete_count} model variants have complete full-suite runs; "
                f"{unrun_count} planned {unrun_noun} {'remains' if unrun_count == 1 else 'remain'} unrun. "
                "See their cache and device-placement notes below."
                if source.get("status") == "partial"
                else "No model completed the shared benchmark. Read the full report for blockers."
            )
        ),
        "planned_candidates_not_run": source.get("planned_candidates_not_run", []),
        "planned_candidate_count": source.get("planned_candidate_count", len(models)),
        "execution": source.get("execution"),
        "memory_cap_gib": source.get("memory_cap_gib"),
        "seed": source.get("seed"),
        "task_order": source.get("task_order", []),
        "task_labels": source.get("task_labels", {}),
        "task_sizes": source.get("task_sizes", {}),
        "leaders": source.get("leaders", {}),
        "models": models,
        "disclosure": "Aggregate-only display data. Per-item prompts, images, and outputs are not included.",
        "source_report": "../results/general_transformers/report.md",
    }
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {SITE / 'data.json'} from report.md and results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
