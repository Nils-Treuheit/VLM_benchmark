#!/usr/bin/env python3
"""Create the compact aggregate-only static-site data from report artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from build_report import latest_runs

HERE = Path(__file__).resolve().parent


def main():
    report_text = (HERE / "report.md").read_text(encoding="utf-8")
    report_title = next((line.lstrip("# ").strip() for line in report_text.splitlines() if line.startswith("# ")), "Local foundation benchmark")
    report_updated = next((line.removeprefix("Updated:").strip() for line in report_text.splitlines() if line.startswith("Updated:")), "")
    models = []
    for path, result in latest_runs():
        m = result["model"]
        panel_tasks = ("vqa_truth", "vqa_count", "vqa_description", "textvqa", "vqa_affordance", "vqa_negation")
        tasks = {}
        for task, value in m.get("tasks", {}).items():
            tasks[task] = {key: value.get(key) for key in (
                "requested_n", "completed_n", "failed_n", "score", "parse_errors",
                "mean_generation_s", "generated_tokens_per_s", "group_scores",
            )}
        preflight = result.get("preflight", {})
        preflight_summary = {
            "passed": preflight.get("passed"),
            "required": preflight.get("required", []),
            "checks": {
                task: {
                    key: check.get(key)
                    for key in ("passed", "nonempty_answer", "answer_format_ok")
                    if key in check
                }
                for task, check in preflight.get("checks", {}).items()
            },
        }
        models.append({
            "model_id": m.get("model_id"),
            "model_card": m.get("model_card"),
            "backend": m.get("backend"),
            "runtime": m.get("runtime"),
            "revision": m.get("revision"),
            "quantization": m.get("quantization"),
            "configured_context": m.get("configured_context"),
            "advertised_context": m.get("advertised_context"),
            "image_processing": m.get("image_processing"),
            "status": m.get("status"),
            "full_benchmark_complete": m.get("full_benchmark_complete", False),
            "full_benchmark_macro": m.get("full_benchmark_macro"),
            "vqa_panel_macro": (
                sum(m["tasks"][task]["score"] for task in panel_tasks) / len(panel_tasks)
                if all(task in m.get("tasks", {}) and m["tasks"][task].get("completed_n") == m["tasks"][task].get("requested_n") and m["tasks"][task].get("score") is not None for task in panel_tasks)
                else None
            ),
            "whole_device_peak_gib": m.get("whole_device_peak_gib"),
            "whole_device_memory_method": m.get("whole_device_memory_method"),
            "runtime_versions": m.get("runtime_versions"),
            "generation": m.get("generation"),
            "weight_bytes": m.get("weight_bytes"),
            "tasks": tasks,
            "preflight": preflight_summary,
            "run_path": str(path.relative_to(HERE)),
            "generated_at": result.get("generated_at"),
        })
    data = {
        "generated_at": max((model.get("generated_at", "") for model in models), default=""),
        "updated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "report_title": report_title,
        "report_updated_at": report_updated,
        "models": models,
        "candidates": json.loads((HERE / "candidate_inventory.json").read_text(encoding="utf-8"))["candidates"],
        "notes": [
            "Scores use the shared 3407 fixed samples and local task scorers.",
            "Only complete 825-example runs qualify for the full-suite macro or selection picks.",
            "Whole-device VRAM samples include the desktop baseline.",
            "Candidate cache and compatibility dispositions are explained in report.md.",
        ],
        "source_report": "report.md",
    }
    (HERE / "data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {HERE / 'data.json'} with {len(models)} run(s).")


if __name__ == "__main__":
    main()
