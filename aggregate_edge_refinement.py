#!/usr/bin/env python3
"""Merge one cap-specific runner artifact into the root Edge Limited data."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TASKS = (
    "ai2d", "textvqa", "vqa_truth", "vqa_count", "vqa_description",
    "vqa_affordance", "vqa_negation", "graph_grab_lite",
)
CAPS = (8, 10, 12, 14, 16)


def complete_task(task: dict[str, Any] | None) -> bool:
    return bool(task and task.get("requested_n", 0) > 0
                and task.get("completed_n") == task.get("requested_n")
                and isinstance(task.get("score"), (int, float)))


def as_record(payload: dict[str, Any], model: dict[str, Any], suite_name: str, suite_href: str,
              expected_cap: int) -> dict[str, Any]:
    tasks = model.get("tasks", {})
    task_complete = {task: complete_task(tasks.get(task)) for task in TASKS}
    full = all(task_complete.values())
    scores = {task: tasks[task]["score"] for task in TASKS if task_complete[task]}
    probe = model.get("context_probe") or {}
    verification = model.get("cap_verification") or {}
    peak = model.get("peak_gpu_used_gib")
    cap = model.get("whole_device_cap_gib")
    context_peak = probe.get("peak_whole_device_gib")
    task_peaks = {task: tasks[task].get("peak_gpu_used_gib") for task in TASKS if task_complete[task]}
    process_peaks = [model.get("model_load_peak_allocated_gib")]
    for task in tasks.values():
        process_peaks.extend(item.get("peak_allocated_gib") for item in task.get("items", []))
    process_peaks = [value for value in process_peaks if isinstance(value, (int, float))]
    actual_cap = payload.get("whole_device_cap_gib", payload.get("memory_cap_gib"))
    context_tokens = probe.get("measured_context_tokens")
    selection_attempts = probe.get("image_selection_attempts") or []
    successful_selection_samples = probe.get("image_selection_success_count")
    if successful_selection_samples is None and probe.get("image_selection_complete") is True:
        successful_selection_samples = 0
        for attempt in selection_attempts:
            sample_count = attempt.get("sample_count", attempt.get("whole_device_sample_count", 0))
            sample_error = next((attempt.get(key) for key in (
                "sampler_error", "whole_device_sampler_error", "whole_device_sample_error"
            ) if attempt.get(key) is not None), None)
            peak_value = attempt.get("peak_whole_device_gib")
            inferred_pass = (
                attempt.get("sampler_verified", attempt.get("whole_device_sampler_verified")) is True
                and isinstance(sample_count, (int, float)) and sample_count > 0
                and not attempt.get("error")
                and not attempt.get("measurement_error")
                and sample_error is None
                and isinstance(peak_value, (int, float))
                and math.isfinite(peak_value)
                and 0 < peak_value <= expected_cap
            )
            pass_evidence = attempt.get("passed") is True if "passed" in attempt else inferred_pass
            if inferred_pass and pass_evidence:
                successful_selection_samples += 1
    load_peak = model.get("load_peak_whole_device_gib")
    whole_peak = model.get("whole_device_peak_gib")
    whole_run_verified = bool(
        model.get("cap_memory_verified") is True
        or (
            model.get("cap_verified") is True
            and verification.get("whole_device_sampling_verified") is True
            and verification.get("task_sampling_verified") is True
            and verification.get("whole_run_peak_within_cap",
                                 verification.get("whole_device_peak_within_cap")) is True
        )
    )
    cap_evidence = bool(
        cap == expected_cap == actual_cap
        and verification.get("whole_device_cap_gib") == expected_cap
        and whole_run_verified
        and probe.get("passed") is True
        and isinstance(context_tokens, (int, float)) and context_tokens > 0
        and isinstance(context_peak, (int, float)) and 0 < context_peak <= expected_cap
        and isinstance(peak, (int, float)) and 0 < peak <= expected_cap
        and isinstance(load_peak, (int, float)) and 0 < load_peak <= expected_cap
        and isinstance(whole_peak, (int, float)) and 0 < whole_peak <= expected_cap
    )
    return {
        "model_id": model.get("model_id"),
        "backend": model.get("backend"),
        "snapshot": model.get("snapshot"),
        "revision": model.get("revision"),
        "quantization": model.get("quantization"),
        "suite_name": suite_name,
        "suite_href": suite_href,
        "status": "verified" if cap_evidence else "unverified",
        "complete": full,
        "task_complete": task_complete,
        "context_probe_passed": bool(probe.get("passed")),
        "whole_device_cap_gib": cap,
        "context_tokens": context_tokens,
        "context_probe_peak_gib": context_peak,
        "context_probe_image": {
            "dataset": probe.get("image_dataset"),
            "index": probe.get("image_index"),
            "size_px": probe.get("image_size_px"),
            "selection_method": probe.get("image_selection_method"),
            "visual_token_count": probe.get("image_visual_token_count"),
            "prompt_tokens": probe.get("image_prompt_tokens"),
            "total_input_token_proxy": probe.get(
                "image_selection_total_input_token_proxy", probe.get("image_prompt_tokens")
            ),
            "token_measure": probe.get(
                "image_selection_token_measure",
                "prompt_tokens" if probe.get("image_prompt_tokens") is not None else None,
            ),
            "sample_count": probe.get("image_selection_sample_count"),
            "successful_samples": successful_selection_samples,
            "complete": probe.get("image_selection_complete"),
        },
        "context_probe_actual_input_tokens": probe.get("actual_input_tokens"),
        "context_probe_output_tokens": probe.get("output_tokens"),
        "context_probe_configured_limit_reached": probe.get("configured_limit_reached"),
        "scoring_peak_gib": peak,
        "whole_device_peak_gib": whole_peak,
        "load_peak_gib": load_peak,
        "process_peak_gib": max(process_peaks) if process_peaks else model.get("process_peak_gib"),
        "task_scores": scores,
        "task_peaks_gib": task_peaks,
        "peak_task": max(task_peaks, key=task_peaks.get) if task_peaks else None,
        "task_completion_counts": {
            task: {"completed": tasks.get(task, {}).get("completed_n", 0),
                   "requested": tasks.get(task, {}).get("requested_n", 0)}
            for task in TASKS
        },
        "score": statistics.mean(scores.values()) if full else None,
        "run_status": model.get("status"),
        "run_date": payload.get("generated_at"),
        "gpu_baseline_gib": payload.get("gpu_baseline_gib"),
        "hardware": payload.get("hardware"),
        "runtime_versions": payload.get("runtime_versions"),
        "cap_enforcement_method": model.get("cap_enforcement_method") or payload.get("cap_enforcement_method"),
        "configured_context_tokens": model.get(
            "configured_context_tokens",
            model.get(
                "configured_context",
                model.get("scored_context_limit_tokens", model.get("context_cap_tokens")),
            ),
        ),
        "llama_fit_target_mib": model.get("llama_fit_target_mib", model.get("cap_fit_target_mib")),
        "source_artifact": model.get("source_artifact"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="runner results.json")
    parser.add_argument("--cap", required=True, type=int, choices=CAPS)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--suite-href", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "edge-refinement-data.json")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("whole_device_cap_gib", payload.get("memory_cap_gib")) != args.cap:
        raise SystemExit("input artifact cap does not match --cap")
    models = payload.get("models") or []
    if not models:
        raise SystemExit("input artifact contains no model records")

    data = json.loads(args.output.read_text(encoding="utf-8"))
    stage = next((item for item in data.get("caps", []) if item.get("cap_gib") == args.cap), None)
    if stage is None:
        raise SystemExit(f"edge data has no {args.cap} GiB stage")
    records = {(record.get("suite_name"), record.get("model_id")): record
               for record in stage.get("records", [])}
    for model in models:
        record = as_record(payload, model, args.suite_name, args.suite_href, args.cap)
        record["source_artifact"] = str(args.input.resolve().relative_to(ROOT)) if args.input.resolve().is_relative_to(ROOT) else str(args.input.resolve())
        records[(record["suite_name"], record["model_id"])] = record
    stage["records"] = list(records.values())
    verified = [record for record in stage["records"] if record["status"] == "verified"]
    full_verified = [record for record in verified if record["complete"]]
    stage["verified_candidate_count"] = len(full_verified)
    stage["status"] = "measured" if verified else "no_verified_runs"

    stage["overall"] = None
    if full_verified:
        best = max(record["score"] for record in full_verified)
        stage["overall"] = [record for record in full_verified if record["score"] == best]
    stage["category_leaders"] = {}
    for task in TASKS:
        eligible = [record for record in verified
                    if record["task_complete"].get(task) and task in record["task_scores"]
                    and record["task_peaks_gib"].get(task) is not None
                    and 0 < record["task_peaks_gib"][task] <= args.cap]
        if eligible:
            best = max(record["task_scores"][task] for record in eligible)
            stage["category_leaders"][task] = [
                {**record, "score": record["task_scores"][task], "task_complete": True}
                for record in eligible if record["task_scores"][task] == best
            ]
        else:
            stage["category_leaders"][task] = []
    data["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["status"] = "measured" if any(c.get("verified_candidate_count") for c in data["caps"]) else "pending_measurements"
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
