#!/usr/bin/env python3
"""Merge completed per-model runs into the group report and results bundle."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TASKS = (
    "ai2d", "textvqa", "vqa_truth", "vqa_count", "vqa_description",
    "vqa_affordance", "vqa_negation", "graph_grab_lite",
)
TASK_LABELS = {
    "ai2d": "AI2D", "textvqa": "TextVQA / OCR", "vqa_truth": "Truth",
    "vqa_count": "Counting", "vqa_description": "Description",
    "vqa_affordance": "Affordance", "vqa_negation": "Negation",
    "graph_grab_lite": "GRAB-Lite",
}
TASK_SIZES = {
    "ai2d": 100, "textvqa": 100, "vqa_truth": 25, "vqa_count": 25,
    "vqa_description": 25, "vqa_affordance": 25, "vqa_negation": 25,
    "graph_grab_lite": 500,
}
MODEL_RUNS = (
    "paligemma2_mix_224",
    "florence2_large_ft",
)


def aggregate(run_dir: Path) -> dict[str, Any]:
    models = []
    for model_dir in MODEL_RUNS:
        path = run_dir / model_dir / "results.json"
        if not path.is_file():
            raise FileNotFoundError(f"model run is missing: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        model = record["model"]
        model["preflight"] = record.get("preflight", {})
        task_results = model.get("tasks", {})
        model["complete_tasks"] = sum(
            task_results.get(task, {}).get("completed_n") == TASK_SIZES[task]
            and task_results.get(task, {}).get("requested_n") == TASK_SIZES[task]
            for task in TASKS
        )
        model["full_benchmark_complete"] = all(
            task_results.get(task, {}).get("completed_n") == TASK_SIZES[task]
            and task_results.get(task, {}).get("requested_n") == TASK_SIZES[task]
            for task in TASKS
        )
        model["full_benchmark_macro"] = (
            statistics.mean(task_results[task]["score"] for task in TASKS)
            if model["full_benchmark_complete"] else None
        )
        model["peak_allocated_gib"] = max(
            (task_results.get(task, {}).get("peak_allocated_gib") or 0 for task in TASKS),
            default=0,
        ) or None
        model["weight_gib"] = model.get("weight_bytes", 0) / (1024**3)
        model["report_file"] = os.path.relpath(
            path.parent / "report.md", start=HERE / "results" / "general_transformers"
        )
        models.append(model)
    complete = [m for m in models if m["full_benchmark_complete"]]
    task_leaders = {}
    if complete:
        for task in TASKS:
            maximum = max(m["tasks"][task]["score"] for m in complete)
            task_leaders[task] = [
                m["model_id"] for m in complete
                if abs(m["tasks"][task]["score"] - maximum) < 1e-12
            ]
    status = "complete" if len(complete) == len(models) else (
        "partial" if any(m.get("status") in {"complete", "partial"} for m in models)
        else "blocked"
    )
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "run_id": run_dir.name,
        "status": status,
        "memory_cap_gib": 30.0,
        "seed": 3407,
        "task_order": list(TASKS),
        "task_labels": TASK_LABELS,
        "task_sizes": TASK_SIZES,
        "models": models,
        "leaders": {
            "overall": ([m["model_id"] for m in complete
                         if abs(m["full_benchmark_macro"] - max(x["full_benchmark_macro"] for x in complete)) < 1e-12]),
            "tasks": task_leaders,
        },
    }


def fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value: Any) -> str:
    return "—" if value is None else f"{value:.1%}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    complete = [m for m in result["models"] if m["full_benchmark_complete"]]
    runtime_versions = sorted({
        "Transformers " + str(m["runtime_versions"]["transformers"])
        + " / PyTorch " + str(m["runtime_versions"]["torch"])
        for m in result["models"]
    })
    lines = [
        "# General Transformers VLM benchmark", "",
        f"Updated: {result['generated_at']}",
        f"Run ID: {result['run_id']}",
        f"Run state: {result['status']}; models in this comparison: {len(result['models'])}; "
        f"complete full-suite runs: {len(complete)}.",
        f"Runtime versions: {', '.join(runtime_versions)}.",
        f"Sampling seed: {result['seed']}; memory ceiling: {result['memory_cap_gib']:.1f} GiB.",
        "The full-suite macro is the unweighted mean of eight local task scores and requires complete coverage on all requested examples.",
        "",
        "## Selection guide", "",
    ]
    if complete:
        best = max(complete, key=lambda model: model["full_benchmark_macro"])
        lines.append(
            f"Best complete overall: **{best['model_id']}** at {pct(best['full_benchmark_macro'])}. "
            "Task leaders below use only complete full-suite models."
        )
    else:
        lines.append("No model completed all eight shared tasks; no quality picks are reported.")
    lines += ["", "## Quality results", "",
              "| Model | Run status | Preflight | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in result["models"]:
        tasks = model.get("tasks", {})
        values = [pct(tasks.get(task, {}).get("score")) for task in TASKS]
        preflight = model.get("preflight", {})
        preflight_text = "passed" if preflight.get("passed") else "failed/not run"
        lines.append(
            f"| {model['model_id']} | {model['status']} | {preflight_text} | "
            + " | ".join(values)
            + f" | {model['complete_tasks']}/8 | {pct(model['full_benchmark_macro'])} |"
        )
    lines += ["", "## Task leaders", ""]
    if complete:
        for task in TASKS:
            leaders = result["leaders"]["tasks"].get(task, [])
            value = max(m["tasks"][task]["score"] for m in complete)
            lines.append(f"- **{TASK_LABELS[task]}:** {', '.join(leaders)} ({pct(value)}).")
    else:
        lines.append("No comparable complete runs.")
    lines += ["", "## GRAB-Lite group scores", "",
              "| Model | Function | Properties | Real figures | Series | Transformations |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in result["models"]:
        groups = model.get("tasks", {}).get("graph_grab_lite", {}).get("group_scores") or {}
        lines.append(
            f"| {model['model_id']} | {pct(groups.get('function'))} | {pct(groups.get('properties'))} | "
            f"{pct(groups.get('real'))} | {pct(groups.get('series'))} | {pct(groups.get('transform'))} |"
        )
    lines += ["", "## Real-image preflight", "",
              "Each model must produce a nonempty TextVQA answer and a parseable AI2D answer before any full task begins. Passing examples are reused in the scored task results.",
              "| Model | Result | AI2D format | TextVQA response |",
              "|---|---|---|---|"]
    for model in result["models"]:
        checks = model.get("preflight", {}).get("checks", {})
        ai2d = checks.get("ai2d", {})
        textvqa = checks.get("textvqa", {})
        lines.append(
            f"| {model['model_id']} | {'passed' if model.get('preflight', {}).get('passed') else 'failed/not run'} | "
            f"{'valid' if ai2d.get('answer_format_ok') else 'invalid/not run'} | "
            f"{'nonempty' if textvqa.get('nonempty_answer') else 'empty/not run'} |"
        )
    lines += ["", "## Runtime and resources", "",
              "Generation time excludes preprocessing. Peak allocated VRAM is process-local, not whole-device memory.",
              "| Model | Task | Completed / requested | Mean seconds | Tokens / second | Peak allocated GiB | Parse errors |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for model in result["models"]:
        for task in TASKS:
            data = model.get("tasks", {}).get(task, {})
            lines.append(
                f"| {model['model_id']} | {TASK_LABELS[task]} | "
                f"{data.get('completed_n', 0)}/{data.get('requested_n', TASK_SIZES[task])} | "
                f"{fmt(data.get('mean_generation_s'))} | "
                f"{fmt(data.get('generated_tokens_per_s'))} | "
                f"{fmt(data.get('peak_allocated_gib'))} | {data.get('parse_errors', 0)} |"
            )
    lines += ["", "## Method", "",
              "Tasks reuse the prepared examples, prompts, sample indices, and scorers described in `../../qwenVL_Bench/README.md`.",
              "AI2D uses 100 test examples; TextVQA uses 100 validation examples; the five additional VQA panel categories use 25 examples each; GRAB-Lite uses all 500 test examples.",
              "Requests run serially with one image and greedy decoding, with a 256-token generation limit. PaliGemma uses `answer en` and its 224-pixel input. Florence uses a separate Transformers 4.44.2 environment and the experimental `<VQA>` wrapper.",
              "Florence's cached processor does not register `<VQA>` and post-processing falls back to plain text. It is scored only if its real-image preflights demonstrate parseable AI2D output and a nonempty TextVQA answer.",
              "", "## Provenance and limitations", ""]
    for model in result["models"]:
        lines.append(
            f"- **{model['model_id']}:** {model['status']}; Transformers {model['runtime_versions']['transformers']}; "
            f"checkpoint weights {model['weight_gib']:.2f} GiB; cache status {model['cache_status']}. "
            f"{model.get('error') or model.get('scoring_route', {}).get('evidence') or ''}"
        )
        lines.append(f"  Source run: `{model['report_file']}`.")
    lines += ["", "Raw per-item outputs and errors remain in each model's `results.json`. The report site publishes aggregate metrics only.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.run_dir.resolve())
    output = HERE / "results" / "general_transformers"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output / "report.md", result)
    print(f"Wrote {output / 'results.json'} and {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
