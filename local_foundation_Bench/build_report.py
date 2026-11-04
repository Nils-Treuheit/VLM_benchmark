#!/usr/bin/env python3
"""Build the aggregate report from recorded per-run benchmark artifacts."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
TASK_LABELS = {
    "ai2d": "AI2D",
    "textvqa": "TextVQA / OCR",
    "vqa_truth": "Truth",
    "vqa_count": "Counting",
    "vqa_description": "Description",
    "vqa_affordance": "Affordance",
    "vqa_negation": "Negation",
    "graph_grab_lite": "GRAB-Lite",
}


def latest_runs():
    found = {}
    for path in RUNS.rglob("results.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            model = data["model"]
            model_id = model["model_id"]
            generated = data.get("generated_at", "")
        except (OSError, ValueError, KeyError, TypeError):
            continue
        prior = found.get(model_id)
        if prior is None or generated > prior[0]:
            found[model_id] = (generated, path, data)
    return [(path, data) for _, path, data in sorted(found.values(), key=lambda item: item[0])]


def fmt(value, suffix=""):
    return "—" if value is None else f"{value:.2f}{suffix}"


def main():
    runs = latest_runs()
    inventory = json.loads((HERE / "candidate_inventory.json").read_text(encoding="utf-8"))["candidates"]
    completed = [data for _, data in runs if data["model"].get("full_benchmark_complete")]
    tasks = list(TASK_LABELS)
    panel_tasks = ("vqa_truth", "vqa_count", "vqa_description", "textvqa", "vqa_affordance", "vqa_negation")
    lines = [
        "# Local foundation and GGUF VLM benchmark", "",
        f"Updated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        f"Run timestamps: {', '.join(data.get('generated_at', '—') for _, data in runs)}.",
        f"Completion: {len(completed)} of {len(runs)} latest model runs completed all 825 requests.",
        "Execution: serial, one image per request, fixed examples, no batching.",
        "",
        "## Summary", "",
        "Measured candidates include the configured Qwen3.8 Ollama route and four Liquid VLM GGUF variants. The corrected LFM2.5-VL 1.6B run is the current result for that model ID; its earlier run remains in the attempt history.",
        "The full-suite macro is the unweighted mean of eight task scores. It is reported only for models that completed every requested example.",
        "",
        "## Quality results", "",
        "| Model and route | Status | Samples | Full-suite macro | VQA panel macro | " + " | ".join(TASK_LABELS[t] for t in tasks) + " |",
        "|---|---|---:|---:|---:|" + "---:|" * len(tasks),
    ]
    for path, result in runs:
        model = result["model"]
        total_completed = sum(value.get("completed_n", 0) for value in model.get("tasks", {}).values())
        values = []
        for task in tasks:
            score = model.get("tasks", {}).get(task, {}).get("score")
            values.append("—" if score is None else f"{score:.1%}")
        route = model.get("runtime") or model.get("backend")
        model_name = f"{model['model_id']} ({route})"
        macro = model.get("full_benchmark_macro")
        panel_values = [model.get("tasks", {}).get(task, {}).get("score") for task in panel_tasks]
        panel_complete = all(
            model.get("tasks", {}).get(task, {}).get("completed_n") == model.get("tasks", {}).get(task, {}).get("requested_n")
            and model.get("tasks", {}).get(task, {}).get("score") is not None
            for task in panel_tasks
        )
        panel_macro = statistics.mean(panel_values) if panel_complete else None
        lines.append(
            f"| {model_name} | {model.get('status')} | {total_completed}/825 | "
            f"{'—' if macro is None else f'{macro:.1%}'} | "
            f"{'—' if panel_macro is None else f'{panel_macro:.1%}'} | " + " | ".join(values) + " |"
        )
    lines += ["", "## Selection guide", ""]
    if completed:
        best_macro = max(completed, key=lambda d: d["model"]["full_benchmark_macro"])
        m = best_macro["model"]
        lines.append(f"Best complete full-suite macro: `{m['model_id']}` at {m['full_benchmark_macro']:.1%}.")
        for task in tasks:
            candidates = [d for d in completed if d["model"].get("tasks", {}).get(task, {}).get("score") is not None]
            if candidates:
                best = max(candidates, key=lambda d: d["model"]["tasks"][task]["score"])
                value = best["model"]["tasks"][task]["score"]
                lines.append(f"Best complete result for {TASK_LABELS[task]}: `{best['model']['model_id']}` at {value:.1%}.")
        lines.append("Picks use only full-suite-complete models; ties retain the first listed model.")
    else:
        lines.append("No model completed the full suite, so no quality pick is made.")
    lines += ["", "## Real-image preflight", "", "| Model | AI2D format | TextVQA nonempty answer |", "|---|---:|---:|"]
    for _, result in runs:
        checks = result.get("preflight", {}).get("checks", {})
        ai2d = checks.get("ai2d", {}).get("answer_format_ok")
        textvqa = checks.get("textvqa", {}).get("nonempty_answer")
        lines.append(f"| {result['model']['model_id']} | {'pass' if ai2d else 'fail' if ai2d is not None else '—'} | {'pass' if textvqa else 'fail' if textvqa is not None else '—'} |")
    lines += ["", "## Runtime and resources", "", "| Model | Context | Peak whole-device VRAM GiB | Memory scope |", "|---|---:|---:|---|"]
    for _, result in runs:
        model = result["model"]
        lines.append(
            f"| {model['model_id']} | {model.get('configured_context', '—')} | "
            f"{fmt(model.get('whole_device_peak_gib'))} | "
            f"{model.get('whole_device_memory_method', 'whole-device sample; desktop baseline included')} |"
        )
    lines += ["", "### Per-task latency and throughput", "", "| Model | Task | Completed | Mean generation seconds | Output tokens/second |", "|---|---|---:|---:|---:|"]
    for _, result in runs:
        model = result["model"]
        for task in tasks:
            value = model.get("tasks", {}).get(task, {})
            lines.append(
                f"| {model['model_id']} | {TASK_LABELS[task]} | "
                f"{value.get('completed_n', 0)}/{value.get('requested_n', 0)} | "
                f"{fmt(value.get('mean_generation_s'))} | {fmt(value.get('generated_tokens_per_s'))} |"
            )
    lines += ["", "## Candidate inventory", "", "| Checkpoint | License metadata | Status | Disposition |", "|---|---|---|---|"]
    for candidate in inventory:
        source = f" ([model card]({candidate['source']}))" if candidate.get("source") else ""
        lines.append(f"| `{candidate['checkpoint']}`{source} | {candidate.get('license', 'See linked model card')} | {candidate['status']} | {candidate['detail']} |")
    lines += [
        "", "## Methodology", "",
        "All measured models use the shared fixed task definitions, examples, prompts, and scorers in the [Qwen-VL benchmark README](../qwenVL_Bench/README.md). The sample seed is 3407. Each request carries one image and runs serially. Liquid GGUF variants use a host-native llama.cpp b12097 CUDA 13.2 build.",
        "Primary task sources: [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d), [TextVQA](https://huggingface.co/datasets/facebook/textvqa), [VQAv2](https://visualqa.org/download.html), [PercepTaxBench](https://huggingface.co/datasets/RyanWW/PercepTaxBench), [NegVQA](https://arxiv.org/abs/2505.22946), and [GRAB-Lite](https://huggingface.co/datasets/jonathan-roberts1/GRAB-lite).",
        "AI2D uses exact four-choice letter accuracy. TextVQA and the truth, count, description, and OCR categories use the shared normalized human-answer consensus. Affordance uses normalized reference matching. Negation uses exact A/B accuracy. GRAB-Lite uses exact match, including its five task-group scores in the JSON artifacts.",
        "The six-category VQA panel macro weights truth, counting, description, OCR, affordance, and negation equally. The full-suite macro weights all eight task scores equally and is local comparison evidence, not an official benchmark score.",
        "",
        "## Backend settings and provenance", "",
    ]
    for path, result in runs:
        m = result["model"]
        lines.append(f"### `{m['model_id']}`")
        lines.append("")
        lines.append(f"Backend: {m.get('backend')}; context: {m.get('configured_context', '—')} tokens; status: {m.get('status')}.")
        if m.get("model_card"):
            lines.append(f"Base model card: [{m['model_card']}]({m['model_card']}).")
        if m.get("revision"):
            lines.append(f"Cached revision: `{m['revision']}`.")
        if m.get("advertised_context"):
            lines.append(f"Advertised context: {m['advertised_context']} tokens; configured run context: {m.get('configured_context', '—')} tokens.")
        if m.get("quantization"):
            lines.append(f"Quantization: {m['quantization']}.")
        if m.get("image_processing"):
            lines.append(f"Image processing: `{json.dumps(m['image_processing'], sort_keys=True)}`.")
        if m.get("generation"):
            lines.append(f"Generation settings: `{json.dumps(m['generation'], sort_keys=True)}`.")
        if m.get("runtime_versions"):
            lines.append(f"Runtime versions: `{json.dumps(m['runtime_versions'], sort_keys=True)}`.")
        elif m.get("runtime"):
            lines.append(f"Runtime route: `{m['runtime']}`; runtime version was unavailable through the benchmark API.")
        if m.get("weight_bytes"):
            lines.append(f"Cached model and projector size: {m['weight_bytes'] / (1024**3):.2f} GiB.")
        if result.get("repair_attempts"):
            lines.append(f"Recovery: {len(result['repair_attempts'])} failed request(s) were retried and scored; the original error remains recorded in `repair_attempts` in the results JSON.")
        lines.append(f"Raw outputs, scores, errors, latency, token counts, and memory samples: [results JSON](results/runs/{path.parent.parent.name}/{path.parent.name}/results.json).")
        lines.append("")
    lines += [
        "## Limitations", "",
        "These fixed samples are selection evidence, not leaderboard scores. The Liquid GGUF runner sends JPEG quality 92. LFM2.5-VL 1.6B uses a 64 image-token cap; the other Liquid variants leave image-token limits at the llama.cpp default. The original model-card processor settings are not fully enforced by the GGUF routes. Whole-device memory samples include the desktop baseline; they are not process-local estimates. Advertised maximum context is distinct from the configured context.",
        "License labels are taken from the linked model cards at their cached revisions where available. This report records model provenance and does not grant redistribution or use rights.",
        "The Qwen MTP4 route uses a 128K configured context with thinking disabled. Liquid GGUF variants use an 8K configured context. The current LFM2.5-VL 1.6B result uses temperature 0.1, min-p 0.15, repetition penalty 1.05, top-k 50, a fixed seed, and the Unsloth tutorial's 64 image-token cap. LFM2.5 3B and both LFM3 quantizations use their model-card sampling settings. LFM3 uses a 4,096-token output allowance because 512 tokens ended before its first final answer. The NVFP4 checkpoint uses an upstream-supported llama.cpp build on the RTX 5090. Unsupported and incomplete artifacts were not ranked.",
        "The runner records whole-device peak VRAM for each model run. Its sampling does not reset at task boundaries, so per-task peak VRAM is not reported.",
        "",
        "## Attempts", "",
    ]
    for path, result in runs:
        lines.append(f"- `{path.parent.parent.name}/{path.parent.name}`: {result['model'].get('status')}.")
    failed = []
    selected_paths = {path for path, _ in runs}
    for path in RUNS.rglob("results.json"):
        if path in selected_paths:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            failed.append((path, result["model"].get("status"), result["model"].get("error")))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for path, status, error in sorted(failed):
        lines.append(f"- `{path.parent.parent.name}/{path.parent.name}`: {status}; {error or 'incomplete run retained'}.")
    lines += ["", "Raw item outputs are kept in run artifacts and are not copied into the static report page.", ""]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {HERE / 'report.md'} from {len(runs)} latest model run(s).")


if __name__ == "__main__":
    main()
