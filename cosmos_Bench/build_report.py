#!/usr/bin/env python3
"""Build human-readable reports from authoritative Cosmos result records."""
from __future__ import annotations

import json
import platform
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "results" / "results.json"
REPORT_PATH = HERE / "report.md"
SITE_DATA_PATH = HERE / "report-site" / "data.json"
TASKS = [
    ("ai2d", "AI2D test"),
    ("textvqa", "TextVQA validation"),
    ("vqa_truth", "VQA truth"),
    ("vqa_count", "VQA counting"),
    ("vqa_description", "VQA description"),
    ("vqa_affordance", "VQA affordance"),
    ("vqa_negation", "VQA negation"),
    ("graph_grab_lite", "GRAB-Lite graph analysis"),
]
GRAPH_GROUPS = ["properties", "function", "series", "transform", "real"]


def pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def pct(x):
    return "—" if x is None else f"{100*x:.1f}%"


def num(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def mean_score(model, task_ids):
    vals = [model.get("tasks", {}).get(task, {}).get("score") for task in task_ids]
    return sum(vals) / len(vals) if vals and all(v is not None for v in vals) else None


def graph_groups(model):
    items = model.get("tasks", {}).get("graph_grab_lite", {}).get("items", [])
    result = {}
    for group in GRAPH_GROUPS:
        scores = [item["score"] for item in items if item.get("task_type") == group and "score" in item]
        result[group] = sum(scores) / len(scores) if scores else None
    return result


def main():
    result = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    completed = [m for m in result["models"] if m.get("status") == "complete"]
    pending = [m for m in result["models"] if m.get("status") == "pending"]
    unsupported = [m for m in result["models"] if m.get("status") == "unsupported"]
    incomplete = [m for m in result["models"] if m.get("status") == "complete_with_failures"]
    overall_tasks = [key for key, _ in TASKS]
    for model in completed:
        model["overall_macro"] = mean_score(model, overall_tasks)
        model["panel_macro"] = mean_score(model, ["textvqa", "vqa_truth", "vqa_count", "vqa_description", "vqa_affordance", "vqa_negation"])
        model["graph_groups"] = graph_groups(model)
        snapshot = Path(model["snapshot"])
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        text_config = config.get("text_config") or config
        model["checkpoint_context"] = text_config.get("max_position_embeddings")
        model["cached_weight_gib"] = round(
            sum(path.stat().st_size for path in snapshot.rglob("*.safetensors")) / (1024**3), 3
        )
    completed.sort(key=lambda m: m["overall_macro"] or 0, reverse=True)
    pending_names = ", ".join(f"`{m['model_id']}`" for m in pending)
    unsupported_names = ", ".join(f"`{m['model_id']}`" for m in unsupported)
    incomplete_names = ", ".join(f"`{m['model_id']}`" for m in incomplete)
    pending_summary = f"{pending_names} {'is' if len(pending) == 1 else 'are'} pending. {pending[0].get('error', 'Benchmark execution has not completed.')}" if pending_names else ""
    unsupported_summary = f"{unsupported_names} {'remains' if len(unsupported) == 1 else 'remain'} unsupported." if unsupported_names else ""
    incomplete_summary = f"{incomplete_names} {'has' if len(incomplete) == 1 else 'have'} an incomplete run; partial scores are excluded from model rankings." if incomplete_names else ""
    extra_summary = " ".join(text for text in (pending_summary, unsupported_summary, incomplete_summary) if text)
    if not extra_summary:
        extra_summary = "All configured checkpoints completed."
    best = {}
    for task, label in TASKS:
        eligible = [m for m in completed if m.get("tasks", {}).get(task, {}).get("score") is not None]
        best[task] = max(eligible, key=lambda m: m["tasks"][task]["score"])["model_id"] if eligible else None
    runtime = {
        "python": platform.python_version(),
        "torch": pkg("torch"),
        "transformers": pkg("transformers"),
        "qwen_vl_utils": pkg("qwen-vl-utils"),
        "hqq": pkg("hqq"),
        "pillow": pkg("Pillow"),
    }
    aggregate_models = []
    for model in completed:
        aggregate_models.append({
            key: model.get(key) for key in (
                "model_id", "status", "backend", "revision", "max_new_tokens",
                "reasoning_mode", "overall_macro", "panel_macro", "graph_groups", "run_context_tokens",
                "gpu_memory_utilization",
                "checkpoint_context", "cached_weight_gib", "model_load_peak_allocated_gib",
                "peak_gpu_used_gib", "peak_gpu_delta_gib", "benchmark_note",
            )
        } | {
            "tasks": {
                task: {key: data.get(key) for key in (
                    "requested_n", "completed_n", "failed_n", "score", "mean_wall_s",
                    "generated_tokens_per_s", "peak_gpu_used_gib", "peak_gpu_delta_gib",
                )}
                for task, data in model.get("tasks", {}).items()
            }
        })
    aggregate_incomplete = []
    for model in incomplete:
        aggregate_incomplete.append({
            "model_id": model.get("model_id"),
            "status": model.get("status"),
            "backend": model.get("backend"),
            "revision": model.get("revision"),
            "error": model.get("error"),
            "peak_gpu_used_gib": model.get("peak_gpu_used_gib"),
            "tasks": {
                task: {key: data.get(key) for key in (
                    "requested_n", "completed_n", "failed_n", "score", "mean_wall_s",
                    "generated_tokens_per_s", "peak_gpu_used_gib", "peak_gpu_delta_gib",
                )}
                for task, data in model.get("tasks", {}).items()
            },
        })
    site_data = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark_generated_at": result["generated_at"],
        "completion": {"completed": len(completed), "total": len(result["models"])},
        "sample_plan": result["samples_per_task"],
        "seed": result["seed"],
        "memory_cap_gib": result["memory_cap_gib"],
        "execution_mode": result["execution_mode"],
        "runtime": runtime,
        "tasks": [{"id": key, "label": label} for key, label in TASKS],
        "best_by_task": best,
        "models": aggregate_models,
        "incomplete": aggregate_incomplete,
        "pending": pending,
        "unsupported": unsupported,
    }
    SITE_DATA_PATH.write_text(json.dumps(site_data, indent=2), encoding="utf-8")
    page_path = HERE / "report-site" / "index.html"
    page = page_path.read_text(encoding="utf-8")
    marker = '<script id="report-data" type="application/json">'
    begin = page.index(marker) + len(marker)
    end = page.index("</script>", begin)
    page = page[:begin] + json.dumps(site_data, separators=(",", ":")) + page[end:]
    page_path.write_text(page, encoding="utf-8")

    lines = [
        "# Cosmos VLM benchmark",
        "",
        f"**Updated:** {site_data['generated_at']}  ",
        f"**Status:** {len(completed)} complete; {len(incomplete)} incomplete; {len(pending)} pending; {len(unsupported)} unsupported  ",
        f"**Runtime:** Python {runtime['python']}, PyTorch {runtime['torch']}, Transformers {runtime['transformers']}, qwen-vl-utils {runtime['qwen_vl_utils']}, HQQ {runtime['hqq']}  ",
        f"**Execution:** {result['execution_mode']}; {result['memory_cap_gib']} GiB per-process target",
        "",
        "## Summary",
        "",
        f"{len(completed)} checkpoints completed all eight shared task groups. Each completed model scored 825 examples across the eight task groups. {extra_summary}",
        "",
        f"The overall score is the unweighted macro average of AI2D, TextVQA, the five additional VQA panel categories, and GRAB-Lite. It is a local comparison score, not an official leaderboard metric. Among complete models, `{completed[0]['model_id']}` has the highest overall macro score. Models with incomplete coverage are excluded from comparisons.",
        "",
        "| Model | Overall macro | VQA panel macro | Completed examples | Peak whole-GPU memory | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model in completed:
        requested = sum(task.get("requested_n", 0) for task in model.get("tasks", {}).values())
        done = sum(task.get("completed_n", 0) for task in model.get("tasks", {}).values())
        lines.append(f"| `{model['model_id']}` | {pct(model['overall_macro'])} | {pct(model['panel_macro'])} | {done}/{requested} | {num(model.get('peak_gpu_used_gib'))} GiB | Complete |")
    for model in unsupported:
        lines.append(f"| `{model['model_id']}` | — | — | 0/0 | — | Unsupported |")
    for model in pending:
        lines.append(f"| `{model['model_id']}` | — | — | 0/0 | — | Pending GPU |")
    for model in incomplete:
        requested = sum(task.get("requested_n", 0) for task in model.get("tasks", {}).values())
        done = sum(task.get("completed_n", 0) for task in model.get("tasks", {}).values())
        lines.append(f"| `{model['model_id']}` | — | — | {done}/{requested} | {num(model.get('peak_gpu_used_gib'))} GiB | Incomplete |")

    lines += [
        "",
        "## Quality results",
        "",
        "Scores use the shared fixed sample IDs selected with seed 3407. Every completed model has full coverage for each listed task.",
        "",
        "| Model | AI2D (100) | TextVQA (100) | Truth (25) | Counting (25) | Description (25) | Affordance (25) | Negation (25) | GRAB-Lite (500) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in completed:
        vals = [pct(model["tasks"].get(task, {}).get("score")) for task, _ in TASKS]
        lines.append(f"| `{model['model_id']}` | " + " | ".join(vals) + " |")
    if incomplete:
        lines += ["", "### Partial runs", "", "These scores use only completed examples and are excluded from rankings.", "", "| Model | Task | Completed | Failed | Partial score |", "|---|---|---:|---:|---:|"]
        for model in incomplete:
            for task, label in TASKS:
                data = model.get("tasks", {}).get(task)
                if data:
                    lines.append(f"| `{model['model_id']}` | {label} | {data.get('completed_n', 0)}/{data.get('requested_n', 0)} | {data.get('failed_n', 0)} | {pct(data.get('score'))} |")
    lines += ["", "### Best complete model by task", "", "| Task | Highest score | Model |", "|---|---:|---|"]
    for task, label in TASKS:
        model_id = best[task]
        score = next((m["tasks"][task]["score"] for m in completed if m["model_id"] == model_id), None)
        lines.append(f"| {label} | {pct(score)} | `{model_id}` |")

    lines += [
        "",
        "## Runtime and resources",
        "",
        "Latency is mean generation-call time per example. Throughput is generated tokens divided by generation-call time. Peak memory is whole-device `nvidia-smi` usage sampled during each task, including desktop use; it is not process-only VRAM. The benchmark used a 30 GiB per-process allocation target.",
        "",
        "| Model | Task | Completed | Mean latency (s) | Output tokens/s | Peak whole-GPU (GiB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in completed:
        for task, label in TASKS:
            data = model["tasks"][task]
            lines.append(f"| `{model['model_id']}` | {label} | {data['completed_n']}/{data['requested_n']} | {num(data.get('mean_wall_s'), 3)} | {num(data.get('generated_tokens_per_s'), 1)} | {num(data.get('peak_gpu_used_gib'), 2)} |")

    lines += [
        "",
        "## Context and hardware fit",
        "",
        "Advertised context comes from the cached checkpoint configuration. The run-context column shows explicit server caps where configured. Cached weight size sums local safetensors shards. Model-load peak is this process's CUDA allocation; task peak is whole-device GPU use.",
        "",
        "| Model | Advertised context | Run context | Cached weights | Model-load peak allocated | Peak whole GPU |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for model in completed:
        context = model.get("checkpoint_context")
        run_context = model.get("run_context_tokens")
        run_context_label = f"{run_context:,} tokens" if run_context else "Checkpoint default"
        lines.append(
            f"| `{model['model_id']}` | {f'{context:,}' if context else '—'} | {run_context_label} | "
            f"{num(model.get('cached_weight_gib'), 3)} GiB | "
            f"{num(model.get('model_load_peak_allocated_gib'), 2)} GiB | "
            f"{num(model.get('peak_gpu_used_gib'), 2)} GiB |"
        )

    lines += ["", "## GRAB-Lite task groups", "", "The overall score is exact-match accuracy across all 500 questions. Group scores below use the five prepared task categories.", "", "| Model | Properties | Function | Series | Transform | Real figures |", "|---|---:|---:|---:|---:|---:|"]
    for model in completed:
        groups = model["graph_groups"]
        lines.append(f"| `{model['model_id']}` | " + " | ".join(pct(groups[g]) for g in GRAPH_GROUPS) + " |")

    lines += [
        "",
        "## Methodology",
        "",
        "- **AI2D:** cached test split, 100 fixed examples, exact A/B/C/D answer accuracy. [Dataset card](https://huggingface.co/datasets/lmms-lab/ai2d).",
        "- **TextVQA:** cached validation split, 100 fixed examples, normalized human-answer consensus score. [Dataset card](https://huggingface.co/datasets/facebook/textvqa).",
        "- **VQA panel:** 25 fixed examples each for truth, counting, visual description, affordance/function, and negation; the same 100 TextVQA examples provide OCR. Panel macro averages these six categories equally. It is a local metric. Sources: [VQAv2](https://visualqa.org/download.html), [PercepTax](https://huggingface.co/datasets/RyanWW/PercepTaxBench), and [NegVQA](https://arxiv.org/abs/2505.22946).",
        "- **GRAB-Lite:** complete 500-question fixed split, normalized exact match, with five category scores. [Dataset card](https://huggingface.co/datasets/jonathan-roberts1/GRAB-lite).",
        "- **Cosmos Reason1 image support:** the official card lists image and video inputs and documents the Qwen2.5-VL architecture. This run loads its Qwen2.5-VL conditional-generation class. [Model card](https://huggingface.co/nvidia/Cosmos-Reason1-7B).",
        "- **Cosmos3 Nano support route:** the official card documents vLLM 0.21.0 for CUDA 13 or 0.19.1 for CUDA 12.8, plus the `vllm-cosmos3` plugin and a reasoner architecture override. This run used 16,384-token context, one sequence, and 88% GPU memory utilization to leave headroom for full-resolution vision activations on the RTX 5090. [Model card](https://huggingface.co/nvidia/Cosmos3-Nano).",
        "- Requests were serial and single-image, using the shared task prompts, scorers, prepared panel data, and seed 3407. Dataset Arrow files and cached model snapshots were read offline.",
        "- AI2D and open-answer tasks used a 256-token cap except Cosmos Reason1 and Cosmos3 Nano, which used a 4096-token cap as recommended by their model cards. No Reason1 output reached that cap. Reason2 used checkpoint generation defaults with deterministic seeds. Cosmos3 Edge used `enable_thinking=False` in its cached chat template so outputs follow the shared answer-only contract; the same deterministic seed and 256-token limit were used. No output reached the cap in the completed Cosmos3 Edge run.",
        "- The Transformers cache used HQQ 8-bit KV storage where supported. The cached model revisions are recorded in `results.json`.",
        "",
        "## Coverage and limitations",
        "",
        "| Checkpoint | Cached revision | License label | Coverage status |",
        "|---|---|---|---|",
    ]
    for model in completed + incomplete + pending + unsupported:
        rid = model.get("revision", "—")
        label = "NVIDIA Open Model License" if "Cosmos-Reason" in model["model_id"] else "OpenMDW1.1"
        status = model.get("status", "unknown")
        reason = model.get("error", "all eight task groups completed")
        lines.append(f"| `{model['model_id']}` | `{rid}` | {label} | {status}: {reason} |")
    nano = next((model for model in completed + incomplete if model["model_id"] == "nvidia/Cosmos3-Nano"), None)
    if nano and nano.get("benchmark_note"):
        nano_note = f"- **Cosmos3 Nano run note:** {nano['benchmark_note']}"
    else:
        nano_note = "- Cosmos3 Nano has no incomplete run in this report."
    lines += [
        "",
        nano_note,
        "- Cosmos3 Edge ran with reasoning disabled through a supported chat-template option. Its scores are therefore for answer-only decoding and are not directly equivalent to a default-thinking run.",
        "- Results are local fixed-sample comparisons, not broad accuracy estimates or leaderboard scores. No model-load, licensing acceptance, or context-capacity claims are inferred from cache presence.",
        "",
        "## Artifacts",
        "",
        "- [Aggregate-only report page](report-site/index.html)",
        "- Authoritative per-item results, errors, latencies, token counts, and memory observations: [`results/results.json`](results/results.json).",
        "- Aggregate site payload: [`report-site/data.json`](report-site/data.json).",
        "- Benchmark runner: [`run_benchmark.py`](run_benchmark.py).",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH} and {SITE_DATA_PATH}")


if __name__ == "__main__":
    main()
