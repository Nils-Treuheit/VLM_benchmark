#!/usr/bin/env python3
"""Prepare compact public-facing benchmark aggregates for the static viewer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "corrected" / "results.json"
PANEL_SOURCE = ROOT / "results" / "vqa_panel" / "results.json"
GRAPH_SOURCE = ROOT / "results" / "graph_panel" / "results.json"
LARGE_PANEL_SOURCE = ROOT / "results" / "vqa_panel_large_30gb" / "results.json"
CONTEXT_SOURCE = ROOT / "results" / "context_profiles_30gb" / "results.json"
DESTINATION = ROOT / "data.json"
DESKTOP_RESERVE_GIB = 1.0
TASK_ORDER = ("ai2d", "textvqa")
PANEL_TASKS = ("vqa_truth", "vqa_count", "vqa_description", "textvqa", "vqa_affordance", "vqa_negation")
TASK_INFO = {
    "ai2d": {
        "title": "AI2D diagram reasoning",
        "dataset": "lmms-lab/ai2d",
        "split": "test",
        "url": "https://huggingface.co/datasets/lmms-lab/ai2d",
        "samples": 100,
        "available": 3088,
        "inspection": "One diagram image and a four-option question. The prompt requests only A, B, C, or D.",
        "measure": "Exact option accuracy. A missing or unparsable letter counts as incorrect.",
    },
    "textvqa": {
        "title": "TextVQA image text reading",
        "dataset": "facebook/textvqa",
        "split": "validation",
        "url": "https://huggingface.co/datasets/facebook/textvqa",
        "samples": 100,
        "available": 5000,
        "inspection": "One natural image and an open question. The prompt requests the shortest exact text visible in the image.",
        "measure": "Normalized answer consensus: mean leave-one-out reference matches, capped at three matches per three annotators.",
    },
    "vqa_truth": {
        "title": "VQAv2 visible truth questions", "dataset": "VQAv2", "split": "validation",
        "url": "https://visualqa.org/download.html", "samples": 25,
        "inspection": "Yes/no questions sampled from validation annotations with answer_type=yes/no.",
        "measure": "VQAv2 human-answer consensus score after normalization.",
    },
    "vqa_count": {
        "title": "VQAv2 counting", "dataset": "VQAv2", "split": "validation",
        "url": "https://visualqa.org/download.html", "samples": 25,
        "inspection": "Number questions whose wording begins with “How many”.",
        "measure": "VQAv2 human-answer consensus score after normalization.",
    },
    "vqa_description": {
        "title": "VQAv2 visual description", "dataset": "VQAv2", "split": "validation",
        "url": "https://visualqa.org/download.html", "samples": 25,
        "inspection": "Open questions about visible content, filtered from answer_type=other.",
        "measure": "VQAv2 human-answer consensus score after normalization.",
    },
    "vqa_affordance": {
        "title": "PercepTax affordance and function", "dataset": "RyanWW/PercepTaxBench", "split": "real/test",
        "url": "https://huggingface.co/datasets/RyanWW/PercepTaxBench", "samples": 25,
        "inspection": "Real-image questions from affordance, function, and object-repurposing types.",
        "measure": "Normalized open-answer match against the dataset answer.",
    },
    "vqa_negation": {
        "title": "NegVQA negation", "dataset": "yuhuizhang/NegVQA", "split": "dev",
        "url": "https://huggingface.co/datasets/yuhuizhang/NegVQA", "samples": 25,
        "inspection": "Two-choice questions with negated wording; one image per request.",
        "measure": "Exact accuracy on the correct A/B option letter.",
    },
}


def edge_context(model: dict, ceiling: float) -> int | None:
    profile = model.get("context_profile") or {}
    tiers = profile.get("tiers") or []
    low = next((tier for tier in tiers if tier.get("tier") == "min"), None)
    if low is None:
        return None
    high = max(tiers, key=lambda tier: tier.get("context_tokens", 0), default=low)
    low_context = int(low.get("context_tokens", 8192))
    high_context = int(high.get("context_tokens", low_context))
    low_vram = float(low.get("estimated_vram_gib", 0))
    high_vram = float(high.get("estimated_vram_gib", low_vram))
    context_cap = min(
        int(profile.get("estimated_max_context") or high_context),
        int(model.get("checkpoint_context") or high_context),
    )
    model_budget = max(low_vram, ceiling - DESKTOP_RESERVE_GIB)
    if context_cap <= low_context or high_context <= low_context or high_vram <= low_vram:
        fit = min(low_context, context_cap)
    elif model_budget >= high_vram:
        fit = context_cap
    else:
        fit = low_context + (model_budget - low_vram) * (high_context - low_context) / (high_vram - low_vram)
        fit = min(context_cap, max(low_context, int(fit // 1024) * 1024))
    return int(fit)


def main() -> None:
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    panel = json.loads(PANEL_SOURCE.read_text(encoding="utf-8")) if PANEL_SOURCE.is_file() else {"models": []}
    large_panel = json.loads(LARGE_PANEL_SOURCE.read_text(encoding="utf-8")) if LARGE_PANEL_SOURCE.is_file() else {"models": []}
    context_data = json.loads(CONTEXT_SOURCE.read_text(encoding="utf-8")) if CONTEXT_SOURCE.is_file() else {"models": []}
    graph = json.loads(GRAPH_SOURCE.read_text(encoding="utf-8")) if GRAPH_SOURCE.is_file() else {"models": []}
    panel_by_id = {model.get("model_id"): model for model in panel.get("models", [])}
    large_by_id = {model.get("model_id"): model for model in large_panel.get("models", [])}
    context_by_id = {model.get("model_id"): model for model in context_data.get("models", [])}
    graph_by_id = {model.get("model_id"): model for model in graph.get("models", [])}
    models = []
    originals = {model.get("model_id"): model for model in result["models"]}
    # Include every model with an existing baseline score or a completed VQA panel.
    all_ids = list(originals)
    all_ids.extend(model["model_id"] for model in panel.get("models", []) if model.get("model_id") not in originals)
    all_ids.extend(model["model_id"] for model in large_panel.get("models", []) if model.get("model_id") not in originals and model.get("model_id") not in panel_by_id)
    for model_id in all_ids:
        original = originals.get(model_id)
        panel_model = panel_by_id.get(model_id, {})
        large_model = large_by_id.get(model_id, {})
        context_model = context_by_id.get(model_id, {})
        if original is None:
            original = {key: value for key, value in (panel_model or large_model).items() if key != "tasks"}
            original["tasks"] = {}
        if original.get("status") != "complete" or any(
            original.get("tasks", {}).get(task, {}).get("completed_n") != result["samples_per_task"]
            for task in TASK_ORDER
        ):
            # New variants may have a completed VQA panel but no AI2D/TextVQA row yet.
            if not panel_model and not large_model:
                continue
        model = {key: value for key, value in original.items() if key != "tasks"}
        model["tasks"] = {}
        for task, metrics in original.get("tasks", {}).items():
            model["tasks"][task] = {key: value for key, value in metrics.items() if key != "items"}
            model["tasks"][task]["parse_errors"] = sum(
                bool(item.get("parse_error")) for item in metrics.get("items", [])
            )
        for task, metrics in panel_model.get("tasks", {}).items():
            model["tasks"][task] = {key: value for key, value in metrics.items() if key != "items"}
            model["tasks"][task]["parse_errors"] = sum(
                bool(item.get("parse_error")) for item in metrics.get("items", [])
            )
        # Use large-context reruns when they completed every requested sample.
        for task, metrics in large_model.get("tasks", {}).items():
            requested = metrics.get("requested_n", large_panel.get("samples_per_task", 25))
            if metrics.get("completed_n") != requested:
                continue
            model["tasks"][task] = {key: value for key, value in metrics.items() if key != "items"}
            model["tasks"][task]["parse_errors"] = sum(bool(item.get("parse_error")) for item in metrics.get("items", []))
            model["configured_context"] = large_model.get("configured_context")
            model["kv_cache"] = large_model.get("kv_cache")
            model["backend"] = large_model.get("backend", model.get("backend"))
        if context_model.get("context_profile"):
            model["context_profile"] = context_model["context_profile"]
            model["verified_context_cap"] = context_model.get("profiled_context_tokens")
        panel_scores = [model["tasks"].get(task, {}).get("score") for task in PANEL_TASKS]
        model["vqa_panel_score"] = sum(panel_scores) / len(PANEL_TASKS) if all(
            score is not None for score in panel_scores
        ) else None
        model["vqa_panel_complete"] = all(
            model["tasks"].get(task, {}).get("completed_n") == (100 if task == "textvqa" else 25)
            for task in PANEL_TASKS
        )
        if large_model:
            model["status"] = "complete" if model["vqa_panel_complete"] else "partial"
            if not model["vqa_panel_complete"]:
                model["preliminary_status"] = "large-context run lacks a complete six-category panel; excluded from complete-score recommendations"
        graph_model = graph_by_id.get(model["model_id"], {})
        graph_metrics = graph_model.get("tasks", {}).get("graph_grab_lite", {})
        graph_items = [item for item in graph_metrics.get("items", []) if "error" not in item]
        graph_types = sorted({item.get("task_type") for item in graph_items if item.get("task_type")})
        model["graph_score"] = graph_metrics.get("score")
        model["graph_complete_n"] = graph_metrics.get("completed_n")
        model["graph_task_scores"] = {
            task: sum(item.get("score", 0) for item in graph_items if item.get("task_type") == task)
                  / sum(1 for item in graph_items if item.get("task_type") == task)
            for task in graph_types
        }
        model["graph_peak_gpu_used_gib"] = graph_metrics.get("peak_gpu_used_gib")
        model["graph_mean_wall_s"] = graph_metrics.get("mean_wall_s")
        model["graph_tokens_s"] = graph_metrics.get("generated_tokens_per_s")
        models.append(model)

    edge_tables = {}
    for ceiling in (12, 16):
        rows = []
        for model in models:
            if model.get("status") != "complete" or not all(task in model["tasks"] for task in TASK_ORDER):
                continue
            tiers = (model.get("context_profile") or {}).get("tiers", [])
            min_tier = next((tier for tier in tiers if tier.get("tier") == "min"), None)
            if min_tier is None:
                continue
            estimated_total = float(min_tier["estimated_vram_gib"]) + DESKTOP_RESERVE_GIB
            if estimated_total >= ceiling:
                continue
            rows.append({
                "model_id": model["model_id"],
                "benchmark_peak_gib": max(model["tasks"][task].get("peak_gpu_used_gib") or 0 for task in TASK_ORDER),
                "estimated_vram_8k_gib": estimated_total,
                "max_context_fit": edge_context(model, ceiling),
                "ai2d_score": model["tasks"]["ai2d"].get("score"),
                "textvqa_score": model["tasks"]["textvqa"].get("score"),
                "vqa_panel_score": model.get("vqa_panel_score"),
                "benchmark_context": model.get("configured_context", model.get("checkpoint_context")),
                "ai2d_mean_s": model["tasks"]["ai2d"].get("mean_wall_s"),
                "textvqa_mean_s": model["tasks"]["textvqa"].get("mean_wall_s"),
                "ai2d_tokens_s": model["tasks"]["ai2d"].get("generated_tokens_per_s"),
                "textvqa_tokens_s": model["tasks"]["textvqa"].get("generated_tokens_per_s"),
            })
        rows.sort(key=lambda row: -((row["ai2d_score"] or 0) + (row["textvqa_score"] or 0)) / 2)
        edge_tables[str(ceiling)] = rows

    score_tasks = ("ai2d", *PANEL_TASKS)
    complete = [model for model in models if model.get("status") == "complete" and model.get("vqa_panel_complete")
                and all(model.get("tasks", {}).get(task, {}).get("score") is not None for task in score_tasks)]

    def mean_score(model):
        return sum(model["tasks"][task]["score"] for task in score_tasks) / len(score_tasks)

    def recommendation(label, model, rationale, extra=None):
        if model is None:
            return None
        result = {"label": label, "model_id": model["model_id"], "rationale": rationale,
                  "combined_score": mean_score(model),
                  "ai2d_score": model["tasks"]["ai2d"]["score"],
                  "textvqa_score": model["tasks"]["textvqa"]["score"],
                  "vqa_panel_score": model.get("vqa_panel_score"),
                  "vqa_category_scores": {task: model["tasks"][task]["score"] for task in PANEL_TASKS},
                  "peak_gpu_gib": max(model["tasks"][task].get("peak_gpu_used_gib") or 0 for task in score_tasks),
                  "context": model.get("configured_context") or model.get("checkpoint_context"),
                  "mean_tokens_s": sum(model["tasks"][task].get("generated_tokens_per_s") or 0 for task in score_tasks) / len(score_tasks),
                  "backend": model.get("backend")}
        if extra:
            result.update(extra)
        return result

    quality_leader = max(complete, key=mean_score, default=None)
    ai2d_leader = max(complete, key=lambda model: model["tasks"]["ai2d"]["score"], default=None)
    textvqa_leader = max(complete, key=lambda model: model["tasks"]["textvqa"]["score"], default=None)
    thinking = [model for model in complete if "thinking" in model["model_id"].lower()]
    thinking_leader = max(thinking, key=mean_score, default=None)
    # Use the same acceptable-quality floor as the speed pick for long context.
    long_context = [model for model in complete
                    if int(model.get("configured_context") or model.get("checkpoint_context") or 0) >= 256 * 1024
                    and (model["tasks"]["ai2d"]["score"] + model["tasks"]["textvqa"]["score"]) / 2 > .75
                    and model.get("vqa_panel_score") is not None
                    and model["vqa_panel_score"] >= .50]
    long_context_leader = max(long_context,
                              key=lambda model: (mean_score(model), int(model.get("configured_context") or model.get("checkpoint_context") or 0)),
                              default=None)
    # Keep the speed pick useful: require >75% mean on AI2D and TextVQA,
    # plus at least 50% on the six-category VQA skill-panel macro score.
    speed_candidates = [model for model in complete
                        if (model["tasks"]["ai2d"]["score"] + model["tasks"]["textvqa"]["score"]) / 2 > .75
                        and model.get("vqa_panel_score") is not None
                        and model["vqa_panel_score"] >= .50]
    speed_leader = max(speed_candidates, key=lambda model: sum(
        model["tasks"][task].get("generated_tokens_per_s") or 0 for task in score_tasks
    ) / len(score_tasks), default=None)
    edge_picks = []
    for ceiling, title in ((12, "Best edge fit under 12 GiB"), (16, "Best edge fit under 16 GiB")):
        candidates = edge_tables[str(ceiling)]
        if candidates:
            row = max(candidates, key=lambda item: ((item["ai2d_score"] or 0) + (item["vqa_panel_score"] or 0)) / 2)
            model = next(m for m in complete if m["model_id"] == row["model_id"])
            edge_picks.append(recommendation(title, model,
                f"Highest mean task score among models whose estimated 8K residency fits the {ceiling} GiB tier.",
                {"edge_limit_gib": ceiling, "edge_max_context": row["max_context_fit"],
                 "edge_estimated_vram_8k_gib": row["estimated_vram_8k_gib"]}))

    graph_specialist = next((m for m in complete if "g-substrate" in m["model_id"].lower()), None)
    if graph_specialist and graph_specialist.get("graph_complete_n") == 500 and graph_specialist.get("graph_score") is not None:
        all_scores = [graph_specialist["tasks"][task]["score"] for task in score_tasks]
        all_scores.append(graph_specialist["graph_score"])
        graph_peak = max(
            [graph_specialist["tasks"][task].get("peak_gpu_used_gib") or 0 for task in score_tasks]
            + [graph_specialist.get("graph_peak_gpu_used_gib") or 0]
        )
        graph_specialist_rec = recommendation(
            "Graph specialist", graph_specialist,
            "G-Substrate is a graph-reasoning fine-tune. Mean Score equally averages AI2D, the six VQA panel measures, and GRAB-Lite. GRAB-Lite covers properties, functions, series, transformations, and real figures.",
            {"graph_score": graph_specialist["graph_score"],
             "graph_task_scores": graph_specialist.get("graph_task_scores", {}),
             "all_benchmark_mean_score": sum(all_scores) / len(all_scores),
             "peak_gpu_gib": graph_peak,
             "context": graph_specialist.get("configured_context") or graph_specialist.get("checkpoint_context")}
        )
    else:
        graph_specialist_rec = recommendation(
            "Graph specialist", graph_specialist,
            "G-Substrate is a graph-reasoning fine-tune for relational visual structure; GRAB-Lite results are not available yet.")

    graph_rows = []
    baseline_id = "Qwen/Qwen3-VL-2B-Instruct"
    baseline_graph_score = graph_by_id.get(baseline_id, {}).get("tasks", {}).get("graph_grab_lite", {}).get("score")
    for model in models:
        if model.get("graph_score") is None:
            continue
        graph_rows.append({
            "model_id": model["model_id"], "score": model["graph_score"],
            "delta_vs_qwen3_2b": None if baseline_graph_score is None else (model["graph_score"] - baseline_graph_score) * 100,
            "completed_n": model.get("graph_complete_n"),
            "task_scores": model.get("graph_task_scores", {}),
            "peak_gpu_used_gib": model.get("graph_peak_gpu_used_gib"),
            "mean_wall_s": model.get("graph_mean_wall_s"),
            "generated_tokens_per_s": model.get("graph_tokens_s"),
            "context": model.get("configured_context") or model.get("checkpoint_context"),
        })

    recommendations = [
        recommendation("Best overall quality", quality_leader,
                       "Highest mean of AI2D accuracy and the six VQA skill-panel category scores."),
        recommendation("AI2D specialist", ai2d_leader,
                       "Highest AI2D diagram-question accuracy."),
        recommendation("TextVQA specialist", textvqa_leader,
                       "Highest TextVQA answer-consensus score."),
        graph_specialist_rec,
        recommendation("Best Thinking model", thinking_leader,
                       "Highest mean task score among Thinking variants."),
        recommendation("Large context with acceptable quality", long_context_leader,
                       "Highest-scoring model configured for at least 256K that exceeds 75% mean on AI2D and TextVQA and reaches 50% on the VQA skill panel."),
        recommendation("Fastest average generation with acceptable quality", speed_leader,
                       "Fastest mean output-token rate among models scoring above 75% on AI2D and TextVQA combined, and at least 50% on the six-category VQA panel."),
        *edge_picks,
    ]

    data = {
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "preliminary": True,
        "preliminary_note": "GRAB-Lite scoring is complete for the four-model graph panel. Some large-model rows still have incomplete task coverage; partial rows are marked and excluded from complete-score recommendations.",
        "samples_per_task": result["samples_per_task"],
        "seed": result["seed"],
        "memory_cap_gib": result.get("memory_cap_gib"),
        "tasks": list(TASK_ORDER),
        "panel_tasks": list(PANEL_TASKS),
        "panel_samples_per_category": panel.get("samples_per_task", 25),
        "graph_benchmark": {
            "title": "GRAB-Lite graph analysis",
            "url": "https://huggingface.co/datasets/jonathan-roberts1/GRAB-lite",
            "homepage": "https://grab-benchmark.github.io/",
            "samples": 500,
            "task_counts": {"properties": 100, "function": 100, "series": 100, "transform": 100, "real": 100},
            "models": graph_rows,
        },
        "task_info": TASK_INFO,
        "models": models,
        "edge_tables": edge_tables,
        "recommendations": [item for item in recommendations if item],
    }
    DESTINATION.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {DESTINATION} ({DESTINATION.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
