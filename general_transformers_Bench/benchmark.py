#!/usr/bin/env python3
"""Full shared-suite runner for PaliGemma 2 and experimental Florence-2."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QWEN_BENCH = ROOT / "qwenVL_Bench"
sys.path.insert(0, str(QWEN_BENCH))
import benchmark as common  # noqa: E402
import benchmark_runners as shared  # noqa: E402

from model_adapters import FLORENCE_ID, PALIGEMMA_ID, generate, load_florence, load_paligemma

HF_HOME = Path(os.environ.get("HF_HOME", "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"))
HUB = HF_HOME / "hub"
MEMORY_CAP_GIB = 30.0
SEED = 3407
TASK_SIZES = {
    "ai2d": 100,
    "textvqa": 100,
    "vqa_truth": 25,
    "vqa_count": 25,
    "vqa_description": 25,
    "vqa_affordance": 25,
    "vqa_negation": 25,
    "graph_grab_lite": 500,
}
TASKS = tuple(TASK_SIZES)
MODEL_IDS = (PALIGEMMA_ID, FLORENCE_ID)
LABELS = {
    "ai2d": "AI2D",
    "textvqa": "TextVQA / OCR",
    "vqa_truth": "Truth",
    "vqa_count": "Counting",
    "vqa_description": "Description",
    "vqa_affordance": "Affordance",
    "vqa_negation": "Negation",
    "graph_grab_lite": "GRAB-Lite",
}


def resolve_snapshot(cache: Path) -> Path | None:
    ref = cache / "refs" / "main"
    if ref.is_file():
        snapshot = cache / "snapshots" / ref.read_text(encoding="utf-8").strip()
    else:
        snapshots = list((cache / "snapshots").glob("*")) if (cache / "snapshots").is_dir() else []
        snapshot = snapshots[0] if len(snapshots) == 1 else None
    return snapshot if snapshot and (snapshot / "config.json").is_file() else None


def cache_entry(model_id: str) -> dict[str, Any]:
    cache = HUB / f"models--{model_id.replace('/', '--')}"
    snapshot = resolve_snapshot(cache)
    config: dict[str, Any] = {}
    reason = None
    weight_bytes = 0
    if snapshot is None:
        reason = "cached snapshot or config.json is missing"
    else:
        try:
            config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            reason = f"invalid config.json: {exc}"
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
            except (OSError, ValueError, TypeError) as exc:
                shards = set()
                reason = f"invalid safetensors index: {exc}"
        else:
            safetensors = {p.name for p in snapshot.glob("*.safetensors")}
            shards = safetensors or {p.name for p in snapshot.glob("pytorch_model*.bin")}
        missing = sorted(name for name in shards if not (snapshot / name).is_file())
        if not shards:
            reason = reason or "model weight files are missing"
        elif missing:
            reason = f"incomplete checkpoint; missing {len(missing)} weight shard(s)"
        else:
            weight_bytes = sum((snapshot / name).stat().st_size for name in shards)
        if not config.get("architectures"):
            reason = reason or "model architecture is absent"
    return {
        "id": model_id,
        "snapshot": str(snapshot) if snapshot else None,
        "config": config,
        "weight_bytes": weight_bytes,
        "cache_status": "ready" if reason is None else "incomplete",
        "cache_error": reason,
        "backend": "Transformers (local cached checkpoint)",
    }


def generation(model_id: str, model, processor, row: dict[str, Any], task: str, prompt: str,
               *, route_validated: bool = False):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        record = generate(model_id, model, processor, row["image"], task, prompt)
    if model_id == FLORENCE_ID and route_validated:
        record["scoring_eligible"] = True
        record["route_note"] = "experimental <VQA> route passed AI2D and TextVQA real-image preflight"
    torch.cuda.synchronize()
    record["generation_s"] = round(time.perf_counter() - started, 4)
    record["peak_allocated_gib"] = round(
        torch.cuda.max_memory_allocated() / (1024**3), 3
    )
    return record


def preflight(model_id: str, model, processor, seed: int) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    """Prove real-image handling and both answer formats before any full task runs."""
    checks: dict[str, Any] = {}
    cached: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in TASKS}
    for task in ("ai2d", "textvqa"):
        dataset = shared.dataset_for(task)
        indices = common.pick_indices(len(dataset), TASK_SIZES[task], seed)
        index = indices[0]
        row = dataset[index]
        prompt, expected = shared.prompt_for(task, row)
        output = generation(model_id, model, processor, row, task, prompt)
        prediction, score, parse_error = shared.score_for(task, output["answer"], expected)
        nonempty = bool(output["answer"].strip())
        format_ok = not parse_error if task == "ai2d" else nonempty
        checks[task] = {
            "index": index,
            "passed": nonempty and format_ok,
            "nonempty_answer": nonempty,
            "answer_format_ok": format_ok,
            "prediction": prediction,
            "score": score,
            "parse_error": parse_error,
            "generation_s": output["generation_s"],
            "output_tokens": output["output_tokens"],
            "answer": output["answer"],
            "prompt": output["prompt"],
            "prompt_prefix": output["prompt_prefix"],
            "postprocess": output["postprocess"],
            "route_note": output["route_note"],
            "scoring_eligible_before_preflight": output["scoring_eligible"],
        }
        cached[task][index] = {
            "generation": output,
            "prediction": prediction,
            "score": score,
            "parse_error": parse_error,
        }
        if not checks[task]["passed"]:
            break
    passed = len(checks) == 2 and all(check["passed"] for check in checks.values())
    return {"passed": passed, "required": ["ai2d", "textvqa"], "checks": checks}, cached


def run_task(model_id: str, model, processor, task: str, count: int, seed: int,
             cached: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import torch

    dataset = shared.dataset_for(task)
    indices = common.pick_indices(len(dataset), count, seed)
    items: list[dict[str, Any]] = []
    scores: list[float] = []
    total_tokens = 0
    total_generation_s = 0.0
    for position, index in enumerate(indices):
        row = dataset[index]
        prompt, expected = shared.prompt_for(task, row)
        try:
            reused = cached.get(index)
            output = reused["generation"] if reused else generation(
                model_id, model, processor, row, task, prompt, route_validated=True
            )
            if not output["scoring_eligible"]:
                raise RuntimeError("model prompt route has not passed the required preflight")
            prediction, score, parse_error = shared.score_for(task, output["answer"], expected)
            item = {
                "index": index,
                "score": score,
                "prediction": prediction,
                "output": output["answer"],
                "prompt": output["prompt"],
                "prompt_prefix": output["prompt_prefix"],
                "scoring_route_validated": True,
                "scoring_eligible": output["scoring_eligible"],
                "input_tokens": output["input_tokens"],
                "output_tokens": output["output_tokens"],
                "parse_error": parse_error,
                "generation_s": output["generation_s"],
                "peak_allocated_gib": output["peak_allocated_gib"],
            }
            if output["postprocess"] is not None:
                item["postprocess"] = output["postprocess"]
            items.append(item)
            scores.append(score)
            total_tokens += output["output_tokens"]
            total_generation_s += output["generation_s"]
        except Exception as exc:
            items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        finally:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(f"PROGRESS {model_id} {task} {position + 1}/{len(indices)}", flush=True)
    successful = [item for item in items if "score" in item]
    parse_errors = sum(bool(item.get("parse_error")) for item in successful)
    group_scores = None
    if task == "graph_grab_lite":
        groups: dict[str, list[float]] = {}
        for item in successful:
            row = dataset[item["index"]]
            groups.setdefault(row["task"], []).append(item["score"])
            item.update(pid=row["pid"], task_type=row["task"],
                        dataset_category=row["category"], complexity=row["complexity"])
        group_scores = {name: statistics.mean(values) for name, values in sorted(groups.items())}
    return {
        "requested_n": len(indices),
        "completed_n": len(successful),
        "failed_n": len(indices) - len(successful),
        "score": statistics.mean(scores) if scores else None,
        "parse_errors": parse_errors,
        "group_scores": group_scores,
        "mean_generation_s": statistics.mean(i["generation_s"] for i in successful) if successful else None,
        "generated_tokens_per_s": total_tokens / total_generation_s if total_generation_s else None,
        "peak_allocated_gib": max((i["peak_allocated_gib"] for i in successful), default=None),
        "items": items,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    model = result["model"]
    task_rows = []
    for task in TASKS:
        data = model.get("tasks", {}).get(task, {})
        score = "—" if data.get("score") is None else f"{data['score']:.1%}"
        task_rows.append(
            f"| {LABELS[task]} | {data.get('completed_n', 0)}/{data.get('requested_n', TASK_SIZES[task])} | {score} |"
        )
    complete = all(
        model.get("tasks", {}).get(task, {}).get("completed_n") == TASK_SIZES[task]
        for task in TASKS
    )
    macro = statistics.mean(model["tasks"][task]["score"] for task in TASKS) if complete else None
    model["full_benchmark_complete"] = complete
    model["full_benchmark_macro"] = macro
    lines = [
        f"# {model['model_id']} shared-suite run", "",
        f"Generated: {result['generated_at']}",
        f"Run status: {model['status']}",
        f"Backend: {model['backend']}; Transformers {model['runtime_versions']['transformers']}; "
        f"PyTorch {model['runtime_versions']['torch']}.",
        f"Seed: {result['seed']}; memory ceiling: {result['memory_cap_gib']:.1f} GiB.",
        "All eight shared tasks run only after AI2D format and TextVQA real-image preflights pass.",
        "The preflight AI2D and TextVQA requests are reused in the scored task results.",
        "Florence's `<VQA>` route is experimental because the cached processor does not register that task;"
        " it is promoted for scoring only after both real-image checks pass.",
        "Full-suite macro is the unweighted mean of all eight task scores and is withheld unless every requested example completes.",
        "", "## Preflight", "",
        "| Task | Passed | Answer format | Parse error | Output tokens |",
        "|---|---:|---|---:|---:|",
    ]
    for task, check in result["preflight"]["checks"].items():
        lines.append(
            f"| {LABELS[task]} | {'yes' if check['passed'] else 'no'} | "
            f"{'valid' if check['answer_format_ok'] else 'invalid'} | "
            f"{'yes' if check['parse_error'] else 'no'} | {check['output_tokens']} |"
        )
    lines += ["", "## Scores", "", "| Task | Completed / requested | Score |", "|---|---:|---:|"]
    lines.extend(task_rows)
    lines += ["", f"Full-suite macro: {'—' if macro is None else f'{macro:.1%}'}.",
              "", "## Runtime and resources", "",
              "Generation time covers model generation. Peak allocated VRAM is process-local and excludes unrelated device use.",
              "", "| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Parse errors |",
              "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        data = model.get("tasks", {}).get(task, {})
        fmt = lambda value: "—" if value is None else f"{value:.2f}"
        lines.append(
            f"| {LABELS[task]} | {fmt(data.get('mean_generation_s'))} | "
            f"{fmt(data.get('generated_tokens_per_s'))} | {fmt(data.get('peak_allocated_gib'))} | "
            f"{data.get('parse_errors', 0)} |"
        )
    lines += ["", "## Method and provenance", "",
              "Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.",
              "See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.",
              f"Cache snapshot: `{model.get('snapshot') or 'missing'}`.",
              f"Cache status: {model.get('cache_status')}; checkpoint weight files: {model.get('weight_gib', 0):.2f} GiB.",
              "Raw per-item outputs and errors are in `results.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model_id = args.model
    output_dir = args.output_dir or HERE / "results" / model_id.split("/")[-1].lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    transformers_version = version("transformers")
    entry = cache_entry(model_id)
    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": SEED,
        "memory_cap_gib": MEMORY_CAP_GIB,
        "task_sizes": TASK_SIZES,
        "task_names": list(TASKS),
        "model": {
            "model_id": model_id,
            "backend": entry["backend"],
            "snapshot": entry["snapshot"],
            "cache_status": entry["cache_status"],
            "cache_error": entry["cache_error"],
            "weight_bytes": entry["weight_bytes"],
            "weight_gib": entry["weight_bytes"] / (1024**3),
            "runtime_versions": {
                "python": sys.version.split()[0],
                "transformers": transformers_version,
                "torch": version("torch"),
            },
            "prompt_route": "PaliGemma answer en" if model_id == PALIGEMMA_ID else "experimental Florence <VQA>",
            "tasks": {},
            "status": "not_started",
        },
        "preflight": {"passed": False, "required": ["ai2d", "textvqa"], "checks": {}},
    }
    if entry["cache_status"] != "ready":
        result["model"].update(status="failed", error=entry["cache_error"])
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    if entry["weight_bytes"] >= MEMORY_CAP_GIB * 1024**3:
        result["model"].update(status="failed", error="cached weights exceed the configured memory ceiling")
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2

    import torch
    if not torch.cuda.is_available():
        result["model"].update(status="failed", error="CUDA is unavailable; inference requires the GPU")
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if total_memory < MEMORY_CAP_GIB * 1024**3:
        result["model"].update(status="failed", error="the configured 30 GiB ceiling exceeds installed GPU memory")
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 1024**3 / total_memory, device)

    model = processor = None
    try:
        max_memory = {device: int(MEMORY_CAP_GIB * 1024**3)}
        print(f"LOAD {model_id} snapshot={entry['snapshot']}", flush=True)
        if model_id == PALIGEMMA_ID:
            model, processor = load_paligemma(entry["snapshot"], max_memory=max_memory)
        else:
            model, processor = load_florence(entry["snapshot"], max_memory=max_memory)
        result["model"]["checkpoint_context"] = getattr(
            getattr(model.config, "text_config", model.config), "max_position_embeddings", None
        )
        if model_id == PALIGEMMA_ID:
            result["model"]["image_resolution_px"] = 224
        preflight_result, cached = preflight(model_id, model, processor, SEED)
        result["preflight"] = preflight_result
        if not preflight_result["passed"]:
            result["model"].update(
                status="preflight_failed",
                error="real-image AI2D and TextVQA checks did not both pass; full suite not started",
            )
        else:
            if model_id == FLORENCE_ID:
                for task_outputs in cached.values():
                    for cached_item in task_outputs.values():
                        cached_item["generation"]["scoring_eligible"] = True
                        cached_item["generation"]["route_note"] = (
                            "experimental <VQA> route passed AI2D and TextVQA real-image preflight"
                        )
                for check in result["preflight"]["checks"].values():
                    check["scoring_eligible_for_full_run"] = True
            result["model"]["scoring_route"] = {
                "status": "empirically_validated",
                "evidence": "nonempty TextVQA response and valid AI2D answer format on real shared examples",
                "experimental_processor_task_fallback": model_id == FLORENCE_ID,
            }
            result["model"]["status"] = "in_progress"
            for task in TASKS:
                print(f"RUN {model_id} {task} n={TASK_SIZES[task]}", flush=True)
                try:
                    result["model"]["tasks"][task] = run_task(
                        model_id, model, processor, task, TASK_SIZES[task], SEED, cached[task]
                    )
                except Exception as exc:
                    result["model"]["tasks"][task] = {
                        "requested_n": TASK_SIZES[task], "completed_n": 0,
                        "failed_n": TASK_SIZES[task], "score": None,
                        "error": f"{type(exc).__name__}: {exc}", "items": [],
                    }
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["model"]["status"] = "complete" if all(
                result["model"]["tasks"].get(task, {}).get("completed_n") == TASK_SIZES[task]
                for task in TASKS
            ) else "partial"
    except Exception as exc:
        result["model"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", result)
    print(f"Wrote {result_path} and {output_dir / 'report.md'}", flush=True)
    return 0 if result["model"]["status"] in {"complete", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
