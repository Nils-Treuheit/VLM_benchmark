#!/usr/bin/env python3
"""Offline, serial Transformers benchmark for cached Phi vision checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from importlib.metadata import version

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QWEN_BENCH = ROOT / "qwenVL_Bench"
sys.path.insert(0, str(QWEN_BENCH))
import benchmark as common  # noqa: E402
import benchmark_runners as shared  # noqa: E402

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
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
MODEL_IDS = (
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/Phi-3-vision-128k-instruct",
    "microsoft/Phi-4-multimodal-instruct",
)


def cache_entry(model_id: str) -> dict[str, Any]:
    cache = HUB / f"models--{model_id.replace('/', '--')}"
    ref = cache / "refs" / "main"
    snapshots = cache / "snapshots"
    snapshot = snapshots / ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    reason = None
    config: dict[str, Any] = {}
    weight_bytes = 0
    if snapshot is None or not snapshot.is_dir():
        reason = "cached snapshot is missing"
    else:
        config_path = snapshot / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                reason = f"invalid config.json: {exc}"
        else:
            reason = "config.json is missing"
        index = snapshot / "model.safetensors.index.json"
        weights: set[str] = set()
        if index.is_file():
            try:
                weights.update(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
            except (OSError, ValueError, TypeError) as exc:
                reason = f"invalid safetensors index: {exc}"
        else:
            weights.update(path.name for path in snapshot.glob("*.safetensors"))
            weights.update(path.name for path in snapshot.glob("pytorch_model*.bin"))
        if not weights:
            reason = reason or "model weight files are missing"
        missing = [name for name in sorted(weights) if not (snapshot / name).is_file()]
        if missing:
            reason = f"incomplete checkpoint; missing {len(missing)} weight shard(s)"
        else:
            weight_bytes = sum((snapshot / name).stat().st_size for name in weights)
        if not config.get("architectures"):
            reason = reason or "model architecture is missing from config.json"
    return {
        "id": model_id,
        "backend": "Transformers (cached checkpoint)",
        "snapshot": str(snapshot) if snapshot else None,
        "config": config,
        "config_identity": {key: config.get(key) for key in (
            "architectures", "model_type", "max_position_embeddings", "torch_dtype", "auto_map"
        )},
        "weight_bytes": weight_bytes,
        "cache_status": "ready" if reason is None else "incomplete",
        "cache_error": reason,
    }


def list_models() -> list[dict[str, Any]]:
    return [cache_entry(model_id) for model_id in MODEL_IDS]


def model_size_gib(entry: dict[str, Any]) -> float:
    return entry["weight_bytes"] / (1024**3)


def load_model(entry: dict[str, Any]):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; benchmark inference requires the GPU")
    device = torch.cuda.current_device()
    snapshot = entry["snapshot"]
    config = entry["config"]
    text_config = config.get("text_config") or {}
    dtype_name = (text_config.get("torch_dtype") or text_config.get("dtype")
                  or config.get("torch_dtype") or "float16")
    dtype_name = str(dtype_name).removeprefix("torch.")
    model_dtype = getattr(torch, dtype_name, None)
    if model_dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(f"unsupported checkpoint dtype {dtype_name!r}")
    is_phi3 = config.get("model_type") == "phi3_v"
    trust_remote_code = is_phi3 or config.get("model_type") == "phi4mm"
    model_config = AutoConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=trust_remote_code
    )
    if is_phi3:
        # Phi-3 remote code expects DynamicCache aliases removed in newer
        # Transformers. Keep this compatibility patch local to the Phi adapter.
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length(a[1] if len(a) > 1 else 0)
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: self.get_max_cache_shape()
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length(0))
        model_config._attn_implementation = "eager"
    elif config.get("model_type") == "phi4mm":
        # Phi-4's cached remote code wraps its inner Phi4MMModel with PEFT as
        # CAUSAL_LM, although that inner class lacks the generation helper
        # PEFT expects. The outer Phi4MMForCausalLM owns the generation loop;
        # this forwards the standard Transformers preparation helper locally.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.generation import GenerationMixin
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length(a[1] if len(a) > 1 else 0)
        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: self.get_max_cache_shape()
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length(0))
        inner_model_class = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMModel", snapshot, local_files_only=True
        )
        if not hasattr(inner_model_class, "prepare_inputs_for_generation"):
            inner_model_class.prepare_inputs_for_generation = (
                GenerationMixin.prepare_inputs_for_generation
            )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        config=model_config,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        device_map={"": device},
        dtype=model_dtype,
        attn_implementation="eager" if is_phi3 else "sdpa",
        max_memory={device: int(MEMORY_CAP_GIB * 1024**3)},
    )
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=trust_remote_code
    )
    model.eval()
    return model, processor


def make_messages(prompt: str):
    return [{"role": "user", "content": f"<|image_1|>{prompt}"}]


def generate(model, processor, image, prompt: str, model_id: str):
    import torch

    model_type = getattr(model.config, "model_type", None)
    if model_type == "phi4mm":
        rendered = processor.tokenizer.apply_chat_template(
            make_messages(prompt), tokenize=False, add_generation_prompt=True
        )
        if rendered.endswith("<|endoftext|>"):
            rendered = rendered.removesuffix("<|endoftext|>")
    elif model_type == "phi3_v":
        rendered = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
    else:
        raise RuntimeError(f"unsupported Phi vision architecture {model_type!r}")
    if model_type == "phi3_v":
        inputs = processor(
            text=rendered, images=image.convert("RGB"), return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            text=rendered, images=[image.convert("RGB")], return_tensors="pt"
        ).to(model.device)
    max_context = getattr(getattr(model.config, "text_config", model.config),
                          "max_position_embeddings", None)
    input_tokens = int(inputs.input_ids.shape[-1])
    if max_context and input_tokens + 256 > max_context:
        raise RuntimeError(
            f"request needs {input_tokens + 256} tokens including output; "
            f"checkpoint context is {max_context}"
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generation_options = {"max_new_tokens": 256, "do_sample": False}
        if model_type == "phi4mm":
            # The cached Phi-4 forward requires an integer here; Transformers
            # 4.57 may pass None unless the value is explicit.
            generation_options["num_logits_to_keep"] = 0
        output_ids = model.generate(**inputs, **generation_options)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output_ids[:, input_tokens:]
    text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text, input_tokens, int(generated.shape[-1]), elapsed


def summarize_task(dataset, task: str, indices: list[int], items: dict[int, dict[str, Any]],
                   failures: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ordered = [items[index] if index in items else failures[index]
               for index in indices if index in items or index in failures]
    successful = [item for item in ordered if "score" in item]
    group_scores = None
    if task == "graph_grab_lite":
        groups: dict[str, list[float]] = {}
        for item in successful:
            row = dataset[item["index"]]
            groups.setdefault(row["task"], []).append(item["score"])
            item.update(pid=row["pid"], task_type=row["task"],
                        dataset_category=row["category"], complexity=row["complexity"])
        group_scores = {name: statistics.mean(values) for name, values in sorted(groups.items())}
    total_tokens = sum(item.get("output_tokens", 0) for item in successful)
    total_generation_s = sum(item.get("generation_s", 0) for item in successful)
    return {
        "requested_n": len(indices),
        "completed_n": len(successful),
        "failed_n": len([item for item in ordered if "error" in item]),
        "score": statistics.mean(item["score"] for item in successful) if successful else None,
        "group_scores": group_scores,
        "mean_generation_s": statistics.mean(item["generation_s"] for item in successful) if successful else None,
        "generated_tokens_per_s": total_tokens / total_generation_s if total_generation_s else None,
        "peak_allocated_gib": max((item["peak_allocated_gib"] for item in successful), default=None),
        "items": ordered,
    }


def run_task(model, processor, model_id: str, task: str, count: int, seed: int,
             prior_task: dict[str, Any] | None = None,
             on_progress: Any | None = None) -> dict[str, Any]:
    import torch

    dataset = shared.dataset_for(task)
    indices = common.pick_indices(len(dataset), count, seed)
    valid_indices = set(indices)
    items = {
        item["index"]: item for item in (prior_task or {}).get("items", [])
        if item.get("index") in valid_indices and "score" in item
    }
    failures: dict[int, dict[str, Any]] = {}
    # The first measured request follows an unmeasured real-image warmup.
    # Clear its CUDA peak so the first item records only its own allocations.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for position, index in enumerate(indices):
        if index in items:
            continue
        row = dataset[index]
        prompt, expected = shared.prompt_for(task, row)
        try:
            output, input_tokens, output_tokens, generation_s = generate(
                model, processor, row["image"], prompt, model_id
            )
            prediction, score, parse_error = shared.score_for(task, output, expected)
            items[index] = {
                "index": index,
                "score": score,
                "prediction": prediction,
                "output": output,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "parse_error": parse_error,
                "generation_s": round(generation_s, 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
        except Exception as exc:
            failures[index] = {"index": index, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        finally:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(f"PROGRESS {model_id} {task} {position + 1}/{len(indices)}", flush=True)
            if on_progress:
                on_progress(summarize_task(dataset, task, indices, items, failures))
    return summarize_task(dataset, task, indices, items, failures)


def write_report(path: Path, result: dict[str, Any]) -> None:
    models = result["models"]
    lines = [
        "# Phi vision family benchmark", "",
        f"Generated: {result['generated_at']}",
        f"Seed: {result['seed']}; GPU memory ceiling: {result['memory_cap_gib']:.1f} GiB.",
        "Runtime: " + "; ".join(
            f"{label} {result.get('runtime_versions', {}).get(key, 'unrecorded')}"
            for key, label in (("python", "Python"), ("torch", "PyTorch"),
                               ("cuda", "CUDA"), ("transformers", "Transformers"))
        ) + ".",
        "Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.",
        "The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.",
        "",
        "## Tasks and scoring", "",
        "| Task | Examples | Dataset and split | Score |",
        "|---|---:|---|---|",
        "| AI2D | 100 | AI2D test | Exact A/B/C/D accuracy |",
        "| TextVQA / OCR | 100 | TextVQA validation | Normalized human-answer consensus |",
        "| Truth, counting, description | 25 each | VQAv2 validation | Normalized consensus per category |",
        "| Affordance | 25 | PercepTaxBench real/test | Normalized answer match |",
        "| Negation | 25 | NegVQA dev | Exact A/B choice accuracy |",
        "| GRAB-Lite | 500 | GRAB-Lite test | Exact match overall and across five groups |",
        "",
        "The same fixed examples, image resize policy, one-image prompt wording, and output constraints were used for every model. Requests ran serially without batching. The macro is a local comparison score, not a benchmark leaderboard score.",
        "", "## Scores", "",
        "| Model | Status | AI2D | TextVQA / OCR | Truth | Counting | Description | Affordance | Negation | GRAB-Lite | Complete tasks | Full-suite macro |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = ("ai2d", "textvqa", "vqa_truth", "vqa_count", "vqa_description",
              "vqa_affordance", "vqa_negation", "graph_grab_lite")
    for model in models:
        tasks = model.get("tasks", {})
        cells = ["—" if tasks.get(task, {}).get("score") is None
                 else f"{tasks[task]['score']:.1%}" for task in labels]
        complete = sum(tasks.get(task, {}).get("completed_n", 0)
                       == result["task_sizes"].get(task, TASK_SIZES[task])
                       for task in labels)
        full_complete = all(
            task in tasks and tasks[task].get("completed_n") == TASK_SIZES[task]
            and tasks[task].get("requested_n") == TASK_SIZES[task]
            for task in TASKS
        )
        full_score = statistics.mean(tasks[task]["score"] for task in TASKS) if full_complete else None
        model["full_benchmark_complete"] = full_complete
        model["full_benchmark_macro"] = full_score
        lines.append(
            f"| {model['model_id']} | {model.get('status', 'unknown')} | "
            + " | ".join(cells) + f" | {complete}/8 | "
            + ("—" if full_score is None else f"{full_score:.1%}") + " |"
        )
    lines += ["", "## Runtime details", "",
              "Times cover generation after preprocessing. Peak allocated VRAM is process-local.",
              "Failed requests and parse failures remain in the per-item JSON and score as incorrect only when a response was produced.",
              "", "| Model | Task | Completed / requested | Mean generation seconds | Output tokens/second | Peak allocated GiB |",
              "|---|---|---:|---:|---:|---:|"]
    for model in models:
        for task in result["task_names"]:
            data = model.get("tasks", {}).get(task, {})
            lines.append(
                f"| {model['model_id']} | {task} | {data.get('completed_n', 0)}/{data.get('requested_n', 0)} | "
                f"{fmt(data.get('mean_generation_s'))} | {fmt(data.get('generated_tokens_per_s'))} | "
                f"{fmt(data.get('peak_allocated_gib'))} |"
            )
    group_names = sorted({
        name for model in models
        for name in (model.get("tasks", {}).get("graph_grab_lite", {}).get("group_scores") or {})
    })
    if group_names:
        lines += ["", "## GRAB-Lite group scores", "",
                  "| Model | " + " | ".join(group_names) + " |",
                  "|---|" + "---:|" * len(group_names)]
        for model in models:
            scores = model.get("tasks", {}).get("graph_grab_lite", {}).get("group_scores") or {}
            lines.append("| " + model["model_id"] + " | " + " | ".join(
                f"{scores[name]:.1%}" if name in scores else "—" for name in group_names
            ) + " |")
    lines += ["", "## Method", "",
              "Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.",
              "Each request contains one still image and one shared task prompt. Model-specific local processors apply Phi image preprocessing and the documented Phi image-token format.",
              "Requests run serially with greedy decoding and at most one model loaded at a time.",
              "Incomplete checkpoints are retained with their cache errors; over-budget checkpoints are skipped before loading.",
              "Both runnable models used Transformers 4.57.1. Resume requires the same Transformers version for comparable outputs.",
              "Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.",
              "", "Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.", ""]
    lines += ["", "## Checkpoint identity and compatibility", "",
              "| Model | Snapshot | Cached weights (GiB) | Configuration identity | Backend | Remote code | Attention | Compatibility patches | Cache and comparison status |",
              "|---|---|---:|---|---|---|---|---|---|"]
    for model in models:
        config = model.get("config_identity") or {}
        comparison = model.get("comparison") or {}
        detail = model.get("error") or comparison.get("status") or model.get("status", "unknown")
        patches = ", ".join(model.get("compatibility_patches") or []) or "none"
        lines.append(
            f"| {model['model_id']} | `{model.get('snapshot') or '—'}` | "
            f"{fmt((model.get('weight_bytes') or 0) / (1024 ** 3))} | "
            f"`{config.get('model_type') or '—'}`; context `{config.get('max_position_embeddings') or '—'}` | "
            f"{model.get('backend') or '—'} | {model.get('trust_remote_code', '—')} | "
            f"{model.get('attention_implementation') or '—'} | {patches} | {detail} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--models", nargs="*", choices=MODEL_IDS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--samples", type=int, help="override the shared per-task sample counts")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "phi_family")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    transformers_version = version("transformers")

    entries = list_models()
    if args.models:
        selected = set(args.models)
        entries = [entry for entry in entries if entry["id"] in selected]
    if args.list_models:
        for entry in entries:
            size = f"{model_size_gib(entry):.2f} GiB weights" if entry["weight_bytes"] else "size unknown"
            status = entry["cache_status"]
            if status == "ready" and entry["weight_bytes"] >= MEMORY_CAP_GIB * 1024**3:
                status = "over 30 GiB budget"
            print(f"{entry['id']}\t{status}\t{size}\t{entry.get('snapshot') or '—'}")
        return 0

    task_names = tuple(args.tasks or TASKS)
    task_sizes = {task: args.samples if args.samples is not None else TASK_SIZES[task]
                  for task in task_names}
    if any(size < 1 for size in task_sizes.values()):
        parser.error("sample counts must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        saved_sizes = result.get("task_sizes") or {}
        if result.get("seed") != args.seed or any(
            task not in result.get("task_names", []) or saved_sizes.get(task) != task_sizes[task]
            for task in task_names
        ):
            parser.error("--resume requires matching seed and sample counts for the selected tasks")
        previous_transformers = result.get("runtime_versions", {}).get("transformers")
        if previous_transformers != transformers_version:
            parser.error(
                "--resume requires matching Transformers versions; "
                f"results use {previous_transformers or 'an unrecorded version'}, "
                f"current runtime uses {transformers_version}"
            )
    else:
        result = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "seed": args.seed,
            "memory_cap_gib": MEMORY_CAP_GIB,
            "runtime_versions": {"transformers": transformers_version},
            "task_names": list(task_names),
            "task_sizes": task_sizes,
            "models": [],
        }

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run inference in host context on a supported GPU")
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if total_memory < MEMORY_CAP_GIB * 1024**3:
        raise SystemExit("the configured 30 GiB ceiling exceeds installed GPU memory")
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 1024**3 / total_memory, device)

    for entry in entries:
        old = next((m for m in result["models"] if m.get("model_id") == entry["id"]), None)
        if args.resume and old and all(
            old.get("tasks", {}).get(task, {}).get("requested_n") == task_sizes[task]
            and old.get("tasks", {}).get(task, {}).get("completed_n") == task_sizes[task]
            and old.get("tasks", {}).get(task, {}).get("score") is not None
            for task in task_names
        ):
            print(f"RESUME SKIP {entry['id']} selected tasks already complete", flush=True)
            continue
        if old and entry["cache_status"] == "ready" and entry["weight_bytes"] < MEMORY_CAP_GIB * 1024**3:
            record = old
            record.update(
                backend=entry["backend"], snapshot=entry["snapshot"],
                weight_bytes=entry["weight_bytes"], config_identity=entry.get("config_identity"),
                comparison=entry.get("comparison"),
                runtime_versions={"transformers": transformers_version}, status="in_progress",
            )
            record.pop("error", None)
        else:
            if old:
                result["models"].remove(old)
            record = {"model_id": entry["id"], "backend": entry["backend"],
                      "snapshot": entry["snapshot"], "weight_bytes": entry["weight_bytes"],
                      "config_identity": entry.get("config_identity"),
                      "comparison": entry.get("comparison"),
                      "runtime_versions": {"transformers": transformers_version},
                      "prompt_template": (
                          "Phi-3 image-token format with cached processor" if entry["config"].get("model_type") == "phi3_v"
                          else "Phi-4 still-image mode with cached processor chat template"
                      ),
                      "trust_remote_code": entry["config"].get("model_type") in {"phi3_v", "phi4mm"},
                      "attention_implementation": "eager" if entry["config"].get("model_type") == "phi3_v" else "sdpa",
                      "tasks": {}, "status": "in_progress"}
            result["models"].append(record)
        if entry["cache_status"] != "ready":
            record.update(status="skipped", error=entry["cache_error"])
        elif entry["weight_bytes"] >= MEMORY_CAP_GIB * 1024**3:
            record.update(status="skipped", error=(
                f"cached weights require {model_size_gib(entry):.2f} GiB before runtime memory; "
                f"exceeds the {MEMORY_CAP_GIB:.1f} GiB ceiling"
            ))
        else:
            model = processor = None
            try:
                print(f"LOAD {entry['id']}", flush=True)
                model, processor = load_model(entry)
                record["checkpoint_context"] = getattr(
                    getattr(model.config, "text_config", model.config),
                    "max_position_embeddings", None
                )
                # Real-image warmup exercises the exact single-image template
                # and loader path, but does not contribute to reported metrics.
                warm_data = shared.dataset_for("ai2d")
                warm = warm_data[common.pick_indices(len(warm_data), 1, args.seed)[0]]
                prompt, _ = shared.prompt_for("ai2d", warm)
                generate(model, processor, warm["image"], prompt, entry["id"])
                record["warmup"] = "one AI2D image request excluded from scores"
                for task in task_names:
                    existing_task = record.get("tasks", {}).get(task, {})
                    if (args.resume
                            and existing_task.get("requested_n") == task_sizes[task]
                            and existing_task.get("completed_n") == task_sizes[task]
                            and existing_task.get("score") is not None):
                        print(f"RESUME SKIP {entry['id']} {task} already complete", flush=True)
                        continue
                    print(f"RUN {entry['id']} {task} n={task_sizes[task]}", flush=True)
                    try:
                        def checkpoint_task(progress: dict[str, Any]) -> None:
                            record["tasks"][task] = progress
                            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

                        record["tasks"][task] = run_task(
                            model, processor, entry["id"], task, task_sizes[task], args.seed,
                            prior_task=existing_task if args.resume else None,
                            on_progress=checkpoint_task,
                        )
                    except Exception as exc:
                        record["tasks"][task] = {
                            "requested_n": task_sizes[task], "completed_n": 0,
                            "failed_n": task_sizes[task], "score": None,
                            "error": f"{type(exc).__name__}: {exc}", "items": [],
                        }
                        print(f"TASK ERROR {task}: {type(exc).__name__}: {exc}", flush=True)
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                saved_task_names = result.get("task_names", task_names)
                saved_task_sizes = result.get("task_sizes", task_sizes)
                record["status"] = "complete" if all(
                    record["tasks"].get(task, {}).get("requested_n") == saved_task_sizes[task]
                    and record["tasks"].get(task, {}).get("completed_n") == saved_task_sizes[task]
                    and record["tasks"].get(task, {}).get("score") is not None
                    for task in saved_task_names
                ) else "partial"
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
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
    write_report(output_dir / "report.md", result)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {result_path} and {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
