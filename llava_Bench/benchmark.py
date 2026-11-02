#!/usr/bin/env python3
"""Offline, serial Transformers benchmark for cached LLaVA family checkpoints."""

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
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "xtuner/llava-phi-3-mini-hf",
    "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
    "llava-hf/LLaVA-NeXT-Video-34B-DPO-hf",
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
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

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
    needs_remote_model_code = entry["id"] == "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=needs_remote_model_code,
        device_map={"": device},
        # Some LLaVA configs declare different root and text dtypes. Casting
        # every submodule to the text dtype avoids mixed-dtype generation paths.
        dtype=model_dtype,
        attn_implementation="sdpa",
        max_memory={device: int(MEMORY_CAP_GIB * 1024**3)},
    )
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=needs_remote_model_code
    )
    if needs_remote_model_code:
        # This checkpoint's repository omits tokenizer vocabulary files. Its
        # bundled processor consequently loads a tokenizer containing only
        # added special tokens, which silently drops every prompt word. Reuse
        # a cached Qwen3 tokenizer with the checkpoint's declared vocabulary
        # and special-token IDs; keep the model-provided chat template below.
        tokenizer_cache = HUB / "models--unsloth--qwen3-4b-instruct-2507-unsloth-bnb-4bit"
        tokenizer_ref = tokenizer_cache / "refs" / "main"
        if not tokenizer_ref.is_file():
            raise RuntimeError(
                "OneVision-2 requires a cached Qwen3 tokenizer; "
                f"missing {tokenizer_ref}"
            )
        tokenizer_snapshot = tokenizer_cache / "snapshots" / tokenizer_ref.read_text(
            encoding="utf-8"
        ).strip()
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_snapshot, local_files_only=True
        )
        expected_vocab_size = model.config.text_config.vocab_size
        if tokenizer.vocab_size > expected_vocab_size:
            raise RuntimeError(
                f"cached Qwen3 tokenizer vocabulary {tokenizer.vocab_size} exceeds "
                f"OneVision-2 vocabulary {expected_vocab_size}"
            )
        expected_special_tokens = {
            "<|im_start|>": 151644,
            "<|im_end|>": model.config.eos_token_id,
            "<|vision_start|>": model.config.vision_start_token_id,
            "<|vision_end|>": model.config.vision_end_token_id,
            "<|image_pad|>": model.config.image_token_id,
            "<|video_pad|>": model.config.video_token_id,
        }
        for token, expected_id in expected_special_tokens.items():
            actual_id = tokenizer.convert_tokens_to_ids(token)
            if actual_id != expected_id:
                raise RuntimeError(
                    f"cached Qwen3 tokenizer maps {token} to {actual_id}, "
                    f"but OneVision-2 expects {expected_id}"
                )
        processor.tokenizer = tokenizer
    if entry["id"] == "xtuner/llava-phi-3-mini-hf":
        # This older checkpoint omits fields that recent LlavaProcessor uses
        # to expand the single <image> marker into patch tokens.
        vision_config = model.config.vision_config
        processor.patch_size = vision_config.patch_size
        processor.num_additional_image_tokens = 1
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    model.eval()
    return model, processor


def make_messages(prompt: str):
    # A single still image is inserted through each checkpoint's own processor
    # template. No video frames or timestamps are supplied to NeXT Video.
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]


def generate(model, processor, image, prompt: str, model_id: str):
    import torch

    messages = make_messages(prompt)
    if getattr(processor, "chat_template", None):
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif model_id == "xtuner/llava-phi-3-mini-hf":
        # XTuner's cached Phi-3 checkpoint predates Transformers chat-template
        # metadata and documents this native LLaVA-Phi message wrapper.
        rendered = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
    else:
        raise RuntimeError("cached processor does not provide apply_chat_template")
    inputs = processor(
        text=[rendered], images=[image.convert("RGB")], padding=True,
        return_tensors="pt",
    ).to(model.device)
    if model_id == "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct":
        grid = getattr(inputs, "image_grid_thw", None)
        image_token_id = int(model.config.image_token_id)
        processor_image_token_id = int(
            processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        )
        if grid is None:
            raise RuntimeError("OneVision-2 processor did not return image_grid_thw")
        merge_size = int(getattr(processor, "spatial_merge_size", 2))
        expected_image_tokens = int(
            (grid.prod(dim=-1).sum() // (merge_size * merge_size)).item()
        )
        processor_marker_count = int(
            (inputs.input_ids == processor_image_token_id).sum().item()
        )
        if processor_marker_count != expected_image_tokens:
            raise RuntimeError(
                "OneVision-2 processor image-marker count does not match "
                f"image features: markers={processor_marker_count}, "
                f"features={expected_image_tokens}"
            )
        if processor_image_token_id != image_token_id:
            inputs.input_ids[inputs.input_ids == processor_image_token_id] = image_token_id
        image_token_count = int((inputs.input_ids == image_token_id).sum().item())
        if image_token_count != expected_image_tokens:
            raise RuntimeError(
                "OneVision-2 model image-token count does not match image features: "
                f"tokens={image_token_count}, features={expected_image_tokens}"
            )
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
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output_ids[:, input_tokens:]
    text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text, input_tokens, int(generated.shape[-1]), elapsed


def run_task(model, processor, model_id: str, task: str, count: int, seed: int):
    import torch

    dataset = shared.dataset_for(task)
    indices = common.pick_indices(len(dataset), count, seed)
    items, scores, total_tokens, total_generation_s = [], [], 0, 0.0
    # The first measured request follows an unmeasured real-image warmup.
    # Clear its CUDA peak so the first item records only its own allocations.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for position, index in enumerate(indices):
        row = dataset[index]
        prompt, expected = shared.prompt_for(task, row)
        try:
            output, input_tokens, output_tokens, generation_s = generate(
                model, processor, row["image"], prompt, model_id
            )
            prediction, score, parse_error = shared.score_for(task, output, expected)
            items.append({
                "index": index,
                "score": score,
                "prediction": prediction,
                "output": output,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "parse_error": parse_error,
                "generation_s": round(generation_s, 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            })
            scores.append(score)
            total_tokens += output_tokens
            total_generation_s += generation_s
        except Exception as exc:
            items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        finally:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(f"PROGRESS {model_id} {task} {position + 1}/{len(indices)}", flush=True)
    successful = [item for item in items if "score" in item]
    if task == "graph_grab_lite":
        groups = {}
        for item in successful:
            row = dataset[item["index"]]
            groups.setdefault(row["task"], []).append(item["score"])
        group_scores = {name: statistics.mean(values) for name, values in sorted(groups.items())}
        for item in successful:
            row = dataset[item["index"]]
            item.update(pid=row["pid"], task_type=row["task"],
                        dataset_category=row["category"], complexity=row["complexity"])
    return {
        "requested_n": len(indices),
        "completed_n": len(successful),
        "failed_n": len(indices) - len(successful),
        "score": statistics.mean(scores) if scores else None,
        "group_scores": group_scores if task == "graph_grab_lite" else None,
        "mean_generation_s": statistics.mean(i["generation_s"] for i in successful) if successful else None,
        "generated_tokens_per_s": total_tokens / total_generation_s if total_generation_s else None,
        "peak_allocated_gib": max((i["peak_allocated_gib"] for i in successful), default=None),
        "items": items,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    models = result["models"]
    lines = [
        "# LLaVA family benchmark", "",
        f"Generated: {result['generated_at']}",
        f"Seed: {result['seed']}; GPU memory ceiling: {result['memory_cap_gib']:.1f} GiB.",
        f"Transformers: {result.get('runtime_versions', {}).get('transformers', 'unrecorded')}.",
        "Prompts, fixed sample selection, and scorers come from the shared Qwen-VL benchmark.",
        "The full-suite macro is the unweighted mean of eight scores and requires every shared example to complete.",
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
    lines += ["", "## Method", "",
              "Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.",
              "Each request contains one still image and one shared task prompt. The processor applies its model-specific image preprocessing and chat template.",
              "Requests run serially with greedy decoding and at most one model loaded at a time.",
              "The 34B checkpoint is skipped before loading when its cached weights alone exceed the 30 GiB budget.",
              "Results record the Transformers version; resume requires the same version for comparable outputs.",
              "Missing prepared task data is recorded per task, allowing available tasks to run and preserving partial results.",
              "", "Raw per-item outputs, errors, cache paths, and run settings are in `results.json`.", ""]
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
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "llava_family")
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
        if (result.get("seed") != args.seed or result.get("task_names") != list(task_names)
                or result.get("task_sizes") != task_sizes):
            parser.error("--resume requires matching seed, task list, and sample counts")
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
            for task in task_names
        ):
            continue
        if old:
            result["models"].remove(old)
        record = {"model_id": entry["id"], "backend": entry["backend"],
                  "snapshot": entry["snapshot"], "weight_bytes": entry["weight_bytes"],
                  "runtime_versions": {"transformers": transformers_version},
                  "prompt_template": (
                      "XTuner LLaVA-Phi-3 native image-token format" if entry["id"] == "xtuner/llava-phi-3-mini-hf"
                      else "LLaVA-OneVision-2 model-provided Qwen3 chat template" if entry["id"] == "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
                      else "cached processor apply_chat_template"
                  ),
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
                    print(f"RUN {entry['id']} {task} n={task_sizes[task]}", flush=True)
                    try:
                        record["tasks"][task] = run_task(
                            model, processor, entry["id"], task, task_sizes[task], args.seed
                        )
                    except Exception as exc:
                        record["tasks"][task] = {
                            "requested_n": task_sizes[task], "completed_n": 0,
                            "failed_n": task_sizes[task], "score": None,
                            "error": f"{type(exc).__name__}: {exc}", "items": [],
                        }
                        print(f"TASK ERROR {task}: {type(exc).__name__}: {exc}", flush=True)
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                record["status"] = "complete" if all(
                    record["tasks"].get(task, {}).get("completed_n") == task_sizes[task]
                    for task in task_names
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
