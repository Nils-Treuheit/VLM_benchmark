#!/usr/bin/env python3
"""Offline, single-GPU Qwen2/3-VL comparison on AI2D and TextVQA."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from gpu_queue import run_queued  # noqa: E402

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
HUB = HF_HOME / "hub"
DATASETS = {
    "ai2d": {"id": "lmms-lab/ai2d", "split": "test", "pattern": "ai2d-test-*.arrow"},
    "textvqa": {"id": "facebook/textvqa", "split": "validation", "pattern": "textvqa-validation.arrow"},
}
MAX_SIDE = 896
DEFAULT_SAMPLES = 100
SEED = 3407
MIN_CONTEXT_TARGET = 8192
DEFAULT_CONTEXT_TARGET = 128 * 1024
MAX_CONTEXT_TARGET = 256 * 1024
DEFAULT_CONTEXT = MAX_CONTEXT_TARGET
DEFAULT_MEMORY_CAP_GIB = 30.0
KV_CACHE_BITS = 8
KV_CACHE_BACKEND = "hqq"
KV_CACHE_RESIDUAL_TOKENS = 128


def discover_models() -> list[dict[str, str]]:
    """Find cached Transformers checkpoints in the Qwen2/3-VL model families."""
    found = []
    if not HUB.is_dir():
        return found
    for cache in sorted(HUB.glob("models--*")):
        if not cache.is_dir():
            continue
        model_id = cache.name.removeprefix("models--").replace("--", "/")
        lower = model_id.lower()
        if "gguf" in lower:
            continue
        if not re.search(r"qwen(?:2(?:\.5)?|3)-vl", lower):
            continue
        found.append({"model_id": model_id, "cache": str(cache)})
    return found


def resolve_snapshot(cache: Path) -> Path | None:
    ref = cache / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = cache / "snapshots" / revision
    else:
        # Some complete HF cache exports have their snapshot and all weight
        # blobs but no refs/main pointer. Resolve only when unambiguous.
        snapshots = list((cache / "snapshots").glob("*")) if (cache / "snapshots").is_dir() else []
        if len(snapshots) != 1:
            return None
        snapshot = snapshots[0]
    if not (snapshot / "config.json").is_file():
        return None
    weights = [path for path in snapshot.glob("*.safetensors") if path.is_file()]
    weights += [path for path in snapshot.glob("pytorch_model*.bin") if path.is_file()]
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            shard_names = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
            if shard_names and all((snapshot / name).is_file() for name in shard_names):
                weights.append(index)
        except (OSError, ValueError, TypeError):
            pass
    return snapshot if weights else None


def detect_mtp_head(snapshot: Path) -> str:
    """Check index metadata or safetensors headers without loading weights to the GPU."""
    index_files = list(snapshot.glob("*.safetensors.index.json"))
    for index in index_files:
        try:
            keys = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            if any(name.startswith("mtp.") or ".mtp." in name for name in keys):
                return "yes"
        except (OSError, ValueError, TypeError):
            continue
    try:
        from safetensors import safe_open

        for shard in snapshot.glob("*.safetensors"):
            with safe_open(str(shard), framework="pt", device="cpu") as reader:
                if any(key.startswith("mtp.") or ".mtp." in key for key in reader.keys()):
                    return "yes"
        if list(snapshot.glob("*.safetensors")):
            return "no"
    except (ImportError, OSError, ValueError):
        pass
    return "unknown"


def estimated_kv_bytes(tokens: int, fp16_bytes_per_token: int) -> float:
    residual = min(tokens, KV_CACHE_RESIDUAL_TOKENS)
    quantized = max(0, tokens - residual)
    q8_bytes_per_token = fp16_bytes_per_token * (KV_CACHE_BITS / 16) * 1.125
    return residual * fp16_bytes_per_token + quantized * q8_bytes_per_token


def context_memory_profile(
    model_allocated_bytes: int,
    fp16_bytes_per_token: int | None,
    checkpoint_context: int | None,
    estimated_context: int | None,
) -> list[dict[str, Any]]:
    """Estimate model + Q8 KV allocation at requested context tiers, falling back to capacity."""
    tiers = (
        ("min", MIN_CONTEXT_TARGET),
        ("default", DEFAULT_CONTEXT_TARGET),
        ("max", MAX_CONTEXT_TARGET),
    )
    capacity = min(
        checkpoint_context or 0,
        estimated_context or 0,
    )
    profile = []
    for name, target in tiers:
        context = target if capacity >= target else capacity
        possible = context >= MIN_CONTEXT_TARGET and fp16_bytes_per_token is not None
        memory = (
            (model_allocated_bytes + estimated_kv_bytes(context, fp16_bytes_per_token)) / (1024**3)
            if possible else None
        )
        profile.append({
            "tier": name,
            "target_context": target,
            "context_tokens": context if possible else None,
            "estimated_vram_gib": round(memory, 2) if memory is not None else None,
            "supported": possible,
        })
    return profile


def local_only_dataset(task: str):
    """Read the already-cached Arrow split directly; never execute dataset scripts."""
    from datasets import Dataset, concatenate_datasets

    spec = DATASETS[task]
    cache_root = HF_HOME / "datasets"
    prefix = spec["id"].replace("/", "___")
    matches = sorted(cache_root.glob(f"{prefix}/**/{spec['pattern']}"))
    if not matches:
        raise FileNotFoundError(
            f"cached Arrow split not found for {spec['id']} {spec['split']} under {cache_root}"
        )
    shards = [Dataset.from_file(str(path)) for path in matches]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def normalize_vqa(answer: Any) -> str:
    value = str(answer or "").lower().strip()
    value = re.sub(r"(?<!\d)\.(?!\d)", "", value)
    value = re.sub(r"[,;:!?()\[\]{}\"/\\_+=<>@`~-]", " ", value)
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    value = " ".join(number_words.get(word, word) for word in value.split())
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def vqa_score(prediction: str, references: list[Any]) -> float:
    pred = normalize_vqa(prediction)
    answers = [normalize_vqa(a.get("answer", "") if isinstance(a, dict) else a) for a in references]
    if not pred or not answers:
        return 0.0
    scores = []
    for held_out in range(len(answers)):
        others = answers[:held_out] + answers[held_out + 1:]
        matches = sum(answer == pred for answer in others)
        scores.append(min(matches / 3.0, 1.0))
    return statistics.mean(scores)


def pick_indices(size: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(size), min(size, count)))


def resize_image(image):
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def last_choice(text: str) -> str | None:
    """Read a choice from the final answer, not from Qwen's reasoning text."""
    text = final_response(text)
    patterns = (
        r"(?i)\b(?:final\s+)?answer\s*(?:is|:)?\s*\(?([A-D])\)?\b",
        r"(?im)^\s*(?:final\s+)?\(?([A-D])\)?[.)]?\s*$",
        r"(?im)^\s*([A-D])[.)]\s+\S.*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    return None


def final_response(text: str) -> str:
    """Remove Qwen thinking text so task scorers see only the final answer."""
    return text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else text.strip()


def task_prompt(task: str, row: dict[str, Any]) -> tuple[str, str | list[Any]]:
    if task == "ai2d":
        options = row["options"]
        labels = "ABCD"
        choices = "\n".join(f"{labels[i]}) {option}" for i, option in enumerate(options))
        prompt = (
            f"{row['question']}\n\nOptions:\n{choices}\n\n"
            "Choose one option. Return only its letter, with no explanation."
        )
        answer = int(row["answer"])
        return prompt, labels[answer] if 0 <= answer < 4 else ""
    prompt = (
        f"{row['question']}\n"
        "Answer using the shortest exact text visible in the image. "
        "Return only the answer, with no explanation or list of alternatives."
    )
    return prompt, row.get("answers") or []


def load_model(model_id: str, snapshot: Path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark only runs on GPU")
    device_index = torch.cuda.current_device()
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        device_map={"": device_index},
        dtype="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    model.eval()
    return model, processor


def generate_one(model, processor, image, prompt: str, model_id: str, sample_seed: int,
                 max_context: int, cache_bits: int):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from qwen_vl_utils import process_vision_info

    image = resize_image(image)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[rendered], images=images, videos=videos, padding=True, return_tensors="pt"
    ).to(model.device)
    input_tokens = int(inputs.input_ids.shape[-1])
    if input_tokens + 256 > max_context:
        raise RuntimeError(
            f"request needs {input_tokens + 256} tokens including output; cap is {max_context}"
        )

    # Use the documented Qwen3-VL profile settings when the model name identifies
    # an Instruct or Thinking checkpoint. Greedy decoding is used for older Qwen2-VL.
    lower = model_id.lower()
    is_thinking = "thinking" in lower
    # G-Substrate is an Instruct-derived Qwen3-VL fine-tune; its repository
    # name does not include the word "Instruct".
    is_qwen3 = "qwen3-vl" in lower or "g-substrate-qwen3-vl" in lower
    generation = {"max_new_tokens": 256, "do_sample": False}
    logits_processor = None
    if is_qwen3:
        if is_thinking:
            # Thinking checkpoints emit their answer after </think>; 256 tokens
            # was truncating many outputs before that point.
            generation["max_new_tokens"] = 1024
        generation.update(
            do_sample=True,
            temperature=1.0 if is_thinking else 0.7,
            top_p=0.95 if is_thinking else 0.8,
            top_k=20,
            repetition_penalty=1.0,
        )
        if not is_thinking:
            class PresencePenalty(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    token_ids = input_ids.unique(dim=1)
                    penalty = torch.full_like(token_ids, -1.5, dtype=scores.dtype)
                    scores.scatter_add_(1, token_ids, penalty)
                    return scores

            logits_processor = LogitsProcessorList([PresencePenalty()])

    torch.manual_seed(sample_seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            logits_processor=logits_processor,
            cache_implementation="quantized",
            cache_config={
                "backend": KV_CACHE_BACKEND,
                "nbits": cache_bits,
                "residual_length": KV_CACHE_RESIDUAL_TOKENS,
            },
            **generation,
        )
    torch.cuda.synchronize()
    generation_s = time.perf_counter() - started
    generated_ids = output_ids[:, input_tokens:]
    output_tokens = int(generated_ids.shape[-1])
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text, input_tokens, output_tokens, generation_s


def run_task(model, processor, model_id: str, task: str, count: int, seed: int,
             max_context: int, cache_bits: int) -> dict[str, Any]:
    import torch

    spec = DATASETS[task]
    dataset = local_only_dataset(task)
    indices = pick_indices(len(dataset), count, seed)
    scores, per_item, total_out = [], [], 0

    for position, index in enumerate(indices):
        row = dataset[index]
        prompt, expected = task_prompt(task, row)
        image = row["image"]
        start = time.perf_counter()
        try:
            text, in_tokens, out_tokens, generation_s = generate_one(
                model, processor, image, prompt, model_id, seed + position, max_context, cache_bits
            )
            wall_s = time.perf_counter() - start
            if task == "ai2d":
                prediction = last_choice(text)
                score = float(prediction == expected)
                correct = score == 1.0
            else:
                prediction = final_response(text)
                score = vqa_score(prediction, expected if isinstance(expected, list) else [])
                correct = score == 1.0
            torch.cuda.synchronize()
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
            scores.append(score)
            total_out += out_tokens
            per_item.append({
                "index": index,
                "score": score,
                "correct": correct,
                "prediction": prediction,
                "output": text,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "wall_s": round(wall_s, 4),
                "generation_s": round(generation_s, 4),
                "peak_allocated_gib": round(peak_allocated, 3),
                "peak_reserved_gib": round(peak_reserved, 3),
            })
        except Exception as exc:
            per_item.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(f"PROGRESS {model_id} {task} {position + 1}/{len(indices)}", flush=True)

    valid = [item for item in per_item if "score" in item]
    latencies = [item["wall_s"] for item in valid]
    rates = [item["output_tokens"] / item["generation_s"] for item in valid if item["generation_s"] > 0]
    return {
        "dataset": spec["id"],
        "split": spec["split"],
        "requested_n": len(indices),
        "completed_n": len(valid),
        "failed_n": len(indices) - len(valid),
        "score": round(statistics.mean(scores), 4) if scores else None,
        "mean_wall_s_per_item": round(statistics.mean(latencies), 4) if latencies else None,
        "median_wall_s_per_item": round(statistics.median(latencies), 4) if latencies else None,
        "mean_generation_tok_s": round(statistics.mean(rates), 2) if rates else None,
        "mean_output_tokens": round(total_out / len(valid), 2) if valid else None,
        "peak_allocated_gib": round(max((i["peak_allocated_gib"] for i in valid), default=0), 3),
        "peak_reserved_gib": round(max((i["peak_reserved_gib"] for i in valid), default=0), 3),
        "items": per_item,
    }


def write_report(path: Path, results: dict[str, Any]) -> None:
    models = results["models"]
    task_names = ("ai2d", "textvqa")
    lines = [
        "# Qwen2/3-VL local benchmark",
        "",
        f"Generated: {results['generated_at']}",
        f"Sample size: {results['samples_per_task']} examples per task; seed {results['seed']}.",
        f"Hard limits: {results['memory_cap_gib']:.1f} GiB VRAM and {results['task_context_cap']:,} total tokens; "
        f"{results['kv_cache_bits']}-bit {KV_CACHE_BACKEND} KV cache.",
        "",
        "## Task performance",
        "",
        "Scores are AI2D exact option accuracy and TextVQA consensus accuracy.",
        "Deltas are percentage points from the best completed model on that task.",
        "",
        "| Model | AI2D score (n) | Δ pp | TextVQA score (n) | Δ pp | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    best = {
        task: max((m.get("tasks", {}).get(task, {}).get("score") for m in models
                   if m.get("tasks", {}).get(task, {}).get("score") is not None), default=None)
        for task in task_names
    }
    for model in models:
        data = [model.get("tasks", {}).get(t, {}) for t in task_names]
        vals = [task_data.get("score") for task_data in data]
        deltas = [None if value is None or best[t] is None else (value - best[t]) * 100
                  for value, t in zip(vals, task_names)]
        status = model.get("status", "ok")
        score_cells = [
            f"{fmt_pct(d.get('score'))} ({d.get('completed_n', 0)}/{d.get('requested_n', 0)})"
            for d in data
        ]
        lines.append(
            f"| {model['model_id']} | {score_cells[0]} | {fmt_delta(deltas[0])} | "
            f"{score_cells[1]} | {fmt_delta(deltas[1])} | {status} |"
        )
    lines += [
        "",
        "## VRAM and speed by task",
        "",
        "VRAM is this process's peak CUDA allocated/reserved memory after model load. "
        "Wall time includes image preprocessing and generation; generation tok/s includes the full generate call. "
        "Counts show successful / requested examples; failures are excluded from the score denominator.",
        "",
        "| Model | Task | Completed / requested | Peak allocated GiB | Peak reserved GiB | Mean wall s/item | Median wall s/item | Generation tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for task in task_names:
            data = model.get("tasks", {}).get(task, {})
            lines.append(
                f"| {model['model_id']} | {task} | {data.get('completed_n', 0)}/{data.get('requested_n', 0)} | "
                f"{fmt_num(data.get('peak_allocated_gib'))} | "
                f"{fmt_num(data.get('peak_reserved_gib'))} | {fmt_num(data.get('mean_wall_s_per_item'))} | "
                f"{fmt_num(data.get('median_wall_s_per_item'))} | {fmt_num(data.get('mean_generation_tok_s'))} |"
            )
    lines += [
        "",
        "## Method",
        "",
        "Models are discovered from the local Hugging Face cache. GGUF repositories are excluded "
        "because this harness keeps one Transformers backend for comparable measurements.",
        "AI2D uses the cached test split; TextVQA uses the cached validation split. Arrow files are read "
        "directly from disk without executing dataset loading scripts. Both use "
        "seeded random examples and a common 896-pixel maximum image side.",
        "Qwen3-VL uses the Unsloth guide's Instruct or Thinking sampling values. Thinking calls allow 1,024 output tokens so they can finish the reasoning turn and emit the final answer; other calls allow 256. "
        "Other checkpoints use greedy decoding. Results describe this local sample, not leaderboard scores.",
        "",
        "Raw per-item outputs and failures are in `results.json`.",
        "",
        "## Context and MTP configuration",
        "",
        "Context VRAM is estimated from measured model-load allocation, checkpoint KV dimensions, HQQ Q8 storage "
        f"after a {KV_CACHE_RESIDUAL_TOKENS}-token FP16 residual, and a 1.5 GiB safety reserve. "
        "Each VRAM cell includes its context size in parentheses. If a target is unreachable, the cell shows "
        "the estimated last context that fits. These are configuration estimates, not long-context generation runs. "
        "Image and video tokens count toward context. Native MTP weights were not found in ready checkpoints.",
        "",
        "| Model | Checkpoint context | Min VRAM / context | Default VRAM / context | Max VRAM / context | MTP head | Max MTP drafts |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for model in models:
        profiles = {p["tier"]: p for p in model.get("context_profile", [])}
        cells = []
        for tier in ("min", "default", "max"):
            item = profiles.get(tier, {})
            context = item.get("context_tokens")
            if not item.get("supported") or context is None:
                cells.append("—")
            else:
                if context >= 1_000_000:
                    context_label = f"{context / 1_000_000:g}M"
                elif context >= 1024:
                    context_label = f"{context / 1024:g}K"
                else:
                    context_label = str(context)
                cells.append(f"{fmt_num(item['estimated_vram_gib'])} GiB ({context_label})")
        lines.append(
            f"| {model['model_id']} | {fmt_num(model.get('checkpoint_context'))} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | "
            f"{model.get('checkpoint_mtp_head', '—')} | 0 |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_num(value: Any) -> str:
    return "—" if value is None else f"{value:.2f}"


def fmt_pct(value: Any) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def fmt_delta(value: Any) -> str:
    return "—" if value is None else f"{value:+.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"random examples per dataset (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-context", type=int, default=DEFAULT_CONTEXT,
                        help=f"cap input plus output tokens per request (default: {DEFAULT_CONTEXT}; image/video tokens count)")
    parser.add_argument("--memory-cap-gib", type=float, default=DEFAULT_MEMORY_CAP_GIB,
                        help=f"per-process CUDA allocator limit (default: {DEFAULT_MEMORY_CAP_GIB})")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="*", default=None,
                        help="model IDs to select; by default scan all cached Qwen2/3-VL checkpoints")
    parser.add_argument("--list-models", action="store_true",
                        help="list discovered checkpoints and cache status, then exit")
    args = parser.parse_args()
    if args.samples < 1 or args.max_context < 512 or args.memory_cap_gib <= 0:
        parser.error("samples, context, and memory cap must be positive; context must be at least 512")
    inventory = discover_models()
    if args.models:
        wanted = set(args.models)
        inventory = [entry for entry in inventory if entry["model_id"] in wanted]
    if args.list_models:
        for entry in inventory:
            snapshot = resolve_snapshot(Path(entry["cache"]))
            print(f"{entry['model_id']}\t{'ready' if snapshot else 'incomplete or absent weights'}")
        return 0
    if not inventory:
        raise SystemExit(f"No cached Qwen2/3-VL checkpoints found under {HUB}")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this benchmark only runs on GPU")
    device_index = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device_index).total_memory
    if args.memory_cap_gib * (1024**3) > total_memory:
        raise SystemExit(
            f"requested {args.memory_cap_gib:.1f} GiB cap exceeds this GPU's "
            f"{total_memory / (1024**3):.2f} GiB"
        )
    torch.cuda.set_per_process_memory_fraction(
        args.memory_cap_gib * (1024**3) / total_memory, device_index
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hf_home": str(HF_HOME),
        "samples_per_task": args.samples,
        "seed": args.seed,
        "memory_cap_gib": args.memory_cap_gib,
        "task_context_cap": args.max_context,
        "kv_cache_bits": KV_CACHE_BITS,
        "models": [],
    }
    result_path = output_dir / "results.json"

    for entry in inventory:
        model_result: dict[str, Any] = {"model_id": entry["model_id"], "tasks": {}}
        result["models"].append(model_result)
        snapshot = resolve_snapshot(Path(entry["cache"]))
        if snapshot is None:
            model_result.update(status="skipped", error="local model snapshot or weights are incomplete")
            print(f"SKIP {entry['model_id']}: local weights unavailable", flush=True)
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            continue

        model = processor = None
        try:
            print(f"LOAD {entry['model_id']}", flush=True)
            model, processor = load_model(entry["model_id"], snapshot)
            torch.cuda.synchronize()
            text_config = getattr(model.config, "text_config", model.config)
            layers = getattr(text_config, "num_hidden_layers", None)
            kv_heads = getattr(text_config, "num_key_value_heads", None)
            head_dim = getattr(text_config, "head_dim", None)
            if head_dim is None and getattr(text_config, "hidden_size", None) and getattr(text_config, "num_attention_heads", None):
                head_dim = text_config.hidden_size // text_config.num_attention_heads
            advertised = getattr(text_config, "max_position_embeddings", None)
            current_allocated = torch.cuda.memory_allocated(device_index)
            kv_bytes_per_token = (2 * layers * kv_heads * head_dim * 2
                                  if layers and kv_heads and head_dim else None)
            safety_bytes = 1.5 * (1024**3)
            estimated_context = None
            if advertised and kv_bytes_per_token:
                remaining = max(0, args.memory_cap_gib * (1024**3) - current_allocated - safety_bytes)
                residual_bytes = kv_bytes_per_token * KV_CACHE_RESIDUAL_TOKENS
                if remaining <= residual_bytes:
                    ceiling = int(remaining // kv_bytes_per_token)
                else:
                    q8_bytes_per_token = kv_bytes_per_token * (KV_CACHE_BITS / 16) * 1.125
                    ceiling = KV_CACHE_RESIDUAL_TOKENS + int(
                        (remaining - residual_bytes) // q8_bytes_per_token
                    )
                ceiling = min(int(advertised), ceiling)
                estimated_context = max(0, (ceiling // 1024) * 1024)
            mtp_head = detect_mtp_head(snapshot)
            model_result.update(
                checkpoint_context=advertised,
                estimated_context_30gib=estimated_context,
                context_profile=context_memory_profile(
                    int(current_allocated), kv_bytes_per_token, advertised, estimated_context
                ),
                checkpoint_mtp_head=mtp_head,
                benchmark_context_cap=args.max_context,
            )
            model_result["model_load_peak_allocated_gib"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            )
            model_result["model_load_peak_reserved_gib"] = round(
                torch.cuda.max_memory_reserved() / (1024**3), 3
            )
            torch.cuda.reset_peak_memory_stats()
            for task in DATASETS:
                print(f"RUN {entry['model_id']} {task} n={args.samples}", flush=True)
                model_result["tasks"][task] = run_task(
                    model, processor, entry["model_id"], task, args.samples, args.seed,
                    args.max_context, KV_CACHE_BITS
                )
                model_result["status"] = "in_progress"
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            requested = sum(task.get("requested_n", 0) for task in model_result["tasks"].values())
            completed = sum(task.get("completed_n", 0) for task in model_result["tasks"].values())
            if completed == 0 and requested:
                model_result.update(status="inference_failed", error="no task examples completed")
            elif completed < requested:
                model_result["status"] = "complete_with_failures"
            else:
                model_result["status"] = "complete"
        except Exception as exc:
            model_result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"FAIL {entry['model_id']}: {model_result['error']}", flush=True)
        finally:
            del model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", result)
    print(f"Wrote {result_path} and {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    from benchmark_runners import main as corrected_main

    raise SystemExit(run_queued(corrected_main, "qwen-vl"))
