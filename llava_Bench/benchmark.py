#!/usr/bin/env python3
"""Offline, serial Transformers benchmark for cached LLaVA family checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
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
EDGE_ALLOCATOR_RESERVE_GIB = 0.75
SEED = 3407
EDGE_MODEL_IDS = {
    "xtuner/llava-phi-3-mini-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
}
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


class WholeDeviceMemorySampler:
    def __init__(self, interval_s: float = 0.15, device_index: int = 0):
        self.interval_s = interval_s
        self.device_index = device_index
        self.peak_mib = 0
        self.sample_count = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            self.error = self.error or "TimeoutError: nvidia-smi sampler worker did not stop within 3 seconds"

    def _sample(self):
        while not self._stop.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "-i", str(self.device_index), "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL, timeout=3,
                ).strip().splitlines()[0]
                self.peak_mib = max(self.peak_mib, int(value))
                self.sample_count += 1
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval_s)

    @property
    def peak_gib(self):
        return round(self.peak_mib / 1024, 3) if self.sample_count else None


def gpu_memory_used_mib(device_index: int = 0):
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "-i", str(device_index), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        ).strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    except Exception:
        pass
    return "out of memory" in f"{type(exc).__name__}: {exc}".lower()


def clear_cuda_after_oom() -> str | None:
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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


def load_model(entry: dict[str, Any], memory_cap_gib: float = MEMORY_CAP_GIB):
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
        max_memory={device: int(memory_cap_gib * 1024**3)},
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


def prepare_inputs(model, processor, image, prompt: str, model_id: str):
    messages = make_messages(prompt)
    if getattr(processor, "chat_template", None):
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif model_id == "xtuner/llava-phi-3-mini-hf":
        rendered = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
    else:
        raise RuntimeError("cached processor does not provide apply_chat_template")
    inputs = processor(
        text=[rendered], images=[image.convert("RGB")], padding=True,
        return_tensors="pt",
    ).to(model.device)
    return inputs


def input_token_count(model, processor, image, prompt: str, model_id: str) -> int:
    inputs = prepare_inputs(model, processor, image, prompt, model_id)
    return int(inputs.input_ids.shape[-1])


def generate(model, processor, image, prompt: str, model_id: str,
             *, max_new_tokens: int = 256, context_limit_tokens: int | None = None):
    import torch

    inputs = prepare_inputs(model, processor, image, prompt, model_id)
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
    allowed_context = context_limit_tokens if context_limit_tokens is not None else max_context
    if allowed_context and input_tokens + max_new_tokens > allowed_context:
        raise RuntimeError(
            f"request needs {input_tokens + max_new_tokens} tokens including output; "
            f"configured context is {allowed_context}"
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output_ids[:, input_tokens:]
    text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text, input_tokens, int(generated.shape[-1]), elapsed


def run_task(model, processor, model_id: str, task: str, count: int, seed: int,
             context_limit_tokens: int | None = None):
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
            input_tokens = input_token_count(model, processor, row["image"], prompt, model_id) \
                if context_limit_tokens is not None else None
            max_new_tokens = (min(256, context_limit_tokens - input_tokens)
                              if input_tokens is not None else 256)
            if max_new_tokens < 1:
                raise RuntimeError(
                    f"input length {input_tokens} reaches measured context {context_limit_tokens}"
                )
            output, input_tokens, output_tokens, generation_s = generate(
                model, processor, row["image"], prompt, model_id,
                max_new_tokens=max_new_tokens, context_limit_tokens=context_limit_tokens,
            )
            actual_context_tokens = input_tokens + output_tokens
            if context_limit_tokens is not None and actual_context_tokens > context_limit_tokens:
                raise RuntimeError(
                    f"scored request used {actual_context_tokens} tokens beyond probed context "
                    f"{context_limit_tokens}"
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
            if context_limit_tokens is not None:
                items[-1].update(
                    context_limit_tokens=context_limit_tokens,
                    generation_max_new_tokens=max_new_tokens,
                    actual_context_tokens=actual_context_tokens,
                    context_limit_enforced=True,
                )
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


def probe_context(model, processor, model_id: str, seed: int, configured_context: int,
                  context_limit: int, cap_gib: float) -> dict[str, Any]:
    """Select a demanding real image and measure bounded deterministic context attempts."""
    dataset = shared.dataset_for("ai2d")
    device_index = int(getattr(model.device, "index", 0) or 0)
    indices = common.pick_indices(len(dataset), min(100, len(dataset)), seed)
    rows, selection_attempts, selection_errors = [], [], []
    selection_peak = None
    for index in indices:
        row = dataset[index]
        prompt, _ = shared.prompt_for("ai2d", row)
        sampler = WholeDeviceMemorySampler(device_index=device_index)
        sampler.start()
        before = gpu_memory_used_mib(device_index)
        failure = None
        try:
            tokens = input_token_count(model, processor, row["image"], prompt, model_id)
        except Exception as caught:
            failure = caught
            tokens = None
        error = f"{type(failure).__name__}: {failure}" if failure is not None else None
        after = gpu_memory_used_mib(device_index)
        sampler.stop()
        measured_peak = max((value for value in (before, after, sampler.peak_mib if sampler.sample_count else None)
                             if value is not None), default=None)
        peak_gib = measured_peak / 1024 if measured_peak is not None else None
        if peak_gib is not None:
            selection_peak = max(selection_peak or 0, peak_gib)
        valid = tokens is not None and sampler.sample_count > 0 and sampler.error is None
        selection_attempts.append({
            "image_index": index, "input_tokens": tokens, "error": error,
            "peak_whole_device_gib": peak_gib,
            "whole_device_sample_count": sampler.sample_count,
            "whole_device_sample_error": sampler.error,
            "whole_device_sampler_verified": sampler.sample_count > 0 and sampler.error is None,
            "passed": valid and peak_gib is not None and peak_gib <= cap_gib,
        })
        if failure is not None and is_cuda_oom(failure):
            cleanup_error = clear_cuda_after_oom()
            selection_attempts[-1]["cuda_cache_cleanup_attempted"] = True
            selection_attempts[-1]["cuda_cache_cleanup_error"] = cleanup_error
            selection_errors.append({
                "image_index": index, "error": error,
                "whole_device_sample_count": sampler.sample_count,
                "whole_device_sampler_error": sampler.error,
                "peak_whole_device_gib": peak_gib,
                "cuda_cache_cleanup_error": cleanup_error,
            })
        elif failure is not None:
            selection_errors.append({"image_index": index, "error": error})
        if tokens is not None:
            rows.append({"index": index, "row": row, "prompt": prompt, "input_tokens": tokens,
                         "sampler_verified": sampler.sample_count > 0 and sampler.error is None,
                         "peak_gib": peak_gib})
    selection_complete = len(indices) == 100 and len(rows) == 100 \
        and all(item["sampler_verified"] for item in rows) \
        and all(item["passed"] for item in selection_attempts)
    if not rows:
        return {"passed": False, "error": "no AI2D image produced a measured input length",
                "image_dataset": "AI2D test", "image_selection_sample_count": len(indices),
                "image_selection_success_count": 0, "image_selection_complete": False,
                "image_selection_attempts": selection_attempts,
                "errors": selection_errors,
                "image_selection_peak_whole_device_gib": selection_peak}
    selected = max(rows, key=lambda item: item["input_tokens"])
    base_tokens = selected["input_tokens"]
    search_limit = min(configured_context, context_limit)
    # Search only requests that fit the configured context. Processor-only checks avoid
    # sending out-of-context generations to the model and remain inside attempt sampling.
    low, high = 0, max(0, search_limit)
    best = None
    attempts = []
    probe_errors = list(selection_errors)
    while low <= high:
        repeats = (low + high) // 2
        prompt = selected["prompt"] + (" context" * repeats)
        sampler = WholeDeviceMemorySampler(device_index=device_index)
        sampler.start()
        before = gpu_memory_used_mib(device_index)
        failure = None
        try:
            measured_input = input_token_count(model, processor, selected["row"]["image"], prompt, model_id)
        except Exception as caught:
            failure = caught
            error = f"{type(caught).__name__}: {caught}"
            after = gpu_memory_used_mib(device_index)
            sampler.stop()
            measured_peak = max((value for value in (before, after, sampler.peak_mib if sampler.sample_count else None)
                                 if value is not None), default=None)
            peak_gib = measured_peak / 1024 if measured_peak is not None else None
            sampler_valid = sampler.sample_count > 0 and sampler.error is None
            oom_attempt = is_cuda_oom(failure)
            attempt = {
                "filler_repetitions": repeats, "input_tokens": None,
                "peak_whole_device_gib": peak_gib,
                "whole_device_sample_count": sampler.sample_count,
                "whole_device_sampler_error": sampler.error,
                "whole_device_sampler_verified": sampler_valid,
                "error": error, "oom_upper_bound": oom_attempt, "passed": False,
            }
            attempts.append(attempt)
            recoverable_oom = oom_attempt and sampler_valid and peak_gib is not None and peak_gib <= cap_gib
            if oom_attempt:
                cleanup_error = clear_cuda_after_oom()
                attempt["cuda_cache_cleanup_attempted"] = True
                attempt["cuda_cache_cleanup_error"] = cleanup_error
                recoverable_oom = recoverable_oom and cleanup_error is None
            attempt["recoverable_oom_upper_bound"] = recoverable_oom
            if not recoverable_oom:
                probe_errors.append(error if not oom_attempt else (
                    sampler.error or "OOM upper bound lacked complete under-cap sampling or cache cleanup"
                ))
            high = repeats - 1
            continue
        if measured_input + 8 > search_limit:
            after = gpu_memory_used_mib(device_index)
            sampler.stop()
            measured_peak = max((value for value in (before, after, sampler.peak_mib if sampler.sample_count else None)
                                 if value is not None), default=None)
            peak_gib = measured_peak / 1024 if measured_peak is not None else None
            sampler_valid = sampler.sample_count > 0 and sampler.error is None
            attempts.append({"filler_repetitions": repeats, "input_tokens": measured_input,
                             "skipped": True, "reason": "would exceed configured context",
                             "peak_whole_device_gib": peak_gib,
                             "whole_device_sample_count": sampler.sample_count,
                             "whole_device_sampler_error": sampler.error,
                             "whole_device_sampler_verified": sampler_valid,
                             "passed": sampler_valid and peak_gib is not None and peak_gib <= cap_gib})
            if not sampler_valid or peak_gib is None or peak_gib > cap_gib:
                probe_errors.append(sampler.error or "unverified or over-cap context-boundary attempt")
            high = repeats - 1
            continue
        failure = None
        try:
            output, input_tokens, output_tokens, _ = generate(
                model, processor, selected["row"]["image"], prompt, model_id,
                max_new_tokens=8, context_limit_tokens=search_limit,
            )
            error = None
        except Exception as caught:
            failure = caught
            input_tokens, output_tokens = measured_input, None
            error = f"{type(caught).__name__}: {caught}"
        after = gpu_memory_used_mib(device_index)
        sampler.stop()
        measured_peak = max((value for value in (before, after, sampler.peak_mib if sampler.sample_count else None)
                             if value is not None), default=None)
        peak_gib = measured_peak / 1024 if measured_peak is not None else None
        sampler_valid = sampler.sample_count > 0 and sampler.error is None
        passed = (error is None and sampler_valid and output_tokens is not None
                  and input_tokens + output_tokens <= search_limit
                  and peak_gib is not None and peak_gib <= cap_gib)
        attempt = {
            "filler_repetitions": repeats, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "actual_context_tokens": (
                input_tokens + output_tokens if output_tokens is not None else None
            ), "peak_whole_device_gib": peak_gib, "whole_device_sample_count": sampler.sample_count,
            "whole_device_sampler_error": sampler.error, "whole_device_sampler_verified": sampler_valid,
            "error": error, "passed": passed,
        }
        attempts.append(attempt)
        oom_attempt = failure is not None and is_cuda_oom(failure)
        recoverable_oom = oom_attempt and sampler_valid and peak_gib is not None and peak_gib <= cap_gib
        attempt["oom_upper_bound"] = oom_attempt
        if oom_attempt:
            # Preserve this attempt as a failed upper bound, then reclaim cache before retrying lower.
            attempt["cuda_cache_cleanup_attempted"] = True
            cleanup_error = clear_cuda_after_oom()
            attempt["cuda_cache_cleanup_error"] = cleanup_error
            recoverable_oom = recoverable_oom and cleanup_error is None
        attempt["recoverable_oom_upper_bound"] = recoverable_oom
        if not passed:
            if not recoverable_oom:
                probe_errors.append(error or sampler.error or "context-probe attempt unverified or exceeded cap")
            high = repeats - 1
        else:
            best = attempt
            low = repeats + 1
    probe_peak_values = [selection_peak]
    probe_peak_values.extend(attempt.get("peak_whole_device_gib") for attempt in attempts)
    probe_peak = max((value for value in probe_peak_values if value is not None), default=None)
    probe_passed = bool(best and selection_complete and not probe_errors and probe_peak is not None
                        and probe_peak <= cap_gib)
    return {
        "passed": probe_passed,
        "image_dataset": "AI2D test", "image_index": selected["index"],
        "image_size_px": list(selected["row"]["image"].size),
        "image_selection_method": "maximum measured total input-token proxy across fixed AI2D examples; visual-token counts unavailable",
        "image_selection_sample_count": len(indices),
        "image_selection_success_count": len(rows),
        "image_selection_complete": selection_complete,
        "image_selection_total_input_token_proxy": base_tokens,
        "image_selection_token_measure": "total_input_token_proxy",
        "image_visual_token_count": None,
        "image_selection_attempts": selection_attempts,
        "configured_context_tokens": configured_context,
        "probe_search_limit_tokens": search_limit,
        "measured_context_tokens": best.get("actual_context_tokens") if best else None,
        "actual_input_tokens": best.get("input_tokens") if best else None,
        "output_tokens": best.get("output_tokens") if best else None,
        "whole_device_cap_gib": cap_gib,
        "peak_whole_device_gib": probe_peak,
        "image_selection_peak_whole_device_gib": selection_peak,
        "search_method": "binary search on deterministic context filler with measured processor tokens",
        "attempts": attempts,
        "errors": probe_errors,
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
    if result.get("whole_device_cap_gib") is not None:
        lines += ["", "## Edge cap verification", ""]
        lines.append(
            f"Allocator allowance: selected cap minus measured idle baseline minus "
            f"{result.get('allocator_reserve_gib', EDGE_ALLOCATOR_RESERVE_GIB):.2f} GiB reserve."
        )
        for model in models:
            probe = model.get("context_probe") or {}
            lines.extend([
                f"### {model['model_id']}", "",
                f"Cap verified: `{model.get('cap_verified', False)}`; evidence: `{model.get('cap_verification', {})}`.",
                f"Whole-device peak: {model.get('whole_device_peak_gib')} GiB; load peak: {model.get('load_peak_whole_device_gib')} GiB; scoring peak: {model.get('peak_gpu_used_gib')} GiB.",
                f"Idle baseline: {result.get('gpu_baseline_gib')} GiB; allocator reserve: {model.get('allocator_reserve_gib')} GiB; allocator/load limit: {model.get('allocator_memory_limit_gib')} GiB.",
                f"Context probe: {probe.get('measured_context_tokens')} tokens; image index {probe.get('image_index')}; input measure `{probe.get('image_selection_token_measure')}`; probe peak {probe.get('peak_whole_device_gib')} GiB.",
                "",
            ])
    lines += ["", "## Method", "",
              "Checkpoints are resolved from the local Hugging Face cache. Inference runs offline.",
              "Each request contains one still image and one shared task prompt. The processor applies its model-specific image preprocessing and chat template.",
              "Requests run serially with greedy decoding and at most one model loaded at a time.",
              f"The 34B checkpoint is skipped before loading when its cached weights alone exceed the {result['memory_cap_gib']:.1f} GiB budget.",
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--whole-device-cap-gib", type=int, choices=(8, 10, 12, 14, 16))
    parser.add_argument("--context-probe-limit", type=int, default=32768)
    args = parser.parse_args()
    edge_mode = args.whole_device_cap_gib is not None
    if edge_mode:
        MEMORY_CAP_GIB = float(args.whole_device_cap_gib)
        if not args.models or len(args.models) != 1 or args.models[0] not in EDGE_MODEL_IDS:
            parser.error("Edge mode requires exactly one screened LLaVA model ID")
        if args.tasks and tuple(args.tasks) != TASKS:
            parser.error("Edge mode requires the unchanged eight shared tasks")
        if args.samples is not None or args.seed != SEED or args.resume:
            parser.error("Edge mode requires shared sample counts and seed 3407; --resume is unsupported")
        if args.context_probe_limit < 1024:
            parser.error("--context-probe-limit must be at least 1024 tokens")
    elif args.context_probe_limit != 32768:
        parser.error("--context-probe-limit requires --whole-device-cap-gib")
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
                status = f"over {MEMORY_CAP_GIB:g} GiB budget"
            print(f"{entry['id']}\t{status}\t{size}\t{entry.get('snapshot') or '—'}")
        return 0

    task_names = tuple(args.tasks or TASKS)
    task_sizes = {task: args.samples if args.samples is not None else TASK_SIZES[task]
                  for task in task_names}
    if any(size < 1 for size in task_sizes.values()):
        parser.error("sample counts must be positive")
    output_dir = args.output_dir or (
        HERE / "results" / "edge_refinement" / f"{args.whole_device_cap_gib}gib"
        / args.models[0].replace("/", "--") if edge_mode else HERE / "results" / "llava_family"
    )
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
        raise SystemExit(f"the configured {MEMORY_CAP_GIB:g} GiB ceiling exceeds installed GPU memory")
    if not edge_mode:
        torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 1024**3 / total_memory, device)
    if edge_mode:
        properties = torch.cuda.get_device_properties(device)
        result.update(
            whole_device_cap_gib=MEMORY_CAP_GIB,
            allocator_reserve_gib=EDGE_ALLOCATOR_RESERVE_GIB,
            gpu_baseline_gib=None,
            hardware={"gpu_name": properties.name, "total_memory_gib": round(total_memory / 1024**3, 3),
                      "cuda_device": int(device)},
            runtime_versions={
                "transformers": transformers_version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            cap_enforcement_method=(
                "whole-device nvidia-smi memory.used polling; allocator fraction and max_memory load limit "
                f"after reserving {EDGE_ALLOCATOR_RESERVE_GIB:.2f} GiB beyond idle baseline"
            ),
        )

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
        if edge_mode:
            record.update(
                revision=Path(entry["snapshot"]).name if entry.get("snapshot") else None,
                quantization=(entry.get("config", {}).get("text_config", {}).get("torch_dtype")
                              or entry.get("config", {}).get("torch_dtype") or "checkpoint-native dtype"),
            )
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
            whole_monitor = None
            allocator_limit_gib = MEMORY_CAP_GIB
            if edge_mode:
                baseline_mib = gpu_memory_used_mib(int(device))
                baseline_gib = baseline_mib / 1024 if baseline_mib is not None else None
                result["gpu_baseline_gib"] = round(baseline_gib, 3) if baseline_gib is not None else None
                if baseline_gib is not None:
                    allocator_limit_gib = MEMORY_CAP_GIB - baseline_gib - EDGE_ALLOCATOR_RESERVE_GIB
                whole_monitor = WholeDeviceMemorySampler(device_index=int(device))
                whole_monitor.start()
                record.update(
                    whole_device_cap_gib=MEMORY_CAP_GIB,
                    allocator_reserve_gib=EDGE_ALLOCATOR_RESERVE_GIB,
                    configured_context_tokens=None,
                    cap_enforcement_method=result["cap_enforcement_method"],
                )
            try:
                if edge_mode and result.get("gpu_baseline_gib") is None:
                    raise RuntimeError("could not measure whole-device idle baseline")
                if edge_mode and result["gpu_baseline_gib"] >= MEMORY_CAP_GIB:
                    raise RuntimeError("whole-device idle baseline already reaches the selected cap")
                if edge_mode and allocator_limit_gib <= 0:
                    raise RuntimeError("no whole-device cap allowance remains after the idle baseline")
                if edge_mode:
                    torch.cuda.set_per_process_memory_fraction(
                        allocator_limit_gib * 1024**3 / total_memory, device
                    )
                print(f"LOAD {entry['id']}", flush=True)
                model, processor = load_model(entry, memory_cap_gib=allocator_limit_gib)
                record["load_peak_whole_device_gib"] = whole_monitor.peak_gib if whole_monitor else None
                record["checkpoint_context"] = getattr(
                    getattr(model.config, "text_config", model.config),
                    "max_position_embeddings", None
                )
                configured_context = int(record["checkpoint_context"] or 0)
                record["configured_context_tokens"] = configured_context
                record["configured_context"] = configured_context
                if edge_mode:
                    record["allocator_memory_limit_gib"] = allocator_limit_gib
                    record["allocator_reserve_gib"] = EDGE_ALLOCATOR_RESERVE_GIB
                context_limit = min(configured_context, args.context_probe_limit) if configured_context else 0
                if edge_mode and (not whole_monitor.sample_count or whole_monitor.error):
                    raise RuntimeError(
                        "whole-device sampler failed during model load: "
                        f"samples={whole_monitor.sample_count}, error={whole_monitor.error}"
                    )
                if edge_mode and (record["load_peak_whole_device_gib"] is None
                                  or record["load_peak_whole_device_gib"] > MEMORY_CAP_GIB):
                    raise RuntimeError("whole-device load peak is unmeasured or exceeded the selected cap")
                context_probe = None
                if edge_mode:
                    if context_limit < 1024:
                        raise RuntimeError(f"unsupported configured model context: {configured_context}")
                    context_probe = probe_context(
                        model, processor, entry["id"], SEED, configured_context,
                        context_limit, MEMORY_CAP_GIB,
                    )
                    record["context_probe"] = context_probe
                    record["scored_context_limit_tokens"] = context_probe.get("measured_context_tokens")
                    record["context_probe_max_context_tokens"] = context_probe.get("measured_context_tokens")
                    record["scored_context_enforcement"] = (
                        "max_new_tokens=min(256, measured_context-input_tokens); actual totals checked"
                    )
                    if not context_probe.get("passed"):
                        record.update(status="context_probe_failed", error="real-image context probe failed closed")
                        continue_after_model = False
                    else:
                        continue_after_model = True
                else:
                    continue_after_model = True
                # Real-image warmup exercises the exact single-image template
                # and loader path, but does not contribute to reported metrics.
                if continue_after_model:
                    warm_data = shared.dataset_for("ai2d")
                    if edge_mode:
                        warm_index = context_probe.get("image_index")
                        if (not isinstance(warm_index, int) or warm_index < 0
                                or warm_index >= len(warm_data)
                                or not context_probe.get("measured_context_tokens")):
                            raise RuntimeError("passing context probe lacks a valid warmup image or measured context")
                        warm = warm_data[warm_index]
                    else:
                        warm = warm_data[common.pick_indices(len(warm_data), 1, args.seed)[0]]
                    prompt, _ = shared.prompt_for("ai2d", warm)
                    if edge_mode:
                        n_tokens = input_token_count(model, processor, warm["image"], prompt, entry["id"])
                        n_new = min(256, context_probe["measured_context_tokens"] - n_tokens)
                        if n_new < 1:
                            raise RuntimeError("warmup input exceeds measured context")
                        generate(model, processor, warm["image"], prompt, entry["id"],
                                 max_new_tokens=n_new,
                                 context_limit_tokens=context_probe["measured_context_tokens"])
                    else:
                        generate(model, processor, warm["image"], prompt, entry["id"])
                    record["warmup"] = "one AI2D image request excluded from scores"
                    for task in task_names:
                        print(f"RUN {entry['id']} {task} n={task_sizes[task]}", flush=True)
                        task_monitor = WholeDeviceMemorySampler(device_index=int(device)) if edge_mode else None
                        if task_monitor:
                            task_monitor.start()
                        try:
                            task_result = run_task(
                                model, processor, entry["id"], task, task_sizes[task], args.seed,
                                context_limit_tokens=(context_probe["measured_context_tokens"] if edge_mode else None),
                            )
                        except Exception as exc:
                            task_result = {
                                "requested_n": task_sizes[task], "completed_n": 0,
                                "failed_n": task_sizes[task], "score": None,
                                "error": f"{type(exc).__name__}: {exc}", "items": [],
                            }
                            print(f"TASK ERROR {task}: {type(exc).__name__}: {exc}", flush=True)
                        finally:
                            if task_monitor:
                                task_monitor.stop()
                                task_result["peak_gpu_used_gib"] = task_monitor.peak_gib
                                task_result["whole_device_sample_count"] = task_monitor.sample_count
                                task_result["whole_device_sample_error"] = task_monitor.error
                        record["tasks"][task] = task_result
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
                if whole_monitor:
                    whole_monitor.stop()
                    record["whole_device_sample_count"] = whole_monitor.sample_count
                    record["whole_device_sample_error"] = whole_monitor.error
                    task_peaks = [task.get("peak_gpu_used_gib") for task in record.get("tasks", {}).values()
                                  if isinstance(task.get("peak_gpu_used_gib"), (int, float))]
                    scoring_peak = max(task_peaks) if task_peaks else None
                    probe_peak = (record.get("context_probe") or {}).get("peak_whole_device_gib")
                    measured_peaks = [whole_monitor.peak_gib, record.get("load_peak_whole_device_gib"),
                                      probe_peak, scoring_peak]
                    measured_peaks = [value for value in measured_peaks
                                      if isinstance(value, (int, float)) and math.isfinite(value)]
                    whole_peak = max(measured_peaks) if measured_peaks else None
                    record["whole_device_peak_gib"] = whole_peak
                    complete_suite = all(
                        record.get("tasks", {}).get(task, {}).get("requested_n") == TASK_SIZES[task]
                        and record.get("tasks", {}).get(task, {}).get("completed_n") == TASK_SIZES[task]
                        for task in TASKS
                    )
                    probe = record.get("context_probe") or {}
                    measured_context = probe.get("measured_context_tokens")
                    context_ok = bool(measured_context) and all(
                        item.get("context_limit_enforced") is True
                        and item.get("context_limit_tokens") == measured_context
                        and item.get("actual_context_tokens") is not None
                        and item.get("actual_context_tokens") <= measured_context
                        for task in record.get("tasks", {}).values()
                        for item in task.get("items", []) if "score" in item
                    )
                    task_sampling = complete_suite and all(
                        task.get("whole_device_sample_count", 0) > 0
                        and task.get("whole_device_sample_error") is None
                        and isinstance(task.get("peak_gpu_used_gib"), (int, float))
                        and 0 < task.get("peak_gpu_used_gib") <= MEMORY_CAP_GIB
                        for task in record.get("tasks", {}).values()
                    )
                    load_peak = record.get("load_peak_whole_device_gib")
                    load_ok = isinstance(load_peak, (int, float)) and 0 < load_peak <= MEMORY_CAP_GIB
                    whole_ok = (whole_monitor.sample_count > 0 and whole_monitor.error is None
                                and isinstance(whole_peak, (int, float))
                                and 0 < whole_peak <= MEMORY_CAP_GIB)
                    scoring_ok = isinstance(scoring_peak, (int, float)) and 0 < scoring_peak <= MEMORY_CAP_GIB
                    record["peak_gpu_used_gib"] = scoring_peak
                    record["scoring_peak_gpu_used_gib"] = scoring_peak
                    record["cap_verified"] = bool(
                        probe.get("passed") and complete_suite and context_ok and task_sampling
                        and load_ok and whole_ok and scoring_ok
                    )
                    record["cap_verification"] = {
                        "whole_device_cap_gib": MEMORY_CAP_GIB,
                        "context_probe_passed": bool(probe.get("passed")),
                        "complete_shared_task_suite": complete_suite,
                        "scored_items_within_probed_context": context_ok,
                        "scoring_peak_within_cap": scoring_ok,
                        "load_peak_whole_device_gib": load_peak,
                        "load_peak_within_cap": load_ok,
                        "whole_device_sampling_verified": whole_monitor.sample_count > 0 and whole_monitor.error is None,
                        "whole_device_sampler_error": whole_monitor.error,
                        "task_sampling_verified": task_sampling,
                        "whole_device_peak_gib": whole_peak,
                        "whole_device_peak_within_cap": whole_ok,
                        "whole_run_peak_within_cap": whole_ok,
                    }
                    if record.get("status") == "complete" and not record["cap_verified"]:
                        record["status"] = "unverified_cap"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
    write_report(output_dir / "report.md", result)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {result_path} and {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
