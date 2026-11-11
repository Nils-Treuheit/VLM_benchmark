#!/usr/bin/env python3
"""Full shared-suite runner for Molmo2, PaliGemma 2, and Florence-2."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import traceback
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QWEN_BENCH = ROOT / "qwenVL_Bench"
sys.path.insert(0, str(QWEN_BENCH))
sys.path.insert(0, str(ROOT))
from gpu_queue import run_queued  # noqa: E402
import benchmark as common  # noqa: E402
import benchmark_runners as shared  # noqa: E402

from model_adapters import (
    FLORENCE_ID, INTERNVL35_CANDIDATE_IDS, INTERNVL35_CANDIDATE_ROOT,
    INTERNVL35_GGUF_CANDIDATES, INTERNVL35_GGUF_IDS, INTERNVL35_HF_CANDIDATES,
    INTERNVL_GPU_IDS, INTERNVL_ID, INTERNVL_REVISION, LLAMA_VISION_ID,
    LLAMA_VISION_ROOT, LLAMA_VISION_SNAPSHOT, MOLMO2_ID, PALIGEMMA_ID,
    generate, load_florence, load_internvl, load_internvl35_hf,
    load_llama_cpp_server, load_llama_vision, load_molmo2, load_paligemma,
    prompt_for as adapter_prompt_for, generate_paligemma, generate_llama_vision,
    generate_internvl,
)

HF_HOME = Path(os.environ.get("HF_HOME", "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"))
HUB = HF_HOME / "hub"
MEMORY_CAP_GIB = 30.0
INTERNVL_ALLOCATED_TARGET_GIB = 26.0
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
    MOLMO2_ID,
    PALIGEMMA_ID,
    FLORENCE_ID,
    *INTERNVL35_CANDIDATE_IDS,
    *INTERNVL35_GGUF_IDS,
    LLAMA_VISION_ID,
)
ACTIVE_GPU_MONITOR = None
ACTIVE_WHOLE_GPU_MONITOR = None


class WholeDeviceMemorySampler:
    """Sample nvidia-smi whole-device memory, including desktop and other processes."""

    def __init__(self, *, interval_s: float = 0.15):
        self.interval_s = interval_s
        self.peak_mib = 0
        self.sample_count = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="whole-device-memory", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL, timeout=3,
                ).strip().splitlines()[0]
                value = int(output)
                self.peak_mib = max(self.peak_mib, value)
                self.sample_count += 1
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval_s)

    @property
    def peak_gib(self) -> float | None:
        return round(self.peak_mib / 1024, 3) if self.sample_count else None


class SampledProcessMemory:
    """Sample this process's CUDA memory from nvidia-smi while a run is active."""

    def __init__(self, pid: int, *, interval_s: float = 0.5):
        self.pid = pid
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.sample_count = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="cuda-memory-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=2,
                )
                for line in completed.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",", 1)]
                    if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == self.pid:
                        if fields[1].isdigit():
                            value = int(fields[1])
                            self.peak_mib = max(self.peak_mib or 0, value)
                            self.sample_count += 1
                        break
            except (OSError, subprocess.SubprocessError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval_s)

    @property
    def peak_gib(self) -> float | None:
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


class ResourceLimitError(RuntimeError):
    pass


def validate_internvl_placement(model) -> dict[str, Any]:
    """Reject any custom-model module or parameter outside cuda:0."""
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        raise RuntimeError("InternVL loader did not expose hf_device_map for placement audit")

    def is_cuda_zero(value: Any) -> bool:
        if isinstance(value, int):
            return value == 0
        return str(value).lower() in {"cuda:0", "0"}

    bad_map = {name: str(device) for name, device in device_map.items() if not is_cuda_zero(device)}
    if bad_map:
        raise RuntimeError(f"InternVL hf_device_map contains non-cuda:0 placement: {bad_map}")
    bad_parameters = [
        f"{name}={parameter.device}"
        for name, parameter in model.named_parameters()
        if parameter.device.type != "cuda" or parameter.device.index not in (None, 0)
    ]
    if bad_parameters:
        raise RuntimeError("InternVL parameters are not fully on cuda:0: " + ", ".join(bad_parameters[:20]))
    return {"hf_device_map": {name: str(device) for name, device in device_map.items()},
            "offloaded_parameter_count": 0}
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
    is_hf_candidate = model_id in INTERNVL35_HF_CANDIDATES
    is_gguf_candidate = model_id in INTERNVL35_GGUF_CANDIDATES
    is_llama_candidate = model_id == LLAMA_VISION_ID
    if is_llama_candidate:
        snapshot = LLAMA_VISION_SNAPSHOT
    elif is_hf_candidate:
        snapshot = INTERNVL35_CANDIDATE_ROOT / INTERNVL35_HF_CANDIDATES[model_id]
    elif is_gguf_candidate:
        snapshot = INTERNVL35_CANDIDATE_ROOT / INTERNVL35_GGUF_CANDIDATES[model_id]["directory"]
    else:
        cache = HUB / f"models--{model_id.replace('/', '--')}"
        snapshot = resolve_snapshot(cache)
    config: dict[str, Any] = {}
    reason = None
    weight_bytes = 0
    model_weight_bytes = 0
    auxiliary_weight_bytes = 0
    revision = None
    if not snapshot or not snapshot.is_dir():
        reason = "cached snapshot or config.json is missing"
    else:
        if model_id == INTERNVL_ID and snapshot.name != INTERNVL_REVISION:
            reason = f"cached InternVL snapshot is not pinned revision {INTERNVL_REVISION}"
        if is_hf_candidate or is_gguf_candidate:
            manifest_path = INTERNVL35_CANDIDATE_ROOT / "download_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                repo_id = (
                    model_id
                    if is_hf_candidate
                    else INTERNVL35_GGUF_CANDIDATES[model_id]["repo_id"]
                )
                revision = manifest["huggingface_revisions"][repo_id]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = f"candidate download manifest is missing or invalid: {exc}"
        elif is_llama_candidate:
            try:
                manifest = json.loads(
                    (LLAMA_VISION_ROOT / "download_manifest.json").read_text(encoding="utf-8")
                )
                revision = manifest["revision"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = f"Llama Vision download manifest is missing or invalid: {exc}"
        if is_gguf_candidate:
            spec = INTERNVL35_GGUF_CANDIDATES[model_id]
            filenames = (spec["model"], spec["mmproj"])
            missing = [name for name in filenames if not (snapshot / name).is_file()]
            if missing:
                reason = "GGUF candidate is missing: " + ", ".join(missing)
            else:
                model_weight_bytes = (snapshot / spec["model"]).stat().st_size
                auxiliary_weight_bytes = (snapshot / spec["mmproj"]).stat().st_size
                weight_bytes = model_weight_bytes + auxiliary_weight_bytes
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
        "model_weight_bytes": model_weight_bytes or weight_bytes,
        "auxiliary_weight_bytes": auxiliary_weight_bytes,
        "cache_status": "ready" if reason is None else "incomplete",
        "cache_error": reason,
        "revision": revision,
        "backend": (
            "Transformers (official InternVL3.5 HF-format checkpoint)"
            if is_hf_candidate else (
                "llama.cpp CUDA (GGUF plus matching mmproj)"
                if is_gguf_candidate else (
                    "Transformers (Llama 3.2 Vision BNB 4-bit)"
                    if is_llama_candidate else "Transformers (local cached checkpoint)"
                )
            )
        ),
    }


def adapter_input_tokens(model_id: str, model, processor, row: dict[str, Any], task: str, prompt: str) -> int:
    """Measure input length through the same adapter/model image path before bounded generation."""
    spec = adapter_prompt_for(model_id, task, prompt)
    if model_id in INTERNVL35_GGUF_CANDIDATES:
        _, input_tokens, _, _ = model.generate(row["image"], spec.text, max_new_tokens=1)
        return int(input_tokens)
    if model_id == PALIGEMMA_ID:
        inputs = processor(images=row["image"].convert("RGB"), text=spec.text, return_tensors="pt")
    elif model_id == LLAMA_VISION_ID:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": spec.text},
        ]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=rendered, images=row["image"].convert("RGB"), return_tensors="pt")
    elif model_id in INTERNVL_GPU_IDS:
        raise ResourceLimitError("InternVL HF adapter cannot measure multimodal input tokens reliably")
    else:
        raise ResourceLimitError(f"no Edge context enforcement adapter for {model_id}")
    return int(inputs.input_ids.shape[-1])


def generate_with_output_limit(model_id: str, model, processor, row: dict[str, Any], task: str,
                               prompt: str, max_new_tokens: int) -> dict[str, Any]:
    """Run the existing model adapter with an Edge-derived output-token ceiling."""
    spec = adapter_prompt_for(model_id, task, prompt)
    if model_id in INTERNVL35_GGUF_CANDIDATES:
        answer, input_tokens, output_tokens, details = model.generate(
            row["image"], spec.text, max_new_tokens=max_new_tokens
        )
    elif model_id == PALIGEMMA_ID:
        answer, input_tokens, output_tokens, details = generate_paligemma(
            model, processor, row["image"], spec, max_new_tokens=max_new_tokens
        )
    elif model_id == LLAMA_VISION_ID:
        answer, input_tokens, output_tokens, details = generate_llama_vision(
            model, processor, row["image"], spec, max_new_tokens=max_new_tokens
        )
    elif model_id in INTERNVL_GPU_IDS:
        answer, input_tokens, output_tokens, details = generate_internvl(
            model, processor, row["image"], spec, max_new_tokens=max_new_tokens
        )
    else:
        raise ResourceLimitError(f"no Edge context enforcement adapter for {model_id}")
    return {
        "prompt": spec.text, "prompt_prefix": spec.prefix, "answer": answer,
        "input_tokens": int(input_tokens), "output_tokens": int(output_tokens),
        "postprocess": details, "scoring_eligible": spec.scoring_eligible,
        "route_note": spec.route_note,
    }


def generation(model_id: str, model, processor, row: dict[str, Any], task: str, prompt: str,
               *, route_validated: bool = False, context_limit_tokens: int | None = None):
    import torch

    is_gguf = model_id in INTERNVL35_GGUF_CANDIDATES
    if not is_gguf:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    max_new_tokens = None
    measured_input_tokens = None
    if context_limit_tokens is not None:
        measured_input_tokens = adapter_input_tokens(model_id, model, processor, row, task, prompt)
        remaining = context_limit_tokens - measured_input_tokens
        if remaining <= 0:
            raise ResourceLimitError(
                f"input length {measured_input_tokens} reaches the probed {context_limit_tokens}-token context"
            )
        max_new_tokens = min(256, remaining)
    started = time.perf_counter()
    with torch.inference_mode():
        if max_new_tokens is None:
            record = generate(model_id, model, processor, row["image"], task, prompt)
        else:
            record = generate_with_output_limit(
                model_id, model, processor, row, task, prompt, max_new_tokens
            )
    if model_id == FLORENCE_ID and route_validated:
        record["scoring_eligible"] = True
        record["route_note"] = "experimental <VQA> route passed AI2D and TextVQA real-image preflight"
    record["generation_s"] = round(time.perf_counter() - started, 4)
    if context_limit_tokens is not None:
        actual_context_tokens = record["input_tokens"] + record["output_tokens"]
        if actual_context_tokens > context_limit_tokens:
            raise ResourceLimitError(
                f"adapter generated {actual_context_tokens} tokens beyond the probed "
                f"{context_limit_tokens}-token context"
            )
        record.update(
            context_limit_tokens=context_limit_tokens,
            generation_max_new_tokens=max_new_tokens,
            measured_input_tokens=measured_input_tokens,
            actual_context_tokens=actual_context_tokens,
            context_limit_enforced=True,
        )
    if is_gguf:
        record["peak_allocated_gib"] = ACTIVE_GPU_MONITOR.peak_gib if ACTIVE_GPU_MONITOR else None
        record["peak_reserved_gib"] = None
        record["peak_memory_method"] = "nvidia-smi sampled llama-server process VRAM"
    else:
        torch.cuda.synchronize()
        record["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        record["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1024**3), 3)
    if model_id in INTERNVL_GPU_IDS or model_id == LLAMA_VISION_ID:
        record["sampled_process_peak_gib"] = ACTIVE_GPU_MONITOR.peak_gib if ACTIVE_GPU_MONITOR else None
        if not is_gguf and record["peak_allocated_gib"] > INTERNVL_ALLOCATED_TARGET_GIB:
            raise ResourceLimitError(
                f"Transformer model allocated peak {record['peak_allocated_gib']:.3f} GiB exceeded "
                f"the {INTERNVL_ALLOCATED_TARGET_GIB:.1f} GiB target"
            )
        if ACTIVE_GPU_MONITOR and ACTIVE_GPU_MONITOR.peak_gib is not None \
                and ACTIVE_GPU_MONITOR.peak_gib >= MEMORY_CAP_GIB:
            raise ResourceLimitError(
                f"sampled nvidia-smi process peak {ACTIVE_GPU_MONITOR.peak_gib:.3f} GiB "
                f"reached the {MEMORY_CAP_GIB:.1f} GiB hard cap"
            )
    return record


def preflight(model_id: str, model, processor, seed: int,
              context_limit_tokens: int | None = None) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    """Prove real-image handling and both answer formats before any full task runs."""
    checks: dict[str, Any] = {}
    cached: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in TASKS}
    for task in ("ai2d", "textvqa"):
        dataset = shared.dataset_for(task)
        indices = common.pick_indices(len(dataset), TASK_SIZES[task], seed)
        index = indices[0]
        row = dataset[index]
        prompt, expected = shared.prompt_for(task, row)
        output = generation(model_id, model, processor, row, task, prompt,
                             context_limit_tokens=context_limit_tokens)
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
             cached: dict[int, dict[str, Any]],
             context_limit_tokens: int | None = None) -> dict[str, Any]:
    import torch

    dataset = shared.dataset_for(task)
    indices = common.pick_indices(len(dataset), count, seed)
    items: list[dict[str, Any]] = []
    scores: list[float] = []
    total_tokens = 0
    total_generation_s = 0.0
    resource_limit_exceeded = False
    for position, index in enumerate(indices):
        row = dataset[index]
        prompt, expected = shared.prompt_for(task, row)
        try:
            reused = cached.get(index)
            output = reused["generation"] if reused else generation(
                model_id, model, processor, row, task, prompt, route_validated=True,
                context_limit_tokens=context_limit_tokens,
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
                "context_limit_tokens": output.get("context_limit_tokens"),
                "generation_max_new_tokens": output.get("generation_max_new_tokens"),
                "actual_context_tokens": output.get("actual_context_tokens"),
                "context_limit_enforced": output.get("context_limit_enforced", False),
                "parse_error": parse_error,
                "generation_s": output["generation_s"],
                "peak_allocated_gib": output["peak_allocated_gib"],
                "peak_reserved_gib": output["peak_reserved_gib"],
            }
            if model_id in INTERNVL_GPU_IDS:
                item["sampled_process_peak_gib"] = output.get("sampled_process_peak_gib")
                item["image_tile_count"] = (
                    output["postprocess"].get("image_tile_count")
                    if output.get("postprocess") is not None else None
                )
            if output["postprocess"] is not None:
                item["postprocess"] = output["postprocess"]
            items.append(item)
            scores.append(score)
            total_tokens += output["output_tokens"]
            total_generation_s += output["generation_s"]
        except Exception as exc:
            items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            if isinstance(exc, ResourceLimitError):
                resource_limit_exceeded = True
        finally:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(f"PROGRESS {model_id} {task} {position + 1}/{len(indices)}", flush=True)
        if resource_limit_exceeded:
            for unrun_index in indices[position + 1:]:
                items.append({
                    "index": unrun_index,
                    "error": "not_run_resource_limit",
                    "detail": "execution stopped after an enforced InternVL memory limit",
                })
            break
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
        "peak_allocated_gib": max(
            (i.get("peak_allocated_gib") for i in successful if i.get("peak_allocated_gib") is not None),
            default=None,
        ),
        "peak_reserved_gib": max(
            (i.get("peak_reserved_gib") for i in successful if i.get("peak_reserved_gib") is not None),
            default=None,
        ),
        "sampled_process_peak_gib": max(
            (i["sampled_process_peak_gib"] for i in successful if i.get("sampled_process_peak_gib") is not None),
            default=None,
        ),
        "resource_limit_exceeded": resource_limit_exceeded,
        "items": items,
    }


def gpu_memory_used_mib() -> int | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        ).strip().splitlines()[0]
        return int(value)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def probe_context(model_id: str, model, processor, task: str, seed: int,
                  context_limit: int, cap_gib: float,
                  configured_context_tokens: int | None = None) -> dict[str, Any]:
    """Measure actual real-image input length and whole-device peak via the model adapter."""
    import torch

    dataset = shared.dataset_for(task)
    indices = common.pick_indices(len(dataset), min(100, len(dataset)), seed)
    rows = []

    def generate_probe(row, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        spec = adapter_prompt_for(model_id, task, prompt)
        if model_id in INTERNVL35_GGUF_CANDIDATES:
            answer, input_tokens, output_tokens, details = model.generate(
                row["image"], spec.text, max_new_tokens=max_new_tokens
            )
        elif model_id == PALIGEMMA_ID:
            answer, input_tokens, output_tokens, details = generate_paligemma(
                model, processor, row["image"], spec, max_new_tokens=max_new_tokens
            )
        elif model_id == LLAMA_VISION_ID:
            answer, input_tokens, output_tokens, details = generate_llama_vision(
                model, processor, row["image"], spec, max_new_tokens=max_new_tokens
            )
        elif model_id in INTERNVL_GPU_IDS:
            answer, input_tokens, output_tokens, details = generate_internvl(
                model, processor, row["image"], spec, max_new_tokens=max_new_tokens
            )
        else:
            return {"error": f"no measured Edge context-probe adapter for {model_id}"}
        reliable_tokens = not (
            model_id in INTERNVL_GPU_IDS and model_id not in INTERNVL35_GGUF_CANDIDATES
        )
        return {
            "answer": answer, "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens), "details": details,
            "visual_token_count": next((
                int(details[key]) for key in (
                    "visual_token_count", "image_visual_token_count", "image_token_count",
                ) if isinstance(details, dict) and isinstance(details.get(key), (int, float))
            ), None),
            "token_count_reliable": reliable_tokens,
        }

    selection_peak_gib = None
    selection_attempts = []
    for index in indices:
        row = dataset[index]
        prompt, _ = shared.prompt_for(task, row)
        attempt_memory = WholeDeviceMemorySampler()
        attempt_memory.start()
        before = gpu_memory_used_mib()
        try:
            measured = generate_probe(row, prompt, 1)
        except Exception as exc:
            measured = {"error": f"{type(exc).__name__}: {exc}"}
        after = gpu_memory_used_mib()
        attempt_memory.stop()
        measured_peak = max(
            (value for value in (before, after, attempt_memory.peak_mib if attempt_memory.sample_count else None)
             if value is not None), default=None
        )
        peak_gib = measured_peak / 1024 if measured_peak is not None else None
        if peak_gib is not None:
            selection_peak_gib = max(selection_peak_gib or 0, peak_gib)
        sampler_valid = attempt_memory.sample_count > 0 and attempt_memory.error is None
        selection_attempts.append({
            "image_index": index,
            "sample_count": attempt_memory.sample_count,
            "sampler_error": attempt_memory.error,
            "measurement_error": measured.get("error"),
            "peak_whole_device_gib": peak_gib,
            "sampler_verified": sampler_valid,
        })
        if "input_tokens" in measured:
            rows.append({"index": index, "row": row, "prompt": prompt,
                         "input_tokens": measured["input_tokens"], "peak_gib": peak_gib,
                         "sampler_sample_count": attempt_memory.sample_count,
                         "sampler_error": attempt_memory.error,
                         "sampler_verified": sampler_valid,
                         "visual_token_count": measured.get("visual_token_count"),
                         "token_count_reliable": measured["token_count_reliable"]})
    selection_complete = (
        len(indices) == min(100, len(dataset))
        and len(rows) == len(indices)
        and all(item["sampler_verified"] for item in rows)
        and all(item["token_count_reliable"] for item in rows)
    )
    if not rows:
        return {"passed": False, "error": "no sampled AI2D image produced measurable input tokens",
                "image_sample_count": len(indices), "image_selection_complete": False,
                "image_selection_attempts": selection_attempts,
                "image_selection_peak_whole_device_gib": selection_peak_gib}
    visual_rows = [item for item in rows if item.get("visual_token_count") is not None]
    use_visual_counts = selection_complete and len(visual_rows) == len(indices)
    selected = max(visual_rows if use_visual_counts else rows, key=lambda item: (
        item["visual_token_count"] if use_visual_counts else item["input_tokens"]
    ))
    row, prompt = selected["row"], selected["prompt"]
    context_limit = min(context_limit, int(getattr(
        getattr(model.config, "text_config", model.config), "max_position_embeddings", context_limit
    ) or context_limit))
    low, high = 0, max(1, context_limit * 2)
    best = None
    attempts = []
    while low <= high:
        repeats = (low + high) // 2
        full_prompt = prompt + (" context" * repeats)
        attempt_memory = WholeDeviceMemorySampler()
        attempt_memory.start()
        before = gpu_memory_used_mib()
        try:
            measured = generate_probe(row, full_prompt, 8)
        except Exception as exc:
            measured = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        after = gpu_memory_used_mib()
        attempt_memory.stop()
        measured_peak = max(
            (value for value in (before, after, attempt_memory.peak_mib if attempt_memory.sample_count else None)
             if value is not None), default=None
        )
        peak_gib = measured_peak / 1024 if measured_peak is not None else None
        sampler_valid = attempt_memory.sample_count > 0 and attempt_memory.error is None
        input_tokens = measured.get("input_tokens")
        output_tokens = measured.get("output_tokens")
        within_context = (input_tokens is not None and output_tokens is not None
                          and input_tokens + output_tokens <= context_limit)
        passed = bool(
            input_tokens is not None and measured.get("token_count_reliable")
            and within_context and sampler_valid and peak_gib is not None and peak_gib <= cap_gib
        )
        attempt = {
            "filler_repetitions": repeats, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "peak_whole_device_gib": peak_gib,
            "passed": passed, "error": measured.get("error"),
            "token_count_reliable": measured.get("token_count_reliable", False),
            "whole_device_sample_count": attempt_memory.sample_count,
            "whole_device_sampler_error": attempt_memory.error,
            "whole_device_sampler_verified": sampler_valid,
        }
        attempts.append(attempt)
        if passed:
            best = attempt
            low = repeats + 1
        else:
            high = repeats - 1
    image = row["image"]
    attempt_peaks = [item.get("peak_whole_device_gib") for item in attempts
                     if isinstance(item.get("peak_whole_device_gib"), (int, float))]
    measured_probe_peak = max(
        ([selection_peak_gib] if selection_peak_gib is not None else []) + attempt_peaks,
        default=None,
    )
    all_probe_attempts_measured = all(
        isinstance(item.get("peak_whole_device_gib"), (int, float))
        and item.get("whole_device_sample_count", 0) > 0
        and item.get("whole_device_sampler_error") is None
        and item.get("whole_device_sampler_verified") is True
        for item in attempts
    )
    all_probe_attempts_within_cap = all(
        item["peak_whole_device_gib"] <= cap_gib
        for item in attempts
        if isinstance(item.get("peak_whole_device_gib"), (int, float))
    )
    probe_passed = bool(
        best and selection_complete and selected["token_count_reliable"] and measured_probe_peak is not None
        and measured_probe_peak <= cap_gib
        and all_probe_attempts_measured and all_probe_attempts_within_cap
    )
    return {
        "passed": probe_passed,
        "image_dataset": "AI2D test", "image_index": selected["index"],
        "image_size_px": list(image.size),
        "image_selection_method": (
            "maximum adapter-reported visual-token count across fixed AI2D examples"
            if use_visual_counts else
            "maximum measured total input-token proxy across fixed AI2D examples; visual-token counts unavailable for the full sample"
        ),
        "image_selection_sample_count": len(indices),
        "image_selection_complete": selection_complete,
        "image_selection_verified_sample_count": len(rows),
        "image_selection_attempts": selection_attempts,
        "image_visual_token_count": selected.get("visual_token_count") if use_visual_counts else None,
        "image_selection_total_input_token_proxy": selected["input_tokens"],
        "image_selection_token_measure": "visual_token_count" if use_visual_counts else "total_input_token_proxy",
        "configured_context_tokens": configured_context_tokens,
        "probe_search_limit_tokens": context_limit,
        "measured_context_tokens": (best["input_tokens"] + best["output_tokens"]) if best else None,
        "actual_input_tokens": best["input_tokens"] if best else None,
        "output_tokens": best["output_tokens"] if best else None,
        "configured_limit_reached": bool(best and best["input_tokens"] + best["output_tokens"] >= context_limit - 32),
        "token_count_reliable": selected["token_count_reliable"],
        "whole_device_cap_gib": cap_gib,
        "peak_whole_device_gib": measured_probe_peak,
        "image_selection_peak_whole_device_gib": selection_peak_gib,
        "all_probe_attempts_measured": all_probe_attempts_measured,
        "all_probe_attempts_within_cap": all_probe_attempts_within_cap,
        "search_method": "binary search on deterministic filler repetitions using adapter-reported token counts",
        "attempts": attempts,
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
              "", "| Task | Mean generation seconds | Output tokens/second | Peak allocated GiB | Peak reserved GiB | Sampled process peak GiB | Parse errors |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        data = model.get("tasks", {}).get(task, {})
        fmt = lambda value: "—" if value is None else f"{value:.2f}"
        lines.append(
            f"| {LABELS[task]} | {fmt(data.get('mean_generation_s'))} | "
            f"{fmt(data.get('generated_tokens_per_s'))} | {fmt(data.get('peak_allocated_gib'))} | "
            f"{fmt(data.get('peak_reserved_gib'))} | {fmt(data.get('sampled_process_peak_gib'))} | "
            f"{data.get('parse_errors', 0)} |"
        )
    lines += ["", "## Method and provenance", "",
              "Prompts, prepared samples, and scorers are shared with the Qwen-VL suite. Requests use one image, no batching, deterministic decoding, and at most 256 new tokens.",
              "See `../../qwenVL_Bench/README.md` for datasets, splits, fixed sampling rules, prompt contracts, and scorer definitions.",
              f"Cache snapshot: `{model.get('snapshot') or 'missing'}`.",
              f"Cache status: {model.get('cache_status')}; checkpoint weight files: {model.get('weight_gib', 0):.2f} GiB.",
              "Raw per-item outputs and errors are in `results.json`.", ""]
    if model.get("model_id") in INTERNVL_GPU_IDS:
        quantization = model.get("quantization") or "bfloat16 full checkpoint weights"
        lines.insert(
            -2,
            f"InternVL uses pinned revision `{model.get('revision')}`, {quantization}, official custom chat, and 448 px dynamic tiling with max_num=12 plus thumbnail. Per-item tile counts are stored in `results.json`.",
        )
        lines.insert(
            lines.index("", lines.index("## Runtime and resources")),
            f"Load peak allocated/reserved: {model.get('load_peak_allocated_gib')} / {model.get('load_peak_reserved_gib')} GiB; sampled process peak: {model.get('sampled_process_peak_gib')} GiB ({model.get('sampled_process_peak_method')}).",
        )
        lines.insert(
            lines.index("", lines.index("## Runtime and resources")) + 1,
            (
                f"The {INTERNVL_ALLOCATED_TARGET_GIB:.2f} GiB process allowance applies to CUDA allocation. "
                f"The {MEMORY_CAP_GIB:.1f} GiB Edge cap uses whole-device nvidia-smi sampling."
                if result.get("whole_device_cap_gib") is not None else
                "The 26 GiB target applies to torch.cuda.max_memory_allocated. The 30 GiB hard cap uses sampled per-process nvidia-smi memory."
            ),
        )
    if model.get("model_id") == FLORENCE_ID:
        lines.insert(
            -2,
            "For AI2D and negation choice questions, an exact generated option-text match is mapped to its displayed label before shared scoring. The raw generated text is retained in each item's post-processing record.",
        )
    elif model.get("model_id") == MOLMO2_ID:
        lines.insert(
            -2,
            "Molmo2 uses its processor chat template with the unchanged shared question and decodes only the generated completion. The cached weights are loaded in bfloat16.",
        )
    if result.get("whole_device_cap_gib") is not None:
        probe = model.get("context_probe") or {}
        lines += [
            "## Edge cap verification", "",
            f"Whole-device cap: {result['whole_device_cap_gib']:.1f} GiB; idle baseline: {result.get('gpu_baseline_gib')} GiB.",
            f"Whole-device peak immediately after model load: {model.get('load_peak_whole_device_gib')} GiB.",
            f"Whole-device peak from model load through scoring: {model.get('whole_device_peak_gib')} GiB.",
            f"Scoring peak across completed tasks: {model.get('scoring_peak_gpu_used_gib')} GiB.",
            f"Cap verified: `{model.get('cap_verified', False)}`; verification: `{model.get('cap_verification', {})}`.",
            "", "### Real-image context probe", "",
            f"Passed: `{probe.get('passed', False)}`; AI2D index: {probe.get('image_index', 'unknown')}; "
            f"image: {probe.get('image_size_px', 'unknown')} px; measured context: "
            f"{probe.get('measured_context_tokens', 'unknown')} tokens; input/output: "
            f"{probe.get('actual_input_tokens', 'unknown')}/{probe.get('output_tokens', 'unknown')} tokens; "
            f"probe peak: {probe.get('peak_whole_device_gib', 'unknown')} GiB.",
            f"Image selection: {probe.get('image_selection_method', 'unknown')}; "
            f"configured limit reached: `{probe.get('configured_limit_reached', False)}`.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global ACTIVE_GPU_MONITOR, ACTIVE_WHOLE_GPU_MONITOR, MEMORY_CAP_GIB, INTERNVL_ALLOCATED_TARGET_GIB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--whole-device-cap-gib", type=float, choices=(8, 10, 12, 14, 16),
                        help="enable Edge mode at this whole-device VRAM ceiling")
    parser.add_argument("--context-probe-limit", type=int, default=32768,
                        help="largest context tested by the real-image Edge context probe")
    args = parser.parse_args()
    if args.context_probe_limit < 1024:
        parser.error("--context-probe-limit must be at least 1024 tokens")
    edge_mode = args.whole_device_cap_gib is not None
    if edge_mode:
        MEMORY_CAP_GIB = float(args.whole_device_cap_gib)
    model_id = args.model
    output_dir = args.output_dir or HERE / "results" / model_id.split("/")[-1].lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    transformers_version = version("transformers")
    import torch

    entry = cache_entry(model_id)
    loaded_weight_estimate_gib = entry["weight_bytes"] / (1024**3)
    loaded_weight_estimate_method = "cached checkpoint file size"
    if model_id == MOLMO2_ID:
        # Molmo2's cached checkpoint is FP32, while this adapter explicitly
        # casts its weights to BF16 during loading.
        loaded_weight_estimate_gib /= 2
        loaded_weight_estimate_method = "FP32 cached weights divided by two for configured BF16 loading"
    if model_id == INTERNVL_ID:
        loaded_weight_estimate_gib = None
        loaded_weight_estimate_method = (
            "not estimated; NF4 loading includes non-quantized modules; see measured CUDA and sampled process peaks"
        )
    package_versions = {}
    for package in ("transformers", "bitsandbytes", "flash-attn", "torchvision", "timm"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    prompt_routes = {
        MOLMO2_ID: "Molmo2 processor chat template",
        PALIGEMMA_ID: "PaliGemma answer en",
        FLORENCE_ID: "experimental Florence <VQA>",
        INTERNVL_ID: "InternVL custom chat API",
    }
    precision_routes = {
        MOLMO2_ID: "bfloat16",
        PALIGEMMA_ID: "bfloat16",
        FLORENCE_ID: "float16",
        INTERNVL_ID: "bfloat16 compute; NF4 double-quantized linear weights",
    }
    attention_routes = {
        MOLMO2_ID: "model default",
        PALIGEMMA_ID: "eager",
        FLORENCE_ID: "eager",
        INTERNVL_ID: "custom InternVL implementation default",
        LLAMA_VISION_ID: "Transformers SDPA (Mllama vision compatibility)",
    }
    prompt_routes[LLAMA_VISION_ID] = "Mllama image-text conversation template"
    precision_routes[LLAMA_VISION_ID] = "Unsloth bitsandbytes 4-bit checkpoint with bfloat16 compute"
    for candidate_id in INTERNVL35_CANDIDATE_IDS:
        prompt_routes[candidate_id] = "InternVL3.5 official custom chat API"
        precision_routes[candidate_id] = "bfloat16 full checkpoint weights; no quantization"
        attention_routes[candidate_id] = "official custom model with FlashAttention enabled"
    for candidate_id, spec in INTERNVL35_GGUF_CANDIDATES.items():
        quant = candidate_id.rsplit(":", 1)[-1]
        prompt_routes[candidate_id] = "llama.cpp multimodal chat with the GGUF model template"
        precision_routes[candidate_id] = f"{quant} GGUF; matching vision projector: {spec['mmproj']}"
        attention_routes[candidate_id] = "llama.cpp CUDA backend with FlashAttention enabled"
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
            "revision": entry.get("revision"),
            "cache_status": entry["cache_status"],
            "cache_error": entry["cache_error"],
            "weight_bytes": entry["weight_bytes"],
            "model_weight_bytes": entry.get("model_weight_bytes", entry["weight_bytes"]),
            "auxiliary_weight_bytes": entry.get("auxiliary_weight_bytes", 0),
            "weight_gib": entry["weight_bytes"] / (1024**3),
            "estimated_loaded_weight_gib": loaded_weight_estimate_gib,
            "loaded_weight_estimate_method": loaded_weight_estimate_method,
            "runtime_versions": {
                "python": sys.version.split()[0],
                "transformers": transformers_version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            "runtime_packages": package_versions,
            "prompt_route": prompt_routes[model_id],
            "precision": precision_routes[model_id],
            "attention_backend": attention_routes[model_id],
            "tasks": {},
            "status": "not_started",
        },
        "preflight": {"passed": False, "required": ["ai2d", "textvqa"], "checks": {}},
    }
    if edge_mode:
        result.update(
            whole_device_cap_gib=MEMORY_CAP_GIB,
            gpu_baseline_gib=None,
            hardware={},
            runtime_versions={"python": sys.version.split()[0], **result["model"]["runtime_versions"]},
            cap_enforcement_method=(
                "torch per-process CUDA memory fraction for Transformers; llama.cpp fit-target margin for GGUF; "
                "nvidia-smi whole-device peak sampled throughout load, probe, and scoring"
            ),
            models=[result["model"]],
        )
        result["model"]["whole_device_cap_gib"] = MEMORY_CAP_GIB
    if entry["cache_status"] != "ready":
        result["model"].update(status="failed", error=entry["cache_error"])
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    if model_id != INTERNVL_ID and loaded_weight_estimate_gib >= MEMORY_CAP_GIB:
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
    baseline_mib = gpu_memory_used_mib() if edge_mode else None
    gpu_name = torch.cuda.get_device_name(device)
    driver_version = None
    total_mib = int(total_memory / (1024**2))
    if edge_mode:
        if baseline_mib is None:
            result["model"].update(status="failed", error="could not measure whole-device GPU baseline")
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            write_report(output_dir / "report.md", result)
            return 2
        if baseline_mib / 1024 >= MEMORY_CAP_GIB:
            result["model"].update(status="unavailable_baseline", error="idle whole-device GPU baseline reaches the selected cap")
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            write_report(output_dir / "report.md", result)
            return 2
        try:
            gpu_name, driver_version = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=3,
            ).strip().splitlines()[0].split(", ", 1)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            result["model"].update(status="failed", error="could not record GPU model and driver version")
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            write_report(output_dir / "report.md", result)
            return 2
        process_allowance_gib = max(0.0, MEMORY_CAP_GIB - baseline_mib / 1024 - 0.5)
        if process_allowance_gib <= 0:
            result["model"].update(status="unavailable_baseline", error="baseline leaves no safe model memory allowance")
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            write_report(output_dir / "report.md", result)
            return 2
        INTERNVL_ALLOCATED_TARGET_GIB = min(INTERNVL_ALLOCATED_TARGET_GIB, process_allowance_gib)
        result.update(
            gpu_baseline_gib=round(baseline_mib / 1024, 3),
            hardware={"gpu_name": gpu_name, "driver_version": driver_version},
        )
        result["model"].update(
            whole_device_cap_gib=MEMORY_CAP_GIB,
            gpu_baseline_gib=round(baseline_mib / 1024, 3),
            process_allocator_allowance_gib=round(process_allowance_gib, 3),
        )
        result["memory_cap_gib"] = MEMORY_CAP_GIB
    if total_memory < MEMORY_CAP_GIB * 1024**3:
        result["model"].update(status="failed", error="the configured 30 GiB ceiling exceeds installed GPU memory")
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    free_memory, _ = torch.cuda.mem_get_info(device)
    process_allowance_gib = (
        max(0.0, MEMORY_CAP_GIB - baseline_mib / 1024 - 0.5) if edge_mode else MEMORY_CAP_GIB
    )
    if edge_mode and model_id != INTERNVL_ID and loaded_weight_estimate_gib >= process_allowance_gib:
        result["model"].update(
            status="unavailable_cap",
            error=(f"cached weights {loaded_weight_estimate_gib:.2f} GiB reach the conservative "
                   f"{process_allowance_gib:.2f} GiB process allowance"),
        )
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    required_internvl_free_gib = (
        process_allowance_gib if edge_mode else (
            24.0 if model_id in INTERNVL35_GGUF_CANDIDATES else INTERNVL_ALLOCATED_TARGET_GIB
        )
    )
    if (model_id in INTERNVL_GPU_IDS or model_id == LLAMA_VISION_ID) \
            and free_memory < required_internvl_free_gib * 1024**3:
        result["model"].update(
            status="failed",
            error=(f"free CUDA memory {free_memory / (1024**3):.2f} GiB is below the "
                   f"{required_internvl_free_gib:.1f} GiB InternVL loading budget"),
        )
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    allocator_limit_gib = process_allowance_gib if edge_mode else MEMORY_CAP_GIB
    torch.cuda.set_per_process_memory_fraction(allocator_limit_gib * 1024**3 / total_memory, device)

    model = processor = None
    load_monitor = None
    whole_device_monitor = WholeDeviceMemorySampler() if edge_mode else None
    if whole_device_monitor is not None:
        whole_device_monitor.start()
        ACTIVE_WHOLE_GPU_MONITOR = whole_device_monitor
    try:
        max_memory = {device: int(allocator_limit_gib * 1024**3)}
        if edge_mode and model_id in INTERNVL35_GGUF_CANDIDATES:
            # llama.cpp reads these before CUDA allocation; fit-target is the free-memory margin.
            margin_mib = max(512, total_mib - int(allocator_limit_gib * 1024))
            os.environ["LLAMA_ARG_FIT"] = "on"
            os.environ["LLAMA_ARG_FIT_TARGET"] = str(margin_mib)
        print(f"LOAD {model_id} snapshot={entry['snapshot']}", flush=True)
        if model_id in INTERNVL35_GGUF_CANDIDATES:
            gguf_spec = INTERNVL35_GGUF_CANDIDATES[model_id]
            candidate_dir = INTERNVL35_CANDIDATE_ROOT / gguf_spec["directory"]
            gguf_index = INTERNVL35_GGUF_IDS.index(model_id)
            model = load_llama_cpp_server(
                candidate_dir / gguf_spec["model"],
                candidate_dir / gguf_spec["mmproj"],
                output_dir / "llama-server.log",
                port=18135 + gguf_index,
                context_size=max(8192, args.context_probe_limit) if edge_mode else 8192,
            )
            load_monitor = SampledProcessMemory(model.process.pid)
            ACTIVE_GPU_MONITOR = load_monitor
            load_monitor.start()
            model.wait_ready()
            gpu_layers, total_layers = model.require_full_gpu_offload()
            result["model"]["placement"] = {
                "llama_cpp_gpu_layers": gpu_layers,
                "llama_cpp_total_layers": total_layers,
                "full_gpu_offload": True,
            }
            result["model"]["load_peak_allocated_gib"] = None
            result["model"]["load_peak_reserved_gib"] = None
            result["model"]["load_peak_process_gib"] = load_monitor.peak_gib
            result["model"]["revision"] = entry.get("revision")
            result["model"]["quantization"] = model_id.rsplit(":", 1)[-1]
            result["model"]["gpu_name"] = torch.cuda.get_device_name(device)
            result["model"]["sampled_process_memory_interval_s"] = load_monitor.interval_s
            result["model"]["sampled_process_peak_method"] = (
                "nvidia-smi query-compute-apps sampled llama-server every 0.5 seconds"
            )
            result["model"]["memory_metric"] = "sampled llama-server process VRAM"
            if load_monitor.sample_count == 0:
                raise RuntimeError("nvidia-smi did not return a llama-server process-memory sample")
            if load_monitor.peak_gib is not None and load_monitor.peak_gib >= MEMORY_CAP_GIB:
                raise ResourceLimitError(
                    f"llama-server load peak {load_monitor.peak_gib:.3f} GiB reached the "
                    f"{MEMORY_CAP_GIB:.1f} GiB hard cap"
                )
        elif model_id in INTERNVL_GPU_IDS:
            load_monitor = SampledProcessMemory(os.getpid())
            ACTIVE_GPU_MONITOR = load_monitor
            load_monitor.start()
            module_cache = HERE / "environments" / "internvl" / "modules-tf519-run"
            if model_id == INTERNVL_ID:
                model, processor = load_internvl(entry["snapshot"], module_cache=module_cache)
            else:
                model, processor = load_internvl35_hf(entry["snapshot"], module_cache=module_cache)
            result["model"]["placement"] = validate_internvl_placement(model)
            torch.cuda.synchronize()
            result["model"]["load_peak_allocated_gib"] = round(
                torch.cuda.max_memory_allocated(device) / (1024**3), 3
            )
            result["model"]["load_peak_reserved_gib"] = round(
                torch.cuda.max_memory_reserved(device) / (1024**3), 3
            )
            result["model"]["revision"] = entry.get("revision") or INTERNVL_REVISION
            result["model"]["quantization"] = (
                "bitsandbytes NF4 with double quantization"
                if model_id == INTERNVL_ID else "none; official BF16 checkpoint weights"
            )
            result["model"]["gpu_name"] = torch.cuda.get_device_name(device)
            result["model"]["memory_target_allocated_gib"] = INTERNVL_ALLOCATED_TARGET_GIB
            result["model"]["sampled_process_memory_interval_s"] = load_monitor.interval_s
            result["model"]["sampled_process_peak_method"] = "nvidia-smi query-compute-apps sampled every 0.5 seconds"
            if result["model"]["load_peak_allocated_gib"] > INTERNVL_ALLOCATED_TARGET_GIB:
                raise ResourceLimitError(
                    f"InternVL load allocated peak {result['model']['load_peak_allocated_gib']:.3f} GiB "
                    f"exceeded the {INTERNVL_ALLOCATED_TARGET_GIB:.1f} GiB target"
                )
            if load_monitor.sample_count == 0:
                raise RuntimeError("nvidia-smi did not return a process-memory sample for InternVL")
            if load_monitor.peak_gib is not None and load_monitor.peak_gib >= MEMORY_CAP_GIB:
                raise ResourceLimitError(
                    f"sampled nvidia-smi process peak {load_monitor.peak_gib:.3f} GiB "
                    f"reached the {MEMORY_CAP_GIB:.1f} GiB hard cap"
                )
        elif model_id == LLAMA_VISION_ID:
            load_monitor = SampledProcessMemory(os.getpid())
            ACTIVE_GPU_MONITOR = load_monitor
            load_monitor.start()
            model, processor = load_llama_vision(entry["snapshot"])
            result["model"]["placement"] = validate_internvl_placement(model)
            torch.cuda.synchronize()
            result["model"]["load_peak_allocated_gib"] = round(
                torch.cuda.max_memory_allocated(device) / (1024**3), 3
            )
            result["model"]["load_peak_reserved_gib"] = round(
                torch.cuda.max_memory_reserved(device) / (1024**3), 3
            )
            result["model"]["revision"] = entry.get("revision")
            result["model"]["quantization"] = "bitsandbytes 4-bit checkpoint"
            result["model"]["gpu_name"] = torch.cuda.get_device_name(device)
            result["model"]["memory_target_allocated_gib"] = INTERNVL_ALLOCATED_TARGET_GIB
            result["model"]["sampled_process_memory_interval_s"] = load_monitor.interval_s
            result["model"]["sampled_process_peak_method"] = (
                "nvidia-smi query-compute-apps sampled every 0.5 seconds"
            )
            if result["model"]["load_peak_allocated_gib"] > INTERNVL_ALLOCATED_TARGET_GIB:
                raise ResourceLimitError(
                    f"Llama Vision load allocated peak {result['model']['load_peak_allocated_gib']:.3f} GiB "
                    f"exceeded the {INTERNVL_ALLOCATED_TARGET_GIB:.1f} GiB target"
                )
            if load_monitor.sample_count == 0:
                raise RuntimeError("nvidia-smi did not return a process-memory sample for Llama Vision")
            if load_monitor.peak_gib is not None and load_monitor.peak_gib >= MEMORY_CAP_GIB:
                raise ResourceLimitError(
                    f"sampled nvidia-smi process peak {load_monitor.peak_gib:.3f} GiB "
                    f"reached the {MEMORY_CAP_GIB:.1f} GiB hard cap"
                )
        elif model_id == PALIGEMMA_ID:
            model, processor = load_paligemma(entry["snapshot"], max_memory=max_memory)
        elif model_id == MOLMO2_ID:
            model, processor = load_molmo2(entry["snapshot"], max_memory=max_memory)
        else:
            model, processor = load_florence(entry["snapshot"])
        if whole_device_monitor is not None:
            # Capture the measured sampler peak at the model-load boundary, before
            # context probing begins, so cap verification can audit load residency.
            result["model"]["load_peak_whole_device_gib"] = whole_device_monitor.peak_gib
        result["model"]["checkpoint_context"] = getattr(
            getattr(model.config, "text_config", model.config), "max_position_embeddings", None
        )
        result["model"]["configured_context"] = int(getattr(
            getattr(model.config, "text_config", model.config),
            "max_position_embeddings", getattr(model, "context_size", 0),
        ) or getattr(model, "context_size", 0) or 0)
        if model_id == PALIGEMMA_ID:
            result["model"]["image_resolution_px"] = 224
        if (model_id in INTERNVL_GPU_IDS and model_id not in INTERNVL35_GGUF_CANDIDATES) \
                or model_id == LLAMA_VISION_ID:
            torch.cuda.reset_peak_memory_stats(device)
        context_probe_ok = True
        if edge_mode:
            if whole_device_monitor is None or whole_device_monitor.sample_count == 0:
                raise RuntimeError("nvidia-smi did not produce a whole-device sample during model load")
            if whole_device_monitor.error is not None:
                raise RuntimeError(f"whole-device sampler failed during model load: {whole_device_monitor.error}")
            if whole_device_monitor.peak_gib is None or whole_device_monitor.peak_gib > MEMORY_CAP_GIB:
                raise ResourceLimitError("whole-device model-load peak exceeded the selected Edge cap")
            context_limit = args.context_probe_limit
            if model_id in INTERNVL35_GGUF_CANDIDATES:
                context_limit = min(context_limit, int(model.context_size))
            configured = int(getattr(model, "context_size", 0) or getattr(
                getattr(model.config, "text_config", model.config),
                "max_position_embeddings", context_limit,
            ) or context_limit)
            context_limit = min(context_limit, configured)
            result["model"]["context_probe"] = probe_context(
                model_id, model, processor, "ai2d", SEED, context_limit, MEMORY_CAP_GIB,
                configured_context_tokens=result["model"]["configured_context"],
            )
            probe = result["model"]["context_probe"]
            result["model"]["context_probe_max_context_tokens"] = probe.get("measured_context_tokens")
            result["model"]["scored_context_limit_tokens"] = probe.get("measured_context_tokens")
            result["model"]["scored_context_enforcement"] = (
                "per-generation max_new_tokens bounded by measured input tokens and probed context; actual totals checked"
            )
            result["model"]["whole_device_cap_gib"] = MEMORY_CAP_GIB
            context_probe_ok = bool(probe.get("passed"))
        if context_probe_ok:
            preflight_result, cached = preflight(
                model_id, model, processor, SEED,
                context_limit_tokens=(result["model"].get("scored_context_limit_tokens") if edge_mode else None),
            )
        else:
            preflight_result = {"passed": False, "required": ["ai2d", "textvqa"],
                                "checks": {}, "error": "context probe did not pass the cap and token-count gates"}
            cached = {task: {} for task in TASKS}
        result["preflight"] = preflight_result
        if not context_probe_ok:
            result["model"].update(
                status="context_probe_failed",
                error="real-image context probe did not pass cap and reliable token-count gates",
            )
        elif not preflight_result["passed"]:
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
                task_memory = WholeDeviceMemorySampler() if edge_mode else None
                if task_memory is not None:
                    task_memory.start()
                try:
                    task_result = run_task(
                        model_id, model, processor, task, TASK_SIZES[task], SEED, cached[task],
                        context_limit_tokens=(result["model"].get("scored_context_limit_tokens") if edge_mode else None),
                    )
                    result["model"]["tasks"][task] = task_result
                except Exception as exc:
                    task_result = {
                        "requested_n": TASK_SIZES[task], "completed_n": 0,
                        "failed_n": TASK_SIZES[task], "score": None,
                        "error": f"{type(exc).__name__}: {exc}", "items": [],
                    }
                    result["model"]["tasks"][task] = task_result
                finally:
                    if task_memory is not None:
                        task_memory.stop()
                        task_result["peak_gpu_used_gib"] = task_memory.peak_gib
                        task_result["whole_device_sample_count"] = task_memory.sample_count
                        task_result["whole_device_sample_error"] = task_memory.error
                if edge_mode and (
                    task_memory is None or task_memory.sample_count == 0
                    or task_memory.error is not None
                    or task_memory.peak_gib is None or task_memory.peak_gib > MEMORY_CAP_GIB
                ):
                    result["model"].update(
                        status="partial",
                        error=(f"whole-device peak could not be verified under the {MEMORY_CAP_GIB:.0f} GiB cap "
                               f"during {task}; sampler_error={task_memory.error if task_memory else 'missing sampler'}"),
                    )
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    break
                if result["model"]["tasks"][task].get("resource_limit_exceeded"):
                    result["model"].update(
                        status="partial",
                        error="InternVL exceeded an enforced memory limit; remaining tasks were stopped",
                    )
                    break
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            if result["model"]["status"] != "partial" or not result["model"].get("error"):
                result["model"]["status"] = "complete" if all(
                    result["model"]["tasks"].get(task, {}).get("completed_n") == TASK_SIZES[task]
                    for task in TASKS
                ) else "partial"
    except Exception as exc:
        result["model"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if model is not None and model_id in INTERNVL35_GGUF_CANDIDATES:
            try:
                model.close()
            except Exception as close_exc:
                result["model"]["server_shutdown_error"] = f"{type(close_exc).__name__}: {close_exc}"
        if load_monitor is not None:
            load_monitor.stop()
            result["model"]["sampled_process_peak_gib"] = load_monitor.peak_gib
            result["model"]["sampled_process_sample_count"] = load_monitor.sample_count
            result["model"]["sampled_process_monitor_error"] = load_monitor.error
            ACTIVE_GPU_MONITOR = None
        if model is not None:
            del model
        if processor is not None:
            del processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if whole_device_monitor is not None:
            whole_device_monitor.stop()
            ACTIVE_WHOLE_GPU_MONITOR = None
            result["model"]["whole_device_peak_gib"] = whole_device_monitor.peak_gib
            result["model"]["whole_device_sample_count"] = whole_device_monitor.sample_count
            result["model"]["whole_device_sample_error"] = whole_device_monitor.error
            scoring_peaks = [
                task.get("peak_gpu_used_gib") for task in result["model"].get("tasks", {}).values()
                if isinstance(task.get("peak_gpu_used_gib"), (int, float))
            ]
            scoring_peak = max(scoring_peaks) if scoring_peaks else None
            complete_suite = all(
                result["model"].get("tasks", {}).get(task, {}).get("completed_n") == TASK_SIZES[task]
                for task in TASKS
            )
            probe = result["model"].get("context_probe") or {}
            measured_context = probe.get("measured_context_tokens")
            suite_context_passed = bool(measured_context) and all(
                item.get("context_limit_enforced") is True
                and item.get("context_limit_tokens") == measured_context
                and (item.get("input_tokens", 0) + item.get("output_tokens", 0)) <= measured_context
                for task in result["model"].get("tasks", {}).values()
                for item in task.get("items", []) if "score" in item
            )
            sampled = (whole_device_monitor.sample_count > 0
                       and whole_device_monitor.peak_gib is not None
                       and whole_device_monitor.error is None)
            load_peak = result["model"].get("load_peak_whole_device_gib")
            task_sampling_verified = all(
                task.get("whole_device_sample_count", 0) > 0
                and task.get("whole_device_sample_error") is None
                and task.get("peak_gpu_used_gib") is not None
                for task in result["model"].get("tasks", {}).values()
            ) and complete_suite
            load_peak_within_cap = (isinstance(load_peak, (int, float))
                                    and 0 < load_peak <= MEMORY_CAP_GIB)
            result["model"].update(
                peak_gpu_used_gib=scoring_peak,
                scoring_peak_gpu_used_gib=scoring_peak,
                cap_verified=bool(
                    sampled and task_sampling_verified and load_peak_within_cap
                    and complete_suite and probe.get("passed")
                    and suite_context_passed
                    and scoring_peak is not None and 0 < scoring_peak <= MEMORY_CAP_GIB
                    and whole_device_monitor.peak_gib is not None
                    and whole_device_monitor.peak_gib <= MEMORY_CAP_GIB
                ),
                cap_verification={
                    "whole_device_cap_gib": MEMORY_CAP_GIB,
                    "load_peak_whole_device_gib": load_peak,
                    "load_peak_within_cap": load_peak_within_cap,
                    "context_probe_passed": bool(probe.get("passed")),
                    "complete_shared_task_suite": complete_suite,
                    "scored_items_within_probed_context": suite_context_passed,
                    "scoring_peak_within_cap": scoring_peak is not None and 0 < scoring_peak <= MEMORY_CAP_GIB,
                    "whole_device_sampling_verified": sampled,
                    "whole_device_sampler_error": whole_device_monitor.error,
                    "task_sampling_verified": task_sampling_verified,
                    "whole_device_peak_within_cap": (
                        sampled and whole_device_monitor.peak_gib is not None
                        and whole_device_monitor.peak_gib <= MEMORY_CAP_GIB
                    ),
                },
            )
            result["model"]["status"] = (
                "complete" if result["model"].get("cap_verified") else
                ("unverified_cap" if result["model"].get("status") == "complete" else result["model"].get("status"))
            )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", result)
    print(f"Wrote {result_path} and {output_dir / 'report.md'}", flush=True)
    return 0 if result["model"]["status"] in {"complete", "partial", "unverified_cap"} else 2


if __name__ == "__main__":
    raise SystemExit(run_queued(main, "general-transformers"))
