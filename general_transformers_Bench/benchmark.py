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
import benchmark as common  # noqa: E402
import benchmark_runners as shared  # noqa: E402

from model_adapters import (
    FLORENCE_ID, INTERNVL35_CANDIDATE_IDS, INTERNVL35_CANDIDATE_ROOT,
    INTERNVL35_GGUF_CANDIDATES, INTERNVL35_GGUF_IDS, INTERNVL35_HF_CANDIDATES,
    INTERNVL_GPU_IDS, INTERNVL_ID, INTERNVL_REVISION, LLAMA_VISION_ID,
    LLAMA_VISION_ROOT, LLAMA_VISION_SNAPSHOT, MOLMO2_ID, PALIGEMMA_ID,
    generate, load_florence, load_internvl, load_internvl35_hf,
    load_llama_cpp_server, load_llama_vision, load_molmo2, load_paligemma,
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


def generation(model_id: str, model, processor, row: dict[str, Any], task: str, prompt: str,
               *, route_validated: bool = False):
    import torch

    is_gguf = model_id in INTERNVL35_GGUF_CANDIDATES
    if not is_gguf:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        record = generate(model_id, model, processor, row["image"], task, prompt)
    if model_id == FLORENCE_ID and route_validated:
        record["scoring_eligible"] = True
        record["route_note"] = "experimental <VQA> route passed AI2D and TextVQA real-image preflight"
    record["generation_s"] = round(time.perf_counter() - started, 4)
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
    resource_limit_exceeded = False
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
            "The 26 GiB target applies to torch.cuda.max_memory_allocated. The 30 GiB hard cap uses sampled per-process nvidia-smi memory.",
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
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global ACTIVE_GPU_MONITOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
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
        PALIGEMMA_ID: "SDPA",
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
    if total_memory < MEMORY_CAP_GIB * 1024**3:
        result["model"].update(status="failed", error="the configured 30 GiB ceiling exceeds installed GPU memory")
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(output_dir / "report.md", result)
        return 2
    free_memory, _ = torch.cuda.mem_get_info(device)
    required_internvl_free_gib = (
        24.0 if model_id in INTERNVL35_GGUF_CANDIDATES else INTERNVL_ALLOCATED_TARGET_GIB
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
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 1024**3 / total_memory, device)

    model = processor = None
    load_monitor = None
    try:
        max_memory = {device: int(MEMORY_CAP_GIB * 1024**3)}
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
                context_size=8192,
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
        result["model"]["checkpoint_context"] = getattr(
            getattr(model.config, "text_config", model.config), "max_position_embeddings", None
        )
        if model_id == PALIGEMMA_ID:
            result["model"]["image_resolution_px"] = 224
        if (model_id in INTERNVL_GPU_IDS and model_id not in INTERNVL35_GGUF_CANDIDATES) \
                or model_id == LLAMA_VISION_ID:
            torch.cuda.reset_peak_memory_stats(device)
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
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", result)
    print(f"Wrote {result_path} and {output_dir / 'report.md'}", flush=True)
    return 0 if result["model"]["status"] in {"complete", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
