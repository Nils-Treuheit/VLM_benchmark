#!/usr/bin/env python3
"""Run the shared VLM suite against locally complete foundation/GGUF candidates."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ROBOTICS = Path("/mnt/HDD1/Project_Code/robotics_VLM")
sys.path.insert(0, str(ROOT / "qwenVL_Bench"))
sys.path.insert(0, str(ROBOTICS / "src"))
sys.path.insert(0, str(ROOT))
from gpu_queue import run_queued  # noqa: E402
import benchmark as common  # noqa: E402
import benchmark_runners as shared  # noqa: E402

from robotics_vlm.contracts.common import ClockDomain
from robotics_vlm.frames import Frame
from robotics_vlm.models.adapters.lfm25_vl import (
    LFM25VLAdapter,
    LFM25VLRequest,
    audit_cache as audit_lfm_cache,
    get_setup as get_lfm_setup,
)

HF_HOME = Path(os.environ.get("HF_HOME", "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"))
CACHE_ROOT = HF_HOME / "hub"
MODEL_ID = "unsloth/LFM2.5-VL-1.6B-GGUF"
MODEL_ALIAS = "lfm25_vl_1_6b_bf16_gguf"
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUT = HERE / "results" / "runs" / RUN_ID / "lfm25_vl_1_6b_bf16_gguf"
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
MEMORY_CAP_GIB = 30.0


class DeviceMemorySampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=3,
                ).stdout.strip().splitlines()[0]
                value = int(output)
                self.peak_mib = max(self.peak_mib or 0, value)
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=4)

    @property
    def peak_gib(self):
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


def make_frame(image, index: int) -> Frame:
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Frame(
        pixels_rgb=rgb,
        uid=f"local-foundation-{index}",
        capture_time_ns=None,
        receive_monotonic_ns=time.monotonic_ns(),
        clock_domain=ClockDomain.MONOTONIC_DERIVED,
        source_id="benchmark",
        source_frame_id=str(index),
        receive_sequence=index,
    )


def run_generation(adapter, row, task: str, prompt: str, index: int, sampler: DeviceMemorySampler):
    request = LFM25VLRequest(text=prompt, frames=(make_frame(row["image"], index),))
    messages = adapter._message(request)
    started = time.perf_counter()
    raw = adapter._model.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.1,
        top_p=0.95,
        min_p=0.05,
        repeat_penalty=1.05,
        seed=None,
    )
    elapsed = time.perf_counter() - started
    answer = str(raw["choices"][0]["message"].get("content") or "").strip()
    usage = raw.get("usage", {})
    return {
        "answer": answer,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "generation_s": round(elapsed, 4),
        "peak_device_vram_gib": sampler.peak_gib,
        "finish_reason": raw["choices"][0].get("finish_reason"),
    }


def summarize_task(task: str, indices: list[int], items: list[dict[str, Any]], dataset):
    successes = [item for item in items if "score" in item]
    scores = [item["score"] for item in successes]
    generation_time = sum(item.get("generation_s", 0) for item in successes)
    output_tokens = sum(item.get("output_tokens") or 0 for item in successes)
    groups = None
    if task == "graph_grab_lite":
        group_values: dict[str, list[float]] = {}
        for item in successes:
            row = dataset[item["index"]]
            group_values.setdefault(row["task"], []).append(item["score"])
            item.update(pid=row["pid"], task_type=row["task"], dataset_category=row["category"], complexity=row["complexity"])
        groups = {name: statistics.mean(values) for name, values in sorted(group_values.items())}
    return {
        "requested_n": len(indices),
        "completed_n": len(successes),
        "failed_n": len(indices) - len(successes),
        "score": statistics.mean(scores) if scores else None,
        "parse_errors": sum(bool(item.get("parse_error")) for item in successes),
        "group_scores": groups,
        "mean_generation_s": statistics.mean(item["generation_s"] for item in successes) if successes else None,
        "generated_tokens_per_s": output_tokens / generation_time if generation_time else None,
        "peak_device_vram_gib": max(
            (item.get("peak_device_vram_gib") for item in successes if item.get("peak_device_vram_gib") is not None),
            default=None,
        ),
        "items": items,
    }


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    setup = get_lfm_setup()
    audit = audit_lfm_cache(MODEL_ALIAS, cache_root=CACHE_ROOT)
    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": 3407,
        "memory_cap_gib": MEMORY_CAP_GIB,
        "task_sizes": TASK_SIZES,
        "task_names": list(TASKS),
        "model": {
            "model_id": MODEL_ID,
            "setup_id": MODEL_ALIAS,
            "backend": "llama-cpp-python 0.3.31 MTMDChatHandler CUDA",
            "snapshot": str(audit.snapshot_path) if audit.snapshot_path else None,
            "revision": setup.revision,
            "quantization": "BF16 GGUF; F16 multimodal projector",
            "configured_context": setup.context_length,
            "generation": {
                "max_new_tokens": 512,
                "temperature": 0.1,
                "top_p": 0.95,
                "min_p": 0.05,
                "repeat_penalty": 1.05,
                "seed": None,
            },
            "cache_status": "ready" if audit.complete else "incomplete",
            "cache_issues": list(audit.issues),
            "weight_bytes": sum(
                (audit.snapshot_path / name).stat().st_size
                for name in ("LFM2.5-VL-1.6B-BF16.gguf", "mmproj-F16.gguf")
            ) if audit.snapshot_path else 0,
            "runtime_versions": {},
            "tasks": {},
            "status": "not_started",
        },
        "preflight": {"passed": False, "required": ["ai2d", "textvqa"], "checks": {}},
    }
    for package in ("llama-cpp-python", "torch", "numpy", "datasets"):
        try:
            result["model"]["runtime_versions"][package] = version(package)
        except PackageNotFoundError:
            result["model"]["runtime_versions"][package] = None
    result_path = OUT / "results.json"
    report_path = HERE / "report.md"
    if not audit.complete:
        result["model"].update(status="failed", error="cached GGUF or projector did not pass the pinned artifact audit")
        write_outputs(result, result_path, report_path)
        return result

    sampler = DeviceMemorySampler()
    adapter = LFM25VLAdapter(MODEL_ALIAS, cache_root=CACHE_ROOT)
    try:
        sampler.start()
        adapter.prepare()
        result["model"]["status"] = "in_progress"
        preflight_cache: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in TASKS}
        for task in ("ai2d", "textvqa"):
            dataset = shared.dataset_for(task)
            index = common.pick_indices(len(dataset), TASK_SIZES[task], 3407)[0]
            row = dataset[index]
            prompt, expected = shared.prompt_for(task, row)
            output = run_generation(adapter, row, task, prompt, index, sampler)
            prediction, score, parse_error = shared.score_for(task, output["answer"], expected)
            format_ok = not parse_error if task == "ai2d" else bool(output["answer"])
            check = {
                "index": index,
                "passed": bool(output["answer"]) and format_ok,
                "nonempty_answer": bool(output["answer"]),
                "answer_format_ok": format_ok,
                "prediction": prediction,
                "score": score,
                "parse_error": parse_error,
                "generation_s": output["generation_s"],
                "input_tokens": output["input_tokens"],
                "output_tokens": output["output_tokens"],
                "answer": output["answer"],
                "prompt": prompt,
            }
            result["preflight"]["checks"][task] = check
            preflight_cache[task][index] = {**output, "prediction": prediction, "score": score, "parse_error": parse_error}
            if not check["passed"]:
                break
        result["preflight"]["passed"] = (
            len(result["preflight"]["checks"]) == 2
            and all(check["passed"] for check in result["preflight"]["checks"].values())
        )
        if not result["preflight"]["passed"]:
            result["model"].update(status="preflight_failed", error="AI2D format and TextVQA nonempty answer checks did not both pass")
            write_outputs(result, result_path, report_path)
            return result

        for task, count in TASK_SIZES.items():
            dataset = shared.dataset_for(task)
            indices = common.pick_indices(len(dataset), count, 3407)
            items: list[dict[str, Any]] = []
            for position, index in enumerate(indices, start=1):
                row = dataset[index]
                prompt, expected = shared.prompt_for(task, row)
                try:
                    cached = preflight_cache.get(task, {}).get(index)
                    output = cached or run_generation(adapter, row, task, prompt, index, sampler)
                    prediction, score, parse_error = (
                        cached["prediction"], cached["score"], cached["parse_error"]
                    ) if cached else shared.score_for(task, output["answer"], expected)
                    items.append({
                        "index": index,
                        "score": score,
                        "prediction": prediction,
                        "output": output["answer"],
                        "prompt": prompt,
                        "input_tokens": output["input_tokens"],
                        "output_tokens": output["output_tokens"],
                        "parse_error": parse_error,
                        "generation_s": output["generation_s"],
                        "peak_device_vram_gib": output["peak_device_vram_gib"],
                        "finish_reason": output["finish_reason"],
                    })
                except Exception as exc:  # preserve item-level failures
                    items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
                if position % 10 == 0 or position == len(indices):
                    print(f"PROGRESS {task} {position}/{len(indices)}", flush=True)
            result["model"]["tasks"][task] = summarize_task(task, indices, items, dataset)
            write_outputs(result, result_path, report_path)
        result["model"]["status"] = "complete" if all(
            result["model"]["tasks"][task]["completed_n"] == count for task, count in TASK_SIZES.items()
        ) else "partial"
        result["model"]["whole_device_peak_gib"] = sampler.peak_gib
        result["model"]["whole_device_memory_method"] = "nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline"
        result["model"]["memory_sample_count"] = sampler.samples
    except Exception as exc:
        result["model"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        try:
            adapter.close()
        except Exception:
            traceback.print_exc()
        sampler.close()
    write_outputs(result, result_path, report_path)
    return result


def write_outputs(result: dict[str, Any], result_path: Path, report_path: Path):
    model = result["model"]
    full = all(model.get("tasks", {}).get(task, {}).get("completed_n") == count for task, count in TASK_SIZES.items())
    model["full_benchmark_complete"] = full
    model["full_benchmark_macro"] = statistics.mean(model["tasks"][task]["score"] for task in TASKS) if full else None
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    runtime_rows = []
    for task in TASKS:
        value = model.get("tasks", {}).get(task, {})
        score = value.get("score")
        rows.append(f"| {LABELS[task]} | {value.get('completed_n', 0)}/{value.get('requested_n', TASK_SIZES[task])} | {'—' if score is None else f'{score:.1%}'} | {value.get('parse_errors', 0)} |")
        runtime_rows.append(f"| {LABELS[task]} | {fmt(value.get('mean_generation_s'))} | {fmt(value.get('generated_tokens_per_s'))} | {fmt(value.get('peak_device_vram_gib'))} |")
    check_rows = [f"| {LABELS[task]} | {'yes' if check.get('passed') else 'no'} | {check.get('output_tokens') or '—'} |" for task, check in result["preflight"]["checks"].items()]
    status = model.get("status", "not_started")
    report = [
        "# Local foundation and GGUF VLM benchmark", "",
        f"Updated: {result['generated_at']}",
        f"Run status: **{status}**. Tested scored models: {1 if model.get('tasks') else 0}; tasks: 8; sample count: 825 per complete model.",
        f"Backend: `{model['backend']}`; runtime: {', '.join(f'{key} {value}' for key, value in model['runtime_versions'].items())}.",
        f"Checkpoint: `{model.get('model_id')}`; revision `{model.get('revision')}`; context {model.get('configured_context')} tokens.",
        "", "## Real-image preflight", "", "| Task | Passed | Output tokens |", "|---|---:|---:|", *check_rows,
        "", "## Quality results", "", "| Task | Completed / requested | Score | Parse errors |", "|---|---:|---:|---:|", *rows,
        "", f"Complete-suite macro: {fmt_score(model.get('full_benchmark_macro'))}.",
        "The macro is the unweighted mean of eight local task scores and appears only when all 825 requests complete.",
        "", "## Runtime and resources", "", "| Task | Mean generation seconds | Output tokens/second | Whole-device peak GiB |", "|---|---:|---:|---:|", *runtime_rows,
        f"Whole-device peak across the run: {fmt(model.get('whole_device_peak_gib'))} GiB. Sampling includes the desktop baseline; no process-local VRAM counter is exposed by the llama.cpp Python adapter.",
        "", "## Method and provenance", "",
        "This run uses the shared fixed examples, prompts, scorers, and seed 3407 from the [Qwen-VL benchmark contract](../qwenVL_Bench/README.md). It runs serially with one image per request and no batching.",
        "LFM2.5-VL uses the pinned `unsloth/LFM2.5-VL-1.6B-GGUF` BF16 GGUF plus `mmproj-F16.gguf` through the robotics project's documented `llama-cpp-python` MTMD adapter. Generation uses max 512 tokens, temperature 0.1, top_p 0.95, min_p 0.05, repeat penalty 1.05, and context 8192.",
        "AI2D and TextVQA preflight requests are reused in their scored samples. Outputs, item scores, token counts, timing, and errors are in the aggregate JSON.",
        f"Cached weight size: {fmt(model.get('weight_bytes', 0) / (1024**3) if model.get('weight_bytes') else None)} GiB, including the projector. Runtime peak is measured, not estimated.",
        "", "## Candidate coverage", "",
        "Only complete local image-QA artifacts with an applicable documented route are eligible for inference. MiMo-VL was not run because its cached snapshot contains no files. DiffusionGemma GGUF was not run because its snapshot contains a text-only GGUF and no projector; its separate NVFP4 artifact uses a diffusion pipeline rather than direct image-conditioned text output. Qwen3.8 artifacts remain in the inventory pending an eligible route under the configured Qwen runtime.",
        "", "Raw per-item results: [`results.json`](results/runs/" + RUN_ID + "/lfm25_vl_1_6b_bf16_gguf/results.json).", "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def fmt_score(value):
    return "—" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    raise SystemExit(run_queued(run, "local-foundation"))
