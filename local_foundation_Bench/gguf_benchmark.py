#!/usr/bin/env python3
"""Run the shared image-VQA suite against locally cached Liquid GGUF models."""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ROBOTICS = Path("/mnt/HDD1/Project_Code/robotics_VLM")
sys_path = [str(ROOT / "qwenVL_Bench"), str(ROBOTICS / "src")]
import sys
for entry in reversed(sys_path):
    sys.path.insert(0, entry)
import benchmark as common  # noqa: E402
import benchmark_runners as shared  # noqa: E402

HF_HOME = Path(os.environ.get("HF_HOME", "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"))
HUB = HF_HOME / "hub"
LLAMA_SERVER = Path(os.environ.get("LLAMA_SERVER", "/tmp/llama-src/build/bin/llama-server"))
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
    "ai2d": "AI2D", "textvqa": "TextVQA / OCR", "vqa_truth": "Truth",
    "vqa_count": "Counting", "vqa_description": "Description",
    "vqa_affordance": "Affordance", "vqa_negation": "Negation", "graph_grab_lite": "GRAB-Lite",
}
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
PORT = 28143
DEFAULT_CONTEXT = 8192
EDGE_CAPS = (8, 10, 12, 14, 16)


def repo_snapshot(repo: str, revision: str) -> Path:
    return HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


MODELS = {
    "lfm25_1_6b": {
        "model_id": "unsloth/LFM2.5-VL-1.6B-GGUF",
        "model_card": "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B",
        "revision": "6d323fde0c0a094eeeb2a2452c590fa17a3ff76b",
        "repo": "unsloth/LFM2.5-VL-1.6B-GGUF",
        "model_file": "LFM2.5-VL-1.6B-BF16.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "quantization": "BF16 GGUF + F16 mmproj",
        "temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.15, "repeat_penalty": 1.05,
        "advertised_context": 32768, "no_reasoning_preserve": False, "image_max_tokens": 64,
    },
    "lfm25_3b": {
        "model_id": "unsloth/LFM2.5-VL-3B-GGUF",
        "model_card": "https://huggingface.co/LiquidAI/LFM2.5-VL-3B",
        "revision": "22f063714556bd4daa81c40dc1ff15ca37ae92ea",
        "repo": "unsloth/LFM2.5-VL-3B-GGUF",
        "model_file": "LFM2.5-VL-3B-UD-Q4_K_XL.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "quantization": "UD-Q4_K_XL + F16 mmproj",
        "temperature": 0.2, "top_p": 1.0, "top_k": 50, "min_p": 0.0, "repeat_penalty": 1.0,
        "advertised_context": 32768, "no_reasoning_preserve": False,
    },
    "lfm3_2b": {
        "model_id": "LiquidAI/LFM3-VL-2B-GGUF",
        "model_card": "https://huggingface.co/LiquidAI/LFM3-VL-2B",
        "revision": "1add221a5ea6c047fcf1eaaf2d247939d7cf8849",
        "repo": "LiquidAI/LFM3-VL-2B-GGUF",
        "model_file": "LFM3-VL-2B-Q4_K_M.gguf",
        "mmproj_file": "mmproj-LFM3-VL-2B-F16.gguf",
        "quantization": "Q4_K_M + F16 mmproj",
        "temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0, "repeat_penalty": 1.1,
        "advertised_context": 131072, "no_reasoning_preserve": True, "max_new_tokens": 4096,
    },
    "lfm3_2b_nvfp4": {
        "model_id": "LiquidAI/LFM3-VL-2B-GGUF:NVFP4",
        "model_card": "https://huggingface.co/LiquidAI/LFM3-VL-2B",
        "revision": "1add221a5ea6c047fcf1eaaf2d247939d7cf8849",
        "repo": "LiquidAI/LFM3-VL-2B-GGUF",
        "model_file": "LFM3-VL-2B-NVFP4.gguf",
        "mmproj_file": "mmproj-LFM3-VL-2B-F16.gguf",
        "quantization": "NVFP4 feed-forward weights, Q8_0 other weights + F16 mmproj",
        "temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0, "repeat_penalty": 1.1,
        "advertised_context": 131072, "no_reasoning_preserve": True, "max_new_tokens": 4096,
    },
}


class DeviceMemorySampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.samples = 0
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                raw = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=3,
                ).stdout.strip().splitlines()[0]
                value = int(raw)
                self.peak_mib = max(self.peak_mib or 0, value)
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.wait(self.interval_s)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            self.error = self.error or "TimeoutError: nvidia-smi sampler worker did not stop within 4 seconds"

    @property
    def peak_gib(self):
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


def required_gpu_value(query: str) -> str:
    try:
        value = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError(f"could not measure GPU field {query}") from exc
    if not value:
        raise RuntimeError(f"GPU measurement {query} returned no value")
    return value


def gpu_used_mib_required() -> int:
    value = required_gpu_value("memory.used")
    try:
        return int(value.split()[0])
    except ValueError as exc:
        raise RuntimeError("nvidia-smi returned an invalid memory.used value") from exc


def exception_detail(exc: BaseException) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    reader = getattr(exc, "read", None)
    if callable(reader):
        try:
            body = reader(4096).decode("utf-8", errors="replace").strip()
            if body:
                detail += f"; response_body={body}"
        except Exception:
            pass
    return detail


def is_oom_error(error: str | None) -> bool:
    text = (error or "").lower()
    return any(marker in text for marker in (
        "out of memory", "failed to allocate", "allocation failed", "cudamalloc failed",
    ))


def is_context_limit_error(error: str | None) -> bool:
    text = (error or "").lower()
    return ("exceed_context_size_error" in text
            or "request exceeds the available context size" in text)


def measured_generation(server: "LlamaServer", image, prompt: str, seed: int,
                        max_tokens: int) -> tuple[dict[str, Any] | None, float | None, str | None, dict[str, Any]]:
    sampler = DeviceMemorySampler(interval_s=0.1)
    before = after = None
    before_error = after_error = generation_error = None
    sampler.start()
    try:
        before = gpu_used_mib_required()
    except RuntimeError as exc:
        before_error = str(exc)
    generation = None
    try:
        generation = server.generate(image, prompt, seed, max_tokens=max_tokens)
    except Exception as exc:
        generation_error = exception_detail(exc)
    finally:
        try:
            after = gpu_used_mib_required()
        except RuntimeError as exc:
            after_error = str(exc)
        sampler.close()
    if sampler.thread.is_alive():
        sampler.error = sampler.error or "TimeoutError: per-attempt nvidia-smi sampler did not stop within 4 seconds"
    peaks = [value for value in (before, after, sampler.peak_mib) if isinstance(value, (int, float))]
    peak = round(max(peaks) / 1024, 3) if peaks else None
    evidence = {
        "whole_device_sample_count": sampler.samples,
        "whole_device_sampler_error": sampler.error,
        "whole_device_sample_before_mib": before,
        "whole_device_sample_after_mib": after,
        "whole_device_sample_peak_gib": sampler.peak_gib,
        "measured_peak_whole_device_gib": peak,
        "before_measurement_error": before_error,
        "after_measurement_error": after_error,
        "generation_error": generation_error,
    }
    errors = [value for value in (generation_error, before_error, after_error, sampler.error) if value]
    return generation, peak, "; ".join(errors) or None, evidence


def sampling_evidence_verified(evidence: dict[str, Any], cap_gib: int) -> bool:
    peak = evidence.get("measured_peak_whole_device_gib")
    return bool(
        evidence.get("whole_device_sample_count", 0) > 0
        and evidence.get("whole_device_sampler_error") is None
        and isinstance(evidence.get("whole_device_sample_peak_gib"), (int, float))
        and evidence.get("whole_device_sample_before_mib") is not None
        and evidence.get("whole_device_sample_after_mib") is not None
        and isinstance(peak, (int, float)) and 0 < peak <= cap_gib
    )


def server_context_info(server: "LlamaServer") -> dict[str, Any]:
    """Read llama.cpp's active context when its slots endpoint exposes it."""
    try:
        with urllib.request.urlopen(server.base_url + "/slots", timeout=5) as response:
            slots = json.loads(response.read())
        values = [int(slot["n_ctx"]) for slot in slots if isinstance(slot, dict) and slot.get("n_ctx")]
        if values:
            actual = max(values)
            return {
                "requested_context_tokens": server.context,
                "configured_context_tokens": actual,
                "fit_adjusted": actual != server.context,
                "fit_context_source": "llama.cpp /slots n_ctx",
            }
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        pass
    return {
        "requested_context_tokens": server.context,
        "configured_context_tokens": None,
        "fit_adjusted": None,
        "fit_context_source": "not exposed by llama.cpp /slots",
    }


class LlamaServer:
    def __init__(self, profile: dict[str, Any], model_path: Path, mmproj_path: Path,
                 log_path: Path, context: int = DEFAULT_CONTEXT,
                 fit_target_mib: int | None = None):
        self.profile = profile
        self.base_url = f"http://127.0.0.1:{PORT}"
        self.context = context
        self.fit_target_mib = fit_target_mib
        command = [
            str(LLAMA_SERVER), "--model", str(model_path), "--mmproj", str(mmproj_path),
            "--ctx-size", str(context), "--n-gpu-layers", "all", "--host", "127.0.0.1",
            "--port", str(PORT), "--parallel", "1", "--batch-size", "512",
            "--temp", str(profile["temperature"]), "--top-p", str(profile["top_p"]),
            "--top-k", str(profile["top_k"]), "--min-p", str(profile["min_p"]),
            "--repeat-penalty", str(profile["repeat_penalty"]),
        ]
        if fit_target_mib is not None:
            command.extend(["--fit", "on", "--fit-target", str(fit_target_mib)])
        self.command = command
        if profile["no_reasoning_preserve"]:
            command.append("--no-reasoning-preserve")
        if profile.get("image_max_tokens") is not None:
            command.extend(["--image-max-tokens", str(profile["image_max_tokens"])])
        self.log_file = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(command, stdout=self.log_file, stderr=subprocess.STDOUT)

    def start(self, timeout_s: int = 300):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not become healthy within 300 seconds")

    def generate(self, image, prompt: str, seed: int, max_tokens: int | None = None):
        import io
        started = time.perf_counter()
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": self.profile["model_id"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "max_tokens": max_tokens if max_tokens is not None else self.profile.get("max_new_tokens", 512),
            "temperature": self.profile["temperature"],
            "top_p": self.profile["top_p"],
            "top_k": self.profile["top_k"],
            "min_p": self.profile["min_p"],
            "repeat_penalty": self.profile["repeat_penalty"],
            "seed": seed,
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = json.load(response)
        elapsed = time.perf_counter() - started
        answer = str(raw["choices"][0]["message"].get("content") or "").strip()
        usage = raw.get("usage", {})
        return {
            "answer": answer,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "generation_s": round(elapsed, 4),
            "finish_reason": raw["choices"][0].get("finish_reason"),
        }

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log_file.close()


def probe_context(server: LlamaServer, cap_gib: int, context_limit: int,
                  seed: int = 3407) -> dict[str, Any]:
    data = shared.dataset_for("ai2d")
    indices = common.pick_indices(len(data), min(100, len(data)), seed)
    measured_images: list[tuple[int, int, float]] = []
    image_attempts = []
    for index in indices:
        row = data[index]
        prompt, _ = shared.prompt_for("ai2d", row)
        generation, peak, error, evidence = measured_generation(
            server, row["image"], prompt, seed, max_tokens=1,
        )
        usage = generation or {}
        input_tokens = usage.get("input_tokens")
        sampling_verified = sampling_evidence_verified(evidence, cap_gib)
        if (error is None and isinstance(input_tokens, int) and input_tokens > 0
                and sampling_verified):
            measured_images.append((input_tokens, index, peak))
            image_attempts.append({"index": index, "prompt_tokens": input_tokens,
                                   "peak_whole_device_gib": peak,
                                   "sampling_verified": sampling_verified,
                                   "passed": True, **evidence})
        else:
            image_attempts.append({"index": index, "prompt_tokens": input_tokens,
                                   "peak_whole_device_gib": peak, "passed": False,
                                   "sampling_verified": sampling_verified,
                                   "error": error or "server returned no prompt-token usage",
                                   **evidence})

    if not measured_images:
        return {
            "passed": False, "image_dataset": "AI2D test", "image_selection_sample_count": len(indices),
            "image_selection_success_count": 0, "image_selection_complete": False,
            "error": "no sampled AI2D image returned prompt-token usage",
            "image_selection_attempts": image_attempts,
            "all_probe_attempts_within_cap": False,
        }
    image_prompt_tokens, image_index, _ = max(measured_images, key=lambda item: item[0])
    image = data[image_index]["image"]
    prompt, _ = shared.prompt_for("ai2d", data[image_index])
    low, high = 0, context_limit
    best = None
    attempts = []
    probe_errors = [
        attempt.get("error") or "AI2D selection attempt lacked verified sampling/token usage"
        for attempt in image_attempts if not attempt.get("passed")
    ]
    while low <= high:
        repeats = (low + high) // 2
        filler_prompt = prompt + (" context" * repeats)
        generation, peak, error, evidence = measured_generation(
            server, image, filler_prompt, seed, max_tokens=8,
        )
        usage = generation or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        valid_usage = isinstance(input_tokens, int) and input_tokens > 0 and isinstance(output_tokens, int)
        within_config = bool(valid_usage and input_tokens + output_tokens <= context_limit)
        sampling_verified = sampling_evidence_verified(evidence, cap_gib)
        within_cap = peak is not None and peak <= cap_gib
        passed = error is None and valid_usage and within_config and sampling_verified
        generation_error = evidence.get("generation_error")
        oom_attempt = is_oom_error(generation_error)
        recoverable_oom = oom_attempt and sampling_verified
        recoverable_context = (
            not oom_attempt and is_context_limit_error(generation_error) and sampling_verified
        )
        attempt = {
            "filler_repetitions": repeats,
            "actual_input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "peak_whole_device_gib": peak,
            "within_configured_context": within_config,
            "within_whole_device_cap": within_cap,
            "sampling_verified": sampling_verified,
            "oom_upper_bound": oom_attempt,
            "recoverable_oom_upper_bound": recoverable_oom,
            "recoverable_context_upper_bound": recoverable_context,
            "passed": passed,
            **evidence,
        }
        if error:
            attempt["error"] = error
        attempts.append(attempt)
        if not passed:
            if not (recoverable_oom or recoverable_context):
                probe_errors.append(error or "context-probe attempt lacked complete under-cap sampling evidence")
            high = repeats - 1
        else:
            best = attempt
            low = repeats + 1

    image_selection_peak = max((
        attempt["peak_whole_device_gib"] for attempt in image_attempts
        if isinstance(attempt.get("peak_whole_device_gib"), (int, float))
    ), default=None)
    attempt_peaks = [attempt.get("peak_whole_device_gib") for attempt in attempts
                     if isinstance(attempt.get("peak_whole_device_gib"), (int, float))]
    probe_peak = max(
        [image_selection_peak, *attempt_peaks] if image_selection_peak is not None else attempt_peaks,
        default=None,
    )
    image_selection_complete = (
        len(measured_images) == len(indices)
        and all(attempt.get("sampling_verified") and attempt.get("passed") for attempt in image_attempts)
    )
    all_attempts = [*image_attempts, *attempts]
    all_probe_attempts_within_cap = all(
        attempt.get("sampling_verified") is True
        and isinstance(attempt.get("peak_whole_device_gib"), (int, float))
        and 0 < attempt["peak_whole_device_gib"] <= cap_gib
        for attempt in all_attempts
    )
    passed = bool(
        image_selection_complete and best and not probe_errors and all_probe_attempts_within_cap and probe_peak is not None
        and all_probe_attempts_within_cap and probe_peak <= cap_gib and image_selection_peak is not None
        and image_selection_peak <= cap_gib
    )
    return {
        "passed": passed,
        "image_dataset": "AI2D test",
        "image_index": image_index,
        "image_size_px": list(image.size),
        "image_prompt_tokens": image_prompt_tokens,
        "image_selection_method": "highest llama.cpp prompt-token count across the fixed AI2D sample",
        "image_selection_sample_count": len(indices),
        "image_selection_success_count": len(measured_images),
        "image_selection_complete": image_selection_complete,
        "image_selection_peak_gib": image_selection_peak,
        "image_selection_attempts": image_attempts,
        "requested_context_tokens": context_limit,
        "measured_context_tokens": (best["actual_input_tokens"] + best["output_tokens"]) if best else None,
        "actual_input_tokens": best.get("actual_input_tokens") if best else None,
        "output_tokens": best.get("output_tokens") if best else None,
        "configured_limit_reached": bool(
            best and best["actual_input_tokens"] + best["output_tokens"] >= context_limit - 32
        ),
        "peak_whole_device_gib": probe_peak,
        "all_probe_attempts_within_cap": all_probe_attempts_within_cap,
        "all_probe_attempts_sampling_verified": all(attempt.get("sampling_verified") is True for attempt in all_attempts),
        "errors": probe_errors,
        "whole_device_cap_gib": cap_gib,
        "search_method": "binary search on deterministic filler repetitions using API-reported prompt tokens",
        "attempts": attempts,
    }


def summarize_task(task: str, indices: list[int], items: list[dict[str, Any]], dataset):
    scored = [item for item in items if "score" in item]
    scores = [item["score"] for item in scored]
    if task == "graph_grab_lite":
        groups: dict[str, list[float]] = {}
        for item in scored:
            row = dataset[item["index"]]
            groups.setdefault(row["task"], []).append(item["score"])
            item.update(pid=row["pid"], task_type=row["task"], dataset_category=row["category"], complexity=row["complexity"])
        group_scores = {key: statistics.mean(value) for key, value in sorted(groups.items())}
    else:
        group_scores = None
    generation_s = sum(item.get("generation_s", 0) for item in scored)
    output_tokens = sum(item.get("output_tokens") or 0 for item in scored)
    return {
        "requested_n": len(indices), "completed_n": len(scored), "failed_n": len(indices) - len(scored),
        "score": statistics.mean(scores) if scores else None,
        "parse_errors": sum(bool(item.get("parse_error")) for item in scored),
        "group_scores": group_scores,
        "mean_generation_s": statistics.mean(item["generation_s"] for item in scored) if scored else None,
        "generated_tokens_per_s": output_tokens / generation_s if generation_s else None,
        "peak_device_vram_gib": max((item.get("peak_device_vram_gib") for item in scored if item.get("peak_device_vram_gib") is not None), default=None),
        "items": items,
    }


def write_edge_report(path: Path, payload: dict[str, Any]) -> None:
    model = payload["model"]
    probe = model.get("context_probe") or {}
    macro = model.get("full_benchmark_macro")
    macro_text = "—" if macro is None else f"{macro:.1%}"
    peak_task = max(
        (task for task in TASKS if model.get("tasks", {}).get(task, {}).get("peak_gpu_used_gib") is not None),
        key=lambda task: model["tasks"][task]["peak_gpu_used_gib"], default="—",
    )
    peak_task = LABELS.get(peak_task, peak_task)
    lines = [
        f"# {model['model_id']} Edge run ({model.get('whole_device_cap_gib')} GiB)", "",
        f"Generated: {payload.get('generated_at')}",
        f"Status: {model.get('status')}; cap verified: {model.get('cap_verified', False)}.",
        f"Backend: {model.get('backend')}; revision: `{model.get('revision')}`; quantization: {model.get('quantization')}.",
        f"GPU: {payload.get('hardware', {}).get('gpu_name')}; driver {payload.get('hardware', {}).get('driver_version')}; "
        f"idle baseline {payload.get('gpu_baseline_gib')} GiB.",
        f"Measured context: {probe.get('measured_context_tokens')} tokens; configured limit reached: "
        f"{probe.get('configured_limit_reached', False)}.",
        f"Context image: AI2D index {probe.get('image_index')}, {probe.get('image_size_px')}; "
        f"selection: {probe.get('image_selection_method')} "
        f"({probe.get('image_selection_success_count')}/{probe.get('image_selection_sample_count')} measured).",
        f"Whole-device peaks: load {model.get('load_peak_whole_device_gib')} GiB; context probe "
        f"{probe.get('peak_whole_device_gib')} GiB; scoring {model.get('peak_gpu_used_gib')} GiB; "
        f"whole run {model.get('whole_device_peak_gib')} GiB.",
        f"Cap enforcement: {model.get('cap_enforcement_method', payload.get('cap_enforcement_method'))}; llama.cpp fit target "
        f"{model.get('cap_fit_target_mib')} MiB.",
        "", "## Shared task results", "",
        "| Task | Completed / requested | Score | Peak whole-device GiB |", "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        record = model.get("tasks", {}).get(task, {})
        score = record.get("score")
        lines.append(
            f"| {LABELS[task]} | {record.get('completed_n', 0)}/{record.get('requested_n', TASK_SIZES[task])} "
            f"| {'—' if score is None else f'{score:.1%}'} | {record.get('peak_gpu_used_gib', '—')} |"
        )
    lines.extend([
        "", f"Full-suite macro: {macro_text}.", f"Peak task: {peak_task}.",
        "", "## Verification", "", "```json",
        json.dumps(model.get("cap_verification", {}), indent=2), "```", "",
        f"Source artifact: `{path.with_name('results.json').name}`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(slug: str, whole_device_cap_gib: int | None = None,
            context_probe_limit: int = 262144, output_dir: Path | None = None):
    profile = MODELS[slug]
    snapshot = repo_snapshot(profile["repo"], profile["revision"])
    model_path, mmproj_path = snapshot / profile["model_file"], snapshot / profile["mmproj_file"]
    edge_run = whole_device_cap_gib is not None
    if output_dir is None:
        run_root = HERE / "results" / "runs" / RUN_ID
        if edge_run:
            run_root /= f"edge_{whole_device_cap_gib}gib"
    else:
        run_root = output_dir if output_dir.is_absolute() else HERE / "results" / "runs" / output_dir
        runs_root = (HERE / "results" / "runs").resolve()
        if not run_root.resolve().is_relative_to(runs_root):
            raise ValueError("--output-dir must remain inside local_foundation_Bench/results/runs")
    result_dir = run_root / slug
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "results.json"

    requested_context = min(profile["advertised_context"], context_probe_limit) if edge_run else DEFAULT_CONTEXT
    baseline_gib = None
    fit_target = None
    hardware = None
    precheck_error = None
    if edge_run:
        try:
            baseline_gib = gpu_used_mib_required() / 1024
            total_mib = int(required_gpu_value("memory.total").split()[0])
            gpu_name = required_gpu_value("name")
            driver = required_gpu_value("driver_version")
            if baseline_gib >= whole_device_cap_gib:
                raise RuntimeError("measured idle whole-device baseline reaches the selected cap")
            if whole_device_cap_gib * 1024 > total_mib:
                raise RuntimeError("selected cap exceeds installed GPU memory")
            fit_target = total_mib - int((whole_device_cap_gib - baseline_gib) * 1024)
            if fit_target < 0:
                raise RuntimeError("computed llama.cpp fit target is invalid")
            hardware = {"gpu_name": gpu_name, "driver_version": driver, "total_memory_mib": total_mib}
        except Exception as exc:
            precheck_error = f"{type(exc).__name__}: {exc}"

    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "seed": 3407,
        "memory_cap_gib": whole_device_cap_gib if edge_run else 30.0,
        "task_sizes": TASK_SIZES,
        "model": {
            "model_id": profile["model_id"], "setup_id": slug,
            "backend": "llama.cpp b12097 CUDA 13.2 OpenAI-compatible server",
            "model_card": profile["model_card"], "snapshot": str(snapshot),
            "revision": profile["revision"], "quantization": profile["quantization"],
            "advertised_context": profile["advertised_context"],
            "configured_context": requested_context, "requested_context": requested_context,
            "configured_context_tokens": None, "fit_adjusted": None,
            "context_fit_adjusted": None, "context_probe": None,
            "peak_gpu_used_gib": None,
            "whole_device_cap_gib": whole_device_cap_gib if edge_run else None,
            "gpu_baseline_gib": baseline_gib, "cap_fit_target_mib": fit_target,
            "image_processing": {"image_format": "JPEG quality 92",
                                 "image_max_tokens": profile.get("image_max_tokens", "llama.cpp default; not explicitly capped"),
                                 "images_per_request": 1},
            "generation": {key: profile[key] for key in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty")}
                | {"max_new_tokens": profile.get("max_new_tokens", 512), "seed": 3407},
            "weight_bytes": model_path.stat().st_size + mmproj_path.stat().st_size
                if model_path.is_file() and mmproj_path.is_file() else 0,
            "tasks": {}, "status": "not_started",
            "runtime_versions": {"llama.cpp": "b12097", "CUDA": "13.2"},
        },
        "preflight": {"passed": False, "required": ["ai2d", "textvqa"], "checks": {}},
    }
    if edge_run:
        output.update({
            "whole_device_cap_gib": whole_device_cap_gib,
            "gpu_baseline_gib": baseline_gib,
            "hardware": hardware,
            "runtime_versions": {"python": sys.version.split()[0], "llama.cpp": "b12097", "CUDA": "13.2"},
            "cap_enforcement_method": (
                f"llama.cpp --fit on --fit-target {fit_target} MiB; whole-device nvidia-smi sampling"
            ),
        })
        output["model"].update(
            cap_verified=False,
            cap_verification={
                "whole_device_cap_gib": whole_device_cap_gib,
                "context_probe_passed": False,
                "complete_shared_task_suite": False,
                "scoring_peak_within_cap": False,
                "whole_run_peak_within_cap": False,
            },
        )

    def save() -> None:
        if edge_run:
            output["models"] = [output["model"]]
        result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if precheck_error:
        output["model"].update(status="cap_precheck_failed", error=precheck_error)
        save()
        return output
    if not LLAMA_SERVER.is_file():
        output["model"].update(status="failed", error=f"llama-server binary is missing: {LLAMA_SERVER}")
        save()
        return output
    if not model_path.is_file() or not mmproj_path.is_file():
        output["model"].update(status="failed", error="Pinned GGUF or matching projector is missing")
        save()
        return output

    sampler = DeviceMemorySampler()
    suite_sampler = None
    server = None
    try:
        # Edge runs include load time; default runs preserve their previous sampling window.
        if edge_run:
            sampler.start()
        server = LlamaServer(profile, model_path, mmproj_path, result_dir / "llama-server.log",
                             context=requested_context, fit_target_mib=fit_target)
        server.start()
        if not edge_run:
            sampler.start()
        model = output["model"]
        model["status"] = "in_progress"
        model.update(server_context_info(server))
        model["context_fit_adjusted"] = model.get("fit_adjusted")
        model["server_command"] = server.command
        model["server_log"] = str(result_dir / "llama-server.log")
        if edge_run:
            load_peak_mib = max(sampler.peak_mib or 0, gpu_used_mib_required())
            model["load_peak_whole_device_gib"] = round(load_peak_mib / 1024, 3)
            if model["load_peak_whole_device_gib"] > whole_device_cap_gib:
                model.update(status="load_exceeded_cap", cap_verified=False,
                             error="model load exceeded the selected whole-device cap")
                model["cap_verification"]["whole_run_peak_within_cap"] = False
                save()
                return output

        if edge_run:
            measured_server_context = model.get("configured_context_tokens") or requested_context
            model["context_probe"] = probe_context(server, whole_device_cap_gib,
                                                   measured_server_context, seed=3407)
            model["context_probe"].update({
                "requested_context_tokens": requested_context,
                "configured_context_tokens": model.get("configured_context_tokens"),
                "fit_adjusted": model.get("fit_adjusted"),
                "whole_device_cap_gib": whole_device_cap_gib,
            })
            context_configuration_known = (
                isinstance(model.get("configured_context_tokens"), int)
                and isinstance(model.get("fit_adjusted"), bool)
            )
            model["context_probe"]["context_configuration_known"] = context_configuration_known
            model["context_probe"]["passed"] = bool(
                model["context_probe"].get("passed") and context_configuration_known
            )
            if not model["context_probe"].get("passed"):
                model["cap_verification"].update({
                    "context_probe_passed": False,
                    "context_fit_adjusted": model.get("fit_adjusted"),
                    "configured_limit_reached": bool(model["context_probe"].get("configured_limit_reached")),
                })
                model.update(status="context_probe_failed", cap_verified=False,
                             error="real-image context probe did not pass the requested cap")
                save()
                return output

        # The scoring sampler excludes the independent context probe.
        suite_sampler = DeviceMemorySampler()
        suite_sampler.start()
        cache: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in TASKS}
        preflight_task_peaks: dict[str, float] = {}
        for task in ("ai2d", "textvqa"):
            dataset = shared.dataset_for(task)
            index = common.pick_indices(len(dataset), TASK_SIZES[task], 3407)[0]
            row = dataset[index]
            prompt, expected = shared.prompt_for(task, row)
            if edge_run:
                generation, preflight_peak, measure_error, preflight_evidence = measured_generation(
                    server, row["image"], prompt, 3407,
                    max_tokens=profile.get("max_new_tokens", 512),
                )
                preflight_sampling_verified = sampling_evidence_verified(
                    preflight_evidence, whole_device_cap_gib
                )
                if generation is None or measure_error or not preflight_sampling_verified:
                    output["preflight"]["checks"][task] = {
                        "index": index, "passed": False,
                        "error": measure_error or f"{task} preflight sampling evidence failed verification",
                        "peak_whole_device_gib": preflight_peak,
                        "sampling_verified": preflight_sampling_verified,
                        **preflight_evidence,
                    }
                    raise RuntimeError(measure_error or f"{task} preflight sampling evidence failed verification")
                generation["peak_device_vram_gib"] = preflight_peak
                if preflight_peak is not None:
                    preflight_task_peaks[task] = preflight_peak
            else:
                generation = server.generate(row["image"], prompt, 3407)
            prediction, score, parse_error = shared.score_for(task, generation["answer"], expected)
            format_ok = not parse_error if task == "ai2d" else bool(generation["answer"])
            check = {
                "index": index, "passed": bool(generation["answer"]) and format_ok,
                "nonempty_answer": bool(generation["answer"]), "answer_format_ok": format_ok,
                "prediction": prediction, "score": score, "parse_error": parse_error,
                "generation_s": generation["generation_s"], "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"], "answer": generation["answer"], "prompt": prompt,
            }
            if edge_run:
                check.update({
                    "peak_whole_device_gib": preflight_peak,
                    "sampling_verified": preflight_sampling_verified,
                    **preflight_evidence,
                })
            output["preflight"]["checks"][task] = check
            cache[task][index] = {**generation, "prediction": prediction, "score": score, "parse_error": parse_error}
            if not check["passed"]:
                break
        output["preflight"]["passed"] = len(output["preflight"]["checks"]) == 2 and all(
            item["passed"] for item in output["preflight"]["checks"].values()
        )
        if not output["preflight"]["passed"]:
            model.update(status="preflight_failed",
                         error="AI2D format and TextVQA nonempty-answer preflight did not both pass")
            save()
            return output

        for task, count in TASK_SIZES.items():
            dataset = shared.dataset_for(task)
            indices = common.pick_indices(len(dataset), count, 3407)
            items = []
            task_sampler = DeviceMemorySampler() if edge_run else None
            if task_sampler:
                task_sampler.start()
            for position, index in enumerate(indices, 1):
                row = dataset[index]
                prompt, expected = shared.prompt_for(task, row)
                try:
                    generation = cache.get(task, {}).get(index) or server.generate(row["image"], prompt, 3407)
                    prediction, score, parse_error = (
                        (generation["prediction"], generation["score"], generation["parse_error"])
                        if "score" in generation else shared.score_for(task, generation["answer"], expected)
                    )
                    items.append({
                        "index": index, "score": score, "prediction": prediction, "output": generation["answer"],
                        "prompt": prompt, "input_tokens": generation["input_tokens"], "output_tokens": generation["output_tokens"],
                        "parse_error": parse_error, "generation_s": generation["generation_s"],
                        "peak_device_vram_gib": generation.get(
                            "peak_device_vram_gib", task_sampler.peak_gib if task_sampler else sampler.peak_gib
                        ), "finish_reason": generation["finish_reason"],
                    })
                except Exception as exc:
                    items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
                if position % 10 == 0 or position == count:
                    print(f"PROGRESS {slug} {task} {position}/{count}", flush=True)
            task_peak = None
            task_sample_count = 0
            task_sample_error = None
            if task_sampler:
                task_sampler.close()
                task_peak = task_sampler.peak_gib
                task_sample_count = task_sampler.samples
                task_sample_error = task_sampler.error
            task_peak = max((value for value in (task_peak, preflight_task_peaks.get(task))
                             if value is not None), default=None)
            task_record = summarize_task(task, indices, items, dataset)
            if edge_run:
                task_record["peak_gpu_used_gib"] = task_peak
                task_record["peak_device_vram_gib"] = task_peak
                task_record["whole_device_sample_count"] = task_sample_count
                task_record["whole_device_sample_error"] = task_sample_error
            model["tasks"][task] = task_record
            full = all(model.get("tasks", {}).get(name, {}).get("completed_n") == size
                       for name, size in TASK_SIZES.items())
            model["full_benchmark_complete"] = full
            model["full_benchmark_macro"] = (
                statistics.mean(model["tasks"][name]["score"] for name in TASKS) if full else None
            )
            save()

        suite_sampler.close()
        task_sampler_peaks = [
            task.get("peak_gpu_used_gib") for task in model.get("tasks", {}).values()
            if isinstance(task.get("peak_gpu_used_gib"), (int, float))
        ]
        scoring_peak = (
            max(value for value in [suite_sampler.peak_gib, *task_sampler_peaks]
                if isinstance(value, (int, float)))
            if edge_run and any(isinstance(value, (int, float))
                                for value in [suite_sampler.peak_gib, *task_sampler_peaks])
            else sampler.peak_gib if not edge_run else None
        )
        whole_run_peaks = [sampler.peak_gib, *task_sampler_peaks]
        if edge_run:
            whole_run_peaks.extend([
                model.get("load_peak_whole_device_gib"),
                (model.get("context_probe") or {}).get("peak_whole_device_gib"),
            ])
        whole_device_peak = (
            max(value for value in whole_run_peaks if isinstance(value, (int, float)))
            if edge_run and any(isinstance(value, (int, float)) for value in whole_run_peaks)
            else sampler.peak_gib
        )
        model["status"] = "complete" if all(
            model["tasks"].get(name, {}).get("completed_n") == size
            for name, size in TASK_SIZES.items()
        ) else "partial"
        model["full_benchmark_complete"] = model["status"] == "complete"
        model["full_benchmark_macro"] = (
            statistics.mean(model["tasks"][name]["score"] for name in TASKS)
            if model["full_benchmark_complete"] else None
        )
        model["peak_gpu_used_gib"] = scoring_peak
        if edge_run:
            model["scoring_peak_gpu_used_gib"] = scoring_peak
        model["whole_device_peak_gib"] = whole_device_peak
        model["whole_device_memory_method"] = "nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline"
        model["memory_sample_count"] = sampler.samples
        model["whole_device_sample_error"] = sampler.error
        model["scoring_sample_count"] = suite_sampler.samples if suite_sampler else 0
        model["scoring_sample_error"] = suite_sampler.error if suite_sampler else "missing scoring sampler"
        if edge_run:
            probe = model.get("context_probe") or {}
            probe_peak = probe.get("peak_whole_device_gib")
            probe_passed = bool(probe.get("passed") and probe_peak is not None
                                and 0 < probe_peak <= whole_device_cap_gib)
            scoring_within = bool(scoring_peak is not None and 0 < scoring_peak <= whole_device_cap_gib)
            whole_peak = model.get("whole_device_peak_gib")
            whole_within = bool(whole_peak is not None and 0 < whole_peak <= whole_device_cap_gib)
            complete = bool(model["full_benchmark_complete"])
            task_sampling_verified = all(
                task.get("whole_device_sample_count", 0) > 0
                and task.get("whole_device_sample_error") is None
                and isinstance(task.get("peak_gpu_used_gib"), (int, float))
                and 0 < task["peak_gpu_used_gib"] <= whole_device_cap_gib
                for task in model.get("tasks", {}).values()
            ) and complete
            whole_sampling_verified = sampler.samples > 0 and sampler.error is None
            scoring_sampling_verified = bool(
                suite_sampler and suite_sampler.samples > 0 and suite_sampler.error is None
            )
            model["cap_verification"] = {
                "whole_device_cap_gib": whole_device_cap_gib,
                "context_probe_passed": probe_passed,
                "complete_shared_task_suite": complete,
                "scoring_peak_within_cap": scoring_within,
                "whole_run_peak_within_cap": whole_within,
                "context_fit_adjusted": model.get("fit_adjusted"),
                "configured_limit_reached": bool(probe.get("configured_limit_reached")),
                "whole_device_sampling_verified": whole_sampling_verified,
                "scoring_sampling_verified": scoring_sampling_verified,
                "task_sampling_verified": task_sampling_verified,
            }
            model["cap_verified"] = bool(
                complete and probe_passed and scoring_within and whole_within
                and whole_sampling_verified and scoring_sampling_verified and task_sampling_verified
            )
            if not model["cap_verified"]:
                model["status"] = "unverified_cap"
        save()
    except Exception as exc:
        output["model"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if suite_sampler:
            suite_sampler.close()
        if server is not None:
            server.close()
        sampler.close()
    model = output["model"]
    model["full_benchmark_complete"] = all(
        model.get("tasks", {}).get(task, {}).get("completed_n") == count
        for task, count in TASK_SIZES.items()
    )
    model["full_benchmark_macro"] = (
        statistics.mean(model["tasks"][task]["score"] for task in TASKS)
        if model["full_benchmark_complete"] else None
    )
    save()
    if edge_run:
        write_edge_report(result_dir / "report.md", output)
    print(f"RESULT {slug} {model['status']} {result_path}", flush=True)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--whole-device-cap-gib", type=int, choices=EDGE_CAPS,
                        help="run an Edge refinement pass under an 8, 10, 12, 14, or 16 GiB whole-device cap")
    parser.add_argument("--context-probe-limit", type=int, default=262144,
                        help="maximum requested context for the real-image probe and scored suite")
    parser.add_argument("--output-dir", type=Path,
                        help="optional cap-specific run root beneath results/runs; model slug is appended")
    args = parser.parse_args()
    if args.context_probe_limit < 1024:
        parser.error("--context-probe-limit must be at least 1,024 tokens")
    execute(args.model, whole_device_cap_gib=args.whole_device_cap_gib,
            context_probe_limit=args.context_probe_limit, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
