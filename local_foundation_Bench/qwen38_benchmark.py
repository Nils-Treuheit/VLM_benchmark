#!/usr/bin/env python3
"""Run Qwen3.8 through the configured Ollama MTP4 route on the shared VLM suite."""

from __future__ import annotations

import base64
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT / "qwenVL_Bench"))
import benchmark as common  # noqa: E402
import benchmark_runners as shared  # noqa: E402

MODEL_ID = "qwen3.8:27b-mtp4-ud_q4_K_XL-128k"
CANONICAL_ID = "unsloth/Qwen3.8-27B-NVFP4"
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUT = HERE / "results" / "runs" / RUN_ID / "qwen3_8_mtp4_ud_q4_k_xl_128k"
TASK_SIZES = {
    "ai2d": 100, "textvqa": 100, "vqa_truth": 25, "vqa_count": 25,
    "vqa_description": 25, "vqa_affordance": 25, "vqa_negation": 25,
    "graph_grab_lite": 500,
}
TASKS = tuple(TASK_SIZES)
LABELS = {
    "ai2d": "AI2D", "textvqa": "TextVQA / OCR", "vqa_truth": "Truth",
    "vqa_count": "Counting", "vqa_description": "Description",
    "vqa_affordance": "Affordance", "vqa_negation": "Negation",
    "graph_grab_lite": "GRAB-Lite",
}
API = "http://127.0.0.1:20261/api/chat"
MEMORY_CAP_GIB = 30.0


class DeviceMemorySampler:
    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self.peak_mib = None
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                value = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=3,
                ).stdout.strip().splitlines()[0]
                used = int(value)
                self.peak_mib = max(self.peak_mib or 0, used)
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval_s)

    def start(self): self.thread.start()
    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
    @property
    def peak_gib(self):
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


def image_data_url(image):
    import io
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate(row, prompt, sampler):
    import time
    started = time.perf_counter()
    request = urllib.request.Request(API, data=json.dumps({
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_data_url(row["image"])],
        }],
        "think": False,
        "options": {"num_predict": 512, "temperature": 0, "top_p": 1, "seed": 3407},
        "stream": False,
        "keep_alive": "5m",
    }).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload.get("message", {}).get("content", "")
    output_tokens = payload.get("eval_count")
    input_tokens = payload.get("prompt_eval_count")
    server_total_ns = payload.get("total_duration")
    server_load_ns = payload.get("load_duration", 0)
    generation_s = (
        max(0.0, (server_total_ns - server_load_ns) / 1_000_000_000)
        if isinstance(server_total_ns, (int, float)) else time.perf_counter() - started
    )
    return {
        "answer": str(content).strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_s": round(generation_s, 4),
        "peak_device_vram_gib": sampler.peak_gib,
        "finish_reason": "stop" if payload.get("done") else "incomplete",
        "response_metadata": {
            "prompt_eval_count": input_tokens,
            "eval_count": output_tokens,
            "done_reason": payload.get("done_reason"),
            "load_duration_ns": server_load_ns,
        },
    }


def write_outputs(result, result_path, report_path):
    model = result["model"]
    full = all(model.get("tasks", {}).get(t, {}).get("completed_n") == n for t, n in TASK_SIZES.items())
    model["full_benchmark_complete"] = full
    model["full_benchmark_macro"] = statistics.mean(model["tasks"][t]["score"] for t in TASKS) if full else None
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows, runtime_rows = [], []
    for task in TASKS:
        data = model.get("tasks", {}).get(task, {})
        score = data.get("score")
        score_text = "—" if score is None else f"{score:.1%}"
        rows.append(f"| {LABELS[task]} | {data.get('completed_n', 0)}/{data.get('requested_n', TASK_SIZES[task])} | {score_text} | {data.get('parse_errors', 0)} |")
        runtime_rows.append(f"| {LABELS[task]} | {fmt(data.get('mean_generation_s'))} | {fmt(data.get('generated_tokens_per_s'))} | {fmt(data.get('peak_device_vram_gib'))} |")
    preflight_rows = [f"| {LABELS[t]} | {'yes' if c.get('passed') else 'no'} | {c.get('output_tokens') or '—'} |" for t, c in result["preflight"]["checks"].items()]
    report = [
        "# Qwen3.8 local foundation benchmark", "",
        f"Updated: {result['generated_at']}",
        f"Run status: **{model.get('status')}**. Tested scored models: 1; tasks: 8; sample count: 825.",
        f"Backend: {model['backend']}; runtime: {model['runtime']}; configured context: 131072 tokens.",
        f"Route: `{MODEL_ID}`; precision {model['quantization']}; deterministic decoding with thinking disabled.",
        "", "## Real-image preflight", "", "| Task | Passed | Output tokens |", "|---|---:|---:|", *preflight_rows,
        "", "## Quality results", "", "| Task | Completed / requested | Score | Parse errors |", "|---|---:|---:|---:|", *rows,
        "", f"Complete-suite macro: {fmt_score(model.get('full_benchmark_macro'))}.",
        "The macro is the unweighted mean of eight local task scores and is shown only when all 825 requests complete.",
        "", "## Runtime and resources", "", "| Task | Mean generation seconds | Output tokens/second | Whole-device peak GiB |", "|---|---:|---:|---:|", *runtime_rows,
        f"Whole-device peak: {fmt(model.get('whole_device_peak_gib'))} GiB. The 0.5-second nvidia-smi sampling includes the desktop baseline.",
        "", "## Method and provenance", "",
        "This run uses the shared fixed examples, prompts, scorers, and seed 3407 from the [Qwen-VL benchmark contract](../qwenVL_Bench/README.md). It runs serially with one image per request and no batching.",
        "The model is the already-installed Ollama UD Q4_K_XL MTP4 128K route. No checkpoint conversion or download was performed. The native Ollama chat API disables thinking and allows 512 output tokens. Results retain service token counters, generation time, scores, outputs, and parse errors.",
        "AI2D and TextVQA real-image preflight requests are reused in their scored samples. The measured device memory includes the desktop baseline.",
        "", f"Raw per-item results: [`results.json`](results/runs/{RUN_ID}/qwen3_8_mtp4_ud_q4_k_xl_128k/results.json).", "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")


def fmt(value): return "—" if value is None else f"{value:.2f}"
def fmt_score(value): return "—" if value is None else f"{value:.1%}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": 3407,
        "memory_cap_gib": MEMORY_CAP_GIB,
        "task_sizes": TASK_SIZES,
        "task_names": list(TASKS),
        "model": {
            "model_id": CANONICAL_ID,
            "backend": "Ollama OpenAI-compatible multimodal chat API",
            "runtime": "qwen3.8:27b-mtp4-ud_q4_K_XL-128k; MTP4; UD Q4_K_XL",
            "quantization": "UD Q4_K_XL; MTP4",
            "configured_context": 131072,
            "generation": {"num_predict": 512, "temperature": 0, "top_p": 1, "seed": 3407, "think": False},
            "cache_status": "installed (local Ollama model inventory)",
            "runtime_versions": {"ollama": "local managed service; version unavailable via benchmark API"},
            "tasks": {},
            "status": "not_started",
        },
        "preflight": {"passed": False, "required": ["ai2d", "textvqa"], "checks": {}},
    }
    sampler = DeviceMemorySampler()
    try:
        sampler.start()
        # Verify the chosen exact alias before sending an image request.
        with urllib.request.urlopen("http://127.0.0.1:20261/api/tags", timeout=5) as response:
            inventory = json.loads(response.read().decode("utf-8"))
        names = {item.get("name") for item in inventory.get("models", [])}
        if MODEL_ID not in names:
            raise RuntimeError(f"configured Ollama alias is not installed: {MODEL_ID}")
        cache: dict[str, dict[int, dict[str, Any]]] = {t: {} for t in TASKS}
        for task in ("ai2d", "textvqa"):
            dataset = shared.dataset_for(task)
            index = common.pick_indices(len(dataset), TASK_SIZES[task], 3407)[0]
            row = dataset[index]
            prompt, expected = shared.prompt_for(task, row)
            output = generate(row, prompt, sampler)
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
                "response_metadata": output["response_metadata"],
                "answer": output["answer"],
                "prompt": prompt,
            }
            result["preflight"]["checks"][task] = check
            cache[task][index] = {**output, "prediction": prediction, "score": score, "parse_error": parse_error}
            if not check["passed"]: break
        result["preflight"]["passed"] = len(result["preflight"]["checks"]) == 2 and all(c["passed"] for c in result["preflight"]["checks"].values())
        if not result["preflight"]["passed"]:
            result["model"].update(status="preflight_failed", error="AI2D format and TextVQA nonempty answer checks did not both pass")
            write_outputs(result, result_path, HERE / "report.md")
            return
        result["model"]["status"] = "in_progress"
        write_outputs(result, result_path, HERE / "report.md")
        for task, count in TASK_SIZES.items():
            dataset = shared.dataset_for(task)
            indices = common.pick_indices(len(dataset), count, 3407)
            items = []
            for position, index in enumerate(indices, 1):
                row = dataset[index]
                prompt, expected = shared.prompt_for(task, row)
                try:
                    output = cache.get(task, {}).get(index) or generate(row, prompt, sampler)
                    cached = cache.get(task, {}).get(index)
                    if cached:
                        prediction, score, parse_error = cached["prediction"], cached["score"], cached["parse_error"]
                    else:
                        prediction, score, parse_error = shared.score_for(task, output["answer"], expected)
                    items.append({
                        "index": index, "score": score, "prediction": prediction,
                        "output": output["answer"], "prompt": prompt,
                        "input_tokens": output["input_tokens"], "output_tokens": output["output_tokens"],
                        "parse_error": parse_error, "generation_s": output["generation_s"],
                        "peak_device_vram_gib": output["peak_device_vram_gib"], "finish_reason": output["finish_reason"],
                        "response_metadata": output["response_metadata"],
                    })
                except Exception as exc:
                    items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                if position % 10 == 0 or position == len(indices):
                    print(f"PROGRESS {task} {position}/{len(indices)}", flush=True)
            successes = [item for item in items if "score" in item]
            generation_total = sum(item["generation_s"] for item in successes)
            tokens_total = sum(item.get("output_tokens") or 0 for item in successes)
            groups = None
            if task == "graph_grab_lite":
                group_scores = {}
                for item in successes:
                    row = dataset[item["index"]]
                    group_scores.setdefault(row["task"], []).append(item["score"])
                    item.update(pid=row["pid"], task_type=row["task"], dataset_category=row["category"], complexity=row["complexity"])
                groups = {name: statistics.mean(values) for name, values in sorted(group_scores.items())}
            result["model"]["tasks"][task] = {
                "requested_n": len(indices), "completed_n": len(successes), "failed_n": len(indices)-len(successes),
                "score": statistics.mean([item["score"] for item in successes]) if successes else None,
                "parse_errors": sum(bool(item.get("parse_error")) for item in successes),
                "group_scores": groups,
                "mean_generation_s": statistics.mean([item["generation_s"] for item in successes]) if successes else None,
                "generated_tokens_per_s": tokens_total / generation_total if generation_total else None,
                "peak_device_vram_gib": max((item["peak_device_vram_gib"] for item in successes if item.get("peak_device_vram_gib") is not None), default=None),
                "items": items,
            }
            result["model"]["whole_device_peak_gib"] = sampler.peak_gib
            write_outputs(result, result_path, HERE / "report.md")
        result["model"]["status"] = "complete" if all(result["model"]["tasks"][t]["completed_n"] == n for t,n in TASK_SIZES.items()) else "partial"
        result["model"]["whole_device_peak_gib"] = sampler.peak_gib
        result["model"]["whole_device_memory_method"] = "nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline"
        result["model"]["memory_sample_count"] = sampler.samples
    except Exception as exc:
        result["model"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Ollama remains controlled by the model-control service. release-local
        # runs after this process so the app confirms model unload state.
        sampler.close()
    write_outputs(result, result_path, HERE / "report.md")


if __name__ == "__main__":
    main()
