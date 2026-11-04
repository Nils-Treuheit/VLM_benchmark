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
CONTEXT = 8192


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
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval_s)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=4)

    @property
    def peak_gib(self):
        return round(self.peak_mib / 1024, 3) if self.peak_mib is not None else None


class LlamaServer:
    def __init__(self, profile: dict[str, Any], model_path: Path, mmproj_path: Path, log_path: Path):
        self.profile = profile
        self.base_url = f"http://127.0.0.1:{PORT}"
        command = [
            str(LLAMA_SERVER), "--model", str(model_path), "--mmproj", str(mmproj_path),
            "--ctx-size", str(CONTEXT), "--n-gpu-layers", "all", "--host", "127.0.0.1",
            "--port", str(PORT), "--parallel", "1", "--batch-size", "512",
            "--temp", str(profile["temperature"]), "--top-p", str(profile["top_p"]),
            "--top-k", str(profile["top_k"]), "--min-p", str(profile["min_p"]),
            "--repeat-penalty", str(profile["repeat_penalty"]),
        ]
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

    def generate(self, image, prompt: str, seed: int):
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
            "max_tokens": self.profile.get("max_new_tokens", 512),
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


def execute(slug: str):
    profile = MODELS[slug]
    snapshot = repo_snapshot(profile["repo"], profile["revision"])
    model_path, mmproj_path = snapshot / profile["model_file"], snapshot / profile["mmproj_file"]
    result_dir = HERE / "results" / "runs" / RUN_ID / slug
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "results.json"
    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "seed": 3407, "memory_cap_gib": 30.0,
        "task_sizes": TASK_SIZES,
        "model": {
            "model_id": profile["model_id"], "setup_id": slug, "backend": "llama.cpp b12097 CUDA 13.2 OpenAI-compatible server",
            "model_card": profile["model_card"],
            "snapshot": str(snapshot), "revision": profile["revision"], "quantization": profile["quantization"],
            "advertised_context": profile["advertised_context"], "configured_context": CONTEXT,
            "image_processing": {"image_format": "JPEG quality 92", "image_max_tokens": profile.get("image_max_tokens", "llama.cpp default; not explicitly capped"), "images_per_request": 1},
            "generation": {key: profile[key] for key in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty")} | {"max_new_tokens": profile.get("max_new_tokens", 512), "seed": 3407},
            "weight_bytes": model_path.stat().st_size + mmproj_path.stat().st_size if model_path.is_file() and mmproj_path.is_file() else 0,
            "tasks": {}, "status": "not_started", "runtime_versions": {"llama.cpp": "b12097", "CUDA": "13.2"},
        },
        "preflight": {"passed": False, "required": ["ai2d", "textvqa"], "checks": {}},
    }
    if not LLAMA_SERVER.is_file():
        output["model"].update(status="failed", error=f"llama-server binary is missing: {LLAMA_SERVER}")
        result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output
    if not model_path.is_file() or not mmproj_path.is_file():
        output["model"].update(status="failed", error="Pinned GGUF or matching projector is missing")
        result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output

    sampler = DeviceMemorySampler()
    server = None
    try:
        server = LlamaServer(profile, model_path, mmproj_path, result_dir / "llama-server.log")
        server.start()
        sampler.start()
        output["model"]["status"] = "in_progress"
        cache: dict[str, dict[int, dict[str, Any]]] = {task: {} for task in TASKS}
        for task in ("ai2d", "textvqa"):
            dataset = shared.dataset_for(task)
            index = common.pick_indices(len(dataset), TASK_SIZES[task], 3407)[0]
            row = dataset[index]
            prompt, expected = shared.prompt_for(task, row)
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
            output["preflight"]["checks"][task] = check
            cache[task][index] = {**generation, "prediction": prediction, "score": score, "parse_error": parse_error}
            if not check["passed"]:
                break
        output["preflight"]["passed"] = len(output["preflight"]["checks"]) == 2 and all(x["passed"] for x in output["preflight"]["checks"].values())
        if not output["preflight"]["passed"]:
            output["model"].update(status="preflight_failed", error="AI2D format and TextVQA nonempty-answer preflight did not both pass")
            result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            return output
        for task, count in TASK_SIZES.items():
            dataset = shared.dataset_for(task)
            indices = common.pick_indices(len(dataset), count, 3407)
            items = []
            for position, index in enumerate(indices, 1):
                row = dataset[index]
                prompt, expected = shared.prompt_for(task, row)
                try:
                    generation = cache.get(task, {}).get(index) or server.generate(row["image"], prompt, 3407)
                    prediction, score, parse_error = (generation["prediction"], generation["score"], generation["parse_error"]) if "score" in generation else shared.score_for(task, generation["answer"], expected)
                    items.append({
                        "index": index, "score": score, "prediction": prediction, "output": generation["answer"],
                        "prompt": prompt, "input_tokens": generation["input_tokens"], "output_tokens": generation["output_tokens"],
                        "parse_error": parse_error, "generation_s": generation["generation_s"],
                        "peak_device_vram_gib": sampler.peak_gib, "finish_reason": generation["finish_reason"],
                    })
                except Exception as exc:
                    items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
                if position % 10 == 0 or position == count:
                    print(f"PROGRESS {slug} {task} {position}/{count}", flush=True)
            output["model"]["tasks"][task] = summarize_task(task, indices, items, dataset)
            full = all(output["model"].get("tasks", {}).get(t, {}).get("completed_n") == n for t, n in TASK_SIZES.items())
            output["model"]["full_benchmark_complete"] = full
            output["model"]["full_benchmark_macro"] = statistics.mean(output["model"]["tasks"][t]["score"] for t in TASKS) if full else None
            result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        output["model"]["status"] = "complete" if all(output["model"]["tasks"][t]["completed_n"] == n for t, n in TASK_SIZES.items()) else "partial"
        output["model"]["whole_device_peak_gib"] = sampler.peak_gib
        output["model"]["whole_device_memory_method"] = "nvidia-smi query-gpu sampled every 0.5 seconds, including desktop baseline"
        output["model"]["memory_sample_count"] = sampler.samples
    except Exception as exc:
        output["model"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if server is not None:
            server.close()
        sampler.close()
    output["model"]["full_benchmark_complete"] = all(output["model"].get("tasks", {}).get(t, {}).get("completed_n") == n for t, n in TASK_SIZES.items())
    output["model"]["full_benchmark_macro"] = statistics.mean(output["model"]["tasks"][t]["score"] for t in TASKS) if output["model"]["full_benchmark_complete"] else None
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESULT {slug} {output['model']['status']} {result_path}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    execute(args.model)


if __name__ == "__main__":
    main()
