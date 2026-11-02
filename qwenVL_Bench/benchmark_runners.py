#!/usr/bin/env python3
"""Run cached Qwen VL checkpoints through their publisher's documented runner."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import benchmark as common

LLAMA_SERVER = Path("/usr/local/lib/ollama/llama-server")
MEMORY_CAP_GIB = 30.0
CONTEXTS = (262144, 131072, 65536, 32768, 16384, 8192)
CONTEXTS_1M = (1048576, 524288, *CONTEXTS)
GPU_BASELINE_GIB = 1.0
ACTIVE_SERVER: subprocess.Popen | None = None
MODEL_PAIRS = {
    "ai2d": ("lmms-lab/ai2d", "test", "ai2d-test-*.arrow"),
    "textvqa": ("facebook/textvqa", "validation", "textvqa-validation.arrow"),
    "vqa_truth": ("panel", "truth", ""),
    "vqa_count": ("panel", "count", ""),
    "vqa_description": ("panel", "description", ""),
    "vqa_affordance": ("panel", "affordance", ""),
    "vqa_negation": ("panel", "negation", ""),
    "graph_grab_lite": ("panel", "graph_grab_lite", ""),
}
BASE_TASKS = ("ai2d", "textvqa")
PANEL_TASKS = ("vqa_truth", "vqa_count", "vqa_description", "vqa_affordance", "vqa_negation")
PANEL_DATA = Path(__file__).resolve().parent / "data" / "vqa_panel"
GRAPH_DATA = Path(__file__).resolve().parent / "data" / "grab_lite"


def list_original_models() -> list[dict[str, Any]]:
    entries = []
    for entry in common.discover_models():
        model_id = entry["model_id"]
        lower = model_id.lower()
        # The two cached Unsloth BNB exports have no GGUF files, so llama.cpp
        # cannot load them as-is. Their model cards document Transformers and
        # the BNB quantization metadata; run those exact cached weights there.
        is_bnb = lower.startswith("unsloth/") and "qwen3-vl-8b-thinking" in lower and "bnb-4bit" in lower
        if lower.startswith("unsloth/") and not is_bnb:
            continue
        snapshot = common.resolve_snapshot(Path(entry["cache"]))
        backend = "Transformers (Unsloth BNB 4-bit model card)" if is_bnb else "Transformers (model card)"
        reference = "Qwen/Qwen3-VL-8B-Thinking" if is_bnb else None
        entries.append({"id": model_id, "backend": backend,
                        "comparison_reference": reference,
                        "snapshot": str(snapshot) if snapshot else None})
    return entries


def list_gguf_models() -> list[dict[str, Any]]:
    entries = []
    for repo in sorted(HUB.glob("models--*--*Qwen*GGUF")):
        repo_id = repo.name.removeprefix("models--").replace("--", "/")
        if not any(f"Qwen{family}" in repo_id for family in ("2-VL", "2.5-VL", "3-VL")):
            continue
        ref = repo / "refs" / "main"
        if not ref.is_file():
            continue
        snapshot = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
        for model_path in sorted(snapshot.glob("*.gguf")):
            if model_path.name.startswith("mmproj-") or not model_path.is_file():
                continue
            match = re.search(r"((?:UD-)?(?:Q\d_K_XL|IQ\d_(?:XXS|XS|S|M)))\.gguf$", model_path.name)
            if not match:
                continue
            mmproj_candidates = [snapshot / "mmproj-F16.gguf", *sorted(snapshot.glob("mmproj-*-f16.gguf"))]
            mmproj = next((path for path in mmproj_candidates if path.is_file()), None)
            if mmproj is None:
                continue
            quant = match.group(1)
            base_name = repo_id.split("/", 1)[-1].removesuffix("-GGUF").replace("-1M", "")
            comparison_reference = f"Qwen/{base_name}"
            entries.append({
                "id": f"{repo_id}:{quant}",
                "backend": f"llama.cpp ({quant})",
                "repo": repo_id,
                "quant": quant,
                "comparison_reference": comparison_reference,
                "model_path": str(model_path),
                "mmproj_path": str(mmproj),
                "snapshot": str(snapshot),
            })
    return entries


def model_config(entry: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = entry.get("snapshot")
    if snapshot and (Path(snapshot) / "config.json").is_file():
        try:
            return json.loads((Path(snapshot) / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    if entry.get("repo"):
        base = entry["repo"].split("/", 1)[-1].removesuffix("-GGUF").replace("-1M", "")
        cache = HUB / f"models--Qwen--{base}"
        ref = cache / "refs" / "main"
        if ref.is_file():
            config_path = cache / "snapshots" / ref.read_text(encoding="utf-8").strip() / "config.json"
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if "-1M" in entry["repo"]:
                    config["max_position_embeddings"] = 1048576
                    if isinstance(config.get("text_config"), dict):
                        config["text_config"]["max_position_embeddings"] = 1048576
                return config
            except (OSError, ValueError):
                pass
    return None


def kv_cache_layout(config: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not config:
        return None, None
    cfg = config.get("text_config") or config
    layers = cfg.get("num_hidden_layers")
    kv_heads = cfg.get("num_key_value_heads")
    head_dim = cfg.get("head_dim")
    if not head_dim and cfg.get("hidden_size") and cfg.get("num_attention_heads"):
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    maximum = cfg.get("max_position_embeddings") or config.get("max_position_embeddings")
    if not all((layers, kv_heads, head_dim, maximum)):
        return None, maximum
    return int(2 * layers * kv_heads * head_dim * 2), int(maximum)


def make_context_profile(
    model_bytes: int | None,
    kv_fp16_bytes_per_token: int | None,
    advertised: int | None,
    *,
    q8_cache: bool,
) -> dict[str, Any]:
    if model_bytes is None or kv_fp16_bytes_per_token is None or not advertised:
        return {"estimated_max_context": advertised, "tiers": []}
    q8_bytes_per_token = kv_fp16_bytes_per_token * (0.53125 if q8_cache else 0.5625)
    residual = min(common.KV_CACHE_RESIDUAL_TOKENS, advertised)
    residual_bytes = residual * kv_fp16_bytes_per_token
    available = max(0, int(MEMORY_CAP_GIB * 1024**3) - model_bytes - int(GPU_BASELINE_GIB * 1024**3))
    if available < residual_bytes:
        ceiling = available // kv_fp16_bytes_per_token
    else:
        ceiling = residual + (available - residual_bytes) // q8_bytes_per_token
    ceiling = min(int(advertised), int(ceiling))
    ceiling = max(0, (ceiling // 1024) * 1024)
    tiers = []
    for name, requested in (("min", 8192), ("default", 262144), ("max", 1048576)):
        context = min(requested, ceiling)
        if context < 8192:
            tiers.append({"tier": name, "context_tokens": None, "estimated_vram_gib": None})
            continue
        kv_bytes = min(context, residual) * kv_fp16_bytes_per_token
        kv_bytes += max(0, context - residual) * q8_bytes_per_token
        memory_gib = (model_bytes + kv_bytes) / 1024**3 + GPU_BASELINE_GIB
        tiers.append({"tier": name, "context_tokens": context,
                      "estimated_vram_gib": round(memory_gib, 2)})
    return {"estimated_max_context": ceiling, "tiers": tiers}


def cap_profile_to_verified_context(
    profile: dict[str, Any], verified_context: int, measured_peak_gib: float | None,
) -> dict[str, Any]:
    """Do not report a KV-cache extrapolation above the largest loaded context."""
    profile["estimated_max_context"] = min(
        int(profile.get("estimated_max_context") or verified_context), verified_context
    )
    for tier in profile.get("tiers", []):
        context = tier.get("context_tokens")
        if context is not None and context > verified_context:
            tier["context_tokens"] = verified_context
            if measured_peak_gib is not None:
                tier["estimated_vram_gib"] = round(measured_peak_gib, 2)
    return profile


def dataset_for(task: str):
    if task == "graph_grab_lite":
        from PIL import Image
        rows = json.loads((GRAPH_DATA / "examples.json").read_text(encoding="utf-8"))
        for row in rows:
            with Image.open(Path(__file__).resolve().parent / row["image_path"]) as image:
                row["image"] = image.convert("RGB")
        return rows
    if task in PANEL_TASKS:
        from PIL import Image
        rows = json.loads((PANEL_DATA / f"{task}.json").read_text(encoding="utf-8"))
        for row in rows:
            with Image.open(Path(__file__).resolve().parent / row["image_path"]) as image:
                row["image"] = image.convert("RGB")
        return rows
    dataset_id, split, pattern = MODEL_PAIRS[task]
    cache_root = HF_HOME / "datasets"
    prefix = dataset_id.replace("/", "___")
    matches = sorted(cache_root.glob(f"{prefix}/**/{pattern}"))
    if not matches:
        raise FileNotFoundError(f"Cached {dataset_id} {split} Arrow files missing in {cache_root}")
    from datasets import Dataset, concatenate_datasets
    shards = [Dataset.from_file(str(path)) for path in matches]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def prompt_for(task: str, row: dict[str, Any]):
    if task == "graph_grab_lite":
        return (f"{row['question']}\nOnly provide the answer, no reasoning steps. "
                "If unsure, still provide your best answer. Answer:"), row["answer"]
    if task == "vqa_negation":
        choices = f"A) {row['options'][0]}\nB) {row['options'][1]}"
        return (f"{row['question']}\n\nOptions:\n{choices}\n\n"
                "Choose the correct option. Return only A or B."), row["answers"]
    if task in PANEL_TASKS:
        return (f"{row['question']}\nAnswer using a short phrase grounded in the image. "
                "Return only the answer."), row["answers"]
    return common.task_prompt(task, row)


def score_for(task: str, content: str, expected: Any) -> tuple[str | None, float, bool]:
    if task == "graph_grab_lite":
        prediction = common.final_response(content)
        ref = str(expected).strip()
        numeric = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", ref)
        if numeric:
            match = re.search(r"[-+]?\d+(?:\.\d+)?", prediction.replace(",", ""))
            if not match:
                return None, 0.0, True
            from decimal import Decimal, InvalidOperation
            try:
                return match.group(0), float(Decimal(match.group(0)) == Decimal(ref)), False
            except InvalidOperation:
                return None, 0.0, True
        normalize = lambda value: re.sub(r"\s+", "", value.casefold().replace("−", "-").replace("–", "-"))
        clean = normalize(prediction)
        expected_clean = normalize(ref)
        return prediction, float(bool(clean) and clean == expected_clean), not bool(clean)
    if task == "ai2d" or task == "vqa_negation":
        prediction = common.last_choice(content)
        if task == "vqa_negation" and prediction not in {"A", "B"}:
            prediction = None
        return prediction, float(prediction == expected), prediction is None
    prediction = common.final_response(content)
    if task == "vqa_affordance":
        reference = expected[0] if isinstance(expected, list) and expected else ""
        correct = bool(reference) and common.normalize_vqa(prediction) == common.normalize_vqa(reference)
        return prediction, float(correct), not bool(prediction)
    refs = expected if isinstance(expected, list) else []
    return prediction, common.vqa_score(prediction, refs), not bool(prediction)


def image_data_url(image) -> str:
    import io
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def sample_settings(model_id: str) -> dict[str, Any]:
    lower = model_id.lower()
    if "thinking" in lower:
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0, "seed": 1234}
    if "qwen3-vl" in lower or "g-substrate" in lower:
        return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 3407}
    return {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0,
            "seed": 3407, "use_checkpoint_defaults": True}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MemorySampler:
    def __init__(self):
        self.peak_mib = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL, timeout=2,
                ).strip().splitlines()[0]
                self.peak_mib = max(self.peak_mib, int(value))
            except Exception:
                pass
            self.stop_event.wait(0.15)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop_event.set()
        self.thread.join(timeout=3)


def gpu_memory_used_mib() -> int | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        ).strip().splitlines()[0]
        return int(value)
    except Exception:
        return None


def wait_ready(proc: subprocess.Popen, port: int, timeout: float = 300) -> str:
    import urllib.error
    import urllib.request
    deadline = time.monotonic() + timeout
    log = Path(f"/tmp/vlm-llama-server-{proc.pid}.log")
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return log.read_text(errors="replace") if log.exists() else "llama-server exited early"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return ""
        except Exception:
            pass
        time.sleep(1)
    return "Timed out waiting for llama-server readiness"


def run_llama(entry: dict[str, Any], count: int, seed: int, result: dict[str, Any], result_path: Path,
              task_names: tuple[str, ...], context_cap: int | None = None,
              profile_context_only: bool = False):
    global ACTIVE_SERVER
    import urllib.error
    import urllib.request

    settings = sample_settings(entry["id"])
    model_record: dict[str, Any] = {
        "model_id": entry["id"], "backend": entry["backend"], "tasks": {},
        "kv_cache": "q8_0 K and V", "mtp_requested": "draft-mtp, up to 4 tokens for Qwen3-VL",
        "comparison_reference": entry.get("comparison_reference"),
        "gpu_baseline_gib": round(GPU_BASELINE_GIB, 3),
        "context_cap_tokens": context_cap,
        "profile_context_only": profile_context_only,
    }
    result["models"].append(model_record)
    chosen_context = None
    proc = None
    log_path = None
    contexts = CONTEXTS_1M if "-1M" in entry["repo"] else CONTEXTS
    _, checkpoint_context = kv_cache_layout(model_config(entry))
    if checkpoint_context:
        contexts = tuple(context for context in contexts if context <= checkpoint_context)
        if checkpoint_context not in contexts:
            contexts = (checkpoint_context, *contexts)
        if not contexts:
            contexts = (min(CONTEXTS),)
    if context_cap is not None:
        contexts = tuple(context for context in contexts if context <= context_cap)
        if not contexts:
            raise ValueError(f"context cap {context_cap} is below the minimum supported context")
    warm_data = dataset_for("ai2d")
    warm_row = warm_data[common.pick_indices(len(warm_data), 1, seed)[0]]
    warm_prompt, _ = common.task_prompt("ai2d", warm_row)
    warm_request = {
        "model": entry["id"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url(warm_row["image"])}},
            {"type": "text", "text": warm_prompt},
        ]}],
        **settings, "max_tokens": 64, "stream": False,
    }
    for context in contexts:
        try_mtp = "Qwen3-VL" in entry["repo"]
        modes = (True, False) if try_mtp else (False,)
        for use_mtp in modes:
            port = free_port()
            command = [
                str(LLAMA_SERVER), "--model", entry["model_path"], "--mmproj", entry["mmproj_path"],
                "--n-gpu-layers", "99", "--ctx-size", str(context), "--parallel", "1",
                "--no-cont-batching", "--flash-attn", "on", "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0", "--jinja", "--host", "127.0.0.1", "--port", str(port),
                "--fit", "on", "--fit-target", "512",
            ]
            if use_mtp:
                command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
                            "--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0"]
            suffix = "mtp" if use_mtp else "plain"
            log_path = Path(f"/tmp/vlm-llama-server-{entry['id'].replace('/', '_').replace(':', '_')}-{context}-{suffix}.log")
            log_file = log_path.open("w", encoding="utf-8")
            runtime_env = os.environ.copy()
            runtime_env["GGML_BACKEND_PATH"] = "/usr/local/lib/ollama/cuda_v13/libggml-cuda.so"
            runtime_env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v13:/usr/local/lib/ollama:" + runtime_env.get("LD_LIBRARY_PATH", "")
            proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True,
                                    env=runtime_env)
            ACTIVE_SERVER = proc
            failure = wait_ready(proc, port)
            if not failure:
                warmup_request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(warm_request).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                try:
                    with urllib.request.urlopen(warmup_request, timeout=900) as response:
                        response.read()
                except Exception as exc:
                    log_file.close()
                    try:
                        proc.terminate()
                        proc.wait(timeout=30)
                    except Exception:
                        proc.kill()
                        proc.wait(timeout=10)
                    proc = None
                    ACTIVE_SERVER = None
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    model_record["startup_attempts"] = model_record.get("startup_attempts", []) + [
                        {"context": context, "mtp": use_mtp,
                         "error": f"real-image warmup failed: {type(exc).__name__}: {exc}"}
                    ]
                    continue
                chosen_context = context
                model_record["mtp_supported"] = "yes (draft-mtp, max 4)" if use_mtp else (
                    model_record.get("mtp_supported", "no (startup did not accept draft-mtp)")
                    if try_mtp else "not applicable"
                )
                log_file.close()
                break
            log_file.close()
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)
            proc = None
            ACTIVE_SERVER = None
            log_text = log_path.read_text(errors="replace") if log_path.exists() else failure
            model_record["startup_attempts"] = model_record.get("startup_attempts", []) + [
                {"context": context, "mtp": use_mtp, "error": log_text[-2500:]}
            ]
            if use_mtp and "doesn't contain mtp layers" in log_text.lower():
                model_record["mtp_supported"] = "no (GGUF has no MTP layers)"
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if chosen_context is not None:
            break
    if proc is None or chosen_context is None:
        model_record.update(status="load_failed", error="llama-server failed at all context tiers")
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    model_record.update(status="in_progress", configured_context=chosen_context,
                        max_context=chosen_context, llama_server=str(LLAMA_SERVER),
                        server_log=str(log_path), mtp_supported=model_record.get("mtp_supported", "check server log"))
    time.sleep(2)
    current_used_mib = gpu_memory_used_mib()
    config = model_config(entry)
    kv_per_token, advertised = kv_cache_layout(config)
    if context_cap is not None:
        model_record["context_profile"] = {"estimated_max_context": None, "tiers": []}
        model_record["context_profile_status"] = (
            "deferred: shared GPU load; this pass measures quality and speed at the capped context only"
        )
    elif current_used_mib is not None and kv_per_token:
        resident_bytes = current_used_mib * 1024**2
        kv_bytes = chosen_context * kv_per_token * 0.53125
        model_bytes = max(0, int(resident_bytes - GPU_BASELINE_GIB * 1024**3 - kv_bytes))
        model_record["context_profile"] = make_context_profile(
            model_bytes, kv_per_token, advertised, q8_cache=True
        )
        # A successful real-image warmup is stronger evidence than the
        # conservative residency estimate, which also includes warmup buffers.
        if model_record["context_profile"].get("estimated_max_context", 0) < chosen_context:
            residual = min(common.KV_CACHE_RESIDUAL_TOKENS, advertised)
            q8_bytes_per_token = kv_per_token * 0.53125
            chosen_kv_bytes = min(chosen_context, residual) * kv_per_token
            chosen_kv_bytes += max(0, chosen_context - residual) * q8_bytes_per_token
            model_bytes = max(
                0, int(MEMORY_CAP_GIB * 1024**3 - GPU_BASELINE_GIB * 1024**3 - chosen_kv_bytes)
            )
            model_record["context_profile"] = make_context_profile(
                model_bytes, kv_per_token, advertised, q8_cache=True
            )
    else:
        model_record["context_profile"] = {"estimated_max_context": chosen_context, "tiers": []}
    if current_used_mib is not None and model_record["context_profile"].get("tiers"):
        model_record["context_profile"] = cap_profile_to_verified_context(
            model_record["context_profile"], chosen_context, current_used_mib / 1024
        )
    model_record["checkpoint_context"] = advertised
    model_record["warmup"] = "one real AI2D image request excluded from scoring and speed"
    if profile_context_only:
        model_record.update(
            status="context_profiled",
            profiled_context_tokens=chosen_context,
            profile_peak_gpu_used_gib=round(current_used_mib / 1024, 3)
            if current_used_mib is not None else None,
            profile_peak_gpu_delta_gib=round(
                max(0, current_used_mib - int(GPU_BASELINE_GIB * 1024)) / 1024, 3
            ) if current_used_mib is not None else None,
        )
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)
        proc = None
        ACTIVE_SERVER = None
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    with MemorySampler() as memory:
        for task in task_names:
            dataset = dataset_for(task)
            indices = common.pick_indices(len(dataset), count, seed)
            items, scores, total_out, total_gen = [], [], 0, 0.0
            for position, index in enumerate(indices):
                row = dataset[index]
                prompt, expected = prompt_for(task, row)
                body = {
                    "model": entry["id"],
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(row["image"])}},
                        {"type": "text", "text": prompt},
                    ]}],
                    "temperature": settings["temperature"], "top_p": settings["top_p"],
                    "top_k": settings["top_k"], "presence_penalty": settings["presence_penalty"],
                    "seed": seed + position, "max_tokens": 2048 if "thinking" in entry["id"].lower() else 256,
                    "stream": False,
                }
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                began = time.perf_counter()
                try:
                    with urllib.request.urlopen(request, timeout=900) as response:
                        raw = json.loads(response.read())
                    wall = time.perf_counter() - began
                    content = raw["choices"][0]["message"]["content"] or ""
                    if isinstance(content, list):
                        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                    usage = raw.get("usage", {})
                    out_tokens = int(usage.get("completion_tokens", 0))
                    prediction, score, parse_error = score_for(task, content, expected)
                    scores.append(score)
                    total_out += out_tokens
                    total_gen += wall
                    metadata = ({"pid": row["pid"], "task_type": row["task"],
                                 "dataset_category": row["category"], "complexity": row["complexity"]}
                                if task == "graph_grab_lite" else {})
                    items.append({"index": index, "score": score, "prediction": prediction,
                                  "output": content, "input_tokens": usage.get("prompt_tokens"),
                                  "output_tokens": out_tokens, "parse_error": parse_error,
                                  "wall_s": round(wall, 4), **metadata})
                except Exception as exc:
                    items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
                if (position + 1) % 10 == 0 or position + 1 == len(indices):
                    print(f"PROGRESS {entry['id']} {task} {position + 1}/{len(indices)}", flush=True)
            successful = [item for item in items if "error" not in item]
            task_record = {
                "requested_n": len(indices), "completed_n": len(successful),
                "score": statistics.mean(scores) if scores else None,
                "mean_wall_s": statistics.mean(item["wall_s"] for item in successful) if successful else None,
                "generated_tokens_per_s": total_out / total_gen if total_gen else None,
                "peak_gpu_used_gib": round(memory.peak_mib / 1024, 3), "items": items,
                "peak_gpu_delta_gib": round(
                    max(0, memory.peak_mib - int(GPU_BASELINE_GIB * 1024)) / 1024, 3
                ),
            }
            model_record["tasks"][task] = task_record
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)
    ACTIVE_SERVER = None
    model_record["status"] = "complete" if all(
        t.get("completed_n") == t.get("requested_n") for t in model_record["tasks"].values()
    ) else "complete_with_failures"
    model_record["peak_gpu_used_gib"] = round(memory.peak_mib / 1024, 3)
    model_record["peak_gpu_delta_gib"] = round(
        max(0, memory.peak_mib - int(GPU_BASELINE_GIB * 1024)) / 1024, 3
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_transformers(entry: dict[str, Any], count: int, seed: int, result: dict[str, Any], result_path: Path,
                     task_names: tuple[str, ...], max_new_tokens_override: int | None = None):
    import torch
    from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList
    from qwen_vl_utils import process_vision_info

    model_record = {"model_id": entry["id"], "backend": entry["backend"], "tasks": {},
                    "kv_cache": "Transformers HQQ int8",
                    "comparison_reference": entry.get("comparison_reference") or (
                        "Qwen/Qwen3-VL-2B-Instruct" if model_id_is_gsubstrate(entry["id"]) else None
                    )}
    result["models"].append(model_record)
    snapshot = Path(entry["snapshot"])
    model_id = entry["id"]
    mtp = common.detect_mtp_head(snapshot)
    model_record["mtp_supported"] = "no (checkpoint has no MTP weights)" if mtp == "no" else (
        "yes (checkpoint has MTP weights; model-card runner has no MTP option)" if mtp == "yes" else "unknown"
    )
    device = torch.cuda.current_device()
    trust = model_id.lower().startswith("zmli/")
    # Match each official Qwen card's architecture class. This environment's
    # Transformers version predates AutoModelForMultimodalLM, so Qwen2 and 2.5
    # use their equivalent conditional-generation classes.
    lower_id = model_id.lower()
    template_options = {"enable_thinking": False} if "cosmos3-edge" in lower_id else {}
    if "cosmos-reason1-7b" in lower_id:
        # Cosmos Reason1 retains the Qwen2.5-VL architecture and accepts still images.
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    elif lower_id.startswith("qwen/qwen2-vl-"):
        from transformers import Qwen2VLForConditionalGeneration as ModelClass
    elif lower_id.startswith("qwen/qwen2.5-vl-"):
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    elif "qwen3-vl-" in lower_id:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    else:
        from transformers import AutoModelForImageTextToText as ModelClass
    # G-Substrate's card uses AutoModelForCausalLM for text-only loading, while
    # its config is Qwen3-VL; image evaluation uses the matching image-text class.
    model = ModelClass.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=trust,
        device_map={"": device}, dtype="auto", attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=trust)
    model.eval()
    model_record["status"] = "in_progress"
    text_config = getattr(model.config, "text_config", model.config)
    model_record["checkpoint_context"] = getattr(text_config, "max_position_embeddings", None)
    model_record["model_load_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    cfg = model_config(entry)
    kv_per_token, advertised = kv_cache_layout(cfg)
    model_record["context_profile"] = make_context_profile(
        torch.cuda.memory_allocated(device), kv_per_token, advertised, q8_cache=False
    )
    torch.cuda.reset_peak_memory_stats()
    with MemorySampler() as memory:
        warm_data = dataset_for("ai2d")
        warm_row = warm_data[common.pick_indices(len(warm_data), 1, seed)[0]]
        warm_prompt, _ = common.task_prompt("ai2d", warm_row)
        warm_messages = [{"role": "user", "content": [
            {"type": "image", "image": warm_row["image"].convert("RGB")},
            {"type": "text", "text": warm_prompt},
        ]}]
        warm_rendered = processor.apply_chat_template(
            warm_messages, tokenize=False, add_generation_prompt=True, **template_options
        )
        warm_images, warm_videos = process_vision_info(warm_messages)
        warm_inputs = processor(text=[warm_rendered], images=warm_images, videos=warm_videos,
                                padding=True, return_tensors="pt").to(model.device)
        torch.manual_seed(seed)
        with torch.inference_mode():
            model.generate(**warm_inputs, max_new_tokens=16, do_sample=False,
                           cache_implementation="quantized",
                           cache_config={"backend": "hqq", "nbits": 8, "residual_length": 128})
        torch.cuda.synchronize()
        model_record["warmup"] = "one real AI2D image request excluded from scoring and speed"
        for task in task_names:
            data = dataset_for(task)
            indices = common.pick_indices(len(data), count, seed)
            items, scores, total_out, total_gen = [], [], 0, 0.0
            for position, index in enumerate(indices):
                row = data[index]
                prompt, expected = prompt_for(task, row)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": row["image"].convert("RGB")},
                    {"type": "text", "text": prompt},
                ]}]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **template_options
                )
                images, videos = process_vision_info(messages)
                inputs = processor(text=[rendered], images=images, videos=videos,
                                   padding=True, return_tensors="pt").to(model.device)
                settings = sample_settings(model_id)
                generation = {"max_new_tokens": max_new_tokens_override or (
                                  2048 if "thinking" in model_id.lower() else 256
                              ),
                              "cache_implementation": "quantized",
                              "cache_config": {"backend": "hqq", "nbits": 8, "residual_length": 128}}
                if settings.get("use_checkpoint_defaults"):
                    pass
                else:
                    generation["do_sample"] = settings["temperature"] > 0
                if settings["temperature"] > 0:
                    generation.update(temperature=settings["temperature"], top_p=settings["top_p"], top_k=settings["top_k"])
                if settings["presence_penalty"]:
                    class PresencePenalty(LogitsProcessor):
                        def __call__(self, input_ids, scores):
                            token_ids = input_ids.unique(dim=1)
                            penalty = torch.full_like(token_ids, -settings["presence_penalty"], dtype=scores.dtype)
                            scores.scatter_add_(1, token_ids, penalty)
                            return scores

                    generation["logits_processor"] = LogitsProcessorList([PresencePenalty()])
                torch.manual_seed(seed + position)
                torch.cuda.synchronize()
                began = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, **generation)
                torch.cuda.synchronize()
                wall = time.perf_counter() - began
                new_ids = output_ids[:, inputs.input_ids.shape[-1]:]
                output = processor.batch_decode(new_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0].strip()
                prediction, score, parse_error = score_for(task, output, expected)
                n_out = int(new_ids.shape[-1])
                total_out += n_out
                total_gen += wall
                scores.append(score)
                metadata = ({"pid": row["pid"], "task_type": row["task"],
                             "dataset_category": row["category"], "complexity": row["complexity"]}
                            if task == "graph_grab_lite" else {})
                items.append({"index": index, "score": score, "prediction": prediction,
                              "output": output, "input_tokens": int(inputs.input_ids.shape[-1]),
                              "output_tokens": n_out, "parse_error": parse_error,
                              "wall_s": round(wall, 4),
                              "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                              **metadata})
                torch.cuda.reset_peak_memory_stats()
                if (position + 1) % 10 == 0 or position + 1 == len(indices):
                    print(f"PROGRESS {model_id} {task} {position + 1}/{len(indices)}", flush=True)
            model_record["tasks"][task] = {
                "requested_n": len(indices), "completed_n": len(scores),
                "score": statistics.mean(scores) if scores else None,
                "mean_wall_s": statistics.mean(item["wall_s"] for item in items) if items else None,
                "generated_tokens_per_s": total_out / total_gen if total_gen else None,
                "peak_gpu_used_gib": round(memory.peak_mib / 1024, 3), "items": items,
                "peak_gpu_delta_gib": round(
                    max(0, memory.peak_mib - int(GPU_BASELINE_GIB * 1024)) / 1024, 3
                ),
            }
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_record["peak_gpu_used_gib"] = round(memory.peak_mib / 1024, 3)
    model_record["peak_gpu_delta_gib"] = round(
        max(0, memory.peak_mib - int(GPU_BASELINE_GIB * 1024)) / 1024, 3
    )
    model_record["status"] = "complete"
    del model, processor
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def write_report(path: Path, result: dict[str, Any]):
    models = result["models"]
    score_by_id = {m.get("model_id"): m.get("tasks", {}) for m in models}
    task_names = tuple(result.get("task_names", BASE_TASKS))

    def fmt_value(value: Any, digits: int) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    if "graph_grab_lite" in task_names:
        lines = [
            "# GRAB-Lite graph analysis benchmark", "",
            "The full official 500-example split was scored with normalized exact-match accuracy.",
            "", "| Model | Overall | Properties | Functions | Series | Transforms | Real | Completed | Peak GPU | Mean seconds | Output tokens/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for model in models:
            metrics = model.get("tasks", {}).get("graph_grab_lite", {})
            items = [item for item in metrics.get("items", []) if "error" not in item]
            group_scores = {}
            for name in ("properties", "function", "series", "transform", "real"):
                values = [item.get("score", 0) for item in items if item.get("task_type") == name]
                group_scores[name] = statistics.mean(values) if values else None
            values = [metrics.get("score"), *(group_scores[name] for name in group_scores)]
            formatted = ["—" if value is None else f"{value:.1%}" for value in values]
            lines.append(
                f"| {model['model_id']} | " + " | ".join(formatted) +
                f" | {metrics.get('completed_n', 0)}/{metrics.get('requested_n', 0)} | "
                f"{fmt_value(metrics.get('peak_gpu_used_gib'), 2)} GiB | "
                f"{fmt_value(metrics.get('mean_wall_s'), 2)} | {fmt_value(metrics.get('generated_tokens_per_s'), 1)} |"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    if any(task in PANEL_TASKS for task in task_names):
        lines = [
            "# VQA skill panel benchmark", "",
            f"Generated {result['generated_at']} with {result['samples_per_task']} fixed examples per question type.",
            "Requests are serial, single image, and single turn. The VQA panel uses category consensus or accuracy. GRAB-Lite uses normalized exact-match accuracy over its complete 500-question split.",
            "", "| Model | Question type | Score | Completed | Peak GPU | Mean seconds | Output tokens/s |", "|---|---|---:|---:|---:|---:|---:|",
        ]
        for model in models:
            for task in task_names:
                data = model.get("tasks", {}).get(task, {})
                score = data.get("score")
                lines.append(
                    f"| {model['model_id']} | {task} | {'—' if score is None else f'{score:.1%}'} | "
                    f"{data.get('completed_n', 0)}/{data.get('requested_n', 0)} | "
                    f"{fmt_value(data.get('peak_gpu_used_gib'), 2)} GiB | "
                    f"{fmt_value(data.get('mean_wall_s'), 2)} | {fmt_value(data.get('generated_tokens_per_s'), 1)} |"
                )
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    def fmt_context(value: Any) -> str:
        if value is None:
            return "—"
        value = int(value)
        if value >= 1_000_000:
            return f"{value / 1_000_000:g}M"
        if value >= 1024:
            return f"{value / 1024:g}K"
        return str(value)

    def fmt_metric(value: Any, digits: int) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Corrected Qwen VL Benchmark",
        "",
        f"Generated {result['generated_at']} with {result['samples_per_task']} fixed examples per task.",
        "All calls are serial, single image, single request, and at most 256 output tokens (2,048 for Thinking).",
        "",
        "## Task scores and deltas",
        "",
        "| Model | Runner | Reference | AI2D | Δ AI2D | TextVQA | Δ TextVQA | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for model in models:
        cells, deltas = [], []
        for task in task_names:
            value = model.get("tasks", {}).get(task, {}).get("score")
            cells.append("—" if value is None else f"{100*value:.1f}%")
            reference = score_by_id.get(model.get("comparison_reference"), {}).get(task, {}).get("score")
            deltas.append("—" if value is None or reference is None else f"{100*(value-reference):+.1f} pp")
        lines.append(f"| {model['model_id']} | {model.get('backend','—')} | {model.get('comparison_reference') or '—'} | {cells[0]} | {deltas[0]} | {cells[1]} | {deltas[1]} | {model.get('status','—')} |")
    lines += ["", "## Per task VRAM and speed", "",
              "| Model | Task | Completed / parse errors | Peak GPU used | Mean seconds/example | Effective output tokens/second | Context | KV cache |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for model in models:
        for task in task_names:
            data = model.get("tasks", {}).get(task, {})
            if not data:
                continue
            parse_errors = sum(bool(item.get("parse_error")) for item in data.get("items", []))
            lines.append(
                f"| {model['model_id']} | {task} | {data.get('completed_n',0)}/{data.get('requested_n',0)}; {parse_errors} | "
                f"{fmt_metric(data.get('peak_gpu_used_gib'),2)} GiB | {fmt_metric(data.get('mean_wall_s'),2)} | "
                f"{fmt_metric(data.get('generated_tokens_per_s'),1)} | "
                f"{fmt_context(model.get('configured_context',model.get('checkpoint_context')))} | {model.get('kv_cache','—')} |"
            )
    edge_rows = []
    for model in models:
        if model.get("status") != "complete":
            continue
        tasks = model.get("tasks", {})
        if not all(task in tasks and tasks[task].get("completed_n") == result["samples_per_task"] for task in task_names):
            continue
        tiers = model.get("context_profile", {}).get("tiers", [])
        min_tier = next((tier for tier in tiers if tier.get("tier") == "min"), None)
        estimated_8k = min_tier.get("estimated_vram_gib") if min_tier else None
        if estimated_8k is None:
            continue
        peak = max((tasks[task].get("peak_gpu_used_gib") or 0) for task in task_names)
        edge_rows.append((estimated_8k, peak, model))

    def append_edge_table(title: str, ceiling: float):
        eligible = sorted((row for row in edge_rows if row[0] + GPU_BASELINE_GIB < ceiling), key=lambda row: -sum(row[2]["tasks"][task]["score"] for task in task_names) / len(task_names))
        lines.extend(["", title, "",
                      "| Model | Peak GPU in benchmark | Est. total VRAM @8K | Est. max context at limit | AI2D | TextVQA | Benchmark context | Mean s/example (AI2D / TextVQA) | Output tokens/s (AI2D / TextVQA) |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for estimated_8k, peak, model in eligible:
            tasks = model["tasks"]
            context = model.get("configured_context", model.get("checkpoint_context"))
            profile = model.get("context_profile", {})
            tiers = {item.get("tier"): item for item in profile.get("tiers", [])}
            low = tiers.get("min", {})
            high = max(profile.get("tiers", []), key=lambda item: item.get("context_tokens", 0), default={})
            low_context = low.get("context_tokens", 8192)
            low_vram = low.get("estimated_vram_gib", estimated_8k)
            high_context = high.get("context_tokens", low_context)
            high_vram = high.get("estimated_vram_gib", low_vram)
            context_cap = min(
                int(profile.get("estimated_max_context") or high_context),
                int(model.get("checkpoint_context") or high_context),
            )
            model_budget = max(low_vram, ceiling - GPU_BASELINE_GIB)
            if context_cap <= low_context or high_context <= low_context or high_vram <= low_vram:
                fit_context = min(low_context, context_cap)
            elif model_budget >= high_vram:
                fit_context = context_cap
            else:
                fit_context = low_context + (model_budget - low_vram) * (high_context - low_context) / (high_vram - low_vram)
                fit_context = min(context_cap, max(low_context, int(fit_context // 1024) * 1024))
            lines.append(
                f"| {model['model_id']} | {peak:.2f} GiB | {estimated_8k + GPU_BASELINE_GIB:.2f} GiB | {fmt_context(fit_context)} | "
                f"{100*tasks['ai2d']['score']:.1f}% | {100*tasks['textvqa']['score']:.1f}% | "
                f"{fmt_context(context)} | "
                f"{fmt_metric(tasks['ai2d'].get('mean_wall_s'),2)} / {fmt_metric(tasks['textvqa'].get('mean_wall_s'),2)} | "
                f"{fmt_metric(tasks['ai2d'].get('generated_tokens_per_s'),1)} / {fmt_metric(tasks['textvqa'].get('generated_tokens_per_s'),1)} |"
            )
        if not eligible:
            lines.append("| No completed models fit this estimated 8K profile | — | — | — | — | — | — | — | — |")

    append_edge_table("## Edge candidates: estimated total VRAM below 12 GiB", 12.0)
    append_edge_table("## Edge candidates: estimated total VRAM below 16 GiB", 16.0)
    lines += ["", "_Eligibility uses the estimated 8K model residency and Q8 KV cache plus a 1 GiB desktop reserve. Maximum context at each limit is estimated by interpolating that memory profile and is capped at the model context limit. Benchmark scores and throughput were measured at the context listed; peak GPU use can therefore exceed the edge limit. Edge-device speed at a different context may vary._"]
    lines += ["", "## VRAM and context configuration", "",
              "| Model | Min / default / max VRAM (context) | Peak resident GPU memory | MTP | Runner |",
              "|---|---|---:|---|---|"]
    for model in models:
        profile = model.get("context_profile", {})
        tiers = {item.get("tier"): item for item in profile.get("tiers", [])}
        cells = []
        for name in ("min", "default", "max"):
            item = tiers.get(name)
            context = item.get("context_tokens") if item else None
            memory = item.get("estimated_vram_gib") if item else None
            if context and memory:
                requested = {"min": 8192, "default": 262144, "max": 1048576}[name]
                suffix = " max" if context < requested else ""
                cells.append(f"{memory:.2f} GiB ({fmt_context(context)}{suffix})")
            else:
                cells.append("—")
        triple = " / ".join(cells)
        if not profile.get("tiers"):
            context = model.get("configured_context", model.get("checkpoint_context"))
            triple = "—" if not context else f"— (8K) / — ({fmt_context(min(256*1024,context))}) / — ({fmt_context(min(1024*1024,context))})"
        lines.append(f"| {model['model_id']} | {triple} | {fmt_metric(model.get('peak_gpu_used_gib'),2)} GiB | {model.get('mtp_supported','no')} | {model.get('backend','—')} |")
    lines += ["", "## Notes", "",
              "- GGUF checkpoints use their publisher's llama.cpp path and Q8_0 K/V cache; MTP is requested for Qwen3-VL models and the server log records whether a model head was found.",
              "- Original Qwen checkpoints follow their model-card Transformers path. G-Substrate uses the Transformers image-text class for its Qwen3-VL config because its card's short AutoModelForCausalLM snippet demonstrates text loading only. These use the HQQ 8-bit KV cache where supported.",
              "- Each task uses the same deterministic sample IDs across models. Results are small-sample selection evidence, not leaderboard scores.",
              "- Dataset images are passed at their stored resolution; the selected model runner applies its own published image preprocessing.",
              "- VRAM is peak whole-GPU usage sampled during the task, including the desktop baseline.",
              "- Output tokens per second is output token count divided by end-to-end generation-call time, including image/prompt prefill.",
              "- Context-tier VRAM values are estimates from model residency and the checkpoint KV dimensions using Q8 cache; maximum context is capped by the 30 GiB budget and the checkpoint limit.",
              "- llama.cpp uses the largest requested context that also completes a real-image warmup under its 30 GiB fit target; it is not a claim that the dataset filled that context.",
              "- Context tiers show 8K / 256K / 1M targets; a cell marked `max` reports the estimated last context that fits below that target.",
              "- Parse failures are counted in the task table and score as incorrect answers.",
              "- G-Substrate is a full SFT fine-tune of Qwen3-VL-2B-Instruct trained on scene/event/molecular graphs and graph reasoning, with a unified graph schema across modalities.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    global GPU_BASELINE_GIB, ACTIVE_SERVER
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--output-dir", type=Path, default=Path("results/corrected"))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=tuple(MODEL_PAIRS))
    parser.add_argument("--context-cap", type=int,
                        help="maximum llama.cpp context to try (useful for shared-GPU short-context passes)")
    parser.add_argument("--profile-context-only", action="store_true",
                        help="measure the largest loadable context without rerunning scored tasks")
    args = parser.parse_args()
    task_names = tuple(args.tasks or BASE_TASKS)
    models = list_original_models() + list_gguf_models()
    if args.models:
        want = set(args.models)
        models = [m for m in models if m["id"] in want]
    if args.list_models:
        for model in models:
            ready = "ready" if model.get("snapshot") or model.get("model_path") else "incomplete"
            print(f"{model['id']}\t{model['backend']}\t{ready}")
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; host-context GPU execution is required")
    device_index = torch.cuda.current_device()
    device_bytes = torch.cuda.get_device_properties(device_index).total_memory
    if device_bytes < MEMORY_CAP_GIB * 1024**3:
        raise SystemExit("30 GiB configured benchmark ceiling exceeds the installed GPU memory")
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 1024**3 / device_bytes, device_index)
    baseline_mib = gpu_memory_used_mib()
    if baseline_mib is not None:
        GPU_BASELINE_GIB = baseline_mib / 1024
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result.get("samples_per_task") != args.samples or result.get("seed") != args.seed
                or tuple(result.get("task_names", BASE_TASKS)) != task_names
                or result.get("profile_context_only", False) != args.profile_context_only):
            raise SystemExit("--resume requires the same tasks, sample count, and seed as the existing results")
    else:
        result = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                  "samples_per_task": args.samples, "seed": args.seed,
                  "memory_cap_gib": MEMORY_CAP_GIB, "task_names": list(task_names),
                  "profile_context_only": args.profile_context_only, "models": []}
    for entry in models:
        print(f"LOAD {entry['id']} via {entry['backend']}", flush=True)
        existing = next((m for m in result["models"] if m.get("model_id") == entry["id"]), None)
        if args.resume and existing:
            if args.profile_context_only and existing.get("status") == "context_profiled":
                print(f"RESUME SKIP {entry['id']}: context already profiled", flush=True)
                continue
            complete_tasks = all(
                existing.get("tasks", {}).get(task, {}).get("completed_n") == args.samples
                and existing.get("tasks", {}).get(task, {}).get("requested_n") == args.samples
                for task in task_names
            )
            if complete_tasks or (existing.get("status") == "skipped" and not entry.get("snapshot")):
                if complete_tasks:
                    existing["status"] = "complete"
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                write_report(args.output_dir / "report.md", result)
                print(f"RESUME SKIP {entry['id']}: already complete", flush=True)
                continue
            result["models"].remove(existing)
        if entry["backend"].startswith("Transformers") and not entry.get("snapshot"):
            result["models"].append({"model_id": entry["id"], "backend": entry["backend"],
                                     "tasks": {}, "status": "skipped",
                                     "error": "local model weights are incomplete"})
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            write_report(args.output_dir / "report.md", result)
            continue
        try:
            if entry["backend"].startswith("llama.cpp"):
                run_llama(entry, args.samples, args.seed, result, result_path, task_names,
                          context_cap=args.context_cap,
                          profile_context_only=args.profile_context_only)
            else:
                run_transformers(entry, args.samples, args.seed, result, result_path, task_names)
        except Exception as exc:
            record = next((m for m in reversed(result["models"]) if m.get("model_id") == entry["id"]), None)
            if record is None:
                record = {"model_id": entry["id"], "backend": entry["backend"], "tasks": {}}
                result["models"].append(record)
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            if ACTIVE_SERVER is not None and ACTIVE_SERVER.poll() is None:
                ACTIVE_SERVER.terminate()
                try:
                    ACTIVE_SERVER.wait(timeout=30)
                except Exception:
                    ACTIVE_SERVER.kill()
                    ACTIVE_SERVER.wait(timeout=10)
                ACTIVE_SERVER = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(args.output_dir / "report.md", result)
    write_report(args.output_dir / "report.md", result)
    print(f"Wrote {result_path} and {args.output_dir / 'report.md'}", flush=True)


def model_id_is_gsubstrate(model_id: str) -> bool:
    return model_id.lower().startswith("zmli/g-substrate")


HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
HUB = HF_HOME / "hub"

if __name__ == "__main__":
    main()
