#!/usr/bin/env python3
"""Run cached Qwen VL checkpoints serially on prepared SGG panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from evaluator import EVALUATOR_VERSION, content_hash, evaluator_source_hash, score_item
from prompt import PROMPT_VERSION, prompt_for, prompt_hash


HERE = Path(__file__).resolve().parent
QWEN_DIR = HERE.parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))
import benchmark_runners as backend  # noqa: E402


DATASETS = {
    "vg150": "VG-150 canonical test",
    "openimages_v7": "Open Images V7 validation relationships",
    "psg": "PSG box-adapted SGG",
}
MAX_NEW_TOKENS = 8192
CONTEXT_SIZE = 32768
IMAGE_MIN_TOKENS = 1024
MEMORY_CAP_GIB = 30.0
RUNTIME_GATE: dict[str, Any] = {}
ACTIVE_LLAMA: dict[str, Any] | None = None


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_panel(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("items", manifest.get("samples"))
    if not isinstance(rows, list):
        raise ValueError(f"manifest has no items list: {path}")
    image_meta = manifest.get("image", {})
    display_name = manifest.get("display_name", manifest.get("dataset", ""))
    for row in rows:
        row["vocabulary"] = manifest["vocabulary"]
        vocab_payload = {"objects": manifest["vocabulary"]["objects"],
                         "predicates": manifest["vocabulary"]["predicates"]}
        row["_vocabulary_hash"] = manifest["vocabulary"].get("hash") or content_hash(vocab_payload)
        row["source_split"] = manifest.get("split")
        image = row.get("image", {})
        if "path" in image:
            row["image_source"] = {"kind": "file", "path": image["path"]}
        elif "parquet_path" in image:
            row["image_source"] = {"kind": "parquet", "path": image["parquet_path"],
                                    "row_index": image["row_index"]}
        row["image_sha256"] = image.get("sha256")
        row["_image_width"] = image.get("width", row.get("width"))
        row["_image_height"] = image.get("height", row.get("height"))
        row["_dataset_display"] = display_name
    return manifest, rows


def image_from_row(row: dict[str, Any]):
    from PIL import Image
    import io

    source = row.get("image_source")
    if isinstance(source, str):
        source = {"kind": "file", "path": source}
    if not isinstance(source, dict):
        if row.get("image_path"):
            source = {"kind": "file", "path": row["image_path"]}
        elif row.get("image_bytes"):
            source = {"kind": "bytes", "data": row["image_bytes"]}
        else:
            raise ValueError("manifest row has no supported image_source")
    expected_hash = row.get("image_sha256")
    if not expected_hash:
        raise ValueError("manifest row has no source image SHA-256")
    if source.get("kind") == "file":
        source_path = Path(source["path"])
        actual_hash = digest_file(source_path)
        if actual_hash != expected_hash:
            raise ValueError(f"source image SHA-256 mismatch: {source_path}")
        image = Image.open(source_path)
        image.load()
    elif source.get("kind") == "bytes":
        value = source["data"]
        if isinstance(value, str):
            value = bytes.fromhex(value)
        actual_hash = hashlib.sha256(value).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("embedded source image SHA-256 mismatch")
        image = Image.open(io.BytesIO(value))
        image.load()
    elif source.get("kind") == "parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(source["path"])
        offset = int(source["row_index"])
        row_group_start = 0
        for row_group in range(parquet.metadata.num_row_groups):
            row_count = parquet.metadata.row_group(row_group).num_rows
            if offset < row_group_start + row_count:
                table = parquet.read_row_group(row_group, columns=["image"])
                cell = table.slice(offset - row_group_start, 1).column(0)[0].as_py()
                break
            row_group_start += row_count
        else:
            raise ValueError(f"PSG Parquet row index is out of range: {offset}")
        value = cell.get("bytes") if isinstance(cell, dict) else None
        if not value:
            raise ValueError("PSG Parquet image bytes are absent")
        actual_hash = hashlib.sha256(value).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("PSG Parquet image SHA-256 mismatch")
        image = Image.open(io.BytesIO(value))
        image.load()
    else:
        raise ValueError(f"unsupported image source kind: {source.get('kind')!r}")
    return image.convert("RGB")


class DeviceMemoryPeak:
    """Sample whole-device usage consistently for both inference backends."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_mib: int | None = None

    def _sample(self) -> None:
        value = backend.gpu_memory_used_mib()
        if value is not None:
            self._peak_mib = max(self._peak_mib or value, value)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, name="image-sgg-gpu-memory", daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._sample()
        return round(self._peak_mib / 1024, 3) if self._peak_mib is not None else None


def item_truth(row: dict[str, Any]) -> dict[str, Any]:
    w, h = row.get("_image_width"), row.get("_image_height")
    if not w or not h:
        raise ValueError("image dimensions are required to normalize ground-truth boxes")
    objects = []
    for obj in row["ground_truth"]["objects"]:
        x1, y1, x2, y2 = obj["box"]
        objects.append({"id": str(obj["id"]), "label": obj["label"],
                        "box": [x1 / w, y1 / h, x2 / w, y2 / h]})
    relations = [{"subject": str(r["subject"]), "predicate": r["predicate"],
                  "object": str(r["object"])} for r in row["ground_truth"]["relations"]]
    return {"objects": objects, "relations": relations}


def model_entries() -> list[dict[str, Any]]:
    entries = backend.list_original_models() + backend.list_gguf_models()
    seen = set()
    ready = []
    for entry in entries:
        identity = entry["id"]
        if identity in seen:
            continue
        seen.add(identity)
        if entry.get("snapshot") and not Path(entry["snapshot"]).is_dir():
            continue
        if entry.get("model_path") and (not Path(entry["model_path"]).is_file()
                                         or not Path(entry["mmproj_path"]).is_file()):
            continue
        ready.append(entry)
    # Run checkpoints using the existing Transformers backend first. GGUF
    # entries follow and launch the local llama.cpp binary directly in host mode.
    return sorted(ready, key=lambda item: (bool(item.get("repo")), item["id"].casefold()))


def discover_requested_model_folders() -> list[str]:
    return [
        "models--Qwen--Qwen2-VL-7B-Instruct",
        "models--Qwen--Qwen2.5-VL-7B-Instruct",
        "models--unsloth--Qwen3-VL-8B-Instruct-GGUF",
        "models--unsloth--Qwen3-VL-30B-A3B-Thinking-GGUF",
        "models--unsloth--Qwen3-VL-30B-A3B-Thinking-1M-GGUF",
        "models--bartowski--Qwen2-VL-72B-Instruct-GGUF",
        "models--zmli--G-Substrate-Qwen3-VL-2B",
    ]


def folder_coverage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    roots = {Path(e.get("snapshot") or e.get("model_path", "")).resolve() for e in entries}
    present = {path for path in roots if str(path) not in {"", "."}}
    all_roots = [Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / x
                 for x in discover_requested_model_folders()]
    coverage = {}
    for repo in all_roots:
        matched = any(repo.resolve() == p or repo.resolve() in p.parents for p in present)
        coverage[str(repo)] = {"present": repo.exists(), "runnable_entry": matched}
    return coverage


def prepare_run(output_dir: Path, dataset_names: list[str], model: dict[str, Any],
                data_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base = data_dir or HERE / "data"
    manifests: dict[str, Any] = {}
    for dataset in dataset_names:
        path = base / dataset / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared panel manifest: {path}")
        manifest, rows = load_panel(path)
        if len(rows) != 200:
            raise ValueError(f"{dataset} manifest contains {len(rows)} items, expected 200")
        manifests[dataset] = {"path": path, "manifest": manifest, "rows": rows,
                              "hash": digest_file(path)}
    revision = Path(model.get("snapshot") or model.get("model_path", "")).name
    record = {
        "schema_version": 1,
        "model_id": model["id"],
        "model_revision": revision,
        "backend": model["backend"],
        "model_snapshot": model.get("snapshot"),
        "model_path": model.get("model_path"),
        "mmproj_path": model.get("mmproj_path"),
        "prompt_version": PROMPT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "decoding": {"do_sample": False, "temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
        "backend_runtime": ({
            "server_binary": str(backend.LLAMA_SERVER),
            "context_size_tokens": CONTEXT_SIZE,
            "image_min_tokens": IMAGE_MIN_TOKENS,
            "parallel": 1,
            "fit": "on",
            "fit_target_gib": MEMORY_CAP_GIB,
        "gpu_layers": "automatic",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "speculative_decoding": "disabled",
        } if model.get("repo") else {
            "device_map": "single CUDA device",
            "attention": "sdpa",
            "kv_cache": "HQQ int8",
        }),
        "datasets": {},
    }
    return record, manifests


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def add_item(record: dict[str, Any], dataset: str, row: dict[str, Any], raw: str | None,
             prompt: str, vocabulary_hash: str, manifest_hash: str, source_hashes: dict[str, str],
             image_dims: list[int] | None, latency: float | None, input_tokens: int | None,
             output_tokens: int | None, peak_gib: float | None, error: str | None = None,
             image_processing: dict[str, Any] | None = None) -> None:
    vocab = row["vocabulary"]
    scored = score_item(raw, item_truth(row), set(vocab["objects"]), set(vocab["predicates"]), error)
    item = {
        "dataset": dataset,
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "backend": record["backend"],
        "split": row.get("source_split"),
        "image_id": row.get("image_id", row.get("id")),
        "image_sha256": row.get("image_sha256"),
        "manifest_hash": manifest_hash,
        "source_hashes": source_hashes,
        "prompt_hash": prompt_hash(prompt),
        "vocabulary_hash": vocabulary_hash,
        "evaluator_hash": evaluator_source_hash(),
        "image_dimensions": image_dims,
        "image_processing": image_processing,
        "raw_response": raw,
        "json_valid": scored["json_valid"],
        "parse_error": scored.get("error"),
        "parsed_graph": scored.get("prediction"),
        "score": scored.get("score"),
        "latency_s": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "peak_vram_gib": peak_gib,
        "peak_vram_method": "whole-device nvidia-smi samples every 0.25 seconds; sampled estimate may miss brief peaks",
        "error": error,
    }
    record["datasets"].setdefault(dataset, {"requested_n": 200, "items": []})["items"].append(item)


def transformers_image_processing(processor: Any, model: Any, inputs: Any,
                                  source_dimensions: list[int]) -> dict[str, Any]:
    """Record processor resizing/grid behavior and the actual visual input shape."""
    image_processor = getattr(processor, "image_processor", None)
    def plain_setting(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return plain_setting(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): plain_setting(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain_setting(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "tolist"):
            try:
                return plain_setting(value.tolist())
            except (TypeError, ValueError):
                pass
        if hasattr(value, "item"):
            try:
                return plain_setting(value.item())
            except (TypeError, ValueError):
                pass
        try:
            return plain_setting(dict(value))
        except (TypeError, ValueError):
            return str(value)

    info: dict[str, Any] = {
        "backend": "transformers",
        "source_dimensions": source_dimensions,
        "processor": type(image_processor).__name__ if image_processor is not None else None,
        "configured_size": plain_setting(getattr(image_processor, "size", None)),
        "configured_crop_size": plain_setting(getattr(image_processor, "crop_size", None)),
        "configured_min_pixels": plain_setting(getattr(image_processor, "min_pixels", None)),
        "configured_max_pixels": plain_setting(getattr(image_processor, "max_pixels", None)),
        "processor_output_shapes": {},
    }
    for key in ("pixel_values", "image_grid_thw"):
        tensor = inputs.get(key)
        if tensor is None:
            continue
        info["processor_output_shapes"][key] = list(tensor.shape)
        if key == "image_grid_thw":
            info["image_grid_thw"] = tensor.detach().cpu().tolist()
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    merge_size = int(getattr(vision_config, "spatial_merge_size", 1) or 1)
    if info.get("image_grid_thw"):
        info["spatial_merge_size"] = merge_size
        info["visual_token_count"] = sum(
            int(t * h * w) // (merge_size * merge_size)
            for t, h, w in info["image_grid_thw"]
        )
    elif "pixel_values" in info["processor_output_shapes"]:
        info["visual_token_count"] = info["processor_output_shapes"]["pixel_values"][0]
    return plain_setting(info)


def gguf_image_processing(source_dimensions: list[int]) -> dict[str, Any]:
    return {
        "backend": "llama.cpp",
        "source_dimensions": source_dimensions,
        "encoding": "RGB PNG data URL",
        "resize_crop_policy": "model/mmproj defaults; chat API does not expose processed crop dimensions or max-pixel settings",
    }


def transformers_model_class(model_id: str):
    low = model_id.lower()
    from transformers import (AutoModelForImageTextToText, Qwen2VLForConditionalGeneration,
                              Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration)
    if low.startswith("qwen/qwen2-vl-"):
        return Qwen2VLForConditionalGeneration
    if low.startswith("qwen/qwen2.5-vl-"):
        return Qwen2_5_VLForConditionalGeneration
    if "qwen3-vl-" in low or "g-substrate" in low:
        return Qwen3VLForConditionalGeneration
    return AutoModelForImageTextToText


def release_local_gate(*, require_llama: bool = False) -> None:
    """Run the legacy managed-mode release gate and confirm control-app state."""
    global RUNTIME_GATE
    pipeline = Path(os.environ.get("PIPELINE_DIR", "/home/nils/Project_Code/codex_opencode_pipeline"))
    executor = pipeline / "src" / "executor.py"
    if not executor.is_file():
        raise RuntimeError(f"model-control executor is missing: {executor}")
    project_root = next((parent for parent in Path(__file__).resolve().parents
                         if (parent / "pyproject.toml").is_file()), None)
    if project_root is None:
        raise RuntimeError("cannot locate the workflow checkout containing pyproject.toml")
    result = subprocess.run([sys.executable, str(executor), "release-local"],
                            cwd=project_root,
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"release-local failed ({result.returncode}): {result.stderr[-2000:]}")
    import urllib.request
    state = {}
    for endpoint in ("status", "models"):
        request = urllib.request.Request(f"http://127.0.0.1:20268/{endpoint}")
        with urllib.request.urlopen(request, timeout=10) as response:
            state[endpoint] = json.loads(response.read().decode("utf-8"))
    if not isinstance(state["status"], dict) or not isinstance(state["models"], dict):
        raise RuntimeError("model-control returned an invalid readiness response")
    if require_llama and not isinstance(state["models"].get("llama_cpp"), dict):
        raise RuntimeError("model-control app lacks the managed llama.cpp lifecycle routes")
    llama = state["status"].get("llama_server")
    if require_llama and (not isinstance(llama, dict)
                          or type(llama.get("running")) is not bool):
        raise RuntimeError("model-control app lacks the managed llama.cpp status route")
    if isinstance(llama, dict) and llama.get("running"):
        raise RuntimeError("release-local left an app-managed llama-server active")
    ollama = state["status"].get("ollama")
    unmatched = ollama.get("unmatched_gpu_processes", []) if isinstance(ollama, dict) else None
    if not isinstance(unmatched, list):
        raise RuntimeError("model-control status lacks the unmatched GPU process inventory")
    if unmatched:
        raise RuntimeError(f"release-local left unmatched GPU processes: {unmatched}")
    RUNTIME_GATE = {"release_local_stdout": result.stdout[-2000:], "control_app": state}
    print("MODEL_CONTROL_GATE confirmed", flush=True)


def configure_gpu_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable; image SGG inference requires CUDA")
    baseline = backend.gpu_memory_used_mib()
    if baseline is None:
        raise RuntimeError("could not measure idle whole-device GPU memory")
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    backend.GPU_BASELINE_GIB = baseline / 1024
    backend.MEMORY_CAP_GIB = MEMORY_CAP_GIB
    total_gib = total / 1024**3
    free_gib = total_gib - backend.GPU_BASELINE_GIB
    RUNTIME_GATE.update({"gpu_name": torch.cuda.get_device_name(device),
                         "idle_whole_device_gib": round(backend.GPU_BASELINE_GIB, 3),
                         "free_whole_device_gib": round(free_gib, 3),
                         "total_device_gib": round(total_gib, 3),
                         "whole_device_cap_gib": MEMORY_CAP_GIB})
    if total_gib < MEMORY_CAP_GIB:
        raise RuntimeError(f"{MEMORY_CAP_GIB} GiB cap exceeds device memory ({total_gib:.3f} GiB)")
    if free_gib < MEMORY_CAP_GIB:
        raise RuntimeError(
            f"free GPU memory is {free_gib:.3f} GiB; waiting for {MEMORY_CAP_GIB:.1f} GiB"
        )
    budget = max(0.25, MEMORY_CAP_GIB - backend.GPU_BASELINE_GIB - 0.5)
    torch.cuda.set_per_process_memory_fraction(budget * 1024**3 / total, device)
    RUNTIME_GATE["allocator_budget_gib"] = round(budget, 3)


def configure_direct_gpu_runtime() -> None:
    """Gate direct host execution on GPU 0's measured whole-device headroom."""
    import torch

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        fields = [field.strip() for field in output.splitlines()[0].split(",")]
        if len(fields) != 3:
            raise ValueError(f"expected 3 GPU fields, got {fields!r}")
        name, total_text, used_text = fields
        total_mib, used_mib = int(total_text), int(used_text)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError("cannot establish direct host GPU 0 memory headroom") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 0 is unavailable to CUDA; direct SGG requires CUDA")
    device = torch.cuda.current_device()
    if device != 0:
        raise RuntimeError(f"CUDA selected device {device}, but direct host mode requires GPU 0")
    cuda_total_mib = torch.cuda.get_device_properties(device).total_memory / 1024**2
    cuda_name = torch.cuda.get_device_name(device)
    if cuda_name.casefold() != name.casefold():
        raise RuntimeError(f"nvidia-smi GPU 0 is {name!r}, but CUDA device 0 is {cuda_name!r}")
    if abs(cuda_total_mib - total_mib) > 1024:
        raise RuntimeError(
            f"nvidia-smi GPU 0 memory ({total_mib} MiB) does not match CUDA device 0 "
            f"({cuda_total_mib:.0f} MiB)"
        )
    # CUDA excludes a small amount of device-reserved memory from its reported
    # capacity. Use the lower reading so both the free-memory gate and allocator
    # budget remain conservative.
    effective_total_mib = min(total_mib, int(cuda_total_mib))
    total_gib = effective_total_mib / 1024
    used_gib = used_mib / 1024
    free_gib = (effective_total_mib - used_mib) / 1024
    RUNTIME_GATE.update({
        "launch_mode": "direct-host",
        "model_control": "not used",
        "gpu_name": name,
        "nvidia_smi_total_gib": round(total_mib / 1024, 3),
        "cuda_usable_total_gib": round(cuda_total_mib / 1024, 3),
        "effective_total_mib": effective_total_mib,
        "baseline_used_mib": used_mib,
        "idle_whole_device_gib": round(used_gib, 3),
        "free_whole_device_gib": round(free_gib, 3),
        "total_device_gib": round(total_gib, 3),
        "whole_device_cap_gib": MEMORY_CAP_GIB,
    })
    if total_gib < MEMORY_CAP_GIB:
        raise RuntimeError(
            f"{MEMORY_CAP_GIB} GiB cap exceeds GPU 0 memory ({total_gib:.3f} GiB)"
        )
    if free_gib < MEMORY_CAP_GIB:
        raise RuntimeError(
            f"free GPU memory is {free_gib:.3f} GiB; waiting for {MEMORY_CAP_GIB:.1f} GiB"
        )
    backend.GPU_BASELINE_GIB = used_gib
    backend.MEMORY_CAP_GIB = MEMORY_CAP_GIB
    budget = max(0.25, MEMORY_CAP_GIB - used_gib - 0.5)
    torch.cuda.set_per_process_memory_fraction(budget / total_gib, device)
    RUNTIME_GATE["allocator_budget_gib"] = round(budget, 3)


def model_control_busy(state: dict[str, Any], *, require_llama: bool) -> bool:
    ollama = state.get("ollama") or {}
    vllm = state.get("vllm") or {}
    unsloth = state.get("unsloth") or {}
    djev = state.get("djev") or {}
    jobs = state.get("jobs")
    llama = state.get("llama_server")
    llama_status_valid = (
        isinstance(llama, dict) and type(llama.get("running")) is bool
        if require_llama or llama is not None else True
    )
    if (not isinstance(state.get("ollama_loaded"), list)
            or type(state.get("vllm_listening")) is not bool
            or type(state.get("unsloth_listening")) is not bool
            or not llama_status_valid
            or not all(isinstance(x, dict) for x in (ollama, vllm, unsloth, djev, jobs))):
        raise RuntimeError("model-control returned an invalid readiness status")
    if (not isinstance(ollama.get("unmatched_gpu_processes"), list)
            or not isinstance(vllm.get("processes"), list)
            or not isinstance(unsloth.get("pids"), list)
            or type(djev.get("upstream_listening")) is not bool
            or type(djev.get("structured_listening")) is not bool
            or not isinstance(djev.get("process_groups_alive"), list)
            or any(not isinstance(job, dict) or not isinstance(job.get("status"), str)
                   for job in jobs.values())):
        raise RuntimeError("model-control readiness status is missing engine fields")
    return bool(
        state.get("ollama_loaded")
        or ollama.get("unmatched_gpu_processes")
        or state.get("vllm_listening") or vllm.get("processes")
        or state.get("unsloth_listening") or unsloth.get("pids")
        or (llama or {}).get("running")
        or djev.get("upstream_listening") is True
        or djev.get("structured_listening") is True
        or djev.get("process_groups_alive") not in (None, [])
        or any(isinstance(job, dict) and job.get("status") not in {"done", "error"}
               for job in jobs.values())
    )


def wait_for_control_idle(entry: dict[str, Any], interval_s: float) -> tuple[dict[str, Any], bool]:
    """Poll read-only app state while busy; do not repeat unload during occupancy."""
    observed_busy = False
    while True:
        models = model_control_request("/models")
        if not isinstance(models, dict):
            raise RuntimeError("model-control returned an invalid model inventory")
        if entry.get("repo") and not isinstance(models.get("llama_cpp"), dict):
            raise RuntimeError("model-control app lacks the managed llama.cpp lifecycle routes")
        state = model_control_request("/status")
        if not isinstance(state.get("gpu_memory_used_mib"), (int, float)):
            raise RuntimeError("model-control status lacks GPU memory telemetry")
        if not model_control_busy(state, require_llama=bool(entry.get("repo"))):
            return state, observed_busy
        observed_busy = True
        print(f"WAIT_FOR_GPU {entry['id']}: model-control reports active GPU work; "
              f"checking again in {interval_s:g}s", flush=True)
        time.sleep(interval_s)


def wait_for_llama_routes(interval_s: float) -> None:
    """Retry control-app outages; fail when a healthy inventory lacks GGUF routes."""
    while True:
        try:
            models = model_control_request("/models")
        except RuntimeError as exc:
            print(f"WAIT_FOR_LLAMA_ROUTES: control app unavailable ({type(exc).__name__}); "
                  f"retrying in {interval_s:g}s", flush=True)
            time.sleep(interval_s)
            continue
        if not isinstance(models, dict):
            raise RuntimeError("model-control returned an invalid model inventory")
        if not isinstance(models.get("llama_cpp"), dict):
            raise RuntimeError("model-control app lacks the managed llama.cpp lifecycle routes")
        return


def wait_until_runtime_ready(entry: dict[str, Any], *, wait_for_gpu: bool,
                             interval_s: float, direct_host: bool = False) -> None:
    """Wait for the whole-device VRAM budget before loading a model."""
    if direct_host:
        while True:
            try:
                configure_direct_gpu_runtime()
                RUNTIME_GATE["launch_mode"] = "direct-host"
                RUNTIME_GATE["model_control"] = "not used"
                return
            except RuntimeError as exc:
                if not wait_for_gpu or "free GPU memory is " not in str(exc):
                    raise
                free_gib = RUNTIME_GATE.get("free_whole_device_gib")
                print(f"WAIT_FOR_GPU {entry['id']}: {free_gib} GiB free; "
                      f"waiting for {MEMORY_CAP_GIB:.1f} GiB", flush=True)
                time.sleep(interval_s)

    while True:
        try:
            release_local_gate(require_llama=bool(entry.get("repo")))
        except RuntimeError as exc:
            if (wait_for_gpu and entry.get("repo")
                    and "model-control app lacks the managed llama.cpp" in str(exc)):
                wait_for_llama_routes(interval_s)
                continue
            if not wait_for_gpu or "release-local failed" not in str(exc):
                raise
            if entry.get("repo"):
                models = model_control_request("/models")
                if not isinstance(models, dict):
                    raise RuntimeError("model-control returned an invalid model inventory")
                if not isinstance(models.get("llama_cpp"), dict):
                    wait_for_llama_routes(interval_s)
                    continue
            _, observed_busy = wait_for_control_idle(entry, interval_s)
            if not observed_busy:
                raise RuntimeError(
                    "release gate failed while model-control reports idle; refusing to retry unload"
                ) from exc
            print(f"WAIT_FOR_GPU {entry['id']}: release gate did not confirm idle; "
                  "retrying after the active work clears", flush=True)
            continue
        try:
            configure_gpu_runtime()
            return
        except RuntimeError as exc:
            if not wait_for_gpu or "free GPU memory is " not in str(exc):
                raise
            total_gib = RUNTIME_GATE.get("total_device_gib")
            if not isinstance(total_gib, (int, float)):
                raise RuntimeError("cannot wait for GPU headroom without device capacity") from exc
            while True:
                state, _ = wait_for_control_idle(entry, interval_s)
                free_gib = total_gib - float(state["gpu_memory_used_mib"]) / 1024
                if free_gib >= MEMORY_CAP_GIB:
                    break
                print(f"WAIT_FOR_GPU {entry['id']}: {free_gib:.3f} GiB free; "
                      f"waiting for {MEMORY_CAP_GIB:.1f} GiB", flush=True)
                time.sleep(interval_s)


def run_transformers_model(entry: dict[str, Any], panels: dict[str, Any], out_path: Path,
                           data_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    record, manifests = prepare_run(out_path, list(panels), entry, data_dir)
    snapshot = Path(entry["snapshot"])
    model_id = entry["id"]
    device = torch.cuda.current_device()
    model_cls = transformers_model_class(model_id)
    model = model_cls.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=model_id.lower().startswith("zmli/"), device_map={"": device},
        dtype="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=model_id.lower().startswith("zmli/"))
    model.eval()
    source_hashes = {d: content_hash(x["manifest"].get("source_provenance", {}))
                     for d, x in manifests.items()}
    try:
        for dataset, panel in panels.items():
            manifest_info = manifests[dataset]
            for i, row in enumerate(panel):
                started = time.perf_counter()
                raw = None
                image_dims = None
                image_processing = None
                input_tokens = output_tokens = None
                error = None
                memory_peak = DeviceMemoryPeak()
                memory_peak.start()
                try:
                    image = image_from_row(row)
                    image_dims = list(image.size)
                    labels = row["vocabulary"]
                    prompt = prompt_for(labels["objects"], labels["predicates"], DATASETS[dataset])
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image}, {"type": "text", "text": prompt}
                    ]}]
                    rendered = processor.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    inputs = processor(text=[rendered], images=images, videos=videos,
                        padding=True, return_tensors="pt").to(model.device)
                    image_processing = transformers_image_processing(processor, model, inputs, image_dims)
                    input_tokens = int(inputs.input_ids.shape[-1])
                    torch.manual_seed(3407 + i)
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False, cache_implementation="quantized",
                            cache_config={"backend": "hqq", "nbits": 8, "residual_length": 128})
                    new_ids = output[:, inputs.input_ids.shape[-1]:]
                    output_tokens = int(new_ids.shape[-1])
                    raw = processor.batch_decode(new_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0].strip()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    peak_gib = memory_peak.stop()
                latency = time.perf_counter() - started
                prompt = prompt_for(row["vocabulary"]["objects"], row["vocabulary"]["predicates"], DATASETS[dataset])
                add_item(record, dataset, row, raw, prompt, row["_vocabulary_hash"],
                         manifest_info["hash"], {dataset: source_hashes[dataset]}, image_dims, latency,
                         input_tokens, output_tokens, peak_gib, error,
                         image_processing=image_processing)
                write_json_atomic(out_path, record)
                if (i + 1) % 10 == 0 or i + 1 == len(panel):
                    print(f"PROGRESS {model_id} {dataset} {i + 1}/{len(panel)}", flush=True)
    finally:
        del model, processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return record


def request_llama(port: int, model_id: str, image: Any, prompt: str) -> tuple[str, int | None, int | None]:
    import urllib.error
    from benchmark_runners import image_data_url

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    usage = payload.get("usage", {})
    return choice["message"]["content"], usage.get("prompt_tokens"), usage.get("completion_tokens")


def model_control_request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:20268{path}", data=body,
        headers={"X-Model-Unload": "1", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"model-control request {path} failed: {type(exc).__name__}: {exc}") from exc


def wait_model_control_job(name: str, timeout: float | None,
                           request_id: str | None = None) -> dict[str, Any]:
    deadline = time.monotonic() + timeout if timeout is not None else None
    next_notice = time.monotonic() + 60
    while deadline is None or time.monotonic() < deadline:
        try:
            state = model_control_request("/status")
        except Exception as exc:
            if deadline is not None and time.monotonic() >= deadline:
                raise
            if time.monotonic() >= next_notice:
                print(f"MODEL_CONTROL_WAIT {name}: status unavailable ({type(exc).__name__})", flush=True)
                next_notice = time.monotonic() + 60
            time.sleep(2)
            continue
        job = state.get("jobs", {}).get(name, {})
        if request_id is not None and job.get("request_id") != request_id:
            raise RuntimeError(f"model-control job {name} is not attributable to this request")
        if job.get("status") == "done":
            return job.get("detail", {})
        if job.get("status") == "error":
            raise RuntimeError(f"model-control job {name} failed: {job.get('error')}")
        if time.monotonic() >= next_notice:
            print(f"MODEL_CONTROL_WAIT {name}: still running", flush=True)
            next_notice = time.monotonic() + 60
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for model-control job {name}")


def start_llama(entry: dict[str, Any]) -> dict[str, Any]:
    global ACTIVE_LLAMA
    expected = {"model_path": entry["model_path"], "mmproj_path": entry["mmproj_path"]}
    request_id = uuid.uuid4().hex
    try:
        model_control_request("/llama/start", {
            **expected, "context_size": CONTEXT_SIZE, "request_id": request_id,
        })
        detail = wait_model_control_job("llama-start", timeout=480, request_id=request_id)
        state = model_control_request("/status").get("llama_server", {})
        if (not state.get("running") or state.get("pid") != detail.get("pid")
                or state.get("model_path") != expected["model_path"]
                or state.get("mmproj_path") != expected["mmproj_path"]
                or state.get("request_id") != request_id
                or detail.get("request_id") != request_id):
            raise RuntimeError(f"model-control reports llama-server is not active: {state}")
        ACTIVE_LLAMA = {**expected, "pid": detail["pid"], "request_id": request_id}
        return detail
    except BaseException as start_error:
        try:
            current = model_control_request("/status")
            start_job = current.get("jobs", {}).get("llama-start", {})
            if start_job.get("request_id") == request_id:
                if start_job.get("status") == "running":
                    wait_model_control_job("llama-start", timeout=None, request_id=request_id)
                    current = model_control_request("/status")
                    start_job = current.get("jobs", {}).get("llama-start", {})
                if start_job.get("status") == "done":
                    detail = start_job.get("detail", {})
                    owned = {**expected, "pid": detail.get("pid"), "request_id": request_id}
                    current_server = current.get("llama_server", {})
                    if (owned["pid"] is not None
                            and all(current_server.get(key) == value for key, value in owned.items())):
                        stop_llama(owned)
                elif start_job.get("status") == "error":
                    current_server = current.get("llama_server", {})
                    owned = {
                        **expected,
                        "pid": current_server.get("pid"),
                        "request_id": request_id,
                    }
                    if (owned["pid"] is not None
                            and all(current_server.get(key) == value for key, value in owned.items())):
                        stop_llama(owned)
        except Exception as cleanup_error:
            raise RuntimeError(
                f"GGUF start failed ({type(start_error).__name__}: {start_error}); "
                f"model-control cleanup could not be confirmed ({type(cleanup_error).__name__}: {cleanup_error})"
            ) from start_error
        raise


def wait_direct_llama_ready(proc: subprocess.Popen, port: int, log_path: Path,
                            timeout: float = 1800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with {proc.returncode}: "
                f"{log_path.read_text(encoding='utf-8', errors='replace')[-5000:]}"
            )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(
        f"llama-server did not become ready within {timeout:g}s: "
        f"{log_path.read_text(encoding='utf-8', errors='replace')[-5000:]}"
    )


def start_direct_llama(entry: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    server_binary = Path(backend.LLAMA_SERVER)
    if not server_binary.is_file() or not os.access(server_binary, os.X_OK):
        raise FileNotFoundError(f"llama-server is unavailable or not executable: {server_binary}")
    port = backend.free_port()
    effective_total_mib = RUNTIME_GATE.get("effective_total_mib")
    baseline_used_mib = RUNTIME_GATE.get("baseline_used_mib")
    if not isinstance(effective_total_mib, int) or not isinstance(baseline_used_mib, int):
        raise RuntimeError("direct GPU gate did not record conservative fit-target inputs")
    allowed_model_mib = max(0, int(MEMORY_CAP_GIB * 1024) - baseline_used_mib)
    fit_target_mib = max(512, effective_total_mib - allowed_model_mib)
    command = [
        str(server_binary), "--model", entry["model_path"], "--mmproj", entry["mmproj_path"],
        "--ctx-size", str(CONTEXT_SIZE), "--parallel", "1", "--no-cont-batching",
        "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--image-min-tokens", str(IMAGE_MIN_TOKENS), "--jinja",
        "--host", "127.0.0.1", "--port", str(port),
        "--fit", "on", "--fit-target", str(fit_target_mib),
    ]
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", entry["id"])
    log_path = output_dir / f"{safe_id}.llama-server.log"
    runtime_env = os.environ.copy()
    runtime_env["GGML_BACKEND_PATH"] = "/usr/local/lib/ollama/cuda_v13/libggml-cuda.so"
    runtime_env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v13:/usr/local/lib/ollama:" + runtime_env.get("LD_LIBRARY_PATH", "")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                text=True, env=runtime_env, start_new_session=True)
    try:
        wait_direct_llama_ready(proc, port, log_path)
    except BaseException:
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        raise
    return {"process": proc, "pid": proc.pid, "port": port,
            "fit_target_mib": fit_target_mib, "server_log": str(log_path),
            "server_binary": str(server_binary)}


def stop_direct_llama(server: dict[str, Any]) -> None:
    proc = server["process"]
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)
    if proc.poll() is None:
        raise RuntimeError(f"direct llama-server process {proc.pid} did not stop")


def install_direct_signal_handlers() -> None:
    """Turn session termination into Python cleanup for the owned server child."""
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}; stopping the owned model child")

    signal.signal(signal.SIGTERM, interrupt)


def stop_llama(expected: dict[str, Any] | None = None) -> None:
    global ACTIVE_LLAMA
    expected = expected or ACTIVE_LLAMA
    if expected is None:
        state = model_control_request("/status").get("llama_server", {})
        if state.get("running"):
            raise RuntimeError("refusing to stop llama-server without a benchmark-owned request ID")
        return
    last_error = None
    for attempt in range(3):
        state = model_control_request("/status")
        server = state.get("llama_server", {})
        if not server.get("running"):
            ACTIVE_LLAMA = None
            return
        if any(server.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"refusing to stop a different llama-server model: {server}")
        request_error = None
        try:
            model_control_request("/llama/stop", {
                "request_id": expected.get("request_id"), "pid": expected.get("pid"),
            })
        except Exception as exc:
            request_error = exc
        job = state.get("jobs", {}).get("llama-stop", {})
        if (job.get("status") != "running"
                or job.get("request_id") != expected.get("request_id")):
            try:
                latest = model_control_request("/status")
                job = latest.get("jobs", {}).get("llama-stop", {})
            except Exception:
                if request_error:
                    raise request_error
                raise
        try:
            if job.get("request_id") == expected.get("request_id"):
                wait_model_control_job("llama-stop", timeout=60,
                                       request_id=expected.get("request_id"))
        except Exception as exc:
            last_error = exc
        after = model_control_request("/status").get("llama_server", {})
        if not after.get("running"):
            ACTIVE_LLAMA = None
            return
        if any(after.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"llama-server ownership changed during stop: {after}")
        if attempt == 2:
            raise RuntimeError(
                f"model-control could not confirm llama-server shutdown: {after}; "
                f"request_error={request_error}; job_error={last_error}"
            )
    raise RuntimeError(f"model-control could not confirm llama-server shutdown: {last_error}")


def run_gguf_model(entry: dict[str, Any], panels: dict[str, Any], out_path: Path,
                   data_dir: Path, *, direct_host: bool = False,
                   existing_record: dict[str, Any] | None = None) -> dict[str, Any]:
    record, manifests = prepare_run(out_path, list(panels), entry, data_dir)
    if existing_record is not None:
        record = existing_record
    server_started = False
    source_hashes = {d: content_hash(x["manifest"].get("source_provenance", {}))
                     for d, x in manifests.items()}
    try:
        server = start_direct_llama(entry, out_path.parent) if direct_host else start_llama(entry)
        server_started = True
        port = int(server["port"])
        record["backend_runtime"].update({
            "launch_mode": "direct-host" if direct_host else "model-control",
            "model_control": "not used" if direct_host else "127.0.0.1:20268",
            "server_pid": server["pid"],
            "server_port": port,
            "fit_target_mib": server["fit_target_mib"],
            "server_log": server.get("server_log", server.get("log")),
            "server_binary": server.get("server_binary"),
        })
        for dataset, panel in panels.items():
            mi = manifests[dataset]
            block = record["datasets"].setdefault(
                dataset, {"requested_n": 200, "items": []})
            if existing_record is not None:
                retry_items = [item for item in block["items"]
                               if item.get("error") or item.get("raw_response") is None]
                if retry_items:
                    record.setdefault("resume_retry_history", []).extend(
                        {"dataset": dataset, "item": item} for item in retry_items
                    )
                    block["items"] = [item for item in block["items"]
                                       if item not in retry_items]
            completed_ids = {str(item.get("image_id")) for item in block["items"]}
            if completed_ids:
                print(f"RESUME {entry['id']} {dataset}: "
                      f"{len(completed_ids)}/{len(panel)} items already saved", flush=True)
            for i, row in enumerate(panel):
                image_id = str(row.get("image_id", row.get("id")))
                if image_id in completed_ids:
                    continue
                raw = None
                image_dims = None
                image_processing = None
                input_tokens = output_tokens = None
                error = None
                started = time.perf_counter()
                memory_peak = DeviceMemoryPeak()
                memory_peak.start()
                try:
                    image = image_from_row(row)
                    image_dims = list(image.size)
                    image_processing = gguf_image_processing(image_dims)
                    vocab = row["vocabulary"]
                    prompt = prompt_for(vocab["objects"], vocab["predicates"], DATASETS[dataset])
                    raw, input_tokens, output_tokens = request_llama(port, entry["id"], image, prompt)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    peak_gib = memory_peak.stop()
                latency = time.perf_counter() - started
                prompt = prompt_for(row["vocabulary"]["objects"], row["vocabulary"]["predicates"], DATASETS[dataset])
                add_item(record, dataset, row, raw, prompt, row["_vocabulary_hash"],
                    mi["hash"], {dataset: source_hashes[dataset]}, image_dims, latency,
                    input_tokens, output_tokens, peak_gib, error,
                    image_processing=image_processing)
                completed_ids.add(image_id)
                write_json_atomic(out_path, record)
                if (i + 1) % 10 == 0 or i + 1 == len(panel):
                    print(f"PROGRESS {entry['id']} {dataset} {i + 1}/{len(panel)}", flush=True)
    finally:
        if server_started:
            if direct_host:
                stop_direct_llama(server)
            else:
                stop_llama()
    return record


def load_incomplete_gguf_record(path: Path, entry: dict[str, Any], data_dir: Path,
                                datasets: list[str], *, direct_host: bool) -> dict[str, Any]:
    """Load an interrupted GGUF run only when its full provenance still matches."""
    record = json.loads(path.read_text(encoding="utf-8"))
    expected, manifests = prepare_run(path, datasets, entry, data_dir)
    for key in ("model_id", "model_revision", "backend", "prompt_version",
                "evaluator_version", "decoding"):
        if record.get(key) != expected.get(key):
            raise ValueError(f"cannot resume {path}: {key} differs from current run")
    if record.get("status") or record.get("run_error") or record.get("load_error"):
        raise ValueError(f"cannot resume {path}: it contains a finalized failure/partial status")
    runtime = record.get("backend_runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"cannot resume {path}: backend runtime metadata is missing")
    expected_runtime = expected["backend_runtime"]
    stable_runtime_keys = ("server_binary", "context_size_tokens", "image_min_tokens",
                           "parallel", "fit", "fit_target_gib", "gpu_layers",
                           "cache_type_k", "cache_type_v", "speculative_decoding")
    if any(runtime.get(key) != expected_runtime.get(key) for key in stable_runtime_keys):
        raise ValueError(f"cannot resume {path}: GGUF runtime settings differ")
    if runtime.get("launch_mode") != ("direct-host" if direct_host else "model-control"):
        raise ValueError(f"cannot resume {path}: launch mode differs")

    blocks = record.get("datasets")
    if not isinstance(blocks, dict):
        raise ValueError(f"cannot resume {path}: dataset records are missing")
    current_evaluator_hash = evaluator_source_hash()
    for dataset in datasets:
        block = blocks.get(dataset)
        if block is None:
            continue
        if (not isinstance(block, dict) or block.get("requested_n") != 200
                or not isinstance(block.get("items"), list)):
            raise ValueError(f"cannot resume {path}: invalid {dataset} record")
        manifest = manifests[dataset]
        _, rows = load_panel(manifest["path"])
        expected_rows = {str(row.get("image_id", row.get("id"))): row for row in rows}
        seen: set[str] = set()
        for item in block["items"]:
            image_id = str(item.get("image_id"))
            row = expected_rows.get(image_id)
            if row is None or image_id in seen:
                raise ValueError(f"cannot resume {path}: duplicate or unknown {dataset} image {image_id}")
            seen.add(image_id)
            prompt = prompt_for(row["vocabulary"]["objects"],
                                row["vocabulary"]["predicates"], DATASETS[dataset])
            if (item.get("manifest_hash") != manifest["hash"]
                    or item.get("prompt_hash") != prompt_hash(prompt)
                    or item.get("evaluator_hash") != current_evaluator_hash
                    or item.get("model_revision") != record["model_revision"]):
                raise ValueError(f"cannot resume {path}: {dataset} item provenance differs")
    return record


def run_model(entry: dict[str, Any], data_dir: Path, output_dir: Path, datasets: list[str],
              max_items: int | None = None, *, direct_host: bool = False,
              resume_incomplete: bool = False) -> Path:
    panels = {}
    for name in datasets:
        manifest_path = data_dir / name / "manifest.json"
        manifest, rows = load_panel(manifest_path)
        panels[name] = rows
        if len(rows) != 200:
            raise ValueError(f"{name} manifest contains {len(rows)} items, expected 200")
        if max_items is not None:
            panels[name] = rows[:max_items]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", entry["id"])
    out_path = output_dir / f"{safe}.json"
    existing_record = None
    if out_path.exists():
        if not resume_incomplete:
            raise FileExistsError(
                f"result already exists; use --resume-incomplete to continue it: {out_path}"
            )
        if not entry.get("repo"):
            raise ValueError("--resume-incomplete currently supports GGUF entries only")
        existing_record = load_incomplete_gguf_record(
            out_path, entry, data_dir, datasets, direct_host=direct_host,
        )
    try:
        if entry.get("repo"):
            record = run_gguf_model(entry, panels, out_path, data_dir,
                                     direct_host=direct_host, existing_record=existing_record)
        else:
            record = run_transformers_model(entry, panels, out_path, data_dir)
        if max_items is not None and max_items < 200:
            record["status"] = "partial"
            record["partial_reason"] = f"limited to first {max_items} items per dataset"
            write_json_atomic(out_path, record)
    except Exception as exc:
        print(f"MODEL_FAILURE {entry['id']}: {type(exc).__name__}: {exc}", flush=True)
        failure = f"{type(exc).__name__}: {exc}"
        if out_path.is_file():
            record = json.loads(out_path.read_text(encoding="utf-8"))
            record["status"] = "partial_failure"
            record["run_error"] = failure
            _, manifests = prepare_run(out_path, datasets, entry, data_dir)
        else:
            record, manifests = prepare_run(out_path, datasets, entry, data_dir)
            record["status"] = "load_failed"
            record["load_error"] = failure
        for dataset, rows in panels.items():
            manifest = manifests[dataset]
            source_hashes = {dataset: content_hash(manifest["manifest"].get("source_provenance", {}))}
            result_block = record["datasets"].setdefault(dataset, {"requested_n": 200, "items": []})
            existing_ids = {str(item.get("image_id")) for item in result_block.get("items", [])}
            for row in rows:
                if str(row.get("image_id")) in existing_ids:
                    continue
                prompt = prompt_for(row["vocabulary"]["objects"], row["vocabulary"]["predicates"], DATASETS[dataset])
                add_item(record, dataset, row, None, prompt, row["_vocabulary_hash"],
                         manifest["hash"], source_hashes, None, None, None, None, None,
                         failure)
                existing_ids.add(str(row.get("image_id")))
        write_json_atomic(out_path, record)
        if entry.get("repo") and not direct_host:
            try:
                status = model_control_request("/status")
            except Exception as status_error:
                raise RuntimeError(
                    "cannot confirm model-control is idle after GGUF failure; "
                    "aborting the remaining model queue"
                ) from status_error
            server_state = status.get("llama_server", {})
            start_job = status.get("jobs", {}).get("llama-start", {})
            stop_job = status.get("jobs", {}).get("llama-stop", {})
            ollama = status.get("ollama", {})
            vllm = status.get("vllm", {})
            unsloth = status.get("unsloth", {})
            djev = status.get("djev", {})
            other_engine_active = bool(
                status.get("ollama_loaded")
                or status.get("vllm_listening") or vllm.get("processes")
                or status.get("unsloth_listening") or unsloth.get("pids")
                or djev.get("upstream_listening") is not False
                or djev.get("structured_listening") is not False
                or djev.get("process_groups_alive") != []
                or ollama.get("unmatched_gpu_processes")
            )
            if (server_state.get("running") or start_job.get("status") == "running"
                    or stop_job.get("status") == "running" or other_engine_active):
                raise RuntimeError(
                    "model-control still reports an active engine after GGUF failure; "
                    "aborting the remaining model queue"
                )
    return out_path


def validate_panels(data_dir: Path, datasets: list[str]) -> None:
    """Reject incomplete panels before probing CUDA or loading a model."""
    for dataset in datasets:
        path = data_dir / dataset / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared panel manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = manifest.get("items")
        if not isinstance(rows, list) or len(rows) != 200:
            count = len(rows) if isinstance(rows, list) else "missing"
            raise ValueError(f"{dataset} manifest contains {count} items, expected 200")


def completed_model_run(output_dir: Path, model: dict[str, Any], data_dir: Path,
                        datasets: list[str]) -> bool:
    """Recognize a full, current-provenance run so resumed batches do not repeat it."""
    model_id = model["id"]
    revision = Path(model.get("snapshot") or model.get("model_path", "")).name
    expected: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        manifest_path = data_dir / dataset / "manifest.json"
        manifest, rows = load_panel(manifest_path)
        if not rows:
            return False
        vocabulary = rows[0]["vocabulary"]
        prompt = prompt_for(vocabulary["objects"], vocabulary["predicates"], DATASETS[dataset])
        expected[dataset] = {
            "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "prompt_hash": prompt_hash(prompt),
            "image_ids": {str(row.get("image_id", row.get("id"))) for row in rows},
        }
    current_evaluator_hash = evaluator_source_hash()
    for path in output_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (record.get("model_id") != model_id
                or record.get("model_revision") != revision
                or record.get("prompt_version") != PROMPT_VERSION
                or record.get("evaluator_version") != EVALUATOR_VERSION):
            continue
        blocks = record.get("datasets", {})
        if all(
            isinstance(blocks.get(dataset), dict)
            and len(blocks[dataset].get("items", [])) == 200
            and blocks[dataset].get("requested_n") == 200
            and {str(item.get("image_id")) for item in blocks[dataset]["items"]}
                == expected[dataset]["image_ids"]
            and all(item.get("manifest_hash") == expected[dataset]["manifest_hash"]
                    and item.get("prompt_hash") == expected[dataset]["prompt_hash"]
                    and item.get("evaluator_hash") == current_evaluator_hash
                    and item.get("model_revision") == revision
                    for item in blocks[dataset]["items"])
            for dataset in datasets
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--model", action="append", help="exact cached model entry ID; repeatable")
    parser.add_argument("--_worker-model", dest="worker_model", help=argparse.SUPPRESS)
    parser.add_argument("--direct-host", action="store_true", default=True,
                        help="run directly on the host GPU (this runner always uses direct mode)")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true",
                        help="wait for the required whole-device free VRAM budget")
    parser.add_argument("--wait-interval-seconds", type=float, default=60.0,
                        help="seconds between readiness attempts when --wait-for-gpu is set")
    parser.add_argument("--max-items", type=int,
                        help="run the first N items from each fixed panel; partial runs cannot be aggregated")
    parser.add_argument("--resume-incomplete", action="store_true",
                        help="resume an interrupted GGUF record after validating its provenance")
    args = parser.parse_args()
    if args.direct_host:
        install_direct_signal_handlers()
    all_entries = model_entries()
    coverage = folder_coverage(all_entries)
    if args.list_models:
        print(json.dumps({"models": all_entries, "coverage": coverage}, indent=2))
        return
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items must be a positive integer")
    if args.wait_interval_seconds <= 0:
        raise SystemExit("--wait-interval-seconds must be positive")
    entries = all_entries
    if args.model:
        wanted = set(args.model)
        entries = [entry for entry in entries if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in entries}
        if missing:
            raise SystemExit(f"requested cached model entries not runnable: {sorted(missing)}")
    if not entries:
        raise SystemExit("no requested cached Qwen VL model entries are runnable")
    if args.worker_model:
        entries = [entry for entry in entries if entry["id"] == args.worker_model]
        if not entries:
            raise SystemExit(f"worker model is not runnable: {args.worker_model}")
    missing_folders = [path for path, entry in coverage.items() if entry["present"] and not entry["runnable_entry"]]
    if missing_folders:
        raise SystemExit(f"requested cached folders have no runnable model entry: {missing_folders}")
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    validate_panels(args.data_dir, datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "inventory.json"
    if not args.worker_model:
        inventory = {"execution_mode": "direct-host" if args.direct_host else "managed",
                     "models": entries, "folder_coverage": coverage, "runtime_gates": {}}
        write_json_atomic(inventory_path, inventory)
        # Keep the orchestrator CPU-only. Each model gets a fresh process so its
        # A fresh child per model releases its CUDA context before the next runtime gate.
        for entry in entries:
            if args.max_items is None and completed_model_run(
                    args.output_dir, entry, args.data_dir, datasets):
                print(f"SKIP {entry['id']}: complete current-provenance panels already exist",
                      flush=True)
                continue
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--_worker-model", entry["id"],
                       "--data-dir", str(args.data_dir.resolve()),
                       "--output-dir", str(args.output_dir.resolve()),
                       "--dataset", args.dataset,
                       "--wait-interval-seconds", str(args.wait_interval_seconds)]
            if args.wait_for_gpu:
                command.append("--wait-for-gpu")
            if args.direct_host:
                command.append("--direct-host")
            if args.resume_incomplete:
                command.append("--resume-incomplete")
            if args.max_items is not None:
                command.extend(("--max-items", str(args.max_items)))
            child_env = os.environ.copy()
            if not args.direct_host:
                # The scheduler owns the host queue lease for the full serial batch.
                child_env["VLM_BENCHMARK_SCHEDULER_OWNS_GPU"] = "1"
            else:
                child_env.pop("VLM_BENCHMARK_GPU_QUEUE_FD", None)
                child_env.pop("VLM_BENCHMARK_SCHEDULER_OWNS_GPU", None)
            worker = subprocess.Popen(command, env=child_env)
            try:
                return_code = worker.wait()
            except BaseException:
                worker.terminate()
                try:
                    worker.wait(timeout=75)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=15)
                raise
            if return_code:
                raise SystemExit(return_code)
        return 0

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for entry in entries:
        wait_until_runtime_ready(
            entry, wait_for_gpu=args.wait_for_gpu,
            interval_s=args.wait_interval_seconds, direct_host=args.direct_host,
        )
        inventory["runtime_gates"][entry["id"]] = RUNTIME_GATE
        write_json_atomic(inventory_path, inventory)
        print(f"START {entry['id']} ({entry['backend']})", flush=True)
        run_model(entry, args.data_dir, args.output_dir, datasets, args.max_items,
                  direct_host=args.direct_host, resume_incomplete=args.resume_incomplete)


if __name__ == "__main__":
    main()
