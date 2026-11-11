#!/usr/bin/env python3
"""Score the Cosmos3 Nano reasoner through its managed vLLM OpenAI endpoint."""
from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_DIR = ROOT / "qwenVL_Bench"
sys.path.insert(0, str(QWEN_DIR))
sys.path.insert(0, str(ROOT))
from gpu_queue import run_queued  # noqa: E402
import benchmark as common  # noqa: E402
import benchmark_runners as runner  # noqa: E402

MODEL_ID = "nvidia/Cosmos3-Nano"
SNAPSHOT = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/e59a53c25979a090fa8706c9acc0c254a6e89b92")
RESULT_PATH = Path(__file__).resolve().parent / "results" / "results.json"
BASE_URL = "http://127.0.0.1:20265/v1"
TASK_GROUPS = (
    (100, ("ai2d", "textvqa")),
    (25, ("vqa_truth", "vqa_count", "vqa_description", "vqa_affordance", "vqa_negation")),
    (500, ("graph_grab_lite",)),
)
MAX_TOKENS = 4096


def api_json(path: str, payload: dict | None = None, timeout: float = 900) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def persist(result: dict) -> None:
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_task_group(result: dict, record: dict, model_name: str, count: int,
                   task_names: tuple[str, ...], seed: int) -> None:
    with runner.MemorySampler() as memory:
        for task in task_names:
            data = runner.dataset_for(task)
            indices = common.pick_indices(len(data), count, seed)
            items, scores = [], []
            total_tokens = total_wall = 0.0
            for position, index in enumerate(indices):
                row = data[index]
                try:
                    prompt, expected = runner.prompt_for(task, row)
                    image_url = runner.image_data_url(row["image"])
                    body = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ]}],
                        "max_tokens": MAX_TOKENS,
                        "seed": seed + position,
                    }
                    began = time.perf_counter()
                    response = api_json("/chat/completions", body)
                    wall = time.perf_counter() - began
                    choice = response["choices"][0]
                    content = choice.get("message", {}).get("content") or ""
                    if isinstance(content, list):
                        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                    usage = response.get("usage", {})
                    out_tokens = int(usage.get("completion_tokens") or 0)
                    prediction, score, parse_error = runner.score_for(task, content, expected)
                    metadata = ({"pid": row["pid"], "task_type": row["task"],
                                 "dataset_category": row["category"], "complexity": row["complexity"]}
                                if task == "graph_grab_lite" else {})
                    items.append({"index": index, "score": score, "prediction": prediction,
                                  "output": content, "input_tokens": usage.get("prompt_tokens"),
                                  "output_tokens": out_tokens, "parse_error": parse_error,
                                  "wall_s": round(wall, 4), **metadata})
                    scores.append(score)
                    total_tokens += out_tokens
                    total_wall += wall
                except Exception as exc:
                    items.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
                if (position + 1) % 10 == 0 or position + 1 == len(indices):
                    print(f"PROGRESS {MODEL_ID} {task} {position + 1}/{len(indices)}", flush=True)
                if (position + 1) % 10 == 0 or position + 1 == len(indices):
                    record["tasks"][task] = {
                        "requested_n": len(indices),
                        "completed_n": sum("error" not in item for item in items),
                        "failed_n": sum("error" in item for item in items),
                        "score": statistics.mean(scores) if scores else None,
                        "mean_wall_s": statistics.mean(item["wall_s"] for item in items if "error" not in item) if scores else None,
                        "generated_tokens_per_s": total_tokens / total_wall if total_wall else None,
                        "peak_gpu_used_gib": round(memory.peak_mib / 1024, 3),
                        "peak_gpu_delta_gib": round(max(0, memory.peak_mib - 1024) / 1024, 3),
                        "items": items,
                    }
                    persist(result)


def main() -> None:
    if not SNAPSHOT.is_dir():
        raise SystemExit(f"Cached Cosmos3 Nano snapshot missing: {SNAPSHOT}")
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    result["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    result["execution_mode"] = "offline cached Transformers and managed vLLM CUDA 13, serial models and requests"
    result["models"] = [m for m in result["models"] if m.get("model_id") != MODEL_ID]
    model_list = api_json("/models")
    if not model_list.get("data"):
        raise RuntimeError("Managed vLLM endpoint returned no served model")
    model_name = model_list["data"][0]["id"]
    record = {
        "model_id": MODEL_ID,
        "backend": "managed vLLM 0.21.0 (CUDA 13) + vllm-cosmos3",
        "snapshot": str(SNAPSHOT),
        "revision": SNAPSHOT.name,
        "tasks": {},
        "status": "in_progress",
        "max_new_tokens": MAX_TOKENS,
        "reasoning_mode": "checkpoint default; 4096-token cap follows the model card recommendation",
        "gpu_memory_utilization": 0.88,
        "max_model_len": 16384,
        "run_context_tokens": 16384,
        "max_num_seqs": 1,
        "managed_profile": "nv_serve_cosmos3_nano_cuda13",
    }
    result["models"].append(record)
    persist(result)
    for count, tasks in TASK_GROUPS:
        print(f"START {MODEL_ID} tasks={','.join(tasks)} samples={count}", flush=True)
        run_task_group(result, record, model_name, count, tasks, 3407)
    expected = {task: n for n, tasks in TASK_GROUPS for task in tasks}
    complete = all(record["tasks"].get(task, {}).get("completed_n") == n
                   for task, n in expected.items())
    record["status"] = "complete" if complete else "complete_with_failures"
    task_peaks = [task.get("peak_gpu_used_gib") for task in record["tasks"].values()]
    record["peak_gpu_used_gib"] = max((v for v in task_peaks if v is not None), default=None)
    persist(result)
    print(f"FINISH {MODEL_ID} status={record['status']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(run_queued(main, "cosmos-nano-vllm"))
