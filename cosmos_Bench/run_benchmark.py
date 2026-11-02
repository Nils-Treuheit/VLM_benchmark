#!/usr/bin/env python3
"""Run cached Cosmos checkpoints through the shared VLM task runners."""
from __future__ import annotations

import json
import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_DIR = ROOT / "qwenVL_Bench"
sys.path.insert(0, str(QWEN_DIR))
import benchmark_runners as runner  # noqa: E402

HF_HUB = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub")
OUT = Path(__file__).resolve().parent / "results"
GROUPS = [
    (100, ("ai2d", "textvqa")),
    (25, ("vqa_truth", "vqa_count", "vqa_description", "vqa_affordance", "vqa_negation")),
    (500, ("graph_grab_lite",)),
]
MODELS = [
    ("nvidia/Cosmos-Reason2-2B", "Cosmos-Reason2-2B"),
    ("nvidia/Cosmos-Reason2-8B", "Cosmos-Reason2-8B"),
    ("nvidia/Cosmos-Reason1-7B", "Cosmos-Reason1-7B"),
    ("nvidia/Cosmos3-Edge", "Cosmos3-Edge"),
]
PENDING = [
    {"model_id": "nvidia/Cosmos3-Nano", "revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92", "status": "pending",
     "backend": "managed vLLM 0.21.0 + vllm-cosmos3 (CUDA 13)", "managed_profile": "nv_serve_cosmos3_nano_cuda13",
     "max_model_len": 16384, "run_context_tokens": 16384, "max_num_seqs": 1, "gpu_memory_utilization": 0.92, "max_new_tokens": 4096,
     "error": "CUDA 13 vLLM 0.21.0 environment, vllm-cosmos3 plugin, and managed server profile are prepared; the benchmark is waiting for the RTX 5090 to become free."},
]


def snapshot_for(folder: str) -> Path:
    repo = HF_HUB / f"models--nvidia--{folder}"
    rev = (repo / "refs/main").read_text().strip()
    return repo / "snapshots" / rev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="restrict execution to exact model IDs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if args.models and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    else:
        result = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "samples_per_task": {"ai2d": 100, "textvqa": 100, "vqa_panel": 25, "graph_grab_lite": 500},
            "seed": 3407,
            "memory_cap_gib": 30.0,
            "task_names": [task for _, tasks in GROUPS for task in tasks],
            "execution_mode": "Transformers, offline cached snapshots, serial models and requests",
            "models": list(PENDING),
        }
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this benchmark only runs on GPU")
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if total < 30:
        raise SystemExit(f"30 GiB per-process target exceeds GPU capacity ({total:.2f} GiB)")
    torch.cuda.set_per_process_memory_fraction(30.0 / total, device)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    selected_models = [item for item in MODELS if not args.models or item[0] in args.models]
    for model_id, folder in selected_models:
        snap = snapshot_for(folder)
        result["models"] = [m for m in result["models"] if m.get("model_id") != model_id]
        record = {
            "model_id": model_id,
            "backend": "Transformers (cached checkpoint, bfloat16, HQQ int8 KV cache)",
            "snapshot": str(snap),
            "revision": snap.name,
            "tasks": {},
            "status": "in_progress",
            "max_new_tokens": 4096 if "Cosmos-Reason1" in folder else 256,
            "reasoning_mode": "checkpoint default; 4096-token cap follows the model card recommendation" if "Cosmos-Reason1" in folder else (
                "disabled with the model's enable_thinking chat-template option to follow the shared answer-only contract" if "Cosmos3" in folder else "checkpoint default"
            ),
        }
        result["models"].append(record)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        entry = {"id": model_id, "backend": record["backend"], "snapshot": str(snap)}
        try:
            for count, tasks in GROUPS:
                print(f"START {model_id} tasks={','.join(tasks)} samples={count}", flush=True)
                runner.run_transformers(
                    entry, count, 3407, result, result_path, tasks,
                    max_new_tokens_override=4096 if "Cosmos-Reason1" in folder else (256 if "Cosmos3" in folder else None),
                )
                run_record = result["models"].pop()
                record.update({k: v for k, v in run_record.items() if k not in {"model_id", "tasks"}})
                record["tasks"].update(run_record.get("tasks", {}))
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            expected = {task: n for n, tasks in GROUPS for task in tasks}
            complete = all(record["tasks"].get(task, {}).get("completed_n") == n
                           for task, n in expected.items())
            record["status"] = "complete" if complete else "complete_with_failures"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"FINISH {model_id} status={record['status']}", flush=True)

    print(f"Wrote {result_path}", flush=True)

if __name__ == "__main__":
    main()
