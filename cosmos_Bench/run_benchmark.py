#!/usr/bin/env python3
"""Run screened cached Cosmos checkpoints under a measured whole-device cap."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
QWEN_DIR = ROOT / "qwenVL_Bench"
sys.path.insert(0, str(QWEN_DIR))
sys.path.insert(0, str(ROOT))
from gpu_queue import run_queued  # noqa: E402
import benchmark_runners as runner  # noqa: E402

HF_HUB = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub")
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
CAPS = (8, 10, 12, 14, 16)
MODELS = [
    ("nvidia/Cosmos-Reason2-2B", "Cosmos-Reason2-2B"),
    ("nvidia/Cosmos-Reason2-8B", "Cosmos-Reason2-8B"),
    ("nvidia/Cosmos-Reason1-7B", "Cosmos-Reason1-7B"),
    ("nvidia/Cosmos3-Edge", "Cosmos3-Edge"),
]


def snapshot_for(folder: str) -> Path:
    repo = HF_HUB / f"models--nvidia--{folder}"
    rev = (repo / "refs/main").read_text(encoding="utf-8").strip()
    return repo / "snapshots" / rev


def screened_model_ids(cap_gib: int) -> set[str]:
    roster = json.loads((ROOT / "edge-refinement-candidates.json").read_text(encoding="utf-8"))
    stage = next((item for item in roster.get("caps", []) if item.get("cap_gib") == cap_gib), None)
    if stage is None:
        raise ValueError(f"candidate roster has no {cap_gib} GiB stage")
    return {item["model_id"] for item in stage.get("candidates", []) if item.get("suite") == "Cosmos"}


def gpu_identity() -> tuple[str, str]:
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL, timeout=3,
    ).strip().splitlines()[0]
    name, driver = line.split(", ", 1)
    return name, driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="restrict to exact screened model IDs")
    parser.add_argument("--whole-device-cap-gib", type=int, choices=CAPS, required=True,
                        help="maximum whole-device GPU use, including the desktop baseline")
    parser.add_argument("--context-probe-limit", type=int, default=262144,
                        help="maximum token context considered by the shared real-image probe")
    parser.add_argument("--profile-context-only", action="store_true",
                        help="load the selected model and run only the real-image context probe")
    parser.add_argument("--output-dir", type=Path,
                        help="cap-specific artifact directory; defaults under results/edge_refinement")
    args = parser.parse_args()
    if args.context_probe_limit < 1024:
        parser.error("--context-probe-limit must be at least 1024 tokens")

    candidates = screened_model_ids(args.whole_device_cap_gib)
    known = {model_id for model_id, _ in MODELS}
    selected_ids = candidates if args.models is None else set(args.models)
    unknown = selected_ids - known
    if unknown:
        parser.error("unknown Cosmos model ID(s): " + ", ".join(sorted(unknown)))
    unscreened = selected_ids - candidates
    if unscreened:
        parser.error(f"not screened for {args.whole_device_cap_gib} GiB: " + ", ".join(sorted(unscreened)))
    selected_models = [item for item in MODELS if item[0] in selected_ids]
    if not selected_models:
        raise SystemExit(f"No Cosmos candidates were selected for the {args.whole_device_cap_gib} GiB stage")

    output_dir = args.output_dir or HERE / "results" / "edge_refinement" / f"{args.whole_device_cap_gib}gib"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this benchmark requires GPU execution in host context")
    device = torch.cuda.current_device()
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    baseline_mib = runner.gpu_memory_used_mib()
    if baseline_mib is None:
        raise SystemExit("could not measure the idle whole-device GPU baseline")
    baseline_gib = baseline_mib / 1024
    cap_gib = float(args.whole_device_cap_gib)
    if baseline_gib >= cap_gib:
        raise SystemExit("the measured idle whole-device baseline reaches the selected cap")
    if total_bytes < cap_gib * 1024**3:
        raise SystemExit("the selected cap exceeds installed GPU memory")
    process_budget_gib = cap_gib - baseline_gib - 0.5
    if process_budget_gib <= 0:
        raise SystemExit("the measured baseline leaves no process budget after the 0.5 GiB reserve")
    try:
        gpu_name, driver_version = gpu_identity()
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise SystemExit("could not record GPU model and driver version") from exc

    runner.GPU_BASELINE_GIB = baseline_gib
    runner.MEMORY_CAP_GIB = cap_gib
    torch.cuda.set_per_process_memory_fraction(process_budget_gib * 1024**3 / total_bytes, device)

    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": 3407,
        "samples_per_task": TASK_SIZES,
        "task_sizes": TASK_SIZES,
        "task_names": list(TASKS),
        "whole_device_cap_gib": cap_gib,
        "memory_cap_gib": cap_gib,
        "gpu_baseline_gib": round(baseline_gib, 3),
        "hardware": {"gpu_name": gpu_name, "driver_version": driver_version},
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "transformers": version("transformers"),
        },
        "cap_enforcement_method": (
            "torch per-process CUDA memory fraction capped to the whole-device limit minus measured baseline and 0.5 GiB reserve; "
            "cap qualification uses sampled whole-device nvidia-smi peaks"
        ),
        "process_allocator_budget_gib": round(process_budget_gib, 3),
        "context_probe_limit_tokens": args.context_probe_limit,
        "edge_suite": True,
        "profile_context_only": args.profile_context_only,
        "execution_mode": "offline cached Transformers, one Cosmos model at a time, serial requests",
        "models": [],
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for model_id, folder in selected_models:
        snap = snapshot_for(folder)
        record: dict[str, Any] = {
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
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        entry = {"id": model_id, "backend": record["backend"], "snapshot": str(snap)}
        try:
            print(f"START {model_id} cap={cap_gib:g}GiB tasks={','.join(TASKS)}", flush=True)
            runner.run_transformers(
                entry, 100, 3407, result, result_path, TASKS,
                max_new_tokens_override=4096 if "Cosmos-Reason1" in folder else 256,
                task_sizes=TASK_SIZES,
                context_cap=args.context_probe_limit,
                context_probe_limit=args.context_probe_limit,
                profile_context_only=args.profile_context_only,
            )
            run_record = next(item for item in reversed(result["models"]) if item.get("model_id") == model_id)
            run_record.update({k: v for k, v in record.items() if k not in {"tasks", "status"}})
        except Exception as exc:
            failed = next((item for item in reversed(result["models"]) if item.get("model_id") == model_id), None)
            if failed is None:
                failed = record
                result["models"].append(failed)
            failed.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        runner.write_report(output_dir / "report.md", result)
        print(f"FINISH {model_id} status={result['models'][-1]['status']}", flush=True)

    runner.write_report(output_dir / "report.md", result)
    print(f"Wrote {result_path} and {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(run_queued(main, "cosmos-edge"))
