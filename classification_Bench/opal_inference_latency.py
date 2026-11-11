#!/usr/bin/env python3
"""Measure OPAL checkpoint accuracy and per-query latency on a fixed test panel."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from gpu_queue import run_queued  # noqa: E402
OPAL_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/opal_protocol")
PANEL_PATH = PROJECT_ROOT / "classification_Bench" / "data" / "opal_test_panels.json"


def read_training_args(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[max(0, int(0.95 * len(values) + 0.999999) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=(
        "cub_200_2011", "stanford_cars", "oxford_iiit_pet", "stanford_dogs", "oxford_flowers_102",
    ))
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="OPAL experiment directory containing logs/args.txt and checkpoint/")
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-items", type=int, default=10)
    args_cli = parser.parse_args()

    # This is a local CUDA inference job. Release and inspect managed GPU
    # workloads before importing torch or probing CUDA availability.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpu_guard import release_and_check_local_gpu
    release_and_check_local_gpu()

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(f"Required inference dependency is unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("OPAL timing requires CUDA; CPU execution is not used as a substitute")

    training_log = args_cli.experiment_dir / "logs" / "args.txt"
    checkpoint_dir = args_cli.experiment_dir / "checkpoint"
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    if not training_log.is_file() or len(checkpoints) != 1:
        raise FileNotFoundError(
            f"expected logs/args.txt and exactly one .pth checkpoint under {args_cli.experiment_dir}"
        )
    train_args = read_training_args(training_log)
    classes = int(train_args["num_classes"])
    model_config = SimpleNamespace(
        base_model=train_args.get("base_model", "convnext_tiny"),
        pretrain=False,  # trained checkpoint supplies the complete initialized model weights
        pooling=train_args.get("pooling", "max"),
        classifier=train_args.get("classifier", "norm_prototypes_dist"),
        metric=train_args.get("metric", "euclidean"),
        p=int(train_args.get("p", "2")),
        ch_last_conv=int(train_args.get("ch_last_conv", "5")),
        num_classes=classes,
    )
    model_name = train_args.get("model", "NNModel_softmax")

    sys.path.insert(0, str(OPAL_ROOT))
    from net.net_model import NNModel, NNModel_softmax
    from utils.data import get_transforms

    model_types = {"NNModel": NNModel, "NNModel_softmax": NNModel_softmax}
    if model_name not in model_types:
        raise ValueError(f"Unsupported OPAL model class in training log: {model_name}")
    model_config.img_size = int(train_args.get("img_size", "224"))
    model_config.data_aug = False
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    load_started = time.perf_counter()
    model = model_types[model_name](model_config).to(device)
    state = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    transform = get_transforms(model_config)["eval"]

    manifest = json.loads(args_cli.panel.read_text(encoding="utf-8"))
    panel = manifest["datasets"][args_cli.dataset]
    rows = panel["examples"]
    if len(rows) != 200 or panel["class_count"] != classes:
        raise ValueError(
            f"Panel/model mismatch for {args_cli.dataset}: {len(rows)} items, "
            f"{panel['class_count']} panel classes, {classes} model classes"
        )

    def load_tensor(image_path: str):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return transform(image).unsqueeze(0).to(device)

    warmup_started = time.perf_counter()
    with torch.inference_mode():
        for row in rows[:max(0, args_cli.warmup_items)]:
            _ = model(load_tensor(row["image"]))
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    latencies: list[float] = []
    item_results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in rows:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            image_tensor = load_tensor(row["image"])
            logits = model(image_tensor)
            prediction = int(logits.argmax(dim=1).item())
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            item_results.append({
                "id": row["id"],
                "label": int(row["label"]),
                "prediction": prediction,
                "correct": prediction == int(row["label"]),
                "latency_s": elapsed,
            })

    accuracy = sum(item["correct"] for item in item_results) / len(item_results)
    output = {
        "dataset": panel["dataset"],
        "split": panel["split"],
        "sample_count": len(rows),
        "class_count": classes,
        "panel_seed": panel["seed"],
        "model": {
            "method": "OPAL",
            "class": model_name,
            "backbone": model_config.base_model,
            "classifier": model_config.classifier,
            "checkpoint": str(checkpoints[0].resolve()),
        },
        "accuracy_top1": accuracy,
        "model_load_duration_s": model_load_seconds,
        "warmup_duration_s": warmup_seconds,
        "query_timing_scope": "image open, RGB conversion, eval preprocessing, device transfer, forward pass, argmax; model load and warmup excluded",
        "runtime": {
            "mean_latency_s": statistics.mean(latencies),
            "median_latency_s": statistics.median(latencies),
            "p95_latency_s": p95(latencies),
            "max_latency_s": max(latencies),
            "images_per_s": len(latencies) / sum(latencies),
        },
        "items": item_results,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{panel['dataset']}: top1={accuracy:.3f}; mean={statistics.mean(latencies):.4f}s; "
        f"median={statistics.median(latencies):.4f}s; p95={p95(latencies):.4f}s; "
        f"load={model_load_seconds:.2f}s; output={args_cli.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_queued(main, "opal-inference-latency"))
