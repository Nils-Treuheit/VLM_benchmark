#!/usr/bin/env python3
"""Train OPAL sequentially on the five prepared ImageFolder datasets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from gpu_queue import run_queued  # noqa: E402
OPAL_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/opal_protocol")
DATA_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/opal_datasets")
DEFAULT_OUTPUT = PROJECT_ROOT / "classification_Bench" / "results" / "opal_training"
TORCH_HOME = Path("/tmp/opal-torch-home")

DATASETS = {
    "cub_200_2011": ("CUB-200-2011", 200),
    "stanford_cars": ("Stanford Cars", 196),
    "oxford_iiit_pet": ("Oxford-IIIT Pets", 37),
    "stanford_dogs": ("Stanford Dogs", 120),
    "oxford_flowers_102": ("Oxford Flowers 102", 102),
}


def validate_imagefolder(dataset_id: str, expected_classes: int) -> Path:
    root = DATA_ROOT / dataset_id / "dataset"
    train = root / "train"
    test = root / "test"
    if not train.is_dir() or not test.is_dir():
        raise FileNotFoundError(f"{dataset_id}: expected train/ and test/ under {root}")
    train_classes = sorted(path.name for path in train.iterdir() if path.is_dir())
    test_classes = sorted(path.name for path in test.iterdir() if path.is_dir())
    if len(train_classes) != expected_classes or train_classes != test_classes:
        raise ValueError(
            f"{dataset_id}: split classes differ or count is wrong "
            f"(train={len(train_classes)}, test={len(test_classes)}, expected={expected_classes})"
        )
    for split in (train, test):
        empty = [name for name in train_classes if not any((split / name).iterdir())]
        if empty:
            raise ValueError(f"{dataset_id}: empty class folders in {split.name}: {empty[:5]}")
    return root


def run_spc_only(dataset_id: str, experiment_dir: Path, summary_dir: Path) -> None:
    """Recover one missing test-split SPC artifact from an existing checkpoint."""
    from gpu_guard import release_and_check_local_gpu

    data_dir = validate_imagefolder(dataset_id, DATASETS[dataset_id][1])
    args_path = experiment_dir / "logs" / "args.txt"
    checkpoint_dir = experiment_dir / "checkpoint"
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    if not args_path.is_file() or len(checkpoints) != 1:
        raise FileNotFoundError(
            f"{dataset_id}: expected training args and one checkpoint in {experiment_dir}"
        )
    train_args = {}
    for line in args_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            train_args[key.strip()] = value.strip()
    class_count = DATASETS[dataset_id][1]
    if int(train_args.get("num_classes", "-1")) != class_count:
        raise ValueError(f"{dataset_id}: checkpoint class count does not match the dataset")

    release_and_check_local_gpu()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("OPAL SPC evaluation requires CUDA; CPU fallback is not allowed")
    sys.path.insert(0, str(OPAL_ROOT))
    from net.net_model import NNModel_softmax
    from utils.viz import compute_spatial_prototype_collapse

    model_config = SimpleNamespace(
        base_model=train_args.get("base_model", "convnext_tiny"),
        pretrain=False,
        pooling=train_args.get("pooling", "max"),
        classifier=train_args.get("classifier", "norm_prototypes_dist"),
        metric=train_args.get("metric", "euclidean"),
        p=int(train_args.get("p", "2")),
        ch_last_conv=int(train_args.get("ch_last_conv", "5")),
        num_classes=class_count,
    )
    device = torch.device("cuda:0")
    model = NNModel_softmax(model_config).to(device)
    model.load_state_dict(torch.load(checkpoints[0], map_location="cpu", weights_only=True))
    model.eval()
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    compute_spatial_prototype_collapse(
        model, str(data_dir), class_count, "test", str(metrics_dir)
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "spc_recovery.json"
    summary_path.write_text(json.dumps({
        "dataset_id": dataset_id,
        "dataset": DATASETS[dataset_id][0],
        "test_images": sum(1 for _ in (data_dir / "test").glob("*/*")),
        "experiment_dir": str(experiment_dir.resolve()),
        "checkpoint": str(checkpoints[0].resolve()),
        "spc_metrics_csv": str((metrics_dir / "spatial_collapse_metrics.csv").resolve()),
        "spc_per_image_csv": str((metrics_dir / "spatial_collapse_per_image.csv").resolve()),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Recovered SPC metrics: {metrics_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--spc-only", choices=tuple(DATASETS),
                        help="compute missing test-split SPC metrics from an existing checkpoint")
    parser.add_argument("--experiment-dir", type=Path,
                        help="existing OPAL experiment directory required by --spc-only")
    args = parser.parse_args()
    if args.spc_only:
        if args.experiment_dir is None:
            raise ValueError("--experiment-dir is required with --spc-only")
        run_spc_only(args.spc_only, args.experiment_dir, args.output_dir.resolve())
        return 0
    if args.experiment_dir is not None:
        raise ValueError("--experiment-dir is only valid with --spc-only")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not (OPAL_ROOT / "main.py").is_file():
        raise FileNotFoundError(f"OPAL entry point not found: {OPAL_ROOT / 'main.py'}")
    if not (TORCH_HOME / "hub" / "checkpoints" / "convnext_tiny-983f1562.pth").is_file():
        raise FileNotFoundError(f"pretrained ConvNeXt checkpoint is missing under {TORCH_HOME}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpu_guard import release_and_check_local_gpu

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id in args.datasets:
        display_name, class_count = DATASETS[dataset_id]
        data_dir = validate_imagefolder(dataset_id, class_count)

        # Recheck the release gate before every training process. The guard
        # rejects unmanaged GPU processes and never unloads them.
        release_and_check_local_gpu()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("OPAL training requires CUDA; CPU fallback is not allowed")
        command = [
            sys.executable, "-u", str(OPAL_ROOT / "main.py"),
            "--dataset", display_name,
            "--data_dir", str(data_dir),
            "--num_classes", str(class_count),
            "--base_model", "convnext_tiny",
            "--pretrain", "True",
            "--pooling", "max",
            "--classifier", "norm_prototypes_dist",
            "--metric", "euclidean",
            "--model", "NNModel_softmax",
            "--seed", "3407",
            "--epochs", str(args.epochs),
            "--batch_size", "64",
            "--lr", "1e-4",
            "--data_aug", "True",
            "--output_dir", str(output_dir),
            # Dataset-wide t-SNE is quadratic and not needed for the reported
            # classification or latency measures. Keep OPAL metrics and SPC.
            "--save_tsne", "False",
            "--save_metrics", "True",
            "--save_curves", "True",
            "--save_collapse_metrics", "True",
        ]
        environment = os.environ.copy()
        environment["TORCH_HOME"] = str(TORCH_HOME)
        print(f"Starting OPAL: {dataset_id} ({class_count} classes)", flush=True)
        subprocess.run(command, cwd=OPAL_ROOT, env=environment, check=True)
        print(f"Completed OPAL: {dataset_id}", flush=True)

    print(f"OPAL training outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_queued(main, "opal-training"))
