#!/usr/bin/env python3
"""Validate the large YOLO26 oriented detector on a local DOTA v1 config.

This is a separate, in-domain OBB track. YOLOE-26 has no released OBB head,
so its results are not mixed with this metric.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vision_benchmark import DeviceMemory, checkpoint_identity, release_local  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Local Ultralytics DOTA v1 YAML; this command will not download images")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Explicit local YOLO26-L-OBB checkpoint, such as yolo26l-obb.pt")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "vision_runs")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"DOTA data YAML not found: {args.data}")
    if not args.weights.is_file():
        parser.error(f"local checkpoint not found: {args.weights}")

    import yaml
    from ultralytics import YOLO

    config = yaml.safe_load(args.data.read_text(encoding="utf-8")) or {}
    if not config.get("val"):
        parser.error("DOTA YAML must define a local validation split")
    if not isinstance(config.get("names"), (list, dict)) or len(config["names"]) != 15:
        parser.error("expected the 15 DOTA v1 categories in the local YAML")
    if args.weights.name.casefold() != "yolo26l-obb.pt":
        parser.error("this DOTA track requires the explicitly named yolo26l-obb.pt large checkpoint")
    dataset_root = Path(config.get("path") or args.data.parent)
    if not dataset_root.is_absolute():
        dataset_root = (args.data.parent / dataset_root).resolve()
    validation_entries = config["val"] if isinstance(config["val"], list) else [config["val"]]
    for entry in validation_entries:
        validation_path = Path(str(entry))
        if not validation_path.is_absolute():
            validation_path = dataset_root / validation_path
        if not validation_path.exists():
            parser.error(f"local DOTA validation path is missing: {validation_path}")

    release_local()
    memory = DeviceMemory()
    memory.start()
    model = YOLO(str(args.weights.resolve()))
    started = time.perf_counter()
    metrics = model.val(
        data=str(args.data.resolve()), split="val", imgsz=args.imgsz, batch=1,
        device=args.device, workers=0, conf=0.001, nms=False, save_json=True,
        plots=False, verbose=False,
    )
    elapsed = time.perf_counter() - started
    memory.stop()
    speed = {str(key): float(value) for key, value in (getattr(metrics, "speed", {}) or {}).items()}
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validation did not return oriented-box metrics")
    names = (list(config["names"].values()) if isinstance(config["names"], dict)
             else list(config["names"]))
    output: dict[str, Any] = {
        "model_id": "yolo26l-obb",
        "label": "YOLO26-L OBB (yolo26l-obb.pt; DOTA-pretrained)",
        "checkpoint": checkpoint_identity(args.weights.resolve()),
        "dataset": "DOTA v1.0 validation",
        "dataset_root": str(dataset_root),
        "data_yaml": str(args.data.resolve()),
        "category_count": len(config["names"]),
        "input_size": args.imgsz,
        "batch_size": 1,
        "split": "val",
        "timing_scope": "Ultralytics validation speed; model load excluded; peak VRAM sampling includes model load",
        "validation_wall_s": elapsed,
        "runtime": {
            "preprocess_ms_per_image": speed.get("preprocess"),
            "inference_ms_per_image": speed.get("inference"),
            "postprocess_ms_per_image": speed.get("postprocess"),
            "images_per_s_inference_only": 1000.0 / speed["inference"] if speed.get("inference", 0) > 0 else None,
            "peak_whole_device_vram_gib": memory.gib(),
            "gpu_memory_samples": memory.samples,
            "gpu_memory_error": memory.error,
        },
        "tasks": {
            "oriented_detection": {
                "status": "complete",
                "metric": "Ultralytics OBB mAP on DOTA v1.0 validation",
                "AP": float(box.map), "AP50": float(box.map50), "AP75": float(box.map75),
                "per_category_AP": {str(key): float(value) for key, value in zip(names, box.maps)},
            }
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "dota-yolo26l-obb.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    inference_ms = speed.get("inference")
    speed_text = f"{inference_ms:.1f} ms/image" if inference_ms is not None else "not recorded"
    print(f"DOTA v1.0 validation -> {target}; AP={output['tasks']['oriented_detection']['AP']:.4f}; "
          f"inference={speed_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
