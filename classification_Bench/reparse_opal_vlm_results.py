#!/usr/bin/env python3
"""Apply the strict single-label parser to preserved OPAL VLM raw outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from opal_vlm_benchmark import parse_class  # noqa: E402


def reparse(source_dir: Path, output_dir: Path, panel_path: Path) -> list[Path]:
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    labels_by_dataset = {}
    for dataset_id, dataset in panel["datasets"].items():
        label_by_index = {int(row["label"]): row["prompt_label"] for row in dataset["examples"]}
        if set(label_by_index) != set(range(int(dataset["class_count"]))):
            raise ValueError(f"{dataset_id}: panel does not cover every class index")
        labels_by_dataset[dataset_id] = [label_by_index[index] for index in range(dataset["class_count"])]

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sorted(source_dir.glob("*.json")):
        run = json.loads(source.read_text(encoding="utf-8"))
        dataset_id = str(run["dataset_id"])
        labels = labels_by_dataset[dataset_id]
        for item in run.get("items", []):
            if item.get("status") != "complete":
                continue
            prediction, parse_status = parse_class(str(item.get("raw_output", "")), labels)
            item["prediction"] = prediction
            item["correct"] = prediction == int(item["label"]) if prediction is not None else False
            item["parse_status"] = parse_status
        items = run.get("items", [])
        complete = [item for item in items if item.get("status") == "complete"]
        valid = [item for item in complete if item.get("parse_status") == "matched"]
        latencies = [float(item["latency_s"]) for item in complete if item.get("latency_s") is not None]
        correct = sum(bool(item.get("correct")) for item in complete)
        run["classification_parser_version"] = "strict-single-label-v2"
        run["summary"].update({
            "attempted": len(items), "completed": len(complete),
            "failed": sum(item.get("status") != "complete" for item in items),
            "parse_valid": len(valid),
            "accuracy_top1": correct / len(items) if items else None,
            "accuracy_on_valid": sum(bool(item.get("correct")) for item in valid) / len(valid) if valid else None,
            "mean_latency_s": statistics.mean(latencies) if latencies else None,
            "median_latency_s": statistics.median(latencies) if latencies else None,
            "p95_latency_s": sorted(latencies)[max(0, (95 * len(latencies) + 99) // 100 - 1)] if latencies else None,
            "max_latency_s": max(latencies) if latencies else None,
            "images_per_s": len(latencies) / sum(latencies) if latencies and sum(latencies) else None,
        })
        target = output_dir / source.name
        target.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=HERE / "data" / "opal_test_panels.json")
    args = parser.parse_args()
    outputs = reparse(args.source_dir, args.output_dir, args.panel)
    print(json.dumps({"files": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
