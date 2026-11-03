#!/usr/bin/env python3
"""Select fixed grounding examples from local, rights-cleared JSONL exports.

The script never downloads or copies images. It stores paths and annotations in
local manifests so both models receive the exact same sampled records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TASKS = ("refcocog", "coco", "screenspot_pro", "hiertext")
TASK_METADATA = {
    "refcocog": {"dataset": "RefCOCOg", "split": "test"},
    "coco": {"dataset": "COCO", "split": "val2017"},
    "screenspot_pro": {"dataset": "ScreenSpot-Pro", "split": "test"},
    "hiertext": {"dataset": "HierText", "split": "validation"},
}
TASK_ROW_REQUIRED = {
    "refcocog": ("image", "phrase", "target_box"),
    "coco": ("image", "image_id", "categories", "targets_by_category"),
    "screenspot_pro": ("image", "instruction", "target_box"),
    "hiertext": ("image", "target_boxes"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            rows.append(row)
    return rows


def validate_row(task: str, row: dict[str, Any], source: Path, line: int) -> None:
    missing = [key for key in TASK_ROW_REQUIRED[task] if key not in row]
    if missing:
        raise ValueError(f"{source}:{line}: missing required fields: {', '.join(missing)}")
    if not isinstance(row["image"], str) or not row["image"].strip():
        raise ValueError(f"{source}:{line}: image must be a non-empty path")
    if task == "coco":
        categories = row["categories"]
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"{source}:{line}: categories must be a non-empty list")
        if any(not isinstance(item, dict) or not item.get("name") or not item.get("id") for item in categories):
            raise ValueError(f"{source}:{line}: each COCO category must contain numeric id and name")
        if not isinstance(row["targets_by_category"], dict):
            raise ValueError(f"{source}:{line}: targets_by_category must map category names to xyxy boxes")
    for key in ("target_box",):
        if key in row:
            box = row[key]
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
                raise ValueError(f"{source}:{line}: {key} must be [x1,y1,x2,y2] normalized to 0..1")
            if not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
                raise ValueError(f"{source}:{line}: {key} must be an ordered normalized box")
    boxes = row.get("target_boxes", [])
    if not isinstance(boxes, list):
        raise ValueError(f"{source}:{line}: target_boxes must be a list")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
            raise ValueError(f"{source}:{line}: each target_boxes entry must be a normalized xyxy box")


def select_rows(task: str, source: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rows = load_jsonl(source)
    for i, row in enumerate(rows, 1):
        validate_row(task, row, source, i)
    if len(rows) < count:
        raise ValueError(f"{task}: source has {len(rows)} rows; {count} are required")
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    selected = []
    for i, row_index in enumerate(indices):
        row = dict(rows[row_index])
        row["sample_index"] = i
        row["source_row"] = row_index
        row["id"] = str(row.get("id", f"{task}-{row_index}"))
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE / "data" / "source",
                        help="Directory containing one canonical <task>.jsonl per task")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "manifests")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        source = args.source_dir / f"{task}.jsonl"
        metadata_path = args.source_dir / f"{task}.meta.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing local annotation export: {source}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing dataset provenance sidecar: {metadata_path}")
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = TASK_METADATA[task]
        if source_metadata.get("dataset") != expected["dataset"] or source_metadata.get("split") != expected["split"]:
            raise ValueError(f"{metadata_path} must identify {expected['dataset']} {expected['split']}")
        if not source_metadata.get("license"):
            raise ValueError(f"{metadata_path} must record the applicable license or access terms")
        rows = select_rows(task, source, args.samples, args.seed)
        payload = {
            "task": task, "sample_count": len(rows), "seed": args.seed,
            "dataset": expected["dataset"], "split": expected["split"],
            "license_or_access_terms": source_metadata["license"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "asset_policy": "image paths only; no image bytes copied",
            "examples": rows,
        }
        target = args.output_dir / f"{task}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{task}: {len(rows)} examples -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
