#!/usr/bin/env python3
"""Download and materialize the official 500-example GRAB-Lite split."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "grab_lite"


def main() -> None:
    rows = load_dataset("jonathan-roberts1/GRAB-lite", split="GRAB_lite")
    image_dir = DEST / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        image_name = f"{row['pid']}.png"
        image = row["decoded_image"].convert("RGB")
        image.save(image_dir / image_name, format="PNG", compress_level=1)
        records.append({
            "pid": str(row["pid"]),
            "question": row["question"],
            "answer": str(row["answer"]),
            "task": str(row["task"]).lower(),
            "category": str(row["category"]),
            "complexity": int(row["complexity"]),
            "image_path": f"data/grab_lite/images/{image_name}",
        })
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "examples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in records:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    manifest = {
        "dataset": "jonathan-roberts1/GRAB-lite",
        "split": "GRAB_lite",
        "official_test_examples": len(records),
        "task_counts": counts,
        "selection": "Complete official 500-example GRAB-Lite split; no subsampling.",
        "scoring": "Exact-match accuracy after answer normalization; numeric answers compared as exact decimal values.",
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
