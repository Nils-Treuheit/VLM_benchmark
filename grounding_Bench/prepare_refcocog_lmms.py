#!/usr/bin/env python3
"""Prepare the fixed RefCOCOg panel from local lmms-lab parquet shards.

Only the sampled images are materialized. The manifest records the parquet
source hashes and uses the first human reference caption as the target phrase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Local lmms-lab RefCOCOg dataset directory containing data/test-*.parquet")
    parser.add_argument("--output", type=Path, default=HERE / "data" / "manifests" / "refcocog.json")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="output image directory (default: a derived panel folder under the dataset root)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.images_dir is None:
        args.images_dir = (args.dataset_dir.parent / "derived" / "grounding_Bench" /
                           f"refcocog_test_{args.samples}_seed{args.seed}" / "images")

    shards = sorted((args.dataset_dir / "data").glob("test-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no test-*.parquet shards under {args.dataset_dir / 'data'}")
    rows: list[dict[str, Any]] = []
    for shard in shards:
        for row in pq.read_table(shard).to_pylist():
            rows.append(row)
    if len(rows) < args.samples:
        raise ValueError(f"RefCOCOg test split has {len(rows)} rows; need {args.samples}")

    # Sort by the stable query ID before sampling so shard reordering cannot
    # change the panel.
    rows.sort(key=lambda row: str(row["question_id"]))
    selected_indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.samples))
    args.images_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for sample_index, source_index in enumerate(selected_indices):
        row = rows[source_index]
        references = [str(value).strip() for value in row.get("answer", []) if str(value).strip()]
        if not references:
            raise ValueError(f"RefCOCOg row {row.get('question_id')} has no reference phrase")
        box = [float(value) for value in row.get("bbox", [])]
        if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"invalid RefCOCOg xywh box for row {row.get('question_id')}: {box}")
        image_struct = row.get("image") or {}
        image_bytes = image_struct.get("bytes")
        if not image_bytes:
            raise ValueError(f"embedded image missing for RefCOCOg row {row.get('question_id')}")
        image_name = f"refcocog_{row['question_id']}.jpg"
        image_path = (args.images_dir / image_name).resolve()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
        image_path.write_bytes(image_bytes)
        x, y, w, h = box
        right, bottom = x + w, y + h
        # Float32 annotation values may overshoot an integer image edge by a
        # few millionths of a pixel after conversion back to decimal.
        if x < 0 or y < 0 or right > width + 1e-3 or bottom > height + 1e-3:
            raise ValueError(f"RefCOCOg xywh box exceeds {width}x{height} image: {box}")
        right, bottom = min(right, float(width)), min(bottom, float(height))
        examples.append({
            "id": f"refcocog-{row['question_id']}",
            "sample_index": sample_index,
            "source_row": source_index,
            "image": str(image_path),
            "phrase": references[0],
            "reference_phrases": references,
            "target_box": [x / width, y / height, right / width, bottom / height],
            "image_width": width,
            "image_height": height,
        })

    source_shards = [{"file": str(shard.resolve()), "sha256": sha256(shard)} for shard in shards]
    provenance = hashlib.sha256("".join(item["sha256"] for item in source_shards).encode()).hexdigest()
    payload = {
        "task": "refcocog",
        "sample_count": len(examples),
        "seed": args.seed,
        "dataset": "RefCOCOg",
        "split": "test",
        "license_or_access_terms": (
            "Local lmms-lab formatted RefCOCOg copy; RefCOCOg and COCO source terms apply. "
            "The local dataset README provides citation information but states no separate license. "
            "Used for local research evaluation; do not redistribute images or annotations."
        ),
        "source_sha256": provenance,
        "source_shards": source_shards,
        "asset_policy": "only fixed-panel image bytes extracted from local parquet shards",
        "sampling": "stable sort by question_id, then seeded sample without replacement",
        "phrase_policy": "first non-empty human reference answer is scored as the target phrase; all references retained for audit",
        "box_source": "parquet bbox in pixel xywh, converted to normalized xyxy",
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RefCOCOg: {len(examples)}/{len(rows)} test examples -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
