#!/usr/bin/env python3
"""Prepare a deterministic shared COCO val2017 manifest for vision benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def build_manifest(coco_root: Path, output: Path, sample_count: int = 100,
                   seed: int = 3407) -> dict[str, Any]:
    annotations_path = coco_root / "annotations" / "instances_val2017.json"
    image_root = coco_root / "val2017"
    source = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = source["images"]
    annotations = source["annotations"]
    categories = source["categories"]
    by_image: dict[int, list[dict[str, Any]]] = {int(item["id"]): [] for item in images}
    for annotation in annotations:
        by_image[int(annotation["image_id"])].append(annotation)
    candidates = [item for item in images if by_image[int(item["id"])]
                  and (image_root / item["file_name"]).is_file()]
    if len(candidates) < sample_count:
        raise ValueError(f"Only {len(candidates)} annotated COCO val images are available; requested {sample_count}")
    selected = sorted(random.Random(seed).sample(candidates, sample_count), key=lambda item: int(item["id"]))
    category_names = {int(category["id"]): str(category["name"]) for category in categories}
    category_prompt = [{"id": int(category["id"]), "name": str(category["name"])}
                       for category in categories]
    rows = []
    for image in selected:
        image_id = int(image["id"])
        width, height = int(image["width"]), int(image["height"])
        targets = {name: [] for name in category_names.values()}
        for annotation in by_image[image_id]:
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            targets[category_names[int(annotation["category_id"])]].append([
                max(0.0, x / width), max(0.0, y / height),
                min(1.0, (x + box_width) / width), min(1.0, (y + box_height) / height),
            ])
        rows.append({
            "id": f"COCO-val2017-{image_id}", "image_id": image_id,
            "image": str((image_root / image["file_name"]).resolve()),
            "categories": category_prompt, "targets_by_category": targets,
        })
    payload = {
        "task": "coco", "dataset": "COCO val2017", "split": "val2017",
        "sample_count": len(rows), "seed": seed, "input_size": 640,
        "license_or_access_terms": "COCO dataset terms; see cocodataset.org/#download",
        "protocol": "Fixed annotated-image subset; official COCO bbox and instance-mask AP",
        "annotation_file": str(annotations_path.resolve()),
        "source_root": str(coco_root.resolve()),
        "selected_image_ids": [int(item["id"]) for item in selected],
        "examples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True,
                        help="Local COCO directory containing annotations/ and val2017/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    result = build_manifest(args.coco_root, args.output, args.sample_count, args.seed)
    print(f"Prepared {result['sample_count']} COCO val2017 images at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
