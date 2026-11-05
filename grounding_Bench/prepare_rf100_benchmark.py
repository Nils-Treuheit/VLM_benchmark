#!/usr/bin/env python3
"""Prepare fixed COCO-shaped panels from the locally available RF100-VL test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/rf100_vl")
DATASETS = (
    ("aerial-airport", "aerial_airport", "Aerial airport aircraft"),
    ("activity-diagrams", "activity_diagrams", "Activity diagrams"),
    ("cable-damage", "cable_damage", "Cable damage"),
    ("marine-sharks", "marine_sharks", "Marine sharks"),
    ("taco-trash-annotations-in-context", "taco_trash", "TACO trash"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(root: Path, output: Path, samples: int, seed: int) -> list[dict[str, Any]]:
    results = []
    for source_name, slug, label in DATASETS:
        source = root / source_name / "test"
        annotation_file = source / "_annotations.coco.json"
        license_file = root / source_name / "README.dataset.txt"
        if not annotation_file.is_file() or not license_file.is_file():
            raise FileNotFoundError(f"{source_name}: missing test annotations or license metadata")
        license_text = license_file.read_text(encoding="utf-8")
        if "License: MIT" not in license_text:
            raise ValueError(f"{source_name}: expected explicit MIT license metadata")
        source_data = json.loads(annotation_file.read_text(encoding="utf-8"))
        images = {int(row["id"]): row for row in source_data["images"]}
        categories = sorted(source_data["categories"], key=lambda row: int(row["id"]))
        annotations = source_data["annotations"]
        by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
        for annotation in annotations:
            by_image[int(annotation["image_id"])].append(annotation)
        candidates = sorted(image_id for image_id, rows in by_image.items() if rows)
        if len(candidates) < samples:
            raise ValueError(f"{source_name}: only {len(candidates)} labeled images, requested {samples}")
        selected = sorted(random.Random(seed).sample(candidates, samples))
        selected_images = []
        selected_annotations = []
        rows = []
        names = {int(category["id"]): str(category["name"]) for category in categories}
        for image_id in selected:
            image = images[image_id]
            image_path = (source / image["file_name"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"{source_name}: image missing: {image_path}")
            width, height = int(image["width"]), int(image["height"])
            selected_images.append({**image, "file_name": str(image_path)})
            targets = {name: [] for name in names.values()}
            for ann in by_image[image_id]:
                selected_annotations.append(ann)
                x, y, w, h = [float(value) for value in ann["bbox"]]
                targets[names[int(ann["category_id"])]].append([
                    max(0.0, x / width), max(0.0, y / height),
                    min(1.0, (x + w) / width), min(1.0, (y + h) / height),
                ])
            rows.append({
                "id": f"RF100-{slug}-{image_id}", "image_id": image_id,
                "image": str(image_path),
                "categories": [{"id": int(category["id"]), "name": str(category["name"])}
                               for category in categories],
                "targets_by_category": targets,
            })
        base = output / "rf100_vl" / slug
        subset_annotations = {
            "info": {"description": f"RF100-VL {source_name} test subset; {samples} fixed images"},
            "licenses": source_data.get("licenses", []), "images": selected_images,
            "annotations": selected_annotations, "categories": categories,
        }
        write(base / "coco_annotations.json", subset_annotations)
        manifest = {
            "task": "coco", "dataset": f"RF100-VL: {label}", "source_dataset": source_name,
            "split": "test", "sample_count": len(rows), "seed": seed, "input_size": 640,
            "license_or_access_terms": "MIT per local README.dataset.txt; source metadata retained in local data tree",
            "source_annotation_sha256": sha256(annotation_file),
            "protocol": "20-image seeded sample from labeled test images; official COCO bbox AP and local class-aware F1",
            "annotation_file": str((base / "coco_annotations.json").resolve()),
            "source_root": str(source.resolve()), "selected_image_ids": selected, "examples": rows,
        }
        write(base / "coco.json", manifest)
        results.append({"dataset": source_name, "slug": slug, "images": len(images),
                        "annotations": len(annotations), "sample_count": len(rows),
                        "categories": len(categories), "annotation_sha256": manifest["source_annotation_sha256"],
                        "manifest": str(base / "coco.json"),
                        "coco_annotations": str(base / "coco_annotations.json")})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "domain_manifests")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    for result in prepare(args.data_root, args.output_dir, args.samples, args.seed):
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
