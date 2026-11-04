#!/usr/bin/env python3
"""Prepare a small, balanced LVIS v1 validation manifest for local vision runs."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zipfile import ZipFile


def prepare(annotation_zip: Path, coco_root: Path, output_dir: Path,
            classes_per_frequency: int = 10, sample_count: int = 100,
            seed: int = 3407) -> dict[str, Any]:
    with ZipFile(annotation_zip) as archive:
        names = archive.namelist()
        if names != ["lvis_v1_val.json"]:
            raise ValueError(f"Expected the LVIS v1 validation archive, got members {names}")
        source = json.load(archive.open(names[0]))

    image_paths: dict[int, Path] = {}
    for image in source["images"]:
        image_id = int(image["id"])
        split = Path(urlparse(str(image.get("coco_url", ""))).path).parent.name
        if split not in {"train2017", "val2017"}:
            continue
        candidate = coco_root / split / f"{image_id:012d}.jpg"
        if candidate.is_file():
            image_paths[image_id] = candidate
    val_images_by_category: dict[int, set[int]] = defaultdict(set)
    for annotation in source["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in image_paths:
            val_images_by_category[int(annotation["category_id"])].add(image_id)
    categories_by_frequency = {
        frequency: [category for category in source["categories"]
                    if category.get("frequency") == frequency
                    and len(val_images_by_category[int(category["id"])]) >= 2]
        for frequency in ("r", "c", "f")
    }
    rng = random.Random(seed)
    selected_categories = []
    for frequency in ("r", "c", "f"):
        pool = sorted(categories_by_frequency[frequency], key=lambda item: int(item["id"]))
        if len(pool) < classes_per_frequency:
            raise ValueError(f"Only {len(pool)} eligible {frequency!r} categories")
        selected_categories.extend(rng.sample(pool, classes_per_frequency))
    selected_category_ids = {int(category["id"]) for category in selected_categories}
    category_name = {int(category["id"]): str(category["name"]) for category in selected_categories}
    aliases = {str(alias).casefold(): category_name[int(category["id"])]
               for category in selected_categories for alias in category.get("synonyms", [])}

    images_by_id = {int(image["id"]): image for image in source["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    categories_by_image: dict[int, set[int]] = {}
    for annotation in source["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if category_id not in selected_category_ids:
            continue
        annotations_by_image.setdefault(image_id, []).append(annotation)
        categories_by_image.setdefault(image_id, set()).add(category_id)
    candidates = [image_id for image_id in categories_by_image if image_id in image_paths]
    if len(candidates) < sample_count:
        raise ValueError(f"Only {len(candidates)} annotated LVIS val images are local; requested {sample_count}")

    # Greedy set cover gives the fixed image panel broad category coverage.
    remaining = set(selected_category_ids)
    available = set(candidates)
    selected_image_ids: list[int] = []
    while available and len(selected_image_ids) < sample_count:
        best_coverage = max(len(categories_by_image[item] & remaining) for item in available)
        if best_coverage:
            tied = sorted(item for item in available
                          if len(categories_by_image[item] & remaining) == best_coverage)
        else:
            tied = sorted(available)
        image_id = rng.choice(tied)
        selected_image_ids.append(image_id)
        remaining.difference_update(categories_by_image[image_id])
        available.remove(image_id)
    selected_image_ids.sort()

    prompt_categories = [{"id": int(category["id"]), "name": str(category["name"]),
                         "frequency": category["frequency"]}
                        for category in selected_categories]
    rows = []
    coco_images = []
    coco_annotations = []
    lvis_images = []
    lvis_annotations = []
    for image_id in selected_image_ids:
        image = images_by_id[image_id]
        width, height = int(image["width"]), int(image["height"])
        filename = image_paths[image_id]
        targets = {name: [] for name in category_name.values()}
        for annotation in annotations_by_image[image_id]:
            category_id = int(annotation["category_id"])
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            targets[category_name[category_id]].append([
                max(0.0, x / width), max(0.0, y / height),
                min(1.0, (x + box_width) / width), min(1.0, (y + box_height) / height),
            ])
            lvis_annotations.append(annotation)
            coco_annotations.append({**annotation, "iscrowd": int(annotation.get("iscrowd", 0))})
        lvis_image = dict(image)
        lvis_image["neg_category_ids"] = [value for value in image.get("neg_category_ids", [])
                                          if int(value) in selected_category_ids]
        lvis_image["not_exhaustive_category_ids"] = [
            value for value in image.get("not_exhaustive_category_ids", [])
            if int(value) in selected_category_ids]
        lvis_images.append(lvis_image)
        coco_images.append({"id": image_id, "file_name": str(filename.resolve()),
                            "width": width, "height": height})
        rows.append({
            "id": f"LVIS-v1-val-{image_id}", "image_id": image_id,
            "image": str(filename.resolve()), "categories": prompt_categories,
            "targets_by_category": targets, "prediction_aliases": aliases,
        })

    categories_frequency = {int(category["id"]): category["frequency"]
                            for category in selected_categories}
    annotation_categories = [dict(category, supercategory="LVIS") for category in selected_categories]
    payload = {
        "task": "coco", "dataset": "LVIS v1 validation (30-category stratified panel)",
        "split": "val", "sample_count": len(rows), "seed": seed, "input_size": 640,
        "category_count": len(selected_categories),
        "category_frequency_counts": {frequency: sum(value == frequency for value in categories_frequency.values())
                                      for frequency in ("r", "c", "f")},
        "selected_category_ids": sorted(selected_category_ids),
        "selected_image_ids": selected_image_ids,
        "license_or_access_terms": "LVIS v1 validation annotations; underlying images are local COCO 2017 train/val files",
        "protocol": "100-image panel; 10 categories per LVIS rare/common/frequent group; official LVIS evaluator required",
        "examples": rows,
    }
    official = {key: value for key, value in source.items()
                if key not in {"images", "annotations", "categories"}}
    official.update({"images": lvis_images, "annotations": lvis_annotations,
                     "categories": selected_categories})
    coco = {"info": {"description": "LVIS v1 validation balanced local panel"},
            "images": coco_images, "annotations": coco_annotations,
            "categories": annotation_categories}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "lvis_annotations.json").write_text(json.dumps(official) + "\n", encoding="utf-8")
    (output_dir / "coco_annotations.json").write_text(json.dumps(coco) + "\n", encoding="utf-8")
    return {key: payload[key] for key in ("dataset", "split", "sample_count", "category_count",
                                          "category_frequency_counts", "selected_image_ids")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes-per-frequency", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    result = prepare(args.annotations_zip, args.coco_root, args.output_dir,
                     args.classes_per_frequency, args.sample_count, args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
