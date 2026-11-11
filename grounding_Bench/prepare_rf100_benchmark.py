#!/usr/bin/env python3
"""Prepare fixed COCO-shaped panels from the locally available RF100-VL test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
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


def dataset_roster(root: Path, all_datasets: bool) -> list[tuple[str, str, str]]:
    """Return the representative five or every local test subset."""
    if not all_datasets:
        return list(DATASETS)
    known = {source: (slug, label) for source, slug, label in DATASETS}
    roster = []
    for source in sorted(path for path in root.iterdir() if path.is_dir()):
        annotation = source / "test" / "_annotations.coco.json"
        if not annotation.is_file():
            continue
        if source.name in known:
            slug, label = known[source.name]
        else:
            slug = re.sub(r"[^a-z0-9]+", "_", source.name.casefold()).strip("_")
            label = source.name.replace("-", " ").replace("_", " ").title()
        roster.append((source.name, slug, label))
    if not roster:
        raise FileNotFoundError(f"no RF100-VL test annotations found under {root}")
    return roster


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_all_subsets_panel(results: list[dict[str, Any]], output: Path,
                            samples: int, seed: int) -> dict[str, Any]:
    """Combine domain panels with globally unique COCO image/category IDs."""
    rows: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    next_image_id = next_category_id = next_annotation_id = 1
    for result in results:
        source_name = result["dataset"]
        manifest_path = Path(result["manifest"])
        annotation_path = Path(result["coco_annotations"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        category_map: dict[int, int] = {}
        for category in source_annotations["categories"]:
            old_category_id = int(category["id"])
            category_map[old_category_id] = next_category_id
            categories.append({**category, "id": next_category_id})
            next_category_id += 1
        image_map: dict[int, int] = {}
        for image in source_annotations["images"]:
            old_image_id = int(image["id"])
            image_map[old_image_id] = next_image_id
            images.append({**image, "id": next_image_id, "source_dataset": source_name,
                           "source_image_id": old_image_id})
            next_image_id += 1
        for annotation in source_annotations["annotations"]:
            annotations.append({
                **annotation,
                "id": next_annotation_id,
                "image_id": image_map[int(annotation["image_id"])],
                "category_id": category_map[int(annotation["category_id"])],
            })
            next_annotation_id += 1
        for source_row in manifest["examples"]:
            row = dict(source_row)
            row["source_dataset"] = source_name
            row["source_image_id"] = int(row["image_id"])
            row["source_categories"] = [dict(category) for category in row["categories"]]
            row["image_id"] = image_map[int(row["image_id"])]
            row["categories"] = [
                {**category, "id": category_map[int(category["id"])]}
                for category in row["categories"]
            ]
            rows.append(row)
    target = output / "manifests" / "rf100_vl_all" / "coco.json"
    coco_path = target.parent / "coco_annotations.json"
    write(coco_path, {
        "info": {"description": "RF100-VL all local test subsets, fixed seeded panels"},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": categories,
    })
    write(target, {
        "task": "coco", "dataset": "RF100-VL all local test subsets",
        "source_dataset_count": len(results), "split": "test",
        "sample_count": len(rows), "seed": seed, "input_size": 640,
        "license_or_access_terms": "Each local export records License: MIT; see per-subset source metadata",
        "protocol": f"up to {samples} seeded labeled test images per subset; boxes only; per-domain COCO bbox AP and class-aware F1",
        "annotation_file": str(coco_path.resolve()),
        "source_manifests": [result["manifest"] for result in results],
        "examples": rows,
    })
    return {"dataset_count": len(results), "sample_count": len(rows),
            "manifest": str(target.resolve()), "coco_annotations": str(coco_path.resolve())}


def prepare(root: Path, output: Path, samples: int, seed: int,
            all_datasets: bool = False) -> list[dict[str, Any]]:
    results = []
    for source_name, slug, label in dataset_roster(root, all_datasets):
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
        sample_count = min(samples, len(candidates))
        if sample_count < 1:
            raise ValueError(f"{source_name}: no labeled test images")
        selected = sorted(random.Random(seed).sample(candidates, sample_count))
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
            "info": {"description": f"RF100-VL {source_name} test subset; {sample_count} fixed images"},
            "licenses": source_data.get("licenses", []), "images": selected_images,
            "annotations": selected_annotations, "categories": categories,
        }
        write(base / "coco_annotations.json", subset_annotations)
        manifest = {
            "task": "coco", "dataset": f"RF100-VL: {label}", "source_dataset": source_name,
            "split": "test", "sample_count": len(rows), "seed": seed, "input_size": 640,
            "license_or_access_terms": "MIT per local README.dataset.txt; source metadata retained in local data tree",
            "source_annotation_sha256": sha256(annotation_file),
            "protocol": f"seeded sample of up to {samples} labeled test images; official COCO bbox AP and local class-aware F1",
            "annotation_file": str((base / "coco_annotations.json").resolve()),
            "source_root": str(source.resolve()), "selected_image_ids": selected, "examples": rows,
        }
        write(base / "coco.json", manifest)
        results.append({"dataset": source_name, "slug": slug, "images": len(images),
                        "annotations": len(annotations), "sample_count": len(rows),
                        "categories": len(categories), "annotation_sha256": manifest["source_annotation_sha256"],
                        "manifest": str(base / "coco.json"),
                        "coco_annotations": str(base / "coco_annotations.json")})
    if all_datasets:
        combined = write_all_subsets_panel(results, output, samples, seed)
        print(json.dumps({"combined_panel": combined}, ensure_ascii=False))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "domain_manifests")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--all-datasets", action="store_true",
                        help="prepare every local RF100-VL test subset with COCO annotations")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    for result in prepare(args.data_root, args.output_dir, args.samples, args.seed, args.all_datasets):
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
