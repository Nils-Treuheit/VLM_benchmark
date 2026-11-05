#!/usr/bin/env python3
"""Prepare a deterministic, local COCO-shaped panel from labeled KITTI training data.

This enables shared 2D detector comparisons on fixed images. It does not replace
the official KITTI AP_R40 evaluator or expose the hidden test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/kitti")
KITTI_CLASSES = ("Car", "Pedestrian", "Cyclist")
ALIASES = {
    "car": "Car", "person": "Pedestrian",
    "bicycle": "Cyclist",
}


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(data_root: Path, output_dir: Path, sample_count: int, seed: int) -> dict[str, Any]:
    from PIL import Image

    training = data_root / "training"
    image_dir, label_dir = training / "image_2", training / "label_2"
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths or not label_dir.is_dir():
        raise FileNotFoundError(f"KITTI training image_2/label_2 directories are missing under {training}")
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    image_stems = {path.stem for path in image_paths}
    if image_stems != set(labels):
        raise ValueError(f"KITTI image/label pairing mismatch: images={len(image_stems)}, labels={len(labels)}")
    candidates = []
    parsed_by_stem: dict[str, list[tuple[str, list[float]]]] = {}
    for image_path in image_paths:
        objects: list[tuple[str, list[float]]] = []
        for line_no, line in enumerate(labels[image_path.stem].read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if len(parts) != 15:
                raise ValueError(f"{labels[image_path.stem]}:{line_no}: expected 15 KITTI label fields")
            class_name = parts[0]
            if class_name not in KITTI_CLASSES:
                continue
            box = [float(value) for value in parts[4:8]]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"{labels[image_path.stem]}:{line_no}: invalid 2D box {box}")
            objects.append((class_name, box))
        if objects:
            parsed_by_stem[image_path.stem] = objects
            candidates.append(image_path)
    if len(candidates) < sample_count:
        raise ValueError(f"only {len(candidates)} labeled KITTI images have the selected classes")
    shuffled_candidates = list(candidates)
    random.Random(seed).shuffle(shuffled_candidates)
    train_end = int(len(shuffled_candidates) * 0.8)
    train_ids = sorted(int(path.stem) for path in shuffled_candidates[:train_end])
    validation_candidates = shuffled_candidates[train_end:]
    if len(validation_candidates) < sample_count:
        raise ValueError(f"only {len(validation_candidates)} validation images; requested {sample_count}")
    chosen = sorted(random.Random(seed + 1).sample(validation_candidates, sample_count), key=lambda path: path.stem)
    validation_ids = sorted(int(path.stem) for path in validation_candidates)
    categories = [{"id": index + 1, "name": name, "supercategory": "KITTI"}
                  for index, name in enumerate(KITTI_CLASSES)]
    category_ids = {name: index + 1 for index, name in enumerate(KITTI_CLASSES)}
    images, annotations, examples = [], [], []
    annotation_id = 1
    for image_id, image_path in enumerate(chosen, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        source_image_id = int(image_path.stem)
        images.append({"id": source_image_id, "file_name": str(image_path.resolve()),
                       "width": width, "height": height})
        targets = {name: [] for name in KITTI_CLASSES}
        for class_name, box in parsed_by_stem[image_path.stem]:
            x1, y1, x2, y2 = box
            clipped = [max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
                       max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            bbox = [clipped[0], clipped[1], clipped[2] - clipped[0], clipped[3] - clipped[1]]
            annotations.append({"id": annotation_id, "image_id": source_image_id,
                                "category_id": category_ids[class_name], "bbox": bbox,
                                "area": bbox[2] * bbox[3], "iscrowd": 0})
            annotation_id += 1
            targets[class_name].append([clipped[0] / width, clipped[1] / height,
                                        clipped[2] / width, clipped[3] / height])
        examples.append({"id": f"KITTI-training-{image_path.stem}", "image_id": source_image_id,
                         "image": str(image_path.resolve()),
                         "categories": categories, "targets_by_category": targets,
                         "prediction_aliases": ALIASES})
    selected_ids = [int(path.stem) for path in chosen]
    all_label_paths = [labels[path.stem] for path in chosen]
    base = output_dir / "kitti_training_2d"
    split_payload = {
        "dataset": "KITTI object detection training",
        "source_split": "training",
        "seed": seed,
        "train_fraction": 0.8,
        "train_image_ids": train_ids,
        "validation_image_ids": validation_ids,
        "protocol": "Deterministic 80/20 partition of the labeled training split; test labels are unavailable",
    }
    write_json(base / "splits.json", split_payload)
    split_sha256 = hashlib.sha256((base / "splits.json").read_bytes()).hexdigest()
    annotations_payload = {
        "info": {"description": "KITTI object detection labeled training panel; selected 2D classes only"},
        "images": images, "annotations": annotations, "categories": categories,
    }
    write_json(base / "coco_annotations.json", annotations_payload)
    write_json(base / "coco.json", {
        "task": "coco", "dataset": "KITTI object training (local 2D panel)",
        "split": "training-derived-validation", "sample_count": len(examples),
        "seed": seed, "partition_seed": seed, "sample_seed": seed + 1, "input_size": 640,
        "license_or_access_terms": "KITTI object dataset terms; source license metadata is not present in the local tree",
        "protocol": "Seeded 80/20 partition of labeled training data; 100-image validation panel; local COCOeval adaptation on Car, Pedestrian, Cyclist; not official KITTI AP_R40",
        "source_root": str(data_root.resolve()), "source_labels_sha256": sha256_files(all_label_paths),
        "split_manifest_sha256": split_sha256,
        "selected_image_ids": selected_ids, "validation_image_ids": validation_ids,
        "excluded_classes": ["Van", "Truck", "Person_sitting", "Tram", "Misc", "DontCare"],
        "examples": examples,
    })
    return {"dataset": "KITTI object training (local 2D panel)", "image_count": len(image_paths),
            "labeled_panel_candidates": len(candidates), "sample_count": len(examples),
            "train_split_size": len(train_ids), "validation_split_size": len(validation_ids),
            "object_count": len(annotations), "categories": list(KITTI_CLASSES),
            "selected_image_ids": selected_ids, "source_labels_sha256": sha256_files(all_label_paths),
            "manifest": str((base / "coco.json").resolve()),
            "coco_annotations": str((base / "coco_annotations.json").resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "domain_manifests")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    print(json.dumps(prepare(args.data_root, args.output_dir, args.samples, args.seed), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
