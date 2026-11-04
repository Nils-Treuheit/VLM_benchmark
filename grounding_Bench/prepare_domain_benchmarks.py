#!/usr/bin/env python3
"""Prepare shared COCO-shaped manifests for local DOTA v1 and BOP-H3 HOPE v2.

The manifests preserve the dataset split and source paths. DOTA quadrilaterals
are reduced to axis-aligned boxes for common prompt-detector comparison; the
native YOLO26-L OBB result remains a separate task.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT_DATA = Path("/mnt/HDD1/Project_Data")
DOTA_NAMES = (
    "plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle",
    "large-vehicle", "ship", "tennis-court", "basketball-court", "storage-tank",
    "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter",
)
HOPE_NAMES = (
    "AlphabetSoup", "BBQSauce", "Butter", "Cherries", "ChocolatePudding", "Cookies",
    "Corn", "CreamCheese", "GranolaBars", "GreenBeans", "Ketchup", "Macaroni&Cheese",
    "Mayo", "Milk", "Mushrooms", "Mustard", "OrangeJuice", "Parmesan", "Peaches",
    "Peas&Carrots", "Pineapple", "Popcorn", "Raisins", "SaladDressing", "Spaghetti",
    "TomatoSauce", "Tuna", "Yogurt",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def manifest_rows(dataset: str, split: str, images: list[dict[str, Any]],
                  annotations: list[dict[str, Any]], categories: list[dict[str, Any]],
                  sample_count: int, seed: int) -> tuple[list[dict[str, Any]], list[int]]:
    by_image: dict[int, list[dict[str, Any]]] = {int(image["id"]): [] for image in images}
    for annotation in annotations:
        by_image[int(annotation["image_id"])].append(annotation)
    candidates = [image for image in images if by_image[int(image["id"])]]
    if len(candidates) < sample_count:
        raise ValueError(f"{dataset}: only {len(candidates)} labeled images; requested {sample_count}")
    selected = sorted(random.Random(seed).sample(candidates, sample_count), key=lambda image: int(image["id"]))
    rows = []
    names = {int(category["id"]): str(category["name"]) for category in categories}
    category_prompt = [{"id": int(category["id"]), "name": str(category["name"])} for category in categories]
    for image in selected:
        image_id = int(image["id"])
        width, height = int(image["width"]), int(image["height"])
        targets = {name: [] for name in names.values()}
        for ann in by_image[image_id]:
            x, y, box_width, box_height = [float(value) for value in ann["bbox"]]
            targets[names[int(ann["category_id"])]].append([
                max(0.0, x / width), max(0.0, y / height),
                min(1.0, (x + box_width) / width), min(1.0, (y + box_height) / height),
            ])
        rows.append({
            "id": f"{dataset}-{image_id}", "image_id": image_id,
            "image": str(Path(image["file_name"]).resolve()),
            "categories": category_prompt, "targets_by_category": targets,
            "prediction_aliases": ({"airplane": "plane", "car": "small-vehicle",
                                    "truck": "large-vehicle", "boat": "ship"}
                                   if dataset == "DOTA-v1" else {}),
        })
    return rows, [int(image["id"]) for image in selected]


def prepare_dota(root: Path, output: Path, sample_count: int, seed: int) -> dict[str, Any]:
    from PIL import Image

    image_dir = root / "val" / "images"
    label_dir = root / "val" / "labels_yolo"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"DOTA validation image/label directories missing under {root}")
    categories = [{"id": index + 1, "name": name, "supercategory": "DOTA"}
                  for index, name in enumerate(DOTA_NAMES)]
    images, annotations = [], []
    ann_id = 1
    image_files = sorted(image_dir.glob("*.png"))
    for image_id, image_path in enumerate(image_files, 1):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": str(image_path.resolve()),
                       "width": width, "height": height})
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 9:
                raise ValueError(f"{label_path}:{line_number}: expected class ID and eight normalized polygon values")
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(DOTA_NAMES):
                raise ValueError(f"{label_path}:{line_number}: class ID {class_id} outside DOTA vocabulary")
            coordinates = [float(value) for value in parts[1:]]
            xs = [coordinates[i] * width for i in (0, 2, 4, 6)]
            ys = [coordinates[i] * height for i in (1, 3, 5, 7)]
            x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
            y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
            if x2 <= x1 or y2 <= y1:
                continue
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": class_id + 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            ann_id += 1
    rows, selected_ids = manifest_rows("DOTA-v1", "val", images, annotations, categories, sample_count, seed)
    annotation_payload = {"info": {"description": "DOTA v1.0 validation; OBB polygons converted to HBB for cross-model comparison"},
                          "images": images, "annotations": annotations, "categories": categories}
    base = output / "dota_v1"
    write_json(base / "coco_annotations.json", annotation_payload)
    write_json(base / "coco.json", {
        "task": "coco", "dataset": "DOTA v1.0", "split": "val",
        "sample_count": len(rows), "seed": seed, "input_size": 1024,
        "license_or_access_terms": "DOTA dataset terms restrict use to academic purposes and prohibit commercial use",
        "protocol": "horizontal-box COCO AP and class-aware F1; native OBB reported separately",
        "annotation_file": str((base / "coco_annotations.json").resolve()),
        "source_root": str(root.resolve()), "selected_image_ids": selected_ids, "examples": rows,
    })
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(DOTA_NAMES))
    (base / "DOTAv1.yaml").write_text(
        f"path: {root.resolve()}\nval: val/images\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    return {"dataset": "DOTA v1.0", "images": len(images), "annotations": len(annotations),
            "sample_count": len(rows), "annotation_file": str(base / "coco_annotations.json"),
            "manifest": str(base / "coco.json"), "obb_config": str(base / "DOTAv1.yaml")}


def prepare_hope(root: Path, output: Path, sample_count: int, seed: int) -> dict[str, Any]:
    categories = [{"id": index + 1, "name": name, "supercategory": "HOPE-v2"}
                  for index, name in enumerate(HOPE_NAMES)]
    images, annotations = [], []
    ann_id = 1
    image_id = 1
    for scene_dir in sorted((root / "val").iterdir()):
        source = scene_dir / "scene_gt_coco.json"
        if not scene_dir.is_dir() or not source.is_file():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        for source_image in payload.get("images", []):
            old_id = int(source_image["id"])
            filename = (scene_dir / source_image["file_name"]).resolve()
            if not filename.is_file():
                raise FileNotFoundError(f"HOPE image is missing: {filename}")
            images.append({"id": image_id, "file_name": str(filename),
                           "width": int(source_image["width"]), "height": int(source_image["height"]),
                           "source_scene": scene_dir.name, "source_image_id": old_id})
            for source_ann in payload.get("annotations", []):
                if int(source_ann["image_id"]) != old_id:
                    continue
                category_id = int(source_ann["category_id"])
                if category_id < 1 or category_id > len(HOPE_NAMES):
                    raise ValueError(f"HOPE category ID {category_id} outside the canonical object list")
                annotation = {key: value for key, value in source_ann.items()
                              if key not in {"id", "image_id", "category_id"}}
                annotation.update({"id": ann_id, "image_id": image_id, "category_id": category_id})
                annotations.append(annotation)
                ann_id += 1
            image_id += 1
    rows, selected_ids = manifest_rows("BOP-H3-HOPE-v2", "val", images, annotations,
                                       categories, sample_count, seed)
    base = output / "bop_h3_hope_v2"
    write_json(base / "coco_annotations.json", {
        "info": {"description": "BOP-H3 HOPE v2 validation"}, "images": images,
        "annotations": annotations, "categories": categories,
    })
    write_json(base / "coco.json", {
        "task": "coco", "dataset": "BOP-H3 HOPE v2", "split": "val",
        "sample_count": len(rows), "seed": seed, "input_size": 640,
        "license_or_access_terms": "CC BY-NC-SA 4.0 as recorded in the local HOPE dataset_info.md",
        "protocol": "2D HBB detection/classification; instance masks scored only for mask-capable models",
        "annotation_file": str((base / "coco_annotations.json").resolve()),
        "source_root": str(root.resolve()), "selected_image_ids": selected_ids, "examples": rows,
    })
    return {"dataset": "BOP-H3 HOPE v2", "images": len(images), "annotations": len(annotations),
            "sample_count": len(rows), "annotation_file": str(base / "coco_annotations.json"),
            "manifest": str(base / "coco.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-data", type=Path, default=DEFAULT_PROJECT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "domain_manifests")
    parser.add_argument("--dota-samples", type=int, default=100)
    parser.add_argument("--hope-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.dota_samples < 1 or args.hope_samples < 1:
        parser.error("sample counts must be positive")
    results = [
        prepare_dota(args.project_data / "public_datasets" / "dotav1", args.output_dir,
                     args.dota_samples, args.seed),
        prepare_hope(args.project_data / "bop_h3" / "hope", args.output_dir,
                     args.hope_samples, args.seed),
    ]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
