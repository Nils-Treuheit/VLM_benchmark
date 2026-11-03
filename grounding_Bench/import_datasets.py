#!/usr/bin/env python3
"""Convert locally obtained benchmark exports into the shared source JSONL schema.

This importer is offline. It reads official evaluation files and local images,
normalizes annotations, and never copies image bytes or fetches data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

HERE = Path(__file__).resolve().parent


def read_lines(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        content = handle.read()
    try:
        document = json.loads(content)
        if isinstance(document, dict) and isinstance(document.get("annotations"), list):
            return [row for row in document["annotations"] if isinstance(row, dict)]
        if isinstance(document, list):
            return [row for row in document if isinstance(row, dict)]
        if isinstance(document, dict):
            return [document]
    except json.JSONDecodeError:
        pass
    rows = []
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def image_size(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"local image is missing: {path}")
    with Image.open(path) as image:
        return image.size


def normalized_box(box: Iterable[float], width: int, height: int) -> list[float]:
    values = [float(value) for value in box]
    if len(values) != 4 or not (0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height):
        raise ValueError(f"box must be pixel xyxy within {width}x{height}: {values}")
    return [values[0] / width, values[1] / height, values[2] / width, values[3] / height]


def write_source(output_dir: Path, task: str, rows: list[dict[str, Any]], license_terms: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{task}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "dataset": {"refcocog": "RefCOCOg", "coco": "COCO", "screenspot_pro": "ScreenSpot-Pro",
                    "hiertext": "HierText"}[task],
        "split": {"refcocog": "test", "coco": "val2017", "screenspot_pro": "test",
                  "hiertext": "validation"}[task],
        "license": license_terms,
    }
    (output_dir / f"{task}.meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{task}: imported {len(rows)} local annotations -> {target}")


def import_refcocog(source: Path, image_root: Path) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(read_lines(source)):
        image_path = (image_root / item["image_path"]).resolve()
        width, height = image_size(image_path)
        categories = item.get("categories") or list(item.get("gt", {}))
        if not categories:
            raise ValueError(f"RefCOCOg row {index} has no referring expression")
        phrase = str(categories[0])
        boxes = item.get("gt", {}).get(phrase)
        if not boxes:
            raise ValueError(f"RefCOCOg row {index} has no target box for {phrase!r}")
        rows.append({"id": f"refcocog-{index}", "image": str(image_path), "phrase": phrase,
                     "target_box": normalized_box(boxes[0], width, height)})
    return rows


def import_coco(source: Path, image_root: Path, instance_json: Path) -> list[dict[str, Any]]:
    annotations = json.loads(instance_json.read_text(encoding="utf-8"))
    category_ids = {str(category["name"]): int(category["id"]) for category in annotations["categories"]}
    rows = []
    for index, item in enumerate(read_lines(source)):
        image_path = (image_root / item["image_path"]).resolve()
        width, height = image_size(image_path)
        image_id_match = re.search(r"(\d{12})\.[^.]+$", image_path.name)
        if not image_id_match:
            raise ValueError(f"COCO image filename lacks a 12-digit image ID: {image_path.name}")
        image_id = int(image_id_match.group(1))
        names = [str(name) for name in item["categories"]]
        unknown = [name for name in names if name not in category_ids]
        if unknown:
            raise ValueError(f"COCO categories absent from instances metadata: {unknown}")
        target_map = item.get("gt", {})
        categories = [{"id": category_ids[name], "name": name} for name in names]
        targets = {name: [normalized_box(box, width, height) for box in target_map.get(name, [])]
                   for name in names}
        rows.append({"id": f"coco-{image_id}", "image": str(image_path), "image_id": image_id,
                     "categories": categories, "targets_by_category": targets})
    return rows


def import_screenspot(annotations_dir: Path, image_root: Path) -> list[dict[str, Any]]:
    rows = []
    index = 0
    for path in sorted(annotations_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, list):
            raise ValueError(f"ScreenSpot-Pro annotation file must contain a list: {path}")
        for item in document:
            width, height = (int(value) for value in item["img_size"])
            image_path = (image_root / item["img_filename"]).resolve()
            actual_size = image_size(image_path)
            if actual_size != (width, height):
                raise ValueError(f"ScreenSpot-Pro annotation dimensions disagree with image: {image_path}")
            box = normalized_box(item["bbox"], width, height)
            rows.append({"id": str(item.get("id", f"screenspot-{index}")), "image": str(image_path),
                         "instruction": str(item["instruction"]), "target_box": box})
            index += 1
    return rows


def polygon_box(vertices: list[list[float]], width: int, height: int) -> list[float] | None:
    if not vertices:
        return None
    xs = [float(point[0]) for point in vertices]
    ys = [float(point[1]) for point in vertices]
    left, top = max(0.0, min(xs)), max(0.0, min(ys))
    right, bottom = min(float(width), max(xs)), min(float(height), max(ys))
    if left >= right or top >= bottom:
        return None
    return normalized_box([left, top, right, bottom], width, height)


def import_hiertext(source: Path, image_root: Path) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(read_lines(source)):
        image_id = str(item["image_id"])
        image_name = image_id if Path(image_id).suffix else image_id + ".jpg"
        image_path = (image_root / image_name).resolve()
        width, height = int(item["image_width"]), int(item["image_height"])
        if image_size(image_path) != (width, height):
            raise ValueError(f"HierText annotation dimensions disagree with image: {image_path}")
        boxes = []
        for paragraph in item.get("paragraphs", []):
            for line in paragraph.get("lines", []):
                for word in line.get("words", []):
                    if not word.get("legible", True):
                        continue
                    box = polygon_box(word.get("vertices", []), width, height)
                    if box:
                        boxes.append(box)
        rows.append({"id": f"hiertext-{image_id}", "image": str(image_path), "target_boxes": boxes})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "source")
    parser.add_argument("--licenses-file", type=Path, required=True,
                        help="JSON object with verified `refcocog`, `coco`, `screenspot_pro`, `hiertext` terms")
    parser.add_argument("--refcocog-jsonl", type=Path, required=True,
                        help="LocateAnything/Rex-Omni RefCOCOg_test.jsonl")
    parser.add_argument("--eval-data-root", type=Path, required=True,
                        help="Root containing the local COCO train2014 images referenced by RefCOCOg")
    parser.add_argument("--coco-jsonl", type=Path, required=True)
    parser.add_argument("--coco-image-root", type=Path, required=True)
    parser.add_argument("--coco-instances", type=Path, required=True,
                        help="Official instances_val2017.json used to map category names to IDs")
    parser.add_argument("--screenspot-annotations", type=Path, required=True)
    parser.add_argument("--screenspot-images", type=Path, required=True)
    parser.add_argument("--hiertext-validation", type=Path, required=True,
                        help="Official HierText validation.jsonl or .jsonl.gz")
    parser.add_argument("--hiertext-images", type=Path, required=True)
    args = parser.parse_args()
    license_terms = json.loads(args.licenses_file.read_text(encoding="utf-8"))
    for task in ("refcocog", "coco", "screenspot_pro", "hiertext"):
        if not license_terms.get(task):
            parser.error(f"--licenses-file must include verified terms for {task}")
    sources = {
        "refcocog": import_refcocog(args.refcocog_jsonl, args.eval_data_root),
        "coco": import_coco(args.coco_jsonl, args.coco_image_root, args.coco_instances),
        "screenspot_pro": import_screenspot(args.screenspot_annotations, args.screenspot_images),
        "hiertext": import_hiertext(args.hiertext_validation, args.hiertext_images),
    }
    for task, rows in sources.items():
        write_source(args.output_dir, task, rows, str(license_terms[task]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
