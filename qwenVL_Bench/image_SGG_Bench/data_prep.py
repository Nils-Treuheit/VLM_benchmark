"""Deterministic real-data panels for image scene graph generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

CODE_VERSION = "image-sgg-prep-v1"
PANEL_SIZE = 200
VG_H5_SHA256 = "ad52a01f8e6142bb7bc62d91f97dad1567f44772a23eac68d60a5b7766ffe4b1"
VG_DICT_SHA256 = "1255873ae0250555f59d20f8d1e5a4bdf22b331119502c850f765173b6963ca1"
VG_H5_REVISION = "kb-kim/LLM4SGG@73ff29cb7587ab5cf011772e863172a6fa468000"
VG_DICT_REVISION = "Prk10/VG150_VisualGenome@1bcd54b861bc5d3604b83da43c0fd2e7d71d7e97"
VG_IMAGE_DATA_ARCHIVE_SHA256 = "b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a"
VG_LOADER_REVISION = "rowanz/neural-motifs@d05a251b705cedaa51599bf2906cfa4653b7a761"
VG_PREPROCESS_REVISION = "danfeiX/scene-graph-TF-release@4c9e3c6"
VG_CORRUPT_IDS = {1592, 1722, 4616, 4617}
PREDICATE_ATTRIBUTE = "is"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, revision: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if revision:
        row["revision"] = revision
    return row


def canonical_json_bytes(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def manifest_write(output_dir: Path, dataset: str, manifest: dict[str, Any]) -> Path:
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / "manifest.json"
    payload = canonical_json_bytes(manifest)
    target.write_bytes(payload)
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{sha256_bytes(payload)}  {target.name}\n", encoding="ascii"
    )
    return target


def _selection_tie(seed: int, dataset: str, image_id: str) -> str:
    return sha256_bytes(f"{seed}|{dataset}|{image_id}".encode("utf-8"))


def select_panel(candidates: list[dict[str, Any]], seed: int, dataset: str) -> list[dict[str, Any]]:
    """Greedily add uncovered predicates, then fill deterministically by seed."""
    if len(candidates) < PANEL_SIZE:
        raise ValueError(f"{dataset}: only {len(candidates)} relation-bearing joined images; need {PANEL_SIZE}")
    by_id = {str(row["image_id"]): row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError(f"{dataset}: duplicate candidate image IDs")
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    uncovered = {str(p) for row in candidates for p in row["predicate_ids"]}
    while remaining and len(selected) < PANEL_SIZE and uncovered:
        ranked = []
        for row in remaining:
            new = uncovered.intersection(str(p) for p in row["predicate_ids"])
            ranked.append((-len(new), _selection_tie(seed, dataset, str(row["image_id"])), row))
        ranked.sort(key=lambda entry: (entry[0], entry[1]))
        if not ranked or ranked[0][0] == 0:
            break
        chosen = ranked[0][2]
        selected.append(chosen)
        uncovered.difference_update(str(p) for p in chosen["predicate_ids"])
        remaining.remove(chosen)
    if len(selected) < PANEL_SIZE:
        filler = [row for row in remaining if str(row["image_id"]) not in {str(x["image_id"]) for x in selected}]
        rng = random.Random(seed)
        rng.shuffle(filler)
        selected.extend(filler[: PANEL_SIZE - len(selected)])
    if len(selected) != PANEL_SIZE or len({str(row["image_id"]) for row in selected}) != PANEL_SIZE:
        raise ValueError(f"{dataset}: deterministic selection did not produce {PANEL_SIZE} unique IDs")
    return sorted(selected, key=lambda row: str(row["image_id"]))


def _box_xyxy(box: Iterable[float]) -> list[float]:
    result = [float(value) for value in box]
    if len(result) != 4 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"invalid box: {result!r}")
    x1, y1, x2, y2 = result
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"non-positive box: {result!r}")
    return [x1, y1, x2, y2]


def _validate_graph(row: dict[str, Any], *, width: int, height: int) -> None:
    objects = row["ground_truth"]["objects"]
    relations = row["ground_truth"]["relations"]
    object_ids = set()
    for item in objects:
        if not item.get("label") or item["label"] == "__background__":
            raise ValueError(f"{row['image_id']}: invalid object label")
        if item["id"] in object_ids:
            raise ValueError(f"{row['image_id']}: duplicate object ID {item['id']}")
        object_ids.add(item["id"])
        box = _box_xyxy(item["box"])
        if box[0] < 0 or box[1] < 0 or box[2] > width + 1e-4 or box[3] > height + 1e-4:
            raise ValueError(f"{row['image_id']}: box outside {width}x{height}: {box}")
    if not relations:
        raise ValueError(f"{row['image_id']}: empty relationship set")
    for rel in relations:
        if rel["subject"] not in object_ids or rel["object"] not in object_ids:
            raise ValueError(f"{row['image_id']}: relation endpoint is not in this image")
        if not rel["predicate"] or rel["predicate"] == PREDICATE_ATTRIBUTE:
            raise ValueError(f"{row['image_id']}: invalid predicate {rel['predicate']!r}")


def _support(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    objects: Counter[str] = Counter()
    predicates: Counter[str] = Counter()
    for row in rows:
        objects.update(obj["label"] for obj in row["ground_truth"]["objects"])
        predicates.update(rel["predicate"] for rel in row["ground_truth"]["relations"])
    return dict(sorted(objects.items())), dict(sorted(predicates.items()))


def _manifest(dataset: str, display_name: str, split: str, seed: int,
              provenance: dict[str, Any], rows: list[dict[str, Any]],
              object_vocabulary: list[str], predicate_vocabulary: list[str]) -> dict[str, Any]:
    obj_support, pred_support = _support(rows)
    for row in rows:
        _validate_graph(row, width=row["image"]["width"], height=row["image"]["height"])
    vocab = {"objects": object_vocabulary, "predicates": predicate_vocabulary}
    vocab_hash = sha256_bytes(json.dumps(
        vocab, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    vocab["hash"] = vocab_hash
    return {
        "schema_version": 1,
        "dataset": dataset,
        "display_name": display_name,
        "split": split,
        "selection": {
            "seed": seed,
            "panel_size": PANEL_SIZE,
            "algorithm": "greedy_max_uncovered_predicates_then_seeded_random_fill",
            "tie_break": "sha256(seed|dataset|image_id)",
            "code_version": CODE_VERSION,
        },
        "source_provenance": provenance,
        "vocabulary": vocab,
        "vocabulary_sha256": vocab_hash,
        "selected_ids": [str(row["image_id"]) for row in rows],
        "class_support": obj_support,
        "predicate_support": pred_support,
        "items": rows,
    }


def _find_vg_image(root: Path, image_id: int) -> Path:
    candidates = [root / "VG_100K" / f"{image_id}.jpg", root / "VG_100K_2" / f"{image_id}.jpg"]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"VG image {image_id}: expected exactly one file across VG_100K and VG_100K_2, found {found}")
    return found[0]


def prepare_vg(root: Path, output_dir: Path, seed: int, audit: bool = False) -> Path:
    import h5py

    h5_path = root / "VG-SGG-with-attri.h5"
    dict_path = root / "VG-SGG-dicts-with-attri.json"
    image_data_path = root / "image_data.json"
    for path in (h5_path, dict_path, image_data_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    h5_hash = sha256_file(h5_path)
    dict_hash = sha256_file(dict_path)
    if h5_hash != VG_H5_SHA256:
        raise ValueError(f"VG HDF5 SHA-256 mismatch: {h5_hash}")
    if dict_hash != VG_DICT_SHA256:
        raise ValueError(f"VG dictionary SHA-256 mismatch: {dict_hash}")
    mappings = json.loads(dict_path.read_text(encoding="utf-8"))
    obj_map = {int(key): value for key, value in mappings["idx_to_label"].items() if int(key) != 0}
    pred_map = {int(key): value.lower().strip() for key, value in mappings["idx_to_predicate"].items() if int(key) != 0}
    if len(set(obj_map.values())) != 150 or len(set(pred_map.values())) != 50:
        raise ValueError("VG dictionary must contain exactly 150 unique object labels and 50 predicates")
    raw_meta = json.loads(image_data_path.read_text(encoding="utf-8"))
    filtered = [row for row in raw_meta if int(row["image_id"]) not in VG_CORRUPT_IDS]
    if len(filtered) != 108073:
        raise ValueError(f"VG official image metadata filtered count {len(filtered)} != 108073")
    ids = [int(row["image_id"]) for row in filtered]
    if len(set(ids)) != len(ids):
        raise ValueError("VG image_data.json contains duplicate image IDs")

    candidates: list[dict[str, Any]] = []
    excluded_out_of_bounds_rows = 0
    with h5py.File(h5_path, "r") as h5:
        required = ("split", "img_to_first_box", "img_to_last_box", "img_to_first_rel", "img_to_last_rel",
                    "labels", "boxes_1024", "relationships", "predicates")
        missing = set(required) - set(h5.keys())
        if missing:
            raise ValueError(f"VG HDF5 missing required arrays: {sorted(missing)}")
        nrows = len(h5["split"])
        if nrows != len(filtered):
            raise ValueError(f"VG HDF5 image axis {nrows} != filtered official metadata rows {len(filtered)}")
        if any(len(h5[name]) != nrows for name in required[:5]):
            raise ValueError("VG image-indexed arrays have inconsistent lengths")
        all_labels = h5["labels"][:, 0]
        all_boxes = h5["boxes_1024"][:]
        all_relations = h5["relationships"][:]
        all_predicates = h5["predicates"][:, 0]
        for row_index, meta in enumerate(filtered):
            if int(h5["split"][row_index]) != 2:
                continue
            first_box = int(h5["img_to_first_box"][row_index])
            last_box = int(h5["img_to_last_box"][row_index])
            first_rel = int(h5["img_to_first_rel"][row_index])
            last_rel = int(h5["img_to_last_rel"][row_index])
            if min(first_box, last_box, first_rel, last_rel) < 0:
                continue
            if last_box < first_box or last_rel < first_rel:
                raise ValueError(f"VG row {row_index}: reversed annotation span")
            if last_box >= len(all_labels) or last_box >= len(all_boxes) or last_rel >= len(all_relations) or last_rel >= len(all_predicates):
                raise ValueError(f"VG row {row_index}: annotation span exceeds flat HDF5 arrays")
            width, height = int(meta["width"]), int(meta["height"])
            if width <= 0 or height <= 0:
                raise ValueError(f"VG image {meta['image_id']}: invalid metadata dimensions")
            scale = 1024.0 / max(width, height)
            labels = [int(x) for x in all_labels[first_box:last_box + 1]]
            boxes = all_boxes[first_box:last_box + 1]
            objects = []
            boxes_in_bounds = True
            for local, (label_id, raw_box) in enumerate(zip(labels, boxes)):
                if label_id not in obj_map:
                    raise ValueError(f"VG row {row_index}: object category {label_id} absent from dictionary")
                xc, yc, box_w, box_h = [float(x) for x in raw_box]
                if not all(math.isfinite(x) for x in (xc, yc, box_w, box_h)) or box_w <= 0 or box_h <= 0:
                    raise ValueError(f"VG row {row_index}: invalid center-size box {raw_box}")
                xyxy = [(xc - box_w / 2) / scale, (yc - box_h / 2) / scale,
                        (xc + box_w / 2) / scale, (yc + box_h / 2) / scale]
                if xyxy[0] < 0 or xyxy[1] < 0 or xyxy[2] > width or xyxy[3] > height:
                    boxes_in_bounds = False
                objects.append({"id": f"vg-row{row_index}-box{first_box + local}",
                                "label": obj_map[label_id], "box": _box_xyxy(xyxy),
                                "source_label_id": label_id, "source_box_index": first_box + local})
            if not boxes_in_bounds:
                excluded_out_of_bounds_rows += 1
                continue
            relations = []
            for rel_index, pair in enumerate(all_relations[first_rel:last_rel + 1], start=first_rel):
                subject_abs, object_abs = int(pair[0]), int(pair[1])
                if subject_abs < first_box or subject_abs > last_box or object_abs < first_box or object_abs > last_box:
                    raise ValueError(f"VG row {row_index}: relationship endpoint outside current image span")
                predicate_id = int(all_predicates[rel_index])
                if predicate_id not in pred_map:
                    raise ValueError(f"VG row {row_index}: predicate {predicate_id} absent from dictionary")
                relations.append({"subject": f"vg-row{row_index}-box{subject_abs}",
                                  "predicate": pred_map[predicate_id],
                                  "object": f"vg-row{row_index}-box{object_abs}",
                                  "source_predicate_id": predicate_id})
            if not relations:
                continue
            candidates.append({"image_id": int(meta["image_id"]), "source_row": row_index,
                               "predicate_ids": sorted({int(h5["predicates"][idx][0]) for idx in range(first_rel, last_rel + 1)}),
                               "meta": meta, "objects": objects, "relations": relations})
        source_shape = {name: list(h5[name].shape) for name in required}
        split_counts = {str(code): int(np.count_nonzero(h5["split"][:] == code)) for code in (0, 2)}

    selected = select_panel(candidates, seed, "VG-150")
    rows = []
    for candidate in selected:
        image_id = int(candidate["image_id"])
        meta = candidate["meta"]
        image_path = _find_vg_image(root, image_id)
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        if (width, height) != (int(meta["width"]), int(meta["height"])):
            raise ValueError(f"VG image {image_id}: local dimensions {(width, height)} != metadata {(meta['width'], meta['height'])}")
        row = {
            "image_id": str(image_id), "source_row": candidate["source_row"],
            "image": {"path": str(image_path), "sha256": sha256_file(image_path), "width": width, "height": height},
            "ground_truth": {"objects": candidate["objects"], "relations": candidate["relations"]},
            "source_split": "test", "source_split_code": 2,
        }
        _validate_graph(row, width=width, height=height)
        rows.append(row)

    provenance = {
        "hdf5": {**file_record(h5_path, VG_H5_REVISION), "sha256": h5_hash},
        "dictionary": {**file_record(dict_path, VG_DICT_REVISION), "sha256": dict_hash},
        "image_data": file_record(image_data_path, "Visual Genome official metadata; origin archive sha256=" + VG_IMAGE_DATA_ARCHIVE_SHA256),
        "terms_reference": {
            "url": "https://visualgenome.org/home",
            "license": "CC BY 4.0",
            "license_reference": "https://osg-htc.org/services/osdf/data/purdue-anvil-visual-genome/",
            "note": "Source images remain at their original paths; this benchmark stores references and hashes, not image copies.",
        },
        "preprocessing": VG_PREPROCESS_REVISION,
        "loader": VG_LOADER_REVISION,
        "split_code": 2,
        "excluded_corrupt_image_ids": sorted(VG_CORRUPT_IDS),
        "hdf5_shapes": source_shape,
        "split_counts": split_counts,
        "test_rows_excluded_for_strict_box_bounds": excluded_out_of_bounds_rows,
    }
    manifest = _manifest("vg150", "VG-150", "canonical_test", seed, provenance, rows,
                         [obj_map[k] for k in sorted(obj_map)], [pred_map[k] for k in sorted(pred_map)])
    target = manifest_write(output_dir, "vg150", manifest)
    if audit:
        render_vg_audit(rows, output_dir)
    return target


def render_vg_audit(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    chosen = sorted(rows, key=lambda row: _selection_tie(3407, "VG-150|visual-audit", row["image_id"]))[:12]
    directory = output_dir / "audit" / "vg150"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in chosen:
        with Image.open(row["image"]["path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for obj in row["ground_truth"]["objects"]:
            x1, y1, x2, y2 = obj["box"]
            draw.rectangle((round(x1), round(y1), round(x2), round(y2)), outline=(255, 32, 32), width=max(2, round(min(width, height) / 300)))
            draw.text((max(0, round(x1)), max(0, round(y1) - 12)), obj["label"], fill=(255, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
        path = directory / f"{row['image_id']}.jpg"
        image.save(path, quality=92)
        paths.append(path)
    (directory / "README.txt").write_text(
        "Deterministic real-image HDF5 audit overlays. Inspect image identity, box alignment, bounds, and labels.\n"
        + "\n".join(path.name for path in paths) + "\n", encoding="utf-8"
    )
    return paths


def _read_open_image_classes(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0] == "LabelName":
                continue
            if row[0] in mapping:
                raise ValueError(f"duplicate Open Images class ID {row[0]!r} in {path}")
            mapping[row[0]] = row[1].strip()
    if not mapping:
        raise ValueError(f"Open Images class mapping is empty: {path}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Open Images class descriptions are not unique: {path}")
    return mapping


def _norm_coord(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite Open Images coordinate {value!r}")
    return number


def prepare_openimages(root: Path, output_dir: Path, seed: int) -> Path:
    base = root / "open_images_v7" / "validation"
    rel_path = base / "labels" / "relationships.csv"
    det_path = base / "labels" / "detections.csv"
    class_path = base / "metadata" / "classes.csv"
    image_dir = base / "data"
    for path in (rel_path, det_path, class_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    class_map = _read_open_image_classes(class_path)
    rel_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predicate_vocab: set[str] = set()
    endpoint_wanted: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    with rel_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"ImageID", "LabelName1", "LabelName2", "XMin1", "XMax1", "YMin1", "YMax1",
                    "XMin2", "XMax2", "YMin2", "YMax2", "RelationshipLabel"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Open Images relationships.csv missing columns {sorted(required - set(reader.fieldnames or []))}")
        for line, raw in enumerate(reader, 2):
            pred = raw["RelationshipLabel"].strip().lower()
            if pred == "is":
                continue
            predicate_vocab.add(pred)
            image_id = raw["ImageID"].strip()
            l1, l2 = raw["LabelName1"].strip(), raw["LabelName2"].strip()
            if l1 not in class_map or l2 not in class_map:
                raise ValueError(f"Open Images relationship line {line}: object label absent from class descriptions")
            def endpoint(num: int, label: str):
                vals = (_norm_coord(raw[f"XMin{num}"]), _norm_coord(raw[f"YMin{num}"]),
                        _norm_coord(raw[f"XMax{num}"]), _norm_coord(raw[f"YMax{num}"]))
                if not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
                    raise ValueError(f"Open Images relationship line {line}: invalid normalized box {vals}")
                endpoint_wanted[(image_id, label)].append(vals)
                return vals
            b1, b2 = endpoint(1, l1), endpoint(2, l2)
            rel_by_image[image_id].append({"line": line, "subject_label_id": l1, "object_label_id": l2,
                                           "subject_box": b1, "object_box": b2, "predicate": pred})

    needed_keys = set(endpoint_wanted)
    detections: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    with det_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Open Images detections.csv missing columns {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            key = (raw["ImageID"].strip(), raw["LabelName"].strip())
            if key not in needed_keys:
                continue
            box = (_norm_coord(raw["XMin"]), _norm_coord(raw["YMin"]),
                   _norm_coord(raw["XMax"]), _norm_coord(raw["YMax"]))
            detections[key].append(box)

    candidates = []
    for image_id, relations in rel_by_image.items():
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            continue
        endpoints: list[dict[str, Any]] = []
        endpoint_id_map: dict[tuple[str, tuple[float, float, float, float]], str] = {}
        joined = True
        for rel in relations:
            for side in ("subject", "object"):
                label_id = rel[f"{side}_label_id"]
                raw_box = rel[f"{side}_box"]
                key = (image_id, label_id)
                matches = [box for box in detections.get(key, [])
                           if max(abs(box[i] - raw_box[i]) for i in range(4)) <= 1.1e-6]
                unique_matches = {tuple(box) for box in matches}
                if len(unique_matches) != 1:
                    joined = False
                    break
                detection_box = next(iter(unique_matches))
                endpoint_key = (label_id, tuple(round(x, 6) for x in raw_box))
                if endpoint_key not in endpoint_id_map:
                    object_id = f"oi:{image_id}:obj{len(endpoints)}"
                    endpoint_id_map[endpoint_key] = object_id
                    endpoints.append({"id": object_id, "label": class_map[label_id], "label_id": label_id,
                                      "normalized_box": list(detection_box)})
            if not joined:
                break
        if not joined:
            continue
        converted_relations = []
        for rel in relations:
            skey = (rel["subject_label_id"], tuple(round(x, 6) for x in rel["subject_box"]))
            okey = (rel["object_label_id"], tuple(round(x, 6) for x in rel["object_box"]))
            converted_relations.append({"subject": endpoint_id_map[skey], "predicate": rel["predicate"],
                                        "object": endpoint_id_map[okey]})
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        for obj in endpoints:
            x1, y1, x2, y2 = obj.pop("normalized_box")
            obj["box"] = _box_xyxy([x1 * width, y1 * height, x2 * width, y2 * height])
        candidate = {"image_id": image_id, "source_row": None,
                     "predicate_ids": sorted({rel["predicate"] for rel in converted_relations}),
                     "image_path": image_path, "width": width, "height": height,
                     "objects": endpoints, "relations": converted_relations}
        candidates.append(candidate)
    if len(candidates) < PANEL_SIZE:
        raise ValueError(f"Open Images only has {len(candidates)} images with unambiguous detection joins")
    selected = select_panel(candidates, seed, "Open-Images-V7-validation")
    rows = []
    for candidate in selected:
        image_path = candidate["image_path"]
        row = {"image_id": candidate["image_id"], "source_row": None,
               "image": {"path": str(image_path), "sha256": sha256_file(image_path),
                         "width": candidate["width"], "height": candidate["height"]},
               "ground_truth": {"objects": candidate["objects"], "relations": candidate["relations"]},
               "source_split": "validation"}
        _validate_graph(row, width=candidate["width"], height=candidate["height"])
        rows.append(row)
    provenance = {
        "relationships": file_record(rel_path, "Open Images V7 relationships validation"),
        "detections": file_record(det_path, "Open Images V7 detections validation"),
        "class_descriptions": file_record(class_path, "Open Images V7 class descriptions"),
        "terms_reference": {
            "url": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
            "annotations": "CC BY 4.0",
            "images": "listed as CC BY 2.0; verify rights for each source image",
        },
        "image_directory": str(image_dir),
        "split": "validation",
        "relationship_attribute_excluded": "is",
        "endpoint_join": "image_id,label_id,normalized_box; max edge delta 1.1e-6 for CSV decimal serialization",
    }
    object_vocab = sorted(set(class_map.values()))
    manifest = _manifest("openimages_v7", "Open Images V7 visual relationships", "validation", seed,
                         provenance, rows, object_vocab, sorted(predicate_vocab))
    return manifest_write(output_dir, "openimages_v7", manifest)


def prepare_psg(root: Path, output_dir: Path, seed: int) -> Path:
    import pyarrow.parquet as pq
    from datasets import Features

    data_dir = root / "psg" / "huggingface_dataset" / "data"
    parquet_files = sorted(data_dir.glob("validation-*.parquet"))
    if len(parquet_files) != 1:
        raise ValueError(f"PSG mirror expected one validation parquet, found {parquet_files}")
    parquet_path = parquet_files[0]
    revision_metadata = root / "psg" / "huggingface_dataset" / ".cache" / "huggingface" / "download" / ".gitattributes.metadata"
    if not revision_metadata.is_file():
        raise FileNotFoundError(f"PSG mirror revision metadata is missing: {revision_metadata}")
    revision_lines = revision_metadata.read_text(encoding="ascii").splitlines()
    if not revision_lines or len(revision_lines[0]) != 40:
        raise ValueError(f"PSG mirror revision metadata has no 40-character commit: {revision_metadata}")
    mirror_revision = f"wliafe/PSG@{revision_lines[0]}"
    metadata = pq.read_metadata(parquet_path).metadata or {}
    hf_metadata = json.loads(metadata[b"huggingface"])
    features = Features.from_dict(hf_metadata["info"]["features"])
    object_names = features["labels"].feature.names
    predicate_names = features["relations"]["predicate"].feature.names
    table_file = pq.ParquetFile(parquet_path)
    candidates = []
    source_row = 0
    for batch in table_file.iter_batches(batch_size=64):
        for record in batch.to_pylist():
            row_index = source_row
            source_row += 1
            image_id = str(record["id"])
            width, height = int(record["width"]), int(record["height"])
            image_struct = record["image"]
            image_bytes = image_struct.get("bytes") if image_struct else None
            if not image_bytes:
                raise ValueError(f"PSG row {row_index} ({image_id}): embedded image bytes missing")
            with Image.open(__import__("io").BytesIO(image_bytes)) as image:
                image.load()
                if image.size != (width, height):
                    raise ValueError(f"PSG row {row_index} ({image_id}): embedded image dimensions disagree with metadata")
            boxes = record["boxes"] or []
            labels = record["labels"] or []
            if len(boxes) != len(labels):
                raise ValueError(f"PSG row {row_index} ({image_id}): boxes/labels length mismatch")
            objects = []
            for index, (box, label_id) in enumerate(zip(boxes, labels)):
                label_id = int(label_id)
                if label_id <= 0 or label_id >= len(object_names):
                    raise ValueError(f"PSG row {row_index}: invalid/background object label {label_id}")
                objects.append({"id": f"psg:{image_id}:node{index}", "label": object_names[label_id],
                                "box": _box_xyxy(box), "source_label_id": label_id})
            rel_data = record["relations"] or {}
            subs = [int(x) for x in (rel_data.get("subject_index") or [])]
            objs = [int(x) for x in (rel_data.get("object_index") or [])]
            preds = [int(x) for x in (rel_data.get("predicate") or [])]
            if not (len(subs) == len(objs) == len(preds)):
                raise ValueError(f"PSG row {row_index} ({image_id}): relation arrays differ in length")
            relations = []
            for subject, object_, predicate in zip(subs, objs, preds):
                if not 0 <= subject < len(objects) or not 0 <= object_ < len(objects):
                    raise ValueError(f"PSG row {row_index}: relation endpoint outside node list")
                if predicate <= 0 or predicate >= len(predicate_names):
                    raise ValueError(f"PSG row {row_index}: invalid/background predicate {predicate}")
                relations.append({"subject": objects[subject]["id"], "predicate": predicate_names[predicate],
                                  "object": objects[object_]["id"], "source_predicate_id": predicate})
            if not relations:
                continue
            candidates.append({"image_id": image_id, "source_row": row_index,
                               "predicate_ids": sorted({rel["source_predicate_id"] for rel in relations}),
                               "image_bytes_sha256": sha256_bytes(image_bytes),
                               "width": width, "height": height, "objects": objects, "relations": relations})
    if source_row != table_file.metadata.num_rows:
        raise ValueError(f"PSG parquet iterator yielded {source_row} rows, metadata declares {table_file.metadata.num_rows}")
    if len({str(row["image_id"]) for row in candidates}) != len(candidates):
        raise ValueError("PSG validation contains duplicate sample IDs")
    selected = select_panel(candidates, seed, "PSG-validation")
    rows = []
    for candidate in selected:
        row = {"image_id": candidate["image_id"], "source_row": candidate["source_row"],
               "image": {"parquet_path": str(parquet_path), "row_index": candidate["source_row"],
                         "sha256": candidate["image_bytes_sha256"], "width": candidate["width"], "height": candidate["height"]},
               "ground_truth": {"objects": candidate["objects"], "relations": candidate["relations"]},
               "source_split": "validation", "upstream_split": "test"}
        _validate_graph(row, width=candidate["width"], height=candidate["height"])
        rows.append(row)
    provenance = {
        "parquet": file_record(parquet_path, mirror_revision),
        "mirror_revision_metadata": file_record(revision_metadata, mirror_revision),
        "readme": file_record(root / "psg" / "huggingface_dataset" / "README.md"),
        "terms_reference": {
            "url": "https://psgdataset.org/term.html",
            "annotations": "CC BY 4.0",
            "images": "sourced from Visual Genome and COCO; original-source terms apply",
            "note": "Source images remain embedded in the local Parquet source; this benchmark stores only the Parquet path and row index.",
        },
        "split": "validation",
        "upstream_split_mapping": "validation is the complete official PSG test split per local mirror README",
        "parquet_rows": table_file.metadata.num_rows,
    }
    object_vocab = list(object_names[1:])
    predicate_vocab = list(predicate_names[1:])
    manifest = _manifest("psg", "PSG box-adapted SGG", "validation (upstream test)", seed,
                         provenance, rows, object_vocab, predicate_vocab)
    return manifest_write(output_dir, "psg", manifest)


def prepare(dataset: str, data_root: Path, output_dir: Path, seed: int, audit: bool) -> list[Path]:
    if seed != 3407:
        # The planned panel is versioned at this fixed sampling seed.
        raise ValueError("the planned SGG panel requires seed 3407")
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "vg150": data_root / "visual_genome" if (data_root / "visual_genome").is_dir() else data_root,
        "openimages_v7": data_root,
        "psg": data_root,
    }
    actions = {"vg150": prepare_vg, "openimages_v7": prepare_openimages, "psg": prepare_psg}
    chosen = list(actions) if dataset == "all" else [dataset]
    results = []
    for name in chosen:
        if name == "vg150":
            results.append(actions[name](roots[name], output_dir, seed, audit))
        else:
            results.append(actions[name](roots[name], output_dir, seed))
    return results
