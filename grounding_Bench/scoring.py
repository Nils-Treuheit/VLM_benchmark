#!/usr/bin/env python3
"""Model-specific grounding parsers and task scorers.

The raw generation text is deliberately retained alongside canonical geometry.
LocateAnything and GroundAnything use different spatial token vocabularies.
"""

from __future__ import annotations

import math
import re
from typing import Any


def _valid_box(box: list[float] | tuple[float, ...]) -> bool:
    return (
        len(box) == 4
        and all(math.isfinite(float(v)) for v in box)
        and 0 <= box[0] <= box[2] <= 1
        and 0 <= box[1] <= box[3] <= 1
        and box[0] < box[2]
        and box[1] < box[3]
    )


def _locate_numbers(raw: str) -> list[int]:
    """Read LocateAnything's integer coordinate tokens without touching labels."""
    return [int(value) for value in re.findall(r"<(\d{1,4})>", raw)]


def parse_locateanything(raw: str) -> dict[str, Any]:
    """Parse `<ref>label</ref><box><x1><y1><x2><y2></box>` and points."""
    records: list[dict[str, Any]] = []
    pattern = re.compile(r"<ref>(.*?)</ref>\s*<box>(.*?)</box>", re.S)
    for match in pattern.finditer(raw):
        label = re.sub(r"\s+", " ", match.group(1)).strip() or None
        body = match.group(2).strip()
        if body.lower() == "none":
            records.append({"label": label, "kind": "none", "box_xyxy_norm": None,
                            "point_xy_norm": None, "valid": True})
            continue
        values = _locate_numbers(body)
        if len(values) >= 4:
            box = [v / 1000 for v in values[:4]]
            valid = _valid_box(box)
            records.append({"label": label, "kind": "box", "box_xyxy_norm": box if valid else None,
                            "point_xy_norm": None, "valid": valid})
    # Unlabelled boxes and GUI points are both emitted inside <box> tags.
    if not records:
        for match in re.finditer(r"<box>(.*?)</box>", raw, re.S):
            body = match.group(1).strip()
            if body.lower() == "none":
                records.append({"label": None, "kind": "none", "box_xyxy_norm": None,
                                "point_xy_norm": None, "valid": True})
                continue
            values = _locate_numbers(body)
            if len(values) == 2:
                point = [v / 1000 for v in values]
                valid = all(math.isfinite(v) and 0 <= v <= 1 for v in point)
                records.append({"label": None, "kind": "point", "box_xyxy_norm": None,
                                "point_xy_norm": point if valid else None, "valid": valid})
            elif len(values) >= 4:
                box = [v / 1000 for v in values[:4]]
                valid = _valid_box(box)
                records.append({"label": None, "kind": "box", "box_xyxy_norm": box if valid else None,
                                "point_xy_norm": None, "valid": valid})
    return {"records": records, "valid": bool(records) and all(r["valid"] for r in records),
            "empty": not bool(records)}


def parse_groundanything(raw: str) -> dict[str, Any]:
    """Parse GroundAnything's GAM box wrappers and coordinate tokens (0..999)."""
    records: list[dict[str, Any]] = []
    wrapper = re.compile(
        r"<\|object_ref_start\|>(?P<label>.*?)<\|object_ref_end\|>"
        r"<\|box_start\|>(?P<body>.*?)<\|box_end\|>", re.S,
    )
    wrappers = list(wrapper.finditer(raw))
    malformed = False
    for match in wrappers:
        label = re.sub(r"\s+", " ", match.group("label")).strip() or None
        chunks = match.group("body").split(",")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                malformed = True
                continue
            if chunk.lower() == "none":
                records.append({"label": label, "kind": "none", "box_xyxy_norm": None,
                                "point_xy_norm": None, "valid": True})
                continue
            values = [int(v) for v in re.findall(r"<(\d{1,4})>", chunk)]
            if len(values) == 4:
                box = [v / 999 for v in values]
                valid = _valid_box(box)
                records.append({"label": label, "kind": "box", "box_xyxy_norm": box if valid else None,
                                "point_xy_norm": None, "valid": valid})
            elif len(values) == 2:
                point = [v / 999 for v in values]
                valid = all(math.isfinite(v) and 0 <= v <= 1 for v in point)
                records.append({"label": label, "kind": "point", "box_xyxy_norm": None,
                                "point_xy_norm": point if valid else None, "valid": valid})
            else:
                malformed = True
    if not wrappers and re.search(r"<\|object_ref_start\|>|<\|box_start\|>|<\|box_end\|>", raw):
        malformed = True
    return {"records": records, "valid": bool(records) and all(r["valid"] for r in records) and not malformed,
            "empty": not bool(records), "malformed": malformed}


def iou_xyxy(a: list[float], b: list[float]) -> float:
    if not _valid_box(a) or not _valid_box(b):
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def score_referring(prediction: dict[str, Any], target: list[float]) -> dict[str, Any]:
    boxes = [r["box_xyxy_norm"] for r in prediction["records"] if r["kind"] == "box" and r["valid"]]
    best = max((iou_xyxy(box, target) for box in boxes), default=0.0)
    return {"iou": best, "acc_iou_05": float(best >= 0.5)}


def score_gui(prediction: dict[str, Any], target_box: list[float],
              target_point: list[float] | None = None) -> dict[str, Any]:
    boxes = [r["box_xyxy_norm"] for r in prediction["records"] if r["kind"] == "box" and r["valid"]]
    points = [r["point_xy_norm"] for r in prediction["records"] if r["kind"] == "point" and r["valid"]]
    iou = max((iou_xyxy(box, target_box) for box in boxes), default=0.0)
    reference = target_point or [(target_box[0] + target_box[2]) / 2,
                                 (target_box[1] + target_box[3]) / 2]
    # Point-in-target is the GUI task's standard hit criterion.
    hit = any(target_box[0] <= p[0] <= target_box[2] and target_box[1] <= p[1] <= target_box[3]
              for p in points)
    return {"point_in_target": float(hit), "acc_iou_05": float(iou >= 0.5),
            "best_box_iou": iou, "target_point": reference}


def match_boxes(predicted: list[list[float]], targets: list[list[float]], threshold: float = 0.5) -> dict[str, int]:
    """Greedy one-to-one matching; deterministic by descending IoU."""
    pairs = sorted(((iou_xyxy(p, t), pi, ti) for pi, p in enumerate(predicted)
                    for ti, t in enumerate(targets)), reverse=True)
    used_p: set[int] = set()
    used_t: set[int] = set()
    for value, pi, ti in pairs:
        if value < threshold:
            break
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
    return {"tp": len(used_p), "fp": len(predicted) - len(used_p), "fn": len(targets) - len(used_t)}


def f1_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
