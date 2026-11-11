"""Strict scene-graph parsing and deterministic box-grounded metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


EVALUATOR_VERSION = "image-sgg-evaluator-v1"


def evaluator_source_hash() -> str:
    """Hash the scorer implementation so result provenance changes with its code."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class GraphParseError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_graph(text: str, object_vocab: set[str], predicate_vocab: set[str]) -> dict[str, Any]:
    """Parse without repair; malformed or unsupported output remains an error."""
    try:
        graph = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GraphParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(graph, dict) or set(graph) != {"objects", "relations"}:
        raise GraphParseError("top level must contain exactly objects and relations")
    objects, relations = graph["objects"], graph["relations"]
    if not isinstance(objects, list) or len(objects) > 50:
        raise GraphParseError("objects must be a list with at most 50 entries")
    if not isinstance(relations, list) or len(relations) > 100:
        raise GraphParseError("relations must be a list with at most 100 entries")
    ids: set[str] = set()
    for item in objects:
        if not isinstance(item, dict) or set(item) != {"id", "label", "box"}:
            raise GraphParseError("each object must contain exactly id, label and box")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
            raise GraphParseError("object IDs must be non-empty and unique strings")
        ids.add(item["id"])
        if not isinstance(item["label"], str) or item["label"] not in object_vocab:
            raise GraphParseError(f"unknown object label: {item['label']!r}")
        box = item["box"]
        if not isinstance(box, list) or len(box) != 4:
            raise GraphParseError("object boxes must contain four normalized xyxy numbers")
        if any(isinstance(x, bool) for x in box):
            raise GraphParseError("box coordinates must be numeric")
        try:
            coords = [float(x) for x in box]
        except (TypeError, ValueError) as exc:
            raise GraphParseError("box coordinates must be numeric") from exc
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
            raise GraphParseError("box coordinates must be finite and in [0,1]")
        if not coords[0] < coords[2] or not coords[1] < coords[3]:
            raise GraphParseError("boxes must have positive area")
        item["box"] = coords
    for relation in relations:
        if not isinstance(relation, dict) or set(relation) != {"subject", "predicate", "object"}:
            raise GraphParseError("each relation must contain subject, predicate and object")
        if (not isinstance(relation["subject"], str) or not isinstance(relation["object"], str)
                or relation["subject"] not in ids or relation["object"] not in ids):
            raise GraphParseError("relation endpoints must reference listed object IDs")
        if not isinstance(relation["predicate"], str) or relation["predicate"] not in predicate_vocab:
            raise GraphParseError(f"unknown predicate: {relation['predicate']!r}")
    return graph


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_objects(predicted: list[dict[str, Any]], truth: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, int]:
    """Class-aware maximum-IoU assignment with stable row/column tie weights."""
    if not predicted or not truth:
        return {}
    weights = np.full((len(predicted), len(truth)), -1.0, dtype=np.float64)
    for i, pred in enumerate(predicted):
        for j, gt in enumerate(truth):
            if pred["label"] == gt["label"]:
                iou = box_iou(pred["box"], gt["box"])
                if iou >= threshold:
                    # Stable tie preference is lower prediction then lower GT index.
                    weights[i, j] = iou - (i * len(truth) + j) * 1e-12
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return {predicted[i]["id"]: int(j) for i, j in zip(rows, cols) if weights[i, j] >= threshold - 1e-9}


def evaluate_graph(prediction: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    pred_objects = prediction["objects"]
    gt_objects = truth["objects"]
    mapping = match_objects(pred_objects, gt_objects)
    gt_rel_counts = Counter((r["subject"], r["predicate"], r["object"]) for r in truth["relations"])
    per_pred_total = Counter(r["predicate"] for r in truth["relations"])
    per_cutoff: dict[int, dict[str, Any]] = {}
    pred_rel = prediction["relations"]
    for cutoff in (20, 50, 100):
        hits: Counter[tuple[int, str, int]] = Counter()
        per_pred_hits = Counter()
        for relation in pred_rel[:cutoff]:
            if relation["subject"] not in mapping or relation["object"] not in mapping:
                continue
            triplet = (mapping[relation["subject"]], relation["predicate"], mapping[relation["object"]])
            gt_key = (gt_objects[triplet[0]]["id"], triplet[1], gt_objects[triplet[2]]["id"])
            if hits[gt_key] < gt_rel_counts[gt_key]:
                hits[gt_key] += 1
                per_pred_hits[triplet[1]] += 1
        total = sum(gt_rel_counts.values())
        recall = sum(hits.values()) / total if total else 0.0
        supported = [p for p, count in per_pred_total.items() if count]
        mean_recall = (
            sum(per_pred_hits[p] / per_pred_total[p] for p in supported) / len(supported)
            if supported else 0.0
        )
        per_cutoff[cutoff] = {
            "recall": recall, "mean_recall": mean_recall,
            "predicate_recall": {p: per_pred_hits[p] / n for p, n in sorted(per_pred_total.items())},
            "predicate_support": dict(sorted(per_pred_total.items())),
            "triplet_precision": sum(hits.values()) / min(cutoff, len(pred_rel)) if pred_rel else 0.0,
            "matched_triplets": sum(hits.values()),
            "ground_truth_triplets": total,
            "predicted_triplets": min(cutoff, len(pred_rel)),
        }
    # Precision/recall/F1 use the first 50 relations; every unmatched or duplicate
    # prediction remains in the precision denominator.
    top50 = pred_rel[:50]
    matched50 = 0
    used50: Counter[tuple[str, str, str]] = Counter()
    for relation in top50:
        if relation["subject"] not in mapping or relation["object"] not in mapping:
            continue
        s = gt_objects[mapping[relation["subject"]]]["id"]
        o = gt_objects[mapping[relation["object"]]]["id"]
        key = (s, relation["predicate"], o)
        if used50[key] < gt_rel_counts[key]:
            used50[key] += 1
            matched50 += 1
    gt_total = len(truth["relations"])
    precision = matched50 / len(top50) if top50 else 0.0
    recall50 = matched50 / gt_total if gt_total else 0.0
    f1 = (2 * precision * recall50 / (precision + recall50)) if precision + recall50 else 0.0
    top_objects = pred_objects[:50]
    object_recall = len({mapping[x["id"]] for x in top_objects if x["id"] in mapping}) / len(gt_objects) if gt_objects else 0.0
    return {
        "r_at_k": {str(k): v["recall"] for k, v in per_cutoff.items()},
        "mr_at_k": {str(k): v["mean_recall"] for k, v in per_cutoff.items()},
        "predicate_recall": {str(k): v["predicate_recall"] for k, v in per_cutoff.items()},
        "predicate_support": dict(sorted(per_pred_total.items())),
        "triplet_precision_at_50": precision,
        "triplet_recall_at_50": recall50,
        "triplet_f1_at_50": f1,
        "object_recall_at_50": object_recall,
        "matched_objects": len(mapping),
        "matched_objects_at_50": len({mapping[x["id"]] for x in top_objects if x["id"] in mapping}),
        "ground_truth_objects": len(gt_objects),
        "ground_truth_relations": gt_total,
        "matched_triplets_at_50": matched50,
        "predicted_triplets_at_50": len(top50),
    }


def score_item(raw_text: str | None, gt: dict[str, Any], object_vocab: set[str], predicate_vocab: set[str], error: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"json_valid": False, "error": error}
    if raw_text is None or error:
        return record
    try:
        prediction = parse_graph(raw_text, object_vocab, predicate_vocab)
    except GraphParseError as exc:
        record["error"] = str(exc)
        return record
    record["json_valid"] = True
    record["prediction"] = prediction
    record["score"] = evaluate_graph(prediction, gt)
    return record


def aggregate_items(items: list[dict[str, Any]], truth_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset aggregate. Failed items remain in the completion denominator."""
    n = len(truth_rows)
    by_id = {str(item.get("image_id")): item for item in items}
    if len(by_id) != len(items):
        raise ValueError("run contains duplicate item image IDs")
    truth_ids = [str(row.get("image_id", row.get("id"))) for row in truth_rows]
    if len(set(truth_ids)) != len(truth_ids):
        raise ValueError("panel contains duplicate item image IDs")
    if set(by_id) != set(truth_ids):
        missing = sorted(set(truth_ids) - set(by_id))
        extra = sorted(set(by_id) - set(truth_ids))
        raise ValueError(f"run item IDs do not match panel; missing={missing[:5]}, extra={extra[:5]}")
    aligned = [(by_id[image_id], row) for image_id, row in zip(truth_ids, truth_rows)]
    successful = [item for item, _ in aligned if item.get("score")]
    output: dict[str, Any] = {
        "requested_n": n,
        "completed_n": sum(item.get("raw_response") is not None for item, _ in aligned),
        "attempted_n": len(aligned),
        "valid_json_n": sum(bool(item.get("json_valid")) for item, _ in aligned),
        "failure_n": sum(not bool(item.get("json_valid")) for item, _ in aligned),
        "metrics": {},
    }
    metrics: dict[str, Any] = {"r_at_k": {}, "mr_at_k": {}, "predicate_support": {}}
    predicate_support = Counter()
    for item, row in aligned:
        predicate_support.update(Counter(rel["predicate"] for rel in row["ground_truth"]["relations"]))
    metrics["predicate_support"] = dict(sorted(predicate_support.items()))
    for k in ("20", "50", "100"):
        matched = sum(x["score"]["r_at_k"][k] * x["score"]["ground_truth_relations"]
                      for x in successful)
        total = sum(len(row["ground_truth"]["relations"]) for _, row in aligned)
        metrics["r_at_k"][k] = matched / total if total else 0.0
        hits = Counter()
        supports = Counter()
        for item, row in aligned:
            row_support = Counter(rel["predicate"] for rel in row["ground_truth"]["relations"])
            supports.update(row_support)
            if item.get("score"):
                score = item["score"]
                for pred, recall_value in score.get("predicate_recall", {}).get(k, {}).items():
                    hits[pred] += recall_value * row_support.get(pred, 0)
        supported = [p for p, count in supports.items() if count]
        metrics["mr_at_k"][k] = (
            sum(hits[p] / supports[p] for p in supported) / len(supported) if supported else 0.0
        )
    matched50 = sum(x["score"].get("matched_triplets_at_50", 0) for x in successful)
    predicted50 = sum(x["score"].get("predicted_triplets_at_50", 0) for x in successful)
    total_rel = sum(len(row["ground_truth"]["relations"]) for _, row in aligned)
    precision = matched50 / predicted50 if predicted50 else 0.0
    recall = matched50 / total_rel if total_rel else 0.0
    metrics["triplet_precision_at_50"] = precision
    metrics["triplet_recall_at_50"] = recall
    metrics["triplet_f1_at_50"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    obj_matched = sum(x["score"].get("matched_objects_at_50", 0) for x in successful)
    obj_total = sum(len(row["ground_truth"]["objects"]) for _, row in aligned)
    metrics["object_recall_at_50"] = obj_matched / obj_total if obj_total else 0.0
    output["metrics"] = metrics
    return output


def compatible_provenance(items: list[dict[str, Any]]) -> bool:
    keys = ("manifest_hash", "prompt_hash", "evaluator_hash", "vocabulary_hash", "source_hashes")
    signatures = {tuple(canonical_json(item.get(key)) for key in keys) for item in items}
    return len(signatures) <= 1
