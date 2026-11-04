#!/usr/bin/env python3
"""Model-specific grounding parsers and task scorers.

The raw generation text is deliberately retained alongside canonical geometry.
LocateAnything and GroundAnything use different spatial token vocabularies.
"""

from __future__ import annotations

import math
import re
import statistics
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
    has_geometry = any(record["kind"] in {"box", "point"} for record in records)
    return {"records": records, "valid": bool(records) and all(r["valid"] for r in records),
            "empty": not has_geometry}


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
    has_geometry = any(record["kind"] in {"box", "point"} for record in records)
    return {"records": records, "valid": bool(records) and all(r["valid"] for r in records) and not malformed,
            "empty": not has_geometry, "malformed": malformed}


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


def normalize_vqa_answer(answer: Any) -> str:
    """Apply the official VQAv2 punctuation, number, article, and contraction rules."""
    value = str(answer or "").lower().replace("\n", " ").replace("\t", " ").strip()
    punctuation = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
                   ">", "<", "@", "`", ",", "?", "!"]
    comma_between_digits = re.search(r"(\d),(\d)", value) is not None
    for char in punctuation:
        if f"{char} " in value or f" {char}" in value or comma_between_digits:
            value = value.replace(char, "")
        else:
            value = value.replace(char, " ")
    value = re.sub(r"(?<!\d)\.(?!\d)", "", value)
    number_words = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
        "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
        "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
        "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "id've": "i'd've",
        "i'dve": "i'd've", "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
        "it'dve": "it'd've", "itll": "it'll", "its": "it's", "lets": "let's", "maam": "ma'am",
        "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
        "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
        "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
        "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
        "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
        "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
        "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
        "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
        "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
        "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
        "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
        "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
        "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
        "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
        "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
        "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
        "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
        "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
        "youre": "you're", "youve": "you've",
    }
    articles = {"a", "an", "the"}
    words = []
    for word in value.split():
        word = number_words.get(word, word)
        if word not in articles:
            words.append(contractions.get(word, word))
    return " ".join(words)


def vqa_consensus_score(prediction: str, references: list[Any]) -> float:
    pred = normalize_vqa_answer(prediction)
    answers = [normalize_vqa_answer(answer.get("answer", "") if isinstance(answer, dict) else answer)
               for answer in references]
    if not pred or not answers:
        return 0.0
    return statistics.mean(min(sum(answer == pred for index, answer in enumerate(answers) if index != held_out) / 3.0, 1.0)
                          for held_out in range(len(answers)))


def edit_similarity(left: str, right: str) -> float:
    """Normalized Levenshtein similarity for OCR words."""
    left = re.sub(r"\s+", " ", left.strip().casefold())
    right = re.sub(r"\s+", " ", right.strip().casefold())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def score_ocr_words(records: list[dict[str, Any]], targets: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    predicted = [r for r in records if r.get("kind") == "box" and r.get("valid") and r.get("box_xyxy_norm")]
    pairs = sorted(((iou_xyxy(p["box_xyxy_norm"], t["box_xyxy_norm"]), pi, ti)
                    for pi, p in enumerate(predicted) for ti, t in enumerate(targets)), reverse=True)
    used_p: set[int] = set()
    used_t: set[int] = set()
    matched = []
    for overlap, pi, ti in pairs:
        if overlap < threshold:
            break
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            matched.append((pi, ti))
    exact = 0
    similarities = []
    for pi, ti in matched:
        predicted_text = str(predicted[pi].get("label") or "")
        target_text = str(targets[ti].get("text") or "")
        exact += int(predicted_text.strip().casefold() == target_text.strip().casefold())
        similarities.append(edit_similarity(predicted_text, target_text))
    return {
        "text_detection_tp": len(used_p), "text_detection_fp": len(predicted) - len(used_p),
        "text_detection_fn": len(targets) - len(used_t), "matched_text_regions": len(used_p),
        "exact_text_matches": exact,
        "exact_text_accuracy": exact / len(used_p) if used_p else None,
        "normalized_edit_similarity": statistics.mean(similarities) if similarities else None,
        "text_target_count": len(targets),
    }
