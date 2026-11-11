"""Versioned closed-vocabulary image SGG prompts."""

from __future__ import annotations

import hashlib
import json

PROMPT_VERSION = "image-sgg-prompt-v2"


def prompt_for(object_labels: list[str], predicate_labels: list[str], dataset_label: str) -> str:
    objects = json.dumps(object_labels, ensure_ascii=False)
    predicates = json.dumps(predicate_labels, ensure_ascii=False)
    example = json.dumps({
        "objects": [
            {"id": "o1", "label": object_labels[0], "box": [0.1, 0.1, 0.4, 0.4]},
            {"id": "o2", "label": object_labels[1], "box": [0.5, 0.5, 0.9, 0.9]},
        ],
        "relations": [{"subject": "o1", "predicate": predicate_labels[0], "object": "o2"}],
    }, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{PROMPT_VERSION}. Inspect the image and generate a scene graph for {dataset_label}.\n"
        "Detect visible objects and ground each object with a normalized [x1,y1,x2,y2] box. "
        "Each box coordinate must be a fraction from 0 to 1 relative to the full image, never "
        "a pixel coordinate. Give every object a unique string ID such as o1. Each relation's "
        "subject and object fields must copy IDs from the objects array, never object labels. "
        "Produce directed relations from subject to object, ordered from most confident to least "
        "confident. Use only the allowed labels below.\n"
        "The top-level JSON value must be one object with exactly objects and relations arrays. "
        "Each object must have exactly id, label, and box. Each relation must have exactly "
        "subject, predicate, and object. Return raw JSON only: do not add markdown fences, "
        "comments, or explanation. The first response character must be { and the last must be }. "
        "The following schema example uses allowed labels but does not describe this image; "
        "include an object only when it is visible in the image. Example: "
        f"{example}. "
        "Use at most 50 objects and 100 relations. Do not include an object outside the image.\n"
        f"Allowed object labels: {objects}\nAllowed predicate labels: {predicates}"
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
