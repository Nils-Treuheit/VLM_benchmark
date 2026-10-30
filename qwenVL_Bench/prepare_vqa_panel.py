#!/usr/bin/env python3
"""Prepare a fixed, source-labeled VQA skill panel and its image files."""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "vqa_panel"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
SAMPLES_PER_TYPE = 25
SEED = 3407
VQA_URLS = {
    "questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "VLM-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)


def save_image(image: Any, name: str) -> str:
    if isinstance(image, dict):
        if image.get("bytes"):
            from io import BytesIO
            image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path"):
            image = Image.open(image["path"])
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported dataset image value: {type(image).__name__}")
    destination = DATA / "images" / f"{name}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, format="JPEG", quality=95)
    return str(destination.relative_to(ROOT))


def sample_vqav2() -> dict[str, list[dict[str, Any]]]:
    cache = DATA / "source" / "vqav2"
    paths = {}
    for kind, url in VQA_URLS.items():
        archive = cache / f"{kind}.zip"
        download(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            json_name = next(name for name in bundle.namelist() if name.endswith(".json"))
            paths[kind] = json.loads(bundle.read(json_name))

    questions = {row["question_id"]: row for row in paths["questions"]["questions"]}
    buckets: dict[str, list[dict[str, Any]]] = {"truth": [], "count": [], "description": []}
    for ann in paths["annotations"]["annotations"]:
        question = questions.get(ann["question_id"])
        if not question:
            continue
        answer_type = ann.get("answer_type", "").lower()
        q = question.get("question", "").strip()
        if answer_type == "yes/no":
            bucket = "truth"
        elif answer_type == "number" and re.match(r"(?i)^\s*how many\b", q):
            bucket = "count"
        elif answer_type == "other" and re.match(r"(?i)^\s*(what|which|where|who|what color|what is)\b", q):
            bucket = "description"
        else:
            continue
        buckets[bucket].append({
            "question": q,
            "answers": [item["answer"] for item in ann.get("answers", [])],
            "canonical_answer": ann.get("multiple_choice_answer", ""),
            "image_id": int(ann["image_id"]),
            "question_id": int(ann["question_id"]),
            "source": "VQAv2 validation",
        })

    output = {}
    for bucket, candidates in buckets.items():
        rng = random.Random(SEED + sum(map(ord, bucket)))
        selected = rng.sample(candidates, min(SAMPLES_PER_TYPE, len(candidates)))
        for row in selected:
            image_id = row["image_id"]
            filename = f"COCO_val2014_{image_id:012d}.jpg"
            image_path = DATA / "images" / filename
            if not image_path.is_file():
                download(f"http://images.cocodataset.org/val2014/{filename}", image_path)
            row["image_path"] = str(image_path.relative_to(ROOT))
            row["category"] = f"vqa_{bucket}"
        output[f"vqa_{bucket}"] = selected
    return output


def image_dataset_rows(dataset_id: str, config: str | None, split: str):
    from datasets import load_dataset
    kwargs = {"split": split}
    if config:
        return load_dataset(dataset_id, config, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def sample_hub_dataset(dataset_id: str, config: str | None, split: str, *, kind: str):
    selected = []
    rng = random.Random(SEED + (1 if kind == "affordance" else 2))
    seen = 0
    for source_index, row in enumerate(image_dataset_rows(dataset_id, config, split)):
        question = row.get("question") or row.get("negated_question") or ""
        answer = row.get("answer") or row.get("negated_answer")
        if kind == "affordance":
            qtype = str(row.get("question_type", "")).lower()
            if not ("affordance" in qtype or "function" in qtype or qtype.startswith("repurposing_")):
                continue
            answers = [answer] if answer else []
            task = "vqa_affordance"
        else:
            a, b = str(row.get("A", "")), str(row.get("B", ""))
            correct = str(row.get("negated_answer", "")).strip().upper()
            if correct not in {"A", "B"} or not a or not b:
                continue
            answers = correct
            task = "vqa_negation"
            question = str(row.get("negated_question", ""))
        image = row.get("image")
        if not question or not image:
            continue
        options = [a, b] if kind == "negation" else []
        candidate = {
            "question": str(question), "answers": answers, "options": options,
            "image": image, "image_name": f"{task}_{row.get('id', source_index)}",
            "category": task, "source": dataset_id, "source_index": source_index,
            "question_type": row.get("question_type"),
        }
        seen += 1
        if len(selected) < SAMPLES_PER_TYPE:
            selected.append(candidate)
        else:
            pick = rng.randrange(seen)
            if pick < SAMPLES_PER_TYPE:
                selected[pick] = candidate
    if len(selected) < SAMPLES_PER_TYPE:
        raise RuntimeError(f"Only prepared {len(selected)} examples for {task} from {dataset_id}")
    for item in selected:
        item["image_path"] = save_image(item.pop("image"), item.pop("image_name"))
    return selected


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tasks = sample_vqav2()
    tasks["vqa_affordance"] = sample_hub_dataset(
        "RyanWW/PercepTaxBench", "real", "test", kind="affordance"
    )
    tasks["vqa_negation"] = sample_hub_dataset(
        "yuhuizhang/NegVQA", None, "dev", kind="negation"
    )
    for task, rows in tasks.items():
        destination = DATA / f"{task}.json"
        destination.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"{task}: {len(rows)} examples -> {destination}")
    manifest = {
        "seed": SEED,
        "samples_per_type": SAMPLES_PER_TYPE,
        "categories": {task: len(rows) for task, rows in tasks.items()},
        "sources": {
            "vqa_truth": "VQAv2 validation; answer_type=yes/no",
            "vqa_count": "VQAv2 validation; answer_type=number and question begins How many",
            "vqa_description": "VQAv2 validation; answer_type=other and direct visual question prefix",
            "vqa_affordance": "PercepTax real test; affordance/function and taxonomy reasoning types",
            "vqa_negation": "NegVQA dev; two-choice negated questions",
        },
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
