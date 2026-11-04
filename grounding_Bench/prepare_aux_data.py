#!/usr/bin/env python3
"""Prepare local VQAv2 panel and cached TextVQA manifests for the specialists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_HF_HOME = Path(os.environ.get(
    "HF_HOME", "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"
))
TEXTVQA_ARROW = (
    DEFAULT_HF_HOME / "datasets/facebook___textvqa/textvqa/0.5.1/"
    "9b89037cc122c3b495b155a1bce4170851829843454e88f236bb8715d977c027/"
    "textvqa-validation.arrow"
)
SEED = 3407


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_manifest(output_dir: Path, task: str, dataset: str, split: str,
                  terms: str, source_hash: str, rows: list[dict[str, Any]], seed: int) -> None:
    payload = {"task": task, "dataset": dataset, "split": split, "sample_count": len(rows),
               "seed": seed, "license_or_access_terms": terms, "source_sha256": source_hash,
               "asset_policy": "sampled image paths and annotations remain local", "examples": rows}
    (output_dir / f"{task}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"{task}: {len(rows)} examples -> {output_dir / f'{task}.json'}")


def prepare_vqa_panel(panel_dir: Path, output_dir: Path, seed: int) -> None:
    for task in ("vqa_count", "vqa_truth", "vqa_description"):
        path = panel_dir / f"{task}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        examples = []
        for index, row in enumerate(rows):
            image = Path(row["image_path"])
            if not image.is_absolute():
                image = (ROOT / image).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"VQAv2 panel image is missing: {image}")
            examples.append({
                "id": str(row.get("question_id", f"{task}-{index}")), "image": str(image),
                "question": str(row["question"]), "answers": [str(a) for a in row["answers"]],
                "canonical_answer": str(row.get("canonical_answer", "")),
                "image_id": row.get("image_id"), "source_question_id": row.get("question_id"),
            })
        save_manifest(output_dir, task, "VQAv2", "validation",
                      "VQA Consortium annotations CC BY 4.0; images are from MS COCO and retain their source terms",
                      sha256(path), examples, seed)


def prepare_textvqa(arrow_path: Path, output_dir: Path, image_dir: Path, count: int, seed: int) -> None:
    if not arrow_path.is_file():
        raise FileNotFoundError(f"cached TextVQA validation Arrow file not found: {arrow_path}")
    from datasets import Dataset

    dataset = Dataset.from_file(str(arrow_path))
    if count < 1 or count > len(dataset):
        raise ValueError(f"TextVQA sample count must be between 1 and {len(dataset)}")
    selected_indices = sorted(random.Random(seed).sample(range(len(dataset)), count))
    image_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for sample_index, source_index in enumerate(selected_indices):
        row = dataset[source_index]
        image_path = image_dir / f"textvqa_{int(row['question_id'])}.jpg"
        row["image"].convert("RGB").save(image_path, format="JPEG", quality=95)
        examples.append({
            "id": str(row["question_id"]), "question_id": int(row["question_id"]),
            "image_id": str(row["image_id"]), "image": str(image_path.resolve()),
            "question": str(row["question"]), "answers": [str(a) for a in row["answers"]],
            "sample_index": sample_index, "source_row": source_index,
        })
    save_manifest(output_dir, "textvqa", "TextVQA", "validation",
                  "TextVQA dataset card declares CC BY 4.0; its OpenImages-derived source images retain their source terms",
                  sha256(arrow_path), examples, seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-panel-dir", type=Path, default=ROOT / "data" / "vqa_panel")
    parser.add_argument("--textvqa-arrow", type=Path, default=TEXTVQA_ARROW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "manifests")
    parser.add_argument("--image-dir", type=Path, default=HERE / "data" / "textvqa" / "images")
    parser.add_argument("--textvqa-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_vqa_panel(args.vqa_panel_dir, args.output_dir, args.seed)
    prepare_textvqa(args.textvqa_arrow, args.output_dir, args.image_dir, args.textvqa_samples, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
