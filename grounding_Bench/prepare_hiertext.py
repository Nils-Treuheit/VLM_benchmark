#!/usr/bin/env python3
"""Prepare a local HierText validation OCR manifest without downloading data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from import_datasets import import_hiertext, read_lines, write_source
from prepare_data import select_rows

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "hiertext"
DEFAULT_TERMS = (
    "HierText annotations: CC-BY-SA-4.0 per the official repository. "
    "Images are hosted by CVDF and sourced from Open Images; verify image-level terms separately."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_DATA / "gt" / "validation.jsonl.gz")
    parser.add_argument("--images", type=Path, default=DEFAULT_DATA / "validation" / "validation")
    parser.add_argument("--source-dir", type=Path, default=HERE / "data" / "source")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "manifests")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--license-notes", default=DEFAULT_TERMS)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"HierText annotations not found: {args.annotations}")
    annotations = read_lines(args.annotations)
    if args.samples > len(annotations):
        raise ValueError(f"HierText validation contains {len(annotations)} rows; {args.samples} requested")
    selected_indices = sorted(random.Random(args.seed).sample(range(len(annotations)), args.samples))
    selected_annotations = [annotations[index] for index in selected_indices]
    selected_annotations_path = args.annotations.parent / "validation_selected.jsonl"
    selected_annotations_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_annotations), encoding="utf-8")
    rows = import_hiertext(selected_annotations_path, args.images)
    for row, source_index in zip(rows, selected_indices):
        row["source_annotation_row"] = source_index
    args.source_dir.mkdir(parents=True, exist_ok=True)
    write_source(args.source_dir, "hiertext", rows, args.license_notes)
    source = args.source_dir / "hiertext.jsonl"
    selected = select_rows("hiertext", source, args.samples, args.seed)
    payload = {
        "task": "hiertext", "sample_count": len(selected), "seed": args.seed,
        "dataset": "HierText", "split": "validation", "license_or_access_terms": args.license_notes,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "asset_policy": "sampled image paths and annotations remain local",
        "examples": selected,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "hiertext.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"hiertext: {len(selected)} examples -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
