#!/usr/bin/env python3
"""Split a completed all-subsets RF100-VL run into report-ready domain records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import benchmark  # noqa: E402


def import_run(run_path: Path, manifest_path: Path, output_root: Path,
               model_id: str, group: str) -> list[Path]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = run.get("tasks", {}).get("coco", {})
    items_by_id = {str(item.get("id")): item for item in task.get("items", [])}
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for source_row in manifest.get("examples", []):
        rows_by_dataset.setdefault(str(source_row["source_dataset"]), []).append(source_row)
    manifest_dirs = {}
    for path in (HERE / "data" / "domain_manifests" / "rf100_vl").glob("*/coco.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_dataset"):
            manifest_dirs[str(payload["source_dataset"])] = path.parent

    written = []
    for source_dataset, rows in sorted(rows_by_dataset.items()):
        domain_dir = manifest_dirs.get(source_dataset)
        if domain_dir is None:
            raise FileNotFoundError(f"domain manifest is missing for {source_dataset}")
        domain_manifest = domain_dir / "coco.json"
        source_payload = json.loads(domain_manifest.read_text(encoding="utf-8"))
        domain_rows = []
        domain_items = []
        for combined_row in rows:
            row = dict(combined_row)
            row["image_id"] = int(row["source_image_id"])
            row["categories"] = row["source_categories"]
            item = items_by_id.get(str(row["id"]))
            if item is None:
                continue
            item_copy = dict(item)
            input_record = dict(item_copy.get("input_record", {}))
            input_record["image_id"] = int(row["source_image_id"])
            input_record["categories"] = row["source_categories"]
            item_copy["input_record"] = input_record
            domain_rows.append(row)
            domain_items.append(item_copy)
        summary = benchmark.aggregate("coco", domain_items)
        ann_path = Path(source_payload["annotation_file"])
        summary["coco_official_ap"] = benchmark.coco_official_ap(ann_path, domain_rows, domain_items)
        summary["peak_whole_device_vram_gib"] = task.get("summary", {}).get("peak_whole_device_vram_gib")
        domain_run = {
            key: run.get(key) for key in (
                "model_id", "revision", "snapshot", "backend", "precision", "generation_mode",
                "client_package_versions", "service_inventory", "model_load_duration_s",
                "model_load_timing", "query_timing_scope", "input_size",
            )
        }
        domain_run["model_id"] = model_id
        domain_run["sample_count"] = len(domain_items)
        domain_run["source_dataset"] = source_dataset
        domain_run["tasks"] = {"coco": {"items": domain_items, "summary": summary}}
        slug = domain_dir.name
        target = output_root / f"rf100_{slug}" / group / f"{model_id.replace('/', '--')}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(domain_run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(target)
    return written


def _slug_for(source_dataset: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", source_dataset.casefold()).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="combined benchmark.py JSON output")
    parser.add_argument("--manifest", type=Path,
                        default=HERE / "data" / "domain_manifests" / "manifests" / "rf100_vl_all" / "coco.json")
    parser.add_argument("--output-root", type=Path, default=HERE / "results" / "domain_runs")
    parser.add_argument("--model", required=True, choices=benchmark.MODEL_IDS)
    parser.add_argument("--group", required=True,
                        help="distinct report run group, such as all_subsets_20")
    args = parser.parse_args()
    written = import_run(args.run, args.manifest, args.output_root, args.model, args.group)
    print(json.dumps({"domains": len(written), "files": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
