#!/usr/bin/env python3
"""Build the aggregate-only static report data from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def build_data(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    inventory_path = results_dir / "inventory.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    models = {}
    for model_id in summary.get("models", []):
        detail = next((row for block in summary.get("datasets", {}).values()
                       for row in block.get("models", []) if row.get("model_id") == model_id), {})
        models[model_id] = {"id": model_id, "backend": detail.get("backend"),
                            "revision": detail.get("model_revision")}
    candidates = inventory.get("models", [])
    candidate_ids = [row.get("id") if isinstance(row, dict) else row for row in candidates]
    datasets = {}
    for key, block in summary.get("datasets", {}).items():
        datasets[key] = {
            "name": block.get("display_name", key),
            "split": block.get("split"),
            "panel_n": block.get("panel_n"),
            "class_count": len(block.get("class_support", {})),
            "predicate_count": len(block.get("predicate_support", {})),
            "models": [{
                "id": row.get("model_id"),
                "backend": row.get("backend"),
                "requested": row.get("requested_n"),
                "completed": row.get("completed_n"),
                "valid_json": row.get("valid_json_n"),
                "failures": row.get("failure_n"),
                "metrics": row.get("metrics", {}),
            } for row in block.get("models", [])],
        }
    complete_ids = set()
    for model_id in models:
        if all(
            (row := next((candidate for candidate in datasets.get(key, {}).get("models", [])
                          if candidate.get("id") == model_id), None)) is not None
            and datasets[key].get("panel_n") == 200
            and row.get("requested") == 200
            and row.get("completed") == 200
            for key in ("vg150", "openimages_v7", "psg")
        ):
            complete_ids.add(model_id)
    incomplete_ids = [model_id for model_id in models if model_id not in complete_ids]
    candidates = inventory.get("models", [])
    candidate_ids = [row.get("id") if isinstance(row, dict) else row for row in candidates]
    pending_ids = [model_id for model_id in candidate_ids if model_id not in complete_ids]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_complete": bool(candidate_ids) and set(candidate_ids).issubset(complete_ids),
        "candidate_count": len(candidate_ids),
        "completed_model_count": len(complete_ids),
        "incomplete_models": incomplete_ids,
        "pending_models": pending_ids,
        "models": list(models.values()),
        "datasets": datasets,
        "suite_macro_mr_at_50": summary.get("suite_macro_mr_at_50", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results" / "20261106_full")
    parser.add_argument("--output", type=Path, default=HERE / "data.json")
    args = parser.parse_args()
    data = build_data(args.results_dir)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
