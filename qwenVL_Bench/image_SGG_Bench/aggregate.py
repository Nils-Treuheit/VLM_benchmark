#!/usr/bin/env python3
"""Validate run provenance and aggregate per-dataset SGG metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluator import aggregate_items, compatible_provenance, content_hash


def read_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "inventory.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read complete run record {path}: {exc}") from exc
        if "model_id" in data and "datasets" in data:
            runs.append(data)
    return runs


def aggregate(results_dir: Path, panel_dir: Path) -> dict[str, Any]:
    runs = read_runs(results_dir)
    if not runs:
        raise ValueError(f"no model run files found under {results_dir}")
    panels = {}
    manifest_hash_by_dataset = {}
    for dataset in ("vg150", "openimages_v7", "psg"):
        p = json.loads((panel_dir / dataset / "manifest.json").read_text(encoding="utf-8"))
        panels[dataset] = p
        manifest_hash_by_dataset[dataset] = hashlib.sha256(
            (panel_dir / dataset / "manifest.json").read_bytes()
        ).hexdigest()
    output: dict[str, Any] = {"schema_version": 1, "datasets": {}, "models": []}
    protocol_by_dataset: dict[str, dict[str, str]] = {}
    for dataset, manifest in panels.items():
        protocol_by_dataset[dataset] = {
            "manifest_hash": manifest_hash_by_dataset[dataset],
            "source_hashes": content_hash(manifest.get("source_provenance", {})),
            "vocabulary_hash": manifest.get("vocabulary", {}).get("hash") or
                               content_hash(manifest.get("vocabulary", {})),
        }
    for dataset in panels:
        model_rows = []
        provenance = []
        for run in runs:
            block = run["datasets"].get(dataset)
            if not isinstance(block, dict):
                continue
            items = block.get("items", [])
            if not compatible_provenance(items):
                raise ValueError(f"run {run['model_id']} contains mixed provenance for {dataset}")
            if items:
                first = items[0]
                expected = protocol_by_dataset[dataset]
                actual = {"manifest_hash": first.get("manifest_hash"),
                          "source_hashes": first.get("source_hashes", {}).get(dataset),
                          "vocabulary_hash": first.get("vocabulary_hash")}
                # Require each runner to use the exact panel's sources and vocab.
                for key in ("source_hashes", "vocabulary_hash"):
                    if actual[key] != expected[key]:
                        raise ValueError(f"{dataset} {key} mismatch for {run['model_id']}")
                if actual["manifest_hash"] != expected["manifest_hash"]:
                    raise ValueError(f"{dataset} manifest_hash mismatch for {run['model_id']}")
                provenance.append(first)
            rows = panels[dataset].get("items", [])
            aggregate_values = aggregate_items(items, rows)
            model_rows.append({
                "model_id": run["model_id"],
                "model_revision": run.get("model_revision"),
                "backend": run.get("backend"),
                **aggregate_values,
            })
        manifest_hashes = {item.get("manifest_hash") for item in provenance}
        prompt_hashes = {item.get("prompt_hash") for item in provenance}
        evaluator_hashes = {item.get("evaluator_hash") for item in provenance}
        vocabulary_hashes = {item.get("vocabulary_hash") for item in provenance}
        source_hashes = {content_hash(item.get("source_hashes", {})) for item in provenance}
        if any(len(values) > 1 for values in (manifest_hashes, prompt_hashes,
                                               evaluator_hashes, vocabulary_hashes, source_hashes)):
            raise ValueError(f"runs for {dataset} have incompatible benchmark provenance")
        output["datasets"][dataset] = {
            "split": panels[dataset].get("split"),
            "panel_n": len(panels[dataset].get("items", [])),
            "display_name": panels[dataset].get("display_name", dataset),
            "provenance": protocol_by_dataset[dataset],
            "source_provenance": panels[dataset].get("source_provenance", {}),
            "class_support": panels[dataset].get("class_support", {}),
            "predicate_support": panels[dataset].get("predicate_support", {}),
            "models": model_rows,
        }
    output["models"] = sorted({run["model_id"] for run in runs})
    # The optional macro is published only for models with all three full panels.
    output["suite_macro_mr_at_50"] = {}
    for model in output["models"]:
        by_dataset = {}
        complete = True
        for dataset, entry in output["datasets"].items():
            match = next((x for x in entry["models"] if x["model_id"] == model), None)
            if not match or match["completed_n"] != 200:
                complete = False
                break
            by_dataset[dataset] = match["metrics"]["mr_at_k"]["50"]
        if complete:
            output["suite_macro_mr_at_50"][model] = sum(by_dataset.values()) / 3
    return output


def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# Image SGG benchmark", "",
             "Fixed 200-image panels; direct image-to-graph generation. Results are subset metrics.",
             "PSG uses box-adapted SGG and is not the official mask-based metric.", ""]
    for dataset, block in result["datasets"].items():
        lines += [f"## {block['display_name']}", "",
                  f"Split: `{block['split']}`; panel size: {block['panel_n']}.", "",
                  "| Model | Backend | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | "
                  "Triplet P@50 | Triplet R@50 | Triplet F1@50 | Obj R@50 | JSON valid | Completed | Failures |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model in block["models"]:
            m = model["metrics"]
            lines.append(
                f"| {model['model_id']} | {model.get('backend') or ''} | "
                f"{m['r_at_k']['20']:.4f} | {m['r_at_k']['50']:.4f} | {m['r_at_k']['100']:.4f} | "
                f"{m['mr_at_k']['20']:.4f} | {m['mr_at_k']['50']:.4f} | {m['mr_at_k']['100']:.4f} | "
                f"{m['triplet_precision_at_50']:.4f} | {m['triplet_recall_at_50']:.4f} | "
                f"{m['triplet_f1_at_50']:.4f} | {m['object_recall_at_50']:.4f} | "
                f"{model['valid_json_n']}/{model['requested_n']} | {model['completed_n']}/{model['requested_n']} | "
                f"{model['failure_n']} |"
            )
        lines.append("")
    if result["suite_macro_mr_at_50"]:
        lines += ["## Optional three-panel macro", "",
                  "Unweighted mean of dataset-level mR@50; shown only for models with all three full panels.", "",
                  "| Model | Macro mR@50 |", "|---|---:|"]
        for model, value in sorted(result["suite_macro_mr_at_50"].items()):
            lines.append(f"| {model} | {value:.4f} |")
        lines.append("")
    lines += ["## Protocol and limitations", "",
              "Each dataset is reported separately. Source hashes, source revisions, class support, and predicate support "
              "are recorded in `summary.json`. "
              "Malformed output and runtime failures remain in the denominator. The panels are small local subsets, "
              "not full-set leaderboard scores. PSG results are box-adapted and must not be compared as official mask scores.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--panel-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    summary = aggregate(args.results_dir, args.panel_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.results_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
