#!/usr/bin/env python3
"""Apply the task-format qualification gate to saved model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from score_official_subsample import (
    convert_official_graph_facts_event,
    parse_event_graph_edges,
)


MAVEN_ALLOWED = {
    "maven_ere_temporal_anchor": {
        "BEFORE", "AFTER", "OVERLAP", "SIMULTANEOUS", "ENDS-ON", "BEGINS-ON",
    },
    "maven_ere_causal_anchor": {"CAUSE", "PRECONDITION"},
    "maven_ere_subevent_anchor": {"SUBEVENT"},
}
HIE_ALLOWED = {"SuperSub", "SubSuper"}
REFUSAL = re.compile(r"^(?:sorry[,.: ]|i (?:cannot|can't|am unable)|i'm unable)", re.I)
TEXT_PANEL_SHA256 = "4541d7ed8e641ca70485180a04c0645acf022cea7a2dd446bc1a502a9d172849"
QUALIFICATION_ROOT = Path(__file__).resolve().parents[2] / ".agentic_workflow" / "workspace" / "qwenvl-four-task" / "qualification"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    if "###" in value:
        value = value.split("###", 1)[1]
    return value.strip()


def valid_gar(subtask: str, text: str) -> tuple[bool, str]:
    value = text.strip()
    if subtask in {"connectivity", "cycle"}:
        ok = re.fullmatch(r"(?:yes|no)[.!]?", value, re.I) is not None
        return ok, "binary answer must be only Yes or No"
    if subtask == "shortest_path":
        ok = re.fullmatch(r"\d+(?:\s*->\s*\d+)+[.]?", value) is not None
        return ok, "path must contain only node IDs joined by ->"
    if subtask == "matching":
        ok = re.fullmatch(r"\d+[.]?", value) is not None
        return ok, "matching size must be one nonnegative integer"
    return False, "unknown GAR subtask"


def valid_mgd(text: str, prompt: str) -> tuple[bool, str]:
    value = text.strip()
    if not value:
        return False, "empty response"
    if value == prompt.strip():
        return False, "prompt echoed without a description"
    if REFUSAL.search(value):
        return False, "refusal instead of a molecular description"
    if len(value) < 20:
        return False, "response is too short to be a molecular description"
    if not re.search(r"\b(molecule|molecular|compound|chemical|structure|atom|bond|ring|group|formula)\b", value, re.I):
        return False, "response does not describe a molecule or chemical structure"
    return True, "nonempty natural-language molecular description"


def valid_ere(subtask: str, text: str, label: str, prompt: str) -> tuple[bool, str]:
    if not text.strip() or text.strip().casefold() == "none":
        gold_edges = parse_event_graph_edges(label)
        if not gold_edges and text.strip().casefold() == "none":
            return True, "None represents empty gold relation set"
        return False, "no relation output for a nonempty gold example"
    candidate = text
    if any(marker in candidate for marker in ("[Graph Facts]", "entity(", "event(", "relation(")):
        candidate = convert_official_graph_facts_event(candidate)
    allowed = HIE_ALLOWED if subtask == "hievents" else MAVEN_ALLOWED.get(subtask, set())
    parsed = parse_event_graph_edges(candidate)
    kept = [edge for edge in parsed if edge[1] in allowed]
    anchor_match = re.search(r"ANCHOR event\s+(E\d+)\b", prompt)
    if not anchor_match:
        return False, "qualification prompt has no identifiable anchor event"
    anchor = anchor_match.group(1)
    anchored = [edge for edge in kept if edge[0] == anchor or edge[2] == anchor]
    if anchored:
        return True, f"{len(anchored)} parsed relation(s) survive the {subtask} filter and involve {anchor}"
    if kept:
        return False, f"{len(kept)} allowed relation(s) parsed, none involves anchor {anchor}"
    if parsed:
        return False, f"{len(parsed)} parsed relation(s), none allowed for {subtask}"
    return False, "no relation parseable by the pinned official evaluator"


def validate(row: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    raw = answer_text(output.get("raw_prediction"))
    family = row["family"]
    subtask = row["subtask"]
    if family == "GAR":
        valid, reason = valid_gar(subtask, raw)
    elif family == "MGD":
        valid, reason = valid_mgd(raw, row["prompt_text"])
    elif family == "ERE":
        valid, reason = valid_ere(subtask, raw, row["label"], row["prompt_text"])
    else:
        valid, reason = False, f"unsupported text-task family {family}"
    return {
        "sample_id": row["sample_id"],
        "family": family,
        "task_tag": row["task_tag"],
        "reported_subtask": row["reported_subtask"],
        "raw_prediction": output.get("raw_prediction"),
        "valid": valid,
        "reason": reason,
        "finish_reason": output.get("finish_reason"),
        "error": output.get("error"),
        "output_tokens": output.get("output_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--families", nargs="+", choices=("GAR", "MGD", "ERE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.attempt <= 3:
        raise ValueError("attempt must be baseline 0 or one of three repair attempts")

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if not isinstance(panel, list):
        raise ValueError("text qualification panel must be a JSON list")
    if sha256_file(args.panel) != TEXT_PANEL_SHA256:
        raise ValueError("refusing a panel other than the frozen qualification text panel")
    families = list(dict.fromkeys(args.families))
    panel = [row for row in panel if row.get("family") in families]
    panel_ids = [row.get("sample_id") for row in panel]
    if len(set(panel_ids)) != len(panel_ids) or any(not isinstance(sample_id, str) for sample_id in panel_ids):
        raise ValueError("qualification panel has missing or duplicate sample IDs")
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["sample_id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("prediction file contains duplicate sample IDs")
    extra = sorted(set(by_id) - set(panel_ids))
    if extra:
        raise ValueError(f"prediction file contains IDs outside the qualification panel: {extra[:3]}")

    items = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for row in panel:
        output = by_id.get(row["sample_id"])
        if output is None:
            missing.append(row["sample_id"])
            continue
        for field in ("source_file", "source_sha256", "source_index", "task_tag", "family",
                      "reported_subtask", "subtask", "label"):
            if output.get(field) != row.get(field):
                raise ValueError(f"prediction join mismatch for {row['sample_id']}: field {field}")
        suffix = output.get("repair_instruction", "")
        expected_prompt = (
            row["prompt_text"].rstrip() + f"\n\nOutput requirement: {suffix}"
            if suffix else row["prompt_text"]
        )
        if output.get("prompt_text") != expected_prompt:
            raise ValueError(f"prediction prompt differs from the frozen task prompt for {row['sample_id']}")
        if output.get("prompt_sha256") != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest():
            raise ValueError(f"prediction prompt hash mismatch for {row['sample_id']}")
        item = validate(row, output)
        items.append(item)
        grouped[row["family"]].append(item)

    summary = {}
    for family, values in grouped.items():
        counts = Counter(item["reported_subtask"] for item in values if item["valid"])
        summary[family] = {
            "n": len(values),
            "valid_n": sum(item["valid"] for item in values),
            "threshold": 3,
            "passed": len(values) == 10 and sum(item["valid"] for item in values) >= 3,
            "valid_by_subtask": dict(counts),
        }

    source_manifest_path = args.predictions.parent / "run_manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError("predictions have no run manifest; refusing to certify a qualification gate")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_rows = 10 * len(source_manifest.get("families", []))
    if (source_manifest.get("status") not in {"complete", "complete_with_errors"}
            or source_manifest.get("model_id") != args.model_id
            or source_manifest.get("attempt") != args.attempt
            or source_manifest.get("panel_sha256") != TEXT_PANEL_SHA256
            or source_manifest.get("panel_n") != expected_rows
            or source_manifest.get("completed_rows") != expected_rows
            or len(predictions) != expected_rows
            or sha256_file(args.predictions) != source_manifest.get("prediction_sha256")):
        raise ValueError("run manifest does not prove complete predictions for this model, attempt, and frozen panel")
    expected_families = set(source_manifest.get("families", []))
    if expected_families != set(families):
        raise ValueError("requested family set differs from the source run manifest")
    if set(summary) != expected_families or any(values["n"] != 10 for values in summary.values()):
        raise ValueError("predictions do not contain exactly ten rows for every requested family")
    source_revision = (source_manifest.get("model") or {}).get("revision")
    if source_revision is None:
        source_revision = source_manifest.get("model_revision")
    if not source_revision:
        raise ValueError("run manifest has no loaded model revision")
    record = {
        "schema_version": 1,
        "model_id": args.model_id,
        "attempt": args.attempt,
        "attempt_kind": "baseline" if args.attempt == 0 else "prompt_repair",
        "panel_path": str(args.panel.resolve()),
        "prediction_path": str(args.predictions.resolve()),
        "panel_n": len(panel),
        "prediction_n": len(predictions),
        "panel_sha256": sha256_file(args.panel),
        "prediction_sha256": sha256_file(args.predictions),
        "missing_sample_ids": missing,
        "task_summary": summary,
        "official_evaluator_sha256": source_manifest.get("official_evaluator_sha256"),
        "source_model_revision": source_revision,
        "source_runner_sha256": source_manifest.get("implementation_sha256", {}).get("runner"),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"preserving an earlier qualification report at {args.output}")
    import re as _re
    model_dir = QUALIFICATION_ROOT / _re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_id)
    if missing:
        raise ValueError("qualification attempt is incomplete; refusing to create a repair gate record")
    gate_records = {}
    for family in families:
        gate_path = model_dir / f"attempt-{args.attempt}-{family}-gate.json"
        if gate_path.exists():
            raise FileExistsError(f"preserving an earlier gate record at {gate_path}")
        family_record = dict(record)
        family_record["families"] = [family]
        family_record["panel_n"] = 10
        family_record["prediction_n"] = 10
        family_record["task_summary"] = {family: summary[family]}
        family_record["items"] = [item for item in items if item["family"] == family]
        gate_records[gate_path] = family_record
    model_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for gate_path, gate_record in gate_records.items():
        gate_path.write_text(json.dumps(gate_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aggregate_path = model_dir / f"attempt-{args.attempt}-gate.json"
    if not aggregate_path.exists():
        aggregate_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_id": args.model_id, "attempt": args.attempt,
                      "task_summary": summary, "missing_n": len(missing),
                      "output": str(args.output), "gate_records": [str(path) for path in gate_records]}, indent=2))


if __name__ == "__main__":
    main()
