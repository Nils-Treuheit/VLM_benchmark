#!/usr/bin/env python3
"""Score completed Video VLM runs and build the report's aggregate-only data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scoring import INTERVAL_PARSER_VERSION, parse_intervals

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data/video_bench"
RAW = DATA_ROOT / "results/raw"
OUT = Path(__file__).resolve().parent / "data.json"
VUE_SOURCE = DATA_ROOT / "VUE-TR/source"
MODELS = {
    "internvideo3_8b_instruct": ("InternVideo3-8B-Instruct", ("mvbench", "vue_tr")),
    "timelens2_8b": ("TimeLens2-8B", ("vue_tr",)),
    "llava_next_video_7b": ("LLaVA-NeXT-Video-7B", ("mvbench", "vue_tr")),
    "videochat3_4b": ("VideoChat3-4B", ("mvbench", "vue_tr")),
}
DATASET_LINKS = {
    "mvbench": "https://github.com/OpenGVLab/Ask-Anything",
    "vue_tr": "https://github.com/bytedance/vidi/tree/main/VUE_TR",
}


def official_vue_scorer():
    source = VUE_SOURCE / "qa_eval.py"
    spec = importlib.util.spec_from_file_location("official_vue_tr_qa_eval", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official VUE-TR scorer at {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite_or_none(value: Any) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def score_mvbench(model: str, expected: list[dict[str, Any]]) -> dict[str, Any]:
    path = RAW / f"mvbench__{model}.jsonl"
    rows = load_jsonl(path) if path.exists() else []
    latest = {str(row.get("item_id")): row for row in rows}
    expected_ids = {f"{item['task']}:{item['source_index']}" for item in expected}
    by_id = {key: row for key, row in latest.items() if row.get("status") == "complete"}
    per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in expected:
        row = by_id.get(f"{item['task']}:{item['source_index']}")
        if row:
            per_task[item["task"]].append(row)
    task_scores = {
        task: {"accuracy": sum(bool(r.get("correct")) for r in values) / len(values),
               "correct": sum(bool(r.get("correct")) for r in values), "completed": len(values)}
        for task, values in per_task.items() if values
    }
    valid = [by_id[key] for key in expected_ids if key in by_id]
    correct = sum(bool(row.get("correct")) for row in valid)
    errors = sum(1 for key in expected_ids if key in latest and latest[key].get("status") != "complete")
    errors += sum(1 for row in rows if row.get("status") == "prepare_error")
    return {
        "status": "complete" if len(valid) == len(expected) else ("incomplete" if rows else "not_run"),
        "expected": len(expected), "completed": len(valid), "correct": correct,
        "accuracy": correct / len(valid) if valid else None,
        "task_macro_accuracy": float(np.mean([v["accuracy"] for v in task_scores.values()])) if task_scores else None,
        "parse_failures": sum(row.get("parse_status") not in ("letter", "exact_choice_text") for row in valid),
        "inference_errors": errors,
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in valid])) if valid else None,
        "task_scores": task_scores,
    }


def score_vue(model: str, expected: list[dict[str, Any]], scorer) -> dict[str, Any]:
    path = RAW / f"vue_tr__{model}.jsonl"
    rows = load_jsonl(path) if path.exists() else []
    latest = {str(row.get("item_id")): row for row in rows}
    expected_ids = {str(item["query_id"]) for item in expected}
    by_id = {str(row.get("query_id")): row for row in latest.values()
             if row.get("status") == "complete" and row.get("query_id") is not None}
    selected = []
    for query_id in expected_ids:
        row = by_id.get(query_id)
        if row is not None:
            parsed, parse_error, parse_status = parse_intervals(row.get("raw_output"))
            selected.append({**row, "prediction_intervals_seconds": parsed,
                             "parse_error": parse_error, "parse_status": parse_status,
                             "interval_parser_version": INTERVAL_PARSER_VERSION})
    temporary = RAW / f".official_input__{model}.jsonl"
    official_rows = []
    for row in selected:
        official_rows.append({"query_id": row["query_id"], "answer": row.get("prediction_intervals_seconds", [])})
    payload = "\n".join(json.dumps(row) for row in official_rows)
    temporary.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    try:
        evaluated = scorer.load_result(str(VUE_SOURCE / "VUE-TR_ground_truth.json"), str(temporary))
        _, auc = scorer.success_overlap(evaluated) if evaluated else (None, None)
        precision, recall = scorer.compute_precision_recall(evaluated) if evaluated else (None, None)
    finally:
        temporary.unlink(missing_ok=True)
    by_query = {int(row["query_id"]): row for row in selected}
    errors = sum(1 for key in expected_ids if key in latest and latest[key].get("status") != "complete")
    errors += sum(1 for row in rows if row.get("status") == "prepare_error")
    strata: dict[str, dict[str, Any]] = {}
    for key in ("duration_category", "query_format"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in expected:
            row = by_query.get(item["query_id"])
            if row:
                groups[item[key]].append(row)
        strata[key] = {
            group: {"queries": len(values), "parse_failures": sum(bool(r.get("parse_error")) for r in values)}
            for group, values in sorted(groups.items())
        }
    return {
        "status": "complete" if len(selected) == len(expected) else ("incomplete" if rows else "not_run"),
        "expected": len(expected), "completed": len(selected),
        "success_auc": finite_or_none(auc),
        "precision_auc": finite_or_none(precision),
        "recall_auc": finite_or_none(recall),
        "parse_failures": sum(bool(row.get("parse_error")) for row in selected),
        "inference_errors": errors,
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in selected])) if selected else None,
        "strata": strata,
        "scorer": "official ByteDance VUE-TR qa_eval.py: success_overlap, compute_precision_recall; prediction seconds floored/ceiled by official loader",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    mv_manifest = json.loads((DATA_ROOT / "prepared/mvbench_manifest.json").read_text())
    vue_manifest = json.loads((DATA_ROOT / "prepared/vue_tr_manifest.json").read_text())
    sampling_policy = json.loads((DATA_ROOT / "prepared/video_qa_sampling_policy.json").read_text())
    inventory = json.loads((DATA_ROOT / "prepared/model_inventory.json").read_text())
    scorer = official_vue_scorer()
    output_path = args.output if args.output.is_absolute() else PROJECT / args.output
    profile_fields = (
        "setup_id", "max_frames", "precision", "attention", "device", "device_map",
        "frame_min_pixels", "frame_max_pixels", "native_video_only", "deterministic_default",
    )
    profiles = {
        model["adapter_profile"]["setup_id"]: {
            key: model["adapter_profile"][key]
            for key in profile_fields if key in model["adapter_profile"]
        }
        for model in inventory.get("models", [])
        if isinstance(model.get("adapter_profile"), dict) and model["adapter_profile"].get("setup_id")
    }
    sampling_summary = {
        "policy_id": sampling_policy["policy_id"],
        "max_frames": sampling_policy["max_frames"],
        "target_positions": sampling_policy["selection"]["target_positions"],
        "frame_choice": sampling_policy["selection"]["frame_choice"],
        "tie_break": sampling_policy["selection"]["tie_break"],
        "duplicate_policy": sampling_policy["selection"]["duplicate_policy"],
        "timestamp_policy": sampling_policy["selection"]["timestamp_policy"],
        "decoder_strategy": sampling_policy["selection"]["decoder_strategy"],
        "source_resolution": sampling_policy["preprocessing"]["source_resolution"],
        "audio": sampling_policy["audio"],
        "subtitles": sampling_policy["subtitles"],
    }
    report: dict[str, Any] = {
        "title": "Video VLM Benchmark",
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "status": "complete",
        "scope": {
            "mvbench": {"items": len(mv_manifest["items"]), "tasks": len({i["task"] for i in mv_manifest["items"]}),
                        "models": [m for m, (_, tracks) in MODELS.items() if "mvbench" in tracks], "source": DATASET_LINKS["mvbench"],
                        "excluded": "Five NTU RGB+D fine-grained-pose examples were excluded at the user's direction."},
            "vue_tr": {"items": len(vue_manifest["items"]), "unique_videos": vue_manifest["unique_videos"],
                       "models": list(MODELS), "source": DATASET_LINKS["vue_tr"],
                       "sampling": "100 vision-only queries, one per video, seed 3407; audio and audiovisual queries excluded."},
        },
        "method": {"frames": sampling_policy["max_frames"], "audio": "not supplied", "temperature": 0.0,
                   "qa_max_new_tokens": 64, "retrieval_max_new_tokens": 512,
                   "cross_track_composite": None,
                   "sampling": sampling_summary,
                   "model_profiles": {
                       "internvideo3_8b_instruct": profiles.get("internvideo3_8b_instruct", {}),
                       "timelens2_8b": profiles.get("timelens2_8b", {}),
                       "llava_next_video_7b": profiles.get("llava_next_video_7b", {}),
                       "videochat3_4b": profiles.get("videochat3_4b", {}),
                   },
                   "interval_parser": {
                       "version": INTERVAL_PARSER_VERSION,
                       "accepted": "numeric endpoint pairs, numeric-string endpoint pairs, and a top-level numeric pair as one interval",
                       "malformed": "counted as parse failures and submitted as empty predictions to the official VUE-TR scorer",
                   },
                   "note": "MVBench answer-choice accuracy and VUE-TR temporal grounding are separate tasks and are not combined."},
        "models": {},
    }
    for model, (name, tracks) in MODELS.items():
        entry: dict[str, Any] = {"name": name, "id": model, "tracks": {}}
        if "mvbench" in tracks:
            entry["tracks"]["mvbench"] = score_mvbench(model, mv_manifest["items"])
        if "vue_tr" in tracks:
            entry["tracks"]["vue_tr"] = score_vue(model, vue_manifest["items"], scorer)
        report["models"][model] = entry
    statuses = [t["status"] for m in report["models"].values() for t in m["tracks"].values()]
    if statuses and all(status == "complete" for status in statuses):
        report["status"] = "complete"
    elif statuses and all(status == "not_run" for status in statuses):
        report["status"] = "not_started"
    else:
        report["status"] = "in_progress"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Video VLM Benchmark Report", "",
        f"**Evaluation status:** {report['status']}  ",
        f"**Generated:** {report['generated_at_utc']}", "",
        "## Scope", "",
        "| Track | Sample | Models | Score |", "|---|---:|---:|---|",
        f"| MVBench video QA | {report['scope']['mvbench']['items']} questions / {report['scope']['mvbench']['tasks']} tasks | {len(report['scope']['mvbench']['models'])} | Exact-choice accuracy and task macro accuracy |",
        f"| VUE-TR temporal grounding | {report['scope']['vue_tr']['items']} queries / {report['scope']['vue_tr']['unique_videos']} videos | {len(report['scope']['vue_tr']['models'])} | Official success, precision, and recall AUC |",
        "", report["method"]["note"], "",
        "## Results", "", "### MVBench", "",
        "| Model | Accuracy | Task macro accuracy | Coverage | Mean latency | Parse failures | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_id, model in report["models"].items():
        score = model["tracks"].get("mvbench")
        if score:
            lines.append(f"| {model['name']} | {pct_text(score['accuracy'])} | {pct_text(score['task_macro_accuracy'])} | {score['completed']}/{score['expected']} | {ms_text(score['mean_latency_ms'])} | {score['parse_failures']} | {score['status']} |")
    lines += ["", "### VUE-TR", "", "| Model | Success AUC | Precision AUC | Recall AUC | Coverage | Mean latency | Parse failures | Status |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for model_id, model in report["models"].items():
        score = model["tracks"].get("vue_tr")
        if score:
            lines.append(f"| {model['name']} | {pct_text(score['success_auc'])} | {pct_text(score['precision_auc'])} | {pct_text(score['recall_auc'])} | {score['completed']}/{score['expected']} | {ms_text(score['mean_latency_ms'])} | {score['parse_failures']} | {score['status']} |")
    sampling = report["method"]["sampling"]
    profile_text = "; ".join(
        f"{model['name']}: {profile.get('precision', 'adapter default')}, max {profile.get('max_frames', 'adapter default')} frames"
        + (f", {profile['frame_min_pixels']} px/frame" if profile.get("frame_min_pixels") else "")
        for model_id, model in report["models"].items()
        if (profile := report["method"]["model_profiles"].get(model_id))
    )
    interval_parser = report["method"]["interval_parser"]
    lines += ["", "## Protocol", "", f"Up to {sampling['max_frames']} shared source-PTS frames per clip. Target positions span the timeline; {sampling['frame_choice']}; {sampling['tie_break']}. {sampling['decoder_strategy']}.", "", f"Original source resolution is retained through decoding; pinned model processors apply each model's configured preprocessing. Profiles: {profile_text}.", "", f"Audio: {sampling['audio']}. Subtitles: {sampling['subtitles']['track_input']}; burned-in text is retained. Generation temperature is {report['method']['temperature']}; max new tokens are {report['method']['qa_max_new_tokens']} for QA and {report['method']['retrieval_max_new_tokens']} for temporal retrieval.", "", f"VUE-TR interval parser {interval_parser['version']} accepts {interval_parser['accepted']}. {interval_parser['malformed']}.", "", report["scope"]["mvbench"]["excluded"], "", report["scope"]["vue_tr"]["sampling"], "", "Dataset videos, annotations, queries, and item-level outputs remain local and are not included in the website bundle.", "", "## Sources", "", f"- [MVBench](<{report['scope']['mvbench']['source']}>)", f"- [VUE-TR](<{report['scope']['vue_tr']['source']}>)", ""]
    report_path = Path(__file__).resolve().parent / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_path}; status={report['status']}")
    return 0


def pct_text(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def ms_text(value: Any) -> str:
    return "—" if value is None else f"{float(value):.0f} ms"


if __name__ == "__main__":
    raise SystemExit(main())
