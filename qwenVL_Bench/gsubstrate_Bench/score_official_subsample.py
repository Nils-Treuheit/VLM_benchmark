#!/usr/bin/env python3
"""Score a completed G-Substrate subset with the pinned official evaluators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "qwenVL_Bench").is_dir() and (parent / ".agentic_workflow").is_dir()
)
OFFICIAL_ROOT = PROJECT_ROOT / ".agentic_workflow" / "workspace" / "g-substrate-official"
OFFICIAL_INFERENCE = OFFICIAL_ROOT / "inference"
SCORING_DEPS = PROJECT_ROOT / ".agentic_workflow/workspace/g-substrate-cross-task/scoring-dependencies"
TARGETS = {
    "GAR-CT": 98.41,
    "GAR-CD": 96.97,
    "GAR-SP": 48.59,
    "GAR-BM": 94.54,
    "MGD-BLEU4": 51.53,
    "MGD-ROUGE-L": 68.47,
    "ERE-MA-S": 52.20,
    "ERE-MA-T": 42.68,
    "ERE-MA-C": 40.91,
    "ERE-HiE": 25.15,
    "SGG-PCIs-R@50": 25.38,
    "SGG-PCIs-mR@50": 8.67,
}
ERE_TO_LABEL = {
    "maven_ere_subevent_anchor": "MA-S",
    "maven_ere_temporal_anchor": "MA-T",
    "maven_ere_causal_anchor": "MA-C",
    "hievents": "HiE",
}
GAR_TO_LABEL = {
    "connectivity": "CT",
    "cycle": "CD",
    "shortest_path": "SP",
    "matching": "BM",
}
SGG_DIR = PROJECT_ROOT / ".agentic_workflow/workspace/image-sgg-recovery/native-pcis/comparisons/pcis-vg50-open-vocabulary-tuples-v3-50"
EXPECTED_SGG_FILES = {
    "comparison_manifest.json": "2ec0cf7fff3d827935c4d21a932b09d107faf018a7389bc5932ac782c14cf118",
    "zmli_G-Substrate-Qwen3-VL-2B.scores.json": "c2c70f54796b55b74e1f3a151fcc67733e900f67efab86a78701fce69e607bd2",
}
EXPECTED_SGG_RECORD_SHA256 = "db8fe5d7016a2a183e5ce96735b7821499f283276492e2ecfd7781abd58228f1"
EXPECTED_SGG_INPUT_SHA256 = "69da14e7ed66695e2ead8b42981cf06d7faf5331887c43cd71f9eb66ba2f096e"
EXPECTED_SGG_PANEL_SHA256 = "d094767acad647c54ad63a266efdde178c229f8950de86ce97b1cb280fea051e"
EXPECTED_SGG_ROUTER_SHA256 = "ebd89bd1de3a89c7684421afb25deace9ab80d5d28dd2cf34dab39c1b5596e66"
EXPECTED_SGG_SCORER_SHA256 = "e9a2c81d0cc7f9fbdac9ab7fa63a2843f184afa1cf48b161db3782fc214e15f3"

if str(SCORING_DEPS) not in sys.path:
    sys.path.insert(0, str(SCORING_DEPS))
if str(OFFICIAL_INFERENCE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_INFERENCE))
from evaluators.event_graph_eval import (  # noqa: E402
    evaluate_event_graph_original_metrics,
    parse_event_graph_edges,
)
from evaluators.graph_search_eval import eval_graph_search  # noqa: E402
from evaluators.mol_eval import eval_molecular  # noqa: E402

# The pinned ROUGE-L implementation recursively reconstructs the LCS path;
# capped model outputs can exceed Python's default recursion limit.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def launched_runner_sha256(run_manifest: dict[str, Any]) -> str:
    recorded = run_manifest.get("implementation_sha256", {}).get("runner")
    if isinstance(recorded, str):
        return recorded
    path = Path(__file__).with_name("run_official_subsample.py")
    source = path.read_text(encoding="utf-8")
    metadata_block = '''        "implementation_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "scorer": sha256_file(Path(__file__).with_name("score_official_subsample.py").resolve()),
        },
'''
    if metadata_block not in source:
        raise ValueError("legacy run lacks runner hash and source cannot be reconstructed")
    source = source.replace(metadata_block, "", 1)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def extract_answer_content(text: Any) -> Any:
    """Match the pinned router's reasoning/final-answer separator handling."""
    if isinstance(text, str) and "###" in text:
        return text.split("###", 1)[1].strip()
    return text


def convert_official_graph_facts_event(text: str) -> str:
    """Port the pinned official router's Graph Facts to Event Graph conversion."""
    lines = ["<event_graph>", "<Node>"]
    event_pattern = re.compile(r'event\s*\(\s*([^,]+)\s*,\s*"?([^")\n]+)')
    for match in event_pattern.finditer(text):
        node_id, content = match.groups()
        content = content.split('"')[0].split(')')[0].strip()
        lines.append(f'({node_id.strip()}, "{content}")')
    lines.extend(("</Node>", "<Edge>"))
    relation_pattern = re.compile(r'relation\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)\n]+)')
    for match in relation_pattern.finditer(text):
        source, relation, target = match.groups()
        target = target.split(')')[0].strip()
        lines.append(f"({source.strip()}, {relation.strip()}, {target})")
    lines.extend(("</Edge>", "</event_graph>"))
    return "\n".join(lines)


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def bootstrap_mean(values: list[float], rng: random.Random, repetitions: int) -> tuple[float, float]:
    estimates = []
    n = len(values)
    for _ in range(repetitions):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def bootstrap_gar(
    groups: dict[str, list[float]], rng: random.Random, repetitions: int
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    intervals: dict[str, tuple[float, float]] = {}
    samples_by_task = {task: [] for task in groups}
    macro_samples: list[float] = []
    for _ in range(repetitions):
        replicate_means = []
        for task, values in groups.items():
            n = len(values)
            sample_mean = sum(values[rng.randrange(n)] for _ in range(n)) / n
            samples_by_task[task].append(sample_mean)
            replicate_means.append(sample_mean)
        macro_samples.append(sum(replicate_means) / len(replicate_means))
    for task, estimates in samples_by_task.items():
        estimates.sort()
        intervals[task] = (percentile(estimates, 0.025), percentile(estimates, 0.975))
    macro_samples.sort()
    return intervals, (percentile(macro_samples, 0.025), percentile(macro_samples, 0.975))


def metric_line(
    name: str, estimate: float, target: float, interval: tuple[float, float] | None
) -> dict[str, Any]:
    estimate_pct = estimate * 100.0
    diff = estimate_pct - target
    interval_pct = None if interval is None else [interval[0] * 100.0, interval[1] * 100.0]
    target_in_interval = (
        interval_pct is not None and interval_pct[0] <= target <= interval_pct[1]
    )
    return {
        "metric": name,
        "estimate_percent": estimate_pct,
        "target_percent": target,
        "difference_percentage_points": diff,
        "ci95_percent": interval_pct,
        "target_in_ci95": target_in_interval,
        "close": abs(diff) <= 5.0 or target_in_interval,
    }


def get_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 600:
        raise ValueError(f"expected 600 saved predictions, found {len(rows)}")
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    panel_path = run_dir / "panel.json"
    panel_manifest_path = run_dir / "panel_manifest.json"
    if sha256_file(path) != run_manifest.get("predictions_sha256"):
        raise ValueError("prediction file hash does not match the run manifest")
    if sha256_file(panel_manifest_path) != run_manifest.get("panel_manifest_sha256"):
        raise ValueError("panel manifest hash does not match the run manifest")
    if sha256_file(panel_path) != run_manifest.get("panel_sha256"):
        raise ValueError("panel rows hash does not match the run manifest")
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if len(panel) != 600 or panel_manifest.get("row_count") != 600:
        raise ValueError("frozen panel artifacts do not both contain 600 rows")
    sample_ids = [row.get("sample_id") for row in panel]
    if len(set(sample_ids)) != 600 or any(not isinstance(sid, str) for sid in sample_ids):
        raise ValueError("panel sample IDs are missing or duplicated")
    expected_ids_hash = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    if panel_manifest.get("sample_ids_sha256") != expected_ids_hash:
        raise ValueError("panel sample-ID hash mismatch")
    if (panel_manifest.get("dataset_revision") != run_manifest.get("dataset_revision")
            or panel_manifest.get("split") != "test"):
        raise ValueError("panel dataset revision or split differs from the run manifest")
    dataset_dir = Path(run_manifest["dataset_dir"])
    dataset_manifest_path = dataset_dir / "manifest.json"
    if sha256_file(dataset_manifest_path) != run_manifest.get("dataset_manifest_sha256"):
        raise ValueError("source dataset manifest hash does not match the run manifest")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if (dataset_manifest.get("dataset") != "zmli/G-Substrate-Data"
            or dataset_manifest.get("revision") != run_manifest.get("dataset_revision")
            or dataset_manifest.get("split") != "test"):
        raise ValueError("source dataset identity, revision, or split is invalid")
    loaded_sources: dict[str, list[dict[str, Any]]] = {}
    for source_file in {row["source_file"] for row in panel}:
        source_path = dataset_dir / source_file
        expected_hash = dataset_manifest["files"][source_file]["sha256"]
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"source dataset file hash mismatch: {source_file}")
        loaded_sources[source_file] = json.loads(source_path.read_text(encoding="utf-8"))
    if [row.get("sample_id") for row in rows] != sample_ids:
        raise ValueError("prediction rows do not follow the exact frozen panel order")
    join_fields = (
        "sample_id", "source_file", "source_sha256", "source_index", "task_tag",
        "family", "reported_subtask", "task", "subtask", "prompt_text", "label", "images",
    )
    for ordinal, (source, prediction) in enumerate(zip(panel, rows), start=1):
        mismatches = [field for field in join_fields if prediction.get(field) != source.get(field)]
        if mismatches:
            raise ValueError(f"prediction row {ordinal} is not joined to its panel source: {mismatches}")
        source_file = source["source_file"]
        source_index = source["source_index"]
        original = loaded_sources[source_file][source_index]
        messages = original.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"source row {source_file}[{source_index}] lacks messages")
        users = [m for m in messages if m.get("role") == "user"]
        labels = [m for m in messages if m.get("role") == "assistant"]
        tag_match = re.match(r"^\[([^\]]+)\]", users[0].get("content", "")) if len(users) == 1 else None
        expected_tag = tag_match.group(1) if tag_match else None
        expected_family = "GAR" if source_file == "graph_search.json" else (
            "MGD" if source_file == "molecular.json" else "ERE"
        )
        expected_task = "graph_search" if expected_family == "GAR" else (
            "mol" if expected_family == "MGD" else "event_graph"
        )
        if (len(users) != 1 or len(labels) != 1
                or users[0].get("content") != source.get("prompt_text")
                or labels[0].get("content") != source.get("label")
                or expected_tag != source.get("task_tag")
                or original.get("images", []) != source.get("images")
                or source.get("source_sha256") != dataset_manifest["files"][source_file]["sha256"]
                or source.get("sample_id") != f"{source_file}:{source.get('source_sha256')}:{source_index}"
                or source.get("family") != expected_family
                or source.get("task") != expected_task
                or source.get("subtask") != source.get("task_tag", "").split(":", 1)[-1]):
            raise ValueError(f"panel row {ordinal} does not match pinned source {source_file}[{source_index}]")
    failures = [row for row in rows if row.get("error") or not row.get("valid_for_aggregate")]
    if failures:
        raise ValueError(f"failed panels cannot be scored: {len(failures)} errored or invalid rows")
    return rows


def per_example_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, list[float]], dict[str, Any]]:
    gar: dict[str, list[float]] = {task: [] for task in GAR_TO_LABEL}
    mgd_bleu: list[float] = []
    mgd_rouge: list[float] = []
    ere: dict[str, list[float]] = {task: [] for task in ERE_TO_LABEL}
    ere_edge_counts = {task: {"parsed_rows": 0, "predicted_edges": 0, "gold_edges": 0} for task in ERE_TO_LABEL}
    empty_outputs = {"GAR": 0, "MGD": 0, "ERE": 0}
    graph_facts_converted = 0

    for row in rows:
        category = row["family"]
        pred = extract_answer_content(row.get("raw_prediction", ""))
        label = row["label"]
        if not pred.strip():
            empty_outputs[category] += 1
        if category == "GAR":
            subtask = row["reported_subtask"]
            if subtask not in gar:
                raise ValueError(f"unsupported GAR task {subtask}")
            score = eval_graph_search({"predict": pred, "label": label})
            gar[subtask].append(float(score))
        elif category == "MGD":
            score = eval_molecular({"predict": pred, "label": label})
            mgd_bleu.append(float(score["bleu"]))
            mgd_rouge.append(float(score["rouge_l"]))
        elif category == "ERE":
            tag = row["task_tag"].split(":", 1)[1]
            if tag not in ere:
                raise ValueError(f"unsupported ERE task tag {tag}")
            graph_facts_markers = ("[Graph Facts]", "entity(", "event(", "relation(")
            if any(marker in pred or marker in label for marker in graph_facts_markers):
                if any(marker in pred for marker in graph_facts_markers):
                    pred = convert_official_graph_facts_event(pred)
                if any(marker in label for marker in graph_facts_markers):
                    label = convert_official_graph_facts_event(label)
                graph_facts_converted += 1
            if any(marker in pred or marker in label for marker in ("<graph>", "<nodes>", "<edges>")):
                raise ValueError("UGS event graphs require the pinned official UGS conversion path")
            event_subtask = "Hievent" if tag == "hievents" else tag
            score = evaluate_event_graph_original_metrics({
                "task": "event_graph", "subtask": event_subtask,
                "predict": pred, "label": label,
            })
            ere[tag].append(float(score["f1"]))
            predicted_edges = parse_event_graph_edges(pred)
            gold_edges = parse_event_graph_edges(label)
            if predicted_edges:
                ere_edge_counts[tag]["parsed_rows"] += 1
            ere_edge_counts[tag]["predicted_edges"] += len(predicted_edges)
            ere_edge_counts[tag]["gold_edges"] += len(gold_edges)
        else:
            raise ValueError(f"unsupported family {category}")

    expected_counts = {
        "connectivity": 50, "cycle": 50, "shortest_path": 50, "matching": 50,
        "molecular": 200,
        "maven_ere_subevent_anchor": 60,
        "maven_ere_temporal_anchor": 60,
        "maven_ere_causal_anchor": 60,
        "hievents": 20,
    }
    actual_counts = {**{key: len(value) for key, value in gar.items()},
                     "molecular": len(mgd_bleu),
                     **{key: len(value) for key, value in ere.items()}}
    if actual_counts != expected_counts:
        raise ValueError(f"sample count mismatch: {actual_counts}")
    return {
        "gar": gar, "mgd_bleu": mgd_bleu, "mgd_rouge": mgd_rouge, "ere": ere,
    }, {"empty_outputs": empty_outputs, "ere_edge_counts": ere_edge_counts,
        "graph_facts_converted_rows": graph_facts_converted}


def render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# G-Substrate cross-task subsample report",
        "",
        f"Model revision: `{metrics['model_revision']}`  ",
        f"Dataset revision: `{metrics['dataset_revision']}`  ",
        f"Official repository commit: `{metrics['official_repository_commit']}`  ",
        f"Decoding: `{json.dumps(metrics['decoding'], sort_keys=True)}`.",
        "",
        "| Metric | n | Estimate | Advertised | Difference (pp) | 95% bootstrap interval | Close? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for item in metrics["comparisons"]:
        count = metrics["sample_counts"].get(item["metric"], "—")
        interval = item["ci95_percent"]
        interval_text = "—" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}]"
        lines.append(
            f"| {item['metric']} | {count} | {item['estimate_percent']:.2f} | "
            f"{item['target_percent']:.2f} | {item['difference_percentage_points']:+.2f} | "
            f"{interval_text} | {'yes' if item['close'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Inference validity",
        "",
        f"Completed rows: {metrics['completed_rows']}/600; errors: {metrics['error_n']}; token-cap hits: {metrics['cap_n']}; empty outputs by family: `{json.dumps(metrics['validity']['empty_outputs'], sort_keys=True)}`.",
        "ERE parsed edge counts by task: `" + json.dumps(metrics['validity']['ere_edge_counts'], sort_keys=True) + "`.",
        f"Official Graph Facts Event Graph conversions: {metrics['validity']['graph_facts_converted_rows']} rows; raw predictions remain unchanged in `predictions.jsonl`.",
        "",
        "## SGG fixed-panel evidence",
        "",
        ("The existing reviewed 50-image VG canonical-test result is included as a separate SGG panel, not merged into the G-Substrate HF test split. It scored PCIs R@50 22.68% and mR@50 8.06% against advertised 25.38% and 8.67%. It has 42/50 strict-valid rows, seven tuple-limit failures, one token-cap hit, and an independent ACCEPT WITH CAVEAT review. Its denominator is 410 ground-truth relations. No interval is reported for this fixed panel."
         if metrics.get("accepted_sgg_evidence_sha256") else
         "SGG is scored in a separate image panel for each candidate. This text-only report does not include an SGG score."),
        "",
        "## Provenance and limitations",
        "",
        f"Model weights SHA-256: `{metrics['model_weights_sha256']}`; chat-template SHA-256: `{metrics['chat_template_sha256']}`.",
        "Implementation source SHA-256: `" + json.dumps(metrics['implementation_sha256'], sort_keys=True) + "`.",
        f"Scorer files: `{json.dumps(metrics['scorer_sha256'], sort_keys=True)}`.",
        "GAR/MGD/ERE estimates use the official test split and fixed deterministic subsamples. They are not full-test reproductions. A sample is called close when its difference is at most five percentage points or the advertised value falls within its 95% bootstrap interval. No prompt or sample tuning was performed against the advertised values.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    implementation_sha256 = {
        "runner": launched_runner_sha256(run_manifest),
        "scorer": sha256_file(Path(__file__).resolve()),
    }
    recoverable_scoring_failure = (
        run_manifest.get("status") == "failed"
        and run_manifest.get("failure", {}).get("phase") == "scoring"
        and isinstance(run_manifest.get("predictions_sha256"), str)
    )
    if (run_manifest.get("status") not in {"inference_complete", "complete"}
            and not recoverable_scoring_failure):
        raise ValueError("only a complete inference run can be scored")
    rows = get_predictions(run_dir)
    predictions_path = run_dir / "official_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "task": row["task"], "subtask": row["subtask"],
                "prompt": row["prompt_text"], "predict": row["raw_prediction"],
                "label": row["label"], "images": row.get("images", []),
                "sample_id": row["sample_id"], "source_file": row["source_file"],
                "source_index": row["source_index"], "task_tag": row["task_tag"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    per_example, validity = per_example_metrics(rows)
    rng = random.Random(20261111)
    gar_intervals, gar_macro_interval = bootstrap_gar(per_example["gar"], rng, 2000)

    comparisons: list[dict[str, Any]] = []
    point_gar: dict[str, float] = {}
    for subtask, values in per_example["gar"].items():
        point = sum(values) / len(values)
        point_gar[subtask] = point
        comparisons.append(metric_line(
            f"GAR-{GAR_TO_LABEL[subtask]}", point,
            TARGETS[f"GAR-{GAR_TO_LABEL[subtask]}"], gar_intervals[subtask],
        ))
    macro_gar = sum(point_gar.values()) / len(point_gar)
    comparisons.append(metric_line("GAR-mean", macro_gar,
                                   sum(TARGETS[f"GAR-{label}"] for label in GAR_TO_LABEL.values()) / 4,
                                   gar_macro_interval))

    for metric, values, target in (
        ("MGD-BLEU4", per_example["mgd_bleu"], TARGETS["MGD-BLEU4"]),
        ("MGD-ROUGE-L", per_example["mgd_rouge"], TARGETS["MGD-ROUGE-L"]),
    ):
        interval = bootstrap_mean(values, rng, 2000)
        comparisons.append(metric_line(metric, sum(values) / len(values), target, interval))

    for subtask, values in per_example["ere"].items():
        interval = bootstrap_mean(values, rng, 2000)
        report_name = ERE_TO_LABEL[subtask]
        point = sum(values) / len(values)
        comparisons.append(metric_line(
            f"ERE-{report_name}", point, TARGETS[f"ERE-{report_name}"], interval
        ))

    sgg_file_hashes = {}
    sgg_provenance = None
    if run_manifest.get("model", {}).get("id") == "zmli/G-Substrate-Qwen3-VL-2B":
        for name, expected in EXPECTED_SGG_FILES.items():
            actual = sha256_file(SGG_DIR / name)
            if actual != expected:
                raise ValueError(f"accepted SGG evidence hash mismatch for {name}: {actual}")
            sgg_file_hashes[name] = actual
        sgg_manifest = json.loads((SGG_DIR / "comparison_manifest.json").read_text(encoding="utf-8"))
        sgg_entry = next((entry for entry in sgg_manifest["results"]
                          if entry.get("model_id") == "zmli/G-Substrate-Qwen3-VL-2B"), None)
        if sgg_entry is None:
            raise ValueError("accepted SGG manifest lacks the G-Substrate result")
        record_path = SGG_DIR / sgg_entry["record_path"]
        input_path = SGG_DIR / sgg_entry["official_input_path"]
        panel_path = (SGG_DIR / "../../../panels/relation50-v2-order11/vg150.json").resolve()
        record_hash = sha256_file(record_path)
        input_hash = sha256_file(input_path)
        panel_hash = sha256_file(panel_path)
        router_hash = sha256_file(OFFICIAL_INFERENCE / "evaluate.py")
        pcis_scorer_path = OFFICIAL_INFERENCE / "evaluators/sgg_eval/vg_sgg_eval.py"
        pcis_scorer_hash = sha256_file(pcis_scorer_path)
        if record_hash != EXPECTED_SGG_RECORD_SHA256 or record_hash != sgg_entry.get("record_sha256"):
            raise ValueError("accepted SGG run-record hash mismatch")
        if input_hash != EXPECTED_SGG_INPUT_SHA256 or input_hash != sgg_entry.get("official_input_sha256"):
            raise ValueError("accepted SGG official-input hash mismatch")
        if panel_hash != EXPECTED_SGG_PANEL_SHA256 or panel_hash != sgg_manifest.get("panel_manifest_sha256"):
            raise ValueError("accepted SGG panel-manifest hash mismatch")
        if router_hash != EXPECTED_SGG_ROUTER_SHA256 or router_hash != sgg_entry.get("official_router_sha256"):
            raise ValueError("accepted SGG official-router hash mismatch")
        if pcis_scorer_hash != EXPECTED_SGG_SCORER_SHA256 or pcis_scorer_hash != sgg_entry.get("official_scorer_sha256"):
            raise ValueError("accepted SGG PCIs-scorer hash mismatch")
        if sgg_entry.get("score_sha256") != EXPECTED_SGG_FILES["zmli_G-Substrate-Qwen3-VL-2B.scores.json"]:
            raise ValueError("accepted SGG official score hash is inconsistent")
        if (sgg_entry.get("model_revision") != run_manifest["model"]["revision"]
                or sgg_manifest.get("official_repository_commit") != run_manifest["official_repository_commit"]
                or sgg_entry.get("n_images") != 50 or sgg_manifest.get("full_panel_ground_truth_relations") != 410
                or sgg_entry.get("strict_valid_n") != 42 or sgg_entry.get("token_cap_n") != 1
                or sgg_entry.get("adjudication") != "ACCEPT WITH CAVEAT"):
            raise ValueError("accepted SGG manifest no longer matches the recorded protocol and adjudication")
        existing_sgg = json.loads((SGG_DIR / "zmli_G-Substrate-Qwen3-VL-2B.scores.json").read_text(encoding="utf-8"))["ssg"]["pcis"]
        for metric_key, report_name in (("R@50", "SGG-PCIs-R@50"), ("mR@50", "SGG-PCIs-mR@50")):
            comparisons.append(metric_line(report_name, existing_sgg[metric_key], TARGETS[report_name], None))
        sgg_provenance = {
            **sgg_file_hashes, "run_record": record_hash, "official_input": input_hash,
            "panel_manifest": panel_hash, "official_router": router_hash,
            "pcis_scorer": pcis_scorer_hash,
        }

    counts = {item["metric"]: 50 for item in comparisons if item["metric"].startswith("GAR-") and item["metric"] != "GAR-mean"}
    counts.update({"MGD-BLEU4": 200, "MGD-ROUGE-L": 200})
    counts.update({f"ERE-{label}": len(per_example["ere"][tag]) for tag, label in ERE_TO_LABEL.items()})
    if sgg_provenance is not None:
        counts.update({"SGG-PCIs-R@50": 50, "SGG-PCIs-mR@50": 50})

    scorer_paths = {
        "evaluate.py": OFFICIAL_INFERENCE / "evaluate.py",
        "graph_search_eval.py": OFFICIAL_INFERENCE / "evaluators/graph_search_eval.py",
        "mol_eval.py": OFFICIAL_INFERENCE / "evaluators/mol_eval.py",
        "event_graph_eval.py": OFFICIAL_INFERENCE / "evaluators/event_graph_eval.py",
    }
    metrics = {
        "model_revision": run_manifest["model"]["revision"],
        "model_weights_sha256": run_manifest["model"]["model_weights_sha256"],
        "chat_template_sha256": run_manifest["model"]["chat_template_sha256"],
        "dataset_revision": run_manifest["dataset_revision"],
        "official_repository_commit": run_manifest["official_repository_commit"],
        "decoding": {
            **run_manifest["sampling"],
            "sampling_seed": run_manifest["sampling_seed"],
            "max_new_tokens": run_manifest["max_new_tokens"],
        },
        "completed_rows": len(rows), "error_n": 0,
        "cap_n": sum(row["finish_reason"] == "length" for row in rows),
        "sample_counts": counts,
        "scoring_mode": "pinned official per-example GAR, MGD, and ERE evaluators; Event Graph Graph Facts converted using the pinned official router's conversion",
        "scoring_python": sys.executable,
        "comparisons": comparisons,
        "validity": validity,
        "bootstrap": {"repetitions": 2000, "seed": 20261111, "unit": "example within task stratum"},
        "scorer_sha256": {name: sha256_file(path) for name, path in scorer_paths.items()},
        "implementation_sha256": implementation_sha256,
        "accepted_sgg_evidence_sha256": sgg_provenance,
        "predictions_sha256": sha256_file(predictions_path),
        "panel_manifest_sha256": sha256_file(run_dir / "panel_manifest.json"),
        "panel_sha256": sha256_file(run_dir / "panel.json"),
    }
    atomic_json(run_dir / "metrics.json", metrics)
    report_path = run_dir / "report.md"
    report_path.write_text(render_report(metrics), encoding="utf-8")
    if recoverable_scoring_failure:
        run_manifest["recovered_scoring_failure"] = run_manifest.pop("failure")
    run_manifest["implementation_sha256"] = implementation_sha256
    run_manifest["status"] = "complete"
    run_manifest["metrics_sha256"] = sha256_file(run_dir / "metrics.json")
    run_manifest["report_sha256"] = sha256_file(report_path)
    atomic_json(run_dir / "run_manifest.json", run_manifest)
    print(f"METRICS {run_dir / 'metrics.json'}", flush=True)
    print(f"REPORT {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
