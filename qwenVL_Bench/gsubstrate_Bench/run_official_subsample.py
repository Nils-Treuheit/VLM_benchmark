#!/usr/bin/env python3
"""Run deterministic official G-Substrate GAR, MGD, and ERE test subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "qwenVL_Bench").is_dir() and (parent / ".agentic_workflow").is_dir()
)
LEASE_MODULE_DIR = Path.home() / ".local/share/model-unload/benchmark-lease"
if str(LEASE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(LEASE_MODULE_DIR))
from benchmark_lease import benchmark_lease  # noqa: E402

SOURCE_SGG = PROJECT_ROOT / "qwenVL_Bench" / "image_SGG_Bench"
EXPECTED_GSUB_MODEL_ID = "zmli/G-Substrate-Qwen3-VL-2B"
EXPECTED_GSUB_MODEL_REVISION = "e3c7d3c8ffd278884522739ae7bb9e05b4c18a1a"
EXPECTED_CHAT_TEMPLATE_SHA256 = "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
EXPECTED_DATASET_REVISION = "388622518ef349efd051930bdca244fde82e7d0b"
EXPECTED_OFFICIAL_COMMIT = "68d00d258e2be282abca41ef4912b323f747f31c"
EXPECTED_FIXED_PANEL_SHA256 = "cee11274848c0a345043bcc500f8362f6abcfc60d0f94d2591d88bf47da602b2"
EXPECTED_QUALIFICATION_PANEL_MANIFEST_SHA256 = "a02a43dbd2730234524eec3939be22ca5295fa6dda75349eae95bb84dc939f17"
EXPECTED_QUALIFICATION_TEXT_PANEL_SHA256 = "4541d7ed8e641ca70485180a04c0645acf022cea7a2dd446bc1a502a9d172849"
MAX_NEW_TOKENS = 4096
SAMPLE_SEED = 42
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
TOKENIZER_SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
FIXED_FAMILIES = ("GAR", "MGD", "ERE")
FIXED_PANEL = PROJECT_ROOT / ".agentic_workflow/workspace/qwenvl-four-task/benchmark-text-panel-50.json"
FIXED_PANEL_MANIFEST = PROJECT_ROOT / ".agentic_workflow/workspace/qwenvl-four-task/qualification-panel-manifest.json"
QUALIFICATION_ROOT = PROJECT_ROOT / ".agentic_workflow/workspace/qwenvl-four-task/qualification"
QUALIFICATION_RUNNER = PROJECT_ROOT / "qwenVL_Bench/gsubstrate_Bench/run_qualification.py"
RELEASE_GATE = PROJECT_ROOT / ".agentic_workflow/src/executor.py"
TAG_RE = re.compile(r"^\[([^\]]+)\]")

if str(SOURCE_SGG) not in sys.path:
    sys.path.insert(0, str(SOURCE_SGG))
import runner as legacy  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def task_tag(row: dict[str, Any], source_name: str, source_index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source_name}[{source_index}] has no ShareGPT messages list")
    users = [message for message in messages if message.get("role") == "user"]
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if len(users) != 1 or len(assistants) != 1:
        raise ValueError(
            f"{source_name}[{source_index}] must have one user message and one assistant label"
        )
    content = users[0].get("content")
    if not isinstance(content, str):
        raise ValueError(f"{source_name}[{source_index}] user content is not text")
    match = TAG_RE.match(content)
    if match is None:
        raise ValueError(f"{source_name}[{source_index}] has no leading task tag")
    if not isinstance(assistants[0].get("content"), str):
        raise ValueError(f"{source_name}[{source_index}] assistant label is not text")
    return match.group(1)


def load_source(path: Path, expected_sha256: str) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise ValueError(f"source hash mismatch for {path.name}: {actual}")
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value, actual


def make_panels(dataset_dir: Path, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset") != "zmli/G-Substrate-Data":
        raise ValueError("unexpected dataset identity")
    if source_manifest.get("revision") != EXPECTED_DATASET_REVISION:
        raise ValueError("dataset manifest revision is not the pinned test revision")
    if source_manifest.get("split") != "test":
        raise ValueError("only the official test split is allowed")

    specifications = (
        ("graph_search.json", {
            "graph_search:connectivity": ("GAR", "connectivity", 50),
            "graph_search:cycle": ("GAR", "cycle", 50),
            "graph_search:shortest_path": ("GAR", "shortest_path", 50),
            "graph_search:matching": ("GAR", "matching", 50),
        }),
        ("molecular.json", {"molecule:molecule_description": ("MGD", "molecular", 200)}),
        ("event_graph.json", {
            "event_graph:maven_ere_subevent_anchor": ("ERE", "MA-S", 60),
            "event_graph:maven_ere_temporal_anchor": ("ERE", "MA-T", 60),
            "event_graph:maven_ere_causal_anchor": ("ERE", "MA-C", 60),
            "event_graph:hievents": ("ERE", "HiE", 20),
        }),
    )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    groups_manifest: dict[str, Any] = {}
    for source_name, wanted_tags in specifications:
        source_info = source_manifest["files"].get(source_name)
        if not isinstance(source_info, dict):
            raise ValueError(f"pinned manifest lacks {source_name}")
        rows, source_hash = load_source(dataset_dir / source_name, source_info["sha256"])
        grouped: dict[str, list[int]] = {tag: [] for tag in wanted_tags}
        for index, row in enumerate(rows):
            tag = task_tag(row, source_name, index)
            if tag in grouped:
                grouped[tag].append(index)

        for tag, (family, label, count) in wanted_tags.items():
            candidates = grouped[tag]
            if len(candidates) < count:
                raise ValueError(
                    f"{source_name} tag {tag} has {len(candidates)} rows; needs {count}"
                )
            chosen = sorted(rng.sample(candidates, count))
            groups_manifest[tag] = {
                "family": family,
                "reported_subtask": label,
                "source_file": source_name,
                "source_sha256": source_hash,
                "source_count": len(rows),
                "available_tag_count": len(candidates),
                "requested_n": count,
                "source_indices": chosen,
            }
            for index in chosen:
                row = rows[index]
                user = next(m["content"] for m in row["messages"] if m["role"] == "user")
                assistant = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
                sample_id = f"{source_name}:{source_hash}:{index}"
                selected.append({
                    "sample_id": sample_id,
                    "source_file": source_name,
                    "source_sha256": source_hash,
                    "source_index": index,
                    "task_tag": tag,
                    "family": family,
                    "reported_subtask": label,
                    "task": "graph_search" if family == "GAR" else (
                        "mol" if family == "MGD" else "event_graph"
                    ),
                    "subtask": tag.split(":", 1)[1],
                    "prompt_text": user,
                    "label": assistant,
                    "images": row.get("images", []),
                    "original_messages": row["messages"],
                })

    selected.sort(key=lambda row: (row["source_file"], row["source_index"]))
    if len(selected) != 600:
        raise ValueError(f"panel has {len(selected)} rows; expected 600")
    panel_manifest = {
        "schema_version": 1,
        "dataset": "zmli/G-Substrate-Data",
        "dataset_revision": EXPECTED_DATASET_REVISION,
        "split": "test",
        "seed": seed,
        "row_count": len(selected),
        "groups": groups_manifest,
        "sample_ids_sha256": sha256_bytes(
            "\n".join(row["sample_id"] for row in selected).encode("utf-8")
        ),
    }
    return selected, panel_manifest


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def infer(args: argparse.Namespace, rows: list[dict[str, Any]], run_manifest: dict[str, Any]) -> None:
    import torch
    from transformers import AutoProcessor

    torch.manual_seed(SAMPLE_SEED)
    torch.cuda.manual_seed_all(SAMPLE_SEED)

    model_entries = {str(entry["id"]): entry for entry in legacy.model_entries()}
    entry = model_entries.get(args.model_id)
    if entry is None or entry.get("repo"):
        raise ValueError(f"exact cached Transformers model is unavailable: {args.model_id}")
    snapshot = Path(str(entry.get("snapshot") or "")).resolve()
    actual_revision = snapshot.name
    if actual_revision != args.model_revision:
        raise ValueError(
            f"cached revision mismatch: requested {args.model_revision}, found {actual_revision}"
        )
    weight_files = sorted(snapshot.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"model safetensors are missing from pinned snapshot: {snapshot}")
    weight_hashes = {path.name: sha256_file(path) for path in weight_files}
    weights_sha256 = sha256_bytes(json.dumps(weight_hashes, sort_keys=True).encode("utf-8"))

    legacy.configure_gpu_runtime()
    device = torch.cuda.current_device()
    model_cls = legacy.transformers_model_class(args.model_id)
    processor = AutoProcessor.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True
    )
    template = getattr(processor, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("pinned processor has no identifiable chat template")
    template_hash = sha256_bytes(template.encode("utf-8"))
    if args.model_id == EXPECTED_GSUB_MODEL_ID and template_hash != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError(f"advertised G-Substrate chat template hash mismatch: {template_hash}")

    model = model_cls.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        device_map={"": device}, dtype="auto", attn_implementation="sdpa",
    )
    model.eval()
    run_manifest["model"] = {
        "id": args.model_id,
        "revision": actual_revision,
        "snapshot": str(snapshot),
        "backend": "Transformers",
        "model_weights_sha256": weights_sha256,
        "model_weight_file_sha256": weight_hashes,
        "chat_template_sha256": template_hash,
        "gpu_runtime": legacy.RUNTIME_GATE,
    }
    run_manifest["status"] = "running"
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)

    output_path = args.output_dir / "predictions.jsonl"
    started_all = time.perf_counter()
    try:
        for ordinal, row in enumerate(rows, start=1):
            if row["images"]:
                raise ValueError(
                    f"text panel unexpectedly contains image references: {row['sample_id']}"
                )
            messages = [{"role": "user", "content": row["prompt_text"]}]
            request_started = time.perf_counter()
            rendered = None
            input_tokens = None
            try:
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[rendered], padding=True, return_tensors="pt"
                ).to(model.device)
                input_tokens = int(inputs.input_ids.shape[-1])
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        **SAMPLING,
                    )
                new_ids = generated[:, input_tokens:]
                output_tokens = int(new_ids.shape[-1])
                raw = processor.batch_decode(
                    new_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                finish_reason = "length" if output_tokens >= MAX_NEW_TOKENS else "stop"
                # Empty decoded text is a completed model response. Preserve
                # it as an empty denominator prediction, not an infrastructure error.
                error = None
            except Exception as exc:
                raw = None
                output_tokens = None
                finish_reason = None
                error = f"{type(exc).__name__}: {exc}"

            record = {
                "sample_id": row["sample_id"],
                "source_file": row["source_file"],
                "source_sha256": row["source_sha256"],
                "source_index": row["source_index"],
                "task_tag": row["task_tag"],
                "family": row["family"],
                "reported_subtask": row["reported_subtask"],
                "task": row["task"],
                "subtask": row["subtask"],
                "prompt_text": row["prompt_text"],
                "prompt_sha256": sha256_bytes(row["prompt_text"].encode("utf-8")),
                "rendered_input_text": rendered,
                "rendered_input_sha256": (
                    sha256_bytes(rendered.encode("utf-8")) if isinstance(rendered, str) else None
                ),
                "label": row["label"],
                "images": row["images"],
                "raw_prediction": raw,
                "predict": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "cap_hit": finish_reason == "length",
                "latency_s": time.perf_counter() - request_started,
                "error": error,
                "valid_for_aggregate": error is None,
            }
            append_jsonl(output_path, record)
            if error is not None:
                run_manifest["status"] = "failed"
                run_manifest["failure"] = {"sample_id": row["sample_id"], "error": error}
                run_manifest["completed_rows"] = ordinal
                atomic_json(args.output_dir / "run_manifest.json", run_manifest)
                raise RuntimeError(f"panel stopped at {row['sample_id']}: {error}")
            print(
                f"PROGRESS {row['family']} {row['reported_subtask']} {ordinal}/{len(rows)} "
                f"tokens={output_tokens} cap={int(record['cap_hit'])} "
                f"latency_s={record['latency_s']:.1f}",
                flush=True,
            )
    finally:
        del model, processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    run_manifest["status"] = "inference_complete"
    run_manifest["completed_rows"] = len(rows)
    run_manifest["inference_elapsed_s"] = time.perf_counter() - started_all
    run_manifest["predictions_sha256"] = sha256_file(output_path)
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)


def _safe_model_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _gate_path(model_id: str, attempt: int, family: str) -> Path:
    return QUALIFICATION_ROOT / _safe_model_id(model_id) / f"attempt-{attempt}-{family}-gate.json"


def _family_attempts(model_id: str, family: str) -> list[tuple[int, Path, dict[str, Any]]]:
    found = []
    model_dir = QUALIFICATION_ROOT / _safe_model_id(model_id)
    for attempt in range(4):
        path = _gate_path(model_id, attempt, family)
        if not path.is_file():
            aggregate = model_dir / f"attempt-{attempt}-gate.json"
            if aggregate.is_file():
                aggregate_gate = json.loads(aggregate.read_text(encoding="utf-8"))
                if family in aggregate_gate.get("task_summary", {}):
                    path = aggregate
        if path.is_file():
            found.append((attempt, path, json.loads(path.read_text(encoding="utf-8"))))
    return found


def _adjudication_for(model_id: str, model_revision: str, attempt: int, family: str,
                      gate_path: Path, gate: dict[str, Any], prediction_path: Path,
                      prediction_sha256: str, run_manifest_path: Path,
                      run_manifest_sha256: str) -> dict[str, Any] | None:
    path = prediction_path.parent / "independent-adjudication.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("model_id") != model_id
            or data.get("attempt") != attempt
            or data.get("model_revision", data.get("source_model_revision")) != model_revision
            or data.get("panel_sha256") != EXPECTED_QUALIFICATION_TEXT_PANEL_SHA256
            or data.get("run_manifest_sha256") != run_manifest_sha256
            or data.get("prediction_sha256") != prediction_sha256):
        raise ValueError(f"independent adjudication is stale or mismatched for {family}: {path}")
    recorded_gate_hash = data.get("gate_sha256")
    if recorded_gate_hash is not None and recorded_gate_hash != sha256_file(gate_path):
        raise ValueError(f"independent adjudication is not linked to the current gate: {path}")
    # Some existing reviewed sidecars predate gate_sha256. Their run and
    # prediction hashes must match exactly, and the gate must independently
    # point to the same prediction file and model/panel/attempt lineage.
    if (gate.get("model_id") != model_id or gate.get("attempt") != attempt
            or gate.get("panel_sha256") != EXPECTED_QUALIFICATION_TEXT_PANEL_SHA256
            or Path(str(gate.get("prediction_path", ""))).resolve() != prediction_path.resolve()
            or gate.get("prediction_sha256") != prediction_sha256
            or run_manifest_path.parent != prediction_path.parent):
        raise ValueError(f"gate/run lineage does not match the adjudication sidecar: {path}")
    families = data.get("families", {})
    if family in families:
        family_review = families[family]
    elif data.get("family") == family:
        family_review = data
    else:
        raise ValueError(f"independent adjudication does not contain family {family}: {path}")
    return {"path": path, "data": data, "family": family_review}


def _gate_quality(model_id: str, model_revision: str, family: str,
                  attempt: int, path: Path, gate: dict[str, Any]) -> tuple[int, bool, dict[str, Any] | None]:
    if gate.get("model_id") != model_id or gate.get("attempt") != attempt:
        raise ValueError(f"{family} qualification gate identity mismatch: {path}")
    panel_hash = gate.get("panel_sha256")
    if panel_hash != EXPECTED_QUALIFICATION_TEXT_PANEL_SHA256:
        raise ValueError(f"{family} qualification panel hash mismatch: {path}")
    recorded_revision = gate.get("source_model_revision", gate.get("model_revision"))
    if recorded_revision != model_revision:
        raise ValueError(f"{family} qualification model revision mismatch: {path}")
    prediction_path = Path(str(gate.get("prediction_path", ""))).resolve(strict=True)
    prediction_sha256 = sha256_file(prediction_path)
    if gate.get("prediction_sha256") != prediction_sha256:
        raise ValueError(f"{family} qualification prediction hash mismatch: {path}")
    run_manifest_path = prediction_path.parent / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError(f"{family} qualification run manifest is missing: {run_manifest_path}")
    run_manifest_sha256 = sha256_file(run_manifest_path)
    qual_run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (qual_run.get("status") != "complete"
            or qual_run.get("model_id") != model_id
            or qual_run.get("model_revision") != model_revision
            or qual_run.get("attempt") != attempt
            or qual_run.get("panel_sha256") != EXPECTED_QUALIFICATION_TEXT_PANEL_SHA256
            or qual_run.get("prediction_sha256") != prediction_sha256):
        raise ValueError(f"{family} qualification run manifest mismatch: {run_manifest_path}")
    runner_hash = gate.get("source_runner_sha256")
    if runner_hash is not None and runner_hash != qual_run.get("implementation_sha256", {}).get("runner"):
        raise ValueError(f"{family} gate/run implementation lineage mismatch: {path}")
    summary = gate.get("task_summary", {}).get(family, gate)
    valid_n = int(summary.get("valid_n", gate.get("valid_n", 0)))
    adjudication = _adjudication_for(
        model_id, model_revision, attempt, family, path, gate,
        prediction_path, prediction_sha256, run_manifest_path, run_manifest_sha256,
    )
    if adjudication is not None:
        family_review = adjudication["family"]
        valid_n = int(family_review.get("adjudicated_valid_n", valid_n))
        passed = bool(family_review.get("passed", family_review.get("passed_gate", valid_n >= 3)))
        verdict = str(family_review.get("verdict", adjudication["data"].get("independent_adjudication", ""))).upper()
        if verdict.startswith("REVISE") or verdict.startswith("REJECT"):
            passed = False
    else:
        passed = bool(summary.get("passed", gate.get("passed", valid_n >= 3)))
    if valid_n < 3:
        passed = False
    return valid_n, passed, adjudication


def resolve_family_gate(model_id: str, revision: str, family: str,
                        rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = _family_attempts(model_id, family)
    passing = []
    for attempt, path, gate in attempts:
        valid_n, passed, adjudication = _gate_quality(model_id, revision, family, attempt, path, gate)
        if passed:
            passing.append((attempt, path, gate, valid_n, adjudication))
    if not passing:
        raise ValueError(f"requested family {family} has no passing 10-example gate for {model_id}")
    attempt, path, gate, valid_n, adjudication = min(passing, key=lambda item: item[0])
    pred_path = Path(gate.get("prediction_path", ""))
    if not pred_path.is_file():
        raise ValueError(f"qualification prediction file is missing for {family}: {pred_path}")
    if gate.get("prediction_sha256") != sha256_file(pred_path):
        raise ValueError(f"qualification prediction hash mismatch for {family}")
    qual_run_path = pred_path.parent / "run_manifest.json"
    if not qual_run_path.is_file():
        raise ValueError(f"qualification run manifest is missing for {family}")
    qual_run = json.loads(qual_run_path.read_text(encoding="utf-8"))
    if (qual_run.get("status") != "complete"
            or qual_run.get("model_id") != model_id
            or qual_run.get("model_revision") != revision
            or qual_run.get("panel_sha256") != EXPECTED_QUALIFICATION_TEXT_PANEL_SHA256
            or qual_run.get("prediction_sha256") != sha256_file(pred_path)):
        raise ValueError(f"qualification run manifest mismatch for {family}")
    suffixes = {}
    pred_rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line]
    tag_rows: dict[str, list[dict[str, Any]]] = {}
    for row in pred_rows:
        if row.get("family") == family:
            tag_rows.setdefault(row["task_tag"], []).append(row)
    for tag in sorted({row["task_tag"] for row in rows}):
        info = qual_run.get("prompt_variants", {}).get(tag)
        family_rows = tag_rows.get(tag, [])
        if isinstance(info, dict):
            suffix = info.get("repair_instruction", "")
        else:
            # Older imported attempt-0 records predate prompt_variants. Their
            # raw qualification rows still bind the no-repair prompt and hash.
            suffixes_in_rows = {item.get("repair_instruction", "") for item in family_rows}
            if len(suffixes_in_rows) != 1 or not family_rows:
                raise ValueError(f"qualification prompt variant missing or ambiguous for {tag}")
            suffix = suffixes_in_rows.pop()
            if suffix:
                raise ValueError(f"cannot recover non-baseline prompt suffix from legacy manifest for {tag}")
        if not isinstance(suffix, str):
            raise ValueError(f"qualification prompt suffix is invalid for {tag}")
        if not family_rows:
            raise ValueError(f"no qualification examples recorded for task tag {tag}")
        for item in family_rows:
            expected = item["prompt_text"].rstrip() + "\n\nOutput requirement: " + suffix if suffix else item["prompt_text"]
            expected_hash = sha256_bytes(expected.encode("utf-8"))
            if item.get("prompt_sha256") != expected_hash:
                raise ValueError(f"qualification prompt hash mismatch for {tag}")
            if item.get("repair_instruction") != suffix:
                raise ValueError(f"qualification prompt suffix mismatch for {tag}")
        suffixes[tag] = suffix
    gate_info = {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "attempt": attempt, "valid_n": valid_n,
        "run_manifest_path": str(qual_run_path.resolve()),
        "run_manifest_sha256": sha256_file(qual_run_path),
        "prediction_path": str(pred_path.resolve()),
        "prediction_sha256": sha256_file(pred_path),
        "prompt_variants": suffixes,
    }
    if adjudication is not None:
        gate_info["independent_adjudication_path"] = str(adjudication["path"].resolve())
        gate_info["independent_adjudication_sha256"] = sha256_file(adjudication["path"])
    return gate_info


def prepare_fixed_panel(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if args.panel.resolve() != FIXED_PANEL.resolve() or sha256_file(args.panel) != EXPECTED_FIXED_PANEL_SHA256:
        raise ValueError("refusing a panel other than the frozen 150-row text panel")
    if args.panel_manifest.resolve() != FIXED_PANEL_MANIFEST.resolve() or sha256_file(args.panel_manifest) != EXPECTED_QUALIFICATION_PANEL_MANIFEST_SHA256:
        raise ValueError("qualification panel manifest hash mismatch")
    all_rows = json.loads(args.panel.read_text(encoding="utf-8"))
    manifest = json.loads(args.panel_manifest.read_text(encoding="utf-8"))
    if len(all_rows) != 150 or manifest.get("benchmark_text_panel_sha256") != EXPECTED_FIXED_PANEL_SHA256:
        raise ValueError("frozen 150-row text panel manifest mismatch")
    families = list(dict.fromkeys(args.families))
    if not families or any(family not in FIXED_FAMILIES for family in families):
        raise ValueError("request one or more supported families: GAR, MGD, ERE")
    rows = [row for row in all_rows if row.get("family") in families]
    expected_ids = [sid for tag in manifest.get("benchmark_groups", {}).values()
                    for sid in tag.get("sample_ids", [])
                    if any(row.get("sample_id") == sid and row.get("family") in families for row in all_rows)]
    if sorted(expected_ids) != sorted(row["sample_id"] for row in rows) or len(rows) != sum(50 for _ in families):
        raise ValueError("requested rows differ from the frozen panel manifest")
    if len({row.get("sample_id") for row in rows}) != len(rows):
        raise ValueError("fixed panel contains missing or duplicate sample IDs")
    source_manifest_path = args.dataset_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (source_manifest.get("dataset") != "zmli/G-Substrate-Data"
            or source_manifest.get("revision") != EXPECTED_DATASET_REVISION
            or source_manifest.get("split") != "test"):
        raise ValueError("dataset identity/revision/split mismatch")
    source_files = sorted({row["source_file"] for row in rows})
    sources = {}
    source_hashes = {}
    for filename in source_files:
        entry = source_manifest.get("files", {}).get(filename)
        source_path = args.dataset_dir / filename
        if not entry or sha256_file(source_path) != entry.get("sha256"):
            raise ValueError(f"source file hash mismatch: {filename}")
        sources[filename] = json.loads(source_path.read_text(encoding="utf-8"))
        source_hashes[filename] = entry["sha256"]
    counts_by_tag = {}
    gates = {}
    for row in rows:
        source = sources[row["source_file"]][row["source_index"]]
        messages = source.get("messages", [])
        users = [m for m in messages if m.get("role") == "user"]
        assistants = [m for m in messages if m.get("role") == "assistant"]
        if (len(users) != 1 or len(assistants) != 1
                or users[0].get("content") != row["prompt_text"]
                or assistants[0].get("content") != row["label"]
                or source.get("images", []) != row.get("images", [])
                or row.get("images", []) != []
                or row.get("source_sha256") != source_hashes[row["source_file"]]
                or row.get("sample_id") != f"{row['source_file']}:{row['source_sha256']}:{row['source_index']}"
                or task_tag(source, row["source_file"], row["source_index"]) != row.get("task_tag")):
            raise ValueError(f"source join mismatch for {row.get('sample_id')}")
        counts_by_tag[row["task_tag"]] = counts_by_tag.get(row["task_tag"], 0) + 1
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        gates[family] = resolve_family_gate(args.model_id, args.model_revision, family, family_rows)
    panel_rows = rows
    panel_manifest = {
        "schema_version": 1, "dataset": "zmli/G-Substrate-Data",
        "dataset_revision": EXPECTED_DATASET_REVISION, "split": "test",
        "seed": manifest["seed"], "parent_panel_sha256": EXPECTED_FIXED_PANEL_SHA256,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_hashes": source_hashes, "row_count": len(panel_rows),
        "family_counts": {family: sum(row["family"] == family for row in panel_rows) for family in families},
        "task_tag_counts": counts_by_tag,
        "sample_ids_sha256": sha256_bytes("\n".join(row["sample_id"] for row in panel_rows).encode("utf-8")),
    }
    return panel_rows, panel_manifest, gates


def _inference_prompt(row: dict[str, Any], suffix: str) -> str:
    if not suffix:
        return row["prompt_text"]
    return f"{row['prompt_text'].rstrip()}\n\nOutput requirement: {suffix}"


def _release_local() -> dict[str, Any]:
    result = subprocess.run([sys.executable, str(RELEASE_GATE), "release-local"],
                            cwd=PROJECT_ROOT / ".agentic_workflow", capture_output=True,
                            text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"release-local failed before GPU inference: {result.stdout[-1000:]} {result.stderr[-1000:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("release-local returned invalid JSON") from exc


def infer_fixed(args: argparse.Namespace, rows: list[dict[str, Any]], panel_manifest: dict[str, Any],
                gates: dict[str, Any], run_manifest: dict[str, Any]) -> None:
    import torch
    from transformers import AutoProcessor

    model_entries = {str(entry["id"]): entry for entry in legacy.model_entries()}
    entry = model_entries.get(args.model_id)
    if entry is None:
        raise ValueError(f"exact cached model entry unavailable: {args.model_id}")
    is_gguf = bool(entry.get("repo"))
    actual_revision = Path(str(entry.get("snapshot") or entry.get("model_path") or "")).resolve().name
    if actual_revision != args.model_revision:
        raise ValueError(f"cached revision mismatch: expected {args.model_revision}, found {actual_revision}")
    weights: dict[str, str] = {}
    projector_hash = None
    snapshot = Path(str(entry.get("snapshot") or entry.get("model_path") or "")).resolve()
    if is_gguf:
        model_path = Path(entry["model_path"]).resolve(strict=True)
        mmproj_path = Path(entry["mmproj_path"]).resolve(strict=True) if entry.get("mmproj_path") else None
        weights[model_path.name] = sha256_file(model_path)
        if mmproj_path:
            projector_hash = sha256_file(mmproj_path)
            weights[mmproj_path.name] = projector_hash
        run_manifest["model"] = {
            "id": args.model_id, "revision": actual_revision, "backend": entry.get("backend"),
            "snapshot": str(model_path.parent), "model_path": str(model_path),
            "projector_path": str(mmproj_path) if mmproj_path else None,
            "projector_sha256": projector_hash, "model_weight_file_sha256": weights,
            "model_weights_sha256": sha256_bytes(json.dumps(weights, sort_keys=True).encode()),
            "chat_template_sha256": None,
            "hf_home_snapshot_path": str(model_path.parent),
        }
    else:
        if not snapshot.is_dir():
            raise FileNotFoundError(f"cached snapshot missing: {snapshot}")
        weight_files = sorted(snapshot.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"cached weights missing in {snapshot}")
        weights = {path.name: sha256_file(path) for path in weight_files}
        run_manifest["model"] = {
            "id": args.model_id, "revision": actual_revision, "backend": "Transformers",
            "snapshot": str(snapshot), "model_weight_file_sha256": weights,
            "model_weights_sha256": sha256_bytes(json.dumps(weights, sort_keys=True).encode()),
            "projector_sha256": "not_applicable",
            "hf_home_snapshot_path": str(snapshot),
        }

    release_state = _release_local()
    run_manifest["release_gate_before_inference"] = release_state
    run_manifest["status"] = "inference_starting"
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)
    model = processor = None
    llama_detail = None
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists():
        raise FileExistsError(f"preserving existing predictions: {output_path}")
    started_all = time.perf_counter()
    try:
        if is_gguf:
            llama_detail = legacy.start_llama(entry)
            from run_qualification import generate_gguf
            run_manifest["model"]["server"] = {k: v for k, v in llama_detail.items() if k != "process"}
            run_manifest["model"]["lifecycle"] = "model-control-owned"
        else:
            processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
            template = getattr(processor, "chat_template", None)
            if not isinstance(template, str) or not template:
                raise ValueError("cached processor has no identifiable chat template")
            run_manifest["model"]["chat_template_sha256"] = sha256_bytes(template.encode("utf-8"))
            legacy.configure_gpu_runtime()
            device = torch.cuda.current_device()
            model_cls = legacy.transformers_model_class(args.model_id)
            model = model_cls.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True,
                                              device_map={"": device}, dtype="auto", attn_implementation="sdpa")
            model.eval()
        atomic_json(args.output_dir / "run_manifest.json", run_manifest)
        for ordinal, source in enumerate(rows, 1):
            suffix = gates[source["family"]]["prompt_variants"][source["task_tag"]]
            sent = _inference_prompt(source, suffix)
            started = time.perf_counter()
            raw = None
            input_tokens = output_tokens = None
            finish_reason = None
            error = None
            try:
                if is_gguf:
                    raw, input_tokens, output_tokens, finish_reason = generate_gguf(
                        llama_detail, args.model_id, sent, ordinal)
                else:
                    torch.manual_seed(42 + ordinal)
                    torch.cuda.manual_seed_all(42 + ordinal)
                    rendered = processor.apply_chat_template(
                        [{"role": "user", "content": sent}], tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[rendered], padding=True, return_tensors="pt").to(model.device)
                    input_tokens = int(inputs.input_ids.shape[-1])
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                                   **TOKENIZER_SAMPLING)
                    new_ids = generated[:, input_tokens:]
                    output_tokens = int(new_ids.shape[-1])
                    raw = processor.batch_decode(new_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0]
                    finish_reason = "length" if output_tokens >= MAX_NEW_TOKENS else "stop"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if raw is None:
                raw = ""
            out = dict(source)
            out.update({
                "inference_prompt": sent,
                "inference_prompt_sha256": sha256_bytes(sent.encode("utf-8")),
                "qualification_attempt": gates[source["family"]]["attempt"],
                "qualification_gate_path": gates[source["family"]]["path"],
                "qualification_gate_sha256": gates[source["family"]]["sha256"],
                "raw_prediction": raw, "predict": raw,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "cap_hit": finish_reason == "length",
                "empty_output": not bool(raw.strip()),
                "latency_s": round(time.perf_counter() - started, 6),
                "error": error, "valid_for_aggregate": error is None,
            })
            append_jsonl(output_path, out)
            run_manifest["completed_rows"] = ordinal
            if error:
                run_manifest["status"] = "failed"
                run_manifest["failure"] = {"sample_id": source["sample_id"], "error": error}
                atomic_json(args.output_dir / "run_manifest.json", run_manifest)
                raise RuntimeError(f"inference stopped on infrastructure failure: {source['sample_id']}: {error}")
            print(f"PROGRESS {source['family']} {source['task_tag']} {ordinal}/{len(rows)} cap={int(out['cap_hit'])} empty={int(out['empty_output'])}", flush=True)
    finally:
        cleanup_errors = []
        if model is not None:
            try:
                del model, processor
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as exc:
                cleanup_errors.append(f"CUDA cleanup: {type(exc).__name__}: {exc}")
        if llama_detail is not None:
            try:
                legacy.stop_llama(legacy.ACTIVE_LLAMA)
            except Exception as exc:
                cleanup_errors.append(f"owned GGUF stop: {type(exc).__name__}: {exc}")
        try:
            run_manifest["release_gate_after_inference"] = _release_local()
        except Exception as exc:
            cleanup_errors.append(f"release-local: {type(exc).__name__}: {exc}")
        run_manifest["lifecycle_result"] = "released" if not cleanup_errors else "cleanup_failed"
        if cleanup_errors:
            run_manifest["cleanup_errors"] = cleanup_errors
            run_manifest["status"] = "failed"
            atomic_json(args.output_dir / "run_manifest.json", run_manifest)
            raise RuntimeError("model cleanup could not be confirmed: " + "; ".join(cleanup_errors))
    run_manifest["status"] = "inference_complete"
    run_manifest["inference_elapsed_s"] = time.perf_counter() - started_all
    run_manifest["predictions_sha256"] = sha256_file(output_path)
    predictions = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    run_manifest["validity"] = {
        "row_count": len(predictions),
        "cap_n": sum(bool(row.get("cap_hit")) for row in predictions),
        "empty_n": sum(bool(row.get("empty_output")) for row in predictions),
        "infrastructure_error_n": sum(bool(row.get("error")) for row in predictions),
    }
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)


def run_fixed_panel(args: argparse.Namespace) -> None:
    official_root = PROJECT_ROOT / ".agentic_workflow/workspace/g-substrate-official"
    repo = subprocess.run(
        ["git", "-c", f"safe.directory={official_root}", "-C", str(official_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if repo != EXPECTED_OFFICIAL_COMMIT:
        raise ValueError(f"official evaluator repository revision mismatch: {repo}")
    rows, panel_manifest, gates = prepare_fixed_panel(args)
    if args.output_dir.exists():
        raise FileExistsError(f"preserving an earlier attempt at {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "panel.json", rows)
    atomic_json(args.output_dir / "panel_manifest.json", panel_manifest)
    family_status = {family: {"status": "qualified", "gate": gates[family]} for family in args.families}
    omitted = [family for family in FIXED_FAMILIES if family not in args.families]
    for family in omitted:
        attempts = _family_attempts(args.model_id, family)
        outcomes = [
            _gate_quality(args.model_id, args.model_revision, family, attempt, path, gate)
            for attempt, path, gate in attempts
        ]
        complete_attempt_sequence = [attempt for attempt, _, _ in attempts] == [0, 1, 2, 3]
        if complete_attempt_sequence and all(not passed for _, passed, _ in outcomes):
            family_status[family] = {"status": "not_run", "verdict": "incapable_after_three_repairs",
                                     "reason": "All four frozen 10-example qualification gates failed after baseline plus three prompt repairs. Each gate was validated against its run manifest and prediction artifact; independent adjudication overrides machine counts when available.",
                                     "attempts": [{"attempt": attempt,
                                                   "valid_n": outcome[0],
                                                   "passed": outcome[1],
                                                   "evidence": "independent_adjudication" if outcome[2] else "validated_machine_gate",
                                                   "adjudication_path": outcome[2]["path"].as_posix() if outcome[2] else None}
                                                  for (attempt, _, _), outcome in zip(attempts, outcomes)]}
        else:
            family_status[family] = {"status": "not_requested"}
    run_manifest = {
        "schema_version": 2, "benchmark_kind": "fixed-50-text-panel",
        "model_id": args.model_id, "requested_model_revision": args.model_revision,
        "dataset_revision": EXPECTED_DATASET_REVISION,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_manifest_sha256": panel_manifest["source_manifest_sha256"],
        "official_repository_commit": repo,
        "panel_path": str(args.panel.resolve()), "panel_source_sha256": sha256_file(args.panel),
        "panel_manifest_path": str(args.panel_manifest.resolve()),
        "qualification_panel_manifest_sha256": sha256_file(args.panel_manifest),
        "panel_sha256": sha256_file(args.output_dir / "panel.json"),
        "panel_manifest_sha256": sha256_file(args.output_dir / "panel_manifest.json"),
        "panel_rows": len(rows), "ordered_sample_ids_sha256": panel_manifest["sample_ids_sha256"],
        "family_status": family_status, "families": list(args.families),
        "qualification_gates": gates,
        "sampling": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                     "seed_base": 42, "seed_mapping": "42 + panel ordinal"},
        "max_new_tokens": MAX_NEW_TOKENS,
        "task_counts": panel_manifest["task_tag_counts"],
        "status": "prepared",
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "scorer": sha256_file(Path(__file__).with_name("score_official_subsample.py").resolve()),
            "qualification_runner": sha256_file(QUALIFICATION_RUNNER),
        },
    }
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)
    try:
        infer_fixed(args, rows, panel_manifest, gates, run_manifest)
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest.setdefault("failure", {"phase": "inference", "error": f"{type(exc).__name__}: {exc}"})
        atomic_json(args.output_dir / "run_manifest.json", run_manifest)
        raise
    print(f"INFERENCE_COMPLETE {args.output_dir}", flush=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--families", nargs="+", choices=FIXED_FAMILIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fixed_args = any((args.panel, args.panel_manifest, args.families))
    if fixed_args:
        if not (args.panel and args.panel_manifest and args.dataset_dir and args.families):
            parser.error("fixed 50-panel mode requires --panel, --panel-manifest, --dataset-dir, and --families")
        if args.image_root is not None:
            parser.error("--image-root is only used by legacy 600-row mode")
        run_fixed_panel(args)
        return
    if args.dataset_dir is None or args.image_root is None:
        parser.error("legacy 600-row mode requires --dataset-dir and --image-root")
    if args.output_dir.exists():
        raise FileExistsError(f"preserving an earlier attempt at {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    run_manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "seed": args.seed,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling": SAMPLING,
        "sampling_seed": SAMPLE_SEED,
        "status": "preparing",
        "image_root": str(args.image_root.resolve()),
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "scorer": sha256_file(Path(__file__).with_name("score_official_subsample.py").resolve()),
        },
    }
    atomic_json(args.output_dir / "run_manifest.json", run_manifest)
    phase = "preflight"
    try:
        if args.model_id == EXPECTED_GSUB_MODEL_ID and args.model_revision != EXPECTED_GSUB_MODEL_REVISION:
            raise ValueError("G-Substrate reproduction is frozen to the advertised model revision")
        if args.seed != 42:
            raise ValueError("the frozen cross-task benchmark uses sample seed 42")
        official_root = PROJECT_ROOT / ".agentic_workflow" / "workspace" / "g-substrate-official"
        repo = subprocess.run(
            ["git", "-c", f"safe.directory={official_root}", "-C", str(official_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if repo != EXPECTED_OFFICIAL_COMMIT:
            raise ValueError(f"official repository commit mismatch: {repo}")

        rows, panel_manifest = make_panels(args.dataset_dir, args.seed)
        atomic_json(args.output_dir / "panel.json", rows)
        atomic_json(args.output_dir / "panel_manifest.json", panel_manifest)
        manifest_file = args.dataset_dir / "manifest.json"
        run_manifest.update({
            "dataset_manifest_sha256": sha256_file(manifest_file),
            "dataset_revision": EXPECTED_DATASET_REVISION,
            "dataset_dir": str(args.dataset_dir.resolve()),
            "official_repository_commit": repo,
            "panel_rows": len(rows),
            "panel_sha256": sha256_file(args.output_dir / "panel.json"),
            "panel_manifest_sha256": sha256_file(args.output_dir / "panel_manifest.json"),
            "status": "prepared",
        })
        atomic_json(args.output_dir / "run_manifest.json", run_manifest)
        phase = "inference"
        infer(args, rows, run_manifest)
        phase = "scoring"
        scorer_python = Path(sys.executable)
        scorer_script = Path(__file__).with_name("score_official_subsample.py")
        subprocess.run(
            [str(scorer_python), str(scorer_script), "--run-dir", str(args.output_dir)],
            cwd=PROJECT_ROOT, check=True,
        )
        run_manifest["status"] = "complete"
        atomic_json(args.output_dir / "run_manifest.json", run_manifest)
        print(f"RESULT_DIR {args.output_dir}", flush=True)
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["failure"] = {
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(args.output_dir / "run_manifest.json", run_manifest)
        raise


def main() -> None:
    with benchmark_lease():
        _main()


if __name__ == "__main__":
    main()
