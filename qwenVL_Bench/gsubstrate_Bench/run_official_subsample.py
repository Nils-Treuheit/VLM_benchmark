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
SOURCE_SGG = PROJECT_ROOT / "qwenVL_Bench" / "image_SGG_Bench"
EXPECTED_GSUB_MODEL_ID = "zmli/G-Substrate-Qwen3-VL-2B"
EXPECTED_GSUB_MODEL_REVISION = "e3c7d3c8ffd278884522739ae7bb9e05b4c18a1a"
EXPECTED_CHAT_TEMPLATE_SHA256 = "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
EXPECTED_DATASET_REVISION = "388622518ef349efd051930bdca244fde82e7d0b"
EXPECTED_OFFICIAL_COMMIT = "68d00d258e2be282abca41ef4912b323f747f31c"
MAX_NEW_TOKENS = 4096
SAMPLE_SEED = 42
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
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
                error = None
                if not raw.strip():
                    error = "generation returned empty text"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
