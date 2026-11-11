#!/usr/bin/env python3
"""Run one fixed 10-example task-family qualification attempt on one cached model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "qwenVL_Bench").is_dir() and (parent / ".agentic_workflow").is_dir()
)
SGG_SOURCE = PROJECT_ROOT / "qwenVL_Bench" / "image_SGG_Bench"
if str(SGG_SOURCE) not in sys.path:
    sys.path.insert(0, str(SGG_SOURCE))
import runner as legacy  # noqa: E402


MAX_NEW_TOKENS = 4096
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 42}
TOKENIZER_SAMPLING = {
    "do_sample": True,
    **{key: value for key, value in SAMPLING.items() if key != "seed"},
}
DATA_DIR = PROJECT_ROOT / ".agentic_workflow" / "workspace" / "g-substrate-cross-task" / "data" / "test"
OFFICIAL_COMMIT = "68d00d258e2be282abca41ef4912b323f747f31c"
QUALIFICATION_ROOT = PROJECT_ROOT / ".agentic_workflow" / "workspace" / "qwenvl-four-task" / "qualification"
TEXT_PANEL = PROJECT_ROOT / ".agentic_workflow" / "workspace" / "qwenvl-four-task" / "qualification-text-panel.json"
TEXT_PANEL_SHA256 = "4541d7ed8e641ca70485180a04c0645acf022cea7a2dd446bc1a502a9d172849"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_source_hashes() -> dict[str, str]:
    official = PROJECT_ROOT / ".agentic_workflow" / "workspace" / "g-substrate-official" / "inference"
    paths = {
        "router": official / "evaluate.py",
        "graph_search": official / "evaluators" / "graph_search_eval.py",
        "molecular": official / "evaluators" / "mol_eval.py",
        "event_graph": official / "evaluators" / "event_graph_eval.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def repair_instruction(row: dict[str, Any], attempt: int) -> str:
    if attempt == 0:
        return ""
    family = row["family"]
    subtask = row["subtask"]
    if family == "GAR":
        answer_shape = {
            "connectivity": "exactly Yes. or No.",
            "cycle": "exactly Yes. or No.",
            "shortest_path": "only the path as node IDs joined by ->, such as 2->5->9.",
            "matching": "only the maximum matching size as a nonnegative integer.",
        }[subtask]
        variants = {
            1: f"Return only the final answer in this exact format: {answer_shape} Do not explain.",
            2: f"Output format is strict: {answer_shape} Put no reasoning, labels, or Markdown in the answer.",
            3: f"Final response must contain only {answer_shape} Compute the answer from the graph above; do not include steps.",
        }
        return variants[attempt]
    if family == "ERE":
        relation_scope = {
            "maven_ere_temporal_anchor": "the temporal relations requested by this task",
            "maven_ere_causal_anchor": "the causal relations requested by this task",
            "maven_ere_subevent_anchor": "the subevent relations requested by this task",
            "hievents": "the SuperSub and SubSuper event hierarchy relations requested by this task",
        }[subtask]
        variants = {
            1: (f"Return only {relation_scope}, one tuple per line, in the form "
                "(E1:trigger, RELATION, E2:trigger). Return None only if no requested relation exists. "
                "Do not use Markdown or explain."),
            2: (f"Output every requested {relation_scope} involving the specified anchor. "
                "Use exactly (E1:trigger, RELATION, E2:trigger) per line; output None only when there are no such relations."),
            3: (f"Final answer only: list {relation_scope} that involve the anchor event. "
                "Each line must be (E<number>:trigger, RELATION, E<number>:trigger). "
                "Use None only when the list is empty."),
        }
        return variants[attempt]
    if family == "MGD":
        return {
            1: "Write a direct plain-language molecular description. Do not repeat the prompt or refuse.",
            2: "Describe the molecule in one concise paragraph using only properties supported by its structure.",
            3: "Return the molecular description as plain text, with no code fence, preamble, or refusal.",
        }[attempt]
    raise ValueError(f"unsupported family {family}")


def prepare_messages(row: dict[str, Any], attempt: int) -> tuple[str, str]:
    suffix = repair_instruction(row, attempt)
    prompt = row["prompt_text"]
    if suffix:
        prompt = f"{prompt.rstrip()}\n\nOutput requirement: {suffix}"
    return prompt, suffix


def load_transformers(entry: dict[str, Any], model_id: str, revision: str):
    import torch
    from transformers import AutoProcessor

    snapshot = Path(str(entry.get("snapshot") or "")).resolve()
    if snapshot.name != revision:
        raise ValueError(f"cached revision mismatch: expected {revision}, found {snapshot.name}")
    weight_files = sorted(snapshot.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors found in {snapshot}")
    weight_hashes = {path.name: sha256_file(path) for path in weight_files}
    weight_hash = sha256_bytes(json.dumps(weight_hashes, sort_keys=True).encode("utf-8"))
    legacy.configure_gpu_runtime()
    device = torch.cuda.current_device()
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    template = getattr(processor, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("cached processor has no identifiable chat template")
    model_cls = legacy.transformers_model_class(model_id)
    model = model_cls.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        device_map={"": device}, dtype="auto", attn_implementation="sdpa",
    )
    model.eval()
    return model, processor, {
        "id": model_id,
        "revision": revision,
        "snapshot": str(snapshot),
        "backend": "Transformers",
        "weights_sha256": weight_hash,
        "weight_file_sha256": weight_hashes,
        "chat_template_sha256": sha256_bytes(template.encode("utf-8")),
        "gpu_runtime": legacy.RUNTIME_GATE,
    }


def generate_transformers(model: Any, processor: Any, prompt: str, ordinal: int) -> tuple[str, int, int, str]:
    import torch

    torch.manual_seed(42 + ordinal)
    torch.cuda.manual_seed_all(42 + ordinal)
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[rendered], padding=True, return_tensors="pt").to(model.device)
    input_tokens = int(inputs.input_ids.shape[-1])
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **TOKENIZER_SAMPLING)
    new_ids = generated[:, input_tokens:]
    output_tokens = int(new_ids.shape[-1])
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    finish = "length" if output_tokens >= MAX_NEW_TOKENS else "stop"
    return raw, input_tokens, output_tokens, finish


def generate_gguf(detail: dict[str, Any], model_id: str, prompt: str, ordinal: int) -> tuple[str, int | None, int | None, str]:
    state = legacy.model_control_request("/status")
    server = state.get("llama_server", {})
    if not server.get("running") or server.get("model_path") is None:
        raise RuntimeError("managed llama.cpp server is not active after owned start")
    port = int(detail.get("port") or server.get("port") or 0)
    if port <= 0:
        raise RuntimeError("managed llama.cpp route did not expose its port")
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        **{**SAMPLING, "seed": 42 + ordinal},
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    usage = payload.get("usage", {})
    finish = "length" if choice.get("finish_reason") == "length" else "stop"
    return choice["message"]["content"], usage.get("prompt_tokens"), usage.get("completion_tokens"), finish


def run(args: argparse.Namespace) -> Path:
    if not 0 <= args.attempt <= 3:
        raise ValueError("attempt must be baseline 0 or one of three repairs")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if args.panel.resolve() != TEXT_PANEL.resolve() or sha256_file(args.panel) != TEXT_PANEL_SHA256:
        raise ValueError("refusing a panel other than the frozen qualification-text-panel.json")
    if len(panel) != 30:
        raise ValueError(f"qualification text panel must contain 30 rows, found {len(panel)}")
    selected = [row for row in panel if row["family"] in args.families]
    if not selected:
        raise ValueError("no rows selected for requested task families")
    if any(sum(row["family"] == family for row in selected) != 10 for family in args.families):
        raise ValueError("each selected family must contain exactly ten fixed qualification rows")
    frozen = json.loads((TEXT_PANEL.parent / "qualification-panel-manifest.json").read_text(encoding="utf-8"))
    for family in args.families:
        expected = sorted(
            sample_id
            for group, values in frozen["groups"].items()
            if ((family == "GAR" and group.startswith("graph_search:"))
                or (family == "ERE" and group.startswith("event_graph:"))
                or (family == "MGD" and group.startswith("molecule:")))
            for sample_id in values["sample_ids"]
        )
        actual = sorted(row["sample_id"] for row in selected if row["family"] == family)
        if actual != expected:
            raise ValueError(f"{family} rows do not match the frozen qualification manifest")
    if args.attempt > 0:
        model_dir = QUALIFICATION_ROOT / re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_id)
        for family in args.families:
            prior_path = model_dir / f"attempt-{args.attempt - 1}-{family}-gate.json"
            if not prior_path.is_file():
                prior_path = model_dir / f"attempt-{args.attempt - 1}-gate.json"
            if not prior_path.is_file():
                raise ValueError(f"{family} repair requires its prior gate record")
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if prior.get("model_id") != args.model_id or prior.get("panel_sha256") != TEXT_PANEL_SHA256:
                raise ValueError("prior gate record model or frozen panel does not match")
            if prior.get("attempt") != args.attempt - 1:
                raise ValueError("repair attempts must be consecutive")
            if prior.get("source_model_revision") != args.model_revision:
                raise ValueError("repair attempt model revision differs from the baseline revision")
            prior_summary = prior.get("task_summary", {})
            result = prior_summary.get(family, {})
            if result.get("n") != 10 or result.get("passed") is not False:
                raise ValueError(f"{family} repair requires exactly ten prior failed qualification rows")
    if args.output_dir.exists():
        raise FileExistsError(f"preserving an earlier attempt at {args.output_dir}")
    model_dir = QUALIFICATION_ROOT / re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_id)
    aggregate_gate = model_dir / f"attempt-{args.attempt}-gate.json"
    for family in args.families:
        family_gate = model_dir / f"attempt-{args.attempt}-{family}-gate.json"
        if family_gate.exists():
            raise FileExistsError(f"{family} attempt {args.attempt} already has a gate record")
        if aggregate_gate.is_file():
            prior_attempt = json.loads(aggregate_gate.read_text(encoding="utf-8"))
            if family in prior_attempt.get("task_summary", {}):
                raise FileExistsError(f"{family} attempt {args.attempt} already has a gate record")
    source_manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("revision") != "388622518ef349efd051930bdca244fde82e7d0b":
        raise ValueError("pinned G-Substrate test dataset revision mismatch")

    model_entries = {str(entry["id"]): entry for entry in legacy.model_entries()}
    entry = model_entries.get(args.model_id)
    if entry is None:
        raise ValueError(f"exact cached model entry is unavailable: {args.model_id}")
    actual_revision = Path(str(entry.get("snapshot") or entry.get("model_path", ""))).name
    if actual_revision != args.model_revision:
        raise ValueError(f"cached model revision mismatch: expected {args.model_revision}, found {actual_revision}")

    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "model_revision": actual_revision,
        "attempt": args.attempt,
        "attempt_kind": "baseline" if args.attempt == 0 else "prompt_repair",
        "families": args.families,
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "panel_n": len(selected),
        "dataset_revision": source_manifest["revision"],
        "dataset_manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "official_repository_commit": OFFICIAL_COMMIT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling": {**SAMPLING, "per_sample_seed": "42 + ordinal"},
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "validator": sha256_file(Path(__file__).with_name("qualify_outputs.py").resolve()),
        },
        "official_evaluator_sha256": official_source_hashes(),
        "prompt_variants": {},
        "status": "starting",
        "model": None,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    atomic_json(manifest_path, manifest)

    transformers = not bool(entry.get("repo"))
    model = processor = None
    llama_detail = None
    row_errors: list[dict[str, str]] = []
    output_path = args.output_dir / "predictions.jsonl"
    try:
        workflow_checkout = PROJECT_ROOT / ".agentic_workflow"
        release = subprocess.run(
            [sys.executable, "src/executor.py", "release-local"],
            cwd=workflow_checkout, text=True, capture_output=True, check=False,
        )
        manifest["release_gate_stdout"] = release.stdout[-2000:]
        manifest["release_gate_stderr"] = release.stderr[-2000:]
        if release.returncode != 0:
            raise RuntimeError(
                f"release-local failed before GPU use: {release.stdout[-1000:]} {release.stderr[-1000:]}"
            )
        official_root = PROJECT_ROOT / ".agentic_workflow" / "workspace" / "g-substrate-official"
        repo_revision = subprocess.run(
            ["git", "-c", f"safe.directory={official_root}", "-C", str(official_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if repo_revision != OFFICIAL_COMMIT:
            raise ValueError(f"official repository revision mismatch: {repo_revision}")
        manifest["official_repository_commit"] = repo_revision
        atomic_json(manifest_path, manifest)
        if transformers:
            model, processor, model_record = load_transformers(entry, args.model_id, args.model_revision)
        else:
            llama_detail = legacy.start_llama(entry)
            model_record = {
                "id": args.model_id, "revision": actual_revision,
                "backend": entry.get("backend"),
                "model_path": entry["model_path"], "mmproj_path": entry["mmproj_path"],
                "model_sha256": sha256_file(Path(entry["model_path"])),
                "mmproj_sha256": sha256_file(Path(entry["mmproj_path"])),
                "server": {k: v for k, v in llama_detail.items() if k != "process"},
                "lifecycle": "model-control-owned",
            }
        manifest["model"] = model_record
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)

        for ordinal, row in enumerate(selected, start=1):
            prompt, suffix = prepare_messages(row, args.attempt)
            prompt_hash = sha256_bytes(prompt.encode("utf-8"))
            manifest["prompt_variants"][row["task_tag"]] = {
                "sha256": prompt_hash,
                "repair_instruction": suffix,
            }
            start = time.perf_counter()
            error = None
            raw = None
            input_tokens = output_tokens = None
            finish_reason = None
            try:
                if transformers:
                    raw, input_tokens, output_tokens, finish_reason = generate_transformers(
                        model, processor, prompt, ordinal
                    )
                else:
                    raw, input_tokens, output_tokens, finish_reason = generate_gguf(
                        llama_detail, args.model_id, prompt, ordinal
                    )
                if not isinstance(raw, str) or not raw.strip():
                    error = "generation returned empty text"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                row_errors.append({"sample_id": row["sample_id"], "error": error})
            output = {
                "sample_id": row["sample_id"], "source_file": row["source_file"],
                "source_sha256": row["source_sha256"], "source_index": row["source_index"],
                "task_tag": row["task_tag"], "family": row["family"],
                "reported_subtask": row["reported_subtask"], "subtask": row["subtask"],
                "prompt_text": prompt, "prompt_sha256": prompt_hash,
                "repair_instruction": suffix, "label": row["label"],
                "raw_prediction": raw, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "finish_reason": finish_reason,
                "cap_hit": finish_reason == "length",
                "latency_s": time.perf_counter() - start, "error": error,
            }
            append_jsonl(output_path, output)
            manifest["completed_rows"] = ordinal
            manifest["errors"] = row_errors
            atomic_json(manifest_path, manifest)
            print(
                f"QUALIFY {args.model_id} attempt={args.attempt} {row['family']} "
                f"{row['reported_subtask']} {ordinal}/{len(selected)} "
                f"tokens={output_tokens} cap={int(output['cap_hit'])} error={int(error is not None)}",
                flush=True,
            )
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["errors"] = row_errors
        atomic_json(manifest_path, manifest)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["errors"] = row_errors
        atomic_json(manifest_path, manifest)
        raise
    finally:
        if model is not None:
            del model, processor
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if llama_detail is not None:
            legacy.stop_llama(legacy.ACTIVE_LLAMA)

    manifest["status"] = "complete" if not row_errors else "complete_with_errors"
    manifest["completed_rows"] = len(selected)
    manifest["prediction_sha256"] = sha256_file(output_path)
    atomic_json(manifest_path, manifest)
    return args.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--attempt", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--families", nargs="+", choices=("GAR", "MGD", "ERE"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.families = list(dict.fromkeys(args.families))
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}; cleaning up the owned model")

    signal.signal(signal.SIGTERM, interrupt)
    result = run(args)
    print(f"RESULT_DIR {result}", flush=True)


if __name__ == "__main__":
    main()
