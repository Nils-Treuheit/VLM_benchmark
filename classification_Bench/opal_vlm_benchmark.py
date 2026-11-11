#!/usr/bin/env python3
"""Run a grounding VLM as a zero-shot classifier on fixed OPAL test panels."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_queue import run_queued  # noqa: E402
PANEL_DEFAULT = Path(__file__).resolve().parent / "data" / "opal_test_panels.json"
LOCATE_ID = "nvidia/LocateAnything-3B"
GROUND_ID = "GroundingPI/GroundAnything-VLM"
DATASET_TASK = {
    "cub_200_2011": "bird species",
    "stanford_cars": "car make and model",
    "oxford_iiit_pet": "pet breed",
    "stanford_dogs": "dog breed",
    "oxford_flowers_102": "flower category",
}
GROUNDANYTHING_SOURCE = ROOT / ".agentic_workflow" / "workspace" / "groundanything-src"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: str) -> str:
    value = value.casefold().replace("_", " ")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def parse_class(raw: str, labels: list[str]) -> tuple[int | None, str]:
    answer = raw.strip()
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[-1].strip()
    answer = re.sub(r"<\|(?:im_end|fim_suffix|end_of_task)\|>", " ", answer)
    references = re.findall(r"<ref>(.*?)</ref>", answer, flags=re.IGNORECASE | re.DOTALL)
    if len(references) == 1:
        answer = references[0]
    elif len(references) > 1:
        return None, "multiple_reference_regions"
    answer = re.sub(r"</?(?:ref|box|point|quad|poly)\b[^>]*>", " ", answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?im)^\s*(?:final\s+)?(?:answer|class|label)\s*:\s*", "", answer).strip()
    answer_norm = normalized(answer)
    exact = [index for index, label in enumerate(labels) if normalized(label) == answer_norm]
    if len(exact) == 1:
        return exact[0], "matched"
    if len(exact) > 1:
        return None, "ambiguous_label_match"
    candidates: list[tuple[str, int]] = []
    for index, label in enumerate(labels):
        label_norm = normalized(label)
        if label_norm and f" {label_norm} " in f" {answer_norm} ":
            candidates.append((label_norm, index))
    if not candidates:
        return None, "no_label_match"
    # Keep the longest label when it only contains shorter aliases. Separate
    # class mentions indicate a list or otherwise ambiguous response.
    longest = [(name, index) for name, index in candidates
               if not any(name != other and name in other for other, _ in candidates)]
    if len(longest) != 1:
        return None, "ambiguous_label_match"
    return longest[0][1], "matched"


def prompt_for(dataset_id: str, labels: list[str]) -> str:
    label_list = "; ".join(labels)
    return (
        f"Classify the {DATASET_TASK[dataset_id]} shown in the image. "
        f"Choose exactly one label from this closed list: {label_list}. "
        "Return only the exact label text, with no explanation."
    )


def stop_groundanything_server(process: subprocess.Popen[bytes]) -> None:
    """Stop only the server process group started by this benchmark job."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def start_groundanything_server(
    source_root: Path, output_dir: Path, base_url: str, timeout_s: float,
) -> tuple[subprocess.Popen[bytes], float]:
    """Start the documented VLM SGLang service and measure API readiness."""
    source_root = source_root.resolve()
    python = source_root / ".venv-serve" / "bin" / "python"
    config = source_root / "configs" / "release" / "vlm_sglang.yaml"
    weights = source_root / "weights" / "vlm"
    if not python.is_file() or not config.is_file() or not weights.is_dir():
        raise FileNotFoundError(
            f"GroundAnything serving prerequisites missing under {source_root}: "
            f"python={python.is_file()}, config={config.is_file()}, weights={weights.is_dir()}"
        )
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=2):
            raise RuntimeError(
                f"GroundAnything endpoint already responds at {models_url}; "
                "refusing to reuse a service with unmeasured startup time"
            )
    except urllib.error.HTTPError:
        raise RuntimeError(
            f"GroundAnything endpoint already responds at {models_url}; "
            "refusing to reuse a service with unmeasured startup time"
        )
    except (urllib.error.URLError, TimeoutError, OSError):
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    server_log = output_dir / "groundanything_server.log"
    command = [
        str(python), "run.py", "serve", "--config", "configs/release/vlm_sglang.yaml",
    ]
    with server_log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    atexit.register(stop_groundanything_server, process)

    started = time.perf_counter()
    deadline = started + timeout_s
    last_error = "API is not ready"
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(
                f"GroundAnything server exited with code {process.returncode}; log tail:\n{tail}"
            )
        try:
            with urllib.request.urlopen(models_url, timeout=3) as response:
                inventory = json.loads(response.read())
            if not isinstance(inventory, dict) or not inventory:
                raise ValueError(f"unexpected empty model inventory: {inventory!r}")
            return process, time.perf_counter() - started
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
    raise TimeoutError(
        f"GroundAnything API did not become ready within {timeout_s:.0f}s: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("locateanything", "groundanything"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_TASK), default=list(DATASET_TASK))
    parser.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groundanything-base-url", default="http://127.0.0.1:8102/v1")
    parser.add_argument("--groundanything-model-load-duration-s", type=float,
                        help="Measured server startup duration from process start through model API readiness")
    parser.add_argument("--groundanything-autostart", action="store_true",
                        help="Start and stop the documented local GroundAnything VLM server for this run")
    parser.add_argument("--groundanything-source-root", type=Path, default=GROUNDANYTHING_SOURCE)
    parser.add_argument("--groundanything-startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--transformers-compat-overlay", type=Path,
                        help="Optional import overlay for the pinned LocateAnything Transformers runtime")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.groundanything_startup_timeout_s <= 0 or not math.isfinite(args.groundanything_startup_timeout_s):
        raise ValueError("--groundanything-startup-timeout-s must be finite and positive")
    if args.groundanything_autostart and args.model != "groundanything":
        raise ValueError("--groundanything-autostart requires --model groundanything")
    if args.model == "groundanything" and args.groundanything_autostart:
        if args.groundanything_model_load_duration_s is not None:
            raise ValueError("do not supply a load duration when autostarting GroundAnything")
    elif args.model == "groundanything" and args.groundanything_model_load_duration_s is None:
        raise ValueError("GroundAnything runs require a measured server startup duration")
    if (args.groundanything_model_load_duration_s is not None
            and (not math.isfinite(args.groundanything_model_load_duration_s)
                 or args.groundanything_model_load_duration_s < 0)):
        raise ValueError("--groundanything-model-load-duration-s must be finite and nonnegative")

    if args.transformers_compat_overlay is not None:
        overlay = args.transformers_compat_overlay.resolve()
        if not (overlay / "transformers" / "__init__.py").is_file():
            raise FileNotFoundError(f"Transformers compatibility overlay not found: {overlay}")
        sys.path.insert(0, str(overlay))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(ROOT / "grounding_Bench"))
    import model_adapters as adapters

    if args.model == "groundanything" and args.groundanything_autostart:
        from gpu_guard import release_and_check_local_gpu
        release_and_check_local_gpu()
        server, duration = start_groundanything_server(
            args.groundanything_source_root,
            args.output_dir,
            args.groundanything_base_url,
            args.groundanything_startup_timeout_s,
        )
        args.groundanything_model_load_duration_s = duration
        print(
            f"GroundAnything API ready after {duration:.3f}s (pid {server.pid}); "
            "startup is excluded from per-query latency",
            flush=True,
        )

    panel_path = args.panel.resolve()
    manifest = json.loads(panel_path.read_text(encoding="utf-8"))
    panels = {key: manifest["datasets"][key] for key in args.datasets}
    labels_by_dataset = {
        key: [row["prompt_label"] for row in panel["examples"][:panel["class_count"]]]
        for key, panel in panels.items()
    }
    # Some class names have multiple sampled images. Derive class order from the
    # manifest's index list, not the repeated image rows.
    for key, panel in panels.items():
        label_by_index = {}
        for row in panel["examples"]:
            label_by_index[int(row["label"])] = row["prompt_label"]
        if set(label_by_index) != set(range(panel["class_count"])):
            raise ValueError(f"{key}: panel does not cover every class index")
        labels_by_dataset[key] = [label_by_index[index] for index in range(panel["class_count"])]
        if len({normalized(label) for label in labels_by_dataset[key]}) != panel["class_count"]:
            raise ValueError(f"{key}: prompt labels are not unique")

    # LocateAnything loads a local model here. GroundAnything uses a separately
    # launched server whose GPU release preflight must occur before startup.
    if args.model == "locateanything":
        from gpu_guard import release_and_check_local_gpu
        release_and_check_local_gpu()
        if args.transformers_compat_overlay is not None:
            import transformers
            if transformers.__version__ != "4.57.1":
                raise RuntimeError(
                    "LocateAnything compatibility overlay must provide Transformers 4.57.1; "
                    f"found {transformers.__version__}"
                )
        snapshot, revision = adapters.snapshot_for(LOCATE_ID)
        load_started = time.perf_counter()
        adapter = adapters.LocateAnythingAdapter(snapshot)
        adapter.torch.cuda.synchronize(adapter.device)
        load_duration = time.perf_counter() - load_started
        model_id = LOCATE_ID
        load_note = "in-process model initialization; query and processor time excluded"
    else:
        # GroundAnything's service already owns the GPU. Its caller must pass
        # the measured startup duration; release-local ran before that service
        # started, and model_inventory below verifies the live endpoint.
        adapter = adapters.GroundAnythingAdapter(args.groundanything_base_url)
        inventory = adapter.model_inventory()
        revision = inventory
        model_id = GROUND_ID
        load_duration = args.groundanything_model_load_duration_s
        load_note = (
            "measured GroundAnything server process start through /models readiness; query timing excludes load"
            if load_duration is not None else
            "service was loaded before this client run; duration unavailable"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_digest = sha256_file(panel_path)
    # One warm-up query after loading. It is logged separately and excluded
    # from the per-image latency distribution.
    warmup_dataset = args.datasets[0]
    warmup_row = panels[warmup_dataset]["examples"][0]
    warmup_started = time.perf_counter()
    adapter.generate(
        Path(warmup_row["image"]), prompt_for(warmup_dataset, labels_by_dataset[warmup_dataset]), 64,
    )
    warmup_duration = time.perf_counter() - warmup_started

    for dataset_id, panel in panels.items():
        output_path = args.output_dir / f"{args.model}--{dataset_id}.json"
        saved: dict[str, dict[str, Any]] = {}
        if args.resume and output_path.is_file():
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            if previous.get("panel_sha256") != panel_digest or previous.get("model_id") != model_id:
                raise ValueError(f"cannot resume {output_path}: model or panel identity differs")
            saved = {row["id"]: row for row in previous.get("items", [])}
        prompt = prompt_for(dataset_id, labels_by_dataset[dataset_id])
        for row in panel["examples"]:
            if row["id"] in saved and saved[row["id"]].get("status") == "complete":
                continue
            started = time.perf_counter()
            try:
                response = adapter.generate(Path(row["image"]), prompt, 64)
                raw = str(response.get("raw_output", ""))
                prediction, parse_status = parse_class(raw, labels_by_dataset[dataset_id])
                elapsed = time.perf_counter() - started
                saved[row["id"]] = {
                    "id": row["id"],
                    "status": "complete",
                    "raw_output": raw,
                    "label": int(row["label"]),
                    "prediction": prediction,
                    "correct": prediction == int(row["label"]),
                    "parse_status": parse_status,
                    "latency_s": elapsed,
                    "generation_latency_s": response.get("generation_latency_s"),
                    "output_tokens": response.get("output_tokens"),
                }
            except Exception as exc:
                elapsed = time.perf_counter() - started
                saved[row["id"]] = {
                    "id": row["id"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_s": elapsed,
                }

            items = [saved.get(row_["id"]) for row_ in panel["examples"] if row_["id"] in saved]
            valid = [item for item in items if item.get("status") == "complete"]
            latencies = [float(item["latency_s"]) for item in valid]
            correct = sum(bool(item.get("correct")) for item in valid)
            output = {
                "model_id": model_id,
                "model_revision": revision,
                "dataset": panel["dataset"],
                "dataset_id": dataset_id,
                "split": panel["split"],
                "panel_sha256": panel_digest,
                "panel_seed": panel["seed"],
                "panel_count": len(panel["examples"]),
                "class_count": panel["class_count"],
                "model_load_duration_s": load_duration,
                "model_load_timing": load_note,
                "warmup_duration_s": warmup_duration,
                "query_timing_scope": "image read through model response and class-label parsing; model load and warm-up excluded",
                "summary": {
                    "attempted": len(items),
                    "completed": len(valid),
                    "failed": sum(item.get("status") != "complete" for item in items),
                    "parse_valid": sum(item.get("parse_status") == "matched" for item in valid),
                    "accuracy_top1": correct / len(panel["examples"]),
                    "accuracy_on_valid": correct / len(valid) if valid else None,
                    "mean_latency_s": statistics.mean(latencies) if latencies else None,
                    "median_latency_s": statistics.median(latencies) if latencies else None,
                    "p95_latency_s": sorted(latencies)[max(0, int(0.95 * len(latencies) + 0.999999) - 1)] if latencies else None,
                    "max_latency_s": max(latencies) if latencies else None,
                    "images_per_s": len(latencies) / sum(latencies) if latencies else None,
                },
                "items": items,
            }
            temporary = output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(output_path)
            print(f"{panel['dataset']}: {len(items)}/{len(panel['examples'])} items recorded")

    print(f"Results written under {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_queued(main, "opal-vlm-classification"))
