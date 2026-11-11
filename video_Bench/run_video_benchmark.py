#!/usr/bin/env python3
"""Run one model at a time over its assigned native-video benchmark tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scoring import INTERVAL_PARSER_VERSION, parse_choice, parse_intervals

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from gpu_queue import run_queued  # noqa: E402
ROBOTICS_SRC = Path("/mnt/HDD1/Project_Code/robotics_VLM/src")
sys.path.insert(0, str(ROBOTICS_SRC))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CACHE_ROOT = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub")
DATA_ROOT = PROJECT / "data/video_bench"
PREPARED_ROOT = DATA_ROOT / "prepared"
RESULT_ROOT = DATA_ROOT / "results/raw"
POLICY_PATH = PREPARED_ROOT / "video_qa_sampling_policy.json"
MODEL_TRACKS = {
    "internvideo3_8b_instruct": ("mvbench", "vue_tr"),
    "timelens2_8b": ("vue_tr",),
    "llava_next_video_7b": ("mvbench", "vue_tr"),
    "videochat3_4b": ("mvbench", "vue_tr"),
}
MODEL_NAMES = {
    "internvideo3_8b_instruct": "InternVideo3-8B-Instruct",
    "timelens2_8b": "TimeLens2-8B",
    "llava_next_video_7b": "LLaVA-NeXT-Video-7B",
    "videochat3_4b": "VideoChat3-4B",
}


def build_adapter(model: str):
    if model == "internvideo3_8b_instruct":
        from robotics_vlm.models.adapters.internvideo3 import InternVideo3Adapter
        return InternVideo3Adapter(max_frames=16)
    if model == "timelens2_8b":
        from robotics_vlm.models.adapters.timelens2 import TimeLens2Adapter
        return TimeLens2Adapter("timelens2_8b", cache_root=CACHE_ROOT)
    if model == "llava_next_video_7b":
        from robotics_vlm.models.adapters.llava_video import LlavaVideoAdapter
        return LlavaVideoAdapter("llava_next_video_7b", cache_root=CACHE_ROOT)
    from robotics_vlm.models.adapters.videochat3 import VideoChat3Adapter
    return VideoChat3Adapter("videochat3_4b", cache_root=CACHE_ROOT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_item_key(
    dataset: str, item: dict[str, Any], media_rel: str, asset_sha256: str,
    policy_id: str, policy_sha256: str,
) -> str:
    payload = json.dumps(
        {"dataset": dataset, "item_id": str(item.get("query_id", item.get("source_index"))),
         "video_id": str(item.get("video_id") or Path(item["video"]).stem),
         "video_path": media_rel, "asset_sha256": asset_sha256, "policy_id": policy_id,
         "policy_sha256": policy_sha256},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_video_frames(dataset: str, item: dict[str, Any]):
    from robotics_vlm.contracts.common import ClockDomain
    from robotics_vlm.frames import Frame
    item_id = str(item["video_id"] if item.get("video_id") else Path(item["video"]).stem)
    cache_file = PREPARED_ROOT / "frame_cache" / dataset / f"{item_id}.npz"
    metadata = json.loads(cache_file.with_suffix(".json").read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy_sha256 = sha256_file(POLICY_PATH)
    media_rel = str(item["prepared_path"] if dataset == "mvbench" else item["video_path"])
    media_file = PROJECT / media_rel
    expected_size = int(item.get("asset_size_bytes", item.get("asset_bytes", -1)))
    expected_hash = str(item.get("asset_sha256", ""))
    if not media_file.is_file() or expected_size < 0 or not expected_hash:
        raise ValueError(f"active manifest has no verified local asset for {dataset}/{item_id}")
    if media_file.stat().st_size != expected_size or sha256_file(media_file) != expected_hash:
        raise ValueError(f"active source asset size/hash mismatch for {dataset}/{item_id}")
    expected_cache_rel = str(cache_file.relative_to(PROJECT))
    expected_key = cache_item_key(
        dataset, item, media_rel, expected_hash, str(policy["policy_id"]), policy_sha256
    )
    identity = {
        "dataset": dataset,
        "video_id": item_id,
        "video_path": media_rel,
        "cache_file": expected_cache_rel,
        "source_sha256": expected_hash,
        "sampling_policy_id": policy["policy_id"],
        "sampling_policy_sha256": policy_sha256,
        "manifest_cache_key": expected_key,
    }
    mismatches = [key for key, value in identity.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f"frame cache identity mismatch for {dataset}/{item_id}: {', '.join(mismatches)}")
    if sha256_file(cache_file) != metadata.get("frame_cache_sha256"):
        raise ValueError(f"frame cache content hash mismatch for {dataset}/{item_id}")
    with np.load(cache_file) as data:
        images = data["frames"]
    pts_values = metadata.get("selected_source_frame_pts", [])
    pts_time_bases = metadata.get("selected_source_frame_time_bases", [])
    timestamp_values = metadata.get("selected_source_timestamps_seconds", [])
    frame_count = int(metadata.get("selected_frame_count", 0))
    if (images.ndim != 4 or images.shape[1] <= 0 or images.shape[2] <= 0
            or images.shape[-1] != 3 or images.dtype != np.uint8
            or not 2 <= frame_count <= int(policy["max_frames"])
            or len(images) != frame_count or len(pts_values) != frame_count
            or len(pts_time_bases) != frame_count or len(timestamp_values) != frame_count):
        raise ValueError(f"frame cache dimensions/counts violate the shared policy for {dataset}/{item_id}")
    if (any(int(b) <= int(a) for a, b in zip(pts_values, pts_values[1:]))
            or any(not np.isfinite(float(t)) for t in timestamp_values)
            or any(float(b) <= float(a) for a, b in zip(timestamp_values, timestamp_values[1:]))):
        raise ValueError(f"frame cache source timestamps are invalid for {dataset}/{item_id}")
    if not metadata.get("source_time_base"):
        raise ValueError(f"frame cache is missing source time-base metadata for {dataset}/{item_id}")
    if Fraction(metadata["source_time_base"]) <= 0 or any(Fraction(base) <= 0 for base in pts_time_bases):
        raise ValueError(f"frame cache has an invalid source time base for {dataset}/{item_id}")
    expected_seconds = [int(pts) * float(Fraction(base)) for pts, base in zip(pts_values, pts_time_bases, strict=True)]
    if any(abs(float(actual) - expected) > 1e-9 for actual, expected in zip(timestamp_values, expected_seconds, strict=True)):
        raise ValueError(f"frame cache PTS/time-base mapping mismatch for {dataset}/{item_id}")
    if (metadata.get("source_resolution", {}).get("width") != images.shape[2]
            or metadata.get("source_resolution", {}).get("height") != images.shape[1]):
        raise ValueError(f"frame cache resolution metadata mismatch for {dataset}/{item_id}")
    frames = []
    receipt = time.monotonic_ns()
    for sequence, (image, seconds, source_pts) in enumerate(zip(images, timestamp_values, pts_values, strict=True)):
        frames.append(Frame(
            pixels_rgb=image,
            uid=f"{dataset}-{item_id}-{source_pts}",
            capture_time_ns=round(seconds * 1_000_000_000),
            receive_monotonic_ns=receipt + sequence,
            clock_domain=ClockDomain.MEDIA_PTS,
            source_id=item_id,
            source_frame_id=str(source_pts),
            receive_sequence=sequence,
        ))
    return tuple(frames), metadata


def request_for(model: str, prompt: str, frames, max_new_tokens: int):
    from robotics_vlm.contracts.worker import GenerationRequest
    from robotics_vlm.models.capabilities import QueryMode
    generation = GenerationRequest(max_new_tokens=max_new_tokens, temperature=0.0)
    if model == "internvideo3_8b_instruct":
        from robotics_vlm.models.adapters.internvideo3 import InternVideo3Request
        return InternVideo3Request(text=prompt, frames=frames, generation=generation)
    if model == "timelens2_8b":
        from robotics_vlm.models.adapters.qwen3_vl import Qwen3VLRequest
        return Qwen3VLRequest(text=prompt, query_mode=QueryMode.VIDEO, frames=frames, generation=generation)
    if model == "llava_next_video_7b":
        from robotics_vlm.models.adapters.llava_video import LlavaVideoRequest
        return LlavaVideoRequest(text=prompt, query_mode=QueryMode.VIDEO, frames=frames, generation=generation)
    from robotics_vlm.models.adapters.videochat3 import VideoChat3Request
    return VideoChat3Request(text=prompt, query_mode=QueryMode.VIDEO, frames=frames, generation=generation)


def mvbench_prompt(item: dict[str, Any]) -> str:
    choices = "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(item["candidates"]))
    return (
        f"Question: {item['question']}\nOptions:\n{choices}\n"
        "Answer with the option letter only."
    )


def track_items(track: str) -> list[dict[str, Any]]:
    file = PREPARED_ROOT / ("mvbench_manifest.json" if track == "mvbench" else "vue_tr_manifest.json")
    return json.loads(file.read_text(encoding="utf-8"))["items"]


def result_item_id(track: str, item: dict[str, Any]) -> str:
    if track == "mvbench":
        return f"{item['task']}:{item['source_index']}"
    return str(item["query_id"])


def run(model: str, track: str, limit: int | None = None) -> int:
    if track not in MODEL_TRACKS[model]:
        raise SystemExit(f"{MODEL_NAMES[model]} is not assigned to {track}")
    items = track_items(track)
    if limit is not None:
        items = items[:limit]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESULT_ROOT / f"{track}__{model}.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done[str(row["item_id"])] = row
    adapter = build_adapter(model)
    try:
        print(f"loading {MODEL_NAMES[model]} for {track}; {len(items)} items", flush=True)
        try:
            adapter.prepare()
        except Exception as error:
            with out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"item_id": "__model_prepare__", "status": "prepare_error",
                                       "error": f"{type(error).__name__}: {error}", "model": model,
                                       "track": track}, ensure_ascii=False) + "\n")
            print(f"MODEL PREPARE FAILED: {type(error).__name__}: {error}", flush=True)
            return 2
        print("model ready", flush=True)
        with out.open("a", encoding="utf-8") as sink:
            for position, item in enumerate(items, 1):
                item_id = result_item_id(track, item)
                if item_id in done and done[item_id].get("status") == "complete":
                    continue
                frames, frame_meta = load_video_frames(track, item)
                if track == "mvbench":
                    prompt = mvbench_prompt(item)
                else:
                    prompt = (
                        f"Given the query: {json.dumps(item['query'], ensure_ascii=False)}, return all "
                        "time spans in seconds where the query is visually relevant. Output only a JSON "
                        "array of [start, end] pairs. Return [] if no relevant span."
                    )
                request = request_for(model, prompt, frames, 64 if track == "mvbench" else 512)
                try:
                    output = adapter.infer(request)
                    if track == "mvbench":
                        prediction, parse_status = parse_choice(output.text, item["candidates"])
                        row = {
                            "item_id": item_id, "source_index": item["source_index"], "task": item["task"],
                            "status": "complete", "question": item["question"], "candidates": item["candidates"],
                            "gold_answer": item["answer"], "prediction": prediction, "parse_status": parse_status,
                            "correct": prediction == item["answer"], "raw_output": output.text,
                        }
                    else:
                        prediction, parse_error, parse_status = parse_intervals(output.text)
                        row = {
                            "item_id": item_id, "query_id": item["query_id"], "video_id": item["video_id"],
                            "duration_category": item["duration_category"], "query_format": item["query_format"],
                            "query_modality": item["query_modality"], "status": "complete",
                            "prediction_intervals_seconds": prediction, "parse_error": parse_error,
                            "parse_status": parse_status, "interval_parser_version": INTERVAL_PARSER_VERSION,
                            "raw_output": output.text,
                        }
                    row.update({"model": model, "track": track, "latency_ms": output.latency_ms,
                                "frames_used": output.frames_used, "frame_sampling": frame_meta,
                                "adapter_metadata": output.raw})
                except Exception as error:
                    row = {"item_id": item_id, "status": "inference_error", "error": f"{type(error).__name__}: {error}", "model": model, "track": track}
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n"); sink.flush()
                    print(f"STOP at {item_id}: {row['error']}", flush=True)
                    return 2
                sink.write(json.dumps(row, ensure_ascii=False) + "\n"); sink.flush()
                if position % 5 == 0 or position == len(items):
                    print(f"{track} {MODEL_NAMES[model]} {position}/{len(items)}", flush=True)
    finally:
        adapter.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_TRACKS, required=True)
    parser.add_argument("--track", choices=("mvbench", "vue_tr"), required=True)
    parser.add_argument("--limit", type=int, help="Run a prefix for a deliberate partial evaluation")
    args = parser.parse_args()
    return run(args.model, args.track, args.limit)


if __name__ == "__main__":
    raise SystemExit(run_queued(main, "video-vlm"))
