#!/usr/bin/env python3
"""Run one model at a time over its assigned native-video benchmark tracks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROBOTICS_SRC = Path("/mnt/HDD1/Project_Code/robotics_VLM/src")
sys.path.insert(0, str(ROBOTICS_SRC))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CACHE_ROOT = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub")
RESULT_ROOT = Path("data/video_bench/results/raw")
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


def load_video_frames(dataset: str, item: dict[str, Any]):
    from robotics_vlm.contracts.common import ClockDomain
    from robotics_vlm.frames import Frame
    item_id = str(item["video_id"] if item.get("video_id") else Path(item["video"]).stem)
    cache_file = Path("data/video_bench/prepared/frame_cache") / dataset / f"{item_id}.npz"
    metadata = json.loads(cache_file.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(cache_file) as data:
        images = data["frames"]
    frames = []
    receipt = time.monotonic_ns()
    for sequence, (image, seconds) in enumerate(zip(images, metadata["selected_source_timestamps_seconds"], strict=True)):
        source_pts = metadata["selected_source_frame_pts"][sequence]
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


def parse_choice(text: str | None, choices: list[str]) -> tuple[str | None, str]:
    if not text:
        return None, "empty_output"
    match = re.search(r"(?:^|\b)([A-E])(?:\b|[).:]|$)", text.strip(), re.IGNORECASE)
    if match:
        index = ord(match.group(1).upper()) - 65
        if index < len(choices):
            return choices[index], "letter"
    normalized = " ".join(text.casefold().split())
    exact = [choice for choice in choices if " ".join(choice.casefold().split()) == normalized]
    if len(exact) == 1:
        return exact[0], "exact_choice_text"
    return None, "unparsed_choice"


def parse_intervals(text: str | None) -> tuple[list[list[float]], str | None]:
    if not text:
        return [], "empty_output"
    content = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group(1).strip()
    left = content.find("[")
    right = content.rfind("]")
    if left < 0 or right < left:
        return [], "no_json_array"
    try:
        value = json.loads(content[left:right + 1])
        if not isinstance(value, list):
            raise ValueError("interval output is not an array")
        result = []
        for interval in value:
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not np.isfinite(n) for n in interval)):
                raise ValueError("each interval must contain two finite numbers")
            if interval[0] > interval[1]:
                raise ValueError("interval start exceeds end")
            result.append([float(interval[0]), float(interval[1])])
        return result, None
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        return [], f"invalid_interval_json: {error}"


def track_items(track: str) -> list[dict[str, Any]]:
    file = Path("data/video_bench/prepared") / ("mvbench_manifest.json" if track == "mvbench" else "vue_tr_manifest.json")
    return json.loads(file.read_text(encoding="utf-8"))["items"]


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
                item_id = str(item.get("query_id", item.get("source_index")))
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
                        prediction, parse_error = parse_intervals(output.text)
                        row = {
                            "item_id": item_id, "query_id": item["query_id"], "video_id": item["video_id"],
                            "duration_category": item["duration_category"], "query_format": item["query_format"],
                            "query_modality": item["query_modality"], "status": "complete",
                            "prediction_intervals_seconds": prediction, "parse_error": parse_error,
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
    raise SystemExit(main())
