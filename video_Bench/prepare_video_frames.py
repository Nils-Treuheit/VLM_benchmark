#!/usr/bin/env python3
"""Decode the fixed MVBench and VUE-TR samples into shared 16-frame caches."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

POLICY = Path("data/video_bench/prepared/video_qa_sampling_policy.json")
MV_MANIFEST = Path("data/video_bench/prepared/mvbench_manifest.json")
VUE_MANIFEST = Path("data/video_bench/prepared/vue_tr_manifest.json")
CACHE_ROOT = Path("data/video_bench/prepared/frame_cache")


def timestamps(path: Path) -> list[tuple[int, float]]:
    found: list[tuple[int, float]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                continue
            seconds = float(Fraction(frame.pts) * Fraction(frame.time_base))
            if np.isfinite(seconds):
                found.append((index, seconds))
    # A duplicate presentation timestamp is represented by its earliest source frame.
    unique: list[tuple[int, float]] = []
    previous: float | None = None
    for index, seconds in found:
        if previous is None or seconds > previous:
            unique.append((index, seconds))
            previous = seconds
    return unique


def choose_frames(values: list[tuple[int, float]], count: int = 16) -> list[tuple[int, float]]:
    if len(values) < 2:
        raise ValueError("fewer than two distinct decoded frames with valid source PTS")
    k = min(count, len(values))
    start, end = values[0][1], values[-1][1]
    targets = np.linspace(start, end, k).tolist()
    # Layered dynamic programming minimizes total absolute target error. Path tuples
    # provide the specified lexicographically earliest tie break.
    previous: list[tuple[float, tuple[int, ...]] | None] = []
    for index, (_, seconds) in enumerate(values):
        previous.append((abs(seconds - targets[0]), (index,)))
    for layer in range(1, k):
        current: list[tuple[float, tuple[int, ...]] | None] = [None] * len(values)
        best: tuple[float, tuple[int, ...]] | None = None
        for index, (_, seconds) in enumerate(values):
            if index:
                candidate = previous[index - 1]
                if candidate is not None and (best is None or candidate < best):
                    best = candidate
            if best is not None:
                current[index] = (best[0] + abs(seconds - targets[layer]), best[1] + (index,))
        previous = current
    best_path = min((entry for entry in previous if entry is not None), key=lambda entry: entry)
    return [values[index] for index in best_path[1]]


def decode_selected(path: Path, selected: list[tuple[int, float]]) -> list[np.ndarray]:
    wanted = {index for index, _ in selected}
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                frames[index] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
    if len(frames) != len(wanted):
        raise ValueError("second decode did not produce every selected source frame")
    return [frames[index] for index, _ in selected]


def process(item: dict[str, Any], dataset: str, media_path: Path) -> None:
    cache_dir = CACHE_ROOT / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = str(item["video_id"] if item.get("video_id") else Path(item["video"]).stem)
    cache_file = cache_dir / f"{stem}.npz"
    metadata_file = cache_dir / f"{stem}.json"
    if cache_file.exists() and metadata_file.exists():
        return
    pts = timestamps(media_path)
    selected = choose_frames(pts)
    images = decode_selected(media_path, selected)
    np.savez_compressed(cache_file, frames=np.stack(images))
    source_times = [seconds for _, seconds in selected]
    metadata = {
        "dataset": dataset,
        "video_id": stem,
        "video_path": str(media_path),
        "source_duration_seconds": pts[-1][1] - pts[0][1],
        "source_frame_count_with_valid_pts": len(pts),
        "selected_frame_count": len(selected),
        "selected_source_frame_indices": [index for index, _ in selected],
        "selected_source_timestamps_seconds": source_times,
        "effective_sample_fps": (len(selected) - 1) / (source_times[-1] - source_times[0]),
        "source_resolution": {"width": images[0].shape[1], "height": images[0].shape[0]},
        "cache_file": str(cache_file),
        "frame_policy": "uniform 16 target timestamps, source PTS, minimum-total-absolute-distance sequence, lexicographically earliest tie break",
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    policy = json.loads(POLICY.read_text())
    if policy["max_frames"] != 16 or "presentation timestamps" not in policy["decoder"]["timestamp_source"]:
        raise SystemExit("unexpected shared frame-sampling policy")
    mv = json.loads(MV_MANIFEST.read_text())
    vue = json.loads(VUE_MANIFEST.read_text())
    jobs = [(item, "mvbench", Path(item["prepared_path"])) for item in mv["items"]]
    jobs += [(item, "vue_tr", Path(item["video_path"])) for item in vue["items"]]
    for index, (item, dataset, path) in enumerate(jobs, 1):
        if not path.is_file():
            raise FileNotFoundError(path)
        process(item, dataset, path)
        if index % 10 == 0 or index == len(jobs):
            print(f"prepared {index}/{len(jobs)} frame caches", flush=True)
    print(f"ready: {len(jobs)} shared video frame caches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
