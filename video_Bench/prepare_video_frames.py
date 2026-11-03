#!/usr/bin/env python3
"""Prepare one shared source-PTS frame sample for each active benchmark video."""

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
FRAME_LIMIT = 16


def sample_frames(path: Path) -> tuple[list[np.ndarray], list[int], dict[str, Any]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = Fraction(stream.time_base)
        duration = (
            float(stream.duration * time_base)
            if stream.duration is not None
            else float(container.duration / av.time_base)
        )
        fps = float(stream.average_rate or stream.base_rate or 0)
        estimated_count = int(stream.frames or round(duration * fps)) if fps else 0
        if duration <= 0 or fps <= 0 or estimated_count < 2:
            raise ValueError("video stream lacks a usable duration, frame rate, or frame count")
        count = min(FRAME_LIMIT, estimated_count)
        start_seconds = float(stream.start_time * time_base) if stream.start_time is not None else 0.0
        frame_period = 1.0 / fps
        end_seconds = max(start_seconds, start_seconds + duration - frame_period)
        targets = np.linspace(start_seconds, end_seconds, count).tolist()
        chosen: list[tuple[float, int, np.ndarray]] = []

        def append_nearest(target: float, candidates: list[tuple[float, int, Any]]) -> None:
            if not candidates:
                return
            nearest = min(candidates, key=lambda candidate: (abs(candidate[0] - target), candidate[0]))
            if chosen and nearest[1] <= chosen[-1][1]:
                return
            chosen.append((nearest[0], nearest[1], nearest[2].to_ndarray(format="rgb24")))

        if duration <= 120:
            target_index = 0
            before: tuple[float, int, Any] | None = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_time = float(Fraction(frame.pts) * Fraction(frame.time_base or time_base))
                current = (frame_time, int(frame.pts), frame)
                while target_index < len(targets) and targets[target_index] <= frame_time:
                    candidates = [item for item in (before, current) if item is not None]
                    append_nearest(targets[target_index], candidates)
                    target_index += 1
                before = current
            while target_index < len(targets) and before is not None:
                append_nearest(targets[target_index], [before])
                target_index += 1
        else:
            for target in targets:
                target_pts = int(round(target / float(time_base)))
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                before = None
                after = None
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    frame_time = float(Fraction(frame.pts) * Fraction(frame.time_base or time_base))
                    current = (frame_time, int(frame.pts), frame)
                    if frame_time <= target:
                        before = current
                        if frame_time == target:
                            after = before
                            break
                    else:
                        after = current
                        break
                append_nearest(target, [item for item in (before, after) if item is not None])

        if len(chosen) < 2:
            raise ValueError("fewer than two distinct source-PTS frames were selected")
        timestamps = [seconds for seconds, _, _ in chosen]
        ppts = [pts for _, pts, _ in chosen]
        images = [image for _, _, image in chosen]
        metadata = {
            "source_duration_seconds": duration,
            "source_fps": fps,
            "source_resolution": {"width": images[0].shape[1], "height": images[0].shape[0]},
            "selected_frame_count": len(images),
            "selected_source_frame_pts": ppts,
            "selected_source_timestamps_seconds": timestamps,
            "effective_sample_fps": (len(images) - 1) / (timestamps[-1] - timestamps[0]),
            "frame_policy": "evenly spaced timeline targets; nearest decoded source-PTS frame per target; earlier frame wins ties; duplicate PTS omitted; sequential scan through 120 seconds and keyframe seek for longer clips",
        }
        return images, ppts, metadata


def process(item: dict[str, Any], dataset: str, media_path: Path) -> None:
    cache_dir = CACHE_ROOT / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = str(item["video_id"] if item.get("video_id") else Path(item["video"]).stem)
    cache_file = cache_dir / f"{stem}.npz"
    metadata_file = cache_dir / f"{stem}.json"
    if cache_file.exists() and metadata_file.exists():
        return
    images, _, metadata = sample_frames(media_path)
    np.savez_compressed(cache_file, frames=np.stack(images))
    metadata.update({"dataset": dataset, "video_id": stem, "video_path": str(media_path), "cache_file": str(cache_file)})
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    policy = json.loads(POLICY.read_text())
    if policy["max_frames"] != FRAME_LIMIT:
        raise SystemExit("frame count differs from the declared shared policy")
    mv = json.loads(MV_MANIFEST.read_text())
    vue = json.loads(VUE_MANIFEST.read_text())
    jobs = [(item, "mvbench", Path(item["prepared_path"])) for item in mv["items"]]
    jobs += [(item, "vue_tr", Path(item["video_path"])) for item in vue["items"]]
    for index, (item, dataset, path) in enumerate(jobs, 1):
        if not path.is_file():
            raise FileNotFoundError(path)
        process(item, dataset, path)
        if index % 10 == 0 or index == len(jobs):
            print(f"prepared {index}/{len(jobs)} shared frame caches", flush=True)
    print(f"ready: {len(jobs)} shared video frame caches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
