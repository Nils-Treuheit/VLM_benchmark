#!/usr/bin/env python3
"""Prepare one shared source-PTS frame sample for each active benchmark video."""

from __future__ import annotations

import json
import hashlib
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data/video_bench"
POLICY = DATA_ROOT / "prepared/video_qa_sampling_policy.json"
MV_MANIFEST = DATA_ROOT / "prepared/mvbench_manifest.json"
VUE_MANIFEST = DATA_ROOT / "prepared/vue_tr_manifest.json"
CACHE_ROOT = DATA_ROOT / "prepared/frame_cache"
FRAME_LIMIT = 16


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
        chosen: list[tuple[float, int, Fraction, np.ndarray]] = []

        def append_nearest(target: float, candidates: list[tuple[float, int, Fraction, Any]]) -> None:
            if not candidates:
                return
            nearest = min(candidates, key=lambda candidate: (abs(candidate[0] - target), candidate[0]))
            if chosen and nearest[1] <= chosen[-1][1]:
                return
            chosen.append((nearest[0], nearest[1], nearest[2], nearest[3].to_ndarray(format="rgb24")))

        if duration <= 120:
            target_index = 0
            before: tuple[float, int, Fraction, Any] | None = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_time_base = Fraction(frame.time_base or time_base)
                frame_time = float(Fraction(frame.pts) * frame_time_base)
                current = (frame_time, int(frame.pts), frame_time_base, frame)
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
                    frame_time_base = Fraction(frame.time_base or time_base)
                    frame_time = float(Fraction(frame.pts) * frame_time_base)
                    current = (frame_time, int(frame.pts), frame_time_base, frame)
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
        timestamps = [seconds for seconds, _, _, _ in chosen]
        ppts = [pts for _, pts, _, _ in chosen]
        pts_time_bases = [str(frame_time_base) for _, _, frame_time_base, _ in chosen]
        images = [image for _, _, _, image in chosen]
        metadata = {
            "source_duration_seconds": duration,
            "source_fps": fps,
            "source_time_base": str(time_base),
            "source_resolution": {"width": images[0].shape[1], "height": images[0].shape[0]},
            "selected_frame_count": len(images),
            "selected_source_frame_pts": ppts,
            "selected_source_frame_time_bases": pts_time_bases,
            "selected_source_timestamps_seconds": timestamps,
            "effective_sample_fps": (len(images) - 1) / (timestamps[-1] - timestamps[0]),
            "frame_policy": "evenly spaced timeline targets; nearest decoded source-PTS frame per target; earlier frame wins ties; duplicate PTS omitted; sequential scan through 120 seconds and keyframe seek for longer clips",
        }
        return images, ppts, metadata


def process(
    item: dict[str, Any], dataset: str, media_path: Path,
    policy: dict[str, Any], policy_sha256: str,
) -> None:
    cache_dir = CACHE_ROOT / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = str(item["video_id"] if item.get("video_id") else Path(item["video"]).stem)
    cache_file = cache_dir / f"{stem}.npz"
    metadata_file = cache_dir / f"{stem}.json"
    media_rel = str(item["prepared_path"] if dataset == "mvbench" else item["video_path"])
    expected_size = int(item.get("asset_size_bytes", item.get("asset_bytes", -1)))
    expected_hash = str(item.get("asset_sha256", ""))
    if media_path.stat().st_size != expected_size or sha256_file(media_path) != expected_hash:
        raise ValueError(f"active manifest asset size/hash mismatch for {dataset}/{stem}")
    item_key = cache_item_key(
        dataset, item, media_rel, expected_hash, str(policy["policy_id"]), policy_sha256
    )
    if cache_file.exists() and metadata_file.exists():
        try:
            old = json.loads(metadata_file.read_text(encoding="utf-8"))
            if (old.get("dataset") == dataset and old.get("video_id") == stem
                    and old.get("video_path") == media_rel
                    and old.get("cache_file") == str(cache_file.relative_to(PROJECT))
                    and old.get("source_sha256") == expected_hash
                    and old.get("sampling_policy_id") == policy["policy_id"]
                    and old.get("sampling_policy_sha256") == policy_sha256
                    and old.get("manifest_cache_key") == item_key
                    and old.get("frame_cache_sha256") == sha256_file(cache_file)):
                return
        except (OSError, ValueError, TypeError):
            pass
    images, _, metadata = sample_frames(media_path)
    with cache_file.open("wb") as sink:
        np.savez_compressed(sink, frames=np.stack(images))
    metadata.update({
        "dataset": dataset,
        "video_id": stem,
        "video_path": media_rel,
        "cache_file": str(cache_file.relative_to(PROJECT)),
        "source_sha256": expected_hash,
        "sampling_policy_id": policy["policy_id"],
        "sampling_policy_sha256": policy_sha256,
        "manifest_cache_key": item_key,
        "frame_cache_sha256": sha256_file(cache_file),
    })
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    policy = json.loads(POLICY.read_text())
    policy_sha256 = sha256_file(POLICY)
    if policy["max_frames"] != FRAME_LIMIT:
        raise SystemExit("frame count differs from the declared shared policy")
    mv = json.loads(MV_MANIFEST.read_text())
    vue = json.loads(VUE_MANIFEST.read_text())
    jobs = [(item, "mvbench", PROJECT / item["prepared_path"]) for item in mv["items"]]
    jobs += [(item, "vue_tr", PROJECT / item["video_path"]) for item in vue["items"]]
    for index, (item, dataset, path) in enumerate(jobs, 1):
        if not path.is_file():
            raise FileNotFoundError(path)
        process(item, dataset, path, policy, policy_sha256)
        if index % 10 == 0 or index == len(jobs):
            print(f"prepared {index}/{len(jobs)} shared frame caches", flush=True)
    print(f"ready: {len(jobs)} shared video frame caches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
