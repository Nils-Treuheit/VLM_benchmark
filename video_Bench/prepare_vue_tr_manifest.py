#!/usr/bin/env python3
"""Build the fixed, vision-only VUE-TR comparison sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("data/video_bench/VUE-TR/source/VUE-TR_ground_truth.json")
DEFAULT_OUTPUT = Path("data/video_bench/prepared/vue_tr_manifest.json")
DURATION_ORDER = ("ultra-short", "short", "medium", "long", "ultra-long")
FORMAT_ORDER = ("keyword", "phrase", "sentence")


def proportional_quotas(counts: Counter[str], total: int, order: tuple[str, ...]) -> dict[str, int]:
    """Allocate an exact total proportionally, with stable largest remainders."""
    denominator = sum(counts.values())
    if total < 0 or denominator == 0 or total > denominator:
        raise ValueError("cannot allocate requested sample across these strata")
    exact = {key: total * counts[key] / denominator for key in order}
    quotas = {key: int(exact[key]) for key in order}
    remaining = total - sum(quotas.values())
    for key in sorted(order, key=lambda k: (-(exact[k] - quotas[k]), order.index(k)))[:remaining]:
        quotas[key] += 1
    return quotas


def choose_unique_video_queries(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample unique videos while matching duration and query-format proportions."""
    rng = random.Random(seed)
    by_duration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_duration[row["duration_category"]].append(row)

    duration_counts = Counter({key: len(by_duration[key]) for key in DURATION_ORDER})
    duration_quotas = proportional_quotas(duration_counts, count, DURATION_ORDER)
    selected: list[dict[str, Any]] = []
    for duration in DURATION_ORDER:
        duration_rows = by_duration[duration]
        quota = duration_quotas[duration]
        format_counts = Counter(row["query_format"] for row in duration_rows)
        format_quotas = proportional_quotas(format_counts, quota, FORMAT_ORDER)
        by_format_video: dict[str, dict[str, list[dict[str, Any]]]] = {
            fmt: defaultdict(list) for fmt in FORMAT_ORDER
        }
        for row in duration_rows:
            by_format_video[row["query_format"]][row["video_id"]].append(row)

        # Allocate rare formats first so a video eligible for multiple formats
        # does not consume the only available video for a smaller stratum.
        format_order = sorted(
            FORMAT_ORDER,
            key=lambda fmt: (len(by_format_video[fmt]), FORMAT_ORDER.index(fmt)),
        )
        used_video_ids: set[str] = set()
        chosen: list[dict[str, Any]] = []
        for fmt in format_order:
            candidates = [
                video_id
                for video_id in by_format_video[fmt]
                if video_id not in used_video_ids
            ]
            rng.shuffle(candidates)
            need = format_quotas[fmt]
            for video_id in candidates[:need]:
                used_video_ids.add(video_id)
                chosen.append(rng.choice(by_format_video[fmt][video_id]))

        if len(chosen) < quota:
            remaining_videos: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in duration_rows:
                if row["video_id"] not in used_video_ids:
                    remaining_videos[row["video_id"]].append(row)
            candidates = list(remaining_videos)
            rng.shuffle(candidates)
            for video_id in candidates[: quota - len(chosen)]:
                used_video_ids.add(video_id)
                chosen.append(rng.choice(remaining_videos[video_id]))

        if len(chosen) != quota:
            raise ValueError(
                f"duration stratum {duration!r} cannot supply {quota} distinct videos"
            )
        rng.shuffle(chosen)
        selected.extend(chosen)

    if len({row["video_id"] for row in selected}) != count:
        raise ValueError("sample selection did not produce unique source videos")
    return selected


def build_manifest(source: Path, output: Path, seed: int = 3407, count: int = 100) -> dict[str, Any]:
    rows = json.loads(source.read_text(encoding="utf-8"))
    visual_rows = [
        row
        for row in rows
        if row.get("query_modality") == "vision"
        and row.get("duration_category") in DURATION_ORDER
        and row.get("query_format") in FORMAT_ORDER
    ]
    selected = choose_unique_video_queries(visual_rows, count=count, seed=seed)
    items = []
    for row in sorted(selected, key=lambda value: value["query_id"]):
        items.append(
            {
                "query_id": row["query_id"],
                "video_id": row["video_id"],
                "video_path": f"data/video_bench/VUE-TR/videos/{row['video_id']}.mp4",
                "duration_seconds": row["duration"],
                "query": row["query"],
                "ground_truth_intervals_seconds": row["gt"],
                "task": row["task"],
                "duration_category": row["duration_category"],
                "query_format": row["query_format"],
                "query_modality": row["query_modality"],
                "asset_state": "not_downloaded",
            }
        )
    manifest = {
        "dataset": "ByteDance/VUE-TR",
        "official_repository": "https://github.com/bytedance/vidi/tree/main/VUE_TR",
        "source_annotation": str(source),
        "source_annotation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_size": count,
        "seed": seed,
        "selection": "vision-only; proportional largest-remainder allocation by duration category and query format; one query per unique video; deterministic Python random seed",
        "selected_task_counts": dict(Counter(item["duration_category"] for item in items)),
        "selected_query_format_counts": dict(Counter(item["query_format"] for item in items)),
        "query_modality_counts": dict(Counter(item["query_modality"] for item in items)),
        "unique_videos": len({item["video_id"] for item in items}),
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def repair_unavailable(source: Path, manifest_path: Path, seed: int = 3407) -> dict[str, Any]:
    """Replace unavailable source videos within the same duration/format strata."""
    rows = json.loads(source.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unavailable = [item for item in manifest["items"] if item.get("asset_state") == "unavailable"]
    excluded_ids = {item["video_id"] for item in unavailable}
    used_ids = {item["video_id"] for item in manifest["items"] if item not in unavailable}
    ledger = list(manifest.get("replacement_ledger", []))
    for index, old in enumerate(unavailable):
        exact = [
            row for row in rows
            if row.get("query_modality") == "vision"
            and row.get("duration_category") == old["duration_category"]
            and row.get("query_format") == old["query_format"]
            and row.get("video_id") not in used_ids
            and row.get("video_id") not in excluded_ids
        ]
        rng = random.Random(seed + int(old["query_id"]) + index)
        rng.shuffle(exact)
        if exact:
            replacement = exact[0]
        else:
            relaxed = [
                row for row in rows
                if row.get("query_modality") == "vision"
                and row.get("duration_category") == old["duration_category"]
                and row.get("video_id") not in used_ids
                and row.get("video_id") not in excluded_ids
            ]
            rng.shuffle(relaxed)
            if not relaxed:
                raise ValueError(f"no replacement video remains for {old['video_id']}")
            replacement = relaxed[0]
        used_ids.add(replacement["video_id"])
        new = {
            "query_id": replacement["query_id"],
            "video_id": replacement["video_id"],
            "video_path": f"data/video_bench/VUE-TR/videos/{replacement['video_id']}.mp4",
            "duration_seconds": replacement["duration"],
            "query": replacement["query"],
            "ground_truth_intervals_seconds": replacement["gt"],
            "task": replacement["task"],
            "duration_category": replacement["duration_category"],
            "query_format": replacement["query_format"],
            "query_modality": replacement["query_modality"],
            "asset_state": "not_downloaded",
        }
        ledger.append(
            {
                "replaced_query_id": old["query_id"],
                "unavailable_video_id": old["video_id"],
                "failure": old.get("download_error"),
                "replacement_query_id": new["query_id"],
                "replacement_video_id": new["video_id"],
                "same_duration_category": old["duration_category"] == new["duration_category"],
                "same_query_format": old["query_format"] == new["query_format"],
            }
        )
        manifest["items"][manifest["items"].index(old)] = new
    manifest["items"].sort(key=lambda item: item["query_id"])
    manifest["replacement_ledger"] = ledger
    manifest["selected_task_counts"] = dict(Counter(item["duration_category"] for item in manifest["items"]))
    manifest["selected_query_format_counts"] = dict(Counter(item["query_format"] for item in manifest["items"]))
    manifest["unique_videos"] = len({item["video_id"] for item in manifest["items"]})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--repair-unavailable", action="store_true")
    args = parser.parse_args()
    if args.repair_unavailable:
        manifest = repair_unavailable(args.source, args.output, seed=args.seed)
    else:
        manifest = build_manifest(args.source, args.output, seed=args.seed, count=args.count)
    print(
        f"Prepared {manifest['sample_size']} queries across {manifest['unique_videos']} unique videos; "
        f"duration={manifest['selected_task_counts']}; formats={manifest['selected_query_format_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
