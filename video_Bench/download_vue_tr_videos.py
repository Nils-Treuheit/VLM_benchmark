#!/usr/bin/env python3
"""Fetch the manifest's vision-only VUE-TR videos at no more than 480p."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

DEFAULT_MANIFEST = Path("data/video_bench/prepared/vue_tr_manifest.json")
DEFAULT_VIDEO_ROOT = Path("data/video_bench/VUE-TR/videos")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    stream = (metadata.get("streams") or [{}])[0]
    return {
        "duration_seconds": float(metadata["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
    }


def download(manifest_path: Path, video_root: Path, limit: int | None = None) -> tuple[int, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_root.mkdir(parents=True, exist_ok=True)
    items = manifest["items"][:limit] if limit is not None else manifest["items"]
    ok = failed = 0
    for item in items:
        video_id = item["video_id"]
        if item.get("asset_state") == "downloaded_and_verified":
            ok += 1
            continue
        existing = sorted(video_root.glob(f"{video_id}.*"))
        existing = [
            path for path in existing
            if path.is_file() and path.stat().st_size > 0 and not path.name.endswith((".part", ".ytdl"))
        ]
        try:
            if existing:
                path = existing[0]
            else:
                options = {
                    "format": "best[height<=480]/bestvideo[height<=480]",
                    "outtmpl": str(video_root / "%(id)s.%(ext)s"),
                    "noplaylist": True,
                    "quiet": True,
                    "no_warnings": True,
                    "noprogress": True,
                    "continuedl": True,
                    "nopart": True,
                    "overwrites": False,
                    "retries": 3,
                    "fragment_retries": 3,
                }
                with YoutubeDL(options) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                    if info is None:
                        raise RuntimeError("yt-dlp returned no video metadata")
                    path = Path(ydl.prepare_filename(info))
                    if not path.exists() or path.name.endswith((".part", ".ytdl")):
                        candidates = sorted(
                            candidate
                            for candidate in video_root.glob(f"{video_id}.*")
                            if candidate.is_file()
                            and not candidate.name.endswith((".part", ".ytdl"))
                        )
                        if not candidates:
                            raise FileNotFoundError("download finished without a final local video file")
                        path = candidates[0]
            properties = probe_video(path)
            if properties["height"] > 480:
                raise ValueError(f"source video exceeds the 480p acquisition cap: {properties['height']}p")
            item.update(
                asset_state="downloaded_and_verified",
                video_path=str(path),
                asset_bytes=path.stat().st_size,
                asset_sha256=sha256(path),
                probed_video=properties,
            )
            ok += 1
            print(f"OK {ok}/{len(items)} {video_id} {properties['width']}x{properties['height']} {path.stat().st_size}", flush=True)
        except Exception as exc:  # preserve per-source failures for later reconciliation
            for partial in video_root.glob(f"{video_id}.*"):
                if partial.is_file():
                    partial.unlink(missing_ok=True)
            item.update(asset_state="unavailable", download_error=f"{type(exc).__name__}: {exc}")
            failed += 1
            print(f"UNAVAILABLE {video_id}: {type(exc).__name__}: {exc}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ok, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Only visit the first N manifest items")
    args = parser.parse_args()
    ok, failed = download(args.manifest, args.video_root, args.limit)
    print(f"download summary: {ok} available, {failed} unavailable")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
