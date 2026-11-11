#!/usr/bin/env python3
"""Autostart GroundAnything for one shared grounding manifest run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "classification_Bench"))
sys.path.insert(0, str(ROOT / "grounding_Bench"))

from gpu_guard import release_and_check_local_gpu  # noqa: E402
from opal_vlm_benchmark import (  # noqa: E402
    GROUNDANYTHING_SOURCE,
    start_groundanything_server,
)
import benchmark as grounding_benchmark  # noqa: E402
import model_adapters as adapters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8102/v1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    release_and_check_local_gpu()
    _, load_duration_s = start_groundanything_server(
        GROUNDANYTHING_SOURCE, args.output_dir, args.base_url, args.startup_timeout_s,
    )
    sys.argv = [
        "grounding_Bench/benchmark.py",
        "--data-dir", str(args.data_dir),
        "--tasks", "coco",
        "--models", adapters.GROUND_ID,
        "--coco-annotations", str(args.coco_annotations),
        "--groundanything-base-url", args.base_url,
        "--groundanything-model-load-duration-s", str(load_duration_s),
        "--max-new-tokens", str(args.max_new_tokens),
        "--output-dir", str(args.output_dir),
    ]
    return grounding_benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
