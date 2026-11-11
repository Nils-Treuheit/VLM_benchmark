#!/usr/bin/env python3
"""Prepare deterministic real-data image SGG panels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from data_prep import prepare
except ImportError:  # package execution from the eventual image_SGG_Bench directory
    from .data_prep import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("vg150", "openimages_v7", "psg", "all"), required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="public datasets root, or the Visual Genome directory for --dataset vg150")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--audit", action="store_true",
                        help="emit 12 deterministic real-image VG-150 annotation overlays for inspection")
    args = parser.parse_args(argv)
    try:
        paths = prepare(args.dataset, args.data_root.resolve(), args.output_dir.resolve(), args.seed, args.audit)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
        print(path.with_suffix(path.suffix + ".sha256"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
