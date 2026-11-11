#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BENCH_DIR/../.." && pwd)"
cd "$REPO_DIR"
uv run python "$BENCH_DIR/runner.py" "$@"
