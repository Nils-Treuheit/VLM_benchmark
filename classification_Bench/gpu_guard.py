#!/usr/bin/env python3
"""Fail closed unless managed GPU workloads are released and the GPU is clear."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def release_and_check_local_gpu() -> None:
    """Release managed workloads, then reject any remaining GPU process."""
    if os.environ.get("VLM_BENCHMARK_ALLOW_RUSTDESK_GPU_BASELINE") == "1":
        sys.path.insert(0, str(PROJECT_ROOT / "grounding_Bench"))
        from gpu_runtime import verify_idle_desktop_baseline

        verify_idle_desktop_baseline()
        return
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / ".agentic_workflow" / "src" / "executor.py"), "release-local"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        cleanup = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"release-local returned an unreadable result: {result.stdout!r}") from exc
    if any(value != "released" for value in cleanup.values()):
        raise RuntimeError(f"release-local did not release all managed resources: {cleanup}")

    with urllib.request.urlopen("http://127.0.0.1:20268/status", timeout=10) as response:
        status = json.loads(response.read())
    with urllib.request.urlopen("http://127.0.0.1:20268/models", timeout=10) as response:
        json.loads(response.read())
    unmatched = status.get("ollama", {}).get("unmatched_gpu_processes") or []
    if unmatched:
        raise RuntimeError(f"unmanaged GPU processes are active; refusing to start inference: {unmatched}")
