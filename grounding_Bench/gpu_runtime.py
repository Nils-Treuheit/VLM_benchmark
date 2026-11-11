"""GPU engine release and explicit desktop-baseline verification."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONTROL_URL = "http://127.0.0.1:20268"
MAX_IDLE_GPU_MIB = 1536
ALLOWED_DESKTOP_PROCESS = "/usr/share/rustdesk/rustdesk"


def _get_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(CONTROL_URL + path, timeout=10) as response:
        value = json.loads(response.read().decode("utf-8", "replace"))
    if not isinstance(value, dict):
        raise RuntimeError(f"model-control {path} response is not an object")
    return value


def verify_idle_desktop_baseline() -> None:
    sessions = _get_json("/sessions").get("sessions")
    state = _get_json("/status")
    if not isinstance(sessions, list) or sessions:
        raise RuntimeError("cannot bypass release guard while OpenCode sessions are active or unknown")
    memory = state.get("gpu_memory_used_mib")
    if (isinstance(memory, bool) or not isinstance(memory, (int, float))
            or not math.isfinite(memory) or memory < 0 or memory >= MAX_IDLE_GPU_MIB):
        raise RuntimeError(f"idle GPU baseline must be below {MAX_IDLE_GPU_MIB} MiB, got {memory!r}")
    if state.get("ollama_loaded") != [] or state.get("ollama", {}).get("gpu_ollama_processes") != []:
        raise RuntimeError("Ollama still has GPU models or runners")
    servers = state.get("ollama", {}).get("servers")
    if not isinstance(servers, list) or not servers or any(
        not isinstance(server, dict)
        or "inventory_error" not in server
        or server.get("inventory_error") is not None
        or server.get("models") != []
        or server.get("gpu_runner_processes") != []
        or server.get("gpu_memory_mib") not in (0, None)
        for server in servers
    ):
        raise RuntimeError("Ollama server inventory is not confirmed idle")
    if (state.get("vllm_listening") is not False
            or state.get("vllm", {}).get("processes") != []
            or state.get("unsloth_listening") is not False
            or state.get("unsloth", {}).get("pids") != []):
        raise RuntimeError("vLLM or Unsloth remains active")
    djev = state.get("djev", {})
    if (djev.get("process_groups_alive") != []
            or djev.get("upstream_listening") is not False
            or djev.get("structured_listening") is not False):
        raise RuntimeError("D-Jev remains active")
    jobs = state.get("jobs", {})
    if not isinstance(jobs, dict) or any(
        not isinstance(job, dict) or job.get("status") not in {"done", "error"}
        for job in jobs.values()
    ):
        raise RuntimeError("model-control reports an active engine job")
    unmatched = state.get("ollama", {}).get("unmatched_gpu_processes")
    if not isinstance(unmatched, list) or any(
        not isinstance(process, dict)
        or process.get("process_name") != ALLOWED_DESKTOP_PROCESS
        for process in unmatched
    ):
        raise RuntimeError("GPU has an unrecognized process outside the managed engines")
    print(f"[gpu-queue] verified idle model engines; allowing desktop baseline ({memory} MiB)", flush=True)


def release_local() -> None:
    """Release managed engines, or verify the explicitly allowed desktop-only baseline."""
    if os.environ.get("VLM_BENCHMARK_ALLOW_RUSTDESK_GPU_BASELINE") == "1":
        verify_idle_desktop_baseline()
        return
    workflow = ROOT / ".agentic_workflow"
    executor = workflow / "src" / "executor.py"
    if not executor.is_file():
        raise FileNotFoundError(f"required local-engine release helper is missing: {executor}")
    subprocess.run([sys.executable, "src/executor.py", "release-local"], cwd=workflow, check=True)
