"""Process-safe queue for benchmark jobs that share the host GPU."""

from __future__ import annotations

import fcntl
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


DEFAULT_QUEUE_DIR = Path(os.environ.get("VLM_BENCHMARK_GPU_QUEUE_DIR", "/tmp/vlm-benchmark-gpu-queue"))


@contextmanager
def gpu_queue(benchmark: str) -> Iterator[int]:
    """Wait until the shared GPU is free, then hold its lock until exit.

    The kernel releases the advisory lock when a benchmark exits or crashes,
    so abandoned lock files do not block later jobs.
    """
    DEFAULT_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DEFAULT_QUEUE_DIR / "gpu-0.lock"
    lock_file = lock_path.open("a+")
    print(f"[gpu-queue] {benchmark}: waiting for GPU", flush=True)
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"[gpu-queue] {benchmark}: acquired GPU", flush=True)
        yield lock_file.fileno()
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            print(f"[gpu-queue] {benchmark}: released GPU", flush=True)


def run_queued(main: Callable[[], int | None], benchmark: str) -> int:
    """Run under the local lock unless an external VRAM scheduler owns dispatch."""
    # Help and inventory queries do not use the GPU and should remain immediate.
    if any(arg in {"-h", "--help", "--list-models"} for arg in sys.argv[1:]):
        result = main()
        return 0 if result is None else result
    if os.environ.get("VLM_BENCHMARK_SCHEDULER_OWNS_GPU") == "1":
        print(f"[gpu-queue] {benchmark}: launched by VRAM scheduler", flush=True)
        result = main()
        return 0 if result is None else result
    lease_value = os.environ.get("VLM_BENCHMARK_GPU_QUEUE_FD")
    if lease_value is not None:
        try:
            lease_fd = int(lease_value)
            lease_path = Path(os.readlink(f"/proc/self/fd/{lease_fd}")).resolve()
            if lease_path == (DEFAULT_QUEUE_DIR / "gpu-0.lock").resolve():
                # The service passes its open file description, keeping the
                # kernel lock alive if the service itself exits unexpectedly.
                fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"[gpu-queue] {benchmark}: using inherited GPU lease", flush=True)
                result = main()
                return 0 if result is None else result
        except (OSError, ValueError):
            pass
    with gpu_queue(benchmark):
        result = main()
    return 0 if result is None else result
