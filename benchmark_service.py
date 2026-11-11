#!/usr/bin/env python3
"""Local API service that queues and runs curated GPU benchmark jobs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
import signal
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SERVICE_DIR = Path(os.environ.get("VLM_BENCHMARK_SERVICE_DIR", ROOT / ".benchmark_service"))
DB_PATH = SERVICE_DIR / "jobs.sqlite3"
DEFAULT_BENCHMARK_PYTHON = ROOT / ".venv" / "bin" / "python"
BENCHMARK_PYTHON = Path(os.environ.get(
    "VLM_BENCHMARK_PYTHON",
    str(DEFAULT_BENCHMARK_PYTHON if DEFAULT_BENCHMARK_PYTHON.is_file() else sys.executable),
))
MAX_REQUEST_BYTES = 64 * 1024
MAX_ARGS = 128
MAX_ARG_LENGTH = 4096
MAX_COMMAND_LENGTH = 8192
GPU_POLL_SECONDS = 5.0
GPU_FREE_USED_GIB = 1.5

# Named benchmark IDs are kept for older agents. New agents can submit a command.
BENCHMARKS: dict[str, dict[str, Any]] = {
    "qwen-vl": {
        "script": "qwenVL_Bench/benchmark_runners.py",
        "description": "Qwen-VL shared image benchmark suite",
        "result_globs": ("results.json", "report.md"),
    },
    "llava": {
        "script": "llava_Bench/benchmark.py",
        "description": "LLaVA family benchmark suite",
        "result_globs": ("results.json", "report.md"),
    },
    "phi": {
        "script": "phi_Bench/benchmark.py",
        "description": "Phi vision benchmark suite",
        "result_globs": ("results.json", "report.md"),
    },
    "cosmos-edge": {
        "script": "cosmos_Bench/run_benchmark.py",
        "description": "Screened Cosmos Edge benchmark run",
        "result_globs": ("results.json", "report.md"),
    },
    "grounding": {
        "script": "grounding_Bench/benchmark.py",
        "description": "Grounding specialists benchmark suite",
        "result_globs": ("*.json", "*.md"),
    },
    "general-transformers": {
        "script": "general_transformers_Bench/benchmark.py",
        "description": "General Transformers VLM benchmark suite",
        "result_globs": ("results.json", "report.md"),
    },
    "grounding-vision": {
        "script": "grounding_Bench/vision_benchmark.py",
        "description": "Grounding and instance segmentation benchmark",
        "result_globs": ("*.json", "*.md"),
    },
    "grounding-dota-obb": {
        "script": "grounding_Bench/dota_obb_benchmark.py",
        "description": "DOTA oriented bounding box benchmark",
        "result_globs": ("*.json", "*.md"),
    },
    "local-foundation-gguf": {
        "script": "local_foundation_Bench/gguf_benchmark.py",
        "description": "Locally cached Liquid GGUF benchmark suite",
        "result_globs": ("*.json", "*.md"),
    },
    "qwen-image-sgg": {
        "script": "qwenVL_Bench/image_SGG_Bench/runner.py",
        "description": "Qwen image scene graph generation benchmark",
        "result_globs": ("*.json", "*.md"),
    },
    "opal-training": {
        "script": "classification_Bench/run_opal_training.py",
        "description": "OPAL image classification training runs",
        "result_globs": ("*.json", "*.md"),
    },
    "opal-vlm-classification": {
        "script": "classification_Bench/opal_vlm_benchmark.py",
        "description": "OPAL zero-shot VLM classification benchmark",
        "result_globs": ("*.json", "*.md"),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


@contextmanager
def db_connection():
    connection = connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def process_identity(pid: int) -> tuple[int, int, str] | None:
    """Return (process group, start ticks, state) for a live Linux process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = stat[stat.rfind(")") + 2 :].split()
        return int(fields[2]), int(fields[19]), fields[0]
    except (OSError, ValueError, IndexError):
        return None


def boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return None


def acquire_service_lock():
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = (SERVICE_DIR / "service.lock").open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError("another benchmark service instance already owns this queue") from exc
    return lock_file


def process_group_alive(pgid: int) -> bool:
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = path.read_text(encoding="ascii")
            fields = stat[stat.rfind(")") + 2 :].split()
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def find_job_process_group(row: sqlite3.Row) -> int | None:
    """Find a just-launched child if the service crashed before saving its PID."""
    if row["benchmark"] == "command":
        command = json.loads(row["args_json"])
        script = (command[1] if len(command) > 1 else command[0]).encode()
        result_dir = row["result_dir"].encode()
    else:
        script = str(ROOT / BENCHMARKS[row["benchmark"]]["script"]).encode()
        result_dir = row["result_dir"].encode()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = cmdline.read_bytes().split(b"\0")
            if script in arguments and result_dir in arguments:
                pid = int(cmdline.parent.name)
                identity = process_identity(pid)
                if identity and identity[0] == pid:
                    return pid
        except (OSError, ValueError):
            continue
    return None


def stop_tracked_job(row: sqlite3.Row) -> bool:
    pid = row["process_pid"]
    current_boot_id = boot_id()
    if row["process_boot_id"] and row["process_boot_id"] != current_boot_id:
        return True
    group_id = None
    if pid:
        identity = process_identity(pid)
        if identity and identity[0] == pid and identity[1] == row["process_start_ticks"]:
            group_id = pid
    if group_id is None:
        group_id = find_job_process_group(row)
    if group_id is None:
        return True
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 5
    while process_group_alive(group_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_group_alive(group_id):
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + 2
        while process_group_alive(group_id) and time.monotonic() < deadline:
            time.sleep(0.1)
    return not process_group_alive(group_id)


def initialize_database() -> None:
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    with db_connection() as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                benchmark TEXT NOT NULL,
                args_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                exit_code INTEGER,
                result_dir TEXT NOT NULL,
                venv_path TEXT,
                result_files_json TEXT NOT NULL DEFAULT '[]',
                log_path TEXT NOT NULL,
                stderr_path TEXT NOT NULL,
                process_pid INTEGER,
                process_start_ticks INTEGER,
                process_boot_id TEXT,
                error TEXT
            )"""
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
        for column, declaration in (
            ("venv_path", "TEXT"),
            ("stderr_path", "TEXT NOT NULL DEFAULT ''"),
            ("process_pid", "INTEGER"),
            ("process_start_ticks", "INTEGER"),
            ("process_boot_id", "TEXT"),
        ):
            if column not in columns:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")
        # A process cannot safely resume a child benchmark after a service crash.
        running = db.execute("SELECT * FROM jobs WHERE status='running'").fetchall()
        for row in running:
            if not stop_tracked_job(row):
                raise RuntimeError(
                    f"cannot confirm benchmark process group stopped for job {row['id']}; "
                    "service startup is blocked to protect the GPU"
                )
            db.execute(
                """UPDATE jobs SET status='failed', finished_at=?, error=?,
                    exit_code=NULL WHERE id=?""",
                (utc_now(), "service restarted; the active benchmark process was stopped", row["id"]),
            )


def row_to_job(
    row: sqlite3.Row, include_args: bool = False, include_trace: bool = True
) -> dict[str, Any]:
    error_trace = ""
    if include_trace and row["status"] == "failed":
        try:
            error_trace = Path(row["stderr_path"]).read_text(encoding="utf-8", errors="replace")
        except (OSError, TypeError):
            error_trace = ""
        if not error_trace.strip():
            try:
                error_trace = Path(row["log_path"]).read_text(encoding="utf-8", errors="replace")
            except (OSError, TypeError):
                pass
    diagnostic_lines = [
        line for line in error_trace.rstrip().splitlines()
        if line and not line.startswith((
            "command: ", "started: ", "finished: ", "exit_code: ", "[gpu-queue] "
        ))
    ]
    error_message = (diagnostic_lines[-1] if diagnostic_lines else row["error"])
    if row["status"] != "failed":
        error_message = None
    job = {
        "id": row["id"],
        "benchmark": row["benchmark"],
        "status": row["status"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "exit_code": row["exit_code"],
        "process_pid": row["process_pid"],
        "result_dir": row["result_dir"],
        "venv_path": row["venv_path"],
        "result_files": json.loads(row["result_files_json"]),
        "log_url": f"/v1/jobs/{row['id']}/logs",
        "stderr_url": f"/v1/jobs/{row['id']}/stderr",
        "error_message": error_message,
        "error_trace": error_trace if row["status"] == "failed" else None,
    }
    if row["status"] == "queued":
        with db_connection() as db:
            position = db.execute(
                """SELECT COUNT(*) FROM jobs AS pending
                   WHERE pending.status='queued' AND pending.rowid <=
                         (SELECT current.rowid FROM jobs AS current WHERE current.id=?)""",
                (row["id"],),
            ).fetchone()[0]
        job["queue_position"] = position
    if include_args:
        stored_args = json.loads(row["args_json"])
        if row["benchmark"] == "command":
            job["command"] = shlex.join(stored_args)
        else:
            job["args"] = stored_args
    elif row["benchmark"] == "command":
        job["command"] = shlex.join(json.loads(row["args_json"]))
    return job


def find_result_files(benchmark: str, result_dir: Path) -> list[str]:
    found: set[Path] = set()
    if benchmark in BENCHMARKS:
        for pattern in BENCHMARKS[benchmark]["result_globs"]:
            found.update(path for path in result_dir.rglob(pattern) if path.is_file())
    else:
        found.update(
            path for path in result_dir.rglob("*")
            if path.is_file() and path.name not in {"worker.log", "worker.stderr.log"}
        )
    return [str(path.resolve()) for path in sorted(found)]


def create_job(
    benchmark: str | None,
    args: list[str] | None,
    venv_path: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if command is not None:
        if not isinstance(command, str) or not command.strip() or len(command) > MAX_COMMAND_LENGTH:
            raise ValueError(f"command must be a non-empty string of at most {MAX_COMMAND_LENGTH} characters")
        if benchmark is not None or args not in (None, []):
            raise ValueError("provide command or benchmark and args, not both")
        try:
            args = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"invalid command quoting: {exc}") from exc
        if not args:
            raise ValueError("command must contain an executable")
        benchmark = "command"
    else:
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unsupported benchmark; choose one of: {', '.join(BENCHMARKS)} or provide command")
        if args is None:
            args = []
    if not isinstance(args, list) or len(args) > MAX_ARGS:
        raise ValueError(f"args must be a list with at most {MAX_ARGS} strings")
    if any(not isinstance(arg, str) or len(arg) > MAX_ARG_LENGTH for arg in args):
        raise ValueError("each argument must be a string of at most 4096 characters")
    forbidden = {"--output-dir", "--list-models", "--help", "-h"}
    if benchmark != "command" and any(
        arg in forbidden or arg.startswith("--output-dir=") for arg in args
    ):
        raise ValueError("output directory and inventory/help options are managed by the service")
    if venv_path is not None:
        if not isinstance(venv_path, str) or not venv_path.strip():
            raise ValueError("venv_path must be a non-empty path string")
        candidate = Path(venv_path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(ROOT):
            raise ValueError("venv_path must be inside this project checkout")
        if not (candidate / "bin" / "python").is_file():
            raise ValueError(f"venv_path has no bin/python interpreter: {candidate}")
        venv_path = str(candidate)
        if benchmark == "command":
            executable = Path(args[0]).name.lower()
            if executable.startswith("python"):
                args[0] = str(candidate / "bin" / "python")
            elif args[0].endswith(".py"):
                args.insert(0, str(candidate / "bin" / "python"))

    job_id = uuid.uuid4().hex
    result_dir = (SERVICE_DIR / "runs" / job_id).resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    log_path = result_dir / "worker.log"
    stderr_path = result_dir / "worker.stderr.log"
    created_at = utc_now()
    with db_connection() as db:
        db.execute(
            """INSERT INTO jobs
                (id, benchmark, args_json, status, created_at, result_dir, venv_path, log_path, stderr_path)
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?)""",
            (job_id, benchmark, json.dumps(args), created_at, str(result_dir), venv_path,
             str(log_path), str(stderr_path)),
        )
    return get_job(job_id)


def used_vram_gib() -> float:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi could not read used VRAM: {result.stderr.strip()}")
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    return values[0] / 1024


def get_job(job_id: str) -> dict[str, Any] | None:
    with db_connection() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row_to_job(row, include_args=True) if row else None


class JobWorker:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._stopping = False
        self._process: subprocess.Popen[bytes] | None = None
        self._thread = threading.Thread(target=self._run, name="benchmark-job-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process_group_alive(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._thread.join(timeout=10)

    def _next_job(self) -> sqlite3.Row | None:
        with db_connection() as db:
            return db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()

    def _wait(self, seconds: float) -> None:
        with self._condition:
            if not self._stopping:
                self._condition.wait(timeout=seconds)

    def _fail_queued_job(self, row: sqlite3.Row, exc: Exception) -> None:
        trace = traceback.format_exc()
        try:
            Path(row["stderr_path"]).write_text(trace, encoding="utf-8")
        except OSError:
            pass
        with db_connection() as db:
            db.execute(
                """UPDATE jobs SET status='failed', finished_at=?, error=?
                   WHERE id=? AND status='queued'""",
                (utc_now(), f"{type(exc).__name__}: {exc}", row["id"]),
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
            row = self._next_job()
            if row is None:
                self._wait(1.0)
                continue

            try:
                last_reported_used: float | None = None
                while True:
                    used_gib = used_vram_gib()
                    if used_gib < GPU_FREE_USED_GIB:
                        break
                    if last_reported_used is None or abs(used_gib - last_reported_used) >= 0.25:
                        print(
                            f"[benchmark-queue] job {row['id']} waits for VRAM: "
                            f"free threshold {GPU_FREE_USED_GIB:.1f} GiB, used {used_gib:.1f} GiB",
                            flush=True,
                        )
                        last_reported_used = used_gib
                    self._wait(GPU_POLL_SECONDS)
                    with self._condition:
                        if self._stopping:
                            return
                self._execute_with_slot(row)
            except Exception as exc:
                self._fail_queued_job(row, exc)
                print(f"[benchmark-queue] job {row['id']} failed before launch: {exc}", flush=True)

    def _execute_with_slot(self, row: sqlite3.Row) -> None:
        benchmark = row["benchmark"]
        result_dir = Path(row["result_dir"])
        if benchmark == "command":
            command = [
                argument.replace("{result_dir}", str(result_dir))
                for argument in json.loads(row["args_json"])
            ]
        else:
            script = ROOT / BENCHMARKS[benchmark]["script"]
            python_path = (Path(row["venv_path"]) / "bin" / "python") if row["venv_path"] else BENCHMARK_PYTHON
            command = [
                str(python_path), str(script), *json.loads(row["args_json"]),
                "--output-dir", str(result_dir),
            ]
            python_path = Path(command[0])
        env = os.environ.copy()
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env["VLM_BENCHMARK_RESULT_DIR"] = str(result_dir)
        env["VLM_BENCHMARK_SCHEDULER_OWNS_GPU"] = "1"
        env.pop("VLM_BENCHMARK_GPU_QUEUE_FD", None)
        exit_code: int | None = None
        error: str | None = None
        process: subprocess.Popen[bytes] | None = None
        try:
            with open(row["log_path"], "ab", buffering=0) as log_file, open(
                row["stderr_path"], "ab", buffering=0
            ) as stderr_file:
                log_file.write((f"command: {json.dumps(command)}\nstarted: {utc_now()}\n\n").encode())
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
                with self._condition:
                    self._process = process
                identity = process_identity(process.pid)
                with db_connection() as db:
                    db.execute(
                        """UPDATE jobs SET status='running', started_at=?, process_pid=?,
                           process_start_ticks=?, process_boot_id=?, error=NULL
                           WHERE id=? AND status='queued'""",
                        (utc_now(), process.pid, identity[1] if identity else None,
                         boot_id(), row["id"]),
                    )
                with self._condition:
                    stopping = self._stopping
                if stopping:
                    process.terminate()
                exit_code = process.wait()
                with self._condition:
                    self._process = None
                with db_connection() as db:
                    db.execute(
                        """UPDATE jobs SET process_pid=NULL, process_start_ticks=NULL,
                           process_boot_id=NULL WHERE id=?""",
                        (row["id"],),
                    )
                log_file.write(f"\nfinished: {utc_now()}\nexit_code: {exit_code}\n".encode())
            if exit_code != 0:
                try:
                    error = Path(row["stderr_path"]).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    error = ""
                if not error.strip():
                    try:
                        error = Path(row["log_path"]).read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        error = ""
                if not error.strip():
                    error = f"Benchmark process exited with code {exit_code}."
        except Exception as exc:  # persist launch errors for API clients
            error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                if process_group_alive(process.pid):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                with self._condition:
                    self._process = None
                try:
                    with db_connection() as db:
                        db.execute(
                            """UPDATE jobs SET process_pid=NULL, process_start_ticks=NULL,
                               process_boot_id=NULL WHERE id=?""",
                            (row["id"],),
                        )
                except sqlite3.Error:
                    pass
            Path(row["stderr_path"]).write_text(traceback.format_exc(), encoding="utf-8")

        result_files = find_result_files(benchmark, result_dir)
        status = "succeeded" if exit_code == 0 else "failed"
        with db_connection() as db:
            db.execute(
                """UPDATE jobs SET status=?, finished_at=?, exit_code=?,
                    result_files_json=?, error=? WHERE id=?""",
                (status, utc_now(), exit_code, json.dumps(result_files), error, row["id"]),
            )


class Handler(BaseHTTPRequestHandler):
    server_version = "VLM-Benchmark-Service/1.0"

    @property
    def worker(self) -> JobWorker:
        return self.server.worker  # type: ignore[attr-defined]

    def _json(self, status: int, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _path(self) -> str:
        return urlparse(self.path).path.rstrip("/") or "/"

    def do_GET(self) -> None:  # noqa: N802
        path = self._path()
        if path == "/healthz":
            self._json(200, {"status": "ok", "queue": "ready"})
            return
        if path == "/v1/benchmarks":
            self._json(200, {"benchmarks": [
                {"id": key, "description": value["description"], "script": value["script"]}
                for key, value in BENCHMARKS.items()
            ]})
            return
        if path == "/v1/jobs":
            status_filter = urlparse(self.path).query
            selected = None
            if status_filter.startswith("status="):
                selected = status_filter.removeprefix("status=")
            query = "SELECT * FROM jobs"
            params: tuple[Any, ...] = ()
            if selected:
                if selected not in {"queued", "running", "succeeded", "failed"}:
                    self._json(400, {"error": "invalid status filter"})
                    return
                query += " WHERE status=?"
                params = (selected,)
            query += " ORDER BY rowid DESC"
            with db_connection() as db:
                rows = db.execute(query, params).fetchall()
            self._json(200, {"jobs": [row_to_job(row, include_trace=False) for row in rows]})
            return
        match = re.fullmatch(r"/v1/jobs/([0-9a-f]{32})(/logs|/stderr)?", path)
        if match:
            job_id, logs_route = match.groups()
            job = get_job(job_id)
            if job is None:
                self._json(404, {"error": "job not found"})
                return
            if logs_route:
                filename = "worker.stderr.log" if logs_route == "/stderr" else "worker.log"
                log_path = SERVICE_DIR / "runs" / job_id / filename
                try:
                    payload = log_path.read_bytes()
                except FileNotFoundError:
                    payload = b""
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self._json(200, {"job": job})
            return
        self._json(404, {"error": "route not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self._path() != "/v1/jobs":
            self._json(404, {"error": "route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._json(413 if length > MAX_REQUEST_BYTES else 400,
                       {"error": f"request body must be 1 to {MAX_REQUEST_BYTES} bytes"})
            return
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            unknown = set(request) - {"benchmark", "args", "venv_path", "command"}
            if unknown:
                raise ValueError(f"unknown request field(s): {', '.join(sorted(unknown))}")
            job = create_job(
                request.get("benchmark"), request.get("args", []), request.get("venv_path"),
                request.get("command"),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self.worker.wake()
        self._json(202, {"job": job, "status_url": f"/v1/jobs/{job['id']}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[benchmark-api] {self.client_address[0]} {format % args}", flush=True)


def serve(host: str, port: int) -> None:
    service_lock = acquire_service_lock()
    worker = None
    server = None
    try:
        initialize_database()
        worker = JobWorker()
        server = ThreadingHTTPServer((host, port), Handler)
        server.worker = worker  # type: ignore[attr-defined]
        worker.start()
        print(f"Benchmark job API listening on http://{host}:{port}", flush=True)
        print("POST /v1/jobs to enqueue; poll GET /v1/jobs/{id} for status and result paths", flush=True)
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping benchmark job service", flush=True)
    finally:
        if server is not None:
            server.server_close()
        if worker is not None:
            worker.stop()
        service_lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", nargs="?", choices=("serve",), default="serve")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; defaults to loopback only")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
