#!/usr/bin/env python3
"""Durable, non-metric launcher for the locked PointMaze Wave 3 resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/dinowm_pointmaze_wave3/formal"
DETACHED = FORMAL / "detached_runs"
PYTHON = ROOT / "outputs/dinowm_native_pusht_audit_v2/venv/bin/python"
RUNNER = ROOT / "scripts/run_dinowm_pointmaze_wave3.py"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
            "sha256": digest(path)}


def worker(run_id: str) -> int:
    directory = DETACHED / run_id
    status_path = directory / "status.json"
    log_path = directory / "stdout_stderr.log"
    command = [str(PYTHON), str(RUNNER), "--formal", "--execute", "--resume"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "2"
    started = time.time()
    with log_path.open("x", buffering=1) as log:
        child = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, text=True)
        atomic_json(status_path, {
            "schema": "dinowm_pointmaze_detached_status_v1",
            "status": "running", "run_id": run_id,
            "worker_pid": os.getpid(), "child_pid": child.pid,
            "worker_started_unix": started, "command": command,
            "cuda_visible_devices": "2", "log_path": str(log_path.relative_to(ROOT)),
        })
        return_code = child.wait()
    completed = time.time()
    terminal = {
        "schema": "dinowm_pointmaze_detached_exit_v1",
        "status": "completed_zero" if return_code == 0 else "exited_nonzero",
        "run_id": run_id, "worker_pid": os.getpid(),
        "child_pid": child.pid, "return_code": return_code,
        "worker_started_unix": started, "completed_unix": completed,
        "elapsed_seconds": completed - started,
        "log": artifact(log_path),
        "progress": artifact(FORMAL / "progress.json"),
        "summary": artifact(FORMAL / "summary.json"),
        "formal_stop_receipt": artifact(FORMAL / "formal_stop_receipt.json"),
        "precarrier_stop_receipt": artifact(FORMAL / "stop_receipt.json"),
    }
    atomic_json(directory / "exit_receipt.json", terminal)
    current = json.loads(status_path.read_text())
    current.update({"status": terminal["status"], "return_code": return_code,
                    "completed_unix": completed,
                    "exit_receipt": str((directory / "exit_receipt.json").relative_to(ROOT))})
    atomic_json(status_path, current)
    return int(return_code)


def launch() -> None:
    DETACHED.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) \
        + f"-{time.time_ns() % 1_000_000_000:09d}"
    directory = DETACHED / run_id
    directory.mkdir(exist_ok=False)
    bootstrap_log = (directory / "bootstrap.log").open("x")
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__).resolve()), "--worker", run_id],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=bootstrap_log,
        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    bootstrap_log.close()
    receipt = {
        "schema": "dinowm_pointmaze_detached_launch_v1",
        "status": "launched", "run_id": run_id,
        "detached_worker_pid": process.pid, "launcher_pid": os.getpid(),
        "launched_unix": time.time(),
        "start_new_session": True, "stdin": "DEVNULL",
        "directory": str(directory.relative_to(ROOT)),
    }
    atomic_json(directory / "launch_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker))
    launch()
