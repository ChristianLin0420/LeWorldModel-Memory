#!/usr/bin/env python3
"""Durably finalize and independently verify the detached PointMaze run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/dinowm_pointmaze_wave3/formal"
DETACHED = FORMAL / "detached_runs"
PYTHON = ROOT / "outputs/dinowm_native_pusht_audit_v2/venv/bin/python"
VERIFY = ROOT / "scripts/verify_dinowm_pointmaze_wave3.py"
LOCATIONS = ROOT / "scripts/audit_dinowm_pointmaze_final_locations.py"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.relative_to(ROOT)),
        "size": path.stat().st_size,
        "sha256": digest(path),
    }


def update_status(path: Path, **fields: Any) -> None:
    current = json.loads(path.read_text()) if path.is_file() else {
        "schema": "dinowm_pointmaze_wave3_finalizer_status_v1"
    }
    current.update(fields)
    atomic_json(path, current)


def worker(run_id: str) -> int:
    directory = DETACHED / run_id
    upstream_status = directory / "status.json"
    status_path = directory / "finalizer_status.json"
    log_path = directory / "finalizer.log"
    started = time.time()
    update_status(
        status_path, status="waiting_for_formal_worker", run_id=run_id,
        finalizer_pid=os.getpid(), started_unix=started,
        upstream_status=str(upstream_status.relative_to(ROOT)),
    )
    try:
        while True:
            if not upstream_status.is_file():
                raise RuntimeError(f"missing upstream status: {upstream_status}")
            upstream = json.loads(upstream_status.read_text())
            state = upstream.get("status")
            if state == "completed_zero":
                break
            if state == "exited_nonzero":
                raise RuntimeError(
                    f"formal worker exited nonzero: {upstream.get('return_code')}")
            if state != "running":
                raise RuntimeError(f"unknown formal worker status: {state!r}")
            time.sleep(30)

        upstream_exit_path = directory / "exit_receipt.json"
        upstream_exit = json.loads(upstream_exit_path.read_text())
        if upstream_exit.get("return_code") != 0:
            raise RuntimeError("formal exit receipt does not report zero")
        if (FORMAL / "formal_stop_receipt.json").exists() \
                or (FORMAL / "stop_receipt.json").exists():
            raise RuntimeError("formal stop receipt exists")

        update_status(status_path, status="running_independent_verifier")
        with log_path.open("x", buffering=1) as log:
            verified = subprocess.run(
                [str(PYTHON), str(VERIFY)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, text=True,
                check=False)
            if verified.returncode != 0:
                raise RuntimeError(
                    f"independent verifier exited {verified.returncode}")
            update_status(status_path, status="auditing_final_locations")
            located = subprocess.run(
                [str(PYTHON), str(LOCATIONS)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, text=True,
                check=False)
            if located.returncode != 0:
                raise RuntimeError(
                    f"final-location audit exited {located.returncode}")

        receipt = {
            "schema": "dinowm_pointmaze_wave3_finalization_receipt_v1",
            "status": "complete_verified",
            "run_id": run_id,
            "finalizer_pid": os.getpid(),
            "started_unix": started,
            "completed_unix": time.time(),
            "upstream_exit_receipt": artifact(upstream_exit_path),
            "formal_summary": artifact(FORMAL / "summary.json"),
            "carrier_summary": artifact(FORMAL / "carrier_summary.json"),
            "external_use_summary": artifact(
                FORMAL / "external_use_summary.json"),
            "verification": artifact(FORMAL / "verification.json"),
            "final_location_index": artifact(
                FORMAL / "final_location_index.json"),
            "finalizer_log": artifact(log_path),
        }
        receipt_path = directory / "finalization_receipt.json"
        atomic_json(receipt_path, receipt)
        update_status(
            status_path, status="complete_verified",
            completed_unix=receipt["completed_unix"],
            receipt=str(receipt_path.relative_to(ROOT)),
        )
        return 0
    except Exception as error:  # terminal receipt must survive any finalizer error
        failed = {
            "schema": "dinowm_pointmaze_wave3_finalization_failure_v1",
            "status": "failed",
            "run_id": run_id,
            "finalizer_pid": os.getpid(),
            "started_unix": started,
            "failed_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "upstream_status": artifact(upstream_status),
            "finalizer_log": artifact(log_path),
        }
        failure_path = directory / "finalization_failure.json"
        atomic_json(failure_path, failed)
        update_status(
            status_path, status="failed", failed_unix=failed["failed_unix"],
            failure=str(failure_path.relative_to(ROOT)),
            error=failed["error"],
        )
        return 1


def launch(run_id: str) -> None:
    directory = DETACHED / run_id
    if not (directory / "status.json").is_file():
        raise RuntimeError(f"unknown detached run: {run_id}")
    if (directory / "finalizer_launch_receipt.json").exists():
        raise RuntimeError("refusing to launch a second finalizer")
    bootstrap_log = (directory / "finalizer_bootstrap.log").open("x")
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__).resolve()), "--worker", run_id],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=bootstrap_log,
        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    bootstrap_log.close()
    receipt = {
        "schema": "dinowm_pointmaze_wave3_finalizer_launch_v1",
        "status": "launched",
        "run_id": run_id,
        "finalizer_pid": process.pid,
        "launcher_pid": os.getpid(),
        "launched_unix": time.time(),
        "start_new_session": True,
        "stdin": "DEVNULL",
    }
    atomic_json(directory / "finalizer_launch_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        raise SystemExit(worker(args.run_id))
    launch(args.run_id)
