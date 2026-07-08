#!/usr/bin/env python3
"""Durably supervise the sealed Wave 2 v1.1 resume and persist its exit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/dinowm_wave2_spatial_carrier_v1_1/formal"
LOG = FORMAL / "durable_resume.log"
LAUNCH = FORMAL / "durable_resume_launch.json"
EXIT = FORMAL / "durable_resume_exit.json"
PROTOCOL = "b1af10f4bc243b9c22aee29e7f2c420905c3f4f38e45c6ea4d9457f819205178"
COMMAND = [
    str(ROOT / "outputs/dinowm_native_pusht_audit_v2/venv/bin/python"),
    str(ROOT / "scripts/run_dinowm_wave2_spatial_carrier_v1_1.py"),
    "--config", str(ROOT / "configs/dinowm_wave2_spatial_carrier_v1_1.yaml"),
    "--formal", "--resume", "--execute",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("durable Wave 2 resume requires CUDA_VISIBLE_DEVICES=1")
    if LAUNCH.exists() or EXIT.exists():
        raise RuntimeError("refusing to overwrite durable resume receipts")
    launch = {
        "schema": "dinowm_wave2_durable_resume_launch_v1",
        "status": "launching",
        "started_at": timestamp(),
        "protocol_sha256": PROTOCOL,
        "supervisor_pid": os.getpid(),
        "supervisor_ppid": os.getppid(),
        "supervisor_session_id": os.getsid(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": COMMAND,
        "cwd": str(ROOT),
        "log": str(LOG.relative_to(ROOT)),
        "progress_sha256_before": digest(FORMAL / "progress.json"),
    }
    atomic_json(LAUNCH, launch)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "1"
    with LOG.open("x", buffering=1) as log:
        log.write(json.dumps({"event": "supervisor_start", **launch},
                             sort_keys=True) + "\n")
        process = subprocess.Popen(
            COMMAND, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, text=True,
            start_new_session=True)
        launch.update({
            "status": "running",
            "child_pid": process.pid,
            "child_session_id": os.getsid(process.pid),
        })
        atomic_json(LAUNCH, launch)
        log.write(json.dumps({"event": "child_started",
                              "child_pid": process.pid,
                              "child_session_id": os.getsid(process.pid),
                              "at": timestamp()}, sort_keys=True) + "\n")
        return_code = process.wait()
        log.write(json.dumps({"event": "child_exit",
                              "return_code": return_code,
                              "at": timestamp()}, sort_keys=True) + "\n")
    receipt = {
        "schema": "dinowm_wave2_durable_resume_exit_v1",
        "status": "complete" if return_code == 0 else "nonzero_exit",
        "completed_at": timestamp(),
        "protocol_sha256": PROTOCOL,
        "supervisor_pid": os.getpid(),
        "child_pid": launch["child_pid"],
        "return_code": return_code,
        "progress_sha256_after": digest(FORMAL / "progress.json"),
        "summary_sha256": digest(FORMAL / "summary.json"),
        "formal_stop_receipt_sha256": digest(FORMAL / "stop_receipt.json"),
        "log_sha256": digest(LOG),
    }
    atomic_json(EXIT, receipt)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
