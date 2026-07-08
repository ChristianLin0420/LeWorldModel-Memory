#!/usr/bin/env python3
"""Detached finalizer: await Wave 2 formal completion, then verify it."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/dinowm_wave2_spatial_carrier_v1_1/formal"
FORMAL_LAUNCH = FORMAL / "durable_resume_launch.json"
FORMAL_EXIT = FORMAL / "durable_resume_exit.json"
LAUNCH = FORMAL / "durable_finalizer_launch.json"
STATUS = FORMAL / "durable_finalizer_status.json"
EXIT = FORMAL / "durable_finalizer_exit.json"
LOG = FORMAL / "durable_verifier.log"
PROTOCOL = "b1af10f4bc243b9c22aee29e7f2c420905c3f4f38e45c6ea4d9457f819205178"
VERIFY_COMMAND = [
    str(ROOT / "outputs/dinowm_native_pusht_audit_v2/venv/bin/python"),
    str(ROOT / "scripts/verify_dinowm_wave2_spatial_carrier_v1_1.py"),
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


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def status(stage: str, **extra: object) -> None:
    atomic_json(STATUS, {
        "schema": "dinowm_wave2_durable_finalizer_status_v1",
        "protocol_sha256": PROTOCOL,
        "finalizer_pid": os.getpid(),
        "finalizer_ppid": os.getppid(),
        "finalizer_session_id": os.getsid(0),
        "stage": stage,
        "updated_at": timestamp(),
        "paper_modified_by_finalizer": False,
        **extra,
    })


def main() -> int:
    if LAUNCH.exists() or EXIT.exists() or LOG.exists():
        raise RuntimeError("refusing to overwrite durable finalizer artifacts")
    formal_launch = json.loads(FORMAL_LAUNCH.read_text())
    child_pid = int(formal_launch["child_pid"])
    supervisor_pid = int(formal_launch["supervisor_pid"])
    launch = {
        "schema": "dinowm_wave2_durable_finalizer_launch_v1",
        "status": "waiting_for_formal",
        "started_at": timestamp(),
        "protocol_sha256": PROTOCOL,
        "finalizer_pid": os.getpid(),
        "finalizer_ppid": os.getppid(),
        "finalizer_session_id": os.getsid(0),
        "formal_supervisor_pid": supervisor_pid,
        "formal_child_pid": child_pid,
        "formal_exit_receipt": str(FORMAL_EXIT.relative_to(ROOT)),
        "verify_command": VERIFY_COMMAND,
        "verifier_log": str(LOG.relative_to(ROOT)),
        "paper_modified_by_finalizer": False,
    }
    atomic_json(LAUNCH, launch)
    status("waiting_for_formal", formal_child_alive=process_alive(child_pid),
           formal_supervisor_alive=process_alive(supervisor_pid))

    while not FORMAL_EXIT.is_file():
        child_alive = process_alive(child_pid)
        supervisor_alive = process_alive(supervisor_pid)
        if not child_alive and not supervisor_alive:
            receipt = {
                "schema": "dinowm_wave2_durable_finalizer_exit_v1",
                "status": "formal_receipt_missing",
                "completed_at": timestamp(),
                "protocol_sha256": PROTOCOL,
                "finalizer_pid": os.getpid(),
                "formal_child_pid": child_pid,
                "formal_supervisor_pid": supervisor_pid,
                "return_code": 2,
                "paper_modified_by_finalizer": False,
            }
            atomic_json(EXIT, receipt)
            status("failed", reason="formal processes exited without receipt")
            return 2
        time.sleep(10)

    formal_exit = json.loads(FORMAL_EXIT.read_text())
    formal_ok = (formal_exit.get("status") == "complete"
                 and int(formal_exit.get("return_code", -1)) == 0)
    if not formal_ok:
        receipt = {
            "schema": "dinowm_wave2_durable_finalizer_exit_v1",
            "status": "formal_nonzero_no_verification",
            "completed_at": timestamp(),
            "protocol_sha256": PROTOCOL,
            "finalizer_pid": os.getpid(),
            "formal_exit_sha256": digest(FORMAL_EXIT),
            "formal_return_code": formal_exit.get("return_code"),
            "return_code": 3,
            "paper_modified_by_finalizer": False,
        }
        atomic_json(EXIT, receipt)
        status("failed", reason="formal execution was not complete")
        return 3

    status("running_independent_verifier",
           formal_exit_sha256=digest(FORMAL_EXIT))
    with LOG.open("x") as stream:
        verification = subprocess.run(
            VERIFY_COMMAND, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, text=True,
            check=False)
    verification_path = FORMAL / "verification.json"
    verified = False
    if verification.returncode == 0 and verification_path.is_file():
        verified = bool(json.loads(verification_path.read_text()).get(
            "preoutcome_numerical_amendment_verified"))
    return_code = 0 if verification.returncode == 0 and verified else 4
    receipt = {
        "schema": "dinowm_wave2_durable_finalizer_exit_v1",
        "status": "complete_verified" if return_code == 0 else "verification_failed",
        "completed_at": timestamp(),
        "protocol_sha256": PROTOCOL,
        "finalizer_pid": os.getpid(),
        "formal_exit_sha256": digest(FORMAL_EXIT),
        "formal_return_code": formal_exit.get("return_code"),
        "verifier_return_code": verification.returncode,
        "verified": verified,
        "verification_sha256": digest(verification_path),
        "summary_sha256": digest(FORMAL / "summary.json"),
        "verifier_log_sha256": digest(LOG),
        "return_code": return_code,
        "paper_modified_by_finalizer": False,
    }
    atomic_json(EXIT, receipt)
    status("complete" if return_code == 0 else "failed", **receipt)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
