#!/usr/bin/env python3
"""Fail-closed continuation from development to the formal SAGE-Mem audit.

This coordinator does not interpret outcomes.  It can attach to an already
running development launcher, waits for that exact process to exit, requires
the complete 180-cell development-audit receipt, and then invokes the sealed
``seal -> prepare -> full`` launch stages.  The full launcher performs
post-grid finalization and the independent formal evidence audit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sage_mem_v1 import FORMAL_CONFIRMATION  # noqa: E402
from scripts.sage_mem_v1_spec import (  # noqa: E402
    DEFAULT_SPEC, load_spec, output_root,
)


class SageMemCampaignError(RuntimeError):
    """The campaign cannot advance without weakening a stage boundary."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def campaign_commands(spec_path: Path, *, resume: bool) -> list[list[str]]:
    base = [
        str(ROOT / ".venv/bin/python"),
        "scripts/launch_sage_mem_v1.py",
        "--spec", str(spec_path.resolve()),
        "--execute",
    ]
    result = []
    for stage in ("seal", "prepare", "full"):
        command = [*base, "--stage", stage]
        if resume:
            command.append("--resume")
        if stage == "full":
            command.extend(("--formal-confirmation", FORMAL_CONFIRMATION))
        result.append(command)
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               SageMemCampaignError(
                                   f"non-finite JSON in {label}: {token}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SageMemCampaignError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SageMemCampaignError(f"{label} is not a JSON mapping")
    return value


def _development_cmdline(pid: int) -> str | None:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SageMemCampaignError(
            f"cannot inspect development PID {pid}") from error
    return payload.replace(b"\0", b" ").decode("utf-8", errors="strict")


def _require_development_identity(pid: int) -> None:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise SageMemCampaignError("development PID must be positive")
    command = _development_cmdline(pid)
    if command is None or "scripts/launch_sage_mem_v1.py" not in command \
            or "--stage development" not in command:
        raise SageMemCampaignError(
            f"PID {pid} is not the registered development launcher")


def wait_for_development(pid: int, *, poll_seconds: float = 30.0) -> None:
    if not isinstance(poll_seconds, (int, float)) \
            or isinstance(poll_seconds, bool) or poll_seconds <= 0:
        raise SageMemCampaignError("poll interval must be positive")
    _require_development_identity(pid)
    while True:
        command = _development_cmdline(pid)
        if command is None:
            return
        if "scripts/launch_sage_mem_v1.py" not in command \
                or "--stage development" not in command:
            raise SageMemCampaignError(
                f"development PID {pid} was reused before completion")
        time.sleep(float(poll_seconds))


def validate_development_audit(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = output_root(spec) / "development" / "audit_receipt.json"
    value = _read_json(path, "development audit receipt")
    expected = 5 * 12 * 3
    if value.get("study") != "sage-mem-v1" \
            or value.get("stage") != "development-audit" \
            or value.get("status") != "complete" \
            or value.get("registered_cells_verified") != expected \
            or value.get("formal_execution_started") is not False:
        raise SageMemCampaignError(
            "development audit is absent, incomplete, or crossed formal data")
    selections = value.get("selection_receipts")
    if not isinstance(selections, dict) \
            or set(selections) != set(spec["cohorts"]):
        raise SageMemCampaignError(
            "development audit lacks all five locked selections")
    return value


def _campaign_lock(spec: Mapping[str, Any]):
    root = output_root(spec)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign.lock"
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise SageMemCampaignError(
            "another SAGE-Mem campaign coordinator is active") from error
    return stream


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--development-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    spec = load_spec(args.spec, verify_parent_paths=False)
    commands = campaign_commands(args.spec, resume=args.resume)
    if not args.execute:
        print(_canonical_json({
            "study": "sage-mem-v1",
            "preview": True,
            "development_pid": args.development_pid,
            "commands": commands,
            "outcomes_interpreted": False,
        }))
        return 0
    lock = _campaign_lock(spec)
    try:
        if args.development_pid is not None:
            wait_for_development(
                args.development_pid, poll_seconds=args.poll_seconds)
        validate_development_audit(spec)
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise SageMemCampaignError(
            f"campaign stage failed with return code {error.returncode}") \
            from error
    finally:
        lock.close()
    print(_canonical_json({
        "study": "sage-mem-v1",
        "status": "formal-audit-complete",
        "outcomes_interpreted": False,
        "paper_updated": False,
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SageMemCampaignError as error:
        print(f"SAGE-Mem campaign stopped: {error}", file=sys.stderr)
        raise SystemExit(2) from error

