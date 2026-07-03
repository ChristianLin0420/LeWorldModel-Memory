#!/usr/bin/env python3
"""Bind a manually verified two-interruption V18 receipt to final result files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import v18_release_common as common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--record", type=Path, required=True,
        help="JSON object containing only the manually verified interruptions array",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--log-root", type=Path, required=True,
        help="private log root used to verify every log named in the manual record",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite restart audit {output}")
    bundle = common.load_complete_bundle(args.root.resolve(), require_failure=True)
    record = common.read_json(args.record.resolve())
    if not isinstance(record, dict) or set(record) != {"interruptions"}:
        raise common.ReleaseValidationError(
            "manual record must be an object containing only 'interruptions'"
        )
    audit = {
        "schema_version": 2,
        "scope": "v18_process_level_restart_audit",
        "protocol_sha256": bundle["hashes"]["confirmation_protocol.json"],
        "commands_sha256": bundle["protocol"]["commands_sha256"],
        "interruptions": record["interruptions"],
        "final_bindings": {
            "completed_valid_cells": 200,
            "summary_sha256": bundle["hashes"]["confirmation_summary.json"],
            "runs_sha256": bundle["hashes"]["confirmation_runs.json"],
            "attempts_sha256": bundle["hashes"]["confirmation_attempts.json"],
            "analysis_sha256": bundle["hashes"]["confirmation_analysis.json"],
        },
        "ledger_limitation": (
            "The runner attempts ledger records terminal subprocess returns only; "
            "process-killed attempts are represented by this separately verified receipt."
        ),
    }
    # Validate counts, paths, final bindings, and private log bytes in staging;
    # an invalid record is never briefly visible at the requested output path.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v18-restart-", dir=output.parent))
    try:
        candidate = staging / output.name
        common.atomic_write_json(candidate, audit)
        common.validate_restart_audit(
            candidate, bundle, log_root=args.log_root.resolve()
        )
        os.replace(candidate, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({
        "output": str(output),
        "sha256": common.sha256(output),
        "interruptions": len(audit["interruptions"]),
        "final_analysis_sha256": audit["final_bindings"]["analysis_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
