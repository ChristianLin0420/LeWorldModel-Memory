#!/usr/bin/env python3
"""Closed-world outcome tests for the independent V13 screen auditor."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_cf_hiro_v13_screen as audit


def test_complete_negative_is_a_successful_audit_outcome() -> None:
    status, passed = audit.audit_status(
        artifact_integrity=True, analyzer_consistent=True, scientific_gate=False)
    assert status == "PASS_COMPLETE_NEGATIVE"
    assert passed
    assert audit.audit_exit_code({"passed": passed}) == 0


def test_complete_positive_is_distinct() -> None:
    status, passed = audit.audit_status(
        artifact_integrity=True, analyzer_consistent=True, scientific_gate=True)
    assert status == "PASS_COMPLETE"
    assert passed


def test_missing_protocol_and_artifacts_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = audit.audit(Path(directory))
    assert report["status"] == "FAIL_CLOSED"
    assert not report["passed"]
    assert not report["artifact_integrity_passed"]
    assert report["validated_cells"] == 0
    assert report["errors"]
    assert audit.audit_exit_code(report) == 2


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V13 independent-audit tests passed.")


if __name__ == "__main__":
    main()
