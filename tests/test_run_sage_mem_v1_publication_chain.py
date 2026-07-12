from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_sage_mem_v1_publication_chain as chain


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chain.canonical_json(value), encoding="utf-8")


def test_preview_reads_and_writes_nothing(capsys: pytest.CaptureFixture[str]) \
        -> None:
    assert chain.main([]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview"] is True
    assert preview["no_files_read"] is True
    assert preview["no_files_written"] is True
    assert preview["no_outcomes_read"] is True
    assert preview["requires_complete_phase_a_cells"] == 600


def test_execute_refuses_incomplete_phase_a_grid(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    fake_python = tmp_path / ".venv/bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    fake_python.chmod(0o755)
    monkeypatch.setattr(chain, "ROOT", tmp_path)
    monkeypatch.setattr(chain, "PYTHON", fake_python)

    assert chain.main(["--execute", "--root", str(tmp_path)]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    assert "Phase-A grid is incomplete" in output["reason"]


def test_operator_hashes_bind_completed_closeout_files(tmp_path: Path) -> None:
    study = tmp_path / "outputs/sage_mem_v1"
    for index in range(600):
        manifest = study / "cells" / f"cohort-{index}" / "manifest.json"
        _write_json(manifest, {"status": "complete-label-free"})

    for relative, payload in (
            ("protocol_lock.json", {"sealed": True}),
            ("raw_context_phase_a/summary.json", {"raw": True}),
            ("formal_preparation/custody/registry.json", {"labels": True}),
            ("formal_preparation/execution_decks/registry.json",
             {"execution": True})):
        _write_json(study / relative, payload)
    verifier = tmp_path / "scripts/audit_sage_mem_v1_phase_b_reproduction.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_text("# verifier\n", encoding="utf-8")

    label_registry = study / "formal_preparation/custody/registry.json"
    summary = {
        "schema": "sage_mem_v1_formal_finalizer_v1",
        "status": "complete",
        "phase_a_cells": 600,
        "finalized_cells": 600,
        "phase_a_grid_sha256": "a" * 64,
        "label_registry_sha256": chain.sha256_file(label_registry),
        "finalized_cells_sha256": "b" * 64,
    }
    _write_json(study / "formal_finalized/summary.json", summary)
    _write_json(study / "formal_audit/report.json", {
        "schema": "sage_mem_v1_formal_evidence_audit_v1",
        "status": "complete",
        "phase_a_cells_verified": 600,
        "finalized_cells_verified": 600,
        "phase_a_grid_sha256": "a" * 64,
    })

    hashes = chain.operator_hashes(tmp_path)
    assert hashes["phase_a_grid"] == "a" * 64
    assert hashes["finalized_cells"] == "b" * 64
    assert hashes["label_registry"] == chain.sha256_file(label_registry)
    assert hashes["formal_report"] == chain.sha256_file(
        study / "formal_audit/report.json")
