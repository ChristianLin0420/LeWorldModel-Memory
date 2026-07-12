"""Run the guarded SAGE-Mem v1 post-closeout publication artifact chain.

This driver starts only after Phase A is complete, the sealed finalizer has
published ``formal_finalized/summary.json``, and the formal auditor has
published ``formal_audit/report.json``.  It does not edit the manuscript
scientific narrative.  Its job is to bind the completed report through the
independent Phase-B receipt, generated claim ledger, generated figures, and
the paper binding auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
STUDY_ROOT = ROOT / "outputs/sage_mem_v1"
REPORT = STUDY_ROOT / "formal_audit/report.json"
FINALIZER_SUMMARY = STUDY_ROOT / "formal_finalized/summary.json"
PHASE_B_RECEIPT = STUDY_ROOT / "receipts/phase_b/reproduction_receipt.json"
LEDGER_JSON = ROOT / "paper_a/generated_results/sage_mem_v1_claim_ledger.json"
LEDGER_TEX = ROOT / "paper_a/generated_results/sage_mem_v1_claim_ledger.tex"
PLOT_MANIFEST = ROOT / "paper_a/generated_results/sage_mem_v1_plot_manifest.json"
BINDING_RECEIPT = ROOT / "outputs/paper_a_sage_mem_binding/receipt.json"


class PublicationChainError(RuntimeError):
    """The post-closeout publication chain cannot safely proceed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationChainError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationChainError(f"cannot read {label}: {path}") from error
    require(isinstance(value, dict), f"{label} root is not a mapping")
    return value


def count_phase_a_cells(root: Path) -> int:
    cells = root / "cells"
    if not cells.is_dir():
        return 0
    return sum(1 for _ in cells.rglob("manifest.json"))


def operator_hashes(root: Path) -> dict[str, str]:
    study = root / "outputs/sage_mem_v1"
    summary_path = study / "formal_finalized/summary.json"
    report_path = study / "formal_audit/report.json"
    protocol_lock = study / "protocol_lock.json"
    raw_context_summary = study / "raw_context_phase_a/summary.json"
    label_registry = study / "formal_preparation/custody/registry.json"
    execution_registry = (
        study / "formal_preparation/execution_decks/registry.json")
    finalizer_summary = summary_path
    verifier = root / "scripts/audit_sage_mem_v1_phase_b_reproduction.py"

    cells = count_phase_a_cells(study)
    require(cells == 600,
            f"Phase-A grid is incomplete: {cells}/600 cells are present")
    summary = read_json(summary_path, "formal finalizer summary")
    report = read_json(report_path, "formal audit report")
    require(summary.get("schema") == "sage_mem_v1_formal_finalizer_v1"
            and summary.get("status") == "complete"
            and summary.get("phase_a_cells") == 600
            and summary.get("finalized_cells") == 600,
            "formal finalizer summary is incomplete")
    require(report.get("schema") == "sage_mem_v1_formal_evidence_audit_v1"
            and report.get("status") == "complete"
            and report.get("phase_a_cells_verified") == 600
            and report.get("finalized_cells_verified") == 600,
            "formal audit report is incomplete")
    require(report.get("phase_a_grid_sha256")
            == summary.get("phase_a_grid_sha256"),
            "formal report and finalizer summary bind different Phase-A grids")
    for path, label in (
            (protocol_lock, "protocol lock"),
            (raw_context_summary, "raw-context summary"),
            (label_registry, "label registry"),
            (execution_registry, "execution registry"),
            (finalizer_summary, "finalizer summary"),
            (verifier, "Phase-B verifier")):
        require(path.is_file() and not path.is_symlink(),
                f"missing or unsafe {label}: {path}")
    require(sha256_file(label_registry)
            == summary.get("label_registry_sha256"),
            "label-registry hash differs from finalizer summary")
    return {
        "verifier_source": sha256_file(verifier),
        "protocol_lock": sha256_file(protocol_lock),
        "phase_a_grid": str(summary["phase_a_grid_sha256"]),
        "raw_context_summary": sha256_file(raw_context_summary),
        "label_registry": sha256_file(label_registry),
        "execution_registry": sha256_file(execution_registry),
        "finalizer_summary": sha256_file(finalizer_summary),
        "finalized_cells": str(summary["finalized_cells_sha256"]),
        "formal_report": sha256_file(report_path),
    }


def run_command(command: list[str], label: str) -> Mapping[str, Any]:
    completed = subprocess.run(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    require(completed.returncode == 0,
            f"{label} failed with exit {completed.returncode}:\n"
            + completed.stdout[-4000:])
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phase-b-receipt", type=Path,
                        default=PHASE_B_RECEIPT)
    parser.add_argument("--ledger-json", type=Path, default=LEDGER_JSON)
    parser.add_argument("--ledger-tex", type=Path, default=LEDGER_TEX)
    parser.add_argument("--plot-manifest", type=Path, default=PLOT_MANIFEST)
    parser.add_argument("--binding-receipt", type=Path,
                        default=BINDING_RECEIPT)
    parser.add_argument("--skip-paper-binding", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def preview(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "sage_mem_v1_publication_chain_v1",
        "study": "sage-mem-v1",
        "preview": True,
        "requires_complete_phase_a_cells": 600,
        "requires_formal_finalizer_summary": str(FINALIZER_SUMMARY),
        "requires_formal_audit_report": str(REPORT),
        "phase_b_receipt": str(args.phase_b_receipt),
        "ledger_json": str(args.ledger_json),
        "ledger_tex": str(args.ledger_tex),
        "plot_manifest": str(args.plot_manifest),
        "binding_receipt": str(args.binding_receipt),
        "paper_binding_skipped": bool(args.skip_paper_binding),
        "no_files_read": True,
        "no_files_written": True,
        "no_outcomes_read": True,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(canonical_json(preview(args)), end="")
        return 0
    try:
        root = args.root.resolve()
        require(root == ROOT,
                "production publication chain must run from repository ROOT")
        require(PYTHON.is_file() and os.access(PYTHON, os.X_OK),
                f"missing or non-executable repository Python interpreter: "
                f"{PYTHON}")
        hashes = operator_hashes(root)
        steps: list[Mapping[str, Any]] = []

        if args.phase_b_receipt.exists():
            require(args.resume,
                    "Phase-B receipt already exists; use --resume to bind it")
            require(args.phase_b_receipt.is_file()
                    and not args.phase_b_receipt.is_symlink(),
                    "existing Phase-B receipt is unsafe")
        else:
            command = [
                str(PYTHON), "scripts/audit_sage_mem_v1_phase_b_reproduction.py",
                "--execute", "--workspace", str(root),
                "--output", str(args.phase_b_receipt),
                "--expected-verifier-source-sha256",
                hashes["verifier_source"],
                "--expected-protocol-lock-sha256", hashes["protocol_lock"],
                "--expected-phase-a-grid-sha256", hashes["phase_a_grid"],
                "--expected-raw-context-summary-sha256",
                hashes["raw_context_summary"],
                "--expected-label-registry-sha256", hashes["label_registry"],
                "--expected-execution-registry-sha256",
                hashes["execution_registry"],
                "--expected-finalizer-summary-sha256",
                hashes["finalizer_summary"],
                "--expected-finalized-cells-sha256",
                hashes["finalized_cells"],
                "--expected-formal-report-sha256", hashes["formal_report"],
            ]
            steps.append(run_command(command, "phase-b reproduction"))
        phase_b_sha = sha256_file(args.phase_b_receipt)

        summarize = [
            str(PYTHON), "scripts/summarize_sage_mem_v1_report.py",
            "--execute", "--report", str(REPORT),
            "--json-output", str(args.ledger_json),
            "--tex-output", str(args.ledger_tex),
            "--phase-b-receipt", str(args.phase_b_receipt),
            "--expected-report-sha256", hashes["formal_report"],
            "--expected-phase-b-receipt-sha256", phase_b_sha,
        ]
        if args.resume:
            summarize.append("--resume")
        steps.append(run_command(summarize, "claim ledger generation"))
        ledger_sha = sha256_file(args.ledger_json)

        plot = [
            str(PYTHON), "scripts/plot_sage_mem_v1_claims.py",
            "--execute", "--ledger", str(args.ledger_json),
            "--expected-ledger-sha256", ledger_sha,
        ]
        if args.resume:
            plot.append("--resume")
        steps.append(run_command(plot, "claim figure generation"))
        plot_manifest_sha = sha256_file(args.plot_manifest)

        if not args.skip_paper_binding:
            binding = [
                str(PYTHON), "scripts/audit_paper_a_sage_mem_binding.py",
                "--execute", "--resume",
                "--report", str(REPORT),
                "--ledger-json", str(args.ledger_json),
                "--ledger-tex", str(args.ledger_tex),
                "--phase-b-receipt", str(args.phase_b_receipt),
                "--figure-manifest", str(args.plot_manifest),
                "--receipt", str(args.binding_receipt),
                "--expected-report-sha256", hashes["formal_report"],
                "--expected-phase-b-receipt-sha256", phase_b_sha,
                "--expected-ledger-sha256", ledger_sha,
                "--expected-figure-manifest-sha256", plot_manifest_sha,
            ]
            steps.append(run_command(binding, "paper binding audit"))

        print(canonical_json({
            "schema": "sage_mem_v1_publication_chain_v1",
            "study": "sage-mem-v1",
            "status": "complete",
            "paper_binding_skipped": bool(args.skip_paper_binding),
            "operator_hashes": hashes,
            "phase_b_receipt_sha256": phase_b_sha,
            "ledger_sha256": ledger_sha,
            "plot_manifest_sha256": plot_manifest_sha,
            "steps": steps,
        }), end="")
        return 0
    except PublicationChainError as error:
        print(canonical_json({
            "schema": "sage_mem_v1_publication_chain_v1",
            "study": "sage-mem-v1",
            "status": "failed",
            "reason": str(error),
        }), end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
