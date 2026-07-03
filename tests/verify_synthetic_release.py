#!/usr/bin/env python3
"""Assertions for the end-to-end synthetic V18 release build."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "scripts"))
import v18_release_common as common
import build_v18_review_artifact as review_builder
import build_v18_code_supplement as supplement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--prepare-canonical", action="store_true",
        help="validate the portable restart receipt, write the rich synthetic receipt, and exit",
    )
    args = parser.parse_args()
    base = args.base.resolve()
    bundle = common.load_complete_bundle(base / "result", require_failure=True)
    portable_path = base / "provenance" / "v18_restart_audit.portable.v2.json"
    audit = common.validate_restart_audit(
        portable_path if portable_path.is_file() else base / "provenance" / "v18_restart_audit.v2.json",
        bundle,
        log_root=base / "logs",
    )
    assert len(audit["interruptions"]) == 2

    # The release validator also accepts the richer repository-specific audit
    # produced after terminal W&B/lineage verification.
    canonical_lineages = []
    for interruption in audit["interruptions"]:
        for attempt in interruption["interrupted_attempts"]:
            canonical_lineages.append({
                "cell": {
                    "task": attempt["task"],
                    "design": attempt["design"],
                    "seed": attempt["seed"],
                },
                "interrupted_attempt": {
                    "last_logged_epoch": attempt["last_logged_epoch"],
                    "wandb": {"state": "crashed", "run_id": "synthetic-crashed"},
                    "log": {
                        "path": "logs/" + attempt["interrupted_log"],
                        "sha256": attempt["interrupted_log_sha256"],
                    },
                },
                "replacement_attempt": {
                    "first_logged_epoch": 1,
                    "last_logged_epoch": 100,
                    "wandb": {"state": "finished", "run_id": "synthetic-finished"},
                    "log": {
                        "path": "logs/" + attempt["restart_log"],
                        "sha256": attempt["restart_log_sha256"],
                    },
                },
            })
    canonical = {
        "schema_version": 2,
        "study": "lewm-v8-v18-confirmation",
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": common.sha256(Path(__file__).resolve()),
        },
        "protocol": {
            "sha256": bundle["hashes"]["confirmation_protocol.json"],
            "commands_sha256": bundle["protocol"]["commands_sha256"],
        },
        "bound_receipts": {
            alias: {"path": filename, "sha256": bundle["hashes"][filename]}
            for alias, filename in {
                "protocol": "confirmation_protocol.json",
                "summary": "confirmation_summary.json",
                "runs": "confirmation_runs.json",
                "attempts": "confirmation_attempts.json",
                "analysis": "confirmation_analysis.json",
                "cells_csv": "confirmation_cells.csv",
                "contrasts_csv": "confirmation_contrasts.csv",
            }.items()
        },
        "terminal_preconditions": {
            "runner_trainer_analyzer_processes_active": 0,
            "runner_lock_absent": True,
        },
        "resume_events": [
            {
                "resume_index": 1,
                "pre_resume_counts": {
                    "valid_complete": 136,
                    "absent": 64,
                    "partial_or_invalid_core": 0,
                },
                "replacement_cell_count": 4,
                "resume_policy": "complete_cell_only",
            },
            {
                "resume_index": 2,
                "pre_resume_counts": {
                    "valid_complete": 180,
                    "absent": 20,
                    "partial_or_invalid_core": 0,
                },
                "replacement_cell_count": 1,
                "resume_policy": "complete_cell_only",
            },
        ],
        "attempt_lineages": canonical_lineages,
        "artifact_binding": {
            "checked_cells": 200,
            "finished_local_wandb_receipts": 200,
            "artifact_manifest_bound_by_analysis": True,
        },
        "remote_wandb_terminal_observation": {
            "runs": [
                {"state": state, "run_id": f"synthetic-{state}-{index}"}
                for state in ("crashed", "finished")
                for index in range(5)
            ]
        },
        "final_study_snapshot": {
            "status": "COMPLETE",
            "planned_cells": 200,
            "completed_valid_cells": 200,
            "absent_cells": 0,
            "failed_or_invalid_cells": 0,
            "analysis_complete": True,
        },
    }
    with tempfile.TemporaryDirectory(prefix="v18-canonical-") as temporary:
        canonical_path = Path(temporary) / "audit.json"
        common.atomic_write_json(canonical_path, canonical)
        validated = common.validate_restart_audit(
            canonical_path, bundle, log_root=base / "logs"
        )
        assert common.restart_interruption_count(validated) == 2
        assert "all five replacements reached epoch 100" in common.restart_text(validated)[0]
    if args.prepare_canonical:
        output = base / "provenance" / "v18_restart_audit.v2.json"
        common.atomic_write_json(output, canonical)
        print(json.dumps({"canonical_restart_audit": str(output)}, indent=2))
        return

    # A normal in-repository build has the same release-kit and private-repo
    # root.  The result root must still receive the most-specific token, and a
    # duplicate tool-root candidate must not change the public command hash.
    replacement_protocol = {
        "wandb_entity": "private-entity",
        "wandb_project": "private-project",
        "git_commit": "a" * 40,
        "git_upstream_commit": "b" * 40,
    }
    fixture_commands = [{
        "task": "fixture.task",
        "design": "fixture_design",
        "seed": 1,
        "argv": [
            "/private/repo/.venv/bin/python",
            "--output-dir",
            "/private/repo/outputs/v18",
        ],
    }]
    replacement_variants = [
        review_builder.build_replacements(
            result_root=Path("/private/repo/outputs/v18"),
            repository_root=Path("/private/repo"),
            release_kit_root=Path(release_kit),
            audit_generator="/private/repo/tools/private-audit.py",
            home=Path("/private/home"),
            protocol=replacement_protocol,
            run_ids={"private-run-id"},
        )
        for release_kit in ("/private/repo", "/external/release-kit")
    ]
    public_variants = [
        common.canonical_redacted_commands(fixture_commands, replacements)
        for replacements in replacement_variants
    ]
    assert public_variants[0] == public_variants[1]
    assert public_variants[0][0][0]["argv"] == [
        "$REPO/.venv/bin/python", "--output-dir", "$RESULTS"
    ]
    assert review_builder.sanitize(
        "/private/repo/tools/private-audit.py", replacement_variants[0]
    ) == "$AUDIT_GENERATOR"
    assert ("/private/repo", "$RELEASE_KIT") not in replacement_variants[0]

    identity_fixture = {
        "wandb": {"run_id": "private123", "scratch_manifest": ["run-private123.wandb"]}
    }
    tokens = review_builder.embedded_identity_tokens(identity_fixture)
    redacted = review_builder.sanitize(
        identity_fixture, tuple((token, "WITHHELD") for token in tokens)
    )
    assert "private123" not in str(redacted)

    manuscript = (base / "generated" / "ICLR.md").read_text(encoding="utf-8")
    assert not common.PLACEHOLDER.search(manuscript)
    assert r"\cite{" not in manuscript and r"\citep{" in manuscript
    assert not any(character in manuscript for character in "“”‘’")
    assert "fixed GRU width gives 35,048" in manuscript
    assert "SAS-PC has 33,286--36,614 carrier parameters" in manuscript
    assert "total models by at most 0.09%" in manuscript
    assert "per-step gate vectors and route weights were not retained" in manuscript
    assert "Two process-level interruptions" in manuscript
    assert "private repository is not itself an anonymous review artifact" in manuscript

    manuscript_manifest = common.read_json(base / "generated" / "ICLR.manifest.json")
    assert manuscript_manifest["schema_version"] == 2
    assert manuscript_manifest["telemetry_disclosure_present"] is True
    assert manuscript_manifest["restart_interruptions"] == 2

    figure_manifest_path = base / "generated" / "figures" / "fig_v18_manifest.json"
    figure_manifest = common.read_json(figure_manifest_path)
    assert figure_manifest["schema_version"] == 3
    assert figure_manifest["artifact_kind"] == "v18_provenance_bound_paper_figures"
    assert manuscript_manifest["figure_manifest_sha256"] == common.sha256(
        figure_manifest_path
    )
    assert set(figure_manifest["figures"]) == {
        f"fig_v18_{stem}.{suffix}"
        for stem in ("architecture", "evidence", "secondary", "task_design")
        for suffix in ("pdf", "png")
    }
    secondary = figure_manifest["descriptive_secondary"]
    assert secondary["official_decision_changed"] is False
    assert secondary["decision_gates_defined"] is False
    assert secondary["claim_scope"] == "descriptive decomposition only"
    assert len(secondary["public_prior_slices"]) == 200
    assert len({
        (row["task"], row["seed"], row["design"])
        for row in secondary["public_prior_slices"]
    }) == 200

    review = base / "generated" / "review_artifact"
    review_manifest = common.read_json(review / "review_manifest.json")
    required = {
        "confirmation_attempts.redacted.json",
        "confirmation_summary.redacted.json",
        "confirmation_runs.redacted.json",
        "restart_audit.v2.json",
    }
    assert required.issubset(review_manifest["files"])
    assert review_manifest["private_repository_review_safe"] is False

    paper_check = common.read_json(base / "generated" / "paper_check.json")
    assert paper_check["status"] == "PASS"
    assert paper_check["style"] == "iclr2026_conference"
    assert paper_check["style_sha256"] == (
        "a4852f68e080d6c5245057ca2039100b409e31727898aa93c03d78ddb84374a3"
    )
    assert paper_check["main_pages"] <= 9
    assert set(paper_check["main_figure_pages"]) == {
        "fig:fig-v18-architecture", "fig:fig-v18-evidence",
        "fig:fig-v18-secondary",
    }
    assert all(
        page <= paper_check["main_pages"]
        for page in paper_check["main_figure_pages"].values()
    )
    assert paper_check["maximum_overfull_pt"] <= 2.0
    assert paper_check["cited_keys"] == 18
    assert paper_check["pdf_metadata"]["Author"] == ""

    for path in (
        base / "generated" / "README.final.md",
        base / "generated" / "LEARNABLE_MEMORY.v18.patch.md",
    ):
        assert not common.PLACEHOLDER.search(path.read_text(encoding="utf-8"))

    supplement_dir = base / "v18-anonymous-supplement"
    supplement_zip = base / "v18-anonymous-supplement.zip"
    repeated_zip = base / "v18-anonymous-supplement-repeat.zip"
    supplement_manifest = common.read_json(supplement_dir / "MANIFEST.json")
    assert supplement_manifest["scope"] == "anonymous_v18_code_and_result_supplement"
    assert supplement_manifest["scientific_label"] == "CONFIRMATION_FAILED"
    assert supplement_manifest["file_count_excluding_manifest"] == len(
        supplement_manifest["files"]
    )
    source_hashes = bundle["protocol"]["source_sha256"]
    records = {row["path"]: row for row in supplement_manifest["files"]}
    for relative, digest in source_hashes.items():
        row = records[relative]
        assert row["source_class"] == "frozen_execution_source"
        assert common.sha256(supplement_dir / relative) == row["public_sha256"]
        if row["redactions"]:
            assert row["frozen_execution_sha256"] == digest
            assert "matches_frozen_execution" not in row
        else:
            assert row["public_sha256"] == digest
            assert row["matches_frozen_execution"] is True
            assert "frozen_execution_sha256" not in row
    assert all("private_original_sha256" not in row for row in records.values())
    assert supplement_manifest["identity_receipt_policy"] == {
        "private_token_values_published": False,
        "private_token_digests_published": False,
        "receipt_granularity": "category_and_scan_passed_only",
    }
    assert all(
        set(row) == {"category", "scan_passed"} and row["scan_passed"] is True
        for field in ("redaction_policy", "additional_forbidden_token_receipts")
        for row in supplement_manifest[field]
    )
    assert any(row["redactions"] for row in records.values())
    assert "synthetic-private" not in json.dumps(supplement_manifest)
    assert common.sha256(supplement_zip) == common.sha256(repeated_zip)
    sidecar = supplement_zip.with_suffix(".zip.sha256").read_text(encoding="utf-8")
    assert sidecar == f"{common.sha256(supplement_zip)}  {supplement_zip.name}\n"
    with zipfile.ZipFile(supplement_zip) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert f"{supplement.ARCHIVE_ROOT}/MANIFEST.json" in names
        assert f"{supplement.ARCHIVE_ROOT}/.gitignore" in names
        assert not any(
            component.casefold() in supplement.BANNED_COMPONENTS
            for name in names
            for component in PurePosixPath(name).parts[1:]
        )
        assert not any(name.endswith(tuple(supplement.BANNED_SUFFIXES)) for name in names)
        assert not any("synthetic-private" in name for name in names)
        with tempfile.TemporaryDirectory(prefix="v18-extracted-") as temporary:
            archive.extractall(temporary)
            extracted = Path(temporary) / supplement.ARCHIVE_ROOT
            verified = subprocess.run(
                [sys.executable, "tools/verify_supplement.py"],
                cwd=extracted,
                capture_output=True,
                text=True,
                check=True,
            )
            assert "public files plus MANIFEST.json" in verified.stdout
            bootstrap = subprocess.run(
                [sys.executable, "tools/bootstrap_anonymous_git.py"],
                cwd=extracted,
                capture_output=True,
                text=True,
                check=True,
            )
            assert len(bootstrap.stdout.strip()) == 40
            assert subprocess.run(
                ["git", "remote"], cwd=extracted,
                capture_output=True, text=True, check=True,
            ).stdout.strip() == ""
            assert subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=extracted,
                capture_output=True, text=True, check=True,
            ).stdout.strip() == ""
            tests = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "scripts/test_train_lewm_v8_v18.py",
                    "scripts/test_run_lewm_v8_v18.py",
                    "scripts/test_analyze_lewm_v8_v18.py",
                ],
                cwd=extracted,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=True,
            )
            assert "3 passed" in tests.stdout

    # The shared guard must fail closed before any renderer sees incomplete data.
    with tempfile.TemporaryDirectory(prefix="v18-negative-") as temporary:
        incomplete = Path(temporary)
        shutil.copy2(
            base / "result" / "confirmation_analysis.json",
            incomplete / "confirmation_analysis.json",
        )
        try:
            common.load_complete_bundle(incomplete)
        except common.ReleaseValidationError as exc:
            assert "refusing incomplete result root" in str(exc)
        else:
            raise AssertionError("incomplete result root unexpectedly validated")

    print(
        "synthetic release verified: complete guard, v2 restarts, manuscript, "
        "review artifact, official-style PDF, citations, pages, metadata, and docs"
    )


if __name__ == "__main__":
    main()
