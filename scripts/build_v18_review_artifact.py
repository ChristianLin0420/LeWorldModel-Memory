#!/usr/bin/env python3
"""Build a checksum-linked, double-blind V18 result artifact.

The public bundle contains aggregate scientific outputs and redacted ledgers.  It
does not claim that the identity-bearing private repository is review-safe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v18_release_common as common

DEFAULT_RESULTS = ROOT / "result"
DEFAULT_OUTPUT = ROOT / "generated" / "review_artifact"
DEFAULT_PROTOCOL_DOCUMENT = ROOT / "provenance" / "V18_LEWM_V8_CONFIRMATION.md"
DEFAULT_RESTART_AUDIT = ROOT / "provenance" / "v18_restart_audit.v2.json"

IDENTITY_KEYS = {
    "wandb_entity",
    "entity",
    "git_branch",
    "git_commit",
    "git_upstream_commit",
    "url",
    "run_id",
    "output_root",
    "log_root",
    "python",
    "directory",
    "log",
}


def sanitize(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, Mapping):
        result = {str(key): sanitize(item, replacements) for key, item in value.items()}
        for key in IDENTITY_KEYS:
            if key in result and result[key] not in (None, ""):
                result[key] = "WITHHELD_FOR_DOUBLE_BLIND_REVIEW"
        if "wandb_project" in result:
            result["wandb_project"] = "anonymous-review-project"
        if "project" in result:
            result["project"] = "anonymous-review-project"
        return result
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
        return value
    return value


def receipt_paths(runs: Iterable[Mapping[str, Any]]) -> Iterable[tuple[Mapping[str, Any], Path]]:
    for row in runs:
        directory = Path(str(row["directory"]))
        yield row, directory / "wandb_run.json"


def embedded_identity_tokens(value: Any) -> set[str]:
    """Collect IDs that may also occur inside W&B filenames or free text."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"run_id", "wandb_id"} and isinstance(item, str) and item:
                found.add(item)
            found.update(embedded_identity_tokens(item))
    elif isinstance(value, list):
        for item in value:
            found.update(embedded_identity_tokens(item))
    return found


def ordered_unique_replacements(
    candidates: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Keep semantic path replacements ahead of overlapping tool paths.

    In the normal repository layout the release-kit root and private repository
    root are the same path.  Applying the release-kit replacement first made a
    staged rebuild emit ``$RELEASE_KIT`` where the checked artifact (and the
    manuscript renderer) emitted ``$REPO``/``$RESULTS``.  Preserve the first
    semantic spelling for duplicate sources so output is independent of where
    this builder itself is installed.
    """

    replacements: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, target in candidates:
        if source and source not in seen:
            replacements.append((source, target))
            seen.add(source)
    return tuple(replacements)


def build_replacements(
    *,
    result_root: Path,
    repository_root: Path,
    release_kit_root: Path,
    audit_generator: str,
    home: Path,
    protocol: Mapping[str, Any],
    run_ids: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    """Build the canonical private-to-public replacement table."""

    return ordered_unique_replacements((
        # Exact/specific paths precede containing directories.  Repository and
        # result tokens are shared with the manuscript renderer; the tool path
        # is only an additional privacy guard when the kit lives elsewhere.
        (audit_generator, "$AUDIT_GENERATOR"),
        (str(result_root), "$RESULTS"),
        (str(repository_root), "$REPO"),
        (str(release_kit_root), "$RELEASE_KIT"),
        (str(home), "$HOME"),
        (str(protocol.get("wandb_entity", "")), "anonymous-review-entity"),
        (str(protocol.get("wandb_project", "")), "anonymous-review-project"),
        (str(protocol.get("git_commit", "")), "WITHHELD_GIT_COMMIT"),
        (str(protocol.get("git_upstream_commit", "")), "WITHHELD_UPSTREAM_COMMIT"),
        *((run_id, "WITHHELD_WANDB_RUN_ID") for run_id in sorted(run_ids)),
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol-document", type=Path, default=DEFAULT_PROTOCOL_DOCUMENT)
    parser.add_argument("--restart-audit", type=Path, default=DEFAULT_RESTART_AUDIT)
    parser.add_argument(
        "--log-root", type=Path,
        help="optional private log root for byte-level restart log verification",
    )
    parser.add_argument(
        "--forbid", action="append", default=[], metavar="TOKEN",
        help="additional case-insensitive identity token forbidden in every public file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    protocol_document = args.protocol_document.resolve()
    restart_path = args.restart_audit.resolve()
    bundle = common.load_complete_bundle(root, require_failure=True)
    report = bundle["report"]
    protocol = bundle["protocol"]
    audit = common.validate_restart_audit(
        restart_path,
        bundle,
        log_root=args.log_root.resolve() if args.log_root else None,
    )
    private_repository_root = root.parent.parent.resolve()
    audit_generator = audit.get("generator", {})
    private_audit_generator = (
        str(audit_generator.get("path", ""))
        if isinstance(audit_generator, Mapping)
        else ""
    )
    if not protocol_document.is_file():
        raise common.ReleaseValidationError("frozen protocol document is absent")
    expected_protocol_document = protocol.get("source_sha256", {}).get(
        "docs/V18_LEWM_V8_CONFIRMATION.md"
    )
    if expected_protocol_document != common.sha256(protocol_document):
        raise common.ReleaseValidationError(
            "frozen protocol document is not the source-hash-bound protocol"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review artifact {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    private_run_ids = embedded_identity_tokens(audit)
    replacements = build_replacements(
        result_root=root,
        repository_root=private_repository_root,
        release_kit_root=ROOT.resolve(),
        audit_generator=private_audit_generator,
        home=Path.home().resolve(),
        protocol=protocol,
        run_ids=private_run_ids,
    )
    canonical_commands, canonical_command_hashes, canonical_commands_sha256 = (
        common.canonical_redacted_commands(protocol.get("commands"), replacements)
    )

    public_protocol = sanitize(protocol, replacements)
    if public_protocol.get("commands") != canonical_commands:
        raise common.ReleaseValidationError("canonical command redaction differs from protocol")
    public_protocol.pop("commands_sha256", None)
    public_protocol["canonical_commands_sha256"] = canonical_commands_sha256

    def public_ledger(rows: Any) -> Any:
        output = sanitize(rows, replacements)
        if not isinstance(output, list):
            raise common.ReleaseValidationError("public ledger is malformed")
        for row in output:
            if not isinstance(row, dict):
                raise common.ReleaseValidationError("public ledger row is malformed")
            key = (str(row.get("task")), str(row.get("design")), int(row.get("seed")))
            row.pop("command_sha256", None)
            row["canonical_command_sha256"] = canonical_command_hashes[key]
        return output

    public_audit = sanitize(audit, replacements)
    if not isinstance(public_audit, dict):
        raise common.ReleaseValidationError("public restart audit is malformed")
    for lineage in public_audit.get("attempt_lineages", []):
        cell = lineage.get("cell", {})
        key = (str(cell.get("task")), str(cell.get("design")), int(cell.get("seed")))
        lineage.pop("command_sha256", None)
        lineage["canonical_command_sha256"] = canonical_command_hashes[key]
    public_audit_protocol = public_audit.get("protocol")
    if not isinstance(public_audit_protocol, dict):
        raise common.ReleaseValidationError("public restart protocol receipt is malformed")
    public_audit_protocol.pop("commands_sha256", None)
    public_audit_protocol["canonical_commands_sha256"] = canonical_commands_sha256

    def reject_private_command_hash_fields(value: Any) -> None:
        if isinstance(value, Mapping):
            forbidden_fields = {"command_sha256", "commands_sha256"} & set(value)
            if forbidden_fields:
                raise common.ReleaseValidationError(
                    f"private command hash field remains public: {sorted(forbidden_fields)}"
                )
            for item in value.values():
                reject_private_command_hash_fields(item)
        elif isinstance(value, list):
            for item in value:
                reject_private_command_hash_fields(item)

    redacted = {
        "confirmation_protocol.redacted.json": public_protocol,
        "confirmation_runs.redacted.json": public_ledger(bundle["runs"]),
        "confirmation_attempts.redacted.json": public_ledger(bundle["attempts"]),
        "confirmation_summary.redacted.json": sanitize(bundle["summary"], replacements),
    }
    reject_private_command_hash_fields(redacted)
    reject_private_command_hash_fields(public_audit)

    remote_receipts = []
    for row, receipt_path in receipt_paths(bundle["runs"]):
        if not receipt_path.is_file():
            raise common.ReleaseValidationError(f"missing remote receipt {receipt_path}")
        receipt = common.read_json(receipt_path)
        if not isinstance(receipt, Mapping) or receipt.get("state") != "finished":
            raise common.ReleaseValidationError(f"non-finished remote receipt {receipt_path}")
        rollout = common.require_hash(
            receipt.get("eval_rollout_sha256"), f"remote rollout {receipt_path}"
        )
        expected = row.get("artifact_sha256", {}).get("wandb_run.json")
        if expected != common.sha256(receipt_path):
            raise common.ReleaseValidationError(f"run ledger receipt hash differs: {receipt_path}")
        remote_receipts.append({
            "task": row["task"],
            "design": row["design"],
            "seed": int(row["seed"]),
            "state": "finished",
            "eval_rollout_sha256": rollout,
            "unredacted_receipt_sha256": expected,
            "remote_identity": "WITHHELD_FOR_DOUBLE_BLIND_REVIEW",
        })
    redacted["remote_receipts.redacted.json"] = remote_receipts

    staging = Path(tempfile.mkdtemp(prefix=".v18-review-", dir=output.parent))
    try:
        for name in (
            "confirmation_analysis.json",
            "confirmation_cells.csv",
            "confirmation_contrasts.csv",
        ):
            shutil.copy2(root / name, staging / name)
        shutil.copy2(protocol_document, staging / "frozen_protocol.md")
        # Rich private audits may contain W&B run IDs/URLs. Preserve their source
        # hash in the manifest but publish only a structurally identical redaction.
        common.atomic_write_json(staging / "restart_audit.v2.json", public_audit)
        for name, value in redacted.items():
            common.atomic_write_json(staging / name, value)
        readme = """# Anonymous V18 review result bundle

This artifact contains the complete write-once decision, 200 cell-level rows,
33 registered contrasts, the source-hash-bound frozen protocol, redacted runs,
attempts and summary ledgers, remote receipt hashes, and restart-audit schema v2.

The restart audit records two process-level interruptions.  Process-killed
attempts are absent from the runner's terminal-return attempt ledger by design;
the separately hash-bound receipt binds interrupted and restart log hashes and the
final result/ledger hashes.  It does not change any scientific value.

The exact pre-core/replacement lineages are `acrobot.swingup/vicreg_ssm/s18005`,
`manipulator.bring_ball/vicreg_ssm/s18005`,
`quadruped.run/vicreg_ssm/s18005`, `swimmer.swimmer15/vicreg_ssm/s18005`, and
`stacker.stack_4/vicreg_hacssmv8_static/s18003`.  The first interruption caused
the four SSM replacements and the second caused the stacker-static replacement.
All five interrupted cells lacked the four core artifacts and every replacement
restarted at epoch one and reached epoch 100.

The frozen protocol also registered V8 gate/route telemetry, but the exporter
retained only final shrinkage coefficients and action-feature norms.  Per-step
gate vectors and route weights are unavailable.  This is a secondary-report
deviation and does not enter or affect any primary preregistered gate.

Public command commitments are recomputed from canonical JSON after replacing
private executable and repository prefixes.  They are labeled
`canonical_command_sha256` and `canonical_commands_sha256`; original
private-path-derived command hashes remain only in the frozen private result.
The review manifest retains the top-level frozen protocol SHA-256 as the
preregistration commitment.

Raw checkpoints, histories, held-out rollout arrays, and identity-bearing remote
metadata remain in the private frozen result root during double-blind review.
The private repository is not represented as anonymous.  A code supplement must
be a separately curated and identity-scanned export.
"""
        common.atomic_write_text(staging / "README.md", readme)

        forbidden = {
            str(protocol.get("wandb_entity", "")),
            str(protocol.get("wandb_project", "")),
            str(protocol.get("git_commit", "")),
            str(protocol.get("git_upstream_commit", "")),
            str(ROOT.resolve()),
            str(root),
            str(private_repository_root),
            private_audit_generator,
            str(Path.home().resolve()),
            "wandb.ai/",
            *private_run_ids,
            *args.forbid,
        } - {""}
        files = sorted(path for path in staging.iterdir() if path.is_file())
        common.scan_forbidden(files, forbidden)
        email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
        for path in files:
            if email.search(path.read_text(encoding="utf-8")):
                raise common.ReleaseValidationError(
                    f"email address remains in review artifact {path.name}"
                )
        manifest = {
            "schema_version": 2,
            "scope": "anonymous_v18_review_result_bundle",
            "scientific_label": report["scientific_label"],
            "cells": report["completed_valid_cells"],
            "contrasts": common.EXPECTED_CONTRASTS,
            "source_result_hashes": bundle["hashes"],
            "source_protocol_document_sha256": common.sha256(protocol_document),
            "source_restart_audit_sha256": common.sha256(restart_path),
            "canonical_commands_sha256": canonical_commands_sha256,
            "builder_sha256": common.sha256(Path(__file__).resolve()),
            "common_validator_sha256": common.sha256(Path(common.__file__).resolve()),
            "restart_schema_version": audit["schema_version"],
            "restart_interruptions": common.restart_interruption_count(audit),
            "files": {path.name: common.sha256(path) for path in files},
            "redactions": [
                "absolute local paths",
                "private repository-root prefixes",
                "restart-audit generator path",
                "W&B entity, project, run IDs, and URLs",
                "Git branch and commit identities",
                "private-path-derived command hashes replaced by canonical public-command hashes",
            ],
            "private_repository_review_safe": False,
        }
        common.atomic_write_json(staging / "review_manifest.json", manifest)
        common.scan_forbidden([staging / "review_manifest.json"], forbidden)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({
        "output": str(output),
        "manifest": str(output / "review_manifest.json"),
        "scientific_label": report["scientific_label"],
        "cells": report["completed_valid_cells"],
        "restart_interruptions": common.restart_interruption_count(audit),
    }, indent=2))


if __name__ == "__main__":
    main()
