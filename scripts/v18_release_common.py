#!/usr/bin/env python3
"""Shared, standalone validation and rendering helpers for the V18 release kit.

This module intentionally does not import the training runner.  A release can be
reproduced from a frozen result directory without placing the mutable experiment
source on ``sys.path``.  Every consumer calls :func:`load_complete_bundle`, which
rejects partial results before reading values into prose, figures, or review files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASKS = (
    "acrobot.swingup",
    "manipulator.bring_ball",
    "quadruped.run",
    "stacker.stack_4",
    "swimmer.swimmer15",
)
SEEDS = tuple(range(18001, 18006))
DESIGNS = (
    "vicreg_none",
    "vicreg_gru",
    "vicreg_ssm",
    "vicreg_hacssmv8",
    "vicreg_hacssmv8_static",
    "vicreg_hacssmv8_dynamic",
    "vicreg_hacssmv8_noaction",
    "vicreg_hacssmv8_single",
)
EXPECTED_CELLS = 200
EXPECTED_CONTRASTS = 33
PRIMARY = "heldout_prior_state_nmse"
CELL_FIELDS = (
    "task",
    "seed",
    "design",
    PRIMARY,
    "clean_prior_state_nmse",
    "val_predictive_loss",
    "deep_prior_state_nmse",
    "encoder_mean_channel_variance",
    "encoder_covariance_effective_rank",
    "predictive_loss_convergence_relative_change",
    "initial_encoder_integrator_probe_nmse",
)
RESULT_FILES = (
    "confirmation_analysis.json",
    "confirmation_cells.csv",
    "confirmation_contrasts.csv",
    "confirmation_protocol.json",
    "confirmation_runs.json",
    "confirmation_attempts.json",
    "confirmation_summary.json",
)
CONTRAST_KEYS = {
    "R": "vicreg_hacssmv8_vs_recurrent_envelope:heldout_prior_state_nmse",
    "N": "vicreg_hacssmv8_vs_vicreg_none:heldout_prior_state_nmse",
    "I": "vicreg_hacssmv8_vs_checkpoint_integrator:heldout_prior_state_nmse",
    "A": "vicreg_hacssmv8_vs_vicreg_hacssmv8_noaction:heldout_prior_state_nmse",
    "J": "vicreg_hacssmv8_vs_vicreg_hacssmv8_single:heldout_prior_state_nmse",
    "E": "vicreg_hacssmv8_vs_endpoint_envelope:heldout_prior_state_nmse",
    "D": "vicreg_hacssmv8_vs_recurrent_envelope:deep_prior_state_nmse",
    "C": "vicreg_hacssmv8_vs_recurrent_envelope:clean_prior_state_nmse",
}
GATE_KEYS = {
    "R": "v8_vs_per_cell_better_gru_ssm",
    "N": "v8_vs_none",
    "I": "v8_vs_checkpoint_integrator",
    "A": "action_causality",
    "J": "joint_state_use",
    "E": "learned_v8_vs_static_dynamic_envelope_noninferiority",
    "D": "deep_vs_per_cell_better_gru_ssm",
    "C": "clean_prior_guard_vs_per_cell_better_gru_ssm",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")


class ReleaseValidationError(RuntimeError):
    """Raised when a purported release input is incomplete or not hash-bound."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_redacted_commands(
    commands: Any,
    replacements: Sequence[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], str], str]:
    """Return public commands and hashes computed only from their redacted bytes."""

    def redact(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str):
            for source, target in replacements:
                if source:
                    value = value.replace(source, target)
            return value
        return value

    public = redact(commands)
    if not isinstance(public, list):
        raise ReleaseValidationError("canonical public command list is malformed")
    indexed: dict[tuple[str, str, int], str] = {}
    for row in public:
        if not isinstance(row, dict):
            raise ReleaseValidationError("canonical public command row is malformed")
        try:
            key = (str(row["task"]), str(row["design"]), int(row["seed"]))
            argv = row["argv"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ReleaseValidationError("canonical public command identity is malformed") from exc
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ReleaseValidationError(f"canonical public argv is malformed for {key}")
        if key in indexed:
            raise ReleaseValidationError(f"duplicate canonical public command {key}")
        indexed[key] = json_sha256(argv)
    return public, indexed, json_sha256(public)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"cannot read valid JSON {path}: {exc}") from exc


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ReleaseValidationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def artifact_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    entries = [
        {
            "task": str(row.get("task")),
            "seed": int(row.get("seed")),
            "design": str(row.get("design")),
            "artifact_sha256": row.get("artifact_sha256"),
        }
        for row in rows
    ]
    entries.sort(key=lambda row: (row["task"], row["seed"], row["design"]))
    if any(not isinstance(row["artifact_sha256"], Mapping) for row in entries):
        raise ReleaseValidationError("run ledger has a malformed artifact hash record")
    for index, row in enumerate(entries):
        for name, digest in row["artifact_sha256"].items():
            require_hash(digest, f"run artifact {index}/{name}")
    return json_sha256(entries)


def expected_grid() -> set[tuple[str, int, str]]:
    return {
        (task, seed, design)
        for task in TASKS
        for seed in SEEDS
        for design in DESIGNS
    }


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseValidationError(f"cannot read valid CSV {path}: {exc}") from exc
    return fields, rows


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_lists = {
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "seeds": list(SEEDS),
    }
    for key, expected in expected_lists.items():
        if protocol.get(key) != expected:
            raise ReleaseValidationError(f"protocol {key} differs from frozen V18 grid")
    if protocol.get("epochs") != 100 or protocol.get("runs") != EXPECTED_CELLS:
        raise ReleaseValidationError("protocol is not the frozen 200-cell/100-epoch study")
    commands = protocol.get("commands")
    if not isinstance(commands, list) or len(commands) != EXPECTED_CELLS:
        raise ReleaseValidationError("protocol command expansion is not 200 rows")
    try:
        command_grid = {
            (str(row["task"]), int(row["seed"]), str(row["design"]))
            for row in commands
            if isinstance(row, Mapping)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseValidationError("protocol command expansion has a malformed cell") from exc
    if len(command_grid) != EXPECTED_CELLS or command_grid != expected_grid():
        raise ReleaseValidationError("protocol command expansion differs from the exact grid")
    if protocol.get("commands_sha256") != json_sha256(commands):
        raise ReleaseValidationError("protocol command expansion hash differs")
    if protocol.get("resume_granularity") != "complete_cell_only":
        raise ReleaseValidationError("protocol does not specify complete-cell-only resume")
    if protocol.get("git_worktree_clean") is not True:
        raise ReleaseValidationError("protocol did not freeze from a clean worktree")
    source = protocol.get("source_sha256")
    data = protocol.get("data")
    if not isinstance(source, Mapping) or not source:
        raise ReleaseValidationError("protocol source hash map is absent")
    if not isinstance(data, Mapping) or "__manifest__" not in data:
        raise ReleaseValidationError("protocol data manifest is absent")
    for name, digest in source.items():
        require_hash(digest, f"protocol source {name}")
    manifest = data["__manifest__"]
    if not isinstance(manifest, Mapping):
        raise ReleaseValidationError("protocol data manifest is malformed")
    for key in ("path_sha256", "sidecar_sha256"):
        require_hash(manifest.get(key), f"protocol data manifest {key}")


def _validate_grid_ledger(rows: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_CELLS:
        raise ReleaseValidationError(f"{label} must contain exactly 200 rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ReleaseValidationError(f"{label} contains a non-object row")
    keys: list[tuple[str, int, str]] = []
    for row in rows:
        try:
            key = (str(row["task"]), int(row["seed"]), str(row["design"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReleaseValidationError(f"{label} has a malformed cell row") from exc
        keys.append(key)
        if row.get("status") != "complete":
            raise ReleaseValidationError(f"{label} contains non-complete row {key}")
        if not isinstance(row.get("artifact_sha256"), Mapping):
            raise ReleaseValidationError(f"{label} lacks artifacts for {key}")
    if len(set(keys)) != EXPECTED_CELLS or set(keys) != expected_grid():
        raise ReleaseValidationError(f"{label} differs from the exact V18 grid")
    return rows


def load_complete_bundle(root: Path, *, require_failure: bool = False) -> dict[str, Any]:
    """Load and independently validate every release-critical result file.

    The attempts ledger is deliberately allowed to have more than 200 rows, but
    it must include a terminal complete receipt for all 200 cells.  The V18
    runner's process-killed attempts are supplied separately by restart-audit v2.
    """

    root = root.resolve()
    missing = [name for name in RESULT_FILES if not (root / name).is_file()]
    if missing:
        raise ReleaseValidationError(
            f"refusing incomplete result root {root}; missing {missing}"
        )
    report = read_json(root / "confirmation_analysis.json")
    protocol = read_json(root / "confirmation_protocol.json")
    runs = read_json(root / "confirmation_runs.json")
    attempts = read_json(root / "confirmation_attempts.json")
    summary = read_json(root / "confirmation_summary.json")
    if not all(isinstance(value, Mapping) for value in (report, protocol, summary)):
        raise ReleaseValidationError("analysis, protocol, or summary is not an object")

    _validate_protocol(protocol)
    run_rows = _validate_grid_ledger(runs, "confirmation_runs.json")
    if not isinstance(attempts, list) or not all(isinstance(row, dict) for row in attempts):
        raise ReleaseValidationError("confirmation_attempts.json is not an object list")
    terminal_attempts = {
        (str(row.get("task")), int(row.get("seed", -1)), str(row.get("design")))
        for row in attempts
        if row.get("status") == "complete"
    }
    if not expected_grid().issubset(terminal_attempts):
        raise ReleaseValidationError("attempt ledger lacks terminal receipts for all cells")

    if summary.get("status") != "COMPLETE" \
            or summary.get("expected_cells") != EXPECTED_CELLS \
            or summary.get("completed_cells") != EXPECTED_CELLS \
            or summary.get("failed_or_invalid_cells") != 0:
        raise ReleaseValidationError("confirmation summary is not complete and clean")
    if report.get("status") != "COMPLETE" \
            or report.get("expected_cells") != EXPECTED_CELLS \
            or report.get("completed_valid_cells") != EXPECTED_CELLS \
            or report.get("artifact_integrity_passed") is not True:
        raise ReleaseValidationError("analysis is not a complete valid 200-cell decision")
    if report.get("artifact_integrity_errors") != [] \
            or report.get("protocol_contract_errors") != []:
        raise ReleaseValidationError("analysis contains integrity or protocol errors")
    frozen_grid = report.get("frozen_grid")
    expected_report_grid = {
        "tasks": 5,
        "designs": 8,
        "seeds": 5,
        "epochs": 100,
        "cells": 200,
        "task_ids": list(TASKS),
        "seed_ids": list(SEEDS),
        "design_ids": list(DESIGNS),
    }
    if not isinstance(frozen_grid, Mapping) or any(
        frozen_grid.get(key) != value for key, value in expected_report_grid.items()
    ):
        raise ReleaseValidationError("analysis frozen-grid receipt differs")
    labels = {"STABILIZED_LEWM_V8_CONFIRMATION_PASS", "CONFIRMATION_FAILED"}
    if report.get("scientific_label") not in labels:
        raise ReleaseValidationError("analysis has an invalid scientific label")
    if bool(report.get("official_confirmation_result")) \
            != (report.get("scientific_label") == "STABILIZED_LEWM_V8_CONFIRMATION_PASS"):
        raise ReleaseValidationError("analysis label and boolean decision disagree")
    if require_failure and report.get("scientific_label") != "CONFIRMATION_FAILED":
        raise ReleaseValidationError("this falsification manuscript requires CONFIRMATION_FAILED")

    protocol_path = root / "confirmation_protocol.json"
    if report.get("input_protocol_sha256") != sha256(protocol_path):
        raise ReleaseValidationError("analysis is not bound to confirmation_protocol.json")
    if report.get("input_artifact_manifest_sha256") != artifact_manifest_sha256(run_rows):
        raise ReleaseValidationError("analysis is not bound to current run artifacts")

    cell_fields, cells = _load_csv(root / "confirmation_cells.csv")
    if cell_fields != list(CELL_FIELDS):
        raise ReleaseValidationError(f"unexpected cell CSV schema: {cell_fields}")
    try:
        cell_keys = [(row["task"], int(row["seed"]), row["design"]) for row in cells]
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseValidationError("cell CSV contains a malformed grid key") from exc
    if len(cells) != EXPECTED_CELLS or len(set(cell_keys)) != EXPECTED_CELLS \
            or set(cell_keys) != expected_grid():
        raise ReleaseValidationError("cell CSV differs from the exact V18 grid")
    for row in cells:
        for field in CELL_FIELDS[3:]:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReleaseValidationError(f"invalid cell CSV number in {field}") from exc
            if not math.isfinite(value):
                raise ReleaseValidationError(f"nonfinite cell CSV number in {field}")

    contrast_fields, contrast_rows = _load_csv(root / "confirmation_contrasts.csv")
    expected_contrast_fields = [
        "contrast", "metric", "mean_paired_relative_reduction", "paired_wins",
        "pairs", "task_mean_wins", "ci95_low", "ci95_high",
    ]
    if contrast_fields != expected_contrast_fields:
        raise ReleaseValidationError("unexpected contrast CSV schema")
    names = [row.get("contrast") for row in contrast_rows]
    if len(contrast_rows) != EXPECTED_CONTRASTS or len(set(names)) != EXPECTED_CONTRASTS:
        raise ReleaseValidationError("contrast CSV is not 33 unique rows")
    contrasts = report.get("contrasts")
    if not isinstance(contrasts, Mapping) or set(names) != set(contrasts):
        raise ReleaseValidationError("analysis and contrast CSV contrast names differ")
    for row in contrast_rows:
        name = str(row["contrast"])
        value = contrasts[name]
        try:
            csv_receipt = (
                row["metric"],
                float(row["mean_paired_relative_reduction"]),
                int(row["paired_wins"]),
                int(row["pairs"]),
                int(row["task_mean_wins"]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
            )
            json_receipt = (
                value["metric"],
                float(value["mean_paired_relative_reduction"]),
                int(value["paired_wins"]),
                int(value["pairs"]),
                int(value["task_mean_wins"]),
                float(value["bootstrap"]["ci95_low"]),
                float(value["bootstrap"]["ci95_high"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReleaseValidationError(f"malformed contrast receipt {name}") from exc
        if csv_receipt != json_receipt or not all(
            math.isfinite(item) for item in (*csv_receipt[1:2], *csv_receipt[5:])
        ):
            raise ReleaseValidationError(f"analysis/CSV contrast receipt differs for {name}")
    for alias, name in CONTRAST_KEYS.items():
        contrast = contrasts.get(name)
        if not isinstance(contrast, Mapping):
            raise ReleaseValidationError(f"missing registered contrast {alias}: {name}")
        for field in ("mean_paired_relative_reduction", "paired_wins", "pairs", "task_mean_wins"):
            if field not in contrast:
                raise ReleaseValidationError(f"contrast {alias} lacks {field}")
        bootstrap = contrast.get("bootstrap")
        if not isinstance(bootstrap, Mapping) \
                or "ci95_low" not in bootstrap or "ci95_high" not in bootstrap:
            raise ReleaseValidationError(f"contrast {alias} lacks its registered CI")
    gates = report.get("gates")
    receipts = report.get("gate_receipts")
    if not isinstance(gates, Mapping) or not isinstance(receipts, Mapping):
        raise ReleaseValidationError("analysis gate maps are absent")
    for alias, key in GATE_KEYS.items():
        if not isinstance(gates.get(key), bool):
            raise ReleaseValidationError(f"missing boolean gate {alias}: {key}")
        if not isinstance(receipts.get(key), Mapping) \
                or receipts[key].get("passed") is not gates[key]:
            raise ReleaseValidationError(f"gate receipt disagrees for {key}")
    for key in ("healthy_representation", "convergence"):
        if not isinstance(gates.get(key), bool) \
                or not isinstance(receipts.get(key), Mapping) \
                or receipts[key].get("passed") is not gates[key]:
            raise ReleaseValidationError(f"validity gate receipt disagrees for {key}")

    hashes = {
        name: sha256(root / name)
        for name in RESULT_FILES
    }
    if report.get("cells_csv_sha256") != hashes["confirmation_cells.csv"] \
            or report.get("contrasts_csv_sha256") != hashes["confirmation_contrasts.csv"]:
        raise ReleaseValidationError("analysis/CSV hash bindings differ")
    return {
        "root": root,
        "report": dict(report),
        "protocol": dict(protocol),
        "runs": list(run_rows),
        "attempts": list(attempts),
        "summary": dict(summary),
        "cells": cells,
        "contrast_rows": contrast_rows,
        "hashes": hashes,
    }


def _validate_canonical_restart_audit(
    audit: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    log_root: Path | None,
) -> dict[str, Any]:
    """Validate the repository-specific rich schema-v2 audit."""

    if audit.get("study") != "lewm-v8-v18-confirmation":
        raise ReleaseValidationError("canonical restart audit study differs")
    generator = audit.get("generator")
    if not isinstance(generator, Mapping):
        raise ReleaseValidationError("canonical restart audit generator receipt is absent")
    generator_digest = require_hash(
        generator.get("sha256"), "canonical restart audit generator"
    )
    generator_path = generator.get("path")
    if isinstance(generator_path, str) and Path(generator_path).is_file() \
            and sha256(Path(generator_path)) != generator_digest:
        raise ReleaseValidationError("canonical restart audit generator bytes differ")
    protocol = audit.get("protocol")
    if not isinstance(protocol, Mapping) \
            or protocol.get("sha256") != bundle["hashes"]["confirmation_protocol.json"] \
            or protocol.get("commands_sha256") != bundle["protocol"]["commands_sha256"]:
        raise ReleaseValidationError("canonical restart audit protocol binding differs")
    bound = audit.get("bound_receipts")
    aliases = {
        "protocol": "confirmation_protocol.json",
        "summary": "confirmation_summary.json",
        "runs": "confirmation_runs.json",
        "attempts": "confirmation_attempts.json",
        "analysis": "confirmation_analysis.json",
        "cells_csv": "confirmation_cells.csv",
        "contrasts_csv": "confirmation_contrasts.csv",
    }
    if not isinstance(bound, Mapping):
        raise ReleaseValidationError("canonical restart audit bound receipts are absent")
    for key, filename in aliases.items():
        receipt = bound.get(key)
        if not isinstance(receipt, Mapping) \
                or receipt.get("sha256") != bundle["hashes"][filename]:
            raise ReleaseValidationError(
                f"canonical restart audit final binding differs for {filename}"
            )
    snapshot = audit.get("final_study_snapshot")
    if not isinstance(snapshot, Mapping) \
            or snapshot.get("status") != "COMPLETE" \
            or snapshot.get("planned_cells") != 200 \
            or snapshot.get("completed_valid_cells") != 200 \
            or snapshot.get("absent_cells") != 0 \
            or snapshot.get("failed_or_invalid_cells") != 0 \
            or snapshot.get("analysis_complete") is not True:
        raise ReleaseValidationError("canonical restart audit final snapshot is incomplete")
    terminal = audit.get("terminal_preconditions")
    if not isinstance(terminal, Mapping) \
            or terminal.get("runner_trainer_analyzer_processes_active") != 0 \
            or terminal.get("runner_lock_absent") is not True:
        raise ReleaseValidationError("canonical restart audit lacks terminal quiescence")
    resume_events = audit.get("resume_events")
    if not isinstance(resume_events, list) or len(resume_events) != 2:
        raise ReleaseValidationError("canonical restart audit must bind two resumes")
    expected_counts = ((136, 64, 4), (180, 20, 1))
    for index, (event, expected) in enumerate(zip(resume_events, expected_counts, strict=True), 1):
        if not isinstance(event, Mapping) or event.get("resume_index") != index:
            raise ReleaseValidationError("canonical restart audit resume order differs")
        counts = event.get("pre_resume_counts")
        complete, absent, replacements = expected
        if not isinstance(counts, Mapping) \
                or counts.get("valid_complete") != complete \
                or counts.get("absent") != absent \
                or counts.get("partial_or_invalid_core") != 0 \
                or event.get("replacement_cell_count") != replacements \
                or event.get("resume_policy") != "complete_cell_only":
            raise ReleaseValidationError("canonical restart audit resume counts differ")
    lineages = audit.get("attempt_lineages")
    expected_cells = {
        ("acrobot.swingup", "vicreg_ssm", 18005),
        ("manipulator.bring_ball", "vicreg_ssm", 18005),
        ("quadruped.run", "vicreg_ssm", 18005),
        ("swimmer.swimmer15", "vicreg_ssm", 18005),
        ("stacker.stack_4", "vicreg_hacssmv8_static", 18003),
    }
    if not isinstance(lineages, list) or len(lineages) != 5:
        raise ReleaseValidationError("canonical restart audit must bind five lineages")
    actual_cells = set()
    for lineage in lineages:
        if not isinstance(lineage, Mapping) or not isinstance(lineage.get("cell"), Mapping):
            raise ReleaseValidationError("canonical restart lineage is malformed")
        cell = lineage["cell"]
        key = (str(cell.get("task")), str(cell.get("design")), int(cell.get("seed", -1)))
        actual_cells.add(key)
        interrupted = lineage.get("interrupted_attempt")
        replacement = lineage.get("replacement_attempt")
        if not isinstance(interrupted, Mapping) or not isinstance(replacement, Mapping):
            raise ReleaseValidationError(f"canonical lineage attempts are absent for {key}")
        if not 0 <= int(interrupted.get("last_logged_epoch", -1)) < 100 \
                or replacement.get("first_logged_epoch") != 1 \
                or replacement.get("last_logged_epoch") != 100:
            raise ReleaseValidationError(f"canonical restart epochs differ for {key}")
        interrupted_wandb = interrupted.get("wandb")
        replacement_wandb = replacement.get("wandb")
        if not isinstance(interrupted_wandb, Mapping) \
                or interrupted_wandb.get("state") != "crashed" \
                or not isinstance(replacement_wandb, Mapping) \
                or replacement_wandb.get("state") != "finished":
            raise ReleaseValidationError(f"canonical W&B lineage state differs for {key}")
        for attempt, name in ((interrupted, "interrupted"), (replacement, "replacement")):
            receipt = attempt.get("log")
            if not isinstance(receipt, Mapping):
                raise ReleaseValidationError(f"canonical {name} log receipt absent for {key}")
            digest = require_hash(receipt.get("sha256"), f"canonical {name} log {key}")
            relative = receipt.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() \
                    or ".." in Path(relative).parts:
                raise ReleaseValidationError(f"unsafe canonical restart log path {relative!r}")
            if log_root is not None:
                candidate = log_root / Path(relative).name
                if not candidate.is_file() or sha256(candidate) != digest:
                    raise ReleaseValidationError(f"canonical restart log hash differs: {candidate}")
    if actual_cells != expected_cells:
        raise ReleaseValidationError("canonical restart lineage cell set differs")
    artifact_binding = audit.get("artifact_binding")
    if not isinstance(artifact_binding, Mapping) \
            or artifact_binding.get("checked_cells") != 200 \
            or artifact_binding.get("finished_local_wandb_receipts") != 200 \
            or artifact_binding.get("artifact_manifest_bound_by_analysis") is not True:
        raise ReleaseValidationError("canonical restart artifact binding is incomplete")
    remote = audit.get("remote_wandb_terminal_observation")
    remote_runs = remote.get("runs") if isinstance(remote, Mapping) else None
    if not isinstance(remote_runs, list) or len(remote_runs) != 10:
        raise ReleaseValidationError("canonical audit lacks ten remote W&B observations")
    states = [row.get("state") for row in remote_runs if isinstance(row, Mapping)]
    if sorted(states) != ["crashed"] * 5 + ["finished"] * 5:
        raise ReleaseValidationError("canonical remote W&B terminal states differ")
    return dict(audit)


def validate_restart_audit(
    path: Path,
    bundle: Mapping[str, Any],
    *,
    log_root: Path | None = None,
) -> dict[str, Any]:
    """Validate restart-audit schema v2 and its final-result bindings."""

    audit = read_json(path)
    if not isinstance(audit, Mapping) or audit.get("schema_version") != 2:
        raise ReleaseValidationError("restart audit is not schema v2 for V18")
    if audit.get("study") == "lewm-v8-v18-confirmation":
        return _validate_canonical_restart_audit(
            audit, bundle, log_root=log_root
        )
    if audit.get("scope") != "v18_process_level_restart_audit":
        raise ReleaseValidationError("restart audit scope is not recognized")
    protocol = bundle["protocol"]
    bindings = {
        "protocol_sha256": bundle["hashes"]["confirmation_protocol.json"],
        "commands_sha256": protocol["commands_sha256"],
    }
    for key, expected in bindings.items():
        if audit.get(key) != expected:
            raise ReleaseValidationError(f"restart audit {key} binding differs")
    interruptions = audit.get("interruptions")
    if not isinstance(interruptions, list) or len(interruptions) != 2:
        raise ReleaseValidationError("restart audit must record exactly two interruptions")
    expected_observations = ((136, 64, 4), (180, 20, 1))
    seen_sequences: list[int] = []
    for interruption, expected in zip(interruptions, expected_observations, strict=True):
        if not isinstance(interruption, Mapping):
            raise ReleaseValidationError("restart audit interruption is malformed")
        seen_sequences.append(int(interruption.get("sequence", -1)))
        observation = interruption.get("observation")
        resume = interruption.get("resume")
        attempts = interruption.get("interrupted_attempts")
        if not isinstance(observation, Mapping) or not isinstance(resume, Mapping) \
                or not isinstance(attempts, list):
            raise ReleaseValidationError("restart audit interruption fields are malformed")
        complete, absent, count = expected
        if observation.get("complete_valid_cells") != complete \
                or observation.get("absent_cells") != absent \
                or observation.get("partial_or_invalid_core_cells") != 0 \
                or observation.get("interrupted_trainers") != count:
            raise ReleaseValidationError("restart audit observed cell counts differ")
        if resume.get("validated_existing_cells") != complete \
                or resume.get("policy") != "complete_cell_only":
            raise ReleaseValidationError("restart audit resume policy/count differs")
        if len(attempts) != count:
            raise ReleaseValidationError("restart audit interrupted-attempt count differs")
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise ReleaseValidationError("restart audit attempt is malformed")
            key = (attempt.get("task"), attempt.get("seed"), attempt.get("design"))
            if key not in expected_grid():
                raise ReleaseValidationError(f"restart audit cell is outside grid: {key}")
            if not 0 <= int(attempt.get("last_logged_epoch", -1)) < 100 \
                    or attempt.get("restart_terminal_epoch") != 100:
                raise ReleaseValidationError(f"restart epoch receipt is invalid for {key}")
            for field in ("interrupted_log_sha256", "restart_log_sha256"):
                digest = require_hash(attempt.get(field), f"restart audit {field}")
                relative = attempt.get(field.removesuffix("_sha256"))
                if relative is not None:
                    if not isinstance(relative, str) or Path(relative).is_absolute() \
                            or ".." in Path(relative).parts:
                        raise ReleaseValidationError(f"unsafe restart log path {relative!r}")
                    if log_root is not None:
                        candidate = log_root / relative
                        if not candidate.is_file() or sha256(candidate) != digest:
                            raise ReleaseValidationError(f"restart log hash differs: {candidate}")
    if seen_sequences != [1, 2]:
        raise ReleaseValidationError("restart interruption sequence is not [1, 2]")
    final = audit.get("final_bindings")
    if not isinstance(final, Mapping) or final.get("completed_valid_cells") != 200:
        raise ReleaseValidationError("restart audit lacks final 200-cell binding")
    final_expected = {
        "summary_sha256": bundle["hashes"]["confirmation_summary.json"],
        "runs_sha256": bundle["hashes"]["confirmation_runs.json"],
        "attempts_sha256": bundle["hashes"]["confirmation_attempts.json"],
        "analysis_sha256": bundle["hashes"]["confirmation_analysis.json"],
    }
    for key, expected in final_expected.items():
        if final.get(key) != expected:
            raise ReleaseValidationError(f"restart final binding differs for {key}")
    return dict(audit)


def restart_interruption_count(audit: Mapping[str, Any]) -> int:
    if audit.get("study") == "lewm-v8-v18-confirmation":
        events = audit.get("resume_events")
        return len(events) if isinstance(events, list) else 0
    interruptions = audit.get("interruptions")
    return len(interruptions) if isinstance(interruptions, list) else 0


def restart_text(audit: Mapping[str, Any]) -> tuple[str, str]:
    if audit.get("study") == "lewm-v8-v18-confirmation":
        events = audit["resume_events"]
        lineages = audit["attempt_lineages"]
        brief = (
            "Two process-level interruptions were audited before final analysis. "
            "The first occurred with 136 valid cells and required four complete-cell "
            "restarts; the second occurred with 180 valid cells and required one. "
            "The schema-v2 audit verifies interrupted and replacement logs, local and "
            "remote W&B terminal states, replacement artifact hashes, terminal runner "
            "quiescence, and the final attempts, runs, summary, CSV, and analysis bytes; "
            "all five replacements reached epoch 100."
        )
        details = []
        for index, lineage in enumerate(lineages):
            cell = lineage["cell"]
            interrupted = lineage["interrupted_attempt"]
            replacement = lineage["replacement_attempt"]
            phase = 1 if index < 4 else 2
            details.append(
                f"Interruption {phase}, `{cell['task']}/{cell['design']}/s{cell['seed']}`: "
                f"last interrupted epoch {interrupted['last_logged_epoch']}; "
                f"interrupted/replacement log hashes "
                f"`{interrupted['log']['sha256'][:12]}`/"
                f"`{replacement['log']['sha256'][:12]}`; replacement epochs 1--100."
            )
        appendix = " ".join(details) + (
            " The runner's JSON attempts ledger records terminal subprocess returns, so "
            "process-killed attempts are supplied by this independently verified schema-v2 "
            "lineage receipt. The audit's separately classified parent-only pause did not "
            "signal the trainer and is not counted as an additional restart."
        )
        return brief, appendix
    interruptions = audit["interruptions"]
    counts = [len(value["interrupted_attempts"]) for value in interruptions]
    brief = (
        "Two process-level interruptions were audited before final analysis. "
        f"The first occurred with 136 valid cells and interrupted {counts[0]} active "
        "trainers; the second occurred with 180 valid cells and interrupted one active "
        "trainer. In both cases the resume validator preserved only complete cells and "
        "reran each interrupted cell from scratch. Restart-audit v2 binds both pairs of "
        "interrupted/restart log hashes and the final attempts, runs, summary, and analysis "
        "hashes; all restarted cells reached epoch 100."
    )
    details: list[str] = []
    for interruption in interruptions:
        cells = []
        for attempt in interruption["interrupted_attempts"]:
            cells.append(
                f"`{attempt['task']}/{attempt['design']}/s{attempt['seed']}` "
                f"(last interrupted epoch {attempt['last_logged_epoch']}; interrupted/restart "
                f"log hashes `{attempt['interrupted_log_sha256'][:12]}`/"
                f"`{attempt['restart_log_sha256'][:12]}`; restart epoch 100)"
            )
        observation = interruption["observation"]
        details.append(
            f"Interruption {interruption['sequence']}: "
            f"{observation['complete_valid_cells']} valid and {observation['absent_cells']} "
            f"absent cells; " + "; ".join(cells) + "."
        )
    appendix = " ".join(details) + (
        " The runner's JSON attempts ledger records only terminal subprocess returns, so "
        "process-killed attempts are intentionally supplied by this separately hash-bound "
        "manual restart receipt rather than inferred from the final ledger."
    )
    return brief, appendix


def render_template(text: str, values: Mapping[str, str], *, label: str) -> str:
    for name, value in values.items():
        text = text.replace("{{" + name + "}}", str(value))
    leftovers = sorted(set(PLACEHOLDER.findall(text)))
    if leftovers:
        raise ReleaseValidationError(f"unrendered {label} placeholders: {leftovers}")
    if re.search(r"(?i)(?:^|\W)(?:nan|[+-]?inf)(?:$|\W)", text):
        raise ReleaseValidationError(f"{label} contains a nonfinite numeric token")
    return text


def scan_forbidden(paths: Iterable[Path], tokens: Iterable[str]) -> None:
    forbidden = sorted({token for token in tokens if token}, key=len, reverse=True)
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="strict")
        leaked = [token for token in forbidden if token.casefold() in content.casefold()]
        if leaked:
            raise ReleaseValidationError(
                f"double-blind identity leak in {path.name}: {leaked}"
            )
