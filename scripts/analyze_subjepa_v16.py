#!/usr/bin/env python3
"""Validate and aggregate the exact 144-cell Sub-JEPA-v16 development grid.

The report is descriptive adaptive-development evidence.  It computes paired
Sub-JEPA-vs-fullSIG, memory-vs-none, and checkpoint-integrator contrasts over
the common task/seed cells, plus representation-rank and convergence summaries.
Missing, duplicate, malformed, non-finite, or provenance-drifted cells make the
report ``INCOMPLETE_OR_INVALID`` while still producing an auditable partial CSV.
The complete report also includes the predeclared Sub-JEPA-vs-VICReg and
K=32-vs-K=16 stress contrasts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_subjepa_v16 import (
    CORE_ARTIFACTS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STUDY,
    DESIGNS,
    EPOCHS,
    FULLSIG_DESIGNS,
    LOCK_NAME,
    MEMORY_VARIANTS,
    OBJECTIVE_FAMILIES,
    PROTOCOL_NAME,
    RUNS_NAME,
    SEEDS,
    SOURCE_PATHS,
    TASKS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    cell_specs,
    command_records,
    data_paths,
    design_parts,
    file_sha256,
    json_sha256,
    load_json,
    run_directory,
    validate_core_artifacts,
)


PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
INTEGRATOR = "initial_encoder_integrator_probe_nmse"
RANK = "encoder_covariance_effective_rank"
CONVERGENCE = "predictive_loss_convergence_relative_change"
EXPECTED_CELLS = len(TASKS) * len(DESIGNS) * len(SEEDS)
RANK_THRESHOLD = 16.0
CONVERGENCE_ABS_THRESHOLD = 0.05

ANALYSIS_NAME = "development_analysis.json"
CELLS_CSV_NAME = "development_cells.csv"
CONTRASTS_CSV_NAME = "development_contrasts.csv"


class AnalysisError(RuntimeError):
    """Closed-world development artifact validation failure."""


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{label} is non-finite")
    return result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not bool(np.isfinite(array).all()):
        raise AnalysisError("cannot summarize an empty or non-finite vector")
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / math.sqrt(len(array))
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "sem": sem,
        "ci95_low": float(array.mean() - 1.96 * sem),
        "ci95_high": float(array.mean() + 1.96 * sem),
    }


def _protocol_errors(root: Path, protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    exact = {
        "schema_version": 1,
        "scope": "subjepa_v16_excluded_adaptive_development",
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "objective_families": list(OBJECTIVE_FAMILIES),
        "memory_variants": list(MEMORY_VARIANTS),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "runs": EXPECTED_CELLS,
        "gpus": ["0", "1", "2", "3"],
        "task_pinned_gpu": dict(zip(TASKS, ("0", "1", "2", "3"), strict=True)),
        "study": DEFAULT_STUDY,
        "output_root": str(root),
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_mode": "online",
        "resume_supported": True,
        "core_artifacts": list(CORE_ARTIFACTS),
    }
    for key, value in exact.items():
        if protocol.get(key) != value:
            errors.append(f"protocol {key} differs")

    python = protocol.get("python")
    log_root = protocol.get("log_root")
    wandb_enabled = protocol.get("wandb_enabled")
    if not isinstance(python, str) or not Path(python).is_file():
        errors.append("protocol Python path is missing")
    if not isinstance(log_root, str) or not Path(log_root).is_absolute():
        errors.append("protocol log_root must be absolute")
    if not isinstance(wandb_enabled, bool):
        errors.append("protocol wandb_enabled must be boolean")

    source = protocol.get("source_sha256")
    expected_source_set = {str(path) for path in SOURCE_PATHS}
    if not isinstance(source, Mapping) or set(source) != expected_source_set:
        errors.append("protocol source manifest differs")
    else:
        for relative in SOURCE_PATHS:
            path = ROOT / relative
            if not path.is_file() or source[str(relative)] != file_sha256(path):
                errors.append(f"source SHA-256 differs: {relative}")

    data = protocol.get("data")
    if not isinstance(data, Mapping) or set(data) != set(TASKS):
        errors.append("protocol data manifest differs")
    else:
        for task in TASKS:
            train_path, val_path = data_paths(task)
            for split, path in (("train", train_path), ("val", val_path)):
                record = data.get(task)
                if not isinstance(record, Mapping):
                    errors.append(f"protocol data record is invalid: {task}")
                    break
                if (record.get(split) != str(path) or not path.is_file()
                        or record.get(f"{split}_sha256") != file_sha256(path)):
                    errors.append(f"data receipt differs: {task}/{split}")

    if isinstance(python, str) and isinstance(wandb_enabled, bool):
        expected_commands = command_records(
            python, root, DEFAULT_STUDY, EPOCHS, wandb=wandb_enabled)
        if protocol.get("commands") != expected_commands:
            errors.append("protocol commands differ token-for-token")
        if protocol.get("commands_sha256") != json_sha256(expected_commands):
            errors.append("protocol command hash differs")
    return errors


def _validate_metric_metadata(
        metrics: Mapping[str, Any], task: str, design: str, seed: int) -> None:
    family, memory, tokens = design_parts(design)
    expected = {
        "regularizer": family,
        "memory_architecture": memory,
        "num_subspaces": (
            16 if family == "subjepa16" else
            32 if family == "subjepa32" else
            1 if family == "fullsig" else None),
        "confirmation_evidence": False,
    }
    del tokens
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise AnalysisError(
                f"{task}/{design}/s{seed}: metric metadata {key} differs")
    for key in (PRIMARY, INTEGRATOR, RANK, CONVERGENCE):
        finite(metrics.get(key), f"{task}/{design}/s{seed}:{key}")


def _load_ledger(root: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    path = root / RUNS_NAME
    value = load_json(path)
    if not isinstance(value, list):
        raise AnalysisError(f"{path} must contain a list")
    result: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise AnalysisError(f"{path} contains a non-object row")
        pair = (str(row.get("task")), str(row.get("design")), int(row.get("seed")))
        if pair in result:
            raise AnalysisError(f"duplicate runner receipt {pair}")
        result[pair] = row
    return result


def load_rows(
        root: Path, protocol: Mapping[str, Any]
        ) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        ledger = _load_ledger(root)
    except (AnalysisError, OSError, ValueError, TypeError) as exc:
        return rows, [str(exc)]

    expected_cells = set(cell_specs())
    if set(ledger) != expected_cells:
        missing = expected_cells - set(ledger)
        extra = set(ledger) - expected_cells
        if missing:
            errors.append(f"runner ledger missing {len(missing)} cells")
        if extra:
            errors.append(f"runner ledger has {len(extra)} unexpected cells")

    for task, design, seed in cell_specs():
        receipt = ledger.get((task, design, seed))
        if receipt is None:
            continue
        if receipt.get("status") != "complete":
            errors.append(
                f"{task}/{design}/s{seed}: runner status {receipt.get('status')!r}")
            continue
        try:
            validation = validate_core_artifacts(
                root, task, design, seed, EPOCHS)
            metrics = validation["metrics"]
            _validate_metric_metadata(metrics, task, design, seed)
            if receipt.get("artifact_sha256") != validation["artifact_sha256"]:
                raise AnalysisError(
                    f"{task}/{design}/s{seed}: runner artifact hashes differ")
            command_index = (
                TASKS.index(task) * len(SEEDS) * len(DESIGNS)
                + SEEDS.index(seed) * len(DESIGNS) + DESIGNS.index(design))
            command = protocol["commands"][command_index]["argv"]
            if receipt.get("command_sha256") != json_sha256(command):
                raise AnalysisError(
                    f"{task}/{design}/s{seed}: runner command hash differs")
            family, memory, tokens = design_parts(design)
            wandb_path = run_directory(root, task, design, seed) / "wandb_run.json"
            wandb_state = None
            if wandb_path.is_file():
                wandb = load_json(wandb_path)
                if isinstance(wandb, Mapping):
                    wandb_state = wandb.get("state")
            rows.append({
                "task": task,
                "design": design,
                "seed": seed,
                "family": family,
                "memory": memory,
                "tokens": tokens,
                "metrics": metrics,
                "directory": validation["directory"],
                "artifact_sha256": validation["artifact_sha256"],
                "wandb_state": wandb_state,
            })
        except (AnalysisError, OSError, ValueError, TypeError, RuntimeError) as exc:
            errors.append(str(exc))
    return rows, errors


def _design_metric(
        rows: Sequence[Mapping[str, Any]], design: str, key: str
        ) -> dict[tuple[str, int], float]:
    result = {
        (str(row["task"]), int(row["seed"])):
            finite(row["metrics"].get(key), f"{design}:{key}")
        for row in rows if row["design"] == design
    }
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    if set(result) != expected:
        raise AnalysisError(f"{design}/{key}: incomplete task-seed grid")
    return result


def paired_contrast(
        rows: Sequence[Mapping[str, Any]], candidate: str, reference: str,
        *, kind: str, candidate_key: str = PRIMARY,
        reference_key: str = PRIMARY) -> dict[str, Any]:
    candidate_values = _design_metric(rows, candidate, candidate_key)
    if reference == "checkpoint_initial_encoder_integrator":
        reference_values = _design_metric(rows, candidate, reference_key)
    else:
        reference_values = _design_metric(rows, reference, reference_key)
    keys = [(task, seed) for task in TASKS for seed in SEEDS]
    cand = np.asarray([candidate_values[key] for key in keys], dtype=np.float64)
    ref = np.asarray([reference_values[key] for key in keys], dtype=np.float64)
    if bool((ref == 0).any()):
        raise AnalysisError(f"{candidate} versus {reference}: zero reference metric")
    relative = (ref - cand) / ref
    absolute = ref - cand
    ties = np.isclose(cand, ref, rtol=1e-12, atol=1e-12)
    by_task = {}
    for task in TASKS:
        selected = np.asarray([name == task for name, _ in keys])
        by_task[task] = {
            "candidate_mean": float(cand[selected].mean()),
            "reference_mean": float(ref[selected].mean()),
            "paired_relative_reduction_mean": float(relative[selected].mean()),
            "wins": int((cand[selected] < ref[selected]).sum()),
            "ties": int(ties[selected].sum()),
            "losses": int((cand[selected] > ref[selected]).sum()),
        }
    result = {
        "kind": kind,
        "candidate": candidate,
        "reference": reference,
        "candidate_metric": candidate_key,
        "reference_metric": reference_key,
        "pairs": len(keys),
        "candidate_mean": float(cand.mean()),
        "reference_mean": float(ref.mean()),
        "equal_cell_relative_reduction": float((ref.mean() - cand.mean()) / ref.mean()),
        "paired_absolute_improvement": _summary(absolute),
        "paired_relative_reduction": _summary(relative),
        "wins": int((cand < ref).sum()),
        "ties": int(ties.sum()),
        "losses": int((cand > ref).sum()),
        "by_task": by_task,
    }
    return result


def design_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for design in DESIGNS:
        selected = [row for row in rows if row["design"] == design]
        if len(selected) != len(TASKS) * len(SEEDS):
            raise AnalysisError(f"{design}: incomplete design cells")
        metrics: dict[str, Any] = {}
        for key in (PRIMARY, CLEAN, INTEGRATOR, RANK, CONVERGENCE,
                    "final_val_loss", "val_predictive_loss",
                    "mean_epoch_seconds", "peak_vram_bytes"):
            values = []
            for row in selected:
                value = row["metrics"].get(key)
                if value is not None:
                    values.append(finite(value, f"{design}:{key}"))
            if values:
                metrics[key] = _summary(values)
        result[design] = {
            "family": selected[0]["family"],
            "memory": selected[0]["memory"],
            "tokens": selected[0]["tokens"],
            "cells": len(selected),
            "metrics": metrics,
        }
    return result


def diagnostics_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_design: dict[str, Any] = {}
    for design in DESIGNS:
        selected = [row for row in rows if row["design"] == design]
        ranks = np.asarray([
            finite(row["metrics"].get(RANK), f"{design}:{RANK}")
            for row in selected], dtype=np.float64)
        convergence = np.asarray([
            finite(row["metrics"].get(CONVERGENCE), f"{design}:{CONVERGENCE}")
            for row in selected], dtype=np.float64)
        by_design[design] = {
            "rank": {
                **_summary(ranks),
                "threshold": RANK_THRESHOLD,
                "cells_at_or_above_threshold": int((ranks >= RANK_THRESHOLD).sum()),
                "all_cells_at_or_above_threshold": bool(
                    (ranks >= RANK_THRESHOLD).all()),
            },
            "convergence": {
                **_summary(convergence),
                "abs_summary": _summary(np.abs(convergence)),
                "abs_threshold": CONVERGENCE_ABS_THRESHOLD,
                "cells_abs_at_or_below_threshold": int(
                    (np.abs(convergence) <= CONVERGENCE_ABS_THRESHOLD).sum()),
                "negative_cells": int((convergence < 0).sum()),
                "all_abs_at_or_below_threshold": bool(
                    (np.abs(convergence) <= CONVERGENCE_ABS_THRESHOLD).all()),
            },
        }
    return {
        "rank_threshold": RANK_THRESHOLD,
        "convergence_abs_threshold": CONVERGENCE_ABS_THRESHOLD,
        "by_design": by_design,
    }


def analyze_rows(
        rows: Sequence[Mapping[str, Any]], errors: Sequence[str]
        ) -> dict[str, Any]:
    complete = len(rows) == EXPECTED_CELLS and not errors
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "subjepa_v16_excluded_adaptive_development",
        "status": "COMPLETE" if complete else "INCOMPLETE_OR_INVALID",
        "official_confirmation_result": False,
        "expected_cells": EXPECTED_CELLS,
        "completed_valid_cells": len(rows),
        "artifact_integrity_passed": complete,
        "artifact_integrity_errors": list(errors),
        "primary_metric": PRIMARY,
    }
    if not complete:
        return result

    summaries = design_summary(rows)
    subjepa: dict[str, Any] = {}
    for family in ("subjepa16", "subjepa32"):
        for memory in MEMORY_VARIANTS:
            candidate = f"{family}_{memory}"
            subjepa[candidate] = paired_contrast(
                rows, candidate, f"fullsig_{memory}",
                kind="subjepa_vs_fullsig")
    subjepa_vs_vicreg = {
        f"subjepa16_{memory}": paired_contrast(
            rows, f"subjepa16_{memory}", f"vicreg_{memory}",
            kind="subjepa16_vs_vicreg")
        for memory in MEMORY_VARIANTS
    }
    subspace_stress = {
        f"subjepa32_{memory}": paired_contrast(
            rows, f"subjepa32_{memory}", f"subjepa16_{memory}",
            kind="subjepa32_vs_subjepa16_stress")
        for memory in MEMORY_VARIANTS
    }
    memory_contrasts: dict[str, Any] = {}
    for family in OBJECTIVE_FAMILIES:
        for memory in ("ssm", "hacssmv8"):
            candidate = f"{family}_{memory}"
            memory_contrasts[candidate] = paired_contrast(
                rows, candidate, f"{family}_none", kind="memory_vs_none")
    integrator = {
        design: paired_contrast(
            rows, design, "checkpoint_initial_encoder_integrator",
            kind="checkpoint_integrator", reference_key=INTEGRATOR)
        for design in DESIGNS
    }
    result.update({
        "design_summary": summaries,
        "subjepa_vs_fullsig": subjepa,
        "subjepa16_vs_vicreg": subjepa_vs_vicreg,
        "subjepa32_vs_subjepa16_stress": subspace_stress,
        "memory_vs_none": memory_contrasts,
        "checkpoint_integrator_comparison": integrator,
        "representation_and_convergence": diagnostics_summary(rows),
        "wandb_receipts": {
            "finished": sum(row.get("wandb_state") == "finished" for row in rows),
            "sync_failed": sum(row.get("wandb_state") == "sync_failed" for row in rows),
            "disabled": sum(row.get("wandb_state") == "not_requested" for row in rows),
            "missing_or_other": sum(row.get("wandb_state") not in {
                "finished", "sync_failed", "not_requested"} for row in rows),
        },
    })
    return result


def cell_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        metrics = row["metrics"]
        result.append({
            "task": row["task"],
            "seed": row["seed"],
            "design": row["design"],
            "family": row["family"],
            "memory": row["memory"],
            "tokens": "" if row["tokens"] is None else row["tokens"],
            PRIMARY: metrics.get(PRIMARY),
            CLEAN: metrics.get(CLEAN),
            INTEGRATOR: metrics.get(INTEGRATOR),
            RANK: metrics.get(RANK),
            CONVERGENCE: metrics.get(CONVERGENCE),
            "final_val_loss": metrics.get("final_val_loss"),
            "val_predictive_loss": metrics.get("val_predictive_loss"),
            "mean_epoch_seconds": metrics.get("mean_epoch_seconds"),
            "peak_vram_bytes": metrics.get("peak_vram_bytes"),
            "wandb_state": row.get("wandb_state"),
            "directory": row["directory"],
        })
    return result


def contrast_csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for section in (
            "subjepa_vs_fullsig", "subjepa16_vs_vicreg",
            "subjepa32_vs_subjepa16_stress", "memory_vs_none",
            "checkpoint_integrator_comparison"):
        values = analysis.get(section)
        if not isinstance(values, Mapping):
            continue
        for contrast in values.values():
            relative = contrast["paired_relative_reduction"]
            absolute = contrast["paired_absolute_improvement"]
            result.append({
                "kind": contrast["kind"],
                "candidate": contrast["candidate"],
                "reference": contrast["reference"],
                "candidate_metric": contrast["candidate_metric"],
                "reference_metric": contrast["reference_metric"],
                "pairs": contrast["pairs"],
                "candidate_mean": contrast["candidate_mean"],
                "reference_mean": contrast["reference_mean"],
                "equal_cell_relative_reduction": contrast[
                    "equal_cell_relative_reduction"],
                "paired_relative_reduction_mean": relative["mean"],
                "paired_relative_reduction_std": relative["std"],
                "paired_relative_reduction_ci95_low": relative["ci95_low"],
                "paired_relative_reduction_ci95_high": relative["ci95_high"],
                "paired_absolute_improvement_mean": absolute["mean"],
                "wins": contrast["wins"],
                "ties": contrast["ties"],
                "losses": contrast["losses"],
            })
    return result


CELL_CSV_FIELDS = (
    "task", "seed", "design", "family", "memory", "tokens", PRIMARY,
    CLEAN, INTEGRATOR, RANK, CONVERGENCE, "final_val_loss",
    "val_predictive_loss", "mean_epoch_seconds", "peak_vram_bytes",
    "wandb_state", "directory",
)
CONTRAST_CSV_FIELDS = (
    "kind", "candidate", "reference", "candidate_metric", "reference_metric",
    "pairs", "candidate_mean", "reference_mean",
    "equal_cell_relative_reduction", "paired_relative_reduction_mean",
    "paired_relative_reduction_std", "paired_relative_reduction_ci95_low",
    "paired_relative_reduction_ci95_high", "paired_absolute_improvement_mean",
    "wins", "ties", "losses",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    write = parser.add_mutually_exclusive_group()
    write.add_argument("--write", dest="write", action="store_true")
    write.add_argument("--no-write", dest="write", action="store_false")
    parser.set_defaults(write=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = (args.root if args.root.is_absolute() else ROOT / args.root).resolve()
    errors: list[str] = []
    protocol: Mapping[str, Any] | None = None
    try:
        value = load_json(root / PROTOCOL_NAME)
        if not isinstance(value, Mapping):
            raise AnalysisError("development protocol is not an object")
        protocol = value
        errors.extend(_protocol_errors(root, protocol))
    except (AnalysisError, OSError, ValueError, TypeError, RuntimeError) as exc:
        errors.append(str(exc))
    if (root / LOCK_NAME).exists():
        errors.append("development runner lock still exists")

    rows: list[dict[str, Any]] = []
    if protocol is not None and not errors:
        loaded, row_errors = load_rows(root, protocol)
        rows.extend(loaded)
        errors.extend(row_errors)
    analysis = analyze_rows(rows, errors)
    if protocol is not None and (root / PROTOCOL_NAME).is_file():
        analysis["development_protocol_sha256"] = file_sha256(root / PROTOCOL_NAME)
        analysis["commands_sha256"] = protocol.get("commands_sha256")
        analysis["wandb_enabled"] = protocol.get("wandb_enabled")
    rendered = json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")

    if args.write:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(root / ANALYSIS_NAME, rendered)
        cell_rows = cell_csv_rows(rows)
        atomic_write_text(
            root / CELLS_CSV_NAME, csv_text(cell_rows, CELL_CSV_FIELDS))
        contrast_rows = contrast_csv_rows(analysis)
        atomic_write_text(
            root / CONTRASTS_CSV_NAME,
            csv_text(contrast_rows, CONTRAST_CSV_FIELDS))
    raise SystemExit(0 if analysis["artifact_integrity_passed"] else 2)


if __name__ == "__main__":
    main()
