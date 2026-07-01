#!/usr/bin/env python3
"""Validate and aggregate the frozen 72-cell AutoVISReg-v17 grid.

All comparisons are paired within task, optimizer seed, and memory backbone.
Missing, malformed, nonfinite, provenance-drifted, or incompletely synchronized
cells make the report fail closed.  Scientific labels remain descriptive
opened-cache adaptive-development labels.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_autovisreg_v17 import (
    CORE_ARTIFACTS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STUDY,
    DESIGNS,
    EPOCHS,
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
VARIANCE = "encoder_mean_channel_variance"
RANK = "encoder_covariance_effective_rank"
CONVERGENCE = "predictive_loss_convergence_relative_change"
VAL_PREDICTIVE = "val_predictive_loss"
GRADIENT_METRICS = (
    "train_gradient_prediction_norm",
    "train_gradient_regularizer_norm",
    "train_gradient_cosine",
    "train_gradient_adaptive_scale",
    "train_gradient_preclip_norm",
    "train_gradient_clip_fraction",
    "train_gradient_conflict_fraction",
)
SUMMARY_METRICS = (
    PRIMARY, CLEAN, INTEGRATOR, VARIANCE, RANK, CONVERGENCE,
    VAL_PREDICTIVE, "final_val_loss", "mean_epoch_seconds",
    "peak_vram_bytes", *GRADIENT_METRICS,
)
EXPECTED_CELLS = len(TASKS) * len(DESIGNS) * len(SEEDS)
RANK_THRESHOLD = 16.0
VARIANCE_FLOOR = 1e-4
CONVERGENCE_ABS_THRESHOLD = 0.05

ANALYSIS_NAME = "development_analysis.json"
CELLS_CSV_NAME = "development_cells.csv"
CONTRASTS_CSV_NAME = "development_contrasts.csv"


class AnalysisError(RuntimeError):
    """Closed-grid analysis invariant failed."""


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{label} is nonfinite")
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
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not bool(np.isfinite(array).all()):
        raise AnalysisError("cannot summarize an empty or nonfinite vector")
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / math.sqrt(len(array))
    return {
        "n": int(len(array)), "mean": float(array.mean()), "std": std,
        "min": float(array.min()), "max": float(array.max()), "sem": sem,
        "ci95_low": float(array.mean() - 1.96 * sem),
        "ci95_high": float(array.mean() + 1.96 * sem),
    }


def _protocol_errors(root: Path, protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    exact = {
        "schema_version": 1,
        "scope": "autovisreg_v17_excluded_adaptive_development",
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
        "resume_granularity": "complete_cell_only",
        "core_artifacts": list(CORE_ARTIFACTS),
        "candidate_ssl_selectable_hyperparameters": [],
        "candidate_gradient_policy": (
            "per_batch_scale_invariant_shared_encoder_angular_bisector"),
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
    expected_source = {str(path) for path in SOURCE_PATHS}
    if not isinstance(source, Mapping) or set(source) != expected_source:
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
            record = data.get(task)
            if not isinstance(record, Mapping):
                errors.append(f"protocol data record is invalid: {task}")
                continue
            for split, path in (("train", train_path), ("val", val_path)):
                if record.get(split) != str(path) or not path.is_file() \
                        or record.get(f"{split}_sha256") != file_sha256(path):
                    errors.append(f"data receipt differs: {task}/{split}")

    if isinstance(python, str) and isinstance(wandb_enabled, bool):
        commands = command_records(
            python, root, DEFAULT_STUDY, EPOCHS, wandb=wandb_enabled)
        if protocol.get("commands") != commands:
            errors.append("protocol commands differ token-for-token")
        if protocol.get("commands_sha256") != json_sha256(commands):
            errors.append("protocol command hash differs")
    return errors


def _validate_metric_metadata(
        metrics: Mapping[str, Any], task: str, design: str, seed: int) -> None:
    family, memory = design_parts(design)
    expected = {
        "regularizer": family,
        "memory_architecture": memory,
        "confirmation_evidence": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise AnalysisError(
                f"{task}/{design}/s{seed}: metric metadata {key} differs")
    for key in (PRIMARY, CLEAN, INTEGRATOR, VARIANCE, RANK, CONVERGENCE,
                VAL_PREDICTIVE, *GRADIENT_METRICS):
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
        try:
            key = (
                str(row.get("task")), str(row.get("design")),
                int(row.get("seed")))
        except (TypeError, ValueError) as exc:
            raise AnalysisError(f"{path} contains an invalid cell key") from exc
        if key in result:
            raise AnalysisError(f"duplicate runner receipt {key}")
        result[key] = row
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
    expected = set(cell_specs())
    if set(ledger) != expected:
        missing = expected - set(ledger)
        extra = set(ledger) - expected
        if missing:
            errors.append(f"runner ledger missing {len(missing)} cells")
        if extra:
            errors.append(f"runner ledger has {len(extra)} unexpected cells")

    wandb_expected = bool(protocol.get("wandb_enabled"))
    for task, design, seed in cell_specs():
        receipt = ledger.get((task, design, seed))
        if receipt is None:
            continue
        if receipt.get("status") != "complete":
            errors.append(
                f"{task}/{design}/s{seed}: runner status "
                f"{receipt.get('status')!r}")
            continue
        try:
            validation = validate_core_artifacts(
                root, task, design, seed, EPOCHS,
                wandb_expected=wandb_expected)
            metrics = validation["metrics"]
            _validate_metric_metadata(metrics, task, design, seed)
            if receipt.get("artifact_sha256") != validation["artifact_sha256"]:
                raise AnalysisError(
                    f"{task}/{design}/s{seed}: runner artifact hashes differ")
            command_index = (
                TASKS.index(task) * len(SEEDS) * len(DESIGNS)
                + SEEDS.index(seed) * len(DESIGNS)
                + DESIGNS.index(design))
            command = protocol["commands"][command_index]["argv"]
            if receipt.get("command_sha256") != json_sha256(command):
                raise AnalysisError(
                    f"{task}/{design}/s{seed}: runner command hash differs")
            family, memory = design_parts(design)
            rows.append({
                "task": task, "design": design, "seed": seed,
                "family": family, "memory": memory, "metrics": metrics,
                "directory": validation["directory"],
                "artifact_sha256": validation["artifact_sha256"],
                "wandb_state": validation["wandb_state"],
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
        *, metric: str, direction: str) -> dict[str, Any]:
    if direction not in ("lower", "higher"):
        raise ValueError(f"invalid contrast direction {direction!r}")
    candidate_values = _design_metric(rows, candidate, metric)
    reference_values = _design_metric(rows, reference, metric)
    keys = [(task, seed) for task in TASKS for seed in SEEDS]
    cand = np.asarray([candidate_values[key] for key in keys], dtype=np.float64)
    ref = np.asarray([reference_values[key] for key in keys], dtype=np.float64)
    sign = 1.0 if direction == "higher" else -1.0
    absolute = sign * (cand - ref)
    relative = absolute / np.maximum(np.abs(ref), 1e-12)
    ties = np.isclose(cand, ref, rtol=1e-12, atol=1e-12)
    wins = cand > ref if direction == "higher" else cand < ref
    losses = cand < ref if direction == "higher" else cand > ref
    by_task: dict[str, Any] = {}
    for task in TASKS:
        selected = np.asarray([name == task for name, _ in keys])
        by_task[task] = {
            "candidate_mean": float(cand[selected].mean()),
            "reference_mean": float(ref[selected].mean()),
            "paired_relative_improvement_mean": float(relative[selected].mean()),
            "wins": int(wins[selected].sum()),
            "ties": int(ties[selected].sum()),
            "losses": int(losses[selected].sum()),
        }
    return {
        "kind": "autovisreg_vs_vicreg",
        "candidate": candidate, "reference": reference,
        "metric": metric, "direction": direction, "pairs": len(keys),
        "candidate_mean": float(cand.mean()),
        "reference_mean": float(ref.mean()),
        "paired_absolute_improvement": _summary(absolute),
        "paired_relative_improvement": _summary(relative),
        "wins": int(wins.sum()), "ties": int(ties.sum()),
        "losses": int(losses.sum()), "by_task": by_task,
    }


def design_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    expected_per_design = len(TASKS) * len(SEEDS)
    for design in DESIGNS:
        selected = [row for row in rows if row["design"] == design]
        if len(selected) != expected_per_design:
            raise AnalysisError(f"{design}: incomplete design cells")
        metrics: dict[str, Any] = {}
        for key in SUMMARY_METRICS:
            values = [
                finite(row["metrics"].get(key), f"{design}:{key}")
                for row in selected if row["metrics"].get(key) is not None
            ]
            if values:
                metrics[key] = _summary(values)
        result[design] = {
            "family": selected[0]["family"],
            "memory": selected[0]["memory"],
            "cells": len(selected), "metrics": metrics,
        }
    return result


def diagnostic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_design: dict[str, Any] = {}
    for design in DESIGNS:
        selected = [row for row in rows if row["design"] == design]
        ranks = np.asarray([
            finite(row["metrics"].get(RANK), f"{design}:{RANK}")
            for row in selected], dtype=np.float64)
        variances = np.asarray([
            finite(row["metrics"].get(VARIANCE), f"{design}:{VARIANCE}")
            for row in selected], dtype=np.float64)
        convergence = np.asarray([
            finite(row["metrics"].get(CONVERGENCE), f"{design}:{CONVERGENCE}")
            for row in selected], dtype=np.float64)
        gradients = {
            key: _summary([
                finite(row["metrics"].get(key), f"{design}:{key}")
                for row in selected])
            for key in GRADIENT_METRICS
        }
        by_design[design] = {
            "rank": {
                **_summary(ranks), "threshold": RANK_THRESHOLD,
                "cells_passing": int((ranks >= RANK_THRESHOLD).sum()),
                "all_cells_pass": bool((ranks >= RANK_THRESHOLD).all()),
            },
            "variance": {
                **_summary(variances), "floor": VARIANCE_FLOOR,
                "cells_passing": int((variances >= VARIANCE_FLOOR).sum()),
                "all_cells_pass": bool((variances >= VARIANCE_FLOOR).all()),
            },
            "convergence": {
                **_summary(convergence),
                "abs_summary": _summary(np.abs(convergence)),
                "abs_threshold": CONVERGENCE_ABS_THRESHOLD,
                "cells_passing": int(
                    (np.abs(convergence) <= CONVERGENCE_ABS_THRESHOLD).sum()),
                "all_cells_pass": bool(
                    (np.abs(convergence) <= CONVERGENCE_ABS_THRESHOLD).all()),
            },
            "gradient_and_clip": gradients,
        }
    candidate_designs = [
        design for design in DESIGNS if design.startswith("autovisreg_")]
    candidate = [row for row in rows if row["design"] in candidate_designs]
    candidate_none = [
        row for row in candidate if row["design"] == "autovisreg_none"]

    def gate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ranks = np.asarray([
            finite(row["metrics"].get(RANK), f"candidate:{RANK}")
            for row in selected], dtype=np.float64)
        variances = np.asarray([
            finite(row["metrics"].get(VARIANCE), f"candidate:{VARIANCE}")
            for row in selected], dtype=np.float64)
        convergence = np.asarray([
            finite(row["metrics"].get(CONVERGENCE), f"candidate:{CONVERGENCE}")
            for row in selected], dtype=np.float64)
        rank_pass = bool((ranks >= RANK_THRESHOLD).all())
        variance_pass = bool((variances >= VARIANCE_FLOOR).all())
        convergence_pass = bool(
            (np.abs(convergence) <= CONVERGENCE_ABS_THRESHOLD).all())
        return {
            "cells": len(selected), "rank_pass": rank_pass,
            "variance_pass": variance_pass,
            "convergence_pass": convergence_pass,
            "all_pass": rank_pass and variance_pass and convergence_pass,
        }

    return {
        "rank_threshold": RANK_THRESHOLD,
        "variance_floor": VARIANCE_FLOOR,
        "convergence_abs_threshold": CONVERGENCE_ABS_THRESHOLD,
        "by_design": by_design,
        "candidate_host_only_gate": gate(candidate_none),
        "candidate_full_grid_gate": gate(candidate),
    }


def analyze_rows(
        rows: Sequence[Mapping[str, Any]], errors: Sequence[str]
        ) -> dict[str, Any]:
    complete = len(rows) == EXPECTED_CELLS and not errors
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "autovisreg_v17_excluded_adaptive_development",
        "status": "COMPLETE" if complete else "INCOMPLETE_OR_INVALID",
        "scientific_label": (
            "NOT_EVALUATED_INCOMPLETE" if not complete else "PENDING"),
        "official_confirmation_result": False,
        "adaptive_descriptive_only": True,
        "expected_cells": EXPECTED_CELLS,
        "completed_valid_cells": len(rows),
        "artifact_integrity_passed": complete,
        "artifact_integrity_errors": list(errors),
        "primary_metric": PRIMARY,
    }
    if not complete:
        return result

    diagnostics = diagnostic_summary(rows)
    contrasts: dict[str, Any] = {}
    directions = {
        PRIMARY: "lower", CLEAN: "lower", VAL_PREDICTIVE: "lower",
        VARIANCE: "higher", RANK: "higher",
    }
    for memory in MEMORY_VARIANTS:
        candidate = f"autovisreg_{memory}"
        reference = f"vicreg_{memory}"
        for metric, direction in directions.items():
            key = f"{candidate}_vs_{reference}:{metric}"
            contrasts[key] = paired_contrast(
                rows, candidate, reference, metric=metric,
                direction=direction)
    full_pass = diagnostics["candidate_full_grid_gate"]["all_pass"]
    host_pass = diagnostics["candidate_host_only_gate"]["all_pass"]
    result.update({
        "scientific_label": (
            "ADAPTIVE_COLLAPSE_REPAIR_FULL_GRID_PASS" if full_pass else
            "ADAPTIVE_HOST_REPAIR_ONLY" if host_pass else
            "ADAPTIVE_COLLAPSE_REPAIR_FAILED"),
        "design_summary": design_summary(rows),
        "autovisreg_vs_vicreg": contrasts,
        "representation_convergence_and_gradient_diagnostics": diagnostics,
        "wandb_receipts": {
            "finished": sum(row["wandb_state"] == "finished" for row in rows),
            "disabled": sum(
                row["wandb_state"] == "not_requested" for row in rows),
        },
    })
    return result


def cell_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        metrics = row["metrics"]
        result.append({
            "task": row["task"], "seed": row["seed"],
            "design": row["design"], "family": row["family"],
            "memory": row["memory"],
            **{key: metrics.get(key) for key in SUMMARY_METRICS},
            "wandb_state": row["wandb_state"],
            "directory": row["directory"],
        })
    return result


def contrast_csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    contrasts = analysis.get("autovisreg_vs_vicreg", {})
    if not isinstance(contrasts, Mapping):
        return []
    result = []
    for contrast in contrasts.values():
        relative = contrast["paired_relative_improvement"]
        absolute = contrast["paired_absolute_improvement"]
        result.append({
            "kind": contrast["kind"], "candidate": contrast["candidate"],
            "reference": contrast["reference"], "metric": contrast["metric"],
            "direction": contrast["direction"], "pairs": contrast["pairs"],
            "candidate_mean": contrast["candidate_mean"],
            "reference_mean": contrast["reference_mean"],
            "paired_relative_improvement_mean": relative["mean"],
            "paired_relative_improvement_std": relative["std"],
            "paired_relative_improvement_ci95_low": relative["ci95_low"],
            "paired_relative_improvement_ci95_high": relative["ci95_high"],
            "paired_absolute_improvement_mean": absolute["mean"],
            "wins": contrast["wins"], "ties": contrast["ties"],
            "losses": contrast["losses"],
        })
    return result


CELL_CSV_FIELDS = (
    "task", "seed", "design", "family", "memory", *SUMMARY_METRICS,
    "wandb_state", "directory",
)
CONTRAST_CSV_FIELDS = (
    "kind", "candidate", "reference", "metric", "direction", "pairs",
    "candidate_mean", "reference_mean",
    "paired_relative_improvement_mean", "paired_relative_improvement_std",
    "paired_relative_improvement_ci95_low",
    "paired_relative_improvement_ci95_high",
    "paired_absolute_improvement_mean", "wins", "ties", "losses",
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
        analysis["development_protocol_sha256"] = file_sha256(
            root / PROTOCOL_NAME)
        analysis["commands_sha256"] = protocol.get("commands_sha256")
        analysis["wandb_enabled"] = protocol.get("wandb_enabled")
    rendered = json.dumps(
        analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.write:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(root / ANALYSIS_NAME, rendered)
        atomic_write_text(
            root / CELLS_CSV_NAME,
            csv_text(cell_csv_rows(rows), CELL_CSV_FIELDS))
        atomic_write_text(
            root / CONTRASTS_CSV_NAME,
            csv_text(contrast_csv_rows(analysis), CONTRAST_CSV_FIELDS))
    raise SystemExit(0 if analysis["artifact_integrity_passed"] else 2)


if __name__ == "__main__":
    main()
