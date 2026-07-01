#!/usr/bin/env python3
"""Write-once analysis for the frozen LeWM+V8 V18 confirmation grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_lewm_v8_v18 as runner

PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
SECONDARY = "val_predictive_loss"
DEEP = "deep_prior_state_nmse"
VARIANCE = "encoder_mean_channel_variance"
RANK = "encoder_covariance_effective_rank"
CONVERGENCE = "predictive_loss_convergence_relative_change"
INTEGRATOR = "initial_encoder_integrator_probe_nmse"
CONDITIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")

NONE = "vicreg_none"
GRU = "vicreg_gru"
SSM = "vicreg_ssm"
CANDIDATE = "vicreg_hacssmv8"
DYNAMIC = "vicreg_hacssmv8_dynamic"
STATIC = "vicreg_hacssmv8_static"
NO_ACTION = "vicreg_hacssmv8_noaction"
SINGLE = "vicreg_hacssmv8_single"
FROZEN_DESIGNS = (
    NONE,
    GRU,
    SSM,
    CANDIDATE,
    STATIC,
    DYNAMIC,
    NO_ACTION,
    SINGLE,
)
DIRECT_REFERENCES = tuple(design for design in FROZEN_DESIGNS if design != CANDIDATE)
RECURRENT_REFERENCES = (GRU, SSM)
ENDPOINT_REFERENCES = (DYNAMIC, STATIC)

FROZEN_TASKS = (
    "acrobot.swingup",
    "manipulator.bring_ball",
    "quadruped.run",
    "stacker.stack_4",
    "swimmer.swimmer15",
)
FROZEN_SEEDS = (18_001, 18_002, 18_003, 18_004, 18_005)
FROZEN_TASK_COUNT = len(FROZEN_TASKS)
FROZEN_SEED_COUNT = len(FROZEN_SEEDS)
FROZEN_EPOCHS = 100
FROZEN_CELL_COUNT = len(FROZEN_TASKS) * len(FROZEN_DESIGNS) * len(FROZEN_SEEDS)
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 18_018
ANALYSIS_NAME = "confirmation_analysis.json"
CELLS_NAME = "confirmation_cells.csv"
CONTRASTS_NAME = "confirmation_contrasts.csv"


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{label} is not a finite scalar")
    return float(value)


def _deep_metric(metrics: Mapping[str, Any]) -> float:
    values = [
        _finite(
            metrics[f"{condition}_prior_state_nmse_deep"],
            f"{condition} deep prior",
        )
        for condition in CONDITIONS
    ]
    return float(np.mean(values))


def _metric(row: Mapping[str, Any], metric: str, label: str) -> float:
    if metric == DEEP:
        return _finite(row[DEEP], label)
    return _finite(row["metrics"][metric], label)


def _contract_errors() -> list[str]:
    errors: list[str] = []
    tasks = tuple(runner.TASKS)
    designs = tuple(runner.DESIGNS)
    seeds = tuple(runner.SEEDS)
    if tasks != FROZEN_TASKS:
        errors.append(
            f"frozen V18 task tuple mismatch: expected={FROZEN_TASKS!r}, "
            f"actual={tasks!r}")
    if seeds != FROZEN_SEEDS:
        errors.append(
            f"frozen V18 seed tuple mismatch: expected={FROZEN_SEEDS!r}, "
            f"actual={seeds!r}")
    if designs != FROZEN_DESIGNS:
        errors.append(
            "frozen V18 design tuple mismatch: "
            f"expected={FROZEN_DESIGNS!r}, actual={designs!r}")
    if int(runner.EPOCHS) != FROZEN_EPOCHS:
        errors.append(
            f"frozen V18 requires {FROZEN_EPOCHS} epochs, got {runner.EPOCHS}")
    return errors


def load_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runner._install_contract()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        protocol, guard = runner.validate_frozen_protocol(root)
        runner._ACTIVE_INTEGRITY_GUARD = guard
        summary = runner.base.load_json(root / runner.base.SUMMARY_NAME)
        if not isinstance(summary, Mapping) or summary.get("status") != "COMPLETE" \
                or summary.get("expected_cells") != FROZEN_CELL_COUNT \
                or summary.get("completed_cells") != FROZEN_CELL_COUNT \
                or summary.get("failed_or_invalid_cells") != 0:
            raise runner.base.ArtifactError(
                "V18 confirmation summary is absent, incomplete, or inconsistent")
        ledger_rows = runner.base.load_json(root / runner.base.RUNS_NAME)
        if not isinstance(ledger_rows, list) or len(ledger_rows) != FROZEN_CELL_COUNT:
            raise runner.base.ArtifactError(
                "V18 confirmation run ledger must contain exactly 200 rows")
        ledger: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        for row in ledger_rows:
            if not isinstance(row, Mapping):
                raise runner.base.ArtifactError("V18 run ledger contains a non-object")
            key = (str(row.get("task")), int(row.get("seed")), str(row.get("design")))
            if key in ledger or row.get("status") != "complete":
                raise runner.base.ArtifactError(
                    f"V18 run ledger has duplicate/noncomplete cell {key}")
            ledger[key] = row
        commands = {
            (str(row["task"]), int(row["seed"]), str(row["design"])): row["argv"]
            for row in protocol["commands"]}
        if set(commands) != set(ledger):
            raise runner.base.ArtifactError("V18 command and run-ledger cell sets differ")
        for key, row in ledger.items():
            if row.get("command_sha256") != runner.base.json_sha256(commands[key]):
                raise runner.base.ArtifactError(
                    f"V18 run-ledger command hash differs for {key}")

        for task, design, seed in runner.base.cell_specs():
            try:
                validated = runner.base.validate_core_artifacts(
                    root, task, design, seed, runner.EPOCHS, wandb_expected=True)
                metrics = dict(validated["metrics"])
                rows.append({
                    "task": task,
                    "design": design,
                    "seed": seed,
                    "metrics": metrics,
                    DEEP: _deep_metric(metrics),
                    "directory": validated["directory"],
                    "artifact_sha256": validated["artifact_sha256"],
                    "wandb_state": validated["wandb_state"],
                })
                ledger_row = ledger[(task, seed, design)]
                if ledger_row.get("artifact_sha256") != validated["artifact_sha256"]:
                    raise runner.base.ArtifactError(
                        "run-ledger artifact hashes differ from validated artifacts")
                if ledger_row.get("wandb_state") != "finished":
                    raise runner.base.ArtifactError(
                        "run-ledger W&B state is not finished")
            except Exception as exc:  # retain every fail-closed diagnosis
                errors.append(
                    f"{task}/{design}/s{seed}: {type(exc).__name__}: {exc}")
        guard.assert_all()
    except Exception as exc:
        errors.append(f"protocol/ledger preflight: {type(exc).__name__}: {exc}")
    finally:
        runner._ACTIVE_INTEGRITY_GUARD = None
        runner._restore_contract()
    return rows, errors


def _index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["seed"]), str(row["design"]))
        if key in result:
            raise ValueError(f"duplicate V18 cell {key}")
        result[key] = row
    return result


def _crossed_bootstrap(
    values: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int | None = None,
) -> dict[str, float | int]:
    expected_shape = (len(runner.TASKS), len(runner.SEEDS))
    if values.shape != expected_shape:
        raise ValueError(f"unexpected crossed matrix {values.shape}; expected {expected_shape}")
    if not np.isfinite(values).all():
        raise ValueError("crossed bootstrap matrix contains nonfinite values")
    actual_draws = BOOTSTRAP_DRAWS if draws is None else int(draws)
    if actual_draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = actual_draws
    while remaining:
        count = min(10_000, remaining)
        task_ids = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        seed_ids = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        sampled = values[task_ids[:, :, None], seed_ids[:, None, :]]
        chunks.append(sampled.mean(axis=(1, 2)))
        remaining -= count
    estimates = np.concatenate(chunks)
    return {
        "ci90_low": float(np.quantile(estimates, 0.05, method="linear")),
        "ci90_high": float(np.quantile(estimates, 0.95, method="linear")),
        "ci95_low": float(np.quantile(estimates, 0.025, method="linear")),
        "ci95_high": float(np.quantile(estimates, 0.975, method="linear")),
        "draws": actual_draws,
        "seed": int(seed),
    }


ReferenceSelector = Callable[[str, int], tuple[float, str]]


def _selected_contrast(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    candidate: str,
    reference_label: str,
    metric: str,
    select_reference: ReferenceSelector,
) -> dict[str, Any]:
    matrix = np.empty((len(runner.TASKS), len(runner.SEEDS)), dtype=np.float64)
    candidate_values: dict[str, list[float]] = {task: [] for task in runner.TASKS}
    reference_values: dict[str, list[float]] = {task: [] for task in runner.TASKS}
    selected_counts: dict[str, int] = {}
    for task_index, task in enumerate(runner.TASKS):
        for seed_index, seed in enumerate(runner.SEEDS):
            candidate_row = index[(task, seed, candidate)]
            cand = _metric(candidate_row, metric, f"{task}/{candidate}/{metric}")
            ref, selected = select_reference(task, seed)
            ref = _finite(ref, f"{task}/{reference_label}/{metric}")
            matrix[task_index, seed_index] = (ref - cand) / max(abs(ref), 1e-12)
            candidate_values[task].append(cand)
            reference_values[task].append(ref)
            selected_counts[selected] = selected_counts.get(selected, 0) + 1

    task_effects = {
        task: float(
            (np.mean(reference_values[task]) - np.mean(candidate_values[task]))
            / max(abs(np.mean(reference_values[task])), 1e-12)
        )
        for task in runner.TASKS
    }
    flattened_candidate = [value for values in candidate_values.values() for value in values]
    flattened_reference = [value for values in reference_values.values() for value in values]
    return {
        "candidate": candidate,
        "reference": reference_label,
        "metric": metric,
        "direction": "lower_is_better",
        "mean_paired_relative_reduction": float(matrix.mean()),
        "paired_wins": int((matrix > 0).sum()),
        "paired_ties": int((matrix == 0).sum()),
        "pairs": int(matrix.size),
        "task_mean_wins": int(sum(value > 0 for value in task_effects.values())),
        "task_effects": task_effects,
        "candidate_mean": float(np.mean(flattened_candidate)),
        "reference_mean": float(np.mean(flattened_reference)),
        "selected_reference_counts": selected_counts,
        "bootstrap": _crossed_bootstrap(matrix),
        "cell_effects": matrix.tolist(),
    }


def paired_contrast(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    candidate: str,
    reference: str,
    metric: str,
) -> dict[str, Any]:
    def select(task: str, seed: int) -> tuple[float, str]:
        row = index[(task, seed, reference)]
        return _metric(row, metric, f"{task}/{reference}/{metric}"), reference

    return _selected_contrast(
        index,
        candidate=candidate,
        reference_label=reference,
        metric=metric,
        select_reference=select,
    )


def envelope_contrast(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    candidate: str,
    references: Sequence[str],
    metric: str,
    *,
    label: str,
) -> dict[str, Any]:
    reference_tuple = tuple(references)
    if not reference_tuple:
        raise ValueError("an envelope requires at least one reference")

    def select(task: str, seed: int) -> tuple[float, str]:
        values = [
            (
                _metric(
                    index[(task, seed, reference)],
                    metric,
                    f"{task}/{reference}/{metric}",
                ),
                reference,
            )
            for reference in reference_tuple
        ]
        return min(values, key=lambda item: (item[0], item[1]))

    result = _selected_contrast(
        index,
        candidate=candidate,
        reference_label=label,
        metric=metric,
        select_reference=select,
    )
    result["envelope_members"] = list(reference_tuple)
    result["envelope_policy"] = "per_task_seed_lower_error"
    return result


def selected_identity_contrast(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    candidate: str,
    references: Sequence[str],
    metric: str,
    *,
    selection_metric: str,
    label: str,
) -> dict[str, Any]:
    """Evaluate a metric using identities selected once on another metric."""
    reference_tuple = tuple(references)
    if not reference_tuple:
        raise ValueError("selected-identity contrast requires a reference")

    def select(task: str, seed: int) -> tuple[float, str]:
        selected = min(
            reference_tuple,
            key=lambda reference: (
                _metric(
                    index[(task, seed, reference)], selection_metric,
                    f"{task}/{reference}/{selection_metric}"),
                reference,
            ),
        )
        return (
            _metric(index[(task, seed, selected)], metric,
                    f"{task}/{selected}/{metric}"),
            selected,
        )

    result = _selected_contrast(
        index,
        candidate=candidate,
        reference_label=label,
        metric=metric,
        select_reference=select,
    )
    result["envelope_members"] = list(reference_tuple)
    result["envelope_policy"] = "per_task_seed_identity_selected_once"
    result["selection_metric"] = selection_metric
    return result


def integrator_contrast(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    def select(task: str, seed: int) -> tuple[float, str]:
        metrics = index[(task, seed, CANDIDATE)]["metrics"]
        return (
            _finite(metrics[INTEGRATOR], f"{task}/{CANDIDATE}/{INTEGRATOR}"),
            "checkpoint_matched_initial_frame_action_integrator",
        )

    return _selected_contrast(
        index,
        candidate=CANDIDATE,
        reference_label="checkpoint_matched_initial_frame_action_integrator",
        metric=PRIMARY,
        select_reference=select,
    )


def _superiority_receipt(
    contrast: Mapping[str, Any],
    *,
    magnitude: float,
    paired_wins: int,
    task_wins: int,
    require_positive_ci95: bool,
) -> dict[str, Any]:
    observed = {
        "mean_paired_relative_reduction": contrast["mean_paired_relative_reduction"],
        "paired_wins": contrast["paired_wins"],
        "task_mean_wins": contrast["task_mean_wins"],
        "ci95_low": contrast["bootstrap"]["ci95_low"],
    }
    thresholds = {
        "minimum_mean_paired_relative_reduction": magnitude,
        "minimum_paired_wins": paired_wins,
        "minimum_task_mean_wins": task_wins,
        "require_ci95_low_strictly_positive": require_positive_ci95,
    }
    passed = (
        observed["mean_paired_relative_reduction"] >= magnitude
        and observed["paired_wins"] >= paired_wins
        and observed["task_mean_wins"] >= task_wins
        and (
            not require_positive_ci95
            or observed["ci95_low"] > 0.0
        )
    )
    return {"passed": bool(passed), "thresholds": thresholds, "observed": observed}


def _invalid_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
    contract_errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scope": "lewm_v8_v18_unopened_task_confirmation",
        "frozen_grid": {
            "tasks": FROZEN_TASK_COUNT,
            "designs": len(FROZEN_DESIGNS),
            "seeds": FROZEN_SEED_COUNT,
            "epochs": FROZEN_EPOCHS,
            "cells": FROZEN_CELL_COUNT,
            "task_ids": list(FROZEN_TASKS),
            "seed_ids": list(FROZEN_SEEDS),
        },
        "expected_cells": FROZEN_CELL_COUNT,
        "completed_valid_cells": len(rows),
        "artifact_integrity_passed": False,
        "artifact_integrity_errors": list(errors),
        "protocol_contract_errors": list(contract_errors),
        "official_confirmation_result": False,
        "status": "INCOMPLETE_OR_INVALID",
        "scientific_label": "INCOMPLETE_OR_INVALID",
    }


def analyze(rows: Sequence[Mapping[str, Any]], errors: Sequence[str]) -> dict[str, Any]:
    contract_errors = _contract_errors()
    row_errors = list(errors)
    try:
        index = _index(rows)
    except (KeyError, TypeError, ValueError) as exc:
        index = {}
        row_errors.append(f"row index error: {type(exc).__name__}: {exc}")

    expected_keys = {
        (task, seed, design)
        for task in runner.TASKS
        for seed in runner.SEEDS
        for design in FROZEN_DESIGNS
    }
    actual_keys = set(index)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing:
            row_errors.append(f"missing V18 cells ({len(missing)}): {missing[:8]!r}")
        if extra:
            row_errors.append(f"unexpected V18 cells ({len(extra)}): {extra[:8]!r}")

    complete = (
        len(rows) == FROZEN_CELL_COUNT
        and not row_errors
        and not contract_errors
        and len(index) == FROZEN_CELL_COUNT
    )
    if not complete:
        return _invalid_report(
            rows=rows, errors=row_errors, contract_errors=contract_errors)

    try:
        metrics_to_report = (PRIMARY, CLEAN, SECONDARY, DEEP)
        contrasts = {
            f"{CANDIDATE}_vs_{reference}:{metric}": paired_contrast(
                index, CANDIDATE, reference, metric)
            for reference in DIRECT_REFERENCES
            for metric in metrics_to_report
        }

        recurrent_primary = selected_identity_contrast(
            index,
            CANDIDATE,
            RECURRENT_REFERENCES,
            PRIMARY,
            selection_metric=PRIMARY,
            label="per_cell_better_of_gru_ssm",
        )
        recurrent_deep = selected_identity_contrast(
            index,
            CANDIDATE,
            RECURRENT_REFERENCES,
            DEEP,
            selection_metric=PRIMARY,
            label="per_cell_better_of_gru_ssm",
        )
        recurrent_clean = selected_identity_contrast(
            index,
            CANDIDATE,
            RECURRENT_REFERENCES,
            CLEAN,
            selection_metric=PRIMARY,
            label="per_cell_better_of_gru_ssm",
        )
        endpoint_primary = envelope_contrast(
            index,
            CANDIDATE,
            ENDPOINT_REFERENCES,
            PRIMARY,
            label="per_cell_better_of_static_dynamic",
        )
        integrator = integrator_contrast(index)

        contrasts.update({
            f"{CANDIDATE}_vs_recurrent_envelope:{PRIMARY}": recurrent_primary,
            f"{CANDIDATE}_vs_recurrent_envelope:{DEEP}": recurrent_deep,
            f"{CANDIDATE}_vs_recurrent_envelope:{CLEAN}": recurrent_clean,
            f"{CANDIDATE}_vs_endpoint_envelope:{PRIMARY}": endpoint_primary,
            f"{CANDIDATE}_vs_checkpoint_integrator:{PRIMARY}": integrator,
        })

        variance_values = [
            _finite(
                row["metrics"][VARIANCE],
                f"{row['task']}/{row['design']}/{VARIANCE}",
            )
            for row in rows
        ]
        rank_values = [
            _finite(
                row["metrics"][RANK],
                f"{row['task']}/{row['design']}/{RANK}",
            )
            for row in rows
        ]
        convergence_values = [
            abs(_finite(
                row["metrics"][CONVERGENCE],
                f"{row['task']}/{row['design']}/{CONVERGENCE}",
            ))
            for row in rows
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return _invalid_report(
            rows=rows,
            errors=[f"scientific metric validation: {type(exc).__name__}: {exc}"],
            contract_errors=contract_errors,
        )

    recurrent_receipt = _superiority_receipt(
        recurrent_primary,
        magnitude=0.03,
        paired_wins=18,
        task_wins=4,
        require_positive_ci95=True,
    )
    none_receipt = _superiority_receipt(
        contrasts[f"{CANDIDATE}_vs_{NONE}:{PRIMARY}"],
        magnitude=0.05,
        paired_wins=20,
        task_wins=4,
        require_positive_ci95=False,
    )
    integrator_receipt = _superiority_receipt(
        integrator,
        magnitude=0.03,
        paired_wins=18,
        task_wins=4,
        require_positive_ci95=False,
    )
    action_receipt = _superiority_receipt(
        contrasts[f"{CANDIDATE}_vs_{NO_ACTION}:{PRIMARY}"],
        magnitude=0.05,
        paired_wins=18,
        task_wins=4,
        require_positive_ci95=True,
    )
    single_receipt = _superiority_receipt(
        contrasts[f"{CANDIDATE}_vs_{SINGLE}:{PRIMARY}"],
        magnitude=0.03,
        paired_wins=18,
        task_wins=4,
        require_positive_ci95=True,
    )
    deep_receipt = {
        "passed": bool(
            recurrent_deep["bootstrap"]["ci95_low"] > 0.0
            and recurrent_deep["task_mean_wins"] >= 3
        ),
        "thresholds": {
            "require_ci95_low_strictly_positive": True,
            "minimum_task_mean_wins": 3,
        },
        "observed": {
            "mean_paired_relative_reduction": recurrent_deep[
                "mean_paired_relative_reduction"],
            "task_mean_wins": recurrent_deep["task_mean_wins"],
            "ci95_low": recurrent_deep["bootstrap"]["ci95_low"],
        },
    }
    endpoint_receipt = {
        "passed": bool(
            endpoint_primary["mean_paired_relative_reduction"] >= -0.01
            and endpoint_primary["bootstrap"]["ci95_low"] >= -0.01
        ),
        "thresholds": {
            "minimum_mean_paired_relative_reduction": -0.01,
            "minimum_ci95_low": -0.01,
            "interpretation": "learned V8 noninferior within one percent",
        },
        "observed": {
            "mean_paired_relative_reduction": endpoint_primary[
                "mean_paired_relative_reduction"],
            "ci95_low": endpoint_primary["bootstrap"]["ci95_low"],
        },
    }
    clean_degradation = -float(recurrent_clean["mean_paired_relative_reduction"])
    clean_receipt = {
        "passed": bool(clean_degradation <= 0.03),
        "thresholds": {"maximum_mean_paired_relative_degradation": 0.03},
        "observed": {
            "mean_paired_relative_degradation": clean_degradation,
            "mean_paired_relative_reduction": recurrent_clean[
                "mean_paired_relative_reduction"],
            "ci95_low_reduction": recurrent_clean["bootstrap"]["ci95_low"],
            "ci95_high_reduction": recurrent_clean["bootstrap"]["ci95_high"],
        },
    }
    representation_receipt = {
        "passed": bool(
            min(variance_values) >= 1e-4
            and min(rank_values) >= 16.0
        ),
        "thresholds": {
            "minimum_every_cell_channel_variance": 1e-4,
            "minimum_every_cell_effective_rank": 16.0,
        },
        "observed": {
            "cells": len(rows),
            "minimum_channel_variance": min(variance_values),
            "minimum_effective_rank": min(rank_values),
            "variance_passing_cells": sum(value >= 1e-4 for value in variance_values),
            "rank_passing_cells": sum(value >= 16.0 for value in rank_values),
        },
    }
    convergence_receipt = {
        "passed": bool(max(convergence_values) <= 0.05),
        "thresholds": {"maximum_every_cell_absolute_relative_change": 0.05},
        "observed": {
            "cells": len(rows),
            "maximum_absolute_relative_change": max(convergence_values),
            "passing_cells": sum(value <= 0.05 for value in convergence_values),
        },
    }

    gate_receipts = {
        "v8_vs_per_cell_better_gru_ssm": recurrent_receipt,
        "v8_vs_none": none_receipt,
        "v8_vs_checkpoint_integrator": integrator_receipt,
        "action_causality": action_receipt,
        "joint_state_use": single_receipt,
        "deep_vs_per_cell_better_gru_ssm": deep_receipt,
        "learned_v8_vs_static_dynamic_envelope_noninferiority": endpoint_receipt,
        "clean_prior_guard_vs_per_cell_better_gru_ssm": clean_receipt,
        "healthy_representation": representation_receipt,
        "convergence": convergence_receipt,
    }
    gates = {name: bool(receipt["passed"]) for name, receipt in gate_receipts.items()}
    gates["integrity"] = True
    passed = all(gates.values())

    return {
        "schema_version": 2,
        "scope": "lewm_v8_v18_unopened_task_confirmation",
        "frozen_grid": {
            "tasks": FROZEN_TASK_COUNT,
            "designs": len(FROZEN_DESIGNS),
            "seeds": FROZEN_SEED_COUNT,
            "epochs": FROZEN_EPOCHS,
            "cells": FROZEN_CELL_COUNT,
            "task_ids": list(FROZEN_TASKS),
            "seed_ids": list(FROZEN_SEEDS),
            "design_ids": list(FROZEN_DESIGNS),
        },
        "expected_cells": FROZEN_CELL_COUNT,
        "completed_valid_cells": len(rows),
        "artifact_integrity_passed": True,
        "artifact_integrity_errors": [],
        "protocol_contract_errors": [],
        "status": "COMPLETE",
        "scientific_label": (
            "STABILIZED_LEWM_V8_CONFIRMATION_PASS"
            if passed
            else "CONFIRMATION_FAILED"
        ),
        "official_confirmation_result": passed,
        "primary_metric": PRIMARY,
        "gates": gates,
        "gate_receipts": gate_receipts,
        "representation": representation_receipt,
        "convergence": convergence_receipt,
        "contrasts": contrasts,
        "integrator_guard": integrator,
        "claim_boundary": (
            "V18 licenses only a persistent causal-memory claim for the stabilized "
            "VICReg LeWM host on the frozen partial-observation cohort. It licenses no "
            "executed-return, planning, original-SIGReg, learned-timescale, semantic-"
            "hierarchy, or calibrated-uncertainty claim."
        ),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csvs(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    cells = root / CELLS_NAME
    with cells.open("x", newline="", encoding="utf-8") as stream:
        fields = [
            "task",
            "seed",
            "design",
            PRIMARY,
            CLEAN,
            SECONDARY,
            DEEP,
            VARIANCE,
            RANK,
            CONVERGENCE,
            INTEGRATOR,
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "task": row["task"],
                "seed": row["seed"],
                "design": row["design"],
                PRIMARY: metrics[PRIMARY],
                CLEAN: metrics[CLEAN],
                SECONDARY: metrics[SECONDARY],
                DEEP: row[DEEP],
                VARIANCE: metrics[VARIANCE],
                RANK: metrics[RANK],
                CONVERGENCE: metrics[CONVERGENCE],
                INTEGRATOR: metrics[INTEGRATOR],
            })

    contrasts = root / CONTRASTS_NAME
    with contrasts.open("x", newline="", encoding="utf-8") as stream:
        fields = [
            "contrast",
            "metric",
            "mean_paired_relative_reduction",
            "paired_wins",
            "pairs",
            "task_mean_wins",
            "ci95_low",
            "ci95_high",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, value in report.get("contrasts", {}).items():
            writer.writerow({
                "contrast": name,
                "metric": value["metric"],
                "mean_paired_relative_reduction": value[
                    "mean_paired_relative_reduction"],
                "paired_wins": value["paired_wins"],
                "pairs": value["pairs"],
                "task_mean_wins": value["task_mean_wins"],
                "ci95_low": value["bootstrap"]["ci95_low"],
                "ci95_high": value["bootstrap"]["ci95_high"],
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=runner.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root if args.root.is_absolute() else ROOT / args.root
    root = root.resolve()
    rows, errors = load_rows(root)
    report = analyze(rows, errors)
    if report["status"] == "COMPLETE":
        report["input_protocol_sha256"] = runner.base.file_sha256(
            root / "confirmation_protocol.json")
        report["input_artifact_manifest_sha256"] = (
            runner.artifact_manifest_sha256(rows))
    if args.write:
        output_paths = (
            root / ANALYSIS_NAME,
            root / CELLS_NAME,
            root / CONTRASTS_NAME,
        )
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"write-once V18 outputs already exist: {existing}")
        if report["status"] == "COMPLETE":
            # CSVs are written first and the JSON decision is the completion
            # marker. A crash can never leave a lone decision file that a
            # resume mistakes for a complete analysis bundle.
            write_csvs(root, rows, report)
            report["cells_csv_sha256"] = runner.base.file_sha256(
                root / CELLS_NAME)
            report["contrasts_csv_sha256"] = runner.base.file_sha256(
                root / CONTRASTS_NAME)
            _write_json_exclusive(root / ANALYSIS_NAME, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
