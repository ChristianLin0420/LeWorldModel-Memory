#!/usr/bin/env python3
"""Fail-closed analysis for the frozen ORBIT-v10 end-to-end study.

The headline metric is ``heldout_state_nmse``: a per-checkpoint state-probe
metric in normalized simulator coordinates, averaged equally over the four
prospectively held-out corruptions.  Private latent MSE is deliberately not a
cross-model estimand because every design learns its own encoder coordinates.

Seeds 0--2 form an immutable pilot.  Seeds 3--4 are mandatory precision runs;
they can never reopen a failed pilot decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_hacssm_v5 as shared


ENVIRONMENTS = (
    "dmc:walker.walk",
    "dmc:hopper.hop",
    "dmc:cartpole.swingup",
    "dmc:pendulum.swingup",
    "dmc:fish.swim",
)
OCC_TO_CLEAN = {environment: environment for environment in ENVIRONMENTS}
DESIGNS = (
    "none",
    "gru",
    "ssm",
    "hacssmv8",
    "orbitv10",
    "orbitv10_noaction",
    "orbitv10_additive",
    "orbitv10_scaled",
    "orbitv10_static",
)
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 100
WINDOW = 10
PRIMARY = "heldout_state_nmse"
CLEAN_METRIC = "clean_state_nmse"
CANDIDATE = "orbitv10"
HEADLINE_REFERENCES = ("ssm", "hacssmv8")
GEOMETRY_CONTROLS = ("orbitv10_additive", "orbitv10_scaled")
NOACTION_CONTROL = "orbitv10_noaction"
STATIC_CONTROL = "orbitv10_static"
HELDOUT_CONDITIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")

BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 10_010
BOOTSTRAP_CONTRACT = {
    "schema_version": 1,
    "algorithm": "crossed_environment_seed_percentile_bootstrap",
    "draws": BOOTSTRAP_DRAWS,
    "seed": BOOTSTRAP_SEED,
    "rng": "numpy.random.Generator(numpy.random.PCG64)",
    "resampling": (
        "independently sample E environment indices and S optimizer-seed indices "
        "with replacement; evaluate their E-by-S Cartesian product; equal-weight mean"
    ),
    "estimand": "mean paired relative NMSE reduction (reference-candidate)/reference",
    "quantiles": {"method": "linear", "reported": [0.05, 0.95]},
}
BOOTSTRAP_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        BOOTSTRAP_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()

ORBIT_RECEIPTS = (
    "orbit_orthogonality_error_max",
    "orbit_streaming_max_abs",
)
ENCODER_RECEIPTS = (
    "encoder_mean_channel_variance",
    "encoder_covariance_effective_rank",
    "encoder_singleton_max_abs",
    "encoder_prefix_max_abs",
)
ROW_METRICS = (
    "val_pred_loss",
    PRIMARY,
    CLEAN_METRIC,
    *(f"{condition}_state_nmse" for condition in HELDOUT_CONDITIONS),
    *(
        f"{condition}_state_nmse_{phase}"
        for condition in HELDOUT_CONDITIONS
        for phase in ("deep", "first_post", "post")
    ),
    "probe_ceiling_state_nmse",
    "probe_ceiling_r2",
    *(f"{condition}_predicted_state_r2" for condition in HELDOUT_CONDITIONS),
    "convergence_relative_change",
    *ORBIT_RECEIPTS,
    *ENCODER_RECEIPTS,
    "clean_mse_first_post",
    "clean_mse_deep_blackout",
    "clean_mse_all",
)


def _finite(value: Any, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{context} is not finite: {value!r}")
    return float(value)


def load_cells(root: Path, seeds: Sequence[int]):
    """Load exactly one complete checkpoint for every locked grid cell."""
    rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for design in DESIGNS:
            for seed in seeds:
                run = f"lewm-{environment}-{design}-s{seed}"
                run_dir = root / run
                try:
                    metrics = shared.read_json(run_dir / "metrics.json")
                    checkpoint = torch.load(
                        run_dir / "model.pt", map_location="cpu", weights_only=False
                    )
                except Exception as exc:
                    raise ValueError(f"cannot load {run_dir}: {exc}") from exc
                if not isinstance(metrics, dict) or not isinstance(checkpoint, dict):
                    raise ValueError(f"{run_dir}: malformed metric/checkpoint objects")
                if metrics != checkpoint.get("final_metrics"):
                    raise ValueError(f"{run_dir}: metrics/checkpoint mismatch")
                history = checkpoint.get("history")
                if not isinstance(history, list) or len(history) != EPOCHS:
                    raise ValueError(f"{run_dir}: expected {EPOCHS} history records")

                row: dict[str, Any] = {
                    "run": run,
                    "env": environment,
                    "design": design,
                    "seed": int(seed),
                    "trainable_parameters": int(metrics["trainable_parameters"]),
                }
                for key in ROW_METRICS:
                    row[key] = _finite(metrics[key], f"{run}.{key}") if key in metrics else ""
                # The V10 trainer records both total and prediction loss in history; private
                # latent losses are diagnostics only and never enter a cross-model gate.
                row["val_pred_loss"] = _finite(
                    history[-1]["val"]["pred_loss"], f"{run}.final_val_pred_loss"
                )
                for key in (PRIMARY, CLEAN_METRIC, "val_pred_loss", *ENCODER_RECEIPTS):
                    if row[key] == "":
                        raise ValueError(f"{run}: required metric {key} is absent")
                    if key in (PRIMARY, CLEAN_METRIC) and float(row[key]) <= 0.0:
                        raise ValueError(f"{run}: {key} must be positive")
                condition_values = []
                for condition in HELDOUT_CONDITIONS:
                    key = f"{condition}_state_nmse"
                    if row[key] == "" or float(row[key]) <= 0.0:
                        raise ValueError(f"{run}: required held-out metric {key} is invalid")
                    condition_values.append(float(row[key]))
                if not math.isclose(
                    float(row[PRIMARY]), mean(condition_values), rel_tol=1e-6, abs_tol=1e-8
                ):
                    raise ValueError(
                        f"{run}: {PRIMARY} is not the equal four-condition mean"
                    )
                if design.startswith("orbitv10"):
                    for key in ORBIT_RECEIPTS:
                        if row[key] == "" or float(row[key]) < 0.0:
                            raise ValueError(f"{run}: invalid ORBIT receipt {key}")

                previous = mean(
                    _finite(item["val"]["pred_loss"], f"{run}.history.previous")
                    for item in history[-2 * WINDOW:-WINDOW]
                )
                recent = mean(
                    _finite(item["val"]["pred_loss"], f"{run}.history.recent")
                    for item in history[-WINDOW:]
                )
                if previous <= 0.0:
                    raise ValueError(f"{run}: non-positive convergence denominator")
                relative_change = (previous - recent) / previous
                emitted_change = _finite(
                    metrics.get("convergence_relative_change"),
                    f"{run}.convergence_relative_change",
                )
                if not math.isclose(
                    relative_change, emitted_change, rel_tol=1e-6, abs_tol=1e-8
                ):
                    raise ValueError(f"{run}: convergence receipt mismatch")
                convergence.append({
                    "run": run,
                    "env": environment,
                    "design": design,
                    "seed": int(seed),
                    "previous_window_mean": previous,
                    "recent_window_mean": recent,
                    "relative_improvement": relative_change,
                })
                rows.append(row)
    expected = len(ENVIRONMENTS) * len(DESIGNS) * len(seeds)
    if len(rows) != expected:
        raise ValueError(f"grid has {len(rows)} rows, expected {expected}")
    return rows, convergence


def _grid(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[dict[tuple[str, str, int], float], tuple[int, ...]]:
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    result: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (str(row["env"]), str(row["design"]), int(row["seed"]))
        if key in result:
            raise ValueError(f"duplicate cell {key}")
        value = _finite(row.get(metric), f"{key}.{metric}")
        if value <= 0.0:
            raise ValueError(f"{key}.{metric} must be positive")
        result[key] = value
    wanted = {
        (environment, design, seed)
        for environment in ENVIRONMENTS
        for design in DESIGNS
        for seed in seeds
    }
    if set(result) != wanted:
        raise ValueError("grid keys do not match the locked study")
    return result, seeds


def pairwise_matrix(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str,
    metric: str = PRIMARY,
) -> np.ndarray:
    lookup, seeds = _grid(rows, metric)
    return np.asarray([
        [
            (lookup[(environment, reference, seed)]
             - lookup[(environment, candidate, seed)])
            / lookup[(environment, reference, seed)]
            for seed in seeds
        ]
        for environment in ENVIRONMENTS
    ], dtype=np.float64)


def pairwise_summary(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str,
    metric: str = PRIMARY,
) -> dict[str, Any]:
    matrix = pairwise_matrix(rows, candidate, reference, metric)
    environment_effects = {
        environment: float(matrix[index].mean())
        for index, environment in enumerate(ENVIRONMENTS)
    }
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "n_pairs": int(matrix.size),
        "mean_paired_relative_reduction": float(matrix.mean()),
        "paired_wins": int((matrix > 0.0).sum()),
        "paired_ties": int((matrix == 0.0).sum()),
        "environment_mean_wins": int(sum(value > 0.0 for value in environment_effects.values())),
        "environment_mean_reductions": environment_effects,
    }


def crossed_bootstrap(matrix: np.ndarray, label: str) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(ENVIRONMENTS), len(FINAL_SEEDS)):
        raise ValueError(f"{label}: final bootstrap matrix shape is {matrix.shape}")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    env_indices = rng.integers(0, matrix.shape[0], size=(BOOTSTRAP_DRAWS, matrix.shape[0]))
    seed_indices = rng.integers(0, matrix.shape[1], size=(BOOTSTRAP_DRAWS, matrix.shape[1]))
    draws = matrix[env_indices[:, :, None], seed_indices[:, None, :]].mean(axis=(1, 2))
    quantiles = np.quantile(draws, (0.05, 0.95), method="linear")
    return {
        "label": label,
        "point": float(matrix.mean()),
        "ci90": [float(quantiles[0]), float(quantiles[1])],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
    }


def convergence_summary(convergence: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = np.abs(np.asarray([
        _finite(row["relative_improvement"], "relative_improvement")
        for row in convergence
    ], dtype=np.float64))
    if values.size == 0:
        raise ValueError("empty convergence records")
    return {
        "median": float(np.quantile(values, 0.5, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "max": float(values.max()),
    }


def candidate_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    candidate = [row for row in rows if row["design"] == CANDIDATE]
    expected = len(ENVIRONMENTS) * len({int(row["seed"]) for row in rows})
    if len(candidate) != expected:
        raise ValueError("candidate receipt rows are incomplete")
    return {
        "orthogonality_max": max(float(row[ORBIT_RECEIPTS[0]]) for row in candidate),
        "streaming_max": max(float(row[ORBIT_RECEIPTS[1]]) for row in candidate),
        "encoder_variance_min": min(float(row[ENCODER_RECEIPTS[0]]) for row in candidate),
        "encoder_effective_rank_min": min(float(row[ENCODER_RECEIPTS[1]]) for row in candidate),
        "encoder_singleton_max": max(float(row[ENCODER_RECEIPTS[2]]) for row in candidate),
        "encoder_prefix_max": max(float(row[ENCODER_RECEIPTS[3]]) for row in candidate),
    }


def observed_summary(rows, convergence) -> dict[str, Any]:
    references = {design: pairwise_summary(rows, CANDIDATE, design) for design in DESIGNS if design != CANDIDATE}
    clean = {
        reference: pairwise_summary(rows, CANDIDATE, reference, CLEAN_METRIC)
        for reference in HEADLINE_REFERENCES
    }
    return {
        "pairwise": references,
        "clean_pairwise": clean,
        "convergence_absolute": convergence_summary(convergence),
        "candidate_receipts": candidate_receipts(rows),
    }


def _pair_gate(
    summary: Mapping[str, Any], reduction: float, wins: int, env_wins: int,
) -> tuple[bool, bool, bool]:
    return (
        float(summary["mean_paired_relative_reduction"]) >= reduction,
        int(summary["paired_wins"]) >= wins,
        int(summary["environment_mean_wins"]) >= env_wins,
    )


def _quality_criteria(observed: Mapping[str, Any]) -> dict[str, bool]:
    convergence = observed["convergence_absolute"]
    receipts = observed["candidate_receipts"]
    return {
        "convergence_absolute_median_lt_1pct": convergence["median"] < 0.01,
        "convergence_absolute_p95_lt_3pct": convergence["p95"] < 0.03,
        "convergence_absolute_max_lt_5pct": convergence["max"] < 0.05,
        "orbit_orthogonality_max_le_1e_5": receipts["orthogonality_max"] <= 1e-5,
        "orbit_streaming_max_le_1e_5": receipts["streaming_max"] <= 1e-5,
        "encoder_variance_min_ge_1e_5": receipts["encoder_variance_min"] >= 1e-5,
        "encoder_effective_rank_min_ge_16": receipts["encoder_effective_rank_min"] >= 16.0,
        "encoder_singleton_max_le_1e_5": receipts["encoder_singleton_max"] <= 1e-5,
        "encoder_prefix_max_le_1e_5": receipts["encoder_prefix_max"] <= 1e-5,
    }


def _decision_criteria(
    observed: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, final: bool,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria: dict[str, bool] = {}
    pairwise = observed["pairwise"]
    wins_headline = 15 if final else 9
    wins_geometry = 14 if final else 9
    wins_noaction = 17 if final else 11
    wins_static = 14 if final else 9

    for reference in HEADLINE_REFERENCES:
        for suffix, passed in zip(
            ("reduction_ge_5pct", f"wins_ge_{wins_headline}", "env_wins_ge_4_of_5"),
            _pair_gate(pairwise[reference], 0.05, wins_headline, 4),
        ):
            criteria[f"vs_{reference}_{suffix}"] = passed
    for reference in GEOMETRY_CONTROLS:
        for suffix, passed in zip(
            ("reduction_ge_2pct", f"wins_ge_{wins_geometry}", "env_wins_ge_3_of_5"),
            _pair_gate(pairwise[reference], 0.02, wins_geometry, 3),
        ):
            criteria[f"vs_{reference}_{suffix}"] = passed
    for suffix, passed in zip(
        ("reduction_ge_5pct", f"wins_ge_{wins_noaction}", "env_wins_ge_3_of_5"),
        _pair_gate(pairwise[NOACTION_CONTROL], 0.05, wins_noaction, 3),
    ):
        criteria[f"vs_{NOACTION_CONTROL}_{suffix}"] = passed
    for suffix, passed in zip(
        ("reduction_ge_1pct", f"wins_ge_{wins_static}", "env_wins_ge_3_of_5"),
        _pair_gate(pairwise[STATIC_CONTROL], 0.01, wins_static, 3),
    ):
        criteria[f"vs_{STATIC_CONTROL}_{suffix}"] = passed

    for reference, summary in observed["clean_pairwise"].items():
        criteria[f"clean_harm_vs_{reference}_le_2pct"] = (
            float(summary["mean_paired_relative_reduction"]) >= -0.02
        )
    criteria.update(_quality_criteria(observed))

    bootstrap: dict[str, Any] = {}
    if final:
        for reference in HEADLINE_REFERENCES:
            receipt = crossed_bootstrap(
                pairwise_matrix(rows, CANDIDATE, reference), reference
            )
            bootstrap[reference] = receipt
            criteria[f"bootstrap90_lower_vs_{reference}_gt_0"] = receipt["ci90"][0] > 0.0
    return criteria, bootstrap


def pilot_decision(rows, convergence) -> dict[str, Any]:
    observed = observed_summary(rows, convergence)
    criteria, _ = _decision_criteria(observed, rows, final=False)
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "phase": "pilot",
        "decision": "PILOT_CONFIRMATION_PASS" if passed else "NO_GO",
        "pilot_screen_passed": passed,
        "criteria": criteria,
        "observed": {
            **observed,
            "bootstrap_contract": BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        },
        "scope": "prospectively_heldout_end_to_end_confirmation",
        "note": "Immutable seeds-0--2 pilot; completion seeds run regardless.",
    }


def final_decision(rows, convergence, *, pilot_screen_passed: bool) -> dict[str, Any]:
    observed = observed_summary(rows, convergence)
    criteria, bootstrap = _decision_criteria(observed, rows, final=True)
    final_gates_passed = all(criteria.values())
    confirmed = bool(pilot_screen_passed and final_gates_passed)
    if confirmed:
        label = "END_TO_END_CONFIRMATION_PASS"
    elif not pilot_screen_passed:
        label = "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    else:
        label = "NO_GO"
    return {
        "schema_version": 1,
        "phase": "final",
        "decision": label,
        "pilot_screen_passed": bool(pilot_screen_passed),
        "final_gates_passed": final_gates_passed,
        "end_to_end_confirmation_passed": confirmed,
        "scoped_component_confirmation_passed": confirmed,
        "iclr_submission_ready": False,
        "criteria": criteria,
        "completed_runs": len(rows),
        "observed": {
            **observed,
            "bootstrap": bootstrap,
            "bootstrap_contract": BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        },
        "scope": "prospectively_heldout_end_to_end_confirmation",
        "limitations": [
            "A positive screen establishes the frozen five-task corruption cohort, not universal superiority.",
            "Private next-latent MSE is diagnostic only and is never pooled across learned encoders.",
            "Executed task return remains a separate outcome unless explicitly present in the sealed artifacts.",
        ],
    }


def contrast_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in (PRIMARY, CLEAN_METRIC):
        for reference in DESIGNS:
            if reference == CANDIDATE:
                continue
            summary = pairwise_summary(rows, CANDIDATE, reference, metric)
            output.append({
                "candidate": CANDIDATE,
                "reference": reference,
                "metric": metric,
                "env": "__overall__",
                "n_pairs": summary["n_pairs"],
                "mean_paired_relative_reduction": summary["mean_paired_relative_reduction"],
                "paired_wins": summary["paired_wins"],
                "paired_ties": summary["paired_ties"],
                "environment_mean_wins": summary["environment_mean_wins"],
            })
            for environment, reduction in summary["environment_mean_reductions"].items():
                output.append({
                    "candidate": CANDIDATE,
                    "reference": reference,
                    "metric": metric,
                    "env": environment,
                    "n_pairs": len({int(row["seed"]) for row in rows}),
                    "mean_paired_relative_reduction": reduction,
                    "paired_wins": "",
                    "paired_ties": "",
                    "environment_mean_wins": "",
                })
    return output


def strict_validate_cells(root: Path, seeds: Sequence[int]) -> None:
    import scripts.run_hacssm_v10 as runner

    original_root = runner.OUTPUT_ROOT
    try:
        runner.OUTPUT_ROOT = root.resolve()
        runner.configure_shared()
        jobs = tuple(
            runner.shared.Job(
                "pilot" if seed in PILOT_SEEDS else "completion",
                seed, environment, environment, design,
            )
            for seed in seeds
            for environment in ENVIRONMENTS
            for design in DESIGNS
        )
        for job in jobs:
            runner.validate_job(job, allow_missing=False)
    finally:
        runner.OUTPUT_ROOT = original_root
        runner.configure_shared()


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze the locked ORBIT-v10 grid.")
    parser.add_argument("--root", type=Path, default=Path("outputs/hacssm_v10_shared"))
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = PILOT_SEEDS if args.phase == "pilot" else FINAL_SEEDS
    strict_validate_cells(args.root, seeds)
    rows, convergence = load_cells(args.root, seeds)
    grouped = shared.grouped_rows(rows)
    contrasts = contrast_rows(rows)
    prefix = "pilot_" if args.phase == "pilot" else ""
    if args.phase == "pilot":
        decision = pilot_decision(rows, convergence)
    else:
        pilot_path = args.root / "pilot_decision.json"
        pilot = shared.read_json(pilot_path)
        pilot_rows = [row for row in rows if int(row["seed"]) in PILOT_SEEDS]
        pilot_convergence = [
            row for row in convergence if int(row["seed"]) in PILOT_SEEDS
        ]
        recomputed = pilot_decision(pilot_rows, pilot_convergence)
        if pilot != recomputed:
            raise ValueError(f"immutable pilot decision mismatch: {pilot_path}")
        decision = final_decision(
            rows, convergence,
            pilot_screen_passed=bool(recomputed["pilot_screen_passed"]),
        )
    shared.atomic_csv(args.root / f"{prefix}per_run.csv", rows)
    shared.atomic_csv(args.root / f"{prefix}grouped.csv", grouped)
    shared.atomic_csv(args.root / f"{prefix}paired_contrasts.csv", contrasts)
    shared.atomic_csv(args.root / f"{prefix}convergence.csv", convergence)
    shared.atomic_json(
        args.root / ("pilot_decision.json" if args.phase == "pilot" else "decision.json"),
        decision,
    )
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
