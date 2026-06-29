#!/usr/bin/env python3
"""Fail-closed analysis for the locked LOIF-v9 adaptive-development grid.

Raw latent MSE is never pooled across environments.  Every cross-environment
estimate is an equal-cell paired relative reduction.  The immutable pilot is
recomputed before the five-seed descriptive/final label is written.
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


OCC_TO_CLEAN = shared.OCC_TO_CLEAN
V9_DESIGNS = (
    "loifv9",
    "loifv9_fixedalpha",
    "loifv9_globalR",
    "loifv9_innovationonly",
    "loifv9_latentonly",
    "loifv9_uniformfusion",
    "loifv9_noaction",
    "loifv9_singlebank",
)
REFERENCE_DESIGNS = (
    "ssm",
    "hacssmv7_sharedaction",
    "hacssmv8",
    "hacssmv8_dynamic",
    "hacssmv8_static",
)
DESIGNS = V9_DESIGNS + REFERENCE_DESIGNS
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
PRIMARY = "clean_mse_first_post"
EPOCHS = 200
WINDOW = 10
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v9"
EVAL_ROLLOUT_EPISODE = 0

CANDIDATE = "loifv9"
V7_REFERENCE = "hacssmv7_sharedaction"
V8_REFERENCE = "hacssmv8"
DYNAMIC_ENDPOINT = "hacssmv8_dynamic"
STATIC_ENDPOINT = "hacssmv8_static"
HEADLINE_REFERENCES = (V7_REFERENCE, V8_REFERENCE)
ADAPTIVE_EVIDENCE_CONTROLS = (
    "loifv9_fixedalpha",
    "loifv9_globalR",
    "loifv9_innovationonly",
    "loifv9_latentonly",
)
FUSION_CONTROL = "loifv9_uniformfusion"
STRUCTURAL_CONTROLS = ("loifv9_noaction", "loifv9_singlebank")

BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 8_008
BOOTSTRAP_CONTRACT = {
    "schema_version": 1,
    "algorithm": "crossed_environment_seed_percentile_bootstrap",
    "draws": BOOTSTRAP_DRAWS,
    "seed": BOOTSTRAP_SEED,
    "rng": "numpy.random.Generator(numpy.random.PCG64)",
    "resampling": (
        "independently sample E environment indices and S optimizer-seed indices "
        "with replacement; evaluate the E-by-S Cartesian product; equal-weight mean"
    ),
    "estimand": "mean paired relative reduction (reference-candidate)/reference",
    "quantiles": {"method": "linear", "reported": [0.05, 0.95]},
}
BOOTSTRAP_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        BOOTSTRAP_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()

PHASE_METRICS = ("clean_mse_deep_blackout", "clean_mse_all")
INTERVENTION_PHASES = ("first_post", "deep_blackout", "all")
INTERVENTION_KINDS = ("permuted", "mean")
DIAGNOSTIC_PHASES = ("visible", "blackout_transition", "deep_blackout", "recovery")
DIAGNOSTIC_STATS = (
    "log_R", "K_fast", "K_slow", "log_P_fast", "log_P_slow",
    "omega_fast", "omega_slow", "pi_fast", "pi_slow",
    "direct_fast", "direct_slow", "innovation_norm",
    "action_state_influence", "action_output_influence",
)

DIAGNOSTIC_BOOLEAN_FIELDS = (
    "loif_pole_collapsed",
    "loif_boundary_saturated",
    "loif_streaming_equivalent",
    *(f"loif_innovation_or_log_R_constant_{phase}" for phase in DIAGNOSTIC_PHASES),
)
DIAGNOSTIC_RECEIPT_FIELDS = (
    "loif_pole_separation",
    "loif_fast_boundary_margin",
    "loif_slow_boundary_margin",
    "loif_pole_boundary_margin",
    "loif_saturation_tolerance",
    "loif_log_scale_extreme_threshold",
    "loif_gain_saturated_fraction",
    "loif_log_R_extreme_fraction",
    "loif_log_P_extreme_fraction",
    "loif_nonfinite_diagnostic_count",
    "loif_streaming_batch_size",
    "loif_streaming_tolerance",
    "loif_streaming_mixed_max_abs",
    "loif_streaming_state_max_abs",
    "loif_streaming_log_P_max_abs",
    *(f"loif_innovation_log_R_corr_{phase}" for phase in DIAGNOSTIC_PHASES),
)

ROW_METRICS = (
    "val_pred_loss",
    "clean_mse_pre",
    "clean_mse_blackout_transition",
    "clean_mse_deep_blackout",
    "clean_mse_first_post",
    "clean_mse_recovery",
    "clean_mse_late_post",
    "clean_mse_all",
    "clean_mse_first_post_ablated",
    "clean_input_mse_first_post",
    "last_visible_mse_first_post",
    "constant_mse_first_post",
    "persistence_mse_first_post",
    "infl_all",
    "infl_fast",
    "infl_slow",
    "alpha_fast",
    "alpha_slow",
    "q_fast",
    "q_slow",
    *DIAGNOSTIC_RECEIPT_FIELDS,
    *DIAGNOSTIC_BOOLEAN_FIELDS,
    *(
        f"loif_{stat}_{phase}"
        for phase in DIAGNOSTIC_PHASES
        for stat in DIAGNOSTIC_STATS
    ),
    *(
        f"clean_mse_{phase}_resistance_{kind}"
        for phase in INTERVENTION_PHASES
        for kind in INTERVENTION_KINDS
    ),
)


def _finite(value: Any, context: str) -> float:
    return shared.finite(value, context)


def load_cells(root: Path, seeds: Sequence[int]):
    """Load already runner-validated cells without V5's legacy 0.5 loss literal."""
    expected = {
        (env, design, seed): root / f"lewm-{env}-{design}-s{seed}"
        for env in OCC_TO_CLEAN
        for design in DESIGNS
        for seed in seeds
    }
    rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for (env, design, seed), run_dir in sorted(expected.items()):
        metrics = shared.read_json(run_dir / "metrics.json")
        checkpoint = torch.load(
            run_dir / "model.pt", map_location="cpu", weights_only=False
        )
        if not isinstance(metrics, dict) or not isinstance(checkpoint, dict):
            raise ValueError(f"{run_dir}: invalid metric/checkpoint objects")
        if metrics != checkpoint.get("final_metrics"):
            raise ValueError(f"{run_dir}: metrics/checkpoint mismatch")
        history = checkpoint.get("history")
        if not isinstance(history, list) or len(history) != EPOCHS:
            raise ValueError(f"{run_dir}: incomplete history")
        row: dict[str, Any] = {
            "run": run_dir.name,
            "env": env,
            "design": design,
            "seed": seed,
            "trainable_parameters": int(metrics["trainable_parameters"]),
        }
        for key in ROW_METRICS:
            if key in metrics and metrics[key] is not None:
                if key in DIAGNOSTIC_BOOLEAN_FIELDS:
                    if type(metrics[key]) is not bool:
                        raise ValueError(f"{run_dir}.{key} is not boolean")
                    row[key] = int(metrics[key])
                else:
                    row[key] = _finite(metrics[key], f"{run_dir}.{key}")
            else:
                row[key] = ""
        for key in (PRIMARY, "val_pred_loss", "last_visible_mse_first_post"):
            if row[key] == "":
                raise ValueError(f"{run_dir}: required metric {key} is absent")
        previous = mean(
            _finite(item["val"]["pred_loss"], f"{run_dir}.previous")
            for item in history[-2 * WINDOW:-WINDOW]
        )
        recent = mean(
            _finite(item["val"]["pred_loss"], f"{run_dir}.recent")
            for item in history[-WINDOW:]
        )
        if previous <= 0.0:
            raise ValueError(f"{run_dir}: non-positive convergence denominator")
        convergence.append({
            "run": run_dir.name,
            "env": env,
            "design": design,
            "seed": seed,
            "previous_window_mean": previous,
            "recent_window_mean": recent,
            "relative_improvement": (previous - recent) / previous,
        })
        rows.append(row)
    return rows, convergence


def _grid(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[dict[tuple[str, str, int], float], tuple[int, ...]]:
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    expected_count = len(OCC_TO_CLEAN) * len(DESIGNS) * len(seeds)
    if len(rows) != expected_count:
        raise ValueError(f"grid has {len(rows)} rows, expected {expected_count}")
    result: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (str(row["env"]), str(row["design"]), int(row["seed"]))
        if key in result:
            raise ValueError(f"duplicate grid cell {key}")
        value = _finite(row.get(metric), f"{key}.{metric}")
        if value <= 0.0:
            raise ValueError(f"{key}.{metric} must be positive")
        result[key] = value
    wanted = {
        (env, design, seed)
        for env in OCC_TO_CLEAN
        for design in DESIGNS
        for seed in seeds
    }
    if set(result) != wanted:
        raise ValueError("grid keys do not match the locked design")
    return result, seeds


def pairwise_matrix(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
    metric: str = PRIMARY,
) -> np.ndarray:
    if candidate == reference or candidate not in DESIGNS or reference not in DESIGNS:
        raise ValueError(f"invalid contrast {candidate}/{reference}")
    lookup, seeds = _grid(rows, metric)
    matrix = np.empty((len(OCC_TO_CLEAN), len(seeds)), dtype=np.float64)
    for env_index, env in enumerate(OCC_TO_CLEAN):
        for seed_index, seed in enumerate(seeds):
            cand = lookup[(env, candidate, seed)]
            ref = lookup[(env, reference, seed)]
            matrix[env_index, seed_index] = (ref - cand) / ref
    return matrix


def _summary_from_matrix(
    matrix: np.ndarray,
    *,
    candidate: str,
    reference: str,
    environment_effects: Mapping[str, float],
    environment_wins: int,
    metric: str,
) -> dict[str, Any]:
    if matrix.shape[0] != len(OCC_TO_CLEAN) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid relative-reduction matrix for {reference}")
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "n_pairs": int(matrix.size),
        "mean_paired_relative_reduction": float(matrix.mean()),
        "paired_wins": int((matrix > 0.0).sum()),
        "paired_ties": int((matrix == 0.0).sum()),
        "environment_mean_wins": int(environment_wins),
        "environment_mean_reductions": dict(environment_effects),
    }


def pairwise_summary(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
    metric: str = PRIMARY,
) -> dict[str, Any]:
    matrix = pairwise_matrix(rows, candidate, reference, metric)
    lookup, seeds = _grid(rows, metric)
    effects = {}
    wins = 0
    for env in OCC_TO_CLEAN:
        cand = mean(lookup[(env, candidate, seed)] for seed in seeds)
        ref = mean(lookup[(env, reference, seed)] for seed in seeds)
        effects[env] = (ref - cand) / ref
        wins += cand < ref
    return _summary_from_matrix(
        matrix, candidate=candidate, reference=reference,
        environment_effects=effects, environment_wins=wins, metric=metric,
    )


def endpoint_matrix(rows: Sequence[Mapping[str, Any]], metric: str = PRIMARY) -> np.ndarray:
    lookup, seeds = _grid(rows, metric)
    matrix = np.empty((len(OCC_TO_CLEAN), len(seeds)), dtype=np.float64)
    for env_index, env in enumerate(OCC_TO_CLEAN):
        for seed_index, seed in enumerate(seeds):
            candidate = lookup[(env, CANDIDATE, seed)]
            reference = min(
                lookup[(env, DYNAMIC_ENDPOINT, seed)],
                lookup[(env, STATIC_ENDPOINT, seed)],
            )
            matrix[env_index, seed_index] = (reference - candidate) / reference
    return matrix


def endpoint_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = endpoint_matrix(rows)
    lookup, seeds = _grid(rows, PRIMARY)
    effects = {}
    wins = 0
    for env in OCC_TO_CLEAN:
        candidate = mean(lookup[(env, CANDIDATE, seed)] for seed in seeds)
        reference = min(
            mean(lookup[(env, DYNAMIC_ENDPOINT, seed)] for seed in seeds),
            mean(lookup[(env, STATIC_ENDPOINT, seed)] for seed in seeds),
        )
        effects[env] = (reference - candidate) / reference
        wins += candidate < reference
    result = _summary_from_matrix(
        matrix, candidate=CANDIDATE, reference="v8_dynamic_static_endpoint_envelope",
        environment_effects=effects, environment_wins=wins, metric=PRIMARY,
    )
    result["references"] = [DYNAMIC_ENDPOINT, STATIC_ENDPOINT]
    return result


def intervention_matrix(
    rows: Sequence[Mapping[str, Any]], kind: str, phase: str = "first_post"
) -> np.ndarray:
    if kind not in INTERVENTION_KINDS or phase not in INTERVENTION_PHASES:
        raise ValueError(f"invalid intervention {phase}/{kind}")
    normal_metric = f"clean_mse_{phase}"
    override_metric = f"clean_mse_{phase}_resistance_{kind}"
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    candidate_rows = [row for row in rows if row["design"] == CANDIDATE]
    lookup = {(str(row["env"]), int(row["seed"])): row for row in candidate_rows}
    wanted = {(env, seed) for env in OCC_TO_CLEAN for seed in seeds}
    if set(lookup) != wanted:
        raise ValueError("candidate intervention grid is incomplete")
    matrix = np.empty((len(OCC_TO_CLEAN), len(seeds)), dtype=np.float64)
    for env_index, env in enumerate(OCC_TO_CLEAN):
        for seed_index, seed in enumerate(seeds):
            row = lookup[(env, seed)]
            candidate = _finite(row.get(normal_metric), f"{env}/{seed}/{normal_metric}")
            reference = _finite(row.get(override_metric), f"{env}/{seed}/{override_metric}")
            if candidate <= 0.0 or reference <= 0.0:
                raise ValueError("intervention contrast requires positive MSE")
            matrix[env_index, seed_index] = (reference - candidate) / reference
    return matrix


def intervention_summary(
    rows: Sequence[Mapping[str, Any]], kind: str, phase: str = "first_post"
) -> dict[str, Any]:
    matrix = intervention_matrix(rows, kind, phase)
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    candidate_rows = [row for row in rows if row["design"] == CANDIDATE]
    lookup = {(str(row["env"]), int(row["seed"])): row for row in candidate_rows}
    normal_metric = f"clean_mse_{phase}"
    override_metric = f"clean_mse_{phase}_resistance_{kind}"
    effects = {}
    wins = 0
    for env in OCC_TO_CLEAN:
        candidate = mean(float(lookup[(env, seed)][normal_metric]) for seed in seeds)
        reference = mean(float(lookup[(env, seed)][override_metric]) for seed in seeds)
        effects[env] = (reference - candidate) / reference
        wins += candidate < reference
    return _summary_from_matrix(
        matrix,
        candidate=CANDIDATE,
        reference=f"resistance_{kind}",
        environment_effects=effects,
        environment_wins=wins,
        metric=normal_metric,
    )


def crossed_bootstrap(matrix: np.ndarray, label: str) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[0] != len(OCC_TO_CLEAN):
        raise ValueError(f"{label}: invalid crossed-bootstrap matrix")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    env_indices = rng.integers(0, matrix.shape[0], (BOOTSTRAP_DRAWS, matrix.shape[0]))
    seed_indices = rng.integers(0, matrix.shape[1], (BOOTSTRAP_DRAWS, matrix.shape[1]))
    sampled = matrix[
        env_indices[:, :, np.newaxis], seed_indices[:, np.newaxis, :]
    ].mean(axis=(1, 2))
    q05, q95 = np.quantile(sampled, (0.05, 0.95), method="linear")
    return {
        "label": label,
        "n_environments": matrix.shape[0],
        "n_seeds": matrix.shape[1],
        "point_mean_paired_relative_reduction": float(matrix.mean()),
        "ci90": [float(q05), float(q95)],
        "contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
    }


def _convergence(convergence: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = np.abs(np.asarray([
        _finite(row.get("relative_improvement"), "convergence")
        for row in convergence
    ], dtype=np.float64))
    if values.size == 0:
        raise ValueError("empty convergence table")
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def phase_contrast_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for metric in PHASE_METRICS:
        summary = pairwise_summary(rows, CANDIDATE, V7_REFERENCE, metric)
        matrix = pairwise_matrix(rows, CANDIDATE, V7_REFERENCE, metric)
        for env_index, env in enumerate((*OCC_TO_CLEAN, "__overall__")):
            output.append({
                "candidate": CANDIDATE,
                "reference": V7_REFERENCE,
                "metric": metric,
                "env": env,
                "n_pairs": len({int(row["seed"]) for row in rows}) if env != "__overall__" else summary["n_pairs"],
                "mean_paired_relative_reduction": (
                    float(matrix[env_index].mean())
                    if env != "__overall__" else summary["mean_paired_relative_reduction"]
                ),
                "paired_wins": "" if env != "__overall__" else summary["paired_wins"],
            })
    return output


def intervention_contrast_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    n_seeds = len({int(row["seed"]) for row in rows})
    for phase in INTERVENTION_PHASES:
        for kind in INTERVENTION_KINDS:
            summary = intervention_summary(rows, kind, phase)
            matrix = intervention_matrix(rows, kind, phase)
            for env_index, env in enumerate((*OCC_TO_CLEAN, "__overall__")):
                output.append({
                    "candidate": CANDIDATE,
                    "intervention": f"resistance_{kind}",
                    "metric": f"clean_mse_{phase}",
                    "env": env,
                    "n_pairs": n_seeds if env != "__overall__" else summary["n_pairs"],
                    "mean_paired_relative_reduction": (
                        float(matrix[env_index].mean())
                        if env != "__overall__" else summary["mean_paired_relative_reduction"]
                    ),
                    "paired_wins": "" if env != "__overall__" else summary["paired_wins"],
                })
    return output


def _diagnostic_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = [row for row in rows if row["design"] == CANDIDATE]
    separation = np.asarray([
        float(row["alpha_slow"]) - float(row["alpha_fast"]) for row in candidate
    ])
    alpha_fast = np.asarray([float(row["alpha_fast"]) for row in candidate])
    alpha_slow = np.asarray([float(row["alpha_slow"]) for row in candidate])
    return {
        "n_cells": len(candidate),
        "pole_separation_mean": float(separation.mean()),
        "pole_separation_min": float(separation.min()),
        "pole_separation_le_0_01_cells": int((separation <= 0.01).sum()),
        "fast_boundary_le_0_01_cells": int((alpha_fast <= 0.01).sum()),
        "slow_boundary_ge_0_99_cells": int((alpha_slow >= 0.99).sum()),
        "pole_collapsed_cells": sum(
            int(row["loif_pole_collapsed"]) for row in candidate
        ),
        "boundary_saturated_cells": sum(
            int(row["loif_boundary_saturated"]) for row in candidate
        ),
        "gain_saturation_max_fraction": max(
            float(row["loif_gain_saturated_fraction"]) for row in candidate
        ),
        "log_R_extreme_max_fraction": max(
            float(row["loif_log_R_extreme_fraction"]) for row in candidate
        ),
        "log_P_extreme_max_fraction": max(
            float(row["loif_log_P_extreme_fraction"]) for row in candidate
        ),
        "nonfinite_diagnostic_count": sum(
            int(row["loif_nonfinite_diagnostic_count"]) for row in candidate
        ),
        "streaming_nonequivalent_cells": sum(
            not bool(row["loif_streaming_equivalent"]) for row in candidate
        ),
        "diversity_loss_imposed": False,
    }


def _scientific_stop_criteria(observed: Mapping[str, Any]) -> dict[str, bool]:
    diagnostics = observed["candidate_diagnostics"]
    return {
        "no_pole_collapse_cells": diagnostics["pole_collapsed_cells"] == 0,
        "no_pole_boundary_saturation_cells": (
            diagnostics["boundary_saturated_cells"] == 0
        ),
        "no_gain_saturation_samples": (
            diagnostics["gain_saturation_max_fraction"] == 0.0
        ),
        "no_extreme_log_R_samples": (
            diagnostics["log_R_extreme_max_fraction"] == 0.0
        ),
        "no_extreme_log_P_samples": (
            diagnostics["log_P_extreme_max_fraction"] == 0.0
        ),
        "batch_size_one_streaming_equivalent_all_cells": (
            diagnostics["streaming_nonequivalent_cells"] == 0
        ),
    }


def _observed(rows, convergence) -> dict[str, Any]:
    pairwise = {
        reference: pairwise_summary(rows, CANDIDATE, reference)
        for reference in DESIGNS if reference != CANDIDATE
    }
    endpoint = endpoint_summary(rows)
    interventions = {
        kind: intervention_summary(rows, kind) for kind in INTERVENTION_KINDS
    }
    phases = {
        metric: pairwise_summary(rows, CANDIDATE, V7_REFERENCE, metric)
        for metric in PHASE_METRICS
    }
    return {
        "pairwise": pairwise,
        "endpoint_envelope": endpoint,
        "interventions": interventions,
        "phase_vs_v7_sharedaction": phases,
        "convergence_absolute": _convergence(convergence),
        "candidate_diagnostics": _diagnostic_summary(rows),
    }


def _pair_gate(
    summary: Mapping[str, Any], reduction: float, wins: int, env_wins: int,
    *, strict_reduction: bool = False,
) -> tuple[bool, bool, bool]:
    observed = float(summary["mean_paired_relative_reduction"])
    reduction_ok = observed > reduction if strict_reduction else observed >= reduction
    return (
        reduction_ok,
        int(summary["paired_wins"]) >= wins,
        int(summary["environment_mean_wins"]) >= env_wins,
    )


def _phase_criteria(observed: Mapping[str, Any]) -> dict[str, bool]:
    result = {}
    for metric, summary in observed["phase_vs_v7_sharedaction"].items():
        label = metric.removeprefix("clean_mse_")
        effects = summary["environment_mean_reductions"]
        result[f"{label}_paired_reduction_gt_minus_1pct"] = (
            float(summary["mean_paired_relative_reduction"]) > -0.01
        )
        result[f"{label}_env_effect_gt_minus_1pct_ge_3_of_5"] = sum(
            float(value) > -0.01 for value in effects.values()
        ) >= 3
    return result


def pilot_decision(rows, convergence, contrasts=None) -> dict[str, Any]:
    observed = _observed(rows, convergence)
    pairwise = observed["pairwise"]
    criteria: dict[str, bool] = {}

    for suffix, passed in zip(
        ("reduction_ge_6pct", "wins_ge_10_of_15", "env_wins_ge_4_of_5"),
        _pair_gate(pairwise["ssm"], 0.06, 10, 4),
    ):
        criteria[f"vs_ssm_{suffix}"] = passed
    for reference in HEADLINE_REFERENCES:
        label = "v7_sharedaction" if reference == V7_REFERENCE else "compact_v8"
        for suffix, passed in zip(
            ("reduction_ge_0_5pct", "wins_ge_9_of_15", "env_wins_ge_3_of_5"),
            _pair_gate(pairwise[reference], 0.005, 9, 3),
        ):
            criteria[f"vs_{label}_{suffix}"] = passed
    for reference in ADAPTIVE_EVIDENCE_CONTROLS:
        label = reference.removeprefix("loifv9_")
        for suffix, passed in zip(
            ("reduction_ge_0_25pct", "wins_ge_9_of_15", "env_wins_ge_3_of_5"),
            _pair_gate(pairwise[reference], 0.0025, 9, 3),
        ):
            criteria[f"vs_{label}_{suffix}"] = passed
    for suffix, passed in zip(
        ("reduction_gt_0", "wins_ge_9_of_15", "env_wins_ge_3_of_5"),
        _pair_gate(observed["endpoint_envelope"], 0.0, 9, 3, strict_reduction=True),
    ):
        criteria[f"vs_endpoint_envelope_{suffix}"] = passed
    for suffix, passed in zip(
        ("reduction_gt_0", "wins_ge_8_of_15", "env_wins_ge_3_of_5"),
        _pair_gate(pairwise[FUSION_CONTROL], 0.0, 8, 3, strict_reduction=True),
    ):
        criteria[f"vs_uniformfusion_{suffix}"] = passed
    for reference in STRUCTURAL_CONTROLS:
        label = reference.removeprefix("loifv9_")
        for suffix, passed in zip(
            ("reduction_ge_3pct", "wins_ge_11_of_15", "env_wins_ge_3_of_5"),
            _pair_gate(pairwise[reference], 0.03, 11, 3),
        ):
            criteria[f"vs_{label}_{suffix}"] = passed
    for kind, summary in observed["interventions"].items():
        for suffix, passed in zip(
            ("reduction_ge_0_25pct", "wins_ge_9_of_15", "env_wins_ge_3_of_5"),
            _pair_gate(summary, 0.0025, 9, 3),
        ):
            criteria[f"vs_resistance_{kind}_{suffix}"] = passed
    criteria.update(_phase_criteria(observed))
    criteria.update(_scientific_stop_criteria(observed))
    conv = observed["convergence_absolute"]
    criteria.update({
        "convergence_absolute_median_lt_1pct": conv["median"] < 0.01,
        "convergence_absolute_p95_lt_3pct": conv["p95"] < 0.03,
        "convergence_absolute_max_lt_5pct": conv["max"] < 0.05,
    })
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "phase": "pilot",
        "decision": "PILOT_OVERALL_BEST_PASS" if passed else "NO_GO",
        "pilot_screen_passed": passed,
        "criteria": criteria,
        "observed": {
            **observed,
            "bootstrap_contract": BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        },
        "adaptive_development_only": True,
        "scope": "adaptive_development_only",
        "note": "Immutable adaptive-development pilot; all five seeds run regardless.",
    }


def _environment_envelopes(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    lookup, seeds = _grid(rows, PRIMARY)
    envelope_wins = 0
    hold_wins = 0
    candidate_rows = [row for row in rows if row["design"] == CANDIDATE]
    for env in OCC_TO_CLEAN:
        candidate = mean(lookup[(env, CANDIDATE, seed)] for seed in seeds)
        reference = min(
            mean(lookup[(env, design, seed)] for seed in seeds)
            for design in DESIGNS if design != CANDIDATE
        )
        envelope_wins += candidate < reference
        hold = mean(
            float(row["last_visible_mse_first_post"])
            for row in candidate_rows if row["env"] == env
        )
        hold_wins += candidate < hold
    return envelope_wins, hold_wins


def final_summary(
    rows,
    convergence,
    contrasts=None,
    *,
    pilot_screen_passed: bool,
) -> dict[str, Any]:
    observed = _observed(rows, convergence)
    pairwise = observed["pairwise"]
    criteria: dict[str, bool] = {}

    for suffix, passed in zip(
        ("reduction_ge_7pct", "wins_ge_20_of_25", "env_wins_ge_4_of_5"),
        _pair_gate(pairwise["ssm"], 0.07, 20, 4),
    ):
        criteria[f"vs_ssm_{suffix}"] = passed
    for reference in HEADLINE_REFERENCES:
        label = "v7_sharedaction" if reference == V7_REFERENCE else "compact_v8"
        for suffix, passed in zip(
            ("reduction_ge_1pct", "wins_ge_15_of_25", "env_wins_ge_3_of_5"),
            _pair_gate(pairwise[reference], 0.01, 15, 3),
        ):
            criteria[f"vs_{label}_{suffix}"] = passed
    for reference in ADAPTIVE_EVIDENCE_CONTROLS:
        label = reference.removeprefix("loifv9_")
        for suffix, passed in zip(
            ("reduction_ge_0_5pct", "wins_ge_14_of_25", "env_wins_ge_3_of_5"),
            _pair_gate(pairwise[reference], 0.005, 14, 3),
        ):
            criteria[f"vs_{label}_{suffix}"] = passed
    for suffix, passed in zip(
        ("reduction_ge_0_5pct", "wins_ge_14_of_25", "env_wins_ge_3_of_5"),
        _pair_gate(observed["endpoint_envelope"], 0.005, 14, 3),
    ):
        criteria[f"vs_endpoint_envelope_{suffix}"] = passed
    for suffix, passed in zip(
        ("reduction_ge_0_5pct", "wins_ge_14_of_25", "env_wins_ge_3_of_5"),
        _pair_gate(pairwise[FUSION_CONTROL], 0.005, 14, 3),
    ):
        criteria[f"vs_uniformfusion_{suffix}"] = passed
    for reference in STRUCTURAL_CONTROLS:
        label = reference.removeprefix("loifv9_")
        for suffix, passed in zip(
            ("reduction_ge_3pct", "wins_ge_17_of_25", "env_wins_ge_3_of_5"),
            _pair_gate(pairwise[reference], 0.03, 17, 3),
        ):
            criteria[f"vs_{label}_{suffix}"] = passed
    for kind, summary in observed["interventions"].items():
        for suffix, passed in zip(
            ("reduction_ge_0_5pct", "wins_ge_14_of_25", "env_wins_ge_3_of_5"),
            _pair_gate(summary, 0.005, 14, 3),
        ):
            criteria[f"vs_resistance_{kind}_{suffix}"] = passed

    bootstrap = {}
    for reference in (*HEADLINE_REFERENCES, *ADAPTIVE_EVIDENCE_CONTROLS):
        receipt = crossed_bootstrap(
            pairwise_matrix(rows, CANDIDATE, reference), reference
        )
        bootstrap[reference] = receipt
        criteria[f"bootstrap90_lower_vs_{reference}_gt_0"] = receipt["ci90"][0] > 0.0
    endpoint_bootstrap = crossed_bootstrap(
        endpoint_matrix(rows), "v8_dynamic_static_endpoint_envelope"
    )
    bootstrap["v8_dynamic_static_endpoint_envelope"] = endpoint_bootstrap
    criteria["bootstrap90_lower_vs_endpoint_envelope_gt_0"] = (
        endpoint_bootstrap["ci90"][0] > 0.0
    )
    criteria.update(_phase_criteria(observed))
    criteria.update(_scientific_stop_criteria(observed))
    envelope_wins, hold_wins = _environment_envelopes(rows)
    criteria.update({
        "full_grid_environment_envelope_wins_ge_3_of_5": envelope_wins >= 3,
        "last_visible_hold_wins_ge_4_of_5": hold_wins >= 4,
    })
    conv = observed["convergence_absolute"]
    criteria.update({
        "convergence_absolute_median_lt_1pct": conv["median"] < 0.01,
        "convergence_absolute_p95_lt_3pct": conv["p95"] < 0.03,
        "convergence_absolute_max_lt_5pct": conv["max"] < 0.05,
    })

    final_gates_passed = all(criteria.values())
    best = bool(pilot_screen_passed and final_gates_passed)
    if best:
        decision = "OVERALL_BEST_ADAPTIVE_DEV"
    elif not pilot_screen_passed:
        decision = "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    else:
        decision = "NO_GO"
    return {
        "schema_version": 1,
        "phase": "final",
        "decision": decision,
        "pilot_screen_passed": bool(pilot_screen_passed),
        "final_gates_passed": final_gates_passed,
        "best_in_locked_grid": best,
        "good_enough_for_overall_best_claim": best,
        "criteria": criteria,
        "completed_runs": len(rows),
        "observed": {
            **observed,
            "bootstrap": bootstrap,
            "bootstrap_contract": BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
            "full_grid_environment_envelope_wins": envelope_wins,
            "last_visible_hold_environment_wins": hold_wins,
        },
        "adaptive_development_only": True,
        "scope": "adaptive_development_only",
        "limitations": [
            "V9 was selected after inspecting V1-V8 on these same development tasks.",
            "Optimizer seeds do not create an untouched task/corruption cohort.",
            "No simulator-state outcome, executed return, or tuned contemporary baseline is measured.",
        ],
        "note": (
            "This deterministic label is adaptive-development evidence only, not an "
            "untouched confirmation or publication claim."
        ),
    }


def contrast_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    lookup, _ = _grid(rows, PRIMARY)
    for reference in DESIGNS:
        if reference == CANDIDATE:
            continue
        summary = pairwise_summary(rows, CANDIDATE, reference)
        for env in (*OCC_TO_CLEAN, "__overall__"):
            if env == "__overall__":
                result.append({
                    "candidate": CANDIDATE,
                    "reference": reference,
                    "env": env,
                    "n_pairs": summary["n_pairs"],
                    "candidate_mean_mse": "",
                    "reference_mean_mse": "",
                    "mean_paired_relative_reduction": summary["mean_paired_relative_reduction"],
                    "paired_wins": summary["paired_wins"],
                    "paired_ties": summary["paired_ties"],
                })
            else:
                candidate_values = [lookup[(env, CANDIDATE, seed)] for seed in seeds]
                reference_values = [lookup[(env, reference, seed)] for seed in seeds]
                reductions = [
                    (ref - cand) / ref
                    for cand, ref in zip(candidate_values, reference_values)
                ]
                result.append({
                    "candidate": CANDIDATE,
                    "reference": reference,
                    "env": env,
                    "n_pairs": len(seeds),
                    "candidate_mean_mse": mean(candidate_values),
                    "reference_mean_mse": mean(reference_values),
                    "mean_paired_relative_reduction": mean(reductions),
                    "paired_wins": sum(cand < ref for cand, ref in zip(candidate_values, reference_values)),
                    "paired_ties": sum(cand == ref for cand, ref in zip(candidate_values, reference_values)),
                })
    return result


def strict_validate_cells(root: Path, seeds: Sequence[int]) -> None:
    import scripts.run_hacssm_v9 as runner

    original_root = runner.OUTPUT_ROOT
    try:
        runner.OUTPUT_ROOT = root.resolve()
        runner.configure_shared()
        jobs = tuple(
            runner.shared.Job(
                "pilot" if seed in PILOT_SEEDS else "completion",
                seed, env, OCC_TO_CLEAN[env], design,
            )
            for seed in seeds
            for env in OCC_TO_CLEAN
            for design in DESIGNS
        )
        for job in jobs:
            runner.shared.validate_job(job, allow_missing=False)
    finally:
        runner.OUTPUT_ROOT = original_root


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze the locked LOIF-v9 grid.")
    parser.add_argument("--root", type=Path, default=Path("outputs/hacssm_v9_shared"))
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = PILOT_SEEDS if args.phase == "pilot" else FINAL_SEEDS
    strict_validate_cells(args.root, seeds)
    rows, convergence = load_cells(args.root, seeds)
    expected = len(OCC_TO_CLEAN) * len(DESIGNS) * len(seeds)
    if len(rows) != expected:
        raise ValueError(f"{args.phase} grid has {len(rows)} rows, expected {expected}")
    grouped = shared.grouped_rows(rows)
    contrasts = contrast_rows(rows)
    phase_rows = phase_contrast_rows(rows)
    intervention_rows = intervention_contrast_rows(rows)
    prefix = "pilot_" if args.phase == "pilot" else ""
    if args.phase == "pilot":
        decision = pilot_decision(rows, convergence, contrasts)
    else:
        pilot_path = args.root / "pilot_decision.json"
        pilot = shared.read_json(pilot_path)
        pilot_rows = [row for row in rows if int(row["seed"]) in PILOT_SEEDS]
        pilot_convergence = [
            row for row in convergence if int(row["seed"]) in PILOT_SEEDS
        ]
        recomputed = pilot_decision(pilot_rows, pilot_convergence)
        if pilot != recomputed:
            raise ValueError(f"invalid immutable pilot decision: {pilot_path}")
        decision = final_summary(
            rows, convergence, contrasts,
            pilot_screen_passed=recomputed["pilot_screen_passed"],
        )
    shared.atomic_csv(args.root / f"{prefix}per_run.csv", rows)
    shared.atomic_csv(args.root / f"{prefix}grouped.csv", grouped)
    shared.atomic_csv(args.root / f"{prefix}paired_contrasts.csv", contrasts)
    shared.atomic_csv(args.root / f"{prefix}phase_contrasts.csv", phase_rows)
    shared.atomic_csv(
        args.root / f"{prefix}intervention_contrasts.csv", intervention_rows
    )
    shared.atomic_csv(args.root / f"{prefix}convergence.csv", convergence)
    shared.atomic_json(
        args.root / ("pilot_decision.json" if args.phase == "pilot" else "decision.json"),
        decision,
    )
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
