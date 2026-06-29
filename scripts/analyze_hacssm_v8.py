#!/usr/bin/env python3
"""Fail-closed analysis for the HACSSM-v8 SAS-PC adaptive-development study.

The nominated compact V8 model has no internal hierarchical auxiliary.  This
analyzer therefore rejects every ``hier_*`` epoch field from V8 runs while still
validating the historical V6/V7 anchors under their original objective contracts.
Raw PCA MSE is never pooled across tasks; cross-task summaries use paired relative
reductions with equal weight for every environment/optimizer-seed cell.

The bootstrap is deliberately deterministic and crossed: environments and seeds
are independently resampled, then their Cartesian product is averaged.  Its full
contract and canonical SHA-256 receipt are emitted in every decision record.
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_hacssm_v5 as shared


OCC_TO_CLEAN = shared.OCC_TO_CLEAN
DESIGNS = (
    "ssm",
    "hacssmv6",
    "hacssmv6_static",
    "hacssmv7_noaux",
    "hacssmv7_sharedaction",
    "hacssmv7_norecovery",
    "hacssmv8_dynamic",
    "hacssmv8_static",
    "hacssmv8_levelaction",
    "hacssmv8_redundant",
    "hacssmv8_noaction",
    "hacssmv8_single",
    "hacssmv8",
)
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
V6_DESIGNS = frozenset({"hacssmv6", "hacssmv6_static"})
V7_DESIGNS = frozenset(
    {"hacssmv7_noaux", "hacssmv7_sharedaction", "hacssmv7_norecovery"}
)
V8_DESIGNS = frozenset(design for design in DESIGNS if design.startswith("hacssmv8"))
HIER_DESIGNS = V6_DESIGNS | V7_DESIGNS
PRIMARY = "clean_mse_first_post"
EPOCHS = 200
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v8"
CANDIDATE = "hacssmv8"

ACTION_REFERENCE = "hacssmv8_levelaction"
ACTION_CANDIDATE = "hacssmv8_redundant"
REDUNDANT = "hacssmv8_redundant"
DYNAMIC_ENDPOINT = "hacssmv8_dynamic"
STATIC_ENDPOINT = "hacssmv8_static"
V7_LEADER = "hacssmv7_sharedaction"

# ``redundant`` is an equivalence/optimization receipt, not a distinct deployable
# architecture.  Including it in a strict envelope would make exact equivalence and
# an envelope win logically incompatible, so it is predeclared as envelope-excluded.
PERFORMANCE_ENVELOPE_DESIGNS = tuple(
    design for design in DESIGNS if design not in {CANDIDATE, REDUNDANT}
)

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
    "quantiles": {"method": "linear", "reported": [0.05, 0.025, 0.975, 0.95]},
}
BOOTSTRAP_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        BOOTSTRAP_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    """Configured weight, schedule, and whether objective gradients are active."""
    if design in V8_DESIGNS:
        return 0.0, "fixed", False
    if design in V6_DESIGNS:
        return 0.02, "v6_bootstrap", True
    if design in V7_DESIGNS:
        return 0.02, "v6_bootstrap", design != "hacssmv7_noaux"
    if design == "ssm":
        return 0.0, "fixed", False
    raise ValueError(f"unknown V8-study design {design!r}")


def scheduled_weight(base: float, schedule: str, epoch: int) -> float:
    if epoch < 1:
        raise ValueError(f"epoch must be positive, got {epoch}")
    if schedule == "fixed":
        return float(base)
    if schedule == "v6_bootstrap":
        if epoch <= 40:
            return float(base)
        if epoch <= 100:
            return float(base) * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - 40) / 60.0)
            )
        return 0.0
    raise ValueError(f"unknown hierarchy schedule {schedule!r}")


def validate_history(history: Any, design: str, run_dir: Path) -> None:
    """Validate complete histories and fail closed on any V8 auxiliary field."""
    if not isinstance(history, list) or len(history) != EPOCHS:
        length = len(history) if isinstance(history, list) else None
        raise ValueError(f"{run_dir}: history length {length}, expected {EPOCHS}")
    base, schedule, active = design_aux_contract(design)
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get("epoch") != epoch:
            raise ValueError(f"{run_dir}: malformed epoch {epoch}")
        if set(record) != {"epoch", "train", "val"}:
            raise ValueError(f"{run_dir}: unexpected history fields at epoch {epoch}")
        for split in ("train", "val"):
            values = record.get(split)
            if not isinstance(values, dict):
                raise ValueError(f"{run_dir}: missing {split} at epoch {epoch}")
            for metric in ("loss", "pred_loss", "sigreg_loss"):
                shared.finite(values.get(metric), f"{run_dir}:{epoch}:{split}.{metric}")
            shared.finite_tree(values, f"{run_dir}:{epoch}:{split}")

            hierarchical_fields = sorted(
                str(key) for key in values if str(key).startswith("hier_")
            )
            if design in V8_DESIGNS:
                if hierarchical_fields:
                    raise ValueError(
                        f"{run_dir}:{epoch}:{split} V8 must have no hierarchical "
                        f"auxiliary fields, got {hierarchical_fields}"
                    )
                continue

            if design in HIER_DESIGNS:
                for metric in ("hier_loss", "hier_loss_fast", "hier_loss_medium"):
                    shared.finite(
                        values.get(metric), f"{run_dir}:{epoch}:{split}.{metric}"
                    )
                observed = shared.finite(
                    values.get("hier_loss_weight"),
                    f"{run_dir}:{epoch}:{split}.hier_loss_weight",
                )
                wanted = scheduled_weight(base, schedule, epoch) if active else 0.0
                if not math.isclose(observed, wanted, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(
                        f"{run_dir}:{epoch}:{split} auxiliary weight {observed}, "
                        f"expected {wanted}"
                    )
            if design in V7_DESIGNS:
                for metric in ("hier_loss_bridge", "hier_loss_recovery", "hier_overlap"):
                    shared.finite(
                        values.get(metric), f"{run_dir}:{epoch}:{split}.{metric}"
                    )
                if float(values["hier_overlap"]) != 0.0:
                    raise ValueError(
                        f"{run_dir}:{epoch}:{split} V7 counterfactual windows overlap "
                        f"original hidden targets: {values['hier_overlap']}"
                    )


def configure_shared() -> None:
    """Point the generic V5 table loader at the exact V8 study contract."""
    shared.DESIGNS = DESIGNS
    shared.PILOT_SEEDS = PILOT_SEEDS
    shared.FINAL_SEEDS = FINAL_SEEDS
    shared.V5_DESIGNS = V8_DESIGNS
    shared.HIER_DESIGNS = HIER_DESIGNS
    shared.NO_AUX_DESIGNS = frozenset({"hacssmv7_noaux", *V8_DESIGNS})
    shared.PRIMARY = PRIMARY
    shared.EPOCHS = EPOCHS
    shared.WANDB_ENTITY = WANDB_ENTITY
    shared.WANDB_PROJECT = WANDB_PROJECT
    shared.WANDB_MODE = WANDB_MODE
    shared.WANDB_STUDY = WANDB_STUDY
    shared.design_aux_contract = design_aux_contract
    shared.scheduled_weight = scheduled_weight
    shared.validate_history = validate_history
    extras = (
        "val_hier_loss_bridge",
        "val_hier_loss_recovery",
        "val_hier_overlap",
        "rho_fast",
        "rho_medium",
        "action_head_shared_norm",
        "action_head_fast_norm",
        "action_head_medium_norm",
        "action_head_cosine",
    )
    shared.METRIC_FIELDS = tuple(dict.fromkeys((*shared.METRIC_FIELDS, *extras)))


def _lookup(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[tuple[str, str, int], float], tuple[int, ...]]:
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    expected = len(OCC_TO_CLEAN) * len(DESIGNS) * len(seeds)
    if len(rows) != expected:
        raise ValueError(f"contrast input has {len(rows)} rows, expected {expected}")
    lookup: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (str(row["env"]), str(row["design"]), int(row["seed"]))
        if key in lookup:
            raise ValueError(f"duplicate contrast cell {key}")
        value = shared.finite(row.get(PRIMARY), f"{key}.{PRIMARY}")
        if value <= 0.0:
            raise ValueError(f"{key}.{PRIMARY} must be positive, got {value}")
        lookup[key] = value
    expected_keys = {
        (env, design, seed)
        for env in OCC_TO_CLEAN
        for design in DESIGNS
        for seed in seeds
    }
    if set(lookup) != expected_keys:
        missing = sorted(expected_keys - set(lookup))
        extra = sorted(set(lookup) - expected_keys)
        raise ValueError(f"contrast grid mismatch: missing={missing[:3]}, extra={extra[:3]}")
    return lookup, seeds


def pairwise_reduction_matrix(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    if candidate == reference or candidate not in DESIGNS or reference not in DESIGNS:
        raise ValueError(f"invalid pairwise contrast {candidate!r} vs {reference!r}")
    lookup, seeds = _lookup(rows)
    matrix = np.empty((len(OCC_TO_CLEAN), len(seeds)), dtype=np.float64)
    for env_index, env in enumerate(OCC_TO_CLEAN):
        for seed_index, seed in enumerate(seeds):
            candidate_mse = lookup[(env, candidate, seed)]
            reference_mse = lookup[(env, reference, seed)]
            matrix[env_index, seed_index] = (
                reference_mse - candidate_mse
            ) / reference_mse
    if not np.isfinite(matrix).all():
        raise ValueError(f"non-finite pairwise contrast {candidate} vs {reference}")
    return matrix, seeds


def pairwise_summary(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    matrix, seeds = pairwise_reduction_matrix(rows, candidate, reference)
    lookup, _ = _lookup(rows)
    # As in the prospective V5--V7 analyzers, an environment win compares raw
    # within-environment means.  Only the cross-environment aggregate uses
    # equal-weight paired relative reductions.
    env_reductions = np.empty(len(OCC_TO_CLEAN), dtype=np.float64)
    env_wins = 0
    for index, env in enumerate(OCC_TO_CLEAN):
        candidate_mean = mean(lookup[(env, candidate, seed)] for seed in seeds)
        reference_mean = mean(lookup[(env, reference, seed)] for seed in seeds)
        env_reductions[index] = (reference_mean - candidate_mean) / reference_mean
        env_wins += candidate_mean < reference_mean
    return {
        "candidate": candidate,
        "reference": reference,
        "n_pairs": int(matrix.size),
        "mean_paired_relative_reduction": float(matrix.mean()),
        "paired_wins": int((matrix > 0.0).sum()),
        "paired_ties": int((matrix == 0.0).sum()),
        "environment_mean_wins": int(env_wins),
        "environment_mean_reductions": {
            env: float(env_reductions[index])
            for index, env in enumerate(OCC_TO_CLEAN)
        },
    }


def crossed_bootstrap_summary(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    matrix, seeds = pairwise_reduction_matrix(rows, candidate, reference)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    environment_indices = rng.integers(
        0, matrix.shape[0], size=(BOOTSTRAP_DRAWS, matrix.shape[0])
    )
    seed_indices = rng.integers(
        0, matrix.shape[1], size=(BOOTSTRAP_DRAWS, matrix.shape[1])
    )
    sampled = matrix[
        environment_indices[:, :, np.newaxis], seed_indices[:, np.newaxis, :]
    ].mean(axis=(1, 2))
    q025, q05, q95, q975 = np.quantile(
        sampled, (0.025, 0.05, 0.95, 0.975), method="linear"
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "n_environments": matrix.shape[0],
        "n_seeds": len(seeds),
        "point_mean_paired_relative_reduction": float(matrix.mean()),
        "environment_mean_reductions": {
            env: float(matrix[index].mean())
            for index, env in enumerate(OCC_TO_CLEAN)
        },
        "ci90": [float(q05), float(q95)],
        "ci95": [float(q025), float(q975)],
        "contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
    }


def endpoint_envelope_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup, seeds = _lookup(rows)
    reductions = np.empty((len(OCC_TO_CLEAN), len(seeds)), dtype=np.float64)
    candidate_env_mse: dict[str, float] = {}
    endpoint_env_mse: dict[str, float] = {}
    for env_index, env in enumerate(OCC_TO_CLEAN):
        candidate_values = []
        dynamic_values = []
        static_values = []
        for seed_index, seed in enumerate(seeds):
            candidate = lookup[(env, CANDIDATE, seed)]
            dynamic = lookup[(env, DYNAMIC_ENDPOINT, seed)]
            static = lookup[(env, STATIC_ENDPOINT, seed)]
            best = min(dynamic, static)
            reductions[env_index, seed_index] = (best - candidate) / best
            candidate_values.append(candidate)
            dynamic_values.append(dynamic)
            static_values.append(static)
        candidate_env_mse[env] = mean(candidate_values)
        endpoint_env_mse[env] = min(mean(dynamic_values), mean(static_values))
    return {
        "candidate": CANDIDATE,
        "references": [DYNAMIC_ENDPOINT, STATIC_ENDPOINT],
        "selection": "minimum retrained endpoint independently within each paired cell",
        "n_pairs": int(reductions.size),
        "mean_paired_relative_reduction": float(reductions.mean()),
        "paired_wins": int((reductions > 0.0).sum()),
        "paired_ties": int((reductions == 0.0).sum()),
        "environment_mean_wins": sum(
            candidate_env_mse[env] < endpoint_env_mse[env] for env in OCC_TO_CLEAN
        ),
        "environment_candidate_mse": candidate_env_mse,
        "environment_best_endpoint_mse": endpoint_env_mse,
    }


def contrast_map(contrasts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        design: shared.overall_contrast(contrasts, design)
        for design in DESIGNS
        if design != CANDIDATE
    }


def environment_wins(rows: Sequence[Mapping[str, Any]], reference: str) -> int:
    candidate = shared.environment_means(rows, CANDIDATE)
    baseline = shared.environment_means(rows, reference)
    return sum(candidate[env] < baseline[env] for env in OCC_TO_CLEAN)


def _equivalence_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair = pairwise_summary(rows, CANDIDATE, REDUNDANT)
    bootstrap = crossed_bootstrap_summary(rows, CANDIDATE, REDUNDANT)
    environment_effects = pair["environment_mean_reductions"]
    criteria = {
        "compact_redundant_abs_mean_le_0_25pct":
            abs(float(pair["mean_paired_relative_reduction"])) <= 0.0025,
        "compact_redundant_all_env_abs_le_1pct": all(
            abs(float(value)) <= 0.01 for value in environment_effects.values()
        ),
        "compact_redundant_ci90_inside_plusminus_1pct":
            float(bootstrap["ci90"][0]) >= -0.01
            and float(bootstrap["ci90"][1]) <= 0.01,
    }
    return {"pairwise": pair, "bootstrap": bootstrap, "criteria": criteria}


def _noninferiority_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair = pairwise_summary(rows, CANDIDATE, V7_LEADER)
    bootstrap = crossed_bootstrap_summary(rows, CANDIDATE, V7_LEADER)
    environment_effects = pair["environment_mean_reductions"]
    criteria = {
        "vs_v7_leader_point_gt_minus_0_5pct":
            float(pair["mean_paired_relative_reduction"]) > -0.005,
        "vs_v7_leader_envs_gt_minus_1pct_ge_4_of_5": sum(
            float(value) > -0.01 for value in environment_effects.values()
        ) >= 4,
        "vs_v7_leader_bootstrap95_lower_gt_minus_1pct":
            float(bootstrap["ci95"][0]) > -0.01,
    }
    return {"pairwise": pair, "bootstrap": bootstrap, "criteria": criteria}


def _convergence_observed(convergence: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not convergence:
        raise ValueError("empty convergence table")
    absolute = np.abs(np.asarray(
        [shared.finite(row.get("relative_improvement"), "convergence")
         for row in convergence], dtype=np.float64
    ))
    return {
        "median": float(np.median(absolute)),
        "p95": float(np.quantile(absolute, 0.95)),
        "max": float(absolute.max()),
    }


def _mechanism_observed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "action_sharing": pairwise_summary(rows, ACTION_CANDIDATE, ACTION_REFERENCE),
        "endpoint_envelope": endpoint_envelope_summary(rows),
        "compact_redundant_equivalence": _equivalence_observed(rows),
        "v7_leader_noninferiority": _noninferiority_observed(rows),
    }


def _pilot_mechanism_criteria(mechanism: Mapping[str, Any]) -> dict[str, bool]:
    action = mechanism["action_sharing"]
    endpoint = mechanism["endpoint_envelope"]
    return {
        "redundant_vs_levelaction_reduction_ge_0_5pct":
            float(action["mean_paired_relative_reduction"]) >= 0.005,
        "redundant_vs_levelaction_wins_ge_9_of_15": int(action["paired_wins"]) >= 9,
        "redundant_vs_levelaction_env_wins_ge_3_of_5":
            int(action["environment_mean_wins"]) >= 3,
        "learned_vs_endpoint_envelope_reduction_ge_0_5pct":
            float(endpoint["mean_paired_relative_reduction"]) >= 0.005,
        "learned_vs_endpoint_envelope_wins_ge_9_of_15":
            int(endpoint["paired_wins"]) >= 9,
        "learned_vs_endpoint_envelope_env_wins_ge_3_of_5":
            int(endpoint["environment_mean_wins"]) >= 3,
        **mechanism["compact_redundant_equivalence"]["criteria"],
    }


def _final_mechanism_criteria(mechanism: Mapping[str, Any]) -> dict[str, bool]:
    action = mechanism["action_sharing"]
    endpoint = mechanism["endpoint_envelope"]
    return {
        "redundant_vs_levelaction_reduction_ge_0_5pct":
            float(action["mean_paired_relative_reduction"]) >= 0.005,
        "redundant_vs_levelaction_wins_ge_15_of_25": int(action["paired_wins"]) >= 15,
        "redundant_vs_levelaction_env_wins_ge_3_of_5":
            int(action["environment_mean_wins"]) >= 3,
        "learned_vs_endpoint_envelope_reduction_ge_1pct":
            float(endpoint["mean_paired_relative_reduction"]) >= 0.01,
        "learned_vs_endpoint_envelope_wins_ge_15_of_25":
            int(endpoint["paired_wins"]) >= 15,
        "learned_vs_endpoint_envelope_env_wins_ge_3_of_5":
            int(endpoint["environment_mean_wins"]) >= 3,
        **mechanism["compact_redundant_equivalence"]["criteria"],
    }


def pilot_decision(rows, convergence, contrasts) -> dict[str, Any]:
    compared = contrast_map(contrasts)
    env_wins = {design: environment_wins(rows, design) for design in compared}
    mechanism = _mechanism_observed(rows)
    convergence_observed = _convergence_observed(convergence)
    criteria = {
        "vs_ssm_reduction_ge_6pct":
            float(compared["ssm"]["mean_paired_relative_reduction"]) >= 0.06,
        "vs_ssm_wins_ge_10_of_15": int(compared["ssm"]["paired_wins"]) >= 10,
        "vs_ssm_env_wins_ge_4_of_5": env_wins["ssm"] >= 4,
        "vs_v7_shared_reduction_ge_0_5pct":
            float(compared[V7_LEADER]["mean_paired_relative_reduction"]) >= 0.005,
        "vs_v7_shared_wins_ge_9_of_15": int(compared[V7_LEADER]["paired_wins"]) >= 9,
        "vs_v7_shared_env_wins_ge_3_of_5": env_wins[V7_LEADER] >= 3,
        "vs_v6_static_positive":
            float(compared["hacssmv6_static"]["mean_paired_relative_reduction"]) > 0.0,
        "vs_v6_static_wins_ge_8_of_15":
            int(compared["hacssmv6_static"]["paired_wins"]) >= 8,
        "vs_v6_static_env_wins_ge_3_of_5": env_wins["hacssmv6_static"] >= 3,
        "vs_full_v6_reduction_ge_1pct":
            float(compared["hacssmv6"]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_full_v6_wins_ge_9_of_15": int(compared["hacssmv6"]["paired_wins"]) >= 9,
        "vs_full_v6_env_wins_ge_3_of_5": env_wins["hacssmv6"] >= 3,
        "vs_v7_norecovery_positive":
            float(compared["hacssmv7_norecovery"]["mean_paired_relative_reduction"]) > 0.0,
        "vs_v7_norecovery_wins_ge_8_of_15":
            int(compared["hacssmv7_norecovery"]["paired_wins"]) >= 8,
        "vs_v7_norecovery_env_wins_ge_3_of_5": env_wins["hacssmv7_norecovery"] >= 3,
        **_pilot_mechanism_criteria(mechanism),
    }
    for reference in (DYNAMIC_ENDPOINT, STATIC_ENDPOINT):
        label = reference.removeprefix("hacssmv8_")
        criteria.update({
            f"vs_{label}_positive":
                float(compared[reference]["mean_paired_relative_reduction"]) > 0.0,
            f"vs_{label}_wins_ge_8_of_15": int(compared[reference]["paired_wins"]) >= 8,
            f"vs_{label}_env_wins_ge_3_of_5": env_wins[reference] >= 3,
        })
    for reference in ("hacssmv8_noaction", "hacssmv8_single"):
        label = reference.removeprefix("hacssmv8_")
        criteria.update({
            f"vs_{label}_reduction_ge_3pct":
                float(compared[reference]["mean_paired_relative_reduction"]) >= 0.03,
            f"vs_{label}_wins_ge_11_of_15": int(compared[reference]["paired_wins"]) >= 11,
            f"vs_{label}_env_wins_ge_3_of_5": env_wins[reference] >= 3,
        })
    criteria.update({
        "convergence_absolute_median_lt_1pct": convergence_observed["median"] < 0.01,
        "convergence_absolute_p95_lt_3pct": convergence_observed["p95"] < 0.03,
        "convergence_absolute_max_lt_5pct": convergence_observed["max"] < 0.05,
    })

    noninferiority_criteria = {
        **mechanism["v7_leader_noninferiority"]["criteria"],
        "noninferiority_vs_ssm_reduction_ge_6pct":
            float(compared["ssm"]["mean_paired_relative_reduction"]) >= 0.06,
        **_pilot_mechanism_criteria(mechanism),
        "noninferiority_convergence_median_lt_1pct": convergence_observed["median"] < 0.01,
        "noninferiority_convergence_p95_lt_3pct": convergence_observed["p95"] < 0.03,
        "noninferiority_convergence_max_lt_5pct": convergence_observed["max"] < 0.05,
    }
    strict_passed = all(criteria.values())
    noninferiority_passed = all(noninferiority_criteria.values())
    # Noninferiority is retained as a diagnostic pilot receipt, but it cannot
    # reopen a failed strict pilot.  The execution harness treats the strict
    # pilot boolean as immutable before running the mandatory completion seeds.
    decision = "PILOT_OVERALL_BEST_PASS" if strict_passed else "NO_GO"
    return {
        "schema_version": 1,
        "phase": "pilot",
        "decision": decision,
        "pilot_screen_passed": strict_passed,
        "pilot_noninferiority_screen_passed": noninferiority_passed,
        "criteria": criteria,
        "noninferiority_criteria": noninferiority_criteria,
        "observed": {
            "overall_contrasts": compared,
            "environment_mean_wins": env_wins,
            "mechanism": mechanism,
            "convergence_absolute": convergence_observed,
            "bootstrap_contract": BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        },
        "adaptive_development_only": True,
        "note": (
            "Immutable adaptive-development pilot. All five seeds run regardless. "
            "Neither pilot label is an untouched-test or publication claim."
        ),
    }


def final_summary(
    rows,
    convergence,
    contrasts,
    *,
    pilot_screen_passed: bool,
    pilot_noninferiority_screen_passed: bool = False,
) -> dict[str, Any]:
    compared = contrast_map(contrasts)
    env_wins = {design: environment_wins(rows, design) for design in compared}
    mechanism = _mechanism_observed(rows)
    convergence_observed = _convergence_observed(convergence)
    candidate_means = shared.environment_means(rows, CANDIDATE)
    design_means = {design: shared.environment_means(rows, design) for design in DESIGNS}
    hold = shared.environment_means(rows, CANDIDATE, "last_visible_mse_first_post")
    envelope = sum(
        candidate_means[env] < min(
            design_means[design][env] for design in PERFORMANCE_ENVELOPE_DESIGNS
        )
        for env in OCC_TO_CLEAN
    )
    hold_wins = sum(candidate_means[env] < hold[env] for env in OCC_TO_CLEAN)

    criteria = {
        "vs_ssm_reduction_ge_7pct":
            float(compared["ssm"]["mean_paired_relative_reduction"]) >= 0.07,
        "vs_ssm_wins_ge_20_of_25": int(compared["ssm"]["paired_wins"]) >= 20,
        "vs_ssm_env_wins_ge_4_of_5": env_wins["ssm"] >= 4,
        "vs_v7_shared_reduction_ge_1pct":
            float(compared[V7_LEADER]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_v7_shared_wins_ge_15_of_25": int(compared[V7_LEADER]["paired_wins"]) >= 15,
        "vs_v7_shared_env_wins_ge_3_of_5": env_wins[V7_LEADER] >= 3,
        "vs_v6_static_reduction_ge_1pct":
            float(compared["hacssmv6_static"]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_v6_static_wins_ge_15_of_25":
            int(compared["hacssmv6_static"]["paired_wins"]) >= 15,
        "vs_v6_static_env_wins_ge_3_of_5": env_wins["hacssmv6_static"] >= 3,
        "vs_full_v6_reduction_ge_1pct":
            float(compared["hacssmv6"]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_full_v6_wins_ge_15_of_25": int(compared["hacssmv6"]["paired_wins"]) >= 15,
        "vs_full_v6_env_wins_ge_3_of_5": env_wins["hacssmv6"] >= 3,
        "vs_v7_norecovery_positive":
            float(compared["hacssmv7_norecovery"]["mean_paired_relative_reduction"]) > 0.0,
        "vs_v7_norecovery_wins_ge_13_of_25":
            int(compared["hacssmv7_norecovery"]["paired_wins"]) >= 13,
        "vs_v7_norecovery_env_wins_ge_3_of_5": env_wins["hacssmv7_norecovery"] >= 3,
        **_final_mechanism_criteria(mechanism),
        "performance_envelope_wins_ge_3_of_5": envelope >= 3,
        "beats_hold_ge_4_of_5": hold_wins >= 4,
    }
    for reference in (DYNAMIC_ENDPOINT, STATIC_ENDPOINT):
        label = reference.removeprefix("hacssmv8_")
        criteria.update({
            f"vs_{label}_positive":
                float(compared[reference]["mean_paired_relative_reduction"]) > 0.0,
            f"vs_{label}_wins_ge_13_of_25": int(compared[reference]["paired_wins"]) >= 13,
            f"vs_{label}_env_wins_ge_3_of_5": env_wins[reference] >= 3,
        })
    for reference in ("hacssmv8_noaction", "hacssmv8_single"):
        label = reference.removeprefix("hacssmv8_")
        criteria.update({
            f"vs_{label}_reduction_ge_3pct":
                float(compared[reference]["mean_paired_relative_reduction"]) >= 0.03,
            f"vs_{label}_wins_ge_17_of_25": int(compared[reference]["paired_wins"]) >= 17,
            f"vs_{label}_env_wins_ge_3_of_5": env_wins[reference] >= 3,
        })
    criteria.update({
        "convergence_absolute_median_lt_1pct": convergence_observed["median"] < 0.01,
        "convergence_absolute_p95_lt_3pct": convergence_observed["p95"] < 0.03,
        "convergence_absolute_max_lt_5pct": convergence_observed["max"] < 0.05,
    })

    noninferiority_criteria = {
        **mechanism["v7_leader_noninferiority"]["criteria"],
        "noninferiority_vs_ssm_reduction_ge_6pct":
            float(compared["ssm"]["mean_paired_relative_reduction"]) >= 0.06,
        **_final_mechanism_criteria(mechanism),
        "noninferiority_convergence_median_lt_1pct": convergence_observed["median"] < 0.01,
        "noninferiority_convergence_p95_lt_3pct": convergence_observed["p95"] < 0.03,
        "noninferiority_convergence_max_lt_5pct": convergence_observed["max"] < 0.05,
    }
    strict_final = all(criteria.values())
    noninferiority_final = all(noninferiority_criteria.values())
    overall_best = bool(pilot_screen_passed and strict_final)
    compact_noninferior = bool(
        pilot_screen_passed and noninferiority_final and not overall_best
    )
    if overall_best:
        decision = "OVERALL_BEST_ADAPTIVE_DEV"
    elif compact_noninferior:
        decision = "COMPACT_NONINFERIOR_ADAPTIVE_DEV"
    elif not pilot_screen_passed:
        decision = "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    else:
        decision = "NO_GO"

    return {
        "schema_version": 1,
        "phase": "final",
        "decision": decision,
        "pilot_screen_passed": bool(pilot_screen_passed),
        "pilot_noninferiority_screen_passed": bool(
            pilot_noninferiority_screen_passed
        ),
        "good_enough_for_overall_best_claim": overall_best,
        "good_enough_for_compact_noninferiority_claim": compact_noninferior,
        "best_in_focused_locked_grid": overall_best,
        "compact_noninferior_to_v7_leader": compact_noninferior,
        "criteria": criteria,
        "noninferiority_criteria": noninferiority_criteria,
        "completed_runs": len(rows),
        "observed": {
            "overall_contrasts": compared,
            "environment_mean_wins": env_wins,
            "mechanism": mechanism,
            "performance_envelope_designs": list(PERFORMANCE_ENVELOPE_DESIGNS),
            "performance_envelope_env_wins": envelope,
            "hold_environment_wins": hold_wins,
            "convergence_absolute": convergence_observed,
            "bootstrap_contract": BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        },
        "adaptive_development_only": True,
        "scope": "adaptive_development_only",
        "limitations": [
            "V8 was selected after inspecting V1-V7 on the same five tasks, corruption, and seed-7777 trajectories.",
            "Optimizer seeds do not make the task/corruption/trajectory grid untouched.",
            "No simulator-state outcome, executed-control return, or tuned contemporary baseline is measured.",
            "The redundant control is excluded from the performance envelope because it is an equivalence receipt.",
        ],
        "note": (
            "All labels are deterministic adaptive-development labels only. Even "
            "OVERALL_BEST_ADAPTIVE_DEV is not an untouched-test or ICLR claim."
        ),
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/hacssm_v8_shared"))
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    return parser.parse_args(argv)


def strict_validate_cells(root: Path, seeds: Sequence[int]) -> None:
    """Reuse the V8 execution harness's complete local artifact contract."""
    import scripts.run_hacssm_v8 as runner

    original_root = runner.OUTPUT_ROOT
    try:
        runner.OUTPUT_ROOT = root.resolve()
        runner.configure_shared()
        jobs = tuple(
            runner.shared.Job(
                "pilot" if seed in PILOT_SEEDS else "completion",
                seed,
                env,
                OCC_TO_CLEAN[env],
                design,
            )
            for seed in seeds
            for env in OCC_TO_CLEAN
            for design in DESIGNS
        )
        for job in jobs:
            runner.shared.validate_job(job, allow_missing=False)
    finally:
        runner.OUTPUT_ROOT = original_root


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = PILOT_SEEDS if args.phase == "pilot" else FINAL_SEEDS
    strict_validate_cells(args.root, seeds)
    configure_shared()
    expected = len(OCC_TO_CLEAN) * len(DESIGNS) * len(seeds)
    rows, convergence = shared.load_cells(args.root, seeds)
    if len(rows) != expected:
        raise ValueError(f"{args.phase} grid has {len(rows)} rows, expected {expected}")
    grouped = shared.grouped_rows(rows)
    contrasts = shared.contrast_rows(rows, candidate=CANDIDATE)
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
        recomputed_pilot = pilot_decision(
            pilot_rows,
            pilot_convergence,
            shared.contrast_rows(pilot_rows, candidate=CANDIDATE),
        )
        if pilot != recomputed_pilot:
            raise ValueError(f"invalid immutable pilot decision: {pilot_path}")
        decision = final_summary(
            rows,
            convergence,
            contrasts,
            pilot_screen_passed=recomputed_pilot["pilot_screen_passed"],
            pilot_noninferiority_screen_passed=recomputed_pilot[
                "pilot_noninferiority_screen_passed"
            ],
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
