#!/usr/bin/env python3
"""Validate and analyze the excluded 28-cell SIRO-v12 four-task screen."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v11_data import sha256_file
from scripts.run_siro_v12_screen import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STUDY,
    SEED,
    TASKS,
    run_directory,
)
from scripts.train_siro_v12 import (
    DESIGNS,
    FLOAT32_STABILITY_CAP,
    SIRO_DESIGNS,
    V11_COMPARATOR_RANKING,
)


PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
CANDIDATE = "sirov12"
MECHANISM_CONTROLS = (
    "sirov12_spectralshrink",
    "sirov12_identityA",
    "sirov12_identityK",
    "sirov12_noaction",
    "sirov12_noanchor",
)


class IntegrityError(RuntimeError):
    pass


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{label} is not finite: {result!r}")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"missing {path}")
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise IntegrityError(f"{path} must contain one JSON object")
    return value


def _validate_common(
        root: Path, task: str, design: str, seed: int,
        epochs: int, study: str) -> dict[str, Any]:
    directory = run_directory(root, task, design)
    metrics = _load_json(directory / "metrics.json")
    expected_env = f"dmc:{task}"
    expected = {
        "env": expected_env,
        "design": design,
        "seed": seed,
        "epochs": epochs,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise IntegrityError(
                f"{task}/{design}: metrics {key}={metrics.get(key)!r}, expected {value!r}")
    for key in (
            PRIMARY, CLEAN, "initial_encoder_integrator_probe_nmse",
            "loss_convergence_relative_change", "eval_rollout_episode"):
        _finite(metrics.get(key), f"{task}/{design}:{key}")
    rollout_path = directory / "eval_rollout.npz"
    if not rollout_path.is_file():
        raise IntegrityError(f"{task}/{design}: missing rollout NPZ")
    rollout_hash = sha256_file(rollout_path)
    if metrics.get("eval_rollout_sha256") != rollout_hash:
        raise IntegrityError(f"{task}/{design}: rollout hash mismatch")
    wandb = _load_json(directory / "wandb_run.json")
    expected_wandb = {
        "state": "finished",
        "mode": "online",
        "study": study,
        "eval_rollout_sha256": rollout_hash,
    }
    for key, value in expected_wandb.items():
        if wandb.get(key) != value:
            raise IntegrityError(
                f"{task}/{design}: W&B {key}={wandb.get(key)!r}, expected {value!r}")
    if not wandb.get("run_id") or not wandb.get("url"):
        raise IntegrityError(f"{task}/{design}: incomplete W&B identity")
    if design == "kdiov11":
        if metrics.get("development_action_ranking") != V11_COMPARATOR_RANKING:
            raise IntegrityError(
                f"{task}/{design}: comparator did not use {V11_COMPARATOR_RANKING}")
    return {
        "task": task,
        "design": design,
        "metrics": metrics,
        "wandb": wandb,
        "directory": str(directory),
    }


def _validate_siro(row: Mapping[str, Any], epochs: int) -> None:
    task, design, metrics = row["task"], row["design"], row["metrics"]
    label = f"{task}/{design}"
    exact = {
        "fit_updates": epochs + 1,
        "siro_fit_fit_index": epochs,
        "siro_fit_fit_finite": True,
        "siro_fit_fit_transition_samples": 1_200 * 47,
        "siro_fit_fit_episodes": 1_200,
        "siro_fit_fit_length": 48,
        "siro_fit_reachability_lags": 47,
        "siro_fit_survival_weight_first": 1.0,
        "siro_fit_survival_weight_last": 1 / 47,
        "siro_fit_innovation_samples": 1_200 * 47,
        "identified_operator_fit": True,
        "fit_gradient_active": False,
        "memory_specific_loss_weight": 0.0,
        "memory_recurrent_floats": 384.0,
        "siro_fit_anchor_centered_fit": design != "sirov12_noanchor",
    }
    for key, value in exact.items():
        actual = metrics.get(key)
        if isinstance(value, float):
            if abs(_finite(actual, f"{label}:{key}") - value) > 1e-12:
                raise IntegrityError(f"{label}:{key}={actual!r}, expected {value!r}")
        elif actual != value:
            raise IntegrityError(f"{label}:{key}={actual!r}, expected {value!r}")
    for key in (
            "siro_fit_identified_A_singular_max",
            "siro_fit_identified_A_singular_min",
            "siro_fit_action_B_singular_max",
            "siro_fit_action_B_singular_min",
            "siro_fit_action_covariance_condition",
            "siro_fit_residual_covariance_condition",
            "siro_fit_signal_trace",
            "siro_fit_noise_trace",
            "siro_fit_reachability_trace",
            "siro_fit_age_tau_min",
            "siro_fit_age_tau_max",
            "siro_fit_lmmse_K_norm",
            "siro_fit_operator_A_relative_frobenius_delta",
            "siro_fit_operator_B_relative_frobenius_delta",
            "siro_fit_operator_K_relative_frobenius_delta",
            "siro_fit_operator_R_relative_frobenius_delta",
            "siro_fit_pre_refit_clean_prior_mse",
            "siro_fit_post_refit_clean_prior_mse",
            "siro_fit_pre_post_refit_clean_prior_shift_mse",
            "siro_fit_pre_post_refit_clean_prior_relative_shift",
            "siro_fit_parity_B_relative_disagreement",
            "siro_fit_parity_B_cosine_alignment",
            "siro_fit_cross_signal_to_full_reachability_trace_ratio",
            "siro_fit_initial_anchor_max_abs_mismatch",
            "siro_fit_centered_x0_max_abs",
            "siro_fit_anchor_mean_channel_variance",
            "siro_fit_anchor_covariance_effective_rank",
            "siro_fit_normalized_residual_anchor_cross_covariance",
            "siro_streaming_max_abs",
            "siro_anchor_invariance_max_abs",
            "siro_action_effect_rms",
            "siro_true_action_suffix_advantage",
            "encoder_mean_channel_variance",
            "encoder_covariance_effective_rank",
            "encoder_singleton_max_abs",
            "encoder_prefix_max_abs"):
        _finite(metrics.get(key), f"{label}:{key}")
    if int(metrics.get("siro_fit_action_B_rank", -1)) != int(metrics["action_dim"]):
        raise IntegrityError(f"{label}: fitted action map is rank deficient")
    if _finite(metrics["siro_streaming_max_abs"], f"{label}:streaming") > 1e-5:
        raise IntegrityError(f"{label}: streaming discrepancy exceeds 1e-5")
    if _finite(
            metrics["siro_anchor_invariance_max_abs"],
            f"{label}:anchor invariance") > 1e-7:
        raise IntegrityError(f"{label}: anchor state changed during streaming")
    if _finite(
            metrics["siro_fit_initial_anchor_max_abs_mismatch"],
            f"{label}:initial anchor mismatch") > 1e-5:
        raise IntegrityError(f"{label}: paired clean/observed initial anchors differ")
    fit_x0 = _finite(
        metrics["siro_fit_centered_x0_max_abs"], f"{label}:fit x0")
    if design == "sirov12_noanchor":
        if fit_x0 <= 0.0:
            raise IntegrityError(f"{label}: noanchor did not fit absolute coordinates")
    elif fit_x0 != 0.0:
        raise IntegrityError(f"{label}: anchor-centered fit did not have exact x0=0")
    if _finite(
            metrics["siro_fit_anchor_mean_channel_variance"],
            f"{label}:fit anchor variance") < 1e-5:
        raise IntegrityError(f"{label}: fit anchors collapsed")
    if _finite(
            metrics["siro_fit_anchor_covariance_effective_rank"],
            f"{label}:fit anchor rank") < 16:
        raise IntegrityError(f"{label}: fit anchor effective rank below 16")
    normalized_cross = _finite(
        metrics["siro_fit_normalized_residual_anchor_cross_covariance"],
        f"{label}:normalized residual-anchor cross covariance")
    if not 0.0 <= normalized_cross <= 1.0 + 1e-6:
        raise IntegrityError(
            f"{label}: normalized residual-anchor cross covariance is out of range")
    if _finite(metrics["encoder_mean_channel_variance"], f"{label}:variance") < 1e-5:
        raise IntegrityError(f"{label}: encoder variance collapsed")
    if _finite(metrics["encoder_covariance_effective_rank"], f"{label}:rank") < 16:
        raise IntegrityError(f"{label}: encoder effective rank below 16")
    if max(
            abs(_finite(metrics["encoder_singleton_max_abs"], f"{label}:singleton")),
            abs(_finite(metrics["encoder_prefix_max_abs"], f"{label}:prefix"))) > 1e-5:
        raise IntegrityError(f"{label}: encoder causality receipt failed")

    identity_A = design == "sirov12_identityA"
    if bool(metrics.get("siro_fit_effective_A_identity")) != identity_A:
        raise IntegrityError(f"{label}: identity-A mode receipt mismatch")
    singular_max = _finite(
        metrics["siro_fit_identified_A_singular_max"], f"{label}:A singular")
    singular_min = _finite(
        metrics["siro_fit_identified_A_singular_min"], f"{label}:A singular")
    if identity_A:
        if abs(singular_max - 1.0) > 1e-6 or abs(singular_min - 1.0) > 1e-6:
            raise IntegrityError(f"{label}: identityA did not retain exact unit singulars")
    elif singular_max > FLOAT32_STABILITY_CAP + 2e-6:
        raise IntegrityError(f"{label}: identified A exceeds its frozen singular cap")

    expected_flags = {
        "siro_fit_effective_action_zero": design == "sirov12_noaction",
        "siro_fit_effective_K_identity": design == "sirov12_identityK",
        "siro_fit_effective_R_identity": design != "sirov12_spectralshrink",
    }
    for key, expected in expected_flags.items():
        if bool(metrics.get(key)) != expected:
            raise IntegrityError(f"{label}: {key} mode receipt mismatch")
    if design == "sirov12_noaction":
        for key in (
                "siro_action_effect_rms", "siro_true_action_one_step_advantage",
                "siro_true_action_suffix_advantage",
                "siro_action_rollout_divergence_h1",
                "siro_action_rollout_divergence_h47"):
            if _finite(metrics.get(key), f"{label}:{key}") != 0.0:
                raise IntegrityError(f"{label}:{key} must be exact zero")


def load_rows(root: Path, seed: int, epochs: int, study: str):
    rows = []
    errors = []
    for task in TASKS:
        for design in DESIGNS:
            try:
                row = _validate_common(root, task, design, seed, epochs, study)
                if design in SIRO_DESIGNS:
                    _validate_siro(row, epochs)
                rows.append(row)
            except (IntegrityError, OSError, ValueError) as exc:
                errors.append(str(exc))
    return rows, errors


def _design_values(rows, design: str, metric: str) -> np.ndarray:
    mapping = {
        row["task"]: _finite(row["metrics"].get(metric), f"{row['task']}/{design}:{metric}")
        for row in rows if row["design"] == design}
    if tuple(sorted(mapping)) != tuple(sorted(TASKS)):
        raise IntegrityError(f"{design}/{metric}: incomplete task grid")
    return np.asarray([mapping[task] for task in TASKS], dtype=np.float64)


def contrast(rows, reference: str, metric: str = PRIMARY) -> dict[str, Any]:
    candidate = _design_values(rows, CANDIDATE, metric)
    baseline = _design_values(rows, reference, metric)
    improvement = (baseline - candidate) / baseline
    return {
        "reference": reference,
        "metric": metric,
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(baseline.mean()),
        "equal_task_reduction": float((baseline.mean() - candidate.mean()) / baseline.mean()),
        "paired_reduction_mean": float(improvement.mean()),
        "wins": int((candidate < baseline).sum()),
        "task_reductions": {
            task: float(value) for task, value in zip(TASKS, improvement, strict=True)},
    }


def _integrator_contrast(rows) -> dict[str, Any]:
    full_rows = {row["task"]: row for row in rows if row["design"] == CANDIDATE}
    candidate = np.asarray([
        _finite(full_rows[task]["metrics"][PRIMARY], f"{task}:candidate")
        for task in TASKS])
    integrator = np.asarray([
        _finite(
            full_rows[task]["metrics"]["initial_encoder_integrator_probe_nmse"],
            f"{task}:integrator")
        for task in TASKS])
    return {
        "reference": "candidate_checkpoint_initial_encoder_integrator",
        "metric": PRIMARY,
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(integrator.mean()),
        "equal_task_reduction": float(
            (integrator.mean() - candidate.mean()) / integrator.mean()),
        "wins": int((candidate < integrator).sum()),
        "task_reductions": {
            task: float((base - value) / base)
            for task, value, base in zip(TASKS, candidate, integrator, strict=True)},
    }


def _action_gate(rows) -> dict[str, Any]:
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    required = ("fish.swim", "walker.walk")
    key = "siro_true_action_suffix_advantage"
    available = all(task in full and key in full[task] for task in required)
    values = {
        task: (_finite(full[task][key], f"{task}:{key}") if available else None)
        for task in required}
    return {
        "available": available,
        "values": values,
        "passed": (all(value > 0.0 for value in values.values()) if available else None),
    }


def analyze(rows, errors, epochs: int, study: str) -> dict[str, Any]:
    complete = len(rows) == len(TASKS) * len(DESIGNS) and not errors
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v12_screen_after_failed_v11",
        "study": study,
        "seed": SEED,
        "epochs": epochs,
        "expected_cells": len(TASKS) * len(DESIGNS),
        "completed_cells": len(rows),
        "integrity_passed": complete,
        "integrity_errors": errors,
        "official_result": False,
        "iclr_confirmation": False,
    }
    if not complete:
        result.update({
            "status": "INCOMPLETE_OR_INVALID",
            "continue_to_100_epochs": False,
            "scientific_gate_passed": False,
        })
        return result

    contrasts = {
        reference: contrast(rows, reference)
        for reference in MECHANISM_CONTROLS + ("kdiov11",)}
    integrator = _integrator_contrast(rows)
    action = _action_gate(rows)
    candidate_values = _design_values(rows, CANDIDATE, PRIMARY)
    v11_values = _design_values(rows, "kdiov11", PRIMARY)
    legal_values = np.asarray([
        next(row for row in rows
             if row["task"] == task and row["design"] == CANDIDATE)["metrics"][
                 "initial_encoder_integrator_probe_nmse"]
        for task in TASKS], dtype=np.float64)
    mechanism = {
        "fit_streaming_integrity": True,
        "nonzero_action_effect_all_tasks": all(
            _finite(
                next(row for row in rows
                     if row["task"] == task and row["design"] == CANDIDATE)["metrics"][
                         "siro_action_effect_rms"], f"{task}:action effect") > 0.0
            for task in TASKS),
        "within_15pct_of_better_v11_or_integrator": float(candidate_values.mean()) <= (
            1.15 * min(float(v11_values.mean()), float(legal_values.mean()))),
        "full_vs_noanchor_2pct": (
            contrasts["sirov12_noanchor"]["equal_task_reduction"] >= .02),
        "fish_walker_action_advantage": action,
    }
    action_continuation = action["passed"] if action["available"] else True
    continue_100 = bool(
        mechanism["fit_streaming_integrity"]
        and mechanism["nonzero_action_effect_all_tasks"]
        and mechanism["within_15pct_of_better_v11_or_integrator"]
        and mechanism["full_vs_noanchor_2pct"]
        and action_continuation)

    late = np.asarray([
        abs(_finite(row["metrics"]["loss_convergence_relative_change"], "late change"))
        for row in rows], dtype=np.float64)
    full_late = np.asarray([
        abs(_finite(row["metrics"]["loss_convergence_relative_change"], "full late"))
        for row in rows if row["design"] == CANDIDATE], dtype=np.float64)
    scientific = {
        "vs_v11_5pct_3wins": (
            contrasts["kdiov11"]["equal_task_reduction"] >= .05
            and contrasts["kdiov11"]["wins"] >= 3),
        "vs_integrator_5pct_3wins": (
            integrator["equal_task_reduction"] >= .05 and integrator["wins"] >= 3),
        "vs_noaction_5pct": contrasts["sirov12_noaction"]["equal_task_reduction"] >= .05,
        "vs_identityA_2pct": contrasts["sirov12_identityA"]["equal_task_reduction"] >= .02,
        "vs_identityK_2pct": contrasts["sirov12_identityK"]["equal_task_reduction"] >= .02,
        "vs_noanchor_2pct": contrasts["sirov12_noanchor"]["equal_task_reduction"] >= .02,
        "no_worse_than_spectralshrink": (
            contrasts["sirov12_spectralshrink"]["equal_task_reduction"] >= 0.0),
        "fish_walker_action_advantage": action_continuation,
        "late_median_below_3pct": float(np.median(late)) < .03,
        "full_late_max_below_3pct": float(full_late.max()) < .03,
        "integrity": True,
    }
    scientific_pass = bool(epochs >= 100 and all(scientific.values()))
    result.update({
        "status": (
            "SCIENTIFIC_GATE_PASS" if scientific_pass else
            "MECHANICS_GATE_PASS_100E_NOT_LAUNCHED" if continue_100 and epochs < 100 else
            "SCIENTIFIC_GATE_FAIL" if epochs >= 100 else
            "MECHANICS_GATE_FAIL"),
        "design_means": {
            design: {
                PRIMARY: float(_design_values(rows, design, PRIMARY).mean()),
                CLEAN: float(_design_values(rows, design, CLEAN).mean()),
            } for design in DESIGNS},
        "contrasts": contrasts,
        "integrator_contrast": integrator,
        "mechanics_gate": mechanism,
        "continue_to_100_epochs": continue_100 if epochs < 100 else False,
        "automatic_100_epoch_launch_performed": False,
        "scientific_gate": scientific,
        "scientific_gate_passed": scientific_pass,
        "late_change": {
            "all_cell_median_abs": float(np.median(late)),
            "all_cell_max_abs": float(late.max()),
            "candidate_max_abs": float(full_late.max()),
        },
        "wandb_runs": [{
            "task": row["task"],
            "design": row["design"],
            "run_id": row["wandb"]["run_id"],
            "url": row["wandb"]["url"],
        } for row in rows],
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.root = args.root if args.root.is_absolute() else (ROOT / args.root).resolve()
    if args.seed != SEED:
        raise ValueError(f"SIRO screen seed is frozen at {SEED}")
    rows, errors = load_rows(args.root, args.seed, args.epochs, args.study)
    analysis = analyze(rows, errors, args.epochs, args.study)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    if args.write:
        for filename, payload in (
                ("screen_analysis.json", analysis),
                ("screen_decision.json", {
                    "status": analysis["status"],
                    "integrity_passed": analysis["integrity_passed"],
                    "continue_to_100_epochs": analysis["continue_to_100_epochs"],
                    "automatic_launch_performed": False,
                    "scientific_gate_passed": analysis["scientific_gate_passed"],
                })):
            path = args.root / filename
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            with path.open("x") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
    if not analysis["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
