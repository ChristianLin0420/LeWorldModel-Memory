#!/usr/bin/env python3
"""Validate and analyze the frozen 36-cell CF-HIRO-v13 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v11_data import sha256_file
from scripts.run_cf_hiro_v13_screen import (
    BLAS_THREADS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STUDY,
    SEED,
    SOURCE_PATHS,
    TASKS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    run_directory,
)
from scripts.train_cf_hiro_v13 import (
    CF_HIRO_DESIGNS,
    CORE_MODES,
    DESIGNS,
    V11_COMPARATOR_RANKING,
)


PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
CANDIDATE = "cfhirov13"
EXTERNAL_REFERENCES = ("ssm", "hacssmv8", "kdiov11")
INTERNAL_CONTROLS = (
    "cfhirov13_fullanchor",
    "cfhirov13_triangular",
    "cfhirov13_noshrink",
    "cfhirov13_noaction",
    "cfhirov13_nocorrect",
)
INTERNAL_THRESHOLDS = {
    design: (.05 if design == "cfhirov13_noaction" else .02)
    for design in INTERNAL_CONTROLS}
EXPECTED_CELLS = len(TASKS) * len(DESIGNS)
FLOAT32_STABILITY_BOUNDARY = 1.0 - math.sqrt(np.finfo(np.float32).eps)
CONTINUATION_SEEDS = (13_002, 13_003, 13_004)
CONTINUATION_DESIGNS = (
    "cfhirov13", "cfhirov13_noaction", "cfhirov13_noshrink",
    "ssm", "hacssmv8", "kdiov11",
)
DIRECT_SUM_MIN_RMS = 1e-8


class IntegrityError(RuntimeError):
    """One missing, malformed, non-finite, or hash-inconsistent artifact."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise IntegrityError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{label} is not finite: {result!r}")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"missing {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{path} must contain one JSON object")
    return value


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise IntegrityError(f"{label} contains a non-finite tensor")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise IntegrityError(f"{label} contains a non-finite array")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise IntegrityError(f"{label} contains {value!r}")


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"{label}: missing model.pt")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise IntegrityError(f"{label}: cannot load model.pt: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{label}: model.pt must contain a dictionary")
    return value


def validate_protocol(root: Path, protocol: Mapping[str, Any]) -> list[str]:
    """Validate the write-once source/data/command receipt before reading cells."""
    errors = []
    expected = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v13_screen_after_failed_v12",
        "seed": SEED,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "runs": EXPECTED_CELLS,
        "epochs": 30,
        "study": DEFAULT_STUDY,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "blas_threads_per_process": BLAS_THREADS,
        "automatic_100_epoch_launch_in_this_process": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            errors.append(f"protocol {key}={protocol.get(key)!r}, expected {value!r}")
    if protocol.get("git_branch") != "learnable-memory":
        errors.append("protocol git branch is not learnable-memory")
    commit = protocol.get("git_commit")
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        errors.append("protocol git commit is not a full lowercase SHA-1")
    if (protocol.get("git_worktree_clean") is not True
            or protocol.get("git_head_pushed") is not True
            or protocol.get("git_upstream_commit") != commit):
        errors.append("protocol does not prove a clean pushed launch commit")
    source = protocol.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != {str(path) for path in SOURCE_PATHS}:
        errors.append("protocol source manifest differs from frozen source set")
    else:
        for relative, expected_hash in source.items():
            path = ROOT / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                errors.append(f"protocol source hash mismatch: {relative}")
    data = protocol.get("data")
    if not isinstance(data, Mapping) or set(data) != set(TASKS):
        errors.append("protocol data manifest differs from frozen task set")
    else:
        for task in TASKS:
            for split in ("train", "val"):
                value = data[task].get(split) if isinstance(data[task], Mapping) else None
                path = Path(value) if isinstance(value, str) else Path("__missing__")
                path = path if path.is_absolute() else ROOT / path
                if (not path.is_file()
                        or sha256_file(path) != data[task].get(f"{split}_sha256")):
                    errors.append(f"protocol data hash mismatch: {task}/{split}")
    commands = protocol.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != set(TASKS):
        errors.append("protocol command grid differs from frozen task set")
    else:
        for task in TASKS:
            if not isinstance(commands[task], list) or len(commands[task]) != len(DESIGNS):
                errors.append(f"protocol command count differs for {task}")
    return errors


def validate_runner_receipt(
        root: Path, rows: Sequence[Mapping[str, Any]],
        protocol: Mapping[str, Any]) -> list[str]:
    errors = []
    path = root / "screen_runs.json"
    if not path.is_file():
        return ["missing screen_runs.json"]
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot load screen_runs.json: {exc}"]
    if not isinstance(records, list) or len(records) != EXPECTED_CELLS:
        return [f"screen_runs.json must contain {EXPECTED_CELLS} records"]
    expected = {(row["task"], row["design"]): row for row in rows}
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("screen_runs.json contains a non-object row")
            continue
        pair = (record.get("task"), record.get("design"))
        if pair in seen or pair not in expected:
            errors.append(f"invalid or duplicate runner cell {pair}")
            continue
        seen.add(pair)
        task, design = pair
        if str(record.get("gpu")) != str(protocol["task_pinned_gpu"][task]):
            errors.append(f"runner GPU mismatch: {task}/{design}")
        if record.get("artifact_sha256") != expected[pair]["artifact_sha256"]:
            errors.append(f"runner artifact hash mismatch: {task}/{design}")
    if seen != set(expected):
        errors.append("runner receipt cell set is incomplete")
    return errors


def _validate_history(payload: Mapping[str, Any], epochs: int, label: str) -> None:
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise IntegrityError(f"{label}: checkpoint must contain {epochs} epoch rows")
    epoch_indices = []
    for row in history:
        if not isinstance(row, Mapping):
            raise IntegrityError(f"{label}: malformed history row")
        epoch_indices.append(row.get("epoch"))
        if not isinstance(row.get("train"), Mapping) or not isinstance(
                row.get("val"), Mapping):
            raise IntegrityError(f"{label}: history row lacks train/val metrics")
        _finite_tree(row, f"{label}.history")
    if epoch_indices != list(range(1, epochs + 1)) or len(set(epoch_indices)) != epochs:
        raise IntegrityError(
            f"{label}: W&B-backed checkpoint history does not have {epochs} unique epochs")


def _validate_candidate_checkpoint(
        payload: Mapping[str, Any], metrics: Mapping[str, Any], epochs: int,
        label: str) -> None:
    fit_history = payload.get("fit_history")
    if not isinstance(fit_history, list) or len(fit_history) != epochs + 1:
        raise IntegrityError(f"{label}: fit history must contain {epochs + 1} rows")
    if [row.get("fit_index") for row in fit_history if isinstance(row, Mapping)] \
            != list(range(epochs + 1)):
        raise IntegrityError(f"{label}: fit-history indices are not exact")
    final_fit = payload.get("final_operator_fit")
    if not isinstance(final_fit, Mapping) or not isinstance(
            final_fit.get("receipts"), Mapping):
        raise IntegrityError(f"{label}: missing final operator fit/receipts")
    _finite_tree(final_fit, f"{label}.final_operator_fit")
    receipts = final_fit["receipts"]
    if receipts.get("fit_index") != epochs:
        raise IntegrityError(f"{label}: final fit index is not {epochs}")
    for key, value in receipts.items():
        if isinstance(value, (bool, int, float, str)):
            metric_key = f"cf_hiro_fit_{key}"
            if metrics.get(metric_key) != value:
                raise IntegrityError(
                    f"{label}: final fit receipt differs from metrics key {metric_key}")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise IntegrityError(f"{label}: missing model state")
    prefix = "world.mem_cfhirov13."
    updates = model_state.get(prefix + "fit_updates")
    installed = model_state.get(prefix + "operators_installed")
    if not isinstance(updates, torch.Tensor) or int(updates) != epochs + 1:
        raise IntegrityError(f"{label}: serialized fit_updates is not {epochs + 1}")
    if not isinstance(installed, torch.Tensor) or not bool(installed):
        raise IntegrityError(f"{label}: fitted operators are not installed")
    if metrics.get("fit_updates") != epochs + 1:
        raise IntegrityError(f"{label}: metrics fit_updates is not {epochs + 1}")
    for name in (
            "state_matrix", "action_matrix", "read_matrix", "process_covariance",
            "measurement_covariance", "initial_covariance", "steady_prior_covariance",
            "steady_gain", "initial_map", "output_mean", "action_mean"):
        saved = model_state.get(prefix + name)
        fitted = final_fit.get(name)
        if (not isinstance(saved, torch.Tensor) or not isinstance(fitted, torch.Tensor)
                or not torch.equal(saved.cpu(), fitted.cpu())):
            raise IntegrityError(f"{label}: serialized {name} differs from final fit")


def _validate_common(
        root: Path, task: str, design: str, *, seed: int,
        epochs: int, study: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    directory = run_directory(root, task, design)
    label = f"{task}/{design}"
    required_paths = {
        name: directory / name for name in
        ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")}
    for path in required_paths.values():
        if not path.is_file():
            raise IntegrityError(f"{label}: missing {path.name}")
    metrics = _load_json(required_paths["metrics.json"])
    expected = {
        "env": f"dmc:{task}", "design": design, "seed": seed, "epochs": epochs}
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise IntegrityError(
                f"{label}: metrics {key}={metrics.get(key)!r}, expected {value!r}")
    for key in (
            PRIMARY, CLEAN, "initial_encoder_integrator_probe_nmse",
            "predictive_loss_convergence_relative_change", "encoder_mean_channel_variance",
            "encoder_covariance_effective_rank", "encoder_singleton_max_abs",
            "encoder_prefix_max_abs", "eval_rollout_episode"):
        _finite(metrics.get(key), f"{label}:{key}")
    _finite(metrics.get("action_dim"), f"{label}:action_dim")
    if design in CF_HIRO_DESIGNS:
        for key in (
                "fit_updates", "memory_state_dim", "cf_hiro_fit_fit_index",
                "cf_hiro_fit_fit_episode_count", "cf_hiro_fit_fit_length",
                "cf_hiro_fit_markov_lag_count", "cf_hiro_fit_even_episodes",
                "cf_hiro_fit_odd_episodes", "cf_hiro_fit_action_refit_lags",
                "cf_hiro_streaming_max_abs", "cf_hiro_projector_algebra_max_abs",
                "cf_hiro_initial_reconstruction_max_abs",
                "cf_hiro_complement_dynamic_orthogonality_max_abs",
                "cf_hiro_core_steady_riccati_relative_residual",
                "cf_hiro_core_state_spectral_radius", "cf_hiro_core_state_operator_norm"):
            _finite(metrics.get(key), f"{label}:{key}")
    if design == CANDIDATE:
        for key in (
                "cf_hiro_fit_held_fold_action_r2_even_to_odd",
                "cf_hiro_fit_held_fold_action_r2_odd_to_even",
                "cf_hiro_true_action_suffix_advantage", "cf_hiro_action_pair_accuracy",
                "cf_hiro_complement_anchor_rms", "cf_hiro_dynamic_initial_rms"):
            _finite(metrics.get(key), f"{label}:{key}")
    data_receipt = protocol["data"][task]
    for split in ("train", "val"):
        if metrics.get(f"{split}_data_sha256") != data_receipt[f"{split}_sha256"]:
            raise IntegrityError(f"{label}: {split} data hash differs from protocol")
    rollout_hash = sha256_file(required_paths["eval_rollout.npz"])
    if metrics.get("eval_rollout_sha256") != rollout_hash:
        raise IntegrityError(f"{label}: rollout hash mismatch")
    wandb = _load_json(required_paths["wandb_run.json"])
    expected_wandb = {
        "state": "finished", "mode": "online", "study": study,
        "entity": WANDB_ENTITY, "project": WANDB_PROJECT,
        "eval_rollout_sha256": rollout_hash,
    }
    for key, value in expected_wandb.items():
        if wandb.get(key) != value:
            raise IntegrityError(
                f"{label}: W&B {key}={wandb.get(key)!r}, expected {value!r}")
    if not isinstance(wandb.get("run_id"), str) or not wandb["run_id"]:
        raise IntegrityError(f"{label}: missing W&B run ID")
    if not isinstance(wandb.get("url"), str) or not wandb["url"]:
        raise IntegrityError(f"{label}: missing W&B URL")
    expected_run_name = f"{study}-{directory.name}"
    if wandb.get("run_name") != expected_run_name:
        raise IntegrityError(f"{label}: W&B run name differs")
    payload = _load_checkpoint(required_paths["model.pt"], label)
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise IntegrityError(f"{label}: checkpoint lacks args")
    for key, value in {
            "memory_mode": design, "seed": seed, "epochs": epochs,
            "wandb": True, "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT, "wandb_mode": "online",
            "wandb_study": study, "eval_rollout_episode": 0}.items():
        if args.get(key) != value:
            raise IntegrityError(f"{label}: checkpoint arg {key} differs")
    if design == "kdiov11" and (
            metrics.get("development_action_ranking") != V11_COMPARATOR_RANKING
            or args.get("development_action_ranking") != V11_COMPARATOR_RANKING):
        raise IntegrityError(f"{label}: KDIO comparator ranking differs")
    _validate_history(payload, epochs, label)
    if payload.get("final_metrics") != metrics:
        raise IntegrityError(f"{label}: checkpoint final metrics differ from metrics.json")
    _finite_tree(payload.get("model_state_dict"), f"{label}.model_state_dict")
    if design in CF_HIRO_DESIGNS:
        _validate_candidate_checkpoint(payload, metrics, epochs, label)
    artifact_hashes = {
        name: sha256_file(path) for name, path in required_paths.items()}
    return {
        "task": task,
        "design": design,
        "metrics": metrics,
        "wandb": wandb,
        "directory": str(directory),
        "wandb_epoch_indices": list(range(1, epochs + 1)),
        "artifact_sha256": artifact_hashes,
    }


def load_rows(
        root: Path, seed: int, epochs: int, study: str,
        protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for task in TASKS:
        for design in DESIGNS:
            try:
                rows.append(_validate_common(
                    root, task, design, seed=seed, epochs=epochs,
                    study=study, protocol=protocol))
            except (IntegrityError, OSError, ValueError) as exc:
                errors.append(str(exc))
    run_ids = [row["wandb"]["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        errors.append("completed cells contain duplicate W&B run IDs")
    return rows, errors


def _design_values(
        rows: Sequence[Mapping[str, Any]], design: str, metric: str) -> np.ndarray:
    mapping = {
        str(row["task"]): _finite(
            row["metrics"].get(metric), f"{row['task']}/{design}:{metric}")
        for row in rows if row["design"] == design}
    if set(mapping) != set(TASKS):
        raise IntegrityError(f"{design}/{metric}: incomplete task grid")
    return np.asarray([mapping[task] for task in TASKS], dtype=np.float64)


def contrast(
        rows: Sequence[Mapping[str, Any]], reference: str,
        metric: str = PRIMARY) -> dict[str, Any]:
    candidate = _design_values(rows, CANDIDATE, metric)
    baseline = _design_values(rows, reference, metric)
    reductions = (baseline - candidate) / baseline
    return {
        "reference": reference,
        "metric": metric,
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(baseline.mean()),
        "equal_task_reduction": float(
            (baseline.mean() - candidate.mean()) / baseline.mean()),
        "paired_reduction_mean": float(reductions.mean()),
        "wins": int((candidate < baseline).sum()),
        "task_reductions": {
            task: float(value) for task, value in zip(TASKS, reductions, strict=True)},
    }


def _integrator_contrast(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    candidate = np.asarray([full[task][PRIMARY] for task in TASKS], dtype=np.float64)
    integrator = np.asarray([
        full[task]["initial_encoder_integrator_probe_nmse"] for task in TASKS],
        dtype=np.float64)
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


def representation_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["design"] != CANDIDATE:
            continue
        metrics = row["metrics"]
        label = f"{row['task']}/{row['design']}"
        checks = {
            "encoder variance below 1e-5": metrics["encoder_mean_channel_variance"] >= 1e-5,
            "encoder effective rank below 16": metrics["encoder_covariance_effective_rank"] >= 16,
            "encoder singleton causality above 1e-5":
                abs(metrics["encoder_singleton_max_abs"]) <= 1e-5,
            "encoder prefix causality above 1e-5":
                abs(metrics["encoder_prefix_max_abs"]) <= 1e-5,
        }
        failures.extend(f"{label}: {reason}" for reason, passed in checks.items() if not passed)
    return {"passed": not failures, "failures": failures}


def numerical_gate(rows: Sequence[Mapping[str, Any]], epochs: int) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["design"] not in CF_HIRO_DESIGNS:
            continue
        metrics = row["metrics"]
        design = str(row["design"])
        label = f"{row['task']}/{design}"
        action_dim = int(metrics["action_dim"])
        expected_state_dim = min(23 * 128, 24 * action_dim)
        exact = {
            "fit_updates": epochs + 1,
            "cf_hiro_fit_fit_index": epochs,
            "cf_hiro_fit_fit_episode_count": 1200,
            "cf_hiro_fit_fit_length": 48,
            "cf_hiro_fit_markov_lag_count": 47,
            "cf_hiro_fit_even_episodes": 600,
            "cf_hiro_fit_odd_episodes": 600,
            "cf_hiro_fit_action_refit_lags": 47,
            "cf_hiro_fit_fit_uses_validation": False,
            "cf_hiro_fit_fit_gradient_active": False,
            "cf_hiro_core_gradient_parameter_count": 0,
            "cf_hiro_core_streaming_covariance_floats": 0,
            "cf_hiro_core_online_covariance_update":
                "none_fixed_steady_gain_mean_only",
            "cf_hiro_fit_fold_agreement_mode": (
                "unit" if design == "cfhirov13_noshrink" else "empirical_bayes"),
            "cf_hiro_fit_transition_deployment": (
                "triangular" if design == "cfhirov13_triangular" else "normal"),
        }
        for key, expected in exact.items():
            if metrics.get(key) != expected:
                failures.append(
                    f"{label}: {key}={metrics.get(key)!r}, expected {expected!r}")
        if int(round(_finite(metrics.get("memory_state_dim"), label))) != expected_state_dim:
            failures.append(f"{label}: fixed full-Hankel state schema differs")
        bounded = {
            "cf_hiro_streaming_max_abs": 1e-5,
            "cf_hiro_projector_algebra_max_abs": 1e-5,
            "cf_hiro_initial_reconstruction_max_abs": 1e-5,
            "cf_hiro_complement_dynamic_orthogonality_max_abs": 1e-5,
            "cf_hiro_core_steady_riccati_relative_residual": 1e-6,
        }
        for key, maximum in bounded.items():
            if abs(_finite(metrics.get(key), f"{label}:{key}")) > maximum:
                failures.append(f"{label}: {key} exceeds {maximum}")
        radius = _finite(
            metrics.get("cf_hiro_core_state_spectral_radius"), f"{label}:radius")
        if radius > FLOAT32_STABILITY_BOUNDARY + 2e-6:
            failures.append(f"{label}: deployed spectral radius exceeds boundary")
        if design != "cfhirov13_triangular":
            if metrics.get("cf_hiro_core_state_is_real_normal_contraction") is not True:
                failures.append(f"{label}: normal-contraction receipt is false")
            operator = _finite(
                metrics.get("cf_hiro_core_state_operator_norm"), f"{label}:operator")
            if operator > FLOAT32_STABILITY_BOUNDARY + 2e-6:
                failures.append(f"{label}: normal operator norm exceeds boundary")
        if design == "cfhirov13_noaction" and metrics.get(
                "cf_hiro_exact_noaction") is not True:
            failures.append(f"{label}: no-action exactness failed")
        if design == "cfhirov13_nocorrect" and metrics.get(
                "cf_hiro_exact_nocorrect") is not True:
            failures.append(f"{label}: no-correction exactness failed")
    return {"passed": not failures, "failures": failures}


def analyze(
        rows: Sequence[Mapping[str, Any]], artifact_errors: Sequence[str],
        *, epochs: int, study: str) -> dict[str, Any]:
    artifact_complete = len(rows) == EXPECTED_CELLS and not artifact_errors
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v13_screen_after_failed_v12",
        "study": study,
        "seed": SEED,
        "epochs": epochs,
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": len(rows),
        "artifact_integrity_passed": artifact_complete,
        "artifact_integrity_errors": list(artifact_errors),
        "official_result": False,
        "iclr_confirmation": False,
        "automatic_100_epoch_launch_performed": False,
    }
    if not artifact_complete:
        result.update({
            "status": "INCOMPLETE_OR_INVALID",
            "representation_gate_passed": None,
            "scientific_gate_passed": False,
            "continue_to_100_epochs": False,
            "contingent_100e_manifest": None,
        })
        return result

    representation = representation_gate(rows)
    numerical = numerical_gate(rows, epochs)
    external = {reference: contrast(rows, reference) for reference in EXTERNAL_REFERENCES}
    internal = {reference: contrast(rows, reference) for reference in INTERNAL_CONTROLS}
    integrator = _integrator_contrast(rows)
    external_pass = all(
        receipt["equal_task_reduction"] >= .05 and receipt["wins"] >= 3
        for receipt in (*external.values(), integrator))
    internal_pass = all(
        internal[design]["equal_task_reduction"] >= INTERNAL_THRESHOLDS[design]
        and internal[design]["wins"] >= 3 for design in INTERNAL_CONTROLS)

    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    action_task_checks = {}
    for task in TASKS:
        metrics = full[task]
        checks = {
            "held_even_to_odd_r2_positive":
                _finite(metrics.get("cf_hiro_fit_held_fold_action_r2_even_to_odd"), task) > 0,
            "held_odd_to_even_r2_positive":
                _finite(metrics.get("cf_hiro_fit_held_fold_action_r2_odd_to_even"), task) > 0,
            "suffix_advantage_positive":
                _finite(metrics.get("cf_hiro_true_action_suffix_advantage"), task) > 0,
            "pair_accuracy_above_chance":
                _finite(metrics.get("cf_hiro_action_pair_accuracy"), task) > .5,
        }
        action_task_checks[task] = {**checks, "passed": all(checks.values())}
    action_pass_count = sum(value["passed"] for value in action_task_checks.values())
    action_gate = {
        "passed": action_pass_count >= 3,
        "passed_tasks": action_pass_count,
        "required_tasks": 3,
        "tasks": action_task_checks,
    }
    energy_values = {
        task: {
            "complement_anchor_rms": _finite(
                full[task].get("cf_hiro_complement_anchor_rms"), task),
            "dynamic_initial_rms": _finite(
                full[task].get("cf_hiro_dynamic_initial_rms"), task),
        } for task in TASKS}
    energy_gate = {
        "passed": all(
            value["complement_anchor_rms"] > DIRECT_SUM_MIN_RMS
            and value["dynamic_initial_rms"] > DIRECT_SUM_MIN_RMS
            for value in energy_values.values()),
        "minimum_rms": DIRECT_SUM_MIN_RMS,
        "tasks": energy_values,
    }
    all_late = np.asarray([
        abs(_finite(
            row["metrics"]["predictive_loss_convergence_relative_change"], "late"))
        for row in rows], dtype=np.float64)
    full_signed = {
        task: _finite(
            full[task]["predictive_loss_convergence_relative_change"], task)
        for task in TASKS}
    convergence = {
        "full_signed_nonnegative_every_task": all(value >= 0 for value in full_signed.values()),
        "full_signed_values": full_signed,
        "full_max_abs": max(abs(value) for value in full_signed.values()),
        "full_max_abs_below_5pct": max(abs(value) for value in full_signed.values()) < .05,
        "all_cell_median_abs": float(np.median(all_late)),
        "all_cell_median_abs_below_3pct": float(np.median(all_late)) < .03,
    }
    convergence["passed"] = bool(
        convergence["full_signed_nonnegative_every_task"]
        and convergence["full_max_abs_below_5pct"]
        and convergence["all_cell_median_abs_below_3pct"])
    scientific_pass = bool(
        representation["passed"] and numerical["passed"]
        and external_pass and internal_pass and action_gate["passed"]
        and energy_gate["passed"] and convergence["passed"])
    result.update({
        "status": "SCREEN_PASS_100E_MANIFEST" if scientific_pass else "SCREEN_NO_GO",
        "representation_gate_passed": representation["passed"],
        "representation_gate": representation,
        "numerical_gate": numerical,
        "design_means": {
            design: {
                PRIMARY: float(_design_values(rows, design, PRIMARY).mean()),
                CLEAN: float(_design_values(rows, design, CLEAN).mean()),
            } for design in DESIGNS},
        "external_contrasts": external,
        "internal_contrasts": internal,
        "integrator_contrast": integrator,
        "external_performance_gate_passed": external_pass,
        "internal_mechanism_gate_passed": internal_pass,
        "action_gate": action_gate,
        "direct_sum_energy_gate": energy_gate,
        "convergence_gate": convergence,
        "scientific_gate_passed": scientific_pass,
        "continue_to_100_epochs": scientific_pass,
        "contingent_100e_manifest": (
            "contingent_100e_launch_manifest.json" if scientific_pass else None),
        "wandb_runs": [{
            "task": row["task"], "design": row["design"],
            "run_id": row["wandb"]["run_id"], "url": row["wandb"]["url"],
            "epoch_indices": row["wandb_epoch_indices"],
        } for row in rows],
        "artifact_sha256": [{
            "task": row["task"], "design": row["design"],
            "sha256": row["artifact_sha256"],
        } for row in rows],
    })
    return result


def analysis_exit_code(analysis: Mapping[str, Any]) -> int:
    """Scientific NO_GO is a valid completed result; only artifact failure is nonzero."""
    return 0 if analysis.get("artifact_integrity_passed") is True else 2


def _continuation_manifest(
        root: Path, protocol: Mapping[str, Any], analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "AUTHORIZED_NOT_LAUNCHED",
        "trigger": "V13 30e screen passed every frozen conjunctive gate",
        "screen_root": str(root),
        "screen_protocol_sha256": sha256_file(root / "screen_protocol.json"),
        "screen_analysis_sha256": sha256_file(root / "screen_analysis.json"),
        "tasks": list(TASKS),
        "designs": list(CONTINUATION_DESIGNS),
        "seeds": list(CONTINUATION_SEEDS),
        "epochs": 100,
        "runs": len(TASKS) * len(CONTINUATION_DESIGNS) * len(CONTINUATION_SEEDS),
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_study": "hacssm-v13-contingent-cfhiro100",
        "data": protocol["data"],
        "source_sha256": protocol["source_sha256"],
        "source_changes_allowed": False,
        "automatic_launch_performed": False,
        "scientific_gate_passed": analysis["scientific_gate_passed"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root if args.root.is_absolute() else (ROOT / args.root).resolve()
    if args.seed != SEED or args.epochs != 30 or args.study != DEFAULT_STUDY:
        raise ValueError("V13 screen seed/epochs/study are frozen")
    protocol = _load_json(root / "screen_protocol.json")
    protocol_errors = validate_protocol(root, protocol)
    rows, errors = load_rows(root, args.seed, args.epochs, args.study, protocol)
    errors = [*protocol_errors, *errors]
    if len(rows) == EXPECTED_CELLS:
        errors.extend(validate_runner_receipt(root, rows, protocol))
    analysis = analyze(rows, errors, epochs=args.epochs, study=args.study)
    rendered = json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.write:
        analysis_path = root / "screen_analysis.json"
        decision_path = root / "screen_decision.json"
        for path in (analysis_path, decision_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        with analysis_path.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
        decision = {
            "status": analysis["status"],
            "artifact_integrity_passed": analysis["artifact_integrity_passed"],
            "representation_gate_passed": analysis["representation_gate_passed"],
            "scientific_gate_passed": analysis["scientific_gate_passed"],
            "continue_to_100_epochs": analysis["continue_to_100_epochs"],
            "automatic_launch_performed": False,
        }
        with decision_path.open("x", encoding="utf-8") as stream:
            json.dump(decision, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if analysis["scientific_gate_passed"]:
            manifest_path = root / "contingent_100e_launch_manifest.json"
            if manifest_path.exists():
                raise FileExistsError(f"refusing to overwrite {manifest_path}")
            manifest = _continuation_manifest(root, protocol, analysis)
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
    raise SystemExit(analysis_exit_code(analysis))


if __name__ == "__main__":
    main()
