#!/usr/bin/env python3
"""Validate and analyze the frozen 40-cell CF-EBO-v14 screen."""

from __future__ import annotations

import argparse
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
from scripts.run_cf_ebo_v14_screen import (
    BLAS_THREADS,
    CONTINUATION_DESIGNS,
    CONTINUATION_EPOCHS,
    CONTINUATION_SEEDS,
    CONTINUATION_STUDY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STUDY,
    LOCK_NAME,
    SEED,
    SOURCE_PATHS,
    TASKS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    _json_sha256,
    data_paths,
    run_directory,
    train_command,
)
from scripts.train_cf_ebo_v14 import (
    CF_EBO_DESIGNS,
    CORE_MODES,
    DESIGNS,
)
from scripts.train_cf_hiro_v13 import V11_COMPARATOR_RANKING


PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
CANDIDATE = "cfebov14"
EXTERNAL_REFERENCES = ("cfhirov13_nocorrect", "ssm", "hacssmv8", "kdiov11")
INTERNAL_CONTROLS = (
    "cfebov14_nocorrect", "cfebov14_noaction", "cfebov14_norisk",
    "cfebov14_noenergycap", "cfebov14_noradial",
)
INTERNAL_THRESHOLDS = {
    design: (.05 if design == "cfebov14_noaction" else .02)
    for design in INTERNAL_CONTROLS}
EXPECTED_CELLS = len(TASKS) * len(DESIGNS)
CONDITIONS = (
    "clean", "val_train_view", "freeze", "gaussian_noise", "checkerboard",
    "long_freeze")
ENERGY_TOLERANCE = 2e-2


class IntegrityError(RuntimeError):
    """A missing, malformed, non-finite, or hash-inconsistent artifact."""


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
        value = json.loads(path.read_text(encoding="utf-8"))
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


def _deep_equal(first: Any, second: Any) -> bool:
    """Exact recursive equality for checkpoint receipts, including tensors."""
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
                and torch.equal(first.cpu(), second.cpu()))
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
                and first.dtype == second.dtype and first.shape == second.shape
                and bool(np.array_equal(first, second)))
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (isinstance(first, Mapping) and isinstance(second, Mapping)
                and set(first) == set(second)
                and all(_deep_equal(first[key], second[key]) for key in first))
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (type(first) is type(second) and len(first) == len(second)
                and all(_deep_equal(left, right)
                        for left, right in zip(first, second, strict=True)))
    return type(first) is type(second) and first == second


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"{label}: missing model.pt")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise IntegrityError(f"{label}: cannot load model.pt: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"{label}: model.pt must contain a dictionary")
    return payload


def _validate_rollout(path: Path, label: str) -> None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not payload.files:
                raise IntegrityError(f"{label}: rollout is empty")
            for key in payload.files:
                value = payload[key]
                if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
                    raise IntegrityError(f"{label}: rollout {key} is non-finite")
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"{label}: cannot read rollout: {exc}") from exc


def validate_protocol(root: Path, protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v14_screen_after_failed_v13",
        "seed": SEED,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "runs": EXPECTED_CELLS,
        "epochs": 30,
        "gpus": ["0", "1", "2", "3"],
        "task_pinned_gpu": dict(zip(TASKS, ("0", "1", "2", "3"), strict=True)),
        "study": DEFAULT_STUDY,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "v11_comparator_action_ranking": V11_COMPARATOR_RANKING,
        "blas_threads_per_process": BLAS_THREADS,
        "automatic_continuation_launch_in_this_process": False,
        "conditional_continuation_manifest": "conditional_continuation_manifest.json",
        "continuation_runs": 96,
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
    expected_sources = {str(path) for path in SOURCE_PATHS}
    if not isinstance(source, Mapping) or set(source) != expected_sources:
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
                entry = data[task] if isinstance(data[task], Mapping) else {}
                value = entry.get(split)
                expected_value = str(data_paths(task)[0 if split == "train" else 1])
                if value != expected_value:
                    errors.append(f"protocol data path mismatch: {task}/{split}")
                path = Path(value) if isinstance(value, str) else Path("__missing__")
                path = path if path.is_absolute() else ROOT / path
                if (not path.is_file()
                        or sha256_file(path) != entry.get(f"{split}_sha256")):
                    errors.append(f"protocol data hash mismatch: {task}/{split}")
    commands = protocol.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != set(TASKS):
        errors.append("protocol command grid differs from frozen task set")
    else:
        expected_commands = {
            task: [train_command(
                str(ROOT / ".venv" / "bin" / "python"), root.resolve(),
                DEFAULT_STUDY, 30, task, design)
                for design in DESIGNS]
            for task in TASKS
        }
        for task in TASKS:
            if not isinstance(commands[task], list) or len(commands[task]) != len(DESIGNS):
                errors.append(f"protocol command count differs for {task}")
        if commands != expected_commands:
            errors.append("protocol commands differ from exact frozen command vectors")
        if protocol.get("commands_sha256") != _json_sha256(commands):
            errors.append("protocol command-grid hash differs")
    return errors


def validate_prospective_continuation(root: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = _load_json(root / "conditional_continuation_manifest.json")
    except IntegrityError as exc:
        return [str(exc)]
    expected = {
        "schema_version": 1,
        "status": "CONDITIONAL_NOT_AUTHORIZED",
        "launch_performed": False,
        "automatic_launch_supported": False,
        "designs": list(CONTINUATION_DESIGNS),
        "tasks": list(TASKS),
        "seeds": list(CONTINUATION_SEEDS),
        "epochs": CONTINUATION_EPOCHS,
        "runs": 96,
        "study": CONTINUATION_STUDY,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(f"prospective continuation {key} differs")
    commands = manifest.get("commands")
    if not isinstance(commands, list) or len(commands) != 96:
        errors.append("prospective continuation does not contain 96 commands")
    elif manifest.get("commands_sha256") != _json_sha256(commands):
        errors.append("prospective continuation command hash differs")
    return errors


def _validate_history(payload: Mapping[str, Any], epochs: int, label: str) -> None:
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise IntegrityError(f"{label}: checkpoint must contain {epochs} history rows")
    indices = []
    for row in history:
        if not isinstance(row, Mapping):
            raise IntegrityError(f"{label}: malformed history row")
        indices.append(row.get("epoch"))
        if not isinstance(row.get("train"), Mapping) or not isinstance(row.get("val"), Mapping):
            raise IntegrityError(f"{label}: history row lacks train/val metrics")
        _finite_tree(row, f"{label}.history")
    if indices != list(range(1, epochs + 1)):
        raise IntegrityError(f"{label}: history epoch identities are not exact")


FIT_BUFFER_NAMES = (
    "state_matrix", "action_matrix", "raw_action_matrix", "read_matrix",
    "correction_matrix", "raw_correction_matrix", "innovation_covariance",
    "innovation_whitener", "initial_map", "output_projector",
    "complement_projector", "energy_support_projector", "output_mean", "action_mean",
    "action_reliability", "correction_reliability", "innovation_rank",
)


def _validate_candidate_checkpoint(
        payload: Mapping[str, Any], metrics: Mapping[str, Any], epochs: int,
        label: str) -> None:
    fit_history = payload.get("fit_history")
    if not isinstance(fit_history, list) or len(fit_history) != epochs + 1:
        raise IntegrityError(f"{label}: fit history must contain {epochs + 1} rows")
    if [row.get("fit_index") for row in fit_history if isinstance(row, Mapping)] \
            != list(range(epochs + 1)):
        raise IntegrityError(f"{label}: fit-history indices are not exact")
    _finite_tree(fit_history, f"{label}.fit_history")
    final_fit = payload.get("final_operator_fit")
    if not isinstance(final_fit, Mapping) or not isinstance(final_fit.get("receipts"), Mapping):
        raise IntegrityError(f"{label}: missing final operator fit/receipts")
    required_fields = set(FIT_BUFFER_NAMES) | {"markov_even", "markov_odd", "receipts"}
    if set(final_fit) != required_fields:
        raise IntegrityError(f"{label}: CEBOFit payload field set differs")
    _finite_tree(final_fit, f"{label}.final_operator_fit")
    receipts = final_fit["receipts"]
    if receipts.get("fit_index") != epochs:
        raise IntegrityError(f"{label}: final fit index is not {epochs}")
    for key, value in receipts.items():
        if isinstance(value, (bool, int, float, str)):
            metric_key = f"cf_ebo_fit_{key}"
            if metrics.get(metric_key) != value:
                raise IntegrityError(
                    f"{label}: final fit receipt differs from metrics key {metric_key}")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise IntegrityError(f"{label}: missing model state")
    prefix = "world.mem_cfebov14."
    updates, installed = state.get(prefix + "fit_updates"), state.get(
        prefix + "operators_installed")
    if not isinstance(updates, torch.Tensor) or int(updates) != epochs + 1:
        raise IntegrityError(f"{label}: serialized fit_updates is not {epochs + 1}")
    if not isinstance(installed, torch.Tensor) or not bool(installed):
        raise IntegrityError(f"{label}: fitted operators are not installed")
    if metrics.get("fit_updates") != epochs + 1:
        raise IntegrityError(f"{label}: metrics fit_updates is not {epochs + 1}")
    for name in FIT_BUFFER_NAMES:
        saved, fitted = state.get(prefix + name), final_fit.get(name)
        if (not isinstance(saved, torch.Tensor) or not isinstance(fitted, torch.Tensor)
                or not torch.equal(saved.cpu(), fitted.cpu())):
            raise IntegrityError(f"{label}: serialized {name} differs from final fit")
    extra = state.get(prefix + "_extra_state")
    if (not isinstance(extra, Mapping)
            or not _deep_equal(extra.get("fit_receipts"), receipts)):
        raise IntegrityError(
            f"{label}: serialized fit receipts differ from final fit receipts")


def _validate_common(
        root: Path, task: str, design: str, *, seed: int, epochs: int,
        study: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    directory = run_directory(root, task, design)
    label = f"{task}/{design}"
    paths = {name: directory / name for name in (
        "model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")}
    for path in paths.values():
        if not path.is_file():
            raise IntegrityError(f"{label}: missing {path.name}")
    metrics = _load_json(paths["metrics.json"])
    for key, value in {
            "env": f"dmc:{task}", "design": design,
            "seed": seed, "epochs": epochs}.items():
        if metrics.get(key) != value:
            raise IntegrityError(f"{label}: metrics {key} differs")
    _finite_tree(metrics, f"{label}.metrics")
    for key in (
            PRIMARY, CLEAN, "initial_encoder_integrator_probe_nmse",
            "predictive_loss_convergence_relative_change", "encoder_mean_channel_variance",
            "encoder_covariance_effective_rank", "encoder_singleton_max_abs",
            "encoder_prefix_max_abs", "eval_rollout_episode", "action_dim"):
        _finite(metrics.get(key), f"{label}:{key}")
    if design in CF_EBO_DESIGNS:
        for key in (
                "fit_updates", "memory_state_dim", "cf_ebo_fit_fit_index",
                "cf_ebo_fit_fit_episode_count", "cf_ebo_fit_fit_length",
                "cf_ebo_fit_markov_lag_count", "cf_ebo_fit_even_episodes",
                "cf_ebo_fit_odd_episodes", "cf_ebo_streaming_max_abs",
                "cf_ebo_initial_reconstruction_max_abs",
                "cf_ebo_core_energy_identity_max_abs",
                "cf_ebo_core_state_spectral_radius",
                "cf_ebo_core_state_operator_norm",
                "cf_ebo_core_deployed_correction_operator_norm",
                "cf_ebo_core_action_reliability",
                "cf_ebo_core_correction_reliability"):
            _finite(metrics.get(key), f"{label}:{key}")
        for condition in CONDITIONS:
            for suffix in (
                    "innovation_score_mean", "radial_gate_mean",
                    "correction_energy_max", "evidence_samples"):
                key = f"cf_ebo_{condition}_{suffix}"
                _finite(metrics.get(key), f"{label}:{key}")
    data = protocol["data"][task]
    for split in ("train", "val"):
        if metrics.get(f"{split}_data_sha256") != data[f"{split}_sha256"]:
            raise IntegrityError(f"{label}: {split} data hash differs")
    _validate_rollout(paths["eval_rollout.npz"], label)
    rollout_hash = sha256_file(paths["eval_rollout.npz"])
    if metrics.get("eval_rollout_sha256") != rollout_hash:
        raise IntegrityError(f"{label}: rollout hash mismatch")
    wandb = _load_json(paths["wandb_run.json"])
    for key, value in {
            "state": "finished", "mode": "online", "study": study,
            "entity": WANDB_ENTITY, "project": WANDB_PROJECT,
            "eval_rollout_sha256": rollout_hash}.items():
        if wandb.get(key) != value:
            raise IntegrityError(f"{label}: W&B {key} differs")
    if not isinstance(wandb.get("run_id"), str) or not wandb["run_id"]:
        raise IntegrityError(f"{label}: missing W&B run ID")
    if not isinstance(wandb.get("url"), str) or not wandb["url"]:
        raise IntegrityError(f"{label}: missing W&B URL")
    if wandb.get("run_name") != f"{study}-{directory.name}":
        raise IntegrityError(f"{label}: W&B run name differs")
    payload = _load_checkpoint(paths["model.pt"], label)
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
    if design in CF_EBO_DESIGNS:
        _validate_candidate_checkpoint(payload, metrics, epochs, label)
    return {
        "task": task,
        "design": design,
        "metrics": metrics,
        "wandb": wandb,
        "wandb_epoch_indices": list(range(1, epochs + 1)),
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def load_rows(
        root: Path, seed: int, epochs: int, study: str,
        protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    for task in TASKS:
        for design in DESIGNS:
            try:
                rows.append(_validate_common(
                    root, task, design, seed=seed, epochs=epochs,
                    study=study, protocol=protocol))
            except (IntegrityError, OSError, ValueError) as exc:
                errors.append(str(exc))
    ids = [row["wandb"]["run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("completed cells contain duplicate W&B run IDs")
    return rows, errors


def validate_runner_receipt(
        root: Path, rows: Sequence[Mapping[str, Any]],
        protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if (root / LOCK_NAME).exists():
        errors.append("runner lock still exists")
    path = root / "screen_runs.json"
    if not path.is_file():
        return [*errors, "missing screen_runs.json"]
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [*errors, f"cannot load screen_runs.json: {exc}"]
    if not isinstance(records, list) or len(records) != EXPECTED_CELLS:
        return [*errors, f"screen_runs.json must contain {EXPECTED_CELLS} records"]
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
        if record.get("seed") != SEED:
            errors.append(f"runner seed mismatch: {task}/{design}")
        expected_command = protocol["commands"][task][DESIGNS.index(design)]
        if record.get("command_sha256") != _json_sha256(expected_command):
            errors.append(f"runner command hash mismatch: {task}/{design}")
        if _finite(record.get("seconds"), f"runner seconds {task}/{design}") <= 0:
            errors.append(f"runner seconds are non-positive: {task}/{design}")
        if record.get("artifact_sha256") != expected[pair]["artifact_sha256"]:
            errors.append(f"runner artifact hash mismatch: {task}/{design}")
    if seen != set(expected):
        errors.append("runner receipt cell set is incomplete")
    return errors


def _design_values(rows: Sequence[Mapping[str, Any]], design: str, metric: str) -> np.ndarray:
    values = {
        str(row["task"]): _finite(row["metrics"].get(metric), f"{design}:{metric}")
        for row in rows if row["design"] == design}
    if set(values) != set(TASKS):
        raise IntegrityError(f"{design}/{metric}: incomplete task grid")
    return np.asarray([values[task] for task in TASKS], dtype=np.float64)


def contrast(rows: Sequence[Mapping[str, Any]], reference: str) -> dict[str, Any]:
    candidate = _design_values(rows, CANDIDATE, PRIMARY)
    baseline = _design_values(rows, reference, PRIMARY)
    reductions = (baseline - candidate) / baseline
    return {
        "reference": reference,
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(baseline.mean()),
        "equal_task_reduction": float((baseline.mean() - candidate.mean()) / baseline.mean()),
        "paired_reduction_mean": float(reductions.mean()),
        "wins": int((candidate < baseline).sum()),
        "task_reductions": dict(zip(TASKS, map(float, reductions), strict=True)),
    }


def representation_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["design"] != CANDIDATE:
            continue
        metrics, label = row["metrics"], f"{row['task']}/{CANDIDATE}"
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


def _close(actual: float, expected: float, tolerance: float = 1e-7) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(actual), abs(expected))


def numerical_gate(rows: Sequence[Mapping[str, Any]], epochs: int) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["design"] not in CF_EBO_DESIGNS:
            continue
        metrics, design = row["metrics"], str(row["design"])
        label = f"{row['task']}/{design}"
        action_dim = int(metrics["action_dim"])
        expected_state_dim = min(23 * 128, 24 * action_dim)
        exact = {
            "fit_updates": epochs + 1,
            "cf_ebo_fit_fit_index": epochs,
            "cf_ebo_fit_fit_episode_count": 1200,
            "cf_ebo_fit_fit_length": 48,
            "cf_ebo_fit_markov_lag_count": 47,
            "cf_ebo_fit_even_episodes": 600,
            "cf_ebo_fit_odd_episodes": 600,
            "cf_ebo_fit_action_even_action_refit_lags": 47,
            "cf_ebo_fit_action_odd_action_refit_lags": 47,
            "cf_ebo_fit_action_pooled_action_refit_lags": 47,
            "cf_ebo_fit_fit_uses_validation": False,
            "cf_ebo_fit_fit_gradient_active": False,
            "cf_ebo_fit_action_combination": "minimum_directional_positive_part_EB",
            "cf_ebo_fit_correction_combination": "minimum_directional_positive_part_EB",
            "cf_ebo_core_gradient_parameter_count": 0,
            "cf_ebo_core_streaming_covariance_floats": 0,
        }
        for key, expected in exact.items():
            if metrics.get(key) != expected:
                failures.append(f"{label}: {key}={metrics.get(key)!r}, expected {expected!r}")
        if int(round(_finite(metrics.get("memory_state_dim"), label))) != expected_state_dim:
            failures.append(f"{label}: fixed full-Hankel state schema differs")
        bounded = {
            "cf_ebo_streaming_max_abs": 1e-5,
            "cf_ebo_initial_reconstruction_max_abs": 1e-5,
            "cf_ebo_core_energy_identity_max_abs": 2e-5,
            "cf_ebo_fit_energy_dissipativity_max_abs": 1e-8,
            "cf_ebo_fit_energy_lyapunov_relative_residual": 1e-8,
            "cf_ebo_fit_output_projector_idempotence_max_abs": 1e-5,
            "cf_ebo_fit_complement_projector_idempotence_max_abs": 1e-5,
            "cf_ebo_fit_direct_sum_projector_sum_max_abs": 1e-5,
            "cf_ebo_fit_complement_read_orthogonality_max_abs": 1e-5,
            "cf_ebo_fit_energy_support_projector_symmetry_max_abs": 1e-8,
            "cf_ebo_fit_energy_support_projector_idempotence_max_abs": 1e-8,
            "cf_ebo_core_energy_support_projector_symmetry_max_abs": 2e-5,
            "cf_ebo_core_energy_support_projector_idempotence_max_abs": 2e-5,
            "cf_ebo_core_energy_support_state_left_max_abs": 2e-5,
            "cf_ebo_core_energy_support_state_right_max_abs": 2e-5,
            "cf_ebo_core_energy_support_read_max_abs": 2e-5,
            "cf_ebo_core_energy_support_action_max_abs": 2e-5,
            "cf_ebo_core_energy_support_raw_action_max_abs": 2e-5,
            "cf_ebo_core_energy_support_correction_max_abs": 2e-5,
            "cf_ebo_core_energy_support_raw_correction_max_abs": 2e-5,
            "cf_ebo_core_energy_support_initial_map_max_abs": 2e-5,
        }
        for key, maximum in bounded.items():
            if abs(_finite(metrics.get(key), f"{label}:{key}")) > maximum:
                failures.append(f"{label}: {key} exceeds {maximum}")
        if _finite(metrics.get("cf_ebo_core_state_spectral_radius"), label) >= 1.0:
            failures.append(f"{label}: transition is not strictly stable")
        if _finite(metrics.get("cf_ebo_core_state_operator_norm"), label) > 1.0 + 2e-5:
            failures.append(f"{label}: energy-coordinate transition is expansive")
        active_rank = int(round(_finite(
            metrics.get("cf_ebo_core_energy_support_rank"), label)))
        inactive = int(round(_finite(
            metrics.get("cf_ebo_core_energy_inactive_padding"), label)))
        fit_rank = int(round(_finite(metrics.get("cf_ebo_fit_energy_state_rank"), label)))
        fit_inactive = int(round(_finite(
            metrics.get("cf_ebo_fit_energy_inactive_padding"), label)))
        fit_support_rank = int(round(_finite(
            metrics.get("cf_ebo_fit_energy_support_projector_rank"), label)))
        if (not 0 <= active_rank <= expected_state_dim
                or inactive != expected_state_dim - active_rank
                or fit_rank != active_rank or fit_support_rank != active_rank
                or fit_inactive != inactive):
            failures.append(f"{label}: active energy rank/inactive padding receipts differ")
        for mechanism in ("action", "correction"):
            first = _finite(metrics.get(
                f"cf_ebo_fit_{mechanism}_first_direction_reliability"), label)
            second = _finite(metrics.get(
                f"cf_ebo_fit_{mechanism}_second_direction_reliability"), label)
            combined = _finite(metrics.get(
                f"cf_ebo_fit_{mechanism}_combined_risk_reliability"), label)
            computed = _finite(metrics.get(
                f"cf_ebo_fit_computed_{mechanism}_reliability"), label)
            if (not _close(combined, min(first, second), 1e-10)
                    or not _close(computed, combined, 1e-10)):
                failures.append(f"{label}: {mechanism} reliability is not directional minimum")
        cap_active = design != "cfebov14_noenergycap"
        radial_active = design != "cfebov14_noradial"
        if metrics.get("cf_ebo_core_energy_cap_active") is not cap_active:
            failures.append(f"{label}: energy-cap intervention differs")
        if metrics.get("cf_ebo_core_radial_gate_active") is not radial_active:
            failures.append(f"{label}: radial intervention differs")
        correction_norm = _finite(
            metrics.get("cf_ebo_core_deployed_correction_operator_norm"), label)
        if cap_active and correction_norm > 1.0 + 2e-5:
            failures.append(f"{label}: capped correction operator exceeds one")
        if design == "cfebov14_noaction":
            if metrics.get("cf_ebo_exact_noaction") is not True or _finite(
                    metrics.get("cf_ebo_core_action_reliability"), label) != 0.0:
                failures.append(f"{label}: no-action exactness failed")
        if design == "cfebov14_nocorrect":
            if metrics.get("cf_ebo_exact_nocorrect") is not True or _finite(
                    metrics.get("cf_ebo_core_correction_reliability"), label) != 0.0:
                failures.append(f"{label}: no-correction exactness failed")
        if design == "cfebov14_norisk" and (
                _finite(metrics.get("cf_ebo_core_action_reliability"), label) != 1.0
                or _finite(metrics.get("cf_ebo_core_correction_reliability"), label) != 1.0):
            failures.append(f"{label}: no-risk exactness failed")
        codimension = int(round(_finite(
            metrics.get("cf_ebo_core_complement_codimension"), label)))
        if not 0 <= codimension <= 128:
            failures.append(f"{label}: complement codimension is invalid")
        if metrics.get("cf_ebo_core_complement_present") is not (codimension > 0):
            failures.append(f"{label}: complement-presence receipt differs from codimension")
        for fold in ("even", "odd", "pooled"):
            prefix = f"cf_ebo_fit_correction_{fold}_fit_"
            score_mean = _finite(metrics.get(prefix + "innovation_score_mean"), label)
            score_max = _finite(metrics.get(prefix + "innovation_score_max"), label)
            gate_mean = _finite(metrics.get(prefix + "radial_gate_mean"), label)
            gate_min = _finite(metrics.get(prefix + "radial_gate_min"), label)
            gate_max = _finite(metrics.get(prefix + "radial_gate_max"), label)
            if (score_mean < 0.0 or score_max < score_mean
                    or not 0.0 <= gate_min <= gate_mean <= gate_max <= 1.0):
                failures.append(f"{label}: {fold} fit-calibration telemetry is invalid")
    return {"passed": not failures, "failures": failures}


def _integrator_contrast(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    candidate = np.asarray([full[task][PRIMARY] for task in TASKS], dtype=np.float64)
    integrator = np.asarray([
        full[task]["initial_encoder_integrator_probe_nmse"] for task in TASKS],
        dtype=np.float64)
    reductions = (integrator - candidate) / integrator
    return {
        "reference": "candidate_checkpoint_initial_encoder_integrator",
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(integrator.mean()),
        "equal_task_reduction": float(
            (integrator.mean() - candidate.mean()) / integrator.mean()),
        "wins": int((candidate < integrator).sum()),
        "task_reductions": dict(zip(TASKS, map(float, reductions), strict=True)),
    }


def analyze(
        rows: Sequence[Mapping[str, Any]], artifact_errors: Sequence[str],
        *, epochs: int, study: str) -> dict[str, Any]:
    complete = len(rows) == EXPECTED_CELLS and not artifact_errors
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v14_screen_after_failed_v13",
        "study": study,
        "seed": SEED,
        "epochs": epochs,
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": len(rows),
        "artifact_integrity_passed": complete,
        "artifact_integrity_errors": list(artifact_errors),
        "official_result": False,
        "iclr_confirmation": False,
        "automatic_100_epoch_launch_performed": False,
    }
    if not complete:
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
    external = {name: contrast(rows, name) for name in EXTERNAL_REFERENCES}
    internal = {name: contrast(rows, name) for name in INTERNAL_CONTROLS}
    integrator = _integrator_contrast(rows)
    external_pass = all(
        value["equal_task_reduction"] >= .05 and value["wins"] >= 3
        for value in (*external.values(), integrator))
    internal_pass = all(
        internal[name]["equal_task_reduction"] >= INTERNAL_THRESHOLDS[name]
        and internal[name]["wins"] >= 3 for name in INTERNAL_CONTROLS)
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}

    mechanism_tasks = {}
    robustness_tasks = {}
    energy_bound_tasks = {}
    for task in TASKS:
        metrics = full[task]
        mechanism_checks = {
            "action_even_to_odd_positive":
                _finite(metrics.get("cf_ebo_fit_action_even_to_odd_mean_improvement"), task) > 0,
            "action_odd_to_even_positive":
                _finite(metrics.get("cf_ebo_fit_action_odd_to_even_mean_improvement"), task) > 0,
            "correction_even_to_odd_positive":
                _finite(metrics.get("cf_ebo_fit_correction_even_to_odd_mean_improvement"), task) > 0,
            "correction_odd_to_even_positive":
                _finite(metrics.get("cf_ebo_fit_correction_odd_to_even_mean_improvement"), task) > 0,
            "action_reliability_positive":
                _finite(metrics.get("cf_ebo_core_action_reliability"), task) > 0,
            "correction_reliability_positive":
                _finite(metrics.get("cf_ebo_core_correction_reliability"), task) > 0,
            "executed_suffix_advantage_positive":
                _finite(metrics.get("cf_ebo_true_action_suffix_advantage"), task) > 0,
            "pair_accuracy_above_chance":
                _finite(metrics.get("cf_ebo_action_pair_accuracy"), task) > .5,
        }
        mechanism_tasks[task] = {**mechanism_checks, "passed": all(mechanism_checks.values())}
        robust_checks = {
            "gaussian_score_above_val_train_view": _finite(
                metrics.get("cf_ebo_gaussian_noise_innovation_score_mean"), task) > _finite(
                metrics.get("cf_ebo_val_train_view_innovation_score_mean"), task),
            "gaussian_gate_below_val_train_view": _finite(
                metrics.get("cf_ebo_gaussian_noise_radial_gate_mean"), task) < _finite(
                metrics.get("cf_ebo_val_train_view_radial_gate_mean"), task),
        }
        robustness_tasks[task] = {**robust_checks, "passed": all(robust_checks.values())}
        rank = _finite(metrics.get("cf_ebo_core_innovation_rank"), task)
        alpha = _finite(metrics.get("cf_ebo_core_correction_reliability"), task)
        bound = alpha * alpha * rank
        condition_values = {
            condition: _finite(
                metrics.get(f"cf_ebo_{condition}_correction_energy_max"), task)
            for condition in CONDITIONS}
        passed = all(value <= bound * (1.0 + ENERGY_TOLERANCE) + 1e-5
                     for value in condition_values.values())
        energy_bound_tasks[task] = {
            "bound": bound,
            "tolerance": ENERGY_TOLERANCE,
            "conditions": condition_values,
            "passed": passed,
        }
    mechanism_count = sum(value["passed"] for value in mechanism_tasks.values())
    robust_count = sum(value["passed"] for value in robustness_tasks.values())
    mechanism_gate = {
        "passed": mechanism_count >= 3,
        "passed_tasks": mechanism_count,
        "required_tasks": 3,
        "tasks": mechanism_tasks,
    }
    robustness_gate = {
        "passed": robust_count >= 3 and all(
            value["passed"] for value in energy_bound_tasks.values()),
        "distribution_shift_passed_tasks": robust_count,
        "required_shift_tasks": 3,
        "tasks": robustness_tasks,
        "correction_energy_bound_tasks": energy_bound_tasks,
    }
    complement_gate = {
        "passed": True,
        "policy": "codimension_zero_is_valid_no_nonzero_energy_requirement",
        "tasks": {
            task: {
                "codimension": int(full[task]["cf_ebo_core_complement_codimension"]),
                "present": bool(full[task]["cf_ebo_core_complement_present"]),
            } for task in TASKS},
    }
    fit_calibration = {
        task: {
            fold: {
                "innovation_score_mean": _finite(full[task].get(
                    f"cf_ebo_fit_correction_{fold}_fit_innovation_score_mean"), task),
                "innovation_score_max": _finite(full[task].get(
                    f"cf_ebo_fit_correction_{fold}_fit_innovation_score_max"), task),
                "radial_gate_mean": _finite(full[task].get(
                    f"cf_ebo_fit_correction_{fold}_fit_radial_gate_mean"), task),
                "radial_gate_min": _finite(full[task].get(
                    f"cf_ebo_fit_correction_{fold}_fit_radial_gate_min"), task),
                "radial_gate_max": _finite(full[task].get(
                    f"cf_ebo_fit_correction_{fold}_fit_radial_gate_max"), task),
            } for fold in ("even", "odd", "pooled")
        } for task in TASKS}
    all_late = np.asarray([
        abs(_finite(row["metrics"]["predictive_loss_convergence_relative_change"], "late"))
        for row in rows], dtype=np.float64)
    full_signed = {
        task: _finite(full[task]["predictive_loss_convergence_relative_change"], task)
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
    scientific = bool(
        representation["passed"] and numerical["passed"]
        and external_pass and internal_pass and mechanism_gate["passed"]
        and robustness_gate["passed"] and complement_gate["passed"]
        and convergence["passed"])
    result.update({
        "status": "SCREEN_PASS_100E_MANIFEST" if scientific else "SCREEN_NO_GO",
        "representation_gate_passed": representation["passed"],
        "representation_gate": representation,
        "numerical_gate": numerical,
        "design_means": {
            design: {
                PRIMARY: float(_design_values(rows, design, PRIMARY).mean()),
                CLEAN: float(_design_values(rows, design, CLEAN).mean()),
            } for design in DESIGNS},
        "external_contrasts": external,
        "integrator_contrast": integrator,
        "internal_contrasts": internal,
        "external_performance_gate_passed": external_pass,
        "internal_mechanism_gate_passed": internal_pass,
        "action_correction_mechanism_gate": mechanism_gate,
        "robustness_gate": robustness_gate,
        "rank_aware_complement_gate": complement_gate,
        "train_fit_calibration_evidence": fit_calibration,
        "convergence_gate": convergence,
        "scientific_gate_passed": scientific,
        "continue_to_100_epochs": scientific,
        "contingent_100e_manifest": (
            "contingent_100e_launch_manifest.json" if scientific else None),
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
    """Scientific NO_GO is complete evidence; only invalid artifacts are nonzero."""
    return 0 if analysis.get("artifact_integrity_passed") is True else 2


def _authorized_manifest(
        root: Path, protocol: Mapping[str, Any], analysis: Mapping[str, Any]) -> dict[str, Any]:
    prospective = _load_json(root / "conditional_continuation_manifest.json")
    return {
        "schema_version": 1,
        "status": "AUTHORIZED_NOT_LAUNCHED",
        "trigger": "V14 30e screen passed every frozen conjunctive gate",
        "screen_root": str(root),
        "screen_protocol_sha256": sha256_file(root / "screen_protocol.json"),
        "screen_analysis_sha256": sha256_file(root / "screen_analysis.json"),
        "tasks": list(TASKS),
        "designs": list(CONTINUATION_DESIGNS),
        "seeds": list(CONTINUATION_SEEDS),
        "epochs": CONTINUATION_EPOCHS,
        "runs": 96,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_study": CONTINUATION_STUDY,
        "data": protocol["data"],
        "source_sha256": protocol["source_sha256"],
        "commands": prospective["commands"],
        "commands_sha256": prospective["commands_sha256"],
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
    root = (args.root if args.root.is_absolute() else ROOT / args.root).resolve()
    if args.seed != SEED or args.epochs != 30 or args.study != DEFAULT_STUDY:
        raise ValueError("V14 screen seed/epochs/study are frozen")
    protocol: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        protocol = _load_json(root / "screen_protocol.json")
        protocol_errors = validate_protocol(root, protocol)
        errors.extend(protocol_errors)
    except (IntegrityError, OSError, ValueError, TypeError) as exc:
        protocol_errors = [str(exc)]
        errors.extend(protocol_errors)
    errors.extend(validate_prospective_continuation(root))
    if protocol is None or protocol_errors:
        rows = []
    else:
        rows, row_errors = load_rows(
            root, args.seed, args.epochs, args.study, protocol)
        errors.extend(row_errors)
    if len(rows) == EXPECTED_CELLS:
        assert protocol is not None
        errors.extend(validate_runner_receipt(root, rows, protocol))
    analysis = analyze(rows, errors, epochs=args.epochs, study=args.study)
    rendered = json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.write:
        root.mkdir(parents=True, exist_ok=True)
        analysis_path, decision_path = root / "screen_analysis.json", root / "screen_decision.json"
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
            assert protocol is not None
            manifest_path = root / "contingent_100e_launch_manifest.json"
            if manifest_path.exists():
                raise FileExistsError(f"refusing to overwrite {manifest_path}")
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(_authorized_manifest(root, protocol, analysis),
                          stream, indent=2, sort_keys=True)
                stream.write("\n")
    raise SystemExit(analysis_exit_code(analysis))


if __name__ == "__main__":
    main()
