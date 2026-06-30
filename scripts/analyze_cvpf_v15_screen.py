#!/usr/bin/env python3
"""Validate and analyze the frozen 52-cell CVPF-v15 development screen.

Scientific failure is a valid completed result.  Missing, malformed, non-finite,
or provenance-inconsistent evidence fails closed.  This program never launches
the prospective continuation.
"""

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
from scripts.run_cvpf_v15_screen import (
    BASELINES,
    BLAS_THREADS,
    CONTINUATION_DESIGNS,
    CONTINUATION_EPOCHS,
    CONTINUATION_SEEDS,
    CONTINUATION_STUDY,
    CVPF_DESIGNS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STUDY,
    DESIGNS,
    EPOCHS,
    LOCK_NAME,
    SEED,
    SOURCE_PATHS,
    TASKS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    V11_COMPARATOR_RANKING,
    data_paths,
    json_sha256,
    run_directory,
    train_command,
)


PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
CANDIDATE = "cvpfv15"
DIRECT_CONTROLS = tuple(design for design in CVPF_DESIGNS if design != CANDIDATE)
EXPECTED_CELLS = len(TASKS) * len(DESIGNS)
STRUCTURAL_TOLERANCE = 1e-5
SHIFT_RELATIVE_TOLERANCE = 16.0 * np.finfo(np.float64).eps

STREAM_KEYS = ("cvpf_streaming_max_abs", "cvpf_core_streaming_max_abs")
PREFIX_KEYS = ("cvpf_prefix_closure_max_abs", "cvpf_core_prefix_closure_max_abs")
SHIFT_KEYS = (
    "cvpf_shift_closure_relative", "cvpf_core_shift_closure_relative",
    "cvpf_shift_closure_ratio", "cvpf_core_shift_closure_ratio")
INNOVATION_EXPOSURE_KEYS = (
    "cvpf_observation_deployed_to_fit_innovation_rms_ratio",
    "cvpf_core_observation_deployed_to_fit_innovation_rms_ratio",
    "cvpf_fit_observation_deployed_to_fit_innovation_rms_ratio")
ACTION_GAIN_KEYS = (
    "cvpf_action_crossfit_mean_gain", "cvpf_fit_action_crossfit_mean_gain",
    "cvpf_fit_action_crossfit_gain", "cvpf_core_action_gain")
CORRECTION_GAIN_KEYS = (
    "cvpf_correction_crossfit_mean_gain", "cvpf_fit_correction_crossfit_mean_gain",
    "cvpf_fit_correction_crossfit_gain", "cvpf_core_correction_gain")
SUFFIX_ADVANTAGE_KEYS = (
    "cvpf_true_action_suffix_advantage", "cvpf_true_action_prior_advantage",
    "cvpf_action_suffix_advantage")
PAIR_ACCURACY_KEYS = ("cvpf_action_pair_accuracy", "cvpf_action_swap_pair_accuracy")


class IntegrityError(RuntimeError):
    """A missing, malformed, non-finite, or inconsistent artifact."""


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise IntegrityError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{label} is not finite: {result!r}")
    return result


def metric(metrics: Mapping[str, Any], keys: Sequence[str], label: str) -> float:
    for key in keys:
        if key in metrics:
            return finite(metrics[key], f"{label}:{key}")
    raise IntegrityError(f"{label}: missing every metric alias {tuple(keys)}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"missing {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"{path} must contain one JSON object")
    return value


def finite_tree(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise IntegrityError(f"{label} contains a non-finite tensor")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise IntegrityError(f"{label} contains a non-finite array")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise IntegrityError(f"{label} contains {value!r}")


def deep_equal(first: Any, second: Any) -> bool:
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
                and all(deep_equal(first[key], second[key]) for key in first))
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (type(first) is type(second) and len(first) == len(second)
                and all(deep_equal(left, right)
                        for left, right in zip(first, second, strict=True)))
    return type(first) is type(second) and first == second


def load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"{label}: missing model.pt")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise IntegrityError(f"{label}: cannot load model.pt: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"{label}: model.pt must contain a dictionary")
    return payload


def validate_rollout(path: Path, label: str) -> None:
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
        "scope": "excluded_adaptive_v15_cvpf_screen",
        "seed": SEED,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "runs": EXPECTED_CELLS,
        "epochs": EPOCHS,
        "gpus": ["0", "1", "2", "3"],
        "task_pinned_gpu": dict(zip(TASKS, ("0", "1", "2", "3"), strict=True)),
        "study": DEFAULT_STUDY,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "v11_comparator_action_ranking": V11_COMPARATOR_RANKING,
        "blas_threads_per_process": BLAS_THREADS,
        "automatic_continuation_launch_in_this_process": False,
        "conditional_continuation_manifest": "conditional_continuation_manifest.json",
        "continuation_runs": 156,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            errors.append(f"protocol {key}={protocol.get(key)!r}, expected {value!r}")
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
            entry = data[task] if isinstance(data[task], Mapping) else {}
            for index, split in enumerate(("train", "val")):
                value = entry.get(split)
                expected_value = str(data_paths(task)[index])
                if value != expected_value:
                    errors.append(f"protocol data path mismatch: {task}/{split}")
                    continue
                path = Path(value)
                path = path if path.is_absolute() else ROOT / path
                if (not path.is_file()
                        or sha256_file(path) != entry.get(f"{split}_sha256")):
                    errors.append(f"protocol data hash mismatch: {task}/{split}")
    commands = protocol.get("commands")
    expected_commands = {
        task: [train_command(
            str(ROOT / ".venv" / "bin" / "python"), root.resolve(),
            DEFAULT_STUDY, EPOCHS, task, design)
            for design in DESIGNS]
        for task in TASKS
    }
    if commands != expected_commands:
        errors.append("protocol commands differ from exact frozen command vectors")
    if protocol.get("commands_sha256") != json_sha256(commands):
        errors.append("protocol command-grid hash differs")
    return errors


def validate_prospective_continuation(root: Path) -> list[str]:
    try:
        manifest = load_json(root / "conditional_continuation_manifest.json")
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
        "runs": 156,
        "study": CONTINUATION_STUDY,
    }
    errors = [
        f"prospective continuation {key} differs"
        for key, value in expected.items() if manifest.get(key) != value]
    commands = manifest.get("commands")
    if not isinstance(commands, list) or len(commands) != 156:
        errors.append("prospective continuation does not contain 156 commands")
    elif manifest.get("commands_sha256") != json_sha256(commands):
        errors.append("prospective continuation command hash differs")
    return errors


def validate_history(payload: Mapping[str, Any], epochs: int, label: str) -> None:
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise IntegrityError(f"{label}: checkpoint must contain {epochs} history rows")
    indices = []
    for row in history:
        if not isinstance(row, Mapping):
            raise IntegrityError(f"{label}: malformed history row")
        indices.append(row.get("epoch"))
        if not isinstance(row.get("train"), Mapping) or not isinstance(
                row.get("val"), Mapping):
            raise IntegrityError(f"{label}: history row lacks train/val metrics")
        finite_tree(row, f"{label}.history")
    if indices != list(range(1, epochs + 1)):
        raise IntegrityError(f"{label}: history epoch identities are not exact")


def validate_candidate_checkpoint(
        payload: Mapping[str, Any], metrics: Mapping[str, Any], epochs: int,
        label: str) -> None:
    fit_history = payload.get("fit_history")
    if not isinstance(fit_history, list) or len(fit_history) != epochs + 1:
        raise IntegrityError(f"{label}: fit history must contain {epochs + 1} rows")
    if [row.get("fit_index") for row in fit_history if isinstance(row, Mapping)] \
            != list(range(epochs + 1)):
        raise IntegrityError(f"{label}: fit-history indices are not exact")
    finite_tree(fit_history, f"{label}.fit_history")
    final_fit = payload.get("final_operator_fit")
    if not isinstance(final_fit, Mapping) or not isinstance(
            final_fit.get("receipts"), Mapping):
        raise IntegrityError(f"{label}: missing final operator fit/receipts")
    finite_tree(final_fit, f"{label}.final_operator_fit")
    receipts = final_fit["receipts"]
    if receipts.get("fit_index") != epochs:
        raise IntegrityError(f"{label}: final fit index is not {epochs}")
    for key, value in receipts.items():
        if isinstance(value, (bool, int, float, str)):
            metric_key = f"cvpf_fit_{key}"
            if metrics.get(metric_key) != value:
                raise IntegrityError(
                    f"{label}: final fit receipt differs from metrics key {metric_key}")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise IntegrityError(f"{label}: missing model state")
    prefix = "world.mem_cvpfv15."
    updates = state.get(prefix + "fit_updates")
    installed = state.get(prefix + "operators_installed")
    if not isinstance(updates, torch.Tensor) or int(updates) != epochs + 1:
        raise IntegrityError(f"{label}: serialized fit_updates is not {epochs + 1}")
    if not isinstance(installed, torch.Tensor) or not bool(installed):
        raise IntegrityError(f"{label}: fitted operators are not installed")
    if metrics.get("fit_updates") != epochs + 1:
        raise IntegrityError(f"{label}: metrics fit_updates is not {epochs + 1}")
    # Every fitted tensor that is also a registered core buffer must match bitwise.
    for name, fitted in final_fit.items():
        saved = state.get(prefix + name)
        if isinstance(fitted, torch.Tensor) and saved is not None:
            if not isinstance(saved, torch.Tensor) or not torch.equal(
                    saved.cpu(), fitted.cpu()):
                raise IntegrityError(f"{label}: serialized {name} differs from final fit")
    extra = state.get(prefix + "_extra_state")
    if (not isinstance(extra, Mapping)
            or not deep_equal(extra.get("fit_receipts"), receipts)):
        raise IntegrityError(
            f"{label}: serialized fit receipts differ from final fit receipts")


def validate_common(
        root: Path, task: str, design: str, *, protocol: Mapping[str, Any]
        ) -> dict[str, Any]:
    directory = run_directory(root, task, design)
    label = f"{task}/{design}"
    paths = {name: directory / name for name in (
        "model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")}
    for path in paths.values():
        if not path.is_file():
            raise IntegrityError(f"{label}: missing {path.name}")
    metrics = load_json(paths["metrics.json"])
    for key, value in {
            "env": f"dmc:{task}", "design": design,
            "seed": SEED, "epochs": EPOCHS}.items():
        if metrics.get(key) != value:
            raise IntegrityError(f"{label}: metrics {key} differs")
    finite_tree(metrics, f"{label}.metrics")
    for key in (
            PRIMARY, CLEAN, "initial_encoder_integrator_probe_nmse",
            "predictive_loss_convergence_relative_change",
            "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
            "encoder_singleton_max_abs", "encoder_prefix_max_abs",
            "eval_rollout_episode", "action_dim"):
        finite(metrics.get(key), f"{label}:{key}")
    if design in CVPF_DESIGNS:
        for keys in (STREAM_KEYS, PREFIX_KEYS, SHIFT_KEYS):
            metric(metrics, keys, label)
        for key in (
                "fit_updates", "cvpf_fit_fit_index", "cvpf_fit_fit_episode_count",
                "cvpf_core_action_gain", "cvpf_core_correction_gain",
                "cvpf_core_risk_gain", "cvpf_core_rho", "cvpf_envelope_weight"):
            finite(metrics.get(key), f"{label}:{key}")
        for key in (
                "cvpf_fit_fit_uses_validation", "cvpf_fit_fit_gradient_active",
                "cvpf_exact_nocorrect", "cvpf_exact_noaction",
                "cvpf_exact_norisk", "cvpf_exact_norho",
                "cvpf_exact_anchoronly", "cvpf_identification_detached",
                "cvpf_envelope_active"):
            if not isinstance(metrics.get(key), bool):
                raise IntegrityError(f"{label}:{key} must be boolean")
        for keys in (
                ACTION_GAIN_KEYS, CORRECTION_GAIN_KEYS,
                SUFFIX_ADVANTAGE_KEYS, PAIR_ACCURACY_KEYS):
            metric(metrics, keys, label)
    data = protocol["data"][task]
    for split in ("train", "val"):
        if metrics.get(f"{split}_data_sha256") != data[f"{split}_sha256"]:
            raise IntegrityError(f"{label}: {split} data hash differs")
    validate_rollout(paths["eval_rollout.npz"], label)
    rollout_hash = sha256_file(paths["eval_rollout.npz"])
    if metrics.get("eval_rollout_sha256") != rollout_hash:
        raise IntegrityError(f"{label}: rollout hash mismatch")
    wandb = load_json(paths["wandb_run.json"])
    for key, value in {
            "state": "finished", "mode": "online", "study": DEFAULT_STUDY,
            "entity": WANDB_ENTITY, "project": WANDB_PROJECT,
            "eval_rollout_sha256": rollout_hash}.items():
        if wandb.get(key) != value:
            raise IntegrityError(f"{label}: W&B {key} differs")
    if not isinstance(wandb.get("run_id"), str) or not wandb["run_id"]:
        raise IntegrityError(f"{label}: missing W&B run ID")
    if not isinstance(wandb.get("url"), str) or not wandb["url"]:
        raise IntegrityError(f"{label}: missing W&B URL")
    if wandb.get("run_name") != f"{DEFAULT_STUDY}-{directory.name}":
        raise IntegrityError(f"{label}: W&B run name differs")
    payload = load_checkpoint(paths["model.pt"], label)
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise IntegrityError(f"{label}: checkpoint lacks args")
    for key, value in {
            "memory_mode": design, "seed": SEED, "epochs": EPOCHS,
            "wandb": True, "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT, "wandb_mode": "online",
            "wandb_study": DEFAULT_STUDY, "eval_rollout_episode": 0}.items():
        if args.get(key) != value:
            raise IntegrityError(f"{label}: checkpoint arg {key} differs")
    if design == "kdiov11" and (
            metrics.get("development_action_ranking") != V11_COMPARATOR_RANKING
            or args.get("development_action_ranking") != V11_COMPARATOR_RANKING):
        raise IntegrityError(f"{label}: KDIO comparator ranking differs")
    validate_history(payload, EPOCHS, label)
    if payload.get("final_metrics") != metrics:
        raise IntegrityError(f"{label}: checkpoint final metrics differ from metrics.json")
    finite_tree(payload.get("model_state_dict"), f"{label}.model_state_dict")
    if design in CVPF_DESIGNS:
        validate_candidate_checkpoint(payload, metrics, EPOCHS, label)
    return {
        "task": task,
        "design": design,
        "metrics": metrics,
        "wandb": wandb,
        "wandb_epoch_indices": list(range(1, EPOCHS + 1)),
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def load_rows(
        root: Path, protocol: Mapping[str, Any]
        ) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for task in TASKS:
        for design in DESIGNS:
            try:
                rows.append(validate_common(root, task, design, protocol=protocol))
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
        command = protocol["commands"][task][DESIGNS.index(design)]
        if record.get("command_sha256") != json_sha256(command):
            errors.append(f"runner command hash mismatch: {task}/{design}")
        try:
            seconds = finite(record.get("seconds"), f"runner seconds {task}/{design}")
            if seconds <= 0:
                errors.append(f"runner seconds are non-positive: {task}/{design}")
        except IntegrityError as exc:
            errors.append(str(exc))
        if record.get("artifact_sha256") != expected[pair]["artifact_sha256"]:
            errors.append(f"runner artifact hash mismatch: {task}/{design}")
    if seen != set(expected):
        errors.append("runner receipt cell set is incomplete")
    return errors


def design_values(
        rows: Sequence[Mapping[str, Any]], design: str, key: str) -> np.ndarray:
    values = {
        str(row["task"]): finite(row["metrics"].get(key), f"{design}:{key}")
        for row in rows if row["design"] == design}
    if set(values) != set(TASKS):
        raise IntegrityError(f"{design}/{key}: incomplete task grid")
    return np.asarray([values[task] for task in TASKS], dtype=np.float64)


def contrast(rows: Sequence[Mapping[str, Any]], reference: str) -> dict[str, Any]:
    candidate = design_values(rows, CANDIDATE, PRIMARY)
    baseline = design_values(rows, reference, PRIMARY)
    reductions = (baseline - candidate) / baseline
    return {
        "reference": reference,
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(baseline.mean()),
        "equal_task_reduction": float(
            (baseline.mean() - candidate.mean()) / baseline.mean()),
        "paired_reduction_mean": float(reductions.mean()),
        "wins": int((candidate < baseline).sum()),
        "task_reductions": dict(zip(TASKS, map(float, reductions), strict=True)),
    }


def integrator_contrast(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    candidate = np.asarray([full[task][PRIMARY] for task in TASKS], dtype=np.float64)
    baseline = np.asarray([
        full[task]["initial_encoder_integrator_probe_nmse"] for task in TASKS],
        dtype=np.float64)
    reductions = (baseline - candidate) / baseline
    return {
        "reference": "candidate_checkpoint_initial_encoder_integrator",
        "candidate_mean": float(candidate.mean()),
        "reference_mean": float(baseline.mean()),
        "equal_task_reduction": float(
            (baseline.mean() - candidate.mean()) / baseline.mean()),
        "paired_reduction_mean": float(reductions.mean()),
        "wins": int((candidate < baseline).sum()),
        "task_reductions": dict(zip(TASKS, map(float, reductions), strict=True)),
    }


def representation_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["design"] != CANDIDATE:
            continue
        metrics = row["metrics"]
        label = f"{row['task']}/{CANDIDATE}"
        checks = {
            "encoder variance below 1e-5": finite(
                metrics.get("encoder_mean_channel_variance"), label) >= 1e-5,
            "encoder effective rank below 16": finite(
                metrics.get("encoder_covariance_effective_rank"), label) >= 16,
            "encoder singleton mismatch": abs(finite(
                metrics.get("encoder_singleton_max_abs"), label)) <= 1e-5,
            "encoder prefix mismatch": abs(finite(
                metrics.get("encoder_prefix_max_abs"), label)) <= 1e-5,
        }
        failures.extend(f"{label}: {name}" for name, passed in checks.items() if not passed)
    return {"passed": not failures, "failures": failures}


def structural_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    receipts: dict[str, Any] = {}
    for row in rows:
        if row["design"] not in CVPF_DESIGNS:
            continue
        metrics = row["metrics"]
        label = f"{row['task']}/{row['design']}"
        values = {
            "streaming_max_abs": metric(metrics, STREAM_KEYS, label),
            "prefix_closure_max_abs": metric(metrics, PREFIX_KEYS, label),
            "shift_closure_relative": metric(metrics, SHIFT_KEYS, label),
            "innovation_exposure_ratio": metric(
                metrics, INNOVATION_EXPOSURE_KEYS, label),
        }
        receipts[label] = values
        for name in ("streaming_max_abs", "prefix_closure_max_abs"):
            value = values[name]
            if abs(value) > STRUCTURAL_TOLERANCE:
                failures.append(f"{label}: {name} exceeds {STRUCTURAL_TOLERANCE}")
        shift = values["shift_closure_relative"]
        if not 0.0 <= shift <= 1.0 + SHIFT_RELATIVE_TOLERANCE:
            failures.append(
                f"{label}: projected shift closure {shift} exceeds the zero-predictor bound")
        exposure = values["innovation_exposure_ratio"]
        if not .5 <= exposure <= 2.0:
            failures.append(
                f"{label}: deployed/fit innovation RMS ratio {exposure} outside [0.5,2]")
        if metrics.get("cvpf_fit_fit_uses_validation") is not False:
            failures.append(f"{label}: operator fit used validation data")
        if int(round(finite(metrics.get("fit_updates"), label))) != EPOCHS + 1:
            failures.append(f"{label}: fit_updates is not {EPOCHS + 1}")
        if int(round(finite(metrics.get("cvpf_fit_fit_index"), label))) != EPOCHS:
            failures.append(f"{label}: final fit index is not {EPOCHS}")
    return {"passed": not failures, "failures": failures, "receipts": receipts}


def mode_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["design"] not in CVPF_DESIGNS:
            continue
        design, metrics = str(row["design"]), row["metrics"]
        label = f"{row['task']}/{design}"
        gains = {
            key: finite(metrics.get(key), label) for key in (
                "cvpf_core_action_gain", "cvpf_core_correction_gain",
                "cvpf_core_risk_gain", "cvpf_core_rho")}
        if any(not 0.0 <= value <= 1.0 for value in gains.values()):
            failures.append(f"{label}: mode gain outside [0,1]")
        envelope_weight = finite(metrics.get("cvpf_envelope_weight"), label)
        if envelope_weight < 0:
            failures.append(f"{label}: envelope weight is negative")
        expected = {
            "cvpf_exact_nocorrect": design in (
                "cvpfv15_nocorrect", "cvpfv15_anchoronly"),
            "cvpf_exact_noaction": design in (
                "cvpfv15_noaction", "cvpfv15_anchoronly"),
            "cvpf_exact_norisk": design == "cvpfv15_norisk",
            "cvpf_exact_norho": design == "cvpfv15_norho",
            "cvpf_exact_anchoronly": design == "cvpfv15_anchoronly",
            "cvpf_identification_detached": design == "cvpfv15_detachid",
            "cvpf_envelope_active": design != "cvpfv15_noenvelope",
        }
        for key, value in expected.items():
            if metrics.get(key) is not value:
                failures.append(f"{label}: {key} differs from exact mode semantics")
        exact_values = {
            "cvpfv15_nocorrect": ("cvpf_core_correction_gain", 0.0),
            "cvpfv15_noaction": ("cvpf_core_action_gain", 0.0),
            "cvpfv15_norisk": ("cvpf_core_risk_gain", 1.0),
            "cvpfv15_norho": ("cvpf_core_rho", 1.0),
        }
        if design in exact_values:
            key, wanted = exact_values[design]
            if gains[key] != wanted:
                failures.append(f"{label}: {key} is not exact {wanted}")
        if design == "cvpfv15_noenvelope" and (
                envelope_weight != 0.0 or metrics.get("cvpf_envelope_active") is not False):
            failures.append(f"{label}: no-envelope intervention is not exact")
    return {"passed": not failures, "failures": failures}


def mechanism_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    tasks: dict[str, Any] = {}
    counts = {"action_gain": 0, "correction_gain": 0,
              "suffix_advantage": 0, "pair_accuracy": 0}
    for task in TASKS:
        metrics = full[task]
        values = {
            "action_gain": metric(metrics, ACTION_GAIN_KEYS, task),
            "correction_gain": metric(metrics, CORRECTION_GAIN_KEYS, task),
            "suffix_advantage": metric(metrics, SUFFIX_ADVANTAGE_KEYS, task),
            "pair_accuracy": metric(metrics, PAIR_ACCURACY_KEYS, task),
        }
        checks = {
            "action_gain": values["action_gain"] > 0,
            "correction_gain": values["correction_gain"] > 0,
            "suffix_advantage": values["suffix_advantage"] > 0,
            "pair_accuracy": values["pair_accuracy"] > .5,
        }
        for key, passed in checks.items():
            counts[key] += int(passed)
        tasks[task] = {**values, **{f"{key}_passed": value
                                     for key, value in checks.items()}}
    passed = all(value >= 3 for value in counts.values())
    return {"passed": passed, "required_tasks": 3,
            "passed_task_counts": counts, "tasks": tasks}


def convergence_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    signed = {
        task: finite(full[task]["predictive_loss_convergence_relative_change"], task)
        for task in TASKS}
    all_abs = np.asarray([
        abs(finite(row["metrics"]["predictive_loss_convergence_relative_change"],
                   "late")) for row in rows], dtype=np.float64)
    maximum = max(abs(value) for value in signed.values())
    median = float(np.median(all_abs))
    result = {
        "full_signed_nonnegative_every_task": all(value >= 0 for value in signed.values()),
        "full_signed_values": signed,
        "full_max_abs": maximum,
        "full_max_abs_below_5pct": maximum < .05,
        "all_cell_median_abs": median,
        "all_cell_median_abs_below_3pct": median < .03,
    }
    result["passed"] = all((
        result["full_signed_nonnegative_every_task"],
        result["full_max_abs_below_5pct"],
        result["all_cell_median_abs_below_3pct"],
    ))
    return result


def analyze(
        rows: Sequence[Mapping[str, Any]], artifact_errors: Sequence[str],
        *, study: str = DEFAULT_STUDY) -> dict[str, Any]:
    complete = len(rows) == EXPECTED_CELLS and not artifact_errors
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v15_cvpf_screen",
        "study": study,
        "seed": SEED,
        "epochs": EPOCHS,
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": len(rows),
        "artifact_integrity_passed": complete,
        "artifact_integrity_errors": list(artifact_errors),
        "official_result": False,
        "automatic_continuation_launch_performed": False,
    }
    if not complete:
        result.update({
            "status": "INCOMPLETE_OR_INVALID",
            "scientific_gate_passed": False,
            "continue_to_100_epochs": False,
            "conditional_authorization_status": "CONDITIONAL_NOT_AUTHORIZED",
        })
        return result
    representation = representation_gate(rows)
    structural = structural_gate(rows)
    modes = mode_gate(rows)
    baselines = {name: contrast(rows, name) for name in BASELINES}
    controls = {name: contrast(rows, name) for name in DIRECT_CONTROLS}
    integrator = integrator_contrast(rows)
    baseline_pass = all(
        value["equal_task_reduction"] >= .05 and value["wins"] >= 3
        for value in (*baselines.values(), integrator))
    controls_pass = all(
        value["equal_task_reduction"] >= .02 and value["wins"] >= 3
        for value in controls.values())
    identification_envelope = {
        "detachid": controls["cvpfv15_detachid"],
        "noenvelope": controls["cvpfv15_noenvelope"],
    }
    identification_envelope_pass = all(
        value["equal_task_reduction"] >= .02 and value["wins"] >= 3
        for value in identification_envelope.values())
    mechanism = mechanism_gate(rows)
    convergence = convergence_gate(rows)
    scientific = bool(
        representation["passed"] and structural["passed"] and modes["passed"]
        and baseline_pass and controls_pass and identification_envelope_pass
        and mechanism["passed"] and convergence["passed"])
    result.update({
        "status": "SCREEN_GO" if scientific else "SCREEN_NO_GO",
        "design_means": {
            design: {
                PRIMARY: float(design_values(rows, design, PRIMARY).mean()),
                CLEAN: float(design_values(rows, design, CLEAN).mean()),
            } for design in DESIGNS},
        "baseline_contrasts": baselines,
        "legal_integrator_contrast": integrator,
        "direct_control_contrasts": controls,
        "baseline_gate_passed": baseline_pass,
        "direct_control_gate_passed": controls_pass,
        "active_identification_envelope_gate": {
            "passed": identification_envelope_pass,
            "contrasts": identification_envelope,
        },
        "representation_gate": representation,
        "structural_integrity_gate": structural,
        "mode_gain_exact_ablation_gate": modes,
        "action_correction_suffix_gate": mechanism,
        "convergence_gate": convergence,
        "scientific_gate_passed": scientific,
        "continue_to_100_epochs": scientific,
        "conditional_authorization_status": (
            "AUTHORIZED_NOT_LAUNCHED" if scientific
            else "CONDITIONAL_NOT_AUTHORIZED"),
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
    return 0 if analysis.get("artifact_integrity_passed") is True else 2


def authorization_payload(
        root: Path, protocol: Mapping[str, Any], analysis: Mapping[str, Any]
        ) -> dict[str, Any]:
    prospective = load_json(root / "conditional_continuation_manifest.json")
    passed = analysis.get("scientific_gate_passed") is True
    return {
        "schema_version": 1,
        "status": "AUTHORIZED_NOT_LAUNCHED" if passed else "CONDITIONAL_NOT_AUTHORIZED",
        "screen_decision": analysis.get("status"),
        "screen_root": str(root),
        "screen_protocol_sha256": sha256_file(root / "screen_protocol.json"),
        "screen_analysis_sha256": sha256_file(root / "screen_analysis.json"),
        "tasks": list(TASKS),
        "designs": list(CONTINUATION_DESIGNS),
        "seeds": list(CONTINUATION_SEEDS),
        "epochs": CONTINUATION_EPOCHS,
        "runs": 156,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_study": CONTINUATION_STUDY,
        "data": protocol.get("data"),
        "source_sha256": protocol.get("source_sha256"),
        "commands": prospective.get("commands"),
        "commands_sha256": prospective.get("commands_sha256"),
        "source_changes_allowed": False,
        "automatic_launch_performed": False,
        "scientific_gate_passed": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = (args.root if args.root.is_absolute() else ROOT / args.root).resolve()
    if args.seed != SEED or args.epochs != EPOCHS or args.study != DEFAULT_STUDY:
        raise ValueError("V15 screen seed/epochs/study are frozen")
    protocol: dict[str, Any] | None = None
    errors: list[str] = []
    protocol_errors: list[str] = []
    try:
        protocol = load_json(root / "screen_protocol.json")
        protocol_errors = validate_protocol(root, protocol)
        errors.extend(protocol_errors)
    except (IntegrityError, OSError, ValueError, TypeError) as exc:
        protocol_errors = [str(exc)]
        errors.extend(protocol_errors)
    errors.extend(validate_prospective_continuation(root))
    if protocol is None or protocol_errors:
        rows = []
    else:
        rows, row_errors = load_rows(root, protocol)
        errors.extend(row_errors)
    if len(rows) == EXPECTED_CELLS and protocol is not None:
        errors.extend(validate_runner_receipt(root, rows, protocol))
    analysis = analyze(rows, errors, study=args.study)
    rendered = json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.write:
        root.mkdir(parents=True, exist_ok=True)
        analysis_path = root / "screen_analysis.json"
        decision_path = root / "screen_decision.json"
        authorization_path = root / "conditional_authorization.json"
        for path in (analysis_path, decision_path, authorization_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        analysis_path.write_text(rendered, encoding="utf-8")
        decision = {
            "status": analysis["status"],
            "artifact_integrity_passed": analysis["artifact_integrity_passed"],
            "scientific_gate_passed": analysis["scientific_gate_passed"],
            "continue_to_100_epochs": analysis["continue_to_100_epochs"],
            "automatic_launch_performed": False,
        }
        decision_path.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        authorization = authorization_payload(root, protocol or {}, analysis)
        authorization_path.write_text(
            json.dumps(authorization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    raise SystemExit(analysis_exit_code(analysis))


if __name__ == "__main__":
    main()
