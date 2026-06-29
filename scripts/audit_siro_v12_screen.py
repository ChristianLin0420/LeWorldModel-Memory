#!/usr/bin/env python3
"""Independent, read-only integrity audit for a completed SIRO-v12 screen.

This auditor deliberately does not import the screen runner, trainer, or analyzer.  The
screen protocol is treated as a receipt whose frozen grid and manifest are independently
checked, rather than as an authority that may redefine them.  A successful audit requires
all 28 cells, their complete local artifacts, the runner completion receipts, and an
analyzer result that agrees with the independently loaded cell records.

By default the program only prints JSON.  It writes a file only when ``--output`` is
explicitly supplied, and refuses to overwrite that file.  Any absent, partial, malformed,
non-finite, or inconsistent artifact makes the audit fail closed with exit status 2.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

TASKS = (
    "cartpole.swingup",
    "fish.swim",
    "pendulum.swingup",
    "walker.walk",
)
DESIGNS = (
    "sirov12",
    "sirov12_spectralshrink",
    "sirov12_identityA",
    "sirov12_identityK",
    "sirov12_noaction",
    "sirov12_noanchor",
    "kdiov11",
)
SIRO_DESIGNS = DESIGNS[:-1]
SEED = 11_201
V11_RANKING = "rawdiff_displacement_detached"
EXPECTED_CELLS = len(TASKS) * len(DESIGNS)
CONDITIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")
COORDINATES = ("prior", "posterior", "encoder", "predictor")
SOURCE_MANIFEST = (
    "lewm/models/siro.py",
    "lewm/models/memory_model.py",
    "scripts/train_siro_v12.py",
    "scripts/run_siro_v12_screen.py",
    "scripts/analyze_siro_v12_screen.py",
    "scripts/train_hacssm_v11.py",
    "scripts/train_hacssm_v10.py",
    "scripts/hacssm_v11_data.py",
)
REQUIRED_METRICS = (
    "schema_version",
    "env",
    "design",
    "seed",
    "epochs",
    "training_objective",
    "train_data",
    "val_data",
    "train_data_sha256",
    "val_data_sha256",
    "train_episodes",
    "val_episodes",
    "length",
    "action_dim",
    "eval_target_dim",
    "final_train_loss",
    "final_val_loss",
    "val_predictive_loss",
    "heldout_prior_state_nmse",
    "clean_prior_state_nmse",
    "initial_encoder_integrator_probe_nmse",
    "loss_convergence_relative_change",
    "eval_rollout_episode",
    "eval_rollout_sha256",
)
ANALYZER_STATUSES = {
    "MECHANICS_GATE_FAIL",
    "MECHANICS_GATE_PASS_100E_NOT_LAUNCHED",
    "SCIENTIFIC_GATE_FAIL",
    "SCIENTIFIC_GATE_PASS",
}
NEGATIVE_ANALYZER_STATUS = "INCOMPLETE_OR_INVALID"
ANCHOR_RANK_THRESHOLD = 16.0


class AuditFailure(RuntimeError):
    """One closed-world integrity violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing JSON file: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"JSON root must be an object: {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise AuditFailure(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditFailure(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AuditFailure(f"{label} is not finite: {value!r}")
    return result


def _assert_finite_tree(value: Any, label: str) -> None:
    """Reject non-finite numeric leaves in histories and receipt dictionaries."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise AuditFailure(f"{label} contains a non-finite tensor")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise AuditFailure(f"{label} contains a non-finite array")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{label}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise AuditFailure(f"{label} is not finite: {value!r}")


def _equal_json_values(left: Any, right: Any) -> bool:
    """Exact recursive equality with NumPy scalar normalization."""
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equal_json_values(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _equal_json_values(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, (float, int, bool, str, type(None))) and isinstance(
            right, (float, int, bool, str, type(None))):
        return left == right
    return left == right


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if not _equal_json_values(actual, expected):
        raise AuditFailure(f"{label}={actual!r}; expected {expected!r}")


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise AuditFailure(f"manifest path escapes repository: {path}")
    return resolved


def _resolve_protocol_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditFailure(f"{label} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return _inside_root(path)


def run_name(task: str, design: str) -> str:
    suffix = f"-rank-{V11_RANKING}" if design == "kdiov11" else ""
    return f"lewm-dmc:{task}-{design}-s{SEED}{suffix}"


def expected_metrics_schema_version(design: str) -> int:
    if design not in DESIGNS:
        raise AuditFailure(f"unknown frozen design: {design!r}")
    return 1 if design in SIRO_DESIGNS else 2


def expected_run_directories(root: Path) -> dict[tuple[str, str], Path]:
    return {
        (task, design): root / run_name(task, design)
        for task in TASKS for design in DESIGNS
    }


def _parse_flag(command: Sequence[Any], flag: str) -> str:
    if not all(isinstance(value, str) for value in command):
        raise AuditFailure("protocol command contains a non-string argument")
    occurrences = [index for index, value in enumerate(command) if value == flag]
    if len(occurrences) != 1 or occurrences[0] + 1 >= len(command):
        raise AuditFailure(f"protocol command must contain exactly one {flag}")
    return str(command[occurrences[0] + 1])


def validate_protocol(root: Path) -> dict[str, Any]:
    protocol = _load_json(root / "screen_protocol.json")
    exact = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v12_screen_after_failed_v11",
        "seed": SEED,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "runs": EXPECTED_CELLS,
        "v11_comparator_action_ranking": V11_RANKING,
        "automatic_100_epoch_launch_in_this_process": False,
    }
    for key, expected in exact.items():
        _require_equal(protocol.get(key), expected, f"protocol.{key}")
    epochs = protocol.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise AuditFailure(f"protocol.epochs must be a positive integer: {epochs!r}")
    study = protocol.get("study")
    entity = protocol.get("wandb_entity")
    project = protocol.get("wandb_project")
    for value, label in ((study, "study"), (entity, "wandb_entity"),
                         (project, "wandb_project")):
        if not isinstance(value, str) or not value:
            raise AuditFailure(f"protocol.{label} must be a non-empty string")
    _require_equal(entity, "crlc112358", "protocol.wandb_entity")
    _require_equal(project, "lewm-memory-popgym", "protocol.wandb_project")

    gpus = protocol.get("gpus")
    pinned = protocol.get("task_pinned_gpu")
    if (not isinstance(gpus, list) or len(gpus) != len(TASKS)
            or len(set(map(str, gpus))) != len(TASKS)):
        raise AuditFailure("protocol must list four distinct GPUs")
    if not isinstance(pinned, dict) or set(pinned) != set(TASKS):
        raise AuditFailure("protocol task_pinned_gpu does not cover the frozen task grid")
    if set(map(str, pinned.values())) != set(map(str, gpus)):
        raise AuditFailure("protocol task/GPU pinning is not a bijection")

    source_hashes = protocol.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise AuditFailure("protocol.source_sha256 must be an object")
    _require_equal(set(source_hashes), set(SOURCE_MANIFEST), "protocol source manifest")
    for relative, expected_hash in source_hashes.items():
        path = _inside_root(ROOT / relative)
        if not path.is_file():
            raise AuditFailure(f"source manifest file is missing: {relative}")
        actual_hash = sha256_file(path)
        _require_equal(actual_hash, expected_hash, f"source hash {relative}")

    data = protocol.get("data")
    if not isinstance(data, dict) or tuple(data) != TASKS:
        raise AuditFailure("protocol data manifest does not match the frozen task order")
    for task in TASKS:
        receipt = data[task]
        if not isinstance(receipt, dict):
            raise AuditFailure(f"protocol.data.{task} must be an object")
        for split in ("train", "val"):
            path = _resolve_protocol_path(receipt.get(split), f"data.{task}.{split}")
            if not path.is_file():
                raise AuditFailure(f"data file is missing: {path}")
            _require_equal(
                sha256_file(path), receipt.get(f"{split}_sha256"),
                f"data hash {task}/{split}")

    commands = protocol.get("commands")
    if not isinstance(commands, dict) or tuple(commands) != TASKS:
        raise AuditFailure("protocol commands do not match the frozen task order")
    for task in TASKS:
        task_commands = commands[task]
        if not isinstance(task_commands, list) or len(task_commands) != len(DESIGNS):
            raise AuditFailure(f"protocol commands for {task} do not contain seven cells")
        for design, command in zip(DESIGNS, task_commands, strict=True):
            if not isinstance(command, list):
                raise AuditFailure(f"protocol command {task}/{design} is not a list")
            expected_flags = {
                "--memory-mode": design,
                "--seed": str(SEED),
                "--epochs": str(epochs),
                "--wandb-entity": entity,
                "--wandb-project": project,
                "--wandb-mode": "online",
                "--wandb-study": study,
            }
            for flag, expected in expected_flags.items():
                _require_equal(
                    _parse_flag(command, flag), expected,
                    f"protocol command {task}/{design} {flag}")
            if "--wandb" not in command:
                raise AuditFailure(f"protocol command {task}/{design} omits --wandb")
            train_path = _resolve_protocol_path(
                _parse_flag(command, "--train-data"), "command train data")
            val_path = _resolve_protocol_path(
                _parse_flag(command, "--val-data"), "command val data")
            _require_equal(
                train_path,
                _resolve_protocol_path(data[task]["train"], "protocol train data"),
                f"protocol command {task}/{design} train data")
            _require_equal(
                val_path,
                _resolve_protocol_path(data[task]["val"], "protocol val data"),
                f"protocol command {task}/{design} val data")
            output_root = Path(_parse_flag(command, "--output-dir")).resolve()
            _require_equal(output_root, root.resolve(), f"protocol command {task}/{design} root")
    return protocol


def validate_root_directory_set(root: Path) -> None:
    expected = {path.name for path in expected_run_directories(root).values()}
    actual = {path.name for path in root.iterdir() if path.is_dir()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(actual) != EXPECTED_CELLS:
        raise AuditFailure(
            f"top-level run-directory grid is not exactly 28 cells; "
            f"missing={missing}, extra={extra}, found={len(actual)}")


def validate_metrics(
        path: Path, *, task: str, design: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _load_json(path)
    missing = sorted(set(REQUIRED_METRICS) - set(metrics))
    if missing:
        raise AuditFailure(f"{task}/{design} metrics missing fields: {missing}")
    _assert_finite_tree(metrics, f"{task}/{design}.metrics")
    expected = {
        "schema_version": expected_metrics_schema_version(design),
        "env": f"dmc:{task}",
        "design": design,
        "seed": SEED,
        "epochs": protocol["epochs"],
        "train_episodes": 1_200,
        "val_episodes": 240,
        "length": 48,
        "eval_rollout_episode": 0,
    }
    for key, value in expected.items():
        _require_equal(metrics.get(key), value, f"{task}/{design}.metrics.{key}")
    data = protocol["data"][task]
    for split in ("train", "val"):
        expected_path = _resolve_protocol_path(data[split], f"{task}/{split}")
        actual_path = _resolve_protocol_path(
            metrics.get(f"{split}_data"), f"{task}/{design}.{split}_data")
        _require_equal(actual_path, expected_path, f"{task}/{design}.{split}_data")
        _require_equal(
            metrics.get(f"{split}_data_sha256"), data[f"{split}_sha256"],
            f"{task}/{design}.{split}_data_sha256")
    for key in (
            "final_train_loss", "final_val_loss", "val_predictive_loss",
            "heldout_prior_state_nmse", "clean_prior_state_nmse",
            "initial_encoder_integrator_probe_nmse",
            "loss_convergence_relative_change"):
        _finite_number(metrics[key], f"{task}/{design}.metrics.{key}")
    action_dim = metrics.get("action_dim")
    target_dim = metrics.get("eval_target_dim")
    if not isinstance(action_dim, int) or action_dim < 1:
        raise AuditFailure(f"{task}/{design}: invalid action_dim {action_dim!r}")
    if not isinstance(target_dim, int) or target_dim < 1:
        raise AuditFailure(f"{task}/{design}: invalid eval_target_dim {target_dim!r}")
    if design in SIRO_DESIGNS:
        for key, value in {
            "identified_operator_fit": True,
            "fit_gradient_active": False,
            "fit_updates": protocol["epochs"] + 1,
            "memory_arch_schema_version": 12,
            "memory_architecture": "stable_identified_residual_observer",
        }.items():
            _require_equal(metrics.get(key), value, f"{task}/{design}.metrics.{key}")
    else:
        _require_equal(
            metrics.get("development_action_ranking"), V11_RANKING,
            f"{task}/{design}.metrics.development_action_ranking")
        _require_equal(
            metrics.get("memory_arch_schema_version"), 11,
            f"{task}/{design}.metrics.memory_arch_schema_version")
    return metrics


def _expected_rollout_keys() -> set[str]:
    result = {"schema_version", "episode_index", "conditions"}
    condition_fields = {
        "target_times", "phase", "gap_start", "gap_end", "observed_rgb",
        "clean_rgb", "actions", "evaluation_target",
    }
    for coordinate in COORDINATES:
        condition_fields.add(f"{coordinate}_state_prediction")
        condition_fields.add(f"{coordinate}_state_nmse_by_target_t")
    for condition in CONDITIONS:
        result.update(f"{condition}_{key}" for key in condition_fields)
    result.update({"condition", "target_times", "phase", "state_target"})
    for coordinate in COORDINATES:
        result.add(f"{coordinate}_state_prediction")
        result.add(f"{coordinate}_state_nmse")
    return result


def _array_equal(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.dtype.kind == "O":
        raise AuditFailure(f"{label} uses forbidden object dtype")
    if not np.array_equal(actual, expected):
        raise AuditFailure(f"{label} is inconsistent with condition arrays")


def validate_rollout(path: Path, metrics: Mapping[str, Any], label: str) -> str:
    if not path.is_file():
        raise AuditFailure(f"{label}: missing rollout NPZ")
    rollout_hash = sha256_file(path)
    _require_equal(metrics["eval_rollout_sha256"], rollout_hash, f"{label} rollout hash")
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AuditFailure(f"{label}: cannot load rollout NPZ: {exc}") from exc
    with archive_context as archive:
        actual_keys = set(archive.files)
        expected_keys = _expected_rollout_keys()
        if actual_keys != expected_keys:
            raise AuditFailure(
                f"{label}: rollout schema keys differ; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}")
        arrays = {key: archive[key] for key in archive.files}

    for key, array in arrays.items():
        if array.dtype.kind == "O":
            raise AuditFailure(f"{label}: rollout {key} uses forbidden object dtype")
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise AuditFailure(f"{label}: rollout {key} contains non-finite values")
    _require_equal(arrays["schema_version"].shape, (), f"{label} schema scalar shape")
    _require_equal(int(arrays["schema_version"]), 2, f"{label} rollout schema")
    _require_equal(arrays["episode_index"].shape, (), f"{label} episode scalar shape")
    _require_equal(
        int(arrays["episode_index"]), metrics["eval_rollout_episode"],
        f"{label} rollout episode")
    _array_equal(arrays["conditions"], np.asarray(CONDITIONS), f"{label}.conditions")

    length = int(metrics["length"])
    action_dim = int(metrics["action_dim"])
    target_dim = int(metrics["eval_target_dim"])
    history_len = 3
    target_count = length - history_len
    target_times = np.arange(history_len, length, dtype=np.int64)
    flat: dict[str, list[np.ndarray]] = {
        "condition": [], "target_times": [], "phase": [], "state_target": []}
    for coordinate in COORDINATES:
        flat[f"{coordinate}_state_prediction"] = []
        flat[f"{coordinate}_state_nmse"] = []
    for condition in CONDITIONS:
        prefix = f"{condition}_"
        expected_shapes = {
            "target_times": (target_count,),
            "phase": (target_count,),
            "gap_start": (),
            "gap_end": (),
            "observed_rgb": (length, 64, 64, 3),
            "clean_rgb": (length, 64, 64, 3),
            "actions": (length - 1, action_dim),
            "evaluation_target": (target_count, target_dim),
        }
        for coordinate in COORDINATES:
            expected_shapes[f"{coordinate}_state_prediction"] = (
                target_count, target_dim)
            expected_shapes[f"{coordinate}_state_nmse_by_target_t"] = (target_count,)
        for key, shape in expected_shapes.items():
            _require_equal(arrays[prefix + key].shape, shape, f"{label}.{prefix}{key}.shape")
        if arrays[prefix + "observed_rgb"].dtype != np.uint8:
            raise AuditFailure(f"{label}.{prefix}observed_rgb must be uint8")
        if arrays[prefix + "clean_rgb"].dtype != np.uint8:
            raise AuditFailure(f"{label}.{prefix}clean_rgb must be uint8")
        _array_equal(arrays[prefix + "target_times"], target_times,
                     f"{label}.{prefix}target_times")
        gap_start = int(arrays[prefix + "gap_start"])
        gap_end = int(arrays[prefix + "gap_end"])
        if not history_len <= gap_start < gap_end <= length:
            raise AuditFailure(f"{label}.{condition}: invalid gap [{gap_start}, {gap_end})")
        phases = arrays[prefix + "phase"]
        if phases.dtype.kind not in "US" or not set(map(str, phases)).issubset(
                {"context", "gap", "deep", "first_post", "post"}):
            raise AuditFailure(f"{label}.{condition}: malformed phase labels")
        flat["condition"].append(np.full(target_count, condition))
        flat["target_times"].append(arrays[prefix + "target_times"])
        flat["phase"].append(phases)
        flat["state_target"].append(arrays[prefix + "evaluation_target"])
        for coordinate in COORDINATES:
            flat[f"{coordinate}_state_prediction"].append(
                arrays[prefix + f"{coordinate}_state_prediction"])
            flat[f"{coordinate}_state_nmse"].append(
                arrays[prefix + f"{coordinate}_state_nmse_by_target_t"])
    for key, pieces in flat.items():
        _array_equal(arrays[key], np.concatenate(pieces), f"{label}.{key}")
    return rollout_hash


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"{label}: missing model.pt")
    try:
        # These checkpoints are locally generated artifacts covered by the frozen source
        # and data receipts above.  They contain NumPy probe payloads that PyTorch's
        # weights-only loader does not support.
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch raises several deserialization-specific classes
        raise AuditFailure(f"{label}: cannot load model.pt: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"{label}: model.pt root must be a dictionary")
    return value


def _validate_history(history: Any, epochs: int, label: str, *, siro: bool) -> None:
    if not isinstance(history, list) or len(history) != epochs:
        raise AuditFailure(f"{label}: checkpoint history must contain {epochs} epochs")
    for expected_epoch, row in enumerate(history, 1):
        if not isinstance(row, dict):
            raise AuditFailure(f"{label}: history row {expected_epoch} is not an object")
        _require_equal(row.get("epoch"), expected_epoch, f"{label}.history epoch")
        for key in ("train", "val"):
            if not isinstance(row.get(key), dict) or not row[key]:
                raise AuditFailure(f"{label}: history row {expected_epoch} lacks {key}")
        if siro and not isinstance(row.get("fit"), dict):
            raise AuditFailure(f"{label}: SIRO history row {expected_epoch} lacks fit receipt")
        _assert_finite_tree(row, f"{label}.history[{expected_epoch - 1}]")


def _tensor_shape_finite(value: Any, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise AuditFailure(f"{label} must be a tensor")
    if tuple(value.shape) != shape:
        raise AuditFailure(f"{label} has shape {tuple(value.shape)}; expected {shape}")
    if not bool(torch.isfinite(value).all()):
        raise AuditFailure(f"{label} contains non-finite values")
    return value


def _validate_siro_checkpoint(
        payload: Mapping[str, Any], metrics: Mapping[str, Any], epochs: int,
        label: str) -> None:
    fit_history = payload.get("fit_history")
    if not isinstance(fit_history, list) or len(fit_history) != epochs + 1:
        raise AuditFailure(f"{label}: fit_history must contain fit0 plus {epochs} refits")
    for fit_index, row in enumerate(fit_history):
        if not isinstance(row, dict):
            raise AuditFailure(f"{label}: fit_history[{fit_index}] is not an object")
        _require_equal(row.get("fit_index"), fit_index, f"{label}.fit_history index")
        receipts = row.get("receipts")
        if not isinstance(receipts, dict):
            raise AuditFailure(f"{label}: fit_history[{fit_index}] lacks receipts")
        _require_equal(
            receipts.get("siro_fit_fit_index"), fit_index,
            f"{label}.fit_history receipt index")
        _require_equal(
            receipts.get("siro_fit_fit_finite"), True,
            f"{label}.fit_history fit_finite")
        _assert_finite_tree(row, f"{label}.fit_history[{fit_index}]")

    initial = payload.get("initial_fit_receipts")
    final_fit = payload.get("final_operator_fit")
    if not isinstance(initial, dict) or not isinstance(final_fit, dict):
        raise AuditFailure(f"{label}: missing SIRO initial/final fit payload")
    _require_equal(initial.get("fit_index"), 0, f"{label}.initial fit index")
    receipts = final_fit.get("receipts")
    if not isinstance(receipts, dict):
        raise AuditFailure(f"{label}: final operator fit lacks receipts")
    _require_equal(receipts.get("fit_index"), epochs, f"{label}.final fit index")
    _require_equal(receipts.get("fit_finite"), True, f"{label}.final fit finite")
    _assert_finite_tree(initial, f"{label}.initial_fit_receipts")
    _assert_finite_tree(receipts, f"{label}.final_fit_receipts")
    for key, value in receipts.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            prefixed = f"siro_fit_{key}"
            _require_equal(
                fit_history[-1]["receipts"].get(prefixed), value,
                f"{label}.fit history final receipt {prefixed}")
            _require_equal(metrics.get(prefixed), value, f"{label}.metrics.{prefixed}")

    embed_dim = int(payload["args"].get("embed_dim", -1))
    action_dim = int(metrics["action_dim"])
    if embed_dim != 128:
        raise AuditFailure(f"{label}: checkpoint embed_dim={embed_dim}; expected 128")
    shapes = {
        "identified_A": (embed_dim, embed_dim),
        "raw_A": (embed_dim, embed_dim),
        "action_B": (embed_dim, action_dim),
        "drift_b": (embed_dim,),
        "action_B0": (embed_dim, action_dim),
        "action_B1": (embed_dim, action_dim),
        "action_mean": (action_dim,),
        "action_std": (action_dim,),
        "Qa": (action_dim, action_dim),
        "Qeps": (embed_dim, embed_dim),
        "signal_S": (embed_dim, embed_dim),
        "noise_N": (embed_dim, embed_dim),
        "reachability_W": (embed_dim, embed_dim),
        "age_J": (embed_dim, embed_dim),
        "action_read_R": (embed_dim, embed_dim),
        "lmmse_K": (embed_dim, embed_dim),
        "clean_innovation_mean": (embed_dim,),
        "observed_innovation_mean": (embed_dim,),
    }
    tensors = {
        key: _tensor_shape_finite(final_fit.get(key), shape, f"{label}.fit.{key}")
        for key, shape in shapes.items()
    }
    model_state = payload["model_state_dict"]
    state_prefix = "world.mem_sirov12."
    for key in (
            "identified_A", "action_B", "drift_b", "action_read_R", "lmmse_K",
            "clean_innovation_mean", "observed_innovation_mean"):
        state_value = model_state.get(state_prefix + key)
        if not isinstance(state_value, torch.Tensor) or not torch.equal(
                state_value.cpu(), tensors[key].cpu()):
            raise AuditFailure(f"{label}: serialized memory {key} differs from final fit")
    fit_updates = model_state.get(state_prefix + "fit_updates")
    installed = model_state.get(state_prefix + "operators_installed")
    if not isinstance(fit_updates, torch.Tensor) or int(fit_updates) != epochs + 1:
        raise AuditFailure(f"{label}: serialized fit_updates is not {epochs + 1}")
    if not isinstance(installed, torch.Tensor) or not bool(installed):
        raise AuditFailure(f"{label}: serialized operators_installed is false")
    extra = model_state.get(state_prefix + "_extra_state")
    if (not isinstance(extra, dict)
            or not _equal_json_values(extra.get("fit_receipts"), receipts)):
        raise AuditFailure(f"{label}: serialized extra-state receipts differ from final fit")


def validate_checkpoint(
        path: Path, metrics: Mapping[str, Any], *, task: str, design: str,
        protocol: Mapping[str, Any]) -> None:
    label = f"{task}/{design}"
    payload = _load_checkpoint(path, label)
    required = {
        "model_state_dict", "args", "final_metrics", "history", "state_probes",
        "inverse_action_probe",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise AuditFailure(f"{label}: checkpoint missing payloads: {missing}")
    model_state = payload["model_state_dict"]
    args = payload["args"]
    if not isinstance(model_state, dict) or not model_state:
        raise AuditFailure(f"{label}: model_state_dict is absent or empty")
    if not isinstance(args, dict):
        raise AuditFailure(f"{label}: checkpoint args is not an object")
    for key, value in model_state.items():
        if isinstance(value, torch.Tensor):
            _assert_finite_tree(value, f"{label}.model_state_dict.{key}")
    expected_args = {
        "memory_mode": design,
        "seed": SEED,
        "epochs": protocol["epochs"],
        "wandb": True,
        "wandb_entity": protocol["wandb_entity"],
        "wandb_project": protocol["wandb_project"],
        "wandb_mode": "online",
        "wandb_study": protocol["study"],
        "eval_rollout_episode": 0,
    }
    for key, value in expected_args.items():
        _require_equal(args.get(key), value, f"{label}.checkpoint.args.{key}")
    if design == "kdiov11":
        _require_equal(
            args.get("development_action_ranking"), V11_RANKING,
            f"{label}.checkpoint comparator ranking")
    if not _equal_json_values(payload["final_metrics"], metrics):
        raise AuditFailure(f"{label}: checkpoint final_metrics differs from metrics.json")
    _validate_history(
        payload["history"], protocol["epochs"], label, siro=design in SIRO_DESIGNS)
    if design in SIRO_DESIGNS:
        _validate_siro_checkpoint(payload, metrics, protocol["epochs"], label)
    else:
        forbidden = {"fit_history", "initial_fit_receipts", "final_operator_fit"}
        present = sorted(forbidden & set(payload))
        if present:
            raise AuditFailure(f"{label}: V11 comparator unexpectedly has SIRO fits: {present}")
    del payload
    gc.collect()


def validate_wandb(
        path: Path, *, task: str, design: str, protocol: Mapping[str, Any],
        rollout_hash: str) -> dict[str, Any]:
    label = f"{task}/{design}"
    receipt = _load_json(path)
    expected = {
        "entity": protocol["wandb_entity"],
        "project": protocol["wandb_project"],
        "mode": "online",
        "study": protocol["study"],
        "state": "finished",
        "eval_rollout_episode": 0,
        "eval_rollout_sha256": rollout_hash,
    }
    for key, value in expected.items():
        _require_equal(receipt.get(key), value, f"{label}.wandb.{key}")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id or any(char.isspace() for char in run_id):
        raise AuditFailure(f"{label}: invalid W&B run ID {run_id!r}")
    expected_name = f"{protocol['study']}-{run_name(task, design)}"
    _require_equal(receipt.get("run_name"), expected_name, f"{label}.wandb.run_name")
    _require_equal(
        receipt.get("eval_rollout_artifact_name"), f"eval-rollout-{run_id}",
        f"{label}.wandb artifact name")
    expected_url = (
        f"https://wandb.ai/{protocol['wandb_entity']}/"
        f"{protocol['wandb_project']}/runs/{run_id}")
    _require_equal(receipt.get("url"), expected_url, f"{label}.wandb.url")
    if receipt.get("schema_version") not in (1, 2):
        raise AuditFailure(f"{label}: unsupported W&B receipt schema")
    return receipt


def validate_cell(
        directory: Path, *, task: str, design: str,
        protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not directory.is_dir():
        raise AuditFailure(f"missing run directory: {directory}")
    required_files = {"model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json"}
    files = {path.name for path in directory.iterdir() if path.is_file()}
    missing = sorted(required_files - files)
    if missing:
        raise AuditFailure(f"{task}/{design}: missing run artifacts: {missing}")
    metrics = validate_metrics(
        directory / "metrics.json", task=task, design=design, protocol=protocol)
    rollout_hash = validate_rollout(
        directory / "eval_rollout.npz", metrics, f"{task}/{design}")
    validate_checkpoint(
        directory / "model.pt", metrics, task=task, design=design, protocol=protocol)
    wandb = validate_wandb(
        directory / "wandb_run.json", task=task, design=design,
        protocol=protocol, rollout_hash=rollout_hash)
    return {
        "task": task,
        "design": design,
        "directory": str(directory),
        "heldout_prior_state_nmse": metrics["heldout_prior_state_nmse"],
        "clean_prior_state_nmse": metrics["clean_prior_state_nmse"],
        "anchor_covariance_effective_rank": (
            _finite_number(
                metrics.get("siro_fit_anchor_covariance_effective_rank"),
                f"{task}/{design}.anchor covariance effective rank")
            if design in SIRO_DESIGNS else None),
        "rollout_sha256": rollout_hash,
        "wandb_run_id": wandb["run_id"],
        "wandb_url": wandb["url"],
    }


def validate_runner_receipt(
        root: Path, rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> None:
    if (root / ".siro_v12_screen.lock").exists():
        raise AuditFailure("screen lock still exists; the runner has not completed")
    path = root / "screen_runs.json"
    if not path.is_file():
        raise AuditFailure("missing runner completion receipt screen_runs.json")
    try:
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read screen_runs.json: {exc}") from exc
    if not isinstance(records, list) or len(records) != EXPECTED_CELLS:
        raise AuditFailure("screen_runs.json must contain exactly 28 records")
    expected_pairs = {(row["task"], row["design"]) for row in rows}
    actual_pairs: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AuditFailure(f"screen_runs[{index}] is not an object")
        pair = (record.get("task"), record.get("design"))
        if pair in actual_pairs:
            raise AuditFailure(f"duplicate runner completion record: {pair}")
        actual_pairs.add(pair)
        if pair not in expected_pairs:
            raise AuditFailure(f"unexpected runner completion record: {pair}")
        task, design = pair
        _require_equal(
            str(record.get("gpu")), str(protocol["task_pinned_gpu"][task]),
            f"screen_runs {task}/{design} GPU")
        if _finite_number(record.get("seconds"), f"screen_runs {task}/{design} seconds") <= 0:
            raise AuditFailure(f"screen_runs {task}/{design} has non-positive runtime")
        expected_metrics = str(root / run_name(task, design) / "metrics.json")
        _require_equal(record.get("metrics"), expected_metrics,
                       f"screen_runs {task}/{design} metrics path")
        log = record.get("log")
        if not isinstance(log, str) or not log:
            raise AuditFailure(f"screen_runs {task}/{design} lacks a log path")
    _require_equal(actual_pairs, expected_pairs, "runner completion cell set")


def _design_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for design in DESIGNS:
        selected = [row for row in rows if row["design"] == design]
        if len(selected) != len(TASKS):
            raise AuditFailure(f"cannot compute complete independent mean for {design}")
        result[design] = {
            "heldout_prior_state_nmse": float(np.mean([
                row["heldout_prior_state_nmse"] for row in selected], dtype=np.float64)),
            "clean_prior_state_nmse": float(np.mean([
                row["clean_prior_state_nmse"] for row in selected], dtype=np.float64)),
        }
    return result


def _close(actual: Any, expected: float, label: str) -> None:
    value = _finite_number(actual, label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise AuditFailure(f"{label}={value}; independent value is {expected}")


def representation_rank_failures(
        rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Recompute the frozen analyzer's anchor-rank representation gate exactly."""
    failures = []
    for row in rows:
        if row["design"] not in SIRO_DESIGNS:
            continue
        rank = _finite_number(
            row.get("anchor_covariance_effective_rank"),
            f"{row['task']}/{row['design']}.anchor covariance effective rank")
        if rank < ANCHOR_RANK_THRESHOLD:
            failures.append(
                f"{row['task']}/{row['design']}: "
                "fit anchor effective rank below 16")
    return failures


def validate_analyzer_output(
        root: Path, rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
        representation_failures: Sequence[str]) -> str:
    analysis = _load_json(root / "screen_analysis.json")
    common = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "study": protocol["study"],
        "seed": SEED,
        "epochs": protocol["epochs"],
        "expected_cells": EXPECTED_CELLS,
        "official_result": False,
        "iclr_confirmation": False,
    }
    for key, value in common.items():
        _require_equal(analysis.get(key), value, f"screen_analysis.{key}")
    status = analysis.get("status")
    if representation_failures:
        # The frozen analyzer historically labels representation-gate exclusions as
        # integrity errors and therefore omits its aggregate tables.  Preserve that
        # immutable negative receipt, but do not conflate it with artifact corruption.
        expected_negative = {
            "status": NEGATIVE_ANALYZER_STATUS,
            "completed_cells": EXPECTED_CELLS - len(representation_failures),
            "integrity_passed": False,
            "integrity_errors": list(representation_failures),
            "continue_to_100_epochs": False,
            "scientific_gate_passed": False,
        }
        for key, value in expected_negative.items():
            _require_equal(analysis.get(key), value, f"screen_analysis.{key}")
    else:
        expected_complete = {
            "completed_cells": EXPECTED_CELLS,
            "integrity_passed": True,
            "integrity_errors": [],
        }
        for key, value in expected_complete.items():
            _require_equal(analysis.get(key), value, f"screen_analysis.{key}")
        if status not in ANALYZER_STATUSES:
            raise AuditFailure(f"screen_analysis has non-complete status: {status!r}")
        means = analysis.get("design_means")
        if not isinstance(means, dict) or set(means) != set(DESIGNS):
            raise AuditFailure("screen_analysis design means do not cover seven designs")
        for design, values in _design_means(rows).items():
            if not isinstance(means[design], dict):
                raise AuditFailure(f"screen_analysis mean for {design} is not an object")
            for metric, expected_value in values.items():
                _close(means[design].get(metric), expected_value,
                       f"screen_analysis.{design}.{metric}")

        analyzer_wandb = analysis.get("wandb_runs")
        if not isinstance(analyzer_wandb, list) or len(analyzer_wandb) != EXPECTED_CELLS:
            raise AuditFailure("screen_analysis W&B list must contain 28 cells")
        expected_wandb = {
            (row["task"], row["design"], row["wandb_run_id"], row["wandb_url"])
            for row in rows}
        actual_wandb = set()
        for receipt in analyzer_wandb:
            if not isinstance(receipt, dict):
                raise AuditFailure("screen_analysis W&B record is not an object")
            actual_wandb.add((
                receipt.get("task"), receipt.get("design"), receipt.get("run_id"),
                receipt.get("url")))
        _require_equal(actual_wandb, expected_wandb, "screen_analysis W&B identities")

    decision = _load_json(root / "screen_decision.json")
    decision_pairs = {
        "status": "status",
        "integrity_passed": "integrity_passed",
        "continue_to_100_epochs": "continue_to_100_epochs",
        "scientific_gate_passed": "scientific_gate_passed",
    }
    for decision_key, analysis_key in decision_pairs.items():
        _require_equal(
            decision.get(decision_key), analysis.get(analysis_key),
            f"screen_decision.{decision_key}")
    _require_equal(
        decision.get("automatic_launch_performed"), False,
        "screen_decision.automatic_launch_performed")
    return str(status)


def audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    protocol: dict[str, Any] | None = None
    analyzer_status: str | None = None
    analyzer_receipt_consistent = False

    if not root.is_dir():
        errors.append(f"screen root is not a directory: {root}")
    else:
        try:
            protocol = validate_protocol(root)
        except (AuditFailure, OSError, ValueError) as exc:
            errors.append(str(exc))
        try:
            validate_root_directory_set(root)
        except (AuditFailure, OSError, ValueError) as exc:
            errors.append(str(exc))

    if protocol is not None:
        for (task, design), directory in expected_run_directories(root).items():
            try:
                rows.append(validate_cell(
                    directory, task=task, design=design, protocol=protocol))
            except (AuditFailure, OSError, ValueError) as exc:
                errors.append(str(exc))
        run_ids = [row["wandb_run_id"] for row in rows]
        if len(run_ids) != len(set(run_ids)):
            errors.append("completed cells contain duplicate W&B run IDs")
        if len(rows) == EXPECTED_CELLS and not errors:
            try:
                validate_runner_receipt(root, rows, protocol)
            except (AuditFailure, OSError, ValueError) as exc:
                errors.append(str(exc))
        artifact_integrity_passed = len(rows) == EXPECTED_CELLS and not errors
        rank_failures = (
            representation_rank_failures(rows) if len(rows) == EXPECTED_CELLS else [])
        if artifact_integrity_passed:
            try:
                analyzer_status = validate_analyzer_output(
                    root, rows, protocol, rank_failures)
                analyzer_receipt_consistent = True
            except (AuditFailure, OSError, ValueError) as exc:
                errors.append(str(exc))
    else:
        artifact_integrity_passed = False
        rank_failures = []

    # Analyzer-receipt disagreement fails the audit but does not retroactively describe
    # the 28 independently checked model/data/rollout chains as corrupt.
    artifact_integrity_passed = bool(
        protocol is not None and artifact_integrity_passed)
    representation_gate_passed = (
        not rank_failures if artifact_integrity_passed else None)
    passed = artifact_integrity_passed and analyzer_receipt_consistent and not errors
    if passed:
        status = (
            "PASS_COMPLETE" if representation_gate_passed
            else "PASS_COMPLETE_NEGATIVE")
    else:
        status = "FAIL_CLOSED"
    return {
        "schema_version": 1,
        "scope": "independent_read_only_siro_v12_screen_audit",
        "root": str(root),
        "status": status,
        "passed": passed,
        "artifact_integrity_passed": artifact_integrity_passed,
        "representation_gate_passed": representation_gate_passed,
        "representation_failures": rank_failures,
        "analyzer_receipt_consistent": analyzer_receipt_consistent,
        "expected_cells": EXPECTED_CELLS,
        "validated_cells": len(rows),
        "protocol_validated": protocol is not None,
        "analyzer_status": analyzer_status,
        "errors": errors,
        "cells": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/hacssm_v12_screen_siro30"))
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional explicit JSON receipt path; must not already exist")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root if args.root.is_absolute() else (ROOT / args.root).resolve()
    report = audit(root)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = args.output if args.output.is_absolute() else (ROOT / args.output).resolve()
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
