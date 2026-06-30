#!/usr/bin/env python3
"""Independent read-only integrity and gate audit for the CF-EBO-v14 screen."""

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


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TASKS = ("cartpole.swingup", "fish.swim", "pendulum.swingup", "walker.walk")
DESIGNS = (
    "cfebov14", "cfebov14_nocorrect", "cfebov14_noaction", "cfebov14_norisk",
    "cfebov14_noenergycap", "cfebov14_noradial", "cfhirov13_nocorrect",
    "ssm", "hacssmv8", "kdiov11",
)
CF_DESIGNS = DESIGNS[:6]
CONTROLS = CF_DESIGNS[1:]
EXTERNAL = DESIGNS[6:]
SEED = 14_001
EPOCHS = 30
EXPECTED_CELLS = 40
STUDY = "hacssm-v14-screen-cfebo30"
ENTITY = "crlc112358"
PROJECT = "lewm-memory-popgym"
V11_RANKING = "rawdiff_displacement_detached"
PRIMARY = "heldout_prior_state_nmse"
LOCK_NAME = ".cf_ebo_v14_screen.lock"
CONDITIONS = (
    "clean", "val_train_view", "freeze", "gaussian_noise", "checkerboard",
    "long_freeze")
FROZEN_PYTHON = str(ROOT / ".venv" / "bin" / "python")
TRAIN_SCRIPT = str(ROOT / "scripts" / "train_cf_ebo_v14.py")
CONTINUATION_DESIGNS = (
    "cfebov14", "cfebov14_nocorrect", "cfebov14_noaction", "cfebov14_norisk",
    "cfhirov13_nocorrect", "ssm", "hacssmv8", "kdiov11",
)
CONTINUATION_SEEDS = (14_002, 14_003, 14_004)
CONTINUATION_STUDY = "hacssm-v14-continuation-cfebo100"
CONTINUATION_ROOT = (ROOT / "outputs/hacssm_v14_continuation_cfebo100").resolve()
SOURCE_MANIFEST = (
    "lewm/models/cf_ebo.py",
    "lewm/models/cf_hiro.py",
    "lewm/models/memory_model.py",
    "lewm/models/memory.py",
    "lewm/models/leworldmodel.py",
    "lewm/models/encoder.py",
    "scripts/train_cf_ebo_v14.py",
    "scripts/run_cf_ebo_v14_screen.py",
    "scripts/analyze_cf_ebo_v14_screen.py",
    "scripts/audit_cf_ebo_v14_screen.py",
    "scripts/train_cf_hiro_v13.py",
    "scripts/train_siro_v12.py",
    "scripts/train_hacssm_v11.py",
    "scripts/train_hacssm_v10.py",
    "scripts/hacssm_v11_data.py",
)
FIT_BUFFERS = (
    "state_matrix", "action_matrix", "raw_action_matrix", "read_matrix",
    "correction_matrix", "raw_correction_matrix", "innovation_covariance",
    "innovation_whitener", "initial_map", "output_projector",
    "complement_projector", "energy_support_projector", "output_mean",
    "action_mean", "action_reliability", "correction_reliability", "innovation_rank",
)


class AuditFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"JSON root must be an object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AuditFailure(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditFailure(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AuditFailure(f"{label} is not finite: {value!r}")
    return result


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise AuditFailure(f"{label} contains a non-finite tensor")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            raise AuditFailure(f"{label} contains a non-finite array")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AuditFailure(f"{label} contains {value!r}")


def _deep_equal(first: Any, second: Any) -> bool:
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


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditFailure(f"{label}={actual!r}; expected {expected!r}")


def _resolve(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditFailure(f"invalid manifest path {value!r}")
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise AuditFailure(f"manifest path escapes repository: {value}")
    return resolved


def run_name(task: str, design: str) -> str:
    suffix = f"-rank-{V11_RANKING}" if design == "kdiov11" else ""
    return f"lewm-dmc:{task}-{design}-s{SEED}{suffix}"


def expected_data_path(task: str, split: str) -> Path:
    if split == "train":
        episodes, seed = 1_200, 37_100
    elif split == "val":
        episodes, seed = 240, 103_710
    else:
        raise AuditFailure(f"unknown split {split!r}")
    safe = f"dmc_{task.replace('.', '_')}"
    return Path("outputs/hacssm_v11_data") / (
        f"{safe}_{split}_n{episodes}_L48_s64_seed{seed}.npz")


def expected_train_command(
        output_root: Path, study: str, epochs: int, task: str, design: str,
        *, seed: int = SEED) -> list[str]:
    """Independent exact reconstruction of the frozen runner command."""
    return [
        FROZEN_PYTHON,
        TRAIN_SCRIPT,
        "--train-data", str(expected_data_path(task, "train")),
        "--val-data", str(expected_data_path(task, "val")),
        "--memory-mode", design,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output-dir", str(output_root.resolve()),
        "--batch-size", "64",
        "--lr", "0.0003",
        "--weight-decay", "0.00001",
        "--num-workers", "2",
        "--img-size", "64",
        "--patch-size", "8",
        "--embed-dim", "128",
        "--encoder-layers", "6",
        "--encoder-heads", "4",
        "--predictor-layers", "4",
        "--predictor-heads", "8",
        "--history-len", "3",
        "--dropout", "0.1",
        "--sigreg-lambda", "0.1",
        "--sigreg-projections", "512",
        "--probe-ridge", "0.001",
        "--eval-target-key", "task_observation",
        "--corruption-seed", "11012",
        "--eval-rollout-episode", "0",
        "--device", "cuda",
        "--wandb",
        "--wandb-entity", ENTITY,
        "--wandb-project", PROJECT,
        "--wandb-mode", "online",
        "--wandb-study", study,
        "--extra-tag", "excluded-adaptive-screen,cf-ebo-v14",
    ]


def validate_protocol(root: Path) -> dict[str, Any]:
    protocol = _load_json(root / "screen_protocol.json")
    exact = {
        "schema_version": 1,
        "scope": "excluded_adaptive_v14_screen_after_failed_v13",
        "seed": SEED,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "runs": EXPECTED_CELLS,
        "epochs": EPOCHS,
        "gpus": ["0", "1", "2", "3"],
        "task_pinned_gpu": dict(zip(TASKS, ("0", "1", "2", "3"), strict=True)),
        "study": STUDY,
        "wandb_entity": ENTITY,
        "wandb_project": PROJECT,
        "v11_comparator_action_ranking": V11_RANKING,
        "blas_threads_per_process": 4,
        "automatic_continuation_launch_in_this_process": False,
        "conditional_continuation_manifest": "conditional_continuation_manifest.json",
        "continuation_runs": 96,
    }
    for key, expected in exact.items():
        _require(protocol.get(key), expected, f"protocol.{key}")
    _require(protocol.get("git_branch"), "learnable-memory", "protocol.git_branch")
    commit = protocol.get("git_commit")
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise AuditFailure("protocol.git_commit is not a full lowercase SHA-1")
    _require(protocol.get("git_upstream_commit"), commit, "protocol.git_upstream_commit")
    _require(protocol.get("git_worktree_clean"), True, "protocol.git_worktree_clean")
    _require(protocol.get("git_head_pushed"), True, "protocol.git_head_pushed")
    source = protocol.get("source_sha256")
    if not isinstance(source, Mapping):
        raise AuditFailure("protocol source manifest is not a map")
    _require(set(source), set(SOURCE_MANIFEST), "protocol source manifest")
    for relative, expected in source.items():
        path = _resolve(relative)
        if not path.is_file():
            raise AuditFailure(f"missing source {relative}")
        _require(sha256_file(path), expected, f"source hash {relative}")
    data = protocol.get("data")
    if not isinstance(data, Mapping) or set(data) != set(TASKS):
        raise AuditFailure("protocol data manifest differs from frozen tasks")
    for task in TASKS:
        for split in ("train", "val"):
            entry = data[task]
            if not isinstance(entry, Mapping):
                raise AuditFailure(f"protocol data {task} is not an object")
            expected_path = expected_data_path(task, split)
            _require(entry.get(split), str(expected_path), f"data path {task}/{split}")
            path = _resolve(entry.get(split))
            if not path.is_file():
                raise AuditFailure(f"missing data {task}/{split}")
            _require(sha256_file(path), entry.get(f"{split}_sha256"),
                     f"data hash {task}/{split}")
    commands = protocol.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != set(TASKS):
        raise AuditFailure("protocol command grid differs")
    _require(protocol.get("commands_sha256"), json_sha256(commands), "command hash")
    expected_commands = {
        task: [expected_train_command(root, STUDY, EPOCHS, task, design)
               for design in DESIGNS]
        for task in TASKS
    }
    _require(commands, expected_commands, "protocol exact command vectors")
    prospective = _load_json(root / "conditional_continuation_manifest.json")
    for key, expected in {
            "status": "CONDITIONAL_NOT_AUTHORIZED", "launch_performed": False,
            "automatic_launch_supported": False,
            "designs": list(CONTINUATION_DESIGNS),
            "tasks": list(TASKS), "seeds": list(CONTINUATION_SEEDS),
            "epochs": 100, "runs": 96,
            "study": CONTINUATION_STUDY,
            "output_root": str(CONTINUATION_ROOT)}.items():
        _require(prospective.get(key), expected, f"prospective.{key}")
    continuation_commands = prospective.get("commands")
    if not isinstance(continuation_commands, list) or len(continuation_commands) != 96:
        raise AuditFailure("prospective continuation is not 96 commands")
    _require(prospective.get("commands_sha256"), json_sha256(continuation_commands),
             "prospective command hash")
    expected_continuation = [
        expected_train_command(
            CONTINUATION_ROOT, CONTINUATION_STUDY, 100, task, design, seed=seed)
        for seed in CONTINUATION_SEEDS
        for task in TASKS
        for design in CONTINUATION_DESIGNS
    ]
    _require(continuation_commands, expected_continuation,
             "prospective exact command vectors")
    return protocol


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise AuditFailure(f"{label}: cannot load model.pt: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"{label}: model.pt is not a dictionary")
    return value


def _validate_rollout(path: Path, label: str) -> None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not payload.files:
                raise AuditFailure(f"{label}: empty rollout")
            for key in payload.files:
                value = payload[key]
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise AuditFailure(f"{label}: non-finite rollout {key}")
    except (OSError, ValueError) as exc:
        raise AuditFailure(f"{label}: unreadable rollout: {exc}") from exc


def validate_cell(
        root: Path, task: str, design: str,
        protocol: Mapping[str, Any]) -> dict[str, Any]:
    directory = root / run_name(task, design)
    label = f"{task}/{design}"
    paths = {name: directory / name for name in (
        "model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")}
    for path in paths.values():
        if not path.is_file():
            raise AuditFailure(f"{label}: missing {path.name}")
    metrics = _load_json(paths["metrics.json"])
    for key, expected in {
            "env": f"dmc:{task}", "design": design,
            "seed": SEED, "epochs": EPOCHS}.items():
        _require(metrics.get(key), expected, f"{label}.metrics.{key}")
    _finite_tree(metrics, f"{label}.metrics")
    for key in (
            PRIMARY, "clean_prior_state_nmse", "initial_encoder_integrator_probe_nmse",
            "predictive_loss_convergence_relative_change", "encoder_mean_channel_variance",
            "encoder_covariance_effective_rank", "encoder_singleton_max_abs",
            "encoder_prefix_max_abs", "action_dim"):
        _finite(metrics.get(key), f"{label}.{key}")
    if design in CF_DESIGNS:
        for key in (
                "fit_updates", "memory_state_dim", "cf_ebo_fit_fit_index",
                "cf_ebo_streaming_max_abs", "cf_ebo_initial_reconstruction_max_abs",
                "cf_ebo_core_energy_identity_max_abs",
                "cf_ebo_core_state_spectral_radius",
                "cf_ebo_core_state_operator_norm",
                "cf_ebo_core_action_reliability",
                "cf_ebo_core_correction_reliability"):
            _finite(metrics.get(key), f"{label}.{key}")
        for condition in CONDITIONS:
            for suffix in (
                    "innovation_score_mean", "radial_gate_mean",
                    "correction_energy_max", "evidence_samples"):
                key = f"cf_ebo_{condition}_{suffix}"
                _finite(metrics.get(key), f"{label}.{key}")
    for split in ("train", "val"):
        _require(metrics.get(f"{split}_data_sha256"),
                 protocol["data"][task][f"{split}_sha256"], f"{label}.{split} hash")
    _validate_rollout(paths["eval_rollout.npz"], label)
    rollout_hash = sha256_file(paths["eval_rollout.npz"])
    _require(metrics.get("eval_rollout_sha256"), rollout_hash, f"{label}.rollout hash")
    wandb = _load_json(paths["wandb_run.json"])
    for key, expected in {
            "state": "finished", "mode": "online", "study": STUDY,
            "entity": ENTITY, "project": PROJECT,
            "eval_rollout_sha256": rollout_hash}.items():
        _require(wandb.get(key), expected, f"{label}.wandb.{key}")
    if not wandb.get("run_id") or not wandb.get("url"):
        raise AuditFailure(f"{label}: incomplete W&B identity")
    _require(wandb.get("run_name"), f"{STUDY}-{directory.name}", f"{label}.run name")
    checkpoint = _load_checkpoint(paths["model.pt"], label)
    args = checkpoint.get("args")
    if not isinstance(args, Mapping):
        raise AuditFailure(f"{label}: checkpoint args are missing")
    for key, expected in {
            "memory_mode": design, "seed": SEED, "epochs": EPOCHS,
            "wandb": True, "wandb_entity": ENTITY, "wandb_project": PROJECT,
            "wandb_mode": "online", "wandb_study": STUDY,
            "eval_rollout_episode": 0}.items():
        _require(args.get(key), expected, f"{label}.args.{key}")
    _require(checkpoint.get("final_metrics"), metrics, f"{label}.final metrics")
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise AuditFailure(f"{label}: checkpoint history is not 30 rows")
    _require([row.get("epoch") for row in history if isinstance(row, Mapping)],
             list(range(1, EPOCHS + 1)), f"{label}.history epochs")
    _finite_tree(history, f"{label}.history")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise AuditFailure(f"{label}: model state is missing")
    _finite_tree(state, f"{label}.model state")
    if design in CF_DESIGNS:
        fits, final = checkpoint.get("fit_history"), checkpoint.get("final_operator_fit")
        if not isinstance(fits, list) or len(fits) != EPOCHS + 1:
            raise AuditFailure(f"{label}: fit history is not 31 rows")
        _require([row.get("fit_index") for row in fits if isinstance(row, Mapping)],
                 list(range(EPOCHS + 1)), f"{label}.fit indices")
        _finite_tree(fits, f"{label}.fit history")
        if not isinstance(final, Mapping) or not isinstance(final.get("receipts"), Mapping):
            raise AuditFailure(f"{label}: final fit is missing")
        expected_fields = set(FIT_BUFFERS) | {"markov_even", "markov_odd", "receipts"}
        _require(set(final), expected_fields, f"{label}.fit fields")
        _finite_tree(final, f"{label}.final fit")
        _require(final["receipts"].get("fit_index"), EPOCHS, f"{label}.fit index")
        for key, value in final["receipts"].items():
            if isinstance(value, (bool, int, float, str)):
                _require(metrics.get(f"cf_ebo_fit_{key}"), value,
                         f"{label}.fit metric {key}")
        prefix = "world.mem_cfebov14."
        _require(int(state[prefix + "fit_updates"]), EPOCHS + 1,
                 f"{label}.fit updates")
        _require(bool(state[prefix + "operators_installed"]), True,
                 f"{label}.operators installed")
        for name in FIT_BUFFERS:
            saved, fitted = state.get(prefix + name), final.get(name)
            if (not isinstance(saved, torch.Tensor) or not isinstance(fitted, torch.Tensor)
                    or not torch.equal(saved.cpu(), fitted.cpu())):
                raise AuditFailure(f"{label}: serialized {name} differs from final fit")
        extra = state.get(prefix + "_extra_state")
        if (not isinstance(extra, Mapping)
                or not _deep_equal(extra.get("fit_receipts"), final["receipts"])):
            raise AuditFailure(
                f"{label}: serialized fit receipts differ from final fit receipts")
    if design == "kdiov11":
        _require(metrics.get("development_action_ranking"), V11_RANKING,
                 f"{label}.KDIO ranking")
        _require(args.get("development_action_ranking"), V11_RANKING,
                 f"{label}.KDIO arg ranking")
    return {
        "task": task, "design": design, "metrics": metrics,
        "wandb_run_id": wandb["run_id"], "wandb_url": wandb["url"],
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def validate_runner(root: Path, rows: Sequence[Mapping[str, Any]], protocol) -> None:
    if (root / LOCK_NAME).exists():
        raise AuditFailure("runner lock still exists")
    path = root / "screen_runs.json"
    if not path.is_file():
        raise AuditFailure("missing screen_runs.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != EXPECTED_CELLS:
        raise AuditFailure("screen_runs.json is not 40 records")
    row_map = {(row["task"], row["design"]): row for row in rows}
    seen = set()
    for record in records:
        pair = (record.get("task"), record.get("design"))
        if pair in seen or pair not in row_map:
            raise AuditFailure(f"invalid runner record pair {pair}")
        seen.add(pair)
        task, design = pair
        _require(str(record.get("gpu")), protocol["task_pinned_gpu"][task],
                 f"runner GPU {task}/{design}")
        _require(record.get("seed"), SEED, f"runner seed {task}/{design}")
        command = protocol["commands"][task][DESIGNS.index(design)]
        _require(record.get("command_sha256"), json_sha256(command),
                 f"runner command hash {task}/{design}")
        if _finite(record.get("seconds"), f"runner seconds {task}/{design}") <= 0:
            raise AuditFailure(f"runner seconds non-positive: {task}/{design}")
        _require(record.get("artifact_sha256"), row_map[pair]["artifact_sha256"],
                 f"runner hashes {task}/{design}")
    _require(seen, set(row_map), "runner cell set")


def _values(rows, design, key):
    mapping = {row["task"]: float(row["metrics"][key])
               for row in rows if row["design"] == design}
    if set(mapping) != set(TASKS):
        raise AuditFailure(f"incomplete {design}/{key}")
    return np.asarray([mapping[task] for task in TASKS])


def _contrast(rows, reference):
    candidate, baseline = _values(rows, "cfebov14", PRIMARY), _values(
        rows, reference, PRIMARY)
    return ((baseline.mean() - candidate.mean()) / baseline.mean(),
            int((candidate < baseline).sum()))


def recompute_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    representation_failures, numerical_failures = [], []
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == "cfebov14"}
    for task, metrics in full.items():
        if metrics["encoder_mean_channel_variance"] < 1e-5:
            representation_failures.append(f"{task}: variance")
        if metrics["encoder_covariance_effective_rank"] < 16:
            representation_failures.append(f"{task}: rank")
        if abs(metrics["encoder_singleton_max_abs"]) > 1e-5:
            representation_failures.append(f"{task}: singleton")
        if abs(metrics["encoder_prefix_max_abs"]) > 1e-5:
            representation_failures.append(f"{task}: prefix")
    for row in rows:
        if row["design"] not in CF_DESIGNS:
            continue
        m, design = row["metrics"], row["design"]
        label = f"{row['task']}/{design}"
        state_dim = min(23 * 128, 24 * int(m["action_dim"]))
        exact = {
            "fit_updates": 31,
            "cf_ebo_fit_fit_index": 30,
            "cf_ebo_fit_fit_episode_count": 1200,
            "cf_ebo_fit_fit_length": 48,
            "cf_ebo_fit_markov_lag_count": 47,
            "cf_ebo_fit_even_episodes": 600,
            "cf_ebo_fit_odd_episodes": 600,
            "cf_ebo_fit_action_combination": "minimum_directional_positive_part_EB",
            "cf_ebo_fit_correction_combination": "minimum_directional_positive_part_EB",
            "cf_ebo_core_gradient_parameter_count": 0,
            "cf_ebo_core_streaming_covariance_floats": 0,
        }
        for key, expected in exact.items():
            if m.get(key) != expected:
                numerical_failures.append(f"{label}: {key}")
        if int(round(_finite(m.get("memory_state_dim"), label))) != state_dim:
            numerical_failures.append(f"{label}: schema")
        for key, maximum in {
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
                "cf_ebo_core_energy_support_initial_map_max_abs": 2e-5}.items():
            if abs(_finite(m.get(key), f"{label}.{key}")) > maximum:
                numerical_failures.append(f"{label}: {key}")
        active = int(round(_finite(m.get("cf_ebo_core_energy_support_rank"), label)))
        inactive = int(round(_finite(m.get("cf_ebo_core_energy_inactive_padding"), label)))
        if (not 0 <= active <= state_dim or inactive != state_dim - active
                or int(m.get("cf_ebo_fit_energy_state_rank", -1)) != active
                or int(m.get("cf_ebo_fit_energy_support_projector_rank", -1)) != active
                or int(m.get("cf_ebo_fit_energy_inactive_padding", -1)) != inactive):
            numerical_failures.append(f"{label}: support rank")
        if _finite(m.get("cf_ebo_core_state_spectral_radius"), label) >= 1.0:
            numerical_failures.append(f"{label}: stability")
        if _finite(m.get("cf_ebo_core_state_operator_norm"), label) > 1.0 + 2e-5:
            numerical_failures.append(f"{label}: contraction")
        for mechanism in ("action", "correction"):
            first = _finite(m.get(
                f"cf_ebo_fit_{mechanism}_first_direction_reliability"), label)
            second = _finite(m.get(
                f"cf_ebo_fit_{mechanism}_second_direction_reliability"), label)
            combined = _finite(m.get(
                f"cf_ebo_fit_{mechanism}_combined_risk_reliability"), label)
            computed = _finite(m.get(
                f"cf_ebo_fit_computed_{mechanism}_reliability"), label)
            if (abs(combined - min(first, second)) > 1e-9
                    or abs(computed - combined) > 1e-9):
                numerical_failures.append(f"{label}: {mechanism} minimum")
        cap = design != "cfebov14_noenergycap"
        radial = design != "cfebov14_noradial"
        if m.get("cf_ebo_core_energy_cap_active") is not cap:
            numerical_failures.append(f"{label}: cap intervention")
        if m.get("cf_ebo_core_radial_gate_active") is not radial:
            numerical_failures.append(f"{label}: radial intervention")
        if cap and m["cf_ebo_core_deployed_correction_operator_norm"] > 1.0 + 2e-5:
            numerical_failures.append(f"{label}: cap")
        if design == "cfebov14_noaction" and (
                m.get("cf_ebo_exact_noaction") is not True
                or m["cf_ebo_core_action_reliability"] != 0.0):
            numerical_failures.append(f"{label}: noaction")
        if design == "cfebov14_nocorrect" and (
                m.get("cf_ebo_exact_nocorrect") is not True
                or m["cf_ebo_core_correction_reliability"] != 0.0):
            numerical_failures.append(f"{label}: nocorrect")
        if design == "cfebov14_norisk" and (
                m["cf_ebo_core_action_reliability"] != 1.0
                or m["cf_ebo_core_correction_reliability"] != 1.0):
            numerical_failures.append(f"{label}: norisk")
        codim = int(m["cf_ebo_core_complement_codimension"])
        if not 0 <= codim <= 128 or m.get(
                "cf_ebo_core_complement_present") is not (codim > 0):
            numerical_failures.append(f"{label}: complement")
        for fold in ("even", "odd", "pooled"):
            prefix = f"cf_ebo_fit_correction_{fold}_fit_"
            score_mean = _finite(m.get(prefix + "innovation_score_mean"), label)
            score_max = _finite(m.get(prefix + "innovation_score_max"), label)
            gate_mean = _finite(m.get(prefix + "radial_gate_mean"), label)
            gate_min = _finite(m.get(prefix + "radial_gate_min"), label)
            gate_max = _finite(m.get(prefix + "radial_gate_max"), label)
            if (score_mean < 0 or score_max < score_mean
                    or not 0 <= gate_min <= gate_mean <= gate_max <= 1):
                numerical_failures.append(f"{label}: calibration {fold}")

    external = {}
    for reference in EXTERNAL:
        reduction, wins = _contrast(rows, reference)
        external[reference] = reduction >= .05 and wins >= 3
    candidate = np.asarray([full[t][PRIMARY] for t in TASKS])
    integrator = np.asarray([full[t]["initial_encoder_integrator_probe_nmse"] for t in TASKS])
    external["integrator"] = bool(
        (integrator.mean() - candidate.mean()) / integrator.mean() >= .05
        and int((candidate < integrator).sum()) >= 3)
    internal = {}
    for control in CONTROLS:
        reduction, wins = _contrast(rows, control)
        internal[control] = reduction >= (
            .05 if control == "cfebov14_noaction" else .02) and wins >= 3
    mechanisms = 0
    shifts = 0
    energy_bounds = True
    for task in TASKS:
        m = full[task]
        mechanisms += int(
            m["cf_ebo_fit_action_even_to_odd_mean_improvement"] > 0
            and m["cf_ebo_fit_action_odd_to_even_mean_improvement"] > 0
            and m["cf_ebo_fit_correction_even_to_odd_mean_improvement"] > 0
            and m["cf_ebo_fit_correction_odd_to_even_mean_improvement"] > 0
            and m["cf_ebo_core_action_reliability"] > 0
            and m["cf_ebo_core_correction_reliability"] > 0
            and m["cf_ebo_true_action_suffix_advantage"] > 0
            and m["cf_ebo_action_pair_accuracy"] > .5)
        shifts += int(
            m["cf_ebo_gaussian_noise_innovation_score_mean"]
            > m["cf_ebo_val_train_view_innovation_score_mean"]
            and m["cf_ebo_gaussian_noise_radial_gate_mean"]
            < m["cf_ebo_val_train_view_radial_gate_mean"])
        bound = m["cf_ebo_core_correction_reliability"] ** 2 \
            * m["cf_ebo_core_innovation_rank"]
        energy_bounds &= all(
            m[f"cf_ebo_{condition}_correction_energy_max"]
            <= bound * 1.02 + 1e-5 for condition in CONDITIONS)
    full_late = [full[t]["predictive_loss_convergence_relative_change"] for t in TASKS]
    all_late = [abs(row["metrics"]["predictive_loss_convergence_relative_change"])
                for row in rows]
    convergence = bool(
        all(value >= 0 for value in full_late)
        and max(map(abs, full_late)) < .05 and float(np.median(all_late)) < .03)
    scientific = bool(
        not representation_failures and not numerical_failures
        and all(external.values()) and all(internal.values())
        and mechanisms >= 3 and shifts >= 3 and energy_bounds and convergence)
    return {
        "representation_passed": not representation_failures,
        "numerical_passed": not numerical_failures,
        "external_passed": all(external.values()),
        "internal_passed": all(internal.values()),
        "mechanism_passed": mechanisms >= 3,
        "robustness_passed": shifts >= 3 and energy_bounds,
        "complement_passed": True,
        "convergence_passed": convergence,
        "scientific_gate_passed": scientific,
        "representation_failures": representation_failures,
        "numerical_failures": numerical_failures,
    }


def audit_status(
        *, artifact_integrity: bool, analyzer_consistent: bool,
        scientific_gate: bool) -> tuple[str, bool]:
    if not artifact_integrity or not analyzer_consistent:
        return "FAIL_CLOSED", False
    return ("PASS_COMPLETE" if scientific_gate else "PASS_COMPLETE_NEGATIVE"), True


def audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    protocol = None
    try:
        protocol = validate_protocol(root)
    except (AuditFailure, OSError, ValueError) as exc:
        errors.append(str(exc))
    if protocol is not None:
        for task in TASKS:
            for design in DESIGNS:
                try:
                    rows.append(validate_cell(root, task, design, protocol))
                except (AuditFailure, OSError, ValueError) as exc:
                    errors.append(str(exc))
    ids = [row["wandb_run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate W&B run IDs")
    integrity = protocol is not None and len(rows) == EXPECTED_CELLS and not errors
    if integrity:
        try:
            validate_runner(root, rows, protocol)
        except (AuditFailure, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            integrity = False
    gates = recompute_gates(rows) if len(rows) == EXPECTED_CELLS else None
    consistent = False
    analyzer_status = None
    if integrity and gates is not None:
        try:
            analysis = _load_json(root / "screen_analysis.json")
            decision = _load_json(root / "screen_decision.json")
            expected_status = (
                "SCREEN_PASS_100E_MANIFEST" if gates["scientific_gate_passed"]
                else "SCREEN_NO_GO")
            checks = {
                "artifact_integrity_passed": True,
                "representation_gate_passed": gates["representation_passed"],
                "external_performance_gate_passed": gates["external_passed"],
                "internal_mechanism_gate_passed": gates["internal_passed"],
                "scientific_gate_passed": gates["scientific_gate_passed"],
                "status": expected_status,
            }
            for key, expected in checks.items():
                _require(analysis.get(key), expected, f"analysis.{key}")
            _require(analysis.get("numerical_gate", {}).get("passed"),
                     gates["numerical_passed"], "analysis numerical")
            _require(analysis.get("action_correction_mechanism_gate", {}).get("passed"),
                     gates["mechanism_passed"], "analysis mechanism")
            _require(analysis.get("robustness_gate", {}).get("passed"),
                     gates["robustness_passed"], "analysis robustness")
            _require(analysis.get("rank_aware_complement_gate", {}).get("passed"),
                     gates["complement_passed"], "analysis complement")
            _require(analysis.get("convergence_gate", {}).get("passed"),
                     gates["convergence_passed"], "analysis convergence")
            _require(decision.get("status"), expected_status, "decision status")
            _require(decision.get("scientific_gate_passed"),
                     gates["scientific_gate_passed"], "decision scientific")
            manifest_path = root / "contingent_100e_launch_manifest.json"
            _require(manifest_path.exists(), gates["scientific_gate_passed"],
                     "authorized continuation presence")
            if manifest_path.exists():
                manifest = _load_json(manifest_path)
                for key, expected in {
                        "status": "AUTHORIZED_NOT_LAUNCHED", "runs": 96,
                        "designs": ["cfebov14", "cfebov14_nocorrect", "cfebov14_noaction",
                                    "cfebov14_norisk", "cfhirov13_nocorrect", "ssm",
                                    "hacssmv8", "kdiov11"],
                        "seeds": [14002, 14003, 14004], "epochs": 100,
                        "automatic_launch_performed": False,
                        "scientific_gate_passed": True}.items():
                    _require(manifest.get(key), expected, f"authorized.{key}")
                commands = manifest.get("commands")
                if not isinstance(commands, list) or len(commands) != 96:
                    raise AuditFailure("authorized continuation is not 96 commands")
                _require(manifest.get("commands_sha256"), json_sha256(commands),
                         "authorized command hash")
                expected_commands = [
                    expected_train_command(
                        CONTINUATION_ROOT, CONTINUATION_STUDY, 100,
                        task, design, seed=seed)
                    for seed in CONTINUATION_SEEDS
                    for task in TASKS
                    for design in CONTINUATION_DESIGNS
                ]
                _require(commands, expected_commands,
                         "authorized exact command vectors")
            analyzer_status = expected_status
            consistent = True
        except (AuditFailure, OSError, ValueError) as exc:
            errors.append(str(exc))
    scientific = bool(gates and gates["scientific_gate_passed"])
    status, passed = audit_status(
        artifact_integrity=integrity, analyzer_consistent=consistent,
        scientific_gate=scientific)
    return {
        "schema_version": 1,
        "scope": "independent_read_only_cf_ebo_v14_screen_audit",
        "root": str(root),
        "status": status,
        "passed": passed,
        "artifact_integrity_passed": integrity,
        "analyzer_receipt_consistent": consistent,
        "scientific_gate_passed": scientific if gates is not None else None,
        "analyzer_status": analyzer_status,
        "expected_cells": EXPECTED_CELLS,
        "validated_cells": len(rows),
        "protocol_validated": protocol is not None,
        "recomputed_gates": gates,
        "errors": errors,
        "cells": [{
            "task": row["task"], "design": row["design"],
            PRIMARY: row["metrics"][PRIMARY],
            "wandb_run_id": row["wandb_run_id"],
            "wandb_url": row["wandb_url"],
            "artifact_sha256": row["artifact_sha256"],
        } for row in rows],
    }


def audit_exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report.get("passed") is True else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/hacssm_v14_screen_cfebo30"))
    parser.add_argument("--output", type=Path, default=None)
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
    raise SystemExit(audit_exit_code(report))


if __name__ == "__main__":
    main()
