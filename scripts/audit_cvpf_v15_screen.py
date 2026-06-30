#!/usr/bin/env python3
"""Independent closed-world audit of the frozen CVPF-v15 screen.

This module deliberately does not import the V15 analyzer.  It reconstructs
the source/data/command freeze, artifact grid, and scientific gates from the
saved evidence.  It never launches a continuation.
"""

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

from scripts.hacssm_v11_data import (
    DEFAULT_IMG_SIZE,
    DEFAULT_LENGTH,
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_EPISODES,
    DEFAULT_VAL_SEED,
    cache_name,
    sha256_file,
)
TASKS = ("cartpole.swingup", "fish.swim", "pendulum.swingup", "walker.walk")
V11_COMPARATOR_RANKING = "rawdiff_displacement_detached"
CVPF_DESIGNS = (
    "cvpfv15", "cvpfv15_nocorrect", "cvpfv15_noaction", "cvpfv15_norisk",
    "cvpfv15_norho", "cvpfv15_anchoronly", "cvpfv15_detachid",
    "cvpfv15_noenvelope")
BASELINES = (
    "cfebov14_norisk", "cfhirov13_nocorrect", "ssm", "hacssmv8", "kdiov11")
DESIGNS = CVPF_DESIGNS + BASELINES
CANDIDATE = "cvpfv15"
DIRECT_CONTROLS = CVPF_DESIGNS[1:]
PRIMARY = "heldout_prior_state_nmse"
SEED = 15001
EPOCHS = 30
STUDY = "hacssm-v15-screen-cvpf30"
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
EXPECTED_CELLS = 52
CONTINUATION_SEEDS = (15002, 15003, 15004)
CONTINUATION_EPOCHS = 100
CONTINUATION_STUDY = "hacssm-v15-continuation-cvpf100"
DATA_ROOT = Path("outputs/hacssm_v11_data")
FROZEN_PYTHON = ROOT / ".venv" / "bin" / "python"
LOCK_NAME = ".cvpf_v15_screen.lock"

SOURCE_MANIFEST = (
    "lewm/models/cvpf.py", "lewm/models/cf_ebo.py", "lewm/models/cf_hiro.py",
    "lewm/models/siro.py", "lewm/models/memory_model.py", "lewm/models/memory.py",
    "lewm/models/leworldmodel.py", "lewm/models/encoder.py", "lewm/models/sigreg.py",
    "scripts/train_cvpf_v15.py", "scripts/run_cvpf_v15_screen.py",
    "scripts/analyze_cvpf_v15_screen.py", "scripts/audit_cvpf_v15_screen.py",
    "scripts/train_cf_ebo_v14.py", "scripts/run_cf_ebo_v14_screen.py",
    "scripts/analyze_cf_ebo_v14_screen.py", "scripts/audit_cf_ebo_v14_screen.py",
    "scripts/train_cf_hiro_v13.py", "scripts/run_cf_hiro_v13_screen.py",
    "scripts/analyze_cf_hiro_v13_screen.py", "scripts/audit_cf_hiro_v13_screen.py",
    "scripts/train_siro_v12.py", "scripts/run_siro_v12_screen.py",
    "scripts/analyze_siro_v12_screen.py", "scripts/train_hacssm_v11.py",
    "scripts/train_hacssm_v10.py", "scripts/hacssm_v10_data.py",
    "scripts/hacssm_v11_data.py",
)

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
SUFFIX_KEYS = (
    "cvpf_true_action_suffix_advantage", "cvpf_true_action_prior_advantage",
    "cvpf_action_suffix_advantage")
PAIR_KEYS = ("cvpf_action_pair_accuracy", "cvpf_action_swap_pair_accuracy")


class AuditFailure(RuntimeError):
    """Independent audit failure."""


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AuditFailure(f"{label} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditFailure(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AuditFailure(f"{label} is non-finite")
    return result


def metric(metrics: Mapping[str, Any], aliases: Sequence[str], label: str) -> float:
    for key in aliases:
        if key in metrics:
            return finite(metrics[key], f"{label}:{key}")
    raise AuditFailure(f"{label}: missing aliases {tuple(aliases)}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise AuditFailure(f"{path} is not a JSON object")
    return result


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


def data_paths(task: str) -> tuple[Path, Path]:
    return (
        DATA_ROOT / cache_name(
            task, "train", DEFAULT_TRAIN_EPISODES, DEFAULT_LENGTH,
            DEFAULT_IMG_SIZE, DEFAULT_TRAIN_SEED),
        DATA_ROOT / cache_name(
            task, "val", DEFAULT_VAL_EPISODES, DEFAULT_LENGTH,
            DEFAULT_IMG_SIZE, DEFAULT_VAL_SEED),
    )


def run_name(task: str, design: str, seed: int = SEED) -> str:
    suffix = f"-rank-{V11_COMPARATOR_RANKING}" if design == "kdiov11" else ""
    return f"lewm-dmc:{task}-{design}-s{seed}{suffix}"


def expected_train_command(
        root: Path, study: str, epochs: int, task: str, design: str,
        *, seed: int = SEED) -> list[str]:
    train_data, val_data = data_paths(task)
    return [
        str(FROZEN_PYTHON), str(ROOT / "scripts" / "train_cvpf_v15.py"),
        "--train-data", str(train_data), "--val-data", str(val_data),
        "--memory-mode", design, "--seed", str(seed), "--epochs", str(epochs),
        "--output-dir", str(root), "--batch-size", "64", "--lr", "0.0003",
        "--weight-decay", "0.00001", "--num-workers", "2", "--img-size", "64",
        "--patch-size", "8", "--embed-dim", "128", "--encoder-layers", "6",
        "--encoder-heads", "4", "--predictor-layers", "4",
        "--predictor-heads", "8", "--history-len", "3", "--dropout", "0.1",
        "--sigreg-lambda", "0.1", "--sigreg-projections", "512",
        "--probe-ridge", "0.001", "--eval-target-key", "task_observation",
        "--corruption-seed", "11012", "--eval-rollout-episode", "0",
        "--device", "cuda", "--wandb", "--wandb-entity", WANDB_ENTITY,
        "--wandb-project", WANDB_PROJECT, "--wandb-mode", "online",
        "--wandb-study", study, "--extra-tag", "excluded-adaptive-screen,cvpf-v15",
    ]


def validate_protocol(root: Path, protocol: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": 1, "scope": "excluded_adaptive_v15_cvpf_screen",
        "seed": SEED, "tasks": list(TASKS), "designs": list(DESIGNS),
        "runs": EXPECTED_CELLS, "epochs": EPOCHS,
        "gpus": ["0", "1", "2", "3"], "study": STUDY,
        "wandb_entity": WANDB_ENTITY, "wandb_project": WANDB_PROJECT,
        "v11_comparator_action_ranking": V11_COMPARATOR_RANKING,
        "automatic_continuation_launch_in_this_process": False,
        "continuation_runs": 156,
    }
    for key, value in exact.items():
        if protocol.get(key) != value:
            raise AuditFailure(f"protocol {key} differs")
    commit = protocol.get("git_commit")
    if (not isinstance(commit, str) or len(commit) != 40
            or protocol.get("git_upstream_commit") != commit
            or protocol.get("git_worktree_clean") is not True
            or protocol.get("git_head_pushed") is not True):
        raise AuditFailure("protocol does not prove clean pushed provenance")
    source = protocol.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_MANIFEST):
        raise AuditFailure("source manifest set differs")
    for relative in SOURCE_MANIFEST:
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != source[relative]:
            raise AuditFailure(f"source hash differs: {relative}")
    data = protocol.get("data")
    if not isinstance(data, Mapping) or set(data) != set(TASKS):
        raise AuditFailure("data manifest differs")
    for task in TASKS:
        for index, split in enumerate(("train", "val")):
            path = data_paths(task)[index]
            if (data[task].get(split) != str(path) or not path.is_file()
                    or data[task].get(f"{split}_sha256") != sha256_file(path)):
                raise AuditFailure(f"data receipt differs: {task}/{split}")
    expected = {
        task: [expected_train_command(root.resolve(), STUDY, EPOCHS, task, design)
               for design in DESIGNS]
        for task in TASKS}
    if protocol.get("commands") != expected:
        raise AuditFailure("protocol commands differ token-for-token")
    if protocol.get("commands_sha256") != json_sha256(expected):
        raise AuditFailure("protocol command hash differs")


def validate_continuation(root: Path) -> None:
    manifest = load_json(root / "conditional_continuation_manifest.json")
    exact = {
        "status": "CONDITIONAL_NOT_AUTHORIZED", "launch_performed": False,
        "automatic_launch_supported": False, "designs": list(DESIGNS),
        "tasks": list(TASKS), "seeds": list(CONTINUATION_SEEDS),
        "epochs": CONTINUATION_EPOCHS, "runs": 156, "study": CONTINUATION_STUDY,
    }
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise AuditFailure(f"continuation {key} differs")
    commands = manifest.get("commands")
    expected = [
        expected_train_command(
            (ROOT / "outputs/hacssm_v15_continuation_cvpf100").resolve(),
            CONTINUATION_STUDY, CONTINUATION_EPOCHS, task, design, seed=seed)
        for seed in CONTINUATION_SEEDS for task in TASKS for design in DESIGNS]
    if commands != expected or manifest.get("commands_sha256") != json_sha256(expected):
        raise AuditFailure("continuation commands differ token-for-token")


def validate_rollout(path: Path, expected_hash: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise AuditFailure(f"{label}: rollout hash differs")
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not payload.files:
                raise AuditFailure(f"{label}: empty rollout")
            for key in payload.files:
                value = payload[key]
                if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
                    raise AuditFailure(f"{label}: non-finite rollout")
    except (OSError, ValueError) as exc:
        raise AuditFailure(f"{label}: unreadable rollout") from exc


def load_artifact_rows(
        root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    run_ids = set()
    for task in TASKS:
        for design in DESIGNS:
            label = f"{task}/{design}"
            directory = root / run_name(task, design)
            metrics = load_json(directory / "metrics.json")
            for key, wanted in {
                    "env": f"dmc:{task}", "design": design,
                    "seed": SEED, "epochs": EPOCHS}.items():
                if metrics.get(key) != wanted:
                    raise AuditFailure(f"{label}: metrics {key} differs")
            for key in (
                    PRIMARY, "clean_prior_state_nmse",
                    "initial_encoder_integrator_probe_nmse",
                    "predictive_loss_convergence_relative_change",
                    "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
                    "encoder_singleton_max_abs", "encoder_prefix_max_abs"):
                finite(metrics.get(key), f"{label}:{key}")
            for split in ("train", "val"):
                if metrics.get(f"{split}_data_sha256") != protocol["data"][task][
                        f"{split}_sha256"]:
                    raise AuditFailure(f"{label}: data hash differs")
            rollout = directory / "eval_rollout.npz"
            rollout_hash = metrics.get("eval_rollout_sha256")
            if not isinstance(rollout_hash, str):
                raise AuditFailure(f"{label}: rollout receipt missing")
            validate_rollout(rollout, rollout_hash, label)
            wandb = load_json(directory / "wandb_run.json")
            for key, wanted in {
                    "state": "finished", "mode": "online", "study": STUDY,
                    "entity": WANDB_ENTITY, "project": WANDB_PROJECT,
                    "eval_rollout_sha256": rollout_hash,
                    "run_name": f"{STUDY}-{directory.name}"}.items():
                if wandb.get(key) != wanted:
                    raise AuditFailure(f"{label}: W&B {key} differs")
            run_id = wandb.get("run_id")
            if not isinstance(run_id, str) or not run_id or run_id in run_ids:
                raise AuditFailure(f"{label}: invalid/duplicate W&B ID")
            run_ids.add(run_id)
            try:
                payload = torch.load(
                    directory / "model.pt", map_location="cpu", weights_only=False)
            except Exception as exc:
                raise AuditFailure(f"{label}: unreadable checkpoint") from exc
            if not isinstance(payload, Mapping) or payload.get("final_metrics") != metrics:
                raise AuditFailure(f"{label}: checkpoint metrics differ")
            args = payload.get("args")
            if not isinstance(args, Mapping):
                raise AuditFailure(f"{label}: checkpoint args missing")
            for key, wanted in {
                    "memory_mode": design, "seed": SEED, "epochs": EPOCHS,
                    "wandb": True, "wandb_mode": "online", "wandb_study": STUDY}.items():
                if args.get(key) != wanted:
                    raise AuditFailure(f"{label}: checkpoint arg {key} differs")
            history = payload.get("history")
            if (not isinstance(history, list) or len(history) != EPOCHS
                    or [row.get("epoch") for row in history] != list(range(1, EPOCHS + 1))):
                raise AuditFailure(f"{label}: history differs")
            if design in CVPF_DESIGNS:
                fit_history = payload.get("fit_history")
                final_fit = payload.get("final_operator_fit")
                if (not isinstance(fit_history, list) or len(fit_history) != EPOCHS + 1
                        or [row.get("fit_index") for row in fit_history]
                        != list(range(EPOCHS + 1))
                        or not isinstance(final_fit, Mapping)
                        or not isinstance(final_fit.get("receipts"), Mapping)):
                    raise AuditFailure(f"{label}: fit serialization differs")
                receipts = final_fit["receipts"]
                if receipts.get("fit_index") != EPOCHS:
                    raise AuditFailure(f"{label}: final fit index differs")
                for key, value in receipts.items():
                    if isinstance(value, (bool, int, float, str)) and metrics.get(
                            f"cvpf_fit_{key}") != value:
                        raise AuditFailure(f"{label}: fit receipt differs")
                state = payload.get("model_state_dict")
                prefix = "world.mem_cvpfv15."
                if (not isinstance(state, Mapping)
                        or not isinstance(state.get(prefix + "fit_updates"), torch.Tensor)
                        or int(state[prefix + "fit_updates"]) != EPOCHS + 1
                        or not isinstance(state.get(prefix + "operators_installed"), torch.Tensor)
                        or not bool(state[prefix + "operators_installed"])):
                    raise AuditFailure(f"{label}: installed fit receipt differs")
                extra = state.get(prefix + "_extra_state")
                if not isinstance(extra, Mapping) or not _deep_equal(
                        extra.get("fit_receipts"), receipts):
                    raise AuditFailure(f"{label}: extra-state fit receipt differs")
            artifacts = {
                name: sha256_file(directory / name) for name in (
                    "model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")}
            rows.append({
                "task": task, "design": design, "metrics": metrics,
                "wandb_run_id": run_id, "wandb_url": wandb.get("url"),
                "artifact_sha256": artifacts,
            })
    if len(rows) != EXPECTED_CELLS:
        raise AuditFailure("artifact grid is incomplete")
    return rows


def validate_runner(
        root: Path, rows: Sequence[Mapping[str, Any]],
        protocol: Mapping[str, Any]) -> None:
    if (root / LOCK_NAME).exists():
        raise AuditFailure("runner lock still exists")
    path = root / "screen_runs.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditFailure("runner receipt unreadable") from exc
    if not isinstance(records, list) or len(records) != EXPECTED_CELLS:
        raise AuditFailure("runner receipt count differs")
    expected = {(row["task"], row["design"]): row for row in rows}
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise AuditFailure("runner row is not an object")
        pair = (record.get("task"), record.get("design"))
        if pair in seen or pair not in expected:
            raise AuditFailure(f"runner cell invalid: {pair}")
        seen.add(pair)
        task, design = pair
        if (str(record.get("gpu")) != str(protocol["task_pinned_gpu"][task])
                or record.get("seed") != SEED):
            raise AuditFailure(f"runner pin/seed differs: {pair}")
        command = protocol["commands"][task][DESIGNS.index(design)]
        if record.get("command_sha256") != json_sha256(command):
            raise AuditFailure(f"runner command hash differs: {pair}")
        if finite(record.get("seconds"), f"{pair}:seconds") <= 0:
            raise AuditFailure(f"runner time is nonpositive: {pair}")
        if record.get("artifact_sha256") != expected[pair]["artifact_sha256"]:
            raise AuditFailure(f"runner artifact hash differs: {pair}")
    if seen != set(expected):
        raise AuditFailure("runner cell set differs")


def values(rows: Sequence[Mapping[str, Any]], design: str, key: str) -> np.ndarray:
    by_task = {
        row["task"]: finite(row["metrics"].get(key), f"{design}:{key}")
        for row in rows if row["design"] == design}
    if set(by_task) != set(TASKS):
        raise AuditFailure(f"{design}/{key}: incomplete")
    return np.asarray([by_task[task] for task in TASKS], dtype=np.float64)


def reduction(rows: Sequence[Mapping[str, Any]], reference: str) -> tuple[float, int]:
    candidate = values(rows, CANDIDATE, PRIMARY)
    baseline = values(rows, reference, PRIMARY)
    return float((baseline.mean() - candidate.mean()) / baseline.mean()), int(
        (candidate < baseline).sum())


def recompute_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: dict[str, list[str]] = {
        "representation": [], "structural": [], "modes": []}
    full = {row["task"]: row["metrics"] for row in rows if row["design"] == CANDIDATE}
    for task in TASKS:
        metrics = full[task]
        if finite(metrics.get("encoder_mean_channel_variance"), task) < 1e-5:
            failures["representation"].append(f"{task}: variance")
        if finite(metrics.get("encoder_covariance_effective_rank"), task) < 16:
            failures["representation"].append(f"{task}: rank")
        if abs(finite(metrics.get("encoder_singleton_max_abs"), task)) > 1e-5:
            failures["representation"].append(f"{task}: singleton")
        if abs(finite(metrics.get("encoder_prefix_max_abs"), task)) > 1e-5:
            failures["representation"].append(f"{task}: encoder prefix")
    for row in rows:
        if row["design"] not in CVPF_DESIGNS:
            continue
        metrics, design = row["metrics"], str(row["design"])
        label = f"{row['task']}/{design}"
        for aliases in (STREAM_KEYS, PREFIX_KEYS):
            if abs(metric(metrics, aliases, label)) > 1e-5:
                failures["structural"].append(f"{label}: {aliases[0]}")
        shift = metric(metrics, SHIFT_KEYS, label)
        if not 0.0 <= shift <= 1.0 + 16.0 * np.finfo(np.float64).eps:
            failures["structural"].append(f"{label}: projected shift closure")
        exposure = metric(metrics, INNOVATION_EXPOSURE_KEYS, label)
        if not .5 <= exposure <= 2.0:
            failures["structural"].append(f"{label}: innovation exposure")
        if (metrics.get("cvpf_fit_fit_uses_validation") is not False
                or int(finite(metrics.get("fit_updates"), label)) != EPOCHS + 1
                or int(finite(metrics.get("cvpf_fit_fit_index"), label)) != EPOCHS):
            failures["structural"].append(f"{label}: fit protocol")
        gain_keys = (
            "cvpf_core_action_gain", "cvpf_core_correction_gain",
            "cvpf_core_risk_gain", "cvpf_core_rho")
        gains = {key: finite(metrics.get(key), label) for key in gain_keys}
        if any(not 0 <= value <= 1 for value in gains.values()):
            failures["modes"].append(f"{label}: gain bound")
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
        if any(metrics.get(key) is not wanted for key, wanted in expected.items()):
            failures["modes"].append(f"{label}: exact semantics")
        if design == "cvpfv15_nocorrect" and gains["cvpf_core_correction_gain"] != 0:
            failures["modes"].append(f"{label}: correction nonzero")
        if design == "cvpfv15_noaction" and gains["cvpf_core_action_gain"] != 0:
            failures["modes"].append(f"{label}: action nonzero")
        if design == "cvpfv15_norisk" and gains["cvpf_core_risk_gain"] != 1:
            failures["modes"].append(f"{label}: risk not one")
        if design == "cvpfv15_norho" and gains["cvpf_core_rho"] != 1:
            failures["modes"].append(f"{label}: rho not one")
        weight = finite(metrics.get("cvpf_envelope_weight"), label)
        if weight < 0 or (design == "cvpfv15_noenvelope" and weight != 0):
            failures["modes"].append(f"{label}: envelope weight")
    baseline_results = {design: reduction(rows, design) for design in BASELINES}
    control_results = {design: reduction(rows, design) for design in DIRECT_CONTROLS}
    candidate = values(rows, CANDIDATE, PRIMARY)
    integrator = np.asarray([
        full[task]["initial_encoder_integrator_probe_nmse"] for task in TASKS],
        dtype=np.float64)
    integrator_result = (
        float((integrator.mean() - candidate.mean()) / integrator.mean()),
        int((candidate < integrator).sum()))
    baseline_passed = all(value[0] >= .05 and value[1] >= 3
                          for value in (*baseline_results.values(), integrator_result))
    controls_passed = all(value[0] >= .02 and value[1] >= 3
                          for value in control_results.values())
    identification_envelope_passed = all(
        control_results[name][0] >= .02 and control_results[name][1] >= 3
        for name in ("cvpfv15_detachid", "cvpfv15_noenvelope"))
    mechanism_counts = [0, 0, 0, 0]
    for task in TASKS:
        metrics = full[task]
        checks = (
            metric(metrics, ACTION_GAIN_KEYS, task) > 0,
            metric(metrics, CORRECTION_GAIN_KEYS, task) > 0,
            metric(metrics, SUFFIX_KEYS, task) > 0,
            metric(metrics, PAIR_KEYS, task) > .5,
        )
        mechanism_counts = [left + int(right)
                            for left, right in zip(mechanism_counts, checks, strict=True)]
    mechanism_passed = all(value >= 3 for value in mechanism_counts)
    signed = [finite(full[task]["predictive_loss_convergence_relative_change"], task)
              for task in TASKS]
    all_abs = [abs(finite(row["metrics"][
        "predictive_loss_convergence_relative_change"], "late")) for row in rows]
    convergence_passed = (
        all(value >= 0 for value in signed)
        and max(map(abs, signed)) < .05 and float(np.median(all_abs)) < .03)
    representation_passed = not failures["representation"]
    structural_passed = not failures["structural"]
    modes_passed = not failures["modes"]
    scientific = all((
        representation_passed, structural_passed, modes_passed,
        baseline_passed, controls_passed, identification_envelope_passed,
        mechanism_passed, convergence_passed))
    return {
        "representation_passed": representation_passed,
        "structural_passed": structural_passed,
        "mode_gain_exact_ablation_passed": modes_passed,
        "baseline_passed": baseline_passed,
        "direct_controls_passed": controls_passed,
        "active_identification_envelope_passed": identification_envelope_passed,
        "mechanism_passed": mechanism_passed,
        "mechanism_passed_task_counts": mechanism_counts,
        "convergence_passed": convergence_passed,
        "scientific_gate_passed": scientific,
        "baseline_results": baseline_results,
        "control_results": control_results,
        "integrator_result": integrator_result,
        "failures": failures,
    }


def audit_status(
        *, artifact_integrity: bool, analyzer_consistent: bool,
        scientific_gate: bool) -> tuple[str, bool]:
    if not artifact_integrity or not analyzer_consistent:
        return "FAIL_CLOSED", False
    return ("PASS_COMPLETE" if scientific_gate
            else "PASS_COMPLETE_NEGATIVE"), True


def audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    protocol: dict[str, Any] | None = None
    gates: dict[str, Any] | None = None
    analyzer_consistent = False
    try:
        protocol = load_json(root / "screen_protocol.json")
        validate_protocol(root, protocol)
        validate_continuation(root)
        rows = load_artifact_rows(root, protocol)
        validate_runner(root, rows, protocol)
        gates = recompute_gates(rows)
        analysis = load_json(root / "screen_analysis.json")
        decision = load_json(root / "screen_decision.json")
        authorization = load_json(root / "conditional_authorization.json")
        expected_status = "SCREEN_GO" if gates["scientific_gate_passed"] else "SCREEN_NO_GO"
        analyzer_consistent = all((
            analysis.get("artifact_integrity_passed") is True,
            analysis.get("completed_cells") == EXPECTED_CELLS,
            analysis.get("scientific_gate_passed") is gates["scientific_gate_passed"],
            analysis.get("status") == expected_status,
            decision.get("status") == expected_status,
            decision.get("automatic_launch_performed") is False,
            authorization.get("automatic_launch_performed") is False,
            authorization.get("status") == (
                "AUTHORIZED_NOT_LAUNCHED" if gates["scientific_gate_passed"]
                else "CONDITIONAL_NOT_AUTHORIZED"),
        ))
        if not analyzer_consistent:
            errors.append("saved analyzer/decision/authorization differs from audit")
    except (AuditFailure, OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    artifact_integrity = len(rows) == EXPECTED_CELLS and not errors
    scientific = bool(gates and gates.get("scientific_gate_passed"))
    status, passed = audit_status(
        artifact_integrity=artifact_integrity,
        analyzer_consistent=analyzer_consistent,
        scientific_gate=scientific)
    return {
        "schema_version": 1,
        "scope": "independent_closed_world_v15_cvpf_screen_audit",
        "status": status,
        "passed": passed,
        "artifact_integrity_passed": artifact_integrity,
        "analyzer_consistent": analyzer_consistent,
        "scientific_gate_passed": scientific,
        "expected_cells": EXPECTED_CELLS,
        "validated_cells": len(rows),
        "gates": gates,
        "errors": errors,
        "automatic_continuation_launch_performed": False,
    }


def audit_exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report.get("passed") is True else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "outputs/hacssm_v15_screen_cvpf30"))
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = (args.root if args.root.is_absolute() else ROOT / args.root).resolve()
    report = audit(root)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.write:
        path = root / "screen_audit.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(rendered, encoding="utf-8")
    raise SystemExit(audit_exit_code(report))


if __name__ == "__main__":
    main()
