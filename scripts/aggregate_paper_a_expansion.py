#!/usr/bin/env python3
"""Validate and aggregate the preregistered Paper-A expansion experiments.

The experiment matrix is read from ``configs/paper_a_expansion.yaml``.  The
script deliberately treats a missing cell and an invalid cell differently:
``--allow-incomplete`` permits the former so progress can be inspected, but
never permits a provenance or schema violation.  A normal publication build
must run without ``--allow-incomplete``.

Outputs are deterministic ``summary.json`` and ``summary.md`` files under the
experiment root.  Confidence intervals resample optimizer/model seeds as the
unit of analysis; fixed-trust contrasts are paired by the same seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

try:
    import yaml
except ImportError as error:  # pragma: no cover - dependency error is explicit
    raise SystemExit(
        "aggregate_paper_a_expansion.py requires PyYAML (import name 'yaml')"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/paper_a_expansion"
DEFAULT_CONFIG = ROOT / "configs/paper_a_expansion.yaml"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_706

TASK_FAMILIES = {
    "t1": "transient-marker",
    "t3": "drifting-color",
    "t4": "occluded-tracking",
}
SEMANTIC_TASK_NAMES = {
    "t1": "Transient-marker recall",
    "t3": "Drifting-color recall",
    "t4": "Occluded-target prediction",
}
ARM_DISPLAY_NAMES = {
    "none": "No carrier",
    "gru": "Action-conditioned GRU",
    "lstm": "Action-conditioned LSTM",
    "ssm": "Diagonal SSM",
    "fixed_trust": "Fixed-trust memory",
}
OBJECTIVE_DISPLAY_NAMES = {
    "one_step": "One-step objective",
    "overshoot_8": "Eight-step overshooting",
}
FROZEN_PARAMETER_KEYS = {
    "none": "none",
    "gru": "acgru",
    "lstm": "aclstm",
    "ssm": "diag_ssm",
    "fixed_trust": "lkc_fixed_trust",
}
ROLLOUT_METRICS = (
    "normalized_latent_mse",
    "copy_last_normalized_mse",
    "shuffled_action_normalized_mse",
    "true_action_advantage",
    "pose_angular_mae",
    "predicted_effective_rank",
    "target_effective_rank",
)
FROZEN_PROBE_SCHEMA = "official-frozen-decision-endpoint-v2"
FROZEN_TRAJECTORY_SCHEMA = "official-frozen-trajectory-diagnostic-v1"
FROZEN_DECISION_INDEX = 63
FROZEN_RAW_CONTEXT_INDICES = [60, 61, 62]


class AggregationError(RuntimeError):
    """Raised after collecting all actionable validation errors."""


@dataclass(frozen=True)
class Cell:
    wave: str
    task: str
    variant: str
    seed: int
    path: Path

    @property
    def label(self) -> str:
        return f"{self.wave}/{self.task}/{self.variant}/seed={self.seed}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def carrier_state_digest(state: Mapping[str, Any]) -> str:
    """Hash every carrier-state tensor including its identity and layout."""

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("carrier state must map string names to tensors")
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"carrier state {name!r} contains a non-finite value")
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cached_sha256(path: Path, cache: dict[Path, str]) -> str:
    """Hash ``path`` once; unlike ``dict.setdefault``, avoid eager rehashing."""
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AggregationError(f"preregistration config does not exist: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise AggregationError(f"preregistration config is not a YAML mapping: {path}")
    required = {
        "schema_version", "display_names", "official_host",
        "availability_gate", "frozen_carrier_swap", "long_context",
        "learned_rollout",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise AggregationError(f"preregistration config misses keys: {missing}")
    if payload.get("schema_version") != 1:
        raise AggregationError("preregistration schema_version must be 1")
    if payload.get("display_names") != SEMANTIC_TASK_NAMES:
        raise AggregationError(
            "display_names must use the publication semantic names exactly: "
            f"{SEMANTIC_TASK_NAMES}"
        )
    return payload


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise AggregationError(f"{label} must be a nonempty YAML list")
    return value


def expected_cells(root: Path, config: Mapping[str, Any]) -> list[Cell]:
    cells: list[Cell] = []
    frozen = config["frozen_carrier_swap"]
    for task in _list(frozen["tasks"], "frozen_carrier_swap.tasks"):
        for arm in _list(frozen["arms"], "frozen_carrier_swap.arms"):
            for seed in _list(frozen["seeds"], "frozen_carrier_swap.seeds"):
                cells.append(Cell(
                    "frozen_swap", str(task), str(arm), int(seed),
                    root / "frozen_swap" / str(task) / str(arm)
                    / f"s{int(seed)}" / "metrics.json",
                ))
    context = config["long_context"]
    for task in _list(context["tasks"], "long_context.tasks"):
        for history in _list(context["contexts"], "long_context.contexts"):
            for seed in _list(context["seeds"], "long_context.seeds"):
                cells.append(Cell(
                    "long_context", str(task), f"h{int(history)}", int(seed),
                    root / "long_context" / str(task) / f"h{int(history)}"
                    / f"s{int(seed)}" / "metrics.json",
                ))
    rollout = config["learned_rollout"]
    for task in _list(rollout["tasks"], "learned_rollout.tasks"):
        for objective in _list(rollout["objectives"],
                               "learned_rollout.objectives"):
            for seed in _list(rollout["seeds"], "learned_rollout.seeds"):
                cells.append(Cell(
                    "rollout", str(task), str(objective), int(seed),
                    root / "rollout" / str(task) / str(objective)
                    / f"s{int(seed)}" / "metrics.json",
                ))
    labels = [cell.label for cell in cells]
    if len(labels) != len(set(labels)):
        raise AggregationError("preregistration produces duplicate grid cells")
    return cells


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _same_number(actual: Any, expected: Any) -> bool:
    return (_is_number(actual) and _is_number(expected)
            and math.isclose(float(actual), float(expected),
                             rel_tol=1e-10, abs_tol=1e-12))


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{label}: cannot read JSON ({error})")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label}: JSON root must be an object")
        return None
    return payload


def _cache_manifest(root: Path, task: str, config: Mapping[str, Any],
                    errors: list[str], hash_cache: dict[Path, str]
                    ) -> dict[str, Any] | None:
    directory = root / "cache" / task
    manifest_path = directory / "manifest.json"
    digest_path = directory / "manifest.sha256"
    label = f"cache/{task}"
    if not manifest_path.is_file():
        errors.append(f"{label}: missing manifest.json")
        return None
    manifest = _load_json(manifest_path, errors, label)
    if manifest is None:
        return None
    actual_manifest_hash = sha256_file(manifest_path)
    if digest_path.is_file():
        tokens = digest_path.read_text().strip().split()
        _expect(errors, len(tokens) >= 1 and tokens[0] == actual_manifest_hash,
                f"{label}: manifest.sha256 does not match manifest.json")
    else:
        errors.append(f"{label}: missing manifest.sha256")

    host = config["official_host"]
    official = manifest.get("official_checkpoint", {})
    _expect(errors, official.get("sha256") == host["weights_sha256"],
            f"{label}: checkpoint hash differs from preregistration")
    _expect(errors, official.get("source") == host["source"],
            f"{label}: checkpoint source differs from preregistration")
    _expect(errors, official.get("source_commit") == host["source_commit"],
            f"{label}: checkpoint commit differs from preregistration")
    _expect(errors, manifest.get("task") == task,
            f"{label}: task field is {manifest.get('task')!r}")
    _expect(errors, manifest.get("source_stream") == "clean",
            f"{label}: source_stream must be 'clean'")

    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict) or "path" not in artifact:
            errors.append(f"{label}: malformed cache artifact ledger")
            continue
        artifact_path = directory / str(artifact["path"])
        if not artifact_path.is_file():
            errors.append(f"{label}: missing cache artifact {artifact_path.name}")
            continue
        digest = cached_sha256(artifact_path, hash_cache)
        _expect(errors, digest == artifact.get("sha256"),
                f"{label}: {artifact_path.name} hash differs from manifest")
        sidecar_name = artifact.get("sidecar")
        if sidecar_name:
            sidecar_path = directory / str(sidecar_name)
            if not sidecar_path.is_file():
                errors.append(f"{label}: missing sidecar {sidecar_path.name}")
            else:
                sidecar_digest = cached_sha256(sidecar_path, hash_cache)
                _expect(errors, sidecar_digest == artifact.get("sidecar_sha256"),
                        f"{label}: {sidecar_path.name} hash differs from manifest")
    return {
        "path": relative_path(manifest_path),
        "sha256": actual_manifest_hash,
        "schema": manifest.get("schema"),
        "source_stream": manifest.get("source_stream"),
        "official_checkpoint": official,
    }


def availability_results(root: Path, config: Mapping[str, Any],
                         errors: list[str], missing: list[str],
                         hash_cache: dict[Path, str]
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = config["frozen_carrier_swap"]
    ranked = [str(task) for task in frozen["tasks"]]
    diagnostic = [str(task) for task in frozen.get("diagnostic_only_tasks", [])]
    tasks = list(dict.fromkeys([*ranked, *diagnostic]))
    displays = config["display_names"]
    gate = config["availability_gate"]
    results: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for task in tasks:
        if task not in displays:
            errors.append(f"display_names has no entry for {task}")
            continue
        manifest = _cache_manifest(root, task, config, errors, hash_cache)
        if manifest is not None:
            manifests[task] = manifest
        path = root / "cache" / task / "availability.json"
        if not path.is_file():
            missing.append(f"availability/{task}")
            continue
        payload = _load_json(path, errors, f"availability/{task}")
        if payload is None:
            continue
        metric = payload.get("metric")
        threshold = (gate["categorical_accuracy_min"]
                     if metric == "accuracy"
                     else gate["continuous_r2_min"] if metric == "r2"
                     else None)
        _expect(errors, threshold is not None,
                f"availability/{task}: unsupported metric {metric!r}")
        _expect(errors, payload.get("task") == task,
                f"availability/{task}: task field mismatch")
        _expect(errors, payload.get("representation_frozen") is True,
                f"availability/{task}: representation_frozen is not true")
        _expect(errors, payload.get("representation_label_training") is False,
                f"availability/{task}: labels affected representation training")
        _expect(errors, _is_number(payload.get("value")),
                f"availability/{task}: value is not finite")
        manifest_full = _load_json(root / "cache" / task / "manifest.json",
                                   errors, f"cache/{task}")
        if manifest_full is not None:
            recorded = manifest_full.get("availability_file", {})
            _expect(errors, recorded.get("sha256") == sha256_file(path),
                    f"availability/{task}: hash differs from cache manifest")
        value = float(payload["value"]) if _is_number(payload.get("value")) else None
        results[task] = {
            "task_id": task,
            "display_name": str(displays[task]),
            "role": "carrier_ranking" if task in ranked else "diagnostic_only",
            "metric": metric,
            "value": value,
            "threshold": float(threshold) if threshold is not None else None,
            "passed": bool(value is not None and threshold is not None
                           and value >= float(threshold)),
            "chance": payload.get("chance"),
            "feature": payload.get("feature"),
            "probe": payload.get("probe"),
            "train_episodes": payload.get("train_episodes"),
            "validation_episodes": payload.get("val_episodes"),
            "source": relative_path(path),
            "source_sha256": sha256_file(path),
        }
    return results, manifests


def validate_frozen(cell: Cell, metrics: Mapping[str, Any],
                    config: Mapping[str, Any], errors: list[str]) -> None:
    label = cell.label
    wave = config["frozen_carrier_swap"]
    host = config["official_host"]
    _expect(errors, metrics.get("schema_version") == 1,
            f"{label}: schema_version must be 1")
    _expect(errors, metrics.get("study") == "official-lewm-frozen-carrier-swap",
            f"{label}: wrong study field")
    _expect(errors, metrics.get("task") == cell.task,
            f"{label}: task field mismatch")
    _expect(errors, metrics.get("arm") == cell.variant,
            f"{label}: arm field mismatch")
    _expect(errors, metrics.get("seed") == cell.seed,
            f"{label}: seed field mismatch")
    _expect(errors, metrics.get("official_host") == host["source"],
            f"{label}: official host source mismatch")
    before = metrics.get("official_host_state_sha256_before")
    after = metrics.get("official_host_state_sha256_after")
    _expect(errors, isinstance(before, str) and len(before) == 64,
            f"{label}: pre-training host state digest is malformed")
    _expect(errors, before == after,
            f"{label}: official host state changed during training")
    _expect(errors, metrics.get("frozen_host_unchanged") is True,
            f"{label}: frozen_host_unchanged is not true")
    _expect(errors, metrics.get("host_trainable_parameters") == 0,
            f"{label}: host_trainable_parameters is not zero")
    expected_epochs = 0 if cell.variant == "none" else int(wave["epochs"])
    _expect(errors, metrics.get("epochs") == expected_epochs,
            f"{label}: epochs={metrics.get('epochs')} expected {expected_epochs}")
    for key, config_key in (("batch_size", "batch_size"),
                            ("learning_rate", "learning_rate")):
        _expect(errors, _same_number(metrics.get(key), wave[config_key]),
                f"{label}: {key} differs from preregistration")
    # Trainer schema v1 did not serialize weight_decay; history/config still
    # attest all serialized settings, while the launcher fixes this argument.
    probe = metrics.get("probe", {})
    _expect(errors, isinstance(probe, dict), f"{label}: probe must be an object")
    if isinstance(probe, dict):
        _expect(errors, probe.get("metric") == "accuracy",
                f"{label}: expected categorical accuracy probe")
        _expect(errors, _is_number(probe.get("mean")),
                f"{label}: probe mean is not finite")
        if _is_number(probe.get("mean")):
            _expect(errors, 0.0 <= float(probe["mean"]) <= 1.0,
                    f"{label}: probe accuracy lies outside [0,1]")
        _expect(errors,
                probe.get("role") == "primary_registered_decision_endpoint",
                f"{label}: probe is not the registered final decision endpoint")
        expected_dim = int(host["latent_dim"]) * (int(host["context"]) + 1)
        _expect(errors, probe.get("feature_dim") == expected_dim,
                f"{label}: probe feature_dim must be {expected_dim} "
                "(three legal raw latents plus one final prior)")
        _expect(errors, "random_state=0" in str(probe.get("readout", "")),
                f"{label}: categorical readout does not fix random_state=0")
        endpoint = probe.get("endpoint_contract", {})
        _expect(errors, isinstance(endpoint, dict),
                f"{label}: probe endpoint_contract must be an object")
        if isinstance(endpoint, dict):
            expected_endpoint = {
                "schema": FROZEN_PROBE_SCHEMA,
                "decision_observation_index": FROZEN_DECISION_INDEX,
                "raw_context_history": int(host["context"]),
                "raw_context_indices": FROZEN_RAW_CONTEXT_INDICES,
                "raw_context_slice": "z[:, q-H:q]",
                "raw_context_cutoff_exclusive": FROZEN_DECISION_INDEX,
                "final_prior_index": FROZEN_DECISION_INDEX,
                "final_prior_timing": (
                    "prior_read[:, q] before consuming z[:, q]"),
                "feature_order": [
                    "raw_predecision_context_flat",
                    "final_preobservation_prior",
                ],
                "current_observation_excluded": True,
                "future_observation_consumed": False,
                "temporal_aggregation": False,
            }
            _expect(errors, endpoint == expected_endpoint,
                    f"{label}: probe endpoint contract is stale or non-causal; "
                    "reevaluate the checkpoint-only probe")
    trajectory = metrics.get("trajectory_probe", {})
    _expect(errors, isinstance(trajectory, dict),
            f"{label}: trajectory_probe must be an object")
    if isinstance(trajectory, dict):
        _expect(errors, trajectory.get("metric") == "accuracy",
                f"{label}: trajectory probe must report accuracy")
        _expect(errors, _is_number(trajectory.get("mean")),
                f"{label}: trajectory probe mean is not finite")
        if _is_number(trajectory.get("mean")):
            _expect(errors, 0.0 <= float(trajectory["mean"]) <= 1.0,
                    f"{label}: trajectory accuracy lies outside [0,1]")
        _expect(errors,
                trajectory.get("role") ==
                "exploratory_secondary_trajectory_probe",
                f"{label}: trajectory probe role changed")
        expected_trajectory_dim = int(host["latent_dim"]) * (
            int(host["context"]) + 2)
        _expect(errors, trajectory.get("feature_dim") == expected_trajectory_dim,
                f"{label}: trajectory feature_dim must be "
                f"{expected_trajectory_dim}")
        _expect(errors,
                "random_state=0" in str(trajectory.get("readout", "")),
                f"{label}: trajectory readout does not fix random_state=0")
        trajectory_endpoint = trajectory.get("endpoint_contract", {})
        expected_trajectory_endpoint = {
            "schema": FROZEN_TRAJECTORY_SCHEMA,
            "decision_observation_index": FROZEN_DECISION_INDEX,
            "raw_context_indices": FROZEN_RAW_CONTEXT_INDICES,
            "aggregation": "mean prior_read[t] for cue_off+2 <= t <= q",
            "prior_timing": "every prior_read[:, t] precedes z[:, t]",
            "current_observation_excluded": True,
            "future_observation_consumed": False,
            "temporal_aggregation": True,
        }
        _expect(errors, trajectory_endpoint == expected_trajectory_endpoint,
                f"{label}: trajectory endpoint contract changed")
    _expect(errors, _is_number(metrics.get("val_next_latent_mse"))
            and float(metrics["val_next_latent_mse"]) >= 0.0,
            f"{label}: val_next_latent_mse is invalid")

    report = metrics.get("parameter_matching", {})
    arms = report.get("arms", {}) if isinstance(report, dict) else {}
    report_key = FROZEN_PARAMETER_KEYS.get(cell.variant)
    arm_report = arms.get(report_key, {}) if report_key else {}
    _expect(errors, arm_report.get("parameters") == metrics.get("carrier_parameters"),
            f"{label}: carrier parameter count differs from match ledger")
    if cell.variant != "none":
        _expect(errors, _is_number(arm_report.get("relative_mismatch"))
                and float(arm_report["relative_mismatch"]) <= 0.005,
                f"{label}: carrier is not within 0.5% of parameter target")


def validate_frozen_artifacts(
        cell: Cell, metrics: Mapping[str, Any], root: Path,
        errors: list[str], hash_cache: dict[Path, str]) -> None:
    """Cross-check a frozen metric against its safe checkpoint and caches."""

    label = cell.label
    checkpoint_path = cell.path.with_name("carrier.pt")
    if not checkpoint_path.is_file():
        errors.append(f"{label}: missing carrier.pt")
        return
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:  # weights_only prevents arbitrary-code loading
        errors.append(f"{label}: cannot safely load carrier.pt ({error})")
        return
    if not isinstance(checkpoint, dict):
        errors.append(f"{label}: carrier.pt root must be a mapping")
        return
    embedded = checkpoint.get("metrics")
    _expect(errors, isinstance(embedded, dict),
            f"{label}: carrier.pt misses embedded metrics")
    if isinstance(embedded, dict):
        _expect(errors, embedded == metrics,
                f"{label}: carrier.pt embedded metrics differ from metrics.json")
    state = checkpoint.get("carrier_state_dict")
    if not isinstance(state, Mapping):
        errors.append(f"{label}: carrier.pt misses carrier_state_dict")
        return
    try:
        state_sha256 = carrier_state_digest(state)
    except (RuntimeError, TypeError, ValueError) as error:
        errors.append(f"{label}: invalid carrier state ({error})")
        return

    probe = metrics.get("probe", {})
    receipt = probe.get("reevaluation", {}) if isinstance(probe, dict) else {}
    _expect(errors, isinstance(receipt, dict) and bool(receipt),
            f"{label}: probe reevaluation receipt is missing")
    if not isinstance(receipt, dict):
        return
    expected_flags = {
        "mode": "checkpoint_only_no_retraining",
        "training_performed": False,
        "host_instantiated": False,
        "carrier_state_unchanged": True,
        "checkpoint_metrics_synchronized": True,
    }
    for key, expected in expected_flags.items():
        _expect(errors, receipt.get(key) == expected,
                f"{label}: reevaluation receipt {key} is invalid")
    _expect(errors, receipt.get("carrier_state_sha256") == state_sha256,
            f"{label}: reevaluation carrier-state digest differs from carrier.pt")

    trajectory = metrics.get("trajectory_probe", {})
    trajectory_receipt = (trajectory.get("reevaluation", {})
                          if isinstance(trajectory, dict) else {})
    _expect(errors, trajectory_receipt == receipt,
            f"{label}: primary and trajectory reevaluation receipts differ")

    cache_paths = {
        "train_cache_sha256": root / "cache" / cell.task / "train.npz",
        "validation_cache_sha256": root / "cache" / cell.task / "val.npz",
    }
    for key, path in cache_paths.items():
        if not path.is_file():
            errors.append(f"{label}: missing reevaluation cache {path.name}")
            continue
        actual = cached_sha256(path, hash_cache)
        _expect(errors, receipt.get(key) == actual,
                f"{label}: reevaluation {key} differs from cache artifact")


def validate_no_carrier_determinism(
        records: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any],
        errors: list[str]) -> None:
    """Require repeated no-carrier seeds to be exact deterministic copies."""

    for task in [str(value) for value in config["frozen_carrier_swap"]["tasks"]]:
        prefix = f"frozen_swap/{task}/none/seed="
        repeated = [
            {
                "probe": metrics.get("probe"),
                "trajectory_probe": metrics.get("trajectory_probe"),
                "val_next_latent_mse": metrics.get("val_next_latent_mse"),
            }
            for label, metrics in sorted(records.items())
            if label.startswith(prefix)
        ]
        canonical = {json.dumps(value, sort_keys=True) for value in repeated}
        _expect(errors, len(canonical) <= 1,
                f"frozen_swap/{task}/none: deterministic metrics vary by seed")


def validate_context(cell: Cell, metrics: Mapping[str, Any], root: Path,
                     config: Mapping[str, Any], errors: list[str],
                     hash_cache: dict[Path, str]) -> None:
    label = cell.label
    wave = config["long_context"]
    host = config["official_host"]
    history = int(cell.variant.removeprefix("h"))
    cfg = metrics.get("config", {})
    initialization = metrics.get("initialization", {})
    _expect(errors, metrics.get("schema_version") == 1,
            f"{label}: schema_version must be 1")
    _expect(errors, metrics.get("study") == "official-lewm-long-context",
            f"{label}: wrong study field")
    _expect(errors, cfg.get("history_len") == history,
            f"{label}: history_len mismatch")
    _expect(errors, cfg.get("seed") == cell.seed,
            f"{label}: seed field mismatch")
    _expect(errors, cfg.get("task_family") == TASK_FAMILIES.get(cell.task),
            f"{label}: semantic task family mismatch")
    _expect(errors, cfg.get("epochs") == int(wave["epochs"]),
            f"{label}: epochs differ from preregistration")
    for key, config_key in (("batch_size", "batch_size"), ("lr", "learning_rate"),
                            ("weight_decay", "weight_decay")):
        _expect(errors, _same_number(cfg.get(key), wave[config_key]),
                f"{label}: {key} differs from preregistration")
    _expect(errors, cfg.get("encoder_frozen") is True,
            f"{label}: encoder_frozen is not true")
    _expect(errors, cfg.get("encoder_instantiated_during_training") is False,
            f"{label}: encoder was instantiated during training")
    _expect(errors, initialization.get("checkpoint_sha256")
            == host["weights_sha256"], f"{label}: official checkpoint hash mismatch")
    _expect(errors, initialization.get("source_history") == int(host["context"]),
            f"{label}: source history mismatch")
    _expect(errors, initialization.get("target_history") == history,
            f"{label}: target history mismatch")

    for split in ("train", "validation"):
        cache = metrics.get("caches", {}).get(split, {})
        filename = "train.npz" if split == "train" else "val.npz"
        actual_path = root / "cache" / cell.task / filename
        if actual_path.is_file():
            digest = cached_sha256(actual_path, hash_cache)
            _expect(errors, cache.get("sha256") == digest,
                    f"{label}: {split} cache hash mismatch")
        else:
            errors.append(f"{label}: missing {split} cache {actual_path}")
        length = cache.get("length")
        episodes = cache.get("episodes")
        windows = cache.get("sliding_target_windows")
        if all(isinstance(value, int) for value in (length, episodes, windows)):
            _expect(errors, windows == episodes * (length - history),
                    f"{label}: {split} sliding-window count is inconsistent")

    checkpoint_mse = metrics.get("best_checkpoint_prediction_mse", {})
    _expect(errors, _is_number(checkpoint_mse.get("validation"))
            and float(checkpoint_mse["validation"]) >= 0,
            f"{label}: validation prediction MSE is invalid")
    semantic = metrics.get("semantic_target_readout", {})
    _expect(errors, semantic.get("task_family") == TASK_FAMILIES.get(cell.task),
            f"{label}: semantic readout task family mismatch")
    _expect(errors, semantic.get("metric") == "accuracy",
            f"{label}: semantic readout metric must be accuracy")
    _expect(errors, _is_number(semantic.get("value"))
            and 0 <= float(semantic["value"]) <= 1,
            f"{label}: semantic accuracy is invalid")
    for split_key in ("train_target_windows", "validation_target_windows"):
        coverage = semantic.get(split_key, {})
        _expect(errors, coverage.get("future_target_observation_consumed") is False,
                f"{label}: {split_key} consumed the target observation")
        requested = coverage.get("requested_target_windows")
        valid = coverage.get("valid_target_windows")
        _expect(errors, isinstance(requested, int) and valid == requested,
                f"{label}: {split_key} has incomplete target coverage")


def validate_rollout(cell: Cell, metrics: Mapping[str, Any],
                     config: Mapping[str, Any], errors: list[str]) -> None:
    label = cell.label
    wave = config["learned_rollout"]
    host = config["official_host"]
    expected_overshoot = 1 if cell.variant == "one_step" else 8
    _expect(errors, metrics.get("schema_version") == 1,
            f"{label}: schema_version must be 1")
    _expect(errors, metrics.get("study") == "official-lewm-learned-rollout",
            f"{label}: wrong study field")
    _expect(errors, metrics.get("task") == cell.task,
            f"{label}: task field mismatch")
    _expect(errors, metrics.get("objective") == cell.variant,
            f"{label}: objective field mismatch")
    _expect(errors, metrics.get("seed") == cell.seed,
            f"{label}: seed field mismatch")
    _expect(errors, metrics.get("overshoot_horizon") == expected_overshoot,
            f"{label}: overshoot horizon mismatch")
    _expect(errors, metrics.get("official_weights_sha256")
            == host["weights_sha256"], f"{label}: official checkpoint hash mismatch")
    _expect(errors, metrics.get("frozen_encoder") is True,
            f"{label}: encoder is not frozen")
    _expect(errors, metrics.get("epochs") == int(wave["epochs"]),
            f"{label}: epochs differ from preregistration")
    _expect(errors, metrics.get("trainable_components")
            == ["action_encoder", "predictor", "pred_proj"],
            f"{label}: trainable component ledger mismatch")
    rows = metrics.get("horizons", {})
    expected_horizons = [int(value) for value in wave["horizons"]]
    found_horizons: list[int] = []
    if isinstance(rows, dict):
        try:
            found_horizons = sorted(int(value) for value in rows)
        except ValueError:
            pass
    _expect(errors, found_horizons == sorted(expected_horizons),
            f"{label}: rollout horizon grid mismatch")
    for horizon in expected_horizons:
        row = rows.get(str(horizon), {}) if isinstance(rows, dict) else {}
        for metric in ROLLOUT_METRICS:
            _expect(errors, _is_number(row.get(metric)),
                    f"{label}: horizon {horizon} has invalid {metric}")
        for metric in ("normalized_latent_mse", "copy_last_normalized_mse",
                       "shuffled_action_normalized_mse", "pose_angular_mae",
                       "predicted_effective_rank", "target_effective_rank"):
            if _is_number(row.get(metric)):
                _expect(errors, float(row[metric]) >= 0,
                        f"{label}: horizon {horizon} has negative {metric}")
    if all(str(horizon) in rows for horizon in (1, 2, 4, 8)):
        recomputed = all(
            float(rows[str(h)]["normalized_latent_mse"])
            < float(rows[str(h)]["copy_last_normalized_mse"])
            and float(rows[str(h)]["true_action_advantage"]) > 0
            for h in (1, 2, 4, 8)
        )
        _expect(errors, metrics.get("rollout_competent_through_8") is recomputed,
                f"{label}: rollout competence flag does not match horizon metrics")


def _bootstrap_rng(key: str) -> np.random.Generator:
    material = f"{BOOTSTRAP_SEED}:{key}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed)


def summarize_seeds(values_by_seed: Mapping[int, float], key: str,
                    draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    seeds = sorted(int(seed) for seed in values_by_seed)
    if not seeds:
        return {"n": 0, "seeds": [], "values": [], "mean": None,
                "sample_sd": None, "ci95": [None, None]}
    values = np.asarray([values_by_seed[seed] for seed in seeds], dtype=np.float64)
    if not np.isfinite(values).all():
        raise AggregationError(f"non-finite value reached seed summary {key}")
    if len(values) == 1:
        low = high = float(values[0])
    else:
        rng = _bootstrap_rng(key)
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        boot = values[indices].mean(axis=1)
        low, high = (float(value) for value in np.quantile(boot, [0.025, 0.975]))
    return {
        "n": len(seeds),
        "seeds": seeds,
        "values": [float(value) for value in values],
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95": [low, high],
        "bootstrap": {"unit": "seed", "draws": draws,
                      "interval": "percentile", "level": 0.95},
    }


def paired_summary(candidate: Mapping[int, float], reference: Mapping[int, float],
                   key: str, candidate_name: str, reference_name: str,
                   expected_seeds: Sequence[int]) -> dict[str, Any]:
    seeds = sorted(set(candidate).intersection(reference))
    differences = {
        seed: float(candidate[seed]) - float(reference[seed]) for seed in seeds
    }
    output = summarize_seeds(differences, key)
    output.update({
        "estimand": f"{candidate_name} minus {reference_name}",
        "candidate": candidate_name,
        "reference": reference_name,
        "expected_pairs": len(expected_seeds),
        "complete": seeds == sorted(int(seed) for seed in expected_seeds),
        "wins": int(sum(value > 0 for value in differences.values())),
        "ties": int(sum(value == 0 for value in differences.values())),
    })
    return output


def _pooled_stratified_contrast(per_task: Mapping[str, Mapping[int, float]],
                                key: str, draws: int = BOOTSTRAP_DRAWS
                                ) -> dict[str, Any]:
    task_arrays = {
        task: np.asarray([values[seed] for seed in sorted(values)], dtype=np.float64)
        for task, values in sorted(per_task.items()) if values
    }
    if not task_arrays:
        return {
            "n_tasks": 0, "mean": None, "ci95": [None, None],
            "positive_task_seed_wins": 0, "ties": 0,
            "total_task_seed_pairs": 0,
        }
    point = float(np.mean([values.mean() for values in task_arrays.values()]))
    rng = _bootstrap_rng(key)
    task_boot = []
    for values in task_arrays.values():
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        task_boot.append(values[indices].mean(axis=1))
    boot = np.stack(task_boot).mean(axis=0)
    low, high = (float(value) for value in np.quantile(boot, [0.025, 0.975]))
    return {
        "n_tasks": len(task_arrays),
        "tasks": sorted(task_arrays),
        "n_seed_pairs": int(sum(len(values) for values in task_arrays.values())),
        "positive_task_seed_wins": int(sum(
            np.count_nonzero(values > 0) for values in task_arrays.values())),
        "ties": int(sum(
            np.count_nonzero(values == 0) for values in task_arrays.values())),
        "total_task_seed_pairs": int(sum(
            len(values) for values in task_arrays.values())),
        "mean": point,
        "ci95": [low, high],
        "per_task_means": {
            task: float(values.mean()) for task, values in task_arrays.items()
        },
        "bootstrap": {"unit": "seed within task", "task_weighting": "equal",
                      "draws": draws, "interval": "percentile", "level": 0.95},
    }


def equal_task_seed_summary(
        per_task: Mapping[str, Mapping[int, float]], key: str,
        tasks: Sequence[str], expected_seeds: Sequence[int],
        draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    """Average tasks within each matched seed, then bootstrap those seeds.

    A seed contributes only when every requested task is present.  This makes
    incomplete progress summaries honest and preserves equal task weighting;
    it never silently substitutes a one-task value for a two-task estimand.
    """
    task_order = [str(task) for task in tasks]
    if not task_order:
        raise AggregationError("equal-task pooling requires at least one task")
    seed_sets = [set(per_task.get(task, {})) for task in task_order]
    common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    seed_values = {
        seed: float(np.mean([
            float(per_task[task][seed]) for task in task_order
        ]))
        for seed in common_seeds
    }
    summary = summarize_seeds(seed_values, key, draws)
    expected = sorted(int(seed) for seed in expected_seeds)
    summary.update({
        "pooling_order": "equal-weight task mean within matched seed, then seed bootstrap",
        "task_weighting": "equal",
        "tasks": task_order,
        "expected_seeds": expected,
        "complete": common_seeds == expected,
        "per_seed_task_values": {
            str(seed): {task: float(per_task[task][seed]) for task in task_order}
            for seed in common_seeds
        },
    })
    return summary


def aggregate_frozen(records: Mapping[str, Mapping[str, Any]],
                     config: Mapping[str, Any]) -> dict[str, Any]:
    wave = config["frozen_carrier_swap"]
    displays = config["display_names"]
    expected_seeds = [int(value) for value in wave["seeds"]]
    task_order = [str(value) for value in wave["tasks"]]
    arm_order = [str(value) for value in wave["arms"]]
    tasks: dict[str, Any] = {}
    pooled_arm_accuracy: dict[str, dict[str, dict[int, float]]] = {
        arm: {} for arm in arm_order
    }
    pooled_arm_trajectory: dict[str, dict[str, dict[int, float]]] = {
        arm: {} for arm in arm_order
    }
    pooled_arm_mse: dict[str, dict[str, dict[int, float]]] = {
        arm: {} for arm in arm_order
    }
    pooled_differences: dict[str, dict[str, dict[int, float]]] = {
        arm: {} for arm in ("gru", "lstm", "ssm")
    }
    for task in task_order:
        arm_values: dict[str, dict[int, float]] = {}
        task_output: dict[str, Any] = {
            "task_id": task,
            "display_name": displays[task],
            "primary_metric": "legal decision-time accuracy",
            "arms": {},
            "paired_contrasts": {},
            "paired_vs_no_carrier": {},
        }
        for arm in arm_order:
            cells = {
                cell.seed: records[cell.label]
                for cell in expected_cells_for(records, "frozen_swap", task, arm)
            }
            scores = {seed: float(cell["probe"]["mean"])
                      for seed, cell in cells.items()}
            trajectory_scores = {
                seed: float(cell["trajectory_probe"]["mean"])
                for seed, cell in cells.items()
            }
            mse = {seed: float(cell["val_next_latent_mse"])
                   for seed, cell in cells.items()}
            arm_values[arm] = scores
            pooled_arm_accuracy[arm][task] = scores
            pooled_arm_trajectory[arm][task] = trajectory_scores
            pooled_arm_mse[arm][task] = mse
            parameter_counts = sorted({int(cell["carrier_parameters"])
                                       for cell in cells.values()})
            chance_values = sorted({float(cell["probe"]["chance"])
                                    for cell in cells.values()
                                    if _is_number(cell["probe"].get("chance"))})
            task_output["arms"][arm] = {
                "arm_id": arm,
                "display_name": ARM_DISPLAY_NAMES.get(arm, arm),
                "accuracy": summarize_seeds(
                    scores, f"frozen/{task}/{arm}/accuracy"),
                "trajectory_accuracy": summarize_seeds(
                    trajectory_scores,
                    f"frozen/{task}/{arm}/trajectory-accuracy"),
                "validation_next_latent_mse": summarize_seeds(
                    mse, f"frozen/{task}/{arm}/mse"),
                "carrier_parameters": parameter_counts[0]
                if len(parameter_counts) == 1 else parameter_counts,
                "chance": chance_values[0] if len(chance_values) == 1 else None,
            }
        no_carrier = arm_values.get("none", {})
        for arm in ("gru", "lstm", "ssm", "fixed_trust"):
            task_output["paired_vs_no_carrier"][arm] = paired_summary(
                arm_values.get(arm, {}), no_carrier,
                f"frozen/{task}/{arm}-vs-none", arm, "none", expected_seeds,
            )
        fixed = arm_values.get("fixed_trust", {})
        for reference in ("gru", "lstm", "ssm"):
            contrast = paired_summary(
                fixed, arm_values.get(reference, {}),
                f"frozen/{task}/fixed_trust-vs-{reference}",
                "fixed_trust", reference, expected_seeds,
            )
            task_output["paired_contrasts"][reference] = contrast
            common = sorted(set(fixed).intersection(arm_values.get(reference, {})))
            pooled_differences[reference][task] = {
                seed: fixed[seed] - arm_values[reference][seed] for seed in common
            }
        tasks[task] = task_output
    pooled = {
        reference: {
            "estimand": f"fixed_trust minus {reference}",
            **_pooled_stratified_contrast(
                per_task, f"frozen/pooled/fixed_trust-vs-{reference}"),
        }
        for reference, per_task in pooled_differences.items()
    }
    pooled_arms = {
        arm: {
            "arm_id": arm,
            "display_name": ARM_DISPLAY_NAMES.get(arm, arm),
            "accuracy": equal_task_seed_summary(
                pooled_arm_accuracy[arm],
                f"frozen/pooled-equal-task/{arm}/accuracy",
                task_order, expected_seeds),
            "trajectory_accuracy": equal_task_seed_summary(
                pooled_arm_trajectory[arm],
                f"frozen/pooled-equal-task/{arm}/trajectory-accuracy",
                task_order, expected_seeds),
            "validation_next_latent_mse": equal_task_seed_summary(
                pooled_arm_mse[arm],
                f"frozen/pooled-equal-task/{arm}/mse",
                task_order, expected_seeds),
        }
        for arm in arm_order
    }
    return {
        "tasks": tasks,
        "pooled_equal_task_arms": pooled_arms,
        "pooled_equal_task_contrasts": pooled,
    }


def expected_cells_for(records: Mapping[str, Mapping[str, Any]], wave: str,
                       task: str, variant: str) -> list[Cell]:
    """Reconstruct record coordinates from their canonical label."""
    prefix = f"{wave}/{task}/{variant}/seed="
    cells = []
    for label in sorted(records):
        if label.startswith(prefix):
            seed = int(label.removeprefix(prefix))
            cells.append(Cell(wave, task, variant, seed, Path()))
    return cells


def _load_raw_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"z", "xi", "meta_json", "event_cue_off"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise AggregationError(f"{path}: raw-context readout misses {missing}")
        meta_raw = np.asarray(archive["meta_json"]).reshape(()).item()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode()
        return {
            "z": np.asarray(archive["z"], dtype=np.float32),
            "xi": np.asarray(archive["xi"]),
            "cue_on": (np.asarray(archive["event_cue_on"], dtype=np.int64)
                       if "event_cue_on" in archive.files
                       else np.asarray(archive["event_cue_off"], dtype=np.int64) - 1),
            "cue_off": np.asarray(archive["event_cue_off"], dtype=np.int64),
            "decision_time": (np.asarray(archive["event_decision_time"],
                                         dtype=np.int64)
                              if "event_decision_time" in archive.files else None),
            "meta": json.loads(str(meta_raw)),
        }


def _raw_context_features(cache: Mapping[str, Any], history: int
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    z = np.asarray(cache["z"], dtype=np.float32)
    labels = np.asarray(cache["xi"])
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise AggregationError("raw-context publication readout currently requires categorical xi")
    if cache["decision_time"] is not None:
        target = np.asarray(cache["decision_time"], dtype=np.int64)
    else:
        target = np.full(len(z), int(cache["meta"].get("decision_time", z.shape[1] - 1)),
                         dtype=np.int64)
    starts = np.maximum(0, target - history)
    features = []
    for episode, (start, stop) in enumerate(zip(starts, target, strict=True)):
        window = z[episode, int(start):int(stop)]
        if not len(window):
            raise AggregationError("raw-context readout encountered an empty legal window")
        features.append(np.concatenate([window.mean(axis=0), window[-1]], axis=0))
    cue_on = np.asarray(cache["cue_on"], dtype=np.int64)
    cue_off = np.asarray(cache["cue_off"], dtype=np.int64)
    reachable = np.maximum(0, np.minimum(target, cue_off) - np.maximum(starts, cue_on))
    full = (starts <= cue_on) & (target >= cue_off)
    lengths = target - starts
    coverage = {
        "episodes": int(len(z)),
        "target_time_min": int(target.min()),
        "target_time_max": int(target.max()),
        "context_length_min": int(lengths.min()),
        "context_length_mean": float(lengths.mean()),
        "context_length_max": int(lengths.max()),
        "cue_any_frame_reachable": int((reachable > 0).sum()),
        "cue_full_window_reachable": int(full.sum()),
        "cue_frames_reachable_min": int(reachable.min()),
        "cue_frames_reachable_mean": float(reachable.mean()),
        "cue_frames_reachable_max": int(reachable.max()),
        "future_target_observation_consumed": False,
        "legal_input_contract": "z[max(0,q-H):q] only",
    }
    return np.asarray(features, dtype=np.float64), labels.astype(np.int64), coverage


def raw_context_readout(root: Path, task: str, history: int) -> dict[str, Any]:
    """Seed-independent access probe on legal frozen-encoder context."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = _load_raw_cache(root / "cache" / task / "train.npz")
    validation = _load_raw_cache(root / "cache" / task / "val.npz")
    train_x, train_y, train_coverage = _raw_context_features(train, history)
    val_x, val_y, val_coverage = _raw_context_features(validation, history)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs"),
    )
    probe.fit(train_x, train_y)
    return {
        "metric": "accuracy",
        "value": float(probe.score(val_x, val_y)),
        "chance": 1.0 / len(np.unique(train_y)),
        "feature_dimension": int(train_x.shape[1]),
        "feature": "concatenate(mean(last H legal latents), last legal latent)",
        "readout": "StandardScaler + LogisticRegression(C=1, lbfgs)",
        "seed_independent": True,
        "train_coverage": train_coverage,
        "validation_coverage": val_coverage,
    }


def _consistent_coverage(cells: Mapping[int, Mapping[str, Any]], key: str
                         ) -> dict[str, Any] | None:
    values = [cell["semantic_target_readout"].get(key) for cell in cells.values()]
    if not values:
        return None
    canonical = json.dumps(values[0], sort_keys=True)
    return values[0] if all(json.dumps(value, sort_keys=True) == canonical
                            for value in values[1:]) else {"varies_by_seed": values}


def aggregate_context(records: Mapping[str, Mapping[str, Any]], root: Path,
                      config: Mapping[str, Any]) -> dict[str, Any]:
    wave = config["long_context"]
    displays = config["display_names"]
    expected_seeds = [int(value) for value in wave["seeds"]]
    tasks: dict[str, Any] = {}
    for task in [str(value) for value in wave["tasks"]]:
        contexts: dict[str, Any] = {}
        semantic_by_history: dict[int, dict[int, float]] = {}
        mse_by_history: dict[int, dict[int, float]] = {}
        for history in sorted(int(value) for value in wave["contexts"]):
            variant = f"h{history}"
            cells = {
                cell.seed: records[cell.label]
                for cell in expected_cells_for(records, "long_context", task, variant)
            }
            semantic = {seed: float(cell["semantic_target_readout"]["value"])
                        for seed, cell in cells.items()}
            mse = {seed: float(cell["best_checkpoint_prediction_mse"]["validation"])
                   for seed, cell in cells.items()}
            semantic_by_history[history] = semantic
            mse_by_history[history] = mse
            raw = (raw_context_readout(root, task, history)
                   if (root / "cache" / task / "train.npz").is_file()
                   and (root / "cache" / task / "val.npz").is_file() else None)
            contexts[str(history)] = {
                "history": history,
                "trained_predictor_semantic_accuracy": summarize_seeds(
                    semantic, f"context/{task}/h{history}/semantic"),
                "validation_next_latent_mse": summarize_seeds(
                    mse, f"context/{task}/h{history}/mse"),
                "raw_legal_context_readout": raw,
                "trained_readout_train_coverage": _consistent_coverage(
                    cells, "train_target_windows"),
                "trained_readout_validation_coverage": _consistent_coverage(
                    cells, "validation_target_windows"),
            }
        baseline = min(int(value) for value in wave["contexts"])
        comparisons: dict[str, Any] = {}
        for history in sorted(int(value) for value in wave["contexts"]):
            if history == baseline:
                continue
            comparisons[str(history)] = {
                "trained_semantic_accuracy_delta": paired_summary(
                    semantic_by_history[history], semantic_by_history[baseline],
                    f"context/{task}/h{history}-h{baseline}/semantic",
                    f"H={history}", f"H={baseline}", expected_seeds),
                "validation_mse_delta": paired_summary(
                    mse_by_history[history], mse_by_history[baseline],
                    f"context/{task}/h{history}-h{baseline}/mse",
                    f"H={history}", f"H={baseline}", expected_seeds),
            }
        tasks[task] = {
            "task_id": task,
            "display_name": displays[task],
            "contexts": contexts,
            "paired_vs_short_context": comparisons,
        }
    return {"tasks": tasks}


def aggregate_rollout(records: Mapping[str, Mapping[str, Any]],
                      config: Mapping[str, Any]) -> dict[str, Any]:
    wave = config["learned_rollout"]
    displays = config["display_names"]
    expected_seeds = [int(value) for value in wave["seeds"]]
    horizons = sorted(int(value) for value in wave["horizons"])
    tasks: dict[str, Any] = {}
    for task in [str(value) for value in wave["tasks"]]:
        objectives: dict[str, Any] = {}
        values: dict[str, dict[int, dict[str, float]]] = {}
        for objective in [str(value) for value in wave["objectives"]]:
            cells = {
                cell.seed: records[cell.label]
                for cell in expected_cells_for(records, "rollout", task, objective)
            }
            values[objective] = {}
            horizon_output: dict[str, Any] = {}
            for horizon in horizons:
                metric_output: dict[str, Any] = {}
                for metric in ROLLOUT_METRICS:
                    by_seed = {
                        seed: float(cell["horizons"][str(horizon)][metric])
                        for seed, cell in cells.items()
                    }
                    metric_output[metric] = summarize_seeds(
                        by_seed, f"rollout/{task}/{objective}/h{horizon}/{metric}")
                    values[objective].setdefault(horizon, {})[metric] = by_seed
                copy_by_seed = values[objective][horizon][
                    "copy_last_normalized_mse"]
                if any(value <= 0.0 for value in copy_by_seed.values()):
                    raise AggregationError(
                        f"rollout/{task}/{objective}/h{horizon}: "
                        "copy-last denominator must be positive")
                ratio_by_seed = {
                    seed: (
                        values[objective][horizon]["normalized_latent_mse"][seed]
                        / copy_by_seed[seed]
                    )
                    for seed in cells
                }
                metric_output["model_to_copy_ratio"] = summarize_seeds(
                    ratio_by_seed,
                    f"rollout/{task}/{objective}/h{horizon}/model-to-copy-ratio",
                )
                horizon_output[str(horizon)] = metric_output
            pass_seeds = sorted(seed for seed, cell in cells.items()
                                if cell["rollout_competent_through_8"])
            objectives[objective] = {
                "objective_id": objective,
                "display_name": OBJECTIVE_DISPLAY_NAMES.get(objective, objective),
                "horizons": horizon_output,
                "competence_gate_through_horizon_8": {
                    "passed_seeds": pass_seeds,
                    "pass_count": len(pass_seeds),
                    "evaluated_seeds": sorted(cells),
                    "expected_seeds": expected_seeds,
                    "pass_rate": len(pass_seeds) / len(cells) if cells else None,
                    "all_evaluated_seeds_pass": bool(cells) and len(pass_seeds) == len(cells),
                    "all_preregistered_seeds_pass": pass_seeds == sorted(expected_seeds),
                },
            }
        objective_contrasts: dict[str, Any] = {}
        if "overshoot_8" in values and "one_step" in values:
            for horizon in horizons:
                objective_contrasts[str(horizon)] = {
                    metric: paired_summary(
                        values["overshoot_8"].get(horizon, {}).get(metric, {}),
                        values["one_step"].get(horizon, {}).get(metric, {}),
                        f"rollout/{task}/overshoot-minus-one/h{horizon}/{metric}",
                        "overshoot_8", "one_step", expected_seeds,
                    )
                    for metric in ROLLOUT_METRICS
                }
        tasks[task] = {
            "task_id": task,
            "display_name": displays[task],
            "objectives": objectives,
            "paired_overshoot_minus_one_step": objective_contrasts,
            "control_policy": wave["control_policy"],
        }
    return {"tasks": tasks}


def _fmt_statistic(statistic: Mapping[str, Any], digits: int = 3) -> str:
    if not statistic or statistic.get("mean") is None:
        return "pending"
    low, high = statistic["ci95"]
    return (f"{statistic['mean']:.{digits}f} "
            f"[{low:.{digits}f}, {high:.{digits}f}]")


def render_markdown(summary: Mapping[str, Any]) -> str:
    completion = summary["completion"]
    lines = ["# Paper-A expansion experiment summary", ""]
    if not completion["complete"]:
        lines += [
            f"> INCOMPLETE: {completion['completed_metrics']}/"
            f"{completion['expected_metrics']} metric cells are complete. "
            "Do not use this progress summary for publication claims.", "",
        ]
    lines += [
        "All intervals below are deterministic 95% percentile bootstrap "
        "intervals over optimizer/model seeds. Contrasts are paired by seed.", "",
        "## Frozen official-encoder availability", "",
        "| Semantic task | Role | Metric | Result | Gate | Status |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for item in summary["availability"].values():
        lines.append(
            f"| {item['display_name']} | {item['role'].replace('_', ' ')} | "
            f"{item['metric']} | {item['value']:.3f} | {item['threshold']:.3f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )

    lines += ["", "## Frozen SIGReg LeWM carrier swap", ""]
    for task in summary["frozen_carrier_swap"]["tasks"].values():
        lines += [f"### {task['display_name']}", "",
                  "| Carrier | Final legal accuracy, mean [95% CI] | "
                  "Trajectory diagnostic [95% CI] | Paired delta vs. no carrier | "
                  "Next-latent MSE | Parameters |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm_id, arm in task["arms"].items():
            delta = ("reference" if arm_id == "none" else
                     _fmt_statistic(task["paired_vs_no_carrier"][arm_id]))
            lines.append(
                f"| {arm['display_name']} | {_fmt_statistic(arm['accuracy'])} | "
                f"{_fmt_statistic(arm['trajectory_accuracy'])} | {delta} | "
                f"{_fmt_statistic(arm['validation_next_latent_mse'], 4)} | "
                f"{arm['carrier_parameters']} |"
            )
        lines += ["", "| Fixed-trust contrast | Paired difference [95% CI] | Wins |",
                  "|---|---:|---:|"]
        for reference, contrast in task["paired_contrasts"].items():
            lines.append(
                f"| vs. {ARM_DISPLAY_NAMES[reference]} | {_fmt_statistic(contrast)} | "
                f"{contrast['wins']}/{contrast['n']} |"
            )
        lines.append("")

    pooled_arms = summary["frozen_carrier_swap"]["pooled_equal_task_arms"]
    lines += [
        "### Equal-task pooled carrier results", "",
        "Each value first averages the semantic tasks within the same seed; "
        "the interval then bootstraps those seed-level task means.", "",
        "| Carrier | Final accuracy [95% CI] | Trajectory diagnostic [95% CI] | "
        "Next-latent MSE [95% CI] | Seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in pooled_arms.values():
        lines.append(
            f"| {arm['display_name']} | {_fmt_statistic(arm['accuracy'])} | "
            f"{_fmt_statistic(arm['trajectory_accuracy'])} | "
            f"{_fmt_statistic(arm['validation_next_latent_mse'], 4)} | "
            f"{arm['accuracy']['n']} |"
        )
    lines += ["", "| Fixed-trust pooled contrast | Difference [95% CI] | "
              "Positive task-seed pairs |",
              "|---|---:|---:|"]
    for reference, contrast in summary["frozen_carrier_swap"][
            "pooled_equal_task_contrasts"].items():
        lines.append(
            f"| vs. {ARM_DISPLAY_NAMES[reference]} | "
            f"{_fmt_statistic(contrast)} | "
            f"{contrast['positive_task_seed_wins']}/"
            f"{contrast['total_task_seed_pairs']} |"
        )
    lines.append("")

    lines += ["## Long-context controls", ""]
    for task in summary["long_context"]["tasks"].values():
        lines += [f"### {task['display_name']}", "",
                  "| Context | Raw legal-context accuracy | Cue frames reachable "
                  "(validation) | Trained-predictor accuracy [95% CI] | "
                  "Next-latent MSE [95% CI] |",
                  "|---:|---:|---:|---:|---:|"]
        for context in task["contexts"].values():
            raw = context.get("raw_legal_context_readout")
            if raw:
                coverage = raw["validation_coverage"]
                raw_value = f"{raw['value']:.3f}"
                cue = (f"{coverage['cue_frames_reachable_mean']:.1f} mean; "
                       f"{coverage['cue_any_frame_reachable']}/{coverage['episodes']} any")
            else:
                raw_value = cue = "pending"
            lines.append(
                f"| {context['history']} | {raw_value} | {cue} | "
                f"{_fmt_statistic(context['trained_predictor_semantic_accuracy'])} | "
                f"{_fmt_statistic(context['validation_next_latent_mse'], 4)} |"
            )
        lines.append("")

    lines += ["## Learned-model rollout", ""]
    metric_headers = (
        "Normalized latent MSE", "Copy-last MSE", "Model/copy ratio",
        "True-action advantage",
        "Pose angular MAE", "Predicted rank",
    )
    for task in summary["learned_rollout"]["tasks"].values():
        lines.append(f"### {task['display_name']}")
        lines.append("")
        for objective in task["objectives"].values():
            gate = objective["competence_gate_through_horizon_8"]
            lines += [
                f"**{objective['display_name']}** — competence gate: "
                f"{gate['pass_count']}/{len(gate['evaluated_seeds'])} evaluated seeds pass.",
                "",
                "| Horizon | " + " | ".join(metric_headers) + " |",
                "|---:|" + "---:|" * len(metric_headers),
            ]
            for horizon, metrics in objective["horizons"].items():
                lines.append(
                    f"| {horizon} | "
                    f"{_fmt_statistic(metrics['normalized_latent_mse'])} | "
                    f"{_fmt_statistic(metrics['copy_last_normalized_mse'])} | "
                    f"{_fmt_statistic(metrics['model_to_copy_ratio'])} | "
                    f"{_fmt_statistic(metrics['true_action_advantage'])} | "
                    f"{_fmt_statistic(metrics['pose_angular_mae'])} | "
                    f"{_fmt_statistic(metrics['predicted_effective_rank'])} |"
                )
            lines.append("")

    lines += ["## Reproducibility ledger", "",
              f"- Pre-specified configuration record: `{summary['provenance']['config']['path']}` "
              f"(SHA-256 `{summary['provenance']['config']['sha256']}`)",
              f"- Official host weights: `"
              f"{summary['provenance']['official_host']['weights_sha256']}`",
              f"- Completed metric files: {completion['completed_metrics']}",
              f"- Bootstrap draws: {summary['analysis']['bootstrap_draws']}", ""]
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def build_summary(root: Path, config_path: Path, allow_incomplete: bool
                  ) -> dict[str, Any]:
    config = load_config(config_path)
    cells = expected_cells(root, config)
    expected_by_path = {cell.path.resolve(): cell for cell in cells}
    errors: list[str] = []
    missing: list[str] = []
    hash_cache: dict[Path, str] = {}

    weights_path = root / "pretrained/lewm-reacher/weights.pt"
    weights_sha256: str | None = None
    if weights_path.is_file():
        weights_sha256 = cached_sha256(weights_path, hash_cache)
        _expect(errors, weights_sha256 == config["official_host"]["weights_sha256"],
                "local official-host weights differ from preregistered SHA-256")
    else:
        errors.append(f"missing preregistered official-host weights: {weights_path}")

    discovered = set()
    for wave in ("frozen_swap", "long_context", "rollout"):
        directory = root / wave
        if directory.exists():
            discovered.update(path.resolve() for path in directory.glob("**/metrics.json"))
    unexpected = sorted(path for path in discovered if path not in expected_by_path)
    errors.extend(f"unexpected metric outside preregistered grid: {relative_path(path)}"
                  for path in unexpected)

    records: dict[str, dict[str, Any]] = {}
    source_metrics: dict[str, str] = {}
    for cell in cells:
        if not cell.path.is_file():
            missing.append(cell.label)
            continue
        metrics = _load_json(cell.path, errors, cell.label)
        if metrics is None:
            continue
        if cell.wave == "frozen_swap":
            validate_frozen(cell, metrics, config, errors)
            validate_frozen_artifacts(cell, metrics, root, errors, hash_cache)
        elif cell.wave == "long_context":
            validate_context(cell, metrics, root, config, errors, hash_cache)
        else:
            validate_rollout(cell, metrics, config, errors)
        records[cell.label] = metrics
        source_metrics[relative_path(cell.path)] = sha256_file(cell.path)

    validate_no_carrier_determinism(records, config, errors)

    frozen_state_digests = {
        str(metrics["official_host_state_sha256_before"])
        for label, metrics in records.items() if label.startswith("frozen_swap/")
    }
    _expect(errors, len(frozen_state_digests) <= 1,
            "frozen carrier cells did not load one byte-identical host state")
    frozen_match_ledgers = {
        json.dumps(metrics.get("parameter_matching"), sort_keys=True)
        for label, metrics in records.items() if label.startswith("frozen_swap/")
    }
    _expect(errors, len(frozen_match_ledgers) <= 1,
            "frozen carrier cells disagree on the parameter-matching ledger")
    for arm in [str(value) for value in config["frozen_carrier_swap"]["arms"]]:
        counts = {
            int(metrics["carrier_parameters"])
            for label, metrics in records.items()
            if label.startswith("frozen_swap/") and f"/{arm}/" in label
        }
        _expect(errors, len(counts) <= 1,
                f"frozen carrier parameter count varies across cells for arm {arm}")

    availability, manifests = availability_results(
        root, config, errors, missing, hash_cache)
    ranked_tasks = [str(value) for value in config["frozen_carrier_swap"]["tasks"]]
    for task in ranked_tasks:
        result = availability.get(task)
        if result is not None:
            _expect(errors, result["passed"],
                    f"availability/{task}: preregistered carrier-ranking gate did not pass")

    if errors:
        raise AggregationError(
            "Paper-A expansion provenance validation failed:\n- "
            + "\n- ".join(sorted(set(errors)))
        )
    if missing and not allow_incomplete:
        preview = "\n- ".join(sorted(missing))
        raise AggregationError(
            f"Paper-A expansion grid is incomplete ({len(missing)} missing products):\n"
            f"- {preview}\nRerun after all jobs finish, or pass --allow-incomplete "
            "for a clearly marked progress summary."
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper-a-expansion",
        "field_guide": {
            "completion": "Preregistered grid coverage; publication use requires complete=true.",
            "availability": "Frozen official-encoder task-admission gates.",
            "frozen_carrier_swap": (
                "Per-task carrier summaries, matched-seed equal-task arm pools, "
                "and paired accuracy contrasts."
            ),
            "long_context": "Raw legal-context access and trained predictor-output readouts by H.",
            "learned_rollout": "Seed aggregates by objective, task, horizon, and competence gate.",
            "analysis": "Bootstrap estimand and deterministic resampling contract.",
            "validation": "Fail-closed grid, host, schema, and parameter-match checks.",
            "provenance": "Configuration, cache, checkpoint, and source-metric SHA-256 ledger.",
        },
        "completion": {
            "complete": not missing,
            "allow_incomplete": bool(allow_incomplete),
            "expected_metrics": len(cells),
            "completed_metrics": len(records),
            "missing_count": len(missing),
            "missing": sorted(missing),
        },
        "semantic_task_names": {
            str(key): str(value) for key, value in config["display_names"].items()
        },
        "availability": availability,
        "frozen_carrier_swap": aggregate_frozen(records, config),
        "long_context": aggregate_context(records, root, config),
        "learned_rollout": aggregate_rollout(records, config),
        "analysis": {
            "confidence_level": 0.95,
            "bootstrap_interval": "percentile",
            "bootstrap_unit": "optimizer/model seed",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_contrast_unit": "same task and seed",
            "raw_context_readout_seed_independent": True,
        },
        "validation": {
            "fail_closed": True,
            "grid_complete": not missing,
            "all_discovered_cells_schema_and_provenance_valid": True,
            "official_host_file_hash_matches_preregistration": True,
            "frozen_host_unchanged_within_every_completed_cell": True,
            "frozen_host_state_consistent_across_completed_cells": True,
            "parameter_matching_ledger_consistent_across_completed_cells": True,
        },
        "provenance": {
            "config": {"path": relative_path(config_path),
                       "sha256": sha256_file(config_path)},
            "official_host": config["official_host"],
            "official_host_weights": {
                "path": relative_path(weights_path), "sha256": weights_sha256},
            "frozen_host_state_sha256": (
                next(iter(frozen_state_digests)) if frozen_state_digests else None),
            "cache_manifests": manifests,
            "source_metric_sha256": dict(sorted(source_metrics.items())),
        },
    }
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Paper-A expansion output root")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="preregistered YAML experiment matrix")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write a prominently marked progress summary")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        summary = build_summary(args.root, args.config, args.allow_incomplete)
    except AggregationError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    json_path = args.root / "summary.json"
    markdown_path = args.root / "summary.md"
    _atomic_write(json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_markdown(summary))
    completion = summary["completion"]
    status = "complete" if completion["complete"] else "INCOMPLETE"
    print(f"[paper-a-aggregate] {status}: "
          f"{completion['completed_metrics']}/{completion['expected_metrics']} cells; "
          f"wrote {json_path} and {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
