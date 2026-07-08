#!/usr/bin/env python3
"""Evaluate one frozen carrier with the locked TwoRoom waypoint consumer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier  # noqa: E402
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    load_verified_npz,
    sha256_arrays,
    stable_json,
)
from lewm.official_tasks.matched_memory import balanced_joint_labels  # noqa: E402
from scripts.paper_a_evidence_age import (  # noqa: E402
    configure_determinism,
    fixed_endpoint_features,
)
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    SEEDS,
    load_locked_spec,
    output_path,
    resolve_input_path,
    sha256_file,
    validate_device,
)
from scripts.prepare_paper_a_matched_host import host_manifest_path  # noqa: E402
from scripts.prepare_paper_a_tworoom_use import (  # noqa: E402
    deck_path,
    gate_path,
)
from scripts.train_frozen_official_swap import state_digest  # noqa: E402
from scripts.train_official_pusht_carrier import _carrier_prior  # noqa: E402
from scripts.train_paper_a_matched_host import (  # noqa: E402
    _aligned_latent,
    _load_base,
    _load_cue,
    carrier_directory,
)


DECK_KEYS = {
    "z", "actions", "color_label", "location_label", "combination_label",
    "decision_state", "goal_waypoints", "success_matrix", "distance_matrix",
    "controller_target_success", "controller_final_state", "reset_replay",
    "random_choice",
}
DECISION_INDEX = 19
ENDPOINT_HISTORY = 3
CUE_AGE = 15
INFERENCE_BATCH_SIZE = 64


class TwoRoomUseEvaluationError(RuntimeError):
    """The locked TwoRoom use cell is incomplete or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TwoRoomUseEvaluationError(message)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def use_cell_directory(spec: Mapping[str, Any], arm: str, seed: int) -> Path:
    return output_path(spec, "use") / "cells" / arm / f"seed-{seed}"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as error:
        raise TwoRoomUseEvaluationError(
            f"formal artifact is outside the repository: {path}") from error


def _record(path: Path) -> dict[str, str]:
    _require(path.is_file(), f"missing formal artifact: {path}")
    return {"path": _relative(path), "sha256": sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TwoRoomUseEvaluationError(
            f"cannot load JSON artifact {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _expected_labels(count: int, seed: int) -> dict[str, np.ndarray]:
    labels = balanced_joint_labels(count, seed)
    return {
        "color_label": labels.color,
        "location_label": labels.location,
        "combination_label": labels.combination,
    }


def _validate_labels(arrays: Mapping[str, np.ndarray], *, count: int,
                     seed: int, label: str) -> None:
    expected = _expected_labels(count, seed)
    for key, reference in expected.items():
        value = np.asarray(arrays[key])
        _require(value.dtype == np.int64 and value.shape == (count,),
                 f"{label} {key} has an unexpected dtype or shape")
        _require(np.array_equal(value, reference),
                 f"{label} {key} differs from its locked seed")
    joint = np.asarray(arrays["combination_label"])
    color = np.asarray(arrays["color_label"])
    location = np.asarray(arrays["location_label"])
    _require(np.array_equal(joint, color * 4 + location),
             f"{label} joint labels do not encode color x location")
    _require(np.array_equal(
        np.bincount(joint, minlength=16), np.full(16, count // 16)),
        f"{label} joint labels are not exactly balanced")


def _load_admitted_deck(
        spec: Mapping[str, Any]) -> tuple[dict[str, np.ndarray],
                                         dict[str, dict[str, str]]]:
    path = deck_path(dict(spec))
    sidecar_path = path.with_suffix(path.suffix + ".json")
    gate_file = gate_path(dict(spec))
    gate = _load_json(gate_file)
    use = spec["tworoom_use"]
    episodes = int(use["heldout_episodes"])
    expected_gate = {
        "schema_version": 1,
        "study": spec["study"],
        "lock": spec["_lock"],
        "status": "admitted",
        "admitted": True,
        "episodes": episodes,
        "oracle_success_min": use["oracle_success_min"],
        "oracle_per_class_success_min": use["oracle_per_class_success_min"],
        "off_diagonal_false_success_max": use[
            "off_diagonal_false_success_max"],
        "replay_fidelity_min": use["replay_fidelity_min"],
        "vendor_commit": use["upstream_environment"]["revision"],
        "vendor_clean": True,
    }
    failed = [key for key, expected in expected_gate.items()
              if gate.get(key) != expected]
    _require(not failed, f"TwoRoom use gate differs in fields {failed}")
    _require(float(gate.get("oracle_executed_success", -1.0))
             >= float(use["oracle_success_min"]),
             "TwoRoom oracle controller gate is below threshold")
    _require(float(gate.get("reset_replay_fidelity", -1.0))
             >= float(use["replay_fidelity_min"]),
             "TwoRoom reset replay gate is below threshold")
    oracle_per_class = gate.get("oracle_per_class_executed_success")
    _require(isinstance(oracle_per_class, list) and len(oracle_per_class) == 4
             and min(map(float, oracle_per_class))
             >= float(use["oracle_per_class_success_min"]),
             "TwoRoom per-class oracle controller gate is below threshold")
    _require(float(gate.get("off_diagonal_false_success", 1.0))
             <= float(use["off_diagonal_false_success_max"]),
             "TwoRoom off-diagonal false-success gate is above threshold")

    arrays, sidecar = load_verified_npz(path)
    expected_sidecar = {
        "schema": "paper_a_matched_tworoom_use_deck_v1",
        "study": spec["study"],
        "lock": spec["_lock"],
        "episodes": episodes,
        "cue_age": CUE_AGE,
        "prefix_label_blind": True,
        "physical_gpu": 0,
        "fixed_physics_seed": int(use["physics_seed"]),
    }
    sidecar_failed = [key for key, expected in expected_sidecar.items()
                      if sidecar.get(key) != expected]
    _require(not sidecar_failed,
             f"TwoRoom use deck differs in fields {sidecar_failed}")
    _require(set(arrays) == DECK_KEYS,
             f"TwoRoom use deck keys differ: {sorted(set(arrays) ^ DECK_KEYS)}")

    deck_record = gate.get("deck")
    _require(isinstance(deck_record, dict), "TwoRoom gate has no deck record")
    _require(Path(str(deck_record.get("path", ""))).resolve() == path.resolve(),
             "TwoRoom gate points to a different deck")
    _require(Path(str(deck_record.get("sidecar", ""))).resolve()
             == sidecar_path.resolve(),
             "TwoRoom gate points to a different deck sidecar")
    deck_hash = sha256_file(path)
    sidecar_hash = sha256_file(sidecar_path)
    _require(deck_record.get("sha256") == deck_hash,
             "TwoRoom gate deck hash differs")
    _require(deck_record.get("sidecar_sha256") == sidecar_hash,
             "TwoRoom gate deck-sidecar hash differs")

    expected_shapes = {
        "z": (episodes, 20, 192),
        "actions": (episodes, 19, 10),
        "color_label": (episodes,),
        "location_label": (episodes,),
        "combination_label": (episodes,),
        "decision_state": (episodes, 2),
        "goal_waypoints": (4, 2),
        "success_matrix": (episodes, 4, 4),
        "distance_matrix": (episodes, 4, 4),
        "controller_target_success": (episodes, 4),
        "controller_final_state": (episodes, 4, 2),
        "reset_replay": (episodes, 4),
        "random_choice": (episodes,),
    }
    for key, shape in expected_shapes.items():
        _require(np.asarray(arrays[key]).shape == shape,
                 f"TwoRoom deck {key} has shape {np.asarray(arrays[key]).shape}, "
                 f"expected {shape}")
    for key in ("z", "actions", "decision_state", "goal_waypoints",
                "distance_matrix", "controller_final_state"):
        _require(np.isfinite(np.asarray(arrays[key])).all(),
                 f"TwoRoom deck {key} contains a non-finite value")
    _require(np.asarray(arrays["z"]).dtype == np.float32,
             "TwoRoom deck latents must be float32")
    _require(np.asarray(arrays["actions"]).dtype == np.float32,
             "TwoRoom deck actions must be float32")
    success = np.asarray(arrays["success_matrix"])
    controller_success = np.asarray(arrays["controller_target_success"])
    reset_replay = np.asarray(arrays["reset_replay"])
    distance = np.asarray(arrays["distance_matrix"])
    random_choice = np.asarray(arrays["random_choice"])
    _require(np.isin(success, (0, 1)).all()
             and np.isin(controller_success, (0, 1)).all()
             and np.isin(reset_replay, (0, 1)).all(),
             "TwoRoom success record is not binary")
    _require(np.array_equal(
        success, (distance < float(use["success_radius"])).astype(np.int8)),
        "TwoRoom success matrix differs from its locked distance threshold")
    _require(np.array_equal(
        controller_success, success[:, np.arange(4), np.arange(4)]),
        "TwoRoom controller-target diagnostic differs from success diagonal")
    _require(random_choice.dtype == np.int64
             and np.all((0 <= random_choice) & (random_choice < 4)),
             "TwoRoom random choices are invalid")
    _validate_labels(
        arrays, count=episodes, seed=int(use["label_seed"]), label="use deck")
    rows = np.arange(episodes)
    labels = np.asarray(arrays["location_label"], dtype=np.int64)
    oracle = success[rows, labels, labels]
    random = success[rows, random_choice, labels]
    reproduced_per_class = [
        float(oracle[labels == label].mean()) for label in range(4)
    ]
    reproduced_off_diagonal = float(
        success[:, ~np.eye(4, dtype=np.bool_)].mean())
    _require(abs(float(oracle.mean())
                 - float(gate["oracle_executed_success"])) <= 1e-12,
             "TwoRoom gate oracle result does not reproduce from the deck")
    _require(abs(float(random.mean())
                 - float(gate["realized_random_executed_success"])) <= 1e-12,
             "TwoRoom gate random result does not reproduce from the deck")
    _require(np.allclose(reproduced_per_class, oracle_per_class,
                         rtol=0.0, atol=1e-12),
             "TwoRoom gate per-class oracle result does not reproduce")
    _require(abs(reproduced_off_diagonal
                 - float(gate["off_diagonal_false_success"])) <= 1e-12,
             "TwoRoom off-diagonal false-success result does not reproduce")
    _require(abs(float(controller_success.mean())
                 - float(gate["controller_selected_target_success"])) <= 1e-12,
             "TwoRoom controller-target diagnostic does not reproduce")
    _require(abs(float(reset_replay.mean())
                 - float(gate["reset_replay_fidelity"])) <= 1e-12,
             "TwoRoom reset-replay diagnostic does not reproduce")
    provenance = {
        "gate": _record(gate_file),
        "deck": {
            **_record(path),
            "sidecar": _relative(sidecar_path),
            "sidecar_sha256": sidecar_hash,
        },
    }
    return arrays, provenance


def _load_carrier(
        spec: Mapping[str, Any], arm: str, seed: int,
        device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    directory = carrier_directory(dict(spec), "tworoom", arm, seed)
    manifest_path = directory / "manifest.json"
    metrics_path = directory / "metrics.json"
    checkpoint_path = directory / "carrier.pt"
    history_path = directory / "history.csv"
    manifest = _load_json(manifest_path)
    metrics = _load_json(metrics_path)
    expected_manifest = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "host": "tworoom", "arm": arm, "seed": seed,
    }
    failed = [key for key, expected in expected_manifest.items()
              if manifest.get(key) != expected]
    _require(not failed, f"source carrier manifest differs in fields {failed}")
    expected_metrics = {
        **expected_manifest,
        "branch": "matched-color-location-fixed-endpoint",
        "device": "cuda:0", "physical_gpu": 0,
        "age_balanced_mixture": True, "frozen_host_unchanged": True,
        "validation_labels_used_for_fitting": False,
    }
    metric_failed = [key for key, expected in expected_metrics.items()
                     if metrics.get(key) != expected]
    _require(not metric_failed,
             f"source carrier metrics differ in fields {metric_failed}")
    _require(metrics.get("ages") == [4, 8, 15]
             and metrics.get("targets") == ["color", "location"],
             "source carrier task grid differs")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict)
             and set(artifacts) == {"metrics", "checkpoint", "history"},
             "source carrier manifest has an unexpected artifact set")
    source_paths = {
        "metrics": metrics_path,
        "checkpoint": checkpoint_path,
        "history": history_path,
    }
    for key, path in source_paths.items():
        _require(path.is_file(), f"source carrier artifact is missing: {path}")
        _require(artifacts[key].get("path") == path.name,
                 f"source carrier {key} path differs")
        _require(artifacts[key].get("sha256") == sha256_file(path),
                 f"source carrier {key} hash differs")

    host_record = metrics.get("host_manifest")
    current_host_manifest = host_manifest_path(dict(spec), "tworoom")
    _require(isinstance(host_record, dict)
             and host_record.get("path") == _relative(current_host_manifest)
             and host_record.get("sha256") == sha256_file(current_host_manifest),
             "source carrier refers to a different TwoRoom admission")
    host_manifest = _load_json(current_host_manifest)
    _require(host_manifest.get("lock") == spec["_lock"]
             and host_manifest.get("status") == "admitted"
             and host_manifest.get("all_targets_ages_admitted") is True
             and host_manifest.get("frozen_host_unchanged") is True,
             "TwoRoom host is not formally admitted")

    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise TwoRoomUseEvaluationError(
            f"cannot safely load source carrier checkpoint: {error}") from error
    _require(isinstance(payload, dict)
             and set(payload) == {"carrier_state_dict", "metrics"},
             "source carrier checkpoint payload differs")
    _require(payload["metrics"] == metrics,
             "source carrier checkpoint embeds different metrics")
    carrier = make_frozen_carrier(arm, 192, 10)
    carrier.load_state_dict(payload["carrier_state_dict"], strict=True)
    checkpoint_digest = state_digest(carrier)
    _require(checkpoint_digest == metrics.get("carrier_state_sha256"),
             "source carrier state digest differs from training metrics")
    carrier.requires_grad_(False)
    carrier = carrier.to(device).eval()
    _require(state_digest(carrier) == checkpoint_digest,
             "device transfer changed source carrier state")
    return carrier, {
        "carrier_manifest": _record(manifest_path),
        "carrier_checkpoint": _record(checkpoint_path),
        "carrier_metrics": _record(metrics_path),
        "carrier_state_sha256": checkpoint_digest,
    }


def _fit_consumer(train_x: np.ndarray, train_y: np.ndarray,
                  eval_x: np.ndarray, protocol: Mapping[str, Any]
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    train_x = np.asarray(train_x, dtype=np.float32)
    eval_x = np.asarray(eval_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    _require(train_x.ndim == 2 and eval_x.ndim == 2
             and train_x.shape[1] == eval_x.shape[1] == 768,
             "TwoRoom consumer requires the locked 768-D endpoint")
    _require(np.isfinite(train_x).all() and np.isfinite(eval_x).all(),
             "TwoRoom consumer features contain a non-finite value")
    _require(np.array_equal(np.unique(train_y), np.arange(4)),
             "TwoRoom consumer training labels do not contain four classes")
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=float(protocol["logistic_c"]), solver=str(protocol["solver"]),
        max_iter=int(protocol["max_iter"]),
        random_state=int(protocol["random_state"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(train_x), train_y)
    prediction = model.predict(scaler.transform(eval_x)).astype(np.int64)
    digest = sha256_arrays({
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "classes": np.asarray(model.classes_, dtype=np.int64),
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "intercept": np.asarray(model.intercept_, dtype=np.float64),
    })
    return prediction, {
        "model": "StandardScaler+multinomial LogisticRegression",
        "arm_id_feature_present": False,
        "target": "location",
        "fit_episodes": int(len(train_y)),
        "feature_dimension": int(train_x.shape[1]),
        "classes": model.classes_.astype(int).tolist(),
        "logistic_c": float(protocol["logistic_c"]),
        "solver": str(protocol["solver"]),
        "max_iter": int(protocol["max_iter"]),
        "random_state": int(protocol["random_state"]),
        "multiclass_objective": "native multinomial for lbfgs with four classes",
        "iterations": [int(value) for value in model.n_iter_],
        "parameter_sha256": digest,
        "sklearn_version": sklearn.__version__,
    }


def _execution_result(prediction: np.ndarray, labels: np.ndarray,
                      joint_labels: np.ndarray, success_matrix: np.ndarray,
                      distance_matrix: np.ndarray,
                      random_choice: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    joint_labels = np.asarray(joint_labels, dtype=np.int64)
    success = np.asarray(success_matrix)
    distance = np.asarray(distance_matrix, dtype=np.float64)
    random_choice = np.asarray(random_choice, dtype=np.int64)
    episodes = len(labels)
    _require(prediction.shape == labels.shape == joint_labels.shape
             == random_choice.shape == (episodes,),
             "TwoRoom episode-level choice arrays are not aligned")
    _require(success.shape == distance.shape == (episodes, 4, 4),
             "TwoRoom execution matrices are not aligned")
    _require(np.all((0 <= prediction) & (prediction < 4))
             and np.all((0 <= labels) & (labels < 4))
             and np.all((0 <= random_choice) & (random_choice < 4)),
             "TwoRoom execution choice is outside the four-way vocabulary")
    _require(np.isfinite(distance).all(),
             "TwoRoom execution distance matrix is non-finite")
    rows = np.arange(episodes)
    goal_correct = prediction == labels
    executed = success[rows, prediction, labels].astype(np.int8)
    selected_distance = distance[rows, prediction, labels]
    oracle_success = success[rows, labels, labels].astype(np.int8)
    random_success = success[rows, random_choice, labels].astype(np.int8)
    oracle_distance = distance[rows, labels, labels]
    random_distance = distance[rows, random_choice, labels]
    regret = selected_distance - oracle_distance
    return {
        "labels": labels.tolist(),
        "joint_labels": joint_labels.tolist(),
        "predictions": prediction.tolist(),
        "goal_correct": goal_correct.astype(np.int8).tolist(),
        "executed_success": executed.tolist(),
        "selected_distance": selected_distance.tolist(),
        "oracle_success": oracle_success.tolist(),
        "random_success": random_success.tolist(),
        "oracle_distance": oracle_distance.tolist(),
        "random_distance": random_distance.tolist(),
        "distance_regret": regret.tolist(),
        "goal_selection_accuracy": float(goal_correct.mean()),
        "goal_selection_balanced_accuracy": float(
            balanced_accuracy_score(labels, prediction)),
        "executed_success_rate": float(executed.mean()),
        "mean_selected_distance": float(selected_distance.mean()),
        "mean_distance_regret": float(regret.mean()),
        "oracle_executed_success_rate": float(oracle_success.mean()),
        "random_executed_success_rate": float(random_success.mean()),
        "mean_oracle_distance": float(oracle_distance.mean()),
        "mean_random_distance": float(random_distance.mean()),
        "executed_success_gain_vs_random": float(
            executed.mean() - random_success.mean()),
        "selected_distance_reduction_vs_random": float(
            random_distance.mean() - selected_distance.mean()),
    }


def _require_physical_gpu_zero(device_name: str) -> torch.device:
    _require(device_name == "cuda:0",
             "TwoRoom use evaluation only permits cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        first = visible.split(",", maxsplit=1)[0].strip()
        _require(first == "0",
                 "logical cuda:0 is not mapped to registered physical GPU 0")
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1,
             "registered physical GPU 0 is unavailable")
    torch.cuda.set_device(0)
    _require(torch.cuda.current_device() == 0,
             "failed to select registered physical GPU 0")
    return torch.device("cuda:0")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing TwoRoom use-cell writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    destination = use_cell_directory(spec, args.arm, args.seed)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    configure_determinism(args.seed)
    device = _require_physical_gpu_zero(args.device)

    deck, provenance = _load_admitted_deck(spec)
    train_base = _load_base(spec, "tworoom", "train")
    train_cue = _load_cue(spec, "tworoom", "train", CUE_AGE)
    train_z = _aligned_latent(train_base, train_cue)
    train_actions = np.asarray(train_base["actions"], dtype=np.float32)
    train_labels = np.asarray(train_cue["location_label"], dtype=np.int64)
    _validate_labels(
        train_cue, count=int(spec["selection"]["train_episodes"]),
        seed=int(spec["selection"]["train_label_seed"]), label="training cache")
    _require(train_z.shape == (1200, 20, 192)
             and train_actions.shape == (1200, 19, 10),
             "TwoRoom age-15 training cache has unexpected shapes")
    _require(np.isfinite(train_z).all() and np.isfinite(train_actions).all(),
             "TwoRoom age-15 training cache contains non-finite values")

    deck_z = np.asarray(deck["z"], dtype=np.float32)
    deck_actions = np.asarray(deck["actions"], dtype=np.float32)
    pooled_train_x: list[np.ndarray] = []
    consumer_sources: dict[str, Any] = {}
    requested_provenance: dict[str, Any] | None = None
    requested_deck_x: np.ndarray | None = None
    carrier_before: str | None = None
    carrier_after: str | None = None
    # One consumer per seed is fit on an equal pool of all five arms.  The arm
    # name is never appended to the feature, and the requested cell changes
    # only which frozen-carrier endpoint is passed through this shared model.
    for training_arm in ARMS:
        carrier, source = _load_carrier(
            spec, training_arm, args.seed, device)
        before = str(source.pop("carrier_state_sha256"))
        train_prior = _carrier_prior(
            carrier, train_z, train_actions, device,
            batch_size=INFERENCE_BATCH_SIZE)
        pooled_train_x.append(fixed_endpoint_features(
            train_z, train_prior, DECISION_INDEX, history=ENDPOINT_HISTORY))
        if training_arm == args.arm:
            deck_prior = _carrier_prior(
                carrier, deck_z, deck_actions, device,
                batch_size=INFERENCE_BATCH_SIZE)
            requested_deck_x = fixed_endpoint_features(
                deck_z, deck_prior, DECISION_INDEX, history=ENDPOINT_HISTORY)
            requested_provenance = source
            carrier_before = before
        after = state_digest(carrier)
        _require(after == before,
                 f"TwoRoom use evaluation mutated the {training_arm} carrier")
        if training_arm == args.arm:
            carrier_after = after
        consumer_sources[training_arm] = {
            **source, "carrier_state_sha256": before,
            "state_unchanged": True,
        }
        del carrier
    _require(requested_deck_x is not None
             and requested_provenance is not None
             and carrier_before is not None and carrier_after is not None,
             "requested carrier was absent from the shared-consumer pool")

    train_x = np.concatenate(pooled_train_x, axis=0)
    pooled_train_labels = np.tile(train_labels, len(ARMS))
    prediction, consumer = _fit_consumer(
        train_x, pooled_train_labels, requested_deck_x, spec["readout"])
    consumer["training_arms"] = list(ARMS)
    consumer["rows_per_arm"] = int(len(train_labels))
    consumer["equal_arm_weighting"] = True
    consumer["shared_across_evaluation_arms"] = True
    consumer["training_sources"] = consumer_sources
    # Held-out location labels are used only after the consumer has emitted its
    # predictions; they never enter scaler or classifier fitting.
    result = _execution_result(
        prediction,
        np.asarray(deck["location_label"], dtype=np.int64),
        np.asarray(deck["combination_label"], dtype=np.int64),
        np.asarray(deck["success_matrix"]),
        np.asarray(deck["distance_matrix"]),
        np.asarray(deck["random_choice"], dtype=np.int64),
    )

    host_weights = spec["inputs"]["tworoom"]["weights"]
    host_weights_path = resolve_input_path(host_weights)
    actual_host_hash = sha256_file(host_weights_path)
    _require(actual_host_hash == host_weights["sha256"],
             "frozen TwoRoom host checkpoint changed after locking")
    provenance.update(requested_provenance)
    provenance["tworoom_host_weights"] = {
        "path": _relative(host_weights_path), "sha256": actual_host_hash,
    }
    metrics = {
        "schema_version": 1,
        "study": spec["study"],
        "branch": "tworoom-external-waypoint-use",
        "lock": spec["_lock"],
        "host": "tworoom",
        "target": "location",
        "arm": args.arm,
        "seed": args.seed,
        "device": str(device),
        "physical_gpu": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cue_age": CUE_AGE,
        "episodes": int(len(prediction)),
        "endpoint": {
            "decision_index": DECISION_INDEX,
            "context_indices": [16, 17, 18],
            "prior_index": 19,
            "current_observation_excluded": True,
            "feature": "concat(z[16],z[17],z[18],prior_read[19])",
        },
        "arm_blind_consumer": True,
        "consumer_training_arms": list(ARMS),
        "consumer_rows": int(len(pooled_train_labels)),
        "consumer_seed": args.seed,
        "consumer_state_sha256": consumer["parameter_sha256"],
        "validation_labels_used_for_fitting": False,
        "frozen_host_instantiated": False,
        "carrier_state_unchanged": True,
        "carrier_state_sha256_before": carrier_before,
        "carrier_state_sha256_after": carrier_after,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "consumer": consumer,
        "provenance": provenance,
        **result,
    }
    destination.mkdir(parents=True, exist_ok=False)
    metrics_path = destination / "metrics.json"
    metrics_hash = atomic_text(metrics_path, stable_json(metrics))
    manifest = {
        "schema_version": 1,
        "study": spec["study"],
        "branch": "tworoom-external-waypoint-use",
        "lock": spec["_lock"],
        "host": "tworoom",
        "arm": args.arm,
        "seed": args.seed,
        "physical_gpu": 0,
        "artifacts": {
            "metrics": {"path": "metrics.json", "sha256": metrics_hash},
        },
    }
    atomic_text(destination / "manifest.json", stable_json(manifest))
    print(
        f"[matched-host/use] {args.arm}/s{args.seed}: "
        f"goal={metrics['goal_selection_accuracy']:.4f}, "
        f"executed={metrics['executed_success_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
