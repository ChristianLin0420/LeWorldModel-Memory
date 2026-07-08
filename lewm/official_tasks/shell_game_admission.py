"""Frozen-representation admission probes for shell-game capacity stages.

Admission is deliberately ordered and fail-closed:

1. paired counterfactual construction must pass exactly on train and
   validation banks;
2. frozen cue features must expose every item's *initial* slot;
3. frozen mid-swap features must expose which visible slot pair moved;
4. post-shuffle and final legal features must not expose final item slots.

The initial slot is the availability coordinate because the final slot also
depends on future swap nuisance.  Cue availability plus swap visibility are
necessary representation conditions for the downstream tracking target.
These are CPU-only linear probes; this module neither loads nor trains a world
model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lewm.official_tasks.shell_game_capacity import (
    ShellGameAdmissionInputs,
    ShellGameCapacityBatch,
    audit_paired_counterfactual,
    build_admission_inputs,
    gather_sequence,
)


PROBE_RANDOM_STATE = 0
PROBE_MAX_ITER = 2000


@dataclass(frozen=True)
class ShellGameAdmissionThresholds:
    """Pre-specified representation and leakage gates."""

    cue_initial_slot_accuracy_min: float = 0.75
    swap_pair_accuracy_min: float = 0.75
    leakage_margin_above_chance: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
                ("cue_initial_slot_accuracy_min",
                 self.cue_initial_slot_accuracy_min),
                ("swap_pair_accuracy_min", self.swap_pair_accuracy_min)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0,1]")
        if not 0.0 <= self.leakage_margin_above_chance < 1.0:
            raise ValueError("leakage margin must be in [0,1)")


@dataclass(frozen=True)
class ShellGameFrozenFeatures:
    """Materialized probe features from one frozen latent bank."""

    cue: np.ndarray
    cue_targets: np.ndarray
    swap: np.ndarray
    swap_targets: np.ndarray
    post_shuffle: np.ndarray
    final_context: np.ndarray
    final_targets: np.ndarray

    def __post_init__(self) -> None:
        episodes, capacity = self.cue_targets.shape
        if self.cue.shape[:2] != (episodes, capacity):
            raise ValueError("cue features and targets disagree")
        if self.final_targets.shape != (episodes, capacity):
            raise ValueError("final target shape disagrees with cue targets")
        if self.post_shuffle.shape[0] != episodes \
                or self.final_context.shape[0] != episodes:
            raise ValueError("leakage features require one row per episode")
        if self.swap.shape[0] != episodes \
                or self.swap.shape[:2] != self.swap_targets.shape:
            raise ValueError("swap features and targets disagree")


def frozen_feature_inputs(latents: np.ndarray,
                          inputs: ShellGameAdmissionInputs
                          ) -> ShellGameFrozenFeatures:
    """Gather and flatten the registered coordinates from ``(E,L,D)``."""

    latents = np.asarray(latents)
    if latents.ndim != 3 or latents.shape[0] != inputs.cue_indices.shape[0]:
        raise ValueError(
            f"latents must be (E,L,D) for the supplied inputs, got {latents.shape}")
    if not np.issubdtype(latents.dtype, np.floating) \
            or not np.isfinite(latents).all():
        raise ValueError("latents must be finite floating-point values")
    episodes, capacity = inputs.cue_initial_slot_targets.shape
    cue = gather_sequence(latents, inputs.cue_indices).reshape(
        episodes, capacity, -1)
    swap = gather_sequence(latents, inputs.swap_indices)
    post_shuffle = gather_sequence(
        latents, inputs.post_shuffle_indices).reshape(episodes, -1)
    final_context = gather_sequence(
        latents, inputs.final_context_indices).reshape(episodes, -1)
    return ShellGameFrozenFeatures(
        cue=cue.astype(np.float32, copy=False),
        cue_targets=np.array(inputs.cue_initial_slot_targets, copy=True),
        swap=swap.astype(np.float32, copy=False),
        swap_targets=np.array(inputs.swap_pair_targets, copy=True),
        post_shuffle=post_shuffle.astype(np.float32, copy=False),
        final_context=final_context.astype(np.float32, copy=False),
        final_targets=np.array(inputs.final_slot_targets, copy=True),
    )


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                 validation_x: np.ndarray) -> np.ndarray:
    classes = np.unique(train_y)
    if len(classes) < 2:
        raise ValueError("categorical probe training data has fewer than two classes")
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=PROBE_MAX_ITER,
        random_state=PROBE_RANDOM_STATE,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.transform(train_x), train_y)
    return classifier.predict(scaler.transform(validation_x))


def _per_item_probe(train_x: np.ndarray, train_y: np.ndarray,
                    validation_x: np.ndarray, validation_y: np.ndarray
                    ) -> dict[str, Any]:
    capacity = train_y.shape[1]
    predictions = np.empty_like(validation_y)
    per_item = []
    for item in range(capacity):
        predictions[:, item] = _fit_predict(
            train_x[:, item], train_y[:, item], validation_x[:, item])
        per_item.append(float(np.mean(
            predictions[:, item] == validation_y[:, item])))
    return {
        "per_item_accuracy": per_item,
        "mean_item_accuracy": float(np.mean(per_item)),
        "minimum_item_accuracy": float(np.min(per_item)),
        "maximum_item_accuracy": float(np.max(per_item)),
        "exact_set_accuracy": float(np.mean(np.all(
            predictions == validation_y, axis=1))),
    }


def _shared_feature_per_item_probe(
        train_x: np.ndarray, train_y: np.ndarray,
        validation_x: np.ndarray, validation_y: np.ndarray) -> dict[str, Any]:
    capacity = train_y.shape[1]
    train_expanded = np.broadcast_to(
        train_x[:, None, :], (len(train_x), capacity, train_x.shape[1]))
    validation_expanded = np.broadcast_to(
        validation_x[:, None, :],
        (len(validation_x), capacity, validation_x.shape[1]))
    return _per_item_probe(
        train_expanded, train_y, validation_expanded, validation_y)


def _swap_probe(train: ShellGameFrozenFeatures,
                validation: ShellGameFrozenFeatures) -> float:
    train_x = train.swap.reshape(-1, train.swap.shape[-1])
    validation_x = validation.swap.reshape(-1, validation.swap.shape[-1])
    train_y = train.swap_targets.reshape(-1)
    validation_y = validation.swap_targets.reshape(-1)
    prediction = _fit_predict(train_x, train_y, validation_x)
    return float(np.mean(prediction == validation_y))


def _gate(value: Any, threshold: float | None,
          direction: str, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "direction": direction,
        "pass": bool(passed),
    }


def evaluate_frozen_admission_inputs(
        *,
        train_latents: np.ndarray,
        train_inputs: ShellGameAdmissionInputs,
        train_counterfactual_report: dict[str, Any],
        validation_latents: np.ndarray,
        validation_inputs: ShellGameAdmissionInputs,
        validation_counterfactual_report: dict[str, Any],
        thresholds: ShellGameAdmissionThresholds = ShellGameAdmissionThresholds(),
        ) -> dict[str, Any]:
    """Score prepared frozen coordinates and binding exact-audit receipts."""

    if train_inputs.display_name != validation_inputs.display_name \
            or train_inputs.capacity != validation_inputs.capacity:
        raise ValueError("train and validation admission contracts differ")
    for name, report in (
            ("train", train_counterfactual_report),
            ("validation", validation_counterfactual_report)):
        if report.get("display_name") != train_inputs.display_name \
                or report.get("capacity") != train_inputs.capacity:
            raise ValueError(f"{name} counterfactual receipt has wrong stage")
    train = frozen_feature_inputs(train_latents, train_inputs)
    validation = frozen_feature_inputs(validation_latents, validation_inputs)

    cue = _per_item_probe(
        train.cue, train.cue_targets,
        validation.cue, validation.cue_targets)
    swap_accuracy = _swap_probe(train, validation)
    post_shuffle = _shared_feature_per_item_probe(
        train.post_shuffle, train.final_targets,
        validation.post_shuffle, validation.final_targets)
    final_context = _shared_feature_per_item_probe(
        train.final_context, train.final_targets,
        validation.final_context, validation.final_targets)
    chance = validation_inputs.per_item_chance
    leakage_threshold = chance + thresholds.leakage_margin_above_chance

    gates = {
        "paired_counterfactual_construction": _gate(
            {
                "train": train_counterfactual_report["overall_pass"],
                "validation": validation_counterfactual_report["overall_pass"],
            },
            None,
            "both exact audits pass",
            train_counterfactual_report["overall_pass"]
            and validation_counterfactual_report["overall_pass"],
        ),
        "cue_initial_slot_availability": _gate(
            cue,
            thresholds.cue_initial_slot_accuracy_min,
            ">= for every item",
            cue["minimum_item_accuracy"]
            >= thresholds.cue_initial_slot_accuracy_min,
        ),
        "swap_pair_visibility": _gate(
            swap_accuracy,
            thresholds.swap_pair_accuracy_min,
            ">=",
            swap_accuracy >= thresholds.swap_pair_accuracy_min,
        ),
        "post_shuffle_target_leakage": _gate(
            post_shuffle,
            leakage_threshold,
            "<= for every item",
            post_shuffle["maximum_item_accuracy"] <= leakage_threshold,
        ),
        "final_context_target_leakage": _gate(
            final_context,
            leakage_threshold,
            "<= for every item",
            final_context["maximum_item_accuracy"] <= leakage_threshold,
        ),
    }
    return {
        "schema": "official_shell_game_frozen_admission_v1",
        "display_name": train_inputs.display_name,
        "capacity": train_inputs.capacity,
        "train_episodes": int(train_inputs.cue_indices.shape[0]),
        "validation_episodes": int(validation_inputs.cue_indices.shape[0]),
        "representation_frozen": True,
        "world_model_training_performed": False,
        "availability_target": "initial slot for each cued item",
        "downstream_target": "final slot for each item after visible swaps",
        "per_item_chance": chance,
        "exact_set_chance": validation_inputs.exact_set_chance,
        "thresholds": {
            "cue_initial_slot_accuracy_min":
                thresholds.cue_initial_slot_accuracy_min,
            "swap_pair_accuracy_min": thresholds.swap_pair_accuracy_min,
            "leakage_margin_above_chance":
                thresholds.leakage_margin_above_chance,
        },
        "counterfactual_audits": {
            "train": train_counterfactual_report,
            "validation": validation_counterfactual_report,
        },
        "gates": gates,
        "admitted": all(gate["pass"] for gate in gates.values()),
    }


def evaluate_frozen_admission(
        *,
        train_latents: np.ndarray,
        train_primary: ShellGameCapacityBatch,
        train_counterfactual: ShellGameCapacityBatch,
        validation_latents: np.ndarray,
        validation_primary: ShellGameCapacityBatch,
        validation_counterfactual: ShellGameCapacityBatch,
        thresholds: ShellGameAdmissionThresholds = ShellGameAdmissionThresholds(),
        cue_probe_frames: int = 3,
        post_shuffle_probe_frames: int = 4) -> dict[str, Any]:
    """Evaluate exact construction and frozen features from in-memory pairs."""

    if train_primary.contract != validation_primary.contract:
        raise ValueError("train and validation stages use different contracts")
    return evaluate_frozen_admission_inputs(
        train_latents=train_latents,
        train_inputs=build_admission_inputs(
            train_primary, cue_probe_frames, post_shuffle_probe_frames),
        train_counterfactual_report=audit_paired_counterfactual(
            train_primary, train_counterfactual),
        validation_latents=validation_latents,
        validation_inputs=build_admission_inputs(
            validation_primary, cue_probe_frames, post_shuffle_probe_frames),
        validation_counterfactual_report=audit_paired_counterfactual(
            validation_primary, validation_counterfactual),
        thresholds=thresholds,
    )
