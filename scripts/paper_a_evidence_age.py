#!/usr/bin/env python3
"""Shared causal endpoints and readouts for evidence-age experiments."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def age_name(age: int | str) -> str:
    if age == "final":
        return "final"
    if isinstance(age, bool) or not isinstance(age, int) or age <= 0:
        raise ValueError(f"invalid evidence age {age!r}")
    return f"age-{age}"


def read_indices(cue_off: np.ndarray, age: int | str, *,
                 length: int, history: int = 3) -> np.ndarray:
    off = np.asarray(cue_off, dtype=np.int64)
    if off.ndim != 1:
        raise ValueError("cue_off must be one-dimensional")
    q = (np.full(len(off), length - 1, dtype=np.int64)
         if age == "final" else off + int(age))
    if np.any(q < history) or np.any(q >= length):
        raise ValueError(f"read indices leave sequence: {q.min()}..{q.max()}")
    if np.any(q - history < off):
        raise ValueError("raw endpoint context still contains a cue frame")
    return q


def endpoint_features(z: np.ndarray, prior: np.ndarray,
                      q: np.ndarray, *, history: int = 3) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    q = np.asarray(q, dtype=np.int64)
    if z.ndim != 3 or prior.shape != z.shape or q.shape != (len(z),):
        raise ValueError("endpoint feature arrays have incompatible shapes")
    offsets = np.arange(-history, 0, dtype=np.int64)
    indices = q[:, None] + offsets[None]
    rows = np.arange(len(z), dtype=np.int64)[:, None]
    context = z[rows, indices].reshape(len(z), -1)
    return np.concatenate((context, prior[np.arange(len(z)), q]), axis=1)


def fit_readout(train_x: np.ndarray, train_y: np.ndarray,
                validation_x: np.ndarray, validation_y: np.ndarray,
                protocol: Mapping[str, Any], *, balanced: bool) -> dict[str, Any]:
    train_y = np.asarray(train_y, dtype=np.int64)
    validation_y = np.asarray(validation_y, dtype=np.int64)
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=float(protocol["logistic_c"]), solver=str(protocol["solver"]),
        max_iter=int(protocol["max_iter"]),
        random_state=int(protocol["random_state"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(train_x), train_y)
    prediction = model.predict(scaler.transform(validation_x)).astype(np.int64)
    accuracy = float(np.mean(prediction == validation_y))
    balanced_accuracy = float(balanced_accuracy_score(validation_y, prediction))
    return {
        "metric": "balanced_accuracy" if balanced else "accuracy",
        "value": balanced_accuracy if balanced else accuracy,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "classes": int(len(np.unique(train_y))),
        "fit_episodes": int(len(train_y)),
        "validation_episodes": int(len(validation_y)),
        "feature_dimension": int(train_x.shape[1]),
        "prediction": prediction.tolist(),
        "correct": (prediction == validation_y).astype(np.int8).tolist(),
        "iterations": [int(value) for value in model.n_iter_],
    }


def fixed_endpoint_features(z: np.ndarray, prior: np.ndarray,
                            decision: int, history: int = 3) -> np.ndarray:
    q = np.full(len(z), decision, dtype=np.int64)
    return endpoint_features(z, prior, q, history=history)


def combine_age_mixture(values: Mapping[int, np.ndarray],
                        ages: Sequence[int]) -> np.ndarray:
    arrays = [np.asarray(values[int(age)]) for age in ages]
    if not arrays or len({array.shape[1:] for array in arrays}) != 1 \
            or len({len(array) for array in arrays}) != 1:
        raise ValueError("age mixture arrays are not paired and shape-matched")
    return np.concatenate(arrays, axis=0)


__all__ = [
    "age_name", "combine_age_mixture", "configure_determinism",
    "endpoint_features", "fit_readout", "fixed_endpoint_features",
    "read_indices",
]
