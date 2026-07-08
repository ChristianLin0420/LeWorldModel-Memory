"""Causal endpoint tests for official frozen-carrier publication probes."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import parameter_report
from lewm.models.official_lewm import OFFICIAL_ACTION_DIM, OFFICIAL_EMBED_DIM
from scripts.aggregate_paper_a_expansion import (
    Cell,
    FROZEN_PROBE_SCHEMA,
    FROZEN_TRAJECTORY_SCHEMA,
    carrier_state_digest,
    sha256_file,
    validate_frozen,
    validate_frozen_artifacts,
    validate_no_carrier_determinism,
)
from scripts.train_frozen_official_swap import (
    categorical_features,
    categorical_probe_contract,
    categorical_trajectory_features,
)


def _endpoint_arrays(dim: int = 2) -> tuple[dict, np.ndarray]:
    z = np.zeros((2, 64, dim), dtype=np.float32)
    for time in range(64):
        z[:, time] = time
    prior = np.zeros_like(z)
    for time in range(64):
        prior[:, time] = 1000 + time
    data = {
        "z": z,
        "xi": np.asarray([0, 1], dtype=np.int64),
        "event_cue_off": np.asarray([58, 59], dtype=np.int64),
    }
    return data, prior


def test_primary_endpoint_uses_z60_to_z62_and_preobservation_prior63() -> None:
    data, prior = _endpoint_arrays()
    features = categorical_features(data, prior)
    assert features.shape == (2, 8)
    np.testing.assert_array_equal(
        features[0, :6], np.asarray([60, 60, 61, 61, 62, 62]))
    np.testing.assert_array_equal(features[0, 6:], np.asarray([1063, 1063]))

    contract = categorical_probe_contract(data)
    assert contract["schema"] == FROZEN_PROBE_SCHEMA
    assert contract["decision_observation_index"] == 63
    assert contract["raw_context_indices"] == [60, 61, 62]
    assert contract["raw_context_cutoff_exclusive"] == 63
    assert contract["current_observation_excluded"] is True
    assert contract["future_observation_consumed"] is False
    assert contract["temporal_aggregation"] is False


def test_primary_endpoint_is_invariant_to_final_decision_observation() -> None:
    data, prior = _endpoint_arrays()
    baseline = categorical_features(data, prior)
    changed = copy.deepcopy(data)
    changed["z"][:, 63] = np.asarray([[-1e9, 1e9], [3e8, -7e8]])
    np.testing.assert_array_equal(categorical_features(changed, prior), baseline)

    changed_predecision = copy.deepcopy(data)
    changed_predecision["z"][:, 62] += 7
    assert not np.array_equal(
        categorical_features(changed_predecision, prior), baseline)


def test_postcue_mean_exists_only_in_exploratory_trajectory_probe() -> None:
    data, prior = _endpoint_arrays()
    primary = categorical_features(data, prior)
    trajectory = categorical_trajectory_features(data, prior)
    assert primary.shape[1] == 8  # three raw frames + final prior
    assert trajectory.shape[1] == 10  # plus exploratory prior mean
    # episode 0: cue_off=58 => mean of pre-observation priors t=60..63
    np.testing.assert_array_equal(
        trajectory[0, 6:8], np.asarray([1061.5, 1061.5]))
    np.testing.assert_array_equal(trajectory[0, 8:], primary[0, 6:])


def _valid_frozen_metrics() -> tuple[dict, dict]:
    digest = "a" * 64
    endpoint = {
        "schema": FROZEN_PROBE_SCHEMA,
        "decision_observation_index": 63,
        "raw_context_history": 3,
        "raw_context_indices": [60, 61, 62],
        "raw_context_slice": "z[:, q-H:q]",
        "raw_context_cutoff_exclusive": 63,
        "final_prior_index": 63,
        "final_prior_timing": "prior_read[:, q] before consuming z[:, q]",
        "feature_order": [
            "raw_predecision_context_flat", "final_preobservation_prior"],
        "current_observation_excluded": True,
        "future_observation_consumed": False,
        "temporal_aggregation": False,
    }
    metrics = {
        "schema_version": 1,
        "study": "official-lewm-frozen-carrier-swap",
        "task": "t1",
        "arm": "none",
        "seed": 0,
        "official_host": "quentinll/lewm-reacher",
        "official_host_state_sha256_before": digest,
        "official_host_state_sha256_after": digest,
        "frozen_host_unchanged": True,
        "host_trainable_parameters": 0,
        "carrier_parameters": 0,
        "parameter_matching": parameter_report(
            OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM),
        "epochs": 0,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "val_next_latent_mse": 0.1,
        "probe": {
            "metric": "accuracy",
            "mean": 0.25,
            "chance": 0.25,
            "feature_dim": 768,
            "role": "primary_registered_decision_endpoint",
            "readout": "LogisticRegression(random_state=0)",
            "endpoint_contract": endpoint,
        },
        "trajectory_probe": {
            "metric": "accuracy",
            "mean": 0.25,
            "chance": 0.25,
            "feature_dim": 960,
            "role": "exploratory_secondary_trajectory_probe",
            "readout": "LogisticRegression(random_state=0)",
            "endpoint_contract": {
                "schema": FROZEN_TRAJECTORY_SCHEMA,
                "decision_observation_index": 63,
                "raw_context_indices": [60, 61, 62],
                "aggregation": (
                    "mean prior_read[t] for cue_off+2 <= t <= q"),
                "prior_timing": (
                    "every prior_read[:, t] precedes z[:, t]"),
                "current_observation_excluded": True,
                "future_observation_consumed": False,
                "temporal_aggregation": True,
            },
        },
    }
    config = {
        "official_host": {
            "source": "quentinll/lewm-reacher",
            "latent_dim": 192,
            "context": 3,
        },
        "frozen_carrier_swap": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 3e-4,
        },
    }
    return metrics, config


def test_aggregator_fails_closed_on_old_or_aggregated_probe_contract() -> None:
    metrics, config = _valid_frozen_metrics()
    cell = Cell("frozen_swap", "t1", "none", 0, Path())
    errors: list[str] = []
    validate_frozen(cell, metrics, config, errors)
    assert errors == []

    stale = copy.deepcopy(metrics)
    del stale["probe"]["endpoint_contract"]
    stale["probe"]["feature_dim"] = 960
    stale_errors: list[str] = []
    validate_frozen(cell, stale, config, stale_errors)
    assert any("feature_dim must be 768" in error for error in stale_errors)
    assert any("stale or non-causal" in error for error in stale_errors)

    aggregated = copy.deepcopy(metrics)
    aggregated["probe"]["endpoint_contract"]["temporal_aggregation"] = True
    aggregate_errors: list[str] = []
    validate_frozen(cell, aggregated, config, aggregate_errors)
    assert any("stale or non-causal" in error for error in aggregate_errors)


def _artifact_fixture(tmp_path: Path) -> tuple[Cell, dict, dict, Path]:
    root = tmp_path / "expansion"
    cache_dir = root / "cache" / "t1"
    cache_dir.mkdir(parents=True)
    train_cache = cache_dir / "train.npz"
    val_cache = cache_dir / "val.npz"
    train_cache.write_bytes(b"fixed-train-cache")
    val_cache.write_bytes(b"fixed-validation-cache")

    metrics, config = _valid_frozen_metrics()
    state = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    receipt = {
        "mode": "checkpoint_only_no_retraining",
        "training_performed": False,
        "host_instantiated": False,
        "carrier_state_sha256": carrier_state_digest(state),
        "carrier_state_unchanged": True,
        "train_cache_sha256": sha256_file(train_cache),
        "validation_cache_sha256": sha256_file(val_cache),
        "checkpoint_metrics_synchronized": True,
    }
    metrics["probe"]["reevaluation"] = copy.deepcopy(receipt)
    metrics["trajectory_probe"]["reevaluation"] = copy.deepcopy(receipt)

    cell_dir = root / "frozen_swap" / "t1" / "none" / "s0"
    cell_dir.mkdir(parents=True)
    metrics_path = cell_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    torch.save({"carrier_state_dict": state, "metrics": copy.deepcopy(metrics)},
               cell_dir / "carrier.pt")
    cell = Cell("frozen_swap", "t1", "none", 0, metrics_path)
    return cell, metrics, config, root


def test_aggregator_cross_checks_checkpoint_state_and_cache_receipts(
        tmp_path: Path) -> None:
    cell, metrics, _, root = _artifact_fixture(tmp_path)
    errors: list[str] = []
    validate_frozen_artifacts(cell, metrics, root, errors, {})
    assert errors == []

    checkpoint_path = cell.path.with_name("carrier.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["carrier_state_dict"]["weight"][0, 0] += 1
    torch.save(checkpoint, checkpoint_path)
    state_errors: list[str] = []
    validate_frozen_artifacts(cell, metrics, root, state_errors, {})
    assert any("state digest differs" in error for error in state_errors)

    checkpoint["carrier_state_dict"]["weight"][0, 0] -= 1
    checkpoint["metrics"]["seed"] = 99
    torch.save(checkpoint, checkpoint_path)
    metric_errors: list[str] = []
    validate_frozen_artifacts(cell, metrics, root, metric_errors, {})
    assert any("embedded metrics differ" in error for error in metric_errors)

    checkpoint["metrics"] = copy.deepcopy(metrics)
    torch.save(checkpoint, checkpoint_path)
    (root / "cache" / "t1" / "train.npz").write_bytes(b"tampered")
    cache_errors: list[str] = []
    validate_frozen_artifacts(cell, metrics, root, cache_errors, {})
    assert any("train_cache_sha256 differs" in error for error in cache_errors)


def test_aggregator_requires_exact_no_carrier_repeats() -> None:
    metrics, config = _valid_frozen_metrics()
    metrics["trajectory_probe"] = {"metric": "accuracy", "mean": 0.25}
    config["frozen_carrier_swap"]["tasks"] = ["t1"]
    records = {
        f"frozen_swap/t1/none/seed={seed}": copy.deepcopy(metrics)
        for seed in (0, 1, 2)
    }
    errors: list[str] = []
    validate_no_carrier_determinism(records, config, errors)
    assert errors == []

    records["frozen_swap/t1/none/seed=2"]["probe"]["mean"] = 0.251
    changed_errors: list[str] = []
    validate_no_carrier_determinism(records, config, changed_errors)
    assert any("deterministic metrics vary" in error for error in changed_errors)
