"""Focused contracts for the development-only SAGE-Mem host adapters."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.models.frozen_swap_carriers import make_frozen_carrier
from lewm.models.sage_mem import SAGEMem
from scripts.prepare_sage_mem_v1_development import build_development_manifest
from scripts.sage_mem_v1_host_adapters import (
    DevelopmentAdapterError,
    FORMAL_PENDING_MESSAGE,
    FormalIntegrationPending,
    SAGE_MEM_HOST_ADAPTER_API_VERSION,
    _carrier_forward,
    _objective_weights,
    _split_indices,
    _validate_episode_arrays,
    build_host_adapter,
)
from scripts.sage_mem_v1_interface import (
    HOST_API_VERSION,
    load_host_adapter_contract,
    load_model_contract,
)
from scripts.sage_mem_v1_spec import COHORTS, DEFAULT_SPEC, load_spec


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec(DEFAULT_SPEC)


@pytest.fixture(scope="module")
def model_contract(spec):
    return load_model_contract(spec["model_interface"])


def _valid_episode_arrays() -> dict[str, np.ndarray]:
    return {
        "episode_id": np.array([104, 108, 115], dtype=np.int64),
        "class_id": np.array([0, 1, 2], dtype=np.int64),
        "evidence_age": np.array([4, 8, 15], dtype=np.int64),
        "retention_correct": np.array([1, 0, 1], dtype=np.int8),
        "reset_correct": np.array([0, 0, 1], dtype=np.int8),
        "exposure_correct": np.array([1, 1, 1], dtype=np.int8),
        "next_feature_mse": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "reset_next_feature_mse": np.array(
            [0.11, 0.22, 0.33], dtype=np.float32),
        "oracle_success": np.zeros(3, dtype=np.int8),
        "execution_success": np.zeros(3, dtype=np.int8),
    }


def test_host_contract_loads_and_all_five_adapters_describe_exactly(spec) -> None:
    contract = load_host_adapter_contract(spec["host_adapter_interface"])
    assert SAGE_MEM_HOST_ADAPTER_API_VERSION == HOST_API_VERSION
    assert contract.builder is build_host_adapter
    for cohort in COHORTS:
        adapter = contract.builder(cohort=cohort, spec=spec)
        description = adapter.describe()
        assert description["api_version"] == HOST_API_VERSION
        assert description["cohort"] == cohort
        assert description["semantic_labels_for_training"] is False
        assert description["formal_status"] == "pending_fresh_bank_builder"


@pytest.mark.parametrize(
    "cohort", ["lewm_reacher_color", "dinowm_pusht_token"])
def test_cpu_smoke_is_label_free_gradient_finite_and_reset_isolated(
        cohort, spec, model_contract, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = build_host_adapter(cohort=cohort, spec=spec).smoke(
        model_contract=model_contract)
    assert result["status"] == "passed"
    assert result["device"] == "cpu"
    assert result["labels_used"] is False
    assert result["gradient_finite"] is True
    assert result["reset_isolates_state"] is True
    assert result["zero_semantic_readouts_fitted"] is True


def test_candidate_uses_native_4d_path_and_baseline_uses_registered_wrapper(
        monkeypatch) -> None:
    features = torch.randn(1, 4, 196, 16)
    actions = torch.randn(1, 3, 3)
    sage = SAGEMem(16, 3)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("candidate was flattened into patch batches")

    monkeypatch.setattr(
        "scripts.sage_mem_v1_host_adapters.spatial_carrier_forward",
        forbidden)
    output = _carrier_forward(
        sage, features, actions, candidate_native=True)
    assert output["fused"].shape == features.shape

    baseline = make_frozen_carrier("ssm", 16, 3)
    with pytest.raises(AssertionError, match="flattened"):
        _carrier_forward(
            baseline, features, actions, candidate_native=False)


def test_actual_lewm_development_manifest_and_parent_train_loader(
        spec) -> None:
    cohort = "lewm_reacher_color"
    manifest = build_development_manifest(spec, cohort)
    adapter = build_host_adapter(cohort=cohort, spec=spec)
    rows = adapter._validate_development_manifest(manifest)
    bank = adapter._open_bank(rows)
    assert bank.count == 240
    assert len(bank.fit_indices) == 180
    assert len(bank.readout_indices) == 60
    assert bank.features(4, bank.fit_indices[:2]).shape == (2, 20, 192)
    assert set(np.unique(bank.labels)) == {0, 1, 2, 3}

    broken = copy.deepcopy(manifest)
    broken["selection"]["rows"][1] = broken["selection"]["rows"][0]
    with pytest.raises(DevelopmentAdapterError, match="rows"):
        adapter._validate_development_manifest(broken)


def test_formal_methods_remain_explicitly_pending(spec, model_contract,
                                                  tmp_path) -> None:
    adapter = build_host_adapter(cohort="lewm_reacher_color", spec=spec)
    with pytest.raises(FormalIntegrationPending,
                       match="parent-disjoint fresh-bank"):
        adapter.prepare_fresh_banks(
            split_counts={}, seed_registry={}, forbidden_parent_artifacts=[],
            model_contract=model_contract)
    with pytest.raises(FormalIntegrationPending,
                       match="parent-disjoint fresh-bank"):
        adapter.run_formal_cell(
            arm="none", seed=0, output_directory=tmp_path,
            model_contract=model_contract, prepared={})
    assert "development parent-TRAIN caches" in FORMAL_PENDING_MESSAGE


def test_objective_controls_are_explicit_and_label_free() -> None:
    assert _objective_weights("sage_mem_next_only") == {
        "next": 1.0, "replay": 0.0, "exposure": 0.0, "reset": 0.0}
    assert _objective_weights("sage_mem_full") == {
        "next": 1.0, "replay": 0.1, "exposure": 0.1, "reset": 0.1}
    assert _objective_weights("fixed_trust_aux")["exposure"] == 0.1
    assert _objective_weights("ssm_aux")["replay"] == 0.1
    assert _objective_weights("sage_mem_no_exposure")["exposure"] == 0.0
    assert _objective_weights("sage_mem_exposure_only")["replay"] == 0.0


def test_label_blind_split_preserves_counterfactual_groups() -> None:
    first = _split_indices(120, 17, groups=4)
    second = _split_indices(120, 17, groups=4)
    assert all(np.array_equal(left, right)
               for left, right in zip(first, second))
    fit, heldout = first
    assert not set(fit // 4).intersection(set(heldout // 4))
    assert len(fit) == 360 and len(heldout) == 120


def test_episode_artifact_validation_rejects_unsafe_values() -> None:
    arrays = _valid_episode_arrays()
    _validate_episode_arrays(arrays, classes=4)
    duplicate = {key: value.copy() for key, value in arrays.items()}
    duplicate["episode_id"][1] = duplicate["episode_id"][0]
    with pytest.raises(DevelopmentAdapterError, match="not unique"):
        _validate_episode_arrays(duplicate, classes=4)
    nonfinite = {key: value.copy() for key, value in arrays.items()}
    nonfinite["next_feature_mse"][0] = np.nan
    with pytest.raises(DevelopmentAdapterError, match="mse"):
        _validate_episode_arrays(nonfinite, classes=4)
    bad_class = {key: value.copy() for key, value in arrays.items()}
    bad_class["class_id"][0] = 4
    with pytest.raises(DevelopmentAdapterError, match="class"):
        _validate_episode_arrays(bad_class, classes=4)


def test_development_cell_writes_atomic_payload_without_formal_claim(
        spec, model_contract, tmp_path, monkeypatch) -> None:
    cohort = "lewm_reacher_color"
    adapter = build_host_adapter(cohort=cohort, spec=spec)
    manifest = build_development_manifest(spec, cohort)
    fake_bank = SimpleNamespace(spatial=False)
    fake_host = SimpleNamespace(digest=lambda: "a" * 64)
    arrays = _valid_episode_arrays()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    monkeypatch.setattr(adapter, "_open_bank", lambda rows: fake_bank)
    monkeypatch.setattr(
        adapter, "_open_host", lambda bank, device: fake_host)
    monkeypatch.setattr(
        adapter, "_train_carrier",
        lambda *args, **kwargs: ([{"epoch": 1, "loss": 0.1}], True))
    monkeypatch.setattr(
        adapter, "_evaluate",
        lambda *args, **kwargs: ({"4": {}, "8": {}, "15": {}},
                                 arrays, 0.2))
    destination = tmp_path / "cell"
    result = adapter.run_development_cell(
        arm="sage_mem_full", seed=101,
        development_manifest=manifest, output_directory=destination,
        model_contract=model_contract)
    assert result["status"] == "complete"
    assert result["stage"] == "development-cell"
    assert result["formal_evidence_permitted"] is False
    assert result["labels_used_for_training"] is False
    assert result["host_hash_before"] == result["host_hash_after"] == "a" * 64
    assert (destination / result["episode_results"]["path"]).is_file()
    assert (destination / result["checkpoint"]["path"]).is_file()
    assert len(result["episode_results"]["sha256"]) == 64


def test_one_epoch_candidate_training_is_executable_without_labels(
        spec, model_contract) -> None:
    local_spec = copy.deepcopy(spec)
    local_spec["optimization"].update({"epochs": 1, "batch_size": 4})
    adapter = build_host_adapter(
        cohort="lewm_reacher_color", spec=local_spec)
    rng = np.random.default_rng(71)
    features = {
        age: rng.normal(size=(8, 20, 192)).astype(np.float32)
        for age in (4, 8, 15)
    }
    actions = rng.normal(size=(8, 19, 10)).astype(np.float32)

    class Bank:
        spatial = False
        fit_indices = np.arange(8, dtype=np.int64)

        def features(self, age, indices):
            return features[int(age)][indices]

        def actions(self, indices):
            return actions[indices]

        def proprio(self, indices):
            del indices
            return None

    class Host:
        @staticmethod
        def predict(latent, action):
            # Frozen differentiable stand-in with the official output shape.
            return latent + action.mean(dim=-1, keepdim=True) * 0.01

    carrier, native = adapter._build_carrier(
        "sage_mem_full", model_contract, torch.device("cpu"))
    history, gradients_finite = adapter._train_carrier(
        Host(), Bank(), carrier, "sage_mem_full", 101, native,
        torch.device("cpu"))
    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])
    assert gradients_finite is True


def test_builder_rejects_unknown_cohort_and_wrong_study(spec) -> None:
    with pytest.raises(DevelopmentAdapterError, match="unknown cohort"):
        build_host_adapter(cohort="unknown", spec=spec)
    broken = dict(spec)
    broken["study"] = "not-sage"
    with pytest.raises(DevelopmentAdapterError, match="requires"):
        build_host_adapter(cohort=COHORTS[0], spec=broken)
