"""Focused contracts for the development-only SAGE-Mem host adapters."""

from __future__ import annotations

import copy
import json
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
    PHASE_A_SCHEMA,
    SAGE_MEM_HOST_ADAPTER_API_VERSION,
    SPATIAL_EVAL_BATCH,
    SPATIAL_TRAIN_MICRO_BATCH,
    _carrier_forward,
    _collect_phase_a_features,
    _cue_offset_reset_index,
    _objective_weights,
    _split_indices,
    _validate_episode_arrays,
    build_host_adapter,
)
import scripts.sage_mem_v1_formal_finalizer as finalizer
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


def test_spatial_chunk_sizes_preserve_registered_effective_batch(spec) -> None:
    effective_batch = int(spec["optimization"]["batch_size"])
    assert SPATIAL_TRAIN_MICRO_BATCH == 32
    assert SPATIAL_EVAL_BATCH == 32
    assert effective_batch % SPATIAL_TRAIN_MICRO_BATCH == 0
    assert SPATIAL_EVAL_BATCH <= effective_batch


@pytest.mark.parametrize(
    "cohort,expected_state_dim",
    [("lewm_reacher_color", 95), ("dinowm_pusht_token", 191)],
)
def test_gdelta_width_is_matched_to_the_current_study_budget(
        cohort, expected_state_dim, spec, model_contract) -> None:
    adapter = build_host_adapter(cohort=cohort, spec=spec)
    carrier, candidate_native = adapter._build_carrier(
        "gdelta", model_contract, torch.device("cpu"))
    target = int(spec["cohorts"][cohort]["target_parameters"])
    margin = float(spec["fairness_reporting"][
        "maximum_parameter_relative_gap"])
    assert candidate_native is False
    assert carrier.state_dim == expected_state_dim
    assert abs(carrier.parameter_count() - target) / target <= margin


def test_reset_index_is_the_actual_cue_offset_for_each_host_design() -> None:
    assert [_cue_offset_reset_index(spatial=False, age=age)
            for age in (4, 8, 15)] == [15, 11, 4]
    assert [_cue_offset_reset_index(spatial=True, age=age)
            for age in (4, 8, 15)] == [4, 4, 4]


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


def test_formal_prepare_returns_public_bank_and_opaque_custody_only(
        spec, model_contract, tmp_path, monkeypatch) -> None:
    adapter = build_host_adapter(cohort="lewm_reacher_color", spec=spec)
    paths = {
        "bank": tmp_path / "bank", "vault": tmp_path / "vault.npz",
        "custody": tmp_path / "custody.json",
    }
    called = {}

    def prepare(**kwargs):
        called.update(kwargs)
        paths["bank"].mkdir()
        paths["vault"].write_bytes(b"opaque")
        paths["custody"].write_text("{}")

    monkeypatch.setattr(
        "scripts.sage_mem_v1_lewm_formal.prepare_lewm_formal_bank", prepare)
    monkeypatch.setattr(adapter, "_formal_device", lambda: torch.device("cpu"))
    counts = spec["cohorts"][adapter.cohort]["split_episodes"]
    bank_handle = {
        "cohort": adapter.cohort, "manifest_sha256": "a" * 64,
        "splits": {
            split: {"count": counts[split],
                    "selection_sha256": str(index + 1) * 64}
            for index, split in enumerate(
                ("formal_train", "consumer_train", "formal_test"))
        },
    }
    custody = {
        "status": "sealed-for-post-grid-finalizer",
        "vault_sha256": "b" * 64,
        "custody_receipt_sha256": "c" * 64,
        "custody_receipt_size": 2,
        "per_cell_api_access": False,
        "path_exposed_to_phase_a": False,
    }
    monkeypatch.setattr(adapter, "_formal_storage_paths", lambda: paths)
    monkeypatch.setattr(
        adapter, "_validated_formal_handles",
        lambda value: (bank_handle, custody))
    monkeypatch.setattr(
        adapter, "_locked_development_selection", lambda: {
            "gdelta_development_healthy": True,
            "locked_comparators": {
                "retention": "ssm", "next_feature": "gru",
                "execution": "ssm"},
        })
    monkeypatch.setattr(
        adapter, "_development_split_handle", lambda: {
            "count": counts["development"],
            "selection_sha256": "d" * 64})
    result = adapter.prepare_fresh_banks(
        split_counts=counts, seed_registry=spec["_seed_registry"],
        forbidden_parent_artifacts=spec["cohorts"][adapter.cohort][
            "forbidden_parent_artifacts"], model_contract=model_contract)
    assert result["formal_labels_hidden"] is True
    assert called["cohort"] == adapter.cohort
    assert called["label_vault_path"] == paths["vault"]
    assert result["label_custody"]["per_cell_api_access"] is False
    assert "path" not in result["label_custody"]
    assert set(result["splits"]) == {
        "development", "formal_train", "consumer_train", "formal_test"}
    assert adapter.describe()["formal_status"] == "pending_fresh_bank_builder"

    with pytest.raises(DevelopmentAdapterError, match="split counts"):
        adapter.prepare_fresh_banks(
            split_counts={}, seed_registry=spec["_seed_registry"],
            forbidden_parent_artifacts=spec["cohorts"][adapter.cohort][
                "forbidden_parent_artifacts"], model_contract=model_contract)
    assert "development parent-TRAIN caches" in FORMAL_PENDING_MESSAGE


def _phase_a_arrays(count: int = 2) -> dict[str, np.ndarray]:
    episode = np.arange(100, 100 + count, dtype=np.int64)
    age = np.repeat(np.asarray((4, 8, 15))[:, None], count, axis=1)
    identity = np.repeat(episode[None], 3, axis=0)
    arrays = {
        "formal_test_episode_id": identity,
        "formal_test_native_cluster_id": identity.copy(),
        "formal_test_evidence_age": age,
        "consumer_train_episode_id": identity.copy(),
        "consumer_train_native_cluster_id": identity.copy(),
        "consumer_train_evidence_age": age.copy(),
        "formal_test_full_mse": np.full((3, count), 0.1, np.float32),
        "formal_test_reset_mse": np.full((3, count), 0.2, np.float32),
        "formal_test_prior_mse": np.full((3, count), 0.3, np.float32),
        "formal_test_full_features": np.zeros((3, count, 192), np.float32),
        "formal_test_reset_features": np.zeros((3, count, 192), np.float32),
        "formal_test_prior_features": np.zeros((3, count, 192), np.float32),
        "consumer_train_full_features": np.zeros(
            (3, count, 192), np.float32),
    }
    return arrays


def test_formal_cell_writes_finalizer_exact_label_free_phase_a(
        spec, model_contract, tmp_path, monkeypatch) -> None:
    adapter = build_host_adapter(cohort="lewm_reacher_color", spec=spec)
    bank_hash = "e" * 64
    prepared = {
        "status": "prepared", "cohort": adapter.cohort,
        "formal_labels_hidden": True,
        "labels_used_for_carrier_training": False,
        "formal_bank": {"cohort": adapter.cohort,
                        "bank_root": str(tmp_path / "bank"),
                        "manifest_sha256": bank_hash},
        "label_custody": {"per_cell_api_access": False,
                          "path_exposed_to_phase_a": False},
    }
    fake_bank = SimpleNamespace(spatial=False, count=2,
                                fit_indices=np.arange(2))
    fake_host = SimpleNamespace(digest=lambda: "a" * 64)
    carrier = make_frozen_carrier("none", 192, 10)
    monkeypatch.setattr(adapter, "_validate_phase_a_prepared", lambda value: None)
    monkeypatch.setattr(adapter, "_formal_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        adapter, "_open_formal_banks",
        lambda value: ({"manifest_sha256": bank_hash}, {
            "formal_train": fake_bank, "consumer_train": fake_bank,
            "formal_test": fake_bank}))
    monkeypatch.setattr(adapter, "_open_formal_host", lambda device: fake_host)
    monkeypatch.setattr(
        adapter, "_build_carrier", lambda *args: (carrier, False))
    monkeypatch.setattr(
        adapter, "_train_carrier",
        lambda *args: ([{"epoch": 1, "loss": 0.125, "lr": 0.0}], True))
    monkeypatch.setattr(
        adapter, "_phase_a_measurements", lambda *args: _phase_a_arrays())
    monkeypatch.setattr(adapter, "_resource_report", lambda *args: {
        "trainable_parameters": 0,
        "forward_flops_per_episode": 0,
        "persistent_state_floats": 0,
        "peak_cuda_bytes": 0,
        "wall_clock_train_seconds": 0.1,
    })
    destination = tmp_path / "cell"
    result = adapter.run_formal_cell(
        arm="none", seed=0, output_directory=destination,
        model_contract=model_contract, prepared=prepared)
    assert result["schema"] == PHASE_A_SCHEMA
    assert result["status"] == "complete-label-free"
    assert result["formal_test_labels_read"] is False
    assert result["formal_test_labels_available"] is False
    assert result["shared_consumer_sha256"] is None
    assert result["physical_gpu"] == 0
    assert result["cuda_visible_devices"] == "0"
    assert set(result["artifacts"]) == {
        "measurements", "checkpoint", "history", "resource_report"}

    (destination / "manifest.json").write_text(
        json.dumps(result, sort_keys=True))
    contract = finalizer._GridContract(
        cohorts=(adapter.cohort,), arms=("none",), seeds=(0,),
        ages=(4, 8, 15), classes={adapter.cohort: 4},
        formal_test_rows={adapter.cohort: 2},
        consumer_train_rows={adapter.cohort: 2},
        variants_per_cluster={adapter.cohort: 1},
        physical_gpus={adapter.cohort: 0})
    validated = finalizer._validate_cell(
        destination, adapter.cohort, "none", 0, contract)
    assert validated.bank_sha256 == bank_hash
    assert validated.representation == "feature_artifact"


def test_phase_a_collector_resets_at_actual_lewm_cue_offset() -> None:
    calls = []

    class Bank:
        spatial = False

        @staticmethod
        def features(age, indices):
            del age
            return np.zeros((len(indices), 20, 4), dtype=np.float32)

        @staticmethod
        def actions(indices):
            return np.zeros((len(indices), 19, 2), dtype=np.float32)

        @staticmethod
        def proprio(indices):
            del indices
            return None

    class Carrier:
        def __call__(self, features, actions):
            del actions
            calls.append(int(features.shape[1]))
            return SimpleNamespace(
                z_tilde=features, prior_read=features,
                telemetry={})

    class Host:
        @staticmethod
        def predict(latent, actions):
            del actions
            return latent

    for age, reset_length in ((4, 5), (8, 9), (15, 16)):
        value = _collect_phase_a_features(
            Host(), Bank(), Carrier(), age, np.array([0]), False,
            torch.device("cpu"))
        assert value["prior_mse"].shape == (1,)
        assert calls[-2:] == [20, reset_length]


def test_phase_a_assembly_never_fits_readout_or_requests_labels(
        spec, monkeypatch) -> None:
    adapter = build_host_adapter(cohort="lewm_reacher_color", spec=spec)
    bank = SimpleNamespace(
        count=3, fit_indices=np.arange(3, dtype=np.int64),
        episode_ids=np.arange(10, 13, dtype=np.int64))

    def collect(host, selected, carrier, age, indices, native, device):
        del host, selected, carrier, native, device
        count = len(indices)
        return {
            "host": np.full((count, 192), age, np.float32),
            "reset": np.zeros((count, 192), np.float32),
            "prior": np.ones((count, 192), np.float32),
            "mse": np.full(count, 0.1, np.float32),
            "reset_mse": np.full(count, 0.2, np.float32),
            "prior_mse": np.full(count, 0.3, np.float32),
        }

    monkeypatch.setattr(
        "scripts.sage_mem_v1_host_adapters._collect_phase_a_features",
        collect)
    monkeypatch.setattr(
        "scripts.sage_mem_v1_host_adapters._fit_shared_readout",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Phase A fitted a semantic readout")))
    arrays = adapter._phase_a_measurements(
        object(), {"consumer_train": bank, "formal_test": bank},
        SimpleNamespace(eval=lambda: None), False, torch.device("cpu"))
    assert set(arrays) == (finalizer._COMMON_ARRAY_KEYS
                           | finalizer._FEATURE_KEYS)
    assert arrays["consumer_train_full_features"].shape == (3, 3, 192)
    assert not hasattr(bank, "labels")


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
