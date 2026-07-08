from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import audit_sage_mem_v1 as audit_module
from scripts import run_sage_mem_v1 as run_module
from scripts.audit_sage_mem_v1 import (
    SageMemAuditError,
    _load_episode_arrays,
    paired_cluster_bootstrap,
)
from scripts.launch_sage_mem_v1 import (
    development_audit_command,
    development_bank_commands,
    formal_audit_command,
    formal_execution_decks_command,
    formal_finalization_command,
    formal_preparation_command,
    formal_raw_context_command,
    planned_commands,
)
from scripts.prepare_sage_mem_v1_development import build_development_manifest
from scripts.run_sage_mem_v1 import (
    FORMAL_CONFIRMATION,
    SageMemRunError,
    _normalize_development_result,
    _prepare,
    preflight_report,
)
from scripts.sage_mem_v1_interface import (
    SageMemInterfaceError,
    load_model_contract,
    validate_forward_output,
    validate_host_adapter_instance,
    validate_model_instance,
)
from scripts.sage_mem_v1_losses import (
    LabelFreeLossError,
    LossWeights,
    compose_label_free_loss,
)
from scripts.sage_mem_v1_spec import (
    AGES,
    ARMS,
    COHORTS,
    DEFAULT_SPEC,
    FORMAL_SEEDS,
    derive_seed,
    formal_cells,
    load_spec,
    seed_registry,
    validate_spec,
)


def _spec() -> dict:
    return load_spec(DEFAULT_SPEC)


def test_preregistered_grid_and_gpu_ownership_are_exact() -> None:
    spec = _spec()
    assert COHORTS == (
        "lewm_reacher_color", "lewm_pusht_color", "dinowm_pusht_token",
        "dinowm_pusht_binding", "dinowm_pointmaze_goal")
    assert len(ARMS) == 12
    assert {"gdelta", "fixed_trust_aux", "ssm_aux",
            "sage_mem_next_only", "sage_mem_no_exposure",
            "sage_mem_exposure_only"}.issubset(ARMS)
    assert AGES == (4, 8, 15) and FORMAL_SEEDS == tuple(range(10))
    assert len(formal_cells(spec)) == 600
    assert [spec["cohorts"][name]["gpu"] for name in COHORTS] == [0, 0, 1, 1, 2]
    assert [spec["cohorts"][name]["target_parameters"] for name in COHORTS] \
        == [76032, 76032, 299520, 299520, 299520]
    assert spec["cohorts"]["dinowm_pusht_token"]["classes"] == 4
    assert spec["cohorts"]["dinowm_pusht_binding"]["classes"] == 6


def test_claim_gates_and_pending_formal_boundary_are_locked() -> None:
    spec = _spec()
    assert spec["confirmatory_gates"]["next_feature_noninferiority"][
        "relative_margin"] == 0.02
    assert spec["confirmatory_gates"]["reset_causality"][
        "reset_to_full_mse_ratio_max"] == 1.25
    assert spec["statistics"]["bootstrap_draws"] == 20_000
    assert spec["confirmatory_gates"]["mechanism_controls"][
        "required_comparisons"][-2:] == [
            "sage_mem_full-fixed_trust_aux", "sage_mem_full-ssm_aux"]
    assert spec["freshness"]["development_source"] == \
        "deterministic subset of parent TRAIN partition only"
    assert spec["freshness"]["formal_preparation_status"] == \
        "pending-executable-fresh-bank-builders"
    changed = copy.deepcopy(spec)
    changed.pop("_spec_path")
    changed.pop("_spec_sha256")
    changed.pop("_seed_registry")
    changed["confirmatory_gates"]["next_feature_noninferiority"][
        "relative_margin"] = 0.05
    with pytest.raises(ValueError):
        validate_spec(changed)


def test_seed_derivation_is_stable_namespaced_and_collision_free() -> None:
    spec = _spec()
    first = seed_registry(spec)
    second = seed_registry(spec)
    assert first == second
    assert len(first) == 5 * 4 * 3
    assert len(set(first.values())) == len(first)
    assert derive_seed(2026070801, "dinowm_pusht_token", "development",
                       "episode_selection") != derive_seed(
                           2026070801, "dinowm_pusht_binding",
                           "development", "episode_selection")


@pytest.mark.parametrize("cohort,expected,eligible", [
    ("lewm_reacher_color", 240, 1200),
    ("lewm_pusht_color", 240, 1200),
    ("dinowm_pusht_token", 320, 1200),
    ("dinowm_pusht_binding", 320, 1200),
    ("dinowm_pointmaze_goal", 120, 300),
])
def test_development_banks_are_deterministic_parent_train_only(
        cohort: str, expected: int, eligible: int) -> None:
    spec = _spec()
    first = build_development_manifest(spec, cohort)
    second = build_development_manifest(spec, cohort)
    assert first == second
    assert first["selection"]["count"] == expected
    assert first["selection"]["eligible_parent_train_rows"] == eligible
    assert len(first["selection"]["rows"]) == expected
    assert first["parent_train_only"] is True
    assert first["parent_validation_or_test_read"] is False
    assert first["semantic_labels_read_for_selection"] is False
    assert first["formal_evidence_permitted"] is False


def test_formal_prepare_is_fail_closed_until_implementation_is_sealed() -> None:
    with pytest.raises(SageMemRunError, match="sealed implementation lock"):
        _prepare(_spec(), "lewm_reacher_color")


def test_adapter_complete_status_normalizes_to_development_receipt() -> None:
    spec = _spec()
    cohort = "lewm_reacher_color"
    bank = build_development_manifest(spec, cohort)
    digest = hashlib.sha256(
        json.dumps(
            bank, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("utf-8")).hexdigest()
    age_record = {
        "host_output_balanced_accuracy": 0.25,
        "prior_balanced_accuracy": 0.25,
        "reset_with_full_readout_balanced_accuracy": 0.25,
        "full_next_feature_mse": 0.1,
        "reset_next_feature_mse": 0.1,
        "reset_to_full_mse_ratio": 1.0,
        "readout_fit_parent_train_rows": 180,
        "readout_eval_parent_train_rows": 60,
    }
    raw = {
        "status": "complete", "cohort": cohort, "arm": "none",
        "seed": 101, "development_only": True,
        "formal_evidence_permitted": False, "parent_train_only": True,
        "labels_used_for_training": False,
        "labels_used_for_posthoc_readout": True,
        "development_manifest_sha256": digest,
        "gradient_finite": True,
        "host_hash_before": "a" * 64, "host_hash_after": "a" * 64,
        "ages": {str(age): dict(age_record) for age in AGES},
        "next_feature_mse": 0.1,
        "resource_report": {
            "trainable_parameters": 0, "forward_flops_per_episode": 0,
            "persistent_state_floats": 0, "peak_cuda_bytes": 0,
            "wall_clock_train_seconds": 0.0,
        },
        "episode_results": {"path": "development_results.npz",
                            "sha256": "b" * 64},
        "checkpoint": {"path": "carrier.pt", "sha256": "c" * 64},
    }
    normalized = _normalize_development_result(
        raw, spec=spec, cohort=cohort, arm="none", seed=101, bank=bank,
        bank_sha256="d" * 64)
    assert normalized["status"] == "complete"
    assert normalized["adapter_status"] == "complete"
    assert normalized["formal_data_read"] is False


def test_actual_sage_model_implements_sealed_interface_and_exact_budgets() -> None:
    spec = _spec()
    contract = load_model_contract(spec["model_interface"])
    for embed_dim, action_dim, target, spatial in (
            (192, 10, 76032, False), (384, 10, 299520, True)):
        model = contract.builder(
            embed_dim=embed_dim, action_dim=action_dim, variant="full",
            config={})
        validate_model_instance(model, contract)
        assert model.parameter_count() == target
        features = (torch.randn(2, 4, 3, embed_dim) if spatial
                    else torch.randn(2, 4, embed_dim))
        actions = torch.randn(2, 3, action_dim)
        output = model.forward_sequence(features, actions)
        validate_forward_output(output, contract)
        assert output["fused"].shape == features.shape
        assert output["prior"].shape == features.shape


def test_label_free_loss_detaches_targets_and_rejects_semantics() -> None:
    predictions = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
    targets = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
    total, metrics = compose_label_free_loss(
        next_prediction=predictions[0], frozen_next_feature=targets[0],
        exposure=predictions[1], frozen_host_output=targets[1],
        replayed_past=predictions[2], frozen_past_feature=targets[2],
        weights=LossWeights(), metadata={"cohort": "safe"})
    total.backward()
    assert all(value.grad is not None for value in predictions)
    assert all(value.grad is None for value in targets)
    assert metrics["labels_used"] == 0.0
    with pytest.raises(LabelFreeLossError, match="semantic metadata"):
        compose_label_free_loss(
            next_prediction=predictions[0], frozen_next_feature=targets[0],
            exposure=predictions[1], frozen_host_output=targets[1],
            replayed_past=predictions[2], frozen_past_feature=targets[2],
            metadata={"goal_id": 3})


def test_preflight_reports_missing_host_integration_without_claiming_ready() -> None:
    report = preflight_report(_spec(), require_integrations=False)
    assert report["status"] == "ready"
    assert report["formal_execution_started"] is False
    assert report["integrations"]["host_adapter"]["path"].endswith(
        "scripts/sage_mem_v1_host_adapters.py")
    assert report["integrations"]["model"]["path"].endswith(
        "lewm/models/sage_mem.py")
    assert len(report["integrations"]["model"]["sha256"]) == 64
    assert len(report["parent_identities"]) == 5
    source_lock = report["preselection_source_lock"]
    assert source_lock["status"] == \
        "frozen-before-complete-development-selection"
    assert len(source_lock["producer_identities"]) >= 50
    assert len(source_lock["source_set_sha256"]) == 64
    assert source_lock["boundary"]["development_audit_present"] is False
    assert source_lock["boundary"][
        "development_or_formal_outcomes_read"] is False


def test_launcher_plans_exact_cells_and_double_confirmation() -> None:
    cells = planned_commands(DEFAULT_SPEC, "full", resume=True)
    assert len(cells) == 600
    assert {gpu for gpu, _ in cells} == {0, 1, 2}
    assert sum(gpu == 0 for gpu, _ in cells) == 240
    assert sum(gpu == 1 for gpu, _ in cells) == 240
    assert sum(gpu == 2 for gpu, _ in cells) == 120
    assert all(command[command.index("--formal-confirmation") + 1]
               == FORMAL_CONFIRMATION for _, command in cells)
    development = planned_commands(DEFAULT_SPEC, "development")
    assert len(development) == 180
    assert sum(gpu == 0 for gpu, _ in development) == 72
    assert sum(gpu == 1 for gpu, _ in development) == 72
    assert sum(gpu == 2 for gpu, _ in development) == 36
    assert all(command[command.index("--stage") + 1] == "development"
               for _, command in development)
    banks = development_bank_commands(DEFAULT_SPEC)
    assert len(banks) == 5
    assert all(any(part.endswith("prepare_sage_mem_v1_development.py")
                   for part in command) for _, command in banks)
    assert "audit_sage_mem_v1.py" in development_audit_command(
        DEFAULT_SPEC)[1][1]
    preparation = formal_preparation_command(DEFAULT_SPEC)[1]
    assert "prepare_sage_mem_v1_formal.py" in preparation[1]
    assert "PREPARE_SAGE_MEM_V1_FORMAL" in preparation
    raw_context = formal_raw_context_command(
        DEFAULT_SPEC, _spec(), resume=True)[1]
    assert "prepare_sage_mem_v1_raw_context_reference.py" in raw_context[1]
    assert "--prepared-root" in raw_context
    assert any("raw_context_phase_a" in part for part in raw_context)
    assert "--resume" in raw_context
    execution_decks = formal_execution_decks_command(
        DEFAULT_SPEC, _spec(), resume=True)[1]
    assert "prepare_sage_mem_v1_execution_decks.py" in execution_decks[1]
    assert "--preparation-root" in execution_decks
    assert "--resume" in execution_decks
    finalization = formal_finalization_command(DEFAULT_SPEC, _spec())[1]
    assert "sage_mem_v1_formal_finalizer.py" in finalization[1]
    assert "--label-registry" in finalization
    assert "--validate-finalized-output" not in finalization
    validation = formal_finalization_command(
        DEFAULT_SPEC, _spec(), validate_existing=True)[1]
    assert "--validate-finalized-output" in validation
    formal_audit = formal_audit_command(DEFAULT_SPEC, resume=True)[1]
    assert formal_audit[formal_audit.index("--stage") + 1] == "formal"
    assert "--resume" in formal_audit


def test_host_adapter_contract_checks_description_and_method_signatures() -> None:
    class Good:
        def describe(self):
            return {
                "api_version": "sage_mem_v1_host_adapter_v1",
                "cohort": "lewm_reacher_color", "family": "lewm",
                "task": "color", "embed_dim": 192, "action_dim": 10,
                "tokens": 1, "classes": 4,
                "development_source": "parent.npz",
                "development_source_policy": (
                    "manifest-selected parent TRAIN only"),
                "semantic_labels_for_training": False,
                "candidate_spatial_path": "native_4d_no_patch_flatten",
                "formal_status": "pending_fresh_bank_builder",
            }

        def smoke(self, *, model_contract):
            pass

        def run_development_cell(
                self, *, arm, seed, output_directory, model_contract,
                development_manifest):
            pass

        def prepare_fresh_banks(
                self, *, split_counts, seed_registry,
                forbidden_parent_artifacts, model_contract):
            pass

        def run_formal_cell(
                self, *, arm, seed, output_directory, model_contract,
                prepared):
            pass

    description = validate_host_adapter_instance(
        Good(), cohort="lewm_reacher_color",
        api_version="sage_mem_v1_host_adapter_v1")
    assert description["embed_dim"] == 192

    bad = Good()
    bad.describe = lambda: {
        **Good().describe(), "semantic_labels_for_training": True}
    with pytest.raises(SageMemInterfaceError, match="boundary"):
        validate_host_adapter_instance(
            bad, cohort="lewm_reacher_color",
            api_version="sage_mem_v1_host_adapter_v1")


def test_development_cell_is_atomic_and_resume_validates(
        tmp_path, monkeypatch) -> None:
    spec = _spec()
    cohort, arm, seed = "lewm_reacher_color", "sage_mem_full", 101
    bank = build_development_manifest(spec, cohort)
    monkeypatch.setattr(run_module, "output_root", lambda unused: tmp_path)
    final = tmp_path / "final-cell"
    monkeypatch.setattr(
        run_module, "development_cell_directory",
        lambda unused_spec, unused_cohort, unused_arm, unused_seed: final)
    bank_path = tmp_path / "development_banks" / cohort / "manifest.json"
    run_module.atomic_json(bank_path, bank)
    bank_sha = run_module.sha256_file(bank_path)

    class Adapter:
        def run_development_cell(self, *, output_directory, **unused):
            artifact = output_directory / "metrics.bin"
            artifact.write_bytes(b"development")
            return {
                "status": "complete", "cohort": cohort, "arm": arm,
                "seed": seed, "labels_used_for_training": False,
                "formal_data_read": False,
                "host_hash_before": "a" * 64,
                "host_hash_after": "a" * 64,
                "development_bank_sha256": bank_sha,
                "gradient_finite": True,
                "resource_report": {
                    "trainable_parameters": 76032,
                    "forward_flops_per_episode": 100,
                    "persistent_state_floats": 193,
                    "peak_cuda_bytes": 100,
                    "wall_clock_train_seconds": 1.0,
                },
                "development_metrics": {
                    "next_feature_mse": 0.1,
                    "retention_balanced_accuracy": 0.5,
                    "execution_success": 0.4,
                },
                "artifacts": [{
                    "path": "metrics.bin",
                    "sha256": run_module.sha256_file(artifact),
                }],
            }

    monkeypatch.setattr(run_module, "_require_ready_receipt",
                        lambda *unused, **unused_kw: {})
    monkeypatch.setattr(run_module, "require_exact_gpu",
                        lambda *unused, **unused_kw: None)
    monkeypatch.setattr(run_module, "_adapter",
                        lambda *unused, **unused_kw: (object(), Adapter()))
    first = run_module._run_development_cell(
        spec, cohort, arm, seed, resume=False)
    assert first["status"] == "complete"
    assert (final / "manifest.json").is_file()
    assert not list(tmp_path.rglob("*.partial-*"))
    second = run_module._run_development_cell(
        spec, cohort, arm, seed, resume=True)
    assert second == first
    with pytest.raises(FileExistsError):
        run_module._run_development_cell(
            spec, cohort, arm, seed, resume=False)


def test_resume_removes_only_dead_runner_owned_partial_directories(
        tmp_path) -> None:
    final = tmp_path / "seed-101"
    stale = tmp_path / ".seed-101.partial-99999999-deadbeef"
    stale.mkdir()
    (stale / "incomplete").write_text("partial")
    unrelated = tmp_path / "user-data"
    unrelated.mkdir()
    run_module._clear_dead_partial_directories(final)
    assert not stale.exists()
    assert unrelated.is_dir()


def test_adapter_native_development_receipt_is_normalized_fail_closed() -> None:
    spec = _spec()
    cohort, arm, seed = "lewm_reacher_color", "sage_mem_full", 101
    bank = build_development_manifest(spec, cohort)
    manifest_sha = run_module.hashlib.sha256(
        run_module.canonical_json(bank).encode("utf-8")).hexdigest()
    age_metrics = {
        str(age): {
            "host_output_balanced_accuracy": .6,
            "prior_balanced_accuracy": .55,
            "reset_with_full_readout_balanced_accuracy": .3,
            "full_next_feature_mse": .1,
            "reset_next_feature_mse": .11,
            "reset_to_full_mse_ratio": 1.1,
            "readout_fit_parent_train_rows": 180,
            "readout_eval_parent_train_rows": 60,
        }
        for age in (4, 8, 15)
    }
    raw = {
        "status": "complete-development", "cohort": cohort, "arm": arm,
        "seed": seed, "development_only": True,
        "formal_evidence_permitted": False, "parent_train_only": True,
        "labels_used_for_training": False,
        "labels_used_for_posthoc_readout": True,
        "development_manifest_sha256": manifest_sha,
        "gradient_finite": True, "host_hash_before": "a" * 64,
        "host_hash_after": "a" * 64, "next_feature_mse": .1,
        "ages": age_metrics,
        "resource_report": {
            "trainable_parameters": 76032,
            "forward_flops_per_episode": 1,
            "persistent_state_floats": 193,
            "peak_cuda_bytes": 1,
            "wall_clock_train_seconds": 1.0,
        },
        "episode_results": {"path": "development_results.npz",
                            "sha256": "b" * 64},
        "checkpoint": {"path": "carrier.pt", "sha256": "c" * 64},
    }
    normalized = run_module._normalize_development_result(
        raw, spec=spec, cohort=cohort, arm=arm, seed=seed, bank=bank,
        bank_sha256="d" * 64)
    assert normalized["status"] == "complete"
    assert normalized["adapter_status"] == "complete-development"
    assert normalized["formal_data_read"] is False
    assert normalized["development_metrics"] == {
        "next_feature_mse": .1, "retention_balanced_accuracy": .55,
        "execution_success": .55}

    broken = copy.deepcopy(raw)
    broken["host_hash_before"] = broken["host_hash_after"] = None
    with pytest.raises(SageMemRunError, match="adapter-native"):
        run_module._normalize_development_result(
            broken, spec=spec, cohort=cohort, arm=arm, seed=seed, bank=bank,
            bank_sha256="d" * 64)


def test_development_audit_locks_healthy_comparators_without_formal_data(
        tmp_path, monkeypatch) -> None:
    spec = _spec()
    monkeypatch.setattr(audit_module, "output_root", lambda unused: tmp_path)
    metrics = {
        "gru": (1.05, .50, .50), "lstm": (1.06, .51, .51),
        "ssm": (1.04, .70, .52), "fixed_trust": (1.03, .55, .53),
        "gdelta": (1.01, .60, .60),
        "fixed_trust_aux": (1.02, .61, .80),
        "ssm_aux": (1.00, .62, .70),
    }
    manifests = {}
    for cohort in COHORTS:
        bank = tmp_path / "development_banks" / cohort / "manifest.json"
        bank.parent.mkdir(parents=True, exist_ok=True)
        bank.write_text(json.dumps({"cohort": cohort}))
        target = spec["cohorts"][cohort]["target_parameters"]
        for arm in ARMS:
            values = metrics.get(arm, (1.20, .40, .40))
            for seed in (101, 102, 103):
                manifests[(cohort, arm, seed)] = {"result": {
                    "gradient_finite": True,
                    "resource_report": {
                        "trainable_parameters": 0 if arm == "none" else target,
                        "forward_flops_per_episode": (
                            1300 if arm == "ssm" else 1000),
                        "persistent_state_floats": 0 if arm == "none" else 192,
                        "peak_cuda_bytes": 100,
                        "wall_clock_train_seconds": 1.0,
                    },
                    "development_metrics": {
                        "next_feature_mse": values[0],
                        "retention_balanced_accuracy": values[1],
                        "execution_success": values[2],
                    },
                }}
    monkeypatch.setattr(audit_module, "_development_grid",
                        lambda unused: manifests)
    global_receipt, selections = audit_module.audit_development(spec)
    assert global_receipt["registered_cells_verified"] == 180
    assert global_receipt["formal_data_read"] is False
    for selection in selections.values():
        assert selection["gdelta_development_healthy"] is True
        assert selection["arm_summary"]["ssm"]["flop_matched"] is False
        assert selection["arm_summary"]["ssm"]["healthy"] is False
        assert selection["locked_comparators"] == {
            "retention": "ssm_aux", "next_feature": "ssm_aux",
            "execution": "ssm_aux"}


def test_paired_cluster_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    rng = np.random.default_rng(7)
    right = rng.normal(size=(5, 24))
    left = right + 0.08
    strata = np.tile(np.arange(6), 4)
    first = paired_cluster_bootstrap(
        left, right, strata, draws=256, seed=9)
    second = paired_cluster_bootstrap(
        left, right, strata, draws=256, seed=9)
    assert first["point"] == pytest.approx(0.08)
    assert first["lower"] > 0.079
    assert np.array_equal(first["samples"], second["samples"])
    assert first["pairing_preserved"] is True


def test_cluster_bootstrap_resamples_intact_multi_age_native_episodes() -> None:
    right = np.zeros((3, 12), dtype=np.float64)
    left = np.full((3, 12), 0.2, dtype=np.float64)
    cluster_ids = np.tile(np.arange(4), 3)
    strata = np.asarray([
        f"{episode % 2}:{age}" for age in (4, 8, 15)
        for episode in range(4)])
    result = paired_cluster_bootstrap(
        left, right, strata, cluster_ids=cluster_ids, draws=128, seed=17)
    assert result["point"] == pytest.approx(0.2)
    assert result["lower"] == pytest.approx(0.2)
    assert result["resampling_unit"] == \
        "formal seed and native episode cluster"


def test_episode_artifact_enforces_class_age_count_and_native_identity(
        tmp_path) -> None:
    ages = (4, 8, 15)
    episode_id = np.tile(np.arange(4, dtype=np.int64), len(ages))
    class_id = np.tile(np.asarray([0, 1, 0, 1], dtype=np.int64), len(ages))
    evidence_age = np.repeat(np.asarray(ages, dtype=np.int64), 4)
    arrays = {
        "episode_id": episode_id, "class_id": class_id,
        "evidence_age": evidence_age,
        "retention_correct": np.ones(12, dtype=np.int8),
        "reset_correct": np.zeros(12, dtype=np.int8),
        "exposure_correct": np.ones(12, dtype=np.int8),
        "next_feature_mse": np.full(12, .1, dtype=np.float32),
        "reset_next_feature_mse": np.full(12, .11, dtype=np.float32),
        "oracle_success": np.ones(12, dtype=np.int8),
        "execution_success": np.ones(12, dtype=np.int8),
    }
    path = tmp_path / "episode.npz"
    np.savez(path, **arrays)
    loaded = _load_episode_arrays(
        path, classes=2, ages=ages, episodes_per_age=4,
        require_shared_episode_ids=True)
    assert loaded["episode_id"].shape == (12,)

    broken = dict(arrays)
    broken["class_id"] = class_id.copy()
    broken["class_id"][4:8] = 0
    np.savez(path, **broken)
    with pytest.raises(SageMemAuditError, match="omits a registered class"):
        _load_episode_arrays(
            path, classes=2, ages=ages, episodes_per_age=4,
            require_shared_episode_ids=True)


def test_formal_grid_rejects_cross_arm_episode_identity_drift(
        tmp_path, monkeypatch) -> None:
    spec = _spec()
    cohort = "lewm_reacher_color"
    cells = ((cohort, "none", 0), (cohort, "gru", 0))
    monkeypatch.setattr(audit_module, "formal_cells", lambda unused: cells)
    monkeypatch.setattr(
        audit_module, "cell_directory",
        lambda unused_spec, unused_cohort, arm, unused_seed: tmp_path / arm)
    monkeypatch.setattr(audit_module, "output_root", lambda unused: tmp_path)
    monkeypatch.setattr(
        audit_module, "validate_cell_directory",
        lambda unused_spec, unused_path, unused_cohort, unused_arm,
        unused_seed: {"result": {"episode_results": {"path": "x.npz"}}})

    def arrays(path, **unused):
        offset = 1 if path.parent.name == "gru" else 0
        ids = np.tile(np.arange(720, dtype=np.int64), 3)
        ids[0] += offset
        return {
            "episode_id": ids,
            "class_id": np.tile(np.arange(720) % 4, 3),
            "evidence_age": np.repeat(np.asarray([4, 8, 15]), 720),
        }

    monkeypatch.setattr(audit_module, "_load_episode_arrays", arrays)
    with pytest.raises(SageMemAuditError, match="cross-arm/seed"):
        audit_module._grid(spec)


def test_formal_exposure_reset_and_controls_use_frozen_host_output(
        monkeypatch) -> None:
    spec = _spec()
    episode_id = np.arange(6, dtype=np.int64)
    class_id = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    evidence_age = np.asarray([4, 4, 8, 8, 15, 15], dtype=np.int64)
    arrays = {}
    for cohort in COHORTS:
        for arm in ARMS:
            for seed in FORMAL_SEEDS:
                full = arm == "sage_mem_full"
                arrays[(cohort, arm, seed)] = {
                    "episode_id": episode_id,
                    "class_id": class_id,
                    "evidence_age": evidence_age,
                    # Deliberately identical priors: using this field would
                    # make every exposure/reset/control contrast zero.
                    "retention_correct": np.zeros(6, dtype=np.int8),
                    "reset_correct": np.zeros(6, dtype=np.int8),
                    "exposure_correct": np.full(
                        6, int(full), dtype=np.int8),
                    "next_feature_mse": np.full(6, .1),
                    "reset_next_feature_mse": np.full(6, .1),
                    "oracle_success": np.ones(6, dtype=np.int8),
                    "execution_success": np.full(
                        6, int(full), dtype=np.int8),
                }

    monkeypatch.setattr(audit_module, "_grid", lambda unused: ({}, arrays))
    monkeypatch.setattr(
        audit_module, "_prepared_comparators",
        lambda unused_spec, unused_cohort: {
            "retention": "gru", "next_feature": "gru", "execution": "gru"})
    monkeypatch.setattr(
        audit_module, "_audit_resource_parity",
        lambda unused_spec, unused_cohort, unused_manifests: {})

    def exact_bootstrap(left, right, unused_strata, **unused_kwargs):
        point = float(np.mean(np.asarray(left) - np.asarray(right)))
        return {
            "point": point, "lower": point, "upper": point,
            "confidence": .95, "draws": 1, "seed": 0,
            "resampling_unit": "test", "pairing_preserved": True,
            "samples": np.asarray([point]),
        }

    monkeypatch.setattr(
        audit_module, "paired_cluster_bootstrap", exact_bootstrap)
    result = audit_module.audit(spec)
    for cohort in COHORTS:
        record = result["cohorts"][cohort]
        assert record["host_output_exposure"]["point"] == pytest.approx(1.0)
        assert record["reset_effect"]["point"] == pytest.approx(1.0)
        assert all(value["point"] == pytest.approx(1.0)
                   for value in record["mechanism_controls"].values())
