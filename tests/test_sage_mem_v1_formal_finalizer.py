"""Contract tests for the two-phase SAGE-Mem v1 formal finalizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from scripts import prepare_sage_mem_v1_raw_context_reference as raw_producer
from scripts import sage_mem_v1_formal_finalizer as finalizer


def test_production_contract_is_exactly_600_cells() -> None:
    assert finalizer.PRODUCTION_CONTRACT.total_cells == 600
    assert finalizer.PRODUCTION_CONTRACT.require_600 is True


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handle(path: Path) -> dict[str, object]:
    return {"path": path.name, "sha256": _sha(path),
            "size": path.stat().st_size}


def _contract(*, cohort: str = "lewm_reacher_color",
              arms: tuple[str, ...] = ("none", "sage_mem_full"),
              seeds: tuple[int, ...] = (0,), classes: int = 4,
              formal_rows: int = 8, consumer_rows: int = 8,
              variants: int = 1) -> finalizer._GridContract:
    value = finalizer._GridContract(
        cohorts=(cohort,), arms=arms, seeds=seeds, ages=finalizer.AGES,
        classes={cohort: classes}, formal_test_rows={cohort: formal_rows},
        consumer_train_rows={cohort: consumer_rows},
        variants_per_cluster={cohort: variants}, physical_gpus={cohort: 0},
        require_600=False)
    value.validate()
    return value


def _identity(count: int, variants: int, offset: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode = np.arange(offset, offset + count, dtype=np.int64)
    cluster = (np.arange(count, dtype=np.int64) // variants) + offset * 10
    return (
        np.repeat(episode[None, :], len(finalizer.AGES), axis=0),
        np.repeat(cluster[None, :], len(finalizer.AGES), axis=0),
        np.stack([np.full(count, age, dtype=np.int64)
                  for age in finalizer.AGES]),
    )


def _arrays(contract: finalizer._GridContract, cohort: str,
            *, representation: str, arm_offset: float = 0.0
            ) -> dict[str, np.ndarray]:
    variants = contract.variants_per_cluster[cohort]
    formal = _identity(contract.formal_test_rows[cohort], variants, 100)
    consumer = _identity(contract.consumer_train_rows[cohort], variants, 1000)
    shape = (len(contract.ages), contract.formal_test_rows[cohort])
    value: dict[str, np.ndarray] = {
        "formal_test_episode_id": formal[0],
        "formal_test_native_cluster_id": formal[1],
        "formal_test_evidence_age": formal[2],
        "consumer_train_episode_id": consumer[0],
        "consumer_train_native_cluster_id": consumer[1],
        "consumer_train_evidence_age": consumer[2],
        "formal_test_full_mse": np.full(shape, 0.10 + arm_offset),
        "formal_test_reset_mse": np.full(shape, 0.12 + arm_offset),
        "formal_test_prior_mse": np.full(shape, 0.11 + arm_offset),
    }
    classes = contract.classes[cohort]
    if representation == "predicted_labels":
        prediction = np.repeat(
            (np.arange(shape[1]) % classes)[None, :], shape[0], axis=0)
        value.update({
            "formal_test_full_pred": prediction,
            "formal_test_reset_pred": np.roll(prediction, 1, axis=1),
            "formal_test_prior_pred": prediction.copy(),
        })
    else:
        dimension = classes
        formal_class = np.arange(shape[1]) % classes
        consumer_class = np.arange(
            contract.consumer_train_rows[cohort]) % classes
        formal_features = np.eye(classes, dtype=np.float64)[formal_class]
        consumer_features = np.eye(classes, dtype=np.float64)[consumer_class]
        # Arm-specific offsets ensure fitting really pools differing features;
        # no arm identity is included as a separate covariate.
        formal_features = formal_features + arm_offset
        consumer_features = consumer_features + arm_offset
        value.update({
            "formal_test_full_features": np.repeat(
                formal_features[None, :, :], len(contract.ages), axis=0),
            "formal_test_reset_features": np.repeat(
                formal_features[None, :, :], len(contract.ages), axis=0),
            "formal_test_prior_features": np.repeat(
                formal_features[None, :, :], len(contract.ages), axis=0),
            "consumer_train_full_features": np.repeat(
                consumer_features[None, :, :], len(contract.ages), axis=0),
        })
    return value


def _write_cell(root: Path, contract: finalizer._GridContract,
                cohort: str, arm: str, seed: int, *,
                representation: str = "predicted_labels",
                mutate: Callable[[dict[str, np.ndarray]], None] | None = None,
                bank_hash: str = "b" * 64,
                consumer_hash: str = "c" * 64) -> None:
    directory = root / "cells" / cohort / arm / f"seed-{seed}"
    directory.mkdir(parents=True)
    arrays = _arrays(
        contract, cohort, representation=representation,
        arm_offset=0.01 * contract.arms.index(arm))
    if mutate is not None:
        mutate(arrays)
    measurement = directory / "measurements.npz"
    np.savez_compressed(measurement, **arrays)
    checkpoint = directory / "checkpoint.bin"
    checkpoint.write_bytes(f"{cohort}/{arm}/{seed}".encode())
    history = directory / "history.json"
    _json(history, {
        "schema": finalizer.HISTORY_SCHEMA,
        "study": "sage-mem-v1",
        "status": "complete",
        "formal_test_labels_read": False,
        "development_outcomes_read": False,
        "bank_manifest_sha256": bank_hash,
        "epochs": ([] if arm == "none" else [
            {"epoch": 0, "train_label_free_loss": 0.5}]),
    })
    resources = directory / "resources.json"
    _json(resources, {
        "schema": finalizer.RESOURCE_SCHEMA,
        "study": "sage-mem-v1",
        "status": "complete",
        "metrics": {key: float(index + 1)
                    for index, key in enumerate(finalizer.RESOURCE_FIELDS)},
    })
    manifest = {
        "schema": finalizer.PHASE_A_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "formal-phase-a",
        "status": "complete-label-free",
        "cohort": cohort,
        "arm": arm,
        "seed": seed,
        "physical_gpu": contract.physical_gpus[cohort],
        "cuda_visible_devices": str(contract.physical_gpus[cohort]),
        "protocol_fingerprint": (
            contract.protocol_fingerprint or "e" * 64),
        "completed_unix_ns": 1_700_000_000_000_000_000 + seed,
        "ages": list(contract.ages),
        "formal_test_labels_read": False,
        "formal_test_labels_available": False,
        "development_outcomes_read": False,
        "labels_used_for_training": False,
        "bank_manifest_sha256": bank_hash,
        "host_hash_before": "a" * 64,
        "host_hash_after": "a" * 64,
        "prediction_representation": representation,
        "consumer_contract": (
            "precomputed-shared-arm-blind"
            if representation == "predicted_labels" else
            "centralized-pooled-consumer-train-features"),
        "shared_consumer_sha256": (
            consumer_hash if representation == "predicted_labels" else None),
        "artifacts": {
            "measurements": _handle(measurement),
            "checkpoint": _handle(checkpoint),
            "history": _handle(history),
            "resource_report": _handle(resources),
        },
    }
    _json(directory / "manifest.json", manifest)


def _write_grid(root: Path, contract: finalizer._GridContract, *,
                representation: str = "predicted_labels",
                mutation: tuple[str, str, int,
                                Callable[[dict[str, np.ndarray]], None]]
                | None = None) -> None:
    for cohort in contract.cohorts:
        for arm in contract.arms:
            for seed in contract.seeds:
                mutate = (mutation[3] if mutation is not None
                          and mutation[:3] == (cohort, arm, seed) else None)
                _write_cell(root, contract, cohort, arm, seed,
                            representation=representation, mutate=mutate)


def _labels(contract: finalizer._GridContract, cohort: str
            ) -> dict[str, np.ndarray]:
    variants = contract.variants_per_cluster[cohort]
    formal = _identity(contract.formal_test_rows[cohort], variants, 100)
    consumer = _identity(contract.consumer_train_rows[cohort], variants, 1000)
    classes = contract.classes[cohort]
    if variants == classes:
        formal_label = np.tile(np.arange(classes), formal[0].shape[1] // classes)
        consumer_label = np.tile(
            np.arange(classes), consumer[0].shape[1] // classes)
    else:
        formal_label = np.arange(formal[0].shape[1]) % classes
        consumer_label = np.arange(consumer[0].shape[1]) % classes
    return {
        "formal_test_episode_id": formal[0][0],
        "formal_test_native_cluster_id": formal[1][0],
        "formal_test_label": formal_label.astype(np.int64),
        "consumer_train_episode_id": consumer[0][0],
        "consumer_train_native_cluster_id": consumer[1][0],
        "consumer_train_label": consumer_label.astype(np.int64),
    }


def _write_registry(root: Path, contract: finalizer._GridContract,
                    bank_hash: str = "b" * 64) -> Path:
    cohorts: dict[str, object] = {}
    for cohort in contract.cohorts:
        artifact = root / f"{cohort}_labels.npz"
        np.savez_compressed(artifact, **_labels(contract, cohort))
        cohorts[cohort] = {
            "bank_manifest_sha256": bank_hash,
            "classes": contract.classes[cohort],
            "artifact": _handle(artifact),
        }
    manifest = root / "sealed_registry.json"
    _json(manifest, {
        "schema": finalizer.LABEL_REGISTRY_SCHEMA,
        "study": "sage-mem-v1",
        "status": "sealed",
        "labels_available_only_after_complete_phase_a_grid": True,
        "development_outcomes_read": False,
        "cohorts": cohorts,
    })
    return manifest


def _write_custody_registry(root: Path,
                            contract: finalizer._GridContract,
                            bank_hash: str = "b" * 64) -> Path:
    cohorts: dict[str, object] = {}
    for cohort in contract.cohorts:
        split_arrays = _labels(contract, cohort)
        vault = root / f"{cohort}_custody.npz"
        np.savez_compressed(vault,
            episode_id=np.concatenate([
                split_arrays["formal_test_episode_id"],
                split_arrays["consumer_train_episode_id"]]),
            native_cluster_id=np.concatenate([
                split_arrays["formal_test_native_cluster_id"],
                split_arrays["consumer_train_native_cluster_id"]]),
            class_id=np.concatenate([
                split_arrays["formal_test_label"],
                split_arrays["consumer_train_label"]]))
        source = {
            "artifact": _handle(vault),
            "keys": {"episode_id": "episode_id",
                     "native_cluster_id": "native_cluster_id",
                     "label": "class_id"},
        }
        cohorts[cohort] = {
            "bank_manifest_sha256": bank_hash,
            "classes": contract.classes[cohort],
            "sources": {"formal_test": source, "consumer_train": source},
        }
    manifest = root / "custody_registry.json"
    _json(manifest, {
        "schema": finalizer.CUSTODY_REGISTRY_SCHEMA,
        "study": "sage-mem-v1",
        "status": "sealed",
        "labels_available_only_after_complete_phase_a_grid": True,
        "development_outcomes_read": False,
        "cohorts": cohorts,
    })
    return manifest


def _write_raw_context(root: Path,
                       contract: finalizer._GridContract) -> Path:
    records: list[dict[str, object]] = []
    for cohort in contract.cohorts:
        arrays = _arrays(contract, cohort, representation="predicted_labels")
        classes = contract.classes[cohort]
        test_class = np.arange(
            contract.formal_test_rows[cohort]) % classes
        consumer_class = np.arange(
            contract.consumer_train_rows[cohort]) % classes
        test_features = np.repeat(
            np.eye(classes, dtype=np.float32)[test_class][None],
            len(contract.ages), axis=0)
        consumer_features = np.repeat(
            np.eye(classes, dtype=np.float32)[consumer_class][None],
            len(contract.ages), axis=0)
        for seed in contract.seeds:
            directory = root / cohort / f"seed-{seed}"
            directory.mkdir(parents=True)
            artifact = directory / "raw_context.npz"
            np.savez_compressed(artifact, **{
                "formal_test_episode_id": arrays["formal_test_episode_id"],
                "formal_test_native_cluster_id":
                    arrays["formal_test_native_cluster_id"],
                "formal_test_evidence_age":
                    arrays["formal_test_evidence_age"],
                "consumer_train_episode_id":
                    arrays["consumer_train_episode_id"],
                "consumer_train_native_cluster_id":
                    arrays["consumer_train_native_cluster_id"],
                "consumer_train_evidence_age":
                    arrays["consumer_train_evidence_age"],
                "formal_test_short_features": test_features,
                "formal_test_long_features": test_features,
                "consumer_train_short_features": consumer_features,
                "consumer_train_long_features": consumer_features,
            })
            manifest_path = directory / "manifest.json"
            _json(manifest_path, {
                "schema": finalizer.RAW_CONTEXT_SCHEMA,
                "study": "sage-mem-v1",
                "stage": "formal-raw-context-reference",
                "status": "complete-label-free",
                "cohort": cohort,
                "seed": seed,
                "ages": list(contract.ages),
                "short_context_frames": 3,
                "long_context_frames": 16,
                "separate_from_parameter_matched_arms": True,
                "formal_test_labels_read": False,
                "development_outcomes_read": False,
                "bank_manifest_sha256": "b" * 64,
                "host_hash_before": "a" * 64,
                "host_hash_after": "a" * 64,
                "consumer_contract":
                    "post-reveal-shared-short-long-arm-blind",
                "shared_consumer_sha256": None,
                "feature_contract": dict(
                    finalizer.RAW_CONTEXT_FEATURE_CONTRACT),
                "artifact": _handle(artifact),
            })
            records.append({
                "cohort": cohort,
                "seed": seed,
                "manifest_sha256": _sha(manifest_path),
                "artifact_sha256": _sha(artifact),
                "bank_manifest_sha256": "b" * 64,
            })
    _json(root / "summary.json", {
        "schema": "sage_mem_v1_raw_context_producer_v1",
        "study": "sage-mem-v1",
        "status": "complete-label-free",
        "cells": len(records),
        "cohorts": list(contract.cohorts),
        "seeds": list(contract.seeds),
        "feature_contract": dict(finalizer.RAW_CONTEXT_FEATURE_CONTRACT),
        "formal_labels_read": False,
        "development_outcomes_read": False,
        "mse_emitted": False,
        "records_sha256": finalizer._sha256_json(records),
    })
    return root


def _write_execution_deck(
        root: Path, contract: finalizer._GridContract, *,
        oracle_fraction: float = 1.0, threshold: float = 0.9) -> Path:
    cohorts: dict[str, object] = {}
    for cohort in contract.cohorts:
        count = contract.formal_test_rows[cohort]
        classes = contract.classes[cohort]
        identity = _identity(
            count, contract.variants_per_cluster[cohort], 100)
        cube = np.zeros((count, classes, classes), dtype=np.uint8)
        eligible_rows = int(round(count * oracle_fraction))
        diagonal = np.arange(classes, dtype=np.int64)
        cube[:eligible_rows, diagonal, diagonal] = 1
        random_class = np.zeros(count, dtype=np.int64)
        artifact = root / f"{cohort}_execution.npz"
        np.savez_compressed(
            artifact,
            formal_test_episode_id=identity[0][0],
            formal_test_native_cluster_id=identity[1][0],
            selected_class_by_true_target_success=cube,
            deterministic_random_class=random_class,
        )
        replay = root / f"{cohort}_execution_replay.json"
        _json(replay, {
            "schema": finalizer.EXECUTION_REPLAY_RECEIPT_SCHEMA,
            "study": "sage-mem-v1",
            "status": "sealed-label-free",
            "cohort": cohort,
            "bank_manifest_sha256": "b" * 64,
            "formal_labels_read": False,
            "development_outcomes_read": False,
            "controller_identity_sha256": "1" * 64,
            "rows": count,
            "classes": classes,
            "native_clusters": count // contract.variants_per_cluster[cohort],
            "executions": count * classes,
            "replayed_executions": count * classes,
            "deterministic_replay_fidelity": 1.0,
            "execution_endpoint": "fixed-test-endpoint",
        })
        cohorts[cohort] = {
            "bank_manifest_sha256": "b" * 64,
            "classes": classes,
            "controller": {
                "controller_identity_sha256": "1" * 64,
                "implementation_sha256": "2" * 64,
                "physics_sha256": "3" * 64,
                "pinned": True,
                "arm_identity_input": False,
                "input": "predicted_class_only",
            },
            "eligibility_gate": {
                "metric": "mean_oracle_success",
                "operator": ">=",
                "threshold": threshold,
                "preregistered": True,
            },
            "artifact": _handle(artifact),
            "replay_receipt": _handle(replay),
        }
    registry = root / "execution_decks.json"
    _json(registry, {
        "schema": finalizer.EXECUTION_DECK_REGISTRY_SCHEMA,
        "study": "sage-mem-v1",
        "status": "sealed",
        "available_only_after_complete_phase_a_grid": True,
        "development_outcomes_read": False,
        "cohorts": cohorts,
        "unavailable_cohorts": {},
    })
    return registry


def test_label_reveal_fails_closed_until_the_grid_is_complete(
        tmp_path: Path) -> None:
    contract = _contract()
    phase = tmp_path / "phase"
    # One of two registered arm cells is intentionally absent.
    _write_cell(phase, contract, contract.cohorts[0], contract.arms[0], 0)
    registry = _write_registry(tmp_path, contract)
    output = tmp_path / "final"
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="directory registry differs|grid is incomplete"):
        finalizer._finalize_with_contract(phase, registry, output, contract)
    assert not (output / "label_reveal_receipt.json").exists()


def test_sealed_labels_cannot_be_loaded_without_a_reveal_receipt(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    grid = finalizer._validate_complete_grid(phase, contract)
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="cannot read label-reveal receipt"):
        finalizer._load_label_registry(
            registry, tmp_path / "absent_receipt.json", grid)


def test_per_age_identity_rows_cannot_collapse_or_drift(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))

    def wrong_age(arrays: dict[str, np.ndarray]) -> None:
        arrays["formal_test_evidence_age"][1] = 4

    phase = tmp_path / "phase"
    _write_grid(phase, contract, mutation=(
        contract.cohorts[0], "none", 0, wrong_age))
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="is not age 8"):
        finalizer._validate_complete_grid(phase, contract)


def test_cross_arm_identity_drift_is_rejected_before_reveal(
        tmp_path: Path) -> None:
    contract = _contract()

    def drift(arrays: dict[str, np.ndarray]) -> None:
        arrays["formal_test_episode_id"][:, 0] += 100_000

    phase = tmp_path / "phase"
    _write_grid(phase, contract, mutation=(
        contract.cohorts[0], contract.arms[1], 0, drift))
    registry = _write_registry(tmp_path, contract)
    output = tmp_path / "final"
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="cross-arm/seed identity drift"):
        finalizer._finalize_with_contract(phase, registry, output, contract)
    assert not (output / "label_reveal_receipt.json").exists()


def test_pointmaze_x4_clusters_survive_finalization_at_every_age(
        tmp_path: Path) -> None:
    cohort = "dinowm_pointmaze_goal"
    contract = _contract(
        cohort=cohort, formal_rows=8, consumer_rows=8, variants=4)
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    output = tmp_path / "final"
    summary = finalizer._finalize_with_contract(
        phase, registry, output, contract)
    assert summary["finalized_cells"] == 2
    assert summary["pointmaze_x4_native_clustering_preserved"] is True
    with np.load(output / "cells" / cohort / contract.arms[1] / "seed-0"
                 / "finalized_results.npz", allow_pickle=False) as result:
        clusters = result["formal_test_native_cluster_id"]
        ages = result["formal_test_evidence_age"]
        assert clusters.shape == (3, 8)
        assert np.array_equal(clusters[0], clusters[1])
        assert np.array_equal(clusters[1], clusters[2])
        assert all(np.all(ages[index] == age)
                   for index, age in enumerate(finalizer.AGES))
        assert sorted(np.unique(clusters[0], return_counts=True)[1].tolist()) \
            == [4, 4]


def test_feature_mode_fits_one_pooled_arm_blind_consumer_per_seed(
        tmp_path: Path) -> None:
    contract = _contract(classes=4, formal_rows=8, consumer_rows=8)
    phase = tmp_path / "phase"
    _write_grid(phase, contract, representation="feature_artifact")
    registry = _write_registry(tmp_path, contract)
    output = tmp_path / "final"
    finalizer._finalize_with_contract(phase, registry, output, contract)
    receipt = json.loads((output / "consumers" / contract.cohorts[0]
                          / "seed-0.json").read_text())
    assert receipt["pooled_arms"] == list(contract.arms)
    assert receipt["arm_identity_used"] is False
    assert receipt["formal_test_labels_used"] is False
    hashes = []
    for arm in contract.arms:
        manifest = json.loads((output / "cells" / contract.cohorts[0]
                               / arm / "seed-0" / "manifest.json").read_text())
        hashes.append(manifest["shared_arm_blind_consumer_sha256"])
        with np.load(output / "cells" / contract.cohorts[0] / arm / "seed-0"
                     / "finalized_results.npz", allow_pickle=False) as result:
            assert np.all(result["formal_test_full_correct"] == 1)
    assert len(set(hashes)) == 1


def test_optional_raw_context_reference_is_separate_and_per_age(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    raw = _write_raw_context(tmp_path / "raw", contract)
    output = tmp_path / "final"
    summary = finalizer._finalize_with_contract(
        phase, registry, output, contract, raw_context_root=raw)
    assert summary["raw_context_reference"] == {
        "status": "complete",
        "short_context_frames": 3,
        "long_context_frames": 16,
        "separate_from_parameter_matched_arms": True,
        "references": 1,
        "records_sha256": summary["raw_context_reference"]["records_sha256"],
    }
    path = (output / "raw_context" / contract.cohorts[0] / "seed-0"
            / "finalized_results.npz")
    with np.load(path, allow_pickle=False) as result:
        assert result["formal_test_short_correct"].shape == (3, 8)
        assert result["formal_test_long_correct"].shape == (3, 8)
        assert np.all(result["formal_test_short_correct"] == 1)
        assert np.all(result["formal_test_long_correct"] == 1)
        assert "formal_test_short_mse" not in result.files
        assert "formal_test_long_mse" not in result.files
    consumer = json.loads((
        output / "raw_context_consumers" / contract.cohorts[0]
        / "seed-0.json").read_text())
    assert consumer["training_rows"] == 2 * 3 * 8
    assert consumer["contexts"] == ["short-3", "long-16"]
    assert consumer["context_identity_used"] is False
    assert consumer["formal_test_labels_used"] is False


def test_raw_context_phase_a_rejects_semantic_label_leak(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    raw = _write_raw_context(tmp_path / "raw", contract)
    cohort = contract.cohorts[0]
    directory = raw / cohort / "seed-0"
    artifact = directory / "raw_context.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy()
                  for name in archive.files}
    arrays["formal_test_label"] = np.zeros(
        contract.formal_test_rows[cohort], dtype=np.int64)
    np.savez_compressed(artifact, **arrays)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact"] = _handle(artifact)
    _json(manifest_path, manifest)
    records = [{
        "cohort": cohort,
        "seed": 0,
        "manifest_sha256": _sha(manifest_path),
        "artifact_sha256": _sha(artifact),
        "bank_manifest_sha256": "b" * 64,
    }]
    summary_path = raw / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["records_sha256"] = finalizer._sha256_json(records)
    _json(summary_path, summary)
    output = tmp_path / "final"
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="measurement schema"):
        finalizer._finalize_with_contract(
            phase, registry, output, contract, raw_context_root=raw)
    assert not (output / "label_reveal_receipt.json").exists()


def test_real_producer_output_validates_and_uses_one_post_reveal_consumer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _contract(arms=("none",))
    cohort = contract.cohorts[0]
    classes = contract.classes[cohort]

    class SyntheticSpatialBank:
        spatial = True

        def __init__(self) -> None:
            self.ids = {
                "consumer_train": _identity(
                    contract.consumer_train_rows[cohort], 1, 1000),
                "formal_test": _identity(
                    contract.formal_test_rows[cohort], 1, 100),
            }

        def indices(self, split: str) -> np.ndarray:
            return np.arange(self.ids[split][0].shape[1], dtype=np.int64)

        def identity(self, split: str) -> dict[str, np.ndarray]:
            return {
                "episode_id": self.ids[split][0][0],
                "native_cluster_id": self.ids[split][1][0],
            }

        def features(self, age: int, indices: np.ndarray) -> np.ndarray:
            rows = np.asarray(indices)
            labels = np.arange(len(rows), dtype=np.int64) % classes
            frame = np.eye(classes, dtype=np.float32)[labels]
            return np.repeat(
                np.repeat(frame[:, None, None, :], 20, axis=1),
                2, axis=2)

    bank = SyntheticSpatialBank()
    view = raw_producer.PreparedBankView(
        cohort=cohort, spatial=True,
        bank_manifest_sha256="b" * 64, host_hash="a" * 64,
        split_banks={split: bank for split in raw_producer.SPLITS})
    monkeypatch.setattr(
        raw_producer, "_load_prepared_bank",
        lambda cohort, prepared_root, split_counts: view)
    calls = 0
    original_ridge = finalizer._ridge_consumer

    def counted_ridge(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original_ridge(*args, **kwargs)

    monkeypatch.setattr(finalizer, "_ridge_consumer", counted_ridge)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    raw = tmp_path / "raw"
    raw_producer._produce_grid(
        config_path=raw_producer.DEFAULT_CONFIG,
        prepared_root=prepared, output_root=raw,
        cohorts=(cohort,), seeds=(0,))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    grid = finalizer._validate_complete_grid(phase, contract)
    references, digest = finalizer._validate_raw_context_references(raw, grid)
    assert len(references) == 1 and len(digest) == 64
    output = tmp_path / "final"
    finalizer._finalize_with_contract(
        phase, registry, output, contract, raw_context_root=raw)
    assert calls == 1
    with np.load(output / "raw_context" / cohort / "seed-0"
                 / "finalized_results.npz", allow_pickle=False) as result:
        assert np.all(result["formal_test_short_correct"] == 1)
        assert np.all(result["formal_test_long_correct"] == 1)
        assert not any(name.endswith("_mse") for name in result.files)


def test_generic_custody_vault_is_normalized_only_after_reveal(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    custody = _write_custody_registry(tmp_path, contract)
    output = tmp_path / "final"
    finalizer._finalize_with_contract(phase, custody, output, contract)
    reveal = output / "label_reveal_receipt.json"
    normalized = output / "normalized_label_registry" / "manifest.json"
    assert reveal.is_file() and normalized.is_file()
    receipt = json.loads(reveal.read_text())
    manifest = json.loads(normalized.read_text())
    assert receipt["complete_grid_validated_before_label_reveal"] is True
    assert manifest["status"] == "normalized-after-complete-grid-reveal"
    assert manifest["label_reveal_receipt_sha256"] == _sha(reveal)


def test_shared_consumer_uses_bounded_memory_lsqr_for_high_dimension(
        monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    # d >> n catches the former dense (d+1)x(d+1) Gram/solve path while
    # remaining quick under LSQR.
    x = rng.normal(size=(24, 2048)).astype(np.float32)
    y = np.tile(np.arange(4, dtype=np.int64), 6)
    test = rng.normal(size=(5, 2048)).astype(np.float32)

    def forbidden_dense_solve(*args: object, **kwargs: object) -> None:
        raise AssertionError("dense primal solve must not be used")

    monkeypatch.setattr(np.linalg, "solve", forbidden_dense_solve)
    predictions, digest = finalizer._ridge_consumer(x, y, [test], classes=4)
    assert predictions[0].shape == (5,)
    assert len(digest) == 64


def test_production_cell_path_uses_seed_prefix_and_public_validator(
        tmp_path: Path) -> None:
    contract = finalizer.PRODUCTION_CONTRACT
    cohort, arm, seed = contract.cohorts[0], contract.arms[0], 0
    phase = tmp_path / "phase"
    _write_cell(phase, contract, cohort, arm, seed)
    path = phase / "cells" / cohort / arm / "seed-0"
    validated = finalizer.validate_phase_a_cell(path, cohort, arm, seed)
    assert validated.directory == path
    wrong = path.with_name("s0")
    path.rename(wrong)
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match=r"seed-\{seed\}"):
        finalizer.validate_phase_a_cell(wrong, cohort, arm, seed)


@pytest.mark.parametrize(("field", "bad_value"), [
    ("cuda_visible_devices", "1"),
    ("protocol_fingerprint", "short"),
])
def test_phase_a_gpu_or_protocol_identity_mismatch_is_rejected(
        tmp_path: Path, field: str, bad_value: str) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    manifest_path = (phase / "cells" / contract.cohorts[0] / "none"
                     / "seed-0" / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = bad_value
    _json(manifest_path, manifest)
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="GPU/protocol/completion"):
        finalizer._validate_complete_grid(phase, contract)


def test_eligible_execution_deck_computes_per_age_arm_blind_success(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    deck = _write_execution_deck(tmp_path, contract)
    output = tmp_path / "final"
    summary = finalizer._finalize_with_contract(
        phase, registry, output, contract,
        execution_deck_registry=deck)
    assert summary["execution_decks"]["eligible_cohorts"] == 1
    receipt = json.loads((output / "execution" / contract.cohorts[0]
                          / "receipt.json").read_text())
    assert receipt["status"] == "computed-class-conditioned-arm-blind"
    assert receipt["arm_identity_used"] is False
    with np.load(output / "cells" / contract.cohorts[0] / "none"
                 / "seed-0" / "finalized_results.npz",
                 allow_pickle=False) as result:
        assert result["formal_test_full_execution_success"].shape == (3, 8)
        assert np.all(result["formal_test_full_execution_success"] == 1)
        assert np.all(result["formal_test_reset_execution_success"] == 0)


def test_failed_oracle_gate_emits_standard_skipped_receipt(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    deck = _write_execution_deck(
        tmp_path, contract, oracle_fraction=0.5, threshold=0.9)
    output = tmp_path / "final"
    summary = finalizer._finalize_with_contract(
        phase, registry, output, contract,
        execution_deck_registry=deck)
    assert summary["execution_decks"]["eligible_cohorts"] == 0
    receipt = json.loads((output / "execution" / contract.cohorts[0]
                          / "receipt.json").read_text())
    assert receipt["status"] == "skipped-oracle-gate"
    assert receipt["skip_reason"] == \
        "oracle-success-below-preregistered-threshold"
    assert receipt["computed_cells"] == 0
    with np.load(output / "cells" / contract.cohorts[0] / "none"
                 / "seed-0" / "finalized_results.npz",
                 allow_pickle=False) as result:
        assert "formal_test_full_execution_success" not in result.files


def test_execution_oracle_is_true_target_diagonal_not_best_cube_entry(
        tmp_path: Path) -> None:
    """A reachable wrong-target action must not inflate the oracle gate."""

    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    labels = _write_registry(tmp_path, contract)
    deck = _write_execution_deck(tmp_path, contract, threshold=0.5)
    registry = json.loads(deck.read_text())
    record = registry["cohorts"][contract.cohorts[0]]
    artifact = tmp_path / record["artifact"]["path"]
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy()
                  for name in archive.files}
    classes = contract.classes[contract.cohorts[0]]
    cube = np.zeros_like(arrays["selected_class_by_true_target_success"])
    for target in range(classes):
        cube[:, (target + 1) % classes, target] = 1
    arrays["selected_class_by_true_target_success"] = cube
    np.savez_compressed(artifact, **arrays)
    record["artifact"] = _handle(artifact)
    _json(deck, registry)

    summary = finalizer._finalize_with_contract(
        phase, labels, tmp_path / "final", contract,
        execution_deck_registry=deck)
    cohort_status = summary["execution_decks"]["cohort_status"][
        contract.cohorts[0]]
    assert cohort_status["eligible"] is False
    assert cohort_status["oracle_success"] == 0.0


def test_legacy_target_conditioned_two_dimensional_deck_is_rejected(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    labels = _write_registry(tmp_path, contract)
    deck = _write_execution_deck(tmp_path, contract)
    registry = json.loads(deck.read_text())
    record = registry["cohorts"][contract.cohorts[0]]
    artifact = tmp_path / record["artifact"]["path"]
    with np.load(artifact, allow_pickle=False) as archive:
        episode = archive["formal_test_episode_id"].copy()
        cluster = archive["formal_test_native_cluster_id"].copy()
        random_class = archive["deterministic_random_class"].copy()
    np.savez_compressed(
        artifact,
        formal_test_episode_id=episode,
        formal_test_native_cluster_id=cluster,
        class_conditioned_success=np.zeros((len(episode), 4), dtype=np.uint8),
        deterministic_random_class=random_class,
    )
    record["artifact"] = _handle(artifact)
    _json(deck, registry)
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="array schema differs"):
        finalizer._finalize_with_contract(
            phase, labels, tmp_path / "final", contract,
            execution_deck_registry=deck)


def test_explicit_unavailable_receipt_cannot_count_as_eligible(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    labels = _write_registry(tmp_path, contract)
    deck = _write_execution_deck(tmp_path, contract)
    registry = json.loads(deck.read_text())
    cohort = contract.cohorts[0]
    registry["cohorts"].pop(cohort)
    receipt = tmp_path / f"{cohort}_unavailable.json"
    _json(receipt, {
        "schema": finalizer.EXECUTION_UNAVAILABLE_RECEIPT_SCHEMA,
        "study": "sage-mem-v1",
        "status": "unavailable",
        "cohort": cohort,
        "reason_code": "native-state-not-exposed",
        "bank_manifest_sha256": "b" * 64,
        "formal_labels_read": False,
        "development_outcomes_read": False,
    })
    registry["unavailable_cohorts"][cohort] = {
        "status": "unavailable",
        "bank_manifest_sha256": "b" * 64,
        "reason_code": "native-state-not-exposed",
        "receipt": _handle(receipt),
    }
    _json(deck, registry)
    summary = finalizer._finalize_with_contract(
        phase, labels, tmp_path / "final", contract,
        execution_deck_registry=deck)
    assert summary["execution_decks"]["eligible_cohorts"] == 0
    assert summary["execution_decks"]["supplied_cohorts"] == []


def test_program_execution_flag_requires_two_eligible_cohorts(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    deck = _write_execution_deck(tmp_path, contract)
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="at least two eligible cohorts"):
        finalizer._finalize_with_contract(
            phase, registry, tmp_path / "final", contract,
            execution_deck_registry=deck,
            require_at_least_two_eligible_execution_cohorts=True)


def test_existing_finalized_root_validates_for_safe_resume(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    output = tmp_path / "final"
    produced = finalizer._finalize_with_contract(
        phase, registry, output, contract)
    validated = finalizer._validate_finalized_with_contract(
        phase, registry, output, contract)
    assert validated == produced


def test_resume_cli_alias_selects_finalized_validation(tmp_path: Path) -> None:
    args = finalizer._parse_args([
        "--phase-a-root", str(tmp_path / "phase"),
        "--label-registry", str(tmp_path / "labels.json"),
        "--output-root", str(tmp_path / "final"),
        "--resume",
    ])
    assert args.validate_finalized_output is True


def test_resume_validator_rejects_finalized_artifact_tampering(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    output = tmp_path / "final"
    finalizer._finalize_with_contract(phase, registry, output, contract)
    artifact = (output / "cells" / contract.cohorts[0] / "none"
                / "seed-0" / "finalized_results.npz")
    with artifact.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="size differs|hash differs"):
        finalizer._validate_finalized_with_contract(
            phase, registry, output, contract)


def test_resume_validator_rechecks_optional_execution_and_raw_receipts(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    raw = _write_raw_context(tmp_path / "raw", contract)
    deck = _write_execution_deck(tmp_path, contract)
    output = tmp_path / "final"
    finalizer._finalize_with_contract(
        phase, registry, output, contract, raw_context_root=raw,
        execution_deck_registry=deck)
    finalizer._validate_finalized_with_contract(
        phase, registry, output, contract, raw_context_root=raw,
        execution_deck_registry=deck)
    receipt_path = output / "execution" / contract.cohorts[0] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["arm_identity_used"] = True
    _json(receipt_path, receipt)
    with pytest.raises(finalizer.SageMemFormalFinalizerError):
        finalizer._validate_finalized_with_contract(
            phase, registry, output, contract, raw_context_root=raw,
            execution_deck_registry=deck)


def test_resume_validator_rejects_summary_grid_hash_tampering(
        tmp_path: Path) -> None:
    contract = _contract(arms=("none",))
    phase = tmp_path / "phase"
    _write_grid(phase, contract)
    registry = _write_registry(tmp_path, contract)
    output = tmp_path / "final"
    finalizer._finalize_with_contract(phase, registry, output, contract)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["phase_a_grid_sha256"] = "0" * 64
    _json(summary_path, summary)
    with pytest.raises(finalizer.SageMemFormalFinalizerError,
                       match="summary identity"):
        finalizer._validate_finalized_with_contract(
            phase, registry, output, contract)
