"""Focused contracts for fresh, label-hidden LeWM formal banks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lewm.official_tasks.matched_memory import balanced_joint_labels
from scripts.paper_a_matched_color_v1_1_spec import (
    DEFAULT_SHA as PARENT_SHA,
    load_locked_spec as load_parent_spec,
)
import scripts.sage_mem_v1_lewm_formal as formal
import scripts.sage_mem_v1_spec as sage_spec
from scripts.sage_mem_v1_formal_finalizer import (
    _COMMON_ARRAY_KEYS as FINALIZER_COMMON_ARRAY_KEYS,
    _FEATURE_KEYS as FINALIZER_FEATURE_KEYS,
    _normalize_custody_source,
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return sage_spec.load_spec()


def _full_arrays(plan: formal.FormalSeedPlan,
                 episode_ids: np.ndarray) -> dict[str, np.ndarray]:
    count = len(episode_ids)
    labels = balanced_joint_labels(count, plan.label_seed)
    arrays: dict[str, np.ndarray] = {
        "z_base": np.zeros((count, 20, 192), dtype=np.float32),
        "actions": np.zeros((count, 19, 10), dtype=np.float32),
        "state": np.zeros((count, 20, 7), dtype=np.float32),
        "episode_index": np.asarray(episode_ids, dtype=np.int64),
        "local_start": np.zeros(count, dtype=np.int64),
        "global_frame_indices": np.arange(
            count * 20, dtype=np.int64).reshape(count, 20),
        "color_label": labels.color,
        "location_label": labels.location,
        "combination_label": labels.combination,
    }
    for age in (4, 8, 15):
        cue_off = 19 - age
        arrays[f"z_cue_age_{age}"] = np.full(
            (count, 3, 192), age, dtype=np.float32)
        arrays[f"cue_on_age_{age}"] = np.full(
            count, cue_off - 3, dtype=np.int64)
        arrays[f"cue_off_age_{age}"] = np.full(
            count, cue_off, dtype=np.int64)
    return arrays


def _synthetic_reacher_manifest(
        tmp_path: Path, spec: dict, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict, Path]:
    local_spec = copy.deepcopy(spec)
    for split in formal.FORMAL_SPLITS:
        local_spec["cohorts"]["lewm_reacher_color"][
            "split_episodes"][split] = 16
    # The validator imports this function lazily.  A 16-episode fixture keeps
    # the test small while preserving the exact registered seed/provenance
    # behavior; production uses the untouched 1200/480/720 counts.
    monkeypatch.setattr(sage_spec, "load_spec", lambda *args, **kwargs:
                        copy.deepcopy(local_spec))

    parent_path = formal.ROOT / local_spec["cohorts"][
        "lewm_reacher_color"]["parent_protocol"]
    parent = load_parent_spec(parent_path, PARENT_SHA, verify_inputs=False)
    plans = formal.formal_seed_plan(local_spec, "lewm_reacher_color")
    splits = {}
    labels_by_split = {}
    episode_sets = {}
    for split in formal.FORMAL_SPLITS:
        ids = (np.int64(plans[split].base_seed) * 10_000
               + np.arange(16, dtype=np.int64))
        splits[split], labels_by_split[split] = formal._write_split_artifacts(
            tmp_path, split, plans[split], _full_arrays(plans[split], ids))
        episode_sets[split] = set(map(int, ids))
    overlaps = {
        f"{left}::{right}": sorted(
            episode_sets[left].intersection(episode_sets[right]))
        for index, left in enumerate(formal.FORMAL_SPLITS)
        for right in formal.FORMAL_SPLITS[index + 1:]
    }
    artifact_hashes = {
        f"{split}/trajectory_artifact": splits[split][
            "trajectory_artifact"]["sha256"]
        for split in formal.FORMAL_SPLITS
    }
    vault = tmp_path.parent / f"{tmp_path.name}-sealed-label-vault.npz"
    vault_sha = formal._atomic_npz(
        vault, formal._label_vault_arrays(labels_by_split))
    vault.chmod(0o400)
    custody_path = tmp_path.parent / f"{tmp_path.name}-label-custody.json"
    custody = {
        "schema": formal.LEWM_LABEL_CUSTODY_SCHEMA,
        "study": "sage-mem-v1",
        "status": "sealed-for-post-grid-finalizer",
        "cohort": "lewm_reacher_color",
        "path": str(vault.resolve()),
        "size": vault.stat().st_size,
        "sha256": vault_sha,
        "mode": "0o400",
        "study_protocol_sha256": local_spec["_spec_sha256"],
        "split_label_hashes": {
            split: splits[split]["sealed_label_arrays_sha256"]
            for split in formal.FORMAL_SPLITS
        },
        "per_cell_api_access": False,
        "available_only_to": formal.POST_GRID_FINALIZER_PHASE,
    }
    formal._atomic_json(custody_path, custody)
    custody_path.chmod(0o400)
    parent_receipt = formal.reacher_parent_seed_receipt(parent)
    parent_rng = formal.parent_rng_registry_receipt()
    formal_rng = sorted({
        value for plan in plans.values()
        for value in (plan.base_seed, plan.label_seed, plan.loader_seed)
    })
    manifest = {
        "schema_version": 1,
        "schema": formal.LEWM_FORMAL_SCHEMA,
        "study": "sage-mem-v1",
        "status": "prepared",
        "cohort": "lewm_reacher_color",
        "formal_only": True,
        "parent_disjoint": True,
        "development_formal_disjoint": True,
        "selection_uses_semantic_labels": False,
        "labels_generated_after_selection": True,
        "labels_used_for_carrier_training": False,
        "trajectory_artifacts_contain_semantic_labels": False,
        "sealed_labels_available_only_to": formal.POST_GRID_FINALIZER_PHASE,
        "semantic_label_vault_inside_bank": False,
        "sealed_label_vault_sha256": vault_sha,
        "ages": [4, 8, 15],
        "splits": splits,
        "freshness": {
            "kind": "seed-disjoint fresh dm_control trajectories",
            "parent": parent_receipt,
            "formal_base_seeds": [
                plans[split].base_seed for split in formal.FORMAL_SPLITS],
            "formal_rng_seeds": formal_rng,
            "formal_parent_rng_overlap": [],
            "parent_rng_registry": parent_rng,
            "all_parent_rng_overlap": [],
            "forbidden_parent_artifacts":
                formal.forbidden_parent_artifact_receipt(
                    local_spec, "lewm_reacher_color"),
            "zero_overlap_proven": True,
        },
        "formal_split_overlap": overlaps,
        "artifact_hashes": artifact_hashes,
        "action_normalization": {
            "source": "fixture", "mean": [0.0] * 10,
            "std_ddof1": [1.0] * 10, "count": 16,
        },
        "study_protocol": str(Path(local_spec["_spec_path"]).resolve(
        ).relative_to(formal.ROOT.resolve())),
        "study_protocol_sha256": local_spec["_spec_sha256"],
        "parent_protocol": local_spec["cohorts"][
            "lewm_reacher_color"]["parent_protocol"],
        "parent_protocol_sha256": formal._sha256_file(parent_path),
        "host_hash_before": "a" * 64,
        "host_hash_after": "a" * 64,
        "host_unchanged": True,
        "admissions": {
            "registered_split_counts_exact": True,
            "exact_joint_label_balance": True,
            "parent_overlap_zero": True,
            "formal_split_overlap_zero": True,
            "trajectory_label_artifacts_separated": True,
            "trajectory_artifacts_label_free": True,
            "host_digest_unchanged": True,
            "artifact_hashes_reverified": True,
        },
        "elapsed_seconds": 0.0,
    }
    path = tmp_path / "manifest.json"
    formal._atomic_json(path, manifest)
    return path, local_spec, custody_path


def test_registered_formal_seeds_and_parent_registries_are_disjoint(
        spec) -> None:
    plans = formal.formal_seed_plan(spec, "lewm_reacher_color")
    all_formal = {
        value for plan in plans.values()
        for value in (plan.base_seed, plan.label_seed, plan.loader_seed)
    }
    parent = load_parent_spec(
        formal.ROOT / spec["cohorts"]["lewm_reacher_color"][
            "parent_protocol"], PARENT_SHA, verify_inputs=False)
    parent_receipt = formal.reacher_parent_seed_receipt(parent)
    parent_rng = formal.parent_rng_registry_receipt()
    assert len(all_formal) == 9
    assert not all_formal.intersection(parent_receipt["parent_reacher_seeds"])
    assert parent_rng["seed_count"] >= 20
    assert not all_formal.intersection(parent_rng["seed_values"])

    excluded, receipt = formal.pusht_parent_exclusion_receipt(parent)
    assert len(excluded) == receipt["union_count"] == 5040
    assert receipt["locked_prior_count"] == 3360
    assert len(receipt["records"]) == 6
    assert receipt["includes_current_development_parent"] is True


def test_fresh_hdf_selection_is_deterministic_and_mutually_disjoint() -> None:
    lengths = np.full(160, 120, dtype=np.int64)
    counts = {"formal_train": 32, "consumer_train": 16, "formal_test": 16}
    plans = {
        split: formal.FormalSeedPlan(100 + index, 200 + index, 300 + index)
        for index, split in enumerate(formal.FORMAL_SPLITS)
    }
    excluded = set(range(20))
    first = formal.select_fresh_hdf_splits(
        lengths, excluded=excluded, split_counts=counts, seed_plan=plans)
    second = formal.select_fresh_hdf_splits(
        lengths, excluded=excluded, split_counts=counts, seed_plan=plans)
    assert first == second
    seen = set()
    for split in formal.FORMAL_SPLITS:
        selected = {item.episode_index for item in first[split]}
        assert len(selected) == counts[split]
        assert not selected.intersection(excluded)
        assert not selected.intersection(seen)
        assert all(0 <= item.local_start <= 24 for item in first[split])
        seen.update(selected)


def test_split_partition_and_trajectory_api_never_expose_labels(spec) -> None:
    plan = formal.formal_seed_plan(spec, "lewm_reacher_color")["formal_train"]
    arrays = _full_arrays(plan, np.arange(16, dtype=np.int64))
    trajectories, labels = formal.partition_formal_split_arrays(arrays)
    assert set(trajectories) == formal.trajectory_split_keys()
    assert set(labels) == formal.label_split_keys()
    assert not set(trajectories).intersection(
        {"color_label", "location_label", "combination_label"})
    bank = formal.FormalLeWMTrajectoryBank(trajectories)
    assert not hasattr(bank, "labels")
    assert bank.features(8, np.array([0, 1])).shape == (2, 20, 192)
    assert np.all(bank.features(8, np.array([0]))[:, 8:11] == 8)
    assert bank.native_state(np.array([0])).shape == (1, 20, 7)

    leaked = dict(trajectories)
    leaked["color_label"] = labels["color_label"]
    with pytest.raises(formal.LeWMFormalError, match="schema"):
        formal.FormalLeWMTrajectoryBank(leaked)


def test_manifest_replays_provenance_and_enforces_post_grid_label_boundary(
        tmp_path, spec, monkeypatch) -> None:
    path, _, custody = _synthetic_reacher_manifest(
        tmp_path, spec, monkeypatch)
    manifest = formal.validate_lewm_formal_manifest(path)
    assert manifest["admissions"]["parent_overlap_zero"] is True
    assert manifest["semantic_label_vault_inside_bank"] is False
    assert {item.name for item in path.parent.iterdir()} == {
        "manifest.json",
        "formal_train.trajectories.npz",
        "consumer_train.trajectories.npz",
        "formal_test.trajectories.npz",
    }

    loaded_paths = []
    real_load = formal._load_npz

    def tracking_load(value):
        loaded_paths.append(Path(value).name)
        return real_load(value)

    monkeypatch.setattr(formal, "_load_npz", tracking_load)
    handle, banks = formal.load_lewm_trajectory_banks(path)
    assert set(banks) == set(formal.FORMAL_SPLITS)
    assert all("sealed-labels" not in value for value in loaded_paths)
    serialized = json.dumps(handle, sort_keys=True)
    assert "sealed-labels" not in serialized
    assert "sealed_label_artifact" not in serialized
    assert handle["formal_labels_hidden"] is True
    assert handle["labels_accessible_through_handle"] is False

    with pytest.raises(formal.LeWMFormalError, match="unavailable"):
        formal.load_lewm_sealed_labels(
            path, custody, phase="carrier-cell")
    vault_handle = formal.sealed_label_vault_handle(path, custody)
    assert vault_handle["per_cell_api_access"] is False
    sealed = formal.load_lewm_sealed_labels(
        path, custody, phase=formal.POST_GRID_FINALIZER_PHASE)
    assert set(np.unique(sealed["formal_test"].color)) == {0, 1, 2, 3}
    record = formal.finalizer_custody_record(
        path, custody, registry_root=custody.parent)
    assert set(record) == {"bank_manifest_sha256", "classes", "sources"}
    episode, cluster, labels = _normalize_custody_source(
        custody.parent / "registry.json", "lewm_reacher_color",
        "formal_test", record["sources"]["formal_test"],
        sealed["formal_test"].episode_ids,
        sealed["formal_test"].episode_ids)
    assert np.array_equal(episode, cluster)
    assert np.array_equal(labels, sealed["formal_test"].color)

    (path.parent / "leaked-labels.npz").write_bytes(b"forbidden")
    with pytest.raises(formal.LeWMFormalError, match="unexpected files"):
        formal.validate_lewm_formal_manifest(path)


def test_manifest_rejects_artifact_tampering(
        tmp_path, spec, monkeypatch) -> None:
    path, _, _ = _synthetic_reacher_manifest(tmp_path, spec, monkeypatch)
    value = json.loads(path.read_text())
    artifact = tmp_path / value["splits"]["formal_test"][
        "trajectory_artifact"]["path"]
    with artifact.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(formal.LeWMFormalError, match="identity failed"):
        formal.validate_lewm_formal_manifest(path)


def _feature_arrays(consumer_ids: np.ndarray,
                    test_ids: np.ndarray) -> dict[str, np.ndarray]:
    arrays = formal.phase_a_identity_arrays(
        consumer_train_episode_ids=consumer_ids,
        formal_test_episode_ids=test_ids)
    for name in ("formal_test_full_mse", "formal_test_reset_mse",
                 "formal_test_prior_mse"):
        arrays[name] = np.full(
            (3, len(test_ids)), 0.1, dtype=np.float32)
    arrays["formal_test_full_features"] = np.zeros(
        (3, len(test_ids), 192), dtype=np.float32)
    arrays["formal_test_reset_features"] = np.zeros(
        (3, len(test_ids), 192), dtype=np.float32)
    arrays["formal_test_prior_features"] = np.zeros(
        (3, len(test_ids), 192), dtype=np.float32)
    arrays["consumer_train_full_features"] = np.zeros(
        (3, len(consumer_ids), 192), dtype=np.float32)
    return arrays


def test_label_free_feature_handle_binds_cells_without_correctness(
        tmp_path, spec, monkeypatch) -> None:
    assert formal.label_free_feature_keys() == (
        FINALIZER_COMMON_ARRAY_KEYS | FINALIZER_FEATURE_KEYS)
    bank_root = tmp_path / "bank"
    bank_root.mkdir()
    bank_manifest, _, _ = _synthetic_reacher_manifest(
        bank_root, spec, monkeypatch)
    trajectory = formal.trajectory_bank_handle(bank_manifest)
    cell = tmp_path / "cell"
    cell.mkdir()
    identities = {}
    for split in formal.FEATURE_SPLITS:
        with np.load(trajectory["splits"][split]["trajectory_artifact"][
                "path"], allow_pickle=False) as archive:
            identities[split] = archive["episode_index"]
    artifact = cell / "measurements.npz"
    formal._atomic_npz(artifact, _feature_arrays(
        identities["consumer_train"], identities["formal_test"]))
    handle = formal.validate_lewm_phase_a_measurement_artifact(
        artifact, trajectory_handle=trajectory)
    assert handle["formal_test_labels_read"] is False
    assert handle["prediction_representation"] == "feature_artifact"
    assert handle["consumer_contract"] == \
        "centralized-pooled-consumer-train-features"

    leaked = _feature_arrays(
        identities["consumer_train"], identities["formal_test"])
    leaked["class_id"] = np.zeros(16, dtype=np.int64)
    with pytest.raises(formal.LeWMFormalError, match="schema"):
        formal.validate_label_free_feature_arrays(
            leaked,
            expected_consumer_train_episode_ids=identities["consumer_train"],
            expected_formal_test_episode_ids=identities["formal_test"])
