from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

import scripts.sage_mem_v1_dino_formal as formal


def _seed_registry(*cohorts: str) -> dict[str, int]:
    result: dict[str, int] = {}
    value = 10_000
    for cohort in cohorts:
        for split in formal.FORMAL_SPLITS:
            for purpose in ("episode_selection", "cue_labels", "loader"):
                result[f"{cohort}/{split}/{purpose}"] = value
                value += 1
    return result


def _parent_registry(*episodes: int) -> formal.ParentExclusionRegistry:
    starts = frozenset((int(value), 0) for value in episodes)
    payload = {
        "episodes": sorted(episodes),
        "starts": sorted([list(value) for value in starts]),
        "seeds": [1, 2, 3],
    }
    return formal.ParentExclusionRegistry(
        episodes=frozenset(map(int, episodes)),
        episode_starts=starts,
        seeds=frozenset((1, 2, 3)),
        artifacts=(),
        registry_sha256=formal.sha256_json(payload),
    )


def _protocol(tmp_path: Path) -> Path:
    path = tmp_path / "parent.yaml"
    path.write_text(yaml.safe_dump({"dataset": {"split_seed": 1}}))
    return path


def test_parent_registry_follows_only_selection_metadata_and_collects_rng(
        tmp_path: Path) -> None:
    protocol = tmp_path / "parent.yaml"
    protocol.write_text(yaml.safe_dump({
        "dataset": {"split_seed": 71, "start_seed": 72},
        "training": {"seeds": [0, 1, 2]},
    }))
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    np.savez_compressed(
        registry_root / "metadata.npz",
        episode_index=np.asarray([3, 7, 11], dtype=np.int64),
        local_start=np.asarray([1, 2, 3], dtype=np.int64),
    )
    (registry_root / "selection.json").write_text(json.dumps({
        "values": [
            {"episode_index": 17, "local_start": 4},
            {"episode_index": 19, "local_start": 5},
        ],
        # Provenance episode lists are intentionally not selection rows.
        "dataset": {"native_train_episodes": list(range(100))},
        "selection_seed": 73,
    }))
    manifest = registry_root / "manifest.json"
    manifest.write_text(json.dumps({
        "artifacts": {
            "metadata": {"path": "registry/metadata.npz"},
            "selection": {"path": "registry/selection.json"},
        }
    }))

    value = formal.collect_parent_exclusion_registry(
        parent_protocol="parent.yaml",
        forbidden_parent_artifacts=["registry/manifest.json"],
        root=tmp_path,
    )

    assert value.episodes == frozenset((3, 7, 11, 17, 19))
    assert value.episode_starts == frozenset(
        ((3, 1), (7, 2), (11, 3), (17, 4), (19, 5)))
    assert {0, 1, 2, 71, 72, 73}.issubset(value.seeds)
    assert len(value.artifacts) == 4
    assert all(len(record["sha256"]) == 64 for record in value.artifacts)
    assert value.public_receipt()["episode_count"] == 5


def test_pusht_pair_is_exact_deterministic_parent_and_peer_disjoint(
        tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    parent = _parent_registry(0, 1, 2, 3)
    counts = {
        "dinowm_pusht_token": {
            "formal_train": 8, "consumer_train": 4, "formal_test": 8},
        "dinowm_pusht_binding": {
            "formal_train": 6, "consumer_train": 6, "formal_test": 6},
    }
    seeds = _seed_registry(*counts)
    kwargs = dict(
        split_counts_by_cohort=counts,
        seed_registry=seeds,
        episode_lengths=[150] * 100,
        parent_registry=parent,
        parent_protocol=protocol,
        eligible_episodes=range(100),
        root=tmp_path,
    )
    first = formal.plan_pusht_formal_pair(**kwargs)
    second = formal.plan_pusht_formal_pair(**kwargs)

    assert {key: value.plan_sha256 for key, value in first.items()} == {
        key: value.plan_sha256 for key, value in second.items()}
    token = first["dinowm_pusht_token"]
    binding = first["dinowm_pusht_binding"]
    assert not token.native_episodes & binding.native_episodes
    assert not token.native_episodes & parent.episodes
    assert not binding.native_episodes & parent.episodes
    for cohort, plan in first.items():
        for split, count in counts[cohort].items():
            record = plan.split_record(split)
            assert record["count"] == count
            assert record["expanded_sequence_count"] == count
            assert record["native_episode_count"] == count
        receipt_text = json.dumps(plan.public_receipt())
        assert '"class_id"' not in receipt_text
        assert plan.public_receipt()["semantic_labels_in_public_receipt"] is False


def test_pointmaze_counts_are_native_bases_and_expand_fourfold(
        tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    counts = {"formal_train": 5, "consumer_train": 3, "formal_test": 4}
    plan = formal.plan_fresh_formal_selection(
        cohort="dinowm_pointmaze_goal",
        split_counts=counts,
        seed_registry=_seed_registry("dinowm_pointmaze_goal"),
        episode_lengths=[100] * 40,
        parent_registry=_parent_registry(0, 1),
        eligible_episodes=range(40),
        parent_protocol=protocol,
        root=tmp_path,
    )

    assert len(plan.native_episodes) == sum(counts.values())
    assert len(plan.rows) == 4 * sum(counts.values())
    for split, count in counts.items():
        rows = plan.rows_for(split)
        assert len(rows) == 4 * count
        assert plan.split_record(split)["count"] == count
        assert plan.split_record(split)["expanded_sequence_count"] == 4 * count
        by_cluster: dict[int, list[formal.FormalSelectionRow]] = {}
        for row in rows:
            by_cluster.setdefault(row.native_cluster_id, []).append(row)
        assert len(by_cluster) == count
        for variants in by_cluster.values():
            assert {row.class_id for row in variants} == {0, 1, 2, 3}
            assert len({row.episode_index for row in variants}) == 1
            assert len({row.local_start for row in variants}) == 1
            assert len({row.episode_id for row in variants}) == 4


def test_seed_collision_with_parent_fails_closed(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    seeds = _seed_registry("dinowm_pusht_token")
    collision = seeds["dinowm_pusht_token/formal_train/episode_selection"]
    parent = replace(_parent_registry(0), seeds=frozenset((collision,)))
    with pytest.raises(formal.DinoFormalError, match="collide"):
        formal.plan_fresh_formal_selection(
            cohort="dinowm_pusht_token",
            split_counts={
                "formal_train": 4, "consumer_train": 4, "formal_test": 4},
            seed_registry=seeds,
            episode_lengths=[150] * 30,
            parent_registry=parent,
            parent_protocol=protocol,
            root=tmp_path,
        )


def _write_fake_label_free_bank(
        tmp_path: Path, plan: formal.FormalSelectionPlan) -> Path:
    root = tmp_path / "bank"
    root.mkdir()
    native = []
    lookup: dict[int, int] = {}
    base_index = []
    for row in plan.rows:
        if row.native_cluster_id not in lookup:
            lookup[row.native_cluster_id] = len(native)
            native.append(row.native_cluster_id)
        base_index.append(lookup[row.native_cluster_id])
    base = np.zeros((len(native), 20, 2, 3), dtype=np.float32)
    cue = np.full((len(plan.rows), 3, 2, 3), 7.0, dtype=np.float32)
    np.save(root / "base_visual.npy", base)
    np.save(root / "cue_visual.npy", cue)
    split_code = {split: index for index, split in enumerate(formal.FORMAL_SPLITS)}
    np.savez_compressed(
        root / "metadata.npz",
        split=np.asarray([split_code[row.split] for row in plan.rows],
                         dtype=np.uint8),
        episode_index=np.asarray([row.episode_index for row in plan.rows],
                                 dtype=np.int64),
        local_start=np.asarray([row.local_start for row in plan.rows],
                               dtype=np.int64),
        episode_id=np.asarray([row.episode_id for row in plan.rows],
                              dtype=np.int64),
        native_cluster_id=np.asarray(
            [row.native_cluster_id for row in plan.rows], dtype=np.int64),
        base_index=np.asarray(base_index, dtype=np.int64),
        base_actions=np.zeros((len(native), 19, 10), dtype=np.float32),
        base_proprio=np.zeros((len(native), 20, 4), dtype=np.float32),
        base_states=np.zeros((len(native), 20, 4), dtype=np.float32),
    )
    (root / "selection.json").write_text(json.dumps(plan.public_receipt()))
    artifacts = {}
    for name, filename in {
            "base_visual": "base_visual.npy",
            "cue_visual": "cue_visual.npy",
            "metadata": "metadata.npz",
            "selection": "selection.json"}.items():
        path = root / filename
        artifacts[name] = {
            "path": filename,
            "size": path.stat().st_size,
            "sha256": formal.sha256_file(path),
        }
    manifest = {
        "schema": "sage_mem_v1_dino_formal_bank_v1",
        "api_version": formal.DINO_FORMAL_API_VERSION,
        "status": "prepared",
        "cohort": plan.cohort,
        "plan_sha256": plan.plan_sha256,
        "host_hash_before": "a" * 64,
        "host_hash_after": "a" * 64,
        "dependency_activation": {
            "status": "activated-before-native-host-import"},
        "admission_proof": {"status": "admitted"},
        "freshness_proof": {
            "parent_episode_overlap_count": 0,
            "cross_split_native_episode_overlap_count": 0,
        },
        "splits": {split: plan.split_record(split)
                   for split in formal.FORMAL_SPLITS},
        "sealed_label_vault_sha256": "b" * 64,
        "semantic_label_vault_inside_bank": False,
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_label_free_handle_exposes_features_trajectory_and_no_labels(
        tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    plan = formal.plan_fresh_formal_selection(
        cohort="dinowm_pointmaze_goal",
        split_counts={
            "formal_train": 2, "consumer_train": 1, "formal_test": 2},
        seed_registry=_seed_registry("dinowm_pointmaze_goal"),
        episode_lengths=[100] * 20,
        parent_registry=_parent_registry(0),
        parent_protocol=protocol,
        root=tmp_path,
    )
    root = _write_fake_label_free_bank(tmp_path, plan)
    bank = formal.open_label_free_bank(root)
    rows = bank.indices("formal_test")

    assert len(rows) == 8
    features = bank.features(15, rows[:2])
    assert features.shape == (2, 20, 2, 3)
    assert np.all(features[:, 1:4] == 7.0)
    trajectory = bank.trajectory(rows[:2])
    assert set(trajectory) == {
        "row_index", "episode_id", "native_episode_index",
        "native_cluster_id", "local_start", "actions", "proprio", "states"}
    assert not hasattr(bank, "labels")
    handle = formal.validate_materialized_bank_provenance(root)
    assert handle["labels_accessible_through_handle"] is False
    assert handle["semantic_label_vault_inside_bank"] is False
    assert not (root / "formal_labels.npz").exists()

    phase = bank.phase_a_identity_inputs()
    assert set(phase) == {
        "formal_test_episode_id", "formal_test_native_cluster_id",
        "formal_test_evidence_age", "consumer_train_episode_id",
        "consumer_train_native_cluster_id",
        "consumer_train_evidence_age",
    }
    assert phase["formal_test_episode_id"].shape == (3, 8)
    assert phase["consumer_train_episode_id"].shape == (3, 4)
    assert np.array_equal(
        phase["formal_test_episode_id"][0],
        phase["formal_test_episode_id"][2])
    assert np.array_equal(
        phase["formal_test_evidence_age"][:, 0],
        np.asarray(formal.AGES))
    _, multiplicity = np.unique(
        phase["formal_test_native_cluster_id"][0], return_counts=True)
    assert np.all(multiplicity == 4)
    age_deck = bank.phase_a_split_inputs("formal_test", 8)
    assert age_deck["features"].shape == (8, 20, 2, 3)
    assert np.all(age_deck["evidence_age"] == 8)


def test_actual_overlap_proof_reopens_metadata_and_rejects_parent(
        tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    parent = _parent_registry(0)
    plan = formal.plan_fresh_formal_selection(
        cohort="dinowm_pusht_token",
        split_counts={
            "formal_train": 4, "consumer_train": 4, "formal_test": 4},
        seed_registry=_seed_registry("dinowm_pusht_token"),
        episode_lengths=[150] * 30,
        parent_registry=parent,
        parent_protocol=protocol,
        root=tmp_path,
    )
    metadata = tmp_path / "metadata.npz"
    public = formal._public_metadata(  # noqa: SLF001 - focused invariant test
        plan, np.arange(len(plan.rows), dtype=np.int64))
    np.savez_compressed(metadata, **public)
    proof = formal._actual_overlap_proof(  # noqa: SLF001
        plan, metadata)
    assert proof["parent_episode_overlap_count"] == 0

    public["episode_index"][0] = 0
    np.savez_compressed(metadata, **public)
    with pytest.raises(formal.DinoFormalError, match="sealed selection"):
        formal._actual_overlap_proof(plan, metadata)  # noqa: SLF001


def test_parent_pusht_admission_artifacts_are_hash_authenticated(
        tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    plan = formal.plan_fresh_formal_selection(
        cohort="dinowm_pusht_token",
        split_counts={
            "formal_train": 4, "consumer_train": 4, "formal_test": 4},
        seed_registry=_seed_registry("dinowm_pusht_token"),
        episode_lengths=[150] * 30,
        parent_registry=_parent_registry(0),
        parent_protocol=protocol,
        root=tmp_path,
    )
    artifact_root = tmp_path / "parent_output"
    cache = artifact_root / "cache"
    formal_root = artifact_root / "formal"
    cache.mkdir(parents=True)
    formal_root.mkdir()
    task = formal_root / "task.json"
    rollout = formal_root / "rollout.json"
    task.write_text(json.dumps({"admitted": True}))
    rollout.write_text(json.dumps({"admitted": True}))
    (cache / "manifest.json").write_text(json.dumps({
        "admissions": {
            "tasks": {
                "transient-visual-token-recall": {
                    "admitted": True,
                    "path": str(task),
                    "sha256": formal.sha256_file(task),
                }
            },
            "rollout_health": {
                "admitted": True,
                "path": str(rollout),
                "sha256": formal.sha256_file(rollout),
            },
        }
    }))
    proof = formal._parent_admission_proof(  # noqa: SLF001
        plan, {"artifacts": {"root": str(artifact_root)}})
    assert proof["status"] == "admitted"
    assert proof["task_admission"]["sha256"] == formal.sha256_file(task)

    task.write_text(json.dumps({"admitted": False}))
    with pytest.raises(formal.DinoFormalError, match="identity"):
        formal._parent_admission_proof(  # noqa: SLF001
            plan, {"artifacts": {"root": str(artifact_root)}})


def test_module_has_no_per_cell_result_or_consumer_api() -> None:
    forbidden = (
        "finalize_dino_formal_cell",
        "formal_identity_arrays",
        "validate_formal_episode_arrays",
        "prepared_payload",
        "execution_identity_sha256",
    )
    assert not any(hasattr(formal, name) for name in forbidden)


def test_pinned_dependency_activation_and_no_auto_execution(
        tmp_path: Path) -> None:
    isolated = tmp_path / "env" / "bin" / "python"
    isolated.parent.mkdir(parents=True)
    isolated.write_text("synthetic isolated python")
    site = tmp_path / "env" / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    shim = tmp_path / "shim"
    shim.mkdir()
    pth = site / "formal.pth"
    pth.write_text(f"# pinned shim\n{shim}\n")
    manifest = tmp_path / "deps.json"
    manifest.write_text(json.dumps({"locked": True}))
    cfg = {
        "execution": {
            "isolated_python": str(isolated),
            "dependency_manifest_path": str(manifest),
            "dependency_manifest_identity": {
                "size": manifest.stat().st_size,
                "sha256": formal.sha256_file(manifest),
            },
        }
    }
    prior = list(sys.path)
    try:
        receipt = formal.activate_parent_dependency_environment(
            cfg, root=tmp_path)
        assert receipt["status"] == "activated-before-native-host-import"
        assert receipt["activated_paths"] == [str(shim.resolve())]
        assert sys.path[0] == str(shim.resolve())
        assert receipt["dependency_manifest"]["sha256"] == \
            formal.sha256_file(manifest)
        assert receipt["pth"]["sha256"] == formal.sha256_file(pth)
    finally:
        sys.path[:] = prior

    source = inspect.getsource(formal.materialize_dino_formal_bank)
    assert source.index("activate_parent_dependency_environment") < \
        source.index("_materialize_pusht")
    tree = ast.parse(Path(formal.__file__).read_text())
    assert not any(isinstance(node, ast.If) for node in tree.body)
    top_imports = [node for node in tree.body
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("run_dinowm" in ast.unparse(node) for node in top_imports)


def test_pointmaze_native_source_split_is_distinct_from_formal_split(
        tmp_path: Path) -> None:
    plan = formal.plan_fresh_formal_selection(
        cohort="dinowm_pointmaze_goal",
        split_counts={
            "formal_train": 2, "consumer_train": 1, "formal_test": 2},
        seed_registry=_seed_registry("dinowm_pointmaze_goal"),
        episode_lengths=[100] * 20,
        parent_registry=_parent_registry(0),
        parent_protocol=_protocol(tmp_path),
        root=tmp_path,
        source_split="train",
    )
    assert {row.source_split for row in plan.rows} == {"train"}
    assert {row.split for row in plan.rows} == set(formal.FORMAL_SPLITS)
    source = inspect.getsource(formal._materialize_pointmaze)  # noqa: SLF001
    assert "split=row.source_split" in source
    assert "split=row.split" not in source
