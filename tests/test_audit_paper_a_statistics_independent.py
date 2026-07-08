from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import audit_paper_a_statistics_independent as audit


def _minimal_configs(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    for filename, output in (
            ("dinowm_wave2_spatial_carrier_v1_1.yaml", "outputs/wave2"),
            ("dinowm_pointmaze_wave3.yaml", "outputs/wave3")):
        (configs / filename).write_text(yaml.safe_dump({
            "artifacts": {"root": output, "formal": "formal"},
        }))


def test_auditor_does_not_import_sealed_producer_helpers() -> None:
    source = Path(audit.__file__).read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = [name for name in imported if
                 name.startswith("scripts.run_dinowm")
                 or name.startswith("lewm.official_tasks")
                 or name.startswith("lewm.models")]
    assert forbidden == []


def test_wave2_bootstraps_match_fixed_independent_fixture() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    left = np.array([
        [0, 1, 1, 1, 2, 0],
        [0, 0, 0, 1, 2, 2],
    ])
    right = np.array([
        [0, 0, 1, 0, 1, 2],
        [1, 0, 1, 1, 2, 0],
    ])
    paired = audit.stratified_paired_bootstrap(
        left, right, truth, classes=3, draws=257, seed=17)
    assert paired == {
        "mean": 0.08333333333333337,
        "ci95": [-0.5, 0.5],
        "draws": 257,
        "seed": 17,
        "confidence": 0.95,
        "paired": True,
        "units": ["matched carrier seed",
                  "class-stratified held-out episode"],
        "ci_excludes_zero": False,
    }
    absolute = audit.stratified_absolute_bootstrap(
        left, truth, classes=3, draws=257, seed=17)
    assert absolute == {
        "mean": 0.75,
        "ci95": [0.5, 1.0],
        "draws": 257,
        "seed": 17,
        "confidence": 0.95,
        "paired": True,
        "units": ["matched carrier seed",
                  "class-stratified held-out episode"],
        "ci_excludes_zero": True,
        "metric": "balanced_accuracy",
    }


def test_wave3_cluster_bootstrap_preserves_native_episode_unit() -> None:
    values = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
    ], dtype=float)
    episodes = np.array([10] * 4 + [20] * 4)
    result = audit.native_episode_bootstrap(
        values, episodes, draws=513, seed=23)
    assert result == {
        "mean": 0.5,
        "ci95": [0.0, 1.0],
        "draws": 513,
        "seed": 23,
        "confidence": 0.95,
        "paired": True,
        "equal_native_episode_weight": True,
        "native_episode_clusters": 2,
        "carrier_seeds": 2,
        "ci_excludes_zero": False,
    }


def test_external_execution_is_recomputed_without_producer_helper() -> None:
    success = np.zeros((2, 4, 4), dtype=np.int8)
    for base in range(2):
        for selected in range(4):
            for truth in range(4):
                success[base, selected, truth] = int(
                    (selected + truth + base) % 3 == 0)
    prediction = np.array([
        [0, 1, 2, 3, 3, 2, 1, 0],
        [1, 1, 1, 1, 2, 2, 2, 2],
    ])
    truth = np.tile(np.arange(4), 2)
    expected = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1],
    ], dtype=float)
    assert np.array_equal(
        audit.executed_success(success, prediction, truth), expected)
    with pytest.raises(audit.AuditFailure, match="base-major"):
        audit.executed_success(success, prediction, truth[::-1])
    with pytest.raises(audit.AuditFailure, match="undeclared class"):
        audit.executed_success(
            success, np.full((1, 8), -1, dtype=np.int64), truth)
    with pytest.raises(audit.AuditFailure, match="stored as integers"):
        audit.executed_success(success, prediction.astype(float), truth)
    invalid_success = success.copy()
    invalid_success[0, 0, 0] = 2
    with pytest.raises(audit.AuditFailure, match="not binary"):
        audit.executed_success(invalid_success, prediction, truth)


def test_exact_comparison_rejects_numeric_and_metadata_tamper() -> None:
    expected = {"record": {"mean": 0.25, "draws": 20_000,
                            "paired": True}, "units": ["seed", "episode"]}
    audit.compare_exact(json.loads(json.dumps(expected)), expected)
    tampered = json.loads(json.dumps(expected))
    tampered["record"]["mean"] += 1e-12
    with pytest.raises(audit.AuditFailure, match="mean"):
        audit.compare_exact(tampered, expected)
    tampered = json.loads(json.dumps(expected))
    tampered["units"] = ["episode", "seed"]
    with pytest.raises(audit.AuditFailure, match=r"units\[0\]"):
        audit.compare_exact(tampered, expected)


def test_manifest_verifier_supports_both_sealed_cell_schemas(
        tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"sealed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    wave2_record = {"size": 6, "sha256": digest}
    wave3_record = {"path": "staging/artifact.bin", "size": 6,
                    "sha256": digest}
    assert audit.verify_manifest_artifact(
        path, wave2_record, "Wave 2 fixture") == digest
    assert audit.verify_manifest_artifact(
        path, wave3_record, "Wave 3 fixture", require_path=True) == digest
    with pytest.raises(audit.AuditFailure, match="artifact path is missing"):
        audit.verify_manifest_artifact(
            path, wave2_record, "path-required fixture", require_path=True)


def test_incomplete_preflight_never_opens_prediction_or_writes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _minimal_configs(tmp_path)
    for wave, count, expected in (("wave2", 14, 50), ("wave3", 9, 25)):
        formal = tmp_path / "outputs" / wave / "formal"
        formal.mkdir(parents=True)
        (formal / "progress.json").write_text(json.dumps({
            "count": count, "expected": expected,
        }))

    def forbidden_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("prediction artifact opened before summaries")

    monkeypatch.setattr(audit.np, "load", forbidden_load)
    result = audit.audit_repository(tmp_path)
    assert result["status"] == "incomplete"
    assert result["statistics_computed"] is False
    assert result["progress"] == {
        "wave2": {"count": 14, "expected": 50},
        "wave3": {"count": 9, "expected": 25},
    }
    assert not list(tmp_path.rglob("receipt.json"))


def test_execute_refuses_incomplete_without_writing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _minimal_configs(tmp_path)
    code = audit.main([
        "--root", str(tmp_path), "--execute",
        "--output", "outputs/audit/receipt.json",
    ])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
    assert not (tmp_path / "outputs/audit/receipt.json").exists()


def test_receipt_is_opt_in_atomic_and_outside_experiment_roots(
        tmp_path: Path) -> None:
    _minimal_configs(tmp_path)
    payload = {"schema": "fixture", "status": "verified"}
    destination = Path("outputs/audit/receipt.json")
    assert audit.emit_receipt(
        tmp_path, destination, payload, execute=False) is False
    assert not (tmp_path / destination).exists()
    assert audit.emit_receipt(
        tmp_path, destination, payload, execute=True) is True
    assert json.loads((tmp_path / destination).read_text()) == payload
    with pytest.raises(audit.AuditFailure, match="already exists"):
        audit.emit_receipt(
            tmp_path, destination, payload, execute=True)
    with pytest.raises(audit.AuditFailure, match="experiment root"):
        audit.emit_receipt(
            tmp_path, Path("outputs/wave2/receipt.json"), payload,
            execute=True)


def test_balanced_accuracy_is_equal_class_not_pooled_accuracy() -> None:
    truth = np.array([0, 0, 0, 1])
    prediction = np.array([0, 0, 0, 0])
    assert audit.class_balanced_accuracy(prediction, truth, 2) == 0.5
    with pytest.raises(audit.AuditFailure, match="every declared class"):
        audit.class_balanced_accuracy(
            np.array([0, 0]), np.array([0, 0]), 2)


def test_lock_source_and_provenance_checks_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("sealed\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    lock = {"protocol_sha256": "a" * 64,
            "source_sha256": {"source.py": digest}}
    assert audit._validate_locked_sources(tmp_path, lock, "fixture") == 1
    source.write_text("changed\n")
    with pytest.raises(audit.AuditFailure, match="locked source changed"):
        audit._validate_locked_sources(tmp_path, lock, "fixture")

    provenance = {
        "schema": "fixture_provenance", "status": "complete",
        "protocol_sha256": "a" * 64,
        "source_sha256": lock["source_sha256"],
        "physical_gpu": 2, "cuda_visible_devices": "2",
        "paper_modified": False,
        "runtime_host_digest": "b" * 64,
        "runtime_host_digest_after": "b" * 64,
    }
    audit._validate_provenance(
        provenance, lock, label="fixture", schema="fixture_provenance",
        gpu=2, paper_flag="paper_modified")
    provenance["status"] = "stopped_fail_closed"
    with pytest.raises(audit.AuditFailure, match="not complete"):
        audit._validate_provenance(
            provenance, lock, label="fixture",
            schema="fixture_provenance", gpu=2,
            paper_flag="paper_modified")


def test_cell_identity_clone_and_mse_checks_are_strict() -> None:
    manifest = {
        "schema": "cell_manifest", "protocol_sha256": "a" * 64,
        "task": "task", "arm": "gru", "seed": 0,
    }
    metrics = {
        "schema": "cell_metrics", "protocol_sha256": "a" * 64,
        "task": "task", "arm": "gru", "seed": 0,
        "physical_gpu": 1, "cuda_visible_devices": "1",
        "gpu_name": "fixture GPU", "elapsed_seconds": 1.0,
        "peak_vram_bytes": 123,
        "host_digest_before": "b" * 64,
        "host_digest_after": "b" * 64, "host_unchanged": True,
        "training_labels_used": False,
        "common_schedule_sha256": "c" * 64,
        "parameter_matching": audit.PARAMETER_MATCHING,
    }
    audit._validate_common_cell(
        manifest, metrics, label="fixture", protocol_sha256="a" * 64,
        manifest_schema="cell_manifest", metrics_schema="cell_metrics",
        task="task", arm="gru", seed=0, physical_gpu=1)
    manifest["arm"] = "lstm"
    with pytest.raises(audit.AuditFailure, match="manifest identity"):
        audit._validate_common_cell(
            manifest, metrics, label="fixture", protocol_sha256="a" * 64,
            manifest_schema="cell_manifest", metrics_schema="cell_metrics",
            task="task", arm="gru", seed=0, physical_gpu=1)

    reference = {"prediction": np.array([0, 1]),
                 "mse": np.array([0.1, 0.2])}
    audit._validate_none_clone(
        reference, {name: value.copy() for name, value in reference.items()},
        "fixture")
    changed = {name: value.copy() for name, value in reference.items()}
    changed["prediction"][1] = 0
    with pytest.raises(audit.AuditFailure, match="clone differs"):
        audit._validate_none_clone(reference, changed, "fixture")
    with pytest.raises(audit.AuditFailure, match="numeric vector"):
        audit.require_numeric_vector(np.zeros(3), 4, "fixture MSE")
    with pytest.raises(audit.AuditFailure, match="non-finite"):
        audit.require_numeric_vector(
            np.array([0.0, np.inf]), 2, "fixture MSE")


def test_shared_consumers_are_refit_from_preserved_features(
        tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    arms = ["none", "gru"]
    train_truth = np.tile(np.arange(2), 4)
    validation_truth = np.tile(np.arange(2), 2)
    for arm_index, arm in enumerate(arms):
        directory = formal / "cells" / arm / "s0"
        directory.mkdir(parents=True)
        train_feature = np.stack([
            np.array([-2.0, float(index)]) if label == 0
            else np.array([2.0, float(index)])
            for index, label in enumerate(train_truth)
        ]).astype(np.float32)
        train_feature[:, 1] += arm_index * 0.01
        validation_feature = np.array([
            [-3.0, 0.0], [3.0, 0.0], [-1.0, 1.0], [1.0, 1.0],
        ], dtype=np.float32)
        np.savez_compressed(
            directory / "use_features.npz",
            train_feature=train_feature,
            validation_feature=validation_feature,
            train_truth=train_truth,
            validation_truth=validation_truth)
    predictions, receipts, truth = audit.refit_shared_consumers(
        formal, arms, [0], train_rows_per_arm=8,
        validation_rows=4, feature_dim=2, classes=2)
    assert np.array_equal(truth, validation_truth)
    assert all(np.array_equal(value, validation_truth[None])
               for value in predictions.values())
    assert receipts[0]["train_examples"] == 16
    assert receipts[0]["training_arms"] == arms
    assert len(receipts[0]["coefficient_sha256"]) == 64


def _artifact_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_pointmaze_clusters_and_execution_are_anchored_to_deck(
        tmp_path: Path) -> None:
    output = tmp_path / "output"
    cache = output / "cache"
    formal = output / "formal"
    cache.mkdir(parents=True)
    formal.mkdir()
    protocol = "a" * 64
    cfg = {
        "dataset": {"train_base_windows": 2,
                    "validation_base_windows": 2},
        "sequence": {"evidence_ages": [4], "endpoint_frames": [7]},
        "external_use": {
            "evidence_age": 4, "success_radius": 0.5,
            "oracle_success_minimum": 0.9,
            "oracle_per_class_success_minimum": 0.85,
            "off_diagonal_false_success_maximum": 0.1,
            "deterministic_reset_replay_minimum": 1.0,
        },
    }
    lock = {"protocol_sha256": protocol}
    split = np.array([0, 0, 1, 1], dtype=np.uint8)
    episode = np.array([1, 2, 10, 20], dtype=np.int64)
    local_start = np.array([0, 1, 2, 3], dtype=np.int64)
    states = np.zeros((4, 20, 4), dtype=np.float32)
    states[2:, 7] = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    metadata_path = cache / "metadata.npz"
    np.savez_compressed(
        metadata_path, actions=np.zeros((4, 19, 10), dtype=np.float32),
        proprio=np.zeros((4, 20, 4), dtype=np.float32), states=states,
        split=split, episode_index=episode, local_start=local_start)
    selection_path = cache / "selection.json"
    selection_path.write_text(json.dumps({
        "schema": "dinowm_pointmaze_wave3_selection_v1",
        "selection_sha256": "selection",
        "values": [
            {"split": "train" if value == 0 else "validation",
             "episode_index": int(episode[index]),
             "local_start": int(local_start[index])}
            for index, value in enumerate(split)
        ],
    }))
    success = np.zeros((2, 4, 4), dtype=np.int8)
    success[:, np.arange(4), np.arange(4)] = 1
    distance = np.ones((2, 4, 4), dtype=np.float32)
    distance[success == 1] = 0.0
    deck_path = cache / "execution_deck.npz"
    np.savez_compressed(
        deck_path, validation_base_index=np.array([2, 3]),
        validation_episode=np.array([10, 20]),
        initial_state=states[2:, 7], goal_waypoints=np.zeros((4, 2)),
        success_matrix=success, distance_matrix=distance,
        final_state=np.zeros((2, 4, 4)),
        steps=np.ones((2, 4), dtype=np.int32),
        replay=np.ones((2, 4), dtype=np.int8),
        selected_goal_success=np.ones((2, 4), dtype=np.int8))
    admission = {
        "schema": "dinowm_pointmaze_wave3_admission_v1",
        "status": "admitted", "admitted": True, "all_gates_required": True,
        "frozen_host": {"pass": True, "digest_before": "b" * 64,
                        "digest_after": "b" * 64},
        "shortcuts": {"4": {"probe": {"pass": True}}},
    }
    controller = {
        "schema": "dinowm_pointmaze_wave3_controller_gate_v1",
        "status": "admitted", "admitted": True,
        "current_mujoco_version": "3.0.0",
        "released_xml_sha256": "c" * 64,
        "validation_base_windows": 2, "executions": 8,
        "replayed_executions": 8, "oracle_executed_success": 1.0,
        "oracle_per_class_executed_success": [1.0] * 4,
        "off_diagonal_false_success": 0.0,
        "deterministic_replay_fidelity": 1.0,
        "thresholds": {
            "oracle_success_minimum": 0.9,
            "oracle_per_class_success_minimum": 0.85,
            "off_diagonal_false_success_maximum": 0.1,
            "deterministic_reset_replay_minimum": 1.0,
        },
        "artifact": _artifact_record(tmp_path, deck_path),
    }
    admission_path = formal / "admission.json"
    controller_path = formal / "controller_gate.json"
    admission_path.write_text(json.dumps(admission))
    controller_path.write_text(json.dumps(controller))
    (cache / "manifest.json").write_text(json.dumps({
        "schema": "dinowm_pointmaze_wave3_cache_v1",
        "status": "admitted", "protocol_sha256": protocol,
        "precarrier_gates_passed": True, "host_unchanged": True,
        "selection_sha256": "selection",
        "admission_path": str(admission_path.relative_to(tmp_path)),
        "controller_gate_path": str(controller_path.relative_to(tmp_path)),
        "admission_sha256": hashlib.sha256(
            admission_path.read_bytes()).hexdigest(),
        "controller_gate_sha256": hashlib.sha256(
            controller_path.read_bytes()).hexdigest(),
        "artifacts": {
            "base_visual": {"path": "unused", "size": 0,
                            "sha256": "d" * 64},
            "cue_visual": {"path": "unused", "size": 0,
                           "sha256": "e" * 64},
            "metadata": _artifact_record(tmp_path, metadata_path),
            "selection": _artifact_record(tmp_path, selection_path),
        },
    }))
    clusters, loaded_success, _ = audit._load_pointmaze_audit_inputs(
        tmp_path, output, formal, cfg, lock, admission, controller)
    assert np.array_equal(clusters, np.repeat([10, 20], 4))
    assert np.array_equal(loaded_success, success)
    with deck_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(audit.AuditFailure, match="referenced size differs"):
        audit._load_pointmaze_audit_inputs(
            tmp_path, output, formal, cfg, lock, admission, controller)
