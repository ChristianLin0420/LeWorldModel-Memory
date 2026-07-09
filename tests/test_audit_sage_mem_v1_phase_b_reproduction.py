"""Synthetic contract tests for the independent Phase-B reproducer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import types
from typing import Any, Callable

import numpy as np
import pytest

from scripts import audit_sage_mem_v1_phase_b_reproduction as reproduction
from scripts import sage_mem_v1_formal_finalizer as finalizer


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handle(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": _sha(path),
            "size": path.stat().st_size}


def _identity(rows: int, offset: int) -> tuple[np.ndarray, ...]:
    episode = np.arange(offset, offset + rows, dtype=np.int64)
    cluster = episode.copy()
    return (
        np.repeat(episode[None], 3, axis=0),
        np.repeat(cluster[None], 3, axis=0),
        np.stack([np.full(rows, age, dtype=np.int64)
                  for age in reproduction.AGES]),
    )


@dataclass
class Fixture:
    workspace: Path
    contract: reproduction.ReproductionContract
    paths: reproduction.InputPaths
    expected: reproduction.ExpectedHashes
    report: dict[str, Any]
    verifier_source: Path


def _lock_record(path: Path, workspace: Path, *, include_path: bool = True) \
        -> dict[str, Any]:
    result: dict[str, Any] = {"sha256": _sha(path),
                              "size": path.stat().st_size}
    if include_path:
        result["path"] = path.relative_to(workspace).as_posix()
    return result


def _make_fixture(tmp_path: Path) -> Fixture:
    workspace = tmp_path / "repo"
    workspace.mkdir(parents=True)
    cohort = "lewm_reacher_color"
    contract = reproduction.ReproductionContract(
        cohorts=(cohort,), arms=("none", "sage_mem_full"), seeds=(0,),
        ages=reproduction.AGES, classes={cohort: 4},
        formal_rows={cohort: 8}, consumer_rows={cohort: 8},
        variants={cohort: 1}, physical_gpus={cohort: 0},
        require_600=False)
    final_contract = finalizer._GridContract(
        cohorts=contract.cohorts, arms=contract.arms, seeds=contract.seeds,
        ages=contract.ages, classes=contract.classes,
        formal_test_rows=contract.formal_rows,
        consumer_train_rows=contract.consumer_rows,
        variants_per_cluster=contract.variants,
        physical_gpus=contract.physical_gpus,
        protocol_fingerprint=None, require_600=False)

    spec_path = workspace / "configs/sage_mem_v1.yaml"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("study: sage-mem-v1\nfixture: true\n",
                         encoding="utf-8")
    spec = {"study": "sage-mem-v1", "fixture": True}
    fingerprint = hashlib.sha256(_canonical(spec).encode()).hexdigest()
    final_contract = replace(final_contract,
                             protocol_fingerprint=fingerprint)
    final_contract.validate()

    phase = workspace / "phase"
    formal_identity = _identity(8, 100)
    consumer_identity = _identity(8, 1000)
    classes = np.arange(8, dtype=np.int64) % 4
    one_hot = np.eye(4, dtype=np.float32)[classes]
    for arm_index, arm in enumerate(contract.arms):
        directory = phase / "cells" / cohort / arm / "seed-0"
        directory.mkdir(parents=True)
        offset = np.float32(arm_index * 0.01)
        features = one_hot + offset
        arrays = {
            "formal_test_episode_id": formal_identity[0],
            "formal_test_native_cluster_id": formal_identity[1],
            "formal_test_evidence_age": formal_identity[2],
            "consumer_train_episode_id": consumer_identity[0],
            "consumer_train_native_cluster_id": consumer_identity[1],
            "consumer_train_evidence_age": consumer_identity[2],
            "formal_test_full_mse": np.full((3, 8), 0.1 + offset),
            "formal_test_reset_mse": np.full((3, 8), 0.2 + offset),
            "formal_test_prior_mse": np.full((3, 8), 0.15 + offset),
            "formal_test_full_features": np.repeat(features[None], 3, 0),
            "formal_test_reset_features": np.repeat(features[None], 3, 0),
            "formal_test_prior_features": np.repeat(features[None], 3, 0),
            "consumer_train_full_features": np.repeat(features[None], 3, 0),
        }
        measurement = directory / "measurements.npz"
        np.savez_compressed(measurement, **arrays)
        checkpoint = directory / "checkpoint.bin"
        checkpoint.write_bytes(f"{arm}/0".encode())
        history = directory / "history.json"
        _json(history, {
            "schema": finalizer.HISTORY_SCHEMA, "study": "sage-mem-v1",
            "status": "complete", "formal_test_labels_read": False,
            "development_outcomes_read": False,
            "bank_manifest_sha256": "b" * 64,
            "epochs": ([] if arm == "none" else [
                {"epoch": 0, "train_label_free_loss": 0.5}]),
        })
        resource = directory / "resources.json"
        _json(resource, {
            "schema": finalizer.RESOURCE_SCHEMA, "study": "sage-mem-v1",
            "status": "complete",
            "metrics": {name: float(index + 1) for index, name in enumerate(
                finalizer.RESOURCE_FIELDS)},
        })
        _json(directory / "manifest.json", {
            "schema": finalizer.PHASE_A_SCHEMA, "study": "sage-mem-v1",
            "stage": "formal-phase-a", "status": "complete-label-free",
            "cohort": cohort, "arm": arm, "seed": 0,
            "physical_gpu": 0, "cuda_visible_devices": "0",
            "protocol_fingerprint": fingerprint,
            "completed_unix_ns": 1_700_000_000_000_000_000,
            "ages": list(contract.ages),
            "formal_test_labels_read": False,
            "formal_test_labels_available": False,
            "development_outcomes_read": False,
            "labels_used_for_training": False,
            "bank_manifest_sha256": "b" * 64,
            "host_hash_before": "a" * 64, "host_hash_after": "a" * 64,
            "prediction_representation": "feature_artifact",
            "consumer_contract":
                "centralized-pooled-consumer-train-features",
            "shared_consumer_sha256": None,
            "artifacts": {
                "measurements": _handle(measurement),
                "checkpoint": _handle(checkpoint),
                "history": _handle(history),
                "resource_report": _handle(resource),
            },
        })

    custody_root = workspace / "custody"
    custody_root.mkdir()
    vault = custody_root / "vault.npz"
    np.savez_compressed(
        vault,
        episode=np.concatenate([formal_identity[0][0],
                                consumer_identity[0][0]]),
        cluster=np.concatenate([formal_identity[1][0],
                                consumer_identity[1][0]]),
        semantic=np.concatenate([classes, classes]))
    source = {
        "artifact": _handle(vault),
        "keys": {"episode_id": "episode", "native_cluster_id": "cluster",
                 "label": "semantic"},
    }
    label_registry = custody_root / "registry.json"
    _json(label_registry, {
        "schema": finalizer.CUSTODY_REGISTRY_SCHEMA,
        "study": "sage-mem-v1", "status": "sealed",
        "labels_available_only_after_complete_phase_a_grid": True,
        "development_outcomes_read": False,
        "cohorts": {cohort: {
            "bank_manifest_sha256": "b" * 64, "classes": 4,
            "sources": {"formal_test": source, "consumer_train": source},
        }},
    })

    raw = workspace / "raw"
    raw_cell = raw / cohort / "seed-0"
    raw_cell.mkdir(parents=True)
    raw_artifact = raw_cell / "raw.npz"
    raw_arrays = {
        "formal_test_episode_id": formal_identity[0],
        "formal_test_native_cluster_id": formal_identity[1],
        "formal_test_evidence_age": formal_identity[2],
        "consumer_train_episode_id": consumer_identity[0],
        "consumer_train_native_cluster_id": consumer_identity[1],
        "consumer_train_evidence_age": consumer_identity[2],
        "formal_test_short_features": np.repeat(one_hot[None], 3, 0),
        "formal_test_long_features": np.repeat(one_hot[None], 3, 0),
        "consumer_train_short_features": np.repeat(one_hot[None], 3, 0),
        "consumer_train_long_features": np.repeat(one_hot[None], 3, 0),
    }
    np.savez_compressed(raw_artifact, **raw_arrays)
    raw_manifest = raw_cell / "manifest.json"
    _json(raw_manifest, {
        "schema": finalizer.RAW_CONTEXT_SCHEMA, "study": "sage-mem-v1",
        "stage": "formal-raw-context-reference",
        "status": "complete-label-free", "cohort": cohort, "seed": 0,
        "ages": list(contract.ages), "short_context_frames": 3,
        "long_context_frames": 16,
        "separate_from_parameter_matched_arms": True,
        "formal_test_labels_read": False,
        "development_outcomes_read": False,
        "bank_manifest_sha256": "b" * 64,
        "host_hash_before": "a" * 64, "host_hash_after": "a" * 64,
        "consumer_contract": "post-reveal-shared-short-long-arm-blind",
        "shared_consumer_sha256": None,
        "feature_contract": reproduction.RAW_FEATURE_CONTRACT,
        "artifact": _handle(raw_artifact),
    })
    raw_records = [{
        "cohort": cohort, "seed": 0,
        "manifest_sha256": _sha(raw_manifest),
        "artifact_sha256": _sha(raw_artifact),
        "bank_manifest_sha256": "b" * 64,
    }]
    raw_summary = raw / "summary.json"
    _json(raw_summary, {
        "schema": "sage_mem_v1_raw_context_producer_v1",
        "study": "sage-mem-v1", "status": "complete-label-free",
        "cells": 1, "cohorts": [cohort], "seeds": [0],
        "feature_contract": reproduction.RAW_FEATURE_CONTRACT,
        "formal_labels_read": False, "development_outcomes_read": False,
        "mse_emitted": False,
        "records_sha256": hashlib.sha256(
            _canonical(raw_records).encode()).hexdigest(),
    })

    execution_root = workspace / "execution"
    execution_root.mkdir()
    cube = np.zeros((8, 4, 4), dtype=np.uint8)
    diagonal = np.arange(4)
    cube[:, diagonal, diagonal] = 1
    execution_artifact = execution_root / "deck.npz"
    np.savez_compressed(
        execution_artifact,
        formal_test_episode_id=formal_identity[0][0],
        formal_test_native_cluster_id=formal_identity[1][0],
        selected_class_by_true_target_success=cube,
        deterministic_random_class=np.zeros(8, dtype=np.int64))
    replay = execution_root / "replay.json"
    _json(replay, {
        "schema": finalizer.EXECUTION_REPLAY_RECEIPT_SCHEMA,
        "study": "sage-mem-v1", "status": "sealed-label-free",
        "cohort": cohort, "bank_manifest_sha256": "b" * 64,
        "formal_labels_read": False, "development_outcomes_read": False,
        "controller_identity_sha256": "1" * 64, "rows": 8,
        "classes": 4, "native_clusters": 8, "executions": 32,
        "replayed_executions": 32, "deterministic_replay_fidelity": 1.0,
        "execution_endpoint": "synthetic-fixed-endpoint",
    })
    execution_registry = execution_root / "registry.json"
    _json(execution_registry, {
        "schema": finalizer.EXECUTION_DECK_REGISTRY_SCHEMA,
        "study": "sage-mem-v1", "status": "sealed",
        "available_only_after_complete_phase_a_grid": True,
        "development_outcomes_read": False,
        "cohorts": {cohort: {
            "bank_manifest_sha256": "b" * 64, "classes": 4,
            "controller": {
                "controller_identity_sha256": "1" * 64,
                "implementation_sha256": "2" * 64,
                "physics_sha256": "3" * 64, "pinned": True,
                "arm_identity_input": False,
                "input": "predicted_class_only",
            },
            "eligibility_gate": {
                "metric": "mean_oracle_success", "operator": ">=",
                "threshold": 0.9, "preregistered": True,
            },
            "artifact": _handle(execution_artifact),
            "replay_receipt": _handle(replay),
        }},
        "unavailable_cohorts": {},
    })

    finalized = workspace / "finalized"
    summary = finalizer._finalize_with_contract(
        phase, label_registry, finalized, final_contract,
        raw_context_root=raw,
        execution_deck_registry=execution_registry)

    report = {
        "schema": reproduction.FORMAL_REPORT_SCHEMA,
        "study": "sage-mem-v1", "stage": "formal-evidence-audit",
        "status": "complete", "phase_a_cells_verified": 2,
        "finalized_cells_verified": 2,
        "phase_a_grid_sha256": summary["phase_a_grid_sha256"],
        "raw_context_references_verified": 1,
        "synthetic_payload": {"kept": "byte-exact"},
    }
    report_path = workspace / "report.json"
    _json(report_path, report)
    prepare = workspace / "prepare"
    prepare.mkdir()

    source = workspace / "source.txt"
    source.write_text("locked source\n", encoding="utf-8")
    locked_scripts: dict[str, Path] = {}
    locked_payloads = {
        "scripts/sage_mem_v1_formal_finalizer.py": "# locked fixture\n",
        "scripts/sage_mem_v1_spec.py": (
            "def load_spec(path, verify_parent_paths=False):\n"
            "    return {'fixture': True}\n"),
        "scripts/audit_sage_mem_v1_formal.py": (
            "def audit_formal_evidence(**kwargs):\n"
            "    return {'isolated': True, 'source': 'locked'}\n"),
    }
    for relative, payload in locked_payloads.items():
        locked_path = workspace / relative
        locked_path.parent.mkdir(parents=True, exist_ok=True)
        locked_path.write_text(payload, encoding="utf-8")
        locked_scripts[relative] = locked_path
    amendment = workspace / "amendment.yaml"
    amendment.write_text("locked: true\n", encoding="utf-8")
    development = workspace / "development.json"
    _json(development, {"locked": True})
    preflight = workspace / "preflight.json"
    _json(preflight, {"locked": True})
    protocol_lock = workspace / "protocol_lock.json"
    _json(protocol_lock, {
        "development_audit": _lock_record(development, workspace),
        "formal_amendment": {
            **_lock_record(amendment, workspace),
            "status": "locked-before-development-selection-or-formal-data",
        },
        "formal_execution_started": False,
        "integration_identities": {
            "host_adapter": _lock_record(source, workspace),
            "model": _lock_record(source, workspace),
        },
        "preselection_source_receipt": _lock_record(preflight, workspace),
        "producer_identities": {
            "configs/sage_mem_v1.yaml":
                _lock_record(spec_path, workspace, include_path=False),
            "source.txt": _lock_record(source, workspace, include_path=False),
            **{relative: _lock_record(path, workspace, include_path=False)
               for relative, path in locked_scripts.items()},
        },
        "protocol_fingerprint": fingerprint, "schema_version": 1,
        "seed_registry": {"fixture/seed": 7},
        "spec_sha256": _sha(spec_path), "stage": "seal",
        "status": "sealed", "study": "sage-mem-v1",
    })

    paths = reproduction.InputPaths(
        protocol_lock=protocol_lock.relative_to(workspace),
        phase_a_root=phase.relative_to(workspace),
        raw_context_root=raw.relative_to(workspace),
        label_registry=label_registry.relative_to(workspace),
        execution_registry=execution_registry.relative_to(workspace),
        finalized_root=finalized.relative_to(workspace),
        prepare_root=prepare.relative_to(workspace),
        formal_report=report_path.relative_to(workspace))
    verifier_source = workspace / "verifier.py"
    verifier_source.write_bytes(Path(reproduction.__file__).read_bytes())
    expected = reproduction.ExpectedHashes(
        verifier_source=_sha(verifier_source),
        protocol_lock=_sha(protocol_lock),
        phase_a_grid=summary["phase_a_grid_sha256"],
        raw_context_summary=_sha(raw_summary),
        label_registry=_sha(label_registry),
        execution_registry=_sha(execution_registry),
        finalizer_summary=_sha(finalized / "summary.json"),
        finalized_cells=summary["finalized_cells_sha256"],
        formal_report=_sha(report_path))
    return Fixture(workspace, contract, paths, expected, report,
                   verifier_source.relative_to(workspace))


def _run(fixture: Fixture, output: str = "receipt.json") -> dict[str, Any]:
    return reproduction.audit_phase_b_reproduction(
        workspace=fixture.workspace, paths=fixture.paths,
        expected=fixture.expected, output=output,
        contract=fixture.contract,
        _report_reproducer=lambda: fixture.report,
        _verifier_source=fixture.verifier_source)


def _rehash_finalized_summary(fixture: Fixture) -> None:
    root = fixture.workspace / fixture.paths.finalized_root
    records = []
    for cohort in fixture.contract.cohorts:
        for arm in fixture.contract.arms:
            for seed in fixture.contract.seeds:
                manifest = json.loads((root / "cells" / cohort / arm
                                       / f"seed-{seed}" /
                                       "manifest.json").read_text())
                records.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "artifact_sha256": manifest["artifact"]["sha256"],
                    "consumer_sha256": manifest[
                        "shared_arm_blind_consumer_sha256"],
                })
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["finalized_cells_sha256"] = hashlib.sha256(
        _canonical(records).encode()).hexdigest()
    _json(summary_path, summary)
    fixture.expected = replace(
        fixture.expected, finalizer_summary=_sha(summary_path),
        finalized_cells=summary["finalized_cells_sha256"])


def test_success_is_value_free_and_receipt_is_deterministic(
        tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    first = _run(fixture, "receipt-a.json")
    second = _run(fixture, "receipt-b.json")
    assert first == second
    assert (fixture.workspace / "receipt-a.json").read_bytes() == \
        (fixture.workspace / "receipt-b.json").read_bytes()
    assert first["authenticated_inventories"]["phase_a_cells"] == 2
    assert first["independent_reproduction"]["all_arrays_exact"] is True
    assert first["independent_reproduction"][
        "formal_report_byte_exact"] is True
    assert first["production_contract_verified"] is False
    assert first["report_reproducer_injected"] is True
    assert first["verifier_source_injected"] is True
    assert first["contract_identity_sha256"] != first[
        "registered_contract_sha256"]
    assert first["authenticated_inventories"]["verifier_source"][
        "sha256"] == fixture.expected.verifier_source
    environment = first["authenticated_inventories"][
        "numerical_environment"]
    assert environment["numpy_version"] == np.__version__
    assert "blas_thread_environment" in environment
    assert "normalized_threadpools" in environment
    assert len(environment["python_executable_identity"]["sha256"]) == 64
    assert set(environment["module_source_identities"]) == {
        "numpy", "scipy", "sklearn", "threadpoolctl"}
    assert set(environment["distribution_record_identities"]) == {
        "numpy", "scipy", "scikit-learn", "threadpoolctl"}
    assert environment["loaded_extension_identities"]
    assert "operator_pins" in first and "semantic_digests" in first
    assert not any(key in first for key in ("accuracy", "effect", "interval"))


def test_preview_performs_no_filesystem_reads(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) \
        -> None:
    monkeypatch.setattr(reproduction, "_read_json",
                        lambda *args, **kwargs: pytest.fail("read JSON"))
    monkeypatch.setattr(reproduction, "_sha256_file",
                        lambda *args, **kwargs: pytest.fail("hash file"))
    assert reproduction.main([]) == 0
    assert '"filesystem_reads":0' in capsys.readouterr().out


def test_tampered_prediction_fails_after_self_consistent_rehash(
        tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    root = fixture.workspace / fixture.paths.finalized_root
    directory = root / "cells/lewm_reacher_color/none/seed-0"
    artifact = directory / "finalized_results.npz"
    with np.load(artifact, allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]).copy() for name in z.files}
    arrays["formal_test_full_pred"][0, 0] = (
        arrays["formal_test_full_pred"][0, 0] + 1) % 4
    arrays["formal_test_full_correct"] = (
        arrays["formal_test_full_pred"] ==
        arrays["formal_test_label"]).astype(np.uint8)
    np.savez_compressed(artifact, **arrays)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact"] = _handle(artifact)
    _json(manifest_path, manifest)
    _rehash_finalized_summary(fixture)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="independent prediction mismatch"):
        _run(fixture)


@pytest.mark.parametrize("target", ["raw", "execution"])
def test_raw_or_execution_tamper_fails_even_when_manifests_are_rehashed(
        tmp_path: Path, target: str) -> None:
    fixture = _make_fixture(tmp_path)
    root = fixture.workspace / fixture.paths.finalized_root
    if target == "raw":
        directory = root / "raw_context/lewm_reacher_color/seed-0"
        artifact = directory / "finalized_results.npz"
        with np.load(artifact, allow_pickle=False) as z:
            arrays = {name: np.asarray(z[name]).copy() for name in z.files}
        arrays["formal_test_short_pred"][0, 0] = (
            arrays["formal_test_short_pred"][0, 0] + 1) % 4
        arrays["formal_test_short_correct"] = (
            arrays["formal_test_short_pred"] ==
            arrays["formal_test_label"]).astype(np.uint8)
        np.savez_compressed(artifact, **arrays)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact"] = _handle(artifact)
        _json(manifest_path, manifest)
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        records = [{
            "cohort": "lewm_reacher_color", "seed": 0,
            "artifact_sha256": _sha(artifact),
            "consumer_sha256": manifest[
                "shared_arm_blind_consumer_sha256"],
        }]
        summary["raw_context_reference"]["records_sha256"] = hashlib.sha256(
            _canonical(records).encode()).hexdigest()
        _json(summary_path, summary)
        fixture.expected = replace(
            fixture.expected, finalizer_summary=_sha(summary_path))
        expected_message = "raw-context prediction mismatch"
    else:
        directory = root / "cells/lewm_reacher_color/none/seed-0"
        artifact = directory / "finalized_results.npz"
        with np.load(artifact, allow_pickle=False) as z:
            arrays = {name: np.asarray(z[name]).copy() for name in z.files}
        values = arrays["formal_test_full_execution_success"]
        values[0, 0] = 1 - values[0, 0]
        np.savez_compressed(artifact, **arrays)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact"] = _handle(artifact)
        manifest["execution"]["per_age_success"]["full"] = list(map(
            float, np.mean(values, axis=1)))
        _json(manifest_path, manifest)
        _rehash_finalized_summary(fixture)
        expected_message = "execution reproduction mismatch"
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match=expected_message):
        _run(fixture)


def test_missing_registry_and_partial_finalization_fail_closed(
        tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    missing = replace(fixture.paths, label_registry=Path("missing.json"))
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="label registry is missing"):
        reproduction.audit_phase_b_reproduction(
            workspace=fixture.workspace, paths=missing,
            expected=fixture.expected, output="receipt.json",
            contract=fixture.contract,
            _report_reproducer=lambda: fixture.report,
            _verifier_source=fixture.verifier_source)
    cell = (fixture.workspace / fixture.paths.finalized_root /
            "cells/lewm_reacher_color/sage_mem_full/seed-0")
    shutil.rmtree(cell)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="finalized seed inventory differs"):
        _run(fixture)


def test_stale_output_path_traversal_and_symlink_fail_closed(
        tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.workspace / "receipt.json").write_text("partial")
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="stale or partial"):
        _run(fixture)
    (fixture.workspace / "receipt.json").unlink()
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="path traversal"):
        reproduction.audit_phase_b_reproduction(
            workspace=fixture.workspace,
            paths=replace(fixture.paths,
                          label_registry=Path("custody/../custody/registry.json")),
            expected=fixture.expected, output="receipt.json",
            contract=fixture.contract,
            _report_reproducer=lambda: fixture.report,
            _verifier_source=fixture.verifier_source)
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(fixture.workspace, target_is_directory=True)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="workspace path contains a symlink"):
        reproduction.audit_phase_b_reproduction(
            workspace=workspace_link, paths=fixture.paths,
            expected=fixture.expected, output="receipt.json",
            contract=fixture.contract,
            _report_reproducer=lambda: fixture.report,
            _verifier_source=fixture.verifier_source)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="overlaps an authenticated input"):
        reproduction.audit_phase_b_reproduction(
            workspace=fixture.workspace, paths=fixture.paths,
            expected=fixture.expected,
            output=fixture.paths.finalized_root / "receipt.json",
            contract=fixture.contract,
            _report_reproducer=lambda: fixture.report,
            _verifier_source=fixture.verifier_source)
    real = fixture.workspace / fixture.paths.label_registry
    link = fixture.workspace / "registry-link.json"
    link.symlink_to(real)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="symlink"):
        reproduction.audit_phase_b_reproduction(
            workspace=fixture.workspace,
            paths=replace(fixture.paths,
                          label_registry=Path("registry-link.json")),
            expected=fixture.expected, output="receipt.json",
            contract=fixture.contract,
            _report_reproducer=lambda: fixture.report,
            _verifier_source=fixture.verifier_source)


def test_tampered_locked_source_and_missing_operator_hash_fail_closed(
        tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.workspace / "source.txt").write_text("tampered\n")
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="(size|hash) differs from protocol lock"):
        _run(fixture)
    fixture = _make_fixture(tmp_path / "second")
    fixture.expected = replace(fixture.expected, formal_report="")
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="operator must supply"):
        _run(fixture)


def test_verifier_source_is_operator_pinned(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.workspace / fixture.verifier_source).write_text(
        "# changed verifier\n", encoding="utf-8")
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="verifier source differs from operator pin"):
        _run(fixture)


def test_internal_manifest_symlink_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = (fixture.workspace / fixture.paths.finalized_root /
                "cells/lewm_reacher_color/none/seed-0/manifest.json")
    real = manifest.with_name("manifest.real.json")
    manifest.rename(real)
    manifest.symlink_to(real.name)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="finalized manifest is absent or unsafe"):
        _run(fixture)


@pytest.mark.parametrize("relative", [
    "raw/lewm_reacher_color/seed-0/manifest.json",
    "finalized/raw_context_consumers/lewm_reacher_color/seed-0.json",
    "finalized/summary.json",
])
def test_every_internal_json_input_rejects_symlinks(
        tmp_path: Path, relative: str) -> None:
    fixture = _make_fixture(tmp_path)
    path = fixture.workspace / relative
    real = fixture.workspace / ("symlink-target-" +
                                hashlib.sha256(relative.encode()).hexdigest())
    path.rename(real)
    path.symlink_to(real)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="symlink|unsafe|inventory differs"):
        _run(fixture)


@pytest.mark.parametrize("relative", [
    "finalized/cells/lewm_reacher_color/none",
    "finalized/consumers/lewm_reacher_color",
    "finalized/raw_context/lewm_reacher_color/seed-0",
    "finalized/raw_context_consumers/lewm_reacher_color",
])
def test_every_nested_finalized_directory_rejects_symlinks(
        tmp_path: Path, relative: str) -> None:
    fixture = _make_fixture(tmp_path)
    path = fixture.workspace / relative
    target = fixture.workspace / ("directory-target-" +
                                  hashlib.sha256(relative.encode()).hexdigest())
    path.rename(target)
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="inventory differs"):
        _run(fixture)


def test_duplicate_and_noncanonical_json_are_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path / "duplicate")
    lock = fixture.workspace / fixture.paths.protocol_lock
    text = lock.read_text().rstrip()
    lock.write_text(text[:-1] + ',"study":"sage-mem-v1"}\n',
                    encoding="utf-8")
    fixture.expected = replace(fixture.expected, protocol_lock=_sha(lock))
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="duplicate JSON key"):
        _run(fixture)

    fixture = _make_fixture(tmp_path / "noncanonical")
    report = fixture.workspace / fixture.paths.formal_report
    report.write_text(json.dumps(fixture.report, indent=2, sort_keys=True)
                      + "\n", encoding="utf-8")
    fixture.expected = replace(fixture.expected, formal_report=_sha(report))
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="not canonical JSON"):
        _run(fixture)


def test_production_lock_must_bind_every_report_replay_source(
        tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    lock_path = fixture.workspace / fixture.paths.protocol_lock
    lock = json.loads(lock_path.read_text())
    del lock["producer_identities"]["scripts/audit_sage_mem_v1_formal.py"]
    _json(lock_path, lock)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="formal-auditor"):
        reproduction._authenticate_protocol_lock(
            lock_path, _sha(lock_path), fixture.workspace,
            reproduction.PRODUCTION_CONTRACT)


def test_exact_production_contract_cannot_be_downgraded_for_hooks() -> None:
    downgraded = replace(reproduction.PRODUCTION_CONTRACT,
                         require_600=False)
    assert downgraded.is_registered_production is True
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="cannot be downgraded"):
        downgraded.validate()
    bool_gpu = dict(reproduction.PHYSICAL_GPUS)
    bool_gpu["lewm_reacher_color"] = False
    substituted = replace(reproduction.PRODUCTION_CONTRACT,
                          physical_gpus=bool_gpu)
    assert substituted.is_registered_production is False
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="GPU ownership"):
        substituted.validate()


def test_isolated_report_replay_ignores_correct_path_in_memory_monkeypatch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _make_fixture(tmp_path)
    lock = json.loads((fixture.workspace /
                       fixture.paths.protocol_lock).read_text())
    fake = types.ModuleType("scripts.audit_sage_mem_v1_formal")
    fake.__file__ = str(fixture.workspace /
                        "scripts/audit_sage_mem_v1_formal.py")
    fake.audit_formal_evidence = lambda **kwargs: {"forged": True}
    monkeypatch.setitem(sys.modules, "scripts.audit_sage_mem_v1_formal", fake)
    malicious = fixture.workspace / "malicious-pythonpath"
    malicious.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(malicious))
    replay = reproduction._run_locked_report_subprocess(
        workspace=fixture.workspace,
        producers=lock["producer_identities"],
        spec_path=fixture.workspace / "configs/sage_mem_v1.yaml",
        phase_root=fixture.workspace / "phase",
        finalized_root=fixture.workspace / "finalized",
        prepare_root=fixture.workspace / "prepare",
        raw_root=fixture.workspace / "raw")
    assert replay == b'{"isolated":true,"source":"locked"}\n'


def test_direct_cli_preview_runs_from_outside_repository(tmp_path: Path) \
        -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(reproduction.__file__).resolve()),
         "--phase-a-root", "/definitely/missing"],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert completed.returncode == 0
    assert '"preview":true' in completed.stdout
    assert '"filesystem_reads":0' in completed.stdout


def test_stable_session_rejects_path_replacement_between_validation_and_fit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _make_fixture(tmp_path)
    measurement = (fixture.workspace / "phase/cells/lewm_reacher_color/"
                   "none/seed-0/measurements.npz")
    original_loader = reproduction._load_labels

    def replacing_loader(*args: Any, **kwargs: Any):
        result = original_loader(*args, **kwargs)
        held = fixture.workspace / "held-original-measurements.npz"
        measurement.rename(held)
        shutil.copy2(held, measurement)
        return result

    monkeypatch.setattr(reproduction, "_load_labels", replacing_loader)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="(stable artifact|inventory directory).*changed "
                       "during audit"):
        _run(fixture)
    assert not (fixture.workspace / "receipt.json").exists()


def test_stable_session_rejects_in_place_mutation_before_publication(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _make_fixture(tmp_path)
    measurement = (fixture.workspace / "phase/cells/lewm_reacher_color/"
                   "none/seed-0/measurements.npz")
    original_replay = reproduction._replay_formal_report

    def mutating_replay(*args: Any, **kwargs: Any):
        result = original_replay(*args, **kwargs)
        with measurement.open("ab") as stream:
            stream.write(b"concurrent-mutation")
            stream.flush()
            os.fsync(stream.fileno())
        return result

    monkeypatch.setattr(reproduction, "_replay_formal_report",
                        mutating_replay)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="stable artifact metadata changed"):
        _run(fixture)
    assert not (fixture.workspace / "receipt.json").exists()


def test_stable_session_rejects_late_directory_inventory_mutation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _make_fixture(tmp_path)
    tracked_directory = (fixture.workspace /
                         "phase/cells/lewm_reacher_color/none/seed-0")
    original_replay = reproduction._replay_formal_report

    def mutating_replay(*args: Any, **kwargs: Any):
        result = original_replay(*args, **kwargs)
        (tracked_directory / "late-unregistered-file").write_bytes(b"late")
        return result

    monkeypatch.setattr(reproduction, "_replay_formal_report",
                        mutating_replay)
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="inventory directory changed during audit"):
        _run(fixture)
    assert not (fixture.workspace / "receipt.json").exists()


def test_receipt_fsyncs_parent_directory_and_is_read_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _make_fixture(tmp_path)
    original_fsync = os.fsync
    directory_syncs = 0

    def recording_fsync(fd: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_syncs += 1
        original_fsync(fd)

    monkeypatch.setattr(reproduction.os, "fsync", recording_fsync)
    _run(fixture)
    receipt = fixture.workspace / "receipt.json"
    assert directory_syncs >= 1
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o400


def test_cli_default_phase_root_is_study_root() -> None:
    args = reproduction._parse_args([])
    assert args.phase_a_root == Path("outputs/sage_mem_v1")
    assert reproduction.PRODUCTION_CONTRACT.total_cells == 600
    assert reproduction.PRODUCTION_CONTRACT.raw_references == 50


def test_report_replay_must_be_byte_exact(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    altered = dict(fixture.report)
    altered["synthetic_payload"] = {"changed": True}
    with pytest.raises(reproduction.PhaseBReproductionError,
                       match="byte-for-byte identical"):
        reproduction.audit_phase_b_reproduction(
            workspace=fixture.workspace, paths=fixture.paths,
            expected=fixture.expected, output="receipt.json",
            contract=fixture.contract,
            _report_reproducer=lambda: altered,
            _verifier_source=fixture.verifier_source)
