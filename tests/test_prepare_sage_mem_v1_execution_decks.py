"""Fail-closed tests for the pre-reveal SAGE-Mem execution producer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from scripts import prepare_sage_mem_v1_execution_decks as producer


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/sage_mem_v1.yaml"


def _banks(root: Path) -> Path:
    preparation = root / "formal_preparation"
    for cohort in producer.COHORTS:
        manifest = preparation / "banks" / cohort / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({
            "schema": "test-label-free-bank",
            "cohort": cohort,
            "semantic_labels_inside": False,
        }, sort_keys=True) + "\n")
    # A tempting vault path is deliberately present.  No producer function
    # accepts or opens it.
    vault = preparation / "vaults" / "forbidden.npz"
    vault.parent.mkdir()
    vault.write_bytes(b"must-not-open")
    os.chmod(vault, 0)
    return preparation


def _controller(tag: str) -> dict[str, Any]:
    digit = str((sum(tag.encode()) % 8) + 1)
    return {
        "controller_identity_sha256": digit * 64,
        "implementation_sha256": "a" * 64,
        "physics_sha256": "b" * 64,
        "pinned": True,
        "arm_identity_input": False,
        "input": "predicted_class_only",
    }


def _builder(cohort: str) -> producer.ReplayBuilder:
    def build(bank_root: Path, spec: Mapping[str, Any], progress) \
            -> producer.ReplayProduct:
        del progress
        classes = int(spec["cohorts"][cohort]["classes"])
        count = producer.FORMAL_TEST_ROWS[cohort]
        variants = producer.VARIANTS_PER_NATIVE_CLUSTER[cohort]
        episode = np.arange(10_000, 10_000 + count, dtype=np.int64)
        cluster = np.repeat(
            np.arange(20_000, 20_000 + count // variants, dtype=np.int64),
            variants)
        cube = np.zeros((count, classes, classes), dtype=np.uint8)
        diagonal = np.arange(classes)
        cube[:, diagonal, diagonal] = 1
        random = producer._deterministic_random_class(
            cohort, cluster, classes)
        return producer.ReplayProduct(
            cohort=cohort,
            bank_manifest_sha256=producer._sha256_file(
                bank_root / "manifest.json"),
            episode_id=episode,
            native_cluster_id=cluster,
            success_cube=cube,
            random_class=random,
            controller=_controller(cohort),
            executions=(count // variants) * classes,
            replayed_executions=(count // variants) * classes,
            replay_fidelity=1.0,
            execution_endpoint="fixed-test-endpoint",
        )
    return build


def _builders() -> dict[str, producer.ReplayBuilder]:
    return {cohort: _builder(cohort) for cohort in producer.SUPPORTED}


def test_preview_is_no_write_and_reports_missing_banks(tmp_path: Path) -> None:
    output = tmp_path / "execution"
    preview = producer.preview_execution_decks(
        SPEC, tmp_path / "missing", output)
    assert preview["formal_labels_read"] is False
    assert preview["development_outcomes_read"] is False
    assert {value["status"] for value in preview["cohorts"].values()} == {
        "blocked"}
    assert not output.exists()


def test_execute_seals_three_cubes_and_two_explicit_unavailable_receipts(
        tmp_path: Path) -> None:
    preparation = _banks(tmp_path)
    output = tmp_path / "execution"
    summary = producer.produce_execution_decks(
        SPEC, preparation, output, builders=_builders(), progress=lambda _: None)
    assert summary["status"] == "sealed-label-free"
    assert summary["supported_cohorts"] == sorted(producer.SUPPORTED)
    assert summary["unavailable_cohorts"] == sorted(producer.UNAVAILABLE)
    assert oct(output.stat().st_mode & 0o777) == "0o555"

    registry = producer.validate_published_registry(output, SPEC, preparation)
    assert set(registry["cohorts"]) == set(producer.SUPPORTED)
    assert set(registry["unavailable_cohorts"]) == set(producer.UNAVAILABLE)
    for cohort, record in registry["cohorts"].items():
        artifact = output / record["artifact"]["path"]
        with np.load(artifact, allow_pickle=False) as archive:
            assert "selected_class_by_true_target_success" in archive.files
            assert "class_conditioned_success" not in archive.files
            assert "oracle_success" not in archive.files
            assert "random_success" not in archive.files
            cube = archive["selected_class_by_true_target_success"]
            assert cube.ndim == 3


def test_resume_revalidates_without_invoking_replay(tmp_path: Path) -> None:
    preparation = _banks(tmp_path)
    output = tmp_path / "execution"
    producer.produce_execution_decks(
        SPEC, preparation, output, builders=_builders(), progress=lambda _: None)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resume must not rerun physics")

    resumed = producer.produce_execution_decks(
        SPEC, preparation, output, resume=True,
        builders={cohort: forbidden for cohort in producer.SUPPORTED},
        progress=lambda _: None)
    assert resumed["status"] == "validated-existing"


def test_tampered_cube_is_rejected_and_not_rebuilt(tmp_path: Path) -> None:
    preparation = _banks(tmp_path)
    output = tmp_path / "execution"
    producer.produce_execution_decks(
        SPEC, preparation, output, builders=_builders(), progress=lambda _: None)
    registry = json.loads((output / "registry.json").read_text())
    artifact = output / registry["cohorts"][
        "lewm_reacher_color"]["artifact"]["path"]
    os.chmod(output, 0o755)
    os.chmod(artifact.parent, 0o755)
    os.chmod(artifact, 0o644)
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    os.chmod(artifact, 0o444)
    os.chmod(artifact.parent, 0o555)
    os.chmod(output, 0o555)
    with pytest.raises(producer.ExecutionDeckProducerError,
                       match="artifact identity differs"):
        producer.produce_execution_decks(
            SPEC, preparation, output, resume=True,
            builders=_builders(), progress=lambda _: None)


def test_fewer_than_two_exact_replays_fails_without_publication(
        tmp_path: Path) -> None:
    preparation = _banks(tmp_path)
    output = tmp_path / "execution"
    with pytest.raises(producer.ExecutionDeckProducerError,
                       match="fewer than two cohorts"):
        producer.produce_execution_decks(
            SPEC, preparation, output,
            builders={"lewm_reacher_color": _builder(
                "lewm_reacher_color")}, progress=lambda _: None)
    assert not output.exists()
    assert not list(tmp_path.glob(".execution.*.staging"))


def test_cluster_variant_cube_or_random_drift_is_rejected(
        tmp_path: Path) -> None:
    preparation = _banks(tmp_path)
    output = tmp_path / "execution"
    builders = _builders()

    def drifting(bank_root: Path, spec: Mapping[str, Any], progress):
        product = _builder("dinowm_pointmaze_goal")(
            bank_root, spec, progress)
        cube = product.success_cube.copy()
        cube[1, 0, 0] ^= 1
        return producer.ReplayProduct(
            **{**product.__dict__, "success_cube": cube})

    builders["dinowm_pointmaze_goal"] = drifting
    with pytest.raises(producer.ExecutionDeckProducerError,
                       match="varies inside native cluster"):
        producer.produce_execution_decks(
            SPEC, preparation, output, builders=builders,
            progress=lambda _: None)
    assert not output.exists()
