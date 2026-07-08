from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

import scripts.prepare_sage_mem_v1_formal as prepare


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _fixture(tmp_path: Path) -> tuple[dict[str, Any],
                                      prepare.PreparationLayout]:
    spec_path = tmp_path / "sage.yaml"
    _write(spec_path, b"synthetic sealed study\n")
    protocol_lock = tmp_path / "protocol.lock.json"
    _write(protocol_lock, b"synthetic implementation lock\n")
    comparator_root = tmp_path / "comparators"
    for cohort in prepare.COHORTS:
        _write(comparator_root / cohort / "receipt.json",
               f"opaque comparator {cohort}\n".encode())
    spec = {
        "_spec_path": str(spec_path),
        "_spec_sha256": prepare._sha256_file(spec_path),  # noqa: SLF001
        "implementation_lock": "unused-under-test-override",
        "execution": {"output_root": "outputs/sage_mem_v1"},
        "cohorts": {
            "lewm_reacher_color": {
                "parent_protocol": "configs/paper_a_matched_color_v1_1.yaml",
                "split_episodes": {"formal_train": 16,
                                   "consumer_train": 16,
                                   "formal_test": 16}},
            "lewm_pusht_color": {
                "parent_protocol": "configs/paper_a_matched_color_v1_1.yaml",
                "split_episodes": {"formal_train": 16,
                                   "consumer_train": 16,
                                   "formal_test": 16}},
            "dinowm_pusht_token": {
                "parent_protocol":
                    "configs/dinowm_wave2_spatial_carrier_v1_1.yaml"},
            "dinowm_pusht_binding": {
                "parent_protocol":
                    "configs/dinowm_wave2_spatial_carrier_v1_1.yaml"},
            "dinowm_pointmaze_goal": {
                "parent_protocol": "configs/dinowm_pointmaze_wave3.yaml"},
        },
    }
    layout = prepare.layout_from_spec(
        spec,
        preparation_root=tmp_path / "formal_preparation",
        comparator_root=comparator_root,
        protocol_lock=protocol_lock,
    )
    layout.ensure_directories()
    return spec, layout


def _fake_materialize(cohort: str, layout: prepare.PreparationLayout) -> None:
    bank = layout.bank(cohort)
    bank.mkdir(parents=True)
    _write(bank / "payload.bin", f"label-free {cohort}".encode())
    (bank / "manifest.json").write_text(json.dumps({
        "cohort": cohort,
        "payload_sha256": prepare._sha256_file(  # noqa: SLF001
            bank / "payload.bin"),
    }))
    vault = layout.vault(cohort)
    vault.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        vault,
        episode_id=np.arange(8, dtype=np.int64),
        native_cluster_id=np.arange(8, dtype=np.int64),
        class_id=np.arange(8, dtype=np.int64) % prepare.CLASSES[cohort],
    )
    os.chmod(vault, 0o400)
    custody = layout.custody_receipt(cohort)
    custody.parent.mkdir(parents=True, exist_ok=True)
    custody.write_text(json.dumps({
        "cohort": cohort,
        "vault_sha256": prepare._sha256_file(vault),  # noqa: SLF001
    }))
    os.chmod(custody, 0o400)


def _fake_validate(cohort: str, layout: prepare.PreparationLayout,
                   *, plan_sha256: str = "f" * 64
                   ) -> prepare.PreparedEvidence:
    bank_manifest = layout.bank(cohort) / "manifest.json"
    vault = layout.vault(cohort)
    custody = layout.custody_receipt(cohort)
    assert bank_manifest.is_file() and vault.is_file() and custody.is_file()
    vault_hash = prepare._sha256_file(vault)  # noqa: SLF001
    relative = vault.resolve().relative_to(layout.custody.resolve())
    artifact = {
        "path": str(relative), "sha256": vault_hash,
        "size": vault.stat().st_size,
    }
    source = {
        "artifact": artifact,
        "keys": {"episode_id": "episode_id",
                 "native_cluster_id": "native_cluster_id",
                 "label": "class_id"},
    }
    bank_identity = prepare._identity(  # noqa: SLF001
        bank_manifest, relative_to=layout.root)
    return prepare.PreparedEvidence(
        cohort=cohort,
        bank_manifest=bank_identity,
        custody_receipt=prepare._identity(  # noqa: SLF001
            custody, relative_to=layout.root),
        vault_sha256=vault_hash,
        custody_record={
            "bank_manifest_sha256": bank_identity["sha256"],
            "classes": prepare.CLASSES[cohort],
            "sources": {"formal_test": source, "consumer_train": source},
        },
        backend_proof={"backend": "fake", "plan_sha256": plan_sha256},
    )


def test_cohort_preparation_is_resumable_and_rehashes_everything(
        tmp_path: Path) -> None:
    spec, layout = _fixture(tmp_path)
    cohort = prepare.COHORTS[0]
    calls = {"materialize": 0, "validate": 0}

    def materialize() -> None:
        calls["materialize"] += 1
        _fake_materialize(cohort, layout)

    def validate() -> prepare.PreparedEvidence:
        calls["validate"] += 1
        return _fake_validate(cohort, layout)

    first = prepare._ensure_cohort(  # noqa: SLF001
        spec=spec, cohort=cohort, layout=layout,
        materialize=materialize, validate=validate)
    assert first.cohort == cohort
    assert calls == {"materialize": 1, "validate": 1}
    receipt = layout.cohort_receipt(cohort)
    assert prepare._mode(receipt) == "0o444"  # noqa: SLF001
    assert prepare._mode(layout.bank(cohort)) == "0o555"  # noqa: SLF001

    second = prepare._ensure_cohort(  # noqa: SLF001
        spec=spec, cohort=cohort, layout=layout,
        materialize=lambda: pytest.fail("resume rematerialized the bank"),
        validate=validate)
    assert second.bank_manifest == first.bank_manifest
    assert calls["materialize"] == 1

    manifest = layout.bank(cohort) / "manifest.json"
    os.chmod(layout.bank(cohort), 0o755)
    os.chmod(manifest, 0o644)
    manifest.write_text(json.dumps({"cohort": cohort, "tampered": True}))
    with pytest.raises(prepare.FormalPreparationError,
                       match="receipt differs"):
        prepare._ensure_cohort(  # noqa: SLF001
            spec=spec, cohort=cohort, layout=layout,
            materialize=lambda: pytest.fail("tamper caused rematerialization"),
            validate=validate)


def test_partial_core_fails_closed_without_deletion_or_rebuild(
        tmp_path: Path) -> None:
    spec, layout = _fixture(tmp_path)
    cohort = prepare.COHORTS[1]
    layout.bank(cohort).mkdir(parents=True)
    with pytest.raises(prepare.FormalPreparationError, match="partial"):
        prepare._ensure_cohort(  # noqa: SLF001
            spec=spec, cohort=cohort, layout=layout,
            materialize=lambda: pytest.fail("partial state was overwritten"),
            validate=lambda: pytest.fail("partial state was trusted"))
    assert layout.bank(cohort).exists()


def test_cohort_partial_check_does_not_race_another_gpu_worker(
        tmp_path: Path) -> None:
    spec, layout = _fixture(tmp_path)
    cohort = prepare.COHORTS[0]
    other = prepare.COHORTS[2]
    (layout.banks / f".{other}.partial-live-worker").mkdir()
    prepare._ensure_cohort(  # noqa: SLF001
        spec=spec, cohort=cohort, layout=layout,
        materialize=lambda: _fake_materialize(cohort, layout),
        validate=lambda: _fake_validate(cohort, layout))
    own_partial = layout.banks / f".{prepare.COHORTS[1]}.partial-dead"
    own_partial.mkdir()
    with pytest.raises(prepare.FormalPreparationError,
                       match="artifacts exist"):
        prepare._ensure_cohort(  # noqa: SLF001
            spec=spec, cohort=prepare.COHORTS[1], layout=layout,
            materialize=lambda: pytest.fail("dead partial was ignored"),
            validate=lambda: pytest.fail("dead partial was trusted"))


@dataclass(frozen=True)
class _FakePlan:
    plan_sha256: str
    native_episodes: frozenset[int]


def test_joint_dino_pusht_uses_one_plan_and_resumes_without_race(
        tmp_path: Path) -> None:
    spec, layout = _fixture(tmp_path)
    planner_calls = 0
    materialized: list[str] = []
    plans = {
        prepare.DINO_PUSHT_COHORTS[0]: _FakePlan(
            "1" * 64, frozenset(range(10))),
        prepare.DINO_PUSHT_COHORTS[1]: _FakePlan(
            "2" * 64, frozenset(range(10, 20))),
    }

    def planner(_spec: Mapping[str, Any]) -> Mapping[str, _FakePlan]:
        nonlocal planner_calls
        planner_calls += 1
        return plans

    def materializer(plan: _FakePlan, cohort: str,
                     target: prepare.PreparationLayout) -> None:
        assert plan is plans[cohort]
        materialized.append(cohort)
        _fake_materialize(cohort, target)

    def validator(*, cohort: str, plan: _FakePlan,
                  layout: prepare.PreparationLayout
                  ) -> prepare.PreparedEvidence:
        return _fake_validate(
            cohort, layout, plan_sha256=plan.plan_sha256)

    result = prepare.prepare_dino_pusht_group(
        spec, layout, planner=planner, materializer=materializer,
        validator=validator)
    assert set(result) == set(prepare.DINO_PUSHT_COHORTS)
    assert planner_calls == 1
    assert materialized == list(prepare.DINO_PUSHT_COHORTS)
    lock = layout.locks / "dinowm_pusht_joint.lock"
    assert lock.is_file()

    materialized.clear()
    prepare.prepare_dino_pusht_group(
        spec, layout, planner=planner,
        materializer=lambda *_: pytest.fail("resume rebuilt PushT"),
        validator=validator)
    assert planner_calls == 2  # exactly one joint plan per invocation
    assert materialized == []


def test_consolidated_registry_is_finalizer_compatible_and_immutable(
        tmp_path: Path) -> None:
    spec, layout = _fixture(tmp_path)
    evidence: dict[str, prepare.PreparedEvidence] = {}
    for cohort in prepare.COHORTS:
        evidence[cohort] = prepare._ensure_cohort(  # noqa: SLF001
            spec=spec, cohort=cohort, layout=layout,
            materialize=lambda cohort=cohort: _fake_materialize(
                cohort, layout),
            validate=lambda cohort=cohort: _fake_validate(cohort, layout),
        )

    def loader(_spec: Mapping[str, Any], _layout: prepare.PreparationLayout
               ) -> Mapping[str, prepare.PreparedEvidence]:
        return evidence

    manifest = prepare.publish_custody_registry(
        spec, layout, evidence_loader=loader)
    registry = json.loads(layout.registry.read_text())
    assert registry["schema"] == prepare.CUSTODY_REGISTRY_SCHEMA
    assert registry["status"] == "sealed"
    assert registry["labels_available_only_after_complete_phase_a_grid"] is True
    assert registry["development_outcomes_read"] is False
    assert set(registry["cohorts"]) == set(prepare.COHORTS)
    assert prepare._mode(layout.registry) == "0o400"  # noqa: SLF001
    assert prepare._mode(layout.manifest) == "0o444"  # noqa: SLF001
    assert manifest["formal_jobs_launched"] is False
    for cohort, record in registry["cohorts"].items():
        assert record["classes"] == prepare.CLASSES[cohort]
        assert set(record["sources"]) == {"formal_test", "consumer_train"}
        for source in record["sources"].values():
            artifact = layout.custody / source["artifact"]["path"]
            assert artifact.is_file()
            assert prepare._sha256_file(artifact) == source[  # noqa: SLF001
                "artifact"]["sha256"]

    resumed = prepare.publish_custody_registry(
        spec, layout, evidence_loader=loader, allow_create=False)
    assert resumed == manifest
    os.chmod(layout.registry, 0o600)
    with pytest.raises(prepare.FormalPreparationError, match="immutable"):
        prepare.publish_custody_registry(
            spec, layout, evidence_loader=loader, allow_create=False)


def test_development_comparator_receipt_is_never_parsed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, layout = _fixture(tmp_path)
    cohort = prepare.COHORTS[0]
    comparator = layout.comparator_receipt(cohort).resolve()
    original = Path.read_text

    def guarded(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.resolve() == comparator:
            raise AssertionError("development comparator JSON was parsed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    boundaries = prepare._opaque_boundary_identities(  # noqa: SLF001
        spec, cohort, layout)
    assert boundaries["locked_comparator_receipt"]["sha256"] == \
        prepare._sha256_file(comparator)  # noqa: SLF001


def test_worker_commands_are_gpu_isolated_preparation_only(
        tmp_path: Path) -> None:
    spec, layout = _fixture(tmp_path)
    commands = prepare.worker_commands(
        spec, Path(spec["_spec_path"]), layout)
    assert len(commands) == 3
    assert [environment["CUDA_VISIBLE_DEVICES"]
            for _, environment in commands] == ["0", "1", "2"]
    assert [command[command.index("--worker") + 1]
            for command, _ in commands] == list(prepare.WORKER_GROUPS)
    for command, _ in commands:
        joined = " ".join(command)
        assert "--stage worker" in joined
        assert "run_sage_mem_v1.py" not in joined
        assert "--stage full" not in joined
        assert prepare.CONFIRMATION in command


def test_source_has_no_formal_cell_or_development_result_access() -> None:
    source = inspect.getsource(prepare)
    assert "run_formal_cell" not in source
    assert "development/cells" not in source
    assert "episode_results" not in source
