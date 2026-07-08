from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.audit_sage_mem_v1_formal as audit


COHORT = "lewm_reacher_color"
TEST_ARMS = (
    "none", "gru", "fixed_trust_aux", "ssm_aux", "sage_mem_full",
    "sage_mem_next_only", "sage_mem_no_exposure",
    "sage_mem_exposure_only",
)


def _contract(*, draws: int = 200) -> audit.AuditContract:
    return audit.AuditContract(
        cohorts=(COHORT,),
        arms=TEST_ARMS,
        seeds=(0, 1, 2),
        ages=(4, 8, 15),
        classes={COHORT: 4},
        formal_rows={COHORT: 40},
        variants={COHORT: 1},
        target_parameters={COHORT: 100},
        bootstrap_draws=draws,
        bootstrap_seed=700,
        parameter_margin=0.05,
        flop_margin=0.10,
        host_comparator_gain=0.05,
        reset_gain=0.03,
        reset_mse_ratio_max=1.25,
        mechanism_gain=0.03,
        mse_relative_margin=0.02,
        execution_gain=0.03,
        execution_oracle_gate=0.90,
        require_raw_context=True,
        require_600_cells=False,
    )


def _binary_rate(rate: float, count: int, shift: int = 0) -> np.ndarray:
    ones = int(round(rate * count))
    value = np.zeros(count, dtype=np.uint8)
    value[:ones] = 1
    return np.roll(value, shift)


def _evidence(*, execution: bool = False) -> audit.LoadedFormalEvidence:
    contract = _contract()
    count = contract.formal_rows[COHORT]
    labels = np.arange(count, dtype=np.int64) % 4
    episode = np.arange(10_000, 10_000 + count, dtype=np.int64)
    ages = np.repeat(np.asarray(contract.ages)[:, None], count, axis=1)
    identity = {
        "formal_test_episode_id": np.repeat(episode[None], 3, axis=0),
        "formal_test_native_cluster_id": np.repeat(episode[None], 3, axis=0),
        "formal_test_evidence_age": ages,
        "formal_test_label": np.repeat(labels[None], 3, axis=0),
    }
    cells = {}
    resources = {}
    for arm in contract.arms:
        for seed in contract.seeds:
            if arm == "sage_mem_full":
                full_rate, prior_rate, reset_rate = 0.50, 0.90, 0.20
            elif arm == "gru":
                full_rate, prior_rate, reset_rate = 0.75, 0.20, 0.20
            elif arm == "none":
                full_rate = prior_rate = reset_rate = 0.25
            else:
                full_rate = prior_rate = reset_rate = 0.10
            arrays = dict(identity)
            arrays.update({
                "formal_test_full_correct": np.stack([
                    _binary_rate(full_rate, count, seed + age_index)
                    for age_index in range(3)]),
                "formal_test_reset_correct": np.stack([
                    _binary_rate(reset_rate, count, seed + age_index)
                    for age_index in range(3)]),
                "formal_test_prior_correct": np.stack([
                    _binary_rate(prior_rate, count, seed + age_index)
                    for age_index in range(3)]),
                "formal_test_full_mse": np.ones((3, count), dtype=np.float32),
                "formal_test_reset_mse": np.full(
                    (3, count), 1.1, dtype=np.float32),
                "formal_test_prior_mse": np.ones((3, count), dtype=np.float32),
            })
            if execution:
                exec_rate = (0.75 if arm == "sage_mem_full" else
                             (0.40 if arm == "gru" else 0.20))
                arrays.update({
                    "formal_test_full_execution_success": np.stack([
                        _binary_rate(exec_rate, count, seed + age_index)
                        for age_index in range(3)]),
                    "formal_test_reset_execution_success": np.zeros(
                        (3, count), dtype=np.uint8),
                    "formal_test_prior_execution_success": np.zeros(
                        (3, count), dtype=np.uint8),
                })
            cells[(COHORT, arm, seed)] = arrays
            resources[(COHORT, arm, seed)] = {
                "trainable_parameters": 0 if arm == "none" else 100,
                "forward_flops_per_episode": 0 if arm == "none" else 1000,
                "persistent_state_floats": 0 if arm == "none" else 10,
                "peak_cuda_bytes": 100,
                "wall_clock_train_seconds": 1.0,
            }
    raw = {}
    for seed in contract.seeds:
        raw[(COHORT, seed)] = {
            **identity,
            "formal_test_short_correct": np.stack([
                _binary_rate(0.25, count, seed + age) for age in range(3)]),
            "formal_test_long_correct": np.stack([
                _binary_rate(0.75, count, seed + age) for age in range(3)]),
        }
    execution_status = ({COHORT: {
        "eligible": True, "oracle_success": 0.95, "random_success": 0.25}
        } if execution else {})
    return audit.LoadedFormalEvidence(
        contract=contract,
        phase_a_grid_sha256="a" * 64,
        cells=cells,
        resources=resources,
        comparators={COHORT: {
            "retention": "gru", "next_feature": "gru", "execution": "gru"}},
        comparator_receipts={COHORT: {
            "path": "/locked/comparator", "size": 1, "sha256": "b" * 64}},
        backend_admissions={COHORT: {
            "backend": "synthetic", "admission_rechecked": True}},
        raw_context=raw,
        execution_status=execution_status,
        execution_random=({COHORT: _binary_rate(0.25, count)}
                          if execution else {}),
        phase_a_cells_verified=len(cells),
        finalized_cells_verified=len(cells),
        raw_context_references_verified=len(raw),
        identity_ledger_sha256="c" * 64,
    )


def test_prior_success_cannot_rescue_failed_frozen_host_output() -> None:
    report = audit.analyze_formal_evidence(_evidence())
    cohort = report["cohorts"][COHORT]
    for age in ("4", "8", "15"):
        result = cohort["ages"][age]
        assert result["prior_diagnostic"]["resolved_positive"] is True
        assert result["prior_diagnostic"]["enters_primary_host_claim"] is False
        assert result["host_full_vs_locked_comparator"]["upper"] < 0
        assert result["gates"]["host_vs_locked_comparator"] is False
        assert result["primary_host_claim_pass"] is False
    assert cohort["all_registered_ages_primary_host_claim_pass"] is False
    assert report["prior_can_substitute_for_host_output"] is False
    assert report["pooled_cross_host_score_computed"] is False


def test_pointmaze_bootstrap_preserves_x4_native_clusters_and_pairing() -> None:
    clusters = np.repeat(np.arange(12), 4)
    labels = np.tile(np.arange(4), 12)
    right = np.zeros((3, 48), dtype=np.float64)
    left = np.ones_like(right)
    first = audit.paired_seed_cluster_bootstrap(
        left, right, labels, clusters, draws=300, seed=9)
    second = audit.paired_seed_cluster_bootstrap(
        left, right, labels, clusters, draws=300, seed=9)
    assert first == second
    assert first["point"] == first["lower"] == first["upper"] == 1.0
    assert first["resampling_unit"] == \
        "paired formal seed x native episode cluster"
    assert first["class_profile_stratified"] is True


def test_parameter_and_flop_mismatch_fail_closed() -> None:
    evidence = _evidence()
    resources = dict(evidence.resources)
    bad = dict(resources[(COHORT, "sage_mem_full", 0)])
    bad["trainable_parameters"] = 120
    resources[(COHORT, "sage_mem_full", 0)] = bad
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="varies across seeds"):
        audit.analyze_formal_evidence(replace(evidence, resources=resources))

    resources = dict(evidence.resources)
    for seed in evidence.contract.seeds:
        bad = dict(resources[(COHORT, "sage_mem_full", seed)])
        bad["forward_flops_per_episode"] = 1300
        resources[(COHORT, "sage_mem_full", seed)] = bad
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="FLOP fairness"):
        audit.analyze_formal_evidence(replace(evidence, resources=resources))


def test_optional_execution_reports_all_three_contrasts_but_program_needs_two() \
        -> None:
    report = audit.analyze_formal_evidence(_evidence(execution=True))
    for value in report["cohorts"][COHORT]["ages"].values():
        execution = value["execution"]
        assert set(execution).issuperset({
            "full_vs_locked_comparator", "full_vs_none",
            "full_vs_random", "pass"})
        assert execution["full_vs_locked_comparator"]["lower"] > 0.03
        assert execution["full_vs_none"]["lower"] > 0.03
        assert execution["full_vs_random"]["lower"] > 0.03
        assert execution["random_reference_is_cohort_rate"] is False
    assert report["execution_program"]["eligible_cohorts"] == 1
    assert report["execution_program"]["program_claim_permitted"] is False
    assert report["execution_program"]["program_claim_pass"] is None
    assert all(not value["claim_pass"] for value in
               report["execution_program"]["per_age"].values())


def test_execution_arrays_cannot_bypass_oracle_eligibility_gate() -> None:
    evidence = _evidence(execution=True)
    status = {COHORT: {
        "eligible": True, "oracle_success": 0.80, "random_success": 0.25}}
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="bypass the oracle gate"):
        audit.analyze_formal_evidence(replace(
            evidence, execution_status=status))

    aggregate_only = {COHORT: np.full(
        evidence.contract.formal_rows[COHORT], 0.25, dtype=np.float64)}
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="bypass the oracle gate"):
        audit.analyze_formal_evidence(replace(
            evidence, execution_random=aggregate_only))


def test_execution_program_gates_each_age_without_cohort_all_age_conjunction() \
        -> None:
    evidence = _evidence(execution=True)
    second = "lewm_pusht_color"
    contract = replace(
        evidence.contract, cohorts=(COHORT, second),
        classes={COHORT: 4, second: 4},
        formal_rows={COHORT: 40, second: 40},
        variants={COHORT: 1, second: 1},
        target_parameters={COHORT: 100, second: 100})
    cells = dict(evidence.cells)
    resources = dict(evidence.resources)
    raw = dict(evidence.raw_context)
    for arm in contract.arms:
        for seed in contract.seeds:
            cells[(second, arm, seed)] = {
                key: np.asarray(value).copy()
                for key, value in evidence.cells[(COHORT, arm, seed)].items()}
            resources[(second, arm, seed)] = dict(
                evidence.resources[(COHORT, arm, seed)])
    for seed in contract.seeds:
        raw[(second, seed)] = {
            key: np.asarray(value).copy()
            for key, value in evidence.raw_context[(COHORT, seed)].items()}
        cells[(COHORT, "sage_mem_full", seed)][
            "formal_test_full_execution_success"][0] = 0
    report = audit.analyze_formal_evidence(replace(
        evidence, contract=contract, cells=cells, resources=resources,
        comparators={
            COHORT: evidence.comparators[COHORT],
            second: evidence.comparators[COHORT]},
        comparator_receipts={
            COHORT: evidence.comparator_receipts[COHORT],
            second: evidence.comparator_receipts[COHORT]},
        backend_admissions={
            COHORT: evidence.backend_admissions[COHORT],
            second: evidence.backend_admissions[COHORT]},
        raw_context=raw,
        execution_status={
            COHORT: evidence.execution_status[COHORT],
            second: evidence.execution_status[COHORT]},
        execution_random={
            COHORT: evidence.execution_random[COHORT],
            second: evidence.execution_random[COHORT]}))
    program = report["execution_program"]
    assert program["per_age"]["4"]["cohorts_passing"] == 1
    assert program["per_age"]["4"]["claim_pass"] is False
    assert program["per_age"]["8"]["cohorts_passing"] == 2
    assert program["per_age"]["8"]["claim_pass"] is True
    assert program["per_age"]["15"]["claim_pass"] is True
    assert program["cross_age_conjunction_computed"] is False
    assert program["program_claim_pass"] is None


def test_raw_short3_long16_is_separate_and_age_specific() -> None:
    report = audit.analyze_formal_evidence(_evidence())
    for age, value in report["cohorts"][COHORT]["ages"].items():
        raw = value["raw_context_reference"]
        assert raw["short3_accuracy"] == 0.25
        assert raw["long16_accuracy"] == 0.75
        assert raw["long16_minus_short3"]["lower"] > 0
        assert raw["resolved_long_context_gain"] is True
        assert raw["separate_from_parameter_matched_grid"] is True
        assert int(age) in (4, 8, 15)


def test_real_raw_producer_to_finalizer_to_auditor_contract(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import prepare_sage_mem_v1_raw_context_reference as producer
    from scripts import sage_mem_v1_formal_finalizer as finalizer
    from tests import test_sage_mem_v1_formal_finalizer as fixtures

    final_contract = fixtures._contract(arms=("none",))  # noqa: SLF001
    cohort = final_contract.cohorts[0]
    classes = final_contract.classes[cohort]

    class SyntheticSpatialBank:
        spatial = True

        def __init__(self) -> None:
            self.ids = {
                "consumer_train": fixtures._identity(  # noqa: SLF001
                    final_contract.consumer_train_rows[cohort], 1, 1000),
                "formal_test": fixtures._identity(  # noqa: SLF001
                    final_contract.formal_test_rows[cohort], 1, 100),
            }

        def indices(self, split: str) -> np.ndarray:
            return np.arange(self.ids[split][0].shape[1], dtype=np.int64)

        def identity(self, split: str) -> dict[str, np.ndarray]:
            return {"episode_id": self.ids[split][0][0],
                    "native_cluster_id": self.ids[split][1][0]}

        def features(self, age: int, indices: np.ndarray) -> np.ndarray:
            rows = np.asarray(indices)
            labels = np.arange(len(rows), dtype=np.int64) % classes
            frame = np.eye(classes, dtype=np.float32)[labels]
            return np.repeat(
                np.repeat(frame[:, None, None, :], 20, axis=1), 2, axis=2)

    bank = SyntheticSpatialBank()
    view = producer.PreparedBankView(
        cohort=cohort, spatial=True,
        bank_manifest_sha256="b" * 64, host_hash="a" * 64,
        split_banks={split: bank for split in producer.SPLITS})
    monkeypatch.setattr(
        producer, "_load_prepared_bank",
        lambda cohort, prepared_root, split_counts: view)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    raw_phase_a = tmp_path / "raw_context_phase_a"
    producer._produce_grid(  # noqa: SLF001
        config_path=producer.DEFAULT_CONFIG, prepared_root=prepared,
        output_root=raw_phase_a, cohorts=(cohort,), seeds=(0,))
    phase = tmp_path / "phase"
    fixtures._write_grid(phase, final_contract)  # noqa: SLF001
    registry = fixtures._write_registry(  # noqa: SLF001
        tmp_path, final_contract)
    output = tmp_path / "finalized"
    finalizer._finalize_with_contract(  # noqa: SLF001
        phase, registry, output, final_contract,
        raw_context_root=raw_phase_a)
    grid = finalizer._validate_complete_grid(  # noqa: SLF001
        phase, final_contract)
    contract = audit.AuditContract(
        cohorts=final_contract.cohorts, arms=final_contract.arms,
        seeds=final_contract.seeds, ages=final_contract.ages,
        classes=final_contract.classes,
        formal_rows=final_contract.formal_test_rows,
        variants=final_contract.variants_per_cluster,
        target_parameters={cohort: 1}, bootstrap_draws=50,
        bootstrap_seed=1, parameter_margin=0.05, flop_margin=0.10,
        host_comparator_gain=0.05, reset_gain=0.03,
        reset_mse_ratio_max=1.25, mechanism_gain=0.03,
        mse_relative_margin=0.02, execution_gain=0.03,
        execution_oracle_gate=0.90, require_raw_context=True,
        require_600_cells=False)
    _, raw, execution, _ = audit._load_finalized_grid(  # noqa: SLF001
        output, grid, contract)
    assert execution == {}
    assert set(raw[(cohort, 0)]) == audit.RAW_ARRAYS
    assert not any(name.endswith("_mse") for name in raw[(cohort, 0)])

    consumer_path = (output / "raw_context_consumers" / cohort /
                     "seed-0.json")
    consumer = json.loads(consumer_path.read_text())
    consumer["feature_dimension"] += 1
    consumer_path.write_text(json.dumps(consumer))
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="source ledger differs"):
        audit._load_finalized_grid(  # noqa: SLF001
            output, grid, contract)


def test_finalized_array_validator_recomputes_host_and_prior_correctness() -> None:
    evidence = _evidence()
    contract = evidence.contract
    source = evidence.cells[(COHORT, "sage_mem_full", 0)]
    labels = source["formal_test_label"]
    arrays = {
        "formal_test_episode_id": source["formal_test_episode_id"],
        "formal_test_native_cluster_id":
            source["formal_test_native_cluster_id"],
        "formal_test_evidence_age": source["formal_test_evidence_age"],
        "formal_test_label": labels,
    }
    for stream, correct in (
            ("full", source["formal_test_full_correct"]),
            ("reset", source["formal_test_reset_correct"]),
            ("prior", source["formal_test_prior_correct"])):
        pred = np.where(correct == 1, labels, (labels + 1) % 4)
        arrays[f"formal_test_{stream}_pred"] = pred
        arrays[f"formal_test_{stream}_correct"] = correct.copy()
        arrays[f"formal_test_{stream}_mse"] = np.ones_like(
            correct, dtype=np.float32)
    phase = {name: arrays[name] for name in (
        "formal_test_episode_id", "formal_test_native_cluster_id",
        "formal_test_evidence_age")}
    audit._validate_finalized_arrays(  # noqa: SLF001
        arrays, cohort=COHORT, contract=contract, phase_identity=phase)
    arrays["formal_test_full_correct"] = 1 - arrays[
        "formal_test_full_correct"]
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="correctness disagrees"):
        audit._validate_finalized_arrays(  # noqa: SLF001
            arrays, cohort=COHORT, contract=contract, phase_identity=phase)


def test_locked_comparator_is_bound_through_prepare_receipt(
        tmp_path: Path) -> None:
    contract = _contract()
    prepare_root = tmp_path / "prepare"
    (prepare_root / "receipts").mkdir(parents=True)
    (prepare_root / "banks" / COHORT).mkdir(parents=True)
    (prepare_root / "custody" / "receipts").mkdir(parents=True)

    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def identity(path: Path, *, relative: bool = False) -> dict[str, object]:
        return {
            "path": (str(path.relative_to(prepare_root)) if relative
                     else str(path.resolve())),
            "size": path.stat().st_size,
            "sha256": audit._sha256_file(path),  # noqa: SLF001
        }

    spec_path = tmp_path / "spec.yaml"
    write(spec_path, {"study": "sage-mem-v1"})
    lock = tmp_path / "protocol-lock.json"
    write(lock, {"locked": True})
    spec = {
        "study": "sage-mem-v1",
        "implementation_lock": str(lock),
        "_spec_path": str(spec_path),
        "_spec_sha256": audit._sha256_file(spec_path),  # noqa: SLF001
    }
    from scripts.sage_mem_v1_spec import spec_fingerprint

    selection = tmp_path / "selection.json"
    write(selection, {
        "schema_version": 1,
        "study": "sage-mem-v1",
        "stage": "development-selection",
        "status": "selected",
        "cohort": COHORT,
        "protocol_fingerprint": spec_fingerprint(spec),
        "locked_comparators": {
            "retention": "gru", "next_feature": "gru", "execution": "gru"},
        "gdelta_development_healthy": True,
        "labels_used_only_for_posthoc_selection_metrics": True,
        "formal_data_read": False,
        "formal_execution_started": False,
    })
    bank = prepare_root / "banks" / COHORT / "manifest.json"
    write(bank, {
        "host_hash_before": "1" * 64,
        "host_hash_after": "1" * 64,
        "admissions": {
            "parent_overlap_zero": True,
            "formal_split_overlap_zero": True,
        },
    })
    custody_receipt = prepare_root / "custody" / "receipts" / f"{COHORT}.json"
    write(custody_receipt, {"sealed": True, "sha256": "f" * 64})
    custody_record = {
        "bank_manifest_sha256": audit._sha256_file(bank),  # noqa: SLF001
        "classes": 4,
        "sources": {
            "formal_test": {"artifact": {"sha256": "f" * 64}},
            "consumer_train": {"artifact": {"sha256": "f" * 64}},
        },
    }
    registry = prepare_root / "custody" / "registry.json"
    write(registry, {
        "schema": "sage_mem_v1_custody_vault_registry_v1",
        "study": "sage-mem-v1",
        "status": "sealed",
        "labels_available_only_after_complete_phase_a_grid": True,
        "development_outcomes_read": False,
        "cohorts": {COHORT: custody_record},
    })
    protocol_identity = identity(spec_path)
    lock_identity = identity(lock)
    receipt = {
        "schema": "sage_mem_v1_formal_prepared_cohort_v1",
        "study": "sage-mem-v1",
        "status": "prepared-and-hash-validated",
        "cohort": COHORT,
        "formal_jobs_launched": False,
        "formal_outcomes_read": False,
        "development_outcomes_read": False,
        "development_access": "opaque locked comparator receipt identity only",
        "boundaries": {
            "study_protocol": protocol_identity,
            "implementation_lock": lock_identity,
            "locked_comparator_receipt": identity(selection),
        },
        "bank_manifest": identity(bank, relative=True),
        "custody_receipt": identity(custody_receipt, relative=True),
        "sealed_vault_sha256": "f" * 64,
        "backend_proof": {
            "backend": "SIGReg-LeWM",
            "host_hash_before": "1" * 64,
            "host_hash_after": "1" * 64,
            "parent_overlap_zero": True,
            "formal_split_overlap_zero": True,
        },
        "finalizer_custody_record": custody_record,
    }
    receipt_path = prepare_root / "receipts" / f"{COHORT}.json"
    write(receipt_path, receipt)
    write(prepare_root / "manifest.json", {
        "schema": "sage_mem_v1_formal_preparation_v1",
        "study": "sage-mem-v1",
        "status": "complete",
        "formal_jobs_launched": False,
        "formal_outcomes_read": False,
        "development_outcomes_read": False,
        "development_access": "opaque locked comparator receipt identities only",
        "study_protocol": protocol_identity,
        "implementation_lock": lock_identity,
        "custody_registry": identity(registry, relative=True),
        "cohort_receipts": {COHORT: identity(receipt_path, relative=True)},
    })
    grid = SimpleNamespace(bank_hashes={
        COHORT: audit._sha256_file(bank)})  # noqa: SLF001
    comparators, identities, admissions, registry_path = \
        audit._load_locked_comparators(  # noqa: SLF001
            prepare_root, grid, contract, spec)
    assert comparators[COHORT]["retention"] == "gru"
    assert identities[COHORT]["locked_comparator_receipt"]["sha256"] == \
        audit._sha256_file(selection)  # noqa: SLF001
    assert admissions[COHORT]["admission_rechecked"] is True
    assert registry_path == registry.resolve()

    selection.write_text("tampered")
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="locked comparator receipt identity changed"):
        audit._load_locked_comparators(  # noqa: SLF001
            prepare_root, grid, contract, spec)

    write(selection, {"jointly": "replaced"})
    receipt["boundaries"]["locked_comparator_receipt"] = identity(selection)
    write(receipt_path, receipt)
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="preparation receipt identity changed"):
        audit._load_locked_comparators(  # noqa: SLF001
            prepare_root, grid, contract, spec)


def _cli_arguments(tmp_path: Path, output: Path) -> list[str]:
    return [
        "--execute",
        "--phase-a-root", str(tmp_path / "phase-a"),
        "--finalized-root", str(tmp_path / "finalized"),
        "--prepare-root", str(tmp_path / "prepare"),
        "--output", str(output),
    ]


def test_cli_resume_recomputes_and_accepts_only_canonical_equal_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    report = {
        "schema": audit.AUDIT_SCHEMA,
        "status": "complete",
        "pooled_cross_host_score_computed": False,
    }
    calls = []

    def recompute(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return dict(report)

    monkeypatch.setattr(audit, "audit_formal_evidence", recompute)
    output = tmp_path / "audit.json"
    arguments = _cli_arguments(tmp_path, output)
    assert audit.main(arguments) == 0
    original = output.read_bytes()
    assert audit.main([*arguments, "--resume"]) == 0
    assert len(calls) == 2
    assert output.read_bytes() == original
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == report


def test_cli_resume_rejects_tampered_or_stale_report_without_overwrite(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = {"value": {
        "schema": audit.AUDIT_SCHEMA, "status": "complete", "revision": 1}}

    def recompute(**unused: object) -> dict[str, object]:
        return dict(current["value"])

    monkeypatch.setattr(audit, "audit_formal_evidence", recompute)
    output = tmp_path / "audit.json"
    arguments = _cli_arguments(tmp_path, output)
    assert audit.main(arguments) == 0

    output.write_text(json.dumps({**current["value"], "tampered": True}))
    tampered = output.read_bytes()
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="differs from freshly authenticated evidence"):
        audit.main([*arguments, "--resume"])
    assert output.read_bytes() == tampered

    output.write_text(json.dumps(current["value"]))
    original = output.read_bytes()
    current["value"] = {**current["value"], "revision": 2}
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="differs from freshly authenticated evidence"):
        audit.main([*arguments, "--resume"])
    assert output.read_bytes() == original


def test_resume_requires_execute_and_output() -> None:
    with pytest.raises(audit.FormalEvidenceAuditError,
                       match="requires --execute and --output"):
        audit.main([
            "--resume", "--phase-a-root", "/unused",
            "--finalized-root", "/unused", "--prepare-root", "/unused"])


def test_registered_raw_context_has_no_missing_data_bypass() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([
            "--phase-a-root", "/unused", "--finalized-root", "/unused",
            "--prepare-root", "/unused", "--allow-missing-raw-context"])


def test_production_loader_invokes_official_finalizer_validation_first(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import sage_mem_v1_formal_finalizer as finalizer
    from scripts.sage_mem_v1_spec import load_spec

    spec = load_spec(audit.DEFAULT_SPEC, verify_parent_paths=False)
    contract = audit.contract_from_spec(spec)
    keys = [(cohort, arm, seed) for cohort in contract.cohorts
            for arm in contract.arms for seed in contract.seeds]
    fake_cells = {key: SimpleNamespace() for key in keys}
    grid = SimpleNamespace(
        cells=fake_cells, grid_sha256="a" * 64,
        bank_hashes={cohort: "b" * 64 for cohort in contract.cohorts})
    study = tmp_path / "study"
    prepare = study / "formal_preparation"
    prepare.mkdir(parents=True)
    raw = study / "raw_context_phase_a"
    raw.mkdir()
    finalized = study / "formal_finalized"
    finalized.mkdir()
    (finalized / "label_reveal_receipt.json").write_text(json.dumps({
        "raw_context_reference": {"sha256": "e" * 64}}))
    registry = prepare / "custody" / "registry.json"
    registry.parent.mkdir()
    registry.write_text("{}")
    events: list[object] = []

    monkeypatch.setattr(
        finalizer, "validate_complete_phase_a_grid", lambda root: grid)

    def official(*args: object, **kwargs: object) -> dict[str, object]:
        events.append(("official", args, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(finalizer, "validate_finalized_output", official)
    monkeypatch.setattr(finalizer, "_load_execution_decks",  # noqa: SLF001
                        lambda *args, **kwargs: {})
    comparators = {cohort: {
        "retention": "gru", "next_feature": "gru", "execution": "gru"}
        for cohort in contract.cohorts}
    receipts = {cohort: {"sha256": "c" * 64}
                for cohort in contract.cohorts}
    admissions = {cohort: {"admission_rechecked": True}
                  for cohort in contract.cohorts}
    monkeypatch.setattr(
        audit, "_load_locked_comparators",
        lambda *args, **kwargs:
        (comparators, receipts, admissions, registry.resolve()))
    monkeypatch.setattr(audit, "_infer_execution_registry", lambda root: None)
    monkeypatch.setattr(
        audit, "_load_resource_report",
        lambda cell: ({
            "trainable_parameters": 1,
            "forward_flops_per_episode": 1,
            "persistent_state_floats": 1,
            "peak_cuda_bytes": 1,
            "wall_clock_train_seconds": 1.0,
        }, "d" * 64))

    def statistical_load(*args: object, **kwargs: object):
        events.append("statistical-load")
        return ({key: {} for key in keys}, {}, {}, "a" * 64)

    monkeypatch.setattr(audit, "_load_finalized_grid", statistical_load)
    evidence = audit.load_formal_evidence(
        spec=spec, phase_a_root=tmp_path / "phase",
        finalized_root=finalized, prepare_root=prepare, contract=contract)
    assert evidence.finalized_cells_verified == 600
    assert events[0][0] == "official"
    assert events[0][2]["raw_context_root"] == raw
    assert events[0][1][1] == registry.resolve()
    assert events[1] == "statistical-load"
