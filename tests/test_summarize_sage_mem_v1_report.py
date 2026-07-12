from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import scripts.summarize_sage_mem_v1_report as adapter
from scripts.sage_mem_v1_spec import DEFAULT_SPEC, load_spec


@pytest.fixture
def _synthetic_sealed_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of ignored, machine-local campaign output."""

    def authenticate(spec: dict[str, object],
                     spec_path: Path) -> dict[str, object]:
        auditor = adapter.AUDITOR_SOURCE.resolve()
        resolved_spec = spec_path.resolve()
        return {
            "implementation_lock": {
                "path": "outputs/sage_mem_v1/protocol_lock.json",
                "size": 100,
                "sha256": "d" * 64,
                "status": "sealed",
            },
            "formal_amendment": {
                "path": "configs/sage_mem_v1_formal_amendment.yaml",
                "size": 100,
                "sha256": "5" * 64,
                "status":
                    "locked-before-development-selection-or-formal-data",
            },
            "formal_auditor": {
                "path": adapter._display_path(auditor),
                "size": auditor.stat().st_size,
                "sha256": adapter._sha256_file(
                    auditor, "formal evidence auditor source"),
                "sealed_by_implementation_lock": True,
            },
            "spec": {
                "path": adapter._display_path(resolved_spec),
                "size": resolved_spec.stat().st_size,
                "sha256": spec["_spec_sha256"],
                "fingerprint": adapter.spec_fingerprint(spec),
            },
        }

    monkeypatch.setattr(adapter, "_authenticate_protocol_lock", authenticate)


def _stat(*, lower: float, confidence: float = 0.95,
          seed: int = 1) -> dict[str, object]:
    return {
        "point": lower + 0.01,
        "lower": lower,
        "upper": lower + 0.02,
        "confidence": confidence,
        "draws": 20_000,
        "seed": seed,
        "resampling_unit": "paired formal seed x native episode cluster",
        "class_profile_stratified": True,
        "pairing_preserved": True,
    }


def _resources(cohort: str) -> dict[str, dict[str, object]]:
    target = 76_032 if cohort.startswith("lewm_") else 299_520
    result: dict[str, dict[str, object]] = {
        "none": {
            "trainable_parameters": 0,
            "forward_flops_per_episode": 0.0,
            "persistent_state_floats": 0,
            "parameter_matched": None,
            "flop_matched": None,
        },
    }
    for arm in adapter.ARMS:
        if arm == "none":
            continue
        result[arm] = {
            "trainable_parameters": target,
            "target_parameters": target,
            "parameter_relative_gap": 0.0,
            "parameter_matched": True,
            "forward_flops_per_episode": 1000.0,
            "persistent_state_floats": 32,
            "peak_cuda_bytes_mean": 100.0,
            "wall_clock_train_seconds_mean": 1.0,
            "baseline_median_flops": 1000.0,
            "flop_relative_gap": 0.0,
            "flop_matched": True,
        }
    return result


def _receipt(character: str) -> dict[str, object]:
    return {
        "formal_preparation_manifest_sha256": character * 64,
        "implementation_lock_sha256": "d" * 64,
        "custody_registry_sha256": "e" * 64,
        "preparation_receipt": {
            "path": "formal_preparation/receipts/cohort.json",
            "size": 123,
            "sha256": "f" * 64,
        },
        "locked_comparator_receipt": {
            "path": "development/selections/cohort.json",
            "size": 456,
            "sha256": "1" * 64,
        },
    }


def _backend(cohort: str) -> dict[str, object]:
    if cohort.startswith("lewm_"):
        return {
            "backend": "SIGReg-LeWM",
            "host_hash_before": "2" * 64,
            "host_hash_after": "2" * 64,
            "parent_overlap_zero": True,
            "formal_split_overlap_zero": True,
            "admission_rechecked": True,
        }
    return {
        "backend": "DINO-WM",
        "plan_sha256": "3" * 64,
        "provenance_manifest_sha256": "4" * 64,
        "parent_episode_overlap_count": 0,
        "cross_split_native_episode_overlap_count": 0,
        "admission_rechecked": True,
    }


def _execution(*, passes: bool, seed: int) -> dict[str, object]:
    lower = 0.04 if passes else 0.02
    values = {
        name: _stat(lower=(lower if name == "full_vs_locked_comparator"
                           else 0.04), seed=seed + index)
        for index, name in enumerate((
            "full_vs_locked_comparator", "full_vs_none", "full_vs_random"))
    }
    return {
        **values,
        "random_reference":
            "sealed per-episode arm-blind random-success deck",
        "random_reference_is_cohort_rate": False,
        "oracle_success": 0.95,
        "random_success_mean": 0.25,
        "pass": passes,
    }


def _age(*, primary_passes: bool, execution: bool | None,
         namespace: int) -> dict[str, object]:
    host_lower = 0.06 if primary_passes else 0.04
    controls = {
        arm: _stat(lower=0.04, seed=namespace + 10 + index)
        for index, arm in enumerate(adapter.MECHANISM_CONTROLS)
    }
    gates = {
        "host_vs_locked_comparator": primary_passes,
        "full_vs_reset": True,
        "full_vs_none": True,
        "all_mechanism_controls": True,
        "next_mse_noninferiority": True,
    }
    return {
        "primary_endpoint": "frozen-host full correctness",
        "host_full_accuracy": _stat(lower=0.50, seed=namespace),
        "host_full_vs_locked_comparator":
            _stat(lower=host_lower, seed=namespace + 2),
        "host_full_vs_reset": _stat(lower=0.04, seed=namespace + 3),
        "host_full_vs_none": _stat(lower=0.04, seed=namespace + 4),
        "reset_to_full_mse_ratio": 1.1,
        "mechanism_controls": controls,
        "next_feature_relative_excess":
            _stat(lower=-0.01, confidence=0.90, seed=namespace + 20),
        "gates": gates,
        "primary_host_claim_pass": primary_passes,
        "prior_diagnostic": {
            "role": "diagnostic-only; cannot establish host use",
            "accuracy": _stat(lower=0.60, seed=namespace + 1),
            "vs_locked_comparator": _stat(
                lower=0.01, seed=namespace + 5),
            "resolved_positive": True,
            "enters_primary_host_claim": False,
        },
        "raw_context_reference": {
            "short3_accuracy": 0.30,
            "long16_accuracy": 0.60,
            "long16_minus_short3": _stat(
                lower=0.20, seed=namespace + 30),
            "resolved_long_context_gain": True,
            "separate_from_parameter_matched_grid": True,
        },
        "execution": (
            None if execution is None
            else _execution(passes=execution, seed=namespace + 40)),
    }


def _report() -> dict[str, object]:
    bootstrap_seed = 2026070821
    cohorts: dict[str, object] = {}
    execution_passing = {age: 0 for age in adapter.AGES}
    for cohort_index, cohort in enumerate(adapter.COHORTS):
        execution_supplied = cohort_index < 2
        ages: dict[str, object] = {}
        pass_by_age: dict[str, bool | None] = {}
        primary_by_age = []
        for age_index, age in enumerate(adapter.AGES):
            primary = (cohort_index + age_index) % 2 == 0
            execution = (
                (cohort_index == 0 or age_index == 0)
                if execution_supplied else None)
            ages[age] = _age(
                primary_passes=primary, execution=execution,
                namespace=(bootstrap_seed + cohort_index * 10_000
                           + age_index * 1_000))
            pass_by_age[age] = execution
            primary_by_age.append(primary)
            if execution is True:
                execution_passing[age] += 1
        cohorts[cohort] = {
            "locked_comparators": {
                "retention": "gru", "next_feature": "gru", "execution": "gru",
            },
            "comparator_receipt": _receipt(chr(ord("a") + cohort_index)),
            "backend_admission": _backend(cohort),
            "resource_enforcement": _resources(cohort),
            "ages": ages,
            "all_registered_ages_primary_host_claim_pass":
                all(primary_by_age),
            "execution_supplied": execution_supplied,
            "execution_pass_by_age": pass_by_age,
        }
    eligible = 2
    return {
        "schema": adapter.REPORT_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "formal-evidence-audit",
        "status": "complete",
        "phase_a_cells_verified": 600,
        "finalized_cells_verified": 600,
        "phase_a_grid_sha256": "a" * 64,
        "identity_ledger_sha256": "b" * 64,
        "comparators_verified": 5,
        "resources_verified": 600,
        "raw_context_references_verified": 50,
        "bootstrap_draws_per_contrast": 20_000,
        "cohorts": cohorts,
        "execution_program": {
            "optional": True,
            "eligible_cohorts": eligible,
            "minimum_eligible_cohorts": 2,
            "program_claim_permitted": True,
            "per_age": {
                age: {
                    "eligible_cohorts": eligible,
                    "cohorts_passing": execution_passing[age],
                    "claim_permitted": True,
                    "claim_pass": execution_passing[age] >= 2,
                }
                for age in adapter.AGES
            },
            "cross_age_conjunction_computed": False,
            "program_claim_pass": None,
        },
        "prior_can_substitute_for_host_output": False,
        "per_age_claims_only": True,
        "pooled_cross_host_score_computed": False,
        "universal_success_claim_permitted": False,
    }


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _materialize_comparator_receipts(
        root: Path, report: dict[str, object]) -> None:
    spec = load_spec(DEFAULT_SPEC)
    fingerprint = adapter.spec_fingerprint(spec)
    receipts_root = root / "synthetic-receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)
    cohorts = report["cohorts"]
    assert isinstance(cohorts, dict)
    for cohort in adapter.COHORTS:
        cohort_record = cohorts[cohort]
        assert isinstance(cohort_record, dict)
        locked = cohort_record["locked_comparators"]
        selection_path = receipts_root / f"{cohort}-selection.json"
        selection_path.write_text(adapter._canonical_json({
            "schema_version": 1,
            "study": "sage-mem-v1",
            "stage": "development-selection",
            "status": "selected",
            "cohort": cohort,
            "protocol_fingerprint": fingerprint,
            "locked_comparators": locked,
        }))
        selection_identity = _identity(selection_path)
        preparation_path = receipts_root / f"{cohort}-preparation.json"
        preparation_path.write_text(adapter._canonical_json({
            "cohort": cohort,
            "boundaries": {
                "locked_comparator_receipt": selection_identity,
            },
        }))
        receipt = cohort_record["comparator_receipt"]
        assert isinstance(receipt, dict)
        receipt["preparation_receipt"] = _identity(preparation_path)
        receipt["locked_comparator_receipt"] = selection_identity


def _write_report(path: Path, report: dict[str, object]) -> bytes:
    payload = adapter._canonical_json(report).encode()
    path.write_bytes(payload)
    return payload


def _write_canonical(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(adapter._canonical_json(value))
    return _identity(path)


def _materialize_phase_b_receipt(
        tmp_path: Path, report_path: Path,
        report: dict[str, object]) -> tuple[Path, str, Path]:
    root = tmp_path / "phase-b-study"
    raw = _write_canonical(
        root / "raw_context_phase_a/summary.json", {"schema": "raw-test"})
    labels = _write_canonical(
        root / "formal_preparation/custody/registry.json",
        {"schema": "labels-test"})
    execution = _write_canonical(
        root / "formal_preparation/execution_decks/registry.json",
        {"schema": "execution-test"})
    finalized_digest = "9" * 64
    finalizer = _write_canonical(
        root / "formal_finalized/summary.json", {
            "schema": "sage_mem_v1_formal_finalizer_v1",
            "study": "sage-mem-v1", "stage": "formal-finalizer",
            "status": "complete", "phase_a_cells": 600,
            "finalized_cells": 600,
            "phase_a_grid_sha256": report["phase_a_grid_sha256"],
            "label_registry_sha256": labels["sha256"],
            "finalized_cells_sha256": finalized_digest,
        })
    report_identity = _identity(report_path)
    verifier = _identity(adapter.PHASE_B_VERIFIER_SOURCE)
    contract_digest = adapter._sha256_json_value(
        adapter.PHASE_B_CONTRACT_IDENTITY)
    pins = {
        "verifier_source_sha256": verifier["sha256"],
        "protocol_lock_sha256": "d" * 64,
        "phase_a_grid_sha256": report["phase_a_grid_sha256"],
        "raw_context_summary_sha256": raw["sha256"],
        "label_registry_sha256": labels["sha256"],
        "execution_registry_sha256": execution["sha256"],
        "finalizer_summary_sha256": finalizer["sha256"],
        "finalized_cells_sha256": finalized_digest,
        "formal_report_sha256": report_identity["sha256"],
    }
    receipt = {
        "schema": adapter.PHASE_B_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "phase-b-independent-reproduction",
        "status": "complete",
        "production_contract_verified": True,
        "report_reproducer_injected": False,
        "verifier_source_injected": False,
        "contract_identity": copy.deepcopy(adapter.PHASE_B_CONTRACT_IDENTITY),
        "contract_identity_sha256": contract_digest,
        "registered_contract_sha256": contract_digest,
        "outcome_values_emitted": False,
        "finalizer_prediction_helpers_called": False,
        "operator_pins": pins,
        "authenticated_inventories": {
            "verifier_source": verifier,
            "bound_input_files": {
                "protocol_lock": {
                    "path": str((root / "protocol_lock.json").resolve()),
                    "size": 100, "sha256": "d" * 64,
                },
                "raw_context_summary": raw,
                "label_registry": labels,
                "execution_registry": execution,
                "finalizer_summary": finalizer,
                "formal_report": report_identity,
            },
            "numerical_environment": {"fixture": True},
            "locked_producers_sha256": "1" * 64,
            "phase_a_artifacts_sha256": "2" * 64,
            "normalized_label_artifacts_sha256": "3" * 64,
            "phase_a_cells": 600,
            "raw_context_references": 50,
            "finalized_cells": 600,
            "execution_registry_status_sha256": "4" * 64,
            "formal_report_sha256": report_identity["sha256"],
            "replayed_formal_report_sha256": report_identity["sha256"],
        },
        "independent_reproduction": {
            "registered_consumer": {
                "estimator": "sklearn.linear_model.RidgeClassifier",
                "alpha": 1e-3, "solver": "lsqr", "tol": 1e-6,
                "max_iter": 5000,
                "standardization": "StandardScaler(mean=True,std=True)",
                "carrier_models_refit": 150,
                "raw_context_models_refit": 50,
            },
            "carrier_streams_reproduced": ["full", "reset", "prior"],
            "raw_context_streams_reproduced": ["short-3", "long-16"],
            "eligible_execution_arrays_recomputed": True,
            "all_arrays_exact": True,
            "formal_report_byte_exact": True,
            "report_timestamp_normalization": "none in fixture",
        },
        "semantic_digests": {
            key: f"{index:x}" * 64
            for index, key in enumerate(
                sorted(adapter.PHASE_B_SEMANTIC_KEYS), start=5)
        },
        "claim_boundary": (
            "provenance-and-reproduction-only; this receipt contains no "
            "accuracy, effect, interval, gate, or universal-success claim"),
    }
    receipt_path = (
        root / "receipts/phase_b/reproduction_receipt.json")
    receipt_identity = _write_canonical(receipt_path, receipt)
    return receipt_path, str(receipt_identity["sha256"]), root


def _phase_b_cli(tmp_path: Path) -> tuple[list[str], Path]:
    root = tmp_path / "phase-b-study"
    receipt = root / "receipts/phase_b/reproduction_receipt.json"
    return [
        "--phase-b-receipt", str(receipt),
        "--expected-phase-b-receipt-sha256",
        hashlib.sha256(receipt.read_bytes()).hexdigest(),
    ], root


def _execute(tmp_path: Path, report: dict[str, object],
             *extra: str, materialize_receipts: bool = True
             ) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if materialize_receipts:
        _materialize_comparator_receipts(tmp_path, report)
    report_path = tmp_path / "report.json"
    json_path = tmp_path / "claim-ledger.json"
    tex_path = tmp_path / "claim-ledger.tex"
    report_payload = _write_report(report_path, report)
    phase_b_path, phase_b_sha256, phase_b_root = \
        _materialize_phase_b_receipt(tmp_path, report_path, report)
    arguments = [
        "--report", str(report_path),
        "--spec", str(DEFAULT_SPEC),
        "--json-output", str(json_path),
        "--tex-output", str(tex_path),
        "--execute",
        "--phase-b-receipt", str(phase_b_path),
        "--expected-phase-b-receipt-sha256", phase_b_sha256,
        *extra,
    ]
    if "--expected-report-sha256" not in arguments:
        arguments.extend([
            "--expected-report-sha256",
            hashlib.sha256(report_payload).hexdigest(),
        ])
    assert adapter.main(
        arguments,
        recompute_formal_audit=lambda _spec: copy.deepcopy(report),
        publication_root=tmp_path,
        phase_b_study_root=phase_b_root) == 0
    return report_path, json_path, tex_path


def test_preview_reads_nothing_and_writes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = tmp_path / "does-not-exist.json"
    json_path = tmp_path / "ledger.json"
    tex_path = tmp_path / "ledger.tex"
    assert adapter.main([
        "--report", str(report),
        "--spec", str(tmp_path / "missing.yaml"),
        "--json-output", str(json_path),
        "--tex-output", str(tex_path),
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview"] is True
    assert preview["no_files_read"] is True
    assert preview["no_outcomes_read"] is True
    assert preview["required_claim_rows"] == 15
    assert not json_path.exists()
    assert not tex_path.exists()


def test_actual_sealed_lock_binds_current_formal_auditor() -> None:
    spec = load_spec(DEFAULT_SPEC)
    lock_path = adapter.ROOT / str(spec["implementation_lock"])
    if not lock_path.is_file():
        pytest.skip("machine-local sealed campaign lock is unavailable")
    binding = adapter._authenticate_protocol_lock(spec, DEFAULT_SPEC)
    assert binding["implementation_lock"]["status"] == "sealed"
    assert binding["spec"]["fingerprint"] == adapter.spec_fingerprint(spec)
    assert binding["formal_auditor"]["sealed_by_implementation_lock"] is True
    assert binding["formal_auditor"]["sha256"] == adapter._sha256_file(
        adapter.AUDITOR_SOURCE, "formal evidence auditor source")


def test_execute_preserves_all_rows_and_separates_integrity_from_science(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        _synthetic_sealed_lock: None) -> None:
    report = _report()
    report_path, json_path, tex_path = _execute(tmp_path, report)
    result = json.loads(capsys.readouterr().out)
    assert result["integrity_status"] == "complete"
    assert result["integrity_is_scientific_pass"] is False
    assert result["claim_rows"] == 15

    payload = json_path.read_bytes()
    ledger = json.loads(payload)
    assert payload == adapter._canonical_json(ledger).encode()
    assert [(row["cohort"], row["age"]) for row in ledger["claim_rows"]] == [
        (cohort, int(age))
        for cohort in adapter.COHORTS for age in adapter.AGES
    ]
    passing = sum(
        row["primary_host_claim_pass"] for row in ledger["claim_rows"])
    assert 0 < passing < 15
    assert ledger["scientific_result"]["primary_claim_rows_passing"] == passing
    assert ledger["integrity_completion"]["status"] == "complete"
    assert "not a scientific pass" in \
        ledger["integrity_completion"]["meaning"]
    failed_with_positive_prior = next(
        row for row in ledger["claim_rows"]
        if not row["primary_host_claim_pass"]
        and row["prior_diagnostic"]["resolved_positive"])
    assert failed_with_positive_prior["prior_diagnostic"][
        "enters_primary_host_claim"] is False
    assert ledger["claim_policy"]["per_age_claims_only"] is True
    assert ledger["claim_policy"]["pooled_cross_host_score_computed"] is False
    assert ledger["source_binding"]["report"]["sha256"] == hashlib.sha256(
        report_path.read_bytes()).hexdigest()
    assert ledger["source_binding"]["report"][
        "expected_sha256_verified"] is True
    assert ledger["source_binding"]["report"][
        "independent_sealed_auditor_recomputation_verified"] is True
    assert len(ledger["source_binding"]["adapter"]["sha256"]) == 64

    tex = tex_path.read_text()
    assert r"\newcommand{\SageMemClaimLedgerTable}" in tex
    table_tex = tex.split(
        r"\newcommand{\SageMemClaimLedgerTable}", 1)[1]
    for label in adapter.COHORT_LABELS.values():
        assert table_tex.count(label.replace("_", r"\_")) == 3
    assert r"\label{sage-mem-v1:complete-claim-ledger}" in table_tex
    assert r"\newcommand{\SageMemPrimaryResultSummary}" in tex
    assert r"\newcommand{\SageMemExecutionResultSummary}" in tex
    assert r"\newcommand{\SageMemClaimBoundary}" in tex
    assert r"\newcommand{\SageMemPrimaryPassList}" in tex
    assert r"\newcommand{\SageMemExecutionPassList}" in tex
    assert tex.count(r"\SageMemGatePass") > 15
    assert hashlib.sha256(tex.encode()).hexdigest() == \
        ledger["publication_artifacts"]["tex"]["sha256"]


def test_refuses_overwrite_and_resume_requires_byte_identical_outputs(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        _synthetic_sealed_lock: None) -> None:
    report = _report()
    report_path, json_path, tex_path = _execute(tmp_path, report)
    capsys.readouterr()
    original_json = json_path.read_bytes()
    original_tex = tex_path.read_bytes()
    phase_b_args, phase_b_root = _phase_b_cli(tmp_path)
    command = [
        "--report", str(report_path),
        "--spec", str(DEFAULT_SPEC),
        "--json-output", str(json_path),
        "--tex-output", str(tex_path),
        "--execute",
        "--expected-report-sha256",
        hashlib.sha256(report_path.read_bytes()).hexdigest(),
        *phase_b_args,
    ]
    with pytest.raises(adapter.SageMemReportAdapterError, match="overwrite"):
        adapter.main(
            command,
            recompute_formal_audit=lambda _spec: copy.deepcopy(report),
            publication_root=tmp_path,
            phase_b_study_root=phase_b_root)
    assert adapter.main(
        [*command, "--resume"],
        recompute_formal_audit=lambda _spec: copy.deepcopy(report),
        publication_root=tmp_path,
        phase_b_study_root=phase_b_root) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["publication"] == "validated-existing"

    changed = copy.deepcopy(report)
    changed["phase_a_grid_sha256"] = "9" * 64
    changed_payload = _write_report(report_path, changed)
    _, _, changed_phase_b_root = _materialize_phase_b_receipt(
        tmp_path, report_path, changed)
    changed_phase_b_args, _ = _phase_b_cli(tmp_path)
    changed_command = [
        "--report", str(report_path),
        "--spec", str(DEFAULT_SPEC),
        "--json-output", str(json_path),
        "--tex-output", str(tex_path),
        "--execute", "--expected-report-sha256",
        hashlib.sha256(changed_payload).hexdigest(),
        *changed_phase_b_args,
    ]
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="differs from authenticated report"):
        adapter.main(
            [*changed_command, "--resume"],
            recompute_formal_audit=lambda _spec: copy.deepcopy(changed),
            publication_root=tmp_path,
            phase_b_study_root=changed_phase_b_root)
    assert json_path.read_bytes() == original_json
    assert tex_path.read_bytes() == original_tex


def test_rejects_selective_or_tampered_claim_grid(
        tmp_path: Path, _synthetic_sealed_lock: None) -> None:
    selective = _report()
    del selective["cohorts"]["lewm_reacher_color"]["ages"]["15"]
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="exactly ages 4, 8, and 15"):
        _execute(tmp_path / "selective", selective)

    tampered = _report()
    age = tampered["cohorts"]["lewm_reacher_color"]["ages"]["4"]
    age["primary_host_claim_pass"] = False
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="primary pass flag is inconsistent"):
        _execute(tmp_path / "tampered", tampered)

    stale_lock = _report()
    stale_lock["cohorts"]["dinowm_pointmaze_goal"]["comparator_receipt"][
        "implementation_lock_sha256"] = "6" * 64
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="do not bind the sealed implementation lock"):
        _execute(tmp_path / "stale-lock", stale_lock)


def test_expected_report_and_protocol_identities_are_enforced(
        tmp_path: Path, _synthetic_sealed_lock: None) -> None:
    report = _report()
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="report SHA-256 differs"):
        _execute(
            tmp_path / "report-hash", report,
            "--expected-report-sha256", "0" * 64)
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="protocol fingerprint differs"):
        _execute(
            tmp_path / "protocol-hash", report,
            "--expected-protocol-fingerprint", "0" * 64)


@pytest.mark.parametrize(
    ("mutation", "message"), [
        ("injected", "no-injection boundary"),
        ("outcomes", "no-injection boundary"),
        ("report-pin", "selected formal report/grid"),
        ("verifier-pin", "verifier source.*differs"),
    ])
def test_phase_b_receipt_flags_and_pins_fail_closed(
        tmp_path: Path, _synthetic_sealed_lock: None,
        mutation: str, message: str) -> None:
    report = _report()
    _materialize_comparator_receipts(tmp_path, report)
    report_path = tmp_path / "report.json"
    report_payload = _write_report(report_path, report)
    receipt_path, _, phase_b_root = _materialize_phase_b_receipt(
        tmp_path, report_path, report)
    receipt = json.loads(receipt_path.read_text())
    if mutation == "injected":
        receipt["report_reproducer_injected"] = True
    elif mutation == "outcomes":
        receipt["outcome_values_emitted"] = True
    elif mutation == "report-pin":
        receipt["operator_pins"]["formal_report_sha256"] = "0" * 64
    else:
        receipt["operator_pins"]["verifier_source_sha256"] = "0" * 64
        receipt["authenticated_inventories"]["verifier_source"][
            "sha256"] = "0" * 64
    receipt_path.write_text(adapter._canonical_json(receipt))
    with pytest.raises(adapter.SageMemReportAdapterError, match=message):
        adapter.main([
            "--report", str(report_path), "--spec", str(DEFAULT_SPEC),
            "--json-output", str(tmp_path / "claim-ledger.json"),
            "--tex-output", str(tmp_path / "claim-ledger.tex"),
            "--execute", "--expected-report-sha256",
            hashlib.sha256(report_payload).hexdigest(),
            "--phase-b-receipt", str(receipt_path),
            "--expected-phase-b-receipt-sha256",
            hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        ], recompute_formal_audit=lambda _spec: copy.deepcopy(report),
            publication_root=tmp_path, phase_b_study_root=phase_b_root)


def test_phase_b_receipt_rejects_symlinked_parent_component(
        tmp_path: Path, _synthetic_sealed_lock: None) -> None:
    report = _report()
    _materialize_comparator_receipts(tmp_path, report)
    report_path = tmp_path / "report.json"
    report_payload = _write_report(report_path, report)
    receipt_path, receipt_sha, phase_b_root = _materialize_phase_b_receipt(
        tmp_path, report_path, report)
    parent = receipt_path.parent
    real_parent = parent.with_name("phase_b_real")
    parent.rename(real_parent)
    parent.symlink_to(real_parent.name, target_is_directory=True)
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="symlink component"):
        adapter.main([
            "--report", str(report_path), "--spec", str(DEFAULT_SPEC),
            "--json-output", str(tmp_path / "claim-ledger.json"),
            "--tex-output", str(tmp_path / "claim-ledger.tex"),
            "--execute", "--expected-report-sha256",
            hashlib.sha256(report_payload).hexdigest(),
            "--phase-b-receipt", str(receipt_path),
            "--expected-phase-b-receipt-sha256", receipt_sha,
        ], recompute_formal_audit=lambda _spec: copy.deepcopy(report),
            publication_root=tmp_path, phase_b_study_root=phase_b_root)


def test_execute_requires_hash_and_independent_audit_equality(
        tmp_path: Path, _synthetic_sealed_lock: None) -> None:
    report = _report()
    _materialize_comparator_receipts(tmp_path, report)
    report_path = tmp_path / "report.json"
    payload = _write_report(report_path, report)
    phase_b_path, phase_b_hash, phase_b_root = _materialize_phase_b_receipt(
        tmp_path, report_path, report)
    arguments = [
        "--report", str(report_path),
        "--spec", str(DEFAULT_SPEC),
        "--json-output", str(tmp_path / "ledger.json"),
        "--tex-output", str(tmp_path / "ledger.tex"),
        "--execute",
        "--phase-b-receipt", str(phase_b_path),
        "--expected-phase-b-receipt-sha256", phase_b_hash,
    ]
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="expected-report-sha256 is required"):
        adapter.main(
            arguments,
            recompute_formal_audit=lambda _spec: copy.deepcopy(report),
            publication_root=tmp_path,
            phase_b_study_root=phase_b_root)

    recomputed = copy.deepcopy(report)
    recomputed["identity_ledger_sha256"] = "9" * 64
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="independent sealed-auditor recomputation"):
        adapter.main(
            [*arguments, "--expected-report-sha256",
             hashlib.sha256(payload).hexdigest()],
            recompute_formal_audit=lambda _spec: recomputed,
            publication_root=tmp_path,
            phase_b_study_root=phase_b_root)
    assert not (tmp_path / "ledger.json").exists()
    assert not (tmp_path / "ledger.tex").exists()


@pytest.mark.parametrize("mutation", ("comparators", "protocol"))
def test_comparator_receipt_content_must_match_report_and_protocol(
        tmp_path: Path, _synthetic_sealed_lock: None,
        mutation: str) -> None:
    report = _report()
    _materialize_comparator_receipts(tmp_path, report)
    cohort = adapter.COHORTS[0]
    cohort_record = report["cohorts"][cohort]
    receipt = cohort_record["comparator_receipt"]
    selection_path = Path(receipt["locked_comparator_receipt"]["path"])
    selection = json.loads(selection_path.read_text())
    if mutation == "comparators":
        selection["locked_comparators"]["retention"] = "lstm"
    else:
        selection["protocol_fingerprint"] = "0" * 64
    selection_path.write_text(adapter._canonical_json(selection))
    selection_identity = _identity(selection_path)
    receipt["locked_comparator_receipt"] = selection_identity
    preparation_path = Path(receipt["preparation_receipt"]["path"])
    preparation = json.loads(preparation_path.read_text())
    preparation["boundaries"]["locked_comparator_receipt"] = \
        selection_identity
    preparation_path.write_text(adapter._canonical_json(preparation))
    receipt["preparation_receipt"] = _identity(preparation_path)

    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="parsed locked comparators/protocol identity"):
        _execute(tmp_path, report, materialize_receipts=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count-gap", "parameter gap was not derived from counts"),
        ("target", "target differs from the sealed protocol"),
        ("bool-zero", "must be an integer"),
        ("flop-gap", "FLOP gap was not derived from the median"),
    ],
)
def test_resource_fairness_is_recomputed(
        tmp_path: Path, _synthetic_sealed_lock: None,
        mutation: str, message: str) -> None:
    report = _report()
    resources = report["cohorts"][adapter.COHORTS[0]]["resource_enforcement"]
    if mutation == "count-gap":
        resources["gru"]["trainable_parameters"] += 1000
    elif mutation == "target":
        resources["gru"]["target_parameters"] = 1
    elif mutation == "bool-zero":
        resources["none"]["trainable_parameters"] = False
    else:
        resources["gru"]["forward_flops_per_episode"] = 1050.0
    with pytest.raises(adapter.SageMemReportAdapterError, match=message):
        _execute(tmp_path, report)


def test_bootstrap_seed_and_metric_domains_are_exact(
        tmp_path: Path, _synthetic_sealed_lock: None) -> None:
    wrong_seed = _report()
    wrong_seed["cohorts"][adapter.COHORTS[0]]["ages"]["4"][
        "host_full_accuracy"]["seed"] += 1
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="bootstrap seed differs"):
        _execute(tmp_path / "seed", wrong_seed)

    impossible_accuracy = _report()
    impossible_accuracy["cohorts"][adapter.COHORTS[0]]["ages"]["4"][
        "host_full_accuracy"]["upper"] = 1.01
    with pytest.raises(adapter.SageMemReportAdapterError, match="above 1.0"):
        _execute(tmp_path / "accuracy", impossible_accuracy)

    impossible_difference = _report()
    impossible_difference["cohorts"][adapter.COHORTS[0]]["ages"]["4"][
        "host_full_vs_none"]["upper"] = 1.01
    with pytest.raises(adapter.SageMemReportAdapterError, match="above 1.0"):
        _execute(tmp_path / "difference", impossible_difference)

    impossible_ratio = _report()
    mse = impossible_ratio["cohorts"][adapter.COHORTS[0]]["ages"]["4"][
        "next_feature_relative_excess"]
    mse.update({"point": -1.0, "lower": -1.01, "upper": -0.99})
    with pytest.raises(adapter.SageMemReportAdapterError, match="below -1.0"):
        _execute(tmp_path / "ratio", impossible_ratio)

    boundary = _report()
    boundary_mse = boundary["cohorts"][adapter.COHORTS[0]]["ages"]["4"][
        "next_feature_relative_excess"]
    boundary_mse.update({"point": -0.99, "lower": -1.0, "upper": -0.98})
    _execute(tmp_path / "boundary", boundary)

    difference_boundary = _report()
    contrast = difference_boundary["cohorts"][adapter.COHORTS[0]][
        "ages"]["4"]["raw_context_reference"]["long16_minus_short3"]
    contrast.update({"point": 0.99, "lower": 0.98, "upper": 1.0})
    _execute(tmp_path / "difference-boundary", difference_boundary)


def test_resume_repairs_only_an_authentic_half_pair(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        _synthetic_sealed_lock: None) -> None:
    report = _report()
    report_path, json_path, tex_path = _execute(tmp_path, report)
    capsys.readouterr()
    expected_tex = tex_path.read_bytes()
    tex_path.unlink()
    phase_b_args, phase_b_root = _phase_b_cli(tmp_path)
    command = [
        "--report", str(report_path),
        "--spec", str(DEFAULT_SPEC),
        "--json-output", str(json_path),
        "--tex-output", str(tex_path),
        "--execute", "--resume",
        "--expected-report-sha256",
        hashlib.sha256(report_path.read_bytes()).hexdigest(),
        *phase_b_args,
    ]
    assert adapter.main(
        command,
        recompute_formal_audit=lambda _spec: copy.deepcopy(report),
        publication_root=tmp_path,
        phase_b_study_root=phase_b_root) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["publication"] == "repaired-missing"
    assert tex_path.read_bytes() == expected_tex

    tex_path.unlink()
    json_path.write_bytes(json_path.read_bytes() + b"tamper")
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="differs from authenticated report"):
        adapter.main(
            command,
            recompute_formal_audit=lambda _spec: copy.deepcopy(report),
            publication_root=tmp_path,
            phase_b_study_root=phase_b_root)
    assert not tex_path.exists()


def test_execute_rejects_outputs_outside_or_through_symlinked_publication_root(
        tmp_path: Path, _synthetic_sealed_lock: None) -> None:
    report = _report()
    _materialize_comparator_receipts(tmp_path, report)
    report_path = tmp_path / "report.json"
    report_payload = _write_report(report_path, report)
    phase_b_path, phase_b_hash, phase_b_root = _materialize_phase_b_receipt(
        tmp_path, report_path, report)
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    base = [
        "--report", str(report_path), "--spec", str(DEFAULT_SPEC),
        "--execute", "--expected-report-sha256",
        hashlib.sha256(report_payload).hexdigest(),
        "--phase-b-receipt", str(phase_b_path),
        "--expected-phase-b-receipt-sha256", phase_b_hash,
    ]
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="must remain inside the publication root"):
        adapter.main([
            *base, "--json-output", str(outside / "ledger.json"),
            "--tex-output", str(allowed / "ledger.tex"),
        ], recompute_formal_audit=lambda _spec: copy.deepcopy(report),
           publication_root=allowed, phase_b_study_root=phase_b_root)
    assert not (outside / "ledger.json").exists()

    linked = allowed / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(adapter.SageMemReportAdapterError,
                       match="symlink component"):
        adapter.main([
            *base, "--json-output", str(linked / "ledger.json"),
            "--tex-output", str(allowed / "ledger.tex"),
        ], recompute_formal_audit=lambda _spec: copy.deepcopy(report),
           publication_root=allowed, phase_b_study_root=phase_b_root)
    assert not (outside / "ledger.json").exists()


def test_generated_tex_has_balanced_structure_and_compiles_when_available(
        tmp_path: Path, _synthetic_sealed_lock: None) -> None:
    _, _, tex_path = _execute(tmp_path, _report())
    tex = tex_path.read_text()
    assert tex.count("{") == tex.count("}")
    table_rows = [
        line for line in tex.splitlines()
        if line.endswith(r"\\")
    ]
    assert len(table_rows) == 16
    assert all(line.count("&") == 8 for line in table_rows)
    assert sum(label in line for label in adapter.COHORT_LABELS.values()
               for line in table_rows[1:]) == 15

    pdflatex_value = shutil.which("pdflatex")
    tinytex_pdflatex = (
        Path.home() / ".TinyTeX/bin/x86_64-linux/pdflatex")
    pdflatex = (Path(pdflatex_value) if pdflatex_value
                else tinytex_pdflatex if tinytex_pdflatex.is_file()
                else None)
    if pdflatex is None:
        return
    wrapper = tmp_path / "compile-generated.tex"
    wrapper.write_text(
        "\\documentclass{article}\n"
        f"\\input{{{tex_path.as_posix()}}}\n"
        "\\begin{document}\n"
        "\\begin{table*}\\centering\\small\n"
        "\\SageMemClaimLedgerTable\n"
        "\\end{table*}\n"
        "\\end{document}\n")
    completed = subprocess.run(
        [str(pdflatex), "-halt-on-error", "-interaction=nonstopmode",
         wrapper.name], cwd=tmp_path, text=True,
        env={
            **os.environ,
            "PATH": f"{pdflatex.parent}:{os.environ.get('PATH', '')}",
        },
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert completed.returncode == 0, completed.stdout
