from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from scripts import audit_paper_a_sage_mem_binding as audit
from scripts import plot_sage_mem_v1_claims as plotter_module


def _identity(root: Path, path: Path, **extra: object) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **extra,
    }


def _write_canonical(path: Path, value: object) -> bytes:
    payload = audit.canonical_json(value).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _write_compiled_evidence(
        root: Path, *, main: Path, appendix: Path, ledger_tex: Path | None,
        sentinel_rendered: bool, extra_inputs: tuple[Path, ...] = ()) -> None:
    paper = main.parent
    inputs = [main, appendix, *extra_inputs]
    if ledger_tex is not None:
        inputs.append(ledger_tex)
    aux = main.with_suffix(".aux")
    fls = main.with_suffix(".fls")
    pdf = main.with_suffix(".pdf")
    aux.write_text(
        (rf"\newlabel{{{audit.LEDGER_SENTINEL}}}{{}}{{1}}" + "\n")
        if sentinel_rendered else "\\relax\n")
    fls.write_text(
        f"PWD {paper}\n" + "".join(
            f"INPUT {path.relative_to(paper)}\n" for path in inputs))
    pdf.write_bytes(b"%PDF-1.4\n% synthetic compiled fixture\n%%EOF\n")
    newest = max(path.stat().st_mtime_ns for path in inputs)
    fresh = newest + 1_000_000
    for path in (aux, fls, pdf):
        os.utime(path, ns=(fresh, fresh))


def _ci(lower: float, *, confidence: float = 0.95,
        seed: int = 1) -> dict[str, object]:
    return {
        "point": lower + 0.02,
        "lower": lower,
        "upper": lower + 0.04,
        "confidence": confidence,
        "draws": 20_000,
        "seed": seed,
        "resampling_unit": "paired formal seed x native episode cluster",
        "class_profile_stratified": True,
        "pairing_preserved": True,
    }


def _execution_program(*, eligible: int = 0,
                       passing: dict[int, int] | None = None
                       ) -> dict[str, object]:
    passing = passing or {age: 0 for age in audit.AGES}
    permitted = eligible >= 2
    return {
        "optional": True,
        "eligible_cohorts": eligible,
        "minimum_eligible_cohorts": 2,
        "program_claim_permitted": permitted,
        "per_age": {
            str(age): {
                "eligible_cohorts": eligible,
                "cohorts_passing": passing[age],
                "claim_permitted": permitted,
                "claim_pass": permitted and passing[age] >= 2,
            }
            for age in audit.AGES
        },
        "cross_age_conjunction_computed": False,
        "program_claim_pass": None,
    }


def _row(*, primary: bool, execution: bool | None = None,
         seed: int = 1
         ) -> dict[str, object]:
    gates = {key: True for key in audit.GATE_KEYS}
    if not primary:
        gates["host_vs_locked_comparator"] = False
    controls = {
        key: _ci(0.04, seed=seed + 10 + index)
        for index, key in enumerate(plotter_module.CONTROL_KEYS)
    }
    execution_record = None
    if execution is not None:
        lower = 0.04 if execution else 0.02
        execution_record = {
            "full_vs_locked_comparator": _ci(lower, seed=seed + 40),
            "full_vs_none": _ci(0.04, seed=seed + 41),
            "full_vs_random": _ci(0.04, seed=seed + 42),
            "random_reference":
                "sealed per-episode arm-blind random-success deck",
            "random_reference_is_cohort_rate": False,
            "oracle_success": 0.95,
            "random_success_mean": 0.25,
            "pass": execution,
        }
    return {
        "primary_endpoint": "frozen-host full correctness",
        "host_full_accuracy": _ci(0.50, seed=seed),
        "host_full_vs_locked_comparator": _ci(
            0.06 if primary else 0.04, seed=seed + 2),
        "host_full_vs_reset": _ci(0.04, seed=seed + 3),
        "host_full_vs_none": _ci(0.04, seed=seed + 4),
        "reset_to_full_mse_ratio": 1.2,
        "mechanism_controls": controls,
        "next_feature_relative_excess": _ci(
            0.02, confidence=0.90, seed=seed + 20),
        "gates": gates,
        "primary_host_claim_pass": primary,
        "prior_diagnostic": {
            "role": "diagnostic-only; cannot establish host use",
            "accuracy": _ci(0.50, seed=seed + 1),
            "vs_locked_comparator": _ci(0.02, seed=seed + 5),
            "resolved_positive": True,
            "enters_primary_host_claim": False,
        },
        "raw_context_reference": {
            "short3_accuracy": 0.30,
            "long16_accuracy": 0.45,
            "long16_minus_short3": _ci(0.02, seed=seed + 30),
            "resolved_long_context_gain": True,
            "separate_from_parameter_matched_grid": True,
        },
        "execution": execution_record,
    }


def _build_fixture(
        root: Path, *, positives: set[tuple[str, int]] | None = None,
        integrated: bool = True, ledger_in_appendix: bool = True,
        execution_cohorts: int = 0) -> dict[str, object]:
    positives = positives if positives is not None \
        else {(audit.COHORTS[0], audit.AGES[0])}
    scripts = root / "scripts"
    configs = root / "configs"
    outputs = root / "outputs/sage_mem_v1"
    generated = root / "paper_a/generated_results"
    for directory in (scripts, configs, outputs, generated):
        directory.mkdir(parents=True, exist_ok=True)
    formal_auditor = scripts / "audit_sage_mem_v1_formal.py"
    adapter = scripts / "summarize_sage_mem_v1_report.py"
    plotter = scripts / "plot_sage_mem_v1_claims.py"
    protocol = configs / "sage_mem_v1.yaml"
    lock = outputs / "protocol_lock.json"
    amendment = configs / "sage_mem_v1_formal_amendment.yaml"
    formal_auditor.write_text("# sealed formal auditor\n")
    adapter.write_text("# authenticated report adapter\n")
    shutil.copyfile(Path(plotter_module.__file__), plotter)
    protocol.write_text("study: sage-mem-v1\n")
    lock.write_text('{"status":"sealed"}\n')
    amendment.write_text("status: locked-before-formal\n")

    cohorts: dict[str, object] = {}
    report_rows: dict[tuple[str, int], dict[str, object]] = {}
    execution_passing = {age: 0 for age in audit.AGES}
    for cohort_index, cohort in enumerate(audit.COHORTS):
        supplied = cohort_index < execution_cohorts
        ages = {}
        pass_by_age = {}
        primary_values = []
        for age in audit.AGES:
            execution = (True if supplied else None)
            row = _row(primary=(cohort, age) in positives,
                       execution=execution,
                       seed=1000 + cohort_index * 100 + age)
            report_rows[(cohort, age)] = row
            ages[str(age)] = row
            pass_by_age[str(age)] = execution
            primary_values.append((cohort, age) in positives)
            if execution is True:
                execution_passing[age] += 1
        cohorts[cohort] = {
            "locked_comparators": {
                "retention": "gru", "next_feature": "gru",
                "execution": "gru",
            },
            "comparator_receipt": {
                "formal_preparation_manifest_sha256":
                    f"{cohort_index + 1:x}" * 64,
                "implementation_lock_sha256": "d" * 64,
                "custody_registry_sha256": "e" * 64,
                "preparation_receipt": {
                    "path": f"outputs/preparation-{cohort_index}.json",
                    "size": 10,
                    "sha256": "f" * 64,
                },
                "locked_comparator_receipt": {
                    "path": f"outputs/comparator-{cohort_index}.json",
                    "size": 10,
                    "sha256": "a" * 64,
                },
            },
            "backend_admission": {"backend": "synthetic"},
            "resource_enforcement": {"verified": True},
            "ages": ages,
            "all_registered_ages_primary_host_claim_pass":
                all(primary_values),
            "execution_supplied": supplied,
            "execution_pass_by_age": pass_by_age,
        }
    report = {
        "schema": audit.REPORT_SCHEMA,
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
        "execution_program": _execution_program(
            eligible=execution_cohorts, passing=execution_passing),
        "prior_can_substitute_for_host_output": False,
        "per_age_claims_only": True,
        "pooled_cross_host_score_computed": False,
        "universal_success_claim_permitted": False,
    }
    report_path = outputs / "formal_audit/report.json"
    report_payload = _write_canonical(report_path, report)

    claim_rows = []
    for cohort in audit.COHORTS:
        for age in audit.AGES:
            source = report_rows[(cohort, age)]
            row = {
                "cohort": cohort,
                "cohort_label": audit.COHORT_LABELS[cohort],
                "age": age,
                **copy.deepcopy(source),
                "locked_comparators": {
                    "retention": "gru", "next_feature": "gru",
                    "execution": "gru",
                },
            }
            claim_rows.append(row)
    passing = len(positives)
    cohort_summaries = {}
    for cohort in audit.COHORTS:
        cohort_record = cohorts[cohort]
        cohort_rows = [row for row in claim_rows if row["cohort"] == cohort]
        cohort_summaries[cohort] = {
            "cohort_label": audit.COHORT_LABELS[cohort],
            "locked_comparators": copy.deepcopy(
                cohort_record["locked_comparators"]),
            "comparator_receipt": copy.deepcopy(
                cohort_record["comparator_receipt"]),
            "backend_admission": copy.deepcopy(
                cohort_record["backend_admission"]),
            "resource_enforcement_verified": True,
            "registered_age_rows": 3,
            "rows_passing_primary_host_claim": sum(
                row["primary_host_claim_pass"] for row in cohort_rows),
            "all_registered_ages_primary_host_claim_pass": all(
                row["primary_host_claim_pass"] for row in cohort_rows),
            "execution_supplied": cohort_record["execution_supplied"],
            "execution_pass_by_age": copy.deepcopy(
                cohort_record["execution_pass_by_age"]),
        }
    ledger = {
        "schema": audit.LEDGER_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "paper-claim-ledger",
        "status": "complete",
        "integrity_completion": {
            "status": "complete",
            "meaning": "authenticated; not a scientific pass",
            "phase_a_cells_verified": 600,
            "finalized_cells_verified": 600,
            "comparators_verified": 5,
            "resources_verified": 600,
            "raw_context_references_verified": 50,
            "phase_a_grid_sha256": "a" * 64,
            "identity_ledger_sha256": "b" * 64,
        },
        "claim_policy": {
            "per_age_claims_only": True,
            "registered_cohorts": list(audit.COHORTS),
            "registered_ages": list(audit.AGES),
            "registered_claim_rows": 15,
            "positive_rows_may_not_be_selected_or_omitted": True,
            "prior_can_substitute_for_host_output": False,
            "pooled_cross_host_score_computed": False,
            "universal_success_claim_permitted": False,
            "thresholds": {
                "host_gain": 0.05,
                "reset_gain": 0.03,
                "reset_mse_ratio_max": 2.0,
                "mechanism_gain": 0.03,
                "mse_relative_margin": 0.10,
                "execution_gain": 0.03,
                "execution_oracle_gate": 0.90,
            },
        },
        "scientific_result": {
            "status": "evaluated",
            "primary_claim_rows_total": 15,
            "primary_claim_rows_passing": passing,
            "primary_claim_rows_failing": 15 - passing,
            "any_primary_claim_row_passed": passing > 0,
            "all_primary_claim_rows_passed": passing == 15,
            "meaning": "complete grid",
        },
        "cohort_summaries": cohort_summaries,
        "claim_rows": claim_rows,
        "execution_program": copy.deepcopy(report["execution_program"]),
        "source_binding": {
            "report": {
                **_identity(root, report_path),
                "schema": audit.REPORT_SCHEMA,
                "expected_sha256_verified": True,
                "independent_sealed_auditor_recomputation_verified": True,
                "standard_roots": {
                    "phase_a": "outputs/sage_mem_v1",
                    "finalized": "outputs/sage_mem_v1/formal_finalized",
                    "preparation": "outputs/sage_mem_v1/formal_preparation",
                    "raw_context": "outputs/sage_mem_v1/raw_context_phase_a",
                },
            },
            "protocol": {
                **_identity(root, protocol),
                "fingerprint": "f" * 64,
                "implementation_lock": {
                    **_identity(root, lock), "status": "sealed"},
                "formal_amendment": {
                    **_identity(root, amendment),
                    "status":
                        "locked-before-development-selection-or-formal-data",
                },
                "report_schema_repeats_protocol_fingerprint": False,
                "binding_note": "synthetic binding",
            },
            "formal_auditor": {
                **_identity(root, formal_auditor),
                "sealed_by_implementation_lock": True,
            },
            "adapter": {
                "path": str(adapter.relative_to(root)),
                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            },
        },
        "publication_artifacts": {},
    }
    ledger_tex_path = generated / "sage_mem_v1_claim_ledger.tex"
    # The TeX hash is the only publication identity needed to finish the JSON.
    provisional = audit.render_expected_ledger_tex(ledger).encode()
    ledger["publication_artifacts"] = {
        "tex": {
            "path": str(ledger_tex_path.relative_to(root)),
            "sha256": hashlib.sha256(provisional).hexdigest(),
        },
    }
    # publication_artifacts is not itself rendered, so the provisional bytes
    # are already final.
    ledger_tex_path.write_bytes(provisional)
    ledger_path = generated / "sage_mem_v1_claim_ledger.json"
    ledger_payload = _write_canonical(ledger_path, ledger)

    paper = root / "paper_a"
    main = paper / "main.tex"
    appendix = paper / "appendix.tex"
    if integrated:
        main.write_text(
            "\\input{generated_results/sage_mem_v1_claim_ledger.tex}\n"
            "\\begin{document}\n"
            "\\SageMemPrimaryResultSummary\\ "
            "\\SageMemClaimBoundary\n"
            "\\appendix\n\\input{appendix.tex}\n\\end{document}\n")
        appendix.write_text(
            ("\\SageMemClaimLedgerTable\n" if ledger_in_appendix else
             "The complete table is intentionally absent.\n"))
    else:
        main.write_text(
            "\\begin{document}\nNo new result artifact is integrated.\n"
            "\\appendix\n\\input{appendix.tex}\n\\end{document}\n")
        appendix.write_text("No SAGE result table.\n")
    _write_compiled_evidence(
        root, main=main, appendix=appendix,
        ledger_tex=ledger_tex_path if integrated else None,
        sentinel_rendered=integrated and ledger_in_appendix)
    return {
        "root": root,
        "report": report_path.relative_to(root),
        "report_sha": hashlib.sha256(report_payload).hexdigest(),
        "ledger": ledger_path.relative_to(root),
        "ledger_sha": hashlib.sha256(ledger_payload).hexdigest(),
        "ledger_tex": ledger_tex_path.relative_to(root),
        "main": main.relative_to(root),
        "appendix": appendix,
        "ledger_value": ledger,
        "expected_publication": {
            "report_payload": report_payload,
            "ledger_payload": ledger_payload,
            "ledger_tex_payload": provisional,
        },
    }


def _audit(fixture: dict[str, object], **kwargs: object) -> dict[str, object]:
    kwargs.setdefault(
        "publication_recomputer",
        lambda _root, _report, _ledger, _tex:
        copy.deepcopy(fixture["expected_publication"]))
    kwargs.setdefault("compiler_runner", _synthetic_compiler_runner)
    return audit.audit_binding(
        fixture["root"], report=fixture["report"],
        ledger_json=fixture["ledger"], ledger_tex=fixture["ledger_tex"],
        main_tex=fixture["main"],
        expected_report_sha256=fixture["report_sha"],
        expected_ledger_sha256=fixture["ledger_sha"], **kwargs)


def _synthetic_compiler_runner(root: Path, main: Path) -> dict[str, object]:
    """Test-only compiler hook: mark prebuilt fixtures as newly emitted."""

    started = time.time_ns()
    stamp = max(time.time_ns(), started + 1)
    for suffix in (".aux", ".fls", ".pdf"):
        path = Path(main).with_suffix(suffix)
        os.utime(path, ns=(stamp, stamp))
    return {
        "engine": "synthetic-test-compiler",
        "engine_sha256": "c" * 64,
        "command": ["synthetic-test-compiler", str(main)],
        "started_ns": started,
        "completed_ns": max(time.time_ns(), stamp),
        "returncode": 0,
    }


def _non_building_compiler_runner(root: Path, main: Path) \
        -> dict[str, object]:
    """Return success without emitting anything, for freshness rejection."""

    started = time.time_ns()
    return {
        "engine": "non-building-test-compiler",
        "engine_sha256": "d" * 64,
        "command": ["non-building-test-compiler", str(main)],
        "started_ns": started,
        "completed_ns": max(time.time_ns(), started),
        "returncode": 0,
    }


def _refresh_compile(
        fixture: dict[str, object], *, sentinel_rendered: bool = True,
        extra_inputs: tuple[Path, ...] = ()) -> None:
    root = fixture["root"]
    _write_compiled_evidence(
        root, main=root / fixture["main"], appendix=fixture["appendix"],
        ledger_tex=root / fixture["ledger_tex"],
        sentinel_rendered=sentinel_rendered, extra_inputs=extra_inputs)


def test_preview_reads_and_writes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing"
    assert audit.main([
        "--root", str(tmp_path),
        "--report", str(missing / "report.json"),
        "--ledger-json", str(missing / "ledger.json"),
        "--ledger-tex", str(missing / "ledger.tex"),
        "--main-tex", str(missing / "main.tex"),
        "--receipt", str(missing / "receipt.json"),
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview"] is True
    assert preview["no_files_read"] is True
    assert preview["no_files_written"] is True
    assert preview["required_claim_rows"] == 15
    assert not missing.exists()


def test_real_tinytex_canonical_build_smoke_when_installed(
        tmp_path: Path) -> None:
    latexmk = audit.discover_latexmk()
    if latexmk is None:
        pytest.skip("latexmk/TinyTeX is not installed")
    paper = tmp_path / "paper_a"
    paper.mkdir()
    main = paper / "main.tex"
    main.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}binding smoke\\end{document}\n")
    receipt = audit.run_canonical_paper_build(tmp_path, main)
    first_pdf_sha = hashlib.sha256((paper / "main.pdf").read_bytes()).hexdigest()
    assert Path(receipt["engine"]) == latexmk
    assert receipt["returncode"] == 0
    assert (paper / "main.aux").stat().st_mtime_ns >= receipt["started_ns"]
    assert (paper / "main.fls").stat().st_mtime_ns >= receipt["started_ns"]
    assert (paper / "main.pdf").stat().st_mtime_ns >= receipt["started_ns"]
    assert (paper / "main.pdf").read_bytes().startswith(b"%PDF-")
    audit.run_canonical_paper_build(tmp_path, main)
    assert hashlib.sha256((paper / "main.pdf").read_bytes()).hexdigest() \
        == first_pdf_sha


def test_complete_mixed_grid_and_appendix_ledger_pass(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    result = _audit(fixture)
    assert result["status"] == "verified"
    assert result["coverage"]["claim_rows_verified"] == 15
    assert result["authorization"]["primary_positive_rows"] == 1
    assert result["integration"]["integrated"] is True
    assert result["integration"][
        "full_negative_or_mixed_ledger_satisfied"] is True
    assert result["integration"]["ledger_table_render_phases"] == ["appendix"]


def test_no_positive_result_cannot_enter_paper(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "integrated", positives=set())
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="without a registered positive"):
        _audit(fixture)

    fixture = _build_fixture(
        tmp_path / "not-integrated", positives=set(), integrated=False)
    result = _audit(fixture)
    assert result["authorization"]["publication_authorized"] is False
    assert result["integration"]["integrated"] is False


def test_permitted_execution_pass_can_authorize_integration(
        tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path, positives=set(), execution_cohorts=2)
    result = _audit(fixture)
    assert result["authorization"]["primary_positive_rows"] == 0
    assert result["authorization"][
        "permitted_execution_positive_rows"] == 6
    assert result["authorization"]["publication_authorized"] is True


def test_exact_15_row_coverage_and_report_equality_are_fail_closed(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    ledger_path = tmp_path / fixture["ledger"]
    ledger = json.loads(ledger_path.read_text())
    ledger["claim_rows"] = ledger["claim_rows"][:-1]
    payload = _write_canonical(ledger_path, ledger)
    fixture["ledger_sha"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="independent sealed publication recomputation"):
        _audit(fixture)

    fixture = _build_fixture(tmp_path / "tampered")
    ledger_path = fixture["root"] / fixture["ledger"]
    ledger = json.loads(ledger_path.read_text())
    ledger["claim_rows"][0]["primary_host_claim_pass"] = False
    payload = _write_canonical(ledger_path, ledger)
    fixture["ledger_sha"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="independent sealed publication recomputation"):
        _audit(fixture)


def test_self_consistent_forged_gate_fails_independent_recomputation(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    root = fixture["root"]
    cohort = audit.COHORTS[0]
    age = audit.AGES[1]
    report_path = root / fixture["report"]
    report = json.loads(report_path.read_text())
    report_row = report["cohorts"][cohort]["ages"][str(age)]
    assert report_row["host_full_vs_locked_comparator"]["lower"] == 0.04
    assert report_row["gates"]["host_vs_locked_comparator"] is False
    report_row["gates"]["host_vs_locked_comparator"] = True
    report_row["primary_host_claim_pass"] = True
    report_payload = _write_canonical(report_path, report)
    fixture["report_sha"] = hashlib.sha256(report_payload).hexdigest()

    ledger_path = root / fixture["ledger"]
    ledger = json.loads(ledger_path.read_text())
    ledger_row = next(
        row for row in ledger["claim_rows"]
        if row["cohort"] == cohort and row["age"] == age)
    ledger_row["gates"]["host_vs_locked_comparator"] = True
    ledger_row["primary_host_claim_pass"] = True
    ledger["cohort_summaries"][cohort][
        "rows_passing_primary_host_claim"] += 1
    ledger["scientific_result"]["primary_claim_rows_passing"] += 1
    ledger["scientific_result"]["primary_claim_rows_failing"] -= 1
    ledger["source_binding"]["report"].update(
        _identity(root, report_path))
    ledger_tex_path = root / fixture["ledger_tex"]
    tex_payload = audit.render_expected_ledger_tex(ledger).encode()
    ledger_tex_path.write_bytes(tex_payload)
    ledger["publication_artifacts"]["tex"]["sha256"] = \
        hashlib.sha256(tex_payload).hexdigest()
    ledger_payload = _write_canonical(ledger_path, ledger)
    fixture["ledger_sha"] = hashlib.sha256(ledger_payload).hexdigest()
    _refresh_compile(fixture)

    with pytest.raises(audit.SageMemBindingAuditError,
                       match="independent sealed publication recomputation"):
        _audit(fixture)


def test_mixed_integration_requires_full_table_in_appendix(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, ledger_in_appendix=False)
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="complete 15-row ledger table in the appendix"):
        _audit(fixture)


def test_inactive_ledger_macro_does_not_count_as_rendered(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    fixture["appendix"].write_text(
        r"\iffalse\SageMemClaimLedgerTable\fi" + "\n")
    _refresh_compile(fixture, sentinel_rendered=False)
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="AUX does not prove"):
        _audit(fixture)


def test_paper_cannot_spoof_generated_ledger_sentinel(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    fixture["appendix"].write_text(
        rf"\label{{{audit.LEDGER_SENTINEL}}}" + "\n"
        + r"\SageMemClaimLedgerTable" + "\n")
    _refresh_compile(fixture, sentinel_rendered=True)
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="sentinel may appear only"):
        _audit(fixture)


def test_stale_compiled_pdf_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    appendix = fixture["appendix"]
    pdf = fixture["root"] / "paper_a/main.pdf"
    newer = pdf.stat().st_mtime_ns + 10_000_000
    os.utime(appendix, ns=(newer, newer))
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="stale relative to sources"):
        _audit(fixture, compiler_runner=_non_building_compiler_runner)


def test_externally_touched_fabricated_build_artifacts_are_rejected(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    paper = fixture["root"] / "paper_a"
    (paper / "main.pdf").write_bytes(
        b"%PDF-1.4\nexternally fabricated\n%%EOF\n")
    future = time.time_ns() + 10_000_000_000
    for suffix in ("aux", "fls", "pdf"):
        os.utime(paper / f"main.{suffix}", ns=(future, future))
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="stale relative to sources"):
        _audit(fixture, compiler_runner=_non_building_compiler_runner)


def test_build_time_tex_source_mutation_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    def mutate_then_build(root: Path, main: Path) -> dict[str, object]:
        appendix = Path(root) / "paper_a/appendix.tex"
        appendix.write_text(
            appendix.read_text() + "Build-time source mutation.\n")
        return _synthetic_compiler_runner(root, main)

    with pytest.raises(audit.SageMemBindingAuditError,
                       match="source graph changed during"):
        _audit(fixture, compiler_runner=mutate_then_build)


def test_alternate_main_and_symlink_inputs_are_rejected(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "alternate")
    dummy = fixture["root"] / "paper_a/dummy.tex"
    dummy.write_text("\\begin{document}clean dummy\\end{document}\n")
    fixture["main"] = dummy.relative_to(fixture["root"])
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="canonical root/paper_a/main.tex"):
        _audit(fixture)

    fixture = _build_fixture(tmp_path / "symlink")
    original = fixture["root"] / fixture["ledger"]
    linked = original.with_name("linked-ledger.json")
    linked.symlink_to(original.name)
    fixture["ledger"] = linked.relative_to(fixture["root"])
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="symlink component"):
        _audit(fixture)


@pytest.mark.parametrize("sentence", [
    "Against no-state, SAGE-Mem universally outperforms every carrier.",
    "SAGE-Mem demonstrates the best pooled cross-host score.",
    "SAGE-Mem enables native-planner success.",
    ("SAGE-Mem does not merely match one baseline; it universally "
     "outperforms every carrier."),
    "Our method universally dominates every carrier.",
    "SAGE-Mem has a pooled cross-host advantage.",
    "SAGE-Mem drives the native planner.",
    "SAGE-Mem does not fail and achieves 99% success.",
    "SAGE-Mem is 99 percentage points ahead.",
    "SAGE-Mem gains [-4.83, 4.42] pp on the pooled endpoint.",
    "Our method is 7 percentage-points ahead.",
    "The proposed carrier passes DINO-WM PointMaze at age 15.",
])
def test_raw_positive_claim_language_is_rejected(
        tmp_path: Path, sentence: str) -> None:
    fixture = _build_fixture(tmp_path)
    appendix = fixture["appendix"]
    appendix.write_text(appendix.read_text() + sentence + "\n")
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="must use only authenticated generated macros"):
        _audit(fixture)


def test_negated_claim_boundaries_are_allowed(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    appendix = fixture["appendix"]
    appendix.write_text(
        appendix.read_text()
        + "SAGE-Mem does not establish native-planner success.\n"
          "SAGE-Mem is not a universally superior architecture.\n"
          "We do not compute a pooled cross-host score.\n"
          "This is not evidence of native planning.\n"
          "No pooled cross-host score is computed.\n"
          "The audit does not test native planning.\n")
    _refresh_compile(fixture)
    assert _audit(fixture)["status"] == "verified"


def test_one_positive_row_does_not_authorize_raw_claim_for_another_row(
        tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path, positives={(audit.COHORTS[0], audit.AGES[0])})
    appendix = fixture["appendix"]
    appendix.write_text(
        appendix.read_text()
        + "SAGE-Mem passes DINO-WM PointMaze at age 15 with a 99 pp gain.\n")
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="must use only authenticated generated macros"):
        _audit(fixture)


def test_any_sage_macro_is_an_artifact_reference(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, positives=set(), integrated=False)
    main = tmp_path / fixture["main"]
    main.write_text(main.read_text().replace(
        "No new result artifact is integrated.",
        r"The passing count is \SageMemPrimaryRowsPassing."))
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="without a registered positive"):
        _audit(fixture)


def test_optional_figure_manifest_is_independently_regenerated(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    root = fixture["root"]
    ledger = root / fixture["ledger"]
    output = root / "paper_a/figures"
    subprocess.run([
        sys.executable, str(root / audit.PLOTTER_SOURCE),
        "--ledger", str(ledger), "--output-dir", str(output),
        "--prefix", "sage_mem_v1",
        "--expected-ledger-sha256", fixture["ledger_sha"], "--execute",
    ], check=True, capture_output=True, text=True)
    manifest_path = output / "sage_mem_v1_plot_manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    figure = output / "sage_mem_v1_claim_ladder.pdf"
    main = root / fixture["main"]
    main.write_text(main.read_text().replace(
        r"\SageMemPrimaryResultSummary\ ",
        r"\SageMemPrimaryResultSummary\ "
        "\\graphicspath{{figures/}}"
        "\\includegraphics{sage_mem_v1_claim_ladder.pdf}"))
    _refresh_compile(fixture, extra_inputs=(figure,))
    result = _audit(
        fixture,
        figure_manifests=(manifest_path.relative_to(root),),
        expected_figure_manifest_sha256=(
            hashlib.sha256(manifest_payload).hexdigest(),))
    assert hashlib.sha256(figure.read_bytes()).hexdigest() in {
        item["sha256"] for item in result["identities"][
            "figure_manifests"][0]["artifacts"]}
    assert result["integration"]["authenticated_figures_included"] == [
        "paper_a/figures/sage_mem_v1_claim_ladder.pdf"]

    figure.write_bytes(b"tampered")
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="size differs|SHA-256 differs"):
        _audit(
            fixture,
            figure_manifests=(manifest_path.relative_to(root),),
            expected_figure_manifest_sha256=(
                hashlib.sha256(manifest_payload).hexdigest(),))


def test_figure_manifest_rejects_fake_generator_and_fake_artifacts(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    root = fixture["root"]
    ledger = root / fixture["ledger"]
    output = root / "paper_a/figures"
    subprocess.run([
        sys.executable, str(root / audit.PLOTTER_SOURCE),
        "--ledger", str(ledger), "--output-dir", str(output),
        "--prefix", "sage_mem_v1",
        "--expected-ledger-sha256", fixture["ledger_sha"], "--execute",
    ], check=True, capture_output=True, text=True)
    manifest_path = output / "sage_mem_v1_plot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    authentic_manifest = copy.deepcopy(manifest)
    fake = root / "scripts/fake_plotter.py"
    fake.write_text("# fake\n")
    manifest["source_binding"]["plotting_script"] = _identity(root, fake)
    payload = _write_canonical(manifest_path, manifest)
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="figure generator path differs"):
        _audit(
            fixture,
            figure_manifests=(manifest_path.relative_to(root),),
            expected_figure_manifest_sha256=(
                hashlib.sha256(payload).hexdigest(),))

    fake_figure = output / "sage_mem_v1_claim_ladder.pdf"
    fake_figure.write_bytes(b"%PDF-1.4\nsynthetic but unsupported\n%%EOF\n")
    manifest = authentic_manifest
    manifest["artifacts"]["claim_ladder_pdf"] = _identity(
        root, fake_figure)
    payload = _write_canonical(manifest_path, manifest)
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="differs from independent render"):
        _audit(
            fixture,
            figure_manifests=(manifest_path.relative_to(root),),
            expected_figure_manifest_sha256=(
                hashlib.sha256(payload).hexdigest(),))


def test_receipt_is_atomic_and_resume_is_exact_match(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    result = _audit(fixture)
    receipt = Path("outputs/binding/receipt.json")
    assert audit.emit_receipt(
        tmp_path, receipt, result, execute=False) == "not-written"
    assert not (tmp_path / receipt).exists()
    assert audit.emit_receipt(
        tmp_path, receipt, result, execute=True) == "created"
    assert (tmp_path / receipt).read_bytes() == \
        audit.canonical_json(result).encode()
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="refusing to overwrite"):
        audit.emit_receipt(tmp_path, receipt, result, execute=True)
    assert audit.emit_receipt(
        tmp_path, receipt, result, execute=True,
        resume=True) == "validated-existing"
    changed = copy.deepcopy(result)
    changed["authorization"]["primary_positive_rows"] = 2
    with pytest.raises(audit.SageMemBindingAuditError,
                       match="differs from the current"):
        audit.emit_receipt(
            tmp_path, receipt, changed, execute=True, resume=True)


def test_execute_requires_pinned_report_and_ledger_hashes(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _build_fixture(tmp_path)
    assert audit.main([
        "--root", str(tmp_path), "--report", str(fixture["report"]),
        "--ledger-json", str(fixture["ledger"]),
        "--ledger-tex", str(fixture["ledger_tex"]),
        "--main-tex", str(fixture["main"]), "--execute",
    ]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert "expected-report-sha256 is required" in failure["reason"]
    assert not (tmp_path / audit.DEFAULT_RECEIPT).exists()
