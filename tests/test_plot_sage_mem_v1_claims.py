from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image
import pytest

import scripts.plot_sage_mem_v1_claims as plotter


@pytest.fixture(autouse=True)
def _isolated_repository_root(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plotter, "ROOT", tmp_path)


def _ci(lower: float, *, confidence: float = 0.95,
        seed: int = 1) -> dict[str, object]:
    upper = min(1.0, lower + 0.04)
    return {
        "point": lower + 0.02,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
        "draws": 20_000,
        "seed": seed,
        "resampling_unit": "paired formal seed x native episode cluster",
        "class_profile_stratified": True,
        "pairing_preserved": True,
    }


def _execution(*, passes: bool, seed: int) -> dict[str, object]:
    lower = 0.04 if passes else 0.02
    return {
        "full_vs_locked_comparator": _ci(lower, seed=seed),
        "full_vs_none": _ci(0.04, seed=seed + 1),
        "full_vs_random": _ci(0.04, seed=seed + 2),
        "random_reference":
            "sealed per-episode arm-blind random-success deck",
        "random_reference_is_cohort_rate": False,
        "oracle_success": 0.95,
        "random_success_mean": 0.25,
        "pass": passes,
    }


def _row(cohort: str, age: int, index: int) -> dict[str, object]:
    primary = index % 4 != 0
    prior = index % 2 == 0
    raw = index % 3 == 0
    execution_supplied = cohort in {
        "dinowm_pusht_token", "dinowm_pointmaze_goal"}
    execution_pass = age == 4 or (
        cohort == "dinowm_pointmaze_goal" and age == 8)
    controls = {
        key: _ci(0.04, seed=1000 + index * 20 + offset)
        for offset, key in enumerate(plotter.CONTROL_KEYS)
    }
    gates = {
        "host_vs_locked_comparator": primary,
        "full_vs_reset": True,
        "full_vs_none": True,
        "all_mechanism_controls": True,
        "next_mse_noninferiority": True,
    }
    return {
        "cohort": cohort,
        "cohort_label": plotter.COHORT_LABELS[cohort],
        "age": age,
        "primary_endpoint": "frozen-host full correctness",
        "host_full_accuracy": _ci(0.50, seed=10 + index),
        "host_full_vs_locked_comparator": _ci(
            0.06 if primary else 0.04, seed=100 + index),
        "host_full_vs_reset": _ci(0.04, seed=200 + index),
        "host_full_vs_none": _ci(0.04, seed=300 + index),
        "reset_to_full_mse_ratio": 1.2,
        "mechanism_controls": controls,
        "next_feature_relative_excess": _ci(
            0.02, confidence=0.90, seed=400 + index),
        "gates": gates,
        "primary_host_claim_pass": primary,
        "prior_diagnostic": {
            "role": "diagnostic-only; cannot establish host use",
            "accuracy": _ci(0.50, seed=500 + index),
            "vs_locked_comparator": _ci(
                0.02 if prior else -0.02, seed=600 + index),
            "resolved_positive": prior,
            "enters_primary_host_claim": False,
        },
        "raw_context_reference": {
            "short3_accuracy": 0.30,
            "long16_accuracy": 0.45,
            "long16_minus_short3": _ci(
                0.02 if raw else -0.02, seed=700 + index),
            "resolved_long_context_gain": raw,
            "separate_from_parameter_matched_grid": True,
        },
        "execution": (
            _execution(passes=execution_pass, seed=800 + index * 3)
            if execution_supplied else None),
        "locked_comparators": {
            "retention": "ssm",
            "next_feature": "gru",
            "execution": "lstm",
        },
    }


def _artifact_identity(character: str) -> dict[str, object]:
    return {
        "path": f"formal_preparation/{character}.json",
        "size": 123,
        "sha256": character * 64,
    }


def _ledger() -> dict[str, object]:
    rows = [
        _row(cohort, age, index)
        for index, (cohort, age) in enumerate(
            (pair for cohort in plotter.COHORTS
             for pair in ((cohort, age) for age in plotter.AGES)))
    ]
    summaries: dict[str, object] = {}
    for cohort_index, cohort in enumerate(plotter.COHORTS):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        supplied = all(row["execution"] is not None for row in cohort_rows)
        passing = sum(bool(row["primary_host_claim_pass"])
                      for row in cohort_rows)
        summaries[cohort] = {
            "cohort_label": plotter.COHORT_LABELS[cohort],
            "locked_comparators": dict(cohort_rows[0]["locked_comparators"]),
            "comparator_receipt": {
                "formal_preparation_manifest_sha256":
                    f"{cohort_index + 1:x}" * 64,
                "implementation_lock_sha256": "d" * 64,
                "custody_registry_sha256": "e" * 64,
                "preparation_receipt": _artifact_identity("f"),
                "locked_comparator_receipt": _artifact_identity("a"),
            },
            "backend_admission": {
                "backend": "synthetic-test-host",
                "admission_rechecked": True,
            },
            "resource_enforcement_verified": True,
            "registered_age_rows": 3,
            "rows_passing_primary_host_claim": passing,
            "all_registered_ages_primary_host_claim_pass": passing == 3,
            "execution_supplied": supplied,
            "execution_pass_by_age": {
                str(row["age"]): (
                    row["execution"]["pass"]
                    if row["execution"] is not None else None)
                for row in cohort_rows
            },
        }
    total_passing = sum(bool(row["primary_host_claim_pass"]) for row in rows)
    eligible = sum(bool(summary["execution_supplied"])
                   for summary in summaries.values())
    permitted = eligible >= 2
    per_age = {}
    for age in plotter.AGES:
        passing = sum(
            row["execution"] is not None and row["execution"]["pass"]
            for row in rows if row["age"] == age)
        per_age[str(age)] = {
            "eligible_cohorts": eligible,
            "cohorts_passing": passing,
            "claim_permitted": permitted,
            "claim_pass": permitted and passing >= 2,
        }
    return {
        "schema": plotter.LEDGER_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "paper-claim-ledger",
        "status": "complete",
        "integrity_completion": {
            "status": "complete",
            "meaning": "authenticated counts; not a scientific pass",
            "phase_a_cells_verified": 600,
            "finalized_cells_verified": 600,
            "comparators_verified": 5,
            "resources_verified": 600,
            "raw_context_references_verified": 50,
            "phase_a_grid_sha256": "b" * 64,
            "identity_ledger_sha256": "c" * 64,
        },
        "claim_policy": {
            "per_age_claims_only": True,
            "registered_cohorts": list(plotter.COHORTS),
            "registered_ages": list(plotter.AGES),
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
            "primary_claim_rows_passing": total_passing,
            "primary_claim_rows_failing": 15 - total_passing,
            "any_primary_claim_row_passed": total_passing > 0,
            "all_primary_claim_rows_passed": total_passing == 15,
            "meaning": "all registered row-level conjunctions",
        },
        "cohort_summaries": summaries,
        "claim_rows": rows,
        "execution_program": {
            "optional": True,
            "eligible_cohorts": eligible,
            "minimum_eligible_cohorts": 2,
            "program_claim_permitted": permitted,
            "per_age": per_age,
            "cross_age_conjunction_computed": False,
            "program_claim_pass": None,
        },
        "source_binding": {
            "report": {
                "path": "outputs/sage_mem_v1/formal_audit/report.json",
                "size": 1000,
                "sha256": "1" * 64,
                "schema": plotter.REPORT_SCHEMA,
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
                "path": "configs/sage_mem_v1.yaml",
                "size": 2000,
                "sha256": "2" * 64,
                "fingerprint": "3" * 64,
                "implementation_lock": {
                    "path": "outputs/sage_mem_v1/protocol_lock.json",
                    "size": 300,
                    "sha256": "4" * 64,
                    "status": "sealed",
                },
                "formal_amendment": {
                    "path": "configs/sage_mem_v1_formal_amendment.yaml",
                    "size": 400,
                    "sha256": "5" * 64,
                    "status":
                        "locked-before-development-selection-or-formal-data",
                },
                "report_schema_repeats_protocol_fingerprint": False,
                "binding_note": "synthetic transitive source binding",
            },
            "formal_auditor": {
                "path": "scripts/audit_sage_mem_v1_formal.py",
                "size": 500,
                "sha256": "6" * 64,
                "sealed_by_implementation_lock": True,
            },
            "adapter": {
                "path": "scripts/summarize_sage_mem_v1_report.py",
                "sha256": "7" * 64,
            },
        },
        "publication_artifacts": {
            "tex": {
                "path": "paper_a/generated_results/sage_mem_v1_claim_ledger.tex",
                "sha256": "8" * 64,
            },
        },
    }


def _write_ledger(path: Path, ledger: dict[str, object] | None = None
                  ) -> str:
    payload = plotter._canonical_json(ledger or _ledger())
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _execute(tmp_path: Path, ledger: dict[str, object] | None = None,
             *, output_name: str = "out", extra: tuple[str, ...] = ()
             ) -> tuple[Path, dict[str, Path]]:
    ledger_path = tmp_path / f"{output_name}-ledger.json"
    digest = _write_ledger(ledger_path, ledger)
    output = tmp_path / "paper_a" / output_name
    arguments = [
        "--ledger", str(ledger_path),
        "--output-dir", str(output),
        "--prefix", "synthetic",
        "--expected-ledger-sha256", digest,
        "--execute",
        *extra,
    ]
    assert plotter.main(arguments) == 0
    return ledger_path, plotter._output_paths(output, "synthetic")


def test_preview_reads_nothing_and_writes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "figures"
    assert plotter.main([
        "--ledger", str(missing), "--output-dir", str(output),
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview"] is True
    assert preview["no_files_read"] is True
    assert preview["no_outcomes_read"] is True
    assert preview["required_claim_rows"] == 15
    assert not output.exists()


def test_execute_renders_all_rows_and_source_bound_publication_artifacts(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger_path, paths = _execute(tmp_path)
    result = json.loads(capsys.readouterr().out)
    assert result["publication"] == "created"
    assert result["registered_rows_rendered"] == 15
    assert result["ledger_sha256"] == hashlib.sha256(
        ledger_path.read_bytes()).hexdigest()
    assert all(path.is_file() and not path.is_symlink()
               for path in paths.values())

    data_payload = paths["plot_data"].read_bytes()
    data = json.loads(data_payload)
    assert data_payload == plotter._canonical_json(data)
    assert len(data["claim_ladder_rows"]) == 15
    assert len(data["effect_rows"]) == 15
    assert [(row["cohort"], row["age"])
            for row in data["claim_ladder_rows"]] == [
                (cohort, age) for cohort in plotter.COHORTS
                for age in plotter.AGES
            ]
    assert any(row["execution"] is None
               for row in data["claim_ladder_rows"])
    assert data["source_binding"]["plotting_script_sha256"] == \
        result["plotting_script_sha256"]

    manifest_payload = paths["manifest"].read_bytes()
    manifest = json.loads(manifest_payload)
    assert manifest_payload == plotter._canonical_json(manifest)
    assert manifest["source_binding"]["claim_ledger"][
        "sha256"] == result["ledger_sha256"]
    assert manifest["display_contract"]["registered_rows_rendered"] == 15
    assert manifest["display_contract"][
        "all_rows_rendered_in_every_channel"] is True
    for key, identity in manifest["artifacts"].items():
        artifact = paths[key]
        assert identity["size"] == artifact.stat().st_size
        assert identity["sha256"] == hashlib.sha256(
            artifact.read_bytes()).hexdigest()

    for key in ("claim_ladder_png", "effects_png"):
        with Image.open(paths[key]) as image:
            assert image.format == "PNG"
            assert image.width >= 1800
            assert image.height >= 900
            assert image.info["ClaimLedgerSHA256"] == result["ledger_sha256"]
    for key in ("claim_ladder_pdf", "effects_pdf"):
        payload = paths[key].read_bytes()
        assert payload.startswith(b"%PDF-")
        assert b"/Subtype /Image" not in payload
        if shutil.which("pdfinfo"):
            info = subprocess.run(
                ["pdfinfo", str(paths[key])], check=True,
                text=True, capture_output=True).stdout
            assert "Pages:           1" in info
        if shutil.which("pdffonts"):
            fonts = subprocess.run(
                ["pdffonts", str(paths[key])], check=True,
                text=True, capture_output=True).stdout
            assert "Type 3" not in fonts
            assert "emb" in fonts and "yes" in fonts


@pytest.mark.parametrize("mutation,match", [
    ("missing_row", "all 15 registered rows"),
    ("reordered_rows", "missing, duplicated, selected, or reordered"),
    ("bad_gate", "gate flags differ"),
    ("bad_source", "formal-report source authentication is incomplete"),
])
def test_strictly_rejects_incomplete_or_inconsistent_ledgers(
        tmp_path: Path, mutation: str, match: str) -> None:
    ledger = _ledger()
    if mutation == "missing_row":
        ledger["claim_rows"].pop()
    elif mutation == "reordered_rows":
        ledger["claim_rows"][0], ledger["claim_rows"][1] = (
            ledger["claim_rows"][1], ledger["claim_rows"][0])
    elif mutation == "bad_gate":
        ledger["claim_rows"][1]["gates"][
            "host_vs_locked_comparator"] = False
    else:
        ledger["source_binding"]["report"][
            "independent_sealed_auditor_recomputation_verified"] = False
    ledger_path = tmp_path / "ledger.json"
    digest = _write_ledger(ledger_path, ledger)
    with pytest.raises(plotter.SageMemPlotError, match=match):
        plotter.main([
            "--ledger", str(ledger_path),
            "--output-dir", str(tmp_path / "paper_a/out"),
            "--expected-ledger-sha256", digest,
            "--execute",
        ])
    assert not (tmp_path / "paper_a/out").exists()


def test_requires_expected_digest_and_canonical_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    digest = _write_ledger(ledger_path)
    with pytest.raises(plotter.SageMemPlotError,
                       match="expected-ledger-sha256 is required"):
        plotter.main([
            "--ledger", str(ledger_path),
            "--output-dir", str(tmp_path / "paper_a/out"), "--execute",
        ])
    with pytest.raises(plotter.SageMemPlotError,
                       match="differs from expected"):
        plotter.main([
            "--ledger", str(ledger_path),
            "--output-dir", str(tmp_path / "paper_a/out"),
            "--expected-ledger-sha256", "0" * 64, "--execute",
        ])
    noncanonical = json.dumps(_ledger(), indent=2).encode()
    ledger_path.write_bytes(noncanonical)
    with pytest.raises(plotter.SageMemPlotError,
                       match="not canonical JSON"):
        plotter.main([
            "--ledger", str(ledger_path),
            "--output-dir", str(tmp_path / "paper_a/out"),
            "--expected-ledger-sha256",
            hashlib.sha256(noncanonical).hexdigest(), "--execute",
        ])
    assert len(digest) == 64


def test_refuses_overwrite_resume_validates_and_repairs_only_identical_set(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger_path, paths = _execute(tmp_path)
    capsys.readouterr()
    digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    command = [
        "--ledger", str(ledger_path),
        "--output-dir", str(tmp_path / "paper_a/out"),
        "--prefix", "synthetic",
        "--expected-ledger-sha256", digest,
        "--execute",
    ]
    with pytest.raises(plotter.SageMemPlotError, match="overwrite"):
        plotter.main(command)
    assert plotter.main([*command, "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)[
        "publication"] == "validated-existing"

    paths["effects_png"].unlink()
    assert plotter.main([*command, "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)[
        "publication"] == "repaired-missing"
    paths["claim_ladder_png"].write_bytes(b"tampered")
    with pytest.raises(plotter.SageMemPlotError, match="differs"):
        plotter.main([*command, "--resume"])


def test_rendering_is_byte_deterministic_for_same_authenticated_ledger(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger = _ledger()
    _, first = _execute(tmp_path, copy.deepcopy(ledger), output_name="one")
    capsys.readouterr()
    _, second = _execute(tmp_path, copy.deepcopy(ledger), output_name="two")
    capsys.readouterr()
    for key in (
            "claim_ladder_pdf", "claim_ladder_png", "effects_pdf",
            "effects_png", "plot_data"):
        assert first[key].read_bytes() == second[key].read_bytes()


def test_execute_rejects_output_outside_paper_and_symlink_parent(
        tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    digest = _write_ledger(ledger)
    with pytest.raises(plotter.SageMemPlotError,
                       match="inside ROOT/paper_a"):
        plotter.main([
            "--ledger", str(ledger), "--output-dir", str(tmp_path / "out"),
            "--expected-ledger-sha256", digest, "--execute",
        ])

    real = tmp_path / "paper_a/real"
    real.mkdir(parents=True)
    linked = tmp_path / "paper_a/linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(plotter.SageMemPlotError,
                       match="symlink component"):
        plotter.main([
            "--ledger", str(ledger), "--output-dir", str(linked),
            "--expected-ledger-sha256", digest, "--execute",
        ])


def test_effect_panel_boundary_tick_labels_do_not_merge() -> None:
    labels = [
        plotter._effect_tick_labels(index, 4, 10.0) for index in range(4)
    ]
    assert labels[0] == ("-10", "0", "")
    assert labels[1] == ("", "0", "")
    assert labels[2] == ("", "0", "")
    assert labels[3] == ("", "0", "10")
    assert all(labels[index][2] == "" or labels[index + 1][0] == ""
               for index in range(3))
