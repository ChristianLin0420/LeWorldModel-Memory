from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from scripts import generate_paper_a_appendix_tables as tables


def _stat(mean: float = 0.5) -> dict[str, object]:
    return {
        "mean": mean,
        "ci95": [mean - 0.05, mean + 0.05],
        "ci90": [mean - 0.04, mean + 0.04],
    }


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matched_summary() -> dict[str, object]:
    hosts = {}
    for host_index, host in enumerate(("reacher", "pusht")):
        arms = {}
        for arm_index, arm in enumerate(tables.ARMS):
            base = 0.3 + 0.03 * host_index + 0.02 * arm_index
            arms[arm] = {
                "age-4": _stat(base + 0.10),
                "age-8": _stat(base + 0.05),
                "age-15": _stat(base),
                "shortest_minus_longest": _stat(0.10),
            }
        hosts[host] = {"arms": arms}
    interaction = _stat(-0.002)
    interaction.update({
        "equivalence_margin": 0.05,
        "equivalent_within_margin": True,
        "resolved_nonzero": False,
    })
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "paper-a-matched-color-v1-1",
        "branch": "admission-informed-matched-color-v1-1",
        "ages": list(tables.AGES),
        "arms": list(tables.ARMS),
        "seeds": list(range(5)),
        "hosts": hosts,
        "primary_ranking_interaction": interaction,
    }


def _pusht_summary() -> dict[str, object]:
    results = {}
    for task in tables.TASK_LABELS:
        age_results = {}
        for age in tables.AGES:
            age_results[str(age)] = {
                "arms": {
                    arm: {"balanced_accuracy": _stat(),
                          "parameters": tables.PARAMETERS[arm]}
                    for arm in tables.ARMS
                },
                "paired_vs_none": {
                    arm: _stat(0.1) for arm in tables.ARMS if arm != "none"
                },
                "full_vs_context_reset": {
                    arm: _stat(0.08) for arm in tables.ARMS
                },
            }
        results[task] = {"ages": age_results}
    return {
        "schema": "dinowm_wave2_spatial_carrier_summary_v1",
        "status": "complete",
        "grid": {"tasks": 2, "arms": 5, "seeds": 5, "cells": 50},
        "results": results,
    }


def _admission() -> dict[str, object]:
    shortcuts = {
        str(age): {
            name: {
                "balanced_accuracy": 0.25,
                "maximum": 0.30,
                "pass": True,
            }
            for name in ("no_cue_visual", "action_only", "proprio_only")
        }
        for age in tables.AGES
    }
    return {
        "admitted": True,
        "all_gates_required": True,
        "requirement": {"pass": True},
        "cue_encoding": {
            "balanced_accuracy": 1.0,
            "per_class_recall": [1.0] * 4,
            "thresholds": {
                "balanced_accuracy_minimum": 0.75,
                "per_class_recall_minimum": 0.70,
            },
            "pass": True,
        },
        "shortcuts": shortcuts,
        "cue_only_counterfactual": {
            "outside_declared_mask_changed_pixels": 0,
            "post_cue_differing_pixels": 0,
            "pass": True,
        },
        "frozen_host": {"pass": True},
    }


def _controller() -> dict[str, object]:
    return {
        "admitted": True,
        "oracle_executed_success": 0.95,
        "oracle_per_class_executed_success": [0.94, 0.95, 0.96, 0.95],
        "off_diagonal_false_success": 0.0,
        "deterministic_replay_fidelity": 1.0,
        "thresholds": {
            "oracle_success_minimum": 0.90,
            "oracle_per_class_success_minimum": 0.85,
            "off_diagonal_false_success_maximum": 0.10,
            "deterministic_reset_replay_minimum": 1.0,
        },
    }


def _pointmaze_summaries() -> tuple[dict[str, object],
                                    dict[str, object],
                                    dict[str, object]]:
    protocol = "a" * 64
    admission = _admission()
    controller = _controller()
    combined = {
        "schema": "dinowm_pointmaze_wave3_summary_v1",
        "status": "complete",
        "protocol_sha256": protocol,
        "admission": admission,
        "controller_gate": controller,
        "carrier_summary_path": "carrier_summary.json",
        "external_use_summary_path": "external_use_summary.json",
    }
    carrier = {
        "schema": "dinowm_pointmaze_wave3_carrier_summary_v1",
        "status": "complete",
        "protocol_sha256": protocol,
        "grid": {"tasks": 1, "arms": 5, "seeds": 5, "cells": 25},
        "results": {
            str(age): {
                "arms": {
                    arm: {"balanced_accuracy": _stat(),
                          "parameters": tables.PARAMETERS[arm]}
                    for arm in tables.ARMS
                },
                "paired_vs_none": {
                    arm: _stat(0.1) for arm in tables.ARMS if arm != "none"
                },
                "full_vs_context_reset": {
                    arm: _stat(0.08) for arm in tables.ARMS
                },
            }
            for age in tables.AGES
        },
    }
    use = {
        "schema": "dinowm_pointmaze_wave3_external_use_v1",
        "status": "complete",
        "protocol_sha256": protocol,
        "controller_gate": controller,
        "arms": {
            arm: {
                "goal_accuracy": _stat(),
                "executed_success": _stat(0.6),
                "contrast_vs_none": _stat(0.1),
                "contrast_vs_random": _stat(0.15),
                "resolved_execution_gain": arm in {"ssm", "fixed_trust"},
            }
            for arm in tables.ARMS
        },
        "realized_random_goal": _stat(0.45),
        "oracle_executed_success": 0.95,
    }
    return combined, carrier, use


def _fixture_repository(root: Path) -> tuple[Path, Path]:
    generator = root / tables.GENERATOR_SCRIPT
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_bytes((tables.ROOT / tables.GENERATOR_SCRIPT).read_bytes())
    matched = _matched_summary()
    pusht = _pusht_summary()
    pointmaze, carrier, use = _pointmaze_summaries()
    hashes = {
        "matched": _write_json(root / tables.MATCHED_SUMMARY, matched),
        "pusht": _write_json(root / tables.PUSHT_SUMMARY, pusht),
        "pointmaze": _write_json(root / tables.POINTMAZE_SUMMARY, pointmaze),
        "pointmaze_carrier": _write_json(
            root / tables.POINTMAZE_CARRIER, carrier),
        "pointmaze_use": _write_json(root / tables.POINTMAZE_USE, use),
    }
    completion = {
        "schema": "paper_a_cross_wave_completion_receipt_v1",
        "status": "complete",
        "read_only_verification": True,
        "scientific_cross_wave_aggregation": False,
        "sealed_locks_modified": False,
        "paper_files_modified": False,
        "waves": {
            "wave1_1": {"status": "verified",
                        "summary_sha256": hashes["matched"]},
            "wave2_v1_1": {"status": "verified",
                            "summary_sha256": hashes["pusht"]},
            "wave3": {
                "status": "verified",
                "summary_sha256": hashes["pointmaze"],
                "carrier_summary_sha256": hashes["pointmaze_carrier"],
                "external_use_summary_sha256": hashes["pointmaze_use"],
            },
        },
    }
    statistics = {
        "schema": "paper_a_statistics_independent_receipt_v1",
        "status": "verified",
        "read_only": True,
        "statistics_computed": True,
        "scientific_cross_family_pooling": False,
        "imports_producer_statistics": False,
        "experiment_roots_modified": False,
        "waves": {
            "wave2": {"status": "verified",
                      "summary_sha256": hashes["pusht"]},
            "wave3": {
                "status": "verified",
                "combined_summary_sha256": hashes["pointmaze"],
                "carrier_summary_sha256": hashes["pointmaze_carrier"],
                "external_use_summary_sha256": hashes["pointmaze_use"],
            },
        },
    }
    completion_path = root / "receipts/completion.json"
    statistics_path = root / "receipts/statistics.json"
    _write_json(completion_path, completion)
    _write_json(statistics_path, statistics)
    return completion_path.relative_to(root), statistics_path.relative_to(root)


def test_missing_receipts_never_inspect_partial_summaries(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / tables.PUSHT_SUMMARY).parent.mkdir(parents=True)
    (tmp_path / tables.PUSHT_SUMMARY).write_text("partial")
    inspected: list[Path] = []

    def forbidden_hash(path: Path) -> str:
        inspected.append(path)
        raise AssertionError("partial summary inspected before receipts")

    monkeypatch.setattr(tables, "sha256_file", forbidden_hash)
    with pytest.raises(tables.NotReady, match="missing independent"):
        tables.load_verified_summaries(
            tmp_path, Path("receipts/missing-a.json"),
            Path("receipts/missing-b.json"))
    assert inspected == []


def test_disagreeing_receipts_fail_before_summary_inspection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completion, statistics = _fixture_repository(tmp_path)
    value = json.loads((tmp_path / statistics).read_text())
    value["waves"]["wave2"]["summary_sha256"] = "f" * 64
    (tmp_path / statistics).write_text(json.dumps(value))
    original = tables.sha256_file
    inspected_summaries: list[Path] = []

    def track(path: Path) -> str:
        if path.name.endswith("summary.json"):
            inspected_summaries.append(path)
        return original(path)

    monkeypatch.setattr(tables, "sha256_file", track)
    with pytest.raises(tables.GenerationFailure, match="receipts disagree"):
        tables.load_verified_summaries(tmp_path, completion, statistics)
    assert inspected_summaries == []


def test_dry_run_is_read_only_and_reports_all_tables(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    completion, statistics = _fixture_repository(tmp_path)
    destination = Path("generated/tables")
    code = tables.main([
        "--root", str(tmp_path),
        "--completion-receipt", str(completion),
        "--statistics-receipt", str(statistics),
        "--output-dir", str(destination),
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["wrote_output"] is False
    assert len(payload["tables"]) == 7
    assert tables.MAIN_CLAIM_LEDGER_FILENAME in payload["tables"]
    assert not (tmp_path / destination).exists()


def test_execute_atomically_generates_dense_readable_tables(
        tmp_path: Path) -> None:
    completion, statistics = _fixture_repository(tmp_path)
    destination = Path("generated/final_appendix")
    result = tables.generate(
        tmp_path, completion, statistics, destination, execute=True)
    output = tmp_path / destination
    assert result["status"] == "complete"
    assert result["wrote_output"] is True
    assert set(path.name for path in output.iterdir()) == {
        "matched_color_results.tex", "dinowm_pusht_results.tex",
        "pointmaze_carrier_results.tex", "pointmaze_gate_checklist.tex",
        "pointmaze_external_use_results.tex", "main_claim_ledger.tex",
        "all_tables.tex",
        "manifest.json",
    }
    for name in (
            "matched_color_results.tex", "dinowm_pusht_results.tex",
            "pointmaze_carrier_results.tex", "pointmaze_gate_checklist.tex",
            "pointmaze_external_use_results.tex"):
        content = (output / name).read_text()
        assert (r"\footnotesize" in content
                or r"\scriptsize" in content)
        captions = re.findall(r"\\caption\{([^}]*)\}", content)
        assert captions and all("wave" not in caption.lower()
                                for caption in captions)
    matched = (output / "matched_color_results.tex").read_text()
    assert "Registered host ranking interaction" in matched
    assert matched.count("No state") == 2
    pusht = (output / "dinowm_pusht_results.tex").read_text()
    assert "Token recall" in pusht and "Binding recall" in pusht
    assert pusht.count("Fixed-trust") == 6
    pointmaze = (output / "pointmaze_carrier_results.tex").read_text()
    assert pointmaze.count("Fixed-trust") == 3
    gates = (output / "pointmaze_gate_checklist.tex").read_text()
    assert gates.count(r"\textsc{pass}") == 11
    assert r"p{0.34\textwidth}" in gates and "	" not in gates
    external = (output / "pointmaze_external_use_results.tex").read_text()
    assert all(tables.ARM_LABELS[arm] in external for arm in tables.ARMS)
    ledger = (output / "main_claim_ledger.tex").read_text()
    assert r"\label{tab:claim-ledger-v2}" in ledger
    assert all(name in ledger for name in (
        "LeWM matched color (Reacher / PushT)",
        "DINO-WM PushT token", "DINO-WM PushT binding",
        "DINO-WM PointMaze goal"))
    assert ("Study & Gates & vs. no state & vs. reset & Executed use") \
        in ledger
    assert all(column not in ledger for column in (
        "Need & Cue & No shortcut", "Necessity & Cue encoding",
        "Shortcut exclusion"))
    assert "Verified" not in ledger and "Not tested" not in ledger
    assert ledger.count(r"\Pass") == 4 and ledger.count(r" & \NA") == 4
    assert "Age curves (both hosts)" in ledger
    assert "State-space, Fixed-trust" in ledger
    assert ledger.count("All learned") >= 2
    assert all(marker not in ledger for marker in (
        r"\checkmark", r"\Fail", r"\textsc{yes}"))
    assert "None resolved" in ledger
    assert "no pooled score or cross-family ranking" in ledger.lower()
    bundle = (output / "all_tables.tex").read_text()
    assert "tab:claim-ledger-v2" not in bundle
    assert all(f"\\long\\def\\{command}" in bundle
               for command in tables.APPENDIX_TABLE_COMMANDS.values())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    generator = tmp_path / tables.GENERATOR_SCRIPT
    assert manifest["generator"] == {
        "path": str(tables.GENERATOR_SCRIPT),
        "size": generator.stat().st_size,
        "sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
    }
    assert manifest["source_identities"]["summaries"]["pusht"]["sha256"]
    assert set(manifest["tables"]) == {
        *tables.APPENDIX_TABLE_FILENAMES,
        tables.MAIN_CLAIM_LEDGER_FILENAME,
        tables.APPENDIX_BUNDLE_FILENAME,
    }
    assert manifest["display_contract"]["all_tables_excludes"] == [
        tables.MAIN_CLAIM_LEDGER_FILENAME]
    assert manifest["display_contract"]["main_claim_ledger_font_size"] \
        == "scriptsize"


def test_main_claim_ledger_uses_strict_registered_resolution_rules() -> None:
    matched = _matched_summary()
    pusht = _pusht_summary()
    pointmaze, carrier, use = _pointmaze_summaries()

    token = pusht["results"]["transient-visual-token-recall"]["ages"]["15"]
    for arm in tables.ARMS:
        if arm != "none":
            # A lower endpoint exactly at zero is not resolved.
            token["paired_vs_none"][arm] = _stat(0.05)
    binding = pusht["results"]["multi-item-visual-binding-recall"]["ages"]["15"]
    for arm in tables.ARMS:
        binding["full_vs_context_reset"][arm] = _stat(0.05)
    for record in use["arms"].values():
        record["resolved_execution_gain"] = False

    ledger = tables.render_tables({
        "matched": matched,
        "pusht": pusht,
        "pointmaze": pointmaze,
        "pointmaze_carrier": carrier,
        "pointmaze_use": use,
    })[tables.MAIN_CLAIM_LEDGER_FILENAME]
    data_rows = [line for line in ledger.splitlines()
                 if " & " in line and line.endswith(r"\\")]
    assert sum(row.count("None resolved") for row in data_rows) == 3
    assert (r"DINO-WM PushT token & \Pass\ 3/3 & None resolved") \
        in ledger
    assert r"None resolved & \NA" in ledger
    assert ledger.count("Age curves (both hosts)") == 1


def test_hash_tamper_and_protected_output_fail_without_writes(
        tmp_path: Path) -> None:
    completion, statistics = _fixture_repository(tmp_path)
    with (tmp_path / tables.PUSHT_SUMMARY).open("a") as stream:
        stream.write("tamper")
    destination = Path("generated/rejected")
    with pytest.raises(tables.GenerationFailure, match="summary hash differs"):
        tables.generate(
            tmp_path, completion, statistics, destination, execute=True)
    assert not (tmp_path / destination).exists()

    # Restore a valid fixture, then exercise the output boundary itself.
    completion, statistics = _fixture_repository(tmp_path)
    protected = tables.EXPERIMENT_ROOTS[0] / "generated_tables"
    with pytest.raises(tables.GenerationFailure, match="outside experiment"):
        tables.generate(
            tmp_path, completion, statistics, protected, execute=True)
    assert not (tmp_path / protected).exists()


def test_atomic_publish_failure_leaves_no_partial_directory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completion, statistics = _fixture_repository(tmp_path)
    summaries, identities = tables.load_verified_summaries(
        tmp_path, completion, statistics)
    files = tables.render_tables(summaries)
    destination = Path("generated/atomic_target")
    plan = tables.build_plan(
        files, identities, destination, tables.generator_identity(tmp_path))

    def fail_rename(source: Path, target: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(tables.os, "rename", fail_rename)
    with pytest.raises(OSError, match="injected"):
        tables.emit_output(
            tmp_path, destination, files, plan, execute=True)
    assert not (tmp_path / destination).exists()
    assert list((tmp_path / "generated").glob(".atomic_target.*.tmp")) == []


def test_publish_rejects_table_generator_drift(tmp_path: Path) -> None:
    completion, statistics = _fixture_repository(tmp_path)
    summaries, identities = tables.load_verified_summaries(
        tmp_path, completion, statistics)
    files = tables.render_tables(summaries)
    destination = Path("generated/stale_generator")
    plan = tables.build_plan(
        files, identities, destination, tables.generator_identity(tmp_path))
    with (tmp_path / tables.GENERATOR_SCRIPT).open("a") as stream:
        stream.write("# drift\n")

    with pytest.raises(tables.GenerationFailure,
                       match="identity changed before publication"):
        tables.emit_output(
            tmp_path, destination, files, plan, execute=True)
    assert not (tmp_path / destination).exists()
