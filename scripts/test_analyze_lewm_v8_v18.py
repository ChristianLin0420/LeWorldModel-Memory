"""Tests for the frozen V18 confirmation analyzer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_lewm_v8_v18 as analyzer


TASKS = analyzer.FROZEN_TASKS
SEEDS = analyzer.FROZEN_SEEDS


@pytest.fixture(autouse=True)
def frozen_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer.runner, "TASKS", TASKS)
    monkeypatch.setattr(analyzer.runner, "DESIGNS", analyzer.FROZEN_DESIGNS)
    monkeypatch.setattr(analyzer.runner, "SEEDS", SEEDS)
    monkeypatch.setattr(analyzer.runner, "EPOCHS", analyzer.FROZEN_EPOCHS)
    # Production analysis is fixed at 100,000 draws. Tests retain the exact
    # crossed resampling implementation with a smaller deterministic budget.
    monkeypatch.setattr(analyzer, "BOOTSTRAP_DRAWS", 2_000)


def _values(task_index: int, seed_index: int, design: str) -> tuple[float, float, float]:
    alternating = (task_index + seed_index) % 2 == 0
    if design == analyzer.NONE:
        return 1.20, 1.10, 1.20
    if design == analyzer.GRU:
        value = 1.00 if alternating else 1.10
        return value, value, value
    if design == analyzer.SSM:
        value = 1.10 if alternating else 1.00
        return value, value, value
    if design == analyzer.CANDIDATE:
        return 0.90, 1.02, 0.90
    if design == analyzer.DYNAMIC:
        value = 0.895 if alternating else 0.905
        return value, 1.02, value
    if design == analyzer.STATIC:
        value = 0.905 if alternating else 0.895
        return value, 1.02, value
    if design == analyzer.NO_ACTION:
        return 1.05, 1.02, 1.05
    if design == analyzer.SINGLE:
        return 1.00, 1.02, 1.00
    raise AssertionError(design)


def make_rows() -> list[dict]:
    rows: list[dict] = []
    for task_index, task in enumerate(TASKS):
        for seed_index, seed in enumerate(SEEDS):
            for design in analyzer.FROZEN_DESIGNS:
                primary, clean, deep = _values(task_index, seed_index, design)
                rows.append({
                    "task": task,
                    "seed": seed,
                    "design": design,
                    analyzer.DEEP: deep,
                    "metrics": {
                        analyzer.PRIMARY: primary,
                        analyzer.CLEAN: clean,
                        analyzer.SECONDARY: 0.05,
                        analyzer.VARIANCE: 0.01,
                        analyzer.RANK: 24.0,
                        analyzer.CONVERGENCE: 0.01,
                        analyzer.INTEGRATOR: 1.00,
                    },
                })
    assert len(rows) == analyzer.FROZEN_CELL_COUNT
    return rows


def _row(rows: list[dict], *, design: str, ordinal: int = 0) -> dict:
    selected = [row for row in rows if row["design"] == design]
    return selected[ordinal]


def test_complete_passing_grid_uses_per_cell_envelopes() -> None:
    report = analyzer.analyze(make_rows(), [])

    assert report["status"] == "COMPLETE"
    assert report["scientific_label"] == "STABILIZED_LEWM_V8_CONFIRMATION_PASS"
    assert report["official_confirmation_result"] is True
    assert report["expected_cells"] == 200
    assert all(report["gates"].values())

    recurrent = report["contrasts"][
        f"{analyzer.CANDIDATE}_vs_recurrent_envelope:{analyzer.PRIMARY}"]
    assert recurrent["envelope_policy"] == "per_task_seed_identity_selected_once"
    assert recurrent["selection_metric"] == analyzer.PRIMARY
    assert recurrent["selected_reference_counts"] == {
        analyzer.GRU: 13,
        analyzer.SSM: 12,
    }
    assert recurrent["mean_paired_relative_reduction"] == pytest.approx(0.10)
    assert recurrent["paired_wins"] == 25
    assert recurrent["task_mean_wins"] == 5
    assert recurrent["bootstrap"]["ci95_low"] > 0

    endpoint = report["gate_receipts"][
        "learned_v8_vs_static_dynamic_envelope_noninferiority"]
    assert endpoint["passed"] is True
    assert endpoint["observed"]["mean_paired_relative_reduction"] == pytest.approx(
        (0.895 - 0.90) / 0.895)
    assert report["representation"]["observed"]["cells"] == 200
    assert report["convergence"]["observed"]["cells"] == 200


@pytest.mark.parametrize(
    ("reference", "gate"),
    (
        (analyzer.NO_ACTION, "action_causality"),
        (analyzer.SINGLE, "joint_state_use"),
    ),
)
def test_causal_gates_enforce_ci_wins_and_task_counts(
    reference: str,
    gate: str,
) -> None:
    rows = make_rows()
    reference_rows = [row for row in rows if row["design"] == reference]
    # Seventeen large wins and eight small losses retain >5% mean reduction,
    # but fail the frozen 18/25 causal-win requirement.
    for index, row in enumerate(reference_rows):
        row["metrics"][analyzer.PRIMARY] = 1.20 if index < 17 else 0.89

    report = analyzer.analyze(rows, [])
    receipt = report["gate_receipts"][gate]
    assert receipt["observed"]["mean_paired_relative_reduction"] > 0.05
    assert receipt["observed"]["paired_wins"] == 17
    assert receipt["passed"] is False
    assert report["scientific_label"] == "CONFIRMATION_FAILED"


def test_recurrent_gate_requires_crossed_ci_above_zero() -> None:
    rows = make_rows()
    # Four tasks improve by 15%; one task regresses by 40%. The equal-task
    # point effect is +4%, with 20/25 wins and 4/5 task wins, but uncertainty
    # must expose the heterogeneous task and reject confirmation.
    for row in rows:
        if row["design"] == analyzer.CANDIDATE:
            row["metrics"][analyzer.PRIMARY] = (
                0.85 if row["task"] != TASKS[-1] else 1.40)

    report = analyzer.analyze(rows, [])
    receipt = report["gate_receipts"]["v8_vs_per_cell_better_gru_ssm"]
    assert receipt["observed"]["mean_paired_relative_reduction"] == pytest.approx(0.04)
    assert receipt["observed"]["paired_wins"] == 20
    assert receipt["observed"]["task_mean_wins"] == 4
    assert receipt["observed"]["ci95_low"] < 0
    assert receipt["passed"] is False


def test_none_and_integrator_guards_enforce_their_win_counts() -> None:
    none_rows = make_rows()
    none_references = [row for row in none_rows if row["design"] == analyzer.NONE]
    for index, row in enumerate(none_references):
        row["metrics"][analyzer.PRIMARY] = 1.20 if index < 19 else 0.89
    none_report = analyzer.analyze(none_rows, [])
    none_receipt = none_report["gate_receipts"]["v8_vs_none"]
    assert none_receipt["observed"]["mean_paired_relative_reduction"] > 0.05
    assert none_receipt["observed"]["paired_wins"] == 19
    assert none_receipt["passed"] is False

    integrator_rows = make_rows()
    candidate_rows = [
        row for row in integrator_rows if row["design"] == analyzer.CANDIDATE]
    for index, row in enumerate(candidate_rows):
        row["metrics"][analyzer.INTEGRATOR] = 1.20 if index < 17 else 0.89
    integrator_report = analyzer.analyze(integrator_rows, [])
    integrator_receipt = integrator_report["gate_receipts"][
        "v8_vs_checkpoint_integrator"]
    assert integrator_receipt["observed"]["mean_paired_relative_reduction"] > 0.03
    assert integrator_receipt["observed"]["paired_wins"] == 17
    assert integrator_receipt["passed"] is False


def test_deep_gate_requires_three_positive_task_means() -> None:
    rows = make_rows()
    for row in rows:
        if row["design"] == analyzer.CANDIDATE:
            row[analyzer.DEEP] = 0.90 if row["task"] in TASKS[:2] else 1.05

    report = analyzer.analyze(rows, [])
    receipt = report["gate_receipts"]["deep_vs_per_cell_better_gru_ssm"]
    assert receipt["observed"]["task_mean_wins"] == 2
    assert receipt["passed"] is False


def test_endpoint_noninferiority_uses_point_and_crossed_ci() -> None:
    rows = make_rows()
    for row in rows:
        if row["design"] in analyzer.ENDPOINT_REFERENCES:
            row["metrics"][analyzer.PRIMARY] = 0.88

    report = analyzer.analyze(rows, [])
    receipt = report["gate_receipts"][
        "learned_v8_vs_static_dynamic_envelope_noninferiority"]
    assert receipt["observed"]["mean_paired_relative_reduction"] < -0.01
    assert receipt["observed"]["ci95_low"] < -0.01
    assert receipt["passed"] is False


def test_representation_and_convergence_gate_every_grid_cell() -> None:
    rows = make_rows()
    _row(rows, design=analyzer.NONE)["metrics"][analyzer.VARIANCE] = 0.00009
    _row(rows, design=analyzer.GRU)["metrics"][analyzer.CONVERGENCE] = -0.051

    report = analyzer.analyze(rows, [])
    representation = report["gate_receipts"]["healthy_representation"]
    convergence = report["gate_receipts"]["convergence"]
    assert representation["observed"]["variance_passing_cells"] == 199
    assert representation["passed"] is False
    assert convergence["observed"]["passing_cells"] == 199
    assert convergence["passed"] is False
    assert report["official_confirmation_result"] is False


def test_clean_guard_is_point_gated_and_reports_uncertainty() -> None:
    rows = make_rows()
    for row in rows:
        if row["design"] == analyzer.CANDIDATE:
            row["metrics"][analyzer.CLEAN] = 1.031

    report = analyzer.analyze(rows, [])
    receipt = report["gate_receipts"][
        "clean_prior_guard_vs_per_cell_better_gru_ssm"]
    assert receipt["observed"]["mean_paired_relative_degradation"] == pytest.approx(0.031)
    assert "ci95_low_reduction" in receipt["observed"]
    assert "ci95_high_reduction" in receipt["observed"]
    assert receipt["passed"] is False


def test_deep_and_clean_reuse_primary_selected_recurrent_identity() -> None:
    rows = make_rows()
    index = analyzer._index(rows)
    key = (TASKS[0], SEEDS[0])
    index[(*key, analyzer.GRU)]["metrics"][analyzer.PRIMARY] = 0.50
    index[(*key, analyzer.SSM)]["metrics"][analyzer.PRIMARY] = 1.50
    index[(*key, analyzer.GRU)][analyzer.DEEP] = 4.0
    index[(*key, analyzer.SSM)][analyzer.DEEP] = 0.25

    contrast = analyzer.selected_identity_contrast(
        index, analyzer.CANDIDATE, analyzer.RECURRENT_REFERENCES,
        analyzer.DEEP, selection_metric=analyzer.PRIMARY,
        label="primary_selected_recurrent")
    assert contrast["selected_reference_counts"][analyzer.GRU] >= 1
    # The first cell must use GRU's deep value (4.0), not the lower SSM deep
    # value (0.25), because recurrent identity was frozen by PRIMARY.
    expected = (4.0 - index[(*key, analyzer.CANDIDATE)][analyzer.DEEP]) / 4.0
    assert contrast["cell_effects"][0][0] == pytest.approx(expected)


def test_incomplete_or_wrong_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = analyzer.analyze(make_rows()[:-1], [])
    assert incomplete["status"] == "INCOMPLETE_OR_INVALID"
    assert incomplete["official_confirmation_result"] is False
    assert incomplete["artifact_integrity_errors"]

    monkeypatch.setattr(analyzer.runner, "EPOCHS", 60)
    wrong_contract = analyzer.analyze(make_rows(), [])
    assert wrong_contract["status"] == "INCOMPLETE_OR_INVALID"
    assert "requires 100 epochs" in wrong_contract["protocol_contract_errors"][0]


def test_nonfinite_comparator_health_fails_closed() -> None:
    rows = make_rows()
    _row(rows, design=analyzer.SSM)["metrics"][analyzer.RANK] = float("nan")
    report = analyzer.analyze(rows, [])
    assert report["status"] == "INCOMPLETE_OR_INVALID"
    assert report["official_confirmation_result"] is False
    assert "not a finite scalar" in report["artifact_integrity_errors"][0]


def test_write_once_outputs(tmp_path: Path) -> None:
    rows = make_rows()
    report = analyzer.analyze(rows, [])
    analysis_path = tmp_path / analyzer.ANALYSIS_NAME
    analyzer._write_json_exclusive(analysis_path, report)
    with analysis_path.open(encoding="utf-8") as stream:
        assert json.load(stream)["status"] == "COMPLETE"
    with pytest.raises(FileExistsError):
        analyzer._write_json_exclusive(analysis_path, report)

    analyzer.write_csvs(tmp_path, rows, report)
    assert (tmp_path / analyzer.CELLS_NAME).is_file()
    assert (tmp_path / analyzer.CONTRASTS_NAME).is_file()
    with pytest.raises(FileExistsError):
        analyzer.write_csvs(tmp_path, rows, report)
