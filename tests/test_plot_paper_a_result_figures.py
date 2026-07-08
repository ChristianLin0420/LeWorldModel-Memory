from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import plot_paper_a_strengthened as plots


ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
LEARNED = ARMS[1:]
REQUIRED_STEMS = (
    "fig_mem_architecture", "fig_mem_tasks", "fig_mem_matched",
    "fig_mem_dinowm_carrier", "fig_mem_pointmaze",
)


def _stat(mean: float, width: float = 0.02) -> dict[str, object]:
    return {"mean": mean, "ci95": [mean - width, mean + width]}


def _dinowm_summary() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for task, chance in (
            ("transient-visual-token-recall", 0.25),
            ("multi-item-visual-binding-recall", 1 / 6)):
        ages: dict[str, Any] = {}
        for age_index, age in enumerate((4, 8, 15)):
            ages[str(age)] = {
                "arms": {
                    arm: {
                        "balanced_accuracy": _stat(
                            chance + 0.20 - 0.035 * age_index
                            + 0.012 * arm_index)
                    }
                    for arm_index, arm in enumerate(ARMS)
                },
                "paired_vs_none": {
                    arm: _stat(0.035 + 0.008 * arm_index
                               - 0.004 * age_index, 0.01)
                    for arm_index, arm in enumerate(LEARNED)
                },
                "full_vs_context_reset": {
                    arm: _stat(0.025 + 0.007 * arm_index
                               - 0.003 * age_index, 0.009)
                    for arm_index, arm in enumerate(LEARNED)
                },
            }
        results[task] = {"chance": chance, "ages": ages}
    return {
        "schema": "dinowm_wave2_spatial_carrier_summary_v1",
        "status": "complete",
        "grid": {"tasks": 2, "arms": 5, "seeds": 5, "cells": 50},
        "results": results,
    }


def _pointmaze_summaries() \
        -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    for age_index, age in enumerate((4, 8, 15)):
        results[str(age)] = {
            "arms": {
                arm: {
                    "balanced_accuracy": _stat(
                        0.27 + 0.045 * arm_index - 0.02 * age_index)
                }
                for arm_index, arm in enumerate(ARMS)
            },
            "paired_vs_none": {
                arm: _stat(0.035 + 0.012 * arm_index
                           - 0.004 * age_index, 0.012)
                for arm_index, arm in enumerate(LEARNED)
            },
            "full_vs_context_reset": {
                arm: _stat(0.025 + 0.010 * arm_index
                           - 0.003 * age_index, 0.010)
                for arm_index, arm in enumerate(LEARNED)
            },
        }
    none_mean = 0.30
    execution_deltas = {
        "none": 0.0, "gru": 0.02, "lstm": -0.01,
        "ssm": 0.05, "fixed_trust": 0.08,
    }
    use_arms = {
        arm: {
            "goal_accuracy": _stat(0.40 + 0.04 * index),
            "executed_success": _stat(none_mean + execution_deltas[arm]),
            "contrast_vs_none": _stat(execution_deltas[arm], 0.015),
            "contrast_vs_random": _stat(
                none_mean + execution_deltas[arm] - 0.24, 0.018),
            "resolved_execution_gain": arm in {"ssm", "fixed_trust"},
        }
        for index, arm in enumerate(ARMS)
    }
    top = {
        "schema": "dinowm_pointmaze_wave3_summary_v1",
        "status": "complete",
    }
    carrier = {
        "schema": "dinowm_pointmaze_wave3_carrier_summary_v1",
        "status": "complete",
        "grid": {"tasks": 1, "arms": 5, "seeds": 5, "cells": 25},
        "results": results,
    }
    use = {
        "schema": "dinowm_pointmaze_wave3_external_use_v1",
        "status": "complete",
        "arms": use_arms,
        "realized_random_goal": _stat(0.24, 0.012),
        "oracle_executed_success": 0.93,
    }
    return top, carrier, use


def _matched_summary() -> dict[str, Any]:
    ages = (4, 8, 15)
    hosts: dict[str, Any] = {}
    for host_index, host in enumerate(("reacher", "pusht")):
        arm_records = {}
        for arm_index, arm in enumerate(ARMS):
            start = 0.78 - 0.12 * host_index + 0.01 * arm_index
            arm_records[arm] = {
                f"age-{age}": _stat(start - 0.018 * age_index, 0.015)
                for age_index, age in enumerate(ages)
            }
        hosts[host] = {
            "arms": arm_records,
            "fixed_minus_ssm": {"age-15": _stat(0.06, 0.02)},
        }
    return {
        "ages": list(ages),
        "hosts": hosts,
        "primary_ranking_interaction": {
            "mean": 0.0,
            "ci95": [-0.04, 0.04],
            "ci90": [-0.03, 0.03],
            "equivalence_margin": 0.05,
            "equivalent_within_margin": True,
        },
    }


def _pointmaze_visuals() \
        -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    frames = []
    for index in range(3):
        value = np.zeros((32, 32, 3), dtype=np.uint8)
        value[..., index] = 80 + 50 * index
        frames.append(value)
    paired = frames[0].copy()
    paired[8:16, 8:16] = 255
    return frames, (frames[0], paired)


def _has_horizontal_reference(axis: plt.Axes, value: float) -> bool:
    for line in axis.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        if y.size >= 2 and np.allclose(y, value):
            return True
    return False


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _synthetic_manifest_inputs(root: Path) -> None:
    _write(root / "scripts/plot_paper_a_strengthened.py",
           Path(plots.__file__).resolve().read_bytes())
    for relative in (
            "outputs/paper_a_cross_wave_completion/receipt.json",
            "outputs/paper_a_statistics_independent/receipt.json",
            "outputs/paper_a_matched_color_v1_1/summary.json",
            "outputs/dinowm_wave2_spatial_carrier_v1_1/formal/summary.json",
            "outputs/dinowm_pointmaze_wave3/formal/summary.json",
            "outputs/dinowm_pointmaze_wave3/formal/carrier_summary.json",
            "outputs/dinowm_pointmaze_wave3/formal/external_use_summary.json"):
        _write(root / relative, (relative + "\n").encode())


def _synthetic_generation() -> None:
    plots.FIG.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(REQUIRED_STEMS):
        _write(plots.FIG / f"{stem}.pdf", f"pdf-{index}".encode())
        _write(plots.FIG / f"{stem}.png", f"png-{index}".encode())


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def test_staged_generation_failure_leaves_published_directory_byte_identical(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = tmp_path / "paper_a/figures"
    _write(final / "fig_mem_existing.pdf", b"old figure")
    _write(final / "manifest.json", b"old manifest")
    _write(final / "notes.txt", b"preserve me")
    before = _snapshot(final)
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    monkeypatch.setattr(plots, "FIG", final)
    monkeypatch.setattr(plots, "_require_final_audit_receipts", lambda: None)

    def fail_after_one_staged_write() -> None:
        _write(plots.FIG / "fig_mem_architecture.pdf", b"partial")
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(plots, "_generate_all_figures",
                        fail_after_one_staged_write)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        plots.main()

    assert _snapshot(final) == before
    assert plots.FIG == final
    assert not list((tmp_path / "paper_a").glob(".figures-staging.*"))
    assert not list((tmp_path / "paper_a").glob(".figures-quarantine.*"))


def test_staged_success_binds_final_paths_and_commits_manifest_last(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = tmp_path / "paper_a/figures"
    _write(final / "fig_mem_obsolete.pdf", b"obsolete")
    _write(final / "manifest.json", b"old manifest")
    _write(final / "notes.txt", b"preserve me")
    _synthetic_manifest_inputs(tmp_path)
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    monkeypatch.setattr(plots, "FIG", final)
    monkeypatch.setattr(plots, "_require_final_audit_receipts", lambda: None)
    monkeypatch.setattr(plots, "_generate_all_figures", _synthetic_generation)

    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def track_replace(source: str | os.PathLike[str],
                      destination: str | os.PathLike[str]) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(plots.os, "replace", track_replace)
    plots.main()

    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_assets = {
        f"{stem}.{suffix}" for stem in REQUIRED_STEMS
        for suffix in ("pdf", "png")
    }
    assert manifest["schema"] == "paper_a_figure_manifest_v1"
    assert manifest["status"] == "complete"
    assert set(manifest["artifacts"]) == expected_assets
    for name, record in manifest["artifacts"].items():
        path = final / name
        assert record == {
            "path": f"paper_a/figures/{name}",
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert not (final / "fig_mem_obsolete.pdf").exists()
    assert (final / "notes.txt").read_bytes() == b"preserve me"
    final_commits = [destination for _source, destination in replacements
                     if destination.parent == final]
    assert final_commits[-1] == manifest_path
    assert all(path.name.startswith("fig_mem_")
               for path in final_commits[:-1])
    assert plots.FIG == final
    assert not list((tmp_path / "paper_a").glob(".figures-staging.*"))
    assert not list((tmp_path / "paper_a").glob(".figures-quarantine.*"))


def test_publish_failure_cannot_leave_stale_old_manifest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = tmp_path / "paper_a/figures"
    _write(final / "fig_mem_existing.pdf", b"old figure")
    _write(final / "manifest.json", b"stale manifest")
    _write(final / "notes.txt", b"preserve me")
    _synthetic_manifest_inputs(tmp_path)
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    monkeypatch.setattr(plots, "FIG", final)
    monkeypatch.setattr(plots, "_require_final_audit_receipts", lambda: None)
    monkeypatch.setattr(plots, "_generate_all_figures", _synthetic_generation)

    real_replace = os.replace
    committed_assets = 0

    def fail_second_asset(source: str | os.PathLike[str],
                          destination: str | os.PathLike[str]) -> None:
        nonlocal committed_assets
        target = Path(destination)
        if target.parent == final and target.name.startswith("fig_mem_"):
            committed_assets += 1
            if committed_assets == 2:
                raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(plots.os, "replace", fail_second_asset)
    with pytest.raises(OSError, match="injected publish failure"):
        plots.main()

    assert not (final / "manifest.json").exists()
    assert (final / "fig_mem_existing.pdf").read_bytes() == b"old figure"
    assert {path.name for path in plots._figure_assets(final)} == {
        "fig_mem_existing.pdf"}
    assert (final / "notes.txt").read_bytes() == b"preserve me"
    assert plots.FIG == final
    assert not list((tmp_path / "paper_a").glob(".figures-staging.*"))
    assert not list((tmp_path / "paper_a").glob(".figures-quarantine.*"))


def test_architecture_uses_concise_names_and_labeled_state_paths(
        monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [np.full((32, 32, 3), 30 + 20 * index, dtype=np.uint8)
              for index in range(3)]
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        plots, "_matched_reacher_visuals",
        lambda: (frames, (frames[0], frames[1])))
    monkeypatch.setattr(
        plots, "save",
        lambda figure, stem: captured.update(figure=figure, stem=stem))

    plots.figure_architecture()
    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert captured["stem"] == "fig_mem_architecture"
    assert np.allclose(figure.get_size_inches(), [5.45, 2.18])
    labels = {text.get_text() for text in figure.axes[0].texts}
    assert {"AUDITED CARRIER", "Frozen predictor", "pre-decision read",
            "executed use"}.issubset(labels)
    assert "Predictive loss" not in labels
    assert "Host-specific legal read" not in labels
    plt.close(figure)


def test_task_cue_crops_are_enlarged_and_keep_square_axes() -> None:
    frames = [np.full((32, 32, 3), 20 + 20 * index, dtype=np.uint8)
              for index in range(3)]
    alternate = frames[0].copy()
    alternate[9:17, 11:19] = 255
    fig, axis = plt.subplots(figsize=(5.45, 1.0))
    plots._compact_task_row(
        axis, host="Synthetic host", task="synthetic cue · 4-way",
        frames=frames, difference=(frames[0], alternate),
        cue_score=1.0, shortcut_score=0.25, shaded=False)
    fig.canvas.draw()

    assert len(axis.child_axes) == 6
    crop_axes = axis.child_axes[-3:]
    parent_width = axis.get_position().width
    crop_widths = [child.get_position().width / parent_width
                   for child in crop_axes]
    assert all(width >= 0.049 for width in crop_widths)
    assert all(child.get_box_aspect() == 1 for child in crop_axes)
    assert all(child.get_aspect() == 1.0 for child in crop_axes)
    plt.close(fig)


def test_matched_forest_uses_full_names_without_micro_legend_sentence(
        monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(plots, "load_json", lambda _path: _matched_summary())
    monkeypatch.setattr(
        plots, "save",
        lambda figure, stem: captured.update(figure=figure, stem=stem))

    plots.figure_matched_lewm()
    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert captured["stem"] == "fig_mem_matched"
    assert len(figure.axes) == 4
    gaps, interaction = figure.axes[2:]
    assert [tick.get_text() for tick in gaps.get_yticklabels()] == [
        "Reacher", "PushT"]
    assert gaps.get_title(loc="left") == "(c) Same age-15 ordering"
    assert interaction.get_title(loc="left") == "(d) Cross-host gap ≈ 0"
    assert not interaction.get_yticks().size
    assert any("TOST equivalent" in text.get_text()
               for text in interaction.texts)
    plt.close(figure)


def test_dinowm_carrier_renders_complete_two_by_three_design(
        monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        plots, "load_verified_dinowm_pusht", _dinowm_summary)
    monkeypatch.setattr(
        plots, "save",
        lambda figure, stem: captured.update(figure=figure, stem=stem))

    plots.figure_dinowm_carrier()
    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert captured["stem"] == "fig_mem_dinowm_carrier"
    assert len(figure.axes) == 6
    expected_titles = [
        "(a) Token · host output",
        "(b) Token · gain vs no state",
        "(c) Token · gain vs reset",
        "(d) Binding · host output",
        "(e) Binding · gain vs no state",
        "(f) Binding · gain vs reset",
    ]
    assert [axis.get_title(loc="left") for axis in figure.axes] \
        == expected_titles

    five_arm_labels = {plots.ARM_SHORT[arm] for arm in ARMS}
    for index in (0, 3):
        assert set(figure.axes[index].get_legend_handles_labels()[1]) \
            == five_arm_labels
    for index in (1, 2, 4, 5):
        assert set(figure.axes[index].get_legend_handles_labels()[1]) \
            == {plots.ARM_SHORT[arm] for arm in LEARNED}
        assert _has_horizontal_reference(figure.axes[index], 0.0)
    assert _has_horizontal_reference(figure.axes[0], 0.25)
    assert _has_horizontal_reference(figure.axes[3], 1 / 6)
    assert any("chance 0.250" in text.get_text()
               for text in figure.axes[0].texts)
    assert any("chance 0.167" in text.get_text()
               for text in figure.axes[3].texts)
    assert len(figure.legends) == 1
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "No state (full-state only)",
        *[plots.ARM_SHORT[arm] for arm in LEARNED],
    ]
    plt.close(figure)


def test_pointmaze_renders_compact_result_only_two_by_two_design(
        monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(plots, "load_verified_pointmaze", _pointmaze_summaries)
    monkeypatch.setattr(
        plots, "_pointmaze_visuals",
        lambda: pytest.fail("result-only figure must not reload task renders"))
    monkeypatch.setattr(
        plots, "goal_card",
        lambda _label: pytest.fail("result-only figure must not draw goal cards"))
    monkeypatch.setattr(
        plots, "save",
        lambda figure, stem: captured.update(figure=figure, stem=stem))

    plots.figure_pointmaze_retention_use()
    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert captured["stem"] == "fig_mem_pointmaze"
    assert np.allclose(figure.get_size_inches(), [5.45, 3.10])
    assert len(figure.axes) == 4
    axes_by_title = {
        axis.get_title(loc="left"): axis for axis in figure.axes
        if axis.get_title(loc="left")}
    expected_titles = {
        "(a) Full-state retention",
        "(b) Full − no state",
        "(c) Full − context reset",
        "(d) External execution · age 15",
    }
    assert set(axes_by_title) == expected_titles
    assert all(title.startswith(f"({panel})")
               for title, panel in zip(sorted(axes_by_title), "abcd"))
    assert all(not axis.child_axes and not axis.images for axis in figure.axes)

    absolute = axes_by_title["(a) Full-state retention"]
    assert set(absolute.get_legend_handles_labels()[1]) == {
        plots.ARM_SHORT[arm] for arm in ARMS}
    assert _has_horizontal_reference(absolute, 0.25)
    assert not absolute.texts
    for title in ("(b) Full − no state", "(c) Full − context reset"):
        axis = axes_by_title[title]
        assert set(axis.get_legend_handles_labels()[1]) == {
            plots.ARM_SHORT[arm] for arm in LEARNED}
        assert _has_horizontal_reference(axis, 0.0)
    execution = axes_by_title["(d) External execution · age 15"]
    assert execution.get_xlabel() == "executed success"
    assert [tick.get_text() for tick in execution.get_yticklabels()] == [
        "No state", "GRU", "LSTM", "State-space", "Fixed-trust"]
    assert any("random" in text.get_text() for text in execution.texts)
    assert any("oracle 0.930" in text.get_text() for text in execution.texts)
    assert any("resolved vs both controls" in text.get_text()
               for text in execution.texts)
    assert len(figure.legends) == 1
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "No state", *[plots.ARM_SHORT[arm] for arm in LEARNED]]
    plt.close(figure)


def test_pointmaze_rejects_inconsistent_absolute_and_paired_execution_means(
        monkeypatch: pytest.MonkeyPatch) -> None:
    top, carrier, use = _pointmaze_summaries()
    use["arms"]["gru"]["executed_success"]["mean"] = 0.5
    monkeypatch.setattr(
        plots, "load_verified_pointmaze", lambda: (top, carrier, use))
    monkeypatch.setattr(
        plots, "save", lambda *_args, **_kwargs: pytest.fail(
            "save must not run after a consistency failure"))

    try:
        with pytest.raises(
                RuntimeError,
                match="PointMaze absolute/paired execution mismatch: gru"):
            plots.figure_pointmaze_retention_use()
    finally:
        plt.close("all")
