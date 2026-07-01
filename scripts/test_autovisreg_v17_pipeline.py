#!/usr/bin/env python3
"""Standalone synthetic tests for the AutoVISReg-v17 run/analyze contract."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.analyze_autovisreg_v17 as analysis
import scripts.run_autovisreg_v17 as runner
import scripts.train_autovisreg_v17 as trainer


def _metrics(
        task: str, design: str, seed: int, *, candidate_factor: float = 1.0
        ) -> dict[str, object]:
    family, memory = runner.design_parts(design)
    task_index = runner.TASKS.index(task)
    seed_index = runner.SEEDS.index(seed)
    base = 1.0 + 0.1 * task_index + 0.01 * seed_index
    candidate = family == "autovisreg"
    factor = candidate_factor if candidate else 1.0
    return {
        "env": f"dmc:{task}", "design": design, "seed": seed,
        "epochs": runner.EPOCHS, "regularizer": family,
        "memory_architecture": memory, "confirmation_evidence": False,
        analysis.PRIMARY: base * factor,
        analysis.CLEAN: base * 0.9 * factor,
        analysis.INTEGRATOR: base * 1.2,
        analysis.VARIANCE: 0.20 if candidate else 0.15,
        analysis.RANK: 20.0 if candidate else 18.0,
        analysis.CONVERGENCE: 0.01,
        analysis.VAL_PREDICTIVE: base * 0.4 * factor,
        "final_val_loss": base * 0.5 * factor,
        "mean_epoch_seconds": 10.0,
        "peak_vram_bytes": 1024,
        "train_gradient_prediction_norm": 2.0,
        "train_gradient_regularizer_norm": 1.0,
        "train_gradient_cosine": -0.25 if candidate else 0.1,
        "train_gradient_adaptive_scale": 2.0 if candidate else 1.0,
        "train_gradient_preclip_norm": 1.5,
        "train_gradient_clip_fraction": 0.25,
        "train_gradient_conflict_fraction": 0.5 if candidate else 0.0,
    }


def _synthetic_rows(candidate_factor: float = 0.8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in runner.TASKS:
        for seed in runner.SEEDS:
            for design in runner.DESIGNS:
                family, memory = runner.design_parts(design)
                rows.append({
                    "task": task, "design": design, "seed": seed,
                    "family": family, "memory": memory,
                    "metrics": _metrics(
                        task, design, seed, candidate_factor=candidate_factor),
                    "directory": f"/tmp/{task}/{design}/{seed}",
                    "artifact_sha256": {"model.pt": "a" * 64},
                    "wandb_state": "finished",
                })
    return rows


def _assert_raises(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"{function.__name__} did not raise {error_type.__name__}")


def test_exact_closed_grid_and_order() -> None:
    assert runner.OBJECTIVE_FAMILIES == ("autovisreg", "vicreg")
    assert runner.MEMORY_VARIANTS == ("none", "ssm", "hacssmv8")
    assert runner.DESIGNS == (
        "autovisreg_none", "vicreg_none",
        "autovisreg_ssm", "vicreg_ssm",
        "autovisreg_hacssmv8", "vicreg_hacssmv8",
    )
    assert runner.SEEDS == (17001, 17002, 17003)
    assert runner.EPOCHS == 30
    assert len(runner.cell_specs()) == 72 == analysis.EXPECTED_CELLS
    assert len(set(runner.cell_specs())) == 72
    assert runner.design_parts("autovisreg_none") == ("autovisreg", "none")
    assert runner.design_parts("vicreg_hacssmv8") == ("vicreg", "hacssmv8")


def test_commands_match_frozen_trainer_and_have_no_ssl_knobs() -> None:
    root = Path("/tmp/autovisreg-v17-test").resolve()
    online = runner.train_command(
        "python", root, runner.DEFAULT_STUDY, runner.EPOCHS,
        runner.TASKS[0], "autovisreg_none", runner.SEEDS[0], wandb=True)
    assert Path(online[1]).name == "train_autovisreg_v17.py"
    assert online[online.index("--design") + 1] == "autovisreg_none"
    assert "--wandb" in online and "--no-wandb" not in online
    assert online[online.index("--wandb-mode") + 1] == "online"
    assert online[online.index("--wandb-entity") + 1] == runner.WANDB_ENTITY
    forbidden_fragments = (
        "lambda", "temperature", "margin", "projection", "subspace",
        "regularizer-weight", "loss-weight")
    assert not any(
        fragment in token
        for token in online for fragment in forbidden_fragments)

    disabled = runner.train_command(
        "python", root, runner.DEFAULT_STUDY, runner.EPOCHS,
        runner.TASKS[0], "autovisreg_none", runner.SEEDS[0], wandb=False)
    assert "--no-wandb" in disabled and "--wandb" not in disabled
    records = runner.command_records(
        "python", root, runner.DEFAULT_STUDY, runner.EPOCHS, wandb=True)
    assert len(records) == 72
    assert len({runner.json_sha256(row["argv"]) for row in records}) == 72
    parsed = trainer.parse_args(online[2:])
    assert parsed.design == "autovisreg_none"
    assert parsed.epochs == runner.EPOCHS
    assert parsed.wandb and parsed.wandb_mode == "online"


def _write_cell(root: Path, *, wandb: bool) -> tuple[str, str, int, Path]:
    task, design, seed = (
        runner.TASKS[0], "autovisreg_none", runner.SEEDS[0])
    directory = runner.run_directory(root, task, design, seed)
    directory.mkdir(parents=True)
    rollout_path = directory / "eval_rollout.npz"
    np.savez_compressed(rollout_path, values=np.arange(5, dtype=np.float32))
    metrics = _metrics(task, design, seed, candidate_factor=0.8)
    rollout_hash = runner.file_sha256(rollout_path)
    metrics["eval_rollout_sha256"] = rollout_hash
    (directory / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8")
    torch.save({
        "args": {"design": design, "seed": seed, "epochs": runner.EPOCHS},
        "final_metrics": metrics,
        "history": [{"epoch": epoch, "loss": 1.0 / epoch}
                    for epoch in range(1, runner.EPOCHS + 1)],
        "model_state_dict": {"weight": torch.ones(2)},
    }, directory / "model.pt")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "mode": "online" if wandb else "disabled",
        "study": runner.DEFAULT_STUDY,
        "state": "finished" if wandb else "not_requested",
        "eval_rollout_sha256": rollout_hash,
    }
    if wandb:
        receipt.update({
            "run_id": "synthetic-run", "run_name": "synthetic-name",
            "url": "https://wandb.ai/synthetic",
            "entity": runner.WANDB_ENTITY, "project": runner.WANDB_PROJECT,
            "eval_rollout_artifact_name": "eval-rollout-synthetic-run",
        })
    (directory / "wandb_run.json").write_text(
        json.dumps(receipt), encoding="utf-8")
    return task, design, seed, directory


def test_core_artifacts_require_finished_online_wandb_receipt() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        task, design, seed, directory = _write_cell(root, wandb=True)
        validated = runner.validate_core_artifacts(
            root, task, design, seed, runner.EPOCHS, wandb_expected=True)
        assert validated["headline_metric"] > 0
        assert validated["wandb_state"] == "finished"
        assert set(validated["artifact_sha256"]) == set(runner.CORE_ARTIFACTS)

        receipt_path = directory / "wandb_run.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["state"] = "sync_failed"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        _assert_raises(
            runner.ArtifactError, runner.validate_core_artifacts,
            root, task, design, seed, runner.EPOCHS, wandb_expected=True)


def test_disabled_wandb_receipt_is_explicitly_validated() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        task, design, seed, _ = _write_cell(root, wandb=False)
        validated = runner.validate_core_artifacts(
            root, task, design, seed, runner.EPOCHS, wandb_expected=False)
        assert validated["wandb_state"] == "not_requested"
        _assert_raises(
            runner.ArtifactError, runner.validate_core_artifacts,
            root, task, design, seed, runner.EPOCHS, wandb_expected=True)


def test_analysis_paired_effects_and_collapse_repair_gates() -> None:
    report = analysis.analyze_rows(_synthetic_rows(), [])
    assert report["status"] == "COMPLETE"
    assert report["artifact_integrity_passed"]
    assert report["scientific_label"] == "ADAPTIVE_COLLAPSE_REPAIR_FULL_GRID_PASS"
    contrasts = report["autovisreg_vs_vicreg"]
    assert len(contrasts) == 15
    primary = contrasts[
        "autovisreg_none_vs_vicreg_none:heldout_prior_state_nmse"]
    assert primary["direction"] == "lower"
    assert primary["pairs"] == 12 and primary["wins"] == 12
    assert abs(primary["paired_relative_improvement"]["mean"] - 0.2) < 1e-12
    rank = contrasts[
        "autovisreg_none_vs_vicreg_none:encoder_covariance_effective_rank"]
    assert rank["direction"] == "higher" and rank["wins"] == 12
    diagnostics = report[
        "representation_convergence_and_gradient_diagnostics"]
    assert diagnostics["candidate_host_only_gate"]["all_pass"]
    assert diagnostics["candidate_full_grid_gate"]["all_pass"]
    assert diagnostics["by_design"]["autovisreg_none"]["rank"][
        "cells_passing"] == 12
    assert len(analysis.cell_csv_rows(_synthetic_rows())) == 72
    assert len(analysis.contrast_csv_rows(report)) == 15


def test_analysis_distinguishes_host_only_and_full_grid_failure() -> None:
    rows = _synthetic_rows()
    for row in rows:
        if row["design"] == "autovisreg_ssm":
            row["metrics"][analysis.RANK] = 3.0
    report = analysis.analyze_rows(rows, [])
    assert report["scientific_label"] == "ADAPTIVE_HOST_REPAIR_ONLY"
    diagnostics = report[
        "representation_convergence_and_gradient_diagnostics"]
    assert diagnostics["candidate_host_only_gate"]["all_pass"]
    assert not diagnostics["candidate_full_grid_gate"]["all_pass"]


def test_incomplete_analysis_fails_closed() -> None:
    report = analysis.analyze_rows(_synthetic_rows()[:-1], ["missing cell"])
    assert report["status"] == "INCOMPLETE_OR_INVALID"
    assert report["scientific_label"] == "NOT_EVALUATED_INCOMPLETE"
    assert not report["artifact_integrity_passed"]
    assert report["completed_valid_cells"] == 71


def test_resume_ledger_roundtrip_retains_attempt_history() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        ledger = runner.RunLedger(root, resume=False)
        failed = {
            "task": runner.TASKS[0], "design": runner.DESIGNS[0],
            "seed": runner.SEEDS[0], "status": "failed", "error": "synthetic",
        }
        ledger.record(failed, attempt=True)
        complete = {**failed, "status": "complete", "error": None}
        ledger.record(complete, attempt=True)
        resumed = runner.RunLedger(root, resume=True)
        key = runner.cell_key(
            runner.TASKS[0], runner.DESIGNS[0], runner.SEEDS[0])
        assert resumed.records[key]["status"] == "complete"
        assert [row["status"] for row in resumed.attempts] == [
            "failed", "complete"]


def test_protocol_declares_no_selectable_ssl_hyperparameters() -> None:
    root = Path("/tmp/autovisreg-v17-test").resolve()
    protocol = runner.protocol_payload(
        python="python", output_root=root, log_root=root / "logs",
        study=runner.DEFAULT_STUDY, epochs=runner.EPOCHS,
        gpu_ids=("0", "1", "2", "3"), wandb=True,
        data={}, source={})
    assert protocol["runs"] == 72
    assert protocol["candidate_ssl_selectable_hyperparameters"] == []
    assert protocol["core_artifacts"] == list(runner.CORE_ARTIFACTS)
    assert "wandb_run.json" in protocol["core_artifacts"]
    assert protocol["resume_granularity"] == "complete_cell_only"


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} AutoVISReg-v17 pipeline tests")


if __name__ == "__main__":
    main()
