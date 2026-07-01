#!/usr/bin/env python3
"""Standalone synthetic tests for the Sub-JEPA-v16 runner/analyzer contract."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.analyze_subjepa_v16 as analysis
import scripts.run_subjepa_v16 as runner


def _synthetic_rows():
    family_factor = {
        "fullsig": 1.0,
        "subjepa16": .80,
        "subjepa32": .75,
        "vicreg": 1.10,
    }
    memory_factor = {"none": 1.0, "ssm": .90, "hacssmv8": .85}
    rows = []
    for task_index, task in enumerate(runner.TASKS):
        for seed_index, seed in enumerate(runner.SEEDS):
            base = 1.0 + .1 * task_index + .01 * seed_index
            for design in runner.DESIGNS:
                family, memory, tokens = runner.design_parts(design)
                primary = base * family_factor[family] * memory_factor[memory]
                rows.append({
                    "task": task,
                    "design": design,
                    "seed": seed,
                    "family": family,
                    "memory": memory,
                    "tokens": tokens,
                    "wandb_state": "finished",
                    "directory": f"/tmp/{task}/{design}/{seed}",
                    "artifact_sha256": {"model.pt": "a" * 64},
                    "metrics": {
                        analysis.PRIMARY: primary,
                        analysis.CLEAN: primary * .9,
                        analysis.INTEGRATOR: base * 1.25,
                        analysis.RANK: 24.0 + task_index,
                        analysis.CONVERGENCE: .01,
                        "final_val_loss": primary * .5,
                        "val_predictive_loss": primary * .4,
                        "mean_epoch_seconds": 10.0,
                        "peak_vram_bytes": 1024,
                    },
                })
    return rows


def test_closed_grid_and_design_parser() -> None:
    assert len(runner.DESIGNS) == 12
    assert len(runner.cell_specs()) == 144 == analysis.EXPECTED_CELLS
    assert len(set(runner.cell_specs())) == 144
    assert runner.design_parts("fullsig_none") == ("fullsig", "none", None)
    assert runner.design_parts("subjepa16_ssm") == ("subjepa16", "ssm", 16)
    assert runner.design_parts("subjepa32_hacssmv8") == (
        "subjepa32", "hacssmv8", 32)


def test_commands_use_trainer_contract_and_wandb_switch() -> None:
    root = Path("/tmp/subjepa-v16-test").resolve()
    online = runner.train_command(
        "python", root, runner.DEFAULT_STUDY, runner.EPOCHS,
        runner.TASKS[0], "subjepa16_none", runner.SEEDS[0], wandb=True)
    assert "--design" in online
    assert online[online.index("--design") + 1] == "subjepa16_none"
    assert "--memory-mode" not in online
    assert "--wandb" in online and "--no-wandb" not in online
    assert online[online.index("--wandb-mode") + 1] == "online"
    assert online[online.index("--wandb-entity") + 1] == runner.WANDB_ENTITY
    offline = runner.train_command(
        "python", root, runner.DEFAULT_STUDY, runner.EPOCHS,
        runner.TASKS[0], "subjepa16_none", runner.SEEDS[0], wandb=False)
    assert "--no-wandb" in offline and "--wandb" not in offline
    records = runner.command_records(
        "python", root, runner.DEFAULT_STUDY, runner.EPOCHS, wandb=True)
    assert len(records) == 144
    assert len({runner.json_sha256(row["argv"]) for row in records}) == 144


def test_local_core_artifact_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        task, design, seed = "cartpole.swingup", "subjepa16_none", 16001
        directory = runner.run_directory(root, task, design, seed)
        directory.mkdir(parents=True)
        rollout_path = directory / "eval_rollout.npz"
        np.savez_compressed(rollout_path, values=np.arange(5, dtype=np.float32))
        metrics = {
            "env": f"dmc:{task}", "design": design, "seed": seed,
            "epochs": runner.EPOCHS,
            analysis.PRIMARY: 1.0, analysis.INTEGRATOR: 1.2,
            analysis.RANK: 20.0, analysis.CONVERGENCE: .01,
            "eval_rollout_sha256": runner.file_sha256(rollout_path),
        }
        (directory / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8")
        torch.save({
            "args": {"design": design, "seed": seed, "epochs": runner.EPOCHS},
            "final_metrics": metrics,
            "history": [{"epoch": epoch}
                        for epoch in range(1, runner.EPOCHS + 1)],
            "model_state_dict": {"weight": torch.ones(2)},
        }, directory / "model.pt")
        validated = runner.validate_core_artifacts(
            root, task, design, seed, runner.EPOCHS)
        assert validated["headline_metric"] == 1.0
        assert set(runner.CORE_ARTIFACTS) <= set(validated["artifact_sha256"])


def test_paired_contrasts_and_full_analysis() -> None:
    rows = _synthetic_rows()
    report = analysis.analyze_rows(rows, [])
    assert report["status"] == "COMPLETE"
    assert report["artifact_integrity_passed"]
    sub16 = report["subjepa_vs_fullsig"]["subjepa16_none"]
    assert sub16["pairs"] == 12
    assert sub16["wins"] == 12 and sub16["losses"] == 0
    assert abs(sub16["paired_relative_reduction"]["mean"] - .20) < 1e-12
    versus_vicreg = report["subjepa16_vs_vicreg"]["subjepa16_ssm"]
    assert versus_vicreg["wins"] == 12
    stress = report["subjepa32_vs_subjepa16_stress"]["subjepa32_none"]
    assert stress["wins"] == 12
    memory = report["memory_vs_none"]["subjepa32_hacssmv8"]
    assert memory["wins"] == 12
    integrator = report["checkpoint_integrator_comparison"]["fullsig_none"]
    assert integrator["wins"] == 12
    assert report["representation_and_convergence"]["by_design"][
        "subjepa16_none"]["rank"]["all_cells_at_or_above_threshold"]
    contrasts = analysis.contrast_csv_rows(report)
    assert len(contrasts) == 6 + 3 + 3 + 8 + 12


def test_resume_ledger_roundtrip_keeps_one_current_cell() -> None:
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


def test_incomplete_analysis_fails_closed() -> None:
    report = analysis.analyze_rows(_synthetic_rows()[:-1], ["missing cell"])
    assert report["status"] == "INCOMPLETE_OR_INVALID"
    assert not report["artifact_integrity_passed"]
    assert report["completed_valid_cells"] == 143


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} Sub-JEPA-v16 pipeline tests passed.")


if __name__ == "__main__":
    main()
