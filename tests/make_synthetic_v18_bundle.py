#!/usr/bin/env python3
"""Create a deterministic, scientifically fictitious complete V18 bundle for tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "scripts"))
import v18_release_common as common
import build_v18_code_supplement as supplement


def fake_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    common.atomic_write_json(path, value)


def contrast(
    name: str,
    effect: float,
    low: float,
    high: float,
    wins: int,
    tasks: int,
    *,
    metric: str = common.PRIMARY,
) -> dict:
    value = {
        "contrast": name,
        "metric": metric,
        "mean_paired_relative_reduction": effect,
        "paired_wins": wins,
        "pairs": 25,
        "task_mean_wins": tasks,
        "bootstrap": {
            "method": "synthetic_crossed_task_seed_bootstrap",
            "replicates": 10000,
            "ci95_low": low,
            "ci95_high": high,
        },
        "task_effects": {
            task: effect + (index - 2) * 0.005
            for index, task in enumerate(common.TASKS)
        },
    }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.output.resolve()
    if base.exists():
        shutil.rmtree(base)
    result = base / "result"
    logs = base / "logs"
    provenance = base / "provenance"
    private_repo = base / "private_repo"
    for directory in (
        result, logs, provenance, private_repo, base / "generated", base / "paper"
    ):
        directory.mkdir(parents=True, exist_ok=True)

    frozen_document = private_repo / "docs" / "V18_LEWM_V8_CONFIRMATION.md"
    common.atomic_write_text(
        frozen_document,
        "# Frozen synthetic V18 protocol\n\nTesting fixture only; no scientific result.\n",
    )
    for relative in sorted(supplement.EXPECTED_FROZEN_SOURCES - {
        "docs/V18_LEWM_V8_CONFIRMATION.md"
    }):
        path = private_repo / relative
        if relative == "scripts/run_autovisreg_v17.py":
            content = (
                '"""Synthetic frozen runner dependency."""\n'
                'import os\n'
                'WANDB_ENTITY = "synthetic-private-entity"\n'
                'WANDB_PROJECT = "synthetic-private-project"\n'
                f'PRIVATE_REPO = {str(private_repo)!r}\n'
            )
        elif relative in {
            "scripts/train_hacssm_v10.py",
            "scripts/train_hacssm_v11.py",
            "scripts/train_subjepa_v16.py",
        }:
            content = (
                f'"""Synthetic source fixture for {relative}."""\n'
                'DEFAULT_WANDB_PROJECT = "synthetic-private-project"\n'
            )
        else:
            content = f'"""Synthetic source fixture for {relative}."""\n'
        common.atomic_write_text(path, content)
    for relative in sorted(supplement.V18_TESTS):
        if relative == "scripts/test_run_lewm_v8_v18.py":
            content = (
                '"""Synthetic public-redaction contract test."""\n'
                'import importlib\n'
                'import scripts.run_autovisreg_v17 as runner\n\n'
                'def test_synthetic_contract_fixture(monkeypatch):\n'
                '    monkeypatch.setenv("V18_WANDB_ENTITY", "fixture-entity")\n'
                '    monkeypatch.setenv("V18_WANDB_PROJECT", "fixture-project")\n'
                '    importlib.reload(runner)\n'
                '    assert runner.WANDB_ENTITY == "fixture-entity"\n'
                '    assert runner.WANDB_PROJECT == "fixture-project"\n'
            )
        else:
            content = (
                f'"""Synthetic contract test {relative}."""\n\n'
                'def test_synthetic_contract_fixture():\n    assert True\n'
            )
        common.atomic_write_text(private_repo / relative, content)
    common.atomic_write_text(
        private_repo / "pyproject.toml",
        '[build-system]\nrequires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "anonymous-v18-fixture"\nversion = "0.0.0"\n',
    )
    common.atomic_write_text(
        private_repo / "configs" / "default.yaml",
        "model:\n  history_len: 3\ntraining:\n  epochs: 100\n",
    )
    forbidden_file = base / "private_identity_tokens.json"
    common.atomic_write_json(
        forbidden_file,
        {"synthetic_identity": ["synthetic-private"]},
    )
    forbidden_file.chmod(0o600)
    protocol_document = provenance / "V18_LEWM_V8_CONFIRMATION.md"
    shutil.copyfile(frozen_document, protocol_document)
    commands = [
        {
            "task": task,
            "seed": seed,
            "design": design,
            "argv": [
                "python", "train.py", "--task", task, "--seed", str(seed),
                "--design", design, "--history-len", "3", "--embed-dim", "128",
                "--img-size", "64", "--patch-size", "8", "--encoder-layers", "6",
                "--encoder-heads", "4", "--predictor-layers", "4",
                "--predictor-heads", "8", "--dropout", "0.1",
                "--sigreg-lambda", "0.0", "--epochs", "100",
                "--eval-target-key", "task_observation",
            ],
        }
        for task in common.TASKS
        for seed in common.SEEDS
        for design in common.DESIGNS
    ]
    protocol = {
        "schema_version": 1,
        "scope": "lewm_v8_v18_unopened_task_confirmation",
        "created_at": "2026-07-03T00:00:00+00:00",
        "tasks": list(common.TASKS),
        "designs": list(common.DESIGNS),
        "seeds": list(common.SEEDS),
        "epochs": 100,
        "runs": 200,
        "resume_supported": True,
        "resume_granularity": "complete_cell_only",
        "git_worktree_clean": True,
        "git_commit": fake_hash("synthetic git commit"),
        "git_upstream_commit": fake_hash("synthetic upstream commit"),
        "git_branch": "synthetic-private-branch",
        "wandb_entity": "synthetic-private-entity",
        "wandb_project": "synthetic-private-project",
        "output_root": str(result),
        "log_root": str(logs),
        "python": "/synthetic/private/python",
        "commands": commands,
        "commands_sha256": common.json_sha256(commands),
        "source_sha256": {
            relative: common.sha256(private_repo / relative)
            for relative in sorted(supplement.EXPECTED_FROZEN_SOURCES)
        },
        "data": {
            "__manifest__": {
                "path_sha256": fake_hash("cache paths"),
                "sidecar_sha256": fake_hash("cache sidecars"),
            }
        },
    }
    write_json(result / "confirmation_protocol.json", protocol)

    design_offsets = {
        "vicreg_none": 0.10,
        "vicreg_gru": 0.02,
        "vicreg_ssm": 0.00,
        "vicreg_hacssmv8": 0.04,
        "vicreg_hacssmv8_noaction": 0.09,
        "vicreg_hacssmv8_single": 0.05,
        "vicreg_hacssmv8_static": 0.045,
        "vicreg_hacssmv8_dynamic": 0.055,
    }
    cells = []
    runs = []
    for task_index, task in enumerate(common.TASKS):
        for seed_index, seed in enumerate(common.SEEDS):
            for design in common.DESIGNS:
                directory = result / (
                    "lewm-dmc_" + task.replace(".", "_") + f"-{design}-s{seed}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                primary = (
                    0.55 + 0.025 * task_index + 0.004 * seed_index
                    + design_offsets[design]
                )
                condition_multipliers = {
                    "freeze": 0.90,
                    "gaussian_noise": 1.10,
                    "checkerboard": 0.95,
                    "long_freeze": 1.05,
                }
                phase_multipliers = {
                    "gap": 1.15,
                    "deep": 1.18,
                    "first_post": 0.82,
                    "post": 0.90,
                }
                metrics = {
                    "schema_version": 1,
                    "env": "dmc:" + task,
                    "design": design,
                    "seed": seed,
                    common.PRIMARY: primary,
                }
                for condition, condition_multiplier in condition_multipliers.items():
                    condition_value = primary * condition_multiplier
                    metrics[f"{condition}_prior_state_nmse"] = condition_value
                    for phase, phase_multiplier in phase_multipliers.items():
                        metrics[f"{condition}_prior_state_nmse_{phase}"] = (
                            condition_value * phase_multiplier
                        )
                metrics["deep_prior_state_nmse"] = float(np.mean([
                    metrics[f"{condition}_prior_state_nmse_deep"]
                    for condition in condition_multipliers
                ]))
                metrics_path = directory / "metrics.json"
                write_json(metrics_path, metrics)
                rollout_hash = fake_hash(f"rollout:{task}:{seed}:{design}")
                receipt = {
                    "state": "finished",
                    "eval_rollout_sha256": rollout_hash,
                    "entity": "synthetic-private-entity",
                    "project": "synthetic-private-project",
                    "run_id": f"private-{task_index}-{seed_index}-{design}",
                    "url": "https://wandb.ai/synthetic-private-entity/private",
                }
                receipt_path = directory / "wandb_run.json"
                write_json(receipt_path, receipt)
                artifacts = {
                    "metrics.json": common.sha256(metrics_path),
                    "model.pt": fake_hash(f"model:{task}:{seed}:{design}"),
                    "eval_rollout.npz": rollout_hash,
                    "wandb_run.json": common.sha256(receipt_path),
                }
                cells.append({
                    "task": task,
                    "seed": seed,
                    "design": design,
                    common.PRIMARY: primary,
                    "clean_prior_state_nmse": primary * 0.75,
                    "val_predictive_loss": 0.12 + primary * 0.03,
                    "deep_prior_state_nmse": metrics["deep_prior_state_nmse"],
                    "encoder_mean_channel_variance": 0.02 + task_index * 0.001,
                    "encoder_covariance_effective_rank": 25.0 + seed_index,
                    "predictive_loss_convergence_relative_change": 0.01,
                    "initial_encoder_integrator_probe_nmse": 0.63 + 0.02 * task_index,
                })
                row = {
                    "task": task,
                    "seed": seed,
                    "design": design,
                    "gpu": str(task_index % 4),
                    "status": "complete",
                    "resumed_existing": False,
                    "seconds": 1.0,
                    "completed_at": "2026-07-03T01:00:00+00:00",
                    "command_sha256": fake_hash(f"command:{task}:{seed}:{design}"),
                    "log": str(logs / f"terminal-{task_index}-{seed}-{design}.log"),
                    "directory": str(directory),
                    "headline_metric": primary,
                    "wandb_state": "finished",
                    "artifact_sha256": artifacts,
                }
                runs.append(row)
    write_json(result / "confirmation_runs.json", runs)
    write_json(result / "confirmation_attempts.json", runs)
    summary = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "status": "COMPLETE",
        "expected_cells": 200,
        "completed_cells": 200,
        "failed_or_invalid_cells": 0,
        "finished_at": "2026-07-03T02:00:00+00:00",
        "resume": True,
        "failures": [],
    }
    write_json(result / "confirmation_summary.json", summary)

    cells_path = result / "confirmation_cells.csv"
    with cells_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=common.CELL_FIELDS)
        writer.writeheader()
        writer.writerows(cells)

    specs = {
        "R": (-0.020, -0.040, 0.010, 10, 2, common.PRIMARY),
        "N": (0.100, 0.070, 0.130, 23, 5, common.PRIMARY),
        "I": (-0.030, -0.060, -0.005, 8, 1, common.PRIMARY),
        "A": (0.060, 0.020, 0.095, 20, 4, common.PRIMARY),
        "J": (0.010, -0.015, 0.035, 14, 3, common.PRIMARY),
        "E": (0.002, -0.005, 0.010, 13, 3, common.PRIMARY),
        "D": (-0.025, -0.050, 0.005, 9, 2, "deep_prior_state_nmse"),
        "C": (-0.010, -0.025, 0.005, 11, 2, "clean_prior_state_nmse"),
    }
    contrasts = {
        common.CONTRAST_KEYS[name]: contrast(
            common.CONTRAST_KEYS[name], *spec[:5], metric=spec[5]
        )
        for name, spec in specs.items()
    }
    cell_index = {
        (row["task"], int(row["seed"]), row["design"]):
        float(row[common.PRIMARY])
        for row in cells
    }
    deep_cell_index = {
        (row["task"], int(row["seed"]), row["design"]):
        float(row["deep_prior_state_nmse"])
        for row in cells
    }

    def recurrent_payload(metric_index):
        matrix = []
        task_effects = {}
        for task in common.TASKS:
            task_candidate, task_reference, task_cells = [], [], []
            for seed in common.SEEDS:
                candidate = metric_index[(task, seed, "vicreg_hacssmv8")]
                selected = min(
                    ("vicreg_gru", "vicreg_ssm"),
                    key=lambda design: (cell_index[(task, seed, design)], design),
                )
                reference = metric_index[(task, seed, selected)]
                task_candidate.append(candidate)
                task_reference.append(reference)
                task_cells.append((reference - candidate) / reference)
            matrix.append(task_cells)
            task_effects[task] = (
                (sum(task_reference) / 5 - sum(task_candidate) / 5)
                / (sum(task_reference) / 5)
            )
        flat = [effect for task in matrix for effect in task]
        return matrix, task_effects, flat

    for key, metric_index in (("R", cell_index), ("D", deep_cell_index)):
        recurrent_cells, recurrent_task_effects, recurrent_flat = recurrent_payload(
            metric_index
        )
        value = contrasts[common.CONTRAST_KEYS[key]]
        value.update({
            "mean_paired_relative_reduction": sum(recurrent_flat) / 25,
            "paired_wins": sum(effect > 0 for effect in recurrent_flat),
            "task_mean_wins": sum(effect > 0 for effect in recurrent_task_effects.values()),
            "task_effects": recurrent_task_effects,
            "cell_effects": recurrent_cells,
            "selected_reference_counts": {"vicreg_gru": 0, "vicreg_ssm": 25},
        })
    for index in range(25):
        name = f"synthetic_registered_contrast_{index + 1:02d}:heldout_prior_state_nmse"
        contrasts[name] = contrast(name, 0.001 * (index - 12), -0.03, 0.03, 12, 2)

    contrast_path = result / "confirmation_contrasts.csv"
    fields = [
        "contrast", "metric", "mean_paired_relative_reduction", "paired_wins",
        "pairs", "task_mean_wins", "ci95_low", "ci95_high",
    ]
    with contrast_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, value in contrasts.items():
            writer.writerow({
                "contrast": name,
                "metric": value["metric"],
                "mean_paired_relative_reduction": value["mean_paired_relative_reduction"],
                "paired_wins": value["paired_wins"],
                "pairs": value["pairs"],
                "task_mean_wins": value["task_mean_wins"],
                "ci95_low": value["bootstrap"]["ci95_low"],
                "ci95_high": value["bootstrap"]["ci95_high"],
            })

    gate_values = {
        common.GATE_KEYS["R"]: False,
        common.GATE_KEYS["N"]: True,
        common.GATE_KEYS["I"]: False,
        common.GATE_KEYS["A"]: True,
        common.GATE_KEYS["J"]: False,
        common.GATE_KEYS["E"]: True,
        common.GATE_KEYS["D"]: False,
        common.GATE_KEYS["C"]: True,
        "healthy_representation": True,
        "convergence": True,
        "integrity": True,
    }
    representation = {
        "passed": True,
        "thresholds": {},
        "observed": {
            "cells": 200,
            "minimum_channel_variance": 0.02,
            "minimum_effective_rank": 25.0,
            "variance_passing_cells": 200,
            "rank_passing_cells": 200,
        },
    }
    convergence = {
        "passed": True,
        "thresholds": {},
        "observed": {
            "cells": 200,
            "maximum_absolute_relative_change": 0.01,
            "passing_cells": 200,
        },
    }
    receipts = {
        key: {"passed": value, "thresholds": {}, "observed": {}}
        for key, value in gate_values.items()
        if key != "integrity"
    }
    receipts["healthy_representation"] = representation
    receipts["convergence"] = convergence
    report = {
        "schema_version": 2,
        "scope": protocol["scope"],
        "frozen_grid": {
            "tasks": 5,
            "designs": 8,
            "seeds": 5,
            "epochs": 100,
            "cells": 200,
            "task_ids": list(common.TASKS),
            "seed_ids": list(common.SEEDS),
            "design_ids": list(common.DESIGNS),
        },
        "expected_cells": 200,
        "completed_valid_cells": 200,
        "artifact_integrity_passed": True,
        "artifact_integrity_errors": [],
        "protocol_contract_errors": [],
        "status": "COMPLETE",
        "scientific_label": "CONFIRMATION_FAILED",
        "official_confirmation_result": False,
        "primary_metric": common.PRIMARY,
        "gates": gate_values,
        "gate_receipts": receipts,
        "representation": representation,
        "convergence": convergence,
        "contrasts": contrasts,
        "input_protocol_sha256": common.sha256(result / "confirmation_protocol.json"),
        "input_artifact_manifest_sha256": common.artifact_manifest_sha256(runs),
        "cells_csv_sha256": common.sha256(cells_path),
        "contrasts_csv_sha256": common.sha256(contrast_path),
    }
    write_json(result / "confirmation_analysis.json", report)

    # Synthetic log pairs reproduce the exact two-interruption topology, not real bytes.
    interruption_specs = (
        (1, 136, 64, (
            ("acrobot.swingup", "vicreg_ssm", 18005, 62),
            ("manipulator.bring_ball", "vicreg_ssm", 18005, 31),
            ("quadruped.run", "vicreg_ssm", 18005, 9),
            ("swimmer.swimmer15", "vicreg_ssm", 18005, 62),
        )),
        (2, 180, 20, (
            ("stacker.stack_4", "vicreg_hacssmv8_static", 18003, 37),
        )),
    )
    interruptions = []
    for sequence, complete, absent, attempts in interruption_specs:
        rows = []
        for task, design, seed, epoch in attempts:
            slug = task.replace(".", "_")
            interrupted_name = f"{slug}-{design}-s{seed}.log"
            restart_name = f"{slug}-{design}-s{seed}.attempt2.log"
            common.atomic_write_text(logs / interrupted_name, f"synthetic interrupted epoch {epoch}\n")
            common.atomic_write_text(logs / restart_name, "synthetic terminal epoch 100\n")
            rows.append({
                "task": task,
                "design": design,
                "seed": seed,
                "last_logged_epoch": epoch,
                "interrupted_log": interrupted_name,
                "interrupted_log_sha256": common.sha256(logs / interrupted_name),
                "restart_log": restart_name,
                "restart_log_sha256": common.sha256(logs / restart_name),
                "restart_terminal_epoch": 100,
            })
        interruptions.append({
            "sequence": sequence,
            "observation": {
                "complete_valid_cells": complete,
                "absent_cells": absent,
                "partial_or_invalid_core_cells": 0,
                "interrupted_trainers": len(rows),
            },
            "resume": {
                "validated_existing_cells": complete,
                "policy": "complete_cell_only",
            },
            "interrupted_attempts": rows,
        })
    manual = provenance / "restart_interruptions.record.json"
    write_json(manual, {"interruptions": interruptions})
    print(json.dumps({
        "base": str(base),
        "result": str(result),
        "logs": str(logs),
        "protocol_document": str(protocol_document),
        "private_repo": str(private_repo),
        "forbidden_file": str(forbidden_file),
        "manual_restart_record": str(manual),
    }, indent=2))


if __name__ == "__main__":
    main()
