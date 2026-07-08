#!/usr/bin/env python3
"""Independent fail-closed verification of DINO-WM PointMaze Wave 3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import parameter_report  # noqa: E402


CONFIG = ROOT / "configs/dinowm_pointmaze_wave3.yaml"
LOCK = CONFIG.with_suffix(".lock.json")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    lock = json.loads(LOCK.read_text())
    require(digest(CONFIG) == lock["protocol_sha256"],
            "protocol hash changed")
    for relative, expected in lock["source_sha256"].items():
        require(digest(resolve(relative)) == expected,
                f"locked source changed: {relative}")
    require(lock["parameter_matching"] == parameter_report(384, 10),
            "parameter ledger changed")
    require(lock["grid"] == {"tasks": 1, "arms": 5,
                             "seeds": 5, "cells": 25},
            "locked grid changed")
    root = resolve(cfg["artifacts"]["root"])
    cache = json.loads((root / "cache/manifest.json").read_text())
    formal = root / "formal"
    admission = json.loads((formal / "admission.json").read_text())
    controller = json.loads((formal / "controller_gate.json").read_text())
    summary = json.loads((formal / "summary.json").read_text())
    carrier = json.loads((formal / "carrier_summary.json").read_text())
    use = json.loads((formal / "external_use_summary.json").read_text())
    provenance = json.loads((formal / "provenance.json").read_text())
    progress = json.loads((formal / "progress.json").read_text())
    require(cache["precarrier_gates_passed"] is True
            and admission["admitted"] is True
            and controller["admitted"] is True,
            "pre-carrier gate failed")
    require(summary["status"] == carrier["status"] == use["status"]
            == provenance["status"] == "complete", "formal study incomplete")
    require(all(value["protocol_sha256"] == lock["protocol_sha256"]
                for value in (summary, carrier, use, provenance)),
            "formal protocol hash differs")
    require(provenance["physical_gpu"] == 2
            and provenance["cuda_visible_devices"] == "2",
            "formal run was not isolated to GPU 2")
    require(provenance["paper_modified_by_wave3"] is False,
            "Wave 3 reports a paper edit")
    require(provenance["runtime_host_digest"]
            == provenance["runtime_host_digest_after"],
            "frozen host digest changed")
    require(progress["count"] == progress["expected"] == 25,
            "formal grid incomplete")
    require(controller["current_mujoco_version"].split(".")[0] >= "3"
            and controller["deterministic_replay_fidelity"] == 1.0,
            "current-MuJoCo/replay gate differs")
    require(controller["oracle_executed_success"]
            >= cfg["external_use"]["oracle_success_minimum"],
            "controller oracle gate differs")
    require(use["scope"]["native_planner"] is False,
            "use result overclaims native planning")
    require(len(use["consumer_receipts"]) == 5
            and all(value["arm_blind"] is True
                    and value["arm_identifier_feature"] is False
                    for value in use["consumer_receipts"]),
            "arm-blind consumer contract differs")

    expected_parameters = {
        "none": 0, "gru": 298_368, "lstm": 299_632,
        "ssm": 299_520, "fixed_trust": 299_520,
    }
    truth = None
    cell_count = 0
    artifact_hashes: dict[str, str] = {}
    for arm in cfg["training"]["arms"]:
        for seed in cfg["training"]["seeds"]:
            directory = formal / "cells" / arm / f"s{seed}"
            manifest = json.loads((directory / "manifest.json").read_text())
            metrics = json.loads((directory / "metrics.json").read_text())
            require(manifest["protocol_sha256"] == lock["protocol_sha256"]
                    and metrics["protocol_sha256"] == lock["protocol_sha256"],
                    f"cell protocol differs: {directory}")
            require(metrics["physical_gpu"] == 2
                    and metrics["cuda_visible_devices"] == "2",
                    f"cell GPU differs: {directory}")
            require(metrics["carrier_parameters"] == expected_parameters[arm],
                    f"cell parameter count differs: {directory}")
            require(metrics["host_unchanged"] is True
                    and metrics["training_labels_used"] is False,
                    f"cell invariant differs: {directory}")
            for name, record in manifest["artifacts"].items():
                path = directory / name
                require(path.stat().st_size == record["size"]
                        and digest(path) == record["sha256"],
                        f"cell artifact changed: {path}")
                artifact_hashes[str(path.relative_to(formal))] = record["sha256"]
            with np.load(directory / "validation_predictions.npz") as values:
                current = values["truth"]
                require(current.shape == (480,), "cell truth shape changed")
                if truth is None:
                    truth = current.copy()
                else:
                    require(np.array_equal(truth, current),
                            "cell validation alignment differs")
                for age in cfg["sequence"]["evidence_ages"]:
                    for suffix in ("full_prediction", "reset_prediction",
                                   "prior_prediction", "full_mse", "reset_mse"):
                        require(values[f"age_{age}_{suffix}"].shape == (480,),
                                f"cell output shape differs: {directory}")
            with np.load(directory / "use_features.npz") as values:
                require(values["train_feature"].shape == (1200, 8064)
                        and values["validation_feature"].shape == (480, 8064),
                        f"use feature shape differs: {directory}")
            cell_count += 1
    require(cell_count == 25, "verified cell count differs")
    for age in cfg["sequence"]["evidence_ages"]:
        record = carrier["results"][str(age)]
        require(set(record["arms"]) == set(cfg["training"]["arms"]),
                f"carrier summary arm set differs at age {age}")
        for contrast in list(record["paired_vs_none"].values()) \
                + list(record["full_vs_context_reset"].values()):
            require(contrast["draws"] == 20_000
                    and contrast["native_episode_clusters"] == 120
                    and contrast["paired"] is True,
                    "carrier bootstrap contract differs")
    require(not (formal / "formal_stop_receipt.json").exists()
            and not (formal / "stop_receipt.json").exists(),
            "formal stop receipt exists")
    result: dict[str, Any] = {
        "schema": "dinowm_pointmaze_wave3_verification_v1",
        "verified": True, "protocol_sha256": lock["protocol_sha256"],
        "physical_gpu": 2, "cells": cell_count,
        "paired_bootstrap_draws": 20_000,
        "native_validation_episode_clusters": 120,
        "host_unchanged": True, "paper_modified_by_wave3": False,
        "current_mujoco_execution": True,
        "artifact_sha256": artifact_hashes,
        "summary_sha256": digest(formal / "summary.json"),
        "carrier_summary_sha256": digest(formal / "carrier_summary.json"),
        "external_use_summary_sha256": digest(
            formal / "external_use_summary.json"),
        "provenance_sha256": digest(formal / "provenance.json"),
    }
    destination = formal / "verification.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
