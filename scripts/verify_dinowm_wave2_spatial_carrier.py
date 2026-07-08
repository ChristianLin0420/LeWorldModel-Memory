#!/usr/bin/env python3
"""Independent fail-closed verification for DINO-WM Wave 2 artifacts."""

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


CONFIG = ROOT / "configs/dinowm_wave2_spatial_carrier.yaml"
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


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    lock = json.loads(LOCK.read_text())
    require(digest(CONFIG) == lock["protocol_sha256"],
            "protocol hash differs")
    for relative, expected in lock["source_sha256"].items():
        require(digest(ROOT / relative) == expected,
                f"locked source differs: {relative}")
    require(lock["parameter_matching"] == parameter_report(384, 10),
            "parameter ledger differs")
    formal = ROOT / cfg["artifacts"]["root"] / cfg["artifacts"]["formal"]
    summary = json.loads((formal / "summary.json").read_text())
    provenance = json.loads((formal / "provenance.json").read_text())
    admissions = json.loads((formal / "admissions.json").read_text())
    progress = json.loads((formal / "progress.json").read_text())
    require(summary["status"] == provenance["status"] == "complete",
            "formal study is not complete")
    require(summary["protocol_sha256"] == provenance["protocol_sha256"]
            == lock["protocol_sha256"], "formal protocol differs")
    require(provenance["physical_gpu"] == 1
            and provenance["cuda_visible_devices"] == "1",
            "formal run was not isolated to physical GPU 1")
    require(provenance["paper_modified_by_wave2"] is False,
            "formal provenance reports paper modification")
    require(provenance["runtime_host_digest"]
            == provenance["runtime_host_digest_after"],
            "runtime host digest changed")
    require(progress["count"] == progress["expected"] == 50,
            "formal grid is incomplete")
    require(all(record["admitted"] for record in admissions["tasks"].values())
            and admissions["rollout_health"]["admitted"],
            "reused prerequisite admission failed")
    require(summary["grid"] == {"tasks": 2, "arms": 5,
                                "seeds": 5, "cells": 50},
            "summary grid differs")
    require(summary["inference"]["draws"] == 20_000
            and summary["inference"]["paired"] is True,
            "paired inference contract differs")

    artifact_hashes: dict[str, str] = {}
    cell_count = 0
    for task in cfg["tasks"]:
        key, classes = task["key"], int(task["classes"])
        task_summary = summary["results"][key]
        require(task_summary["classes"] == classes,
                f"class count differs for {key}")
        reference_truth = None
        for arm in cfg["training"]["arms"]:
            expected_parameters = {
                "none": 0, "gru": 298_368, "lstm": 299_632,
                "ssm": 299_520, "fixed_trust": 299_520,
            }[arm]
            for seed in cfg["training"]["seeds"]:
                directory = formal / "cells" / key / arm / f"s{seed}"
                manifest = json.loads((directory / "manifest.json").read_text())
                metrics = json.loads((directory / "metrics.json").read_text())
                require(manifest["protocol_sha256"] == lock["protocol_sha256"],
                        f"cell lock differs: {directory}")
                require(metrics["carrier_parameters"] == expected_parameters,
                        f"parameter count differs: {directory}")
                require(metrics["physical_gpu"] == 1
                        and metrics["cuda_visible_devices"] == "1",
                        f"GPU provenance differs: {directory}")
                require(metrics["host_unchanged"] is True
                        and metrics["host_digest_before"]
                        == metrics["host_digest_after"],
                        f"host changed: {directory}")
                require(metrics["training_labels_used"] is False,
                        f"label use differs: {directory}")
                for name, record in manifest["artifacts"].items():
                    path = directory / name
                    require(path.stat().st_size == record["size"]
                            and digest(path) == record["sha256"],
                            f"artifact differs: {path}")
                    artifact_hashes[str(path.relative_to(formal))] = record["sha256"]
                with np.load(directory / "validation_predictions.npz") as values:
                    truth = values["truth"]
                    require(truth.shape == (480,),
                            f"truth shape differs: {directory}")
                    if reference_truth is None:
                        reference_truth = truth.copy()
                    else:
                        require(np.array_equal(reference_truth, truth),
                                f"validation alignment differs: {directory}")
                    for age in cfg["sequence"]["evidence_ages"]:
                        for suffix in ("full_prediction", "reset_prediction",
                                       "prior_prediction"):
                            require(values[f"age_{age}_{suffix}"].shape == (480,),
                                    f"prediction shape differs: {directory}")
                        require(np.isfinite(values[f"age_{age}_full_mse"]).all()
                                and np.isfinite(values[f"age_{age}_reset_mse"]).all(),
                                f"MSE is non-finite: {directory}")
                cell_count += 1
        for age in cfg["sequence"]["evidence_ages"]:
            record = task_summary["ages"][str(age)]
            require(set(record["arms"]) == set(cfg["training"]["arms"]),
                    f"missing arms: {key}/age{age}")
            for arm, contrast in record["paired_vs_none"].items():
                require(arm != "none" and contrast["draws"] == 20_000
                        and contrast["paired"] is True,
                        f"paired contrast differs: {key}/{age}/{arm}")
            for contrast in record["full_vs_context_reset"].values():
                require(contrast["draws"] == 20_000
                        and contrast["paired"] is True,
                        f"reset contrast differs: {key}/{age}")
    require(cell_count == 50, "verified cell count differs")
    require(not (formal / "stop_receipt.json").exists(),
            "formal stop receipt exists")

    result: dict[str, Any] = {
        "schema": "dinowm_wave2_spatial_verification_v1",
        "verified": True,
        "protocol_sha256": lock["protocol_sha256"],
        "physical_gpu": 1,
        "cells": cell_count,
        "paired_bootstrap_draws": 20_000,
        "host_unchanged": True,
        "paper_modified_by_wave2": False,
        "artifact_sha256": artifact_hashes,
        "summary_sha256": digest(formal / "summary.json"),
        "provenance_sha256": digest(formal / "provenance.json"),
    }
    destination = formal / "verification.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
