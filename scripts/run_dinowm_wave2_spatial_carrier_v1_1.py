#!/usr/bin/env python3
"""Pre-outcome numerical amendment for the sealed DINO-WM Wave 2 audit.

The parent v1 attempt stopped before carrier outcomes because a sparse B=5
replay used an over-tight absolute tolerance.  A full-bank diagnostic then
showed bit-exact replay under the original B=12 execution layout and bounded
FP32 differences only when the same frozen computation used reusable-cache
batch shapes.  This wrapper changes no experimental field.  It validates the
two amended numerical gates and hard-links the hash-pinned v1 cache into an
independent v1.1 namespace.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_dinowm_wave2_spatial_carrier as base


DEFAULT_CONFIG = ROOT / "configs/dinowm_wave2_spatial_carrier_v1_1.yaml"
PARENT_CACHE = ROOT / "outputs/dinowm_wave2_spatial_carrier/cache"


def _amendment_evidence(cfg: Mapping[str, Any]) -> dict[str, Any]:
    scope = cfg["scope"]["amendment"]
    identities = {
        "parent_protocol": (
            ROOT / scope["parent_protocol"], scope["parent_protocol_sha256"]),
        "parent_lock": (
            ROOT / scope["parent_lock"], scope["parent_lock_sha256"]),
        "parent_cache_stop": (
            ROOT / scope["parent_cache_stop"],
            scope["parent_cache_stop_sha256"]),
        "numerical_diagnosis": (
            ROOT / scope["numerical_diagnosis"],
            scope["numerical_diagnosis_sha256"]),
    }
    checked = {}
    for name, (path, expected) in identities.items():
        base.require(path.is_file(), f"missing amendment evidence: {path}")
        observed = base.sha256_file(path)
        base.require(observed == expected,
                     f"amendment evidence changed: {path}")
        checked[name] = {"path": str(path.relative_to(ROOT)),
                         "size": path.stat().st_size, "sha256": observed}

    stop = json.loads(identities["parent_cache_stop"][0].read_text())
    diagnosis = json.loads(identities["numerical_diagnosis"][0].read_text())
    base.require(stop.get("status") == "failed_preserved"
                 and "1.8358230590820312e-05" in stop.get("reason", ""),
                 "parent fail-closed receipt differs")
    exact = diagnosis["direct_original_batch_vs_preserved_v2r2"]
    cached = diagnosis["reusable_cache_vs_direct_original_batch"]
    threshold = float(cfg["cache"]["reusable_cache_layout_threshold"])
    exact_threshold = float(cfg["cache"]["exact_teacher_replay_threshold"])
    base.require(float(exact["max_abs"]) <= exact_threshold
                 and float(exact["mean_abs"]) == 0.0
                 and float(exact["exact_fraction"]) == 1.0,
                 "original-layout teacher replay is not exact")
    base.require(float(cached["max_abs"]) <= threshold,
                 "reusable-cache numerical drift exceeds amended tolerance")
    base.require(
        float(diagnosis["cache_vs_direct_relative_l2"]["max"])
        <= 5e-6
        and float(diagnosis["cache_vs_direct_cosine"]["min"])
        >= 0.99999999999,
        "cache drift is not confined to the diagnosed FP32 envelope")
    base.require(all(
        int(record["prediction_flips"]) == 0
        for record in diagnosis["existing_teacher_readout_stability"].values()),
        "cache drift changed an existing frozen-teacher decision")
    base.require(diagnosis.get("carrier_outcomes_computed") is False,
                 "amendment evidence contains carrier outcomes")
    base.require(diagnosis.get("host_unchanged") is True,
                 "diagnostic host identity changed")
    return {
        "identities": checked,
        "parent_stop": stop,
        "exact_original_layout_gate": {
            "value": float(exact["max_abs"]),
            "threshold": exact_threshold,
            "direction": "<=",
            "pass": True,
            "episodes": int(diagnosis["episodes"]),
        },
        "reusable_cache_layout_gate": {
            "value": float(cached["max_abs"]),
            "threshold": threshold,
            "direction": "<=",
            "pass": True,
            "mean_abs": float(cached["mean_abs"]),
            "p99_abs": float(cached["p99_abs"]),
            "rmse": float(cached["rmse"]),
        },
        "carrier_outcomes_seen": False,
        "threshold_source": "preoutcome full-bank numerical audit",
    }


def run_smoke(config_path: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = _ORIGINAL_SMOKE(config_path, cfg)
    evidence = _amendment_evidence(cfg)
    value["amendment"] = evidence
    value["amendment_validated"] = True
    destination = (base.resolve(cfg["artifacts"]["root"])
                   / cfg["artifacts"]["smoke"] / "receipt.json")
    base.atomic_json(destination, value)
    return value


def prepare_cache(config_path: Path, cfg: Mapping[str, Any],
                  lock: Mapping[str, Any]) -> dict[str, Any]:
    root = base.resolve(cfg["artifacts"]["root"])
    cache_root = root / "cache"
    base.require(not cache_root.exists(),
                 "refusing to overwrite Wave 2 v1.1 cache namespace")
    cache_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        base.configure_cuda(cfg, seed=9070)
        pins = base.verify_pins(cfg)
        admissions = base.validate_prior_admissions(cfg)
        amendment = _amendment_evidence(cfg)
        _, selections_by_task = base.dataset_and_selections(cfg)

        destination_names = {
            "base_visual": "base_visual.npy",
            "cue__transient-visual-token-recall":
                "transient-visual-token-recall_cue_visual.npy",
            "cue__multi-item-visual-binding-recall":
                "multi-item-visual-binding-recall_cue_visual.npy",
            "metadata": "metadata.npz",
        }
        artifacts, links = {}, {}
        for name, destination_name in destination_names.items():
            expected = cfg["cache"]["parent_artifacts"][name]
            source = base.resolve(expected["path"])
            base.require(source.is_file()
                         and source.stat().st_size == int(expected["size"])
                         and base.sha256_file(source) == expected["sha256"],
                         f"hash-pinned parent cache changed: {source}")
            destination = cache_root / destination_name
            os.link(source, destination)
            base.require(os.path.samefile(source, destination),
                         f"cache reuse is not a hard link: {destination}")
            base.require(destination.stat().st_size == int(expected["size"])
                         and base.sha256_file(destination) == expected["sha256"],
                         f"hard-linked cache identity differs: {destination}")
            artifacts[name] = {
                "path": str(destination.relative_to(ROOT)),
                "size": destination.stat().st_size,
                "sha256": expected["sha256"],
                "parent_path": str(source.relative_to(ROOT)),
                "reuse": "hard_link_read_only_consumer",
            }
            links[name] = {
                "samefile": True,
                "inode": int(destination.stat().st_ino),
                "link_count": int(destination.stat().st_nlink),
            }

        base_visual = np.load(cache_root / "base_visual.npy", mmap_mode="r")
        transient = np.load(
            cache_root / "transient-visual-token-recall_cue_visual.npy",
            mmap_mode="r")
        binding = np.load(
            cache_root / "multi-item-visual-binding-recall_cue_visual.npy",
            mmap_mode="r")
        with np.load(cache_root / "metadata.npz") as metadata:
            base.require(
                base_visual.shape == (1680, 20, 196, 384)
                and transient.shape == binding.shape == (1680, 3, 196, 384)
                and metadata["actions"].shape == (1680, 19, 10)
                and metadata["proprio"].shape == (1680, 20, 4),
                "hard-linked cache shape differs")

        manifest = {
            "schema": "dinowm_wave2_full_patch_cache_v1_1_hardlink",
            "protocol_sha256": lock["protocol_sha256"],
            "shape": [1680, 20, 196, 384],
            "dtype": "float32",
            "selection_sha256": base._canonical_sha256({
                key: [asdict(item) for item in values]
                for key, values in selections_by_task.items()}),
            "amendment": amendment,
            "exact_original_layout_replay_max_abs":
                amendment["exact_original_layout_gate"]["value"],
            "exact_original_layout_replay_threshold":
                amendment["exact_original_layout_gate"]["threshold"],
            "reusable_cache_layout_max_abs":
                amendment["reusable_cache_layout_gate"]["value"],
            "reusable_cache_layout_threshold":
                amendment["reusable_cache_layout_gate"]["threshold"],
            "reuse_policy": cfg["cache"]["reuse_policy"],
            "hard_links": links,
            "admissions": admissions,
            "pins": pins,
            "artifacts": artifacts,
            "carrier_outcomes_computed_before_gate": False,
            "elapsed_seconds": time.time() - started,
        }
        base.atomic_json(cache_root / "manifest.json", manifest)
        return manifest
    except Exception as error:
        base.atomic_json(cache_root / "stop_receipt.json", {
            "schema": "dinowm_wave2_cache_stop_v1_1",
            "status": "failed_preserved", "reason": repr(error),
            "elapsed_seconds": time.time() - started,
            "carrier_outcomes_computed": False,
        })
        raise


_ORIGINAL_SMOKE = base.run_smoke
base.run_smoke = run_smoke
base.prepare_cache = prepare_cache
base.DEFAULT_CONFIG = DEFAULT_CONFIG


if __name__ == "__main__":
    base.main()
