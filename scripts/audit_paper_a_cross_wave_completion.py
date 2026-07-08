#!/usr/bin/env python3
"""Read-only completion audit across Paper-A Waves 1.1, 2 v1.1, and 3.

This is deliberately a post-lock consumer.  It is not imported by, and must
not be added to, any sealed experiment lock.  All experiment/config/artifact
inputs are read-only.  Without ``--execute`` the validated receipt is printed
to stdout only.  With ``--execute`` exactly one new cross-wave receipt is
created outside the three experiment roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import parameter_report  # noqa: E402


WAVE1_ROOT = Path("outputs/paper_a_matched_color_v1_1")
WAVE2_CONFIG = Path("configs/dinowm_wave2_spatial_carrier_v1_1.yaml")
WAVE3_CONFIG = Path("configs/dinowm_pointmaze_wave3.yaml")
DEFAULT_RECEIPT = Path("outputs/paper_a_cross_wave_completion/receipt.json")
PARAMETERS = {
    "none": 0,
    "gru": 298_368,
    "lstm": 299_632,
    "ssm": 299_520,
    "fixed_trust": 299_520,
}


class AuditFailure(RuntimeError):
    """A required completion or integrity condition did not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


class HashVerifier:
    """SHA-256 verifier with inode de-duplication and mutation detection."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int, int], str] = {}

    def digest(self, path: Path) -> str:
        require(path.is_file(), f"missing file: {path}")
        before = path.stat()
        key = (before.st_dev, before.st_ino, before.st_size,
               before.st_mtime_ns)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(chunk)
        after = path.stat()
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"file changed while hashing: {path}")
        result = value.hexdigest()
        self._cache[key] = result
        return result


def repository_path(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    candidate = Path(value)
    result = candidate.resolve() if candidate.is_absolute() \
        else (root / candidate).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise AuditFailure(f"path leaves repository: {value}") from error
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AuditFailure(f"invalid {label}: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} is not a mapping: {path}")
    return value


def read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise AuditFailure(f"invalid {label}: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} is not a mapping: {path}")
    return value


def verify_hash_record(root: Path, record: Mapping[str, Any],
                       hasher: HashVerifier, label: str) -> Path:
    require(isinstance(record, Mapping), f"{label} record is not a mapping")
    path = repository_path(root, record.get("path", ""))
    require(hasher.digest(path) == record.get("sha256"),
            f"{label} SHA-256 differs: {path}")
    if "size" in record:
        require(path.stat().st_size == record["size"],
                f"{label} size differs: {path}")
    return path


def verify_sha256_sidecar(target: Path, sidecar: Path,
                          hasher: HashVerifier) -> str:
    require(sidecar.is_file(), f"missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text().strip().split()
    digest = hasher.digest(target)
    require(fields == [digest, target.name],
            f"SHA-256 sidecar differs: {sidecar}")
    return digest


def verify_artifact(path: Path, record: Mapping[str, Any],
                    hasher: HashVerifier, label: str) -> str:
    require(path.is_file(), f"missing {label}: {path}")
    if "size" in record:
        require(path.stat().st_size == record["size"],
                f"{label} size differs: {path}")
    digest = hasher.digest(path)
    require(digest == record.get("sha256"),
            f"{label} SHA-256 differs: {path}")
    return digest


def verify_bootstrap_record(record: Mapping[str, Any], *,
                            native_clusters: int | None = None) -> None:
    require(record.get("draws") == 20_000,
            "bootstrap draw count differs")
    require(record.get("paired") is True, "bootstrap is not paired")
    if native_clusters is not None:
        require(record.get("native_episode_clusters") == native_clusters,
                "bootstrap native-episode unit differs")


def preflight_completion(root: Path) -> None:
    """Reject incomplete waves before expensive artifact hashing."""
    wave1 = root / WAVE1_ROOT
    receipt = read_json(
        wave1 / "independent_verifier_receipt.json", "Wave 1 receipt")
    final_audit = read_json(wave1 / "final_audit.json", "Wave 1 final audit")
    require(receipt.get("status") == "verified"
            and final_audit.get("status") == "complete"
            and final_audit.get("complete_cells")
            == final_audit.get("expected_cells") == 50,
            "Wave 1.1 is incomplete")

    for label, relative, expected in (
            ("Wave 2 v1.1", WAVE2_CONFIG, 50),
            ("Wave 3", WAVE3_CONFIG, 25)):
        cfg = read_yaml(root / relative, f"{label} config")
        output = repository_path(root, cfg["artifacts"]["root"])
        formal = output / cfg["artifacts"]["formal"]
        progress = read_json(formal / "progress.json", f"{label} progress")
        require(progress.get("count") == progress.get("expected") == expected,
                f"{label} is incomplete: {progress.get('count')}/{expected} cells")
        summary = read_json(formal / "summary.json", f"{label} summary")
        provenance = read_json(
            formal / "provenance.json", f"{label} provenance")
        require(summary.get("status") == provenance.get("status") == "complete",
                f"{label} is not formally complete")
        require((formal / "verification.json").is_file(),
                f"missing {label} official verification: "
                f"{formal / 'verification.json'}")


def verify_wave1(root: Path, hasher: HashVerifier) -> dict[str, Any]:
    wave = root / WAVE1_ROOT
    receipt_path = wave / "independent_verifier_receipt.json"
    sidecar_path = wave / "independent_verifier_receipt.sha256"
    receipt_digest = verify_sha256_sidecar(
        receipt_path, sidecar_path, hasher)
    receipt = read_json(receipt_path, "Wave 1 receipt")
    require(receipt.get("schema_version") == 1
            and receipt.get("status") == "verified"
            and receipt.get("locked_artifacts_mutated") is False,
            "Wave 1 receipt status differs")

    identities = receipt.get("identities", {})
    for name in ("spec", "spec_sidecar", "implementation_lock"):
        verify_hash_record(root, identities.get(name, {}), hasher,
                           f"Wave 1 {name}")
    lock_path = repository_path(
        root, identities["implementation_lock"]["path"])
    lock = read_json(lock_path, "Wave 1 implementation lock")
    producers = lock.get("producers", {})
    require(len(producers) == identities["implementation_lock"].get(
        "verified_producers") == 21, "Wave 1 producer count differs")
    for relative, expected in producers.items():
        path = repository_path(root, relative)
        require(hasher.digest(path) == expected,
                f"Wave 1 locked producer differs: {relative}")

    aggregate = receipt.get("aggregate_artifacts", {})
    summary_path = verify_hash_record(
        root, aggregate.get("summary", {}), hasher, "Wave 1 summary")
    audit_path = verify_hash_record(
        root, aggregate.get("final_audit", {}), hasher,
        "Wave 1 final audit")
    summary = read_json(summary_path, "Wave 1 summary")
    final_audit = read_json(audit_path, "Wave 1 final audit")
    require(summary.get("status") == final_audit.get("status") == "complete",
            "Wave 1 aggregate is incomplete")
    require(final_audit.get("summary", {}).get("sha256")
            == hasher.digest(summary_path), "Wave 1 summary receipt differs")
    require(final_audit.get("complete_cells")
            == final_audit.get("expected_cells") == 50,
            "Wave 1 cell count differs")
    require(final_audit.get("hashed_cell_artifacts") == 200
            and final_audit.get("unexpected_carrier_directories") == 0,
            "Wave 1 artifact ledger differs")
    require(final_audit.get("physical_gpu_counts")
            == {"0": 50, "1": 0, "2": 0, "3": 0}
            and final_audit.get("cuda3_used") is False,
            "Wave 1 GPU provenance differs")
    bootstrap = summary.get("bootstrap", {})
    require(bootstrap == {
        "draws": 20_000,
        "episode_stratification": "16-way color-location joint label",
        "independent_host_episode_resampling": True,
        "joint_seed_resampling_across_hosts": True,
        "paired_within_host": ["age", "arm"],
        "seed": 20261021,
    }, "Wave 1 bootstrap units differ")
    receipt_bootstrap = receipt.get("bootstrap_verification", {})
    require(receipt_bootstrap.get("draws") == 20_000
            and receipt_bootstrap.get(
                "independent_reimplementation_matches_formal_summary") is True
            and receipt_bootstrap.get("host_episode_weight_streams_distinct")
            is True, "Wave 1 independent bootstrap verification differs")

    expected_keys = {
        f"{host}/{arm}/seed-{seed}"
        for host in ("reacher", "pusht")
        for arm in ("none", "gru", "lstm", "ssm", "fixed_trust")
        for seed in range(5)
    }
    cells = final_audit.get("cell_artifacts", {})
    require(set(cells) == expected_keys, "Wave 1 cell directory set differs")
    artifact_count = 0
    seen: set[Path] = set()
    for key, records in cells.items():
        require(set(records) == {"checkpoint", "history", "manifest", "metrics"},
                f"Wave 1 artifact set differs: {key}")
        for name, record in records.items():
            path = verify_hash_record(
                root, record, hasher, f"Wave 1 {key}/{name}")
            require(path not in seen, f"Wave 1 artifact repeated: {path}")
            seen.add(path)
            artifact_count += 1
        metrics = read_json(
            repository_path(root, records["metrics"]["path"]),
            f"Wave 1 {key} metrics")
        require(metrics.get("physical_gpu") == 0
                and metrics.get("device") == "cuda:0"
                and metrics.get("frozen_host_unchanged") is True,
                f"Wave 1 cell invariant differs: {key}")
    require(artifact_count == 200, "Wave 1 verified artifact count differs")
    for name, record in final_audit.get("admissions", {}).items():
        verify_hash_record(root, record, hasher, f"Wave 1 admission {name}")

    return {
        "study": "paper-a-matched-color-v1-1",
        "status": "verified",
        "cells": 50,
        "cell_artifacts": artifact_count,
        "physical_gpu": 0,
        "bootstrap": {
            "draws": 20_000,
            "seed_unit": "joint carrier-seed indices across hosts",
            "episode_unit": "independent host, 16-way-stratified episode",
            "paired": ["arm", "age"],
        },
        "summary_sha256": hasher.digest(summary_path),
        "final_audit_sha256": hasher.digest(audit_path),
        "independent_receipt_sha256": receipt_digest,
        "independent_receipt_sidecar_sha256": hasher.digest(sidecar_path),
    }


def _verify_lock(root: Path, config_relative: Path,
                 hasher: HashVerifier, label: str) \
        -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    config_path = root / config_relative
    lock_path = config_path.with_suffix(".lock.json")
    cfg = read_yaml(config_path, f"{label} config")
    lock = read_json(lock_path, f"{label} lock")
    require(hasher.digest(config_path) == lock.get("protocol_sha256"),
            f"{label} protocol hash differs")
    for relative, expected in lock.get("source_sha256", {}).items():
        path = repository_path(root, relative)
        require(hasher.digest(path) == expected,
                f"{label} locked source differs: {relative}")
    require(lock.get("parameter_matching") == parameter_report(384, 10),
            f"{label} parameter ledger differs")
    output = repository_path(root, cfg["artifacts"]["root"])
    formal = output / cfg["artifacts"]["formal"]
    return cfg, lock, output, formal


def _require_official_verification(
        formal: Path, hasher: HashVerifier, expected_schema: str,
        protocol: str, artifacts: Mapping[str, str],
        summary_hashes: Mapping[str, str]) -> str:
    path = formal / "verification.json"
    value = read_json(path, "official verification")
    require(value.get("schema") == expected_schema
            and value.get("verified") is True
            and value.get("protocol_sha256") == protocol,
            f"official verification differs: {path}")
    require(value.get("artifact_sha256") == dict(artifacts),
            f"official artifact ledger differs: {path}")
    for key, expected in summary_hashes.items():
        require(value.get(key) == expected,
                f"official {key} differs: {path}")
    return hasher.digest(path)


def verify_wave2(root: Path, hasher: HashVerifier) -> dict[str, Any]:
    label = "Wave 2 v1.1"
    cfg, lock, output, formal = _verify_lock(
        root, WAVE2_CONFIG, hasher, label)
    require(lock.get("grid") == {
        "tasks": 2, "arms": 5, "seeds": 5, "cells": 50,
    }, "Wave 2 locked grid differs")
    summary = read_json(formal / "summary.json", "Wave 2 summary")
    provenance = read_json(formal / "provenance.json", "Wave 2 provenance")
    admissions = read_json(formal / "admissions.json", "Wave 2 admissions")
    progress = read_json(formal / "progress.json", "Wave 2 progress")
    protocol = lock["protocol_sha256"]
    require(summary.get("status") == provenance.get("status") == "complete",
            "Wave 2 formal study is incomplete")
    require(summary.get("protocol_sha256")
            == provenance.get("protocol_sha256") == protocol,
            "Wave 2 formal protocol differs")
    require(provenance.get("physical_gpu") == 1
            and provenance.get("cuda_visible_devices") == "1"
            and provenance.get("paper_modified_by_wave2") is False,
            "Wave 2 GPU/paper provenance differs")
    require(provenance.get("runtime_host_digest")
            == provenance.get("runtime_host_digest_after"),
            "Wave 2 frozen host changed")
    require(progress.get("count") == progress.get("expected") == 50,
            "Wave 2 grid is incomplete")
    require(summary.get("grid") == lock["grid"],
            "Wave 2 summary grid differs")
    require(all(record.get("admitted") is True
                for record in admissions.get("tasks", {}).values())
            and admissions.get("rollout_health", {}).get("admitted") is True,
            "Wave 2 reused admission failed")
    inference = summary.get("inference", {})
    require(inference.get("draws") == 20_000
            and inference.get("paired") is True
            and inference.get("method")
            == "matched-seed x class-stratified held-out-episode bootstrap",
            "Wave 2 bootstrap units differ")

    cache = output / "cache"
    cache_manifest_path = cache / "manifest.json"
    cache_manifest = read_json(cache_manifest_path, "Wave 2 cache manifest")
    amendment = cache_manifest.get("amendment", {})
    exact = amendment.get("exact_original_layout_gate", {})
    reusable = amendment.get("reusable_cache_layout_gate", {})
    require(exact.get("pass") is True and exact.get("value") == 0.0,
            "Wave 2 exact replay gate differs")
    require(reusable.get("pass") is True
            and reusable.get("value") <= reusable.get("threshold") == 0.0001,
            "Wave 2 cache-layout gate differs")
    require(cache_manifest.get("carrier_outcomes_computed_before_gate") is False
            and amendment.get("carrier_outcomes_seen") is False,
            "Wave 2 numerical amendment was not pre-outcome")
    for name, record in cache_manifest.get("artifacts", {}).items():
        parent = repository_path(root, record["parent_path"])
        child = repository_path(root, record["path"])
        require(parent.is_file() and child.is_file()
                and os.path.samefile(parent, child),
                f"Wave 2 cache is not a hard link: {name}")
        require(parent.stat().st_size == child.stat().st_size == record["size"],
                f"Wave 2 cache size differs: {name}")
        require(hasher.digest(parent) == hasher.digest(child)
                == record["sha256"], f"Wave 2 cache hash differs: {name}")

    tasks = [record["key"] for record in cfg["tasks"]]
    arms = list(cfg["training"]["arms"])
    seeds = list(map(int, cfg["training"]["seeds"]))
    expected_dirs = {
        (formal / "cells" / task / arm / f"s{seed}").resolve()
        for task in tasks for arm in arms for seed in seeds
    }
    actual_dirs = {
        path.resolve() for path in (formal / "cells").glob("*/*/s*")
        if path.is_dir()
    }
    require(actual_dirs == expected_dirs, "Wave 2 cell directory set differs")
    artifacts: dict[str, str] = {}
    truth_by_task: dict[str, np.ndarray] = {}
    task_classes = {record["key"]: int(record["classes"])
                    for record in cfg["tasks"]}
    for task in tasks:
        task_summary = summary.get("results", {}).get(task, {})
        require(task_summary.get("classes") == task_classes[task],
                f"Wave 2 class count differs: {task}")
        for arm in arms:
            for seed in seeds:
                directory = formal / "cells" / task / arm / f"s{seed}"
                manifest = read_json(directory / "manifest.json",
                                     f"Wave 2 cell manifest {task}/{arm}/s{seed}")
                metrics = read_json(directory / "metrics.json",
                                    f"Wave 2 metrics {task}/{arm}/s{seed}")
                require(manifest.get("protocol_sha256") == protocol,
                        f"Wave 2 cell lock differs: {directory}")
                require(metrics.get("carrier_parameters") == PARAMETERS[arm]
                        and metrics.get("physical_gpu") == 1
                        and metrics.get("cuda_visible_devices") == "1",
                        f"Wave 2 cell parameters/GPU differ: {directory}")
                require(metrics.get("host_unchanged") is True
                        and metrics.get("host_digest_before")
                        == metrics.get("host_digest_after")
                        and metrics.get("training_labels_used") is False,
                        f"Wave 2 cell invariant differs: {directory}")
                records = manifest.get("artifacts", {})
                require(set(records) == {
                    "carrier.pt", "history.csv", "metrics.json",
                    "validation_predictions.npz",
                }, f"Wave 2 artifact set differs: {directory}")
                for name, record in records.items():
                    path = directory / name
                    digest = verify_artifact(
                        path, record, hasher, f"Wave 2 {task}/{arm}/s{seed}/{name}")
                    artifacts[str(path.relative_to(formal))] = digest
                with np.load(directory / "validation_predictions.npz",
                             allow_pickle=False) as values:
                    truth = values["truth"]
                    require(truth.shape == (480,),
                            f"Wave 2 truth shape differs: {directory}")
                    if task not in truth_by_task:
                        truth_by_task[task] = truth.copy()
                    else:
                        require(np.array_equal(truth_by_task[task], truth),
                                f"Wave 2 validation alignment differs: {directory}")
                    for age in cfg["sequence"]["evidence_ages"]:
                        for suffix in ("full_prediction", "reset_prediction",
                                       "prior_prediction"):
                            require(values[f"age_{age}_{suffix}"].shape == (480,),
                                    f"Wave 2 prediction shape differs: {directory}")
                        require(np.isfinite(values[f"age_{age}_full_mse"]).all()
                                and np.isfinite(
                                    values[f"age_{age}_reset_mse"]).all(),
                                f"Wave 2 non-finite MSE: {directory}")
        for age in cfg["sequence"]["evidence_ages"]:
            record = task_summary.get("ages", {}).get(str(age), {})
            require(set(record.get("arms", {})) == set(arms),
                    f"Wave 2 arm set differs: {task}/age{age}")
            for arm, contrast in record.get("paired_vs_none", {}).items():
                require(arm != "none", "Wave 2 none arm has paired contrast")
                verify_bootstrap_record(contrast)
            require(set(record.get("paired_vs_none", {})) == set(arms) - {"none"},
                    f"Wave 2 paired arm set differs: {task}/age{age}")
            require(set(record.get("full_vs_context_reset", {})) == set(arms),
                    f"Wave 2 reset arm set differs: {task}/age{age}")
            for contrast in record["full_vs_context_reset"].values():
                verify_bootstrap_record(contrast)
    require(len(artifacts) == 200, "Wave 2 verified artifact count differs")
    require(not (formal / "stop_receipt.json").exists(),
            "Wave 2 formal stop receipt exists")

    summary_hash = hasher.digest(formal / "summary.json")
    provenance_hash = hasher.digest(formal / "provenance.json")
    official_hash = _require_official_verification(
        formal, hasher, "dinowm_wave2_spatial_verification_v1_1",
        protocol, artifacts,
        {"summary_sha256": summary_hash,
         "provenance_sha256": provenance_hash})
    return {
        "study": cfg["study"],
        "status": "verified",
        "cells": 50,
        "cell_artifacts": len(artifacts),
        "physical_gpu": 1,
        "bootstrap": {
            "draws": 20_000,
            "seed_unit": "matched carrier seed",
            "episode_unit": "class-stratified held-out episode",
            "paired": True,
        },
        "preoutcome_numerical_amendment_verified": True,
        "cache_reused_by_hard_link": True,
        "summary_sha256": summary_hash,
        "provenance_sha256": provenance_hash,
        "cache_manifest_sha256": hasher.digest(cache_manifest_path),
        "official_verification_sha256": official_hash,
    }


def verify_wave3(root: Path, hasher: HashVerifier) -> dict[str, Any]:
    label = "Wave 3"
    cfg, lock, output, formal = _verify_lock(
        root, WAVE3_CONFIG, hasher, label)
    require(lock.get("grid") == {
        "tasks": 1, "arms": 5, "seeds": 5, "cells": 25,
    }, "Wave 3 locked grid differs")
    protocol = lock["protocol_sha256"]
    cache = read_json(output / "cache/manifest.json", "Wave 3 cache manifest")
    admission = read_json(formal / "admission.json", "Wave 3 admission")
    controller = read_json(
        formal / "controller_gate.json", "Wave 3 controller gate")
    summary = read_json(formal / "summary.json", "Wave 3 summary")
    carrier = read_json(
        formal / "carrier_summary.json", "Wave 3 carrier summary")
    use = read_json(
        formal / "external_use_summary.json", "Wave 3 use summary")
    provenance = read_json(formal / "provenance.json", "Wave 3 provenance")
    progress = read_json(formal / "progress.json", "Wave 3 progress")
    require(cache.get("precarrier_gates_passed") is True
            and admission.get("admitted") is True
            and controller.get("admitted") is True,
            "Wave 3 pre-carrier gate failed")
    require(summary.get("status") == carrier.get("status")
            == use.get("status") == provenance.get("status") == "complete",
            "Wave 3 formal study is incomplete")
    require(all(record.get("protocol_sha256") == protocol
                for record in (summary, carrier, use, provenance)),
            "Wave 3 formal protocol differs")
    require(provenance.get("physical_gpu") == 2
            and provenance.get("cuda_visible_devices") == "2"
            and provenance.get("paper_modified_by_wave3") is False,
            "Wave 3 GPU/paper provenance differs")
    require(provenance.get("runtime_host_digest")
            == provenance.get("runtime_host_digest_after"),
            "Wave 3 frozen host changed")
    require(progress.get("count") == progress.get("expected") == 25,
            "Wave 3 grid is incomplete")
    require(controller.get("current_mujoco_version", "0").split(".")[0] >= "3"
            and controller.get("deterministic_replay_fidelity") == 1.0,
            "Wave 3 current-MuJoCo/replay gate differs")
    require(controller.get("oracle_executed_success")
            >= cfg["external_use"]["oracle_success_minimum"],
            "Wave 3 controller oracle gate differs")
    require(use.get("scope", {}).get("native_planner") is False,
            "Wave 3 use result overclaims native planning")
    consumers = use.get("consumer_receipts", [])
    require(len(consumers) == 5
            and all(record.get("arm_blind") is True
                    and record.get("arm_identifier_feature") is False
                    for record in consumers),
            "Wave 3 arm-blind consumer contract differs")
    inference = carrier.get("inference", {})
    require(inference.get("draws") == 20_000
            and inference.get("paired") is True
            and inference.get("resampling_units")
            == ["matched carrier seed", "native episode"],
            "Wave 3 bootstrap units differ")

    arms = list(cfg["training"]["arms"])
    seeds = list(map(int, cfg["training"]["seeds"]))
    expected_dirs = {
        (formal / "cells" / arm / f"s{seed}").resolve()
        for arm in arms for seed in seeds
    }
    actual_dirs = {
        path.resolve() for path in (formal / "cells").glob("*/s*")
        if path.is_dir()
    }
    require(actual_dirs == expected_dirs, "Wave 3 cell directory set differs")
    artifacts: dict[str, str] = {}
    truth: np.ndarray | None = None
    for arm in arms:
        for seed in seeds:
            directory = formal / "cells" / arm / f"s{seed}"
            manifest = read_json(directory / "manifest.json",
                                 f"Wave 3 manifest {arm}/s{seed}")
            metrics = read_json(directory / "metrics.json",
                                f"Wave 3 metrics {arm}/s{seed}")
            require(manifest.get("protocol_sha256") == protocol
                    and metrics.get("protocol_sha256") == protocol,
                    f"Wave 3 cell protocol differs: {directory}")
            require(metrics.get("physical_gpu") == 2
                    and metrics.get("cuda_visible_devices") == "2"
                    and metrics.get("carrier_parameters") == PARAMETERS[arm],
                    f"Wave 3 cell parameters/GPU differ: {directory}")
            require(metrics.get("host_unchanged") is True
                    and metrics.get("training_labels_used") is False,
                    f"Wave 3 cell invariant differs: {directory}")
            records = manifest.get("artifacts", {})
            require(set(records) == {
                "carrier.pt", "history.csv", "metrics.json",
                "use_features.npz", "validation_predictions.npz",
            }, f"Wave 3 artifact set differs: {directory}")
            for name, record in records.items():
                path = directory / name
                digest = verify_artifact(
                    path, record, hasher, f"Wave 3 {arm}/s{seed}/{name}")
                artifacts[str(path.relative_to(formal))] = digest
            with np.load(directory / "validation_predictions.npz",
                         allow_pickle=False) as values:
                current = values["truth"]
                require(current.shape == (480,),
                        f"Wave 3 truth shape differs: {directory}")
                if truth is None:
                    truth = current.copy()
                else:
                    require(np.array_equal(truth, current),
                            f"Wave 3 validation alignment differs: {directory}")
                for age in cfg["sequence"]["evidence_ages"]:
                    for suffix in ("full_prediction", "reset_prediction",
                                   "prior_prediction", "full_mse", "reset_mse"):
                        require(values[f"age_{age}_{suffix}"].shape == (480,),
                                f"Wave 3 cell output shape differs: {directory}")
            with np.load(directory / "use_features.npz",
                         allow_pickle=False) as values:
                require(values["train_feature"].shape == (1200, 8064)
                        and values["validation_feature"].shape == (480, 8064),
                        f"Wave 3 use feature shape differs: {directory}")
    require(len(artifacts) == 125, "Wave 3 verified artifact count differs")
    for age in cfg["sequence"]["evidence_ages"]:
        record = carrier.get("results", {}).get(str(age), {})
        require(set(record.get("arms", {})) == set(arms),
                f"Wave 3 arm set differs at age {age}")
        require(set(record.get("paired_vs_none", {})) == set(arms) - {"none"},
                f"Wave 3 paired arm set differs at age {age}")
        require(set(record.get("full_vs_context_reset", {})) == set(arms),
                f"Wave 3 reset arm set differs at age {age}")
        for contrast in list(record["paired_vs_none"].values()) \
                + list(record["full_vs_context_reset"].values()):
            verify_bootstrap_record(contrast, native_clusters=120)
    for arm, record in use.get("arms", {}).items():
        require(arm in arms, f"Wave 3 unexpected use arm: {arm}")
        for name in ("goal_accuracy", "executed_success", "contrast_vs_none",
                     "contrast_vs_random"):
            verify_bootstrap_record(record[name], native_clusters=120)
    require(set(use.get("arms", {})) == set(arms),
            "Wave 3 use arm set differs")
    verify_bootstrap_record(use["realized_random_goal"], native_clusters=120)
    use_artifact = use.get("artifact", {})
    use_artifact_path = repository_path(root, use_artifact.get("path", ""))
    verify_artifact(use_artifact_path, use_artifact, hasher,
                    "Wave 3 external-use predictions")
    require(not (formal / "formal_stop_receipt.json").exists()
            and not (formal / "stop_receipt.json").exists(),
            "Wave 3 formal stop receipt exists")

    summary_hash = hasher.digest(formal / "summary.json")
    carrier_hash = hasher.digest(formal / "carrier_summary.json")
    use_hash = hasher.digest(formal / "external_use_summary.json")
    provenance_hash = hasher.digest(formal / "provenance.json")
    official_hash = _require_official_verification(
        formal, hasher, "dinowm_pointmaze_wave3_verification_v1",
        protocol, artifacts,
        {"summary_sha256": summary_hash,
         "carrier_summary_sha256": carrier_hash,
         "external_use_summary_sha256": use_hash,
         "provenance_sha256": provenance_hash})
    return {
        "study": cfg["study"],
        "status": "verified",
        "cells": 25,
        "cell_artifacts": len(artifacts),
        "top_level_artifacts": 1,
        "physical_gpu": 2,
        "bootstrap": {
            "draws": 20_000,
            "units": ["matched carrier seed", "native episode"],
            "native_validation_episode_clusters": 120,
            "paired": True,
        },
        "current_mujoco_execution": True,
        "arm_blind_external_consumer": True,
        "summary_sha256": summary_hash,
        "carrier_summary_sha256": carrier_hash,
        "external_use_summary_sha256": use_hash,
        "provenance_sha256": provenance_hash,
        "official_verification_sha256": official_hash,
    }


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    preflight_completion(root)
    hasher = HashVerifier()
    waves = {
        "wave1_1": verify_wave1(root, hasher),
        "wave2_v1_1": verify_wave2(root, hasher),
        "wave3": verify_wave3(root, hasher),
    }
    require(all(value["status"] == "verified" for value in waves.values()),
            "not every wave is verified")
    for name in ("wave2_v1_1", "wave3"):
        official_hash = waves[name].get("official_verification_sha256")
        require(isinstance(official_hash, str) and len(official_hash) == 64,
                f"{name} official verification hash is absent")
    script_path = Path(__file__).resolve()
    verifier_paths = {
        "wave2_base": root / "scripts/verify_dinowm_wave2_spatial_carrier.py",
        "wave2_v1_1": root
        / "scripts/verify_dinowm_wave2_spatial_carrier_v1_1.py",
        "wave3": root / "scripts/verify_dinowm_pointmaze_wave3.py",
    }
    return {
        "schema": "paper_a_cross_wave_completion_receipt_v1",
        "status": "complete",
        "scope": [
            "paper-a-matched-color-v1-1",
            "dinowm-wave2-spatial-carrier-v1-1",
            "dinowm-pointmaze-wave3",
        ],
        "read_only_verification": True,
        "scientific_cross_wave_aggregation": False,
        "sealed_locks_modified": False,
        "paper_files_modified": False,
        "waves": waves,
        "totals": {
            "formal_cells": sum(value["cells"] for value in waves.values()),
            "cell_artifacts_hashed": sum(
                value["cell_artifacts"] for value in waves.values()),
            "physical_gpu_cell_counts": {"0": 50, "1": 50, "2": 25,
                                         "3": 0},
            "cuda3_used": False,
        },
        "auditor": {
            "path": str(script_path.relative_to(root)),
            "sha256": hasher.digest(script_path),
            "official_verifier_sha256": {
                name: hasher.digest(path)
                for name, path in verifier_paths.items()
            },
        },
        "claim_boundary": (
            "This receipt establishes completion and artifact integrity only; "
            "it does not pool scientific endpoints or create a cross-family "
            "performance comparison."
        ),
    }


def emit_receipt(root: Path, destination: Path,
                 payload: Mapping[str, Any], *, execute: bool) -> bool:
    """Print-only unless explicitly authorized; then create one atomic file."""
    if not execute:
        return False
    root = root.resolve()
    target = repository_path(root, destination)
    protected = [
        (root / WAVE1_ROOT).resolve(),
        repository_path(root, read_yaml(
            root / WAVE2_CONFIG, "Wave 2 config")["artifacts"]["root"]),
        repository_path(root, read_yaml(
            root / WAVE3_CONFIG, "Wave 3 config")["artifacts"]["root"]),
    ]
    require(not any(target == path or path in target.parents
                    for path in protected),
            "cross-wave receipt may not be written inside an experiment root")
    require(not target.exists(), f"cross-wave receipt already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(stable_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return True


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root (primarily for isolated tests)")
    parser.add_argument("--execute", action="store_true",
                        help="atomically create the new cross-wave receipt")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT,
                        help="new receipt path, repository-relative by default")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = audit_repository(args.root)
        wrote = emit_receipt(
            args.root, args.receipt, payload, execute=bool(args.execute))
    except (AuditFailure, KeyError, TypeError, ValueError, OSError) as error:
        print(f"[cross-wave-audit] FAIL: {error}", file=sys.stderr)
        return 2
    print(stable_json(payload), end="")
    if wrote:
        print(f"[cross-wave-audit] wrote {args.receipt}", file=sys.stderr)
    else:
        print("[cross-wave-audit] dry run; no file written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
