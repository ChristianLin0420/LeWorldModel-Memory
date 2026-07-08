#!/usr/bin/env python3
"""Exhaustively audit and aggregate the locked matched-host Wave-1 grid."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, stable_json  # noqa: E402
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    AGES,
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    TARGETS,
    load_locked_spec,
    output_path,
    sha256_file,
)
from scripts.prepare_paper_a_matched_host import host_manifest_path  # noqa: E402
from scripts.train_paper_a_matched_host import carrier_directory  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _stat(point: float, samples: np.ndarray,
          seed_values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(point),
        "ci95": [float(value) for value in np.quantile(values, (.025, .975))],
        "ci90": [float(value) for value in np.quantile(values, (.05, .95))],
        "seed_values": [float(value) for value in np.asarray(seed_values)],
    }


def _stratified_weights(joint: np.ndarray, draws: int,
                        rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(joint, dtype=np.int64)
    if labels.shape != (480,) or not np.array_equal(
            np.bincount(labels, minlength=16), np.full(16, 30)):
        raise ValueError("validation joint labels are not exactly 16-way balanced")
    weights = np.zeros((draws, len(labels)), dtype=np.float32)
    rows = np.arange(draws)[:, None]
    for label in range(16):
        indices = np.flatnonzero(labels == label)
        sampled = indices[rng.integers(0, len(indices), size=(draws, len(indices)))]
        np.add.at(weights, (np.broadcast_to(rows, sampled.shape), sampled), 1.0)
    weights /= float(len(labels))
    return weights


def _load_grid(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray,
                                                    dict[str, Any]]:
    # H,A,S,T,G,E
    correct = np.empty((len(HOSTS), len(ARMS), len(SEEDS), len(TARGETS),
                        len(AGES), 480), dtype=np.float32)
    observed = np.empty(correct.shape[:-1], dtype=np.float64)
    joint = np.empty((len(HOSTS), 480), dtype=np.int64)
    manifests: dict[str, Any] = {}
    expected_directories = set()
    artifact_count = 0
    for host_index, host in enumerate(HOSTS):
        admission_path = host_manifest_path(spec, host)
        if not admission_path.is_file():
            raise FileNotFoundError(f"missing host admission {admission_path}")
        admission = json.loads(admission_path.read_text())
        if admission.get("lock") != spec["_lock"] \
                or admission.get("status") != "admitted" \
                or admission.get("all_targets_ages_admitted") is not True \
                or admission.get("frozen_host_unchanged") is not True:
            raise ValueError(f"invalid host admission {admission_path}")
        manifests[host] = {
            "path": str(admission_path.relative_to(ROOT)),
            "sha256": sha256_file(admission_path),
        }
        reference_labels: dict[str, list[int]] | None = None
        reference_joint: list[int] | None = None
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                directory = carrier_directory(spec, host, arm, seed)
                expected_directories.add(directory.resolve())
                metrics_path = directory / "metrics.json"
                manifest_path = directory / "manifest.json"
                history_path = directory / "history.csv"
                checkpoint_path = directory / "carrier.pt"
                if not all(path.is_file() for path in (
                        metrics_path, manifest_path, history_path,
                        checkpoint_path)):
                    raise FileNotFoundError(f"incomplete matched carrier {directory}")
                value = json.loads(metrics_path.read_text())
                manifest = json.loads(manifest_path.read_text())
                expected = {
                    "schema_version": 1, "study": spec["study"],
                    "branch": "matched-color-location-fixed-endpoint",
                    "lock": spec["_lock"], "host": host,
                    "arm": arm, "seed": seed, "device": "cuda:0",
                    "physical_gpu": 0, "age_balanced_mixture": True,
                    "frozen_host_unchanged": True,
                    "validation_labels_used_for_fitting": False,
                }
                failed = [key for key, expected_value in expected.items()
                          if value.get(key) != expected_value]
                artifacts = manifest.get("artifacts", {})
                hashes = {
                    "metrics": sha256_file(metrics_path),
                    "history": sha256_file(history_path),
                    "checkpoint": sha256_file(checkpoint_path),
                }
                if failed or value.get("ages") != list(AGES) \
                        or value.get("targets") != list(TARGETS) \
                        or set(artifacts) != set(hashes) \
                        or any(artifacts[key].get("sha256") != digest
                               for key, digest in hashes.items()):
                    raise ValueError(f"invalid matched carrier {directory}: {failed}")
                artifact_count += 3
                if reference_labels is None:
                    reference_labels = value["labels"]
                    reference_joint = value["joint_labels"]
                elif value["labels"] != reference_labels \
                        or value["joint_labels"] != reference_joint:
                    raise ValueError(f"labels differ across {host} carrier cells")
                for target_index, target in enumerate(TARGETS):
                    labels = np.asarray(value["labels"][target], dtype=np.int64)
                    if labels.shape != (480,) or not np.array_equal(
                            np.bincount(labels, minlength=4), np.full(4, 120)):
                        raise ValueError(f"{host}/{target} labels are not balanced")
                    for age_index, age in enumerate(AGES):
                        result = value["readouts"][target][f"age-{age}"]
                        current = np.asarray(result["correct"], dtype=np.float32)
                        if current.shape != (480,) \
                                or result.get("metric") != "balanced_accuracy" \
                                or abs(float(current.mean())
                                       - float(result["balanced_accuracy"])) > 1e-12:
                            raise ValueError(
                                f"invalid readout {host}/{arm}/s{seed}/{target}/{age}")
                        correct[host_index, arm_index, seed_index,
                                target_index, age_index] = current
                        observed[host_index, arm_index, seed_index,
                                 target_index, age_index] = float(
                                     result["balanced_accuracy"])
        joint[host_index] = np.asarray(reference_joint, dtype=np.int64)
    carrier_root = output_path(spec, "carriers")
    actual = {path.parent.resolve() for path in carrier_root.glob(
        "*/*/seed-*/metrics.json")}
    if actual != expected_directories:
        missing = sorted(map(str, expected_directories - actual))
        extra = sorted(map(str, actual - expected_directories))
        raise ValueError(f"carrier directory set differs: missing={missing}, extra={extra}")
    audit = {
        "host_admissions": manifests,
        "complete_cells": int(np.prod((len(HOSTS), len(ARMS), len(SEEDS)))),
        "expected_cells": 75,
        "hashed_cell_artifacts": artifact_count,
        "expected_hashed_cell_artifacts": 225,
        "physical_gpu_counts": {"0": 75, "1": 0, "2": 0, "3": 0},
        "unexpected_carrier_directories": 0,
    }
    return correct, joint, {"observed": observed, "audit": audit}


def _bootstrap(correct: np.ndarray, joint: np.ndarray, *, draws: int,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # The same carrier-seed resample is used for every host; episode resamples
    # are independent between hosts and paired within a host.
    seed_counts = np.zeros((draws, len(SEEDS)), dtype=np.float32)
    selected = rng.integers(0, len(SEEDS), size=(draws, len(SEEDS)))
    rows = np.arange(draws)[:, None]
    np.add.at(seed_counts, (np.broadcast_to(rows, selected.shape), selected), 1.0)
    seed_weights = seed_counts / float(len(SEEDS))
    point = correct.mean(axis=(2, 5))  # H,A,T,G
    samples = np.empty((draws, len(HOSTS), len(ARMS), len(TARGETS),
                        len(AGES)), dtype=np.float32)
    for host_index in range(len(HOSTS)):
        episode_weights = _stratified_weights(joint[host_index], draws, rng)
        # b,a,s,t,g = sum_e correct[a,s,t,g,e] * episode_weight[b,e]
        episode_mean = np.einsum(
            "astge,be->bastg", correct[host_index], episode_weights,
            optimize=True)
        samples[:, host_index] = np.einsum(
            "bastg,bs->batg", episode_mean, seed_weights, optimize=True)
    return point, samples


def _summarize(spec: dict[str, Any], correct: np.ndarray,
               observed: np.ndarray, point: np.ndarray,
               samples: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    none = ARMS.index("none")
    fixed = ARMS.index("fixed_trust")
    ssm = ARMS.index("ssm")
    age4, age15 = AGES.index(4), AGES.index(15)
    for host_index, host in enumerate(HOSTS):
        host_value: dict[str, Any] = {
            "metric": "balanced_accuracy", "validation_episodes": 480,
            "targets": {}, "fixed_minus_ssm": {},
        }
        for target_index, target in enumerate(TARGETS):
            target_value: dict[str, Any] = {
                "arms": {}, "paired_vs_none": {},
                "shortest_minus_longest": {},
            }
            for arm_index, arm in enumerate(ARMS):
                target_value["arms"][arm] = {}
                for age_index, age in enumerate(AGES):
                    target_value["arms"][arm][f"age-{age}"] = _stat(
                        point[host_index, arm_index, target_index, age_index],
                        samples[:, host_index, arm_index, target_index, age_index],
                        observed[host_index, arm_index, :, target_index, age_index])
                if arm != "none":
                    target_value["paired_vs_none"][arm] = {}
                    for age_index, age in enumerate(AGES):
                        target_value["paired_vs_none"][arm][f"age-{age}"] = _stat(
                            point[host_index, arm_index, target_index, age_index]
                            - point[host_index, none, target_index, age_index],
                            samples[:, host_index, arm_index, target_index, age_index]
                            - samples[:, host_index, none, target_index, age_index],
                            observed[host_index, arm_index, :, target_index, age_index]
                            - observed[host_index, none, :, target_index, age_index])
                target_value["shortest_minus_longest"][arm] = _stat(
                    point[host_index, arm_index, target_index, age4]
                    - point[host_index, arm_index, target_index, age15],
                    samples[:, host_index, arm_index, target_index, age4]
                    - samples[:, host_index, arm_index, target_index, age15],
                    observed[host_index, arm_index, :, target_index, age4]
                    - observed[host_index, arm_index, :, target_index, age15])
            host_value["targets"][target] = target_value
            host_value["fixed_minus_ssm"][target] = {}
            for age_index, age in enumerate(AGES):
                host_value["fixed_minus_ssm"][target][f"age-{age}"] = _stat(
                    point[host_index, fixed, target_index, age_index]
                    - point[host_index, ssm, target_index, age_index],
                    samples[:, host_index, fixed, target_index, age_index]
                    - samples[:, host_index, ssm, target_index, age_index],
                    observed[host_index, fixed, :, target_index, age_index]
                    - observed[host_index, ssm, :, target_index, age_index])
        result[host] = host_value

    reacher, pusht = HOSTS.index("reacher"), HOSTS.index("pusht")
    per_host_rank_point = (
        point[:, fixed, :, age15] - point[:, ssm, :, age15]).mean(axis=1)
    per_host_rank_samples = (
        samples[:, :, fixed, :, age15]
        - samples[:, :, ssm, :, age15]).mean(axis=2)
    interaction_point = per_host_rank_point[pusht] - per_host_rank_point[reacher]
    interaction_samples = (per_host_rank_samples[:, pusht]
                           - per_host_rank_samples[:, reacher])
    primary = _stat(
        interaction_point, interaction_samples,
        (observed[pusht, fixed, :, :, age15]
         - observed[pusht, ssm, :, :, age15]).mean(axis=1)
        - (observed[reacher, fixed, :, :, age15]
           - observed[reacher, ssm, :, :, age15]).mean(axis=1))
    margin = float(spec["statistics"]["equivalence_margin"])
    primary["equivalent_within_margin"] = bool(
        primary["ci90"][0] >= -margin and primary["ci90"][1] <= margin)
    primary["resolved_nonzero"] = bool(
        primary["ci95"][0] > 0 or primary["ci95"][1] < 0)
    primary["equivalence_margin"] = margin

    attribute_interactions = {}
    color, location = TARGETS.index("color"), TARGETS.index("location")
    for arm_name in ("ssm", "fixed_trust"):
        arm = ARMS.index(arm_name)
        gain_point = point[:, arm, :, age15] - point[:, none, :, age15]
        gain_samples = (samples[:, :, arm, :, age15]
                        - samples[:, :, none, :, age15])
        value_point = ((gain_point[pusht, color] - gain_point[pusht, location])
                       - (gain_point[reacher, color]
                          - gain_point[reacher, location]))
        value_samples = (
            (gain_samples[:, pusht, color] - gain_samples[:, pusht, location])
            - (gain_samples[:, reacher, color]
               - gain_samples[:, reacher, location]))
        seed_gain = observed[:, arm, :, :, age15] - observed[:, none, :, :, age15]
        seed_values = ((seed_gain[pusht, :, color]
                        - seed_gain[pusht, :, location])
                       - (seed_gain[reacher, :, color]
                          - seed_gain[reacher, :, location]))
        attribute_interactions[arm_name] = _stat(
            value_point, value_samples, seed_values)
    return {
        "hosts": result,
        "primary_ranking_interaction": primary,
        "attribute_interactions_at_age_15": attribute_interactions,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched aggregation writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    destination = output_path(spec, "root") / "summary.json"
    audit_path = output_path(spec, "root") / "final_audit.json"
    if destination.exists() or audit_path.exists():
        raise FileExistsError("matched-host aggregation is already closed")
    correct, joint, loaded = _load_grid(spec)
    draws = int(spec["statistics"]["bootstrap_draws"])
    seed = int(spec["statistics"]["bootstrap_seed"])
    point, samples = _bootstrap(correct, joint, draws=draws, seed=seed)
    analysis = _summarize(
        spec, correct, loaded["observed"], point, samples)
    summary = {
        "schema_version": 1, "study": spec["study"],
        "branch": "matched-color-location-fixed-endpoint",
        "lock": spec["_lock"], "complete": True,
        "hosts": list(HOSTS), "targets": list(TARGETS),
        "ages": list(AGES), "arms": list(ARMS), "seeds": list(SEEDS),
        "bootstrap": {
            "draws": draws, "seed": seed,
            "joint_seed_resampling_across_hosts": True,
            "independent_host_episode_resampling": True,
            "episode_stratification": "16-way joint color-location label",
            "paired_within_host": ["target", "age", "arm"],
        },
        **analysis,
        "claim_boundary": (
            "Matched cue attributes, ages, endpoint, class count, renderer, and "
            "carrier budget identify a frozen host-system interaction; checkpoint "
            "training distribution and native environment remain bundled."),
    }
    summary_hash = atomic_text(destination, stable_json(summary))
    audit = {
        "schema_version": 1, "study": spec["study"],
        "lock": spec["_lock"], "status": "complete",
        **loaded["audit"],
        "summary": {"path": str(destination.relative_to(ROOT)),
                    "sha256": summary_hash},
        "bootstrap_draws": draws,
        "cuda3_used": False,
    }
    atomic_text(audit_path, stable_json(audit))
    lines = ["# Matched-host Wave-1 audit", "", "Complete: true", "",
             "## Primary ranking interaction", "",
             json.dumps(analysis["primary_ranking_interaction"], indent=2), ""]
    atomic_text(destination.with_suffix(".md"), "\n".join(lines))
    print(f"[matched-host] wrote {destination} and {audit_path}", flush=True)


if __name__ == "__main__":
    main()
