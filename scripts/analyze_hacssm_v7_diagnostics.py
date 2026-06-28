#!/usr/bin/env python3
"""Deterministic post-hoc diagnostics for the sealed HACSSM/HCRD-v7 study.

The prospective pilot, final decision, and primary manifest are immutable inputs.  This
script verifies them, recomputes their decision arithmetic, and publishes a new sibling
package.  It never writes into ``outputs/hacssm_v7_shared``.  Cross-environment summaries
use matched relative reductions; raw PCA MSE is summarized only within an environment.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

V7_ROOT = REPO_ROOT / "outputs" / "hacssm_v7_shared"
V6_ROOT = REPO_ROOT / "outputs" / "hacssm_v6_shared"
V7_MANIFEST_SHA256 = "98eda8abec229753381bed5f22c70317428242470cc6f40b6a3f9c16d0f55c11"
V6_MANIFEST_SHA256 = "915484f69ec78b4dead79a25e5fa096667a4c76034e88338bd46476f6fbc495c"
V7_PRODUCER_COMMIT = "56a294a67b0d8bf8d04a75f31f275b6063c4c8f6"
V6_PRODUCER_COMMIT = "5ae7de8e31780a3892ebc08f250532fa5661e313"

ENVIRONMENTS = (
    "dmc:reacher.hard.occ",
    "dmc:ball_in_cup.catch.occ",
    "dmc:finger.spin.occ",
    "dmc:cheetah.run.occ",
    "ogbench:cube-single.occ",
)
DESIGNS = (
    "ssm",
    "hacsmv4_two_noaux",
    "hacssmv6",
    "hacssmv6_static",
    "hacssmv7_noaux",
    "hacssmv7_sharedaction",
    "hacssmv7_noshrink",
    "hacssmv7_actiononly",
    "hacssmv7_uniform",
    "hacssmv7_norecovery",
    "hacssmv7_noaction",
    "hacssmv7_single",
    "hacssmv7",
)
V7_DESIGNS = tuple(design for design in DESIGNS if design.startswith("hacssmv7"))
ANCHORS = ("ssm", "hacsmv4_two_noaux", "hacssmv6", "hacssmv6_static")
SEEDS = (0, 1, 2, 3, 4)
PILOT_SEEDS = (0, 1, 2)
COMPLETION_SEEDS = (3, 4)
PRIMARY = "clean_mse_first_post"
PHASE_METRICS = (
    "clean_mse_pre",
    "clean_mse_blackout_transition",
    "clean_mse_deep_blackout",
    "clean_mse_first_post",
    "clean_mse_recovery",
    "clean_mse_late_post",
    "clean_mse_all",
)
PREDICTIVE_IDENTITY_METRICS = (
    "val_pred_loss",
    *PHASE_METRICS,
    "clean_mse_first_post_ablated",
    "clean_input_mse_first_post",
    "last_visible_mse_first_post",
    "constant_mse_first_post",
    "persistence_mse_first_post",
)
V7_PRIMARY_INPUTS = (
    "protocol.json",
    "pilot_per_run.csv",
    "pilot_grouped.csv",
    "pilot_paired_contrasts.csv",
    "pilot_convergence.csv",
    "pilot_decision.json",
    "per_run.csv",
    "grouped.csv",
    "paired_contrasts.csv",
    "convergence.csv",
    "decision.json",
    "hacssm_v7_manifest.json",
    "hacssm_v7_manifest.sha256",
)
OUTPUT_FILES = (
    "summary.json",
    "pairwise_contrasts.csv",
    "env_ranks.csv",
    "cell_ranks.csv",
    "phase_contrasts.csv",
    "seed_stage.csv",
    "convergence_summary.csv",
    "mechanism_attribution.csv",
    "learned_parameters_per_run.csv",
    "learned_parameters_summary.csv",
    "objective_history_summary.csv",
    "anchor_reproducibility.csv",
    "anchor_reproducibility_summary.json",
)


def reject_non_rfc_json(token: str) -> None:
    raise ValueError(f"non-RFC JSON constant {token}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=reject_non_rfc_json)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} is not finite: {value!r}")
    return number


def percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile requires values and probability in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def population_std(values: Sequence[float]) -> float:
    center = mean(values)
    return math.sqrt(mean((value - center) ** 2 for value in values))


def run_name(env: str, design: str, seed: int) -> str:
    return f"lewm-{env}-{design}-s{seed}"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent CSV fields: {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_manifest(
    root: Path,
    manifest_name: str,
    expected_sha: str,
    expected_commit: str,
    expected_study: str,
) -> dict[str, Any]:
    manifest_path = root / manifest_name
    sidecar_path = manifest_path.with_suffix(".sha256")
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"missing manifest pair under {root}")
    observed = sha256(manifest_path)
    if observed != expected_sha:
        raise ValueError(f"{manifest_path}: {observed} != frozen {expected_sha}")
    wanted_sidecar = f"{observed}  {manifest_path.name}\n"
    if sidecar_path.read_text() != wanted_sidecar:
        raise ValueError(f"manifest sidecar mismatch: {sidecar_path}")
    manifest = read_json(manifest_path)
    cloud = manifest.get("wandb_cloud_verification")
    expected_cloud = {
        "verified_finished_runs": 325,
        "verified_complete_epoch_histories": 325,
        "verified_rollout_artifacts": 325,
        "verified_rollout_tables": 325,
        "verified_rollout_videos": 325,
    }
    if not isinstance(cloud, dict) or any(cloud.get(k) != v for k, v in expected_cloud.items()):
        raise ValueError(f"incomplete W&B cloud verification in {manifest_path}")
    if (
        manifest.get("completed_runs") != 325
        or manifest.get("expected_runs") != 325
        or manifest.get("all_requested_runs_completed") is not True
        or manifest.get("producer_git_clean") is not True
        or manifest.get("producer_git_commit") != expected_commit
        or expected_study.lower() not in str(manifest.get("study", "")).lower()
    ):
        raise ValueError(f"manifest does not attest the frozen complete grid: {manifest_path}")
    if not isinstance(manifest.get("output_artifacts"), dict):
        raise ValueError(f"manifest lacks output_artifacts: {manifest_path}")
    return manifest


class ArtifactReader:
    """Hash-gated reader for files recorded by one primary manifest."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.hashes: dict[str, str] = {}

    def verify(self, path: Path, section: str = "output_artifacts") -> str:
        absolute = path.absolute()
        relative = absolute.relative_to(REPO_ROOT).as_posix()
        cache_key = f"{section}:{relative}"
        if cache_key in self.hashes:
            return self.hashes[cache_key]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected regular artifact: {path}")
        records = self.manifest.get(section)
        record = records.get(relative) if isinstance(records, dict) else None
        if not isinstance(record, dict):
            raise ValueError(f"artifact absent from {section}: {relative}")
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"artifact size differs from manifest: {relative}")
        digest = sha256(path)
        if record.get("sha256") != digest:
            raise ValueError(f"artifact hash differs from manifest: {relative}")
        self.hashes[cache_key] = digest
        return digest

    def json(self, path: Path) -> Any:
        self.verify(path)
        return read_json(path)

    def checkpoint(self, path: Path) -> dict[str, Any]:
        self.verify(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "model_state_dict", "args", "final_metrics", "history"
        }:
            raise ValueError(f"invalid checkpoint schema: {path}")
        state, history = checkpoint["model_state_dict"], checkpoint["history"]
        if not isinstance(state, dict) or not state or not isinstance(history, list) or len(history) != 200:
            raise ValueError(f"invalid checkpoint payload: {path}")
        for key, tensor in state.items():
            if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"malformed state entry in {path}: {key!r}")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"non-finite state tensor in {path}: {key}")
        return checkpoint

    def rollout(self, path: Path) -> dict[str, np.ndarray]:
        self.verify(path)
        with np.load(path, allow_pickle=False) as archive:
            result = {key: np.array(archive[key], copy=True) for key in archive.files}
        if not result:
            raise ValueError(f"empty rollout: {path}")
        return result

    def verify_unchanged(self) -> None:
        for cache_key, digest in self.hashes.items():
            _section, relative = cache_key.split(":", 1)
            if sha256(REPO_ROOT / relative) != digest:
                raise RuntimeError(f"locked artifact changed during analysis: {relative}")


def verify_v7_primary_inputs(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    artifacts = manifest["output_artifacts"]
    for name in V7_PRIMARY_INPUTS:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing V7 primary input: {path}")
        hashes[name] = sha256(path)
        if name in {"hacssm_v7_manifest.json", "hacssm_v7_manifest.sha256"}:
            continue
        relative = path.absolute().relative_to(REPO_ROOT).as_posix()
        record = artifacts.get(relative)
        if not isinstance(record, dict) or record.get("sha256") != hashes[name]:
            raise ValueError(f"primary input differs from V7 manifest: {name}")
    return hashes


def validate_rows(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, int], Mapping[str, str]]:
    required = {"run", "env", "design", "seed", PRIMARY, *PHASE_METRICS}
    if not required.issubset(rows[0]):
        raise ValueError(f"per_run.csv missing fields: {sorted(required - set(rows[0]))}")
    lookup: dict[tuple[str, str, int], Mapping[str, str]] = {}
    for index, row in enumerate(rows):
        env, design, seed = row["env"], row["design"], int(row["seed"])
        key = (env, design, seed)
        if key in lookup:
            raise ValueError(f"duplicate grid cell: {key}")
        if row["run"] != run_name(env, design, seed):
            raise ValueError(f"unexpected run name in per_run row {index}")
        lookup[key] = row
        for metric in PHASE_METRICS:
            value = finite(row[metric], f"row {index} {metric}")
            if value <= 0.0:
                raise ValueError(f"non-positive predictive metric at row {index}: {metric}")
    expected = {(env, design, seed) for env in ENVIRONMENTS for design in DESIGNS for seed in SEEDS}
    if set(lookup) != expected:
        raise ValueError("per_run.csv is not the exact 325-cell V7 grid")
    return lookup


def paired_summary(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    candidate: str,
    reference: str,
    *,
    metric: str = PRIMARY,
    envs: Iterable[str] = ENVIRONMENTS,
    seeds: Iterable[int] = SEEDS,
) -> dict[str, Any]:
    selected_envs, selected_seeds = tuple(envs), tuple(seeds)
    pairs = [
        (
            finite(lookup[(env, candidate, seed)][metric], f"{env}/{candidate}/{seed}/{metric}"),
            finite(lookup[(env, reference, seed)][metric], f"{env}/{reference}/{seed}/{metric}"),
        )
        for env in selected_envs for seed in selected_seeds
    ]
    if any(reference_value <= 0.0 for _, reference_value in pairs):
        raise ValueError(f"non-positive reference in {candidate} vs {reference}/{metric}")
    reductions = [(reference_value - candidate_value) / reference_value
                  for candidate_value, reference_value in pairs]
    return {
        "n_pairs": len(pairs),
        "candidate_mean_mse": mean(pair[0] for pair in pairs),
        "reference_mean_mse": mean(pair[1] for pair in pairs),
        "mean_paired_relative_reduction": mean(reductions),
        "median_paired_relative_reduction": median(reductions),
        "paired_wins": sum(candidate < reference_value for candidate, reference_value in pairs),
        "paired_ties": sum(candidate == reference_value for candidate, reference_value in pairs),
    }


def environment_wins(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    candidate: str,
    reference: str,
    *,
    metric: str = PRIMARY,
    seeds: Iterable[int] = SEEDS,
) -> int:
    selected = tuple(seeds)
    return sum(
        mean(finite(lookup[(env, candidate, seed)][metric], metric) for seed in selected)
        < mean(finite(lookup[(env, reference, seed)][metric], metric) for seed in selected)
        for env in ENVIRONMENTS
    )


def contrast_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for candidate in DESIGNS:
        for reference in DESIGNS:
            if candidate == reference:
                continue
            for env in (*ENVIRONMENTS, "__overall__"):
                overall = env == "__overall__"
                summary = paired_summary(
                    lookup, candidate, reference,
                    envs=ENVIRONMENTS if overall else (env,),
                )
                if overall:
                    summary["candidate_mean_mse"] = ""
                    summary["reference_mean_mse"] = ""
                result.append({
                    "candidate": candidate,
                    "reference": reference,
                    "env": env,
                    **summary,
                    "environment_mean_wins": (
                        environment_wins(lookup, candidate, reference) if overall else ""
                    ),
                })
    return result


def rank_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env_result, cell_result = [], []
    for env in ENVIRONMENTS:
        centers = {
            design: mean(finite(lookup[(env, design, seed)][PRIMARY], PRIMARY) for seed in SEEDS)
            for design in DESIGNS
        }
        values = {
            design: [finite(lookup[(env, design, seed)][PRIMARY], PRIMARY) for seed in SEEDS]
            for design in DESIGNS
        }
        for design in sorted(DESIGNS, key=lambda item: (centers[item], item)):
            center = centers[design]
            env_result.append({
                "env": env,
                "rank": 1 + sum(other < center for other in centers.values()),
                "design": design,
                "n_seeds": len(SEEDS),
                "clean_mse_first_post_mean": center,
                "clean_mse_first_post_population_std": population_std(values[design]),
            })
        for seed in SEEDS:
            per_cell = {
                design: finite(lookup[(env, design, seed)][PRIMARY], PRIMARY)
                for design in DESIGNS
            }
            for design in sorted(DESIGNS, key=lambda item: (per_cell[item], item)):
                value = per_cell[design]
                cell_result.append({
                    "env": env,
                    "seed": seed,
                    "rank": 1 + sum(other < value for other in per_cell.values()),
                    "design": design,
                    "clean_mse_first_post": value,
                })
    return env_result, cell_result


def phase_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for reference in DESIGNS:
        if reference == "hacssmv7":
            continue
        for metric in PHASE_METRICS:
            for env in (*ENVIRONMENTS, "__overall__"):
                overall = env == "__overall__"
                summary = paired_summary(
                    lookup, "hacssmv7", reference, metric=metric,
                    envs=ENVIRONMENTS if overall else (env,),
                )
                if overall:
                    summary["candidate_mean_mse"] = ""
                    summary["reference_mean_mse"] = ""
                result.append({
                    "metric": metric,
                    "candidate": "hacssmv7",
                    "reference": reference,
                    "env": env,
                    **summary,
                    "environment_mean_wins": (
                        environment_wins(lookup, "hacssmv7", reference, metric=metric)
                        if overall else ""
                    ),
                })
    return result


def seed_stage_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for stage, seeds in (("pilot", PILOT_SEEDS), ("completion", COMPLETION_SEEDS), ("all", SEEDS)):
        for reference in DESIGNS:
            if reference == "hacssmv7":
                continue
            result.append({
                "stage": stage,
                "seeds": ",".join(str(seed) for seed in seeds),
                "candidate": "hacssmv7",
                "reference": reference,
                **paired_summary(lookup, "hacssmv7", reference, seeds=seeds),
                "environment_mean_wins": environment_wins(
                    lookup, "hacssmv7", reference, seeds=seeds),
            })
    return result


def convergence_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    expected = {(env, design, seed) for env in ENVIRONMENTS for design in DESIGNS for seed in SEEDS}
    observed: set[tuple[str, str, int]] = set()
    by_design: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["env"], row["design"], int(row["seed"]))
        if key in observed:
            raise ValueError(f"duplicate convergence row: {key}")
        observed.add(key)
        by_design[row["design"]].append(abs(finite(row["relative_improvement"], f"conv {index}")))
    if observed != expected:
        raise ValueError("convergence.csv is not the exact V7 grid")
    result = []
    for design in (*DESIGNS, "__all__"):
        values = ([value for group in by_design.values() for value in group]
                  if design == "__all__" else by_design[design])
        result.append({
            "design": design,
            "n_runs": len(values),
            "absolute_window_change_median": median(values),
            "absolute_window_change_p95": percentile(values, 0.95),
            "absolute_window_change_max": max(values),
        })
    return result


def find_overall(
    contrasts: Sequence[Mapping[str, Any]], candidate: str, reference: str,
) -> Mapping[str, Any]:
    found = [row for row in contrasts if row["candidate"] == candidate
             and row["reference"] == reference and row["env"] == "__overall__"]
    if len(found) != 1:
        raise ValueError(f"expected one overall contrast: {candidate} vs {reference}")
    return found[0]


def mechanism_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
) -> list[dict[str, Any]]:
    pairs = (
        ("hacssmv7", "hacssmv7_noaux", "counterfactual objective increment"),
        ("hacssmv7", "hacssmv7_sharedaction", "level-specific action heads"),
        ("hacssmv7", "hacssmv7_noshrink", "learned static/dynamic shrinkage"),
        ("hacssmv7", "hacssmv7_actiononly", "counterfactual recovery vs action rollout"),
        ("hacssmv7", "hacssmv7_uniform", "hierarchical horizon assignment"),
        ("hacssmv7", "hacssmv7_norecovery", "restored-frame recovery term"),
        ("hacssmv7", "hacssmv7_noaction", "action pathway"),
        ("hacssmv7", "hacssmv7_single", "joint hierarchical read"),
        ("hacssmv7_noaux", "hacssmv6", "V7 inference without auxiliary vs dynamic V6"),
        ("hacssmv7_noaux", "hacssmv6_static", "V7 inference without auxiliary vs static V6"),
        ("hacssmv7_noaux", "hacsmv4_two_noaux", "V7 inference without auxiliary vs V4-two"),
    )
    result = []
    for candidate, reference, mechanism in pairs:
        summary = paired_summary(lookup, candidate, reference)
        summary["candidate_mean_mse"] = ""
        summary["reference_mean_mse"] = ""
        result.append({
            "mechanism": mechanism,
            "candidate": candidate,
            "reference": reference,
            **summary,
            "environment_mean_wins": environment_wins(lookup, candidate, reference),
        })
    return result


def tensor_l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().to(torch.float64)).item())


def state_group_l2(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], names: Iterable[str],
) -> float:
    total = 0.0
    for name in names:
        if name not in left or name not in right or left[name].shape != right[name].shape:
            raise ValueError(f"incompatible state group tensor: {name}")
        delta = left[name].detach().to(torch.float64) - right[name].detach().to(torch.float64)
        total += float(torch.sum(delta * delta).item())
    return math.sqrt(total)


MEMORY_GROUPS = {
    "memory_total": ("w_z", "w_e", "gate_bias", "route_logits", "shrink_logits",
                     "W_x.weight", "W_a.weight", "W_o.weight"),
    "action": ("W_a.weight",),
    "gate": ("w_z", "w_e", "gate_bias"),
    "shrink": ("shrink_logits",),
    "route": ("route_logits",),
    "input": ("W_x.weight",),
    "output": ("W_o.weight",),
}


def split_memory_state(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    result = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    required = {*MEMORY_GROUPS["memory_total"], "betas"}
    if set(result) != required:
        raise ValueError(f"unexpected V7 memory state keys for {prefix}: {sorted(result)}")
    return result


def history_summary(history: Sequence[Mapping[str, Any]], run: str) -> dict[str, Any]:
    if len(history) != 200:
        raise ValueError(f"{run}: expected 200 history entries")
    active_train: list[Mapping[str, Any]] = []
    for epoch, item in enumerate(history, 1):
        if item.get("epoch") != epoch or set(item) != {"epoch", "train", "val"}:
            raise ValueError(f"{run}: malformed epoch {epoch}")
        for split in ("train", "val"):
            values = item[split]
            for key in ("loss", "pred_loss", "sigreg_loss", "hier_loss", "hier_loss_weight",
                        "hier_loss_fast", "hier_loss_medium", "hier_loss_bridge",
                        "hier_loss_recovery", "hier_overlap"):
                finite(values.get(key), f"{run}/{epoch}/{split}/{key}")
            if float(values["hier_overlap"]) != 0.0:
                raise ValueError(f"{run}: nonzero hidden overlap at epoch {epoch}")
        if float(item["train"]["hier_loss_weight"]) > 0.0:
            active_train.append(item["train"])
    active_epochs = len(active_train)
    if active_epochs not in (0, 99):
        raise ValueError(f"{run}: active auxiliary epochs={active_epochs}, expected 0 or 99")
    def active_mean(key: str) -> float | str:
        return mean(finite(row[key], f"{run}/{key}") for row in active_train) if active_train else ""
    return {
        "active_aux_epochs": active_epochs,
        "active_train_hier_loss_mean": active_mean("hier_loss"),
        "active_train_hier_loss_fast_mean": active_mean("hier_loss_fast"),
        "active_train_hier_loss_medium_mean": active_mean("hier_loss_medium"),
        "active_train_hier_loss_bridge_mean": active_mean("hier_loss_bridge"),
        "active_train_hier_loss_recovery_mean": active_mean("hier_loss_recovery"),
        "active_weighted_aux_term_mean": (
            mean(finite(row["hier_loss"], "hier") * finite(row["hier_loss_weight"], "weight")
                 for row in active_train) if active_train else 0.0
        ),
        "final_val_hier_loss": finite(history[-1]["val"]["hier_loss"], f"{run}/final_hier"),
        "final_val_bridge": finite(history[-1]["val"]["hier_loss_bridge"], f"{run}/final_bridge"),
        "final_val_recovery": finite(history[-1]["val"]["hier_loss_recovery"], f"{run}/final_recovery"),
    }


def validate_metrics_against_row(
    row: Mapping[str, str], metrics: Mapping[str, Any], context: str,
) -> None:
    if metrics.get("env") != row["env"] or metrics.get("design") != row["design"]:
        raise ValueError(f"{context}: metrics identity mismatch")
    for key, text in row.items():
        if key in {"run", "env", "design", "seed", "trainable_parameters"}:
            continue
        if text == "":
            if metrics.get(key) is not None:
                raise ValueError(f"{context}: row omits available metric {key}")
        elif finite(text, f"{context}/{key}") != finite(metrics.get(key), f"{context}/metrics/{key}"):
            raise ValueError(f"{context}: per_run differs from metrics at {key}")


def parameter_diagnostics(
    root: Path,
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    reader: ArtifactReader,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_run: list[dict[str, Any]] = []
    objective_runs: list[dict[str, Any]] = []
    for env in ENVIRONMENTS:
        for seed in SEEDS:
            noaux_path = root / run_name(env, "hacssmv7_noaux", seed) / "model.pt"
            noaux_checkpoint = reader.checkpoint(noaux_path)
            noaux_state = noaux_checkpoint["model_state_dict"]
            noaux_student = split_memory_state(noaux_state, "mem_hacssmv7.")
            noaux_teacher = split_memory_state(noaux_state, "mem_hacssmv7_teacher.")
            for design in V7_DESIGNS:
                run = run_name(env, design, seed)
                run_dir = root / run
                checkpoint = noaux_checkpoint if design == "hacssmv7_noaux" else reader.checkpoint(run_dir / "model.pt")
                state = checkpoint["model_state_dict"]
                metrics = reader.json(run_dir / "metrics.json")
                if metrics != checkpoint["final_metrics"]:
                    raise ValueError(f"{run}: metrics/checkpoint mismatch")
                validate_metrics_against_row(lookup[(env, design, seed)], metrics, run)
                student = split_memory_state(state, "mem_hacssmv7.")
                teacher = split_memory_state(state, "mem_hacssmv7_teacher.")
                action = student["W_a.weight"].detach().to(torch.float64).reshape(2, 256, 6)
                action_norms = [tensor_l2(action[level]) for level in (0, 1)]
                cosine = ""
                if action_norms[0] > 0.0 and action_norms[1] > 0.0:
                    cosine = float(torch.sum(action[0] * action[1]).item()
                                   / (action_norms[0] * action_norms[1]))
                learned_rho = torch.sigmoid(student["shrink_logits"].detach().to(torch.float64))
                functional_rho = torch.ones_like(learned_rho) if design == "hacssmv7_noshrink" else learned_rho
                static_gate = torch.sigmoid(student["gate_bias"].detach().to(torch.float64))
                learned_route = torch.softmax(student["route_logits"].detach().to(torch.float64), dim=0)
                functional_route = (torch.tensor([0.0, 1.0], dtype=torch.float64)
                                    if design == "hacssmv7_single" else learned_route)
                # Reconcile state-derived values with the trainer's final receipts.
                receipt = (float(functional_rho[0]), float(functional_rho[1]),
                           action_norms[0], action_norms[1])
                logged = tuple(float(metrics[key]) for key in (
                    "rho_fast", "rho_medium", "action_head_fast_norm", "action_head_medium_norm"))
                if any(not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7)
                       for left, right in zip(receipt, logged)):
                    raise ValueError(f"{run}: learned parameter receipt mismatch")
                row: dict[str, Any] = {
                    "run": run,
                    "env": env,
                    "design": design,
                    "seed": seed,
                    "rho_fast_learned": float(learned_rho[0]),
                    "rho_medium_learned": float(learned_rho[1]),
                    "rho_fast_functional": float(functional_rho[0]),
                    "rho_medium_functional": float(functional_rho[1]),
                    "static_gate_fast": float(static_gate[0]),
                    "static_gate_medium": float(static_gate[1]),
                    "route_fast_learned": float(learned_route[0]),
                    "route_medium_learned": float(learned_route[1]),
                    "route_fast_functional": float(functional_route[0]),
                    "route_medium_functional": float(functional_route[1]),
                    "action_head_fast_l2": action_norms[0],
                    "action_head_medium_l2": action_norms[1],
                    "action_head_fast_d_l2": tensor_l2(action[0, :128]),
                    "action_head_fast_v_l2": tensor_l2(action[0, 128:]),
                    "action_head_medium_d_l2": tensor_l2(action[1, :128]),
                    "action_head_medium_v_l2": tensor_l2(action[1, 128:]),
                    "action_head_cosine": cosine,
                    "infl_all": finite(metrics["infl_all"], f"{run}/infl_all"),
                    "infl_fast": finite(metrics["infl_fast"], f"{run}/infl_fast"),
                    "infl_slow": finite(metrics["infl_slow"], f"{run}/infl_slow"),
                }
                for group, names in MEMORY_GROUPS.items():
                    row[f"student_teacher_{group}_l2"] = state_group_l2(student, teacher, names)
                    row[f"student_{group}_delta_l2_vs_noaux"] = state_group_l2(
                        student, noaux_student, names)
                    row[f"teacher_{group}_delta_l2_vs_noaux"] = state_group_l2(
                        teacher, noaux_teacher, names)
                predictor_keys = sorted(key for key in state if key.startswith("predictor."))
                row["predictor_delta_l2_vs_noaux"] = state_group_l2(
                    state, noaux_state, predictor_keys)
                history = history_summary(checkpoint["history"], run)
                row.update(history)
                per_run.append(row)
                objective_runs.append({"design": design, **history})

    numeric_fields = [key for key in per_run[0] if key not in {"run", "env", "design", "seed"}]
    summaries = []
    for design in V7_DESIGNS:
        selected = [row for row in per_run if row["design"] == design]
        output: dict[str, Any] = {"design": design, "n_runs": len(selected)}
        for field in numeric_fields:
            values = [float(row[field]) for row in selected if row[field] != ""]
            output[f"{field}_mean"] = mean(values) if values else ""
            output[f"{field}_population_std"] = population_std(values) if values else ""
        summaries.append(output)

    objective_summaries = []
    fields = [key for key in objective_runs[0] if key != "design"]
    for design in V7_DESIGNS:
        selected = [row for row in objective_runs if row["design"] == design]
        output = {"design": design, "n_runs": len(selected)}
        for field in fields:
            values = [float(row[field]) for row in selected if row[field] != ""]
            output[f"{field}_mean"] = mean(values) if values else ""
            output[f"{field}_population_std"] = population_std(values) if values else ""
        objective_summaries.append(output)
    return per_run, summaries, objective_summaries


def tensor_state_difference(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor],
) -> tuple[bool, float]:
    if set(left) != set(right):
        return False, math.inf
    exact, maximum = True, 0.0
    for key in sorted(left):
        one, two = left[key], right[key]
        if one.shape != two.shape or one.dtype != two.dtype:
            return False, math.inf
        equal = torch.equal(one, two)
        exact = exact and equal
        if one.is_floating_point():
            maximum = max(maximum, float((one.to(torch.float64) - two.to(torch.float64)).abs().max()))
        elif not equal:
            maximum = math.inf
    return exact, maximum


def rollout_difference(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray],
) -> tuple[bool, float]:
    if set(left) != set(right):
        return False, math.inf
    exact, maximum = True, 0.0
    for key in sorted(left):
        one, two = left[key], right[key]
        if one.shape != two.shape or one.dtype != two.dtype:
            return False, math.inf
        exact = exact and bool(np.array_equal(one, two))
        if np.issubdtype(one.dtype, np.number):
            maximum = max(maximum, float(np.max(np.abs(one.astype(np.float64) - two.astype(np.float64)))))
    return exact, maximum


HISTORY_FIELDS = (
    "loss", "pred_loss", "sigreg_loss", "pred_loss_all_valid", "pred_loss_first_post",
    "hier_loss", "hier_loss_fast", "hier_loss_medium", "hier_loss_weight",
)


def history_difference(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]],
) -> tuple[bool, float, int]:
    if len(left) != 200 or len(right) != 200:
        return False, math.inf, 0
    exact, maximum, compared = True, 0.0, 0
    for one, two in zip(left, right):
        if one.get("epoch") != two.get("epoch"):
            return False, math.inf, compared
        for split in ("train", "val"):
            for field in HISTORY_FIELDS:
                if field not in one[split] or field not in two[split]:
                    continue
                delta = abs(finite(one[split][field], field) - finite(two[split][field], field))
                exact = exact and delta == 0.0
                maximum = max(maximum, delta)
                compared += 1
    return exact, maximum, compared


def anchor_reproducibility(
    v7_root: Path,
    v6_root: Path,
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    v7_reader: ArtifactReader,
    v6_reader: ArtifactReader,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for env in ENVIRONMENTS:
        for design in ANCHORS:
            for seed in SEEDS:
                run = run_name(env, design, seed)
                v7_dir, v6_dir = v7_root / run, v6_root / run
                v7_checkpoint = v7_reader.checkpoint(v7_dir / "model.pt")
                v6_checkpoint = v6_reader.checkpoint(v6_dir / "model.pt")
                v7_metrics = v7_reader.json(v7_dir / "metrics.json")
                v6_metrics = v6_reader.json(v6_dir / "metrics.json")
                if v7_metrics != v7_checkpoint["final_metrics"] or v6_metrics != v6_checkpoint["final_metrics"]:
                    raise ValueError(f"{run}: checkpoint/metrics mismatch")
                validate_metrics_against_row(lookup[(env, design, seed)], v7_metrics, run)
                differences = [abs(finite(v7_metrics[key], key) - finite(v6_metrics[key], key))
                               for key in PREDICTIVE_IDENTITY_METRICS]
                state_exact, state_max = tensor_state_difference(
                    v7_checkpoint["model_state_dict"], v6_checkpoint["model_state_dict"])
                rollout_exact, rollout_max = rollout_difference(
                    v7_reader.rollout(v7_dir / "eval_rollout.npz"),
                    v6_reader.rollout(v6_dir / "eval_rollout.npz"),
                )
                history_exact, history_max, history_values = history_difference(
                    v7_checkpoint["history"], v6_checkpoint["history"])
                rows.append({
                    "run": run,
                    "env": env,
                    "design": design,
                    "seed": seed,
                    "predictive_metrics_compared": len(differences),
                    "predictive_metrics_exact": all(delta == 0.0 for delta in differences),
                    "max_abs_predictive_metric_difference": max(differences),
                    "model_state_exact": state_exact,
                    "max_abs_model_state_difference": state_max,
                    "rollout_exact": rollout_exact,
                    "max_abs_rollout_difference": rollout_max,
                    "history_values_compared": history_values,
                    "optimization_history_exact": history_exact,
                    "max_abs_optimization_history_difference": history_max,
                })
    summary_by_design = {}
    for design in ANCHORS:
        selected = [row for row in rows if row["design"] == design]
        summary_by_design[design] = {
            "cells": len(selected),
            "exact_predictive_metric_cells": sum(bool(row["predictive_metrics_exact"]) for row in selected),
            "exact_model_state_cells": sum(bool(row["model_state_exact"]) for row in selected),
            "exact_rollout_cells": sum(bool(row["rollout_exact"]) for row in selected),
            "exact_optimization_history_cells": sum(bool(row["optimization_history_exact"]) for row in selected),
            "max_abs_predictive_metric_difference": max(float(row["max_abs_predictive_metric_difference"])
                                                        for row in selected),
            "max_abs_model_state_difference": max(float(row["max_abs_model_state_difference"])
                                                  for row in selected),
            "max_abs_rollout_difference": max(float(row["max_abs_rollout_difference"])
                                              for row in selected),
            "max_abs_optimization_history_difference": max(
                float(row["max_abs_optimization_history_difference"]) for row in selected),
        }
    return rows, {
        "schema_version": 1,
        "matched_cells": len(rows),
        "all_predictive_metrics_exact": all(bool(row["predictive_metrics_exact"]) for row in rows),
        "all_model_states_exact": all(bool(row["model_state_exact"]) for row in rows),
        "all_rollouts_exact": all(bool(row["rollout_exact"]) for row in rows),
        "all_optimization_histories_exact": all(bool(row["optimization_history_exact"]) for row in rows),
        "by_design": summary_by_design,
    }


def recompute_decisions(
    rows: Sequence[Mapping[str, str]], convergence: Sequence[Mapping[str, str]], root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import scripts.analyze_hacssm_v7 as prospective

    prospective.configure_shared()
    pilot_rows = [row for row in rows if int(row["seed"]) in PILOT_SEEDS]
    pilot_convergence = [row for row in convergence if int(row["seed"]) in PILOT_SEEDS]
    pilot_contrasts = prospective.shared.contrast_rows(pilot_rows, candidate="hacssmv7")
    recomputed_pilot = prospective.pilot_decision(pilot_rows, pilot_convergence, pilot_contrasts)
    locked_pilot = read_json(root / "pilot_decision.json")
    if recomputed_pilot != locked_pilot:
        raise ValueError("recomputed V7 pilot differs from immutable pilot_decision.json")
    final_contrasts = prospective.shared.contrast_rows(rows, candidate="hacssmv7")
    recomputed_final = prospective.final_summary(
        rows, convergence, final_contrasts,
        pilot_screen_passed=recomputed_pilot["pilot_screen_passed"],
    )
    locked_final = read_json(root / "decision.json")
    if recomputed_final != locked_final:
        raise ValueError("recomputed V7 final differs from immutable decision.json")
    return recomputed_pilot, recomputed_final


def summary_document(
    contrasts: Sequence[Mapping[str, Any]],
    env_ranks: Sequence[Mapping[str, Any]],
    mechanism: Sequence[Mapping[str, Any]],
    convergence: Sequence[Mapping[str, Any]],
    parameter_summary: Sequence[Mapping[str, Any]],
    anchor_summary: Mapping[str, Any],
    pilot: Mapping[str, Any],
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rank_summary = {
        design: {
            "environment_rank_wins": sum(
                int(row["rank"]) == 1 for row in env_ranks if row["design"] == design),
            "mean_environment_rank": mean(
                float(row["rank"]) for row in env_ranks if row["design"] == design),
        }
        for design in DESIGNS
    }
    best = min(DESIGNS, key=lambda design: (
        -rank_summary[design]["environment_rank_wins"],
        rank_summary[design]["mean_environment_rank"], design))
    full_parameters = next(row for row in parameter_summary if row["design"] == "hacssmv7")
    all_convergence = next(row for row in convergence if row["design"] == "__all__")
    compact = lambda candidate, reference: {
        key: find_overall(contrasts, candidate, reference)[key]
        for key in ("mean_paired_relative_reduction", "median_paired_relative_reduction",
                    "paired_wins", "paired_ties", "n_pairs", "environment_mean_wins")
    }
    mechanism_map = {
        row["mechanism"]: {
            key: row[key] for key in (
                "candidate", "reference", "mean_paired_relative_reduction",
                "paired_wins", "paired_ties", "n_pairs", "environment_mean_wins")
        }
        for row in mechanism
    }
    return {
        "schema_version": 1,
        "scope": "descriptive post-hoc diagnostics; cannot alter the prospective pilot or final decision",
        "frozen_record": {
            "producer_git_commit": manifest["producer_git_commit"],
            "producer_git_clean": manifest["producer_git_clean"],
            "primary_manifest_sha256": V7_MANIFEST_SHA256,
            "completed_runs": manifest["completed_runs"],
            "wandb_cloud_verification": manifest["wandb_cloud_verification"],
        },
        "locked_pilot_decision": pilot["decision"],
        "locked_final_decision": decision["decision"],
        "good_enough_for_overall_best_claim": decision["good_enough_for_overall_best_claim"],
        "failed_final_criteria": sorted(key for key, value in decision["criteria"].items() if not value),
        "best_development_grid_design": {
            "design": best,
            **rank_summary[best],
            "vs_ssm": compact(best, "ssm") if best != "ssm" else None,
            "qualification": "Best by environment-rank wins on the adaptive locked grid, not an untouched-test claim.",
        },
        "key_contrasts": {
            "full_v7_vs_ssm": compact("hacssmv7", "ssm"),
            "full_v7_vs_v4_two": compact("hacssmv7", "hacsmv4_two_noaux"),
            "full_v7_vs_v6": compact("hacssmv7", "hacssmv6"),
            "full_v7_vs_v6_static": compact("hacssmv7", "hacssmv6_static"),
            "full_v7_vs_noaux": compact("hacssmv7", "hacssmv7_noaux"),
        },
        "mechanism_attribution": mechanism_map,
        "full_v7_parameter_means": {
            key: value for key, value in full_parameters.items()
            if key in {
                "rho_fast_learned_mean", "rho_medium_learned_mean",
                "static_gate_fast_mean", "static_gate_medium_mean",
                "route_fast_functional_mean", "route_medium_functional_mean",
                "action_head_fast_l2_mean", "action_head_medium_l2_mean",
                "action_head_cosine_mean", "student_teacher_memory_total_l2_mean",
                "student_memory_total_delta_l2_vs_noaux_mean",
                "predictor_delta_l2_vs_noaux_mean",
            }
        },
        "anchor_reproducibility": dict(anchor_summary),
        "convergence": {
            "absolute_window_change_median": all_convergence["absolute_window_change_median"],
            "absolute_window_change_p95": all_convergence["absolute_window_change_p95"],
            "absolute_window_change_max": all_convergence["absolute_window_change_max"],
        },
        "interpretation": [
            "V7 clears the locked SSM threshold, but its 0.30% gain over its no-auxiliary control is below the predeclared 1% requirement.",
            "V7 is positive against full V6 but remains below V6-static overall, so it is not the overall best locked-grid design.",
            "Deleting actions or using a single memory read causes large regressions; these support the action pathway and joint hierarchy, not the counterfactual objective itself.",
            "Shared action heads and removal of recovery do not hurt on average, so level-specific actions and the recovery term are not supported by this grid.",
        ],
        "limitations": [
            "All results reuse adaptive-development trajectories and the black-sentinel corruption.",
            "Completion seeds were run after a failed pilot and are descriptive, not untouched confirmation.",
            "The study measures latent prediction and has no simulator-state or executed-return endpoint.",
            "Raw PCA MSE is never pooled across environments in this package.",
        ],
    }


def publish_package(
    output_root: Path,
    outputs: Mapping[str, Any],
    *,
    primary_hashes: Mapping[str, str],
    v7_reader: ArtifactReader,
    v6_reader: ArtifactReader,
) -> Path:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite posthoc package: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"stale staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        for name in OUTPUT_FILES:
            value = outputs[name]
            path = staging / name
            if name.endswith(".csv"):
                write_csv(path, value)
            else:
                write_json(path, value)
        # Inputs must still be byte-identical before publication.
        for name, digest in primary_hashes.items():
            if sha256(V7_ROOT / name) != digest:
                raise RuntimeError(f"locked primary input changed: {name}")
        v7_reader.verify_unchanged()
        v6_reader.verify_unchanged()
        output_records = {name: file_record(staging / name) for name in OUTPUT_FILES}
        posthoc_manifest = {
            "schema_version": 1,
            "study": "HACSSM/HCRD-v7 deterministic post-hoc diagnostics",
            "generator": {
                "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "locked_v7_primary_manifest_sha256": V7_MANIFEST_SHA256,
            "locked_v6_anchor_manifest_sha256": V6_MANIFEST_SHA256,
            "locked_v7_primary_inputs": dict(primary_hashes),
            "verified_v7_artifacts": dict(sorted(v7_reader.hashes.items())),
            "verified_v6_anchor_artifacts": dict(sorted(v6_reader.hashes.items())),
            "diagnostic_outputs": output_records,
            "immutability_check_passed": True,
            "pilot_decision_unchanged": "NO_GO",
            "final_decision_unchanged": "PILOT_NO_GO_FINAL_DESCRIPTIVE",
        }
        write_json(staging / "posthoc_manifest.json", posthoc_manifest)
        manifest_hash = sha256(staging / "posthoc_manifest.json")
        sidecar = staging / "posthoc_manifest.sha256"
        with sidecar.open("x") as stream:
            stream.write(f"{manifest_hash}  posthoc_manifest.json\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-root", type=Path, default=V7_ROOT)
    parser.add_argument("--v6-root", type=Path, default=V6_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    global V7_ROOT, V6_ROOT
    V7_ROOT, V6_ROOT = args.v7_root.resolve(), args.v6_root.resolve()
    output_root = (args.output_root.resolve() if args.output_root is not None else
                   REPO_ROOT / "outputs" / f"hacssm_v7_posthoc_{V7_MANIFEST_SHA256[:12]}")
    if V7_ROOT == output_root or output_root.is_relative_to(V7_ROOT):
        raise ValueError("posthoc output must be outside the sealed V7 root")

    lock_path = V7_ROOT / ".run_hacssm_v7.lock"
    lock_stream = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("official V7 runner still holds its provenance lock") from exc

        v7_manifest = verify_manifest(
            V7_ROOT, "hacssm_v7_manifest.json", V7_MANIFEST_SHA256,
            V7_PRODUCER_COMMIT, "v7")
        v6_manifest = verify_manifest(
            V6_ROOT, "hacssm_v6_manifest.json", V6_MANIFEST_SHA256,
            V6_PRODUCER_COMMIT, "v6")
        primary_hashes = verify_v7_primary_inputs(V7_ROOT, v7_manifest)
        pilot_locked = read_json(V7_ROOT / "pilot_decision.json")
        final_locked = read_json(V7_ROOT / "decision.json")
        if (pilot_locked.get("decision") != "NO_GO"
                or pilot_locked.get("pilot_screen_passed") is not False
                or final_locked.get("decision") != "PILOT_NO_GO_FINAL_DESCRIPTIVE"
                or final_locked.get("pilot_screen_passed") is not False
                or final_locked.get("good_enough_for_overall_best_claim") is not False
                or final_locked.get("completed_runs") != 325):
            raise ValueError("frozen V7 decisions have unexpected contents")

        rows = load_csv(V7_ROOT / "per_run.csv")
        lookup = validate_rows(rows)
        locked_convergence = load_csv(V7_ROOT / "convergence.csv")
        pilot, decision = recompute_decisions(rows, locked_convergence, V7_ROOT)
        contrasts = contrast_rows(lookup)
        env_ranks, cell_ranks = rank_rows(lookup)
        phases = phase_rows(lookup)
        stages = seed_stage_rows(lookup)
        convergence = convergence_rows(locked_convergence)
        mechanism = mechanism_rows(lookup)
        v7_reader = ArtifactReader(V7_ROOT, v7_manifest)
        v6_reader = ArtifactReader(V6_ROOT, v6_manifest)
        parameter_rows, parameter_summary, objective_summary = parameter_diagnostics(
            V7_ROOT, lookup, v7_reader)
        anchor_rows, anchor_summary = anchor_reproducibility(
            V7_ROOT, V6_ROOT, lookup, v7_reader, v6_reader)
        summary = summary_document(
            contrasts, env_ranks, mechanism, convergence, parameter_summary,
            anchor_summary, pilot, decision, v7_manifest)
        outputs = {
            "summary.json": summary,
            "pairwise_contrasts.csv": contrasts,
            "env_ranks.csv": env_ranks,
            "cell_ranks.csv": cell_ranks,
            "phase_contrasts.csv": phases,
            "seed_stage.csv": stages,
            "convergence_summary.csv": convergence,
            "mechanism_attribution.csv": mechanism,
            "learned_parameters_per_run.csv": parameter_rows,
            "learned_parameters_summary.csv": parameter_summary,
            "objective_history_summary.csv": objective_summary,
            "anchor_reproducibility.csv": anchor_rows,
            "anchor_reproducibility_summary.json": anchor_summary,
        }
        published = publish_package(
            output_root, outputs, primary_hashes=primary_hashes,
            v7_reader=v7_reader, v6_reader=v6_reader)
        print(json.dumps({
            "output_root": published.relative_to(REPO_ROOT).as_posix(),
            "posthoc_manifest_sha256": sha256(published / "posthoc_manifest.json"),
            "locked_final_decision": decision["decision"],
            "best_development_grid_design": summary["best_development_grid_design"],
            "key_contrasts": summary["key_contrasts"],
            "anchor_reproducibility": anchor_summary,
        }, indent=2, sort_keys=True, allow_nan=False))
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    main()
