#!/usr/bin/env python3
"""Read-only post-hoc diagnostics for the completed HACSSM-v6 study.

The prospective analyzer, pilot decision, final decision, and primary manifest are
locked inputs.  This script verifies those inputs and the checkpoint/rollout files
that it reads, then writes only ``v6_posthoc_*`` descriptive artifacts.  Raw PCA
MSE is never pooled across environments; cross-environment summaries use matched
environment/seed relative reductions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    "hacssmv5_noaux",
    "hacssmv6_noaux",
    "hacssmv6_aux_noaction",
    "hacssmv6_uniform",
    "hacssmv6_sourcegrad",
    "hacssmv6_fastonly",
    "hacssmv6_mediumonly",
    "hacssmv6_noaction",
    "hacssmv6_static",
    "hacssmv6_single",
    "hacssmv6",
)
V6_DESIGNS = tuple(design for design in DESIGNS if design.startswith("hacssmv6"))
SEEDS = (0, 1, 2, 3, 4)
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
PRIMARY_MANIFEST_SHA256 = "915484f69ec78b4dead79a25e5fa096667a4c76034e88338bd46476f6fbc495c"
PRIMARY_INPUTS = (
    "decision.json",
    "pilot_decision.json",
    "per_run.csv",
    "grouped.csv",
    "paired_contrasts.csv",
    "convergence.csv",
    "protocol.json",
    "hacssm_v6_manifest.json",
    "hacssm_v6_manifest.sha256",
)
OUTPUTS = (
    "v6_posthoc_diagnostics.json",
    "v6_posthoc_contrasts.csv",
    "v6_posthoc_env_ranks.csv",
    "v6_posthoc_phase_contrasts.csv",
    "v6_posthoc_seed_stage.csv",
    "v6_posthoc_identity_controls.csv",
    "v6_posthoc_aux_action.csv",
    "v6_posthoc_convergence.csv",
)


def reject_non_rfc_json(token: str) -> None:
    raise ValueError(f"non-RFC JSON constant {token}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=reject_non_rfc_json)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
        raise ValueError("percentile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output: {temporary}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent columns in {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output: {temporary}")
    try:
        with temporary.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def verify_primary_inputs(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    paths = {name: root / name for name in PRIMARY_INPUTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing primary inputs: {missing}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    sidecar = (root / "hacssm_v6_manifest.sha256").read_text().split()
    if sidecar != [hashes["hacssm_v6_manifest.json"], "hacssm_v6_manifest.json"]:
        raise ValueError("primary manifest sidecar does not match the manifest")
    if hashes["hacssm_v6_manifest.json"] != PRIMARY_MANIFEST_SHA256:
        raise ValueError("primary manifest differs from the finalized V6 manifest")

    manifest = read_json(root / "hacssm_v6_manifest.json")
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("primary manifest has no output_artifacts dictionary")
    for name in PRIMARY_INPUTS[:7]:
        key = f"outputs/hacssm_v6_shared/{name}"
        record = artifacts.get(key)
        if not isinstance(record, dict) or record.get("sha256") != hashes[name]:
            raise ValueError(f"{name} differs from the primary manifest")

    expected_cloud = {
        "verified_complete_epoch_histories": 325,
        "verified_finished_runs": 325,
        "verified_rollout_artifacts": 325,
        "verified_rollout_tables": 325,
        "verified_rollout_videos": 325,
    }
    cloud = manifest.get("wandb_cloud_verification")
    if not isinstance(cloud, dict) or any(cloud.get(key) != value
                                          for key, value in expected_cloud.items()):
        raise ValueError(f"incomplete W&B cloud receipt: {cloud!r}")
    if (manifest.get("completed_runs") != 325
            or manifest.get("expected_runs") != 325
            or manifest.get("all_requested_runs_completed") is not True
            or manifest.get("producer_git_clean") is not True
            or manifest.get("producer_git_commit") !=
            "5ae7de8e31780a3892ebc08f250532fa5661e313"):
        raise ValueError("primary manifest does not attest the frozen complete V6 grid")
    return hashes, manifest


def validate_rows(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, int], Mapping[str, str]]:
    required = {
        "run", "env", "design", "seed", PRIMARY, *PHASE_METRICS,
        *PREDICTIVE_IDENTITY_METRICS, "val_hier_loss", "val_hier_loss_fast",
        "val_hier_loss_medium", "val_hier_loss_h1", "val_hier_loss_h2",
        "val_hier_loss_h4", "val_hier_loss_h8",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"per_run.csv is missing columns: {sorted(required - set(rows[0]))}")
    lookup: dict[tuple[str, str, int], Mapping[str, str]] = {}
    for index, row in enumerate(rows):
        env, design, seed = row["env"], row["design"], int(row["seed"])
        key = (env, design, seed)
        if key in lookup:
            raise ValueError(f"duplicate grid cell: {key}")
        lookup[key] = row
        for metric in PREDICTIVE_IDENTITY_METRICS:
            value = finite(row[metric], f"row {index} {metric}")
            if value <= 0.0:
                raise ValueError(f"row {index} {metric} must be positive")
    expected = {(env, design, seed) for env in ENVIRONMENTS
                for design in DESIGNS for seed in SEEDS}
    if set(lookup) != expected:
        raise ValueError("per_run.csv does not contain the exact 325-cell grid")
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
    reductions = [(reference_value - candidate_value) / reference_value
                  for candidate_value, reference_value in pairs]
    return {
        "n_pairs": len(pairs),
        "candidate_mean_mse": mean(value[0] for value in pairs),
        "reference_mean_mse": mean(value[1] for value in pairs),
        "mean_paired_relative_reduction": mean(reductions),
        "paired_wins": sum(candidate_value < reference_value
                           for candidate_value, reference_value in pairs),
        "paired_ties": sum(candidate_value == reference_value
                           for candidate_value, reference_value in pairs),
    }


def environment_wins(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    candidate: str,
    reference: str,
    *,
    metric: str = PRIMARY,
    seeds: Iterable[int] = SEEDS,
) -> int:
    selected_seeds = tuple(seeds)
    return sum(
        mean(finite(lookup[(env, candidate, seed)][metric], metric)
             for seed in selected_seeds)
        < mean(finite(lookup[(env, reference, seed)][metric], metric)
               for seed in selected_seeds)
        for env in ENVIRONMENTS
    )


def contrast_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in DESIGNS:
        for reference in ("ssm", "hacsmv4_two_noaux"):
            for env in (*ENVIRONMENTS, "__overall__"):
                summary = paired_summary(
                    lookup, candidate, reference,
                    envs=ENVIRONMENTS if env == "__overall__" else (env,),
                )
                if env == "__overall__":
                    summary["candidate_mean_mse"] = ""
                    summary["reference_mean_mse"] = ""
                result.append({
                    "candidate": candidate,
                    "reference": reference,
                    "env": env,
                    **summary,
                    "environment_mean_wins": (
                        environment_wins(lookup, candidate, reference)
                        if env == "__overall__" else ""
                    ),
                })
    return result


def rank_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for env in ENVIRONMENTS:
        values = {
            design: [finite(lookup[(env, design, seed)][PRIMARY], PRIMARY) for seed in SEEDS]
            for design in DESIGNS
        }
        centers = {design: mean(observed) for design, observed in values.items()}
        ordered = sorted(DESIGNS, key=lambda design: (centers[design], design))
        for design in ordered:
            observed, center = values[design], centers[design]
            variance = mean((value - center) ** 2 for value in observed)
            result.append({
                "env": env,
                "rank": 1 + sum(value < center for value in centers.values()),
                "design": design,
                "n_seeds": len(observed),
                "clean_mse_first_post_mean": center,
                "clean_mse_first_post_population_std": math.sqrt(variance),
            })
    return result


def phase_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for reference in ("ssm", "hacsmv4_two_noaux", "hacssmv6_noaux", "hacssmv6_static"):
        for metric in PHASE_METRICS:
            summary = paired_summary(lookup, "hacssmv6", reference, metric=metric)
            result.append({
                "metric": metric,
                "candidate": "hacssmv6",
                "reference": reference,
                **summary,
                "environment_mean_wins": environment_wins(
                    lookup, "hacssmv6", reference, metric=metric),
            })
    return result


def seed_stage_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for stage, seeds in (("pilot", (0, 1, 2)), ("completion", (3, 4)), ("all", SEEDS)):
        for reference in ("ssm", "hacsmv4_two_noaux", "hacssmv6_noaux", "hacssmv6_static"):
            summary = paired_summary(lookup, "hacssmv6", reference, seeds=seeds)
            result.append({
                "stage": stage,
                "seeds": ",".join(str(seed) for seed in seeds),
                "candidate": "hacssmv6",
                "reference": reference,
                **summary,
                "environment_mean_wins": environment_wins(
                    lookup, "hacssmv6", reference, seeds=seeds),
            })
    return result


def convergence_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    expected = {(env, design, seed) for env in ENVIRONMENTS
                for design in DESIGNS for seed in SEEDS}
    observed_keys = set()
    by_design: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["env"], row["design"], int(row["seed"]))
        if key in observed_keys:
            raise ValueError(f"duplicate convergence cell {key}")
        observed_keys.add(key)
        by_design[row["design"]].append(abs(finite(
            row["relative_improvement"], f"convergence row {index}")))
    if observed_keys != expected:
        raise ValueError("convergence.csv does not contain the exact final grid")
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


def canonical_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize the inference-identical V4/V6 memory namespace."""
    result = {}
    for key, value in state.items():
        normalized = key.replace("mem_hacsmv4.", "mem_hacssmv6.", 1)
        if normalized in result:
            raise ValueError(f"canonical state collision: {normalized}")
        result[normalized] = value
    return result


def state_distances(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    if set(candidate) != set(reference):
        raise ValueError("checkpoint state keys differ")
    action_keys = [key for key in candidate if key.endswith(".W_a.weight")]
    if len(action_keys) != 1:
        raise ValueError(f"expected one action-map tensor, got {action_keys}")
    action_key = action_keys[0]
    exact = True
    max_abs = 0.0
    action_delta_sq = 0.0
    nonaction_delta_sq = 0.0
    for key in sorted(candidate):
        left, right = candidate[key], reference[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"checkpoint tensor schema differs at {key}")
        equal = torch.equal(left, right)
        exact = exact and equal
        if left.dtype.is_floating_point:
            delta = left.detach().to(torch.float64) - right.detach().to(torch.float64)
            max_abs = max(max_abs, float(delta.abs().max().item()))
            square = float(torch.sum(delta * delta).item())
            if key == action_key:
                action_delta_sq += square
            else:
                nonaction_delta_sq += square
        elif not equal:
            max_abs = math.inf
    action = candidate[action_key].detach().to(torch.float64)
    ref_action = reference[action_key].detach().to(torch.float64)
    action_norm = float(torch.linalg.vector_norm(action).item())
    reference_norm = float(torch.linalg.vector_norm(ref_action).item())
    split = action.shape[0] // 2
    if action.ndim != 2 or action.shape[0] % 2:
        raise ValueError(f"action map cannot be split into two levels: {tuple(action.shape)}")
    action_delta = action - ref_action
    cosine: float | str = ""
    if action_norm > 0.0 and reference_norm > 0.0:
        cosine = float(torch.sum(action * ref_action).item() / (action_norm * reference_norm))
    return {
        "state_exact": exact,
        "state_max_abs_difference": max_abs,
        "action_weight_l2": action_norm,
        "action_weight_delta_l2_vs_noaux": math.sqrt(action_delta_sq),
        "action_weight_delta_relative_vs_noaux": (
            math.sqrt(action_delta_sq) / reference_norm if reference_norm > 0.0 else ""),
        "action_weight_fast_l2": float(torch.linalg.vector_norm(action[:split]).item()),
        "action_weight_medium_l2": float(torch.linalg.vector_norm(action[split:]).item()),
        "action_weight_fast_delta_l2_vs_noaux": float(
            torch.linalg.vector_norm(action_delta[:split]).item()),
        "action_weight_medium_delta_l2_vs_noaux": float(
            torch.linalg.vector_norm(action_delta[split:]).item()),
        "action_weight_cosine_vs_noaux": cosine,
        "nonaction_parameter_delta_l2_vs_noaux": math.sqrt(nonaction_delta_sq),
    }


class ArtifactReader:
    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.hashes: dict[str, str] = {}

    def verify(self, path: Path) -> None:
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        if relative in self.hashes:
            return
        record = self.manifest["output_artifacts"].get(relative)
        if not isinstance(record, dict):
            raise ValueError(f"artifact is absent from primary manifest: {relative}")
        digest = sha256(path)
        if record.get("sha256") != digest:
            raise ValueError(f"artifact differs from primary manifest: {relative}")
        self.hashes[relative] = digest

    def checkpoint(self, run: str) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        path = self.root / run / "model.pt"
        self.verify(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict")
        history = checkpoint.get("history")
        if not isinstance(state, dict) or not isinstance(history, list) or len(history) != 200:
            raise ValueError(f"invalid checkpoint schema: {path}")
        return canonical_state(state), history

    def rollout(self, run: str) -> dict[str, np.ndarray]:
        path = self.root / run / "eval_rollout.npz"
        self.verify(path)
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def verify_unchanged(self) -> None:
        for relative, digest in self.hashes.items():
            if sha256(REPO_ROOT / relative) != digest:
                raise RuntimeError(f"a locked artifact changed during diagnostics: {relative}")


def rollout_difference(candidate: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]) -> tuple[bool, float]:
    if set(candidate) != set(reference):
        return False, math.inf
    exact = True
    maximum = 0.0
    for key in sorted(candidate):
        left, right = candidate[key], reference[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, math.inf
        exact = exact and bool(np.array_equal(left, right))
        if np.issubdtype(left.dtype, np.number):
            maximum = max(maximum, float(np.max(np.abs(
                left.astype(np.float64) - right.astype(np.float64)))))
    return exact, maximum


def identity_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]], reader: ArtifactReader
) -> list[dict[str, Any]]:
    pairs = (
        ("hacsmv4_two_noaux", "hacssmv6_noaux"),
        ("hacssmv6_noaux", "hacssmv6_aux_noaction"),
        ("hacsmv4_two_noaux", "hacssmv6_aux_noaction"),
    )
    result = []
    for reference, candidate in pairs:
        metric_exact = checkpoint_exact = rollout_exact = 0
        max_metric = max_state = max_rollout = 0.0
        for env in ENVIRONMENTS:
            for seed in SEEDS:
                reference_row, candidate_row = lookup[(env, reference, seed)], lookup[(env, candidate, seed)]
                differences = [abs(finite(candidate_row[field], field) - finite(reference_row[field], field))
                               for field in PREDICTIVE_IDENTITY_METRICS]
                max_metric = max(max_metric, max(differences))
                metric_exact += int(all(value == 0.0 for value in differences))
                reference_state, _ = reader.checkpoint(reference_row["run"])
                candidate_state, _ = reader.checkpoint(candidate_row["run"])
                state = state_distances(candidate_state, reference_state)
                checkpoint_exact += int(state["state_exact"])
                max_state = max(max_state, state["state_max_abs_difference"])
                exact, difference = rollout_difference(
                    reader.rollout(candidate_row["run"]), reader.rollout(reference_row["run"]))
                rollout_exact += int(exact)
                max_rollout = max(max_rollout, difference)
        result.append({
            "reference": reference,
            "candidate": candidate,
            "matched_cells": 25,
            "predictive_metrics_compared": len(PREDICTIVE_IDENTITY_METRICS),
            "exact_predictive_metric_cells": metric_exact,
            "max_abs_predictive_metric_difference": max_metric,
            "exact_canonical_checkpoint_cells": checkpoint_exact,
            "max_abs_canonical_state_difference": max_state,
            "exact_rollout_array_cells": rollout_exact,
            "max_abs_rollout_array_difference": max_rollout,
        })
    return result


def aux_action_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]], reader: ArtifactReader
) -> list[dict[str, Any]]:
    run_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for env in ENVIRONMENTS:
        for seed in SEEDS:
            reference_row = lookup[(env, "hacssmv6_noaux", seed)]
            reference_state, _ = reader.checkpoint(reference_row["run"])
            for design in V6_DESIGNS:
                row = lookup[(env, design, seed)]
                state, history = reader.checkpoint(row["run"])
                distances = state_distances(state, reference_state)
                active = []
                for item in history:
                    train = item.get("train", {})
                    weight = finite(train.get("hier_loss_weight"), f"{row['run']} hier weight")
                    loss = finite(train.get("hier_loss"), f"{row['run']} hier loss")
                    if weight > 0.0:
                        active.append((weight, loss))
                run_rows[design].append({
                    **distances,
                    "active_aux_epochs": len(active),
                    "active_train_hier_loss_mean": mean(value[1] for value in active) if active else "",
                    "active_weighted_aux_term_mean": mean(value[0] * value[1] for value in active)
                    if active else 0.0,
                    "val_hier_loss": finite(row["val_hier_loss"], f"{row['run']} val_hier_loss"),
                    "val_hier_loss_fast": finite(row["val_hier_loss_fast"], "val_hier_loss_fast"),
                    "val_hier_loss_medium": finite(row["val_hier_loss_medium"], "val_hier_loss_medium"),
                    "val_hier_loss_h1": (finite(row["val_hier_loss_h1"], "val_hier_loss_h1")
                                         if row["val_hier_loss_h1"] != "" else ""),
                    "val_hier_loss_h2": (finite(row["val_hier_loss_h2"], "val_hier_loss_h2")
                                         if row["val_hier_loss_h2"] != "" else ""),
                    "val_hier_loss_h4": (finite(row["val_hier_loss_h4"], "val_hier_loss_h4")
                                         if row["val_hier_loss_h4"] != "" else ""),
                    "val_hier_loss_h8": (finite(row["val_hier_loss_h8"], "val_hier_loss_h8")
                                         if row["val_hier_loss_h8"] != "" else ""),
                })
    result = []
    numeric = (
        "action_weight_l2", "action_weight_delta_l2_vs_noaux",
        "action_weight_delta_relative_vs_noaux", "action_weight_fast_l2",
        "action_weight_medium_l2", "action_weight_fast_delta_l2_vs_noaux",
        "action_weight_medium_delta_l2_vs_noaux",
        "nonaction_parameter_delta_l2_vs_noaux", "active_aux_epochs",
        "active_weighted_aux_term_mean", "val_hier_loss", "val_hier_loss_fast",
        "val_hier_loss_medium", "val_hier_loss_h1", "val_hier_loss_h2",
        "val_hier_loss_h4", "val_hier_loss_h8",
    )
    for design in V6_DESIGNS:
        rows = run_rows[design]
        cosine = [float(row["action_weight_cosine_vs_noaux"]) for row in rows
                  if row["action_weight_cosine_vs_noaux"] != ""]
        active_loss = [float(row["active_train_hier_loss_mean"]) for row in rows
                       if row["active_train_hier_loss_mean"] != ""]
        out: dict[str, Any] = {
            "design": design,
            "n_runs": len(rows),
            "checkpoint_exact_vs_noaux_cells": sum(bool(row["state_exact"]) for row in rows),
        }
        for field in numeric:
            observed = [float(row[field]) for row in rows if row[field] != ""]
            out[f"{field}_mean"] = mean(observed) if observed else ""
            center = out[f"{field}_mean"]
            out[f"{field}_population_std"] = (
                math.sqrt(mean((value - float(center)) ** 2 for value in observed))
                if observed else ""
            )
        out["action_weight_cosine_vs_noaux_mean"] = mean(cosine) if cosine else ""
        out["active_train_hier_loss_mean"] = mean(active_loss) if active_loss else ""
        for metric in ("val_hier_loss", "val_hier_loss_fast", "val_hier_loss_medium"):
            design_values = [finite(
                lookup[(env, design, seed)][metric], f"{env}/{design}/{seed}/{metric}")
                for env in ENVIRONMENTS for seed in SEEDS]
            noaux_values = [finite(
                lookup[(env, "hacssmv6_noaux", seed)][metric],
                f"{env}/hacssmv6_noaux/{seed}/{metric}")
                for env in ENVIRONMENTS for seed in SEEDS]
            out[f"{metric}_reduction_vs_noaux"] = mean(
                (reference - candidate) / reference
                for candidate, reference in zip(design_values, noaux_values))
            out[f"{metric}_wins_vs_noaux"] = sum(
                candidate < reference
                for candidate, reference in zip(design_values, noaux_values))
            reverse_pairs = [(candidate, reference)
                             for candidate, reference in zip(noaux_values, design_values)
                             if reference > 0.0]
            out[f"noaux_{metric}_reduction_vs_design"] = (
                mean((reference - candidate) / reference
                     for candidate, reference in reverse_pairs)
                if reverse_pairs else ""
            )
            out[f"noaux_{metric}_wins_vs_design"] = sum(
                candidate < reference for candidate, reference in reverse_pairs)
            out[f"noaux_{metric}_pairs_vs_design"] = len(reverse_pairs)
        result.append(out)
    return result


def find_overall(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> Mapping[str, Any]:
    matches = [row for row in rows if row["candidate"] == candidate
               and row["reference"] == reference and row["env"] == "__overall__"]
    if len(matches) != 1:
        raise ValueError(f"expected one {candidate} vs {reference} overall row")
    return matches[0]


def summarize(
    contrasts: Sequence[Mapping[str, Any]],
    ranks: Sequence[Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
    identities: Sequence[Mapping[str, Any]],
    aux_action: Sequence[Mapping[str, Any]],
    convergence: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    pilot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    def compact(candidate: str, reference: str) -> dict[str, Any]:
        row = find_overall(contrasts, candidate, reference)
        return {
            "mean_paired_relative_reduction": row["mean_paired_relative_reduction"],
            "paired_wins": row["paired_wins"],
            "n_pairs": row["n_pairs"],
            "environment_mean_wins": row["environment_mean_wins"],
        }

    rank_summary = {
        design: {
            "environment_rank_wins": sum(
                row["rank"] == 1 for row in ranks if row["design"] == design),
            "mean_environment_rank": mean(
                float(row["rank"]) for row in ranks if row["design"] == design),
        }
        for design in DESIGNS
    }
    best = min(DESIGNS, key=lambda design: (
        -rank_summary[design]["environment_rank_wins"],
        rank_summary[design]["mean_environment_rank"], design))
    all_convergence = next(row for row in convergence if row["design"] == "__all__")
    candidate_aux = next(row for row in aux_action if row["design"] == "hacssmv6")
    zero_action_aux = next(
        row for row in aux_action if row["design"] == "hacssmv6_aux_noaction")
    phase_map = {
        reference: {
            row["metric"]: {
                "mean_paired_relative_reduction": row["mean_paired_relative_reduction"],
                "paired_wins": row["paired_wins"],
                "environment_mean_wins": row["environment_mean_wins"],
            }
            for row in phases if row["reference"] == reference
        }
        for reference in ("ssm", "hacsmv4_two_noaux", "hacssmv6_noaux", "hacssmv6_static")
    }
    stage_map = {
        stage: {
            row["reference"]: {
                "mean_paired_relative_reduction": row["mean_paired_relative_reduction"],
                "paired_wins": row["paired_wins"],
                "n_pairs": row["n_pairs"],
                "environment_mean_wins": row["environment_mean_wins"],
            }
            for row in stages if row["stage"] == stage
        }
        for stage in ("pilot", "completion", "all")
    }
    return {
        "schema_version": 1,
        "scope": "descriptive post-hoc diagnostics; cannot change the prospective pilot/final decisions",
        "frozen_prelaunch_record": {
            "producer_git_commit": manifest["producer_git_commit"],
            "producer_git_clean": manifest["producer_git_clean"],
            "primary_manifest_sha256": PRIMARY_MANIFEST_SHA256,
        },
        "locked_pilot_decision": pilot["decision"],
        "locked_final_decision": decision["decision"],
        "trigger_v7": decision["trigger_v7"],
        "failed_final_criteria": sorted(key for key, value in decision["criteria"].items() if not value),
        "completed_runs": 325,
        "wandb_cloud_verification": manifest["wandb_cloud_verification"],
        "primary_input_sha256": dict(hashes),
        "best_development_grid_design": {
            "design": best,
            **rank_summary[best],
            "vs_ssm": compact(best, "ssm"),
            "vs_v4_two_noaux": compact(best, "hacsmv4_two_noaux"),
            "qualification": (
                "Best by environment-rank wins on this locked development grid; "
                "not an untouched-test or paper-level claim."
            ),
        },
        "key_primary_contrasts": {
            "full_v6_vs_ssm": compact("hacssmv6", "ssm"),
            "full_v6_vs_v4_two_noaux": compact("hacssmv6", "hacsmv4_two_noaux"),
            "full_v6_vs_noaux": {
                "mean_paired_relative_reduction": decision["observed"]["overall_contrasts"]
                ["hacssmv6_noaux"]["mean_paired_relative_reduction"],
                "paired_wins": decision["observed"]["overall_contrasts"]
                ["hacssmv6_noaux"]["paired_wins"],
                "n_pairs": 25,
                "environment_mean_wins": decision["observed"]["environment_mean_wins"]
                ["hacssmv6_noaux"],
            },
            "full_v6_vs_static": {
                "mean_paired_relative_reduction": decision["observed"]["overall_contrasts"]
                ["hacssmv6_static"]["mean_paired_relative_reduction"],
                "paired_wins": decision["observed"]["overall_contrasts"]
                ["hacssmv6_static"]["paired_wins"],
                "n_pairs": 25,
                "environment_mean_wins": decision["observed"]["environment_mean_wins"]
                ["hacssmv6_static"],
            },
        },
        "full_v6_by_prediction_phase": phase_map,
        "full_v6_by_seed_stage": stage_map,
        "exact_identity_controls": list(identities),
        "full_v6_aux_action_diagnostics": dict(candidate_aux),
        "auxiliary_objective_checks": {
            "full_v6_vs_noaux": {
                "val_hier_loss_reduction": candidate_aux["val_hier_loss_reduction_vs_noaux"],
                "wins": candidate_aux["val_hier_loss_wins_vs_noaux"],
            },
            "noaux_real_action_vs_identical_zero_action": {
                "val_hier_loss_reduction": zero_action_aux[
                    "noaux_val_hier_loss_reduction_vs_design"],
                "wins": zero_action_aux["noaux_val_hier_loss_wins_vs_design"],
                "fast_reduction": zero_action_aux[
                    "noaux_val_hier_loss_fast_reduction_vs_design"],
                "medium_reduction": zero_action_aux[
                    "noaux_val_hier_loss_medium_reduction_vs_design"],
            },
        },
        "convergence": {
            "absolute_window_change_median": all_convergence["absolute_window_change_median"],
            "absolute_window_change_p95": all_convergence["absolute_window_change_p95"],
            "absolute_window_change_max": all_convergence["absolute_window_change_max"],
            "passes_locked_bounds": (
                all_convergence["absolute_window_change_median"] < 0.01
                and all_convergence["absolute_window_change_p95"] < 0.03
                and all_convergence["absolute_window_change_max"] < 0.05
            ),
        },
        "interpretation": [
            "The dense detached objective produces a small positive effect over the exact no-auxiliary anchor, but 0.39% misses the predeclared 1% requirement.",
            "The V4-two, V6-noaux, and auxiliary-no-action controls are tensor- and rollout-identical in every matched cell; the no-action auxiliary has exactly zero gradient and therefore contributes no optimization effect.",
            "Full V6 clears the SSM performance clauses, but static correction is stronger on three environments and beats full V6 overall, so the dynamic-gate hierarchy claim is not supported.",
            "The action path and joint read remain important because deleting actions or using a single state causes large, consistent regressions; this is mechanism evidence rather than evidence for the dense objective.",
            "Pilot and completion directions agree and convergence passes, so the failed screen is not explained by the two added seeds or the declared final-window convergence check.",
        ],
        "limitations": [
            "All contrasts reuse adaptive-development trajectories and the exact black-sentinel corruption.",
            "The study measures latent prediction only; it has no simulator-state metric or executed-control return.",
            "The strongest observed static variant was selected descriptively after opening the grid and needs a separately frozen untouched test.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/hacssm_v6_shared"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    hashes_before, manifest = verify_primary_inputs(root)
    decision = read_json(root / "decision.json")
    pilot = read_json(root / "pilot_decision.json")
    if (decision.get("decision") != "PILOT_NO_GO_FINAL_DESCRIPTIVE"
            or decision.get("completed_runs") != 325
            or decision.get("pilot_screen_passed") is not False
            or decision.get("trigger_v7") is not True
            or pilot.get("decision") != "NO_GO"
            or pilot.get("pilot_screen_passed") is not False):
        raise ValueError("locked pilot/final decisions do not match the completed V6 study")

    rows = load_csv(root / "per_run.csv")
    lookup = validate_rows(rows)
    contrasts = contrast_rows(lookup)
    ranks = rank_rows(lookup)
    phases = phase_rows(lookup)
    stages = seed_stage_rows(lookup)
    convergence = convergence_rows(load_csv(root / "convergence.csv"))
    reader = ArtifactReader(root, manifest)
    identities = identity_rows(lookup, reader)
    aux_action = aux_action_rows(lookup, reader)
    diagnostics = summarize(
        contrasts, ranks, phases, stages, identities, aux_action, convergence,
        decision, pilot, manifest, hashes_before,
    )

    atomic_csv(root / "v6_posthoc_contrasts.csv", contrasts)
    atomic_csv(root / "v6_posthoc_env_ranks.csv", ranks)
    atomic_csv(root / "v6_posthoc_phase_contrasts.csv", phases)
    atomic_csv(root / "v6_posthoc_seed_stage.csv", stages)
    atomic_csv(root / "v6_posthoc_identity_controls.csv", identities)
    atomic_csv(root / "v6_posthoc_aux_action.csv", aux_action)
    atomic_csv(root / "v6_posthoc_convergence.csv", convergence)
    atomic_json(root / "v6_posthoc_diagnostics.json", diagnostics)

    hashes_after = {name: sha256(root / name) for name in PRIMARY_INPUTS}
    if hashes_after != hashes_before:
        raise RuntimeError("a locked primary input changed while diagnostics were generated")
    reader.verify_unchanged()
    output_records = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
        for name in OUTPUTS
    }
    diagnostics_manifest = {
        "schema_version": 1,
        "study": "HACSSM-v6 read-only post-hoc diagnostics",
        "generator": {
            "path": "scripts/analyze_hacssm_v6_diagnostics.py",
            "sha256": sha256(Path(__file__).resolve()),
        },
        "locked_primary_inputs": {
            name: {"bytes": (root / name).stat().st_size, "sha256": digest}
            for name, digest in hashes_before.items()
        },
        "verified_checkpoint_and_rollout_inputs": {
            name: {"bytes": (REPO_ROOT / name).stat().st_size, "sha256": digest}
            for name, digest in sorted(reader.hashes.items())
        },
        "diagnostic_outputs": output_records,
        "immutability_check_passed": True,
        "pilot_decision_unchanged": pilot["decision"],
        "final_decision_unchanged": decision["decision"],
    }
    atomic_json(root / "v6_posthoc_diagnostics_manifest.json", diagnostics_manifest)
    print(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
