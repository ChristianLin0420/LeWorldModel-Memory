#!/usr/bin/env python3
"""Deterministic post-hoc diagnostics for the sealed HACSSM-v8 study.

The primary manifest, decisions, and identity receipts are immutable inputs.  This
script verifies their frozen hashes, recomputes descriptive diagnostics, and publishes
one atomic sibling package.  It never writes below ``outputs/hacssm_v8_shared`` and
never pools raw PCA MSE across environments.
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
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROOT = REPO_ROOT / "outputs" / "hacssm_v8_shared"
PRIMARY_MANIFEST_SHA256 = "e92b9c0f5435ae8252f1f6f0a7e5995ac3647de136c889ff79b9b3aa5adb841c"
FINAL_DECISION_SHA256 = "28388137708d27211bbb8ee3f1a4f7842e4b329e3c383185720ccac25cc4645b"
PILOT_DECISION_SHA256 = "1c5e47386cb75325f22e1a20d19190615774eb0e483b9673d67f78d8ed5a1cb7"
EQUIVALENCE_RECEIPTS_SHA256 = "28e4c6dceef671196fe992cbf0b84df273ec1e0fc61ad86b90f363a3b5603b7c"
PRODUCER_GIT_COMMIT = "c75d628b635d6e6ad158753aa9bbe34a0534bac6"
SEALED_V7_MANIFEST_SHA256 = "98eda8abec229753381bed5f22c70317428242470cc6f40b6a3f9c16d0f55c11"

ENVIRONMENTS = (
    "dmc:reacher.hard.occ",
    "dmc:ball_in_cup.catch.occ",
    "dmc:finger.spin.occ",
    "dmc:cheetah.run.occ",
    "ogbench:cube-single.occ",
)
DESIGNS = (
    "ssm",
    "hacssmv6",
    "hacssmv6_static",
    "hacssmv7_noaux",
    "hacssmv7_sharedaction",
    "hacssmv7_norecovery",
    "hacssmv8_dynamic",
    "hacssmv8_static",
    "hacssmv8_levelaction",
    "hacssmv8_redundant",
    "hacssmv8_noaction",
    "hacssmv8_single",
    "hacssmv8",
)
V8_DESIGNS = tuple(design for design in DESIGNS if design.startswith("hacssmv8"))
CANDIDATE = "hacssmv8"
REDUNDANT = "hacssmv8_redundant"
LEVEL_ACTION = "hacssmv8_levelaction"
V7_LEADER = "hacssmv7_sharedaction"
DYNAMIC = "hacssmv8_dynamic"
STATIC = "hacssmv8_static"
PERFORMANCE_ENVELOPE_DESIGNS = tuple(
    design for design in DESIGNS if design not in {CANDIDATE, REDUNDANT}
)
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

BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 8_008
BOOTSTRAP_CONTRACT = {
    "schema_version": 1,
    "algorithm": "crossed_environment_seed_percentile_bootstrap",
    "draws": BOOTSTRAP_DRAWS,
    "seed": BOOTSTRAP_SEED,
    "rng": "numpy.random.Generator(numpy.random.PCG64)",
    "resampling": (
        "independently sample E environment indices and S optimizer-seed indices "
        "with replacement; evaluate the E-by-S Cartesian product; equal-weight mean"
    ),
    "estimand": "mean paired relative reduction (reference-candidate)/reference",
    "quantiles": {"method": "linear", "reported": [0.05, 0.025, 0.975, 0.95]},
}
BOOTSTRAP_CONTRACT_SHA256 = "b387010d207f96e9e6777c272ec51629764bfc190cbfd3f323fe6196c38f969e"

PRIMARY_INPUTS = (
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
    "equivalence_receipts.json",
)
OUTPUT_FILES = (
    "summary.json",
    "ssm_ranking.csv",
    "environment_ranks.csv",
    "environment_envelopes.csv",
    "phase_contrasts.csv",
    "stage_stability.csv",
    "learned_v8_parameters.csv",
    "convergence_by_design.csv",
    "bootstrap_intervals.csv",
    "receipt_summary.json",
)
ARTIFACT_SECTIONS = (
    "protocol",
    "source_artifacts",
    "feature_artifacts",
    "eval_rollout_artifacts",
    "log_artifacts",
    "output_artifacts",
)


def reject_non_rfc_json(token: str) -> None:
    raise ValueError(f"non-RFC JSON constant {token}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=reject_non_rfc_json)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} is not finite: {value!r}")
    return result


def population_std(values: Sequence[float]) -> float:
    center = mean(values)
    return math.sqrt(mean((value - center) ** 2 for value in values))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _safe_repo_path(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe manifest artifact path: {relative!r}")
    return REPO_ROOT.joinpath(*pure.parts)


def verify_artifact(path: Path, record: Mapping[str, Any], context: str) -> None:
    kind = record.get("kind", "file")
    if kind == "symlink":
        if not path.is_symlink() or os.readlink(path) != record.get("target"):
            raise ValueError(f"symlink differs from manifest: {context}")
        if not path.exists():
            raise ValueError(f"broken manifest symlink: {context}")
        return
    if kind != "file" or path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular manifest artifact: {context}")
    if path.stat().st_size != record.get("bytes"):
        raise ValueError(f"artifact size differs from manifest: {context}")
    digest = sha256_file(path)
    if digest != record.get("sha256"):
        raise ValueError(f"artifact hash differs from manifest: {context}")


def verify_manifest_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for section in ARTIFACT_SECTIONS:
        records = manifest.get(section)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"manifest lacks artifact section {section}")
        files = symlinks = 0
        for relative, record in records.items():
            if not isinstance(relative, str) or not isinstance(record, dict):
                raise ValueError(f"malformed record in {section}")
            verify_artifact(_safe_repo_path(relative), record, f"{section}:{relative}")
            if record.get("kind", "file") == "symlink":
                symlinks += 1
            else:
                files += 1
        counts[section] = {"records": len(records), "files": files, "symlinks": symlinks}
    return counts


def _manifest_record(manifest: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    relative = path.absolute().relative_to(REPO_ROOT).as_posix()
    records = manifest.get("output_artifacts")
    record = records.get(relative) if isinstance(records, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"primary input absent from manifest: {relative}")
    return record


def verify_manifest_pair(root: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = root / "hacssm_v8_manifest.json"
    sidecar_path = root / "hacssm_v8_manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"missing V8 manifest pair under {root}")
    observed = sha256_file(manifest_path)
    if observed != expected_sha256:
        raise ValueError(f"primary manifest hash {observed} != frozen {expected_sha256}")
    if sidecar_path.read_text() != f"{observed}  hacssm_v8_manifest.json\n":
        raise ValueError("primary manifest sidecar mismatch")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("primary manifest is not an object")
    return manifest


def _require_bound_file(
    root: Path, manifest: Mapping[str, Any], name: str, expected_sha256: str
) -> Any:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing regular frozen input: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{name} hash {digest} != frozen {expected_sha256}")
    verify_artifact(path, _manifest_record(manifest, path), name)
    return read_json(path)


def verify_primary(
    root: Path = PRIMARY_ROOT, *, full_artifact_audit: bool = True
) -> dict[str, Any]:
    root = root.resolve()
    manifest = verify_manifest_pair(root, PRIMARY_MANIFEST_SHA256)
    decision = _require_bound_file(root, manifest, "decision.json", FINAL_DECISION_SHA256)
    pilot = _require_bound_file(root, manifest, "pilot_decision.json", PILOT_DECISION_SHA256)
    receipts = _require_bound_file(
        root, manifest, "equivalence_receipts.json", EQUIVALENCE_RECEIPTS_SHA256
    )

    input_hashes: dict[str, str] = {}
    for name in PRIMARY_INPUTS:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing regular primary input: {path}")
        verify_artifact(path, _manifest_record(manifest, path), name)
        input_hashes[name] = sha256_file(path)

    if canonical_json_sha256(BOOTSTRAP_CONTRACT) != BOOTSTRAP_CONTRACT_SHA256:
        raise RuntimeError("source bootstrap contract hash is inconsistent")
    for label, locked in (("pilot", pilot), ("final", decision)):
        observed = locked.get("observed", {})
        if (
            observed.get("bootstrap_contract") != BOOTSTRAP_CONTRACT
            or observed.get("bootstrap_contract_sha256") != BOOTSTRAP_CONTRACT_SHA256
        ):
            raise ValueError(f"{label} bootstrap contract differs from frozen contract")
    if manifest.get("pilot_decision") != pilot or manifest.get("final_decision") != decision:
        raise ValueError("manifest-embedded decision differs from bound decision file")
    if (
        pilot.get("decision") != "NO_GO"
        or pilot.get("pilot_screen_passed") is not False
        or decision.get("decision") != "PILOT_NO_GO_FINAL_DESCRIPTIVE"
        or decision.get("pilot_screen_passed") is not False
        or decision.get("completed_runs") != 325
        or decision.get("good_enough_for_overall_best_claim") is not False
        or decision.get("good_enough_for_compact_noninferiority_claim") is not False
    ):
        raise ValueError("bound V8 decisions have unexpected contents")
    if (
        manifest.get("completed_runs") != 325
        or manifest.get("expected_runs") != 325
        or manifest.get("all_requested_runs_completed") is not True
        or manifest.get("producer_git_clean") is not True
        or manifest.get("producer_git_commit") != PRODUCER_GIT_COMMIT
        or manifest.get("pilot_screen_passed") is not False
    ):
        raise ValueError("manifest does not attest the sealed complete V8 grid")

    receipt_record = manifest.get("equivalence_receipts")
    if (
        not isinstance(receipt_record, dict)
        or receipt_record.get("sha256") != EQUIVALENCE_RECEIPTS_SHA256
        or receipts.get("validated_jobs") != 325
        or receipts.get("sealed_v7_manifest_sha256") != SEALED_V7_MANIFEST_SHA256
        or receipts.get("counts")
        != {
            "sealed_anchor_identities": 150,
            "v8_levelaction_v7_noaux_identities": 25,
            "v8_redundant_head_receipts": 25,
        }
    ):
        raise ValueError("identity receipt header differs from frozen contract")
    if (
        len(receipts.get("sealed_anchor_identities", [])) != 150
        or len(receipts.get("v8_levelaction_v7_noaux_identities", [])) != 25
        or len(receipts.get("v8_redundant_head_receipts", [])) != 25
    ):
        raise ValueError("identity receipt arrays have unexpected lengths")
    exact_fields = {
        "sealed_anchor_identities": (
            "history_exact", "model_tensors_exact", "primary_metrics_exact", "rollout_exact"
        ),
        "v8_levelaction_v7_noaux_identities": (
            "base_history_exact", "student_tensors_exact", "primary_metrics_exact", "rollout_exact"
        ),
        "v8_redundant_head_receipts": ("head_blocks_exact",),
    }
    for group, fields in exact_fields.items():
        if any(item.get(field) is not True for item in receipts[group] for field in fields):
            raise ValueError(f"non-exact identity receipt in {group}")

    expected_cloud = {
        "verified_finished_runs": 325,
        "verified_complete_epoch_histories": 325,
        "verified_rollout_artifacts": 325,
        "verified_rollout_tables": 325,
        "verified_rollout_videos": 325,
    }
    cloud = manifest.get("wandb_cloud_verification")
    wandb_runs = manifest.get("wandb_runs")
    if (
        not isinstance(cloud, dict)
        or any(cloud.get(key) != value for key, value in expected_cloud.items())
        or not isinstance(wandb_runs, dict)
        or len(wandb_runs) != 325
        or any(
            run.get("state") != "finished"
            or run.get("mode") != "online"
            or run.get("study") != "hacssm-v8"
            for run in wandb_runs.values()
        )
    ):
        raise ValueError("W&B receipt is incomplete")

    audit = verify_manifest_artifacts(manifest) if full_artifact_audit else {}
    return {
        "manifest": manifest,
        "decision": decision,
        "pilot": pilot,
        "receipts": receipts,
        "input_hashes": input_hashes,
        "artifact_audit": audit,
    }


def run_name(env: str, design: str, seed: int) -> str:
    return f"lewm-{env}-{design}-s{seed}"


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    environments: Sequence[str] = ENVIRONMENTS,
    designs: Sequence[str] = DESIGNS,
    seeds: Sequence[int] = SEEDS,
) -> dict[tuple[str, str, int], Mapping[str, str]]:
    required = {"run", "env", "design", "seed", "trainable_parameters", *PHASE_METRICS}
    if not rows or not required.issubset(rows[0]):
        missing = required - set(rows[0] if rows else {})
        raise ValueError(f"per_run.csv missing fields: {sorted(missing)}")
    lookup: dict[tuple[str, str, int], Mapping[str, str]] = {}
    for index, row in enumerate(rows):
        env, design, seed = row["env"], row["design"], int(row["seed"])
        key = (env, design, seed)
        if key in lookup:
            raise ValueError(f"duplicate per-run cell: {key}")
        if row["run"] != run_name(env, design, seed):
            raise ValueError(f"unexpected run name at row {index}")
        for metric in PHASE_METRICS:
            if finite(row[metric], f"row {index}/{metric}") <= 0.0:
                raise ValueError(f"non-positive MSE at row {index}/{metric}")
        finite(row["trainable_parameters"], f"row {index}/trainable_parameters")
        lookup[key] = row
    expected = {
        (env, design, seed)
        for env in environments
        for design in designs
        for seed in seeds
    }
    if set(lookup) != expected:
        missing, extra = expected - set(lookup), set(lookup) - expected
        raise ValueError(
            f"per_run.csv is not the exact grid: missing={sorted(missing)[:3]}, "
            f"extra={sorted(extra)[:3]}"
        )
    return lookup


def reduction_matrix(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    candidate: str,
    reference: str,
    *,
    metric: str = PRIMARY,
    environments: Sequence[str] = ENVIRONMENTS,
    seeds: Sequence[int] = SEEDS,
) -> np.ndarray:
    matrix = np.empty((len(environments), len(seeds)), dtype=np.float64)
    for env_index, env in enumerate(environments):
        for seed_index, seed in enumerate(seeds):
            candidate_value = finite(lookup[(env, candidate, seed)][metric], metric)
            reference_value = finite(lookup[(env, reference, seed)][metric], metric)
            if reference_value <= 0.0:
                raise ValueError(f"non-positive reference: {env}/{reference}/{seed}/{metric}")
            matrix[env_index, seed_index] = (
                reference_value - candidate_value
            ) / reference_value
    if not np.isfinite(matrix).all():
        raise ValueError(f"non-finite contrast: {candidate} vs {reference}/{metric}")
    return matrix


def endpoint_envelope_matrix(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    *,
    metric: str = PRIMARY,
    environments: Sequence[str] = ENVIRONMENTS,
    seeds: Sequence[int] = SEEDS,
) -> np.ndarray:
    matrix = np.empty((len(environments), len(seeds)), dtype=np.float64)
    for env_index, env in enumerate(environments):
        for seed_index, seed in enumerate(seeds):
            candidate = finite(lookup[(env, CANDIDATE, seed)][metric], metric)
            endpoint = min(
                finite(lookup[(env, DYNAMIC, seed)][metric], metric),
                finite(lookup[(env, STATIC, seed)][metric], metric),
            )
            matrix[env_index, seed_index] = (endpoint - candidate) / endpoint
    return matrix


def matrix_summary(matrix: np.ndarray, environment_wins: int) -> dict[str, Any]:
    return {
        "n_pairs": int(matrix.size),
        "mean_paired_relative_reduction": float(matrix.mean()),
        "median_paired_relative_reduction": float(np.median(matrix)),
        "paired_wins": int((matrix > 0.0).sum()),
        "paired_ties": int((matrix == 0.0).sum()),
        "environment_mean_wins": int(environment_wins),
    }


def paired_summary(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    candidate: str,
    reference: str,
    *,
    metric: str = PRIMARY,
    environments: Sequence[str] = ENVIRONMENTS,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    matrix = reduction_matrix(
        lookup, candidate, reference, metric=metric,
        environments=environments, seeds=seeds,
    )
    env_wins = sum(
        mean(finite(lookup[(env, candidate, seed)][metric], metric) for seed in seeds)
        < mean(finite(lookup[(env, reference, seed)][metric], metric) for seed in seeds)
        for env in environments
    )
    return matrix_summary(matrix, env_wins)


def endpoint_summary(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    *, seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    matrix = endpoint_envelope_matrix(lookup, seeds=seeds)
    env_wins = 0
    for env in ENVIRONMENTS:
        candidate = mean(finite(lookup[(env, CANDIDATE, seed)][PRIMARY], PRIMARY) for seed in seeds)
        endpoint = min(
            mean(finite(lookup[(env, DYNAMIC, seed)][PRIMARY], PRIMARY) for seed in seeds),
            mean(finite(lookup[(env, STATIC, seed)][PRIMARY], PRIMARY) for seed in seeds),
        )
        env_wins += candidate < endpoint
    return matrix_summary(matrix, env_wins)


def ssm_ranking_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    *, designs: Sequence[str] = DESIGNS,
) -> list[dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for design in designs:
        summaries[design] = paired_summary(lookup, design, "ssm")
    centers = {
        design: float(summary["mean_paired_relative_reduction"])
        for design, summary in summaries.items()
    }
    result = []
    for design in sorted(designs, key=lambda item: (-centers[item], item)):
        result.append({
            "rank": 1 + sum(other > centers[design] for other in centers.values()),
            "design": design,
            **summaries[design],
        })
    return result


def environment_rank_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    *,
    environments: Sequence[str] = ENVIRONMENTS,
    designs: Sequence[str] = DESIGNS,
    seeds: Sequence[int] = SEEDS,
) -> list[dict[str, Any]]:
    result = []
    for env in environments:
        values = {
            design: [finite(lookup[(env, design, seed)][PRIMARY], PRIMARY) for seed in seeds]
            for design in designs
        }
        centers = {design: mean(items) for design, items in values.items()}
        for design in sorted(designs, key=lambda item: (centers[item], item)):
            center = centers[design]
            result.append({
                "env": env,
                "rank": 1 + sum(other < center for other in centers.values()),
                "design": design,
                "n_seeds": len(seeds),
                "clean_mse_first_post_mean": center,
                "clean_mse_first_post_population_std": population_std(values[design]),
            })
    return result


def environment_envelope_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    env_ranks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rank_lookup = {
        (str(row["env"]), str(row["design"])): int(row["rank"])
        for row in env_ranks
    }
    result = []
    for env in ENVIRONMENTS:
        centers = {
            design: mean(
                finite(lookup[(env, design, seed)][PRIMARY], PRIMARY) for seed in SEEDS
            )
            for design in DESIGNS
        }
        performance_winner = min(
            PERFORMANCE_ENVELOPE_DESIGNS, key=lambda design: (centers[design], design)
        )
        endpoint_winner = min((DYNAMIC, STATIC), key=lambda design: (centers[design], design))
        hold = mean(
            finite(
                lookup[(env, CANDIDATE, seed)]["last_visible_mse_first_post"],
                "last_visible_mse_first_post",
            )
            for seed in SEEDS
        )
        result.append({
            "env": env,
            "candidate_mse": centers[CANDIDATE],
            "candidate_rank_all_13": rank_lookup[(env, CANDIDATE)],
            "performance_winner": performance_winner,
            "performance_winner_mse": centers[performance_winner],
            "candidate_beats_performance_envelope": centers[CANDIDATE] < centers[performance_winner],
            "endpoint_environment_winner": endpoint_winner,
            "endpoint_environment_winner_mse": centers[endpoint_winner],
            "candidate_beats_endpoint_environment_mean": centers[CANDIDATE] < centers[endpoint_winner],
            "last_visible_hold_mse": hold,
            "candidate_beats_last_visible_hold": centers[CANDIDATE] < hold,
        })
    return result


def phase_contrast_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]]
) -> list[dict[str, Any]]:
    # Only paired relative effects cross environment boundaries; no raw MSE aggregate is emitted.
    result = []
    for reference in DESIGNS:
        if reference == CANDIDATE:
            continue
        for metric in PHASE_METRICS:
            result.append({
                "metric": metric,
                "candidate": CANDIDATE,
                "reference": reference,
                **paired_summary(lookup, CANDIDATE, reference, metric=metric),
            })
    return result


def stage_stability_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]]
) -> list[dict[str, Any]]:
    comparisons = [
        (CANDIDATE, reference, "direct")
        for reference in DESIGNS if reference != CANDIDATE
    ] + [(REDUNDANT, LEVEL_ACTION, "action_tying")]
    result = []
    for candidate, reference, kind in comparisons:
        summaries = {
            stage: paired_summary(lookup, candidate, reference, seeds=seeds)
            for stage, seeds in (
                ("pilot", PILOT_SEEDS),
                ("completion", COMPLETION_SEEDS),
                ("all", SEEDS),
            )
        }
        pilot = float(summaries["pilot"]["mean_paired_relative_reduction"])
        completion = float(summaries["completion"]["mean_paired_relative_reduction"])
        result.append({
            "comparison_kind": kind,
            "candidate": candidate,
            "reference": reference,
            "pilot_mean_paired_relative_reduction": pilot,
            "pilot_wins": summaries["pilot"]["paired_wins"],
            "pilot_environment_wins": summaries["pilot"]["environment_mean_wins"],
            "completion_mean_paired_relative_reduction": completion,
            "completion_wins": summaries["completion"]["paired_wins"],
            "completion_environment_wins": summaries["completion"]["environment_mean_wins"],
            "all_mean_paired_relative_reduction": summaries["all"]["mean_paired_relative_reduction"],
            "all_wins": summaries["all"]["paired_wins"],
            "all_environment_wins": summaries["all"]["environment_mean_wins"],
            "completion_minus_pilot": completion - pilot,
            "same_nonzero_sign": pilot * completion > 0.0,
        })
    endpoint_stages = {
        stage: endpoint_summary(lookup, seeds=seeds)
        for stage, seeds in (
            ("pilot", PILOT_SEEDS),
            ("completion", COMPLETION_SEEDS),
            ("all", SEEDS),
        )
    }
    pilot = float(endpoint_stages["pilot"]["mean_paired_relative_reduction"])
    completion = float(endpoint_stages["completion"]["mean_paired_relative_reduction"])
    result.append({
        "comparison_kind": "per_cell_endpoint_envelope",
        "candidate": CANDIDATE,
        "reference": "min(hacssmv8_dynamic,hacssmv8_static)_per_cell",
        "pilot_mean_paired_relative_reduction": pilot,
        "pilot_wins": endpoint_stages["pilot"]["paired_wins"],
        "pilot_environment_wins": endpoint_stages["pilot"]["environment_mean_wins"],
        "completion_mean_paired_relative_reduction": completion,
        "completion_wins": endpoint_stages["completion"]["paired_wins"],
        "completion_environment_wins": endpoint_stages["completion"]["environment_mean_wins"],
        "all_mean_paired_relative_reduction": endpoint_stages["all"]["mean_paired_relative_reduction"],
        "all_wins": endpoint_stages["all"]["paired_wins"],
        "all_environment_wins": endpoint_stages["all"]["environment_mean_wins"],
        "completion_minus_pilot": completion - pilot,
        "same_nonzero_sign": pilot * completion > 0.0,
    })
    return result


def convergence_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    required = {"env", "design", "seed", "relative_improvement"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("convergence.csv has an invalid schema")
    expected = {(env, design, seed) for env in ENVIRONMENTS for design in DESIGNS for seed in SEEDS}
    observed: set[tuple[str, str, int]] = set()
    by_design: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["env"], row["design"], int(row["seed"]))
        if key in observed:
            raise ValueError(f"duplicate convergence cell: {key}")
        observed.add(key)
        by_design[row["design"]].append(
            abs(finite(row["relative_improvement"], f"convergence row {index}"))
        )
    if observed != expected:
        raise ValueError("convergence.csv is not the exact 325-cell grid")
    result = []
    for design in (*DESIGNS, "__all__"):
        values = (
            [value for group in by_design.values() for value in group]
            if design == "__all__" else by_design[design]
        )
        result.append({
            "design": design,
            "n_runs": len(values),
            "absolute_window_change_median": median(values),
            "absolute_window_change_p95": float(np.quantile(values, 0.95, method="linear")),
            "absolute_window_change_max": max(values),
        })
    return result


def crossed_bootstrap(
    matrix: np.ndarray,
    *, draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("crossed bootstrap requires a finite non-empty 2D matrix")
    rng = np.random.Generator(np.random.PCG64(seed))
    env_indices = rng.integers(0, matrix.shape[0], size=(draws, matrix.shape[0]))
    seed_indices = rng.integers(0, matrix.shape[1], size=(draws, matrix.shape[1]))
    sampled = matrix[
        env_indices[:, :, np.newaxis], seed_indices[:, np.newaxis, :]
    ].mean(axis=(1, 2))
    q025, q05, q95, q975 = np.quantile(
        sampled, (0.025, 0.05, 0.95, 0.975), method="linear"
    )
    return {
        "point_mean_paired_relative_reduction": float(matrix.mean()),
        "ci90_low": float(q05),
        "ci90_high": float(q95),
        "ci95_low": float(q025),
        "ci95_high": float(q975),
    }


def bootstrap_rows(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]]
) -> list[dict[str, Any]]:
    comparisons = (
        (CANDIDATE, "ssm", "direct"),
        (CANDIDATE, V7_LEADER, "v7_leader_noninferiority"),
        (CANDIDATE, REDUNDANT, "compact_redundant_equivalence"),
        (REDUNDANT, LEVEL_ACTION, "action_tying"),
    )
    result = []
    for stage, seeds in (("pilot", PILOT_SEEDS), ("final", SEEDS)):
        for candidate, reference, kind in comparisons:
            matrix = reduction_matrix(lookup, candidate, reference, seeds=seeds)
            result.append({
                "stage": stage,
                "comparison_kind": kind,
                "candidate": candidate,
                "reference": reference,
                "n_environments": matrix.shape[0],
                "n_seeds": matrix.shape[1],
                "draws": BOOTSTRAP_DRAWS,
                "contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
                **crossed_bootstrap(matrix),
            })
        endpoint = endpoint_envelope_matrix(lookup, seeds=seeds)
        result.append({
            "stage": stage,
            "comparison_kind": "per_cell_endpoint_envelope",
            "candidate": CANDIDATE,
            "reference": "min(hacssmv8_dynamic,hacssmv8_static)_per_cell",
            "n_environments": endpoint.shape[0],
            "n_seeds": endpoint.shape[1],
            "draws": BOOTSTRAP_DRAWS,
            "contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
            **crossed_bootstrap(endpoint),
        })
    return result


def _close(left: Any, right: Any, context: str, tolerance: float = 2e-15) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"locked arithmetic mismatch {context}: {left} != {right}")


def validate_locked_arithmetic(
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    convergence: Sequence[Mapping[str, Any]],
    bootstraps: Sequence[Mapping[str, Any]],
    pilot: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    for locked, seeds, label in ((pilot, PILOT_SEEDS, "pilot"), (decision, SEEDS, "final")):
        observed = locked["observed"]
        for reference, receipt in observed["overall_contrasts"].items():
            summary = paired_summary(lookup, CANDIDATE, reference, seeds=seeds)
            for field in ("mean_paired_relative_reduction", "paired_wins", "paired_ties", "n_pairs"):
                if field == "mean_paired_relative_reduction":
                    _close(summary[field], receipt[field], f"{label}/{reference}/{field}")
                elif int(summary[field]) != int(receipt[field]):
                    raise ValueError(f"locked arithmetic mismatch {label}/{reference}/{field}")
            if summary["environment_mean_wins"] != observed["environment_mean_wins"][reference]:
                raise ValueError(f"locked environment wins mismatch {label}/{reference}")
        endpoint = endpoint_summary(lookup, seeds=seeds)
        endpoint_receipt = observed["mechanism"]["endpoint_envelope"]
        _close(
            endpoint["mean_paired_relative_reduction"],
            endpoint_receipt["mean_paired_relative_reduction"],
            f"{label}/endpoint",
        )
        if any(
            int(endpoint[field]) != int(endpoint_receipt[field])
            for field in ("n_pairs", "paired_wins", "paired_ties", "environment_mean_wins")
        ):
            raise ValueError(f"locked endpoint count mismatch {label}")

        for kind, receipt_key in (
            ("compact_redundant_equivalence", "compact_redundant_equivalence"),
            ("v7_leader_noninferiority", "v7_leader_noninferiority"),
        ):
            row = next(
                item for item in bootstraps
                if item["stage"] == label and item["comparison_kind"] == kind
            )
            receipt = observed["mechanism"][receipt_key]["bootstrap"]
            _close(row["point_mean_paired_relative_reduction"], receipt["point_mean_paired_relative_reduction"], f"{label}/{kind}/point")
            for output_field, receipt_field, index in (
                ("ci90_low", "ci90", 0), ("ci90_high", "ci90", 1),
                ("ci95_low", "ci95", 0), ("ci95_high", "ci95", 1),
            ):
                _close(row[output_field], receipt[receipt_field][index], f"{label}/{kind}/{output_field}")

    all_convergence = next(row for row in convergence if row["design"] == "__all__")
    locked_convergence = decision["observed"]["convergence_absolute"]
    for output_field, receipt_field in (
        ("absolute_window_change_median", "median"),
        ("absolute_window_change_p95", "p95"),
        ("absolute_window_change_max", "max"),
    ):
        _close(all_convergence[output_field], locked_convergence[receipt_field], f"convergence/{output_field}")


def _memory_state(checkpoint: Mapping[str, Any], run: str) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{run}: missing model_state_dict")
    prefix = "mem_hacssmv8."
    memory = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    required = {"gate_bias", "route_logits", "shrink_logits", "W_a.weight"}
    if not required.issubset(memory) or any(not isinstance(value, torch.Tensor) for value in memory.values()):
        raise ValueError(f"{run}: invalid compact V8 memory state")
    return memory


def learned_parameter_rows(
    root: Path,
    lookup: Mapping[tuple[str, str, int], Mapping[str, str]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for env in ENVIRONMENTS:
        for seed in SEEDS:
            run = run_name(env, CANDIDATE, seed)
            path = root / run / "model.pt"
            verify_artifact(path, _manifest_record(manifest, path), f"{run}/model.pt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError(f"{run}: invalid checkpoint")
            memory = _memory_state(checkpoint, run)
            shrink = torch.sigmoid(memory["shrink_logits"].detach().to(torch.float64))
            gate = torch.sigmoid(memory["gate_bias"].detach().to(torch.float64))
            route = torch.softmax(memory["route_logits"].detach().to(torch.float64), dim=0)
            action = memory["W_a.weight"].detach().to(torch.float64)
            if shrink.shape != (2,) or gate.shape != (2,) or route.shape != (2,) or action.shape != (256, 6):
                raise ValueError(f"{run}: compact V8 tensor shape mismatch")
            action_norm = float(torch.linalg.vector_norm(action).item())
            source = lookup[(env, CANDIDATE, seed)]
            for field, derived in (
                ("rho_fast", float(shrink[0])),
                ("rho_medium", float(shrink[1])),
                ("action_head_shared_norm", action_norm),
            ):
                if not math.isclose(finite(source[field], f"{run}/{field}"), derived, rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError(f"{run}: learned parameter receipt mismatch for {field}")
            result.append({
                "run": run,
                "env": env,
                "seed": seed,
                "trainable_parameters": int(float(source["trainable_parameters"])),
                "rho_fast": float(shrink[0]),
                "rho_medium": float(shrink[1]),
                "static_gate_fast": float(gate[0]),
                "static_gate_medium": float(gate[1]),
                "route_fast": float(route[0]),
                "route_medium": float(route[1]),
                "action_head_l2": action_norm,
                "action_head_delta_l2": float(torch.linalg.vector_norm(action[:128]).item()),
                "action_head_velocity_l2": float(torch.linalg.vector_norm(action[128:]).item()),
            })
    return result


def parameter_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = tuple(key for key in rows[0] if key not in {"run", "env", "seed"})
    result: dict[str, Any] = {"design": CANDIDATE, "n_runs": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        result[field] = {
            "mean": mean(values),
            "population_std": population_std(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def receipt_summary(
    verified: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, receipts = verified["manifest"], verified["receipts"]
    wandb_runs = manifest["wandb_runs"]
    return {
        "schema_version": 1,
        "locked_primary_manifest_sha256": PRIMARY_MANIFEST_SHA256,
        "locked_final_decision_sha256": FINAL_DECISION_SHA256,
        "locked_pilot_decision_sha256": PILOT_DECISION_SHA256,
        "locked_equivalence_receipts_sha256": EQUIVALENCE_RECEIPTS_SHA256,
        "sealed_v7_manifest_sha256": receipts["sealed_v7_manifest_sha256"],
        "validated_jobs": receipts["validated_jobs"],
        "identity_counts": receipts["counts"],
        "identity_all_exact": {
            "sealed_anchors": True,
            "v8_levelaction_v7_noaux": True,
            "v8_redundant_heads": True,
        },
        "wandb_cloud_verification": manifest["wandb_cloud_verification"],
        "wandb_run_records": len(wandb_runs),
        "wandb_states": dict(sorted(Counter(run["state"] for run in wandb_runs.values()).items())),
        "wandb_modes": dict(sorted(Counter(run["mode"] for run in wandb_runs.values()).items())),
        "artifact_audit": verified["artifact_audit"],
    }


def summary_document(
    ranking: Sequence[Mapping[str, Any]],
    env_ranks: Sequence[Mapping[str, Any]],
    envelopes: Sequence[Mapping[str, Any]],
    stages: Sequence[Mapping[str, Any]],
    convergence: Sequence[Mapping[str, Any]],
    parameters: Sequence[Mapping[str, Any]],
    bootstraps: Sequence[Mapping[str, Any]],
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    decision, pilot = verified["decision"], verified["pilot"]
    candidate_rank = next(row for row in ranking if row["design"] == CANDIDATE)
    leader = min(ranking, key=lambda row: (int(row["rank"]), str(row["design"])))
    final_endpoint = next(
        row for row in stages if row["comparison_kind"] == "per_cell_endpoint_envelope"
    )
    all_convergence = next(row for row in convergence if row["design"] == "__all__")
    key_bootstraps = {
        str(row["comparison_kind"]): {
            key: row[key] for key in (
                "candidate", "reference", "point_mean_paired_relative_reduction",
                "ci90_low", "ci90_high", "ci95_low", "ci95_high",
            )
        }
        for row in bootstraps if row["stage"] == "final"
    }
    return {
        "schema_version": 1,
        "scope": "descriptive sealed-record post-hoc diagnostics; cannot alter the pilot or final decision",
        "locked_record": {
            "primary_manifest_sha256": PRIMARY_MANIFEST_SHA256,
            "producer_git_commit": verified["manifest"]["producer_git_commit"],
            "completed_runs": verified["manifest"]["completed_runs"],
            "pilot_decision": pilot["decision"],
            "final_decision": decision["decision"],
            "pilot_screen_passed": decision["pilot_screen_passed"],
            "good_enough_for_overall_best_claim": decision["good_enough_for_overall_best_claim"],
            "good_enough_for_compact_noninferiority_claim": decision["good_enough_for_compact_noninferiority_claim"],
        },
        "ssm_relative_ranking": {
            "leader": {key: leader[key] for key in ("rank", "design", "mean_paired_relative_reduction")},
            "candidate": {key: candidate_rank[key] for key in (
                "rank", "design", "mean_paired_relative_reduction", "paired_wins", "environment_mean_wins"
            )},
        },
        "candidate_environment_ranks": {
            row["env"]: row["rank"]
            for row in env_ranks if row["design"] == CANDIDATE
        },
        "envelope_summary": {
            "performance_environment_wins": sum(bool(row["candidate_beats_performance_envelope"]) for row in envelopes),
            "endpoint_environment_wins": sum(bool(row["candidate_beats_endpoint_environment_mean"]) for row in envelopes),
            "last_visible_hold_environment_wins": sum(bool(row["candidate_beats_last_visible_hold"]) for row in envelopes),
            "endpoint_per_cell": {
                "mean_paired_relative_reduction": final_endpoint["all_mean_paired_relative_reduction"],
                "paired_wins": final_endpoint["all_wins"],
                "environment_mean_wins": final_endpoint["all_environment_wins"],
            },
        },
        "bootstrap_contract": BOOTSTRAP_CONTRACT,
        "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        "key_final_bootstrap_intervals": key_bootstraps,
        "learned_compact_v8_parameters": parameter_summary(parameters),
        "convergence_all_designs": {
            key: all_convergence[key] for key in (
                "n_runs", "absolute_window_change_median",
                "absolute_window_change_p95", "absolute_window_change_max",
            )
        },
        "failed_final_criteria": sorted(
            key for key, value in decision["criteria"].items() if not value
        ),
        "interpretation": [
            "Compact V8 is second by paired reduction versus SSM, behind V7 shared-action.",
            "Action transport and the joint read remain large positive structural mechanisms.",
            "Action tying, compact/redundant equivalence, and dominance of the retrained endpoint envelope do not clear their locked gates.",
            "Pilot/completion sign changes in close contrasts make them descriptive rather than confirmation.",
        ],
        "limitations": [
            "V8 was selected adaptively after V1-V7 on the same tasks and trajectories.",
            "The package reports latent prediction rather than simulator state or executed return.",
            "Raw PCA MSE is reported only within an environment and is never pooled across environments.",
        ],
    }


def publish_package(
    output_root: Path,
    outputs: Mapping[str, Any],
    *,
    primary_input_hashes: Mapping[str, str],
    reverify: Callable[[], None],
) -> Path:
    output_root = output_root.resolve()
    if set(outputs) != set(OUTPUT_FILES):
        raise ValueError("diagnostic output set differs from the frozen package schema")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite post-hoc package: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"stale staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        for name in OUTPUT_FILES:
            value = outputs[name]
            if name.endswith(".csv"):
                write_csv_new(staging / name, value)
            else:
                write_json_new(staging / name, value)
        reverify()
        output_records = {name: file_record(staging / name) for name in OUTPUT_FILES}
        manifest = {
            "schema_version": 1,
            "study": "HACSSM-v8 sealed deterministic post-hoc diagnostics",
            "scope": "descriptive only; primary decisions remain immutable",
            "generator": {
                "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "locked_primary_manifest_sha256": PRIMARY_MANIFEST_SHA256,
            "locked_final_decision_sha256": FINAL_DECISION_SHA256,
            "locked_pilot_decision_sha256": PILOT_DECISION_SHA256,
            "locked_equivalence_receipts_sha256": EQUIVALENCE_RECEIPTS_SHA256,
            "locked_primary_inputs": dict(sorted(primary_input_hashes.items())),
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
            "diagnostic_outputs": output_records,
            "immutability_check_passed": True,
            "pilot_decision_unchanged": "NO_GO",
            "final_decision_unchanged": "PILOT_NO_GO_FINAL_DESCRIPTIVE",
            "raw_pca_mse_pooled_across_environments": False,
        }
        write_json_new(staging / "posthoc_manifest.json", manifest)
        manifest_hash = sha256_file(staging / "posthoc_manifest.json")
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
        if output_root.exists():
            raise FileExistsError(f"post-hoc package appeared during publication: {output_root}")
        os.rename(staging, output_root)
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
    parser.add_argument("--root", type=Path, default=PRIMARY_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    output_root = (
        args.output_root.resolve() if args.output_root is not None
        else root.parent / f"hacssm_v8_posthoc_{PRIMARY_MANIFEST_SHA256[:12]}"
    )
    if output_root.parent != root.parent or output_root == root:
        raise ValueError("post-hoc output must be a sibling of the sealed V8 root")

    lock_path = root / ".run_hacssm_v8.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing provenance lock: {lock_path}")
    lock_stream = lock_path.open("r")
    try:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("official V8 runner still holds its provenance lock") from exc

        verified = verify_primary(root, full_artifact_audit=True)
        rows = load_csv(root / "per_run.csv")
        lookup = validate_rows(rows)
        ranking = ssm_ranking_rows(lookup)
        env_ranks = environment_rank_rows(lookup)
        envelopes = environment_envelope_rows(lookup, env_ranks)
        phases = phase_contrast_rows(lookup)
        stages = stage_stability_rows(lookup)
        convergence = convergence_rows(load_csv(root / "convergence.csv"))
        bootstraps = bootstrap_rows(lookup)
        validate_locked_arithmetic(
            lookup, convergence, bootstraps, verified["pilot"], verified["decision"]
        )
        learned = learned_parameter_rows(root, lookup, verified["manifest"])
        receipts = receipt_summary(verified)
        summary = summary_document(
            ranking, env_ranks, envelopes, stages, convergence, learned,
            bootstraps, verified,
        )
        outputs = {
            "summary.json": summary,
            "ssm_ranking.csv": ranking,
            "environment_ranks.csv": env_ranks,
            "environment_envelopes.csv": envelopes,
            "phase_contrasts.csv": phases,
            "stage_stability.csv": stages,
            "learned_v8_parameters.csv": learned,
            "convergence_by_design.csv": convergence,
            "bootstrap_intervals.csv": bootstraps,
            "receipt_summary.json": receipts,
        }

        def reverify() -> None:
            repeated = verify_primary(root, full_artifact_audit=True)
            if repeated["input_hashes"] != verified["input_hashes"]:
                raise RuntimeError("locked primary input changed during diagnostics")

        published = publish_package(
            output_root,
            outputs,
            primary_input_hashes=verified["input_hashes"],
            reverify=reverify,
        )
        print(json.dumps({
            "output_root": published.relative_to(REPO_ROOT).as_posix(),
            "posthoc_manifest_sha256": sha256_file(published / "posthoc_manifest.json"),
            "summary_sha256": sha256_file(published / "summary.json"),
            "locked_final_decision": verified["decision"]["decision"],
            "candidate_ssm_rank": next(row["rank"] for row in ranking if row["design"] == CANDIDATE),
        }, indent=2, sort_keys=True, allow_nan=False))
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    main()
