#!/usr/bin/env python3
"""Read-only post-hoc diagnostics for the completed HACSSM-v5 study.

The prospective analyzer and its decision remain immutable.  This script reads the
attested final tables, verifies them against the primary manifest, and writes only
separately named descriptive diagnostics.  Cross-environment summaries use paired
relative reductions because raw PCA MSE scales are environment-specific.
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


ENVIRONMENTS = (
    "dmc:reacher.hard.occ",
    "dmc:ball_in_cup.catch.occ",
    "dmc:finger.spin.occ",
    "dmc:cheetah.run.occ",
    "ogbench:cube-single.occ",
)
DESIGNS = (
    "none",
    "ssm",
    "hacsmv4",
    "hacsmv4_noaux",
    "hacsmv4_two_noaux",
    "hacssmv5_ssmcontrol",
    "hacssmv5_fixedbeta_noaux",
    "hacssmv5_noaux",
    "hacssmv5_noaction",
    "hacssmv5_static",
    "hacssmv5_single",
    "hacssmv5",
)
SEEDS = (0, 1, 2, 3, 4)
PRIMARY = "clean_mse_first_post"
PRIMARY_MANIFEST_SHA256 = "99d25180d63d5b0ebaaf85d44ee1744b0d34df4ba4fb0c0567853e5f49ab7950"
PHASE_METRICS = (
    "clean_mse_pre",
    "clean_mse_blackout_transition",
    "clean_mse_deep_blackout",
    "clean_mse_first_post",
    "clean_mse_recovery",
    "clean_mse_late_post",
    "clean_mse_all",
)
PRIMARY_INPUTS = (
    "decision.json",
    "pilot_decision.json",
    "per_run.csv",
    "grouped.csv",
    "paired_contrasts.csv",
    "convergence.csv",
    "protocol.json",
    "hacssm_v5_manifest.json",
    "hacssm_v5_manifest.sha256",
)
OUTPUTS = (
    "v5_posthoc_diagnostics.json",
    "v5_posthoc_contrasts.csv",
    "v5_posthoc_env_ranks.csv",
    "v5_posthoc_phase_contrasts.csv",
    "v5_posthoc_seed_stage.csv",
    "v5_posthoc_rate_ranges.csv",
    "v5_posthoc_convergence.csv",
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
    """Linear percentile matching NumPy's default quantile for finite sequences."""
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
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

    sidecar = (root / "hacssm_v5_manifest.sha256").read_text().split()
    if sidecar != [hashes["hacssm_v5_manifest.json"], "hacssm_v5_manifest.json"]:
        raise ValueError("primary manifest sidecar does not match the manifest")
    if hashes["hacssm_v5_manifest.json"] != PRIMARY_MANIFEST_SHA256:
        raise ValueError("primary manifest differs from the finalized V5 manifest")
    manifest = read_json(root / "hacssm_v5_manifest.json")
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("primary manifest has no output_artifacts dictionary")
    for name in ("decision.json", "pilot_decision.json", "per_run.csv", "grouped.csv",
                 "paired_contrasts.csv", "convergence.csv", "protocol.json"):
        key = f"outputs/hacssm_v5_shared/{name}"
        record = artifacts.get(key)
        if not isinstance(record, dict) or record.get("sha256") != hashes[name]:
            raise ValueError(f"{name} differs from the primary manifest")

    cloud = manifest.get("wandb_cloud_verification")
    expected_cloud = {
        "verified_finished_runs": 300,
        "verified_rollout_artifacts": 300,
        "verified_rollout_tables": 300,
        "verified_rollout_videos": 300,
    }
    if not isinstance(cloud, dict) or any(cloud.get(key) != value
                                          for key, value in expected_cloud.items()):
        raise ValueError(f"incomplete W&B cloud receipt: {cloud!r}")
    if (manifest.get("completed_runs") != 300
            or manifest.get("expected_runs") != 300
            or manifest.get("all_requested_runs_completed") is not True):
        raise ValueError("primary manifest does not attest the complete 300-run grid")
    return hashes, manifest


def validate_rows(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, int], Mapping[str, str]]:
    required = {"run", "env", "design", "seed", PRIMARY, *PHASE_METRICS,
                "tau_fast", "tau_slow", "tau_fast_min", "tau_fast_max",
                "tau_medium_min", "tau_medium_max"}
    if not required.issubset(rows[0]):
        raise ValueError(f"per_run.csv is missing columns: {sorted(required - set(rows[0]))}")
    lookup: dict[tuple[str, str, int], Mapping[str, str]] = {}
    numeric = (*PHASE_METRICS, "last_visible_mse_first_post")
    for index, row in enumerate(rows):
        env, design = row["env"], row["design"]
        seed = int(row["seed"])
        key = (env, design, seed)
        if key in lookup:
            raise ValueError(f"duplicate grid cell: {key}")
        lookup[key] = row
        for metric in numeric:
            value = finite(row.get(metric), f"row {index} {metric}")
            if value <= 0.0:
                raise ValueError(f"row {index} {metric} must be positive")
    expected = {(env, design, seed) for env in ENVIRONMENTS
                for design in DESIGNS for seed in SEEDS}
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"grid mismatch; missing={missing[:3]}, extra={extra[:3]}")
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
    pairs = [
        (
            finite(lookup[(env, candidate, seed)][metric], f"{env}/{candidate}/{seed}/{metric}"),
            finite(lookup[(env, reference, seed)][metric], f"{env}/{reference}/{seed}/{metric}"),
        )
        for env in envs for seed in seeds
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
    pairs = [("hacsmv4_two_noaux", design) for design in DESIGNS
             if design != "hacsmv4_two_noaux"]
    pairs += [
        ("hacssmv5", "ssm"),
        ("hacssmv5", "hacsmv4_noaux"),
        ("hacssmv5", "hacsmv4_two_noaux"),
        ("hacssmv5", "hacssmv5_noaux"),
        ("hacssmv5", "hacssmv5_noaction"),
        ("hacssmv5", "hacssmv5_static"),
        ("hacssmv5", "hacssmv5_single"),
        ("hacssmv5_noaux", "ssm"),
        ("hacssmv5_noaux", "hacssmv5_fixedbeta_noaux"),
        ("hacssmv5_noaux", "hacsmv4_two_noaux"),
        ("hacssmv5_fixedbeta_noaux", "ssm"),
        ("hacssmv5_fixedbeta_noaux", "hacsmv4_two_noaux"),
    ]
    result: list[dict[str, Any]] = []
    for candidate, reference in pairs:
        for env in (*ENVIRONMENTS, "__overall__"):
            summary = paired_summary(
                lookup, candidate, reference,
                envs=ENVIRONMENTS if env == "__overall__" else (env,),
            )
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
        ordered = sorted(DESIGNS, key=lambda design: (mean(values[design]), design))
        for rank, design in enumerate(ordered, 1):
            observed = values[design]
            center = mean(observed)
            variance = mean((value - center) ** 2 for value in observed)
            result.append({
                "env": env,
                "rank": rank,
                "design": design,
                "n_seeds": len(observed),
                "clean_mse_first_post_mean": center,
                "clean_mse_first_post_population_std": math.sqrt(variance),
            })
    return result


def phase_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for metric in PHASE_METRICS:
        summary = paired_summary(lookup, "hacssmv5", "ssm", metric=metric)
        result.append({
            "metric": metric,
            "candidate": "hacssmv5",
            "reference": "ssm",
            **summary,
            "environment_mean_wins": environment_wins(
                lookup, "hacssmv5", "ssm", metric=metric),
        })
    return result


def seed_stage_rows(lookup: Mapping[tuple[str, str, int], Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for stage, seeds in (("pilot", (0, 1, 2)), ("completion", (3, 4)), ("all", SEEDS)):
        summary = paired_summary(lookup, "hacssmv5", "ssm", seeds=seeds)
        result.append({
            "stage": stage,
            "seeds": ",".join(str(seed) for seed in seeds),
            "candidate": "hacssmv5",
            "reference": "ssm",
            **summary,
            "environment_mean_wins": environment_wins(
                lookup, "hacssmv5", "ssm", seeds=seeds),
        })
    return result


def rate_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    fields = ("tau_fast", "tau_slow", "tau_fast_min", "tau_fast_max",
              "tau_medium_min", "tau_medium_max")
    result = []
    for design in DESIGNS:
        if not design.startswith("hacssmv5"):
            continue
        selected = [row for row in rows if row["design"] == design]
        out: dict[str, Any] = {"design": design, "n_runs": len(selected)}
        for field in fields:
            observed = [finite(row[field], f"{design}/{field}") for row in selected if row[field] != ""]
            out[f"{field}_mean"] = mean(observed) if observed else ""
            out[f"{field}_min_across_runs"] = min(observed) if observed else ""
            out[f"{field}_max_across_runs"] = max(observed) if observed else ""
        result.append(out)
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
    rates: Sequence[Mapping[str, Any]],
    convergence: Sequence[Mapping[str, Any]],
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
    phase_map = {row["metric"]: {
        "mean_paired_relative_reduction": row["mean_paired_relative_reduction"],
        "paired_wins": row["paired_wins"],
        "environment_mean_wins": row["environment_mean_wins"],
    } for row in phases}
    all_convergence = next(row for row in convergence if row["design"] == "__all__")
    full_rates = next(row for row in rates if row["design"] == "hacssmv5")
    cloud = manifest["wandb_cloud_verification"]
    return {
        "schema_version": 1,
        "scope": "descriptive post-hoc diagnostics; cannot change the prospective decision",
        "locked_decision": "PILOT_NO_GO_FINAL_DESCRIPTIVE",
        "completed_runs": 300,
        "wandb_cloud_verification": cloud,
        "primary_input_sha256": dict(hashes),
        "best_development_grid_design": {
            "design": best,
            **rank_summary[best],
            "vs_ssm": compact(best, "ssm"),
            "qualification": (
                "Best by environment-rank envelope on the locked development grid; "
                "not an untouched-test or paper-level claim."
            ),
        },
        "key_primary_contrasts": {
            "full_v5_vs_ssm": compact("hacssmv5", "ssm"),
            "full_v5_vs_v4_two_noaux": compact(
                "hacssmv5", "hacsmv4_two_noaux"),
            "v4_two_noaux_vs_ssm": compact("hacsmv4_two_noaux", "ssm"),
            "full_v5_vs_noaux_auxiliary_effect": compact(
                "hacssmv5", "hacssmv5_noaux"),
            "v5_noaux_vs_fixedbeta_rate_learning_effect": compact(
                "hacssmv5_noaux", "hacssmv5_fixedbeta_noaux"),
            "fixedbeta_v5_vs_v4_two_parameterization_effect": compact(
                "hacssmv5_fixedbeta_noaux", "hacsmv4_two_noaux"),
            "full_v5_vs_noaction": compact("hacssmv5", "hacssmv5_noaction"),
            "full_v5_vs_static": compact("hacssmv5", "hacssmv5_static"),
            "full_v5_vs_single": compact("hacssmv5", "hacssmv5_single"),
        },
        "full_v5_vs_ssm_by_prediction_phase": phase_map,
        "full_v5_vs_ssm_by_seed_stage": {
            row["stage"]: {
                "mean_paired_relative_reduction": row["mean_paired_relative_reduction"],
                "paired_wins": row["paired_wins"],
                "n_pairs": row["n_pairs"],
                "environment_mean_wins": row["environment_mean_wins"],
            } for row in stages
        },
        "full_v5_learned_rate_ranges": full_rates,
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
            "Removing V4's tau=32 state is the strongest tested structural change: the two-level fixed-rate V4 bridge is the locked-grid rank winner.",
            "The V5 boundary auxiliary is counterproductive because full V5 loses to its no-auxiliary counterpart overall and in every environment mean.",
            "Learning V5 gains does not rescue the spectral design; the frozen per-channel spectrum already trails the fixed scalar-rate V4 bridge, so the result cannot be blamed only on gain optimization.",
            "Action transport and the joint two-state read remain essential, while dynamic correction provides a smaller mixed improvement.",
            "The completed seeds agree with the failed pilot direction; convergence passes, so the negative result is not explained by the predeclared final-window test.",
        ],
        "limitations": [
            "All contrasts are descriptive and use the same adaptive-development trajectories and exact black-sentinel corruption.",
            "No downstream control return, simulator-state prediction, or untouched corruption family is measured.",
            "The fixed-beta versus V4-two contrast changes the complete rate parameterization, not one isolated scalar; interpret it as a bundled parameterization effect.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/hacssm_v5_shared"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root
    hashes_before, manifest = verify_primary_inputs(root)
    decision = read_json(root / "decision.json")
    pilot = read_json(root / "pilot_decision.json")
    if (decision.get("decision") != "PILOT_NO_GO_FINAL_DESCRIPTIVE"
            or decision.get("completed_runs") != 300
            or pilot.get("decision") != "NO_GO"
            or pilot.get("pilot_screen_passed") is not False):
        raise ValueError("locked pilot/final decisions do not match the completed V5 study")

    rows = load_csv(root / "per_run.csv")
    lookup = validate_rows(rows)
    contrasts = contrast_rows(lookup)
    ranks = rank_rows(lookup)
    phases = phase_rows(lookup)
    stages = seed_stage_rows(lookup)
    rates = rate_rows(rows)
    convergence = convergence_rows(load_csv(root / "convergence.csv"))
    diagnostics = summarize(
        contrasts, ranks, phases, stages, rates, convergence, manifest, hashes_before)

    root.mkdir(parents=True, exist_ok=True)
    atomic_csv(root / "v5_posthoc_contrasts.csv", contrasts)
    atomic_csv(root / "v5_posthoc_env_ranks.csv", ranks)
    atomic_csv(root / "v5_posthoc_phase_contrasts.csv", phases)
    atomic_csv(root / "v5_posthoc_seed_stage.csv", stages)
    atomic_csv(root / "v5_posthoc_rate_ranges.csv", rates)
    atomic_csv(root / "v5_posthoc_convergence.csv", convergence)
    atomic_json(root / "v5_posthoc_diagnostics.json", diagnostics)

    hashes_after = {name: sha256(root / name) for name in PRIMARY_INPUTS}
    if hashes_after != hashes_before:
        raise RuntimeError("a locked primary input changed while diagnostics were generated")
    output_records = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
        for name in OUTPUTS
    }
    diagnostics_manifest = {
        "schema_version": 1,
        "study": "HACSSM-v5 read-only post-hoc diagnostics",
        "generator": {
            "path": "scripts/analyze_hacssm_v5_diagnostics.py",
            "sha256": sha256(Path(__file__).resolve()),
        },
        "locked_primary_inputs": {
            name: {"bytes": (root / name).stat().st_size, "sha256": digest}
            for name, digest in hashes_before.items()
        },
        "diagnostic_outputs": output_records,
        "immutability_check_passed": True,
        "decision_unchanged": decision["decision"],
    }
    atomic_json(root / "v5_posthoc_diagnostics_manifest.json", diagnostics_manifest)
    print(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
