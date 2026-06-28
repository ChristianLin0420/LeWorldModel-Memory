#!/usr/bin/env python3
"""Fail-closed aggregation for the staged HACSSM-v5 shared-feature study.

Raw PCA MSE is summarized only within an environment.  Cross-environment comparisons use
paired relative reductions, preserving the five environment-specific target scales.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch


OCC_TO_CLEAN = {
    "dmc:reacher.hard.occ": "dmc:reacher.hard",
    "dmc:ball_in_cup.catch.occ": "dmc:ball_in_cup.catch",
    "dmc:finger.spin.occ": "dmc:finger.spin",
    "dmc:cheetah.run.occ": "dmc:cheetah.run",
    "ogbench:cube-single.occ": "ogbench:cube-single",
}
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
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
V5_DESIGNS = frozenset(design for design in DESIGNS if design.startswith("hacssmv5"))
HIER_DESIGNS = frozenset(
    design for design in DESIGNS if design.startswith(("hacsmv4", "hacssmv5"))
)
NO_AUX_DESIGNS = frozenset(
    {
        "hacsmv4_noaux",
        "hacsmv4_two_noaux",
        "hacssmv5_ssmcontrol",
        "hacssmv5_fixedbeta_noaux",
        "hacssmv5_noaux",
    }
)
PRIMARY = "clean_mse_first_post"
EPOCHS = 200
WINDOW = 10
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v5"
EVAL_ROLLOUT_EPISODE = 0


def reject_non_rfc_json(token: str) -> None:
    raise ValueError(f"non-RFC JSON constant {token}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=reject_non_rfc_json)


def finite(value: Any, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{context} is not a finite number: {value!r}")
    return float(value)


def finite_tree(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} is non-finite: {value!r}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError(f"inconsistent CSV columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    """Return configured base weight, schedule, and whether auxiliary gradients are active."""
    if design in V5_DESIGNS:
        return 0.05, "v5_frontload", design not in NO_AUX_DESIGNS
    if design.startswith("hacsmv4"):
        return 0.1, "fixed", design == "hacsmv4"
    return 0.0, "fixed", False


def scheduled_weight(base: float, schedule: str, epoch: int) -> float:
    if schedule == "fixed":
        return base
    if schedule != "v5_frontload" or epoch < 1:
        raise ValueError(f"invalid schedule/epoch: {schedule!r}/{epoch}")
    if epoch <= 20:
        return base
    if epoch <= 120:
        progress = (epoch - 20) / 100.0
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.0


def eval_rollout_cache(clean_env: str) -> str:
    safe = clean_env.replace(":", "_")
    return (
        f"outputs/popgym_data/{safe}_v3_proto0_n150_L32_s64_seed7777.npz"
    )


def expected_args_subset(env: str, design: str, seed: int) -> dict[str, Any]:
    base, schedule, _active = design_aux_contract(design)
    return {
        "env_id": env,
        "target_env_id": OCC_TO_CLEAN[env],
        "memory_mode": design,
        "seed": seed,
        "num_episodes": 600,
        "val_episodes": 150,
        "prototype_seed": 0,
        "mask_occluded_target_loss": True,
        "freeze_encoder": False,
        "encoder_type": "precomputed",
        "length": 32,
        "img_size": 64,
        "epochs": 200,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "num_workers": 2,
        "no_amp": False,
        "patch_size": 8,
        "embed_dim": 128,
        "encoder_layers": 6,
        "encoder_heads": 4,
        "predictor_layers": 4,
        "predictor_heads": 8,
        "predictor_norm": "none",
        "history_len": 3,
        "dropout": 0.1,
        "sigreg_lambda": 0.1,
        "sigreg_projections": 512,
        "hier_loss_weight": base,
        "hier_loss_schedule": schedule,
        "tau_fast": 3.0,
        "tau_slow": 25.0,
        "fixed_alpha": True,
        "wandb": True,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
        "wandb_mode": WANDB_MODE,
        "wandb_study": WANDB_STUDY,
        "eval_rollout_cache": eval_rollout_cache(OCC_TO_CLEAN[env]),
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
        "device": "cuda",
        "first_post_loss_weight": 0.5,
    }


def validate_history(history: Any, design: str, run_dir: Path) -> None:
    if not isinstance(history, list) or len(history) != EPOCHS:
        length = len(history) if isinstance(history, list) else None
        raise ValueError(f"{run_dir}: history length {length}, expected {EPOCHS}")
    base, schedule, aux_active = design_aux_contract(design)
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get("epoch") != epoch:
            raise ValueError(f"{run_dir}: malformed epoch {epoch}")
        for split in ("train", "val"):
            values = record.get(split)
            if not isinstance(values, dict):
                raise ValueError(f"{run_dir}: missing {split} at epoch {epoch}")
            for metric in ("loss", "pred_loss", "sigreg_loss"):
                finite(values.get(metric), f"{run_dir}:{epoch}:{split}.{metric}")
            finite_tree(values, f"{run_dir}:{epoch}:{split}")
            if design in HIER_DESIGNS:
                for metric in ("hier_loss", "hier_loss_fast", "hier_loss_medium"):
                    finite(values.get(metric), f"{run_dir}:{epoch}:{split}.{metric}")
                observed = finite(
                    values.get("hier_loss_weight"),
                    f"{run_dir}:{epoch}:{split}.hier_loss_weight",
                )
                wanted = scheduled_weight(base, schedule, epoch) if aux_active else 0.0
                # The trainer logs this scalar through a float32 loss tensor.
                if not math.isclose(observed, wanted, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(
                        f"{run_dir}:{epoch}:{split} auxiliary weight {observed}, expected {wanted}"
                    )


METRIC_FIELDS = (
    "val_pred_loss",
    "clean_mse_pre",
    "clean_mse_blackout_transition",
    "clean_mse_deep_blackout",
    "clean_mse_first_post",
    "clean_mse_recovery",
    "clean_mse_late_post",
    "clean_mse_all",
    "clean_mse_first_post_ablated",
    "clean_input_mse_first_post",
    "last_visible_mse_first_post",
    "constant_mse_first_post",
    "persistence_mse_first_post",
    "val_hier_loss",
    "val_hier_loss_fast",
    "val_hier_loss_medium",
    "val_hier_loss_slow",
    "val_hier_loss_h1",
    "val_hier_loss_h2",
    "val_hier_loss_h4",
    "val_hier_loss_h8",
    "val_hier_loss_h16",
    "tau_fast",
    "tau_slow",
    "tau_fast_min",
    "tau_fast_max",
    "tau_medium_min",
    "tau_medium_max",
)


def load_cells(root: Path, seeds: Sequence[int]):
    expected = {
        (env, design, seed): root / f"lewm-{env}-{design}-s{seed}"
        for env in OCC_TO_CLEAN
        for design in DESIGNS
        for seed in seeds
    }
    rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for (env, design, seed), run_dir in sorted(expected.items()):
        model_path = run_dir / "model.pt"
        metrics_path = run_dir / "metrics.json"
        if not model_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"missing complete checkpoint pair: {run_dir}")
        metrics = read_json(metrics_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or metrics != checkpoint.get("final_metrics"):
            raise ValueError(f"{run_dir}: metrics/checkpoint mismatch")
        finite_tree(metrics, f"{run_dir}.metrics")
        cfg = checkpoint.get("args")
        if not isinstance(cfg, dict):
            raise ValueError(f"{run_dir}: missing args dictionary")
        for key, wanted in expected_args_subset(env, design, seed).items():
            if cfg.get(key) != wanted:
                raise ValueError(f"{run_dir}: {key}={cfg.get(key)!r}, expected {wanted!r}")
        history = checkpoint.get("history")
        validate_history(history, design, run_dir)

        base, schedule, active = design_aux_contract(design)
        metadata = {
            "env": env,
            "design": design,
            "target_env": OCC_TO_CLEAN[env],
            "predictor_norm": "none",
            "first_post_loss_weight": 0.5,
            "hier_loss_weight": base,
            "hier_loss_schedule": schedule,
            "hier_loss_weight_final": scheduled_weight(base, schedule, EPOCHS),
            "hier_loss_weight_effective": (
                scheduled_weight(base, schedule, EPOCHS) if active else 0.0
            ),
            "masked_clean_blackout_loss": True,
            "primary_common_target_metric": PRIMARY,
            "external_features_fixed": True,
            "wandb_enabled": True,
            "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT,
            "wandb_mode": WANDB_MODE,
            "wandb_study": WANDB_STUDY,
            "eval_rollout_cache": eval_rollout_cache(OCC_TO_CLEAN[env]),
            "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
        }
        for key, wanted in metadata.items():
            if metrics.get(key) != wanted:
                raise ValueError(
                    f"{run_dir}: metric {key}={metrics.get(key)!r}, expected {wanted!r}"
                )
        for key in (PRIMARY, "val_pred_loss", "clean_input_mse_first_post",
                    "last_visible_mse_first_post"):
            finite(metrics.get(key), f"{run_dir}.{key}")
        if metrics["val_pred_loss"] != history[-1]["val"].get("pred_loss"):
            raise ValueError(f"{run_dir}: final validation loss differs from history")

        row: dict[str, Any] = {
            "run": run_dir.name,
            "env": env,
            "design": design,
            "seed": seed,
            "trainable_parameters": int(metrics["trainable_parameters"]),
        }
        for key in METRIC_FIELDS:
            # Some non-hierarchical controls deliberately report unavailable horizons
            # as JSON null after the trainer normalizes NaN-valued summaries.
            row[key] = (
                finite(metrics[key], f"{run_dir}.{key}")
                if key in metrics and metrics[key] is not None else ""
            )
        rows.append(row)
        previous = mean(finite(item["val"]["pred_loss"], "previous")
                        for item in history[-2 * WINDOW:-WINDOW])
        recent = mean(finite(item["val"]["pred_loss"], "recent")
                      for item in history[-WINDOW:])
        if previous <= 0.0:
            raise ValueError(f"{run_dir}: non-positive previous convergence window {previous}")
        convergence.append({
            "run": run_dir.name,
            "env": env,
            "design": design,
            "seed": seed,
            "previous_window_mean": previous,
            "recent_window_mean": recent,
            "relative_improvement": (previous - recent) / previous,
        })
    return rows, convergence


def grouped_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    numeric = [name for name in rows[0] if name not in {"run", "env", "design", "seed"}]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["env"]), str(row["design"]))].append(row)
    result = []
    for (env, design), values in sorted(groups.items()):
        out: dict[str, Any] = {"env": env, "design": design, "n_seeds": len(values)}
        for name in numeric:
            observed = [float(row[name]) for row in values if row[name] != ""]
            out[f"{name}_mean"] = mean(observed) if observed else ""
            out[f"{name}_std"] = pstdev(observed) if observed else ""
        result.append(out)
    return result


def contrast_rows(rows: Sequence[Mapping[str, Any]], candidate: str = "hacssmv5"):
    available_designs = {str(row["design"]) for row in rows}
    if candidate not in available_designs:
        raise ValueError(f"candidate {candidate!r} is absent from contrast rows")
    references = tuple(
        design for design in DESIGNS
        if design in available_designs and design != candidate
    )
    if not references:
        raise ValueError(f"candidate {candidate!r} has no reference designs")
    seeds = sorted({int(row["seed"]) for row in rows})
    lookup = {(row["env"], row["design"], int(row["seed"])): row for row in rows}
    result = []
    for reference in references:
        for env in (*OCC_TO_CLEAN, "__overall__"):
            envs = tuple(OCC_TO_CLEAN) if env == "__overall__" else (env,)
            pairs = [
                (
                    float(lookup[(current_env, candidate, seed)][PRIMARY]),
                    float(lookup[(current_env, reference, seed)][PRIMARY]),
                )
                for current_env in envs
                for seed in seeds
            ]
            reductions = [(reference_mse - candidate_mse) / reference_mse
                          for candidate_mse, reference_mse in pairs]
            result.append({
                "candidate": candidate,
                "reference": reference,
                "env": env,
                "n_pairs": len(pairs),
                "candidate_mean_mse": mean(pair[0] for pair in pairs)
                if env != "__overall__" else "",
                "reference_mean_mse": mean(pair[1] for pair in pairs)
                if env != "__overall__" else "",
                "mean_paired_relative_reduction": mean(reductions),
                "paired_wins": sum(candidate_mse < reference_mse
                                   for candidate_mse, reference_mse in pairs),
                "paired_ties": sum(candidate_mse == reference_mse
                                   for candidate_mse, reference_mse in pairs),
            })
    return result


def environment_means(rows: Sequence[Mapping[str, Any]], design: str,
                      metric: str = PRIMARY) -> dict[str, float]:
    return {
        env: mean(float(row[metric]) for row in rows
                  if row["env"] == env and row["design"] == design)
        for env in OCC_TO_CLEAN
    }


def overall_contrast(contrasts: Sequence[Mapping[str, Any]], reference: str):
    matches = [row for row in contrasts
               if row["reference"] == reference and row["env"] == "__overall__"]
    if len(matches) != 1:
        raise ValueError(f"expected one overall contrast for {reference}, got {len(matches)}")
    return matches[0]


def pilot_decision(rows, convergence, contrasts) -> dict[str, Any]:
    full = environment_means(rows, "hacssmv5")
    ssm_env = environment_means(rows, "ssm")
    v4_env = environment_means(rows, "hacsmv4_noaux")
    hold = environment_means(rows, "hacssmv5", "last_visible_mse_first_post")
    lookup = {(row["env"], row["design"], int(row["seed"])): row for row in rows}
    clean_worsening = mean(
        (float(lookup[(env, "hacssmv5", seed)]["clean_input_mse_first_post"])
         - float(lookup[(env, "ssm", seed)]["clean_input_mse_first_post"]))
        / float(lookup[(env, "ssm", seed)]["clean_input_mse_first_post"])
        for env in OCC_TO_CLEAN for seed in PILOT_SEEDS
    )
    references = (
        "ssm", "hacsmv4_noaux", "hacssmv5_noaux", "hacssmv5_noaction",
        "hacssmv5_static", "hacssmv5_single",
    )
    observed = {reference: overall_contrast(contrasts, reference) for reference in references}
    noaux_rows = [row for row in rows if row["design"] == "hacssmv5_noaux"]
    noaux_contrasts = contrast_rows(noaux_rows + [
        row for row in rows if row["design"] in {"hacssmv5_fixedbeta_noaux", "ssm"}
    ], candidate="hacssmv5_noaux")
    fixedbeta = overall_contrast(noaux_contrasts, "hacssmv5_fixedbeta_noaux")
    noaux_ssm = overall_contrast(noaux_contrasts, "ssm")
    absolute_convergence = np.abs(
        np.asarray([float(row["relative_improvement"]) for row in convergence], dtype=np.float64)
    )

    criteria = {
        "full_vs_ssm_reduction_ge_1pct":
            float(observed["ssm"]["mean_paired_relative_reduction"]) >= 0.01,
        "full_vs_ssm_wins_ge_9_of_15": int(observed["ssm"]["paired_wins"]) >= 9,
        "full_vs_ssm_env_wins_ge_3_of_5":
            sum(full[env] < ssm_env[env] for env in OCC_TO_CLEAN) >= 3,
        "full_vs_v4_noaux_reduction_ge_1pct":
            float(observed["hacsmv4_noaux"]["mean_paired_relative_reduction"]) >= 0.01,
        "full_vs_v4_noaux_wins_ge_9_of_15":
            int(observed["hacsmv4_noaux"]["paired_wins"]) >= 9,
        "full_vs_v4_noaux_env_wins_ge_3_of_5":
            sum(full[env] < v4_env[env] for env in OCC_TO_CLEAN) >= 3,
        "full_beats_hold_ge_3_of_5": sum(full[env] < hold[env] for env in OCC_TO_CLEAN) >= 3,
        "clean_input_worsening_vs_ssm_le_5pct": clean_worsening <= 0.05,
    }
    for reference in ("hacssmv5_noaux", "hacssmv5_noaction",
                      "hacssmv5_static", "hacssmv5_single"):
        label = reference.removeprefix("hacssmv5_")
        criteria[f"full_vs_{label}_positive"] = (
            float(observed[reference]["mean_paired_relative_reduction"]) > 0.0
        )
        criteria[f"full_vs_{label}_wins_ge_8_of_15"] = (
            int(observed[reference]["paired_wins"]) >= 8
        )
    criteria.update({
        "v5_noaux_vs_fixedbeta_noaux_positive":
            float(fixedbeta["mean_paired_relative_reduction"]) > 0.0,
        "v5_noaux_vs_fixedbeta_noaux_wins_ge_8_of_15": int(fixedbeta["paired_wins"]) >= 8,
        "v5_noaux_vs_ssm_positive": float(noaux_ssm["mean_paired_relative_reduction"]) > 0.0,
        "v5_noaux_vs_ssm_wins_ge_8_of_15": int(noaux_ssm["paired_wins"]) >= 8,
        "convergence_absolute_median_lt_1pct": float(np.median(absolute_convergence)) < 0.01,
        "convergence_absolute_p95_lt_3pct": float(np.quantile(absolute_convergence, 0.95)) < 0.03,
        "convergence_absolute_max_lt_5pct": float(absolute_convergence.max()) < 0.05,
    })
    pilot_screen_passed = all(criteria.values())
    return {
        "schema_version": 1,
        "phase": "pilot",
        "decision": "PILOT_PASS" if pilot_screen_passed else "NO_GO",
        "pilot_screen_passed": pilot_screen_passed,
        "criteria": criteria,
        "observed": {
            "full_overall_contrasts": {
                reference: {
                    "mean_paired_relative_reduction": row["mean_paired_relative_reduction"],
                    "paired_wins": row["paired_wins"],
                    "n_pairs": row["n_pairs"],
                }
                for reference, row in observed.items()
            },
            "full_vs_ssm_env_mean_wins": sum(full[e] < ssm_env[e] for e in OCC_TO_CLEAN),
            "full_vs_v4_noaux_env_mean_wins": sum(full[e] < v4_env[e] for e in OCC_TO_CLEAN),
            "full_hold_env_wins": sum(full[e] < hold[e] for e in OCC_TO_CLEAN),
            "clean_input_relative_worsening_vs_ssm": clean_worsening,
            "v5_noaux_vs_fixedbeta_noaux": fixedbeta,
            "v5_noaux_vs_ssm": noaux_ssm,
            "convergence_absolute_median": float(np.median(absolute_convergence)),
            "convergence_absolute_p95": float(np.quantile(absolute_convergence, 0.95)),
            "convergence_absolute_max": float(absolute_convergence.max()),
        },
        "note": (
            "Prospective deterministic pilot screen; not a hypothesis test or paper claim. "
            "All five seeds run regardless, but a failed screen remains immutable."
        ),
    }


def final_summary(rows, convergence, contrasts, *, pilot_screen_passed: bool) -> dict[str, Any]:
    full = environment_means(rows, "hacssmv5")
    design_means = {design: environment_means(rows, design) for design in DESIGNS}
    hold = environment_means(rows, "hacssmv5", "last_visible_mse_first_post")
    absolute = np.abs(np.asarray(
        [float(row["relative_improvement"]) for row in convergence], dtype=np.float64))
    compared = {
        design: overall_contrast(contrasts, design)
        for design in DESIGNS if design != "hacssmv5"
    }
    environment_wins = {
        design: sum(full[env] < design_means[design][env] for env in OCC_TO_CLEAN)
        for design in DESIGNS if design != "hacssmv5"
    }
    envelope_wins = sum(
        full[env] <= min(design_means[design][env]
                         for design in DESIGNS)
        for env in OCC_TO_CLEAN
    )
    criteria = {
        "vs_ssm_reduction_ge_5pct":
            float(compared["ssm"]["mean_paired_relative_reduction"]) >= 0.05,
        "vs_ssm_wins_ge_18_of_25": int(compared["ssm"]["paired_wins"]) >= 18,
        "vs_ssm_env_wins_ge_4_of_5": environment_wins["ssm"] >= 4,
        "vs_v4_noaux_reduction_ge_3pct":
            float(compared["hacsmv4_noaux"]["mean_paired_relative_reduction"]) >= 0.03,
        "vs_v4_noaux_wins_ge_17_of_25":
            int(compared["hacsmv4_noaux"]["paired_wins"]) >= 17,
        "vs_v4_noaux_env_wins_ge_4_of_5": environment_wins["hacsmv4_noaux"] >= 4,
        "locked_grid_envelope_wins_ge_4_of_5": envelope_wins >= 4,
        "beats_hold_ge_4_of_5": sum(full[env] < hold[env] for env in OCC_TO_CLEAN) >= 4,
        "convergence_absolute_median_lt_1pct": float(np.median(absolute)) < 0.01,
        "convergence_absolute_p95_lt_3pct": float(np.quantile(absolute, 0.95)) < 0.03,
        "convergence_absolute_max_lt_5pct": float(absolute.max()) < 0.05,
    }
    for reference in ("hacssmv5_noaux", "hacssmv5_noaction",
                      "hacssmv5_static", "hacssmv5_single"):
        label = reference.removeprefix("hacssmv5_")
        criteria[f"vs_{label}_reduction_ge_3pct"] = (
            float(compared[reference]["mean_paired_relative_reduction"]) >= 0.03
        )
        criteria[f"vs_{label}_wins_ge_17_of_25"] = (
            int(compared[reference]["paired_wins"]) >= 17
        )
        criteria[f"vs_{label}_env_wins_ge_3_of_5"] = environment_wins[reference] >= 3

    overall_best = all(criteria.values())
    positive_primary_directions = all(
        float(compared[reference]["mean_paired_relative_reduction"]) > 0.0
        for reference in ("ssm", "hacsmv4_noaux")
    )
    if not pilot_screen_passed:
        decision = "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    else:
        decision = (
            "OVERALL_BEST_IN_LOCKED_GRID" if overall_best else
            "PROMISING_NOT_OVERALL_BEST" if positive_primary_directions else
            "NO_GO"
        )
    return {
        "schema_version": 1,
        "phase": "final",
        "decision": decision,
        "pilot_screen_passed": pilot_screen_passed,
        "criteria": criteria,
        "completed_runs": len(rows),
        "observed": {
            "overall_contrasts": compared,
            "full_environment_mean_wins": environment_wins,
            "locked_grid_envelope_env_wins": envelope_wins,
            "full_hold_env_wins": sum(full[env] < hold[env] for env in OCC_TO_CLEAN),
            "convergence_absolute_median": float(np.median(absolute)),
            "convergence_absolute_p95": float(np.quantile(absolute, 0.95)),
            "convergence_absolute_max": float(absolute.max()),
        },
        "limitations": [
            "Same fixed validation trajectories and black-token corruption as the V4 study.",
            "Equal training budget; no per-baseline hyperparameter tuning or downstream return.",
            "No post-training causal diagnostics are part of this analyzer version.",
        ],
        "note": (
            "The three-seed pilot screen is immutable. OVERALL_BEST_IN_LOCKED_GRID is a "
            "deterministic development-grid label, not an untouched-test or publication claim."
        ),
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/hacssm_v5_shared"))
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = PILOT_SEEDS if args.phase == "pilot" else FINAL_SEEDS
    expected_count = 180 if args.phase == "pilot" else 300
    rows, convergence = load_cells(args.root, seeds)
    if len(rows) != expected_count:
        raise ValueError(f"{args.phase} grid has {len(rows)} rows, expected {expected_count}")
    grouped = grouped_rows(rows)
    contrasts = contrast_rows(rows)
    prefix = "pilot_" if args.phase == "pilot" else ""
    if args.phase == "pilot":
        decision = pilot_decision(rows, convergence, contrasts)
    else:
        pilot_path = args.root / "pilot_decision.json"
        pilot = read_json(pilot_path)
        if (not isinstance(pilot, dict)
                or type(pilot.get("pilot_screen_passed")) is not bool):
            raise ValueError(f"invalid immutable pilot decision: {pilot_path}")
        decision = final_summary(
            rows, convergence, contrasts,
            pilot_screen_passed=pilot["pilot_screen_passed"])
    atomic_csv(args.root / f"{prefix}per_run.csv", rows)
    atomic_csv(args.root / f"{prefix}grouped.csv", grouped)
    atomic_csv(args.root / f"{prefix}paired_contrasts.csv", contrasts)
    atomic_csv(args.root / f"{prefix}convergence.csv", convergence)
    if args.phase == "pilot":
        atomic_json(args.root / "pilot_decision.json", decision)
    else:
        atomic_json(args.root / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
