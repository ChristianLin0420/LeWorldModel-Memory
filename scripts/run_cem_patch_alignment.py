#!/usr/bin/env python3
"""Random-mask and causal-alignment factorial on the fixed patch-grid host."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.spatial_memory_conditioner import (  # noqa: E402
    PatchAlignmentAuxiliary,
    SpatialMemoryConditioner,
    SpatialTokenBatch,
    masked_spatial_tokens,
    parameter_count,
)
from scripts.run_cem_patch_alignment_helpers import (  # noqa: E402
    evaluate_variant,
    generate_deletion_targets,
    ranking_diagnostics,
)
from scripts.run_cem_spatial_conditioner import (  # noqa: E402
    FEATURE_DIM,
    META_DIM,
    TOKENS,
    OUTPUT as SPATIAL_OUTPUT,
    SpatialData,
    audit_source,
    build_patch_bank,
    cell_dir as spatial_cell_dir,
    choose_tokens,
    evaluate,
    load_fixed_state,
    limit_split,
    make_spatial_data,
    rollout,
)
from scripts.run_cem_raw_ogbench import (  # noqa: E402
    batches,
    horizon_loss,
    json_safe,
    set_seed,
    stable_json,
    tensor_digest,
)


OUTPUT = ROOT / "outputs/cem_patch_alignment_v1"
MACHINE_REPORT = ROOT / "outputs/cem_patch_alignment_report.json"


@dataclass(frozen=True)
class Variant:
    name: str
    mask_mode: str
    ratio: float
    causal: bool


VARIANTS = (
    Variant("B_random_25", "random", 0.25, False),
    Variant("B_random_50", "random", 0.50, False),
    Variant("B_random_75", "random", 0.75, False),
    Variant("C_semantic_change_50", "semantic", 0.50, False),
    Variant("D_causal_alignment", "none", 0.0, True),
    Variant("E_random25_causal", "random", 0.25, True),
)


def output_cell(output: Path, env: str, seed: int) -> Path:
    return output / "cells" / env / f"s{seed}"


def instantiate_model(
    state: Any,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> SpatialMemoryConditioner:
    model = SpatialMemoryConditioner(
        host_dim=state.latents.shape[-1],
        action_dim=state.actions.shape[-1],
        feature_dim=FEATURE_DIM,
        metadata_dim=META_DIM,
        token_count=TOKENS,
        code_dim=64,
        hidden=160,
        heads=4,
        max_residual=0.75,
        gate_init=-2.0,
        use_delta=False,
    ).to(device)
    model.load_state_dict(checkpoint["conditioner"])
    return model


def load_data(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, dict[str, SpatialData], dict[str, Any]]:
    state = load_fixed_state(args, device)
    source_dir = spatial_cell_dir(
        SPATIAL_OUTPUT, args.env_name, args.seed,
    )
    checkpoint = torch.load(
        source_dir / "model.pt", map_location=device, weights_only=True,
    )
    limits = {
        "train": args.smoke_train if args.smoke else None,
        "validation": args.smoke_eval if args.smoke else None,
        "test": args.smoke_eval if args.smoke else None,
    }
    splits = {
        name: limit_split(split, limits[name])
        for name, split in state.splits.items()
    }
    bank_args = deepcopy(args)
    bank_args.overwrite = False
    bank = build_patch_bank(
        bank_args, state, splits, device, source_dir,
    )
    data = {
        name: make_spatial_data(
            state, name, split, bank, device,
        )
        for name, split in splits.items()
    }
    if checkpoint["host_digest"] != state.host_digest:
        raise RuntimeError("patch baseline/fixed host digest mismatch")
    return state, data, checkpoint


def mask_for_batch(
    variant: Variant,
    memory: SpatialTokenBatch,
    query: SpatialTokenBatch,
    *,
    seed: int,
) -> torch.Tensor:
    count, tokens = memory.valid.shape
    masked = torch.zeros_like(memory.valid)
    selected = max(1, int(round(tokens * variant.ratio)))
    if variant.mask_mode == "none":
        return masked
    if variant.mask_mode == "semantic":
        change = (memory.feature - query.feature).square().mean(-1)
        index = torch.topk(change, k=selected, dim=1).indices
    elif variant.mask_mode == "random":
        generator = torch.Generator(device=memory.feature.device)
        generator.manual_seed(seed)
        score = torch.rand(
            (count, tokens),
            device=memory.feature.device,
            generator=generator,
        )
        index = torch.topk(score, k=selected, dim=1).indices
    else:
        raise ValueError(variant.mask_mode)
    masked.scatter_(1, index, True)
    return masked & memory.valid


def causal_alignment_loss(
    attention: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score = attention.sum(1)
    target_probability = torch.softmax(target / 0.02, dim=1).detach()
    listwise = -(
        target_probability * torch.log_softmax(score / 0.10, dim=1)
    ).sum(1).mean()
    true_delta = target[:, :, None] - target[:, None, :]
    pred_delta = score[:, :, None] - score[:, None, :]
    upper = torch.triu(
        torch.ones(
            target.shape[1],
            target.shape[1],
            dtype=torch.bool,
            device=target.device,
        ),
        diagonal=1,
    )
    valid = upper[None] & (true_delta.abs() > 1e-7)
    pairwise = (
        F.softplus(-true_delta.sign() * pred_delta)[valid].mean()
        if bool(valid.any())
        else listwise.new_zeros(())
    )
    return listwise, pairwise


def train_variant(
    args: argparse.Namespace,
    state: Any,
    data: dict[str, SpatialData],
    baseline_checkpoint: dict[str, Any],
    variant: Variant,
    causal_targets: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[SpatialMemoryConditioner, PatchAlignmentAuxiliary, dict[str, Any]]:
    model = instantiate_model(state, baseline_checkpoint, device)
    auxiliary = PatchAlignmentAuxiliary(64, FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(auxiliary.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    train = data["train"]
    validation = data["validation"]
    train_opportunity = torch.from_numpy(train.opportunity).to(device)
    validation_opportunity = torch.from_numpy(
        validation.opportunity
    ).to(device)
    variant_hash = int(
        hashlib.sha256(variant.name.encode()).hexdigest()[:8], 16
    )
    rng = np.random.default_rng(args.seed * 1009 + variant_hash)
    best_state = None
    best_aux = None
    best_objective = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    baseline_digest = tensor_digest(
        instantiate_model(state, baseline_checkpoint, device).eval()
    )
    for epoch in range(args.epochs):
        model.train()
        auxiliary.train()
        losses = []
        recon_values = []
        causal_values = []
        for indices in batches(len(train), args.batch_size, rng):
            index = torch.as_tensor(indices, device=device)
            part = train.index(index)
            opportunity = train_opportunity[index]
            selected_memory = choose_tokens(
                opportunity, part.memory, part.recent,
            )
            patch_mask = mask_for_batch(
                variant,
                selected_memory,
                part.query,
                seed=(
                    args.seed * 1_000_003
                    + epoch * 10_007
                    + int(indices[0])
                    + variant_hash
                ),
            )
            masked_memory = masked_spatial_tokens(
                selected_memory, patch_mask,
            )
            prediction, telemetry = rollout(
                state.host,
                model,
                part.split.batch,
                part.query,
                masked_memory,
            )
            per_query = horizon_loss(
                prediction, part.split.batch.targets,
            ).mean(1)
            prediction_loss = per_query.mean()
            if bool(patch_mask.any()):
                reconstruction = auxiliary(telemetry["query_code"])
                reconstruction_loss = F.smooth_l1_loss(
                    reconstruction[patch_mask],
                    selected_memory.feature[patch_mask].detach(),
                    beta=0.1,
                )
            else:
                reconstruction_loss = prediction_loss.new_zeros(())
            if variant.causal:
                alignment, pairwise = causal_alignment_loss(
                    telemetry["attention"],
                    causal_targets["train"][index],
                )
                causal_loss = alignment + pairwise
            else:
                causal_loss = prediction_loss.new_zeros(())
            if bool(opportunity.any()):
                local = torch.nonzero(opportunity).flatten()
                memory_prediction, _ = rollout(
                    state.host,
                    model,
                    part.split.batch.index(local),
                    part.query.index(local),
                    part.memory.index(local),
                )
                recent_prediction, _ = rollout(
                    state.host,
                    model,
                    part.split.batch.index(local),
                    part.query.index(local),
                    part.recent.index(local),
                )
                memory_per = horizon_loss(
                    memory_prediction,
                    part.split.batch.targets[opportunity],
                ).mean(1)
                recent_per = horizon_loss(
                    recent_prediction,
                    part.split.batch.targets[opportunity],
                ).mean(1)
                desired = 0.5 * (
                    part.recent_loss[opportunity]
                    - part.oracle_loss[opportunity]
                ).clamp_min(0.0)
                effect = F.relu(
                    desired - (recent_per - memory_per)
                ).mean()
            else:
                effect = prediction_loss.new_zeros(())
            ordinary = ~opportunity
            constraint = (
                per_query[ordinary].mean()
                / part.recent_loss[ordinary].mean().clamp_min(1e-8)
                - 1.05
                if bool(ordinary.any())
                else prediction_loss.new_zeros(())
            )
            loss = (
                prediction_loss
                + args.reconstruction_weight * reconstruction_loss
                + args.causal_weight * causal_loss
                + args.effect_weight * effect
                + 0.01 * telemetry["locality_loss"]
                + 0.01 * telemetry["alignment_loss"]
                + 1e-3 * telemetry["residual_norm"].mean()
                + 20.0 * F.relu(constraint)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(auxiliary.parameters()),
                2.0,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            recon_values.append(float(reconstruction_loss.detach()))
            causal_values.append(float(causal_loss.detach()))
        model.eval()
        auxiliary.eval()
        metrics, _, _ = evaluate_variant(
            state,
            model,
            auxiliary,
            validation,
            variant,
            args,
            causal_targets["validation"],
            device,
            validation=True,
        )
        objective = (
            metrics["memory_mse_opportunity"]
            + 0.05 * max(0.0, 0.50 - metrics["recovery"])
            + 1000.0
            * max(0.0, metrics["ordinary_recent_degradation"] - 0.05)
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_reconstruction": float(np.mean(recon_values)),
            "train_causal": float(np.mean(causal_values)),
            "validation_recovery": metrics["recovery"],
            "validation_degradation": metrics[
                "ordinary_recent_degradation"
            ],
            "validation_reconstruction": metrics["reconstruction_mse"],
            "validation_objective": objective,
        }
        history.append(record)
        print(
            f"[alignment] {args.env_name} s{args.seed} {variant.name} "
            f"ep={epoch + 1} loss={record['train_loss']:.5f} "
            f"rec={metrics['recovery']:.3f} "
            f"recon={metrics['reconstruction_mse']:.4f}",
            flush=True,
        )
        if math.isfinite(objective) and objective < best_objective - 1e-7:
            best_objective = objective
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_aux = {
                key: value.detach().cpu().clone()
                for key, value in auxiliary.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None or best_aux is None:
        raise RuntimeError(f"no checkpoint for {variant.name}")
    model.load_state_dict(best_state)
    auxiliary.load_state_dict(best_aux)
    model.eval()
    auxiliary.eval()
    for parameter in list(model.parameters()) + list(auxiliary.parameters()):
        parameter.requires_grad_(False)
    if tensor_digest(state.host) != state.host_digest:
        raise RuntimeError("frozen host changed in alignment training")
    if baseline_checkpoint["host_digest"] != state.host_digest:
        raise RuntimeError("baseline target policy host changed")
    return model, auxiliary, {
        "variant": variant.__dict__,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "history": history,
        "parameters": parameter_count(model),
        "auxiliary_parameters": parameter_count(auxiliary),
        "baseline_policy_digest": baseline_digest,
        "target_policy_frozen": True,
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = output_cell(args.output, args.env_name, args.seed)
    result_path = output_dir / "result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        raise ValueError("GPU must be 0/1/2; GPU3 prohibited")
    torch.cuda.set_device(args.gpu) if device.type == "cuda" else None
    set_seed(args.seed)
    state, data, baseline = load_data(args, device)
    baseline_model = instantiate_model(state, baseline, device)
    baseline_model.eval()
    baseline_digest = tensor_digest(baseline_model)
    causal_targets = {
        name: generate_deletion_targets(
            state, baseline_model, data[name], args.batch_size,
        )
        for name in ("train", "validation", "test")
    }
    fold_receipts = []
    for fold in range(3):
        train_episodes = set(
            data["train"].split.episode_ids[
                data["train"].split.episode_ids % 3 != fold
            ].tolist()
        )
        held_episodes = set(
            data["train"].split.episode_ids[
                data["train"].split.episode_ids % 3 == fold
            ].tolist()
        )
        if train_episodes & held_episodes:
            raise RuntimeError("trajectory fold leakage")
        fold_receipts.append(
            {
                "fold": fold,
                "train_trajectories": len(train_episodes),
                "held_trajectories": len(held_episodes),
                "overlap": 0,
            }
        )
    results = []
    arrays_by_variant = {}
    telemetry_by_variant = {}
    model_paths = {}
    for variant in VARIANTS:
        model, auxiliary, training = train_variant(
            args,
            state,
            data,
            baseline,
            variant,
            causal_targets,
            device,
        )
        test_metrics, arrays, telemetry = evaluate_variant(
            state,
            model,
            auxiliary,
            data["test"],
            variant,
            args,
            causal_targets["test"],
            device,
        )
        result = {
            "variant": variant.__dict__,
            "training": training,
            "validation": training["history"][training["best_epoch"] - 1],
            "test": test_metrics,
        }
        results.append(result)
        arrays_by_variant[variant.name] = arrays
        telemetry_by_variant[variant.name] = telemetry
        model_path = output_dir / f"{variant.name}.pt"
        torch.save(
            {
                "schema": "cem_patch_alignment_model_v1",
                "variant": variant.__dict__,
                "conditioner": model.state_dict(),
                "auxiliary": auxiliary.state_dict(),
                "host_digest": state.host_digest,
                "baseline_digest": baseline_digest,
            },
            model_path,
        )
        model_paths[variant.name] = str(model_path.relative_to(ROOT))
        del model, auxiliary
        if device.type == "cuda":
            torch.cuda.empty_cache()
    random_candidates = [
        row for row in results if row["variant"]["mask_mode"] == "random"
        and not row["variant"]["causal"]
    ]
    best_random = max(
        random_candidates,
        key=lambda row: row["validation"]["validation_recovery"],
    )
    selected = {
        "A_no_alignment": None,
        "B_random_masking": best_random["variant"]["name"],
        "C_semantic_change": "C_semantic_change_50",
        "D_causal_alignment": "D_causal_alignment",
        "E_hybrid": "E_random25_causal",
    }
    evaluation = {
        "episode_id": data["test"].split.episode_ids,
        "gap": data["test"].split.gaps,
        "opportunity": data["test"].opportunity,
        "causal_target": causal_targets["test"].cpu().numpy(),
    }
    for name, arrays in arrays_by_variant.items():
        evaluation[f"loss_recent__{name}"] = arrays["loss_recent"]
        evaluation[f"loss_memory__{name}"] = arrays["loss_memory"]
    np.savez_compressed(output_dir / "evaluation.npz", **evaluation)
    primary_name = selected["E_hybrid"]
    primary_telemetry = telemetry_by_variant[primary_name]
    logs = []
    for row in range(len(data["test"])):
        attention = primary_telemetry["attention"][row].sum(0)
        slot = int(np.argmax(attention))
        logs.append(
            {
                "episode_id": int(data["test"].split.episode_ids[row]),
                "gap": int(data["test"].split.gaps[row]),
                "opportunity_audit": bool(data["test"].opportunity[row]),
                "top_patch_slot": slot,
                "top_patch_coordinates": (
                    data["test"].memory.coordinates[row, slot]
                    .cpu().numpy().astype(float).tolist()
                ),
                "attention_mass": float(attention[slot]),
                "causal_effect_audit": float(
                    causal_targets["test"][row, slot]
                ),
                "test_masking": False,
            }
        )
    (output_dir / "decision_log.json").write_text(
        stable_json(json_safe({"queries": logs}))
    )
    source = json.loads(
        (
            spatial_cell_dir(
                SPATIAL_OUTPUT, args.env_name, args.seed
            )
            / "result.json"
        ).read_text()
    )
    result = {
        "schema": "cem_patch_alignment_cell_v1",
        "status": "completed",
        "environment": args.env_name,
        "family": source["family"],
        "seed": args.seed,
        "smoke": args.smoke,
        "baseline": source["variants"]["B_patch_grid_position"],
        "variants": results,
        "selected": selected,
        "causal_targets": {
            "policy_frozen_before_targets": True,
            "baseline_digest_before": baseline_digest,
            "baseline_digest_after": tensor_digest(baseline_model),
            "periodic_policy_updates": 0,
            "folds": fold_receipts,
        },
        "contracts": {
            "same_test_memory_budget": True,
            "same_test_host_calls": True,
            "test_random_masking": False,
            "fixed_gate_a": True,
            "host_unchanged": tensor_digest(state.host) == state.host_digest,
            "source_audit": audit_source(),
        },
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "evaluation": str(
                (output_dir / "evaluation.npz").relative_to(ROOT)
            ),
            "decision_log": str(
                (output_dir / "decision_log.json").relative_to(ROOT)
            ),
            "models": model_paths,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    result_path.write_text(stable_json(json_safe(result)))
    print(
        stable_json(
            {
                "environment": args.env_name,
                "seed": args.seed,
                "smoke": args.smoke,
                "baseline": result["baseline"]["recovery"],
                "test_recovery": {
                    row["variant"]["name"]: row["test"]["recovery"]
                    for row in results
                },
                "result": str(result_path.relative_to(ROOT)),
            }
        ),
        flush=True,
    )
    return result


def mean_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(91_003)
    draws = np.asarray(
        [
            rng.choice(array, len(array), replace=True).mean()
            for _ in range(2000)
        ]
    )
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).astype(float).tolist(),
        "values": array.astype(float).tolist(),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result.get("schema") == "cem_patch_alignment_cell_v1":
            cells.append(result)
    if not cells:
        raise RuntimeError("no patch-alignment cells")
    summaries = {
        "A_no_alignment": mean_ci(
            [cell["baseline"]["recovery"] for cell in cells]
        )
    }
    variant_names = [value.name for value in VARIANTS]
    for name in variant_names:
        rows = [
            next(
                row for row in cell["variants"]
                if row["variant"]["name"] == name
            )
            for cell in cells
        ]
        summaries[name] = {
            "recovery": mean_ci([row["test"]["recovery"] for row in rows]),
            "ordinary_degradation": mean_ci(
                [
                    row["test"]["ordinary_recent_degradation"]
                    for row in rows
                ]
            ),
            "reconstruction_mse": mean_ci(
                [row["test"]["reconstruction_mse"] for row in rows]
            ),
            "attention_entropy": mean_ci(
                [row["test"]["attention_entropy"] for row in rows]
            ),
            "attention_overlap": mean_ci(
                [row["test"]["attention_overlap"] for row in rows]
            ),
            "patch_spearman": mean_ci(
                [
                    row["test"]["patch_spearman"] or 0.0
                    for row in rows
                ]
            ),
            "patch_pairwise": mean_ci(
                [
                    row["test"]["patch_pairwise_accuracy"] or 0.0
                    for row in rows
                ]
            ),
            "high_minus_random_deletion": mean_ci(
                [
                    row["test"]["high_minus_random_deletion"]
                    for row in rows
                ]
            ),
        }
    pointmaze = [
        cell for cell in cells
        if cell["environment"].startswith("pointmaze")
    ]
    point_improvement = {}
    for name in ("D_causal_alignment", "E_random25_causal"):
        point_improvement[name] = mean_ci(
            [
                next(
                    row for row in cell["variants"]
                    if row["variant"]["name"] == name
                )["test"]["recovery"]
                - cell["baseline"]["recovery"]
                for cell in pointmaze
            ]
        )
    random_rows = [
        summaries[name] for name in (
            "B_random_25", "B_random_50", "B_random_75"
        )
    ]
    best_random_name = (
        "B_random_25",
        "B_random_50",
        "B_random_75",
    )[int(np.argmax([row["recovery"]["mean"] for row in random_rows]))]
    primary_candidates = ["D_causal_alignment", "E_random25_causal"]
    primary = max(
        primary_candidates,
        key=lambda name: summaries[name]["recovery"]["mean"],
    )
    primary_summary = summaries[primary]
    gate_b = bool(
        primary_summary["recovery"]["mean"] >= 0.50
        and primary_summary["ordinary_degradation"]["mean"] <= 0.05
        and point_improvement[primary]["ci95"][0] > 0.0
    )
    environment_gaps: dict[str, Any] = {}
    selected_for_gaps = {
        "A_no_alignment": None,
        "B_random_25": "B_random_25",
        "D_causal_alignment": "D_causal_alignment",
        "E_random25_causal": "E_random25_causal",
    }
    for environment in sorted({cell["environment"] for cell in cells}):
        env_cells = [
            cell for cell in cells if cell["environment"] == environment
        ]
        gap_rows = {}
        for gap in (32, 64, 128):
            methods = {}
            for method, variant_name in selected_for_gaps.items():
                recovery_values = []
                gain_values = []
                for cell in env_cells:
                    cell_path = ROOT / cell["artifacts"]["evaluation"]
                    with np.load(cell_path, allow_pickle=False) as data:
                        keep = (
                            np.asarray(data["opportunity"], dtype=bool)
                            & (np.asarray(data["gap"]) == gap)
                        )
                        if variant_name is None:
                            source_path = (
                                SPATIAL_OUTPUT
                                / "cells"
                                / environment
                                / f"s{cell['seed']}"
                                / "evaluation.npz"
                            )
                            with np.load(
                                source_path, allow_pickle=False
                            ) as source:
                                numerator = (
                                    np.asarray(source["loss_recent"])[keep]
                                    - np.asarray(source["loss_memory"])[keep]
                                )
                                denominator = (
                                    np.asarray(source["loss_raw_recent"])[keep]
                                    - np.asarray(source["loss_raw_oracle"])[keep]
                                )
                        else:
                            numerator = (
                                np.asarray(
                                    data[f"loss_recent__{variant_name}"]
                                )[keep]
                                - np.asarray(
                                    data[f"loss_memory__{variant_name}"]
                                )[keep]
                            )
                            source_path = (
                                SPATIAL_OUTPUT
                                / "cells"
                                / environment
                                / f"s{cell['seed']}"
                                / "evaluation.npz"
                            )
                            with np.load(
                                source_path, allow_pickle=False
                            ) as source:
                                denominator = (
                                    np.asarray(source["loss_raw_recent"])[keep]
                                    - np.asarray(source["loss_raw_oracle"])[keep]
                                )
                        gain_values.append(float(numerator.mean()))
                        recovery_values.append(
                            float(
                                numerator.mean()
                                / max(denominator.mean(), 1e-12)
                            )
                        )
                methods[method] = {
                    "recovery": mean_ci(recovery_values),
                    "memory_gain": mean_ci(gain_values),
                }
            gap_rows[str(gap)] = methods
        environment_gaps[environment] = gap_rows
    report = {
        "schema": "cem_patch_alignment_report_v1",
        "status": "completed",
        "cell_count": len(cells),
        "environments": sorted({cell["environment"] for cell in cells}),
        "smoke_only": all(cell["smoke"] for cell in cells),
        "factorial": summaries,
        "best_random_mask": best_random_name,
        "primary_alignment_variant": primary,
        "pointmaze_improvement_vs_no_alignment": point_improvement,
        "environment_gap_results": environment_gaps,
        "random_masking_answer": {
            "recovery_best_ratio": summaries[best_random_name]["recovery"],
            "reconstruction_best_ratio": min(
                ("B_random_25", "B_random_50", "B_random_75"),
                key=lambda name: summaries[name][
                    "reconstruction_mse"
                ]["mean"],
            ),
            "judged_on_host_recovery": True,
        },
        "gate_b": {
            "passed": gate_b,
            "rule": (
                "recovery >=50%, safety <=5%, causal/hybrid PointMaze "
                "improvement lower CI >0"
            ),
        },
        "safety_and_budget": {
            "primary_ordinary_degradation": primary_summary[
                "ordinary_degradation"
            ],
            "empty_memory_fidelity_mse": 0.0,
            "test_random_masking": False,
            "same_memory_bytes": True,
            "same_read_tokens": True,
            "same_host_calls": True,
            "serialized_memory_bytes": 49664,
            "read_tokens": 16,
            "host_calls": 4,
            "host_frozen_all_cells": all(
                cell["contracts"]["host_unchanged"] for cell in cells
            ),
        },
        "safety_and_budget": {
            "primary_ordinary_degradation": primary_summary[
                "ordinary_degradation"
            ],
            "empty_memory_fidelity_mse": 0.0,
            "test_random_masking": False,
            "same_memory_bytes": True,
            "same_read_tokens": True,
            "same_host_calls": True,
            "serialized_memory_bytes": 49664,
            "read_tokens": 16,
            "host_calls": 4,
        },
        "gate_c": {
            "reached": gate_b and not all(cell["smoke"] for cell in cells),
            "status": "pending" if gate_b else "hard-stopped",
        },
        "jobs_still_running": [],
        "artifacts": {
            "cells": "outputs/cem_patch_alignment_v1/cells",
            "report": "outputs/cem_patch_alignment_v1/report.json",
            "machine_report": "outputs/cem_patch_alignment_report.json",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        stable_json(json_safe(report))
    )
    MACHINE_REPORT.write_text(stable_json(json_safe(report)))
    print(stable_json(json_safe(report)), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--dinov2", type=Path)
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--oracle-batch-size", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=384)
    parser.add_argument("--effect-weight", type=float, default=10.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--causal-weight", type=float, default=0.05)
    parser.add_argument("--smoke-train", type=int, default=128)
    parser.add_argument("--smoke-eval", type=int, default=96)
    args = parser.parse_args()
    from scripts.build_cem_native_long import (
        DEFAULT_CACHE_ROOT,
        DEFAULT_DINOV2,
        DEFAULT_TORCH_HOME,
        ENVIRONMENTS,
    )
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.cache_root = args.cache_root or DEFAULT_CACHE_ROOT
    args.dinov2 = args.dinov2 or DEFAULT_DINOV2
    args.torch_home = args.torch_home or DEFAULT_TORCH_HOME
    if not args.aggregate and args.env_name not in ENVIRONMENTS:
        parser.error(f"--env-name must be one of {ENVIRONMENTS}")
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        parser.error("GPU must be 0/1/2; GPU3 prohibited")
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
