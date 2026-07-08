#!/usr/bin/env python3
"""Train one paired cue-residual repair diagnostic on the frozen LeWM host."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier  # noqa: E402
from lewm.models.official_lewm import (  # noqa: E402
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
    OFFICIAL_HISTORY,
    load_official_reacher_checkpoint,
)
from scripts.delayed_repair_residual_v2_objective import (  # noqa: E402
    DECISION_INDEX,
    EpochPlan,
    cue_residual_target,
    fit_target_standardizer,
    load_label_free_bank,
    make_epoch_plans,
    reconstruction_metrics,
)
from scripts.delayed_repair_residual_v2_spec import (  # noqa: E402
    ARMS,
    CONDITIONS,
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    SEEDS,
    TASKS,
    load_locked_spec,
    lock_receipt,
    repair_directory,
    require_development_health,
    resolve_path,
    sha256_file,
    stable_json,
    validate_device,
)
from scripts.reevaluate_frozen_official_probes import (  # noqa: E402
    Cell,
    load_config,
    preflight_cell,
    state_dict_digest,
)
from scripts.train_frozen_official_swap import (  # noqa: E402
    prediction_mse,
    sampled_windows,
    state_digest,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _authenticate_parent(spec: dict[str, Any], task: str, arm: str,
                         seed: int):
    task_record = spec["tasks"][task]
    parent_id = task_record["parent_task_id"]
    config = load_config(resolve_path(spec["parent"]["expansion_config"]["path"]))
    prepared = preflight_cell(
        Cell(parent_id, arm, seed),
        resolve_path(spec["parent"]["checkpoint_root"]), config)
    expected = task_record["parent_carrier_state_sha256"][arm][seed]
    if prepared.state_sha256 != expected:
        raise ValueError(
            f"authenticated parent state differs for {task}/{arm}/seed-{seed}")
    return prepared


def _train_epoch(
        host: torch.nn.Module, carrier: torch.nn.Module,
        head: torch.nn.Module, optimizer: torch.optim.Optimizer,
        z: np.ndarray, actions: np.ndarray, normalized_target: np.ndarray,
        plan: EpochPlan, *, batch_size: int, next_latent_weight: float,
        repair_weight: float, gradient_clip_norm: float,
        device: torch.device,
        ) -> tuple[float, float, float]:
    carrier.train()
    head.train()
    totals, next_losses, repair_losses = [], [], []
    batch_index = 0
    for offset in range(0, len(plan.order), batch_size):
        index = plan.order[offset:offset + batch_size]
        if len(index) < 4:
            continue
        starts = plan.starts_by_batch[batch_index]
        batch_index += 1
        latent = torch.from_numpy(z[index]).to(device)
        action = torch.from_numpy(actions[index]).to(device)
        target = torch.from_numpy(normalized_target[index]).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = carrier(latent, action)
        context, action_context, next_target = sampled_windows(
            output.z_tilde, action, latent, starts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            next_prediction = host.predict(context, action_context)[:, -1]
            next_loss = F.mse_loss(
                next_prediction.float(), next_target.float())
            reconstruction = head(
                output.prior_read[:, DECISION_INDEX].float())
            repair_loss = F.mse_loss(reconstruction.float(), target.float())
            loss = next_latent_weight * next_loss + repair_weight * repair_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*carrier.parameters(), *head.parameters()], gradient_clip_norm)
        optimizer.step()
        totals.append(float(loss.detach()))
        next_losses.append(float(next_loss.detach()))
        repair_losses.append(float(repair_loss.detach()))
    if batch_index != len(plan.starts_by_batch) or not totals:
        raise RuntimeError("deterministic V2 epoch plan was not consumed exactly")
    return (float(np.mean(totals)), float(np.mean(next_losses)),
            float(np.mean(repair_losses)))


@torch.no_grad()
def _carrier_outputs(carrier: torch.nn.Module, z: np.ndarray,
                     actions: np.ndarray, device: torch.device,
                     batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    carrier.eval()
    fused, prior = [], []
    for offset in range(0, len(z), batch_size):
        output = carrier(
            torch.from_numpy(z[offset:offset + batch_size]).to(device),
            torch.from_numpy(actions[offset:offset + batch_size]).to(device))
        fused.append(output.z_tilde.float().cpu().numpy())
        prior.append(output.prior_read.float().cpu().numpy())
    return np.concatenate(fused), np.concatenate(prior)


@torch.no_grad()
def _causal_decision_check(carrier: torch.nn.Module, z: np.ndarray,
                           actions: np.ndarray, device: torch.device
                           ) -> dict[str, Any]:
    sample = np.array(z[:min(8, len(z))], copy=True)
    altered = np.array(sample, copy=True)
    altered[:, DECISION_INDEX] = np.float32(123.0)
    action = torch.from_numpy(actions[:len(sample)]).to(device)
    first = carrier(torch.from_numpy(sample).to(device), action)
    second = carrier(torch.from_numpy(altered).to(device), action)
    difference = (first.prior_read[:, DECISION_INDEX]
                  - second.prior_read[:, DECISION_INDEX]).abs()
    maximum = float(difference.max().cpu())
    return {
        "intervention": "replace z[:,63] before recomputing prior_read[:,63]",
        "episodes": int(len(sample)),
        "maximum_absolute_prior_difference": maximum,
        "exact_invariance": maximum == 0.0,
        "decision_frame_used_for_optimization": False,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing V2 training without explicit --execute")
    spec = load_locked_spec(args.spec, args.lock)
    validate_device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("V2 repair training requires CUDA; no CPU fallback")
    development = require_development_health(spec, args.task)
    output = repair_directory(
        spec, args.task, args.arm, args.seed, args.condition)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    prepared = _authenticate_parent(spec, args.task, args.arm, args.seed)

    repair = spec["formal_repair"]
    task_record = spec["tasks"][args.task]
    train_record = task_record["training_cache"]
    train = load_label_free_bank(resolve_path(train_record["path"]))
    train_target, train_target_audit = cue_residual_target(train)
    standardizer = fit_target_standardizer(
        train_target, float(spec["cue_residual_target"]["scale_floor"]))
    normalized_train = standardizer.transform(train_target)
    plans, plan_digest = make_epoch_plans(
        len(normalized_train), epochs=int(repair["epochs"]),
        batch_size=int(repair["batch_size"]),
        windows_per_batch=int(repair["next_latent_windows_per_batch"]),
        sequence_length=64, history=OFFICIAL_HISTORY,
        seed=int(repair["batch_plan_seed_base"]) + args.seed)

    torch_seed = int(repair["torch_seed_base"]) + args.seed
    _configure_determinism(torch_seed)
    device = torch.device(args.device)
    host = load_official_reacher_checkpoint(
        resolve_path(spec["parent"]["official_weights"]["path"]), device)
    host.eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    host_before = state_digest(host)
    carrier = make_frozen_carrier(
        args.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)
    carrier.load_state_dict(
        prepared.checkpoint["carrier_state_dict"], strict=True)
    parent_state = state_dict_digest(prepared.checkpoint["carrier_state_dict"])
    head = torch.nn.Linear(OFFICIAL_EMBED_DIM, OFFICIAL_EMBED_DIM).to(device)
    head_initial = state_digest(head)
    optimizer = torch.optim.AdamW(
        [*carrier.parameters(), *head.parameters()],
        lr=float(repair["learning_rate"]),
        weight_decay=float(repair["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(repair["epochs"]))
    repair_weight = float(repair["cue_residual_weight"][args.condition])

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        rows = []
        history_path = stage / "history.csv"
        with history_path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "epoch", "total_loss", "next_latent_loss",
                "cue_residual_loss", "repair_weight", "lr"))
            writer.writeheader()
            for epoch, plan in enumerate(plans, start=1):
                total, next_loss, residual_loss = _train_epoch(
                    host, carrier, head, optimizer,
                    np.asarray(train["z"], dtype=np.float32),
                    np.asarray(train["actions"], dtype=np.float32),
                    normalized_train, plan,
                    batch_size=int(repair["batch_size"]),
                    next_latent_weight=float(repair["next_latent_weight"]),
                    repair_weight=repair_weight,
                    gradient_clip_norm=float(repair["gradient_clip_norm"]),
                    device=device)
                scheduler.step()
                row = {
                    "epoch": epoch,
                    "total_loss": total,
                    "next_latent_loss": next_loss,
                    "cue_residual_loss": residual_loss,
                    "repair_weight": repair_weight,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                rows.append(row)
                writer.writerow(row)
                stream.flush()

        # Validation arrays are opened only after all optimizer steps finish.
        validation_record = task_record["validation_cache"]
        validation = load_label_free_bank(
            resolve_path(validation_record["path"]))
        validation_target, validation_target_audit = cue_residual_target(
            validation)
        normalized_validation = standardizer.transform(validation_target)
        fused, prior = _carrier_outputs(
            carrier, np.asarray(validation["z"], dtype=np.float32),
            np.asarray(validation["actions"], dtype=np.float32), device)
        with torch.no_grad():
            prediction = head(torch.from_numpy(
                prior[:, DECISION_INDEX]).to(device)).float().cpu().numpy()
        reconstruction = reconstruction_metrics(
            prediction, normalized_validation)
        next_mse = prediction_mse(
            host, fused, np.asarray(validation["actions"], dtype=np.float32),
            np.asarray(validation["z"], dtype=np.float32), device)
        causal = _causal_decision_check(
            carrier, np.asarray(validation["z"], dtype=np.float32),
            np.asarray(validation["actions"], dtype=np.float32), device)
        if not causal["exact_invariance"]:
            raise RuntimeError("prior_read[:,63] depends on forbidden z[:,63]")
        host_after = state_digest(host)
        if host_before != host_after:
            raise RuntimeError("frozen official host changed during V2 repair")
        metrics = {
            "schema": "paper_a_delayed_repair_residual_metrics_v2",
            "study": spec["study"],
            "scientific_role": spec["scientific_role"]["classification"],
            "preregistered_primary_result": False,
            "downstream_label_use_claim": False,
            "task": args.task,
            "display_name": task_record["display_name"],
            "arm": args.arm,
            "condition": args.condition,
            "checkpoint_seed": args.seed,
            "device": args.device,
            "formal_lock": lock_receipt(spec),
            "development_health": development,
            "parent_task_id": task_record["parent_task_id"],
            "parent_carrier_state_sha256": parent_state,
            "authenticated_parent_state_sha256":
                task_record["parent_carrier_state_sha256"][args.arm][args.seed],
            "carrier_state_sha256": state_dict_digest(carrier.state_dict()),
            "repair_head_initial_state_sha256": head_initial,
            "repair_head_state_sha256": state_digest(head),
            "repair_head": spec["cue_residual_target"]["repair_head"],
            "training_plan_sha256": plan_digest,
            "torch_seed": torch_seed,
            "official_host_state_sha256_before": host_before,
            "official_host_state_sha256_after": host_after,
            "frozen_host_unchanged": True,
            "epochs": int(repair["epochs"]),
            "batch_size": int(repair["batch_size"]),
            "learning_rate": float(repair["learning_rate"]),
            "weight_decay": float(repair["weight_decay"]),
            "next_latent_weight": float(repair["next_latent_weight"]),
            "cue_residual_weight": repair_weight,
            "optimizer": repair["optimizer"],
            "scheduler": repair["scheduler"],
            "target_standardizer_fit_split": "formal training cache only",
            "target_standardizer_sha256": standardizer.digest(),
            "target_scale_floor": standardizer.scale_floor,
            "target_scale_floored_coordinates": int(np.sum(
                standardizer.scale <= standardizer.scale_floor)),
            "training_target_audit": train_target_audit,
            "validation_target_audit": validation_target_audit,
            "repair_read": spec["cue_residual_target"]["read"],
            "causal_decision_intervention": causal,
            "label_arrays_loaded": False,
            "label_values_consumed": False,
            "decision_frame_used_for_optimization": False,
            "validation_used_for_optimization": False,
            "development_statistics_used_for_formal_normalization": False,
            "validation_cue_residual_mse": reconstruction["mse"],
            "validation_zero_predictor_mse":
                reconstruction["zero_predictor_mse"],
            "validation_normalized_mse_to_zero":
                reconstruction["normalized_mse_to_zero"],
            "validation_r2_vs_training_mean":
                reconstruction["r2_vs_training_mean"],
            "validation_next_latent_mse": next_mse,
            "source_caches": {
                "training": train_record,
                "validation": validation_record,
            },
            "last_epoch": rows[-1],
        }
        metrics_path = stage / "metrics.json"
        metrics_path.write_text(stable_json(metrics))
        checkpoint_path = stage / "repair.pt"
        torch.save({
            "carrier_state_dict": carrier.state_dict(),
            "repair_head_state_dict": head.state_dict(),
            "target_mean": torch.from_numpy(standardizer.mean),
            "target_scale": torch.from_numpy(standardizer.scale),
            "metrics": metrics,
        }, checkpoint_path)
        export_path = stage / "validation_export.npz"
        np.savez_compressed(
            export_path,
            per_episode_mse=reconstruction["per_episode_mse"],
            per_episode_zero_mse=reconstruction["per_episode_zero_mse"],
            prediction=prediction.astype(np.float32),
            normalized_target=normalized_validation.astype(np.float32),
        )
        manifest = {
            "schema": "paper_a_delayed_repair_residual_manifest_v2",
            "study": spec["study"],
            "task": args.task,
            "arm": args.arm,
            "condition": args.condition,
            "checkpoint_seed": args.seed,
            "formal_lock": lock_receipt(spec),
            "artifacts": {
                "history": {"path": history_path.name,
                            "sha256": sha256_file(history_path)},
                "metrics": {"path": metrics_path.name,
                            "sha256": sha256_file(metrics_path)},
                "checkpoint": {"path": checkpoint_path.name,
                               "sha256": sha256_file(checkpoint_path)},
                "validation_export": {"path": export_path.name,
                                      "sha256": sha256_file(export_path)},
            },
        }
        (stage / "manifest.json").write_text(stable_json(manifest))
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"[delayed-residual-v2] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
