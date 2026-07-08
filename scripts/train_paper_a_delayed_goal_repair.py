#!/usr/bin/env python3
"""Train the locked label-free delayed cue-latent repair isolation.

Every pair starts from one authenticated parent carrier.  Both conditions
continue the frozen-host next-latent objective for the same batches and steps;
only ``cue_repair`` gives nonzero weight to reconstructing early cue latents
from the prior available before the final frame.  Labels are never loaded.
"""

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

from lewm.models.frozen_swap_carriers import make_frozen_carrier
from lewm.models.official_lewm import (
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
    OFFICIAL_HISTORY,
    load_official_reacher_checkpoint,
)
from scripts.paper_a_delayed_goal_spec import (
    DEFAULT_SPEC,
    REPAIR_ARMS,
    REPAIR_CONDITIONS,
    SEEDS,
    load_locked_spec,
    repair_directory,
    resolve_path,
    validate_device,
)
from scripts.paper_a_delayed_goal_use import cue_repair_target
from scripts.reevaluate_frozen_official_probes import (
    Cell,
    load_config,
    preflight_cell,
    state_dict_digest,
)
from scripts.train_frozen_official_swap import (
    prediction_mse,
    sampled_windows,
    state_digest,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--task", required=True, choices=("t1", "t3"))
    parser.add_argument("--arm", required=True, choices=REPAIR_ARMS)
    parser.add_argument("--condition", required=True,
                        choices=REPAIR_CONDITIONS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _label_free_cache(path: Path) -> dict[str, np.ndarray]:
    """Load only registered representation arrays; ``xi`` is never read."""

    with np.load(path) as source:
        required = ("z", "actions", "event_cue_on", "event_cue_off")
        missing = [key for key in required if key not in source.files]
        if missing:
            raise ValueError(f"{path} misses label-free arrays {missing}")
        data = {key: np.asarray(source[key]) for key in required}
    if data["z"].shape[1:] != (64, OFFICIAL_EMBED_DIM):
        raise ValueError("repair cache has an unexpected latent shape")
    if data["actions"].shape != (
            len(data["z"]), 63, OFFICIAL_ACTION_DIM):
        raise ValueError("repair cache has an unexpected action shape")
    return data


def _train_epoch(host: torch.nn.Module, carrier: torch.nn.Module,
                 decoder: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 z: np.ndarray, actions: np.ndarray,
                 normalized_target: np.ndarray, *, batch_size: int,
                 next_latent_weight: float, repair_weight: float,
                 rng: np.random.Generator,
                 device: torch.device) -> tuple[float, float, float]:
    carrier.train()
    decoder.train()
    order = rng.permutation(len(z))
    totals: list[float] = []
    base_losses: list[float] = []
    repair_losses: list[float] = []
    for offset in range(0, len(order), batch_size):
        index = order[offset:offset + batch_size]
        if len(index) < 4:
            continue
        latent = torch.from_numpy(z[index]).to(device)
        action = torch.from_numpy(actions[index]).to(device)
        target = torch.from_numpy(normalized_target[index]).to(device)
        starts = rng.choice(
            z.shape[1] - OFFICIAL_HISTORY,
            size=min(8, z.shape[1] - OFFICIAL_HISTORY), replace=False)
        optimizer.zero_grad(set_to_none=True)
        output = carrier(latent, action)
        context, action_context, next_target = sampled_windows(
            output.z_tilde, action, latent, starts)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with amp:
            prediction = host.predict(context, action_context)[:, -1]
            base_loss = F.mse_loss(prediction.float(), next_target.float())
            # prior_read[:, 63] is emitted before carrier consumption of z[:,63].
            reconstruction = decoder(output.prior_read[:, 63].float())
            repair_loss = F.mse_loss(reconstruction.float(), target.float())
            loss = next_latent_weight * base_loss + repair_weight * repair_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*carrier.parameters(), *decoder.parameters()], 1.0)
        optimizer.step()
        totals.append(float(loss.detach()))
        base_losses.append(float(base_loss.detach()))
        repair_losses.append(float(repair_loss.detach()))
    if not totals:
        raise RuntimeError("repair epoch produced no optimization batches")
    return (float(np.mean(totals)), float(np.mean(base_losses)),
            float(np.mean(repair_losses)))


@torch.no_grad()
def _outputs(carrier: torch.nn.Module, z: np.ndarray, actions: np.ndarray,
             device: torch.device, batch_size: int = 32
             ) -> tuple[np.ndarray, np.ndarray]:
    carrier.eval()
    fused: list[np.ndarray] = []
    prior: list[np.ndarray] = []
    for offset in range(0, len(z), batch_size):
        output = carrier(
            torch.from_numpy(z[offset:offset + batch_size]).to(device),
            torch.from_numpy(actions[offset:offset + batch_size]).to(device))
        fused.append(output.z_tilde.float().cpu().numpy())
        prior.append(output.prior_read.float().cpu().numpy())
    return np.concatenate(fused), np.concatenate(prior)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing repair training without explicit --execute")
    spec = load_locked_spec(args.spec)
    device_name = args.device or spec["execution"]["default_device"]
    validate_device(spec, device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("locked repair execution requires CUDA; no CPU fallback")
    device = torch.device(device_name)
    repair = spec["repair"]
    repair_weight = float(repair["cue_repair_weight"][args.condition])
    output_dir = repair_directory(
        spec, args.task, args.arm, args.seed, args.condition)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    parent_config_path = resolve_path(spec["parent"]["config"]["path"])
    parent_config = load_config(parent_config_path)
    prepared = preflight_cell(
        Cell(args.task, args.arm, args.seed),
        resolve_path(spec["parent"]["checkpoint_root"]), parent_config)
    train_path = resolve_path(
        spec["parent"]["train_caches"][args.task]["path"])
    validation_path = resolve_path(
        spec["parent"]["validation_caches"][args.task]["path"])
    train = _label_free_cache(train_path)
    train_target, train_indices = cue_repair_target(train)
    target_mean = train_target.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = train_target.std(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(target_scale, np.float32(1e-6))
    normalized_train = ((train_target - target_mean) / target_scale).astype(
        np.float32)

    torch.manual_seed(170_000 + args.seed)
    np.random.seed(170_000 + args.seed)
    torch.cuda.manual_seed_all(170_000 + args.seed)
    torch.use_deterministic_algorithms(True)
    host = load_official_reacher_checkpoint(
        resolve_path(spec["parent"]["official_weights"]["path"]), device)
    host.eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    host_before = state_digest(host)
    carrier = make_frozen_carrier(
        args.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)
    carrier.load_state_dict(prepared.checkpoint["carrier_state_dict"], strict=True)
    parent_state = state_dict_digest(prepared.checkpoint["carrier_state_dict"])
    decoder = torch.nn.Linear(
        OFFICIAL_EMBED_DIM, 4 * OFFICIAL_EMBED_DIM).to(device)
    decoder_initial = state_digest(decoder)
    optimizer = torch.optim.AdamW(
        [*carrier.parameters(), *decoder.parameters()],
        lr=float(repair["learning_rate"]),
        weight_decay=float(repair["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(repair["epochs"]))
    rng = np.random.default_rng(270_000 + args.seed)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        rows: list[dict[str, Any]] = []
        with (stage / "history.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "epoch", "total_loss", "next_latent_loss",
                "cue_repair_loss", "repair_weight", "lr"))
            writer.writeheader()
            for epoch in range(1, int(repair["epochs"]) + 1):
                total, base, cue = _train_epoch(
                    host, carrier, decoder, optimizer,
                    np.asarray(train["z"], dtype=np.float32),
                    np.asarray(train["actions"], dtype=np.float32),
                    normalized_train,
                    batch_size=int(repair["batch_size"]),
                    next_latent_weight=float(repair["next_latent_weight"]),
                    repair_weight=repair_weight, rng=rng, device=device)
                scheduler.step()
                row = {
                    "epoch": epoch, "total_loss": total,
                    "next_latent_loss": base, "cue_repair_loss": cue,
                    "repair_weight": repair_weight,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                rows.append(row)
                writer.writerow(row)
                stream.flush()

        # Validation is opened only after every optimization step completes.
        validation = _label_free_cache(validation_path)
        validation_target, validation_indices = cue_repair_target(validation)
        val_fused, val_prior = _outputs(
            carrier, np.asarray(validation["z"], dtype=np.float32),
            np.asarray(validation["actions"], dtype=np.float32), device)
        with torch.no_grad():
            val_reconstruction = decoder(torch.from_numpy(
                val_prior[:, 63]).to(device)).float().cpu().numpy()
        normalized_val = (validation_target - target_mean) / target_scale
        val_repair_mse = float(np.mean(np.square(
            val_reconstruction - normalized_val)))
        val_next_mse = prediction_mse(
            host, val_fused, np.asarray(validation["actions"], dtype=np.float32),
            np.asarray(validation["z"], dtype=np.float32), device)
        if state_digest(host) != host_before:
            raise RuntimeError("frozen official host changed during repair")
        metrics = {
            "schema_version": 1,
            "study": spec["study"],
            "task": args.task,
            "arm": args.arm,
            "condition": args.condition,
            "seed": args.seed,
            "device": device_name,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "spec": spec["_spec_record"],
            "parent_carrier_state_sha256": parent_state,
            "carrier_state_sha256": state_digest(carrier),
            "repair_head_initial_state_sha256": decoder_initial,
            "repair_head_state_sha256": state_digest(decoder),
            "official_host_state_sha256_before": host_before,
            "official_host_state_sha256_after": state_digest(host),
            "repair_weight": repair_weight,
            "next_latent_weight": float(repair["next_latent_weight"]),
            "epochs": int(repair["epochs"]),
            "batch_size": int(repair["batch_size"]),
            "learning_rate": float(repair["learning_rate"]),
            "weight_decay": float(repair["weight_decay"]),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "validation_next_latent_mse": val_next_mse,
            "validation_cue_repair_mse": val_repair_mse,
            "label_arrays_loaded": False,
            "label_values_consumed": False,
            "validation_used_for_optimization": False,
            "repair_read": repair["read"],
            "target_gradient": repair["target_gradient"],
            "target_frame_index_min": int(train_indices.min()),
            "target_frame_index_max": int(train_indices.max()),
            "validation_target_frame_index_max": int(validation_indices.max()),
            "final_frame_index": 63,
            "final_frame_excluded_from_target": bool(
                train_indices.max() < 63 and validation_indices.max() < 63),
            "source_caches": {
                "train": spec["parent"]["train_caches"][args.task],
                "validation": spec["parent"]["validation_caches"][args.task],
            },
            "last_epoch": rows[-1],
        }
        _write_json(stage / "metrics.json", metrics)
        torch.save({
            "carrier_state_dict": carrier.state_dict(),
            "repair_head_state_dict": decoder.state_dict(),
            "target_mean": torch.from_numpy(target_mean),
            "target_scale": torch.from_numpy(target_scale),
            "metrics": metrics,
        }, stage / "repair.pt")
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"[delayed-goal-repair] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
