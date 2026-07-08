#!/usr/bin/env python3
"""Train one unchanged frozen carrier on an admitted V3 semantic stage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (  # noqa: E402
    FROZEN_CARRIER_NAMES,
    make_frozen_carrier,
    parameter_report,
)
from lewm.models.official_lewm import (  # noqa: E402
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
    load_official_reacher_checkpoint,
)
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    load_verified_npz,
    sha256_file,
    stable_json,
    write_npz,
)
from lewm.official_tasks.shell_game_capacity_v3 import V3_SALIENCE  # noqa: E402
from lewm.official_tasks.shell_game_pipeline_v3 import (  # noqa: E402
    admission_path_v3,
    cache_manifest_path_v3,
    cache_path_v3,
    carrier_directory_v3,
    lock_receipt_v3,
    require_all_selected_salience_v3,
    require_selected_salience_v3,
    stage_contract_v3,
)
from lewm.official_tasks.shell_game_spec_v3 import (  # noqa: E402
    DEFAULT_LOCK_V3,
    DEFAULT_SPEC_V3,
    load_locked_spec_v3,
    resolve_path_v3,
    validate_device_v3,
)
from scripts.train_official_shell_game_capacity import (  # noqa: E402
    _carrier_outputs,
    _configure_determinism,
    _prediction_mse,
    _primary_probe,
    _state_digest,
    _train_epoch,
)


STAGES = ("single-item", "two-item", "four-item")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--arm", required=True, choices=FROZEN_CARRIER_NAMES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_V3)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_V3)
    return parser.parse_args(argv)


def _load_admitted_cache_v3(spec: dict, stage: str, split: str) -> dict:
    require_all_selected_salience_v3(spec)
    selection = require_selected_salience_v3(spec, stage)
    manifest_path = cache_manifest_path_v3(spec, stage)
    admission_file = admission_path_v3(spec, stage)
    if not manifest_path.is_file() or not admission_file.is_file():
        raise FileNotFoundError(f"V3 stage {stage} has no formal admission")
    manifest = json.loads(manifest_path.read_text())
    admission = json.loads(admission_file.read_text())
    record = manifest.get("admission", {})
    if record.get("sha256") != sha256_file(admission_file) \
            or record.get("admitted") is not True \
            or admission.get("admitted") is not True:
        raise RuntimeError(f"V3 stage {stage} did not pass unchanged gates")
    if manifest.get("formal_lock") != lock_receipt_v3(spec) \
            or admission.get("formal_lock") != lock_receipt_v3(spec) \
            or manifest.get("development_selection") != selection \
            or admission.get("development_selection") != selection \
            or manifest.get("threshold_changed_from_v1_or_v2") is not False \
            or admission.get("threshold_changed_from_v1_or_v2") is not False:
        raise ValueError("V3 admission was produced under a different amendment")
    arrays, sidecar = load_verified_npz(cache_path_v3(spec, stage, split))
    required = {
        "z", "actions", "initial_slots", "final_slots", "cue_on",
        "cue_off", "swap_pairs", "shuffle_off",
    }
    if set(arrays) != required:
        raise ValueError(f"V3 cache fields differ: {sorted(arrays)}")
    if sidecar.get("schema") != "official_shell_game_cache_v3" \
            or sidecar.get("stage") != stage \
            or sidecar.get("split") != split \
            or sidecar.get("formal_lock") != lock_receipt_v3(spec) \
            or sidecar.get("cue_salience") != V3_SALIENCE.describe() \
            or sidecar.get("development_selection") != selection:
        raise ValueError("V3 cache metadata differs from the locked amendment")
    if arrays["z"].shape[-1] != OFFICIAL_EMBED_DIM \
            or arrays["actions"].shape[-1] != OFFICIAL_ACTION_DIM:
        raise ValueError("V3 cache does not use official latent/action dimensions")
    arrays["meta"] = sidecar
    arrays["admission"] = admission
    return arrays


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec_v3(args.spec, args.lock)
    validate_device_v3(args.device)
    training = spec["carrier_training"]
    if args.arm not in training["arms"] or args.seed not in training["seeds"]:
        raise ValueError("requested V3 carrier cell lies outside the locked grid")

    train = _load_admitted_cache_v3(spec, args.stage, "train")
    validation = _load_admitted_cache_v3(spec, args.stage, "validation")
    output = carrier_directory_v3(spec, args.stage, args.arm, args.seed)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V3 carrier {output}")
    weights = resolve_path_v3(spec["official_host"]["weights_path"])
    if not weights.is_file() \
            or sha256_file(weights) != spec["official_host"]["weights_sha256"]:
        raise ValueError("official checkpoint differs from the V3 lock")
    if not torch.cuda.is_available():
        raise RuntimeError("formal V3 training requires an allowed CUDA device")
    _configure_determinism(args.seed)
    device = torch.device(args.device)
    model = load_official_reacher_checkpoint(weights, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    host_before = _state_digest(model)
    carrier = make_frozen_carrier(
        args.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
        writer.writeheader()
        if carrier.parameter_count() > 0:
            optimizer = torch.optim.AdamW(
                carrier.parameters(), lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
            # Retain V1's optimizer-window RNG convention exactly; only the
            # separately locked data/render amendment differs.
            rng = np.random.default_rng(71_000 + args.seed)
            for epoch in range(1, epochs + 1):
                loss = _train_epoch(
                    model, carrier, optimizer,
                    np.asarray(train["z"], dtype=np.float32),
                    np.asarray(train["actions"], dtype=np.float32),
                    batch_size, int(training["windows_per_episode"]),
                    rng, device)
                scheduler.step()
                row = {"epoch": epoch, "loss": loss,
                       "lr": optimizer.param_groups[0]["lr"]}
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                    print(f"[shell-game-v3-carrier] {args.stage}/{args.arm}/"
                          f"seed-{args.seed} {epoch}/{epochs} loss={loss:.6f}")

    _, train_prior = _carrier_outputs(
        carrier, np.asarray(train["z"], dtype=np.float32),
        np.asarray(train["actions"], dtype=np.float32), device)
    fused, validation_prior = _carrier_outputs(
        carrier, np.asarray(validation["z"], dtype=np.float32),
        np.asarray(validation["actions"], dtype=np.float32), device)
    prediction_mse = _prediction_mse(
        model, fused, np.asarray(validation["actions"], dtype=np.float32),
        np.asarray(validation["z"], dtype=np.float32), device)
    probe = _primary_probe(train, train_prior, validation, validation_prior)
    host_after = _state_digest(model)
    if host_before != host_after:
        raise RuntimeError("frozen official host changed during V3 training")

    selection = require_selected_salience_v3(spec, args.stage)
    metrics = {
        "schema": "official_shell_game_carrier_metrics_v3",
        "study": spec["study"],
        "stage": args.stage,
        "display_name": stage_contract_v3(args.stage).stage.display_name,
        "capacity": stage_contract_v3(args.stage).stage.capacity,
        "arm": args.arm,
        "seed": args.seed,
        "formal_lock": lock_receipt_v3(spec),
        "amendment": spec["amendment"]["kind"],
        "threshold_changed_from_v1_or_v2": False,
        "semantic_capacity_contract_changed_from_v1_or_v2": False,
        "carrier_definitions_changed_from_v1_or_v2": False,
        "cue_salience": V3_SALIENCE.describe(),
        "development_selection": selection,
        "official_checkpoint_sha256": spec["official_host"]["weights_sha256"],
        "official_host_state_sha256_before": host_before,
        "official_host_state_sha256_after": host_after,
        "frozen_host_unchanged": True,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(
            OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM),
        "carrier_config": carrier.describe(),
        "epochs": epochs if carrier.parameter_count() else 0,
        "batch_size": batch_size,
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "validation_next_latent_mse": prediction_mse,
        "primary_probe": probe,
        "admission_sha256": sha256_file(
            admission_path_v3(spec, args.stage)),
        "source_caches": {
            split: {
                "path": str(cache_path_v3(spec, args.stage, split)),
                "sha256": sha256_file(cache_path_v3(spec, args.stage, split)),
            }
            for split in ("train", "validation")
        },
    }
    metrics_path = output / "metrics.json"
    metrics_hash = atomic_text(metrics_path, stable_json(metrics))
    checkpoint_path = output / "carrier.pt"
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, checkpoint_path)
    export_path = output / "validation_export.npz"
    export_hash = write_npz(export_path, {
        "prior_read": validation_prior,
        "z": np.asarray(validation["z"], dtype=np.float32),
        "actions": np.asarray(validation["actions"], dtype=np.float32),
        "final_slots": np.asarray(validation["final_slots"], dtype=np.int64),
        "meta_json": np.asarray(json.dumps({
            "stage": args.stage,
            "display_name": stage_contract_v3(args.stage).stage.display_name,
            "arm": args.arm,
            "seed": args.seed,
            "amendment": spec["amendment"]["kind"],
        }, sort_keys=True)),
    })
    manifest_path = output / "manifest.json"
    atomic_text(manifest_path, stable_json({
        "schema": "official_shell_game_carrier_manifest_v3",
        "study": spec["study"],
        "stage": args.stage,
        "arm": args.arm,
        "seed": args.seed,
        "formal_lock": lock_receipt_v3(spec),
        "amendment": spec["amendment"]["kind"],
        "artifacts": {
            "metrics": {"path": metrics_path.name, "sha256": metrics_hash},
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
            },
            "validation_export": {
                "path": export_path.name,
                "sha256": export_hash,
            },
            "history": {
                "path": history_path.name,
                "sha256": sha256_file(history_path),
            },
        },
    }))
    print(json.dumps({
        "manifest": str(manifest_path),
        "mean_per_item_balanced_accuracy":
            probe["mean_per_item_balanced_accuracy"],
        "exact_set_accuracy": probe["exact_set_accuracy"],
        "per_item_chance": probe["per_item_chance"],
        "exact_set_chance": probe["exact_set_chance"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
