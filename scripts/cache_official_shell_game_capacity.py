#!/usr/bin/env python3
"""Cache frozen official LeWM features and score one capacity admission gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm import (  # noqa: E402
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
    load_official_reacher_checkpoint,
)
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    sha256_arrays,
    sha256_file,
    stable_json,
    write_npz_with_sidecar,
)
from lewm.official_tasks.shell_game_admission import (  # noqa: E402
    ShellGameAdmissionThresholds,
    evaluate_frozen_admission_inputs,
)
from lewm.official_tasks.shell_game_capacity import (  # noqa: E402
    build_admission_inputs,
)
from lewm.official_tasks.shell_game_pipeline import (  # noqa: E402
    SPLITS,
    admission_path,
    audit_path,
    batch_arrays,
    cache_manifest_path,
    cache_path,
    load_stage,
    lock_receipt,
    stage_path,
)
from lewm.official_tasks.shell_game_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    load_locked_spec,
    resolve_path,
    validate_device,
)
from scripts.cache_official_lewm import (  # noqa: E402
    action_statistics,
    configure_determinism,
    encode_frames,
    transform_actions,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("single-item", "two-item", "four-item"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    return parser.parse_args(argv)


def _load_audit(spec: dict[str, Any], stage: str, split: str,
                batch, sidecar: dict) -> dict[str, Any]:
    path = audit_path(spec, stage, split)
    expected = sidecar.get("counterfactual_receipt", {})
    if path.name != expected.get("path") or sha256_file(path) != expected.get("sha256"):
        raise ValueError(f"counterfactual receipt mismatch for {stage}/{split}")
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "official_shell_game_counterfactual_receipt_v1" \
            or receipt.get("formal_lock") != lock_receipt(spec) \
            or receipt.get("audit", {}).get("overall_pass") is not True:
        raise ValueError(f"invalid counterfactual audit for {stage}/{split}")
    if receipt.get("primary_content_sha256") != sha256_arrays(batch_arrays(batch)):
        raise ValueError(f"primary content differs from audit for {stage}/{split}")
    return receipt


def _cache_arrays(z: np.ndarray, actions: np.ndarray, batch) -> dict[str, np.ndarray]:
    return {
        "z": z,
        "actions": actions,
        "initial_slots": batch.initial_slots,
        "final_slots": batch.final_slots,
        "cue_on": batch.cue_on,
        "cue_off": batch.cue_off,
        "swap_pairs": batch.swap_pairs,
        "shuffle_off": batch.shuffle_off,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive")
    spec = load_locked_spec(args.spec, args.lock)
    validate_device(args.device)
    for split in SPLITS:
        destination = cache_path(spec, args.stage, split)
        for path in (destination, destination.with_suffix(".npz.json")):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
    for path in (admission_path(spec, args.stage),
                 cache_manifest_path(spec, args.stage)):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    weights = resolve_path(spec["official_host"]["weights_path"])
    if not weights.is_file():
        raise FileNotFoundError(f"missing official checkpoint {weights}")
    weights_hash = sha256_file(weights)
    if weights_hash != spec["official_host"]["weights_sha256"]:
        raise ValueError("official checkpoint differs from locked hash")
    configure_determinism(0)
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("formal frozen caching requires an allowed CUDA device")
    model = load_official_reacher_checkpoint(weights, device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("official model must remain fully frozen")

    batches, sidecars, audits = {}, {}, {}
    for split in SPLITS:
        batches[split], sidecars[split] = load_stage(spec, args.stage, split)
        audits[split] = _load_audit(
            spec, args.stage, split, batches[split], sidecars[split])
    action_mean, action_std = action_statistics(batches["train"].actions)

    latent, records = {}, {}
    for split in SPLITS:
        batch = batches[split]
        latent[split] = encode_frames(
            model, batch.frames, device, args.frame_batch_size,
            f"{args.stage}/{split}")
        if latent[split].shape[-1] != OFFICIAL_EMBED_DIM:
            raise ValueError("official encoder returned wrong latent width")
        actions = transform_actions(
            batch.actions, action_mean, action_std, "clean")
        if actions.shape[-1] != OFFICIAL_ACTION_DIM:
            raise ValueError("official action transform returned wrong width")
        metadata = {
            "schema": "official_shell_game_cache_v1",
            "study": spec["study"],
            "stage": args.stage,
            "display_name": batch.display_name,
            "capacity": batch.contract.stage.capacity,
            "split": split,
            "formal_lock": lock_receipt(spec),
            "source_stage": {
                "path": str(stage_path(spec, args.stage, split)),
                "sha256": sidecars[split]["artifact"]["sha256"],
            },
            "official_checkpoint": {
                "path": str(weights),
                "sha256": weights_hash,
                "source": spec["official_host"]["source"],
                "source_commit": spec["official_host"]["source_commit"],
            },
            "representation_frozen": True,
            "labels_used_only_for": "post_hoc_admission_and_final_readout",
            "action_transform": {
                "method": "per_column_train_zscore_of_native_5x2_action_block",
                "training_mean": [float(value) for value in action_mean],
                "training_std_ddof0": [float(value) for value in action_std],
            },
        }
        records[split] = write_npz_with_sidecar(
            cache_path(spec, args.stage, split),
            _cache_arrays(latent[split], actions, batch), metadata,
            compression_level=int(spec["data"]["compression_level"]),
        )

    admission_spec = spec["admission"]
    thresholds = ShellGameAdmissionThresholds(
        cue_initial_slot_accuracy_min=float(
            admission_spec["cue_initial_slot_accuracy_min"]),
        swap_pair_accuracy_min=float(
            admission_spec["swap_pair_accuracy_min"]),
        leakage_margin_above_chance=float(
            admission_spec["leakage_margin_above_chance"]),
    )
    admission = evaluate_frozen_admission_inputs(
        train_latents=latent["train"],
        train_inputs=build_admission_inputs(
            batches["train"],
            int(admission_spec["cue_probe_frames"]),
            int(admission_spec["post_shuffle_probe_frames"])),
        train_counterfactual_report=audits["train"]["audit"],
        validation_latents=latent["validation"],
        validation_inputs=build_admission_inputs(
            batches["validation"],
            int(admission_spec["cue_probe_frames"]),
            int(admission_spec["post_shuffle_probe_frames"])),
        validation_counterfactual_report=audits["validation"]["audit"],
        thresholds=thresholds,
    )
    admission.update({
        "formal_lock": lock_receipt(spec),
        "official_checkpoint_sha256": weights_hash,
        "cache_artifacts": records,
    })
    admission_hash = atomic_text(
        admission_path(spec, args.stage), stable_json(admission))
    manifest = {
        "schema": "official_shell_game_cache_manifest_v1",
        "study": spec["study"],
        "stage": args.stage,
        "formal_lock": lock_receipt(spec),
        "official_checkpoint_sha256": weights_hash,
        "artifacts": records,
        "admission": {
            "path": str(admission_path(spec, args.stage)),
            "sha256": admission_hash,
            "admitted": admission["admitted"],
        },
    }
    atomic_text(cache_manifest_path(spec, args.stage), stable_json(manifest))
    print(json.dumps({
        "stage": args.stage,
        "admitted": admission["admitted"],
        "gates": admission["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
