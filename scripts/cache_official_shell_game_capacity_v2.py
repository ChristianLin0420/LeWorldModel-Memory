#!/usr/bin/env python3
"""Cache formal frozen features and evaluate unchanged V1 gates for V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

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
    sha256_file,
    stable_json,
    write_npz_with_sidecar,
)
from lewm.official_tasks.shell_game_admission import (  # noqa: E402
    ShellGameAdmissionThresholds,
    evaluate_frozen_admission_inputs,
)
from lewm.official_tasks.shell_game_capacity import build_admission_inputs  # noqa: E402
from lewm.official_tasks.shell_game_capacity_v2 import V2_SALIENCE  # noqa: E402
from lewm.official_tasks.shell_game_pipeline_v2 import (  # noqa: E402
    admission_path_v2,
    cache_manifest_path_v2,
    cache_path_v2,
    load_counterfactual_audit_v2,
    load_stage_v2,
    lock_receipt_v2,
    require_selected_salience_v2,
    stage_path_v2,
)
from lewm.official_tasks.shell_game_spec_v2 import (  # noqa: E402
    DEFAULT_LOCK_V2,
    DEFAULT_SPEC_V2,
    FORMAL_SPLITS_V2,
    load_locked_spec_v2,
    resolve_path_v2,
    validate_device_v2,
)
from scripts.cache_official_lewm import (  # noqa: E402
    action_statistics,
    configure_determinism,
    encode_frames,
    transform_actions,
)


STAGES = ("single-item", "two-item", "four-item")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_V2)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_V2)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    return parser.parse_args(argv)


def _cache_arrays(z: np.ndarray, actions: np.ndarray, batch
                  ) -> dict[str, np.ndarray]:
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
    spec = load_locked_spec_v2(args.spec, args.lock)
    validate_device_v2(args.device)
    selection = require_selected_salience_v2(spec, args.stage)
    for split in FORMAL_SPLITS_V2:
        destination = cache_path_v2(spec, args.stage, split)
        for path in (destination,
                     destination.with_suffix(destination.suffix + ".json")):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite V2 artifact {path}")
    for path in (admission_path_v2(spec, args.stage),
                 cache_manifest_path_v2(spec, args.stage)):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V2 artifact {path}")

    batches, sidecars, audits = {}, {}, {}
    for split in FORMAL_SPLITS_V2:
        batches[split], sidecars[split] = load_stage_v2(
            spec, args.stage, split)
        if sidecars[split].get("development_selection") != selection:
            raise ValueError(
                f"formal V2 stage has stale salience evidence: {args.stage}/{split}")
        audits[split] = load_counterfactual_audit_v2(
            spec, args.stage, split, batches[split], sidecars[split])
    action_mean, action_std = action_statistics(batches["train"].actions)

    weights = resolve_path_v2(spec["official_host"]["weights_path"])
    if not weights.is_file() \
            or sha256_file(weights) != spec["official_host"]["weights_sha256"]:
        raise ValueError("official checkpoint differs from the V2 lock")
    configure_determinism(0)
    if not torch.cuda.is_available():
        raise RuntimeError("formal V2 caching requires an allowed CUDA device")
    device = torch.device(args.device)
    model = load_official_reacher_checkpoint(weights, device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("official model must remain fully frozen")

    latents, records = {}, {}
    for split in FORMAL_SPLITS_V2:
        batch = batches[split]
        latents[split] = encode_frames(
            model, batch.frames, device, args.frame_batch_size,
            f"v2-formal/{args.stage}/{split}")
        if latents[split].shape[-1] != OFFICIAL_EMBED_DIM:
            raise ValueError("official encoder returned wrong latent width")
        actions = transform_actions(
            batch.actions, action_mean, action_std, "clean")
        if actions.shape[-1] != OFFICIAL_ACTION_DIM:
            raise ValueError("official action transform returned wrong width")
        metadata = {
            "schema": "official_shell_game_cache_v2",
            "study": spec["study"],
            "stage": args.stage,
            "display_name": batch.display_name,
            "capacity": batch.contract.stage.capacity,
            "split": split,
            "formal_lock": lock_receipt_v2(spec),
            "amendment": spec["amendment"]["kind"],
            "threshold_changed_from_v1": False,
            "cue_salience": V2_SALIENCE.describe(),
            "development_selection": selection,
            "source_stage": {
                "path": str(stage_path_v2(spec, args.stage, split)),
                "sha256": sidecars[split]["artifact"]["sha256"],
            },
            "official_checkpoint": {
                "path": str(weights),
                "sha256": spec["official_host"]["weights_sha256"],
                "source": spec["official_host"]["source"],
                "source_commit": spec["official_host"]["source_commit"],
            },
            "representation_frozen": True,
            "labels_used_only_for": "post-hoc admission and final readout",
            "action_transform": {
                "method": "train-only per-column zscore of native 5x2 block",
                "training_mean": [float(value) for value in action_mean],
                "training_std_ddof0": [float(value) for value in action_std],
            },
        }
        records[split] = write_npz_with_sidecar(
            cache_path_v2(spec, args.stage, split),
            _cache_arrays(latents[split], actions, batch), metadata,
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
        train_latents=latents["train"],
        train_inputs=build_admission_inputs(
            batches["train"],
            int(admission_spec["cue_probe_frames"]),
            int(admission_spec["post_shuffle_probe_frames"])),
        train_counterfactual_report=audits["train"]["audit"],
        validation_latents=latents["validation"],
        validation_inputs=build_admission_inputs(
            batches["validation"],
            int(admission_spec["cue_probe_frames"]),
            int(admission_spec["post_shuffle_probe_frames"])),
        validation_counterfactual_report=audits["validation"]["audit"],
        thresholds=thresholds,
    )
    admission.update({
        "schema": "official_shell_game_frozen_admission_v2",
        "study": spec["study"],
        "stage": args.stage,
        "formal_lock": lock_receipt_v2(spec),
        "amendment": spec["amendment"]["kind"],
        "threshold_changed_from_v1": False,
        "semantic_capacity_contract_changed_from_v1": False,
        "cue_salience": V2_SALIENCE.describe(),
        "development_selection": selection,
        "v1_failed_gate_evidence":
            spec["amendment"]["parent_v1"]["evidence"][args.stage],
        "official_checkpoint_sha256": spec["official_host"]["weights_sha256"],
        "cache_artifacts": records,
    })
    admission_hash = atomic_text(
        admission_path_v2(spec, args.stage), stable_json(admission))
    manifest = {
        "schema": "official_shell_game_cache_manifest_v2",
        "study": spec["study"],
        "stage": args.stage,
        "formal_lock": lock_receipt_v2(spec),
        "amendment": spec["amendment"]["kind"],
        "threshold_changed_from_v1": False,
        "development_selection": selection,
        "official_checkpoint_sha256": spec["official_host"]["weights_sha256"],
        "artifacts": records,
        "admission": {
            "path": str(admission_path_v2(spec, args.stage)),
            "sha256": admission_hash,
            "admitted": admission["admitted"],
        },
    }
    atomic_text(cache_manifest_path_v2(spec, args.stage), stable_json(manifest))
    print(json.dumps({
        "stage": args.stage,
        "admitted": admission["admitted"],
        "threshold_changed_from_v1": False,
        "gates": admission["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
