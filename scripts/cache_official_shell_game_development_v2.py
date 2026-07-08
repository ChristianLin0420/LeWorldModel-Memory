#!/usr/bin/env python3
"""Run the development-only frozen salience gate for one V2 capacity stage."""

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
from lewm.official_tasks.shell_game_capacity import (  # noqa: E402
    build_admission_inputs,
)
from lewm.official_tasks.shell_game_capacity_v2 import V2_SALIENCE  # noqa: E402
from lewm.official_tasks.shell_game_pipeline_v2 import (  # noqa: E402
    development_selection_decision_v2,
    development_cache_path_v2,
    development_manifest_path_v2,
    development_receipt_path_v2,
    load_counterfactual_audit_v2,
    load_stage_v2,
    lock_receipt_v2,
    slice_admission_inputs_v2,
    stage_path_v2,
)
from lewm.official_tasks.shell_game_spec_v2 import (  # noqa: E402
    DEFAULT_LOCK_V2,
    DEFAULT_SPEC_V2,
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
    destination = development_cache_path_v2(spec, args.stage)
    paths = (
        destination,
        destination.with_suffix(destination.suffix + ".json"),
        development_receipt_path_v2(spec, args.stage),
        development_manifest_path_v2(spec, args.stage),
    )
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V2 artifact {path}")

    weights = resolve_path_v2(spec["official_host"]["weights_path"])
    if not weights.is_file() \
            or sha256_file(weights) != spec["official_host"]["weights_sha256"]:
        raise ValueError("official checkpoint differs from the V2 lock")
    configure_determinism(0)
    if not torch.cuda.is_available():
        raise RuntimeError("V2 development encoding requires an allowed CUDA device")
    device = torch.device(args.device)
    model = load_official_reacher_checkpoint(weights, device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("official model must remain fully frozen")

    batch, sidecar = load_stage_v2(spec, args.stage, "development")
    audit_receipt = load_counterfactual_audit_v2(
        spec, args.stage, "development", batch, sidecar)
    latents = encode_frames(
        model, batch.frames, device, args.frame_batch_size,
        f"v2-development/{args.stage}")
    if latents.shape[-1] != OFFICIAL_EMBED_DIM:
        raise ValueError("official encoder returned wrong latent width")
    mean, std = action_statistics(batch.actions)
    actions = transform_actions(batch.actions, mean, std, "clean")
    if actions.shape[-1] != OFFICIAL_ACTION_DIM:
        raise ValueError("official action transform returned wrong width")
    metadata = {
        "schema": "official_shell_game_development_cache_v2",
        "study": spec["study"],
        "stage": args.stage,
        "display_name": batch.display_name,
        "capacity": batch.contract.stage.capacity,
        "split": "development",
        "formal_lock": lock_receipt_v2(spec),
        "amendment": spec["amendment"]["kind"],
        "threshold_changed_from_v1": False,
        "cue_salience": V2_SALIENCE.describe(),
        "source_stage": {
            "path": str(stage_path_v2(spec, args.stage, "development")),
            "sha256": sidecar["artifact"]["sha256"],
        },
        "official_checkpoint": {
            "path": str(weights),
            "sha256": spec["official_host"]["weights_sha256"],
        },
        "representation_frozen": True,
        "formal_data_read": False,
        "labels_used_only_for": "development-only salience gate",
        "action_transform": {
            "method": "development-bank per-column zscore",
            "mean": [float(value) for value in mean],
            "std_ddof0": [float(value) for value in std],
        },
    }
    cache_record = write_npz_with_sidecar(
        destination, _cache_arrays(latents, actions, batch), metadata,
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
    full_inputs = build_admission_inputs(
        batch,
        int(admission_spec["cue_probe_frames"]),
        int(admission_spec["post_shuffle_probe_frames"]),
    )
    selection_spec = spec["development_selection"]
    fit_stop = int(selection_spec["fit_episodes"])
    check_stop = fit_stop + int(selection_spec["check_episodes"])
    fit_inputs = slice_admission_inputs_v2(full_inputs, 0, fit_stop)
    check_inputs = slice_admission_inputs_v2(
        full_inputs, fit_stop, check_stop)
    diagnostic = evaluate_frozen_admission_inputs(
        train_latents=latents[:fit_stop],
        train_inputs=fit_inputs,
        train_counterfactual_report=audit_receipt["audit"],
        validation_latents=latents[fit_stop:check_stop],
        validation_inputs=check_inputs,
        validation_counterfactual_report=audit_receipt["audit"],
        thresholds=thresholds,
    )
    decision = development_selection_decision_v2(
        diagnostic, float(selection_spec["threshold"]))
    selected = decision["selected"]
    receipt = {
        "schema": "official_shell_game_salience_selection_v2",
        "study": spec["study"],
        "stage": args.stage,
        "display_name": batch.display_name,
        "capacity": batch.contract.stage.capacity,
        "formal_lock": lock_receipt_v2(spec),
        "amendment": spec["amendment"]["kind"],
        "threshold_changed_from_v1": False,
        "semantic_capacity_contract_changed_from_v1": False,
        "formal_data_read": False,
        "development": {
            "episodes": int(batch.num_episodes),
            "fit_indices": [0, fit_stop],
            "check_indices": [fit_stop, check_stop],
            "base_seed": spec["data"]["development"]["base_seed"],
            "counterfactual_seed":
                spec["data"]["development"]["counterfactual_seed"],
        },
        "candidate": V2_SALIENCE.describe(),
        "candidate_count": 1,
        "criterion": {
            "metric": decision["metric"],
            "value": decision["value"],
            "per_item_accuracy": decision["per_item_accuracy"],
            "threshold": decision["threshold"],
            "direction": decision["direction"],
            "pass": decision["cue_pass"],
        },
        "exact_counterfactual_pass":
            decision["exact_counterfactual_pass"],
        "selected": selected,
        "on_failure": selection_spec["action_if_failed"],
        "v1_failed_gate_evidence":
            spec["amendment"]["parent_v1"]["evidence"][args.stage],
        "development_diagnostics": diagnostic,
    }
    receipt_path = development_receipt_path_v2(spec, args.stage)
    receipt_hash = atomic_text(receipt_path, stable_json(receipt))
    manifest = {
        "schema": "official_shell_game_development_manifest_v2",
        "study": spec["study"],
        "stage": args.stage,
        "formal_lock": lock_receipt_v2(spec),
        "development_cache": cache_record,
        "salience_selection": {
            "path": receipt_path.name,
            "sha256": receipt_hash,
            "selected": selected,
        },
        "formal_data_read": False,
    }
    atomic_text(
        development_manifest_path_v2(spec, args.stage), stable_json(manifest))
    print(json.dumps({
        "stage": args.stage,
        "selected": selected,
        "criterion": receipt["criterion"],
        "formal_data_read": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
