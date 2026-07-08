#!/usr/bin/env python3
"""Evaluate one Wave-1b carrier with the shared color waypoint consumer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier  # noqa: E402
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text, load_verified_npz, stable_json,
)
from scripts.evaluate_paper_a_tworoom_use import (  # noqa: E402
    _execution_result, _fit_consumer,
)
from scripts.paper_a_evidence_age import (  # noqa: E402
    configure_determinism, fixed_endpoint_features,
)
from scripts.paper_a_matched_color_spec import (  # noqa: E402
    ARMS, DEFAULT_SHA, DEFAULT_SPEC, SEEDS, load_locked_spec, output_path,
    resolve_input_path, sha256_file, validate_device,
)
from scripts.prepare_paper_a_matched_color import host_manifest_path  # noqa: E402
from scripts.prepare_paper_a_matched_color_use import deck_path, gate_path  # noqa: E402
from scripts.train_frozen_official_swap import state_digest  # noqa: E402
from scripts.train_official_pusht_carrier import _carrier_prior  # noqa: E402
from scripts.train_paper_a_matched_color import (  # noqa: E402
    _aligned_latent, _load_base, _load_cue, carrier_directory,
)


DECK_KEYS = {
    "z", "actions", "color_label", "location_label", "combination_label",
    "episode_index", "local_start", "global_frame_indices",
    "decision_position", "goal_waypoints", "success_matrix",
    "distance_matrix", "controller_target_success", "controller_final_state",
    "reset_replay", "random_choice",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def use_cell_directory(spec: Mapping[str, Any], arm: str, seed: int) -> Path:
    return output_path(spec, "use") / "cells" / arm / f"seed-{seed}"


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _load_deck(spec: dict) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path, gate_file = deck_path(spec), gate_path(spec)
    arrays, sidecar = load_verified_npz(path)
    gate = json.loads(gate_file.read_text())
    use = spec["tworoom_use"]
    expected_sidecar = {
        "schema": "paper_a_matched_color_tworoom_use_deck_v1",
        "study": spec["study"], "lock": spec["_lock"], "episodes": 480,
        "cue_age": 15, "target": "color", "nuisance": "location",
        "physical_gpu": 0, "fresh_zero_overlap": True,
        "fixed_physics_seed": 0,
    }
    if any(sidecar.get(key) != value
           for key, value in expected_sidecar.items()) \
            or set(arrays) != DECK_KEYS:
        raise ValueError("Wave-1b use deck schema differs")
    expected_gate = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "status": "admitted", "admitted": True, "episodes": 480,
        "oracle_success_min": use["oracle_success_min"],
        "oracle_per_class_success_min": use["oracle_per_class_success_min"],
        "off_diagonal_false_success_max": use[
            "off_diagonal_false_success_max"],
        "replay_fidelity_min": use["replay_fidelity_min"],
        "vendor_commit": use["upstream_environment"]["revision"],
        "vendor_clean": True, "frozen_host_unchanged": True,
    }
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise ValueError("Wave-1b use gate differs")
    labels = np.asarray(arrays["color_label"], dtype=np.int64)
    nuisance = np.asarray(arrays["location_label"], dtype=np.int64)
    joint = np.asarray(arrays["combination_label"], dtype=np.int64)
    if not np.array_equal(joint, labels * 4 + nuisance) \
            or not np.array_equal(
                np.bincount(joint, minlength=16), np.full(16, 30)):
        raise ValueError("Wave-1b use labels are not 16-way balanced")
    success = np.asarray(arrays["success_matrix"])
    distance = np.asarray(arrays["distance_matrix"])
    replay = np.asarray(arrays["reset_replay"])
    target_success = np.asarray(arrays["controller_target_success"])
    if success.shape != (480, 4, 4) \
            or distance.shape != (480, 4, 4) \
            or not np.array_equal(
                success, distance < float(use["success_radius"])) \
            or replay.shape != (480, 4) \
            or target_success.shape != (480, 4):
        raise ValueError("Wave-1b crossed physical matrices differ")
    rows = np.arange(480)
    random_choice = np.asarray(arrays["random_choice"], dtype=np.int64)
    oracle = success[rows, labels, labels]
    random = success[rows, random_choice, labels]
    per_class = [float(oracle[labels == value].mean()) for value in range(4)]
    offdiag = float(success[:, ~np.eye(4, dtype=np.bool_)].mean())
    checks = {
        "oracle_executed_success": float(oracle.mean()),
        "realized_random_executed_success": float(random.mean()),
        "off_diagonal_false_success": offdiag,
        "reset_replay_fidelity": float(replay.mean()),
        "controller_selected_target_success": float(target_success.mean()),
    }
    if any(abs(float(gate.get(key, -1)) - value) > 1e-12
           for key, value in checks.items()) \
            or not np.allclose(
                gate.get("oracle_per_class_executed_success"), per_class,
                rtol=0, atol=1e-12):
        raise ValueError("Wave-1b controller gate does not reproduce")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    provenance = {
        "deck": {**_record(path), "sidecar": str(sidecar_path.relative_to(ROOT)),
                 "sidecar_sha256": sha256_file(sidecar_path)},
        "gate": _record(gate_file),
    }
    return arrays, provenance


def _load_carrier(spec: dict, arm: str, seed: int,
                  device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    directory = carrier_directory(spec, "tworoom", arm, seed)
    paths = {
        "carrier_manifest": directory / "manifest.json",
        "carrier_checkpoint": directory / "carrier.pt",
        "carrier_metrics": directory / "metrics.json",
    }
    manifest = json.loads(paths["carrier_manifest"].read_text())
    metrics = json.loads(paths["carrier_metrics"].read_text())
    if manifest.get("lock") != spec["_lock"] \
            or manifest.get("host") != "tworoom" \
            or manifest.get("arm") != arm or manifest.get("seed") != seed \
            or metrics.get("branch") != "matched-color-only-fixed-endpoint" \
            or metrics.get("frozen_host_unchanged") is not True:
        raise ValueError(f"invalid Wave-1b source carrier {arm}/{seed}")
    artifacts = manifest.get("artifacts", {})
    for name, filename in (("metrics", "carrier_metrics"),
                           ("checkpoint", "carrier_checkpoint")):
        if artifacts.get(name, {}).get("sha256") \
                != sha256_file(paths[filename]):
            raise ValueError("Wave-1b carrier hash differs")
    payload = torch.load(
        paths["carrier_checkpoint"], map_location="cpu", weights_only=True)
    carrier = make_frozen_carrier(arm, 192, 10)
    carrier.load_state_dict(payload["carrier_state_dict"], strict=True)
    digest = state_digest(carrier)
    if digest != metrics.get("carrier_state_sha256"):
        raise ValueError("Wave-1b carrier state digest differs")
    carrier.requires_grad_(False)
    carrier = carrier.to(device).eval()
    return carrier, {name: _record(path) for name, path in paths.items()} | {
        "carrier_state_sha256": digest}


def _gpu0(device_name: str) -> torch.device:
    if device_name != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("Wave-1b use requires cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "0"):
        raise RuntimeError("Wave-1b logical cuda:0 must be physical GPU0")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing Wave-1b use-cell writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    destination = use_cell_directory(spec, args.arm, args.seed)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    configure_determinism(args.seed)
    device = _gpu0(args.device)
    deck, provenance = _load_deck(spec)
    base = _load_base(spec, "tworoom", "train")
    cue = _load_cue(spec, "tworoom", "train", 15)
    train_z = _aligned_latent(base, cue)
    train_actions = np.asarray(base["actions"], dtype=np.float32)
    train_labels = np.asarray(cue["color_label"], dtype=np.int64)
    deck_z = np.asarray(deck["z"], dtype=np.float32)
    deck_actions = np.asarray(deck["actions"], dtype=np.float32)
    pooled: list[np.ndarray] = []
    consumer_sources: dict[str, Any] = {}
    requested_x = None
    requested_source = None
    requested_before = requested_after = None
    for training_arm in ARMS:
        carrier, source = _load_carrier(
            spec, training_arm, args.seed, device)
        before = source.pop("carrier_state_sha256")
        prior = _carrier_prior(carrier, train_z, train_actions, device)
        pooled.append(fixed_endpoint_features(
            train_z, prior, 19, history=3))
        if training_arm == args.arm:
            deck_prior = _carrier_prior(
                carrier, deck_z, deck_actions, device)
            requested_x = fixed_endpoint_features(
                deck_z, deck_prior, 19, history=3)
            requested_source = source
            requested_before = before
        after = state_digest(carrier)
        if after != before:
            raise RuntimeError(f"use inference mutated {training_arm}")
        if training_arm == args.arm:
            requested_after = after
        consumer_sources[training_arm] = {
            **source, "carrier_state_sha256": before,
            "state_unchanged": True}
    assert requested_x is not None and requested_source is not None
    train_x = np.concatenate(pooled)
    pooled_labels = np.tile(train_labels, len(ARMS))
    prediction, consumer = _fit_consumer(
        train_x, pooled_labels, requested_x, spec["readout"])
    consumer.update({
        "training_arms": list(ARMS), "rows_per_arm": len(train_labels),
        "equal_arm_weighting": True, "shared_across_evaluation_arms": True,
        "training_sources": consumer_sources,
    })
    result = _execution_result(
        prediction, np.asarray(deck["color_label"], dtype=np.int64),
        np.asarray(deck["combination_label"], dtype=np.int64),
        np.asarray(deck["success_matrix"]),
        np.asarray(deck["distance_matrix"]),
        np.asarray(deck["random_choice"], dtype=np.int64))
    provenance.update(requested_source)
    weights = resolve_input_path(spec["inputs"]["tworoom"]["weights"])
    provenance["tworoom_host_weights"] = _record(weights)
    metrics = {
        "schema_version": 1, "study": spec["study"],
        "branch": "matched-color-tworoom-external-waypoint-use",
        "lock": spec["_lock"], "host": "tworoom",
        "target": "color", "nuisance": "location", "cue_age": 15,
        "arm": args.arm, "seed": args.seed,
        "device": str(device), "physical_gpu": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "arm_blind_consumer": True,
        "consumer_training_arms": list(ARMS),
        "consumer_rows": len(pooled_labels), "consumer_seed": args.seed,
        "consumer_state_sha256": consumer["parameter_sha256"],
        "consumer": consumer, "validation_labels_used_for_fitting": False,
        "frozen_host_instantiated": False,
        "carrier_state_unchanged": True,
        "carrier_state_sha256_before": requested_before,
        "carrier_state_sha256_after": requested_after,
        "provenance": provenance, **result,
    }
    destination.mkdir(parents=True, exist_ok=False)
    metrics_path = destination / "metrics.json"
    digest = atomic_text(metrics_path, stable_json(metrics))
    manifest = {
        "schema_version": 1, "study": spec["study"],
        "branch": metrics["branch"], "lock": spec["_lock"],
        "host": "tworoom", "arm": args.arm, "seed": args.seed,
        "physical_gpu": 0,
        "artifacts": {"metrics": {"path": "metrics.json", "sha256": digest}},
    }
    atomic_text(destination / "manifest.json", stable_json(manifest))
    print(f"[matched-color/use] {args.arm}/s{args.seed}: "
          f"goal={metrics['goal_selection_accuracy']:.4f}, "
          f"executed={metrics['executed_success_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
