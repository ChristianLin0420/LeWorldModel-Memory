#!/usr/bin/env python3
"""Evaluate one matched-token carrier with shared TwoRoom consumer."""

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
from lewm.official_tasks.artifacts import atomic_text, load_verified_npz, stable_json  # noqa: E402
from scripts.evaluate_paper_a_tworoom_use import _execution_result, _fit_consumer  # noqa: E402
from scripts.paper_a_evidence_age import configure_determinism, fixed_endpoint_features  # noqa: E402
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    ARMS, DEFAULT_SHA, DEFAULT_SPEC, SEEDS, load_locked_spec, output_path,
    resolve_input_path, sha256_file, validate_device,
)
from scripts.prepare_paper_a_matched_token_use import deck_path, gate_path  # noqa: E402
from scripts.prepare_paper_a_matched_token import host_manifest_path  # noqa: E402
from scripts.train_frozen_official_swap import state_digest  # noqa: E402
from scripts.train_official_pusht_carrier import _carrier_prior  # noqa: E402
from scripts.train_paper_a_matched_token import (  # noqa: E402
    _aligned_latent, _load_base, _load_cue, carrier_directory,
)


DECK_KEYS = {"z", "actions", "token_label", "location_label",
             "combination_label", "episode_index", "local_start",
             "global_frame_indices", "decision_position", "goal_waypoints",
             "success_matrix", "distance_matrix", "controller_target_success",
             "controller_final_state", "reset_replay", "random_choice"}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def use_cell_directory(spec: Mapping[str, Any], arm: str, seed: int) -> Path:
    return output_path(spec, "use") / "cells" / arm / f"seed-{seed}"


def _record(path: Path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _deck(spec: dict):
    path, gate_file = deck_path(spec), gate_path(spec)
    arrays, sidecar = load_verified_npz(path)
    gate = json.loads(gate_file.read_text())
    use = spec["tworoom_use"]
    expected_sidecar = {
        "schema": "paper_a_matched_token_tworoom_use_deck_v1",
        "study": spec["study"], "lock": spec["_lock"], "episodes": 480,
        "target": "token", "nuisance": "location", "cue_age": 15,
        "fresh_exclusion_count": 3360, "fresh_zero_overlap": True,
        "physical_gpu": 0,
        "fixed_physics_seed": use["physics_seed"],
    }
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
    if set(arrays) != DECK_KEYS \
            or any(sidecar.get(key) != value
                   for key, value in expected_sidecar.items()) \
            or any(gate.get(key) != value
                   for key, value in expected_gate.items()):
        raise ValueError("matched-token use deck/gate differs")
    expected_shapes = {
        "z": (480, 20, 192), "actions": (480, 19, 10),
        "token_label": (480,), "location_label": (480,),
        "combination_label": (480,), "episode_index": (480,),
        "local_start": (480,), "global_frame_indices": (480, 20),
        "decision_position": (480, 2), "goal_waypoints": (4, 2),
        "success_matrix": (480, 4, 4), "distance_matrix": (480, 4, 4),
        "controller_target_success": (480, 4),
        "controller_final_state": (480, 4, 2),
        "reset_replay": (480, 4), "random_choice": (480,),
    }
    if any(np.asarray(arrays[key]).shape != shape
           for key, shape in expected_shapes.items()):
        raise ValueError("matched-token use deck shapes differ")
    if arrays["z"].dtype != np.float32 or arrays["actions"].dtype != np.float32:
        raise ValueError("matched-token use feature dtypes differ")
    for key in ("z", "actions", "decision_position", "goal_waypoints",
                "distance_matrix", "controller_final_state"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"matched-token deck {key} is non-finite")
    for key in ("token_label", "location_label", "combination_label",
                "episode_index", "local_start", "global_frame_indices",
                "random_choice"):
        if arrays[key].dtype != np.int64:
            raise ValueError(f"matched-token deck {key} dtype differs")
    if len(np.unique(arrays["episode_index"])) != 480 \
            or np.any(arrays["local_start"] < 0) \
            or not np.all(np.diff(arrays["global_frame_indices"], axis=1) == 5):
        raise ValueError("matched-token held-out sequence identity differs")
    labels = np.asarray(arrays["token_label"], dtype=np.int64)
    nuisance = np.asarray(arrays["location_label"], dtype=np.int64)
    joint = np.asarray(arrays["combination_label"], dtype=np.int64)
    if not np.array_equal(joint, labels * 4 + nuisance) \
            or not np.array_equal(np.bincount(joint, minlength=16),
                                  np.full(16, 30)) \
            or not np.isin(arrays["random_choice"], np.arange(4)).all():
        raise ValueError("matched-token use labels differ")
    frozen_before = gate.get("frozen_host_sha256_before")
    if not isinstance(frozen_before, str) or len(frozen_before) != 64 \
            or gate.get("frozen_host_sha256_after") != frozen_before:
        raise ValueError("matched-token use frozen-host hashes differ")
    admission = json.loads(host_manifest_path(spec, "tworoom").read_text())
    provenance = admission.get("provenance", {})
    if admission.get("lock") != spec["_lock"] \
            or admission.get("status") != "admitted" \
            or not np.array_equal(
                gate.get("raw_action_mean"), provenance.get("raw_action_mean")) \
            or not np.array_equal(
                gate.get("raw_action_std_ddof1"),
                provenance.get("raw_action_std_ddof1")) \
            or not np.array_equal(
                gate.get("raw_action_count"), provenance.get("raw_action_count")):
        raise ValueError("matched-token use normalization provenance differs")
    success, distance = arrays["success_matrix"], arrays["distance_matrix"]
    target_success, replay = (arrays["controller_target_success"],
                              arrays["reset_replay"])
    if not np.isin(success, (0, 1)).all() \
            or not np.isin(target_success, (0, 1)).all() \
            or not np.isin(replay, (0, 1)).all() \
            or np.any(distance < 0) \
            or not np.array_equal(
                success, distance < float(use["success_radius"])) \
            or not np.array_equal(
                target_success, success[:, np.arange(4), np.arange(4)]):
        raise ValueError("matched-token physical success differs")
    rows = np.arange(480)
    random = arrays["random_choice"]
    checks = {"oracle_executed_success": success[rows, labels, labels].mean(),
              "realized_random_executed_success": success[
                  rows, random, labels].mean(),
              "off_diagonal_false_success": success[
                  :, ~np.eye(4, dtype=bool)].mean(),
              "reset_replay_fidelity": arrays["reset_replay"].mean(),
              "controller_selected_target_success": arrays[
                  "controller_target_success"].mean()}
    if any(abs(float(gate[key]) - float(value)) > 1e-12
           for key, value in checks.items()):
        raise ValueError("matched-token gate does not reproduce")
    oracle = success[rows, labels, labels]
    per_class = np.asarray([
        oracle[labels == value].mean() for value in range(4)])
    if not np.allclose(
            gate.get("oracle_per_class_executed_success"), per_class,
            rtol=0, atol=1e-12) \
            or float(oracle.mean()) < float(use["oracle_success_min"]) \
            or float(per_class.min()) < float(
                use["oracle_per_class_success_min"]) \
            or float(checks["off_diagonal_false_success"]) > float(
                use["off_diagonal_false_success_max"]) \
            or float(checks["reset_replay_fidelity"]) < float(
                use["replay_fidelity_min"]):
        raise ValueError("matched-token gate thresholds do not pass")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    deck_record = gate.get("deck", {})
    expected_record = {
        "path": str(path), "sha256": sha256_file(path),
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
    }
    if any(deck_record.get(key) != value
           for key, value in expected_record.items()):
        raise ValueError("matched-token gate does not authenticate deck")
    sidecar_record = _record(sidecar_path)
    return arrays, {"deck": {**_record(path),
                             "sidecar": sidecar_record["path"],
                             "sidecar_sha256": sidecar_record["sha256"]},
                    "gate": _record(gate_file)}


def _carrier(spec: dict, arm: str, seed: int, device: torch.device):
    directory = carrier_directory(spec, "tworoom", arm, seed)
    paths = {"carrier_manifest": directory / "manifest.json",
             "carrier_checkpoint": directory / "carrier.pt",
             "carrier_metrics": directory / "metrics.json",
             "carrier_history": directory / "history.csv"}
    manifest = json.loads(paths["carrier_manifest"].read_text())
    metrics = json.loads(paths["carrier_metrics"].read_text())
    if manifest.get("schema_version") != 1 \
            or manifest.get("study") != spec["study"] \
            or manifest.get("lock") != spec["_lock"] \
            or manifest.get("host") != "tworoom" \
            or manifest.get("arm") != arm or manifest.get("seed") != seed \
            or metrics.get("schema_version") != 1 \
            or metrics.get("study") != spec["study"] \
            or metrics.get("lock") != spec["_lock"] \
            or metrics.get("host") != "tworoom" \
            or metrics.get("branch") \
            != "matched-composite-token-fixed-endpoint" \
            or metrics.get("arm") != arm or metrics.get("seed") != seed \
            or metrics.get("target") != "token" \
            or metrics.get("nuisance") != "location" \
            or metrics.get("frozen_host_unchanged") is not True \
            or metrics.get("validation_labels_used_for_fitting") is not False:
        raise ValueError("matched-token source carrier differs")
    artifacts = manifest.get("artifacts", {})
    expected_artifacts = {"metrics": "carrier_metrics",
                          "checkpoint": "carrier_checkpoint",
                          "history": "carrier_history"}
    if set(artifacts) != set(expected_artifacts):
        raise ValueError("source carrier artifact grid differs")
    for key, artifact in expected_artifacts.items():
        if artifacts[key].get("path") != paths[artifact].name:
            raise ValueError("source carrier artifact path differs")
        if manifest["artifacts"][key]["sha256"] != sha256_file(paths[artifact]):
            raise ValueError("source carrier hash differs")
    payload = torch.load(paths["carrier_checkpoint"], map_location="cpu",
                         weights_only=True)
    if not isinstance(payload, dict) \
            or set(payload) != {"carrier_state_dict", "metrics"} \
            or payload["metrics"] != metrics:
        raise ValueError("source carrier checkpoint payload differs")
    carrier = make_frozen_carrier(arm, 192, 10)
    carrier.load_state_dict(payload["carrier_state_dict"], strict=True)
    digest = state_digest(carrier)
    if digest != metrics["carrier_state_sha256"]:
        raise ValueError("source carrier state differs")
    carrier.requires_grad_(False)
    return carrier.to(device).eval(), {
        **{name: _record(path) for name, path in paths.items()},
        "carrier_state_sha256": digest}


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched-token use cell without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
        raise RuntimeError("matched-token use requires physical GPU0")
    configure_determinism(args.seed)
    device = torch.device(args.device)
    destination = use_cell_directory(spec, args.arm, args.seed)
    if destination.exists():
        raise FileExistsError(destination)
    deck, provenance = _deck(spec)
    base, cue = (_load_base(spec, "tworoom", "train"),
                 _load_cue(spec, "tworoom", "train", 15))
    train_z = _aligned_latent(base, cue)
    pooled, sources = [], {}
    requested_x = requested_source = before_requested = after_requested = None
    for arm in ARMS:
        carrier, source = _carrier(spec, arm, args.seed, device)
        before = source.pop("carrier_state_sha256")
        prior = _carrier_prior(carrier, train_z, base["actions"], device)
        pooled.append(fixed_endpoint_features(train_z, prior, 19, 3))
        if arm == args.arm:
            deck_prior = _carrier_prior(
                carrier, deck["z"], deck["actions"], device)
            requested_x = fixed_endpoint_features(deck["z"], deck_prior, 19, 3)
            requested_source, before_requested = source, before
        after = state_digest(carrier)
        if after != before:
            raise RuntimeError("use inference mutated carrier")
        if arm == args.arm:
            after_requested = after
        sources[arm] = {**source, "carrier_state_sha256": before,
                        "state_unchanged": True}
    prediction, consumer = _fit_consumer(
        np.concatenate(pooled), np.tile(cue["token_label"], 5),
        requested_x, spec["readout"])
    consumer.update({"training_arms": list(ARMS), "rows_per_arm": 1200,
                     "equal_arm_weighting": True,
                     "shared_across_evaluation_arms": True,
                     "training_sources": sources})
    result = _execution_result(
        prediction, deck["token_label"], deck["combination_label"],
        deck["success_matrix"], deck["distance_matrix"], deck["random_choice"])
    provenance.update(requested_source)
    provenance["tworoom_host_weights"] = _record(resolve_input_path(
        spec["inputs"]["tworoom"]["weights"]))
    metrics = {"schema_version": 1, "study": spec["study"],
               "branch": "matched-token-tworoom-external-waypoint-use",
               "lock": spec["_lock"], "host": "tworoom",
               "target": "token", "nuisance": "location", "cue_age": 15,
               "arm": args.arm, "seed": args.seed,
               "device": str(device), "physical_gpu": 0,
               "arm_blind_consumer": True,
               "consumer_training_arms": list(ARMS), "consumer_rows": 6000,
               "consumer_seed": args.seed,
               "consumer_state_sha256": consumer["parameter_sha256"],
               "consumer": consumer, "validation_labels_used_for_fitting": False,
               "carrier_state_unchanged": True,
               "carrier_state_sha256_before": before_requested,
               "carrier_state_sha256_after": after_requested,
               "provenance": provenance, **result}
    destination.mkdir(parents=True, exist_ok=False)
    metrics_path = destination / "metrics.json"
    digest = atomic_text(metrics_path, stable_json(metrics))
    atomic_text(destination / "manifest.json", stable_json({
        "schema_version": 1, "study": spec["study"], "branch": metrics["branch"],
        "lock": spec["_lock"], "host": "tworoom", "arm": args.arm,
        "seed": args.seed, "physical_gpu": 0,
        "artifacts": {"metrics": {"path": "metrics.json", "sha256": digest}}}))


if __name__ == "__main__":
    main()
