#!/usr/bin/env python3
"""Train one age-balanced carrier for adaptive Wave-1b color recall."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (  # noqa: E402
    make_frozen_carrier,
    parameter_report,
)
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    load_verified_npz,
    stable_json,
)
from scripts.paper_a_evidence_age import (  # noqa: E402
    combine_age_mixture,
    configure_determinism,
    fit_readout,
    fixed_endpoint_features,
)
from scripts.paper_a_matched_color_spec import (  # noqa: E402
    AGES,
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    load_locked_spec,
    output_path,
    sha256_file,
    validate_device,
)
from scripts.prepare_paper_a_matched_color import (  # noqa: E402
    base_cache_path,
    cue_cache_path,
    host_manifest_path,
)
from scripts.prepare_paper_a_matched_host import _load_host  # noqa: E402
from scripts.train_frozen_official_swap import state_digest, train_epoch  # noqa: E402
from scripts.train_official_pusht_carrier import _carrier_prior  # noqa: E402


BASE_KEYS = {
    "z_base", "actions", "state", "episode_index", "local_start",
    "global_frame_indices",
}
CUE_KEYS = {
    "z_cue", "combination_label", "color_label", "location_label",
    "episode_index", "local_start", "cue_on", "cue_off",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def carrier_directory(spec: Mapping[str, Any], host: str, arm: str,
                      seed: int) -> Path:
    return output_path(spec, "carriers") / host / arm / f"seed-{seed}"


def _load_base(spec: Mapping[str, Any], host: str,
               split: str) -> dict[str, np.ndarray]:
    path = base_cache_path(dict(spec), host, split)
    arrays, sidecar = load_verified_npz(path)
    expected = {
        "schema": "paper_a_matched_color_base_cache_v1",
        "study": spec["study"], "lock": spec["_lock"],
        "host": host, "split": split, "num_frames": 20,
        "decision_index": 19, "label_independent_base": True,
    }
    failed = [key for key, value in expected.items()
              if sidecar.get(key) != value]
    if failed or set(arrays) != BASE_KEYS:
        raise ValueError(f"Wave-1b base cache differs {path}: {failed}")
    return arrays


def _load_cue(spec: Mapping[str, Any], host: str, split: str,
              age: int) -> dict[str, np.ndarray]:
    path = cue_cache_path(dict(spec), host, split, age)
    arrays, sidecar = load_verified_npz(path)
    expected = {
        "schema": "paper_a_matched_color_cue_cache_v1",
        "study": spec["study"], "lock": spec["_lock"],
        "host": host, "split": split, "age": age,
        "decision_index": 19, "cue_length": 3,
        "target": "color",
        "location_role": "exact-balanced randomized nuisance",
    }
    failed = [key for key, value in expected.items()
              if sidecar.get(key) != value]
    if failed or set(arrays) != CUE_KEYS:
        raise ValueError(f"Wave-1b cue cache differs {path}: {failed}")
    cue_on = 19 - age - 3
    if not np.all(arrays["cue_on"] == cue_on) \
            or not np.all(arrays["cue_off"] == cue_on + 3):
        raise ValueError(f"Wave-1b cue timing differs at age {age}")
    return arrays


def _aligned_latent(base: Mapping[str, np.ndarray],
                    cue: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in ("episode_index", "local_start"):
        if not np.array_equal(base[key], cue[key]):
            raise ValueError(f"Wave-1b base/cue pairing differs for {key}")
    result = np.asarray(base["z_base"], dtype=np.float32).copy()
    cue_on, cue_off = int(cue["cue_on"][0]), int(cue["cue_off"][0])
    result[:, cue_on:cue_off] = cue["z_cue"]
    return result


def _nuisance_breakdown(correct: np.ndarray,
                        location: np.ndarray) -> dict[str, Any]:
    correct = np.asarray(correct, dtype=np.float64)
    location = np.asarray(location, dtype=np.int64)
    values = [float(correct[location == value].mean()) for value in range(4)]
    return {
        "location_role": "randomized balanced nuisance",
        "per_location_accuracy": values,
        "worst_location_accuracy": float(min(values)),
        "best_minus_worst_location": float(max(values) - min(values)),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing Wave-1b carrier writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("Wave-1b carrier training requires physical GPU 0")
    admission_path = host_manifest_path(spec, args.host)
    if not admission_path.is_file():
        raise FileNotFoundError(f"Wave-1b admission missing: {admission_path}")
    admission = json.loads(admission_path.read_text())
    if admission.get("lock") != spec["_lock"] \
            or admission.get("status") != "admitted" \
            or admission.get("all_color_ages_admitted") is not True \
            or admission.get("frozen_host_unchanged") is not True:
        raise RuntimeError(f"Wave-1b host is not admitted: {args.host}")
    provenance = admission.get("provenance", {})
    fresh = (provenance.get("freshness", {}).get(
        "wave1_trajectories_reused") is False if args.host == "reacher"
        else provenance.get("v1_fresh_data_exclusion", {}).get(
            "zero_overlap_proven") is True)
    if not fresh:
        raise RuntimeError(f"Wave-1b freshness proof is absent: {args.host}")
    destination = carrier_directory(spec, args.host, args.arm, args.seed)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    configure_determinism(args.seed)
    device = torch.device(args.device)
    base = {split: _load_base(spec, args.host, split)
            for split in ("train", "validation")}
    cue = {split: {age: _load_cue(spec, args.host, split, age)
                   for age in AGES}
           for split in ("train", "validation")}
    z = {split: {age: _aligned_latent(base[split], cue[split][age])
                 for age in AGES}
         for split in ("train", "validation")}
    for split in ("train", "validation"):
        reference = cue[split][AGES[0]]["combination_label"]
        for age in AGES[1:]:
            if not np.array_equal(
                    reference, cue[split][age]["combination_label"]):
                raise ValueError("Wave-1b joint labels differ across ages")

    train_z = combine_age_mixture(z["train"], AGES)
    train_actions = np.concatenate(
        [base["train"]["actions"] for _ in AGES], axis=0)
    model = _load_host(spec, args.host, device)
    host_before = state_digest(model)
    carrier = make_frozen_carrier(args.arm, 192, 10).to(device)
    training = spec["carrier_training"]
    epochs = int(training["epochs"])
    rows: list[dict[str, Any]] = []
    if carrier.parameter_count():
        optimizer = torch.optim.AdamW(
            carrier.parameters(), lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        rng = np.random.default_rng(
            int(training["training_rng_offset"]) + args.seed)
        for epoch in range(1, epochs + 1):
            loss = train_epoch(
                model, carrier, optimizer, train_z, train_actions,
                int(training["batch_size"]), rng, device)
            scheduler.step()
            rows.append({"epoch": epoch, "loss": loss,
                         "lr": optimizer.param_groups[0]["lr"]})
            if epoch == 1 or epoch % 10 == 0:
                print(f"[matched-color] {args.host}/{args.arm}/s{args.seed} "
                      f"e{epoch}/{epochs} loss={loss:.6g}", flush=True)

    readouts: dict[str, Any] = {}
    state_before_eval = state_digest(carrier)
    labels = cue["validation"][AGES[0]]["color_label"]
    nuisance = cue["validation"][AGES[0]]["location_label"]
    for age in AGES:
        train_prior = _carrier_prior(
            carrier, z["train"][age], base["train"]["actions"], device)
        validation_prior = _carrier_prior(
            carrier, z["validation"][age],
            base["validation"]["actions"], device)
        result = fit_readout(
            fixed_endpoint_features(
                z["train"][age], train_prior, 19, history=3),
            cue["train"][age]["color_label"],
            fixed_endpoint_features(
                z["validation"][age], validation_prior, 19, history=3),
            cue["validation"][age]["color_label"],
            spec["readout"], balanced=True)
        result["endpoint"] = {
            "evidence_age": age, "decision_index": 19,
            "context_indices": [16, 17, 18], "prior_index": 19,
            "current_observation_excluded": True,
        }
        result["nuisance_location"] = _nuisance_breakdown(
            np.asarray(result["correct"]), nuisance)
        readouts[f"age-{age}"] = result
    if state_digest(carrier) != state_before_eval:
        raise RuntimeError("Wave-1b evaluation mutated the carrier")
    host_after = state_digest(model)
    if host_before != host_after:
        raise RuntimeError("Wave-1b training mutated the frozen host")

    destination.mkdir(parents=True, exist_ok=False)
    history_path = destination / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "schema_version": 1, "study": spec["study"],
        "branch": "matched-color-only-fixed-endpoint",
        "adaptive_origin": "Wave-1 location availability stop",
        "lock": spec["_lock"], "host": args.host,
        "target": "color", "nuisance": "location",
        "arm": args.arm, "seed": args.seed,
        "device": str(device), "physical_gpu": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "host_manifest": {
            "path": str(admission_path.relative_to(ROOT)),
            "sha256": sha256_file(admission_path),
        },
        "ages": list(AGES), "age_balanced_mixture": True,
        "training_episodes_per_age": len(base["train"]["z_base"]),
        "training_episodes_total": len(train_z),
        "epochs": epochs if carrier.parameter_count() else 0,
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(192, 10),
        "carrier_state_sha256": state_digest(carrier),
        "frozen_host_sha256_before": host_before,
        "frozen_host_sha256_after": host_after,
        "frozen_host_unchanged": True,
        "labels": labels.tolist(),
        "nuisance_location_labels": nuisance.tolist(),
        "joint_labels": cue["validation"][AGES[0]][
            "combination_label"].tolist(),
        "readouts": readouts,
        "validation_labels_used_for_fitting": False,
        "v1_admission_metrics_used_for_inference": False,
    }
    metrics_path = destination / "metrics.json"
    metrics_hash = atomic_text(metrics_path, stable_json(metrics))
    checkpoint_path = destination / "carrier.pt"
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, checkpoint_path)
    manifest = {
        "schema_version": 1, "study": spec["study"],
        "lock": spec["_lock"], "host": args.host,
        "arm": args.arm, "seed": args.seed,
        "artifacts": {
            "metrics": {"path": "metrics.json", "sha256": metrics_hash},
            "checkpoint": {"path": "carrier.pt",
                           "sha256": sha256_file(checkpoint_path)},
            "history": {"path": "history.csv",
                        "sha256": sha256_file(history_path)},
        },
    }
    atomic_text(destination / "manifest.json", stable_json(manifest))
    print(f"[matched-color] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
