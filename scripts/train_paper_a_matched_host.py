#!/usr/bin/env python3
"""Train one shared age-balanced carrier for the matched-host audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
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
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    AGES,
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    TARGETS,
    load_locked_spec,
    output_path,
    sha256_file,
    validate_device,
)
from scripts.prepare_paper_a_matched_host import (  # noqa: E402
    base_cache_path,
    cue_cache_path,
    host_manifest_path,
    _load_host,
)
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


def carrier_directory(spec: dict[str, Any], host: str, arm: str,
                      seed: int) -> Path:
    return output_path(spec, "carriers") / host / arm / f"seed-{seed}"


def _load_base(spec: dict[str, Any], host: str,
               split: str) -> dict[str, np.ndarray]:
    path = base_cache_path(spec, host, split)
    arrays, sidecar = load_verified_npz(path)
    expected = {
        "schema": "paper_a_matched_base_cache_v1", "study": spec["study"],
        "lock": spec["_lock"], "host": host, "split": split,
        "num_frames": 20, "decision_index": 19,
        "label_independent_base": True,
    }
    failed = [key for key, value in expected.items()
              if sidecar.get(key) != value]
    if failed or set(arrays) != BASE_KEYS:
        raise ValueError(f"matched base cache schema mismatch {path}: {failed}")
    return arrays


def _load_cue(spec: dict[str, Any], host: str, split: str,
              age: int) -> dict[str, np.ndarray]:
    path = cue_cache_path(spec, host, split, age)
    arrays, sidecar = load_verified_npz(path)
    expected = {
        "schema": "paper_a_matched_cue_cache_v1", "study": spec["study"],
        "lock": spec["_lock"], "host": host, "split": split,
        "age": age, "decision_index": 19, "cue_length": 3,
        "same_rendered_episode_for_both_targets": True,
    }
    failed = [key for key, value in expected.items()
              if sidecar.get(key) != value]
    if failed or set(arrays) != CUE_KEYS:
        raise ValueError(f"matched cue cache schema mismatch {path}: {failed}")
    cue_on = 19 - age - 3
    if not np.all(arrays["cue_on"] == cue_on) \
            or not np.all(arrays["cue_off"] == cue_on + 3):
        raise ValueError(f"matched cue timing differs at age {age}")
    return arrays


def _aligned_latent(base: dict[str, np.ndarray],
                    cue: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("episode_index", "local_start"):
        if not np.array_equal(base[key], cue[key]):
            raise ValueError(f"base/cue pairing differs for {key}")
    result = np.asarray(base["z_base"], dtype=np.float32).copy()
    cue_on, cue_off = int(cue["cue_on"][0]), int(cue["cue_off"][0])
    result[:, cue_on:cue_off] = cue["z_cue"]
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched carrier writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("matched carrier training requires physical GPU 0")
    manifest_path = host_manifest_path(spec, args.host)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"host admission manifest missing: {manifest_path}")
    host_manifest = json.loads(manifest_path.read_text())
    if host_manifest.get("lock") != spec["_lock"] \
            or host_manifest.get("status") != "admitted" \
            or host_manifest.get("all_targets_ages_admitted") is not True:
        raise RuntimeError(f"host is not formally admitted: {args.host}")
    destination = carrier_directory(spec, args.host, args.arm, args.seed)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    configure_determinism(args.seed)
    device = torch.device(args.device)
    base = {split: _load_base(spec, args.host, split)
            for split in ("train", "validation")}
    cues = {split: {age: _load_cue(spec, args.host, split, age)
                    for age in AGES}
            for split in ("train", "validation")}
    z = {split: {age: _aligned_latent(base[split], cues[split][age])
                 for age in AGES}
         for split in ("train", "validation")}
    for split in ("train", "validation"):
        reference = cues[split][AGES[0]]["combination_label"]
        for age in AGES[1:]:
            if not np.array_equal(reference,
                                  cues[split][age]["combination_label"]):
                raise ValueError("joint labels differ across ages")

    train_z = combine_age_mixture(z["train"], AGES)
    train_actions = np.concatenate(
        [base["train"]["actions"] for _ in AGES], axis=0)
    model = _load_host(spec, args.host, device)
    host_before = state_digest(model)
    carrier = make_frozen_carrier(args.arm, 192, 10).to(device)
    training = spec["carrier_training"]
    epochs = int(training["epochs"])
    rows = []
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
                print(f"[matched-host] {args.host}/{args.arm}/s{args.seed} "
                      f"e{epoch}/{epochs} loss={loss:.6g}", flush=True)

    readouts: dict[str, dict[str, Any]] = {target: {} for target in TARGETS}
    state_before_eval = state_digest(carrier)
    labels: dict[str, list[int]] = {}
    for age in AGES:
        train_prior = _carrier_prior(
            carrier, z["train"][age], base["train"]["actions"], device)
        validation_prior = _carrier_prior(
            carrier, z["validation"][age],
            base["validation"]["actions"], device)
        for target in TARGETS:
            train_y = cues["train"][age][f"{target}_label"]
            validation_y = cues["validation"][age][f"{target}_label"]
            result = fit_readout(
                fixed_endpoint_features(
                    z["train"][age], train_prior, 19, history=3),
                train_y,
                fixed_endpoint_features(
                    z["validation"][age], validation_prior, 19, history=3),
                validation_y, spec["readout"], balanced=True)
            result["endpoint"] = {
                "evidence_age": age, "decision_index": 19,
                "context_indices": [16, 17, 18], "prior_index": 19,
                "current_observation_excluded": True,
            }
            readouts[target][f"age-{age}"] = result
            if target not in labels:
                labels[target] = validation_y.tolist()
            elif labels[target] != validation_y.tolist():
                raise ValueError("validation labels differ across ages")
    if state_digest(carrier) != state_before_eval:
        raise RuntimeError("readout evaluation mutated the carrier")
    host_after = state_digest(model)
    if host_before != host_after:
        raise RuntimeError("carrier training mutated the frozen host")

    destination.mkdir(parents=True, exist_ok=False)
    history_path = destination / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "schema_version": 1, "study": spec["study"],
        "branch": "matched-color-location-fixed-endpoint",
        "lock": spec["_lock"], "host": args.host,
        "arm": args.arm, "seed": args.seed,
        "device": str(device), "physical_gpu": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "host_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
        },
        "ages": list(AGES), "targets": list(TARGETS),
        "age_balanced_mixture": True,
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
        "labels": labels,
        "joint_labels": cues["validation"][AGES[0]][
            "combination_label"].tolist(),
        "readouts": readouts,
        "validation_labels_used_for_fitting": False,
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
    print(f"[matched-host] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
