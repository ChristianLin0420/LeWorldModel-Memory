#!/usr/bin/env python3
"""Train one age-balanced carrier for matched composite-token recall."""

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

from lewm.models.frozen_swap_carriers import make_frozen_carrier, parameter_report  # noqa: E402
from lewm.official_tasks.artifacts import atomic_text, load_verified_npz, stable_json  # noqa: E402
from scripts.paper_a_evidence_age import (  # noqa: E402
    combine_age_mixture, configure_determinism, fit_readout,
    fixed_endpoint_features,
)
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    AGES, ARMS, DEFAULT_SHA, DEFAULT_SPEC, HOSTS, SEEDS, load_locked_spec,
    output_path, sha256_file, validate_device,
)
from scripts.prepare_paper_a_matched_host import _load_host  # noqa: E402
from scripts.prepare_paper_a_matched_token import (  # noqa: E402
    BASE_KEYS, BASE_SCHEMA, CUE_KEYS, CUE_SCHEMA, base_cache_path,
    cue_cache_path, host_manifest_path,
)
from scripts.train_frozen_official_swap import state_digest, train_epoch  # noqa: E402
from scripts.train_official_pusht_carrier import _carrier_prior  # noqa: E402


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


def _load_base(spec: Mapping[str, Any], host: str, split: str):
    path = base_cache_path(dict(spec), host, split)
    arrays, sidecar = load_verified_npz(path)
    expected = {"schema": BASE_SCHEMA, "study": spec["study"],
                "lock": spec["_lock"], "host": host, "split": split,
                "num_frames": 20, "decision_index": 19,
                "fresh_from_v1": True, "label_independent_base": True}
    if any(sidecar.get(k) != v for k, v in expected.items()) \
            or tuple(arrays) != BASE_KEYS:
        raise ValueError(f"matched-token base cache differs: {path}")
    return arrays


def _load_cue(spec: Mapping[str, Any], host: str, split: str, age: int):
    path = cue_cache_path(dict(spec), host, split, age)
    arrays, sidecar = load_verified_npz(path)
    expected = {"schema": CUE_SCHEMA, "study": spec["study"],
                "lock": spec["_lock"], "host": host, "split": split,
                "age": age, "decision_index": 19, "cue_length": 3,
                "target": "token", "nuisance": "location"}
    if any(sidecar.get(k) != v for k, v in expected.items()) \
            or tuple(arrays) != CUE_KEYS:
        raise ValueError(f"matched-token cue cache differs: {path}")
    return arrays


def _aligned_latent(base: Mapping[str, np.ndarray],
                    cue: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in ("episode_index", "local_start"):
        if not np.array_equal(base[key], cue[key]):
            raise ValueError("matched-token base/cue pairing differs")
    value = np.asarray(base["z_base"], dtype=np.float32).copy()
    start, stop = int(cue["cue_on"][0]), int(cue["cue_off"][0])
    value[:, start:stop] = cue["z_cue"]
    return value


def _nuisance(correct: np.ndarray, location: np.ndarray) -> dict[str, Any]:
    correct = np.asarray(correct, dtype=np.float64)
    values = [float(correct[location == group].mean()) for group in range(4)]
    return {"per_location_accuracy": values,
            "worst_location_accuracy": min(values),
            "best_minus_worst_location": max(values) - min(values)}


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched-token carrier writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    admission_path = host_manifest_path(spec, args.host)
    admission = json.loads(admission_path.read_text())
    if admission.get("lock") != spec["_lock"] \
            or admission.get("status") != "admitted" \
            or admission.get("all_token_ages_admitted") is not True \
            or admission.get("frozen_host_unchanged") is not True:
        raise RuntimeError(f"matched-token host is not admitted: {args.host}")
    fresh = (admission["provenance"].get("freshness", {}).get(
        "v1_trajectories_reused") is False if args.host == "reacher"
        else admission["provenance"].get(
            "v1_fresh_data_exclusion", {}).get("zero_overlap_proven") is True)
    if not fresh:
        raise RuntimeError("matched-token freshness proof is absent")
    destination = carrier_directory(spec, args.host, args.arm, args.seed)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    configure_determinism(args.seed)
    device = torch.device(args.device)
    base = {split: _load_base(spec, args.host, split)
            for split in ("train", "validation")}
    cues = {split: {age: _load_cue(spec, args.host, split, age)
                    for age in AGES} for split in ("train", "validation")}
    z = {split: {age: _aligned_latent(base[split], cues[split][age])
                 for age in AGES} for split in ("train", "validation")}
    train_z = combine_age_mixture(z["train"], AGES)
    train_actions = np.concatenate(
        [base["train"]["actions"] for _ in AGES])
    model = _load_host(spec, args.host, device)
    host_before = state_digest(model)
    carrier = make_frozen_carrier(args.arm, 192, 10).to(device)
    training = spec["carrier_training"]
    rows = []
    if carrier.parameter_count():
        optimizer = torch.optim.AdamW(
            carrier.parameters(), lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training["epochs"]))
        rng = np.random.default_rng(
            int(training["training_rng_offset"]) + args.seed)
        for epoch in range(1, int(training["epochs"]) + 1):
            loss = train_epoch(
                model, carrier, optimizer, train_z, train_actions,
                int(training["batch_size"]), rng, device)
            scheduler.step()
            rows.append({"epoch": epoch, "loss": loss,
                         "lr": optimizer.param_groups[0]["lr"]})
            if epoch == 1 or epoch % 10 == 0:
                print(f"[matched-token] {args.host}/{args.arm}/s{args.seed} "
                      f"e{epoch}/100 loss={loss:.6g}", flush=True)
    carrier_before_eval = state_digest(carrier)
    readouts = {}
    labels = cues["validation"][AGES[0]]["token_label"]
    nuisance = cues["validation"][AGES[0]]["location_label"]
    for age in AGES:
        train_prior = _carrier_prior(
            carrier, z["train"][age], base["train"]["actions"], device)
        val_prior = _carrier_prior(
            carrier, z["validation"][age],
            base["validation"]["actions"], device)
        result = fit_readout(
            fixed_endpoint_features(z["train"][age], train_prior, 19, 3),
            cues["train"][age]["token_label"],
            fixed_endpoint_features(z["validation"][age], val_prior, 19, 3),
            cues["validation"][age]["token_label"],
            spec["readout"], balanced=True)
        result["endpoint"] = {"evidence_age": age, "decision_index": 19,
                              "context_indices": [16, 17, 18],
                              "prior_index": 19,
                              "current_observation_excluded": True}
        result["nuisance_location"] = _nuisance(
            np.asarray(result["correct"]), nuisance)
        readouts[f"age-{age}"] = result
    if state_digest(carrier) != carrier_before_eval:
        raise RuntimeError("matched-token evaluation mutated carrier")
    host_after = state_digest(model)
    if host_before != host_after:
        raise RuntimeError("matched-token training mutated frozen host")
    destination.mkdir(parents=True, exist_ok=False)
    history_path = destination / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
        writer.writeheader(); writer.writerows(rows)
    metrics = {
        "schema_version": 1, "study": spec["study"],
        "branch": "matched-composite-token-fixed-endpoint",
        "lock": spec["_lock"], "host": args.host,
        "target": "token", "nuisance": "location",
        "arm": args.arm, "seed": args.seed,
        "device": str(device), "physical_gpu": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "host_manifest": {"path": str(admission_path.relative_to(ROOT)),
                          "sha256": sha256_file(admission_path)},
        "ages": list(AGES), "age_balanced_mixture": True,
        "training_episodes_per_age": 1200,
        "training_episodes_total": len(train_z),
        "epochs": 100 if carrier.parameter_count() else 0,
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(192, 10),
        "carrier_state_sha256": state_digest(carrier),
        "frozen_host_sha256_before": host_before,
        "frozen_host_sha256_after": host_after,
        "frozen_host_unchanged": True,
        "labels": labels.tolist(),
        "nuisance_location_labels": nuisance.tolist(),
        "joint_labels": cues["validation"][AGES[0]][
            "combination_label"].tolist(),
        "readouts": readouts,
        "validation_labels_used_for_fitting": False,
        "v1_metrics_used_for_inference": False,
    }
    metrics_path = destination / "metrics.json"
    metrics_hash = atomic_text(metrics_path, stable_json(metrics))
    checkpoint_path = destination / "carrier.pt"
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, checkpoint_path)
    manifest = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "host": args.host, "arm": args.arm, "seed": args.seed,
        "artifacts": {
            "metrics": {"path": "metrics.json", "sha256": metrics_hash},
            "checkpoint": {"path": "carrier.pt",
                           "sha256": sha256_file(checkpoint_path)},
            "history": {"path": "history.csv",
                        "sha256": sha256_file(history_path)},
        }}
    atomic_text(destination / "manifest.json", stable_json(manifest))
    print(f"[matched-token] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
