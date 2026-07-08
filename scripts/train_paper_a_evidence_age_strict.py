#!/usr/bin/env python3
"""Train one carrier on the locked age-balanced strict mixture."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier, parameter_report
from lewm.models.official_lewm import load_official_reacher_checkpoint
from lewm.models.official_lewm_pusht import load_official_pusht_checkpoint
from lewm.official_tasks.artifacts import (
    atomic_text,
    load_verified_npz,
    stable_json,
)
from scripts.paper_a_evidence_age import (
    combine_age_mixture,
    configure_determinism,
    fit_readout,
    fixed_endpoint_features,
)
from scripts.paper_a_evidence_age_spec import (
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    host_tasks,
    load_locked_spec,
    output_root,
    resolve_path,
    sha256_file,
    validate_device,
)
from scripts.prepare_paper_a_evidence_age_strict import strict_cache_path
from scripts.train_frozen_official_swap import state_digest, train_epoch
from scripts.train_official_pusht_carrier import _carrier_prior


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def carrier_directory(spec: dict[str, Any], host: str, task: str,
                      arm: str, seed: int) -> Path:
    return (output_root(spec, "strict") / "carriers" / host / task / arm
            / f"seed-{seed}")


def _load_age(spec: dict[str, Any], host: str, task: str,
              split: str, age: int) -> dict[str, np.ndarray]:
    path = strict_cache_path(spec, host, task, split, age)
    arrays, sidecar = load_verified_npz(path)
    expected = {
        "schema": "paper_a_strict_age_cache_v1", "study": spec["study"],
        "lock": spec["_lock"], "host": host, "task": task,
        "split": split, "age": age,
        "decision_index": 63 if host == "reacher" else 19,
        "paired_base_episode": True, "current_observation_excluded": True,
    }
    failed = [key for key, value in expected.items()
              if sidecar.get(key) != value]
    required = {"z", "actions", "labels", "cue_on", "cue_off",
                "episode_index"}
    if failed or set(arrays) != required:
        raise ValueError(f"strict cache mismatch {path}: {failed}")
    return arrays


def _load_host(spec: dict[str, Any], host: str,
               device: torch.device) -> torch.nn.Module:
    parent = spec["parents"][host]
    if host == "reacher":
        return load_official_reacher_checkpoint(
            resolve_path(parent["weights"]["path"]), device).eval()
    return load_official_pusht_checkpoint(
        resolve_path("outputs/paper_a_strengthening/pretrained/lewm-pusht"),
        device).eval()


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing strict carrier writes without --execute")
    spec = load_locked_spec(args.spec, args.sha)
    validate_device(spec, args.device)
    if args.task not in host_tasks(spec, args.host):
        raise ValueError(f"task {args.task!r} is not registered for {args.host}")
    if not torch.cuda.is_available():
        raise RuntimeError("strict carrier training requires cuda:0")
    manifest_path = (output_root(spec, "strict") / "cache" / args.host
                     / args.task / "manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"strict cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("lock") != spec["_lock"] \
            or manifest.get("status") != "admitted" \
            or manifest.get("all_ages_admitted") is not True:
        raise RuntimeError(f"strict branch is not admitted: {manifest_path}")
    destination = carrier_directory(
        spec, args.host, args.task, args.arm, args.seed)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    configure_determinism(args.seed)
    device = torch.device(args.device)
    ages = [int(value) for value in spec["strict_fixed_endpoint"]
            [args.host]["ages"]]
    train_by_age = {
        age: _load_age(spec, args.host, args.task, "train", age)
        for age in ages}
    validation_by_age = {
        age: _load_age(spec, args.host, args.task, "validation", age)
        for age in ages}
    train_z = combine_age_mixture(
        {age: value["z"] for age, value in train_by_age.items()}, ages)
    train_actions = combine_age_mixture(
        {age: value["actions"] for age, value in train_by_age.items()}, ages)
    model = _load_host(spec, args.host, device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("official strict host is not frozen")
    host_before = state_digest(model)
    carrier = make_frozen_carrier(args.arm, 192, 10).to(device)
    training = spec["strict_fixed_endpoint"]["carrier_training"]
    epochs = int(training["epochs"])
    rows = []
    if carrier.parameter_count():
        optimizer = torch.optim.AdamW(
            carrier.parameters(), lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        rng = np.random.default_rng(571_000 + args.seed)
        for epoch in range(1, epochs + 1):
            loss = train_epoch(
                model, carrier, optimizer, train_z, train_actions,
                int(training["batch_size"]), rng, device)
            scheduler.step()
            rows.append({"epoch": epoch, "loss": loss,
                         "lr": optimizer.param_groups[0]["lr"]})
            if epoch == 1 or epoch % 10 == 0:
                print(f"[evidence-age/strict] {args.host}/{args.task}/"
                      f"{args.arm}/s{args.seed} e{epoch}/{epochs} "
                      f"loss={loss:.6g}", flush=True)

    readouts = {}
    labels = None
    state_before = state_digest(carrier)
    for age in ages:
        train = train_by_age[age]
        validation = validation_by_age[age]
        train_prior = _carrier_prior(
            carrier, train["z"].astype(np.float32),
            train["actions"].astype(np.float32), device)
        validation_prior = _carrier_prior(
            carrier, validation["z"].astype(np.float32),
            validation["actions"].astype(np.float32), device)
        decision = 63 if args.host == "reacher" else 19
        result = fit_readout(
            fixed_endpoint_features(train["z"], train_prior, decision),
            train["labels"],
            fixed_endpoint_features(
                validation["z"], validation_prior, decision),
            validation["labels"], spec["read_time"]["readout"],
            balanced=args.host == "pusht")
        result["endpoint"] = {
            "evidence_age": age, "decision_index": decision,
            "context_indices": [decision - 3, decision - 2, decision - 1],
            "prior_index": decision, "current_observation_excluded": True,
        }
        readouts[f"age-{age}"] = result
        if labels is None:
            labels = np.asarray(validation["labels"], dtype=np.int64)
        elif not np.array_equal(labels, validation["labels"]):
            raise ValueError("strict validation labels differ across ages")
    if state_digest(carrier) != state_before:
        raise RuntimeError("strict evaluation mutated trained carrier")
    host_after = state_digest(model)
    if host_before != host_after:
        raise RuntimeError("strict carrier training mutated official host")

    destination.mkdir(parents=True, exist_ok=False)
    history_path = destination / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "loss", "lr"))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "schema_version": 1, "study": spec["study"],
        "branch": "strict-fixed-endpoint-cue-offset",
        "lock": spec["_lock"], "host": args.host, "task": args.task,
        "arm": args.arm, "seed": args.seed, "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cache_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
        },
        "ages": ages, "age_balanced_mixture": True,
        "training_episodes_per_age": int(len(next(iter(train_by_age.values()))["z"])),
        "training_episodes_total": int(len(train_z)),
        "epochs": epochs if carrier.parameter_count() else 0,
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(192, 10),
        "carrier_state_sha256": state_digest(carrier),
        "frozen_host_sha256_before": host_before,
        "frozen_host_sha256_after": host_after,
        "frozen_host_unchanged": True,
        "labels": labels.tolist(), "readouts": readouts,
        "validation_labels_used_for_fitting": False,
    }
    metrics_path = destination / "metrics.json"
    metrics_sha = atomic_text(metrics_path, stable_json(metrics))
    checkpoint_path = destination / "carrier.pt"
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, checkpoint_path)
    manifest_out = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "host": args.host, "task": args.task, "arm": args.arm,
        "seed": args.seed,
        "artifacts": {
            "metrics": {"path": "metrics.json", "sha256": metrics_sha},
            "checkpoint": {"path": "carrier.pt",
                           "sha256": sha256_file(checkpoint_path)},
            "history": {"path": "history.csv",
                        "sha256": sha256_file(history_path)},
        },
    }
    atomic_text(destination / "manifest.json", stable_json(manifest_out))
    print(f"[evidence-age/strict] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
