#!/usr/bin/env python3
"""Evaluate one V2 cell only after the development-selected controller is sealed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier
from lewm.models.official_lewm import OFFICIAL_ACTION_DIM, OFFICIAL_EMBED_DIM
from scripts.evaluate_paper_a_delayed_goal_use import (
    _consumer_kwargs,
    _load_repair_carrier,
    _source_record,
)
from scripts.paper_a_delayed_goal_spec import (
    CARRIER_ARMS,
    REPAIR_ARMS,
    REPAIR_CONDITIONS,
    SOURCE_IDS,
    load_locked_spec as load_v1_spec,
    source_slug,
)
from scripts.paper_a_delayed_goal_use import (
    action_time_interface,
    carrier_interface,
    cue_window_interface,
    decision_metrics,
    execute_reacher_choices,
    fit_shared_consumer,
    fit_shortcut_consumer,
    long_context_interface,
)
from scripts.paper_a_delayed_goal_v2_spec import (
    DEFAULT_SPEC,
    SEEDS,
    evaluation_directory,
    load_controller_lock,
    load_locked_spec,
    load_v1_provenance,
    resolve_path,
    sha256_file,
    validate_device,
)
from scripts.reevaluate_frozen_official_probes import (
    Cell,
    load_config,
    preflight_cell,
)
from scripts.train_frozen_official_swap import (
    carrier_outputs,
    load_cache,
    state_digest,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--task", required=True, choices=("t1", "t3"))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _cuda_device(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("V2 evaluation requires CUDA; no CPU fallback")
    index = int(device_name.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device is unavailable: {device_name}")
    return torch.device(device_name)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing V2 evaluation without explicit --execute")
    spec = load_locked_spec(args.spec)
    controller_lock, controller_record = load_controller_lock(spec)
    device_name = args.device or spec["execution"]["default_device"]
    validate_device(spec, device_name)
    device = _cuda_device(device_name)
    output_dir = evaluation_directory(spec, args.task, args.seed)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    v1_spec = load_v1_spec(resolve_path(spec["v1"]["spec"]["path"]))
    if v1_spec["_spec_record"] != spec["v1"]["spec"]:
        raise ValueError("loaded V1 spec differs from the V2 amendment parent")
    provenance = load_v1_provenance(spec, verify_artifacts=True)
    parent_config = load_config(resolve_path(spec["parent"]["config"]["path"]))
    train = load_cache(resolve_path(
        spec["parent"]["train_caches"][args.task]["path"]))
    validation = load_cache(resolve_path(
        spec["parent"]["validation_caches"][args.task]["path"]))
    train_labels = np.asarray(train["xi"], dtype=np.int64)
    validation_labels = np.asarray(validation["xi"], dtype=np.int64)
    train_features: dict[str, np.ndarray] = {}
    validation_features: dict[str, np.ndarray] = {}
    artifacts: dict[str, Any] = {}
    prepared: dict[str, Any] = {}

    for arm in CARRIER_ARMS:
        cell = preflight_cell(
            Cell(args.task, arm, args.seed),
            resolve_path(spec["parent"]["checkpoint_root"]), parent_config)
        prepared[arm] = cell
        carrier = make_frozen_carrier(
            arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)
        carrier.load_state_dict(cell.checkpoint["carrier_state_dict"], strict=True)
        before = state_digest(carrier)
        _, train_prior = carrier_outputs(
            carrier, np.asarray(train["z"], dtype=np.float32),
            np.asarray(train["actions"], dtype=np.float32), device)
        _, validation_prior = carrier_outputs(
            carrier, np.asarray(validation["z"], dtype=np.float32),
            np.asarray(validation["actions"], dtype=np.float32), device)
        if state_digest(carrier) != before:
            raise RuntimeError(f"V2 evaluation mutated frozen {arm} carrier")
        train_features[arm] = carrier_interface(train, train_prior)
        validation_features[arm] = carrier_interface(
            validation, validation_prior)
        checkpoint_path = cell.directory / "carrier.pt"
        artifacts[arm] = {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint_path),
            "carrier_state_sha256": cell.state_sha256,
            "source": "authenticated parent frozen-carrier cohort",
        }

    repair_initializations: dict[str, set[str]] = {
        arm: set() for arm in REPAIR_ARMS}
    repair_devices: dict[str, set[tuple[str, str]]] = {
        arm: set() for arm in REPAIR_ARMS}
    repair_hashes = {
        record["path"]: record["sha256"]
        for record in provenance["repair_checkpoints"]
    }
    for arm in REPAIR_ARMS:
        for condition in REPAIR_CONDITIONS:
            source = f"{arm}_{condition}"
            carrier, record = _load_repair_carrier(
                v1_spec, args.task, arm, args.seed, condition,
                prepared[arm].state_sha256, device)
            if repair_hashes.get(record["path"]) != record["sha256"]:
                raise ValueError(f"V2 repair provenance mismatch for {source}")
            repair_initializations[arm].add(
                record["repair_head_initial_state_sha256"])
            repair_devices[arm].add(
                (record["device"], record["cuda_device_name"]))
            before = state_digest(carrier)
            _, train_prior = carrier_outputs(
                carrier, np.asarray(train["z"], dtype=np.float32),
                np.asarray(train["actions"], dtype=np.float32), device)
            _, validation_prior = carrier_outputs(
                carrier, np.asarray(validation["z"], dtype=np.float32),
                np.asarray(validation["actions"], dtype=np.float32), device)
            if state_digest(carrier) != before:
                raise RuntimeError(f"V2 evaluation mutated {source}")
            train_features[source] = carrier_interface(train, train_prior)
            validation_features[source] = carrier_interface(
                validation, validation_prior)
            artifacts[source] = {**record, "source": "sealed V1 repair"}
    if any(len(values) != 1 for values in repair_initializations.values()) \
            or any(len(values) != 1 for values in repair_devices.values()):
        raise ValueError("sealed V1 repair twins fail pairing checks")

    train_features["long_context_56"] = long_context_interface(train)
    validation_features["long_context_56"] = long_context_interface(validation)
    train_features["cue_window"] = cue_window_interface(train)
    validation_features["cue_window"] = cue_window_interface(validation)
    consumer = fit_shared_consumer(
        train_features, train_labels, list(SOURCE_IDS),
        **_consumer_kwargs(v1_spec))
    permutation = np.random.default_rng(
        int(spec["controls"]["label_shuffle"]["seed_base"]) + args.seed
    ).permutation(len(train_labels))
    shuffled_consumer = fit_shared_consumer(
        train_features, train_labels, list(SOURCE_IDS),
        label_permutation=permutation, **_consumer_kwargs(v1_spec))
    shortcut_consumer = fit_shortcut_consumer(
        action_time_interface(train), train_labels, v1_spec["consumer"])

    # Physics never needs a render context; force the CPU backend so CUDA 0/3
    # cannot be touched through EGL while representation inference uses 1/2.
    os.environ["MUJOCO_GL"] = spec["execution"][
        "controller_selection_gl_backend"]
    execution = execute_reacher_choices(
        np.asarray(validation["endo_state"], dtype=np.float64)[:, 63],
        validation_labels, controller_lock["selected_protocol"])
    sources: dict[str, Any] = {}
    shuffled: dict[str, Any] = {}
    for source in SOURCE_IDS:
        slug = source_slug(v1_spec, source)
        sources[slug] = _source_record(
            v1_spec, source, decision_metrics(
                consumer.predict(validation_features[source]),
                validation_labels, execution))
        shuffled[slug] = _source_record(
            v1_spec, source, decision_metrics(
                shuffled_consumer.predict(validation_features[source]),
                validation_labels, execution))
    shortcut_metrics = decision_metrics(
        shortcut_consumer.predict(action_time_interface(validation)),
        validation_labels, execution)
    oracle_metrics = decision_metrics(
        validation_labels, validation_labels, execution)
    max_shuffled = max(record["metrics"]["goal_decision_accuracy"]
                       for record in shuffled.values())
    shortcut_limit = float(spec["controls"]["shortcut_accuracy_max"])
    validity = {
        "label_oracle_success_pass": (
            oracle_metrics["executed_success_rate"] >=
            float(spec["executed_choice"]["validation_oracle_success_min"])),
        "label_shuffle_accuracy_pass": max_shuffled <= shortcut_limit,
        "action_time_accuracy_pass": (
            shortcut_metrics["goal_decision_accuracy"] <= shortcut_limit),
        "max_label_shuffle_accuracy": max_shuffled,
        "shortcut_accuracy_limit": shortcut_limit,
    }
    validity["valid_for_use_claim"] = all(
        validity[key] for key in (
            "label_oracle_success_pass", "label_shuffle_accuracy_pass",
            "action_time_accuracy_pass"))
    result = {
        "schema_version": 1,
        "study": spec["study"],
        "spec": spec["_spec_record"],
        "task": args.task,
        "task_name": spec["tasks"][args.task]["name"],
        "task_slug": spec["tasks"][args.task]["slug"],
        "checkpoint_seed": args.seed,
        "controller_lock": controller_record,
        "selected_candidate_id": controller_lock["selected_candidate_id"],
        "selected_protocol": controller_lock["selected_protocol"],
        "v1_failure_provenance": spec["v1"]["provenance_manifest"],
        "v1_repairs_reused": True,
        "repair_retraining_performed": False,
        "consumer": {
            "digest": consumer.digest(),
            "shared_across_all_sources": True,
            "fit_split": "authenticated parent training bank only",
            "source_order": list(SOURCE_IDS),
            "validation_labels_available_during_fit": False,
        },
        "labels": validation_labels.tolist(),
        "sources": sources,
        "label_oracle": oracle_metrics,
        "controls": {
            "label_shuffle": shuffled,
            "label_shuffle_consumer_digest": shuffled_consumer.digest(),
            "training_episode_permutation": permutation.tolist(),
            "action_time": shortcut_metrics,
            "action_time_consumer_digest": shortcut_consumer.digest(),
        },
        "repair_pairing": {
            arm: {
                "shared_head_initialization": True,
                "repair_head_initial_state_sha256": next(iter(values)),
                "shared_cuda_device": True,
                "cuda_device": list(next(iter(repair_devices[arm]))),
            }
            for arm, values in repair_initializations.items()
        },
        "validity": validity,
        "artifacts": artifacts,
        "claim_boundary": spec["claim_boundary"],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (stage / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"[delayed-goal-v2] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
