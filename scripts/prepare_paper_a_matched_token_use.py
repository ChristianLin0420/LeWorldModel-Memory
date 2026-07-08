#!/usr/bin/env python3
"""Prepare fresh held-out TwoRoom token-to-waypoint execution deck."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, stable_json, write_npz_with_sidecar  # noqa: E402
from lewm.official_tasks.matched_token import balanced_token_labels, render_token_cue  # noqa: E402
from lewm.official_tasks.native_sequence_hdf5 import NativeSequenceHDF5, SequenceSelection, normalize_action_blocks  # noqa: E402
from lewm.official_tasks.tworoom_downstream import execute_waypoint, load_pinned_tworoom, make_tworoom_env  # noqa: E402
from scripts.paper_a_evidence_age import configure_determinism  # noqa: E402
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    DEFAULT_SHA, DEFAULT_SPEC, HOSTS, load_locked_spec, output_path,
    resolve_input_path, resolve_path, validate_device, v1_excluded_episode_indices,
)
from scripts.prepare_paper_a_matched_host import _encode_stream, _load_host  # noqa: E402
from scripts.prepare_paper_a_matched_token import host_manifest_path  # noqa: E402
from scripts.train_frozen_official_swap import state_digest  # noqa: E402
from scripts.train_paper_a_matched_token import _load_base  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def deck_path(spec: dict) -> Path:
    return output_path(spec, "use") / "deck.npz"


def gate_path(spec: dict) -> Path:
    return output_path(spec, "use") / "deck_gate.json"


def _selection(dataset: NativeSequenceHDF5, spec: dict):
    excluded = set(v1_excluded_episode_indices(spec, "tworoom"))
    for split in ("train", "validation"):
        excluded.update(map(int, _load_base(
            spec, "tworoom", split)["episode_index"]))
    use = spec["tworoom_use"]
    available = np.asarray([value for value in np.flatnonzero(
        dataset.episode_lengths >= 96) if int(value) not in excluded])
    selected = np.random.default_rng(use["split_seed"]).permutation(
        available)[:480]
    result = []
    for episode in selected:
        maximum = int(dataset.episode_lengths[int(episode)] - 96)
        rng = np.random.default_rng(np.random.SeedSequence(
            [use["start_seed"], int(episode)]))
        result.append(SequenceSelection(
            "use", int(episode), int(rng.integers(maximum + 1))))
    if len(result) != 480 or set(map(int, selected)).intersection(excluded):
        raise RuntimeError("matched-token use selection is not fresh")
    return tuple(result), excluded


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched-token use writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    tworoom_admission = None
    for host in HOSTS:
        value = json.loads(host_manifest_path(spec, host).read_text())
        if value.get("status") != "admitted" \
                or value.get("all_token_ages_admitted") is not True \
                or value.get("lock") != spec["_lock"]:
            raise RuntimeError(f"token use stopped: {host} not admitted")
        if host == "tworoom":
            tworoom_admission = value
    if deck_path(spec).exists() or gate_path(spec).exists():
        raise FileExistsError("matched-token use deck exists")
    configure_determinism(0)
    device = torch.device(args.device)
    source, use = spec["inputs"]["tworoom"], spec["tworoom_use"]
    record = source["dataset"]
    dataset = NativeSequenceHDF5(
        resolve_input_path(record), expected_sha256=record["sha256"],
        expected_size=record["size"], state_key=source["state_key"])
    selections, excluded = _selection(dataset, spec)
    labels = balanced_token_labels(480, use["label_seed"])
    frames = np.empty((480, 20, *dataset.pixel_shape), dtype=np.uint8)
    raw_actions = np.empty((480, 19, 10), dtype=np.float32)
    state = np.empty((480, 20, dataset.state_dim), dtype=np.float32)
    global_indices = np.empty((480, 20), dtype=np.int64)
    for row, native in enumerate(dataset.read_sequences(selections, 20)):
        frames[row], raw_actions[row], state[row] = (
            native.frames, native.actions, native.state)
        global_indices[row] = native.global_frame_indices
    mean, std, count = dataset.raw_action_statistics(ddof=1)
    admitted_provenance = tworoom_admission["provenance"]
    if not np.array_equal(
            mean, np.asarray(admitted_provenance["raw_action_mean"])) \
            or not np.array_equal(
                std, np.asarray(admitted_provenance["raw_action_std_ddof1"])) \
            or not np.array_equal(
                count, np.asarray(admitted_provenance["raw_action_count"])):
        raise RuntimeError(
            "TwoRoom use action normalization differs from admitted host")
    actions = normalize_action_blocks(raw_actions, mean, std)
    cue_frames = np.empty((480, 3, *dataset.pixel_shape), dtype=np.uint8)
    for row in range(480):
        cue_frames[row] = render_token_cue(
            frames[row], int(labels.token[row]), int(labels.location[row]),
            1, 3)[1:4]
    model = _load_host(spec, "tworoom", device)
    before = state_digest(model)

    def base_stream() -> Iterator[tuple[int, np.ndarray]]:
        for row in range(480):
            for step in range(20):
                yield row * 20 + step, frames[row, step]

    def cue_stream() -> Iterator[tuple[int, np.ndarray]]:
        for row in range(480):
            for step in range(3):
                yield row * 3 + step, cue_frames[row, step]

    z = _encode_stream(model, base_stream(), 9600, 128, device).reshape(
        480, 20, 192)
    z[:, 1:4] = _encode_stream(
        model, cue_stream(), 1440, 128, device).reshape(480, 3, 192)
    after = state_digest(model)
    if before != after:
        raise RuntimeError("token use encoding changed frozen host")
    positions = state[:, 19, :2]
    goals = np.asarray(use["goal_waypoints"], dtype=np.float32)
    env_source = resolve_path(use["upstream_environment"]["path"])
    TwoRoomEnv, ExpertPolicy, commit = load_pinned_tworoom(
        env_source.parents[3], use["upstream_environment"]["revision"])
    env = make_tworoom_env(TwoRoomEnv)
    success = np.empty((480, 4, 4), dtype=np.int8)
    distance = np.empty((480, 4, 4), dtype=np.float32)
    target_success = np.empty((480, 4), dtype=np.int8)
    final_state = np.empty((480, 4, 2), dtype=np.float32)
    replay = np.empty((480, 4), dtype=np.int8)
    for episode in range(480):
        for selected in range(4):
            outcome = execute_waypoint(
                env, ExpertPolicy, state=positions[episode],
                target=goals[selected], max_steps=use["max_execution_steps"],
                physics_seed=use["physics_seed"])
            final = np.asarray(outcome["final_state"], dtype=np.float32)
            final_state[episode, selected] = final
            distance[episode, selected] = np.linalg.norm(
                goals - final[None], axis=1)
            success[episode, selected] = (
                distance[episode, selected] < float(use["success_radius"]))
            target_success[episode, selected] = outcome["success"]
            replay[episode, selected] = np.array_equal(
                outcome["reset_state"], positions[episode])
    rows = np.arange(480)
    oracle = success[rows, labels.token, labels.token]
    random_choice = np.random.default_rng(use["random_goal_seed"]).integers(
        0, 4, size=480)
    random = success[rows, random_choice, labels.token]
    per_class = [float(oracle[labels.token == value].mean()) for value in range(4)]
    offdiag = float(success[:, ~np.eye(4, dtype=bool)].mean())
    admitted = (oracle.mean() >= use["oracle_success_min"]
                and min(per_class) >= use["oracle_per_class_success_min"]
                and offdiag <= use["off_diagonal_false_success_max"]
                and replay.mean() >= use["replay_fidelity_min"])
    episode_index = np.asarray([x.episode_index for x in selections], dtype=np.int64)
    local_start = np.asarray([x.local_start for x in selections], dtype=np.int64)
    arrays = {"z": z, "actions": actions,
              "token_label": labels.token, "location_label": labels.location,
              "combination_label": labels.combination,
              "episode_index": episode_index, "local_start": local_start,
              "global_frame_indices": global_indices,
              "decision_position": positions, "goal_waypoints": goals,
              "success_matrix": success, "distance_matrix": distance,
              "controller_target_success": target_success,
              "controller_final_state": final_state, "reset_replay": replay,
              "random_choice": random_choice.astype(np.int64)}
    deck_record = write_npz_with_sidecar(deck_path(spec), arrays, {
        "schema": "paper_a_matched_token_tworoom_use_deck_v1",
        "study": spec["study"], "lock": spec["_lock"], "episodes": 480,
        "target": "token", "nuisance": "location", "cue_age": 15,
        "fresh_exclusion_count": len(excluded),
        "fresh_zero_overlap": not bool(set(episode_index).intersection(excluded)),
        "physical_gpu": 0, "fixed_physics_seed": 0,
    }, compression_level=1)
    gate = {"schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
            "status": "admitted" if admitted else "stopped-controller-gate",
            "admitted": bool(admitted), "episodes": 480,
            "oracle_executed_success": float(oracle.mean()),
            "oracle_per_class_executed_success": per_class,
            "realized_random_executed_success": float(random.mean()),
            "off_diagonal_false_success": offdiag,
            "reset_replay_fidelity": float(replay.mean()),
            "controller_selected_target_success": float(target_success.mean()),
            "oracle_success_min": use["oracle_success_min"],
            "oracle_per_class_success_min": use["oracle_per_class_success_min"],
            "off_diagonal_false_success_max": use["off_diagonal_false_success_max"],
            "replay_fidelity_min": use["replay_fidelity_min"],
            "raw_action_mean": mean.tolist(), "raw_action_std_ddof1": std.tolist(),
            "raw_action_count": count.tolist(), "vendor_commit": commit,
            "vendor_clean": True, "frozen_host_sha256_before": before,
            "frozen_host_sha256_after": after, "frozen_host_unchanged": True,
            "deck": deck_record}
    atomic_text(gate_path(spec), stable_json(gate))
    if not admitted:
        raise SystemExit("matched-token TwoRoom controller gate failed")


if __name__ == "__main__":
    main()
