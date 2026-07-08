#!/usr/bin/env python3
"""Build the fresh held-out TwoRoom color-to-waypoint execution deck."""

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

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text, stable_json, write_npz_with_sidecar,
)
from lewm.official_tasks.matched_memory import (  # noqa: E402
    balanced_joint_labels, render_joint_cue,
)
from lewm.official_tasks.native_sequence_hdf5 import (  # noqa: E402
    NativeSequenceHDF5, SequenceSelection, normalize_action_blocks,
)
from lewm.official_tasks.tworoom_downstream import (  # noqa: E402
    execute_waypoint, load_pinned_tworoom, make_tworoom_env,
)
from scripts.paper_a_evidence_age import configure_determinism  # noqa: E402
from scripts.paper_a_matched_color_spec import (  # noqa: E402
    DEFAULT_SHA, DEFAULT_SPEC, HOSTS, load_locked_spec, output_path,
    resolve_input_path, resolve_path, validate_device,
    v1_excluded_episode_indices,
)
from scripts.prepare_paper_a_matched_color import host_manifest_path  # noqa: E402
from scripts.prepare_paper_a_matched_host import (  # noqa: E402
    _encode_stream, _load_host,
)
from scripts.train_paper_a_matched_color import _load_base  # noqa: E402
from scripts.train_frozen_official_swap import state_digest  # noqa: E402


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


def _vendor(spec: dict):
    source = resolve_path(spec["tworoom_use"]["upstream_environment"]["path"])
    return load_pinned_tworoom(
        source.parents[3],
        spec["tworoom_use"]["upstream_environment"]["revision"])


def _fresh_selection(dataset: NativeSequenceHDF5, spec: dict
                     ) -> tuple[tuple[SequenceSelection, ...], set[int]]:
    excluded = set(v1_excluded_episode_indices(spec, "tworoom"))
    for split in ("train", "validation"):
        base = _load_base(spec, "tworoom", split)
        excluded.update(map(int, base["episode_index"]))
    use = spec["tworoom_use"]
    raw_span = 96
    eligible = np.flatnonzero(dataset.episode_lengths >= raw_span)
    available = np.asarray(
        [value for value in eligible if int(value) not in excluded],
        dtype=np.int64)
    episodes = int(use["heldout_episodes"])
    if len(available) < episodes:
        raise RuntimeError("insufficient fresh TwoRoom use episodes")
    selected = np.random.default_rng(int(use["split_seed"])).permutation(
        available)[:episodes]
    result = []
    for episode in selected:
        maximum = int(dataset.episode_lengths[int(episode)] - raw_span)
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(use["start_seed"]), int(episode)]))
        result.append(SequenceSelection(
            split="use", episode_index=int(episode),
            local_start=int(rng.integers(maximum + 1))))
    if set(map(int, selected)).intersection(excluded):
        raise RuntimeError("TwoRoom use selection overlaps a prior split")
    return tuple(result), excluded


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing Wave-1b use-deck writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    if deck_path(spec).exists() or gate_path(spec).exists():
        raise FileExistsError("Wave-1b use deck is already resolved")
    for host in HOSTS:
        path = host_manifest_path(spec, host)
        value = json.loads(path.read_text())
        if value.get("lock") != spec["_lock"] \
                or value.get("status") != "admitted" \
                or value.get("all_color_ages_admitted") is not True:
            raise RuntimeError(
                f"Wave-1b use stopped because {host} is not admitted")
    if not torch.cuda.is_available():
        raise RuntimeError("Wave-1b use deck requires physical GPU 0")
    configure_determinism(0)
    device = torch.device(args.device)
    use = spec["tworoom_use"]
    source = spec["inputs"]["tworoom"]
    dataset_record = source["dataset"]
    dataset = NativeSequenceHDF5(
        resolve_input_path(dataset_record),
        expected_sha256=dataset_record["sha256"],
        expected_size=int(dataset_record["size"]),
        state_key=source["state_key"])
    selections, excluded = _fresh_selection(dataset, spec)
    episodes = len(selections)
    labels = balanced_joint_labels(episodes, int(use["label_seed"]))
    frames = np.empty((episodes, 20, *dataset.pixel_shape), dtype=np.uint8)
    raw_actions = np.empty((episodes, 19, 10), dtype=np.float32)
    state = np.empty((episodes, 20, dataset.state_dim), dtype=np.float32)
    global_indices = np.empty((episodes, 20), dtype=np.int64)
    for row, native in enumerate(dataset.read_sequences(selections, 20)):
        frames[row] = native.frames
        raw_actions[row] = native.actions
        state[row] = native.state
        global_indices[row] = native.global_frame_indices
    mean, std, count = dataset.raw_action_statistics(ddof=1)
    actions = normalize_action_blocks(raw_actions, mean, std)
    cue_frames = np.empty((episodes, 3, *dataset.pixel_shape), dtype=np.uint8)
    for row in range(episodes):
        rendered = render_joint_cue(
            frames[row], int(labels.color[row]), int(labels.location[row]), 1, 3)
        cue_frames[row] = rendered[1:4]
    model = _load_host(spec, "tworoom", device)
    host_before = state_digest(model)

    def base_stream() -> Iterator[tuple[int, np.ndarray]]:
        for row in range(episodes):
            for step in range(20):
                yield row * 20 + step, frames[row, step]

    def cue_stream() -> Iterator[tuple[int, np.ndarray]]:
        for row in range(episodes):
            for step in range(3):
                yield row * 3 + step, cue_frames[row, step]

    batch = int(spec["cache"]["frame_batch_size"])
    z = _encode_stream(
        model, base_stream(), episodes * 20, batch, device).reshape(
            episodes, 20, 192)
    z_cue = _encode_stream(
        model, cue_stream(), episodes * 3, batch, device).reshape(
            episodes, 3, 192)
    z[:, 1:4] = z_cue
    host_after = state_digest(model)
    if host_before != host_after:
        raise RuntimeError("Wave-1b use encoding mutated the frozen host")
    position = state[:, 19, np.asarray(use["state_position_indices"], dtype=int)]
    goals = np.asarray(use["goal_waypoints"], dtype=np.float32)
    TwoRoomEnv, ExpertPolicy, commit = _vendor(spec)
    env = make_tworoom_env(TwoRoomEnv)
    success = np.empty((episodes, 4, 4), dtype=np.int8)
    distance = np.empty((episodes, 4, 4), dtype=np.float32)
    final_state = np.empty((episodes, 4, 2), dtype=np.float32)
    target_success = np.empty((episodes, 4), dtype=np.int8)
    replay = np.empty((episodes, 4), dtype=np.int8)
    for episode in range(episodes):
        for selected in range(4):
            outcome = execute_waypoint(
                env, ExpertPolicy, state=position[episode],
                target=goals[selected],
                max_steps=int(use["max_execution_steps"]),
                physics_seed=int(use["physics_seed"]))
            final = np.asarray(outcome["final_state"], dtype=np.float32)
            final_state[episode, selected] = final
            distance[episode, selected] = np.linalg.norm(
                goals - final[None], axis=1)
            success[episode, selected] = (
                distance[episode, selected] < float(use["success_radius"]))
            target_success[episode, selected] = int(outcome["success"])
            replay[episode, selected] = int(np.array_equal(
                outcome["reset_state"], position[episode]))
    rows = np.arange(episodes)
    oracle = success[rows, labels.color, labels.color]
    random_choice = np.random.default_rng(
        int(use["random_goal_seed"])).integers(0, 4, size=episodes)
    random_success = success[rows, random_choice, labels.color]
    per_class = [float(oracle[labels.color == value].mean())
                 for value in range(4)]
    offdiag = float(success[:, ~np.eye(4, dtype=np.bool_)].mean())
    replay_rate = float(replay.mean())
    admitted = (
        float(oracle.mean()) >= float(use["oracle_success_min"])
        and min(per_class) >= float(use["oracle_per_class_success_min"])
        and offdiag <= float(use["off_diagonal_false_success_max"])
        and replay_rate >= float(use["replay_fidelity_min"]))
    episode_index = np.asarray(
        [value.episode_index for value in selections], dtype=np.int64)
    local_start = np.asarray(
        [value.local_start for value in selections], dtype=np.int64)
    arrays = {
        "z": z, "actions": actions,
        "color_label": labels.color, "location_label": labels.location,
        "combination_label": labels.combination,
        "episode_index": episode_index, "local_start": local_start,
        "global_frame_indices": global_indices,
        "decision_position": position, "goal_waypoints": goals,
        "success_matrix": success, "distance_matrix": distance,
        "controller_target_success": target_success,
        "controller_final_state": final_state, "reset_replay": replay,
        "random_choice": random_choice.astype(np.int64),
    }
    record = write_npz_with_sidecar(deck_path(spec), arrays, {
        "schema": "paper_a_matched_color_tworoom_use_deck_v1",
        "study": spec["study"], "lock": spec["_lock"],
        "episodes": episodes, "cue_age": 15, "target": "color",
        "nuisance": "location", "physical_gpu": 0,
        "fresh_exclusion_count": len(excluded),
        "fresh_zero_overlap": not bool(set(episode_index).intersection(excluded)),
        "fixed_physics_seed": int(use["physics_seed"]),
    }, compression_level=int(spec["cache"]["compression_level"]))
    gate = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "status": "admitted" if admitted else "stopped-controller-gate",
        "admitted": bool(admitted), "episodes": episodes,
        "oracle_executed_success": float(oracle.mean()),
        "oracle_per_class_executed_success": per_class,
        "realized_random_executed_success": float(random_success.mean()),
        "off_diagonal_false_success": offdiag,
        "reset_replay_fidelity": replay_rate,
        "controller_selected_target_success": float(target_success.mean()),
        "oracle_success_min": use["oracle_success_min"],
        "oracle_per_class_success_min": use["oracle_per_class_success_min"],
        "off_diagonal_false_success_max": use[
            "off_diagonal_false_success_max"],
        "replay_fidelity_min": use["replay_fidelity_min"],
        "raw_action_mean": mean.tolist(), "raw_action_std_ddof1": std.tolist(),
        "raw_action_count": count.tolist(),
        "vendor_commit": commit, "vendor_clean": True,
        "frozen_host_sha256_before": host_before,
        "frozen_host_sha256_after": host_after,
        "frozen_host_unchanged": True,
        "deck": record,
    }
    atomic_text(gate_path(spec), stable_json(gate))
    if not admitted:
        raise SystemExit("Wave-1b TwoRoom color controller failed locked gate")
    print(f"[matched-color/use] admitted {gate_path(spec)}", flush=True)


if __name__ == "__main__":
    main()
