#!/usr/bin/env python3
"""Build the held-out TwoRoom execution deck after Wave-1 admission."""

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
    atomic_text,
    stable_json,
    write_npz_with_sidecar,
)
from lewm.official_tasks.matched_memory import (  # noqa: E402
    balanced_joint_labels,
    render_joint_cue,
)
from lewm.official_tasks.native_sequence_hdf5 import normalize_action_blocks  # noqa: E402
from lewm.official_tasks.tworoom_downstream import (  # noqa: E402
    execute_waypoint,
    load_pinned_tworoom,
    make_tworoom_env,
)
from scripts.paper_a_evidence_age import configure_determinism  # noqa: E402
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    load_locked_spec,
    output_path,
    resolve_path,
    validate_device,
)
from scripts.prepare_paper_a_matched_host import (  # noqa: E402
    _encode_stream,
    _load_host,
    host_manifest_path,
)


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


def _load_vendor(spec: dict):
    env_source = resolve_path(
        spec["tworoom_use"]["upstream_environment"]["path"])
    vendor = env_source.parents[3]
    return load_pinned_tworoom(
        vendor, spec["tworoom_use"]["upstream_environment"]["revision"])


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing TwoRoom use-deck writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    validate_device(spec, args.device)
    if deck_path(spec).exists() or gate_path(spec).exists():
        raise FileExistsError("TwoRoom use deck is already resolved")
    for host in HOSTS:
        path = host_manifest_path(spec, host)
        if not path.is_file():
            raise FileNotFoundError(f"matched admission missing: {host}")
        value = json.loads(path.read_text())
        if value.get("status") != "admitted" \
                or value.get("all_targets_ages_admitted") is not True:
            raise RuntimeError(f"TwoRoom use stopped because {host} was not admitted")
    if not torch.cuda.is_available():
        raise RuntimeError("TwoRoom deck encoding requires physical GPU 0")
    configure_determinism(0)
    device = torch.device(args.device)
    use = spec["tworoom_use"]
    episodes = int(use["heldout_episodes"])
    labels = balanced_joint_labels(episodes, int(use["label_seed"]))
    goals = np.asarray(use["goal_waypoints"], dtype=np.float32)
    prefix_target = np.asarray(use["shared_prefix_target"], dtype=np.float32)
    rng = np.random.default_rng(int(use["reset_seed"]))
    initial_y = rng.uniform(
        float(use["initial_y_range"][0]), float(use["initial_y_range"][1]),
        size=episodes).astype(np.float32)
    TwoRoomEnv, ExpertPolicy, commit = _load_vendor(spec)
    env = make_tworoom_env(TwoRoomEnv)
    policy = ExpertPolicy(action_noise=0.0, action_repeat_prob=0.0, seed=0)
    policy.set_env(env)
    frames = np.empty((episodes, 20, 224, 224, 3), dtype=np.uint8)
    raw_actions = np.empty((episodes, 19, 10), dtype=np.float32)
    decision_state = np.empty((episodes, 2), dtype=np.float32)
    for episode in range(episodes):
        initial = np.asarray([use["initial_x"], initial_y[episode]],
                             dtype=np.float32)
        _, info = env.reset(
            seed=int(use["physics_seed"]),
            options={"state": initial, "target_state": prefix_target})
        frames[episode, 0] = env.render()
        for transition in range(19):
            block = []
            for _ in range(5):
                action = policy.get_action(info)
                block.append(action)
                _, _, _, _, info = env.step(action)
            raw_actions[episode, transition] = np.concatenate(block)
            frames[episode, transition + 1] = env.render()
        decision_state[episode] = np.asarray(info["state"], dtype=np.float32)
    cue_frames = np.empty((episodes, 3, 224, 224, 3), dtype=np.uint8)
    for row in range(episodes):
        rendered = render_joint_cue(
            frames[row], int(labels.color[row]), int(labels.location[row]), 1, 3)
        cue_frames[row] = rendered[1:4]

    model = _load_host(spec, "tworoom", device)

    def base_stream() -> Iterator[tuple[int, np.ndarray]]:
        for row in range(episodes):
            for step in range(20):
                yield row * 20 + step, frames[row, step]

    def cue_stream() -> Iterator[tuple[int, np.ndarray]]:
        for row in range(episodes):
            for step in range(3):
                yield row * 3 + step, cue_frames[row, step]

    z = _encode_stream(
        model, base_stream(), episodes * 20,
        int(spec["cache"]["frame_batch_size"]), device).reshape(
            episodes, 20, 192)
    z_cue = _encode_stream(
        model, cue_stream(), episodes * 3,
        int(spec["cache"]["frame_batch_size"]), device).reshape(
            episodes, 3, 192)
    z[:, 1:4] = z_cue
    host_manifest = json.loads(host_manifest_path(spec, "tworoom").read_text())
    mean = np.asarray(host_manifest["provenance"]["raw_action_mean"])
    std = np.asarray(host_manifest["provenance"]["raw_action_std_ddof1"])
    actions = normalize_action_blocks(raw_actions, mean, std)

    # Axis 1 is the controller-selected waypoint; axis 2 is the true cued
    # waypoint against which the final physical state is scored.  This crossed
    # matrix is essential: success means executing the memory-conditioned
    # *correct* goal, not merely reaching whichever goal the readout selected.
    success = np.empty((episodes, 4, 4), dtype=np.int8)
    distance = np.empty((episodes, 4, 4), dtype=np.float32)
    controller_target_success = np.empty((episodes, 4), dtype=np.int8)
    controller_final_state = np.empty((episodes, 4, 2), dtype=np.float32)
    replay = np.empty((episodes, 4), dtype=np.int8)
    for episode in range(episodes):
        for selected_goal in range(4):
            outcome = execute_waypoint(
                env, ExpertPolicy, state=decision_state[episode],
                target=goals[selected_goal],
                max_steps=int(use["max_execution_steps"]),
                physics_seed=int(use["physics_seed"]))
            final = np.asarray(outcome["final_state"], dtype=np.float32)
            controller_final_state[episode, selected_goal] = final
            distance[episode, selected_goal] = np.linalg.norm(
                goals - final[None], axis=1)
            success[episode, selected_goal] = (
                distance[episode, selected_goal]
                < float(use["success_radius"])).astype(np.int8)
            controller_target_success[episode, selected_goal] = int(
                outcome["success"])
            replay[episode, selected_goal] = int(np.array_equal(
                outcome["reset_state"], decision_state[episode]))
    oracle = success[
        np.arange(episodes), labels.location, labels.location]
    random_choice = np.random.default_rng(
        int(use["reset_seed"]) + 1).integers(0, 4, size=episodes)
    random_success = success[
        np.arange(episodes), random_choice, labels.location]
    oracle_rate = float(oracle.mean())
    oracle_per_class = [
        float(oracle[labels.location == label].mean()) for label in range(4)
    ]
    off_diagonal = success[:, ~np.eye(4, dtype=np.bool_)].reshape(-1)
    false_success_rate = float(off_diagonal.mean())
    replay_rate = float(replay.mean())
    admitted = (oracle_rate >= float(use["oracle_success_min"])
                and min(oracle_per_class)
                >= float(use["oracle_per_class_success_min"])
                and false_success_rate
                <= float(use["off_diagonal_false_success_max"])
                and replay_rate >= float(use["replay_fidelity_min"]))
    arrays = {
        "z": z, "actions": actions,
        "color_label": labels.color, "location_label": labels.location,
        "combination_label": labels.combination,
        "decision_state": decision_state, "goal_waypoints": goals,
        "success_matrix": success, "distance_matrix": distance,
        "controller_target_success": controller_target_success,
        "controller_final_state": controller_final_state,
        "reset_replay": replay,
        "random_choice": random_choice.astype(np.int64),
    }
    record = write_npz_with_sidecar(deck_path(spec), arrays, {
        "schema": "paper_a_matched_tworoom_use_deck_v1",
        "study": spec["study"], "lock": spec["_lock"],
        "episodes": episodes, "cue_age": 15,
        "prefix_label_blind": True, "physical_gpu": 0,
        "fixed_physics_seed": int(use["physics_seed"]),
    }, compression_level=int(spec["cache"]["compression_level"]))
    gate = {
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "status": "admitted" if admitted else "stopped-controller-gate",
        "admitted": bool(admitted), "episodes": episodes,
        "oracle_executed_success": oracle_rate,
        "oracle_per_class_executed_success": oracle_per_class,
        "off_diagonal_false_success": false_success_rate,
        "realized_random_executed_success": float(random_success.mean()),
        "reset_replay_fidelity": replay_rate,
        "controller_selected_target_success": float(
            controller_target_success.mean()),
        "oracle_success_min": use["oracle_success_min"],
        "oracle_per_class_success_min": use[
            "oracle_per_class_success_min"],
        "off_diagonal_false_success_max": use[
            "off_diagonal_false_success_max"],
        "replay_fidelity_min": use["replay_fidelity_min"],
        "vendor_commit": commit, "vendor_clean": True, "deck": record,
    }
    atomic_text(gate_path(spec), stable_json(gate))
    if not admitted:
        raise SystemExit("TwoRoom external controller failed its locked gate")
    print(f"[matched-host/use] admitted {gate_path(spec)}", flush=True)


if __name__ == "__main__":
    main()
