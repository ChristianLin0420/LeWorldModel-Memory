#!/usr/bin/env python3
"""Collect memory-task banks at the official LeWM Reacher time scale.

The released Reacher model consumes one observation every five simulator
steps and a flattened 5x2 action block.  The older V19 banks contain one
simulator step per observation.  This script regenerates the same semantic
overlays with five independently sampled actions between rendered frames, so
the frozen official action encoder receives its native 10-D contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import EpisodeBatch, IMG_SIZE, save_bank

TASKS = ("t1", "t3", "t4")
TRAIN_EPISODES = 1200
VAL_EPISODES = 240
TRAIN_SEED = 270701
VAL_SEED = 270702
LENGTH = 64
FRAME_SKIP = 5
RAW_ACTION_DIM = 2


def collect_base(num_episodes: int, length: int, seed: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite

    env = suite.load("reacher", "easy",
                     task_kwargs={"random": seed % 2**32})
    env.physics.named.model.geom_rgba["target", 3] = 0.0
    spec = env.action_spec()
    if spec.shape != (RAW_ACTION_DIM,):
        raise RuntimeError(f"unexpected Reacher action shape {spec.shape}")
    low = np.asarray(spec.minimum, dtype=np.float32)
    high = np.asarray(spec.maximum, dtype=np.float32)
    center, half_range = (low + high) * 0.5, (high - low) * 0.5
    rng = np.random.default_rng(seed)
    state_dim = env.physics.data.qpos.size + env.physics.data.qvel.size
    frames = np.empty((num_episodes, length, IMG_SIZE, IMG_SIZE, 3),
                      dtype=np.uint8)
    action_blocks = np.empty(
        (num_episodes, length - 1, FRAME_SKIP * RAW_ACTION_DIM),
        dtype=np.float32)
    endo_state = np.empty((num_episodes, length, state_dim),
                          dtype=np.float32)

    def state() -> np.ndarray:
        return np.concatenate([env.physics.data.qpos,
                               env.physics.data.qvel]).astype(np.float32)

    for episode in range(num_episodes):
        env.reset()
        frames[episode, 0] = env.physics.render(
            IMG_SIZE, IMG_SIZE, camera_id=0)
        endo_state[episode, 0] = state()
        for step in range(length - 1):
            block = []
            for _ in range(FRAME_SKIP):
                squashed = np.tanh(rng.standard_normal(RAW_ACTION_DIM))
                action = np.clip(
                    center + half_range * squashed.astype(np.float32),
                    low, high)
                timestep = env.step(action)
                if timestep.last():
                    raise RuntimeError(
                        f"unexpected termination at episode={episode}, step={step}")
                block.append(action)
            action_blocks[episode, step] = np.concatenate(block)
            frames[episode, step + 1] = env.physics.render(
                IMG_SIZE, IMG_SIZE, camera_id=0)
            endo_state[episode, step + 1] = state()
        if (episode + 1) % 100 == 0:
            print(f"[official-data] {episode + 1}/{num_episodes}", flush=True)
    return frames, action_blocks, endo_state


def generate(task_name: str, episodes: int, seed: int) -> EpisodeBatch:
    task = make_task(task_name)
    base_seed, nuisance_rng, xi_rng = task._rngs(seed)
    script = task._sample_script(episodes, nuisance_rng, xi_rng, xi_shift=0)
    frames, actions, endo_state = collect_base(
        episodes, LENGTH, base_seed)
    exo_state = task._render(frames, script)
    return EpisodeBatch(
        frames=frames,
        actions=actions,
        xi=script["xi"],
        xi_kind=task.xi_kind,
        n_classes=task.n_classes,
        endo_state=endo_state,
        exo_state=exo_state,
        events={key: script[key] for key in task.event_keys},
        # EpisodeBatch's frozen schema admits the iid/script family only;
        # collection.json records that each iid observation step is a block
        # of five independently sampled simulator actions.
        stream="iid",
        task=task_name,
        seed=seed,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--output",
                        default="outputs/paper_a_expansion/data")
    parser.add_argument("--train-episodes", type=int,
                        default=TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=VAL_EPISODES)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output) / args.task
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output / f"train_clean_e{args.train_episodes}_s{TRAIN_SEED}.npz",
        "val": output / f"val_clean_e{args.val_episodes}_s{VAL_SEED}.npz",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    started = time.time()
    for split, episodes, seed in (
            ("train", args.train_episodes, TRAIN_SEED),
            ("val", args.val_episodes, VAL_SEED)):
        split_started = time.time()
        bank = generate(args.task, episodes, seed)
        save_bank(bank, paths[split])
        print(f"[official-data] {args.task}/{split} wrote {paths[split]} "
              f"in {(time.time() - split_started) / 60:.1f} min", flush=True)
    metadata = {
        "schema_version": 1,
        "task": args.task,
        "display_name": {
            "t1": "Transient-marker recall",
            "t3": "Drifting-color recall",
            "t4": "Occluded-target prediction",
        }[args.task],
        "frame_skip": FRAME_SKIP,
        "raw_action_dim": RAW_ACTION_DIM,
        "action_block_dim": FRAME_SKIP * RAW_ACTION_DIM,
        "episode_length": LENGTH,
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "extra_corruption": None,
        "elapsed_minutes": (time.time() - started) / 60,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    (output / "collection.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[official-data] finished {args.task}", flush=True)


if __name__ == "__main__":
    main()
