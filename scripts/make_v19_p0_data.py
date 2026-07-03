#!/usr/bin/env python3
"""Build the V19 P0 host-preflight data caches (docs/V19_PROPOSAL.md section 4.1).

Per certified task (t1-t4): 1,200 train + 240 validation episodes from the
frozen P1a generators (stream 'iid', L=64), fixed data seeds under the V18
convention (train 270701 / val 270702), written ONCE per task as compressed
NPZ banks with sha256 sidecars (``lewm.tasks_v19.base.save_bank/load_bank``)
and shared by every arm and seed.

The corruption-on regime is baked in at cache time on the *train-stream
inputs* of both splits, mirroring the V18 'train' view: per episode one
contiguous window of 6-12 steps is replaced by the episode pixel-mean frame
(even episode index) or a 55% spatial cutout filled with the episode channel
mean (odd episode index), with window position and cutout placement drawn from
corruption seed 270711 in the V11 per-episode rng namespace.  The clean banks
are stored alongside: the VICReg reference arm trains against active clean
targets (paired views, the V18 recipe) while the exact-SIGReg arm never reads
them -- single-stream training is the registered exactness delta.

t4 note: the task's own freeze gap stays as generated; training corruption
windows are drawn independently and may overlap it (the P2/P3 reality).

Set V19_P0_EPISODES for smoke-sized caches (validation scales to
``max(n // 5, 16)``).  Existing caches are skipped when their sidecar sha256
still matches the bytes on disk.
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
from lewm.tasks_v19.base import EpisodeBatch, save_bank, sha256_file
from scripts.hacssm_v11_data import _episode_rng

P0_TASKS = ("t1", "t2", "t3", "t4")
DEFAULT_ROOT = "outputs/v19_p0/data"
DEFAULT_TRAIN_EPISODES = 1_200
DEFAULT_VAL_EPISODES = 240
TRAIN_SEED = 270_701          # V18 data-seed convention
VAL_SEED = 270_702
CORRUPTION_SEED = 270_711
STREAM = "iid"
EPISODE_LENGTH = 64
GAP_RANGE = (6, 12)           # inclusive contiguous-window length, registered for P0
CUTOUT_FRACTION = 0.55
HISTORY_LEN = 3               # earliest-onset guard of the V11 interval contract
MODES = ("meanframe", "cutout")
_INTERVAL_SALT = 17           # V11 corruption rng salts, kept verbatim
_SPATIAL_SALT = 101


def episode_sizes() -> tuple[int, int]:
    """(train, val) episode counts; V19_P0_EPISODES shrinks both for smoke."""
    override = os.environ.get("V19_P0_EPISODES")
    if override is None:
        return DEFAULT_TRAIN_EPISODES, DEFAULT_VAL_EPISODES
    train_episodes = int(override)
    if train_episodes < 8:
        raise ValueError(f"V19_P0_EPISODES must be >= 8, got {train_episodes}")
    return train_episodes, max(train_episodes // 5, 16)


def corruption_window(length: int, episode: int, seed: int = CORRUPTION_SEED,
                      history_len: int = HISTORY_LEN) -> tuple[int, int]:
    """One contiguous 6-12 step corruption window in the V11 interval namespace.

    Mirrors ``scripts.hacssm_v11_data.corruption_interval`` (same per-episode
    rng derivation and start bounds) with the P0-registered gap range.
    """
    rng = _episode_rng(seed, episode, _INTERVAL_SALT)
    gap = int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1))
    earliest = history_len + 2
    latest = length - gap - 2
    if latest < earliest:
        raise ValueError(f"length {length} is too short for a {gap}-step window")
    start = int(rng.integers(earliest, latest + 1))
    return start, start + gap


def corruption_mode(episode: int) -> str:
    """Deterministic mode alternation by episode index (registered for P0)."""
    return MODES[episode % 2]


def corrupt_episode(clean: np.ndarray, episode: int,
                    seed: int = CORRUPTION_SEED
                    ) -> tuple[np.ndarray, int, int, str]:
    """Apply the P0 training corruption to one (L, H, W, 3) uint8 episode.

    Returns (observed, start, end, mode); ``clean`` is never mutated.
    """
    if clean.ndim != 4 or clean.dtype != np.uint8:
        raise ValueError(f"expected uint8 (L,H,W,3) frames, got "
                         f"{clean.dtype} {clean.shape}")
    observed = clean.copy()
    start, end = corruption_window(clean.shape[0], episode, seed)
    mode = corruption_mode(episode)
    if mode == "meanframe":
        mean_frame = np.rint(clean.mean(axis=0)).clip(0, 255).astype(np.uint8)
        observed[start:end] = mean_frame
    else:
        rng = _episode_rng(seed, episode, _SPATIAL_SALT)
        height, width = clean.shape[1:3]
        cut_h = max(1, int(round(height * CUTOUT_FRACTION)))
        cut_w = max(1, int(round(width * CUTOUT_FRACTION)))
        top = int(rng.integers(0, height - cut_h + 1))
        left = int(rng.integers(0, width - cut_w + 1))
        fill = np.rint(clean.mean(axis=(0, 1, 2))).clip(0, 255).astype(np.uint8)
        observed[start:end, top:top + cut_h, left:left + cut_w] = fill
    return observed, start, end, mode


def corrupt_bank(batch: EpisodeBatch, seed: int = CORRUPTION_SEED) -> EpisodeBatch:
    """Corrupted copy of a bank, with corrupt_on/off/mode event annotations."""
    frames = batch.frames.copy()
    episodes = batch.num_episodes
    corrupt_on = np.empty(episodes, dtype=np.int64)
    corrupt_off = np.empty(episodes, dtype=np.int64)
    corrupt_mode = np.empty(episodes, dtype=np.int64)
    for episode in range(episodes):
        frames[episode], start, end, mode = corrupt_episode(
            batch.frames[episode], episode, seed)
        corrupt_on[episode] = start
        corrupt_off[episode] = end
        corrupt_mode[episode] = MODES.index(mode)
    events = dict(batch.events)
    events.update({"corrupt_on": corrupt_on, "corrupt_off": corrupt_off,
                   "corrupt_mode": corrupt_mode})
    return EpisodeBatch(
        frames=frames, actions=batch.actions, xi=batch.xi,
        xi_kind=batch.xi_kind, n_classes=batch.n_classes,
        endo_state=batch.endo_state, exo_state=batch.exo_state,
        events=events, stream=batch.stream, task=batch.task, seed=batch.seed)


def bank_path(root: str | Path, task: str, split: str, view: str,
              episodes: int, seed: int) -> Path:
    """Cache filenames carry episode count and data seed, so smoke-sized and
    full caches coexist without ambiguity (V11 cache-name discipline)."""
    return Path(root) / task / f"{split}_{view}_e{episodes}_s{seed}.npz"


def task_bank_paths(root: str | Path, task: str, train_episodes: int,
                    val_episodes: int) -> dict[str, dict[str, Path]]:
    splits = {"train": (train_episodes, TRAIN_SEED),
              "val": (val_episodes, VAL_SEED)}
    return {split: {view: bank_path(root, task, split, view, episodes, seed)
                    for view in ("clean", "observed")}
            for split, (episodes, seed) in splits.items()}


def _cache_valid(path: Path) -> bool:
    """True iff the bank and its sidecar exist and the sha256 still matches."""
    sidecar = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not sidecar.exists():
        return False
    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (metadata.get("format") == "v19_p1a_bank_v1"
            and sha256_file(path) == metadata.get("npz_sha256"))


def ensure_task_cache(task_name: str, root: str | Path = DEFAULT_ROOT,
                      force: bool = False) -> dict:
    """Generate (or verify) the four banks of one task; returns their metadata."""
    if task_name not in P0_TASKS:
        raise ValueError(f"task must be one of {P0_TASKS}, got {task_name!r}")
    train_episodes, val_episodes = episode_sizes()
    paths = task_bank_paths(root, task_name, train_episodes, val_episodes)
    result: dict = {"task": task_name, "train_episodes": train_episodes,
                    "val_episodes": val_episodes, "corruption_seed": CORRUPTION_SEED,
                    "splits": {}}
    for split, (episodes, seed) in (("train", (train_episodes, TRAIN_SEED)),
                                    ("val", (val_episodes, VAL_SEED))):
        clean_path = paths[split]["clean"]
        observed_path = paths[split]["observed"]
        if not force and _cache_valid(clean_path) and _cache_valid(observed_path):
            metadata = {view: json.loads(
                paths[split][view].with_suffix(".npz.json").read_text())
                for view in ("clean", "observed")}
            result["splits"][split] = {
                "regenerated": False,
                "paths": {view: str(paths[split][view]) for view in metadata},
                "sha256": {view: metadata[view]["npz_sha256"] for view in metadata},
            }
            print(f"[v19-p0-data] {task_name}/{split}: cache valid, skipping",
                  flush=True)
            continue
        started = time.time()
        print(f"[v19-p0-data] {task_name}/{split}: generating {episodes} episodes "
              f"(stream={STREAM}, seed={seed})", flush=True)
        clean = make_task(task_name).generate(STREAM, episodes, seed)
        if clean.length != EPISODE_LENGTH:
            raise RuntimeError(f"expected L={EPISODE_LENGTH}, got {clean.length}")
        observed = corrupt_bank(clean, CORRUPTION_SEED)
        clean_meta = save_bank(clean, clean_path)
        observed_meta = save_bank(observed, observed_path)
        result["splits"][split] = {
            "regenerated": True,
            "seconds": round(time.time() - started, 1),
            "paths": {"clean": str(clean_path), "observed": str(observed_path)},
            "sha256": {"clean": clean_meta["npz_sha256"],
                       "observed": observed_meta["npz_sha256"]},
        }
        print(f"[v19-p0-data] {task_name}/{split}: wrote "
              f"{clean_path.name} + {observed_path.name} "
              f"({result['splits'][split]['seconds']}s)", flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=P0_TASKS,
                        help="task to cache (default: all four)")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if a valid cache exists")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = tuple(args.task) if args.task else P0_TASKS
    manifests = [ensure_task_cache(task, args.root, force=args.force)
                 for task in tasks]
    print(json.dumps({"root": str(Path(args.root).resolve()),
                      "tasks": manifests}, indent=2), flush=True)


if __name__ == "__main__":
    main()
