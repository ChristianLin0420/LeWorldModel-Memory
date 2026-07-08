#!/usr/bin/env python3
"""Collect one deterministic native-timescale Reacher base bank with EGL."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import write_npz_with_sidecar  # noqa: E402
from lewm.official_tasks.shell_game_pipeline import (  # noqa: E402
    base_path,
    lock_receipt,
    split_spec,
)
from lewm.official_tasks.shell_game_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    load_locked_spec,
    validate_device,
)
from scripts.make_official_lewm_memory_data import collect_base  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True,
                        choices=("train", "validation"))
    parser.add_argument(
        "--device", required=True,
        help="EGL render device; the formal protocol permits cuda:1 or cuda:2")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec, args.lock)
    validate_device(args.device)
    split = split_spec(spec, args.split)
    destination = base_path(spec, args.split)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    for path in (destination, sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    # dm_control imports MuJoCo lazily inside collect_base.  Set the absolute
    # EGL device before that import so an unspecified default can never use
    # a forbidden GPU.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = args.device.rsplit(":", 1)[-1]
    frames, actions, endo_state = collect_base(
        int(split["episodes"]),
        int(spec["official_host"]["observation_length"]),
        int(split["base_seed"]),
    )
    metadata = {
        "schema": "official_shell_game_base_v1",
        "study": spec["study"],
        "split": args.split,
        "episodes": int(split["episodes"]),
        "base_seed": int(split["base_seed"]),
        "frame_skip": int(spec["data"]["frame_skip"]),
        "raw_action_dim": int(spec["data"]["raw_action_dim"]),
        "action_block_dim": int(spec["official_host"]["action_block_dim"]),
        "egl_device": args.device,
        "formal_lock": lock_receipt(spec),
    }
    record = write_npz_with_sidecar(
        destination,
        {"frames": frames, "actions": actions, "endo_state": endo_state},
        metadata,
        compression_level=int(spec["data"]["compression_level"]),
    )
    print(f"[shell-game-base] wrote {record['path']} {record['sha256']}")


if __name__ == "__main__":
    main()
