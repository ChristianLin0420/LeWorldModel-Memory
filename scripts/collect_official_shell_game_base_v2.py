#!/usr/bin/env python3
"""Collect a deterministic V2 Reacher base bank on an allowed EGL GPU."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import write_npz_with_sidecar  # noqa: E402
from lewm.official_tasks.shell_game_capacity import CAPACITY_STAGES  # noqa: E402
from lewm.official_tasks.shell_game_pipeline_v2 import (  # noqa: E402
    base_path_v2,
    lock_receipt_v2,
    require_selected_salience_v2,
    split_spec_v2,
)
from lewm.official_tasks.shell_game_spec_v2 import (  # noqa: E402
    ALL_SPLITS_V2,
    DEFAULT_LOCK_V2,
    DEFAULT_SPEC_V2,
    FORMAL_SPLITS_V2,
    load_locked_spec_v2,
    validate_device_v2,
)
from scripts.make_official_lewm_memory_data import collect_base  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=ALL_SPLITS_V2)
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_V2)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_V2)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec_v2(args.spec, args.lock)
    validate_device_v2(args.device)
    if args.split in FORMAL_SPLITS_V2:
        # Formal base pixels are not collected until the development-only
        # frozen cue gate passes at every registered semantic capacity.
        for stage in CAPACITY_STAGES:
            require_selected_salience_v2(spec, stage.key)
    split = split_spec_v2(spec, args.split)
    destination = base_path_v2(spec, args.split)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    for path in (destination, sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V2 artifact {path}")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = args.device.rsplit(":", 1)[-1]
    frames, actions, endo_state = collect_base(
        int(split["episodes"]),
        int(spec["official_host"]["observation_length"]),
        int(split["base_seed"]),
    )
    metadata = {
        "schema": "official_shell_game_base_v2",
        "study": spec["study"],
        "split": args.split,
        "episodes": int(split["episodes"]),
        "base_seed": int(split["base_seed"]),
        "frame_skip": int(spec["data"]["frame_skip"]),
        "raw_action_dim": int(spec["data"]["raw_action_dim"]),
        "action_block_dim": int(spec["official_host"]["action_block_dim"]),
        "egl_device": args.device,
        "formal_lock": lock_receipt_v2(spec),
        "amendment": spec["amendment"]["kind"],
        "formal_split_gated_by_development": args.split in FORMAL_SPLITS_V2,
    }
    record = write_npz_with_sidecar(
        destination,
        {"frames": frames, "actions": actions, "endo_state": endo_state},
        metadata,
        compression_level=int(spec["data"]["compression_level"]),
    )
    print(f"[shell-game-v2-base] wrote {record['path']} {record['sha256']}")


if __name__ == "__main__":
    main()
