#!/usr/bin/env python3
"""Collect immutable fresh validation banks for Paper-A robustness checks.

This program never writes below ``outputs/paper_a_expansion``.  It produces
only the two validation banks fixed in ``configs/paper_a_robustness.yaml``;
the original training bank remains the readout/training source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19.base import save_bank  # noqa: E402
from scripts.cache_official_lewm import sha256_file  # noqa: E402
from scripts.make_official_lewm_memory_data import generate  # noqa: E402
from scripts.paper_a_robustness_spec import (  # noqa: E402
    DEFAULT_SPEC,
    TASKS,
    bank_by_id,
    load_locked_spec,
    resolve_spec_path,
)


def bank_path(spec: Mapping, task: str, bank_id: str) -> Path:
    bank = bank_by_id(spec, bank_id)
    episodes = int(spec["fresh_validation"]["episodes_per_bank"])
    seed = int(bank["collection_seed"])
    root = resolve_spec_path(spec, spec["output"]["fresh_validation_data"])
    return root / task / bank_id / f"val_clean_e{episodes}_s{seed}.npz"


def selected_pairs(spec: Mapping, task: str | None, bank_id: str | None,
                   all_banks: bool) -> list[tuple[str, str]]:
    if all_banks:
        if task is not None or bank_id is not None:
            raise ValueError("--all cannot be combined with --task/--bank")
        return [
            (name, bank["id"])
            for name in spec["tasks"]
            for bank in spec["fresh_validation"]["banks"]
        ]
    if task is None or bank_id is None:
        raise ValueError("one-bank mode requires --task and --bank")
    if task not in TASKS:
        raise ValueError(f"unsupported robustness task {task!r}")
    bank_by_id(spec, bank_id)
    return [(task, bank_id)]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--bank")
    parser.add_argument("--egl-device-id", type=int, choices=(1, 2), default=1)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    pairs = selected_pairs(spec, args.task, args.bank, args.all)
    outputs = [bank_path(spec, task, bank) for task, bank in pairs]
    protected = resolve_spec_path(spec, spec["parent"]["root"])
    for output in outputs:
        if protected == output or protected in output.parents:
            raise RuntimeError(f"refusing to write into parent artifacts: {output}")
        sidecar = output.with_suffix(output.suffix + ".json")
        receipt = output.parent / "collection_receipt.json"
        existing = [path for path in (output, sidecar, receipt) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite robustness data: "
                + ", ".join(str(path) for path in existing))

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device_id)
    episodes = int(spec["fresh_validation"]["episodes_per_bank"])
    for task, bank_id in pairs:
        bank = bank_by_id(spec, bank_id)
        seed = int(bank["collection_seed"])
        destination = bank_path(spec, task, bank_id)
        destination.parent.mkdir(parents=True, exist_ok=False)
        generated = generate(task, episodes, seed)
        save_bank(generated, destination)
        sidecar = destination.with_suffix(destination.suffix + ".json")
        receipt = {
            "schema_version": 1,
            "study": spec["study"],
            "task": task,
            "bank_id": bank_id,
            "collection_seed": seed,
            "episodes": episodes,
            "split": "validation",
            "parent_artifacts_modified": False,
            "strengthening_spec": spec["_spec_record"],
            "artifact": {
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256_file(destination),
                "sidecar": str(sidecar.relative_to(ROOT)),
                "sidecar_sha256": sha256_file(sidecar),
            },
        }
        receipt_path = destination.parent / "collection_receipt.json"
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"[robust-data] wrote {task}/{bank_id}: {destination}",
              flush=True)


if __name__ == "__main__":
    main()
