#!/usr/bin/env python3
"""Render one semantic capacity stage and issue its exact audit receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    sha256_arrays,
    stable_json,
    write_npz_with_sidecar,
)
from lewm.official_tasks.shell_game_capacity import (  # noqa: E402
    paired_counterfactual_batches,
    require_paired_counterfactual,
)
from lewm.official_tasks.shell_game_pipeline import (  # noqa: E402
    audit_path,
    batch_arrays,
    load_base,
    lock_receipt,
    split_spec,
    stage_contract,
    stage_path,
)
from lewm.official_tasks.shell_game_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    load_locked_spec,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("single-item", "two-item", "four-item"))
    parser.add_argument("--split", required=True,
                        choices=("train", "validation"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec, args.lock)
    destinations = (
        audit_path(spec, args.stage, args.split),
        stage_path(spec, args.stage, args.split),
        stage_path(spec, args.stage, args.split).with_suffix(".npz.json"),
    )
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    base, base_sidecar = load_base(spec, args.split)
    seed = int(split_spec(spec, args.split)["counterfactual_seed"])
    primary, counterfactual = paired_counterfactual_batches(
        base, stage_contract(args.stage), seed)
    audit = require_paired_counterfactual(primary, counterfactual)
    primary_arrays = batch_arrays(primary)
    counterfactual_arrays = batch_arrays(counterfactual)
    receipt = {
        "schema": "official_shell_game_counterfactual_receipt_v1",
        "study": spec["study"],
        "stage": args.stage,
        "display_name": primary.display_name,
        "split": args.split,
        "counterfactual_seed": seed,
        "formal_lock": lock_receipt(spec),
        "base_artifact_sha256": base_sidecar["artifact"]["sha256"],
        "primary_content_sha256": sha256_arrays(primary_arrays),
        "counterfactual_content_sha256": sha256_arrays(counterfactual_arrays),
        "audit": audit,
    }
    receipt_path = audit_path(spec, args.stage, args.split)
    receipt_hash = atomic_text(receipt_path, stable_json(receipt))
    metadata = {
        "schema": "official_shell_game_stage_v1",
        "study": spec["study"],
        "stage": args.stage,
        "display_name": primary.display_name,
        "capacity": primary.contract.stage.capacity,
        "split": args.split,
        "counterfactual_seed": seed,
        "formal_lock": lock_receipt(spec),
        "base_artifact_sha256": base_sidecar["artifact"]["sha256"],
        "counterfactual_receipt": {
            "path": receipt_path.name,
            "sha256": receipt_hash,
        },
        "contract": primary.contract.describe(),
    }
    record = write_npz_with_sidecar(
        stage_path(spec, args.stage, args.split), primary_arrays, metadata,
        compression_level=int(spec["data"]["compression_level"]),
    )
    print(json.dumps({"stage_artifact": record, "audit": audit},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
