#!/usr/bin/env python3
"""Render one V3 semantic stage and issue its exact cue-only audit receipt."""

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
    require_paired_counterfactual,
)
from lewm.official_tasks.shell_game_capacity_v3 import (  # noqa: E402
    V3_SALIENCE,
    paired_counterfactual_batches_v3,
    v3_contract_description,
)
from lewm.official_tasks.shell_game_pipeline_v3 import (  # noqa: E402
    audit_path_v3,
    batch_arrays_v3,
    load_base_v3,
    lock_receipt_v3,
    require_all_selected_salience_v3,
    require_selected_salience_v3,
    split_spec_v3,
    stage_contract_v3,
    stage_path_v3,
)
from lewm.official_tasks.shell_game_spec_v3 import (  # noqa: E402
    ALL_SPLITS_V3,
    DEFAULT_LOCK_V3,
    DEFAULT_SPEC_V3,
    FORMAL_SPLITS_V3,
    load_locked_spec_v3,
)


STAGES = ("single-item", "two-item", "four-item")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--split", required=True, choices=ALL_SPLITS_V3)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_V3)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_V3)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec_v3(args.spec, args.lock)
    selection = None
    if args.split in FORMAL_SPLITS_V3:
        require_all_selected_salience_v3(spec)
        selection = require_selected_salience_v3(spec, args.stage)
    destinations = (
        audit_path_v3(spec, args.stage, args.split),
        stage_path_v3(spec, args.stage, args.split),
        stage_path_v3(spec, args.stage, args.split).with_suffix(".npz.json"),
    )
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V3 artifact {path}")
    base, base_sidecar = load_base_v3(spec, args.split)
    seed = int(split_spec_v3(spec, args.split)["counterfactual_seed"])
    contract = stage_contract_v3(args.stage)
    primary, counterfactual = paired_counterfactual_batches_v3(
        base, contract, seed)
    audit = require_paired_counterfactual(primary, counterfactual)
    primary_arrays = batch_arrays_v3(primary)
    counterfactual_arrays = batch_arrays_v3(counterfactual)
    receipt = {
        "schema": "official_shell_game_counterfactual_receipt_v3",
        "study": spec["study"],
        "amendment": spec["amendment"]["kind"],
        "threshold_changed_from_v1_or_v2": False,
        "stage": args.stage,
        "display_name": primary.display_name,
        "split": args.split,
        "counterfactual_seed": seed,
        "formal_lock": lock_receipt_v3(spec),
        "cue_salience": V3_SALIENCE.describe(),
        "base_artifact_sha256": base_sidecar["artifact"]["sha256"],
        "primary_content_sha256": sha256_arrays(primary_arrays),
        "counterfactual_content_sha256": sha256_arrays(counterfactual_arrays),
        "audit": audit,
    }
    receipt_path = audit_path_v3(spec, args.stage, args.split)
    receipt_hash = atomic_text(receipt_path, stable_json(receipt))
    metadata = {
        "schema": "official_shell_game_stage_v3",
        "study": spec["study"],
        "amendment": spec["amendment"]["kind"],
        "threshold_changed_from_v1_or_v2": False,
        "stage": args.stage,
        "display_name": primary.display_name,
        "capacity": primary.contract.stage.capacity,
        "split": args.split,
        "counterfactual_seed": seed,
        "formal_lock": lock_receipt_v3(spec),
        "base_artifact_sha256": base_sidecar["artifact"]["sha256"],
        "cue_salience": V3_SALIENCE.describe(),
        "counterfactual_receipt": {
            "path": receipt_path.name,
            "sha256": receipt_hash,
        },
        "development_selection": selection,
        "contract": v3_contract_description(primary.contract),
    }
    record = write_npz_with_sidecar(
        stage_path_v3(spec, args.stage, args.split), primary_arrays, metadata,
        compression_level=int(spec["data"]["compression_level"]),
    )
    print(json.dumps({"stage_artifact": record, "audit": audit},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
