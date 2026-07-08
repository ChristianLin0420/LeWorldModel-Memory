#!/usr/bin/env python3
"""Seal the Wave-1 protocol and producer hashes before formal outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, stable_json  # noqa: E402
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    output_path,
    sha256_file,
    validate_spec,
)

import yaml  # noqa: E402


PRODUCERS = (
    "lewm/models/frozen_swap_carriers.py",
    "lewm/models/official_lewm.py",
    "lewm/models/official_lewm_config.py",
    "lewm/models/official_lewm_pusht.py",
    "lewm/models/official_lewm_tworoom.py",
    "lewm/official_tasks/artifacts.py",
    "lewm/official_tasks/matched_memory.py",
    "lewm/official_tasks/native_sequence_hdf5.py",
    "lewm/official_tasks/tworoom_downstream.py",
    "scripts/aggregate_paper_a_matched_host.py",
    "scripts/aggregate_paper_a_tworoom_use.py",
    "scripts/evaluate_paper_a_tworoom_use.py",
    "scripts/launch_paper_a_matched_host.py",
    "scripts/launch_paper_a_tworoom_use.py",
    "scripts/make_official_lewm_memory_data.py",
    "scripts/paper_a_evidence_age.py",
    "scripts/paper_a_matched_host_spec.py",
    "scripts/prepare_paper_a_matched_host.py",
    "scripts/prepare_paper_a_tworoom_use.py",
    "scripts/seal_paper_a_matched_host.py",
    "scripts/train_frozen_official_swap.py",
    "scripts/train_official_pusht_carrier.py",
    "scripts/train_paper_a_matched_host.py",
    "tests/test_matched_memory.py",
    "tests/test_native_sequence_hdf5.py",
    "tests/test_aggregate_paper_a_tworoom_use.py",
    "tests/test_paper_a_matched_host.py",
    "tests/test_paper_a_tworoom_use_eval.py",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if DEFAULT_SHA.exists() or DEFAULT_LOCK.exists():
        raise FileExistsError("Wave-1 protocol is already sealed")
    value = yaml.safe_load(DEFAULT_SPEC.read_text())
    validate_spec(value, verify_inputs=True)
    formal_root = ROOT / value["outputs"]["root"]
    forbidden = list(formal_root.glob("cache/**/*")) \
        + list(formal_root.glob("carriers/**/*")) \
        + list(formal_root.glob("use/**/*")) \
        + [formal_root / "summary.json", formal_root / "final_audit.json"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("formal Wave-1 outputs exist before protocol sealing")
    producers = {}
    for relative in PRODUCERS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        producers[relative] = sha256_file(path)
    spec_hash = sha256_file(DEFAULT_SPEC)
    payload = {
        "schema_version": 1, "study": value["study"],
        "spec_path": str(DEFAULT_SPEC.relative_to(ROOT)),
        "spec_sha256": spec_hash,
        "locked_before_formal_outcomes": True,
        "formal_output_absent_at_lock": True,
        "producers": producers,
    }
    if not args.execute:
        print(stable_json(payload), end="")
        return
    atomic_text(DEFAULT_LOCK, stable_json(payload))
    atomic_text(DEFAULT_SHA, f"{spec_hash}  {DEFAULT_SPEC.name}\n")
    print(json.dumps({
        "lock": str(DEFAULT_LOCK.relative_to(ROOT)),
        "lock_sha256": sha256_file(DEFAULT_LOCK),
        "spec_sha256": spec_hash,
        "producers": len(producers),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
