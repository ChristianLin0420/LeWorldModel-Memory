#!/usr/bin/env python3
"""Seal matched-token protocol, sources, and V1 episode exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, load_verified_npz, stable_json  # noqa: E402
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    DEFAULT_LOCK, DEFAULT_SHA, DEFAULT_SPEC, HDF_HOSTS, output_path,
    resolve_path, sha256_file, validate_spec,
)


PRODUCERS = (
    "lewm/models/frozen_swap_carriers.py",
    "lewm/models/official_lewm.py",
    "lewm/models/official_lewm_config.py",
    "lewm/models/official_lewm_pusht.py",
    "lewm/models/official_lewm_tworoom.py",
    "lewm/official_tasks/artifacts.py",
    "lewm/official_tasks/matched_token.py",
    "lewm/official_tasks/native_sequence_hdf5.py",
    "lewm/official_tasks/tworoom_downstream.py",
    "scripts/make_official_lewm_memory_data.py",
    "scripts/paper_a_evidence_age.py",
    "scripts/paper_a_matched_host_spec.py",
    "scripts/prepare_paper_a_matched_host.py",
    "scripts/train_frozen_official_swap.py",
    "scripts/train_official_pusht_carrier.py",
    "scripts/evaluate_paper_a_tworoom_use.py",
    "scripts/aggregate_paper_a_tworoom_use.py",
    "scripts/paper_a_matched_token_spec.py",
    "scripts/prepare_paper_a_matched_token.py",
    "scripts/train_paper_a_matched_token.py",
    "scripts/aggregate_paper_a_matched_token.py",
    "scripts/prepare_paper_a_matched_token_use.py",
    "scripts/evaluate_paper_a_matched_token_use.py",
    "scripts/aggregate_paper_a_matched_token_use.py",
    "scripts/launch_paper_a_matched_token.py",
    "scripts/seal_paper_a_matched_token.py",
    "tests/test_matched_token.py",
    "tests/test_paper_a_matched_token.py",
)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _indices_hash(values):
    return hashlib.sha256(json.dumps(
        values, separators=(",", ":")).encode()).hexdigest()


def _exclusions(spec):
    output = {}
    for host in HDF_HOSTS:
        artifacts, arrays = [], []
        for split, relative in zip(
                ("train", "validation"),
                spec["inputs"][host]["v1_base_cache_paths"], strict=True):
            path = resolve_path(relative)
            values, sidecar = load_verified_npz(path)
            if sidecar.get("study") != "paper-a-matched-host-v1" \
                    or sidecar.get("host") != host \
                    or sidecar.get("split") != split \
                    or sidecar.get("lock", {}).get("sha256") \
                    != "5febf1c31d8a9f73c83a4d26ba4ed0f9934a23e6099840e50ceba32b4b740b7f" \
                    or sidecar.get("lock", {}).get(
                        "implementation", {}).get("sha256") \
                    != "83c2ae48ea53e2080606fa7e15e30339668deb338fd556527025e9418b076998":
                raise ValueError("V1 exclusion cache identity differs")
            indices = np.asarray(values["episode_index"], dtype=np.int64)
            expected = 1200 if split == "train" else 480
            if indices.shape != (expected,) or len(np.unique(indices)) != expected:
                raise ValueError("V1 exclusion episode indices differ")
            arrays.append(indices)
            sidecar_path = path.with_suffix(path.suffix + ".json")
            for artifact in (path, sidecar_path):
                artifacts.append({"path": str(artifact.relative_to(ROOT)),
                                  "size": artifact.stat().st_size,
                                  "sha256": sha256_file(artifact)})
        union = sorted(map(int, np.concatenate(arrays)))
        if len(union) != 1680 or len(set(union)) != 1680:
            raise ValueError("V1 exclusion union differs")
        receipt = resolve_path(spec["adaptive_origin"][
            "v1_host_receipts"][host]["path"])
        artifacts.append({"path": str(receipt.relative_to(ROOT)),
                          "size": receipt.stat().st_size,
                          "sha256": sha256_file(receipt)})
        output[host] = {"episode_indices": union, "count": 1680,
                        "indices_sha256": _indices_hash(union),
                        "artifacts": artifacts}
    return output


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if DEFAULT_SHA.exists() or DEFAULT_LOCK.exists():
        raise FileExistsError("matched-token protocol already sealed")
    spec = yaml.safe_load(DEFAULT_SPEC.read_text())
    validate_spec(spec, verify_inputs=True)
    root = output_path(spec, "root")
    if root.exists() and any(root.rglob("*")):
        raise RuntimeError("matched-token outputs exist before seal")
    exclusions = _exclusions(spec)
    producers = {}
    for relative in PRODUCERS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        producers[relative] = sha256_file(path)
    spec_hash = sha256_file(DEFAULT_SPEC)
    payload = {"schema_version": 1, "study": spec["study"],
               "spec_path": str(DEFAULT_SPEC.relative_to(ROOT)),
               "spec_sha256": spec_hash,
               "locked_before_matched_token_outcomes": True,
               "formal_output_absent_at_lock": True,
               "v1_metrics_used_only_for_adaptation": True,
               "v1_hdf_exclusions": exclusions, "producers": producers}
    if not args.execute:
        print(stable_json(payload), end=""); return
    atomic_text(DEFAULT_LOCK, stable_json(payload))
    atomic_text(DEFAULT_SHA, f"{spec_hash}  {DEFAULT_SPEC.name}\n")
    print(json.dumps({"spec_sha256": spec_hash,
                      "lock_sha256": sha256_file(DEFAULT_LOCK),
                      "producers": len(producers),
                      "exclusions": {host: 1680 for host in HDF_HOSTS}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
