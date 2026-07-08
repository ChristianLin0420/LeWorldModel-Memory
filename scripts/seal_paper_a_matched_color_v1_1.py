#!/usr/bin/env python3
"""Seal Wave 1.1 code, prior receipts, and fresh-data exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    load_verified_npz,
    stable_json,
)
from scripts.paper_a_matched_color_v1_1_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    indices_sha256,
    output_path,
    resolve_path,
    sha256_file,
    validate_spec,
)


PRODUCERS = (
    "lewm/models/frozen_swap_carriers.py",
    "lewm/models/official_lewm.py",
    "lewm/models/official_lewm_config.py",
    "lewm/models/official_lewm_pusht.py",
    "lewm/models/official_lewm_tworoom.py",
    "lewm/official_tasks/artifacts.py",
    "lewm/official_tasks/matched_memory.py",
    "lewm/official_tasks/native_sequence_hdf5.py",
    "scripts/make_official_lewm_memory_data.py",
    "scripts/paper_a_evidence_age.py",
    "scripts/paper_a_matched_host_spec.py",
    "scripts/prepare_paper_a_matched_host.py",
    "scripts/train_frozen_official_swap.py",
    "scripts/train_official_pusht_carrier.py",
    "scripts/paper_a_matched_color_v1_1_spec.py",
    "scripts/prepare_paper_a_matched_color_v1_1.py",
    "scripts/train_paper_a_matched_color_v1_1.py",
    "scripts/aggregate_paper_a_matched_color_v1_1.py",
    "scripts/launch_paper_a_matched_color_v1_1.py",
    "scripts/seal_paper_a_matched_color_v1_1.py",
    "tests/test_paper_a_matched_color_v1_1.py",
)

PRIOR_IDENTITIES = {
    "paper_a_matched_host_v1": {
        "study": "paper-a-matched-host-v1",
        "schema": "paper_a_matched_base_cache_v1",
        "spec_sha": "5febf1c31d8a9f73c83a4d26ba4ed0f9934a23e6099840e50ceba32b4b740b7f",
        "lock_sha": "83c2ae48ea53e2080606fa7e15e30339668deb338fd556527025e9418b076998",
    },
    "paper_a_matched_token_v1": {
        "study": "paper-a-matched-token-v1",
        "schema": "paper_a_matched_token_base_cache_v1",
        "spec_sha": "b279ffaa89b63d9c1994799cd1fe01039f2ac0fb70243986a6c6858ed188ebd3",
        "lock_sha": "025be77f9f40173778fde41759a5e7d6b5a24ee41cf7d604d6e459285c4a224b",
    },
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verified_base(relative: str, split: str) -> tuple[dict[str, Any], np.ndarray]:
    path = resolve_path(relative)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    arrays, sidecar = load_verified_npz(path)
    source_name = next((name for name in PRIOR_IDENTITIES if name in relative), None)
    if source_name is None:
        raise ValueError(f"unregistered prior cache: {relative}")
    identity = PRIOR_IDENTITIES[source_name]
    lock = sidecar.get("lock", {})
    if sidecar.get("schema") != identity["schema"] \
            or sidecar.get("study") != identity["study"] \
            or sidecar.get("host") != "pusht" \
            or sidecar.get("split") != split \
            or lock.get("sha256") != identity["spec_sha"] \
            or lock.get("implementation", {}).get("sha256") \
            != identity["lock_sha"]:
        raise ValueError(f"prior cache lock identity differs: {relative}")
    indices = np.asarray(arrays.get("episode_index"))
    expected = 1200 if split == "train" else 480
    if indices.dtype != np.int64 or indices.shape != (expected,) \
            or len(np.unique(indices)) != expected or np.any(indices < 0):
        raise ValueError(f"prior episode indices differ: {relative}")
    record = {
        "source_study": identity["study"], "split": split,
        "cache": _artifact(path), "sidecar": _artifact(sidecar_path),
        "episode_count": expected,
        "episode_indices_sha256": indices_sha256(sorted(map(int, indices))),
        "embedded_spec_sha256": identity["spec_sha"],
        "embedded_implementation_lock_sha256": identity["lock_sha"],
    }
    return record, indices


def _prior_exclusions(spec: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    by_study: dict[str, list[np.ndarray]] = {}
    paths = spec["inputs"]["pusht"]["prior_base_cache_paths"]
    for index, relative in enumerate(paths):
        split = "train" if index % 2 == 0 else "validation"
        record, values = _verified_base(relative, split)
        records.append(record)
        by_study.setdefault(record["source_study"], []).append(values)
    study_unions: dict[str, list[int]] = {}
    for study, arrays in by_study.items():
        if len(arrays) != 2 or len(np.intersect1d(arrays[0], arrays[1])):
            raise ValueError(f"prior train/validation overlap: {study}")
        values = sorted(map(int, np.concatenate(arrays)))
        if len(values) != 1680 or len(set(values)) != 1680:
            raise ValueError(f"prior union count differs: {study}")
        study_unions[study] = values
    first, second = tuple(study_unions.values())
    if set(first).intersection(second):
        raise ValueError("the two prior PushT screens are not disjoint")
    union = sorted(first + second)
    screens = spec["adaptive_origin"]["prior_screens"]
    manifests = {
        name: _artifact(resolve_path(screen["receipts"]["pusht"]["path"]))
        for name, screen in screens.items()
    }
    return {
        "pusht": {
            "policy": spec["adaptive_origin"]["hdf_exclusion_policy"],
            "episode_indices": union, "count": len(union),
            "indices_sha256": indices_sha256(union),
            "per_study": {
                study: {"count": len(values),
                        "indices_sha256": indices_sha256(values)}
                for study, values in study_unions.items()
            },
            "cache_candidates": records,
            "host_manifests": manifests,
            "cross_screen_overlap_count": 0,
        }
    }


def _reacher_rng_exclusion(spec: dict[str, Any]) -> dict[str, Any]:
    source = spec["inputs"]["reacher"]
    prior = sorted(value for pair in source["prior_seed_pairs"] for value in pair)
    new = [int(source["train_base_seed"]), int(source["validation_base_seed"])]
    if set(prior).intersection(new) or len(set(prior + new)) != 6:
        raise ValueError("Reacher RNG seeds are not disjoint")
    registry = {
        "prior_screens": {
            "paper-a-matched-host-v1": prior[:2],
            "paper-a-matched-token-v1": prior[2:],
        },
        "wave1_1": new,
    }
    digest = hashlib.sha256(json.dumps(
        registry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "policy": spec["adaptive_origin"]["reacher_rng_exclusion_policy"],
        "prior_seeds": prior, "new_seeds": new,
        "all_seed_values_unique": True, "registry": registry,
        "registry_sha256": digest,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if DEFAULT_SHA.exists() or DEFAULT_LOCK.exists():
        raise FileExistsError("Wave 1.1 is already sealed")
    spec = yaml.safe_load(DEFAULT_SPEC.read_text())
    if not isinstance(spec, dict):
        raise ValueError("Wave 1.1 protocol is not a mapping")
    validate_spec(spec, verify_inputs=True)
    root = output_path(spec, "root")
    if root.exists() and any(root.rglob("*")):
        raise RuntimeError("formal Wave 1.1 output exists before sealing")
    exclusions = _prior_exclusions(spec)
    if exclusions["pusht"]["count"] != 3360:
        raise ValueError("PushT prior exclusion must contain exactly 3360 episodes")
    rng = _reacher_rng_exclusion(spec)
    producers: dict[str, str] = {}
    for relative in PRODUCERS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        producers[relative] = sha256_file(path)
    spec_hash = sha256_file(DEFAULT_SPEC)
    payload = {
        "schema_version": 1, "study": spec["study"],
        "spec_path": str(DEFAULT_SPEC.relative_to(ROOT)),
        "spec_sha256": spec_hash,
        "locked_before_wave1_1_outcomes": True,
        "formal_output_absent_at_lock": True,
        "admission_informed_selection_disclosed": True,
        "prior_carrier_outcomes_observed": False,
        "prior_hdf_exclusions": exclusions,
        "reacher_rng_exclusion": rng,
        "producers": producers,
    }
    if not args.execute:
        print(stable_json(payload), end="")
        return
    atomic_text(DEFAULT_LOCK, stable_json(payload))
    atomic_text(DEFAULT_SHA, f"{spec_hash}  {DEFAULT_SPEC.name}\n")
    print(json.dumps({
        "spec_sha256": spec_hash,
        "lock_sha256": sha256_file(DEFAULT_LOCK),
        "producers": len(producers),
        "pusht_prior_exclusion_count": exclusions["pusht"]["count"],
        "pusht_prior_exclusion_sha256": exclusions["pusht"]["indices_sha256"],
        "reacher_rng_registry_sha256": rng["registry_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
