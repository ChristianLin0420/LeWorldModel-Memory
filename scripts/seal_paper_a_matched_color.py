#!/usr/bin/env python3
"""Seal adaptive Wave-1b code and V1 fresh-data exclusions before outcomes."""

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
from scripts.paper_a_matched_color_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HDF_HOSTS,
    output_path,
    resolve_path,
    sha256_file,
    validate_spec,
)


# This list intentionally includes the locked V1 implementation imported by
# the adaptive preparer. Grid producers may be appended before sealing, but no
# listed source may change after the implementation lock is written.
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
    "scripts/paper_a_matched_color_spec.py",
    "scripts/prepare_paper_a_matched_color.py",
    "scripts/train_paper_a_matched_color.py",
    "scripts/aggregate_paper_a_matched_color.py",
    "scripts/launch_paper_a_matched_color.py",
    "scripts/seal_paper_a_matched_color.py",
    "tests/test_paper_a_matched_color.py",
    "tests/test_paper_a_matched_color_grid.py",
)

V1_BASE_SCHEMA = "paper_a_matched_base_cache_v1"
V1_STUDY = "paper-a-matched-host-v1"
V1_SPEC_SHA = "5febf1c31d8a9f73c83a4d26ba4ed0f9934a23e6099840e50ceba32b4b740b7f"
V1_LOCK_SHA = "83c2ae48ea53e2080606fa7e15e30339668deb338fd556527025e9418b076998"


def _indices_sha256(indices: list[int]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _verified_v1_manifest(spec: dict[str, Any], host: str) -> dict[str, Any]:
    relative = spec["inputs"][host]["v1_host_manifest_path"]
    path = resolve_path(relative)
    if not path.is_file():
        raise FileNotFoundError(
            f"cannot seal until V1 {host} host receipt exists: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid V1 {host} host receipt") from error
    lock = value.get("lock", {})
    if value.get("schema_version") != 1 or value.get("study") != V1_STUDY \
            or value.get("host") != host \
            or value.get("status") not in (
                "admitted", "stopped-admission-failure") \
            or lock.get("sha256") != V1_SPEC_SHA \
            or lock.get("implementation", {}).get("sha256") != V1_LOCK_SHA:
        raise ValueError(f"V1 {host} host receipt has wrong locked identity")
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": value["status"],
        "v1_spec_sha256": lock["sha256"],
        "v1_implementation_lock_sha256": lock["implementation"]["sha256"],
    }


def _verified_v1_base(path_value: str, host: str,
                      split: str) -> tuple[dict[str, Any], np.ndarray]:
    path = resolve_path(path_value)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            f"cannot seal until V1 {host}/{split} cache and sidecar exist")
    arrays, sidecar = load_verified_npz(path)
    lock = sidecar.get("lock", {})
    if sidecar.get("schema") != V1_BASE_SCHEMA \
            or sidecar.get("study") != V1_STUDY \
            or sidecar.get("host") != host \
            or sidecar.get("split") != split \
            or lock.get("sha256") != V1_SPEC_SHA \
            or lock.get("implementation", {}).get("sha256") != V1_LOCK_SHA:
        raise ValueError(
            f"V1 {host}/{split} cache does not embed the pinned V1 lock")
    if "episode_index" not in arrays:
        raise ValueError(f"V1 {host}/{split} cache omits episode_index")
    indices = np.asarray(arrays["episode_index"])
    if indices.dtype != np.int64 or indices.ndim != 1 \
            or np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError(f"V1 {host}/{split} episode_index is invalid")
    expected_count = 1200 if split == "train" else 480
    if len(indices) != expected_count:
        raise ValueError(
            f"V1 {host}/{split} has {len(indices)} rather than {expected_count}")
    values = sorted(map(int, indices))
    receipt = {
        "path": path_value,
        "present_at_lock": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "sidecar_path": str(sidecar_path.relative_to(ROOT)),
        "sidecar_size": sidecar_path.stat().st_size,
        "sidecar_sha256": sha256_file(sidecar_path),
        "split": split,
        "episode_count": len(values),
        "episode_indices_sha256": _indices_sha256(values),
        "embedded_v1_spec_sha256": lock["sha256"],
        "embedded_v1_implementation_lock_sha256": (
            lock["implementation"]["sha256"]),
    }
    return receipt, indices


def collect_v1_hdf_exclusions(spec: dict[str, Any]) -> dict[str, Any]:
    """Verify all V1 HDF receipts before extracting their episode indices."""

    output: dict[str, Any] = {}
    for host in HDF_HOSTS:
        candidates: list[dict[str, Any]] = []
        arrays: list[np.ndarray] = []
        paths = spec["inputs"][host]["v1_base_cache_paths"]
        for split, path in zip(("train", "validation"), paths, strict=True):
            receipt, indices = _verified_v1_base(path, host, split)
            candidates.append(receipt)
            arrays.append(indices)
        if len(np.intersect1d(arrays[0], arrays[1])):
            raise ValueError(f"V1 {host} train/validation caches overlap")
        indices = sorted(map(int, np.concatenate(arrays)))
        if len(indices) != 1680 or len(set(indices)) != len(indices):
            raise ValueError(f"V1 {host} exclusion union must contain 1680 episodes")
        output[host] = {
            "policy": spec["adaptive_origin"]["hdf_v1_exclusion_policy"],
            "v1_host_manifest": _verified_v1_manifest(spec, host),
            "cache_candidates": candidates,
            "episode_indices": indices,
            "count": len(indices),
            "indices_sha256": _indices_sha256(indices),
        }
    return output


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if DEFAULT_SHA.exists() or DEFAULT_LOCK.exists():
        raise FileExistsError("adaptive Wave-1b protocol is already sealed")
    value = yaml.safe_load(DEFAULT_SPEC.read_text())
    if not isinstance(value, dict):
        raise ValueError("Wave-1b protocol must contain a mapping")
    validate_spec(value, verify_inputs=True)
    formal_root = output_path(value, "root")
    if formal_root.exists() and any(formal_root.rglob("*")):
        raise RuntimeError(
            "refusing to seal after any Wave-1b output was created")
    exclusions = collect_v1_hdf_exclusions(value)
    producers: dict[str, str] = {}
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
        "adaptive_locked_before_formal_outcomes": True,
        "formal_output_absent_at_lock": True,
        "v1_metrics_used_for_wave1b_inference": False,
        "v1_hdf_exclusions": exclusions,
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
        "v1_hdf_exclusion_counts": {
            host: record["count"] for host, record in exclusions.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["PRODUCERS", "collect_v1_hdf_exclusions"]
