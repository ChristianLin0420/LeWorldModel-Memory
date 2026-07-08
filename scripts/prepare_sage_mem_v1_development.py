#!/usr/bin/env python3
"""Build deterministic SAGE-Mem development banks from parent TRAIN rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sage_mem_v1 import atomic_json  # noqa: E402
from scripts.sage_mem_v1_spec import (  # noqa: E402
    COHORTS, DEFAULT_SPEC, canonical_json, load_spec, output_root,
    resolve_repo_path, sha256_file, spec_fingerprint,
)


SOURCES = {
    "lewm_reacher_color": (
        "outputs/paper_a_matched_color_v1_1/cache/reacher/base/train.npz",
        None,
    ),
    "lewm_pusht_color": (
        "outputs/paper_a_matched_color_v1_1/cache/pusht/base/train.npz",
        None,
    ),
    "dinowm_pusht_token": (
        "outputs/dinowm_wave2_spatial_carrier_v1_1/cache/metadata.npz", 0,
    ),
    "dinowm_pusht_binding": (
        "outputs/dinowm_wave2_spatial_carrier_v1_1/cache/metadata.npz", 0,
    ),
    "dinowm_pointmaze_goal": (
        "outputs/dinowm_pointmaze_wave3/cache/metadata.npz", 0,
    ),
}


class DevelopmentBankError(RuntimeError):
    """A source cannot prove that selected rows belong to parent TRAIN."""


def _selection_sha256(rows: np.ndarray, episodes: np.ndarray,
                      starts: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (rows, episodes, starts):
        canonical = np.ascontiguousarray(value.astype("<i8", copy=False))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def build_development_manifest(spec: Mapping[str, Any], cohort: str
                               ) -> dict[str, Any]:
    if cohort not in COHORTS:
        raise DevelopmentBankError(f"unknown cohort: {cohort}")
    source_relative, train_code = SOURCES[cohort]
    source = resolve_repo_path(source_relative)
    if not source.is_file():
        raise DevelopmentBankError(f"parent TRAIN source missing: {source}")
    if train_code is None:
        sidecar = source.with_suffix(source.suffix + ".json")
        if not sidecar.is_file():
            raise DevelopmentBankError(f"TRAIN sidecar missing: {sidecar}")
        identity = json.loads(sidecar.read_text())
        if identity.get("split") != "train":
            raise DevelopmentBankError("LeWM source sidecar is not TRAIN")
    else:
        sidecar = None
    with np.load(source, allow_pickle=False) as archive:
        required = {"episode_index"}
        if not required.issubset(archive.files):
            raise DevelopmentBankError("source lacks episode_index")
        episodes = np.asarray(archive["episode_index"], dtype=np.int64)
        starts = (np.asarray(archive["local_start"], dtype=np.int64)
                  if "local_start" in archive.files
                  else np.zeros_like(episodes))
        if train_code is None:
            eligible = np.arange(episodes.size, dtype=np.int64)
            split_proof = "sidecar split=train; file contains TRAIN rows only"
        else:
            if "split" not in archive.files:
                raise DevelopmentBankError("source lacks explicit split codes")
            split = np.asarray(archive["split"])
            eligible = np.flatnonzero(split == train_code).astype(np.int64)
            if np.any(split[eligible] != train_code):
                raise DevelopmentBankError("non-TRAIN row entered eligibility")
            split_proof = f"metadata split == {train_code}"
    requested = int(spec["cohorts"][cohort]["split_episodes"]["development"])
    if eligible.size < requested:
        raise DevelopmentBankError(
            f"only {eligible.size} TRAIN rows available; need {requested}")
    seed_key = f"{cohort}/development/episode_selection"
    seed = int(spec["_seed_registry"][seed_key])
    permutation = np.random.default_rng(seed).permutation(eligible)
    rows = np.sort(permutation[:requested])
    selected_episodes = episodes[rows]
    selected_starts = starts[rows]
    if np.unique(np.stack((selected_episodes, selected_starts), axis=1),
                 axis=0).shape[0] != requested:
        raise DevelopmentBankError("development windows are not unique")
    return {
        "schema_version": 1,
        "study": "sage-mem-v1",
        "stage": "development-bank",
        "status": "prepared-parent-train-only",
        "cohort": cohort,
        "protocol_fingerprint": spec_fingerprint(spec),
        "source": {
            "path": source_relative,
            "size": source.stat().st_size,
            "sha256": sha256_file(source),
            "sidecar": (str(sidecar.relative_to(ROOT)) if sidecar else None),
            "sidecar_sha256": (sha256_file(sidecar) if sidecar else None),
        },
        "selection": {
            "count": requested,
            "eligible_parent_train_rows": int(eligible.size),
            "seed": seed,
            "rows": rows.tolist(),
            "episode_indices": selected_episodes.tolist(),
            "local_starts": selected_starts.tolist(),
            "sha256": _selection_sha256(
                rows, selected_episodes, selected_starts),
        },
        "split_proof": split_proof,
        "parent_train_only": True,
        "parent_validation_or_test_read": False,
        "semantic_labels_read_for_selection": False,
        "permitted_use": "hyperparameter development only",
        "formal_evidence_permitted": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_spec(args.spec)
    destination = (output_root(spec) / "development_banks" / args.cohort
                   / "manifest.json")
    if not args.execute:
        print(canonical_json({
            "study": "sage-mem-v1", "preview": True,
            "stage": "development-bank", "cohort": args.cohort,
            "source": SOURCES[args.cohort][0],
            "parent_train_only": True, "formal_evidence_permitted": False,
        }))
        return
    if destination.exists():
        if not args.resume:
            raise FileExistsError(f"development bank exists: {destination}")
        existing = json.loads(destination.read_text())
        expected = build_development_manifest(spec, args.cohort)
        if existing != expected:
            raise DevelopmentBankError("existing development manifest changed")
        print(canonical_json(existing))
        return
    value = build_development_manifest(spec, args.cohort)
    atomic_json(destination, value)
    print(canonical_json(value))


if __name__ == "__main__":
    main()
