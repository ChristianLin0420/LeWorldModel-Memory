from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from lewm.official_tasks.artifacts import write_npz_with_sidecar
from scripts.paper_a_matched_color_spec import (
    AGES,
    ARMS,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    validate_spec,
)
from scripts.prepare_paper_a_matched_color import (
    BASE_KEYS,
    BASE_SCHEMA,
    CUE_KEYS,
    CUE_SCHEMA,
    _fresh_hdf_selections,
    base_cache_path,
    cue_cache_path,
    host_manifest_path,
)
import scripts.seal_paper_a_matched_color as sealer


def _spec() -> dict:
    value = yaml.safe_load(DEFAULT_SPEC.read_text())
    assert isinstance(value, dict)
    return value


def test_adaptive_protocol_is_color_only_and_valid() -> None:
    spec = _spec()
    validate_spec(spec, verify_inputs=False)
    assert tuple(spec["targets"]) == ("color",)
    assert spec["cue"]["location_role"] == "exact-balanced randomized nuisance"
    assert spec["adaptive_origin"]["preserve_v1_failure"] is True
    assert spec["adaptive_origin"][
        "v1_admission_metrics_used_for_wave1b_inference"] is False
    assert spec["outputs"]["root"] == "outputs/paper_a_matched_color_v1"


def test_public_grid_interface_and_cache_contract() -> None:
    assert HOSTS == ("reacher", "pusht", "tworoom")
    assert AGES == (4, 8, 15)
    assert ARMS == ("none", "gru", "lstm", "ssm", "fixed_trust")
    assert SEEDS == (0, 1, 2, 3, 4)
    assert BASE_SCHEMA == "paper_a_matched_color_base_cache_v1"
    assert CUE_SCHEMA == "paper_a_matched_color_cue_cache_v1"
    assert BASE_KEYS == (
        "z_base", "actions", "state", "episode_index", "local_start",
        "global_frame_indices")
    assert CUE_KEYS == (
        "z_cue", "combination_label", "color_label", "location_label",
        "episode_index", "local_start", "cue_on", "cue_off")
    spec = _spec()
    assert base_cache_path(spec, "pusht", "train").name == "train.npz"
    assert cue_cache_path(spec, "pusht", "validation", 15).name == "age-15.npz"
    assert host_manifest_path(spec, "tworoom").name == "manifest.json"


def test_fresh_hdf_selection_excludes_locked_v1_union() -> None:
    spec = _spec()
    excluded = list(range(1680))
    spec["_lock"] = {"implementation": {"v1_hdf_exclusions": {
        "pusht": {"episode_indices": excluded},
    }}}
    dataset = SimpleNamespace(
        frame_skip=5, num_episodes=4000,
        episode_lengths=np.full(4000, 200, dtype=np.int64))
    first = _fresh_hdf_selections(dataset, spec, "pusht")
    second = _fresh_hdf_selections(dataset, spec, "pusht")
    assert first == second
    assert len(first) == 1680
    selected = {item.episode_index for item in first}
    assert len(selected) == 1680
    assert selected.isdisjoint(excluded)
    assert sum(item.split == "train" for item in first) == 1200
    assert sum(item.split == "validation" for item in first) == 480


def test_v1_cache_exclusion_reader_requires_embedded_v1_lock(
        tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    relative = "v1/train.npz"
    path = root / relative
    metadata = {
        "schema": sealer.V1_BASE_SCHEMA,
        "study": sealer.V1_STUDY,
        "host": "pusht", "split": "train",
        "lock": {
            "sha256": sealer.V1_SPEC_SHA,
            "implementation": {"sha256": sealer.V1_LOCK_SHA},
        },
    }
    write_npz_with_sidecar(path, {
        "episode_index": np.arange(1200, dtype=np.int64),
    }, metadata)
    monkeypatch.setattr(sealer, "ROOT", root)
    monkeypatch.setattr(sealer, "resolve_path", lambda value: root / value)
    receipt, indices = sealer._verified_v1_base(relative, "pusht", "train")
    assert len(indices) == 1200
    assert receipt["episode_count"] == 1200
    assert receipt["embedded_v1_spec_sha256"] == sealer.V1_SPEC_SHA
    sidecar = path.with_suffix(".npz.json")
    value = json.loads(sidecar.read_text())
    value["lock"]["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(value))
    try:
        sealer._verified_v1_base(relative, "pusht", "train")
    except ValueError:
        pass
    else:
        raise AssertionError("tampered V1 sidecar was accepted")


def test_sealer_locks_new_and_imported_preparers() -> None:
    required = {
        "scripts/paper_a_matched_color_spec.py",
        "scripts/prepare_paper_a_matched_color.py",
        "scripts/seal_paper_a_matched_color.py",
        "scripts/prepare_paper_a_matched_host.py",
        "lewm/official_tasks/matched_memory.py",
        "lewm/official_tasks/native_sequence_hdf5.py",
        "tests/test_paper_a_matched_color.py",
    }
    assert required.issubset(sealer.PRODUCERS)
