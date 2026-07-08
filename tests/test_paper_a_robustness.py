from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_paper_a_fresh_validation import cached_cue_features
from scripts.launch_paper_a_robustness import build_jobs
from scripts.make_paper_a_robustness_data import bank_path, selected_pairs
from scripts.paper_a_robustness_spec import (
    DEFAULT_SPEC,
    RobustnessSpecError,
    load_locked_spec,
    resolve_spec_path,
    validate_device,
)
from scripts.train_frozen_official_swap import strengthening_provenance


def test_locked_spec_matches_parent_artifacts_and_disjoint_roots() -> None:
    spec = load_locked_spec()
    assert spec["_spec_record"]["sha256"] == (
        "10637be6d29a5b8ec8b7ade9684e009da3f0dc7f66d89ac12a2b8c73d5120a56"
    )
    parent = resolve_spec_path(spec, spec["parent"]["root"])
    output = resolve_spec_path(spec, spec["output"]["root"])
    assert parent != output
    assert parent not in output.parents
    assert spec["carrier_seed_extension"]["seeds"] == [5, 6, 7, 8, 9]


def test_locked_spec_rejects_byte_change(tmp_path: Path) -> None:
    config = tmp_path / DEFAULT_SPEC.name
    lock = config.with_suffix(".sha256")
    shutil.copyfile(DEFAULT_SPEC, config)
    shutil.copyfile(DEFAULT_SPEC.with_suffix(".sha256"), lock)
    config.write_text(config.read_text() + "\n# unauthorized change\n")
    with pytest.raises(RobustnessSpecError, match="hash mismatch"):
        load_locked_spec(config, verify_parent=False, root=tmp_path)


@pytest.mark.parametrize("device", ["cuda:0", "cuda:3"])
def test_forbidden_gpus_fail_closed(device: str) -> None:
    spec = load_locked_spec(verify_parent=False)
    with pytest.raises(RobustnessSpecError, match="forbidden"):
        validate_device(spec, device)


def test_robustness_job_grids_are_complete_and_isolated() -> None:
    spec = load_locked_spec()
    expected = {
        "fresh-data": 4,
        "fresh-cache": 4,
        "fresh-eval": 100,
        "seed-extension": 30,
    }
    robustness_root = resolve_spec_path(spec, spec["output"]["root"])
    parent_root = resolve_spec_path(spec, spec["parent"]["root"])
    for wave, count in expected.items():
        jobs = build_jobs(spec, wave, "cuda:1", DEFAULT_SPEC)
        assert len(jobs) == count
        assert len({job.done_file for job in jobs}) == count
        assert all(robustness_root in job.done_file.parents for job in jobs)
        assert all(parent_root not in job.done_file.parents for job in jobs)
        command_text = "\n".join(" ".join(job.command) for job in jobs)
        assert "cuda:0" not in command_text
        assert "cuda:3" not in command_text


def test_seed_extension_jobs_use_only_locked_primary_deck() -> None:
    spec = load_locked_spec()
    jobs = build_jobs(spec, "seed-extension", "cuda:2", DEFAULT_SPEC)
    assert {job.name.rsplit("_s", 1)[1] for job in jobs} == {
        "5", "6", "7", "8", "9"
    }
    assert all("--provenance-spec" in job.command for job in jobs)
    assert all("cuda:2" in job.command for job in jobs)
    assert not any("_none_" in f"_{job.name}_" for job in jobs)
    assert not any("_lstm_" in f"_{job.name}_" for job in jobs)


def test_fresh_bank_paths_and_selection_are_locked() -> None:
    spec = load_locked_spec(verify_parent=False)
    pairs = selected_pairs(spec, None, None, True)
    assert pairs == [
        ("t1", "fresh-a"), ("t1", "fresh-b"),
        ("t3", "fresh-a"), ("t3", "fresh-b"),
    ]
    assert bank_path(spec, "t1", "fresh-a").name == (
        "val_clean_e240_s270703.npz")
    with pytest.raises(ValueError, match="requires --task and --bank"):
        selected_pairs(spec, "t1", None, False)


def test_cached_cue_features_obey_per_episode_windows() -> None:
    z = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    data = {
        "z": z,
        "event_cue_on": np.asarray([1, 2]),
        "event_cue_off": np.asarray([5, 6]),
    }
    features = cached_cue_features(data)
    assert features.shape == (2, 12)
    np.testing.assert_array_equal(features[0].reshape(4, 3), z[0, [1, 2, 3, 4]])
    np.testing.assert_array_equal(features[1].reshape(4, 3), z[1, [2, 3, 4, 5]])


def _extension_args(spec: dict) -> argparse.Namespace:
    extension = spec["carrier_seed_extension"]
    return argparse.Namespace(
        task="t1",
        arm="gru",
        seed=5,
        epochs=extension["epochs"],
        batch_size=extension["batch_size"],
        lr=extension["learning_rate"],
        weight_decay=extension["weight_decay"],
        study="official-lewm-frozen-carrier-seed-extension-v1",
        provenance_spec=DEFAULT_SPEC,
        output=str(resolve_spec_path(
            spec, spec["output"]["carrier_seed_extension"])),
        weights=str(resolve_spec_path(
            spec, spec["parent"]["official_weights"]["path"])),
        device="cuda:1",
    )


def test_trainer_accepts_only_locked_seed_extension_inputs() -> None:
    spec = load_locked_spec()
    args = _extension_args(spec)
    train = resolve_spec_path(
        spec, spec["parent"]["train_caches"]["t1"]["path"])
    validation = resolve_spec_path(
        spec, spec["parent"]["validation_caches"]["t1"]["path"])
    assert strengthening_provenance(args, train, validation) == \
        spec["_spec_record"]
    args.seed = 4
    with pytest.raises(ValueError, match="outside the locked"):
        strengthening_provenance(args, train, validation)


def test_trainer_rejects_forbidden_gpu_even_when_called_directly() -> None:
    spec = load_locked_spec()
    args = _extension_args(spec)
    args.device = "cuda:3"
    train = resolve_spec_path(
        spec, spec["parent"]["train_caches"]["t1"]["path"])
    validation = resolve_spec_path(
        spec, spec["parent"]["validation_caches"]["t1"]["path"])
    with pytest.raises(RobustnessSpecError, match="forbidden"):
        strengthening_provenance(args, train, validation)
