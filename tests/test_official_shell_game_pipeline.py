"""CPU/synthetic tests for the locked official shell-game pipeline."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    load_verified_npz,
    write_npz_with_sidecar,
)
from lewm.official_tasks.shell_game_spec import (  # noqa: E402
    load_locked_spec,
    validate_device,
)
from scripts.launch_official_shell_game_capacity import (  # noqa: E402
    WAVES,
    build_plan,
    parse_gpu_ids,
    preview_lines,
)
from scripts.train_official_shell_game_capacity import (  # noqa: E402
    _decision_features,
    _primary_probe,
)


def test_default_formal_spec_and_all_producers_match_lock() -> None:
    spec = load_locked_spec()
    assert spec["protocol_status"] == "locked_before_formal_run"
    assert [stage["key"] for stage in spec["semantic_stages"]] == [
        "single-item", "two-item", "four-item"]
    assert spec["task_contract"]["final_context_indices"] == [60, 61, 62]
    assert spec["task_contract"]["decision_index"] == 63
    assert spec["task_contract"]["decision_observation_excluded"] is True
    assert len(spec["_lock_record"]["producer_sha256"]) >= 10


@pytest.mark.parametrize("device", ("cuda:1", "cuda:2"))
def test_formal_device_allowlist_accepts_only_registered_devices(device) -> None:
    assert validate_device(device) == device


@pytest.mark.parametrize(
    "device", ("cuda:0", "cuda:3", "cuda", "cpu", "cuda:4"))
def test_formal_device_allowlist_rejects_forbidden_devices(device) -> None:
    with pytest.raises(ValueError, match="permit only"):
        validate_device(device)


@pytest.mark.parametrize("raw", ("0", "3", "0,1", "2,3", "1,1"))
def test_launcher_rejects_forbidden_or_duplicate_gpu_ids(raw) -> None:
    with pytest.raises(ValueError):
        parse_gpu_ids(raw)
    assert parse_gpu_ids("1,2") == (1, 2)
    assert parse_gpu_ids("cuda:2") == (2,)


def test_deterministic_artifact_roundtrip_and_tamper_detection(tmp_path) -> None:
    arrays = {
        "integer": np.arange(12, dtype=np.int64).reshape(3, 4),
        "floating": np.linspace(-1, 1, 9, dtype=np.float32),
    }
    metadata = {"schema": "synthetic_test_v1", "split": "validation"}
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_record = write_npz_with_sidecar(first, arrays, metadata)
    second_record = write_npz_with_sidecar(second, arrays, metadata)
    assert first_record["sha256"] == second_record["sha256"]
    loaded, sidecar = load_verified_npz(first)
    assert sidecar["schema"] == "synthetic_test_v1"
    np.testing.assert_array_equal(loaded["integer"], arrays["integer"])
    np.testing.assert_array_equal(loaded["floating"], arrays["floating"])

    payload = bytearray(first.read_bytes())
    payload[len(payload) // 2] ^= 1
    first.write_bytes(payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_npz(first)


def _balanced_targets(episodes: int, capacity: int, shift: int) -> np.ndarray:
    row = np.arange(episodes, dtype=np.int64)[:, None]
    item = np.arange(capacity, dtype=np.int64)[None, :]
    return (row + 2 * item + shift) % 3


def _synthetic_probe_bank(stage: str, episodes: int, shift: int) -> dict:
    capacity = {"single-item": 1, "two-item": 2, "four-item": 4}[stage]
    targets = _balanced_targets(episodes, capacity, shift)
    z = np.zeros((episodes, 64, 192), dtype=np.float32)
    for episode in range(episodes):
        for item in range(capacity):
            coordinate = item * 3 + int(targets[episode, item])
            # Only the final legal raw context carries this synthetic signal.
            z[episode, 60:63, coordinate] = 8.0
        # A deliberately strong decision-frame value must never enter raw
        # endpoint features.
        z[episode, 63] = 10_000 + episode
    return {"z": z, "final_slots": targets, "meta": {"stage": stage}}


@pytest.mark.parametrize(
    ("stage", "capacity", "exact_chance"),
    (("single-item", 1, 1 / 3),
     ("two-item", 2, 1 / 9),
     ("four-item", 4, 1 / 81)),
)
def test_primary_metrics_and_legal_endpoint_are_synthetic_exact(
        stage, capacity, exact_chance) -> None:
    train = _synthetic_probe_bank(stage, 90, shift=0)
    validation = _synthetic_probe_bank(stage, 45, shift=1)
    train_prior = np.zeros_like(train["z"])
    validation_prior = np.zeros_like(validation["z"])
    report = _primary_probe(
        train, train_prior, validation, validation_prior)
    assert report["mean_per_item_balanced_accuracy"] == pytest.approx(1.0)
    assert report["minimum_per_item_balanced_accuracy"] == pytest.approx(1.0)
    assert report["exact_set_accuracy"] == pytest.approx(1.0)
    assert len(report["per_item"]) == capacity
    assert all(item["balanced_accuracy"] == pytest.approx(1.0)
               for item in report["per_item"])
    assert report["per_item_chance"] == pytest.approx(1 / 3)
    assert report["exact_set_chance"] == pytest.approx(exact_chance)
    endpoint = report["endpoint_contract"]
    assert endpoint["raw_context_indices"] == [60, 61, 62]
    assert endpoint["decision_observation_index"] == 63
    assert endpoint["decision_observation_excluded"] is True
    assert endpoint["prior_index"] == 63
    assert endpoint["prior_timing"] == (
        "before consuming the decision observation")


def test_raw_decision_frame_cannot_change_primary_features() -> None:
    bank = _synthetic_probe_bank("four-item", 6, shift=0)
    prior = np.zeros_like(bank["z"])
    before, endpoint = _decision_features(bank, prior)
    bank["z"][:, 63] = np.random.default_rng(4).normal(
        size=bank["z"][:, 63].shape)
    after, changed_endpoint = _decision_features(bank, prior)
    np.testing.assert_array_equal(before, after)
    assert endpoint == changed_endpoint


def test_launcher_plan_is_semantic_complete_and_read_only(tmp_path) -> None:
    spec = copy.deepcopy(load_locked_spec())
    destination = tmp_path / "formal-output"
    spec["artifacts"]["root"] = str(destination)
    plan = build_plan(spec, "all", (1, 2))
    assert [wave for wave, _ in plan] == list(WAVES)
    counts = {wave: len(jobs) for wave, jobs in plan}
    assert counts == {
        "base": 2,
        "stages": 6,
        "frozen-cache": 3,
        "carriers": 75,
    }
    all_jobs = [job for _, jobs in plan for job in jobs]
    assert len(all_jobs) == 86
    assert all(job.device in (None, "cuda:1", "cuda:2")
               for job in all_jobs)
    assert all(not re.search(r"(^|[-_])[tT]\d+($|[-_])", job.name)
               for job in all_jobs)
    assert sum(job.device is None for job in all_jobs) == 6
    lines = preview_lines(plan)
    assert len(lines) == 86
    assert all("\tpending\t" in line for line in lines)
    assert not destination.exists(), "preview must not create output roots"
