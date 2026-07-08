from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_a_delayed_goal_use import crossed_bootstrap_ci
from scripts.launch_paper_a_delayed_goal_use import build_jobs, select_shard
from scripts.paper_a_delayed_goal_spec import (
    DEFAULT_SPEC,
    DelayedGoalSpecError,
    SOURCE_IDS,
    load_locked_spec,
    resolve_path,
    validate_device,
)
from scripts.paper_a_delayed_goal_use import (
    action_time_interface,
    carrier_interface,
    cue_repair_target,
    cue_window_interface,
    decision_metrics,
    fit_shared_consumer,
    long_context_interface,
    pd_action,
    wrapped_rms,
)
from scripts.train_paper_a_delayed_goal_repair import _label_free_cache


def _bank(episodes: int = 8) -> dict[str, np.ndarray]:
    z = np.arange(episodes * 64 * 192, dtype=np.float32).reshape(
        episodes, 64, 192)
    actions = np.arange(episodes * 63 * 10, dtype=np.float32).reshape(
        episodes, 63, 10)
    return {
        "z": z,
        "actions": actions,
        "event_cue_on": np.full(episodes, 4, dtype=np.int64),
        "event_cue_off": np.full(episodes, 12, dtype=np.int64),
    }


def test_locked_spec_is_parent_disjoint_and_causally_explicit() -> None:
    spec = load_locked_spec()
    parent = resolve_path(spec["parent"]["root"])
    output = resolve_path(spec["output"]["root"])
    assert parent != output and parent not in output.parents
    assert spec["repair"]["repair_objective_forbidden_inputs"] == [
        "xi", "z[:,63]", "validation data"]
    assert spec["repair"]["cue_repair_weight"] == {
        "objective_off": 0.0, "cue_repair": 1.0}
    assert "model-predictive control" in spec[
        "claim_boundary"]["does_not_establish"]


def test_locked_spec_rejects_byte_change(tmp_path: Path) -> None:
    config = tmp_path / DEFAULT_SPEC.name
    lock = config.with_suffix(".sha256")
    shutil.copyfile(DEFAULT_SPEC, config)
    shutil.copyfile(DEFAULT_SPEC.with_suffix(".sha256"), lock)
    config.write_text(config.read_text() + "\n# changed\n")
    with pytest.raises(DelayedGoalSpecError, match="hash mismatch"):
        load_locked_spec(config, verify_parent=False, root=tmp_path)


@pytest.mark.parametrize("device", ["cuda:0", "cuda:3"])
def test_forbidden_devices_fail_closed(device: str) -> None:
    spec = load_locked_spec(verify_parent=False)
    with pytest.raises(DelayedGoalSpecError, match="forbidden"):
        validate_device(spec, device)
    with pytest.raises(DelayedGoalSpecError, match="forbidden"):
        build_jobs(spec, "repair", device, DEFAULT_SPEC)


def test_job_grids_are_complete_semantic_and_parent_disjoint() -> None:
    spec = load_locked_spec()
    counts = {"repair": 40, "evaluate": 10, "aggregate": 1}
    parent = resolve_path(spec["parent"]["root"])
    output = resolve_path(spec["output"]["root"])
    for wave, count in counts.items():
        jobs = build_jobs(spec, wave, "cuda:2", DEFAULT_SPEC)
        assert len(jobs) == count
        assert len({job.name for job in jobs}) == count
        assert len({job.done_file for job in jobs}) == count
        assert all(output == job.done_file.parent
                   or output in job.done_file.parents for job in jobs)
        assert all(parent not in job.done_file.parents for job in jobs)
        assert all("--execute" in job.command for job in jobs)
    evaluation_names = [job.name for job in build_jobs(
        spec, "evaluate", "cuda:1", DEFAULT_SPEC)]
    assert "transient-marker-recall_checkpoint-seed-0" in evaluation_names
    assert all("t1_" not in name and "t3_" not in name
               for name in evaluation_names)


def test_repair_sharding_keeps_objective_twins_on_one_device() -> None:
    spec = load_locked_spec()
    jobs = build_jobs(spec, "repair", "cuda:1", DEFAULT_SPEC)
    shards = [select_shard(jobs, "repair", 2, index) for index in range(2)]
    assert [len(shard) for shard in shards] == [20, 20]
    for shard in shards:
        stems = [job.name.rsplit("_", 1)[0] for job in shard]
        assert len(stems) == 20
        assert all(stems.count(stem) == 2 for stem in set(stems))


def test_registered_interfaces_have_equal_dimension_and_exclude_final_frame() -> None:
    data = _bank(4)
    prior = np.asarray(data["z"] + 100.0, dtype=np.float32)
    carrier = carrier_interface(data, prior)
    context = long_context_interface(data)
    cue = cue_window_interface(data)
    repair, indices = cue_repair_target(data)
    assert carrier.shape == context.shape == cue.shape == repair.shape == (4, 768)
    np.testing.assert_array_equal(carrier[:, :576], data["z"][:, 60:63].reshape(4, -1))
    np.testing.assert_array_equal(carrier[:, 576:], prior[:, 63])
    assert indices.max() < 63
    altered = {**data, "z": data["z"].copy()}
    altered["z"][:, 63] = -9e8
    np.testing.assert_array_equal(cue_repair_target(altered)[0], repair)


def test_action_time_control_is_label_free_and_padded() -> None:
    data = _bank(5)
    first = action_time_interface(data)
    second = action_time_interface({**data, "xi": np.arange(5) % 4})
    assert first.shape == (5, 768)
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first[:, 33:]) == 0


def test_repair_cache_loader_never_returns_labels(tmp_path: Path) -> None:
    data = _bank(4)
    path = tmp_path / "cache.npz"
    np.savez(path, **data, xi=np.asarray([0, 1, 2, 3]))
    loaded = _label_free_cache(path)
    assert set(loaded) == {
        "z", "actions", "event_cue_on", "event_cue_off"}
    assert "xi" not in loaded


def test_one_arm_blind_consumer_fits_the_complete_source_deck() -> None:
    labels = np.tile(np.arange(4, dtype=np.int64), 12)
    base = np.zeros((len(labels), 768), dtype=np.float32)
    base[np.arange(len(labels)), labels] = 6.0
    features = {
        source: base + np.float32(index * 1e-3)
        for index, source in enumerate(SOURCE_IDS)
    }
    consumer = fit_shared_consumer(
        features, labels, list(SOURCE_IDS), c=1.0, solver="lbfgs",
        max_iter=2000, random_state=0)
    assert all(np.array_equal(consumer.predict(value), labels)
               for value in features.values())
    shuffled = fit_shared_consumer(
        features, labels, list(SOURCE_IDS), c=1.0, solver="lbfgs",
        max_iter=2000, random_state=0,
        label_permutation=np.roll(np.arange(len(labels)), 1))
    assert consumer.digest() != shuffled.digest()


def test_physical_choice_helpers_and_regret_are_interpretable() -> None:
    assert wrapped_rms(np.asarray([np.pi, -np.pi]),
                       np.asarray([-np.pi, np.pi])) < 1e-6
    action = pd_action(
        np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0]),
        np.asarray([2.0, -2.0]), 1.5, 0.25,
        np.asarray([-1.0, -1.0]), np.asarray([1.0, 1.0]))
    np.testing.assert_array_equal(action, np.asarray([1.0, -1.0]))
    labels = np.asarray([0, 1])
    prediction = np.asarray([0, 0])
    execution = {
        "distance": np.asarray([[0.1, 0.8], [0.9, 0.2]]),
        "return": np.asarray([[0.9, 0.2], [0.1, 0.8]]),
        "success": np.asarray([[True, False], [False, True]]),
    }
    metrics = decision_metrics(prediction, labels, execution)
    assert metrics["goal_decision_accuracy"] == 0.5
    assert metrics["executed_success_rate"] == 0.5
    np.testing.assert_allclose(metrics["regret_to_label_oracle"], [0.0, 0.7])


def test_crossed_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    reference = np.arange(5 * 240, dtype=np.float64).reshape(5, 240)
    difference = (reference + 0.125) - reference
    first = crossed_bootstrap_ci(difference, draws=1000, seed=19)
    second = crossed_bootstrap_ci(difference, draws=1000, seed=19)
    assert first == second
    assert first["mean"] == pytest.approx(0.125)
    assert first["ci_low"] == pytest.approx(0.125)
    assert first["ci_high"] == pytest.approx(0.125)
