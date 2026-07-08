"""CPU/synthetic tests for the isolated label-free delayed repair V2."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier  # noqa: E402
from scripts.delayed_repair_residual_v2_objective import (  # noqa: E402
    cue_residual_target,
    development_health,
    fit_target_standardizer,
    load_label_free_bank,
    make_epoch_plans,
    reconstruction_metrics,
)
from scripts.delayed_repair_residual_v2_spec import (  # noqa: E402
    load_locked_spec,
    validate_device,
)
from scripts.launch_delayed_repair_residual_v2 import (  # noqa: E402
    WAVES,
    build_plan,
    parse_gpu_ids,
    preview_lines,
)


def _linear_cue_bank(episodes: int = 6) -> dict:
    time = np.arange(64, dtype=np.float32)[None, :, None]
    coordinate = np.linspace(-0.2, 0.2, 192, dtype=np.float32)[None, None]
    z = np.broadcast_to(time * coordinate, (episodes, 64, 192)).copy()
    cue_on = np.arange(episodes, dtype=np.int64) % 5 + 8
    cue_off = cue_on + 4
    delta = np.linspace(-2, 2, 192, dtype=np.float32)
    for episode in range(episodes):
        z[episode, cue_on[episode]:cue_off[episode]] += delta
    return {
        "z": z,
        "actions": np.zeros((episodes, 63, 10), dtype=np.float32),
        "event_cue_on": cue_on,
        "event_cue_off": cue_off,
        "delta": delta,
    }


def test_cue_residual_exactly_removes_linear_scene_and_excludes_decision() -> None:
    bank = _linear_cue_bank()
    target, audit = cue_residual_target(bank)
    np.testing.assert_allclose(
        target, np.broadcast_to(bank["delta"], target.shape), atol=2e-6)
    assert target.shape == (6, 192)
    assert audit["decision_frame_excluded"] is True
    assert audit["post_index_max"] < audit["decision_index"] == 63
    before = target.copy()
    bank["z"][:, 63] = 1_000_000
    after, changed_audit = cue_residual_target(bank)
    np.testing.assert_array_equal(before, after)
    assert audit == changed_audit


def test_label_free_loader_never_returns_xi(tmp_path) -> None:
    bank = _linear_cue_bank(3)
    path = tmp_path / "bank.npz"
    np.savez(
        path, z=bank["z"], actions=bank["actions"],
        event_cue_on=bank["event_cue_on"],
        event_cue_off=bank["event_cue_off"],
        xi=np.asarray([3, 2, 1], dtype=np.int64))
    loaded = load_label_free_bank(path)
    assert "xi" not in loaded
    assert loaded["label_arrays_present_but_not_loaded"] == ["xi"]
    assert loaded["label_arrays_loaded"] is False


def test_standardizer_uses_supplied_training_target_only() -> None:
    rng = np.random.default_rng(7)
    training = rng.normal(size=(300, 192)).astype(np.float32)
    validation = rng.normal(loc=100, size=(40, 192)).astype(np.float32)
    standardizer = fit_target_standardizer(training, 1e-6)
    normalized = standardizer.transform(training)
    np.testing.assert_allclose(normalized.mean(0), 0, atol=2e-6)
    np.testing.assert_allclose(normalized.std(0), 1, atol=2e-6)
    before = standardizer.digest()
    standardizer.transform(validation)
    assert standardizer.digest() == before


def test_development_health_is_unlabeled_and_fail_closed() -> None:
    bank = _linear_cue_bank(6)
    target, audit = cue_residual_target(bank)
    protocol = {
        "required_episodes": 6,
        "cue_duration_min": 4,
        "cue_duration_max": 6,
        "coordinate_std_min": 0.05,
        "coordinate_std_fraction_min": 0.0,
        "median_episode_residual_rms_min": 0.1,
        "median_episode_residual_rms_max": 10.0,
    }
    report = development_health(target, audit, protocol)
    assert report["passed"] is True
    assert report["labels_loaded"] is False
    protocol["coordinate_std_fraction_min"] = 1.0
    assert development_health(target, audit, protocol)["passed"] is False


def test_objective_twins_receive_identical_deterministic_plans() -> None:
    first, first_digest = make_epoch_plans(
        1200, epochs=20, batch_size=64, windows_per_batch=8,
        sequence_length=64, history=3, seed=270003)
    second, second_digest = make_epoch_plans(
        1200, epochs=20, batch_size=64, windows_per_batch=8,
        sequence_length=64, history=3, seed=270003)
    assert first_digest == second_digest
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.order, b.order)
        for x, y in zip(a.starts_by_batch, b.starts_by_batch, strict=True):
            np.testing.assert_array_equal(x, y)
    assert make_epoch_plans(
        1200, epochs=20, batch_size=64, windows_per_batch=8,
        sequence_length=64, history=3, seed=270004)[1] != first_digest


def test_reconstruction_metrics_use_training_mean_zero_baseline() -> None:
    target = np.ones((5, 192), dtype=np.float32)
    perfect = reconstruction_metrics(target, target)
    assert perfect["mse"] == 0.0
    assert perfect["normalized_mse_to_zero"] == 0.0
    assert perfect["r2_vs_training_mean"] == 1.0
    zero = reconstruction_metrics(np.zeros_like(target), target)
    assert zero["mse"] == zero["zero_predictor_mse"] == 1.0
    assert zero["normalized_mse_to_zero"] == 1.0


@pytest.mark.parametrize("arm", ("gru", "ssm"))
def test_prior_63_is_exactly_invariant_to_decision_frame(arm) -> None:
    torch.manual_seed(5)
    carrier = make_frozen_carrier(arm, 192, 10).eval()
    z = torch.randn(3, 64, 192)
    actions = torch.randn(3, 63, 10)
    altered = z.clone()
    altered[:, 63] = 500
    with torch.no_grad():
        original = carrier(z, actions).prior_read[:, 63]
        intervened = carrier(altered, actions).prior_read[:, 63]
    torch.testing.assert_close(original, intervened, rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cuda:1", "cuda:2"))
def test_only_registered_cuda_devices_are_allowed(device) -> None:
    assert validate_device(device) == device


@pytest.mark.parametrize("raw", ("0", "3", "0,1", "2,3", "1,1"))
def test_launcher_rejects_forbidden_or_duplicate_devices(raw) -> None:
    with pytest.raises(ValueError):
        parse_gpu_ids(raw)


def test_locked_spec_and_preview_plan_are_isolated_and_read_only() -> None:
    spec = load_locked_spec()
    assert spec["scientific_role"]["classification"] \
        == "post_v1_diagnostic_repair"
    assert spec["scientific_role"]["preregistered_primary_result"] is False
    assert spec["cue_residual_target"]["dimension"] == 192
    assert spec["formal_repair"]["conditions"] == [
        "objective-off", "cue-residual-repair"]
    isolated = copy.deepcopy(spec)
    root = Path("outputs/.test-delayed-repair-residual-v2-preview")
    absolute_root = ROOT / root
    assert not absolute_root.exists()
    isolated["output"] = {
        "root": str(root),
        "development": str(root / "development"),
        "repairs": str(root / "repairs"),
        "summary": str(root / "summary.json"),
        "logs": str(root / "logs"),
    }
    plan = build_plan(isolated, "all", (1, 2))
    assert [wave for wave, _ in plan] == list(WAVES)
    assert {wave: len(jobs) for wave, jobs in plan} == {
        "development-health": 2, "formal-repair": 40, "aggregate": 1}
    repair_jobs = dict(plan)["formal-repair"]
    for offset in range(0, len(repair_jobs), 2):
        assert repair_jobs[offset].device == repair_jobs[offset + 1].device
        assert repair_jobs[offset].done_file.parent.parent \
            == repair_jobs[offset + 1].done_file.parent.parent
    lines = preview_lines(plan)
    assert len(lines) == 43
    assert all("\tpending\t" in line for line in lines)
    assert not absolute_root.exists()
