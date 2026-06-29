#!/usr/bin/env python3
"""Unit tests for the frozen LOIF-v9 diagnostic and donor contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hacssm_v9_diagnostics import (
    DONOR_CONTRACT,
    DONOR_CONTRACT_SHA256,
    build_resistance_overrides,
    diagnostic_phase_masks,
    donor_phase_labels,
    summarize_loif_details,
)
from lewm.models.memory_model import MemoryLeWorldModel


def test_contract_hash_and_phase_partitions() -> None:
    observed = hashlib.sha256(json.dumps(
        DONOR_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    assert observed == DONOR_CONTRACT_SHA256
    labels = donor_phase_labels(32, 3)
    assert labels.tolist() == (
        [0] * 10 + [1] * 3 + [2] * 3 + [3] + [4] * 2 + [5] * 13
    )
    masks = diagnostic_phase_masks(32, 3)
    assert {name: int(mask.sum()) for name, mask in masks.items()} == {
        "visible": 22,
        "blackout_transition": 3,
        "deep_blackout": 3,
        "recovery": 3,
    }
    covered = sum(mask.astype(np.int64) for mask in masks.values())
    assert covered[0] == 0
    assert np.array_equal(covered[1:], np.ones(31, dtype=np.int64))


def test_training_donor_overrides_are_deterministic_causal_and_phase_matched() -> None:
    # The decimal encoding lets the test recover source episode and time from every donor.
    train = np.asarray([
        [episode * 100 + time + 1 for time in range(32)]
        for episode in range(7)
    ], dtype=np.float64)
    one_perm, one_mean = build_resistance_overrides(
        train, validation_episodes=11, history_len=3
    )
    two_perm, two_mean = build_resistance_overrides(
        train, validation_episodes=11, history_len=3
    )
    assert np.array_equal(one_perm, two_perm)
    assert np.array_equal(one_mean, two_mean)
    labels = donor_phase_labels(32, 3)
    for target_t in range(32):
        source_t = ((one_perm[:, target_t].astype(np.int64) - 1) % 100)
        assert np.all(source_t <= target_t)
        assert np.all(labels[source_t] == labels[target_t])
        eligible = np.flatnonzero(
            (labels == labels[target_t]) & (np.arange(32) <= target_t)
        )
        expected_mean = train[:, eligible].mean()
        assert np.allclose(one_mean[:, target_t], expected_mean)


def test_summary_schema_and_nominal_direct_coefficients() -> None:
    episodes, length = 4, 32
    alpha_fast, alpha_slow = 0.5, 0.75
    details = {
        "log_R": np.full((episodes, length), np.log(2.0)),
        "gains": np.stack([
            np.full((episodes, length), 0.2),
            np.full((episodes, length), 0.4),
        ], axis=-1),
        "log_P": np.stack([
            np.full((episodes, length), np.log(0.3)),
            np.full((episodes, length), np.log(0.6)),
        ], axis=-1),
        "prior_weights": np.stack([
            np.full((episodes, length), 0.6),
            np.full((episodes, length), 0.4),
        ], axis=-1),
        "read_weights": np.stack([
            np.full((episodes, length), 0.7),
            np.full((episodes, length), 0.3),
        ], axis=-1),
        "innovation_norm": np.full((episodes, length), 1.25),
    }
    observed = summarize_loif_details(
        details,
        alpha_fast=alpha_fast,
        alpha_slow=alpha_slow,
        q_fast=1.0 - alpha_fast ** 2,
        q_slow=1.0 - alpha_slow ** 2,
        history_len=3,
    )
    assert len(observed) >= 4 + 13 * 4
    for phase in ("visible", "blackout_transition", "deep_blackout", "recovery"):
        assert np.isclose(observed[f"loif_log_R_{phase}"], np.log(2.0))
        assert np.isclose(observed[f"loif_direct_fast_{phase}"], 0.4)
        assert np.isclose(observed[f"loif_direct_slow_{phase}"], 0.45)

    # A finite collapsed/boundary solution is a recorded scientific stop, not a missing cell.
    collapsed = summarize_loif_details(
        details, alpha_fast=0.0, alpha_slow=0.0, q_fast=1.0, q_slow=1.0,
        history_len=3,
    )
    assert collapsed["loif_pole_collapsed"] is True
    assert collapsed["loif_boundary_saturated"] is True
    assert collapsed["loif_pole_separation"] == 0.0
    assert collapsed["loif_fast_boundary_margin"] == 0.0


def test_end_to_end_candidate_diagnostics_are_finite_and_complete() -> None:
    torch.manual_seed(902)
    length, dimension, action_dim = 32, 8, 3
    model = MemoryLeWorldModel(
        img_size=8, patch_size=4, embed_dim=dimension, action_dim=action_dim,
        encoder_layers=1, encoder_heads=2, predictor_layers=1, predictor_heads=2,
        predictor_norm="none", history_len=2, dropout=0.0, sigreg_projections=8,
        encoder_type="precomputed", memory_impl="loifv9", memory_mode="both",
        hier_loss_weight=0.0,
    )

    def dataset(episodes: int) -> TensorDataset:
        observed = torch.randn(episodes, length, dimension)
        actions = torch.randn(episodes, length - 1, action_dim)
        target = torch.randn(episodes, length, dimension)
        visible = torch.ones(episodes, length, dtype=torch.bool)
        visible[:, 10:16] = False
        return TensorDataset(observed, actions, target, visible)

    from hacssm_v9_diagnostics import evaluate_loif_v9_diagnostics
    observed = evaluate_loif_v9_diagnostics(
        model, dataset(8), dataset(4), device=torch.device("cpu"), use_amp=False,
        history_len=2, batch_size=4,
    )
    assert observed["loif_diagnostics_schema_version"] == 1
    assert observed["loif_donor_contract_sha256"] == DONOR_CONTRACT_SHA256
    assert observed["loif_donor_train_episodes"] == 8
    assert observed["loif_donor_val_episodes"] == 4
    assert len(observed) >= 99
    required_interventions = {
        f"clean_mse_{phase}_resistance_{kind}"
        for phase in ("first_post", "deep_blackout", "all")
        for kind in ("permuted", "mean")
    }
    assert required_interventions <= set(observed)
    for key, value in observed.items():
        if isinstance(value, (int, float)):
            assert np.isfinite(value), key


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v9 diagnostic tests passed.")
