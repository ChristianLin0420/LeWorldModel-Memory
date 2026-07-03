"""Fast unit tests for the V19 P0 host-preflight machinery.

Covers the invariants the preflight relies on without touching MuJoCo or a
GPU: cache-time corruption determinism and shapes, the analytic projected-zero
Epps-Pulley plateau against an empirical delta-distribution computation, the
V18 covariance effective-rank definition, the sliding H=3 window sampler
alignment, and the registered gate logic (rank/variance/convergence plus the
V16 plateau and gradient-ratio collapse flags).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.sigreg import MultiSubspaceSIGReg
from lewm.tasks_v19.base import EpisodeBatch
import scripts.make_v19_p0_data as p0_data
import scripts.train_v19_p0 as p0_train
import scripts.train_lewm_v8_v18 as v18

LENGTH = 64


def _synthetic_bank(episodes: int = 6, length: int = LENGTH) -> EpisodeBatch:
    rng = np.random.default_rng(0)
    return EpisodeBatch(
        frames=rng.integers(0, 256, size=(episodes, length, 64, 64, 3),
                            dtype=np.uint8),
        actions=rng.standard_normal((episodes, length - 1, 2)).astype(np.float32),
        xi=rng.integers(0, 2, size=episodes).astype(np.int64),
        xi_kind="cat", n_classes=2,
        endo_state=np.zeros((episodes, length, 3), dtype=np.float32),
        exo_state=np.zeros((episodes, length, 2), dtype=np.float32),
        events={}, stream="iid", task="t1", seed=0)


# --------------------------------------------------------------------------
# Corruption pipeline
# --------------------------------------------------------------------------

def test_corruption_determinism_shapes_and_modes():
    bank = _synthetic_bank()
    for episode in range(bank.num_episodes):
        clean = bank.frames[episode]
        first, start_a, end_a, mode_a = p0_data.corrupt_episode(clean, episode)
        second, start_b, end_b, mode_b = p0_data.corrupt_episode(clean, episode)
        # Deterministic in (episode index, corruption seed).
        assert (start_a, end_a, mode_a) == (start_b, end_b, mode_b)
        assert np.array_equal(first, second)
        assert first.shape == clean.shape and first.dtype == np.uint8
        # Registered window contract: 6-12 contiguous steps inside the V11
        # start bounds [history+2, length-gap-2].
        gap = end_a - start_a
        assert p0_data.GAP_RANGE[0] <= gap <= p0_data.GAP_RANGE[1]
        assert start_a >= p0_data.HISTORY_LEN + 2
        assert end_a <= LENGTH - 2
        # Deterministic mode alternation by episode index.
        assert mode_a == ("meanframe" if episode % 2 == 0 else "cutout")
        # Untouched outside the window, and the input is not mutated.
        assert np.array_equal(first[:start_a], clean[:start_a])
        assert np.array_equal(first[end_a:], clean[end_a:])
        assert np.array_equal(bank.frames[episode], clean)
        if mode_a == "meanframe":
            expected = np.rint(clean.mean(axis=0)).clip(0, 255).astype(np.uint8)
            assert np.array_equal(first[start_a:end_a],
                                  np.broadcast_to(expected, (gap, *expected.shape)))
        else:
            # Exactly one 55% rectangle differs, filled with the episode
            # channel mean.
            fill = np.rint(clean.mean(axis=(0, 1, 2))).clip(0, 255).astype(np.uint8)
            difference = (first[start_a:end_a] != clean[start_a:end_a]).any(axis=(0, 3))
            rows = np.flatnonzero(difference.any(axis=1))
            columns = np.flatnonzero(difference.any(axis=0))
            cut = int(round(64 * p0_data.CUTOUT_FRACTION))
            assert rows.size <= cut and columns.size <= cut  # ties with fill allowed
            top, left = rows[0], columns[0]
            region = first[start_a:end_a, top:top + cut, left:left + cut]
            assert np.array_equal(region,
                                  np.broadcast_to(fill, region.shape))


def test_corrupt_bank_events_and_windows():
    bank = _synthetic_bank(episodes=5)
    observed = p0_data.corrupt_bank(bank)
    for key in ("corrupt_on", "corrupt_off", "corrupt_mode"):
        assert key in observed.events
        assert observed.events[key].shape == (5,)
    for episode in range(5):
        start, end = p0_data.corruption_window(LENGTH, episode)
        assert observed.events["corrupt_on"][episode] == start
        assert observed.events["corrupt_off"][episode] == end
        assert observed.events["corrupt_mode"][episode] == episode % 2
    # Clean bank untouched; every episode actually corrupted.
    assert not np.array_equal(observed.frames, bank.frames)
    assert np.array_equal(observed.actions, bank.actions)


# --------------------------------------------------------------------------
# Analytic projected-zero plateau (the V16 collapse signature)
# --------------------------------------------------------------------------

def test_analytic_plateau_matches_empirical_delta():
    torch.manual_seed(0)
    sigreg = MultiSubspaceSIGReg(embed_dim=16, num_subspaces=1,
                                 num_projections=64)
    batch, time_steps = 8, 5
    collapsed = torch.zeros(batch, time_steps, 16)
    plateau = p0_train.analytic_delta_plateau(sigreg, batch)
    # The forward on an exactly collapsed batch is the empirical
    # delta-distribution Epps-Pulley value; it must equal the analytic form.
    empirical = float(sigreg(collapsed))
    assert math.isfinite(plateau) and plateau > 0
    assert abs(empirical - plateau) <= 1e-5 * plateau
    # Every sketch direction sits at the same plateau for a delta at zero.
    directions = p0_train.per_direction_ep_statistics(sigreg, collapsed)
    assert directions.shape == (64,)
    assert torch.allclose(directions,
                          torch.full_like(directions, plateau), rtol=1e-5)
    # The documented V16 anchor: 25.731 at batch size 64.
    assert abs(p0_train.analytic_delta_plateau(sigreg, 64) - 25.731) < 5e-3
    # A healthy (Gaussian) batch sits far below the plateau.
    gaussian = torch.randn(256, 1, 16)
    assert float(sigreg(gaussian)) < 0.5 * p0_train.analytic_delta_plateau(
        sigreg, 256)


# --------------------------------------------------------------------------
# V18 effective-rank definition
# --------------------------------------------------------------------------

def test_effective_rank_matches_v18_definition():
    rng = np.random.default_rng(1)
    samples, active, width = 4000, 5, 12
    data = np.zeros((samples, width))
    data[:, :active] = rng.standard_normal((samples, active))
    variance, eigenvalues = p0_train.covariance_spectrum(torch.from_numpy(data))
    rank = p0_train.effective_rank(eigenvalues)
    # k isotropic dimensions -> effective rank ~ k.
    assert abs(rank - active) < 0.15
    # Independent numpy replication of the V18 formula
    # (scripts/train_hacssm_v10.encoder_diagnostics).
    centered = data - data.mean(axis=0)
    covariance = centered.T @ centered / (samples - 1)
    spectrum = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    probabilities = spectrum / max(spectrum.sum(), 1e-30)
    expected = float(np.exp(
        -(probabilities * np.log(np.clip(probabilities, 1e-30, None))).sum()))
    assert abs(rank - expected) < 1e-8
    assert abs(float(variance.mean())
               - float(centered.var(axis=0).mean())) < 1e-10


# --------------------------------------------------------------------------
# Sliding H=3 window sampler (reused V18 code path)
# --------------------------------------------------------------------------

def test_window_sampler_alignment():
    batch, length, dim, action_dim, history = 2, 8, 4, 2, 3
    z = (torch.arange(length, dtype=torch.float32).view(1, length, 1)
         + 100.0 * torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
         ).expand(batch, length, dim).contiguous()
    actions = (torch.arange(length - 1, dtype=torch.float32).view(1, length - 1, 1)
               ).expand(batch, length - 1, action_dim).contiguous()
    latent_windows, action_windows, targets = v18.sliding_predictor_windows(
        z, actions, z, history=history)
    windows = length - history
    assert latent_windows.shape == (batch * windows, history, dim)
    assert action_windows.shape == (batch * windows, history, action_dim)
    assert targets.shape == (batch * windows, dim)
    for episode in range(batch):
        for window in range(windows):
            index = episode * windows + window
            expected_latents = torch.tensor(
                [window + offset + 100.0 * episode for offset in range(history)])
            assert torch.equal(latent_windows[index, :, 0], expected_latents)
            # Window actions are a_i..a_{i+2}: a_t maps z_t to z_{t+1}.
            assert torch.equal(
                action_windows[index, :, 0],
                torch.tensor([float(window + offset) for offset in range(history)]))
            # Target is z_{i+history}: teacher-forcing next latent.
            assert float(targets[index, 0]) == window + history + 100.0 * episode


def test_dataset_shapes_and_range():
    bank = _synthetic_bank(episodes=4)
    observed = p0_data.corrupt_bank(bank)
    dataset = p0_train.P0EpisodeDataset(observed, bank)
    item = dataset[1]
    assert item["observed"].shape == (LENGTH, 3, 64, 64)
    assert item["clean"].shape == (LENGTH, 3, 64, 64)
    assert item["actions"].shape == (LENGTH - 1, 2)
    assert float(item["observed"].min()) >= 0.0
    assert float(item["observed"].max()) <= 1.0
    single = p0_train.P0EpisodeDataset(observed, None)
    assert "clean" not in single[0]


# --------------------------------------------------------------------------
# Gate logic
# --------------------------------------------------------------------------

def _rows(epochs=100, loss=0.1, rank=32.0, variance=1e-2, ep_ratio=float("nan"),
          grad_ratio=1.0):
    return [{
        "epoch": epoch + 1,
        "val_predictive_loss": loss,
        "encoder_covariance_effective_rank": rank,
        "encoder_mean_channel_variance": variance,
        "ep_ratio": ep_ratio,
        "grad_ratio": grad_ratio,
    } for epoch in range(epochs)]


def test_gates_pass_and_convergence_window():
    rows = _rows()
    gates = p0_train.compute_gates(rows, "vicreg")
    assert gates["rank_pass"] and gates["variance_pass"]
    assert gates["convergence_pass"] and gates["overall_pass"]
    assert gates["convergence_window_epochs"] == 10
    assert not gates["plateau_flag"] and not gates["grad_ratio_flag"]
    # Convergence uses mean(ep81-90) vs mean(ep91-100) of val predictive loss.
    drifting = _rows()
    for row in drifting[90:]:
        row["val_predictive_loss"] = 0.05   # 50% late change -> fail
    gates = p0_train.compute_gates(drifting, "vicreg")
    assert abs(gates["convergence_relative_change"] - 0.5) < 1e-9
    assert not gates["convergence_pass"] and not gates["overall_pass"]


def test_gates_rank_and_variance_thresholds():
    rows = _rows()
    rows[-1]["encoder_covariance_effective_rank"] = 15.9
    assert not p0_train.compute_gates(rows, "vicreg")["overall_pass"]
    rows = _rows()
    rows[-1]["encoder_mean_channel_variance"] = 5e-5
    assert not p0_train.compute_gates(rows, "vicreg")["overall_pass"]


def test_gates_plateau_flag_sigreg_only():
    rows = _rows(ep_ratio=1.5)
    for row in rows[30:42]:                      # 12 consecutive within 2%
        row["ep_ratio"] = 1.01
    gates = p0_train.compute_gates(rows, "sigreg")
    assert gates["plateau_flag"] and gates["plateau_max_streak_epochs"] == 12
    assert not gates["overall_pass"]             # sigreg gates on the plateau
    # A 9-epoch streak stays under the 10-epoch flag threshold.
    rows = _rows(ep_ratio=1.5)
    for row in rows[30:39]:
        row["ep_ratio"] = 0.99
    assert not p0_train.compute_gates(rows, "sigreg")["plateau_flag"]
    # NaN ratios (vicreg arm) can never trip the plateau flag.
    gates = p0_train.compute_gates(_rows(), "vicreg")
    assert not gates["plateau_flag"]


def test_gates_grad_ratio_flag_reported_not_gated():
    rows = _rows(grad_ratio=200.0)               # sustained for all epochs
    gates = p0_train.compute_gates(rows, "sigreg")
    assert gates["grad_ratio_flag"]
    assert gates["overall_pass"]                 # reported, not a gate conjunct
    rows = _rows()
    for row in rows[10:19]:                      # 9 epochs only
        row["grad_ratio"] = 500.0
    assert not p0_train.compute_gates(rows, "sigreg")["grad_ratio_flag"]


def test_streak_helper():
    assert p0_train._longest_streak([]) == 0
    assert p0_train._longest_streak([True, True, False, True]) == 2
    assert p0_train._longest_streak([False]) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
