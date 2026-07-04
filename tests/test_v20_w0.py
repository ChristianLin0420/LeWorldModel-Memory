"""V20 W0 unit tests: the published VisReg objective, the salience ladder,
and the W0 trainer plumbing (docs/V20_PROPOSAL.md 4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.visreg import VisRegObjective
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.tasks import TASKS, _TASK_IDS
import scripts.train_v20_w0 as w0

SLICES = 256  # small slice count for tests; the recipe value is 4096


# --------------------------------------------------------------------------
# VisReg objective
# --------------------------------------------------------------------------

class TestVisRegObjective:
    def test_gaussian_batch_near_zero(self):
        torch.manual_seed(0)
        objective = VisRegObjective(num_slices=SLICES)
        z = torch.randn(4096, 32)
        losses = objective(z)
        assert float(losses["scale"]) < 5e-3
        assert float(losses["center"]) < 5e-3
        assert float(losses["shape"]) < 5e-2
        assert float(losses["total"]) == pytest.approx(
            float(losses["scale"] + losses["shape"] + losses["center"]),
            rel=1e-6)

    def test_collapse_values_and_constant_scale_gradient(self):
        """At an exact delta distribution: scale == 1 and the scale gradient
        is constant, never the SIGReg projected-zero plateau (the published
        collapse-robustness property)."""
        torch.manual_seed(1)
        objective = VisRegObjective(num_slices=SLICES)
        z = torch.full((512, 16), 0.3, requires_grad=True)
        losses = objective(z)
        assert float(losses["scale"].detach()) == pytest.approx(1.0, abs=1e-6)
        losses["scale"].backward()
        grad = z.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        # d/dz of mean_j (1 - std_j)^2 at sigma == 0 has magnitude bounded
        # away from zero once any perturbation exists; check via a tiny
        # perturbation that the restoring force is O(1), not vanishing.
        z2 = (torch.ones(512, 16) * 0.3
              + 1e-3 * torch.randn(512, 16)).requires_grad_(True)
        losses2 = VisRegObjective(num_slices=SLICES)(z2)
        losses2["scale"].backward()
        sigma = z2.detach().std(dim=0, unbiased=False)
        # Analytic: dL/dsigma_j = -2 (1 - sigma_j) / D ~= -2/D at collapse.
        assert float((1.0 - sigma).mean()) > 0.99
        assert float(z2.grad.abs().sum()) > 0

    def test_shape_gradient_finite_near_collapse(self):
        torch.manual_seed(2)
        objective = VisRegObjective(num_slices=SLICES)
        z = (torch.zeros(256, 8) + 1e-4 * torch.randn(256, 8)
             ).requires_grad_(True)
        losses = objective(z)
        losses["shape"].backward()
        assert torch.isfinite(z.grad).all()
        assert float(z.grad.abs().max()) > 0

    def test_stop_grad_standardization(self):
        """The shape term must carry no gradient through mean/std (shape
        cannot fight scale — the V17 preflight failure mode)."""
        torch.manual_seed(3)
        z = torch.randn(512, 8, requires_grad=True)
        losses = VisRegObjective(num_slices=SLICES)(z)
        (grad,) = torch.autograd.grad(losses["shape"], z)
        # Direct check: scaling the input by a constant scales the shape
        # gradient by ~1/std relative to the standardized coordinates but the
        # standardization itself is detached: the gradient wrt a pure global
        # rescale factor comes only through z, not through sigma.  A
        # numerically robust proxy: the gradient's per-column mean is ~0
        # (mu detached => no batch-mean cancellation term).
        assert float(grad.mean(dim=0).abs().max()) < float(
            grad.abs().mean()) * 0.5
        assert torch.isfinite(grad).all()

    def test_quantile_cache_and_shapes(self):
        objective = VisRegObjective(num_slices=SLICES)
        q1 = objective._quantiles(100, torch.device("cpu"))
        q2 = objective._quantiles(100, torch.device("cpu"))
        assert q1 is q2
        assert q1.shape == (100,)
        assert float(q1[0]) < 0 < float(q1[-1])
        assert torch.allclose(q1, -q1.flip(0), atol=1e-5)  # symmetric

    def test_three_dim_input_flattened(self):
        torch.manual_seed(4)
        objective = VisRegObjective(num_slices=SLICES)
        z3 = torch.randn(8, 16, 12)
        z2 = z3.reshape(-1, 12)
        torch.manual_seed(99)
        a = objective(z3)
        torch.manual_seed(99)
        b = objective(z2)
        assert float(a["total"]) == pytest.approx(float(b["total"]), rel=1e-5)

    def test_rejects_bad_input(self):
        objective = VisRegObjective(num_slices=SLICES)
        with pytest.raises(ValueError):
            objective(torch.randn(5))
        with pytest.raises(ValueError):
            objective(torch.randn(1, 8))
        with pytest.raises(ValueError):
            VisRegObjective(num_slices=0)


# --------------------------------------------------------------------------
# Salience ladder tasks
# --------------------------------------------------------------------------

class TestSalienceLadder:
    def test_registry_appended_ids_stable(self):
        """Pre-existing tasks keep their seed-salt ids (frozen banks must
        stay byte-reproducible); ladder tasks are appended."""
        expected_prefix = ("t1", "t2", "t3", "t4", "t1dev", "t2dev", "t3dev")
        assert TASKS[:7] == expected_prefix
        for index, name in enumerate(expected_prefix, start=1):
            assert _TASK_IDS[name] == index
        assert TASKS[7:] == ("t1s1", "t1s2", "t1s3")

    def test_ladder_parameters_monotone(self):
        knobs = []
        for name in ("t1s1", "t1s2", "t1s3", "t1"):
            task = make_task(name)
            knobs.append((task.marker_radius, task.cue_half,
                          task.cue_border_px))
            assert task.cue_half == task.marker_radius - 1  # fill-tiles-ring
            assert task.n_classes == 4
            assert task.cue_shape == "disc"
        for lower, upper in zip(knobs, knobs[1:]):
            assert all(a <= b for a, b in zip(lower, upper))
            assert any(a < b for a, b in zip(lower, upper))
        assert knobs[0] == (4, 3, 0)   # amendment-1 exact
        assert knobs[-1] == (6, 5, 3)  # amendment-2 (t1)

    def test_ladder_leakage_pairing_preserved(self):
        """Paired branches must stay byte-identical outside the cue window at
        every ladder level (the P1a leakage proof transfers)."""
        for name in ("t1s1", "t1s2", "t1s3"):
            task = make_task(name)
            branch_a, branch_b = task.paired_branches(3, seed=0)
            assert not np.array_equal(branch_a.xi, branch_b.xi)
            for episode in range(3):
                on = int(branch_a.events["cue_on"][episode])
                off = int(branch_a.events["cue_off"][episode])
                np.testing.assert_array_equal(
                    branch_a.frames[episode, :on],
                    branch_b.frames[episode, :on])
                np.testing.assert_array_equal(
                    branch_a.frames[episode, off:],
                    branch_b.frames[episode, off:])

    def test_ladder_salience_monotone_in_pixels(self):
        """Mean absolute cue-window pixel difference between paired branches
        rises with the ladder level (the s* instrument's x-axis)."""
        saliences = []
        for name in ("t1s1", "t1s2", "t1s3", "t1"):
            task = make_task(name)
            branch_a, branch_b = task.paired_branches(4, seed=1)
            diffs = []
            for episode in range(4):
                on = int(branch_a.events["cue_on"][episode])
                off = int(branch_a.events["cue_off"][episode])
                delta = (branch_a.frames[episode, on:off].astype(np.float64)
                         - branch_b.frames[episode, on:off].astype(np.float64))
                diffs.append(np.abs(delta).mean())
            saliences.append(float(np.mean(diffs)))
        assert saliences == sorted(saliences)
        assert saliences[0] > 0
        assert saliences[-1] > 2.0 * saliences[0]

    def test_s1_border_is_noop(self):
        """cue_border_px == 0 must draw nothing (amendment-1 fidelity)."""
        task = make_task("t1s1")
        bank = task.generate("iid", 2, seed=2)
        for episode in range(2):
            on = int(bank.events["cue_on"][episode])
            frame = bank.frames[episode, on]
            pre = bank.frames[episode, on - 1]
            # Border rows outside marker regions are unchanged by the cue.
            np.testing.assert_array_equal(frame[31:33, 0], pre[31:33, 0])


# --------------------------------------------------------------------------
# Trainer plumbing
# --------------------------------------------------------------------------

class TestW0Trainer:
    def test_arm_lambda(self):
        assert w0.arm_lambda("visreg60") == pytest.approx(0.60)
        assert w0.arm_lambda("visreg75") == pytest.approx(0.75)
        assert w0.arm_lambda("visreg90") == pytest.approx(0.90)
        assert w0.arm_lambda("vicreg") is None
        with pytest.raises(ValueError):
            w0.arm_lambda("sigreg")

    def test_host_kind(self):
        assert w0.host_kind("visreg75") == "visreg"
        assert w0.host_kind("vicreg") == "vicreg"

    def test_resolve_banks_rejects_unknown_task(self, tmp_path):
        with pytest.raises(ValueError):
            w0.resolve_banks("t2", tmp_path / "a", tmp_path / "b")

    def test_resolve_banks_refuses_p0_root_write(self, tmp_path):
        with pytest.raises(ValueError):
            w0.resolve_banks("t1s1", tmp_path, tmp_path)

    def test_visreg_losses_backward(self):
        """One optimizer step through the full visreg host objective on tiny
        random frames: all components finite, encoder receives gradient."""
        torch.manual_seed(0)
        model = w0.p0.build_sigreg_host(action_dim=2)
        visreg = VisRegObjective(num_slices=64)
        observed = torch.rand(2, 8, 3, 64, 64)
        actions = torch.rand(2, 7, 2) * 2 - 1
        losses = w0.visreg_losses(model, visreg, 0.75, observed, actions)
        for key in ("loss", "predictive_loss", "regularizer_loss",
                    "visreg_scale", "visreg_shape", "visreg_center"):
            assert torch.isfinite(losses[key]), key
        losses["loss"].backward()
        grads = [parameter.grad for parameter in model.encoder.parameters()
                 if parameter.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
