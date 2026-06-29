#!/usr/bin/env python3
"""Core invariants for HACSSM-v9 Learned Ordered Innovation Filter (LOIF)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewm.models.memory import LearnedOrderedInnovationFilterMemory as V9Memory
from lewm.models.memory_model import MemoryLeWorldModel
from scripts.train_popgym import learned_ordered_innovation_metadata


MODE_MAP = {
    "loifv9": "learned",
    "loifv9_fixedalpha": "fixedalpha",
    "loifv9_globalR": "globalR",
    "loifv9_innovationonly": "innovationonly",
    "loifv9_latentonly": "latentonly",
    "loifv9_uniformfusion": "uniformfusion",
    "loifv9_noaction": "noaction",
    "loifv9_singlebank": "singlebank",
}


def make_model(mode: str = "loifv9") -> MemoryLeWorldModel:
    return MemoryLeWorldModel(
        img_size=8,
        patch_size=4,
        embed_dim=8,
        action_dim=3,
        encoder_layers=1,
        encoder_heads=2,
        predictor_layers=1,
        predictor_heads=2,
        history_len=2,
        dropout=0.0,
        predictor_norm="none",
        sigreg_projections=8,
        encoder_type="precomputed",
        memory_impl=mode,
        memory_mode="both",
        # A legacy nonzero value must not silently create a V9 auxiliary.
        hier_loss_weight=0.2,
    )


def batch() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(91)
    observations = torch.randn(2, 32, 8)
    targets = torch.randn(2, 32, 8)
    actions = torch.randn(2, 31, 3)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[:, 10:16] = False
    return observations, targets, actions, mask


def _randomize(memory: V9Memory) -> None:
    """Move the zero/symmetric initialization so structural controls are observable."""
    generator = torch.Generator().manual_seed(92)
    with torch.no_grad():
        for parameter in memory.parameters():
            parameter.copy_(0.2 * torch.randn(
                parameter.shape, dtype=parameter.dtype, device=parameter.device,
                generator=generator))


def _run(memory: V9Memory) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(93)
    z = torch.randn(3, 17, memory.embed_dim)
    actions = torch.randn(3, 16, memory.action_dim)
    return memory(z, actions, return_details=True)


def _assert_disconnected(mode: str, disconnected: tuple[str, ...]) -> None:
    """Changing a retained-but-disconnected tensor cannot alter the control."""
    left = V9Memory(8, 3, mode=mode)
    _randomize(left)
    right = V9Memory(8, 3, mode=mode)
    right.load_state_dict(left.state_dict(), strict=True)
    with torch.no_grad():
        for name in disconnected:
            getattr(right, name).add_(torch.randn_like(getattr(right, name)) * 100.0)
    torch.manual_seed(94)
    z = torch.randn(2, 13, 8)
    actions = torch.randn(2, 12, 3)
    left_output, left_details = left(z, actions, return_details=True)
    right_output, right_details = right(z, actions, return_details=True)
    assert torch.equal(left_output, right_output)
    for key in left_details:
        assert torch.equal(left_details[key], right_details[key]), key


def test_parameter_count_schema_and_shared_action() -> None:
    memories = [V9Memory(128, 6, mode=mode) for mode in MODE_MAP.values()]
    expected = 2 * 128**2 + 2 * 6 * 128 + 2 * 128 + 3
    assert expected == 34_563
    assert V9Memory.expected_parameter_count(128, 6) == expected
    assert all(memory.parameter_count() == expected for memory in memories)
    assert all(tuple(memory.W_a.weight.shape) == (256, 6) for memory in memories)

    signatures = [
        [(name, tuple(value.shape)) for name, value in memory.named_parameters()]
        for memory in memories
    ]
    assert all(signature == signatures[0] for signature in signatures[1:])
    assert {name for name, _ in memories[0].named_parameters()} == {
        "u_fast", "u_delta", "b_R", "W_x.weight", "W_a.weight", "w_z", "w_e",
        "W_o.weight",
    }
    assert not any(
        token in name.lower()
        for name, _ in memories[0].named_parameters()
        for token in ("route", "teacher", "tau", "timescale", "aux")
    )


def test_ordered_alphas_and_coupled_process_scales() -> None:
    memory = V9Memory(8, 3)
    fixed = V9Memory(8, 3, mode="fixedalpha")
    assert torch.allclose(
        memory.ordered_alphas(), fixed.ordered_alphas(), rtol=0.0, atol=1e-7
    )
    # Non-saturating float32 logits exercise both ordered regimes. At truly extreme logits,
    # numerical pole collapse is a logged scientific stop condition rather than a fake margin.
    for fast, delta in ((0.0, 0.0), (-4.0, 4.0), (4.0, -4.0)):
        with torch.no_grad():
            memory.u_fast.fill_(fast)
            memory.u_delta.fill_(delta)
        alphas = memory.ordered_alphas()
        qs = memory.process_scales(alphas)
        alpha_values = alphas.detach().tolist()
        assert 0.0 < alpha_values[0] < alpha_values[1] < 1.0
        assert torch.equal(qs, (1.0 - alphas) * (1.0 + alphas))
        assert torch.isfinite(qs).all() and bool((qs > 0.0).all())

    alphas = fixed.ordered_alphas()
    expected = torch.tensor(
        [math.exp(-0.5), math.exp(-0.125)], dtype=alphas.dtype,
        device=alphas.device)
    assert torch.allclose(alphas, expected, rtol=0.0, atol=1e-7)


def test_all_modes_construct_forward_and_expose_finite_filter_details() -> None:
    required = {
        "x", "priors", "states", "log_prior_scales", "log_scales",
        "resistance", "log_resistance", "gains", "prior_weights",
        "read_weights", "nominal_direct_coefficients", "alphas", "qs",
    }
    for mode in MODE_MAP.values():
        memory = V9Memory(8, 3, mode=mode)
        output, details = _run(memory)
        assert output.shape == (3, 17, 8)
        assert required <= set(details)
        assert details["x"].shape == (3, 17, 8)
        assert details["priors"].shape == (3, 17, 2, 8)
        assert details["states"].shape == (3, 17, 2, 8)
        for key in (
            "log_prior_scales", "log_scales", "gains", "prior_weights",
            "read_weights", "nominal_direct_coefficients",
        ):
            assert details[key].shape == (3, 17, 2), (mode, key, details[key].shape)
        assert details["alphas"].shape == (2,)
        assert details["qs"].shape == (2,)
        assert all(torch.isfinite(value).all() for value in details.values())
        assert bool((details["gains"] >= 0.0).all())
        assert bool((details["gains"] <= 1.0).all())
        assert torch.allclose(
            details["prior_weights"].sum(dim=-1),
            torch.ones_like(details["prior_weights"][..., 0]))
        assert torch.allclose(
            details["read_weights"].sum(dim=-1),
            torch.ones_like(details["read_weights"][..., 0]))

        # The filter contract warm-starts both banks at x_0 with unit scale.
        expected_initial = details["x"][:, :1].expand(-1, 2, -1)
        assert torch.allclose(details["states"][:, 0], expected_initial)
        assert torch.equal(
            details["log_scales"][:, 0],
            torch.zeros_like(details["log_scales"][:, 0]))


def test_structural_interventions_are_exact() -> None:
    # Fixed-alpha and the evidence-source controls retain tensors for parameter matching but
    # disconnect precisely the tensors named in the frozen design table.
    _assert_disconnected("fixedalpha", ("u_fast", "u_delta"))
    _assert_disconnected("globalR", ("w_z", "w_e"))
    _assert_disconnected("innovationonly", ("w_z",))
    _assert_disconnected("latentonly", ("w_e",))

    global_r = V9Memory(8, 3, mode="globalR")
    _, details = _run(global_r)
    log_r = details["log_resistance"]
    assert torch.equal(log_r, log_r[:, :1].expand_as(log_r))

    uniform = V9Memory(8, 3, mode="uniformfusion")
    _, details = _run(uniform)
    assert torch.equal(details["prior_weights"], torch.full_like(
        details["prior_weights"], 0.5))
    assert torch.equal(details["read_weights"], torch.full_like(
        details["read_weights"], 0.5))

    noaction = V9Memory(8, 3, mode="noaction")
    _randomize(noaction)
    torch.manual_seed(95)
    z = torch.randn(2, 15, 8)
    actions = torch.randn(2, 14, 3)
    assert torch.equal(noaction(z, actions), noaction(z, torch.zeros_like(actions)))

    single = V9Memory(8, 3, mode="singlebank")
    _, details = _run(single)
    expected = torch.zeros_like(details["prior_weights"])
    expected[..., 1] = 1.0
    assert torch.equal(details["prior_weights"], expected)
    assert torch.equal(details["read_weights"], expected)


def test_log_scale_path_is_stable_and_differentiable() -> None:
    torch.manual_seed(96)
    for offset in (-80.0, 80.0):
        memory = V9Memory(8, 3)
        _randomize(memory)
        with torch.no_grad():
            memory.b_R.fill_(offset)
        z = (1_000.0 * torch.randn(2, 257, 8)).requires_grad_(True)
        actions = 1_000.0 * torch.randn(2, 256, 3)
        read, details = memory(z, actions, return_details=True)
        fused = memory.fuse(z, read)
        objective = fused.square().mean() + details["log_scales"].square().mean()
        objective.backward()

        assert torch.isfinite(fused).all()
        assert all(torch.isfinite(value).all() for value in details.values())
        assert z.grad is not None and torch.isfinite(z.grad).all()
        gradients = [parameter.grad for parameter in memory.parameters()
                     if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_model_has_no_teacher_auxiliary_or_hidden_clean_target_path() -> None:
    model = make_model()
    keys = model.state_dict()
    assert any(key.startswith("mem_loifv9.") for key in keys)
    assert not any("teacher" in key or "hier_" in key for key in keys)
    assert not hasattr(model, "mem_loifv9_teacher")

    observations, targets, actions, mask = batch()
    changed = targets.clone()
    changed[:, 10:16] += 100_000.0
    model.eval()
    with torch.no_grad():
        left = model.compute_loss(
            observations, actions, targets, mask,
            memory_update_mask=mask, first_post_loss_weight=0.0)
        right = model.compute_loss(
            observations, actions, changed, mask,
            memory_update_mask=mask, first_post_loss_weight=0.0)
    assert set(left) == {
        "loss", "pred_loss", "sigreg_loss", "pred_loss_all_valid",
        "pred_loss_first_post",
    }
    expected = left["pred_loss"] + model.sigreg_lambda * left["sigreg_loss"]
    assert torch.equal(left["loss"], expected)
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)
    model.update_hierarchical_teacher()  # deliberate no-op for V9


def test_all_external_modes_construct_and_cli_exposes_them() -> None:
    observations, _, actions, mask = batch()
    for mode in MODE_MAP:
        model = make_model(mode)
        with torch.no_grad():
            injected = model._inject(
                observations, actions=actions, memory_update_mask=mask)
        assert injected.shape == observations.shape
        assert torch.isfinite(injected).all()
        assert not any("teacher" in key or "hier_" in key for key in model.state_dict())
        metadata = learned_ordered_innovation_metadata(mode)
        assert metadata["memory_arch_schema_version"] == 9
        assert metadata["memory_internal_auxiliary"] == "none"
        assert metadata["memory_teacher_present"] is False
        assert metadata["memory_objective"] == "unweighted_visible_next_latent_mse"
        assert metadata["memory_hidden_clean_targets_used"] is False

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_popgym.py"), "--help"],
        cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    help_text = result.stdout + result.stderr
    assert all(mode in help_text for mode in MODE_MAP)


if __name__ == "__main__":
    tests = (
        test_parameter_count_schema_and_shared_action,
        test_ordered_alphas_and_coupled_process_scales,
        test_all_modes_construct_forward_and_expose_finite_filter_details,
        test_structural_interventions_are_exact,
        test_log_scale_path_is_stable_and_differentiable,
        test_model_has_no_teacher_auxiliary_or_hidden_clean_target_path,
        test_all_external_modes_construct_and_cli_exposes_them,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v9 tests passed.")
