#!/usr/bin/env python3
"""Focused contract, construction, and causality tests for LeWM+V8 V18."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.hacssm_v11_data as v11_data


V11_TASKS_BEFORE_V18_IMPORT = v11_data.TASKS

from scripts import hacssm_v18_data as data
import scripts.train_hacssm_v11 as v11
import scripts.train_lewm_v8_v18 as trainer
import scripts.train_subjepa_v16 as base


def _args(design: str) -> argparse.Namespace:
    return argparse.Namespace(
        design=design,
        img_size=16,
        patch_size=8,
        embed_dim=32,
        encoder_layers=1,
        encoder_heads=4,
        predictor_layers=1,
        predictor_heads=4,
        history_len=3,
        dropout=0.0,
        sigreg_lambda=0.0,
        sigreg_projections=5,
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(18020)
    clean = torch.rand(4, 4, 3, 16, 16)
    observed = clean.clone()
    observed[:, 2] = 0.0
    actions = torch.randn(4, 3, 2)
    return observed, clean, actions


def _assert_raises(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(
        f"{function.__name__} did not raise {error_type.__name__}")


def test_frozen_data_and_grid_contract() -> None:
    assert data.TASKS == (
        "acrobot.swingup",
        "manipulator.bring_ball",
        "quadruped.run",
        "stacker.stack_4",
        "swimmer.swimmer15",
    )
    assert data.DEFAULT_TRAIN_SEED == 270701
    assert data.DEFAULT_VAL_SEED == 270702
    assert data.DEFAULT_CORRUPTION_SEED == 270711
    assert trainer.MEMORIES == (
        "none", "gru", "ssm", "hacssmv8", "hacssmv8_static",
        "hacssmv8_dynamic", "hacssmv8_noaction", "hacssmv8_single",
    )
    assert trainer.DESIGNS == tuple(
        f"vicreg_{memory}" for memory in trainer.MEMORIES)
    assert trainer.SEEDS == (18001, 18002, 18003, 18004, 18005)
    assert trainer.DEFAULT_EPOCHS == 100
    assert len(data.TASKS) * len(trainer.DESIGNS) * len(trainer.SEEDS) == 200


def test_v18_import_does_not_mutate_v11_task_registry() -> None:
    assert v11_data.TASKS == V11_TASKS_BEFORE_V18_IMPORT
    with data._v18_task_registry():
        assert v11_data.TASKS == data.TASKS
    assert v11_data.TASKS == V11_TASKS_BEFORE_V18_IMPORT


def test_design_parser_and_metadata_cover_direct_controls() -> None:
    for memory in trainer.MEMORIES:
        design = f"vicreg_{memory}"
        assert trainer.parse_design(design) == ("vicreg", memory, None)
        metadata = trainer.design_metadata(design, embed_dim=32)
        assert metadata["memory_architecture"] == memory
        assert metadata["memory_specific_loss_weight"] == 0.0
        assert metadata["new_memory_architecture"] is False
        assert metadata["one_token_predictor"] is False
        assert metadata["predictor_history"] == 3
        assert metadata["hidden_clean_targets_included"] is True
        assert metadata["wandb_method_tag"] == "lewm-v8-v18"
        assert metadata["wandb_scope_tag"] == "unopened-task-confirmation"
    _assert_raises(ValueError, trainer.parse_design, "vicreg_unknown")

    assert trainer.design_metadata("vicreg_hacssmv8_static")[
        "v8_correction_mode"] == "static"
    assert trainer.design_metadata("vicreg_hacssmv8_dynamic")[
        "v8_correction_mode"] == "dynamic"
    assert trainer.design_metadata("vicreg_hacssmv8")[
        "v8_correction_mode"] == "learned_shrinkage"
    assert trainer.design_metadata("vicreg_hacssmv8_noaction")[
        "v8_action_transport_enabled"] is False
    assert trainer.design_metadata("vicreg_hacssmv8_single")[
        "v8_joint_read_enabled"] is False
    assert trainer.design_metadata("vicreg_gru")[
        "gru_probe_prior_contract"] == (
            "D_dimensional_read_of_h_t_minus_1_before_z_t")


def test_builds_all_eight_existing_memory_modes() -> None:
    expected_v8_modes = {
        "hacssmv8": "learned",
        "hacssmv8_static": "rho0",
        "hacssmv8_dynamic": "rho1",
        "hacssmv8_noaction": "noaction",
        "hacssmv8_single": "single",
    }
    for design in trainer.DESIGNS:
        torch.manual_seed(18021)
        model = trainer.build_model(_args(design), action_dim=2)
        _, memory, _ = trainer.parse_design(design)
        expected_impl = "ema" if memory == "none" else memory
        assert model.world.memory_impl == expected_impl
        assert model.world.encoder_norm == "causal"
        assert model.world.predictor_norm == "none"
        if memory == "none":
            assert not any(
                parameter.requires_grad
                for parameter in model.world.memory.parameters())
            assert not any(
                parameter.requires_grad
                for parameter in model.world.fusion.parameters())
        elif memory in expected_v8_modes:
            assert model.world.mem_hacssmv8.v8_mode == expected_v8_modes[memory]
        elif memory == "gru":
            assert model.world.mem_gru.readout.out_features == _args(design).embed_dim


def test_gru_probe_prior_is_d_dimensional_and_strictly_pre_observation() -> None:
    torch.manual_seed(18022)
    model = trainer.build_model(_args("vicreg_gru"), action_dim=2)
    # The deployed readout is intentionally zero-initialized.  Give it a
    # deterministic nonzero value so the causality assertion observes changes
    # in the hidden trajectory rather than the initialization invariant.
    with torch.no_grad():
        model.world.mem_gru.readout.weight.copy_(
            torch.randn_like(model.world.mem_gru.readout.weight))
    z = torch.randn(3, 6, 32)
    actions = torch.randn(3, 5, 2)
    result = trainer.memory_representations(model, z, actions)
    raw_prior = result["details"]["prior_read"]
    raw_posterior = result["details"]["posterior_read"]

    assert tuple(raw_prior.shape) == tuple(z.shape)
    assert tuple(result["prior"].shape) == tuple(z.shape)
    assert torch.equal(raw_prior[:, 0], torch.zeros_like(raw_prior[:, 0]))
    assert torch.equal(raw_prior[:, 1:], raw_posterior[:, :-1])

    changed = z.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 17.0
    changed_result = trainer.memory_representations(model, changed, actions)
    # Prior at t=3 has consumed only z_0,z_1,z_2.  Changes to z_3 and later
    # therefore cannot affect any prior through t=3.
    assert torch.equal(result["prior"][:, :4], changed_result["prior"][:, :4])
    assert not torch.equal(
        result["details"]["posterior_read"][:, 3:],
        changed_result["details"]["posterior_read"][:, 3:])


def test_sliding_h3_windows_include_earlier_tokens_and_align_targets() -> None:
    batch, length, dimension, action_dim = 2, 6, 4, 2
    fused = torch.arange(
        batch * length * dimension, dtype=torch.float32).reshape(
            batch, length, dimension)
    actions = torch.arange(
        batch * (length - 1) * action_dim, dtype=torch.float32).reshape(
            batch, length - 1, action_dim)
    targets = 1000.0 + fused
    latent_windows, action_windows, next_targets = (
        trainer.sliding_predictor_windows(fused, actions, targets, history=3))

    assert latent_windows.shape == (batch * 3, 3, dimension)
    assert action_windows.shape == (batch * 3, 3, action_dim)
    assert next_targets.shape == (batch * 3, dimension)
    assert torch.equal(latent_windows[0], fused[0, 0:3])
    assert torch.equal(latent_windows[1], fused[0, 1:4])
    assert torch.equal(action_windows[0], actions[0, 0:3])
    assert torch.equal(
        next_targets.reshape(batch, 3, dimension), targets[:, 3:])

    class _AllTokenPredictor(torch.nn.Module):
        def forward(self, latent, action):
            del action
            return latent.cumsum(dim=1)

    world = SimpleNamespace(history_len=3, predictor=_AllTokenPredictor())
    original = trainer.sliding_next_predictions(world, fused, actions)
    changed = fused.clone()
    changed[:, 0] += 100.0
    modified = trainer.sliding_next_predictions(world, changed, actions)
    # The first prediction sees tokens 0,1,2, including the changed earlier
    # token even though its last context token is unchanged.  Later windows do
    # not contain token 0 and therefore remain identical.
    assert not torch.equal(original[:, 0], modified[:, 0])
    assert torch.equal(original[:, 1:], modified[:, 1:])


def test_none_prior_uses_h3_predictions_after_zero_warmup() -> None:
    torch.manual_seed(18024)
    model = trainer.build_model(_args("vicreg_none"), action_dim=2)
    z = torch.randn(2, 7, 32)
    actions = torch.randn(2, 6, 2)
    expected = trainer.sliding_next_predictions(model.world, z, actions)
    result = trainer.memory_representations(model, z, actions)
    assert torch.equal(result["prior"][:, :3], torch.zeros_like(z[:, :3]))
    assert torch.equal(result["prior"][:, 3:], expected)


def test_inherited_probe_and_evaluation_hook_uses_h3_predictions() -> None:
    batch, steps, dimension, action_dim = 2, 5, 4, 2
    beliefs = torch.arange(
        batch * steps * dimension, dtype=torch.float32).reshape(
            batch, steps, dimension)
    actions = torch.zeros(batch, steps, action_dim)

    class _AllTokenPredictor(torch.nn.Module):
        def forward(self, latent, action):
            del action
            return latent.cumsum(dim=1)

    world = SimpleNamespace(history_len=3, predictor=_AllTokenPredictor())
    trainer._install_contract()
    try:
        # Both inherited functions resolve the module-level hook at call time;
        # ``torch.no_grad`` wraps their bodies, so inspect the shared module
        # binding rather than the decorator wrapper's globals.
        assert v11.one_token_prediction is trainer.finite_context_prediction
        assert v11.collect_representations.__wrapped__.__globals__ is vars(v11)
        assert v11.evaluate_condition.__wrapped__.__globals__ is vars(v11)
        prediction = v11.one_token_prediction(world, beliefs, actions)
        assert torch.equal(prediction[:, :2], torch.zeros_like(prediction[:, :2]))
        assert torch.equal(prediction[:, 2], beliefs[:, :3].sum(dim=1))
        changed = beliefs.clone()
        changed[:, 0] += 100.0
        changed_prediction = v11.one_token_prediction(world, changed, actions)
        assert not torch.equal(prediction[:, 2], changed_prediction[:, 2])
        assert torch.equal(prediction[:, 3:], changed_prediction[:, 3:])
    finally:
        trainer._restore_contract()


def test_representative_vicreg_objectives_are_finite_and_differentiable() -> None:
    trainer._install_contract()
    try:
        observed, clean, actions = _batch()
        for design in (
                "vicreg_none", "vicreg_gru", "vicreg_ssm",
                "vicreg_hacssmv8", "vicreg_hacssmv8_static",
                "vicreg_hacssmv8_noaction", "vicreg_hacssmv8_single"):
            torch.manual_seed(18023)
            model = trainer.build_model(_args(design), action_dim=2)
            losses = base.compute_losses(
                model, observed, clean, actions, sigreg_lambda=0.0)
            assert set(losses) == {
                "loss", "predictive_loss", "regularizer_loss", "sigreg_loss",
                "variance_loss", "covariance_loss",
            }
            assert all(bool(value.isfinite()) for value in losses.values())
            losses["loss"].backward()
            assert any(
                parameter.grad is not None
                for parameter in model.world.encoder.parameters())
    finally:
        trainer._restore_contract()


def test_standalone_main_defaults_to_100_epochs() -> None:
    captured: list[list[str]] = []
    original = base.main
    try:
        base.main = lambda argv: captured.append(list(argv))
        trainer.main(["--train-data", "train.npz", "--val-data", "val.npz",
                      "--design", "vicreg_hacssmv8", "--seed", "18001"])
        assert captured[0][captured[0].index("--epochs") + 1] == "100"

        captured.clear()
        trainer.main(["--train-data", "train.npz", "--val-data", "val.npz",
                      "--design", "vicreg_hacssmv8", "--seed", "18001",
                      "--epochs", "7"])
        assert captured[0].count("--epochs") == 1
        assert captured[0][captured[0].index("--epochs") + 1] == "7"
    finally:
        base.main = original


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} LeWM+V8 V18 focused tests")


if __name__ == "__main__":
    main()
