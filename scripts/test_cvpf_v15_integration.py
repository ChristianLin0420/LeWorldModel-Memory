#!/usr/bin/env python3
"""Host, differentiable-envelope, and trainer integration tests for CVPF-v15."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cvpf import CVPFFit, fit_cvpf
from lewm.models.memory_model import MemoryLeWorldModel
import scripts.train_cvpf_v15 as train


def _synthetic(episodes: int = 80, length: int = 7):
    generator = torch.Generator().manual_seed(15_777)
    transition = torch.tensor([[.74, -.14], [.14, .74]], dtype=torch.float64)
    action_map = torch.tensor([[.22, -.07], [.04, .18]], dtype=torch.float64)
    read = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [.55, -.25], [.15, .72]], dtype=torch.float64)
    actions = torch.randn(
        episodes, length - 1, 2, generator=generator, dtype=torch.float64)
    state = torch.zeros(episodes, length, 2, dtype=torch.float64)
    state[:, 0] = torch.randn(episodes, 2, generator=generator, dtype=torch.float64)
    for step in range(length - 1):
        state[:, step + 1] = (
            state[:, step] @ transition.T + actions[:, step] @ action_map.T)
    clean = state @ read.T
    noise = .025 * torch.randn(clean.shape, generator=generator, dtype=torch.float64)
    noise[::3, 2:5] += .25 * torch.randn(
        noise[::3, 2:5].shape, generator=generator, dtype=torch.float64)
    observed = clean + noise
    observed[:, 0] = clean[:, 0]
    return clean, observed, actions


def _world(mode: str, horizon: int = 6) -> MemoryLeWorldModel:
    return MemoryLeWorldModel(
        img_size=8, patch_size=4, embed_dim=4, action_dim=2,
        encoder_layers=1, encoder_heads=2,
        predictor_layers=1, predictor_heads=2,
        predictor_norm="none", encoder_norm="causal", history_len=2,
        dropout=0.0, sigreg_projections=4, memory_impl=mode,
        cvpf_horizon=horizon)


def _args(design: str) -> argparse.Namespace:
    return argparse.Namespace(
        train_data="train.npz", val_data="val.npz", memory_mode=design,
        seed=15, output_dir="out", epochs=3, batch_size=4, lr=3e-4,
        weight_decay=1e-5, num_workers=0, img_size=8, patch_size=4,
        embed_dim=4, encoder_layers=1, encoder_heads=2,
        predictor_layers=1, predictor_heads=2, history_len=2, dropout=0.0,
        sigreg_lambda=.1, sigreg_projections=4, probe_ridge=1e-3,
        eval_target_key="task_observation", corruption_seed=1,
        eval_rollout_episode=0, no_amp=True, device="cpu", wandb=False,
        wandb_entity=None, wandb_project="test", wandb_mode="online",
        wandb_study="test", extra_tag="", cvpf_horizon=6)


def test_registry_installs_every_mode_with_zero_memory_parameters() -> None:
    for design, core_mode in train.CORE_MODES.items():
        world = _world(design)
        assert world.mem_cvpfv15.mode == core_mode
        assert world.mem_cvpfv15.parameter_count() == 0
        assert world.mem_cvpfv15.horizon == 6
        assert world.horizons()["future_horizon"] == 6.0


def test_build_model_installs_registered_zero_parameter_memory() -> None:
    for design in train.CVPF_DESIGNS:
        model = train.build_model(_args(design), action_dim=2)
        assert model.world.memory_impl == design
        assert model.world.mem_cvpfv15.parameter_count() == 0


def test_fit_install_and_direct_clean_coordinate_read_for_every_core_mode() -> None:
    clean, observed, actions = _synthetic()
    for core_mode in sorted(set(train.CORE_MODES.values())):
        fit = fit_cvpf(clean, observed, actions, mode=core_mode)
        assert isinstance(fit, CVPFFit)
        design = next(key for key, value in train.CORE_MODES.items()
                      if value == core_mode)
        world = _world(design).double()
        world.mem_cvpfv15.install_fit(fit)
        direct, direct_details = world.mem_cvpfv15(
            observed[:3], actions[:3], return_details=True)
        injected, injected_details = world._inject(
            observed[:3], actions[:3], return_memory_details=True)
        assert torch.equal(injected, direct)
        assert torch.equal(injected_details["prior_reads"], direct_details["prior_reads"])
        assert torch.equal(injected_details["posterior_reads"], direct)
        assert torch.allclose(direct[:, 0], observed[:3, 0], atol=2e-12, rtol=0.0)
        assert direct.shape == observed[:3].shape
        if core_mode in ("noaction", "anchoronly"):
            assert torch.count_nonzero(direct_details["action_effects"]) == 0
        if core_mode in ("nocorrect", "anchoronly"):
            assert torch.count_nonzero(direct_details["corrections"]) == 0


def test_trainer_representation_contract_is_strict_and_direct() -> None:
    clean, observed, actions = _synthetic()
    fit = fit_cvpf(clean, observed, actions, mode="full")
    model = train.CVPFExperimentModel(_world("cvpfv15").double())
    model.world.mem_cvpfv15.install_fit(fit)
    representations = train.memory_representations(model, observed[:4], actions[:4])
    assert torch.equal(representations["fused"], representations["posterior"])
    assert representations["prior"].shape == observed[:4].shape


def test_complete_candidate_loss_backpropagates_through_host_and_envelope() -> None:
    clean_z, observed_z, actions = _synthetic()
    fit = fit_cvpf(clean_z, observed_z, actions, mode="full")
    model = train.CVPFExperimentModel(_world("cvpfv15"))
    model.world.mem_cvpfv15.install_fit(fit)
    generator = torch.Generator().manual_seed(15_778)
    clean = torch.rand(4, 7, 3, 8, 8, generator=generator)
    observed = clean.clone()
    observed[:, 2:5] = 0.0
    losses = train.compute_cvpf_losses(
        model, observed, clean, actions[:4].float())
    assert set(losses) == set(train.HISTORY_KEYS)
    assert all(torch.isfinite(value) for value in losses.values())
    assert float(losses["filtration_loss"].detach()) >= 0.0
    losses["loss"].backward()
    assert model.world.encoder.patch_embed.weight.grad is not None
    assert model.world.predictor.projector[0].weight.grad is not None


def test_symmetric_envelope_detachid_is_value_identical_but_changes_gradient() -> None:
    clean, observed, actions = _synthetic(episodes=8)
    clean_full = clean.float().requires_grad_()
    observed_full = observed.float().requires_grad_()
    action_full = actions.float().requires_grad_()
    full = train.filtration_envelope_loss(
        clean_full, observed_full, action_full, "cvpfv15")
    full_grad = torch.autograd.grad(
        full, (clean_full, observed_full, action_full), retain_graph=False)

    clean_detached = clean.float().requires_grad_()
    observed_detached = observed.float().requires_grad_()
    action_detached = actions.float().requires_grad_()
    detached = train.filtration_envelope_loss(
        clean_detached, observed_detached, action_detached, "cvpfv15_detachid")
    detached_grad = torch.autograd.grad(
        detached, (clean_detached, observed_detached, action_detached),
        retain_graph=False)
    assert torch.allclose(full, detached, atol=1e-7, rtol=1e-6)
    assert all(torch.isfinite(value).all() for value in (*full_grad, *detached_grad))
    assert any(not torch.allclose(left, right, atol=1e-7, rtol=1e-5)
               for left, right in zip(full_grad, detached_grad))


def test_noenvelope_is_exact_zero_with_exact_zero_gradient() -> None:
    clean, observed, actions = _synthetic(episodes=8)
    clean = clean.float().requires_grad_()
    observed = observed.float().requires_grad_()
    actions = actions.float().requires_grad_()
    loss = train.filtration_envelope_loss(
        clean, observed, actions, "cvpfv15_noenvelope")
    assert float(loss.detach()) == 0.0
    clean_gradient, = torch.autograd.grad(loss, (clean,), allow_unused=False)
    assert torch.count_nonzero(clean_gradient) == 0


def test_fit_payload_preserves_every_dataclass_field() -> None:
    clean, observed, actions = _synthetic()
    fit = fit_cvpf(clean, observed, actions, mode="full")
    payload = train.operator_fit_payload(fit)
    assert set(payload) == {field.name for field in dataclasses.fields(fit)}
    assert isinstance(payload["receipts"], dict)


def test_core_diagnostics_emit_the_frozen_shift_key() -> None:
    clean, observed, actions = _synthetic()
    memory = _world("cvpfv15").double().mem_cvpfv15
    memory.install_fit(fit_cvpf(clean, observed, actions, mode="full"))
    values = train.scalar_core_diagnostics(memory)
    assert "cvpf_shift_closure_relative" in values
    assert 0.0 <= values["cvpf_shift_closure_relative"] <= 1.0 + 1e-12
    assert values["cvpf_core_shift_closure_max_abs"] == \
        values["cvpf_shift_closure_relative"]


def test_recursive_action_suffix_is_observation_free_and_noaction_exact() -> None:
    clean, observed, actions = _synthetic()
    memory = _world("cvpfv15_noaction").double().mem_cvpfv15
    memory.install_fit(fit_cvpf(clean, observed, actions, mode="noaction"))
    _, details = memory(clean, actions, return_details=True)
    metrics = train.recursive_action_suffix_diagnostics(
        memory, clean, actions, torch.roll(actions, 1, 0), details)
    assert metrics["cvpf_action_suffix_semantics"] == \
        "all_sources_observation_free_equal_horizon_mean"
    assert metrics["cvpf_true_action_suffix_advantage"] == 0.0
    assert abs(metrics["cvpf_action_pair_accuracy"] - .5) <= 1e-7
    assert metrics["cvpf_prior_rollout_divergence"] == 0.0


def test_comparator_delegation_preserves_design_and_common_settings() -> None:
    for design in train.BASELINES:
        args = _args(design)
        delegated = train._delegate_argv(args)
        assert delegated[delegated.index("--memory-mode") + 1] == design
        assert delegated[delegated.index("--epochs") + 1] == str(args.epochs)
        assert delegated[delegated.index("--seed") + 1] == str(args.seed)
        assert "--development-action-ranking" not in delegated


def test_iid_contract_registers_full_future_horizon() -> None:
    original = train.load_cache
    base = dict(
        env_id="dummy", length=7, img_size=8, action_dim=2, state_dim=5,
        task_observation_dim=5, task_observation_keys=("q",),
        task_observation_shapes=((5,),), episodes=9,
        file_sha256="f", content_sha256="c")
    train_meta = SimpleNamespace(
        **base, split="train", seed=train.DEFAULT_TRAIN_SEED,
        smooth_rho=0.0, path=Path("/tmp/train.npz"))
    val_meta = SimpleNamespace(
        **base, split="val", seed=train.DEFAULT_VAL_SEED,
        smooth_rho=0.0, path=Path("/tmp/val.npz"))
    args = _args("cvpfv15")
    try:
        train.load_cache = lambda path: train_meta if path == args.train_data else val_meta
        train.validate_data_contract(args)
        assert args.cvpf_horizon == 6
        assert args.cvpf_fit_path == str(train_meta.path.resolve())
        correlated = SimpleNamespace(**{**train_meta.__dict__, "smooth_rho": .2})
        train.load_cache = lambda path: correlated if path == args.train_data else val_meta
        try:
            train.validate_data_contract(args)
        except ValueError as error:
            assert "IID actions" in str(error)
        else:
            raise AssertionError("correlated actions were accepted")
    finally:
        train.load_cache = original


def test_fit_collection_refuses_validation_dataset_before_loading() -> None:
    metadata = SimpleNamespace(split="val", path=Path("/tmp/train.npz"))
    dataset = SimpleNamespace(metadata=metadata, view="clean")
    args = SimpleNamespace(cvpf_fit_path="/tmp/train.npz")
    try:
        train.collect_detached_fit_views(
            None, dataset, dataset, args, torch.device("cpu"))
    except RuntimeError as error:
        assert "not the registered train cache" in str(error)
    else:
        raise AssertionError("validation data entered the CVPF fitter")


def test_metadata_exposes_true_envelope_gradient_controls() -> None:
    assert train.design_metadata("cvpfv15")["fit_gradient_active"] is True
    assert train.design_metadata("cvpfv15_detachid")["fit_gradient_active"] is False
    no_envelope = train.design_metadata("cvpfv15_noenvelope")
    assert no_envelope["memory_specific_loss_weight"] == 0.0
    assert no_envelope["envelope_weight"] == 0.0


def test_runtime_harness_hooks_are_restored() -> None:
    original_main = train.v12.main
    original_build = train.v12.build_model
    called = []
    try:
        def stub(argv):
            assert argv == []
            assert train.v12.build_model is train.build_model
            assert train.v12.compute_siro_losses is train.compute_cvpf_losses
            assert train.v12.HISTORY_KEYS is train.HISTORY_KEYS
            called.append(True)
        train.v12.main = stub
        train._run_candidate(_args("cvpfv15"))
    finally:
        train.v12.main = original_main
    assert called == [True]
    assert train.v12.build_model is original_build


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} CVPF-v15 integration tests passed.")


if __name__ == "__main__":
    main()
