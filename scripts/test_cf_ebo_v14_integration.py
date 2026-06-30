#!/usr/bin/env python3
"""Host-registry and trainer integration tests for CF-EBO-v14."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cf_ebo import fit_cf_ebo
from lewm.models.memory_model import MemoryLeWorldModel
import scripts.train_cf_ebo_v14 as train


def _synthetic(episodes: int = 120, length: int = 7):
    generator = torch.Generator().manual_seed(14_777)
    transition = torch.tensor([[.72, -.16], [.16, .72]], dtype=torch.float64)
    action_map = torch.tensor([[.24, -.08], [.05, .19]], dtype=torch.float64)
    read = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [.6, -.3], [.2, .7]], dtype=torch.float64)
    actions = torch.randn(
        episodes, length - 1, 2, generator=generator, dtype=torch.float64)
    state = torch.zeros(episodes, length, 2, dtype=torch.float64)
    state[:, 0] = torch.randn(episodes, 2, generator=generator, dtype=torch.float64)
    for step in range(length - 1):
        state[:, step + 1] = (
            state[:, step] @ transition.T + actions[:, step] @ action_map.T)
    clean = state @ read.T
    noise = .02 * torch.randn(clean.shape, generator=generator, dtype=torch.float64)
    # Include sparse large paired corruptions so the energy-cap fit is identifiable.
    noise[::3, 2:4] += .6 * torch.randn(
        noise[::3, 2:4].shape, generator=generator, dtype=torch.float64)
    observed = clean + noise
    observed[:, 0] = clean[:, 0]
    return clean, observed, actions


def _world(mode: str, state_dim: int) -> MemoryLeWorldModel:
    return MemoryLeWorldModel(
        img_size=8, patch_size=4, embed_dim=4, action_dim=2,
        encoder_layers=1, encoder_heads=2,
        predictor_layers=1, predictor_heads=2,
        predictor_norm="none", encoder_norm="causal", history_len=2,
        dropout=0.0, sigreg_projections=4, memory_impl=mode,
        cf_hiro_state_dim=state_dim)


def _args(design: str) -> argparse.Namespace:
    return argparse.Namespace(
        train_data="train.npz", val_data="val.npz", memory_mode=design,
        seed=14, output_dir="out", epochs=3, batch_size=4, lr=3e-4,
        weight_decay=1e-5, num_workers=0, img_size=8, patch_size=4,
        embed_dim=4, encoder_layers=1, encoder_heads=2,
        predictor_layers=1, predictor_heads=2, history_len=2, dropout=0.0,
        sigreg_lambda=.1, sigreg_projections=4, probe_ridge=1e-3,
        eval_target_key="task_observation", corruption_seed=1,
        eval_rollout_episode=0, no_amp=True, device="cpu", wandb=False,
        wandb_entity=None, wandb_project="test", wandb_mode="online",
        wandb_study="test", extra_tag="")


def test_registry_installs_every_mode_with_zero_memory_parameters() -> None:
    for design, core_mode in train.CORE_MODES.items():
        world = _world(design, 6)
        assert world.mem_cfebov14.mode == core_mode
        assert world.mem_cfebov14.parameter_count() == 0
        assert world.horizons() == {"state_dim": 6.0}


def test_fit_install_and_direct_predictor_fusion_for_every_mode() -> None:
    clean, observed, actions = _synthetic()
    state_dim = train.full_hankel_state_dim(
        clean.shape[1], clean.shape[2], actions.shape[2])
    assert state_dim == 6
    for design, mode in train.CORE_MODES.items():
        fit = fit_cf_ebo(clean, observed, actions, mode=mode)
        world = _world(design, state_dim).double()
        world.mem_cfebov14.install_fit(fit)
        direct, direct_details = world.mem_cfebov14(
            observed[:3], actions[:3], return_details=True)
        injected, injected_details = world._inject(
            observed[:3], actions[:3], return_memory_details=True)
        assert torch.equal(injected, direct)
        assert torch.equal(injected_details["prior_reads"], direct_details["prior_reads"])
        assert torch.equal(injected_details["posterior_reads"], direct)
        assert torch.allclose(direct[:, 0], observed[:3, 0], atol=2e-12, rtol=0.0)
        if mode == "noaction":
            assert torch.count_nonzero(direct_details["action_effects"]) == 0
        if mode == "nocorrect":
            assert torch.count_nonzero(direct_details["corrections"]) == 0
        if mode == "norisk":
            assert float(world.mem_cfebov14.action_reliability) == 1.0
            assert float(world.mem_cfebov14.correction_reliability) == 1.0
        if mode == "noenergycap":
            assert torch.allclose(
                world.mem_cfebov14.correction_matrix,
                world.mem_cfebov14.raw_correction_matrix,
                atol=2e-12, rtol=2e-12)
        if mode == "noradial":
            assert torch.equal(
                direct_details["radial_gates"],
                torch.ones_like(direct_details["radial_gates"]))


def test_common_lewm_loss_path_has_no_auxiliary_and_backpropagates() -> None:
    clean, observed, actions = _synthetic()
    fit = fit_cf_ebo(clean, observed, actions, mode="full")
    world = _world("cfebov14", fit.state_matrix.shape[0])
    world.mem_cfebov14.install_fit(fit)
    generator = torch.Generator().manual_seed(14_778)
    images = torch.rand(2, 7, 3, 8, 8, generator=generator)
    losses = world.compute_loss(images, actions[:2].float())
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert world.encoder.patch_embed.weight.grad is not None
    assert world.predictor.projector[0].weight.grad is not None
    assert list(world.mem_cfebov14.parameters()) == []


def test_trainer_representation_contract_is_strict_and_direct() -> None:
    clean, observed, actions = _synthetic()
    fit = fit_cf_ebo(clean, observed, actions, mode="full")
    model = train.CFEBOExperimentModel(
        _world("cfebov14", fit.state_matrix.shape[0]).double())
    model.world.mem_cfebov14.install_fit(fit)
    representations = train.memory_representations(model, observed[:4], actions[:4])
    assert torch.equal(representations["fused"], representations["posterior"])
    assert representations["prior"].shape == observed[:4].shape


def test_comparator_delegation_preserves_design_and_common_settings() -> None:
    for design in train.BASELINES:
        args = _args(design)
        delegated = train._delegate_argv(args)
        assert delegated[delegated.index("--memory-mode") + 1] == design
        assert delegated[delegated.index("--epochs") + 1] == str(args.epochs)
        assert delegated[delegated.index("--seed") + 1] == str(args.seed)
        # KDIO's registered ranking is added by the frozen V13/V11 delegation,
        # not silently overridden by V14.
        assert "--development-action-ranking" not in delegated


def test_iid_contract_and_fixed_state_schema() -> None:
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
    args = _args("cfebov14")
    try:
        train.load_cache = lambda path: train_meta if path == args.train_data else val_meta
        train.validate_data_contract(args)
        assert args.cf_hiro_state_dim == 6
        assert args.cf_ebo_fit_path == str(train_meta.path.resolve())
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
    args = SimpleNamespace(cf_ebo_fit_path="/tmp/train.npz")
    try:
        train.collect_detached_fit_views(
            None, dataset, dataset, args, torch.device("cpu"))
    except RuntimeError as error:
        assert "not the registered train cache" in str(error)
    else:
        raise AssertionError("validation data entered the CF-EBO fitter")


def test_runtime_harness_hooks_are_restored() -> None:
    original_main = train.v12.main
    original_build = train.v12.build_model
    called = []
    try:
        def stub(argv):
            assert argv == []
            assert train.v12.build_model is train.build_model
            assert train.v12.memory_representations is train.memory_representations
            called.append(True)
        train.v12.main = stub
        train._run_candidate(_args("cfebov14"))
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
    print(f"All {len(tests)} CF-EBO-v14 integration tests passed.")


if __name__ == "__main__":
    main()
