#!/usr/bin/env python3
"""Train the frozen LeWM+V8 end-to-end confirmation cells.

This study changes no memory mechanism.  It reuses the repaired causal LeWM
pixel encoder/predictor, the stable clean-target VICReg host, paired corrupted
context and clean future targets, and the V11 evaluation/probe machinery.
The grid compares no memory, GRU, SSM, compact V8, its static/dynamic
correction endpoints, and V8's two strongest causal mechanisms (action
transport and joint two-state readout) on unopened tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.memory_model import MemoryLeWorldModel
from scripts import hacssm_v18_data as data
import scripts.train_hacssm_v11 as v11
import scripts.train_subjepa_v16 as base
from scripts.train_hacssm_v10 import _matched_gru_hidden


MEMORIES = (
    "none",
    "gru",
    "ssm",
    "hacssmv8",
    "hacssmv8_static",
    "hacssmv8_dynamic",
    "hacssmv8_noaction",
    "hacssmv8_single",
)
DESIGNS = tuple(f"vicreg_{memory}" for memory in MEMORIES)
SEEDS = (18_001, 18_002, 18_003, 18_004, 18_005)
DEFAULT_EPOCHS = 100
PREDICTOR_HISTORY = 3
OBJECTIVE = (
    "v18_paired_next_clean_sliding_h3_plus_vicreg_causal_memory_confirmation")

_BASE_CONTRACT_FIELDS = (
    "REGULARIZERS", "MEMORIES", "DESIGNS", "OBJECTIVE",
    "DEFAULT_TRAIN_SEED", "DEFAULT_VAL_SEED", "DEFAULT_CORRUPTION_SEED",
    "V11TrajectoryDataset", "load_cache", "parse_design", "design_metadata",
    "build_model", "memory_representations", "compute_losses",
)
_BASE_CONTRACT_ORIGINALS = {
    name: getattr(base, name) for name in _BASE_CONTRACT_FIELDS}
_V11_MEMORY_REPRESENTATIONS_ORIGINAL = v11.memory_representations
_V11_ONE_TOKEN_PREDICTION_ORIGINAL = v11.one_token_prediction


def parse_design(design: str) -> tuple[str, str, None]:
    if design not in DESIGNS:
        raise ValueError(f"unknown V18 design {design!r}")
    return "vicreg", design.removeprefix("vicreg_"), None


def design_metadata(design: str, embed_dim: int = 128) -> dict[str, Any]:
    _, memory, _ = parse_design(design)
    is_v8 = memory.startswith("hacssmv8")
    correction_mode = (
        "static" if memory == "hacssmv8_static" else
        "dynamic" if memory == "hacssmv8_dynamic" else
        "learned_shrinkage" if is_v8 else None)
    return {
        "method": "LeWM-SAS-PC-v18-confirmation",
        "evidence_scope": "unopened_task_confirmation",
        "wandb_method_tag": "lewm-v8-v18",
        "wandb_scope_tag": "unopened-task-confirmation",
        "confirmation_evidence": True,
        "executed_return_evaluation": False,
        "regularizer": "vicreg",
        "regularizer_family": "clean_target_variance_covariance",
        "regularizer_source": "active_clean_target",
        "memory_architecture": memory,
        "memory_specific_loss_weight": 0.0,
        "new_memory_architecture": False,
        "one_token_predictor": False,
        "predictor_history": PREDICTOR_HISTORY,
        "predictor_window_policy": "all_aligned_length_h_windows",
        "evaluation_predictor_window_policy": (
            "zero_padded_aligned_length_h_windows"),
        "paired_clean_target": True,
        "hidden_clean_targets_included": True,
        "clean_target_gradient_active": True,
        "target_stop_gradient": False,
        "reward_used_for_training": False,
        "state_labels_used_for_training": False,
        "training_objective": OBJECTIVE,
        "unopened_task_cohort": True,
        "causal_action_timing": "a_t_maps_z_t_to_z_t_plus_1",
        "v8_action_transport_enabled": (
            memory != "hacssmv8_noaction" if is_v8 else None),
        "v8_joint_read_enabled": (
            memory != "hacssmv8_single" if is_v8 else None),
        "v8_correction_mode": correction_mode,
        "memory_timescales": [2.0, 8.0] if is_v8 else [],
        "gru_probe_prior_contract": (
            "D_dimensional_read_of_h_t_minus_1_before_z_t"
            if memory == "gru" else None),
        "embedding_dimension": int(embed_dim),
    }


def build_model(args, action_dim: int) -> base.V16ExperimentModel:
    _, memory, _ = parse_design(args.design)
    if memory == "none":
        memory_impl, memory_mode = "ema", "none"
    elif memory in {"gru", "ssm"}:
        memory_impl, memory_mode = memory, "both"
    else:
        memory_impl, memory_mode = memory, "both"
    world = MemoryLeWorldModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        action_dim=action_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        predictor_norm="none",
        encoder_norm="causal",
        history_len=args.history_len,
        dropout=args.dropout,
        sigreg_lambda=0.0,
        sigreg_projections=args.sigreg_projections,
        memory_impl=memory_impl,
        memory_mode=memory_mode,
        gru_hidden=_matched_gru_hidden(args.embed_dim),
        hier_loss_weight=0.0,
        encoder_type="vit",
    )
    if world.encoder_norm != "causal" or world.predictor_norm != "none":
        raise RuntimeError("V18 requires the repaired causal/no-output-norm LeWM host")
    if world.history_len != PREDICTOR_HISTORY:
        raise RuntimeError(
            f"V18 requires predictor history {PREDICTOR_HISTORY}, "
            f"got {world.history_len}")
    if memory == "none":
        world.memory.requires_grad_(False)
        world.fusion.requires_grad_(False)
    return base.V16ExperimentModel(world, args.design)


def sliding_predictor_windows(
        fused: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor,
        history: int = PREDICTOR_HISTORY,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all aligned LeWM context/action windows and next targets."""
    if fused.dim() != 3 or targets.shape != fused.shape:
        raise ValueError(
            "fused and targets must share shape (B,L,D), got "
            f"{tuple(fused.shape)} and {tuple(targets.shape)}")
    batch, length, dimension = fused.shape
    if not isinstance(history, int) or isinstance(history, bool) or history < 1:
        raise ValueError("history must be a positive integer")
    if length < history + 1:
        raise ValueError(
            f"sequence length {length} must be at least history+1={history + 1}")
    if actions.dim() != 3 or actions.shape[:2] != (batch, length - 1):
        raise ValueError(
            "actions must have shape (B,L-1,A), got "
            f"{tuple(actions.shape)} for B={batch}, L={length}")
    windows = length - history
    latent_windows = fused.unfold(1, history, 1)[:, :windows]
    latent_windows = latent_windows.permute(0, 1, 3, 2).reshape(
        batch * windows, history, dimension)
    action_dim = actions.shape[-1]
    action_windows = actions.unfold(1, history, 1)[:, :windows]
    action_windows = action_windows.permute(0, 1, 3, 2).reshape(
        batch * windows, history, action_dim)
    next_targets = targets[:, history:].reshape(batch * windows, dimension)
    return latent_windows, action_windows, next_targets


def sliding_next_predictions(
        world: MemoryLeWorldModel, fused: torch.Tensor,
        actions: torch.Tensor) -> torch.Tensor:
    """Predict every next latent from the true finite LeWM history window."""
    latent_windows, action_windows, _ = sliding_predictor_windows(
        fused, actions, fused, history=world.history_len)
    batch, length, dimension = fused.shape
    windows = length - world.history_len
    prediction = world.predictor(latent_windows, action_windows)[:, -1]
    return prediction.reshape(batch, windows, dimension)


def finite_context_prediction(
        world: MemoryLeWorldModel, beliefs: torch.Tensor,
        actions: torch.Tensor) -> torch.Tensor:
    """V11-compatible predictor read using the true LeWM history window.

    The inherited probe API expects one output for every belief/action token
    and later discards the first ``H-1`` outputs.  We therefore zero-pad only
    those warm-up positions and fill indices ``H-1:`` with aligned H-token
    predictions.  Output index ``i`` predicts the latent after belief ``i``.
    """
    if beliefs.dim() != 3:
        raise ValueError(f"beliefs must have shape (B,T,D), got {beliefs.shape}")
    batch, steps, dimension = beliefs.shape
    history = world.history_len
    if steps < history:
        raise ValueError(
            f"belief sequence length {steps} is shorter than history {history}")
    if actions.dim() != 3 or actions.shape[:2] != (batch, steps):
        raise ValueError(
            "V18 evaluation actions must align one-for-one with beliefs, got "
            f"{tuple(actions.shape)} for beliefs {tuple(beliefs.shape)}")
    windows = steps - history + 1
    latent_windows = beliefs.unfold(1, history, 1)[:, :windows]
    latent_windows = latent_windows.permute(0, 1, 3, 2).reshape(
        batch * windows, history, dimension)
    action_dim = actions.shape[-1]
    action_windows = actions.unfold(1, history, 1)[:, :windows]
    action_windows = action_windows.permute(0, 1, 3, 2).reshape(
        batch * windows, history, action_dim)
    prediction = world.predictor(latent_windows, action_windows)[:, -1]
    padded = beliefs.new_zeros(batch, steps, dimension)
    padded[:, history - 1:] = prediction.reshape(batch, windows, dimension)
    return padded


def memory_representations(
        model: base.V16ExperimentModel, z: torch.Tensor,
        actions: torch.Tensor) -> dict[str, Any]:
    world = model.world
    _, memory, _ = parse_design(model.design)
    if memory == "none":
        prior = torch.zeros_like(z)
        history = world.history_len
        prior[:, history:] = sliding_next_predictions(world, z, actions)
        return {"fused": z, "prior": prior, "posterior": z, "details": {}}
    if memory == "ssm":
        states = world.mem_ssm(z)
        decay = torch.sigmoid(world.mem_ssm.raw_decay).to(dtype=states.dtype)
        prior = torch.zeros_like(states)
        bias = world.mem_ssm.in_proj.bias.to(dtype=states.dtype)
        prior[:, 0] = states[:, 0]
        prior[:, 1:] = (1.0 - decay) * states[:, :-1] + decay * bias
        return {
            "fused": world.mem_ssm.fuse(z, states),
            "prior": v11._rms_read(prior),
            "posterior": v11._rms_read(states),
            "details": {"states": states, "priors": prior},
        }
    if memory == "gru":
        # ``states[:, t]`` is the GRU hidden state after consuming z_t.  The
        # probe prior at t must therefore read ``states[:, t-1]``.  Applying
        # the deployed D-dimensional readout before shifting also prevents a
        # hidden-width mismatch from changing the state-probe contract.
        states = world.mem_gru(z)
        posterior_read = world.mem_gru.readout(states)
        prior_read = torch.zeros_like(posterior_read)
        prior_read[:, 1:] = posterior_read[:, :-1]
        return {
            "fused": world.mem_gru.fuse(z, states),
            "prior": v11._rms_read(prior_read),
            "posterior": v11._rms_read(posterior_read),
            "details": {
                "states": states,
                "prior_read": prior_read,
                "posterior_read": posterior_read,
            },
        }
    if memory.startswith("hacssmv8"):
        mixed, details = world.mem_hacssmv8(z, actions, return_details=True)
        route = details["route"].to(
            device=z.device, dtype=z.dtype).view(1, 1, -1, 1)
        prior = (details["priors"] * route).sum(dim=2)
        posterior = (details["states"] * route).sum(dim=2)
        return {
            "fused": world.mem_hacssmv8.fuse(z, mixed),
            "prior": v11._rms_read(prior),
            "posterior": v11._rms_read(posterior),
            "details": details,
        }
    raise AssertionError(f"unhandled V18 memory {memory!r}")


def compute_losses(
        model: base.V16ExperimentModel, observed: torch.Tensor,
        clean: torch.Tensor, actions: torch.Tensor,
        sigreg_lambda: float) -> dict[str, torch.Tensor]:
    """Ordinary H=3 LeWM prediction plus unit-weight VICReg stabilization."""
    del sigreg_lambda  # V18 has no selectable regularizer coefficient.
    clean_z = v11.encode_clean_active(model, clean)
    observed_z = model.world.encode(observed)
    representations = memory_representations(model, observed_z, actions)
    latent_windows, action_windows, targets = sliding_predictor_windows(
        representations["fused"], actions, clean_z,
        history=model.world.history_len)
    prediction = model.world.predictor(latent_windows, action_windows)[:, -1]
    predictive_loss = F.mse_loss(prediction.float(), targets.float())
    variance_loss, covariance_loss = v11._vicreg_terms(clean_z)
    regularizer_loss = variance_loss + covariance_loss
    zero = predictive_loss.new_zeros(())
    return {
        "loss": predictive_loss + regularizer_loss,
        "predictive_loss": predictive_loss,
        "regularizer_loss": regularizer_loss,
        "sigreg_loss": zero,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
    }


def _install_contract() -> None:
    # The base module supplies the audited training, probe, rollout, W&B, and
    # artifact-writing shell.  All scientific identities are replaced here.
    base.REGULARIZERS = ("vicreg",)
    base.MEMORIES = MEMORIES
    base.DESIGNS = DESIGNS
    base.OBJECTIVE = OBJECTIVE
    base.DEFAULT_TRAIN_SEED = data.DEFAULT_TRAIN_SEED
    base.DEFAULT_VAL_SEED = data.DEFAULT_VAL_SEED
    base.DEFAULT_CORRUPTION_SEED = data.DEFAULT_CORRUPTION_SEED
    base.V11TrajectoryDataset = data.V18TrajectoryDataset
    base.load_cache = data.load_cache
    base.parse_design = parse_design
    base.design_metadata = design_metadata
    base.build_model = build_model
    base.memory_representations = memory_representations
    base.compute_losses = compute_losses
    v11.one_token_prediction = finite_context_prediction


def _restore_contract() -> None:
    for name, value in _BASE_CONTRACT_ORIGINALS.items():
        setattr(base, name, value)
    v11.memory_representations = _V11_MEMORY_REPRESENTATIONS_ORIGINAL
    v11.one_token_prediction = _V11_ONE_TOKEN_PREDICTION_ORIGINAL


def _main_installed(argv: Iterable[str] | None = None) -> None:
    forwarded = list(argv) if argv is not None else list(sys.argv[1:])
    if "--epochs" not in forwarded:
        forwarded.extend(("--epochs", str(DEFAULT_EPOCHS)))
    base.main(forwarded)


def main(argv: Iterable[str] | None = None) -> None:
    _install_contract()
    try:
        _main_installed(argv)
    finally:
        _restore_contract()


if __name__ == "__main__":
    main()
