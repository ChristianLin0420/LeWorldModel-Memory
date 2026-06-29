#!/usr/bin/env python3
"""Frozen LOIF-v9 scale diagnostics and causal resistance interventions.

The intervention donor set is training-only.  For every validation episode/time, the
permuted override draws from a training episode and a time in the same frozen phase at
an index no later than the evaluation index.  The mean override averages exactly that
eligible training phase/prefix set.  No validation value or future index is a donor.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


DIAGNOSTICS_SCHEMA_VERSION = 1
DONOR_SEED = 9_009
SATURATION_TOLERANCE = 1e-4
LOG_SCALE_EXTREME_THRESHOLD = 20.0
STREAMING_TOLERANCE = 1e-5
DONOR_CONTRACT = {
    "schema_version": 1,
    "rng": "numpy.random.Generator(numpy.random.PCG64)",
    "seed": DONOR_SEED,
    "donor_split": "training_only",
    "recipient_split": "validation_only",
    "permuted_rule": (
        "for each validation episode e and memory time t, sample a training episode "
        "and a donor time t_prime in the same frozen phase with t_prime <= t"
    ),
    "mean_rule": (
        "arithmetic mean over all training episodes and donor times in the same "
        "frozen phase with t_prime <= t"
    ),
    "phase_strata": (
        "pre", "blackout_transition", "deep_blackout", "first_post", "recovery",
        "late_post",
    ),
    "future_or_validation_donors": False,
}
DONOR_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        DONOR_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()


def _amp_context(use_amp: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()


def _fixed_boundaries(length: int, history_len: int) -> tuple[int, int, int, int]:
    if length <= history_len or history_len < 1:
        raise ValueError("invalid sequence/history length for V9 diagnostics")
    occ_start = length // 3
    occ_end = min(length, occ_start + max(4, length // 5))
    deep_start = min(occ_end, occ_start + history_len)
    late_start = min(length, occ_end + history_len)
    return occ_start, deep_start, occ_end, late_start


def donor_phase_labels(length: int, history_len: int) -> np.ndarray:
    """Return the six non-overlapping donor strata for memory indices ``0..L-1``."""
    occ_start, deep_start, occ_end, late_start = _fixed_boundaries(
        length, history_len
    )
    labels = np.empty(length, dtype=np.int64)
    labels[:occ_start] = 0
    labels[occ_start:deep_start] = 1
    labels[deep_start:occ_end] = 2
    if occ_end < length:
        labels[occ_end] = 3
    labels[occ_end + 1:late_start] = 4
    labels[late_start:] = 5
    return labels


def diagnostic_phase_masks(length: int, history_len: int) -> dict[str, np.ndarray]:
    """Four disjoint diagnostic phases, excluding the unused initialization at t=0."""
    occ_start, deep_start, occ_end, late_start = _fixed_boundaries(
        length, history_len
    )
    time = np.arange(length)
    active = time > 0
    result = {
        "visible": active & ((time < occ_start) | (time >= late_start)),
        "blackout_transition": active & (time >= occ_start) & (time < deep_start),
        "deep_blackout": active & (time >= deep_start) & (time < occ_end),
        "recovery": active & (time >= occ_end) & (time < late_start),
    }
    if not all(mask.any() for mask in result.values()):
        raise ValueError("one or more V9 diagnostic phases are empty")
    covered = sum(mask.astype(np.int64) for mask in result.values())
    if not np.array_equal(covered, active.astype(np.int64)):
        raise ValueError("V9 diagnostic phases do not partition active memory indices")
    return result


def build_resistance_overrides(
    train_resistance: np.ndarray,
    *,
    validation_episodes: int,
    history_len: int,
    seed: int = DONOR_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic training-donor permutation and prefix-mean overrides."""
    values = np.asarray(train_resistance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("train_resistance must have shape (episodes,length)")
    if validation_episodes < 1:
        raise ValueError("validation_episodes must be positive")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("training resistance donors must be finite and strictly positive")

    n_train, length = values.shape
    labels = donor_phase_labels(length, history_len)
    time = np.arange(length)
    rng = np.random.Generator(np.random.PCG64(seed))
    permuted = np.empty((validation_episodes, length), dtype=np.float32)
    mean = np.empty((validation_episodes, length), dtype=np.float32)
    for target_t in range(length):
        eligible = np.flatnonzero((labels == labels[target_t]) & (time <= target_t))
        if eligible.size == 0:
            raise ValueError(f"empty resistance donor set at time {target_t}")
        mean[:, target_t] = float(values[:, eligible].mean())
        source_episodes = rng.integers(0, n_train, size=validation_episodes)
        source_times = eligible[rng.integers(0, eligible.size, size=validation_episodes)]
        permuted[:, target_t] = values[source_episodes, source_times]
    if not np.isfinite(permuted).all() or not np.isfinite(mean).all():
        raise ValueError("constructed resistance overrides are non-finite")
    if (permuted <= 0).any() or (mean <= 0).any():
        raise ValueError("constructed resistance overrides must be strictly positive")
    return permuted, mean


def _coerce_detail(
    details: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = details.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"LOIF details missing tensor {name!r}")
    if value.shape == (*shape, 1):
        value = value.squeeze(-1)
    if tuple(value.shape) != shape:
        raise ValueError(
            f"LOIF detail {name!r} has shape {tuple(value.shape)}, expected {shape}"
        )
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError(f"LOIF detail {name!r} is non-finite")
    return value


@torch.no_grad()
def collect_loif_details(
    model,
    dataset,
    *,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    include_action_influence: bool = False,
) -> dict[str, np.ndarray]:
    """Collect the complete candidate scale trace in deterministic dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "resistance", "log_R", "gains", "log_P", "prior_weights", "read_weights"
        )
    }
    collected["innovation_norm"] = []
    if include_action_influence:
        collected["action_state_influence"] = []
        collected["action_output_influence"] = []
    model.eval()
    expected_length = None
    for batch in loader:
        if len(batch) != 4:
            raise ValueError("LOIF diagnostics require paired features and a validity mask")
        obs, actions, _, target_mask = batch
        obs = obs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        with _amp_context(use_amp):
            z = model.encode(obs)
            fused, details = model._inject(
                z,
                actions=actions,
                memory_update_mask=target_mask,
                return_memory_details=True,
            )
            if include_action_influence:
                fused_zero_action, zero_action_details = model._inject(
                    z,
                    actions=actions,
                    memory_update_mask=target_mask,
                    action_override=0.0,
                    return_memory_details=True,
                )
        B, length, _ = z.shape
        if expected_length is None:
            expected_length = length
        elif expected_length != length:
            raise ValueError("inconsistent sequence length in LOIF diagnostics")
        shapes = {
            "resistance": (B, length),
            "log_R": (B, length),
            "gains": (B, length, 2),
            "log_P": (B, length, 2),
            "prior_weights": (B, length, 2),
            "read_weights": (B, length, 2),
        }
        batch_values = {}
        for name, shape in shapes.items():
            value = _coerce_detail(details, name, shape)
            batch_values[name] = value
            collected[name].append(value.cpu().numpy())
        x = _coerce_detail(details, "x", (B, length, z.shape[-1]))
        priors = _coerce_detail(
            details, "priors", (B, length, 2, z.shape[-1])
        )
        prior_mixture = (
            priors * batch_values["prior_weights"].unsqueeze(-1)
        ).sum(dim=2)
        innovation_norm = (x - prior_mixture).square().mean(dim=-1).sqrt()
        collected["innovation_norm"].append(innovation_norm.cpu().numpy())
        if include_action_influence:
            zero_states = _coerce_detail(
                zero_action_details, "states", (B, length, 2, z.shape[-1])
            )
            states = _coerce_detail(
                details, "states", (B, length, 2, z.shape[-1])
            )
            state_influence = (
                (states - zero_states).square().mean(dim=-1).sqrt().mean(dim=-1)
            )
            output_influence = (
                (fused.float() - fused_zero_action.float()).square().mean(dim=-1).sqrt()
            )
            collected["action_state_influence"].append(
                state_influence.cpu().numpy()
            )
            collected["action_output_influence"].append(
                output_influence.cpu().numpy()
            )

    if not collected["resistance"]:
        raise ValueError("empty dataset in LOIF diagnostics")
    result = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
    resistance = result["resistance"]
    if (resistance <= 0).any():
        raise ValueError("LOIF resistance must be strictly positive")
    if not np.allclose(result["log_R"], np.log(resistance), rtol=3e-3, atol=3e-3):
        raise ValueError("LOIF log_R is inconsistent with resistance")
    gains = result["gains"]
    if (gains < 0).any() or (gains > 1).any():
        raise ValueError("LOIF gains must lie in [0,1]")
    for name in ("prior_weights", "read_weights"):
        weights = result[name]
        if (weights < 0).any() or not np.allclose(
            weights.sum(axis=-1), 1.0, rtol=3e-3, atol=3e-3
        ):
            raise ValueError(f"LOIF {name} must be a non-negative simplex")
    return result


def summarize_loif_details(
    details: Mapping[str, np.ndarray],
    *,
    alpha_fast: float,
    alpha_slow: float,
    q_fast: float,
    q_slow: float,
    history_len: int,
) -> dict[str, float]:
    """Return the frozen flat per-phase diagnostic schema used by the analyzer."""
    alpha = np.asarray([alpha_fast, alpha_slow], dtype=np.float64)
    q = np.asarray([q_fast, q_slow], dtype=np.float64)
    if not np.isfinite(alpha).all() or not (0 <= alpha[0] <= alpha[1] < 1):
        raise ValueError("LOIF alphas are non-finite or outside the closed ordered domain")
    if not np.allclose(q, (1.0 - alpha) * (1.0 + alpha), rtol=1e-6, atol=1e-7):
        raise ValueError("LOIF q is inconsistent with alpha")
    length = np.asarray(details["log_R"]).shape[1]
    masks = diagnostic_phase_masks(length, history_len)
    gains = np.asarray(details["gains"], dtype=np.float64)
    direct = alpha.reshape(1, 1, 2) * (1.0 - gains)
    values = {
        "log_R": np.asarray(details["log_R"], dtype=np.float64),
        "K_fast": gains[..., 0],
        "K_slow": gains[..., 1],
        "log_P_fast": np.asarray(details["log_P"], dtype=np.float64)[..., 0],
        "log_P_slow": np.asarray(details["log_P"], dtype=np.float64)[..., 1],
        "omega_fast": np.asarray(details["prior_weights"], dtype=np.float64)[..., 0],
        "omega_slow": np.asarray(details["prior_weights"], dtype=np.float64)[..., 1],
        "pi_fast": np.asarray(details["read_weights"], dtype=np.float64)[..., 0],
        "pi_slow": np.asarray(details["read_weights"], dtype=np.float64)[..., 1],
        "direct_fast": direct[..., 0],
        "direct_slow": direct[..., 1],
        "innovation_norm": np.asarray(details["innovation_norm"], dtype=np.float64),
    }
    if "action_state_influence" in details:
        values["action_state_influence"] = np.asarray(
            details["action_state_influence"], dtype=np.float64
        )
        values["action_output_influence"] = np.asarray(
            details["action_output_influence"], dtype=np.float64
        )
    separation = float(alpha_slow - alpha_fast)
    boundary_margin = float(min(alpha_fast, 1.0 - alpha_slow))
    active_gains = gains[:, 1:]
    active_log_r = np.asarray(details["log_R"], dtype=np.float64)[:, 1:]
    active_log_p = np.asarray(details["log_P"], dtype=np.float64)[:, 1:]
    result: dict[str, float] = {
        "alpha_fast": float(alpha_fast),
        "alpha_slow": float(alpha_slow),
        "q_fast": float(q_fast),
        "q_slow": float(q_slow),
        "loif_pole_separation": separation,
        "loif_fast_boundary_margin": float(alpha_fast),
        "loif_slow_boundary_margin": float(1.0 - alpha_slow),
        "loif_pole_boundary_margin": boundary_margin,
        "loif_pole_collapsed": bool(separation <= SATURATION_TOLERANCE),
        "loif_boundary_saturated": bool(
            boundary_margin <= SATURATION_TOLERANCE
        ),
        "loif_saturation_tolerance": SATURATION_TOLERANCE,
        "loif_log_scale_extreme_threshold": LOG_SCALE_EXTREME_THRESHOLD,
        "loif_gain_saturated_fraction": float(np.mean(
            (active_gains <= SATURATION_TOLERANCE)
            | (active_gains >= 1.0 - SATURATION_TOLERANCE)
        )),
        "loif_log_R_extreme_fraction": float(np.mean(
            np.abs(active_log_r) >= LOG_SCALE_EXTREME_THRESHOLD
        )),
        "loif_log_P_extreme_fraction": float(np.mean(
            np.abs(active_log_p) >= LOG_SCALE_EXTREME_THRESHOLD
        )),
        "loif_nonfinite_diagnostic_count": 0,
    }
    for phase, mask in masks.items():
        for stat, array in values.items():
            value = float(array[:, mask].mean())
            if not np.isfinite(value):
                raise ValueError(f"non-finite LOIF summary {stat}/{phase}")
            result[f"loif_{stat}_{phase}"] = value
        innovation = values["innovation_norm"][:, mask].reshape(-1)
        log_r = values["log_R"][:, mask].reshape(-1)
        innovation_std = float(innovation.std())
        log_r_std = float(log_r.std())
        correlation = 0.0 if min(innovation_std, log_r_std) == 0.0 else float(
            np.clip(np.corrcoef(innovation, log_r)[0, 1], -1.0, 1.0)
        )
        if not np.isfinite(correlation):
            raise ValueError(f"non-finite innovation/log_R correlation in {phase}")
        result[f"loif_innovation_log_R_corr_{phase}"] = correlation
        result[f"loif_innovation_or_log_R_constant_{phase}"] = bool(
            min(innovation_std, log_r_std) == 0.0
        )
    return result


@torch.no_grad()
def intervention_mse(
    model,
    dataset,
    resistance_override: np.ndarray,
    *,
    label: str,
    device: torch.device,
    use_amp: bool,
    history_len: int,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate clean-target MSE with one complete exogenous R trajectory."""
    override = np.asarray(resistance_override, dtype=np.float32)
    if override.ndim != 2 or override.shape[0] != len(dataset):
        raise ValueError("resistance override episode count mismatch")
    if not np.isfinite(override).all() or (override <= 0).any():
        raise ValueError("resistance override must be finite and strictly positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sums = None
    count = 0
    offset = 0
    length = override.shape[1]
    model.eval()
    for batch in loader:
        if len(batch) != 4:
            raise ValueError("LOIF intervention requires paired features and a validity mask")
        obs, actions, target_obs, target_mask = batch
        B = obs.shape[0]
        obs = obs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        target_obs = target_obs.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        override_batch = torch.from_numpy(override[offset:offset + B]).to(
            device=device, dtype=obs.dtype
        )
        offset += B
        with _amp_context(use_amp):
            z = model.encode(obs)
            z_target = model.encode(target_obs)
            fused = model._inject(
                z,
                actions=actions,
                memory_update_mask=target_mask,
                resistance_override=override_batch,
            )
            if z.shape[1] != length:
                raise ValueError("resistance override sequence length mismatch")
            windows = length - history_len
            fused_win = fused.unfold(1, history_len, 1)[:, :windows]
            fused_win = fused_win.permute(0, 1, 3, 2).reshape(
                B * windows, history_len, z.shape[-1]
            )
            action_win = actions.unfold(1, history_len, 1)[:, :windows]
            action_win = action_win.permute(0, 1, 3, 2).reshape(
                B * windows, history_len, actions.shape[-1]
            )
            prediction = model.predictor(fused_win, action_win)[:, -1].reshape(
                B, windows, z.shape[-1]
            )
            target = z_target[:, history_len:length]
            error = (prediction - target).float().square().mean(-1).sum(0).cpu().numpy()
        sums = error if sums is None else sums + error
        count += B
    if offset != len(dataset) or count == 0:
        raise ValueError("LOIF intervention loader accounting failed")
    per_t = sums / count
    target_t = np.arange(history_len, length)
    _, deep_start, occ_end, _ = _fixed_boundaries(length, history_len)
    masks = {
        "first_post": target_t == occ_end,
        "deep_blackout": (target_t >= deep_start) & (target_t < occ_end),
        "all": np.ones_like(target_t, dtype=np.bool_),
    }
    result = {}
    for phase, mask in masks.items():
        if not mask.any():
            raise ValueError(f"empty LOIF intervention phase {phase}")
        value = float(per_t[mask].mean())
        if not np.isfinite(value):
            raise ValueError(f"non-finite LOIF intervention MSE {phase}/{label}")
        result[f"clean_mse_{phase}_resistance_{label}"] = value
    return result


@torch.no_grad()
def streaming_equivalence_receipt(
    model,
    dataset,
    *,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    """Compare the batched scan with the explicit 2D+2 batch-size-one recurrence."""
    if len(dataset) < 1 or not hasattr(model, "mem_loifv9"):
        raise ValueError("streaming receipt requires a non-empty LOIF dataset/model")
    sample = dataset[0]
    if len(sample) != 4:
        raise ValueError("streaming receipt requires paired features and a validity mask")
    obs, actions, _, target_mask = (
        value.unsqueeze(0).to(device) for value in sample
    )
    model.eval()
    memory = model.mem_loifv9
    with _amp_context(use_amp):
        z = model.encode(obs)
        full_mixed, full_details = memory(
            z, actions, memory_update_mask=target_mask, return_details=True
        )
        state = full_details["x"][:, 0].unsqueeze(1).expand(-1, memory.K, -1)
        log_scale = torch.zeros(
            1, memory.K, device=z.device,
            dtype=(torch.float32 if z.dtype in {torch.float16, torch.bfloat16} else z.dtype),
        )
        initial_weights = memory._fusion_weights(log_scale)
        stream_mixed = [memory._rms_norm((
            state * initial_weights.to(dtype=state.dtype).unsqueeze(-1)
        ).sum(dim=1))]
        for time_index in range(1, z.shape[1]):
            mixed, state, log_scale, _ = memory.filter_step(
                state, log_scale, z[:, time_index], actions[:, time_index - 1]
            )
            stream_mixed.append(mixed)
        stream_mixed = torch.stack(stream_mixed, dim=1)
    mixed_error = float((stream_mixed.float() - full_mixed.float()).abs().max())
    state_error = float((state.float() - full_details["states"][:, -1].float()).abs().max())
    scale_error = float((
        log_scale.float() - full_details["log_P"][:, -1].float()
    ).abs().max())
    maximum = max(mixed_error, state_error, scale_error)
    if not np.isfinite(maximum):
        raise ValueError("non-finite LOIF streaming equivalence error")
    return {
        "loif_streaming_batch_size": 1,
        "loif_streaming_tolerance": STREAMING_TOLERANCE,
        "loif_streaming_mixed_max_abs": mixed_error,
        "loif_streaming_state_max_abs": state_error,
        "loif_streaming_log_P_max_abs": scale_error,
        "loif_streaming_equivalent": bool(maximum <= STREAMING_TOLERANCE),
    }


@torch.no_grad()
def evaluate_loif_v9_diagnostics(
    model,
    train_dataset,
    validation_dataset,
    *,
    device: torch.device,
    use_amp: bool,
    history_len: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run the complete frozen LOIF-v9 diagnostic/intervention contract."""
    train = collect_loif_details(
        model, train_dataset, device=device, use_amp=use_amp, batch_size=batch_size
    )
    validation = collect_loif_details(
        model, validation_dataset, device=device, use_amp=use_amp, batch_size=batch_size,
        include_action_influence=True,
    )
    if train["resistance"].shape[1] != validation["resistance"].shape[1]:
        raise ValueError("train/validation LOIF sequence length mismatch")
    horizons = model.horizons()
    required = ("alpha_fast", "alpha_slow", "q_fast", "q_slow")
    if not all(name in horizons for name in required):
        raise ValueError("LOIF horizons missing alpha/q diagnostics")
    result: dict[str, Any] = {
        "loif_diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "loif_donor_contract_sha256": DONOR_CONTRACT_SHA256,
        "loif_donor_seed": DONOR_SEED,
        "loif_donor_train_episodes": int(train["resistance"].shape[0]),
        "loif_donor_val_episodes": int(validation["resistance"].shape[0]),
    }
    result.update(summarize_loif_details(
        validation,
        alpha_fast=float(horizons["alpha_fast"]),
        alpha_slow=float(horizons["alpha_slow"]),
        q_fast=float(horizons["q_fast"]),
        q_slow=float(horizons["q_slow"]),
        history_len=history_len,
    ))
    permuted, mean = build_resistance_overrides(
        train["resistance"],
        validation_episodes=len(validation_dataset),
        history_len=history_len,
    )
    result.update(intervention_mse(
        model, validation_dataset, permuted, label="permuted", device=device,
        use_amp=use_amp, history_len=history_len, batch_size=batch_size,
    ))
    result.update(intervention_mse(
        model, validation_dataset, mean, label="mean", device=device,
        use_amp=use_amp, history_len=history_len, batch_size=batch_size,
    ))
    result.update({
        "loif_resistance_permuted_sha256": hashlib.sha256(
            np.ascontiguousarray(permuted.astype("<f4")).tobytes()
        ).hexdigest(),
        "loif_resistance_mean_sha256": hashlib.sha256(
            np.ascontiguousarray(mean.astype("<f4")).tobytes()
        ).hexdigest(),
    })
    result.update(streaming_equivalence_receipt(
        model, validation_dataset, device=device, use_amp=use_amp
    ))
    return result
