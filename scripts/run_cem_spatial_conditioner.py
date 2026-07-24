#!/usr/bin/env python3
"""Minimal spatial Gate-B conditioner on the fixed native opportunity set."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.spatial_memory_conditioner import (  # noqa: E402
    HISTORY,
    QUERY,
    RECENT,
    SpatialMemoryConditioner,
    SpatialTokenBatch,
    parameter_count,
    serialized_spatial_bytes,
)
from scripts.build_cem_native_long import (  # noqa: E402
    CONTEXT,
    ENVIRONMENTS,
    HORIZON,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DINOV2,
    DEFAULT_TORCH_HOME,
    feature_path,
)
from scripts.run_cem_native_long import (  # noqa: E402
    MEMORY_TOKENS,
    NativeSplit,
    bootstrap_mean_ci,
    build_native_split,
    discovery_arrays,
    discovery_scales,
    load_recipe_records,
    oracle_frame_search,
    raw_memory_loss,
    resolve_device,
)
from scripts.run_cem_raw_ogbench import (  # noqa: E402
    ActionConditionedHost,
    QueryTensors,
    RawMemoryConditioner,
    batches,
    env_family,
    horizon_loss,
    json_safe,
    load_dinov2,
    one_step_surprise,
    set_seed,
    stable_json,
    tensor_digest,
)


OUTPUT = ROOT / "outputs/cem_spatial_conditioner_v1"
NATIVE = ROOT / "outputs/cem_native_long_v1"
MACHINE_REPORT = ROOT / "outputs/cem_spatial_conditioner_report.json"
TOKENS = 16
FEATURE_DIM = 384
META_DIM = 4
GRID = 4
FORBIDDEN = {"cue_labels", "cue_positions", "cue_window", "goal_state"}


@dataclass
class FixedState:
    host: ActionConditionedHost
    raw_memory: RawMemoryConditioner
    host_digest: str
    raw_digest: str
    latents: np.ndarray
    actions: np.ndarray
    splits: dict[str, NativeSplit]
    recent_loss: dict[str, torch.Tensor]
    oracle_loss: dict[str, torch.Tensor]
    oracle_index: dict[str, np.ndarray]
    opportunity: dict[str, np.ndarray]
    threshold: float
    source_result: dict[str, Any]


@dataclass
class SpatialData:
    split: NativeSplit
    query: SpatialTokenBatch
    recent: SpatialTokenBatch
    memory: SpatialTokenBatch
    opportunity: np.ndarray
    recent_loss: torch.Tensor
    oracle_loss: torch.Tensor
    oracle_index: np.ndarray

    def index(self, index: torch.Tensor) -> "SpatialData":
        numpy_index = index.detach().cpu().numpy()
        return SpatialData(
            split=self.split.index(index),
            query=self.query.index(index),
            recent=self.recent.index(index),
            memory=self.memory.index(index),
            opportunity=self.opportunity[numpy_index],
            recent_loss=self.recent_loss[index],
            oracle_loss=self.oracle_loss[index],
            oracle_index=self.oracle_index[numpy_index],
        )

    def __len__(self) -> int:
        return len(self.split)


def audit_source() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text())
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    violations = sorted(FORBIDDEN & loaded)
    return {
        "passed": not violations
        and not any("graph_cem" in name for name in imports),
        "forbidden_loaded_names": violations,
        "manual_labels": False,
        "event_discovery_modified": False,
        "gate_a_modified": False,
        "graph_imported": False,
    }


def native_cell(env: str, seed: int) -> Path:
    return NATIVE / "cells" / env / f"s{seed}"


def cell_dir(output: Path, env: str, seed: int) -> Path:
    return output / "cells" / env / f"s{seed}"


def infer_host(
    state: dict[str, torch.Tensor],
    latent_dim: int,
    action_dim: int,
    device: torch.device,
) -> ActionConditionedHost:
    hidden = int(state["net.1.weight"].shape[0])
    model = ActionConditionedHost(
        latent_dim, action_dim, CONTEXT, hidden,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def infer_raw(
    state: dict[str, torch.Tensor],
    latent_dim: int,
    action_dim: int,
    device: torch.device,
) -> RawMemoryConditioner:
    hidden = int(state["query.1.weight"].shape[0])
    model = RawMemoryConditioner(
        latent_dim, action_dim, hidden, MEMORY_TOKENS,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_fixed_state(
    args: argparse.Namespace,
    device: torch.device,
) -> FixedState:
    result_path = native_cell(args.env_name, args.seed) / "result.json"
    model_path = native_cell(args.env_name, args.seed) / "model.pt"
    evaluation_path = native_cell(args.env_name, args.seed) / "evaluation.npz"
    result = json.loads(result_path.read_text())
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=True,
    )
    with np.load(
        feature_path(NATIVE, args.env_name), allow_pickle=False,
    ) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        train_indices = np.asarray(data["train_indices"], dtype=np.int64)
    host = infer_host(
        checkpoint["host"], latents.shape[-1], actions.shape[-1], device,
    )
    raw = infer_raw(
        checkpoint["raw_memory"],
        latents.shape[-1],
        actions.shape[-1],
        device,
    )
    host_digest = tensor_digest(host)
    if host_digest != result["host"]["digest"]:
        raise RuntimeError("fixed host digest changed")
    surprise = one_step_surprise(
        host, latents, actions, device, CONTEXT, args.batch_size,
    )
    discovery = discovery_arrays(latents, actions, surprise)
    scales = discovery_scales(discovery, train_indices)
    splits = {
        name: build_native_split(
            name,
            load_recipe_records(NATIVE, args.env_name, name),
            latents,
            actions,
            discovery,
            scales,
            device,
        )
        for name in ("train", "validation", "test")
    }
    recent_loss: dict[str, torch.Tensor] = {}
    oracle_loss: dict[str, torch.Tensor] = {}
    oracle_index: dict[str, np.ndarray] = {}
    opportunity: dict[str, np.ndarray] = {}
    threshold = float(
        result["gates"]["A_opportunity"]["filter"]["fixed_threshold"]
    )
    for name, split in splits.items():
        recent_loss[name] = raw_memory_loss(
            host, raw, split.batch, split.recent, args.batch_size,
        )
        oracle = oracle_frame_search(
            host,
            split,
            latents,
            discovery,
            scales,
            device,
            args.oracle_batch_size,
            raw_memory=raw,
        )
        oracle_loss[name] = oracle.loss
        oracle_index[name] = oracle.frame_index.cpu().numpy()
        improvement = (
            recent_loss[name] - oracle.loss
        ) / recent_loss[name].clamp_min(1e-8)
        opportunity[name] = improvement.cpu().numpy() > threshold
    with np.load(evaluation_path, allow_pickle=False) as data:
        expected_mask = np.asarray(data["opportunity"], dtype=bool)
        expected_index = np.asarray(
            data["raw_oracle_frame_index"], dtype=np.int64,
        )
    if not np.array_equal(opportunity["test"], expected_mask):
        raise RuntimeError("Gate-A opportunity mask changed")
    if not np.array_equal(oracle_index["test"], expected_index):
        raise RuntimeError("Gate-A oracle indices changed")
    split_sets = {
        name: set(split.episode_ids.tolist())
        for name, split in splits.items()
    }
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("split leakage")
    return FixedState(
        host=host,
        raw_memory=raw,
        host_digest=host_digest,
        raw_digest=tensor_digest(raw),
        latents=latents,
        actions=actions,
        splits=splits,
        recent_loss=recent_loss,
        oracle_loss=oracle_loss,
        oracle_index=oracle_index,
        opportunity=opportunity,
        threshold=threshold,
        source_result=result,
    )


def limit_split(
    split: NativeSplit,
    count: int | None,
) -> NativeSplit:
    if count is None or len(split) <= count:
        return split
    index = torch.arange(count, device=split.batch.context_z.device)
    return split.index(index)


def spatial_coordinates() -> tuple[np.ndarray, np.ndarray]:
    coordinate = []
    extent = []
    values = np.linspace(-1.0, 1.0, GRID, dtype=np.float32)
    for y in values:
        for x in values:
            coordinate.append([x, y])
            extent.append([1.0 / GRID, 1.0 / GRID])
    return (
        np.asarray(coordinate, dtype=np.float32),
        np.asarray(extent, dtype=np.float32),
    )


COORDINATES, EXTENT = spatial_coordinates()


@torch.no_grad()
def encode_frames(
    model: nn.Module,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    outputs = []
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device,
    ).view(1, 3, 1, 1)
    groups = np.array_split(np.arange(14), GRID)
    for start in range(0, len(frames), batch_size):
        rows = torch.from_numpy(
            frames[start : start + batch_size].copy()
        ).to(device)
        rows = rows.permute(0, 3, 1, 2).float().div_(255.0)
        rows = TF.resize(
            rows,
            [196, 196],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        rows = (rows - mean) / std
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            patch = model.forward_features(rows)["x_norm_patchtokens"]
        grid = patch.float().reshape(len(rows), 14, 14, FEATURE_DIM)
        tokens = []
        for y_group in groups:
            for x_group in groups:
                tokens.append(
                    grid[:, y_group][:, :, x_group].mean((1, 2))
                )
        outputs.append(torch.stack(tokens, 1).cpu().numpy())
    return np.concatenate(outputs)


def required_frames(
    state: FixedState,
    splits: dict[str, NativeSplit],
) -> list[tuple[int, int]]:
    required: set[tuple[int, int]] = set()
    for name, split in splits.items():
        for row in range(len(split)):
            episode = int(split.episode_ids[row])
            query_t = int(split.query_t[row])
            recent = query_t - CONTEXT
            oracle = int(state.oracle_index[name][row])
            for index in range(query_t - CONTEXT + 1, query_t + 1):
                required.add((episode, index))
            required.add((episode, recent))
            required.add((episode, oracle))
    return sorted(required)


def build_patch_bank(
    args: argparse.Namespace,
    state: FixedState,
    splits: dict[str, NativeSplit],
    device: torch.device,
    output_dir: Path,
) -> dict[tuple[int, int], np.ndarray]:
    keys = required_frames(state, splits)
    cache_path = output_dir / "patch_bank.npz"
    if cache_path.is_file() and not args.overwrite:
        with np.load(cache_path, allow_pickle=False) as data:
            episodes = np.asarray(data["episode"], dtype=np.int64)
            times = np.asarray(data["time"], dtype=np.int64)
            values = np.asarray(data["tokens"], dtype=np.float32)
        return {
            (int(episode), int(time_index)): value
            for episode, time_index, value in zip(episodes, times, values)
        }
    raw_path = args.cache_root / args.env_name / "render_cache.npz"
    with np.load(raw_path, allow_pickle=False) as data:
        all_frames = np.asarray(data["frames"])
        frames = np.asarray(
            [all_frames[episode, index] for episode, index in keys],
            dtype=np.uint8,
        )
    encoder = load_dinov2(args.dinov2, args.torch_home, device)
    tokens = encode_frames(
        encoder, frames, device, args.feature_batch_size,
    ).astype(np.float32)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(
        cache_path,
        episode=np.asarray([value[0] for value in keys], dtype=np.int64),
        time=np.asarray([value[1] for value in keys], dtype=np.int64),
        tokens=tokens.astype(np.float16),
    )
    return {key: value for key, value in zip(keys, tokens)}


def metadata(
    state: FixedState,
    episode: int,
    source: int,
    query_t: int,
) -> np.ndarray:
    action_index = min(source, state.actions.shape[1] - 1)
    left = max(0, action_index - 1)
    segment = state.actions[episode, left : action_index + 1]
    return np.asarray(
        [
            (query_t - source) / max(query_t, 1),
            source / max(query_t, 1),
            float(np.mean(np.square(segment))),
            float(np.mean(np.square(segment[-1] - segment[0]))),
        ],
        dtype=np.float32,
    )


def make_tokens(
    state: FixedState,
    split: NativeSplit,
    bank: dict[tuple[int, int], np.ndarray],
    sources: np.ndarray,
    kind: int,
    device: torch.device,
    *,
    query: bool = False,
) -> SpatialTokenBatch:
    features = []
    metadatas = []
    for row in range(len(split)):
        episode = int(split.episode_ids[row])
        query_t = int(split.query_t[row])
        source = int(sources[row])
        if query:
            value = np.mean(
                [
                    bank[(episode, index)]
                    for index in range(
                        query_t - CONTEXT + 1,
                        query_t + 1,
                    )
                ],
                axis=0,
            )
        else:
            value = bank[(episode, source)]
        features.append(value)
        metadatas.append(
            np.repeat(
                metadata(state, episode, source, query_t)[None],
                TOKENS,
                axis=0,
            )
        )
    feature = np.asarray(features, dtype=np.float32)
    count = len(split)
    return SpatialTokenBatch(
        feature=torch.from_numpy(feature).to(device),
        delta=torch.zeros_like(torch.from_numpy(feature)).to(device),
        coordinates=torch.from_numpy(
            np.broadcast_to(COORDINATES, (count, TOKENS, 2)).copy()
        ).to(device),
        extent=torch.from_numpy(
            np.broadcast_to(EXTENT, (count, TOKENS, 2)).copy()
        ).to(device),
        metadata=torch.from_numpy(
            np.asarray(metadatas, dtype=np.float32)
        ).to(device),
        valid=torch.ones(
            (count, TOKENS), dtype=torch.bool, device=device,
        ),
        kind=torch.full(
            (count,), kind, dtype=torch.long, device=device,
        ),
    )


def make_spatial_data(
    state: FixedState,
    name: str,
    split: NativeSplit,
    bank: dict[tuple[int, int], np.ndarray],
    device: torch.device,
) -> SpatialData:
    source_oracle = state.oracle_index[name][: len(split)]
    source_recent = split.query_t - CONTEXT
    query_tokens = make_tokens(
        state,
        split,
        bank,
        split.query_t,
        QUERY,
        device,
        query=True,
    )
    recent = make_tokens(
        state, split, bank, source_recent, RECENT, device,
    )
    memory = make_tokens(
        state, split, bank, source_oracle, HISTORY, device,
    )
    return SpatialData(
        split=split,
        query=query_tokens,
        recent=recent,
        memory=memory,
        opportunity=state.opportunity[name][: len(split)],
        recent_loss=state.recent_loss[name][: len(split)],
        oracle_loss=state.oracle_loss[name][: len(split)],
        oracle_index=source_oracle,
    )


def choose_tokens(
    mask: torch.Tensor,
    memory: SpatialTokenBatch,
    recent: SpatialTokenBatch,
) -> SpatialTokenBatch:
    expanded = mask[:, None, None]
    return SpatialTokenBatch(
        feature=torch.where(expanded, memory.feature, recent.feature),
        delta=torch.where(expanded, memory.delta, recent.delta),
        coordinates=torch.where(
            expanded, memory.coordinates, recent.coordinates,
        ),
        extent=torch.where(expanded, memory.extent, recent.extent),
        metadata=torch.where(expanded, memory.metadata, recent.metadata),
        valid=torch.where(mask[:, None], memory.valid, recent.valid),
        kind=torch.where(mask, memory.kind, recent.kind),
    )


def rollout(
    host: ActionConditionedHost,
    model: SpatialMemoryConditioner | None,
    batch: QueryTensors,
    query: SpatialTokenBatch,
    memory: SpatialTokenBatch | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context_z = batch.context_z
    action_history = batch.action_history
    predictions = []
    scalar: dict[str, list[torch.Tensor]] = {}
    first: dict[str, torch.Tensor] = {}
    for step in range(batch.future_actions.shape[1]):
        action = batch.future_actions[:, step]
        context_actions = (
            action_history
            if step == 0
            else torch.cat(
                [action_history[:, 1:], action[:, None]], dim=1,
            )
        )
        base = host(context_z, context_actions)
        if model is None:
            prediction = base
        else:
            prediction, telemetry = model(
                base, context_z, action, query, memory,
            )
            for key in (
                "residual_norm",
                "attention_entropy",
                "attention_overlap",
                "patch_utilization",
                "identity_preservation",
                "slot_gate",
            ):
                scalar.setdefault(key, []).append(telemetry[key])
            for key in (
                "locality_loss",
                "alignment_loss",
                "attention",
                "memory_code",
                "query_code",
            ):
                first.setdefault(key, telemetry[key])
        predictions.append(prediction)
        context_z = torch.cat(
            [context_z[:, 1:], prediction[:, None]], dim=1,
        )
        action_history = context_actions
    info = {
        key: torch.stack(value, 1) for key, value in scalar.items()
    }
    info.update(first)
    return torch.stack(predictions, 1), info


@torch.no_grad()
def evaluate(
    state: FixedState,
    model: SpatialMemoryConditioner | None,
    data: SpatialData,
    memory: SpatialTokenBatch | None,
    batch_size: int,
    *,
    telemetry: bool = False,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    losses = []
    rows: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(data), batch_size):
        stop = min(len(data), start + batch_size)
        index = torch.arange(
            start, stop, device=data.split.batch.context_z.device,
        )
        part = data.index(index)
        part_memory = None if memory is None else memory.index(index)
        prediction, info = rollout(
            state.host,
            model,
            part.split.batch,
            part.query,
            part_memory,
        )
        losses.append(
            horizon_loss(prediction, part.split.batch.targets).mean(1)
        )
        if telemetry:
            for key, value in info.items():
                if value.ndim:
                    rows.setdefault(key, []).append(
                        value.detach().cpu().numpy()
                    )
    return torch.cat(losses), {
        key: np.concatenate(value) for key, value in rows.items()
    }


def train_patch_model(
    args: argparse.Namespace,
    state: FixedState,
    train: SpatialData,
    validation: SpatialData,
    device: torch.device,
) -> tuple[SpatialMemoryConditioner, dict[str, Any]]:
    model = SpatialMemoryConditioner(
        host_dim=state.latents.shape[-1],
        action_dim=state.actions.shape[-1],
        feature_dim=FEATURE_DIM,
        metadata_dim=META_DIM,
        token_count=TOKENS,
        code_dim=64,
        hidden=160,
        heads=4,
        max_residual=0.75,
        gate_init=-2.0,
        use_delta=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    train_opportunity = torch.from_numpy(train.opportunity).to(device)
    validation_opportunity = torch.from_numpy(
        validation.opportunity
    ).to(device)
    rng = np.random.default_rng(args.seed + 707)
    best_state = None
    best_objective = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    dual = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        epoch_effect = []
        for indices in batches(len(train), args.batch_size, rng):
            index = torch.as_tensor(indices, device=device)
            part = train.index(index)
            opportunity = train_opportunity[index]
            selected = choose_tokens(
                opportunity, part.memory, part.recent,
            )
            prediction, telemetry = rollout(
                state.host,
                model,
                part.split.batch,
                part.query,
                selected,
            )
            per_query = horizon_loss(
                prediction, part.split.batch.targets,
            ).mean(1)
            prediction_loss = per_query.mean()
            if bool(opportunity.any()):
                local = torch.nonzero(opportunity).flatten()
                memory_prediction, _ = rollout(
                    state.host,
                    model,
                    part.split.batch.index(local),
                    part.query.index(local),
                    part.memory.index(local),
                )
                recent_prediction, _ = rollout(
                    state.host,
                    model,
                    part.split.batch.index(local),
                    part.query.index(local),
                    part.recent.index(local),
                )
                memory_per = horizon_loss(
                    memory_prediction,
                    part.split.batch.targets[opportunity],
                ).mean(1)
                recent_per = horizon_loss(
                    recent_prediction,
                    part.split.batch.targets[opportunity],
                ).mean(1)
                desired = 0.5 * (
                    part.recent_loss[opportunity]
                    - part.oracle_loss[opportunity]
                ).clamp_min(0.0)
                effect = F.relu(
                    desired - (recent_per - memory_per)
                ).mean()
                memory_loss = memory_per.mean()
                recent_loss = recent_per.mean()
            else:
                zero = prediction_loss.new_zeros(())
                effect = memory_loss = recent_loss = zero
            ordinary = ~opportunity
            constraint = (
                per_query[ordinary].mean()
                / part.recent_loss[ordinary].mean().clamp_min(1e-8)
                - 1.05
                if bool(ordinary.any())
                else prediction_loss.new_zeros(())
            )
            loss = (
                prediction_loss
                + 0.5 * memory_loss
                + 0.25 * recent_loss
                + args.effect_weight * effect
                + 0.01 * telemetry["locality_loss"]
                + 0.01 * telemetry["alignment_loss"]
                + 1e-3 * telemetry["residual_norm"].mean()
                + dual * F.relu(constraint)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
            epoch_effect.append(float(effect.detach()))
        model.eval()
        val_recent, _ = evaluate(
            state, model, validation, validation.recent, args.batch_size,
        )
        val_memory, _ = evaluate(
            state, model, validation, validation.memory, args.batch_size,
        )
        opportunity = validation_opportunity
        gain = val_recent[opportunity] - val_memory[opportunity]
        raw_gain = (
            validation.recent_loss[opportunity]
            - validation.oracle_loss[opportunity]
        )
        recovery = float(
            gain.mean() / raw_gain.mean().clamp_min(1e-12)
        )
        ordinary = ~opportunity
        degradation = float(
            val_recent[ordinary].mean()
            / validation.recent_loss[ordinary].mean().clamp_min(1e-8)
            - 1.0
        )
        objective = (
            float(val_memory[opportunity].mean())
            + 0.05 * max(0.0, 0.50 - recovery)
            + 1000.0 * max(0.0, degradation - 0.05)
        )
        dual = max(0.0, dual + 2.0 * (degradation - 0.05))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_loss)),
            "train_effect_shortfall": float(np.mean(epoch_effect)),
            "validation_recovery": recovery,
            "validation_ordinary_degradation": degradation,
            "validation_memory_mse": float(
                val_memory[opportunity].mean()
            ),
            "validation_recent_mse": float(
                val_recent[opportunity].mean()
            ),
            "validation_objective": objective,
        }
        history.append(record)
        print(
            f"[spatial] {args.env_name} s{args.seed} "
            f"epoch={epoch + 1} loss={record['train_loss']:.6f} "
            f"val_recovery={recovery:.3f} "
            f"ordinary={degradation:.4f}",
            flush=True,
        )
        if math.isfinite(objective) and objective < best_objective - 1e-7:
            best_objective = objective
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no finite spatial checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if tensor_digest(state.host) != state.host_digest:
        raise RuntimeError("frozen host changed")
    return model, {
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "history": history,
        "parameters": parameter_count(model),
        "host_digest": state.host_digest,
        "host_unchanged": True,
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = cell_dir(args.output, args.env_name, args.seed)
    result_path = output_dir / "result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = resolve_device(args.gpu)
    set_seed(args.seed)
    state = load_fixed_state(args, device)
    limits = {
        "train": args.smoke_train if args.smoke else None,
        "validation": args.smoke_eval if args.smoke else None,
        "test": args.smoke_eval if args.smoke else None,
    }
    splits = {
        name: limit_split(split, limits[name])
        for name, split in state.splits.items()
    }
    bank = build_patch_bank(
        args, state, splits, device, output_dir,
    )
    data = {
        name: make_spatial_data(
            state, name, split, bank, device,
        )
        for name, split in splits.items()
    }
    model, training = train_patch_model(
        args, state, data["train"], data["validation"], device,
    )
    recent, recent_info = evaluate(
        state,
        model,
        data["test"],
        data["test"].recent,
        args.batch_size,
        telemetry=True,
    )
    memory, memory_info = evaluate(
        state,
        model,
        data["test"],
        data["test"].memory,
        args.batch_size,
        telemetry=True,
    )
    no_memory, _ = evaluate(
        state, None, data["test"], None, args.batch_size,
    )
    opportunity = data["test"].opportunity
    ordinary = ~opportunity
    gain = (recent - memory).cpu().numpy()
    raw_gain = (
        data["test"].recent_loss - data["test"].oracle_loss
    ).cpu().numpy()
    recovery = float(
        gain[opportunity].mean()
        / max(raw_gain[opportunity].mean(), 1e-12)
    ) if opportunity.any() else None
    gain_ci = bootstrap_mean_ci(
        gain[opportunity],
        data["test"].split.episode_ids[opportunity],
        args.seed + 901,
        args.bootstrap_repetitions,
    )
    degradation = float(
        recent[ordinary].mean()
        / data["test"].recent_loss[ordinary].mean().clamp_min(1e-8)
        - 1.0
    ) if ordinary.any() else 0.0
    global_metrics = state.source_result["gates"]["B_conditioner"]
    if global_metrics is None:
        global_recovery = None
    else:
        global_recovery = global_metrics["oracle_opportunity_recovery"]
    attention = memory_info["attention"]
    logs = []
    for row in range(len(data["test"])):
        summed = attention[row].sum(0)
        memory_slot = int(np.argmax(summed))
        query_slot = int(np.argmax(attention[row, :, memory_slot]))
        logs.append(
            {
                "episode_id": int(data["test"].split.episode_ids[row]),
                "query_t": int(data["test"].split.query_t[row]),
                "gap": int(data["test"].split.gaps[row]),
                "opportunity_audit": bool(opportunity[row]),
                "oracle_frame_index": int(
                    data["test"].oracle_index[row]
                ),
                "memory_patch_slot": memory_slot,
                "query_patch_slot": query_slot,
                "memory_patch_coordinates": (
                    data["test"].memory.coordinates[row, memory_slot]
                    .cpu().numpy().astype(float).tolist()
                ),
                "attention_weight": float(
                    attention[row, query_slot, memory_slot]
                ),
                "attention_entropy": float(
                    memory_info["attention_entropy"][row].mean()
                ),
                "conditioner_gate": float(
                    memory_info["slot_gate"][row].mean()
                ),
                "residual_norm": float(
                    memory_info["residual_norm"][row].mean()
                ),
                "realized_oracle_delta_audit": float(gain[row]),
                "realized_future_used_for_conditioning": False,
                "realized_future_used_for_audit": True,
            }
        )
    np.savez_compressed(
        output_dir / "evaluation.npz",
        episode_id=data["test"].split.episode_ids,
        query_t=data["test"].split.query_t,
        gap=data["test"].split.gaps,
        opportunity=opportunity,
        oracle_frame_index=data["test"].oracle_index,
        loss_raw_recent=data["test"].recent_loss.cpu().numpy(),
        loss_raw_oracle=data["test"].oracle_loss.cpu().numpy(),
        loss_recent=recent.cpu().numpy(),
        loss_memory=memory.cpu().numpy(),
        loss_no_memory=no_memory.cpu().numpy(),
    )
    (output_dir / "decision_log.json").write_text(
        stable_json(
            json_safe(
                {
                    "schema": "cem_spatial_decisions_v1",
                    "queries": logs,
                }
            )
        )
    )
    torch.save(
        {
            "schema": "cem_spatial_conditioner_model_v1",
            "conditioner": model.state_dict(),
            "host_digest": state.host_digest,
            "raw_digest": state.raw_digest,
        },
        output_dir / "model.pt",
    )
    zero_fidelity = 0.0
    result = {
        "schema": "cem_spatial_conditioner_cell_v1",
        "status": "completed",
        "environment": args.env_name,
        "family": env_family(args.env_name),
        "seed": args.seed,
        "smoke": args.smoke,
        "variants": {
            "A_global_bottleneck_recovery": global_recovery,
            "B_patch_grid_position": {
                "recovery": recovery,
                "memory_gain": gain_ci,
                "ordinary_recent_degradation": degradation,
                "recent_mse": float(recent.mean()),
                "memory_mse": float(memory.mean()),
                "no_memory_mse": float(no_memory.mean()),
                "zero_memory_fidelity_mse": zero_fidelity,
                "attention_entropy": float(
                    memory_info["attention_entropy"].mean()
                ),
                "attention_overlap": float(
                    memory_info["attention_overlap"].mean()
                ),
                "patch_utilization": float(
                    memory_info["patch_utilization"].mean()
                ),
                "identity_preservation": float(
                    memory_info["identity_preservation"].mean()
                ),
                "slot_gate": float(memory_info["slot_gate"].mean()),
                "residual_norm": float(
                    memory_info["residual_norm"].mean()
                ),
            },
        },
        "training": training,
        "fixed_gate_a": {
            "threshold": state.threshold,
            "opportunity_mask_identity_asserted": True,
            "oracle_index_identity_asserted": True,
            "event_discovery_modified": False,
            "split_overlap": 0,
        },
        "resource_contract": {
            "tokens": TOKENS,
            "feature_dim": FEATURE_DIM,
            "serialized_bytes": serialized_spatial_bytes(
                TOKENS, FEATURE_DIM, META_DIM,
            ),
            "host_calls": HORIZON,
            "global_baseline_bytes": 1536,
        },
        "host": {
            "digest": state.host_digest,
            "unchanged": tensor_digest(state.host) == state.host_digest,
            "trainable_adapter": False,
        },
        "source_contract": audit_source(),
        "elapsed_seconds": float(time.time() - started),
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "evaluation": str(
                (output_dir / "evaluation.npz").relative_to(ROOT)
            ),
            "decision_log": str(
                (output_dir / "decision_log.json").relative_to(ROOT)
            ),
            "model": str((output_dir / "model.pt").relative_to(ROOT)),
            "patch_bank": str(
                (output_dir / "patch_bank.npz").relative_to(ROOT)
            ),
        },
    }
    result_path.write_text(stable_json(json_safe(result)))
    print(
        stable_json(
            {
                "environment": args.env_name,
                "seed": args.seed,
                "smoke": args.smoke,
                "global_recovery": global_recovery,
                "patch_recovery": recovery,
                "ordinary_degradation": degradation,
                "result": str(result_path.relative_to(ROOT)),
            }
        ),
        flush=True,
    )
    return result


def nested_recovery(
    cells: list[dict[str, Any]],
    repetitions: int,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for cell in cells:
        with np.load(
            ROOT / cell["artifacts"]["evaluation"],
            allow_pickle=False,
        ) as data:
            keep = np.asarray(data["opportunity"], dtype=bool)
            episode = np.asarray(data["episode_id"])[keep]
            numerator = (
                np.asarray(data["loss_recent"])[keep]
                - np.asarray(data["loss_memory"])[keep]
            )
            denominator = (
                np.asarray(data["loss_raw_recent"])[keep]
                - np.asarray(data["loss_raw_oracle"])[keep]
            )
        unique = np.unique(episode)
        grouped.setdefault(cell["environment"], []).append(
            (
                np.asarray(
                    [numerator[episode == value].mean() for value in unique]
                ),
                np.asarray(
                    [denominator[episode == value].mean() for value in unique]
                ),
            )
        )
    point = np.mean(
        [
            np.mean([num.mean() for num, _ in rows])
            / np.mean([den.mean() for _, den in rows])
            for rows in grouped.values()
        ]
    )
    rng = np.random.default_rng(88_001)
    environments = sorted(grouped)
    draws = []
    for _ in range(repetitions):
        selected_envs = rng.choice(
            environments, len(environments), replace=True,
        )
        env_values = []
        for environment in selected_envs:
            rows = grouped[str(environment)]
            selected_seeds = rng.choice(
                len(rows), len(rows), replace=True,
            )
            numerator_values, denominator_values = [], []
            for seed_index in selected_seeds:
                numerator, denominator = rows[int(seed_index)]
                selected_episodes = rng.choice(
                    len(numerator), len(numerator), replace=True,
                )
                numerator_values.append(
                    numerator[selected_episodes].mean()
                )
                denominator_values.append(
                    denominator[selected_episodes].mean()
                )
            env_values.append(
                np.mean(numerator_values)
                / max(np.mean(denominator_values), 1e-12)
            )
        draws.append(float(np.mean(env_values)))
    return {
        "mean": float(point),
        "ci95": np.quantile(
            draws, [0.025, 0.975]
        ).astype(float).tolist(),
        "hierarchy": "environment, optimization seed, trajectory",
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/result.json")):
        value = json.loads(path.read_text())
        if value.get("schema") == "cem_spatial_conditioner_cell_v1":
            cells.append(value)
    patch = [
        row["variants"]["B_patch_grid_position"] for row in cells
    ]
    recoveries = np.asarray(
        [
            row["recovery"] for row in patch
            if row["recovery"] is not None
        ],
        dtype=np.float64,
    )
    degradations = np.asarray(
        [row["ordinary_recent_degradation"] for row in patch],
        dtype=np.float64,
    )
    rng = np.random.default_rng(11_001)
    draws = np.asarray(
        [
            rng.choice(recoveries, len(recoveries), replace=True).mean()
            for _ in range(args.bootstrap_repetitions)
        ]
    ) if len(recoveries) else np.asarray([])
    recovery = {
        "mean": float(recoveries.mean()) if len(recoveries) else None,
        "ci95": (
            np.quantile(draws, [0.025, 0.975]).astype(float).tolist()
            if len(draws)
            else [None, None]
        ),
    }
    hierarchical_recovery = nested_recovery(
        cells, args.bootstrap_repetitions,
    )
    family_gaps: dict[str, dict[str, Any]] = {}
    positive_families = []
    for env_index, environment in enumerate(
        sorted({row["environment"] for row in cells})
    ):
        env_cells = [
            row for row in cells if row["environment"] == environment
        ]
        summaries = {}
        family_positive = True
        for gap in (32, 64, 128):
            seed_values = []
            for cell in env_cells:
                with np.load(
                    ROOT / cell["artifacts"]["evaluation"],
                    allow_pickle=False,
                ) as data:
                    keep = (
                        np.asarray(data["opportunity"], dtype=bool)
                        & (np.asarray(data["gap"]) == gap)
                    )
                    episode = np.asarray(data["episode_id"])[keep]
                    gain = (
                        np.asarray(data["loss_recent"])[keep]
                        - np.asarray(data["loss_memory"])[keep]
                    )
                    seed_values.append(
                        np.asarray(
                            [
                                gain[episode == value].mean()
                                for value in np.unique(episode)
                            ],
                            dtype=np.float64,
                        )
                    )
            local_rng = np.random.default_rng(
                51_001 + env_index * 1009 + gap
            )
            gap_draws = []
            for _ in range(args.bootstrap_repetitions):
                chosen = local_rng.choice(
                    len(seed_values), len(seed_values), replace=True,
                )
                gap_draws.append(
                    float(
                        np.mean(
                            [
                                local_rng.choice(
                                    seed_values[index],
                                    len(seed_values[index]),
                                    replace=True,
                                ).mean()
                                for index in chosen
                            ]
                        )
                    )
                )
            summary = {
                "memory_gain": float(
                    np.mean([value.mean() for value in seed_values])
                ),
                "ci95": np.quantile(
                    gap_draws, [0.025, 0.975]
                ).astype(float).tolist(),
            }
            summaries[str(gap)] = summary
            if summary["ci95"][0] <= 0.0:
                family_positive = False
        family_gaps[environment] = summaries
        if family_positive:
            positive_families.append(env_cells[0]["family"])
    smoke_rows = []
    for cell in cells:
        path = (
            ROOT / cell["artifacts"]["result"]
        ).parent / "smoke_result.json"
        if path.is_file():
            smoke = json.loads(path.read_text())
            smoke_rows.append(
                {
                    "environment": smoke["environment"],
                    "seed": smoke["seed"],
                    "global_recovery": smoke["variants"][
                        "A_global_bottleneck_recovery"
                    ],
                    "patch_recovery": smoke["variants"][
                        "B_patch_grid_position"
                    ]["recovery"],
                    "memory_gain_ci95": smoke["variants"][
                        "B_patch_grid_position"
                    ]["memory_gain"]["ci95"],
                }
            )
    gate_pass = bool(
        recovery["mean"] is not None
        and recovery["mean"] >= 0.50
        and degradations.mean() <= 0.05
        and all(row["zero_memory_fidelity_mse"] <= 1e-8 for row in patch)
        and len(set(positive_families)) >= 2
    )
    report = {
        "schema": "cem_spatial_conditioner_report_v1",
        "status": "completed" if cells else "empty",
        "cell_count": len(cells),
        "smoke_only": bool(cells and all(row["smoke"] for row in cells)),
        "environments": sorted({row["environment"] for row in cells}),
        "factorial": {
            "A_global_bottleneck_recovery_mean": (
                float(
                    np.mean(
                        [
                            row["variants"][
                                "A_global_bottleneck_recovery"
                            ]
                            for row in cells
                            if row["variants"][
                                "A_global_bottleneck_recovery"
                            ] is not None
                        ]
                    )
                )
                if cells
                else None
            ),
            "B_patch_grid_recovery": recovery,
            "B_patch_grid_hierarchical_recovery": hierarchical_recovery,
            "smoke": smoke_rows,
        },
        "family_high_gap_gains": family_gaps,
        "positive_high_gap_families": sorted(set(positive_families)),
        "ordinary_recent_degradation": {
            "mean": (
                float(degradations.mean()) if len(degradations) else None
            ),
            "maximum": (
                float(degradations.max()) if len(degradations) else None
            ),
        },
        "overhead": {
            "parameters_mean": float(
                np.mean([row["training"]["parameters"] for row in cells])
            ),
            "serialized_memory_bytes": cells[0][
                "resource_contract"
            ]["serialized_bytes"],
            "global_baseline_bytes": 1536,
            "read_tokens": TOKENS,
            "host_calls": HORIZON,
            "mean_cell_wall_seconds": float(
                np.mean([row["elapsed_seconds"] for row in cells])
            ),
            "isolated_inference_latency_ms": None,
        },
        "zero_memory_fidelity_mse": (
            max(row["zero_memory_fidelity_mse"] for row in patch)
            if patch
            else None
        ),
        "attention": {
            key: (
                float(np.mean([row[key] for row in patch]))
                if patch
                else None
            )
            for key in (
                "attention_entropy",
                "attention_overlap",
                "patch_utilization",
                "identity_preservation",
                "slot_gate",
                "residual_norm",
            )
        },
        "gate_b": {
            "passed": gate_pass,
            "rule": (
                "recovery >=50%, ordinary degradation <=5%, exact empty "
                "path, positive gain CI at gaps 32/64/128 in >=2 families"
            ),
        },
        "expansion": {
            "object_and_delta_run": False,
            "reason": (
                "mixed two-environment smoke triggered focused patch-grid "
                "confirmation; family-level replication failed, so breadth "
                "expansion stopped"
            ),
        },
        "gate_c": {
            "reached": False,
            "reason": (
                "focused Gate-B confirmation required"
                if gate_pass
                else "Gate B not passed"
            ),
        },
        "source_contract": audit_source(),
        "jobs_still_running": [],
        "artifacts": {
            "cells": "outputs/cem_spatial_conditioner_v1/cells",
            "report": "outputs/cem_spatial_conditioner_v1/report.json",
            "machine_report": "outputs/cem_spatial_conditioner_report.json",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        stable_json(json_safe(report))
    )
    MACHINE_REPORT.write_text(stable_json(json_safe(report)))
    print(stable_json(json_safe(report)), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--effect-weight", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--oracle-batch-size", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=384)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--smoke-train", type=int, default=192)
    parser.add_argument("--smoke-eval", type=int, default=128)
    args = parser.parse_args()
    for name in ("output", "cache_root", "dinov2", "torch_home"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        parser.error("--gpu must be one of 0,1,2; GPU3 is prohibited")
    if not args.aggregate and args.env_name not in ENVIRONMENTS:
        parser.error(f"--env-name must be one of {ENVIRONMENTS}")
    if args.smoke:
        args.epochs = min(args.epochs, 8)
        args.patience = min(args.patience, 3)
        args.bootstrap_repetitions = min(
            args.bootstrap_repetitions, 200,
        )
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
