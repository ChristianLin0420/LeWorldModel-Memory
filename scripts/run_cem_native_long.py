#!/usr/bin/env python3
"""Native long-trajectory conditioner and conservative memory activation gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.native_long_memory import (  # noqa: E402
    EVENT,
    FRAME,
    RECENT,
    TOKEN_TYPE_NAMES,
    MemoryTokenBatch,
    NativeMemoryConditioner,
    UtilityGateEnsemble,
    geometry_regularization,
    parameter_count,
    variance_regularization,
)
from scripts.build_cem_native_long import (  # noqa: E402
    CONTEXT,
    ENVIRONMENTS,
    GAPS,
    HORIZON,
    RECENT_TOKENS,
    feature_path,
    recipe_path,
    source_audit,
)
from scripts.run_cem_raw_ogbench import (  # noqa: E402
    ActionConditionedHost,
    QueryTensors,
    RawMemoryConditioner,
    batches,
    env_family,
    horizon_loss,
    json_safe,
    one_step_surprise,
    rollout,
    set_seed,
    stable_json,
    tensor_digest,
    train_host,
    train_memory,
)


DEFAULT_OUTPUT = ROOT / "outputs/cem_native_long_v1"
ROOT_REPORT = ROOT / "outputs/cem_native_long_report.json"
POOL_SLOTS = 12
MEMORY_TOKENS = 4
METADATA_DIM = 7
GATE_FOLDS = 3


@dataclass
class NativeSplit:
    name: str
    batch: QueryTensors
    recent: MemoryTokenBatch
    robust: MemoryTokenBatch
    pool: MemoryTokenBatch
    source_index: torch.Tensor
    proposal_score: torch.Tensor
    robust_slot: torch.Tensor
    episode_ids: np.ndarray
    query_t: np.ndarray
    gaps: np.ndarray
    query_signals: np.ndarray

    def index(self, index: torch.Tensor) -> "NativeSplit":
        numpy_index = index.detach().cpu().numpy()
        return NativeSplit(
            name=self.name,
            batch=self.batch.index(index),
            recent=self.recent.index(index),
            robust=self.robust.index(index),
            pool=self.pool.index(index),
            source_index=self.source_index[index],
            proposal_score=self.proposal_score[index],
            robust_slot=self.robust_slot[index],
            episode_ids=self.episode_ids[numpy_index],
            query_t=self.query_t[numpy_index],
            gaps=self.gaps[numpy_index],
            query_signals=self.query_signals[numpy_index],
        )

    def __len__(self) -> int:
        return len(self.batch)


@dataclass
class OracleResult:
    loss: torch.Tensor
    frame_index: torch.Tensor
    memory: MemoryTokenBatch


def cell_dir(output: Path, env_name: str, seed: int) -> Path:
    return output / "cells" / env_name / f"s{seed}"


def resolve_device(gpu: int) -> torch.device:
    if gpu == 3:
        raise ValueError("GPU3 is prohibited for this campaign")
    if gpu not in (0, 1, 2):
        raise ValueError(f"GPU must be one of 0,1,2; received {gpu}")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def concatenate_query_tensors(values: list[QueryTensors]) -> QueryTensors:
    return QueryTensors(
        **{
            field: torch.cat([getattr(value, field) for value in values])
            for field in QueryTensors.__dataclass_fields__
        }
    )


def tokens_to_queries(
    batch: QueryTensors,
    memory: MemoryTokenBatch,
) -> QueryTensors:
    return QueryTensors(
        context_z=batch.context_z,
        action_history=batch.action_history,
        future_actions=batch.future_actions,
        targets=batch.targets,
        events=memory.values,
        metadata=memory.metadata[..., :3],
        valid=memory.valid,
        recent_event=memory.values[:, :1],
    )


def choose_memory(
    mask: torch.Tensor,
    when_true: MemoryTokenBatch,
    when_false: MemoryTokenBatch,
) -> MemoryTokenBatch:
    if when_true.values.shape != when_false.values.shape:
        raise ValueError("memory alternatives must have identical shapes")
    expanded = mask[:, None, None]
    return MemoryTokenBatch(
        values=torch.where(expanded, when_true.values, when_false.values),
        metadata=torch.where(
            expanded,
            when_true.metadata,
            when_false.metadata,
        ),
        token_type=torch.where(
            mask[:, None],
            when_true.token_type,
            when_false.token_type,
        ),
        valid=torch.where(
            mask[:, None],
            when_true.valid,
            when_false.valid,
        ),
    )


def _cosine(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return (values @ query) / np.maximum(
        np.linalg.norm(values, axis=1) * np.linalg.norm(query),
        1e-8,
    )


def _kcenter(
    values: np.ndarray,
    available: list[int],
    selected: list[int],
    limit: int,
) -> list[int]:
    output = list(selected)
    candidates = [value for value in available if value not in output]
    if not output and candidates:
        center = values[candidates].mean(0, keepdims=True)
        distance = np.mean(np.square(values[candidates] - center), axis=1)
        output.append(candidates[int(np.argmax(distance))])
        candidates = [value for value in candidates if value not in output]
    while candidates and len(output) < limit:
        distance = np.min(
            np.mean(
                np.square(
                    values[candidates, None] - values[np.asarray(output)][None]
                ),
                axis=-1,
            ),
            axis=1,
        )
        chosen = candidates[int(np.argmax(distance))]
        output.append(chosen)
        candidates.remove(chosen)
    return output


def discovery_arrays(
    latents: np.ndarray,
    actions: np.ndarray,
    surprise: np.ndarray,
) -> dict[str, np.ndarray]:
    change = np.zeros(latents.shape[:2], dtype=np.float32)
    change[:, 1:] = np.mean(
        np.square(latents[:, 1:] - latents[:, :-1]),
        axis=-1,
    )
    action_structure = np.zeros(latents.shape[:2], dtype=np.float32)
    action_structure[:, 1 : actions.shape[1]] = np.mean(
        np.square(actions[:, 1:] - actions[:, :-1]),
        axis=-1,
    )
    action_structure[:, : actions.shape[1]] += 0.25 * np.mean(
        np.square(actions),
        axis=-1,
    )
    return {
        "surprise": np.nan_to_num(surprise, nan=0.0),
        "change": change,
        "action_structure": action_structure,
    }


def discovery_scales(
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
) -> dict[str, float]:
    return {
        key: max(1e-8, float(np.quantile(value[train_indices], 0.75)))
        for key, value in arrays.items()
    }


def _frame_metadata(
    episode: int,
    source_index: int,
    query_t: int,
    current: np.ndarray,
    recent_mean: np.ndarray,
    latents: np.ndarray,
    discovery: dict[str, np.ndarray],
    scales: dict[str, float],
) -> np.ndarray:
    value = latents[episode, source_index]
    similarity = float(_cosine(value[None], current)[0])
    middle_stop = max(source_index + 1, query_t - CONTEXT)
    middle = latents[episode, source_index + 1 : middle_stop]
    middle_similarity = (
        float(_cosine(middle, current).mean()) if len(middle) else similarity
    )
    reappearance = similarity - middle_similarity
    proposal = (
        0.30 * (similarity + 1.0) / 2.0
        + 0.20
        * discovery["surprise"][episode, source_index]
        / scales["surprise"]
        + 0.20
        * discovery["change"][episode, source_index]
        / scales["change"]
        + 0.10
        * discovery["action_structure"][episode, source_index]
        / scales["action_structure"]
        + 0.10 * max(0.0, reappearance)
        + 0.10
        * float(np.mean(np.square(value - recent_mean)))
        / max(scales["change"], 1e-8)
    )
    return np.asarray(
        [
            (query_t - source_index) / max(1, query_t),
            math.log1p(
                max(0.0, float(discovery["surprise"][episode, source_index]))
            ),
            math.log1p(
                max(0.0, float(discovery["change"][episode, source_index]))
            ),
            math.log1p(
                max(
                    0.0,
                    float(
                        discovery["action_structure"][episode, source_index]
                    ),
                )
            ),
            similarity,
            reappearance,
            proposal,
        ],
        dtype=np.float32,
    )


def _mine_query_candidates(
    latents: np.ndarray,
    episode: int,
    query_t: int,
    gap: int,
    discovery: dict[str, np.ndarray],
    scales: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_stop = query_t - gap
    old_indices = np.arange(old_stop + 1, dtype=np.int64)
    current = latents[
        episode,
        query_t - CONTEXT + 1 : query_t + 1,
    ].mean(0)
    recent_stop = query_t - CONTEXT
    recent_start = recent_stop - RECENT_TOKENS + 1
    recent_mean = latents[episode, recent_start : recent_stop + 1].mean(0)
    similarity = _cosine(latents[episode, old_indices], current)
    reappearance = np.zeros_like(similarity)
    for offset, source in enumerate(old_indices):
        middle = latents[episode, source + 1 : recent_stop]
        reappearance[offset] = similarity[offset] - (
            float(_cosine(middle, current).mean())
            if len(middle)
            else similarity[offset]
        )
    surprise = discovery["surprise"][episode, old_indices]
    change = discovery["change"][episode, old_indices]
    action_structure = discovery["action_structure"][episode, old_indices]
    recent_distance = np.mean(
        np.square(latents[episode, old_indices] - recent_mean),
        axis=-1,
    )
    eventness = np.maximum(
        surprise / scales["surprise"],
        change / scales["change"],
    )
    interaction = np.sqrt(
        np.maximum(0.0, action_structure / scales["action_structure"])
        * np.maximum(0.0, change / scales["change"])
    )
    definitions = [
        (int(old_indices[np.argmax(similarity)]), FRAME),
        (int(old_indices[np.argmax(reappearance)]), FRAME),
        (int(old_indices[np.argmax(eventness)]), EVENT),
        (int(old_indices[np.argmax(interaction)]), EVENT),
        (int(old_indices[np.argmax(recent_distance)]), EVENT),
        (int(old_indices[np.argmax(surprise)]), EVENT),
    ]
    selected_frame_indices = [
        index for index, type_id in definitions if type_id == FRAME
    ]
    coverage = _kcenter(
        latents[episode],
        old_indices.astype(int).tolist(),
        selected_frame_indices,
        POOL_SLOTS,
    )
    for index in coverage:
        if len(definitions) >= POOL_SLOTS:
            break
        if (index, FRAME) not in definitions:
            definitions.append((index, FRAME))
    values = np.zeros((POOL_SLOTS, latents.shape[-1]), dtype=np.float32)
    metadata = np.zeros((POOL_SLOTS, METADATA_DIM), dtype=np.float32)
    token_type = np.zeros(POOL_SLOTS, dtype=np.int64)
    valid = np.zeros(POOL_SLOTS, dtype=bool)
    source = np.zeros(POOL_SLOTS, dtype=np.int64)
    for slot, (index, type_id) in enumerate(definitions[:POOL_SLOTS]):
        if type_id == EVENT:
            left, right = max(0, index - 1), min(old_stop + 1, index + 2)
            values[slot] = latents[episode, left:right].mean(0)
        else:
            values[slot] = latents[episode, index]
        metadata[slot] = _frame_metadata(
            episode,
            index,
            query_t,
            current,
            recent_mean,
            latents,
            discovery,
            scales,
        )
        if type_id == EVENT:
            metadata[slot, 6] += 0.05 * eventness[index]
        token_type[slot] = type_id
        valid[slot] = True
        source[slot] = index
    score = metadata[:, 6].copy()
    score[~valid] = -np.inf
    return values, metadata, token_type, valid, source


def load_recipe_records(
    output: Path,
    env_name: str,
    split: str,
) -> dict[str, np.ndarray]:
    path = recipe_path(output, env_name)
    with np.load(path, allow_pickle=False) as data:
        rows: dict[str, list[np.ndarray]] = {}
        for gap in GAPS:
            prefix = f"{split}_g{gap}_"
            for key in (
                "episode_id",
                "query_t",
                "gap",
                "revisit",
                "reappearance",
                "transition",
                "action_structure",
                "region_change",
                "proposal_score",
            ):
                rows.setdefault(key, []).append(np.asarray(data[prefix + key]))
    return {key: np.concatenate(values) for key, values in rows.items()}


def build_native_split(
    name: str,
    records: dict[str, np.ndarray],
    latents: np.ndarray,
    actions: np.ndarray,
    discovery: dict[str, np.ndarray],
    scales: dict[str, float],
    device: torch.device,
) -> NativeSplit:
    count = len(records["episode_id"])
    latent_dim = latents.shape[-1]
    action_dim = actions.shape[-1]
    context_z = np.zeros((count, CONTEXT, latent_dim), dtype=np.float32)
    action_history = np.zeros((count, CONTEXT, action_dim), dtype=np.float32)
    future_actions = np.zeros((count, HORIZON, action_dim), dtype=np.float32)
    targets = np.zeros((count, HORIZON, latent_dim), dtype=np.float32)
    pool_values = np.zeros((count, POOL_SLOTS, latent_dim), dtype=np.float32)
    pool_metadata = np.zeros(
        (count, POOL_SLOTS, METADATA_DIM),
        dtype=np.float32,
    )
    pool_type = np.zeros((count, POOL_SLOTS), dtype=np.int64)
    pool_valid = np.zeros((count, POOL_SLOTS), dtype=bool)
    source_index = np.zeros((count, POOL_SLOTS), dtype=np.int64)
    recent_values = np.zeros(
        (count, MEMORY_TOKENS, latent_dim),
        dtype=np.float32,
    )
    recent_metadata = np.zeros(
        (count, MEMORY_TOKENS, METADATA_DIM),
        dtype=np.float32,
    )
    robust_values = np.zeros_like(recent_values)
    robust_metadata = np.zeros_like(recent_metadata)
    robust_type = np.full((count, MEMORY_TOKENS), RECENT, dtype=np.int64)
    robust_slot = np.zeros(count, dtype=np.int64)
    query_signals = np.stack(
        [
            records[key].astype(np.float32)
            for key in (
                "revisit",
                "reappearance",
                "transition",
                "action_structure",
                "region_change",
                "proposal_score",
            )
        ],
        axis=1,
    )
    for row in range(count):
        episode = int(records["episode_id"][row])
        query_t = int(records["query_t"][row])
        gap = int(records["gap"][row])
        context_z[row] = latents[
            episode,
            query_t - CONTEXT + 1 : query_t + 1,
        ]
        action_history[row] = actions[
            episode,
            query_t - CONTEXT + 1 : query_t + 1,
        ]
        future_actions[row] = actions[
            episode,
            query_t : query_t + HORIZON,
        ]
        targets[row] = latents[
            episode,
            query_t + 1 : query_t + HORIZON + 1,
        ]
        values, metadata, type_id, valid, source = _mine_query_candidates(
            latents,
            episode,
            query_t,
            gap,
            discovery,
            scales,
        )
        pool_values[row] = values
        pool_metadata[row] = metadata
        pool_type[row] = type_id
        pool_valid[row] = valid
        source_index[row] = source
        slot = int(np.argmax(np.where(valid, metadata[:, 6], -np.inf)))
        robust_slot[row] = slot
        recent_stop = query_t - CONTEXT
        recent_indices = np.arange(
            recent_stop - MEMORY_TOKENS + 1,
            recent_stop + 1,
        )
        current = context_z[row].mean(0)
        recent_mean = latents[episode, recent_indices].mean(0)
        for column, index in enumerate(recent_indices):
            recent_values[row, column] = latents[episode, index]
            recent_metadata[row, column] = _frame_metadata(
                episode,
                int(index),
                query_t,
                current,
                recent_mean,
                latents,
                discovery,
                scales,
            )
        robust_values[row, 0] = values[slot]
        robust_metadata[row, 0] = metadata[slot]
        robust_type[row, 0] = type_id[slot]
        robust_values[row, 1:] = recent_values[row, 1:]
        robust_metadata[row, 1:] = recent_metadata[row, 1:]
    if not np.all(pool_valid.any(1)):
        raise RuntimeError("every query must have a historical candidate")
    if robust_values.nbytes // count != recent_values.nbytes // count:
        raise RuntimeError("recent and memory byte budgets differ")
    recent_type = np.full(
        (count, MEMORY_TOKENS),
        RECENT,
        dtype=np.int64,
    )
    memory_valid = np.ones((count, MEMORY_TOKENS), dtype=bool)
    batch = QueryTensors(
        context_z=torch.from_numpy(context_z).to(device),
        action_history=torch.from_numpy(action_history).to(device),
        future_actions=torch.from_numpy(future_actions).to(device),
        targets=torch.from_numpy(targets).to(device),
        events=torch.from_numpy(pool_values).to(device),
        metadata=torch.from_numpy(pool_metadata[..., :3]).to(device),
        valid=torch.from_numpy(pool_valid).to(device),
        recent_event=torch.from_numpy(recent_values[:, :1]).to(device),
    )
    return NativeSplit(
        name=name,
        batch=batch,
        recent=MemoryTokenBatch(
            values=torch.from_numpy(recent_values).to(device),
            metadata=torch.from_numpy(recent_metadata).to(device),
            token_type=torch.from_numpy(recent_type).to(device),
            valid=torch.from_numpy(memory_valid).to(device),
        ),
        robust=MemoryTokenBatch(
            values=torch.from_numpy(robust_values).to(device),
            metadata=torch.from_numpy(robust_metadata).to(device),
            token_type=torch.from_numpy(robust_type).to(device),
            valid=torch.from_numpy(memory_valid).to(device),
        ),
        pool=MemoryTokenBatch(
            values=torch.from_numpy(pool_values).to(device),
            metadata=torch.from_numpy(pool_metadata).to(device),
            token_type=torch.from_numpy(pool_type).to(device),
            valid=torch.from_numpy(pool_valid).to(device),
        ),
        source_index=torch.from_numpy(source_index).to(device),
        proposal_score=torch.from_numpy(pool_metadata[..., 6]).to(device),
        robust_slot=torch.from_numpy(robust_slot).to(device),
        episode_ids=records["episode_id"].astype(np.int64),
        query_t=records["query_t"].astype(np.int64),
        gaps=records["gap"].astype(np.int64),
        query_signals=query_signals,
    )


def raw_memory_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed + 31_007,
        memory_hidden=args.raw_memory_hidden,
        max_events=MEMORY_TOKENS,
        memory_lr=args.raw_memory_lr,
        memory_epochs=args.raw_memory_epochs,
        memory_patience=args.raw_memory_patience,
        residual_cost=1e-3,
        batch_size=args.batch_size,
    )


@torch.no_grad()
def raw_memory_loss(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    tokens: MemoryTokenBatch,
    batch_size: int,
) -> torch.Tensor:
    query = tokens_to_queries(batch, tokens)
    values = []
    for start in range(0, len(query), batch_size):
        stop = min(len(query), start + batch_size)
        index = torch.arange(start, stop, device=batch.context_z.device)
        part = query.index(index)
        prediction, _ = rollout(host, memory, part, mask=part.valid)
        values.append(horizon_loss(prediction, part.targets).mean(1))
    return torch.cat(values)


def conditioned_rollout(
    host: ActionConditionedHost,
    conditioner: NativeMemoryConditioner | None,
    batch: QueryTensors,
    memory: MemoryTokenBatch | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context_z = batch.context_z
    action_history = batch.action_history
    predictions = []
    residual_norm = []
    attention_entropy = []
    first_token_code = None
    for step in range(batch.future_actions.shape[1]):
        action = batch.future_actions[:, step]
        context_actions = (
            action_history
            if step == 0
            else torch.cat(
                [action_history[:, 1:], action[:, None]],
                dim=1,
            )
        )
        base = host(context_z, context_actions)
        if conditioner is None:
            prediction = base
        else:
            prediction, telemetry = conditioner(
                base,
                context_z,
                action,
                memory,
            )
            residual_norm.append(telemetry["residual_norm"])
            attention_entropy.append(telemetry["attention_entropy"])
            if first_token_code is None and "token_code" in telemetry:
                first_token_code = telemetry["token_code"]
        predictions.append(prediction)
        context_z = torch.cat(
            [context_z[:, 1:], prediction[:, None]],
            dim=1,
        )
        action_history = context_actions
    info: dict[str, torch.Tensor] = {}
    if residual_norm:
        info["residual_norm"] = torch.stack(residual_norm, dim=1)
        info["attention_entropy"] = torch.stack(
            attention_entropy,
            dim=1,
        )
    if first_token_code is not None:
        info["token_code"] = first_token_code
    return torch.stack(predictions, dim=1), info


@torch.no_grad()
def conditioned_loss(
    host: ActionConditionedHost,
    conditioner: NativeMemoryConditioner | None,
    batch: QueryTensors,
    memory: MemoryTokenBatch | None,
    batch_size: int,
) -> torch.Tensor:
    values = []
    for start in range(0, len(batch), batch_size):
        stop = min(len(batch), start + batch_size)
        index = torch.arange(start, stop, device=batch.context_z.device)
        part = batch.index(index)
        part_memory = None if memory is None else memory.index(index)
        prediction, _ = conditioned_rollout(
            host,
            conditioner,
            part,
            part_memory,
        )
        values.append(horizon_loss(prediction, part.targets).mean(1))
    return torch.cat(values)


def make_frame_memory(
    split: NativeSplit,
    rows: np.ndarray,
    frame_indices: np.ndarray,
    latents: np.ndarray,
    discovery: dict[str, np.ndarray],
    scales: dict[str, float],
    device: torch.device,
) -> MemoryTokenBatch:
    rows = np.asarray(rows, dtype=np.int64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    values = split.recent.values[
        torch.as_tensor(rows, device=device)
    ].detach().cpu().numpy().copy()
    metadata = split.recent.metadata[
        torch.as_tensor(rows, device=device)
    ].detach().cpu().numpy().copy()
    token_type = np.full((len(rows), MEMORY_TOKENS), RECENT, dtype=np.int64)
    for output_row, (split_row, source_index) in enumerate(
        zip(rows, frame_indices)
    ):
        episode = int(split.episode_ids[split_row])
        query_t = int(split.query_t[split_row])
        current = latents[
            episode,
            query_t - CONTEXT + 1 : query_t + 1,
        ].mean(0)
        recent_stop = query_t - CONTEXT
        recent_mean = latents[
            episode,
            recent_stop - MEMORY_TOKENS + 1 : recent_stop + 1,
        ].mean(0)
        values[output_row, 0] = latents[episode, source_index]
        metadata[output_row, 0] = _frame_metadata(
            episode,
            int(source_index),
            query_t,
            current,
            recent_mean,
            latents,
            discovery,
            scales,
        )
        token_type[output_row, 0] = FRAME
    return MemoryTokenBatch(
        values=torch.from_numpy(values).to(device),
        metadata=torch.from_numpy(metadata).to(device),
        token_type=torch.from_numpy(token_type).to(device),
        valid=torch.ones(
            (len(rows), MEMORY_TOKENS),
            dtype=torch.bool,
            device=device,
        ),
    )


def repeat_query_tensors(
    batch: QueryTensors,
    rows: torch.Tensor,
) -> QueryTensors:
    return batch.index(rows)


@torch.no_grad()
def oracle_frame_search(
    host: ActionConditionedHost,
    split: NativeSplit,
    latents: np.ndarray,
    discovery: dict[str, np.ndarray],
    scales: dict[str, float],
    device: torch.device,
    batch_size: int,
    *,
    raw_memory: RawMemoryConditioner | None = None,
    conditioner: NativeMemoryConditioner | None = None,
) -> OracleResult:
    if (raw_memory is None) == (conditioner is None):
        raise ValueError("choose exactly one memory interface")
    repeated_rows = []
    frame_indices = []
    offsets = [0]
    for row, (query_t, gap) in enumerate(zip(split.query_t, split.gaps)):
        indices = np.arange(int(query_t - gap + 1), dtype=np.int64)
        repeated_rows.append(np.full(len(indices), row, dtype=np.int64))
        frame_indices.append(indices)
        offsets.append(offsets[-1] + len(indices))
    row_np = np.concatenate(repeated_rows)
    frame_np = np.concatenate(frame_indices)
    all_loss = []
    for start in range(0, len(row_np), batch_size):
        stop = min(len(row_np), start + batch_size)
        rows = row_np[start:stop]
        frames = frame_np[start:stop]
        row_tensor = torch.as_tensor(rows, device=device)
        part = repeat_query_tensors(split.batch, row_tensor)
        tokens = make_frame_memory(
            split,
            rows,
            frames,
            latents,
            discovery,
            scales,
            device,
        )
        if raw_memory is not None:
            loss = raw_memory_loss(
                host,
                raw_memory,
                part,
                tokens,
                batch_size,
            )
        else:
            loss = conditioned_loss(
                host,
                conditioner,
                part,
                tokens,
                batch_size,
            )
        all_loss.append(loss)
    flat_loss = torch.cat(all_loss)
    best_loss = torch.empty(len(split), device=device)
    best_index = torch.empty(len(split), dtype=torch.long, device=device)
    for row in range(len(split)):
        local = flat_loss[offsets[row] : offsets[row + 1]]
        choice = int(local.argmin())
        best_loss[row] = local[choice]
        best_index[row] = int(frame_np[offsets[row] + choice])
    selected_memory = make_frame_memory(
        split,
        np.arange(len(split), dtype=np.int64),
        best_index.detach().cpu().numpy(),
        latents,
        discovery,
        scales,
        device,
    )
    return OracleResult(
        loss=best_loss,
        frame_index=best_index,
        memory=selected_memory,
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    unique = np.unique(groups)
    if not len(unique):
        return {"mean": None, "ci95": [None, None], "group_count": 0}
    grouped = np.asarray(
        [values[groups == group].mean() for group in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(grouped, size=len(grouped), replace=True).mean()
            for _ in range(repetitions)
        ],
        dtype=np.float64,
    )
    return {
        "mean": float(grouped.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).astype(float).tolist(),
        "group_count": int(len(grouped)),
        "bootstrap_unit": "native trajectory",
    }


def opportunity_summary(
    split: NativeSplit,
    recent_loss: torch.Tensor,
    oracle_loss: torch.Tensor,
    threshold: float,
    seed: int,
    repetitions: int,
) -> tuple[dict[str, Any], np.ndarray]:
    recent = recent_loss.detach().cpu().numpy()
    oracle = oracle_loss.detach().cpu().numpy()
    relative = (recent - oracle) / np.maximum(recent, 1e-8)
    retained = relative > threshold
    by_gap = []
    for gap in GAPS:
        available = split.gaps == gap
        chosen = available & retained
        gain = recent - oracle
        summary = bootstrap_mean_ci(
            gain[chosen],
            split.episode_ids[chosen],
            seed + gap * 101,
            repetitions,
        )
        by_gap.append(
            {
                "gap": gap,
                "available_count": int(available.sum()),
                "retained_count": int(chosen.sum()),
                "retained_fraction": float(
                    chosen.sum() / max(1, available.sum())
                ),
                "oracle_gain_vs_recent": summary,
            }
        )
    return {
        "threshold_relative_improvement": float(threshold),
        "available_count": int(len(split)),
        "retained_count": int(retained.sum()),
        "retained_fraction": float(retained.mean()),
        "by_gap": by_gap,
    }, retained


def train_conditioner(
    host: ActionConditionedHost,
    train: NativeSplit,
    validation: NativeSplit,
    train_opportunity: np.ndarray,
    validation_opportunity: np.ndarray,
    train_oracle: MemoryTokenBatch,
    validation_oracle: MemoryTokenBatch,
    train_raw_recent: torch.Tensor,
    validation_raw_recent: torch.Tensor,
    train_raw_oracle_loss: torch.Tensor,
    validation_raw_oracle_loss: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[NativeMemoryConditioner, dict[str, Any]]:
    conditioner = NativeMemoryConditioner(
        latent_dim=train.batch.context_z.shape[-1],
        action_dim=train.batch.future_actions.shape[-1],
        metadata_dim=METADATA_DIM,
        code_dim=args.conditioner_code_dim,
        hidden=args.conditioner_hidden,
        heads=args.conditioner_heads,
        max_residual=args.conditioner_max_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(
        conditioner.parameters(),
        lr=args.conditioner_lr,
        weight_decay=1e-4,
    )
    train_opp = torch.from_numpy(train_opportunity).to(device)
    val_opp = torch.from_numpy(validation_opportunity).to(device)
    rng = np.random.default_rng(args.seed + 41_009)
    best_state = None
    best_objective = float("inf")
    best_epoch = 0
    stale = 0
    dual = 0.0
    history = []
    host_digest = tensor_digest(host)
    for epoch in range(args.conditioner_epochs):
        conditioner.train()
        epoch_metrics: dict[str, list[float]] = {
            key: []
            for key in (
                "prediction",
                "oracle",
                "effect_shortfall",
                "geometry",
                "variance",
                "covariance",
                "constraint",
                "total",
            )
        }
        for indices in batches(len(train), args.batch_size, rng):
            index = torch.as_tensor(indices, device=device)
            batch = train.batch.index(index)
            opportunity = train_opp[index]
            main_memory = choose_memory(
                opportunity,
                train.robust.index(index),
                train.recent.index(index),
            )
            prediction, telemetry = conditioned_rollout(
                host,
                conditioner,
                batch,
                main_memory,
            )
            per_query = horizon_loss(prediction, batch.targets).mean(1)
            prediction_loss = per_query.mean()
            if bool(opportunity.any()):
                opportunity_index = torch.nonzero(opportunity).flatten()
                oracle_prediction, _ = conditioned_rollout(
                    host,
                    conditioner,
                    batch.index(opportunity_index),
                    train_oracle.index(index).index(
                        opportunity_index
                    ),
                )
                oracle_per_query = horizon_loss(
                    oracle_prediction,
                    batch.targets[opportunity],
                ).mean(1)
                recent_prediction, _ = conditioned_rollout(
                    host,
                    conditioner,
                    batch.index(opportunity_index),
                    train.recent.index(index).index(opportunity_index),
                )
                recent_per_query = horizon_loss(
                    recent_prediction,
                    batch.targets[opportunity],
                ).mean(1)
                oracle_loss = oracle_per_query.mean()
                raw_gain = (
                    train_raw_recent[index][opportunity]
                    - train_raw_oracle_loss[index][opportunity]
                ).clamp_min(0.0)
                desired_gain = args.conditioner_recovery_target * raw_gain
                effect_shortfall = F.relu(
                    desired_gain - (recent_per_query - oracle_per_query)
                ).mean()
                recent_reference = recent_per_query.mean()
            else:
                oracle_loss = prediction_loss.new_zeros(())
                effect_shortfall = prediction_loss.new_zeros(())
                recent_reference = prediction_loss.new_zeros(())
            geometry = geometry_regularization(
                telemetry["token_code"],
                main_memory.values,
                main_memory.valid,
            )
            variance, covariance = variance_regularization(
                telemetry["token_code"],
                main_memory.valid,
            )
            ordinary = ~opportunity
            if bool(ordinary.any()):
                baseline = train_raw_recent[index][ordinary].mean()
                constraint = (
                    per_query[ordinary].mean()
                    / baseline.clamp_min(1e-8)
                    - 1.05
                )
            else:
                constraint = prediction_loss.new_zeros(())
            loss = (
                prediction_loss
                + args.conditioner_oracle_weight * oracle_loss
                + 0.25 * recent_reference
                + args.conditioner_effect_weight * effect_shortfall
                + args.geometry_weight * geometry
                + args.variance_weight * variance
                + args.covariance_weight * covariance
                + args.residual_weight
                * telemetry["residual_norm"].mean()
                + dual * F.relu(constraint)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(conditioner.parameters(), 2.0)
            optimizer.step()
            for key, value in (
                ("prediction", prediction_loss),
                ("oracle", oracle_loss),
                ("effect_shortfall", effect_shortfall),
                ("geometry", geometry),
                ("variance", variance),
                ("covariance", covariance),
                ("constraint", constraint),
                ("total", loss),
            ):
                epoch_metrics[key].append(float(value.detach()))
        conditioner.eval()
        val_recent = conditioned_loss(
            host,
            conditioner,
            validation.batch,
            validation.recent,
            args.batch_size,
        )
        val_oracle_loss = conditioned_loss(
            host,
            conditioner,
            validation.batch,
            validation_oracle,
            args.batch_size,
        )
        ordinary = ~val_opp
        val_degradation = float(
            val_recent[ordinary].mean()
            / validation_raw_recent[ordinary].mean().clamp_min(1e-8)
            - 1.0
        ) if bool(ordinary.any()) else 0.0
        opportunity_objective = float(
            val_oracle_loss[val_opp].mean()
            if bool(val_opp.any())
            else val_oracle_loss.mean()
        )
        if bool(val_opp.any()):
            desired = args.conditioner_recovery_target * (
                validation_raw_recent[val_opp]
                - validation_raw_oracle_loss[val_opp]
            ).clamp_min(0.0)
            achieved = (
                val_recent[val_opp] - val_oracle_loss[val_opp]
            )
            validation_shortfall = float(
                F.relu(desired - achieved).mean()
            )
            validation_recovery = float(
                achieved.mean() / desired.div(
                    args.conditioner_recovery_target
                ).mean().clamp_min(1e-12)
            )
        else:
            validation_shortfall = 0.0
            validation_recovery = None
        feasible = val_degradation <= 0.05
        objective = (
            opportunity_objective
            + args.conditioner_effect_weight * validation_shortfall
            + 1000.0 * max(0.0, val_degradation - 0.05)
        )
        dual = max(
            0.0,
            dual
            + args.dual_lr * (val_degradation - 0.05),
        )
        history.append(
            {
                "epoch": epoch + 1,
                **{
                    key: float(np.mean(value))
                    for key, value in epoch_metrics.items()
                },
                "validation_opportunity_mse": opportunity_objective,
                "validation_effect_shortfall": validation_shortfall,
                "validation_oracle_recovery": validation_recovery,
                "validation_ordinary_degradation": val_degradation,
                "validation_constraint_feasible": feasible,
                "dual": dual,
                "checkpoint_objective": objective,
            }
        )
        if math.isfinite(objective) and objective < best_objective - 1e-7:
            best_objective = objective
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in conditioner.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.conditioner_patience:
            break
    if best_state is None:
        raise RuntimeError("conditioner produced no finite checkpoint")
    conditioner.load_state_dict(best_state)
    conditioner.eval()
    for parameter in conditioner.parameters():
        parameter.requires_grad_(False)
    if tensor_digest(host) != host_digest:
        raise RuntimeError("frozen host changed during conditioner training")
    with torch.no_grad():
        no_memory_prediction, _ = conditioned_rollout(
            host,
            conditioner,
            validation.batch,
            None,
        )
        base_prediction, _ = conditioned_rollout(
            host,
            None,
            validation.batch,
            None,
        )
        zero_memory_distillation = float(
            F.mse_loss(no_memory_prediction, base_prediction)
        )
    return conditioner, {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "final_dual": dual,
        "parameters": parameter_count(conditioner),
        "base_encoder_frozen": True,
        "zero_memory_residual_initialization": True,
        "zero_memory_distillation_mse": zero_memory_distillation,
        "explicit_non_degradation_constraint": (
            "validation ordinary recent loss <= raw recent baseline +5%"
        ),
        "semantic_bottleneck_dim": args.conditioner_code_dim,
        "distinct_tokens_until_cross_attention": MEMORY_TOKENS,
        "separate_recent_and_historical_residual_paths": True,
        "target_oracle_recovery": args.conditioner_recovery_target,
    }


@torch.no_grad()
def gate_features(
    host: ActionConditionedHost,
    conditioner: NativeMemoryConditioner,
    split: NativeSplit,
    batch_size: int,
) -> np.ndarray:
    descriptor_rows = []
    for start in range(0, len(split), batch_size):
        stop = min(len(split), start + batch_size)
        index = torch.arange(
            start,
            stop,
            device=split.batch.context_z.device,
        )
        batch = split.batch.index(index)
        base = host(batch.context_z, batch.action_history)
        robust, robust_info = conditioner(
            base,
            batch.context_z,
            batch.future_actions[:, 0],
            split.robust.index(index),
        )
        recent, recent_info = conditioner(
            base,
            batch.context_z,
            batch.future_actions[:, 0],
            split.recent.index(index),
        )
        candidate = split.robust.values[index, 0]
        query = batch.context_z.mean(1)
        cosine = F.cosine_similarity(candidate, query, dim=-1)
        candidate_recent = (
            candidate - split.recent.values[index].mean(1)
        ).square().mean(-1)
        prediction_delta = (robust - recent).square().mean(-1)
        descriptor_rows.append(
            torch.stack(
                [
                    cosine,
                    candidate_recent,
                    robust_info["residual_norm"],
                    recent_info["residual_norm"],
                    robust_info["attention_entropy"],
                    recent_info["attention_entropy"],
                    prediction_delta,
                ],
                dim=1,
            ).cpu().numpy()
        )
    descriptors = np.concatenate(descriptor_rows)
    gap_one_hot = np.stack(
        [(split.gaps == gap).astype(np.float32) for gap in GAPS],
        axis=1,
    )
    robust_metadata = split.robust.metadata[:, 0].detach().cpu().numpy()
    robust_type = split.robust.token_type[:, 0].detach().cpu().numpy()
    type_one_hot = np.stack(
        [(robust_type == type_id).astype(np.float32) for type_id in (FRAME, EVENT)],
        axis=1,
    )
    return np.concatenate(
        [
            np.log2(split.gaps.astype(np.float32))[:, None] / 7.0,
            gap_one_hot,
            split.query_signals.astype(np.float32),
            robust_metadata.astype(np.float32),
            type_one_hot,
            descriptors.astype(np.float32),
        ],
        axis=1,
    )


def gate_member_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    logit: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    gaussian = 0.5 * (
        torch.exp(-log_variance) * torch.square(mean - target)
        + log_variance
    ).mean()
    huber = F.smooth_l1_loss(mean, target, beta=0.01)
    binary = F.binary_cross_entropy_with_logits(
        logit,
        label,
        pos_weight=positive_weight,
    )
    return gaussian + 0.25 * huber + 0.5 * binary


def calibration_ece(
    probability: np.ndarray,
    label: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probability)
    value = 0.0
    for index in range(bins):
        selected = (
            (probability >= edges[index])
            & (
                probability <= edges[index + 1]
                if index == bins - 1
                else probability < edges[index + 1]
            )
        )
        if selected.any():
            value += selected.mean() * abs(
                probability[selected].mean() - label[selected].mean()
            )
    return float(value if total else 0.0)


def train_gate(
    train_features_np: np.ndarray,
    validation_features_np: np.ndarray,
    train_utility: np.ndarray,
    validation_utility: np.ndarray,
    train_episode_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[UtilityGateEnsemble, dict[str, Any], np.ndarray, np.ndarray]:
    mean = train_features_np.mean(0)
    std = np.maximum(train_features_np.std(0), 1e-5)
    train_x = torch.from_numpy(
        ((train_features_np - mean) / std).astype(np.float32)
    ).to(device)
    val_x = torch.from_numpy(
        ((validation_features_np - mean) / std).astype(np.float32)
    ).to(device)
    train_y = torch.from_numpy(
        np.clip(train_utility, -0.5, 0.5).astype(np.float32)
    ).to(device)
    val_y = torch.from_numpy(
        np.clip(validation_utility, -0.5, 0.5).astype(np.float32)
    ).to(device)
    train_label = (train_y > args.gate_help_margin).float()
    val_label = (val_y > args.gate_help_margin).float()
    ensemble = UtilityGateEnsemble(
        train_x.shape[1],
        hidden=args.gate_hidden,
        members=GATE_FOLDS,
    ).to(device)
    folds = np.asarray(
        [
            int(hashlib.sha256(str(int(value)).encode()).hexdigest()[:8], 16)
            % GATE_FOLDS
            for value in train_episode_ids
        ],
        dtype=np.int64,
    )
    oof_mean = np.full(len(train_y), np.nan, dtype=np.float32)
    oof_probability = np.full(len(train_y), np.nan, dtype=np.float32)
    member_receipts = []
    for fold, member in enumerate(ensemble.members):
        training_np = folds != fold
        held_np = folds == fold
        training = torch.as_tensor(np.flatnonzero(training_np), device=device)
        held = torch.as_tensor(np.flatnonzero(held_np), device=device)
        train_episodes = set(train_episode_ids[training_np].tolist())
        held_episodes = set(train_episode_ids[held_np].tolist())
        if train_episodes & held_episodes:
            raise RuntimeError("cross-fitted gate trajectory leakage")
        positive = float(train_label[training].sum())
        negative = float(len(training) - positive)
        positive_weight = train_y.new_tensor(
            max(1.0, negative / max(1.0, positive))
        )
        optimizer = torch.optim.AdamW(
            member.parameters(),
            lr=args.gate_lr,
            weight_decay=1e-4,
        )
        rng = np.random.default_rng(args.seed * 10_003 + fold)
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        for epoch in range(args.gate_epochs):
            member.train()
            losses = []
            order = rng.permutation(training.detach().cpu().numpy())
            for start in range(0, len(order), args.batch_size):
                index = torch.as_tensor(
                    order[start : start + args.batch_size],
                    device=device,
                )
                prediction = member(train_x[index])
                loss = gate_member_loss(
                    *prediction,
                    train_y[index],
                    train_label[index],
                    positive_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            member.eval()
            with torch.no_grad():
                prediction = member(val_x)
                validation_loss = float(
                    gate_member_loss(
                        *prediction,
                        val_y,
                        val_label,
                        positive_weight,
                    )
                )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in member.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= args.gate_patience:
                break
        if best_state is None:
            raise RuntimeError("utility gate member produced no checkpoint")
        member.load_state_dict(best_state)
        member.eval()
        with torch.no_grad():
            held_mean, _, held_logit = member(train_x[held])
        oof_mean[held_np] = held_mean.cpu().numpy()
        oof_probability[held_np] = torch.sigmoid(held_logit).cpu().numpy()
        member_receipts.append(
            {
                "fold": fold,
                "training_trajectories": len(train_episodes),
                "held_out_trajectories": len(held_episodes),
                "trajectory_overlap": 0,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "history": history,
            }
        )
    if not np.isfinite(oof_mean).all():
        raise RuntimeError("cross-fitted utility predictions are incomplete")
    ensemble.eval()
    with torch.no_grad():
        validation_output = ensemble(val_x)
    val_mean = validation_output["mean"].cpu().numpy()
    val_std = validation_output["std"].cpu().numpy()
    standardized = (val_mean - validation_utility) / np.maximum(val_std, 1e-8)
    conformal = max(
        0.0,
        float(np.quantile(standardized, args.gate_conformal_coverage)),
    )
    ensemble.conformal_quantile.fill_(conformal)
    logits = validation_output["member_logit"].cpu()
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.geomspace(0.25, 4.0, 33):
        probability = torch.sigmoid(logits / float(temperature)).mean(0)
        nll = float(
            F.binary_cross_entropy(
                probability.clamp(1e-6, 1.0 - 1e-6),
                val_label.cpu(),
            )
        )
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    ensemble.probability_temperature.fill_(best_temperature)
    for parameter in ensemble.parameters():
        parameter.requires_grad_(False)
    ensemble.eval()
    with torch.no_grad():
        calibrated = ensemble(val_x)
    probability = calibrated["probability"].cpu().numpy()
    lower = (
        calibrated["mean"]
        - ensemble.conformal_quantile * calibrated["std"]
    ).cpu().numpy()
    empirical_lower_coverage = float(
        np.mean(validation_utility >= lower)
    )
    summary = {
        "members": member_receipts,
        "crossfit_folds": GATE_FOLDS,
        "crossfit_trajectory_overlap": 0,
        "normalizer_mean": mean.astype(float).tolist(),
        "normalizer_std": std.astype(float).tolist(),
        "conformal_target_coverage": args.gate_conformal_coverage,
        "conformal_quantile": conformal,
        "validation_lower_bound_coverage": empirical_lower_coverage,
        "probability_temperature": best_temperature,
        "validation_probability_ece": calibration_ece(
            probability,
            val_label.cpu().numpy(),
        ),
        "oof_utility_correlation": float(
            np.corrcoef(oof_mean, train_utility)[0, 1]
        ),
        "oof_probability_ece": calibration_ece(
            oof_probability,
            (train_utility > args.gate_help_margin).astype(np.float32),
        ),
        "parameters": parameter_count(ensemble),
    }
    return ensemble, summary, mean, std


def selective_metrics(
    active: np.ndarray,
    relative_utility: np.ndarray,
    absolute_gain: np.ndarray,
    episode_ids: np.ndarray,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    active = np.asarray(active, dtype=bool)
    coverage = float(active.mean())
    activated_relative = relative_utility[active]
    activated_absolute = absolute_gain[active]
    groups = episode_ids[active]
    activated_summary = bootstrap_mean_ci(
        activated_absolute,
        groups,
        seed,
        repetitions,
    )
    policy_summary = bootstrap_mean_ci(
        np.where(active, absolute_gain, 0.0),
        episode_ids,
        seed + 101,
        repetitions,
    )
    return {
        "coverage": coverage,
        "activated_count": int(active.sum()),
        "precision_positive_utility": (
            float(np.mean(activated_relative > 0.0))
            if len(activated_relative)
            else None
        ),
        "activated_gain": activated_summary,
        "overall_policy_gain": policy_summary,
        "selective_risk": (
            float(np.mean(-activated_relative))
            if len(activated_relative)
            else None
        ),
        "low_quantile_relative_utility": (
            float(np.quantile(activated_relative, 0.05))
            if len(activated_relative)
            else None
        ),
        "worst_relative_utility": (
            float(np.min(activated_relative))
            if len(activated_relative)
            else None
        ),
        "worst_case_degradation": (
            float(max(0.0, -np.min(activated_relative)))
            if len(activated_relative)
            else None
        ),
    }


def delta_sweep(
    lower: np.ndarray,
    probability: np.ndarray,
    relative_utility: np.ndarray,
    absolute_gain: np.ndarray,
    episode_ids: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    quantiles = np.linspace(0.0, 0.95, 40)
    thresholds = sorted(
        set(
            [0.0]
            + np.quantile(lower, quantiles).astype(float).tolist()
            + [float(np.max(lower) + 1e-6)]
        )
    )
    rows = []
    feasible = []
    labels = relative_utility > args.gate_help_margin
    ece = calibration_ece(probability, labels.astype(np.float32))
    for index, delta in enumerate(thresholds):
        active = lower > delta
        metrics = selective_metrics(
            active,
            relative_utility,
            absolute_gain,
            episode_ids,
            seed + index * 17,
            args.bootstrap_repetitions,
        )
        metrics.update(
            {
                "delta": float(delta),
                "probability_ece": ece,
            }
        )
        lower_ci = metrics["activated_gain"]["ci95"][0]
        overall = metrics["overall_policy_gain"]["mean"]
        low_quantile = metrics["low_quantile_relative_utility"]
        precision = metrics["precision_positive_utility"]
        metrics["validation_safe"] = bool(
            metrics["coverage"] >= args.minimum_gate_coverage
            and lower_ci is not None
            and lower_ci > 0.0
            and overall is not None
            and overall >= 0.0
            and precision is not None
            and precision >= args.minimum_gate_precision
            and low_quantile is not None
            and low_quantile >= -args.maximum_low_quantile_degradation
        )
        rows.append(metrics)
        if metrics["validation_safe"]:
            feasible.append(metrics)
    selected = (
        max(
            feasible,
            key=lambda row: (
                row["coverage"],
                row["overall_policy_gain"]["mean"],
            ),
        )
        if feasible
        else None
    )
    return rows, selected


@torch.no_grad()
def evaluate_gate_output(
    ensemble: UtilityGateEnsemble,
    features_np: np.ndarray,
    normalizer_mean: np.ndarray,
    normalizer_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = torch.from_numpy(
        ((features_np - normalizer_mean) / normalizer_std).astype(np.float32)
    ).to(device)
    lower, output = ensemble.lower_confidence_bound(features)
    return (
        lower.cpu().numpy(),
        output["probability"].cpu().numpy(),
        output["mean"].cpu().numpy(),
        output["std"].cpu().numpy(),
    )


def mean_latency_ms(
    host: ActionConditionedHost,
    conditioner: NativeMemoryConditioner,
    split: NativeSplit,
    repetitions: int,
) -> float:
    count = min(len(split), 256)
    index = torch.arange(count, device=split.batch.context_z.device)
    batch = split.batch.index(index)
    memory = split.robust.index(index)
    for _ in range(5):
        conditioned_rollout(host, conditioner, batch, memory)
    if batch.context_z.device.type == "cuda":
        torch.cuda.synchronize(batch.context_z.device)
    started = time.perf_counter()
    for _ in range(repetitions):
        conditioned_rollout(host, conditioner, batch, memory)
    if batch.context_z.device.type == "cuda":
        torch.cuda.synchronize(batch.context_z.device)
    return 1000.0 * (time.perf_counter() - started) / (
        repetitions * count
    )


def _host_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        context=CONTEXT,
        host_hidden=args.host_hidden,
        host_epochs=args.host_epochs,
        host_patience=args.host_patience,
        host_lr=args.host_lr,
        batch_size=args.batch_size,
    )


def _phase_result_path(output_dir: Path) -> Path:
    return output_dir / "result.json"


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = cell_dir(args.output, args.env_name, args.seed)
    result_path = _phase_result_path(output_dir)
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    set_seed(args.seed)
    device = resolve_device(args.gpu)
    feature = feature_path(args.output, args.env_name)
    recipe = recipe_path(args.output, args.env_name)
    if not feature.is_file() or not recipe.is_file():
        raise FileNotFoundError(
            f"run build_cem_native_long.py first for {args.env_name}"
        )
    with np.load(feature, allow_pickle=False) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        train_idx = np.asarray(data["train_indices"], dtype=np.int64)
        val_idx = np.asarray(data["val_indices"], dtype=np.int64)
        test_idx = np.asarray(data["test_indices"], dtype=np.int64)
    host, host_training = train_host(
        latents,
        actions,
        train_idx,
        val_idx,
        _host_args(args),
        device,
    )
    host_digest = tensor_digest(host)
    surprise = one_step_surprise(
        host,
        latents,
        actions,
        device,
        CONTEXT,
        args.batch_size,
    )
    discovery = discovery_arrays(latents, actions, surprise)
    scales = discovery_scales(discovery, train_idx)
    splits = {
        name: build_native_split(
            name,
            load_recipe_records(args.output, args.env_name, name),
            latents,
            actions,
            discovery,
            scales,
            device,
        )
        for name in ("train", "validation", "test")
    }
    split_episode_sets = {
        name: set(value.episode_ids.tolist()) for name, value in splits.items()
    }
    if (
        split_episode_sets["train"] & split_episode_sets["validation"]
        or split_episode_sets["train"] & split_episode_sets["test"]
        or split_episode_sets["validation"] & split_episode_sets["test"]
    ):
        raise RuntimeError("query trajectory leakage")
    train_raw = concatenate_query_tensors(
        [
            tokens_to_queries(splits["train"].batch, splits["train"].robust),
            tokens_to_queries(splits["train"].batch, splits["train"].recent),
        ]
    )
    validation_raw = concatenate_query_tensors(
        [
            tokens_to_queries(
                splits["validation"].batch,
                splits["validation"].robust,
            ),
            tokens_to_queries(
                splits["validation"].batch,
                splits["validation"].recent,
            ),
        ]
    )
    raw_memory, raw_training = train_memory(
        host,
        train_raw,
        validation_raw,
        raw_memory_args(args),
        device,
    )
    raw_memory.eval()
    for parameter in raw_memory.parameters():
        parameter.requires_grad_(False)
    raw_recent: dict[str, torch.Tensor] = {}
    raw_robust: dict[str, torch.Tensor] = {}
    raw_oracle: dict[str, OracleResult] = {}
    for split_name, split in splits.items():
        raw_recent[split_name] = raw_memory_loss(
            host,
            raw_memory,
            split.batch,
            split.recent,
            args.batch_size,
        )
        raw_robust[split_name] = raw_memory_loss(
            host,
            raw_memory,
            split.batch,
            split.robust,
            args.batch_size,
        )
        raw_oracle[split_name] = oracle_frame_search(
            host,
            split,
            latents,
            discovery,
            scales,
            device,
            args.oracle_batch_size,
            raw_memory=raw_memory,
        )
    train_relative = (
        raw_recent["train"] - raw_oracle["train"].loss
    ) / raw_recent["train"].clamp_min(1e-8)
    opportunity_threshold = max(
        args.minimum_opportunity_effect,
        float(torch.quantile(train_relative.float(), 0.75)),
    )
    opportunities = {}
    opportunity_masks = {}
    for split_index, split_name in enumerate(
        ("train", "validation", "test")
    ):
        opportunities[split_name], opportunity_masks[split_name] = (
            opportunity_summary(
                splits[split_name],
                raw_recent[split_name],
                raw_oracle[split_name].loss,
                opportunity_threshold,
                args.seed + split_index * 10_007,
                args.bootstrap_repetitions,
            )
        )
    high_test = [
        row
        for row in opportunities["test"]["by_gap"]
        if row["gap"] >= 32
    ]
    phase_a_local = bool(
        opportunities["test"]["retained_fraction"]
        >= args.minimum_opportunity_coverage
        and all(
            row["retained_count"] > 0
            and row["oracle_gain_vs_recent"]["ci95"][0] is not None
            and row["oracle_gain_vs_recent"]["ci95"][0] > 0.0
            for row in high_test
        )
    )
    phase_a = {
        "passed_local": phase_a_local,
        "filter": {
            "definition": (
                "raw-adapter oracle frame relative improvement exceeds max("
                "fixed minimum, train-only 75th percentile)"
            ),
            "minimum_effect": args.minimum_opportunity_effect,
            "train_quantile": 0.75,
            "fixed_threshold": opportunity_threshold,
            "test_future_used_to_apply_fixed_oracle_filter": True,
            "test_future_used_for_training": False,
        },
        "splits": opportunities,
        "required_test_coverage": args.minimum_opportunity_coverage,
        "required_high_gaps": [32, 64, 128],
    }
    conditioner = None
    conditioner_training = None
    conditioner_metrics = None
    gate_training = None
    gate_metrics = None
    decision_logs: list[dict[str, Any]] = []
    evaluation: dict[str, np.ndarray] = {
        "episode_id": splits["test"].episode_ids,
        "query_t": splits["test"].query_t,
        "gap": splits["test"].gaps,
        "loss_raw_recent": raw_recent["test"].cpu().numpy(),
        "loss_raw_robust": raw_robust["test"].cpu().numpy(),
        "loss_raw_oracle": raw_oracle["test"].loss.cpu().numpy(),
        "raw_oracle_frame_index": (
            raw_oracle["test"].frame_index.cpu().numpy()
        ),
        "opportunity": opportunity_masks["test"].astype(bool),
    }
    phase_b_local = False
    phase_c_local = False
    selected_delta = None
    no_memory_test = conditioned_loss(
        host,
        None,
        splits["test"].batch,
        None,
        args.batch_size,
    )
    evaluation["loss_no_memory"] = no_memory_test.cpu().numpy()
    execute_conditioner = bool(
        phase_a_local or (args.smoke and args.exercise_all_phases)
    )
    if execute_conditioner:
        conditioner, conditioner_training = train_conditioner(
            host,
            splits["train"],
            splits["validation"],
            opportunity_masks["train"],
            opportunity_masks["validation"],
            raw_oracle["train"].memory,
            raw_oracle["validation"].memory,
            raw_recent["train"],
            raw_recent["validation"],
            raw_oracle["train"].loss,
            raw_oracle["validation"].loss,
            args,
            device,
        )
        if any(parameter.requires_grad for parameter in conditioner.parameters()):
            raise RuntimeError("conditioner was not frozen before gate evaluation")
        conditioner_digest = tensor_digest(conditioner)
        conditioned_recent = {}
        conditioned_robust = {}
        conditioned_oracle = {}
        for split_name, split in splits.items():
            conditioned_recent[split_name] = conditioned_loss(
                host,
                conditioner,
                split.batch,
                split.recent,
                args.batch_size,
            )
            conditioned_robust[split_name] = conditioned_loss(
                host,
                conditioner,
                split.batch,
                split.robust,
                args.batch_size,
            )
            conditioned_oracle[split_name] = oracle_frame_search(
                host,
                split,
                latents,
                discovery,
                scales,
                device,
                args.oracle_batch_size,
                conditioner=conditioner,
            )
        test_opp = opportunity_masks["test"]
        test_ordinary = ~test_opp
        raw_opportunity_gain = (
            raw_recent["test"] - raw_oracle["test"].loss
        ).cpu().numpy()
        conditioned_oracle_gain = (
            conditioned_recent["test"]
            - conditioned_oracle["test"].loss
        ).cpu().numpy()
        recovery = (
            float(
                conditioned_oracle_gain[test_opp].mean()
                / max(raw_opportunity_gain[test_opp].mean(), 1e-12)
            )
            if test_opp.any()
            else None
        )
        ordinary_degradation = float(
            conditioned_recent["test"][test_ordinary].mean()
            / raw_recent["test"][test_ordinary].mean().clamp_min(1e-8)
            - 1.0
        ) if test_ordinary.any() else 0.0
        oracle_gain_summary = bootstrap_mean_ci(
            conditioned_oracle_gain[test_opp],
            splits["test"].episode_ids[test_opp],
            args.seed + 71_003,
            args.bootstrap_repetitions,
        )
        latency = mean_latency_ms(
            host,
            conditioner,
            splits["test"],
            args.latency_repetitions,
        )
        phase_b_local = bool(
            recovery is not None
            and recovery >= 0.50
            and ordinary_degradation <= 0.05
            and oracle_gain_summary["ci95"][0] is not None
            and oracle_gain_summary["ci95"][0] > 0.0
        )
        conditioner_metrics = {
            "passed_local": phase_b_local,
            "oracle_opportunity_recovery": recovery,
            "oracle_conditioned_gain": oracle_gain_summary,
            "ordinary_recent_degradation": ordinary_degradation,
            "ordinary_recent_conditioned_mse": float(
                conditioned_recent["test"][test_ordinary].mean()
            ) if test_ordinary.any() else None,
            "ordinary_recent_raw_adapter_mse": float(
                raw_recent["test"][test_ordinary].mean()
            ) if test_ordinary.any() else None,
            "no_memory_mse": float(no_memory_test.mean()),
            "zero_memory_output_fidelity_mse": (
                conditioner_training["zero_memory_distillation_mse"]
            ),
            "parameters": parameter_count(conditioner),
            "latency_ms_per_query": latency,
            "conditioner_frozen_before_gate": True,
            "conditioner_digest": conditioner_digest,
            "comparisons": {
                "frozen_raw_adapter_recent_mse": float(
                    raw_recent["test"].mean()
                ),
                "frozen_raw_adapter_memory_mse": float(
                    raw_robust["test"].mean()
                ),
                "trained_recent_mse": float(
                    conditioned_recent["test"].mean()
                ),
                "trained_robust_memory_mse": float(
                    conditioned_robust["test"].mean()
                ),
                "trained_oracle_frame_mse": float(
                    conditioned_oracle["test"].loss.mean()
                ),
            },
        }
        evaluation.update(
            {
                "loss_conditioned_recent": (
                    conditioned_recent["test"].cpu().numpy()
                ),
                "loss_conditioned_robust": (
                    conditioned_robust["test"].cpu().numpy()
                ),
                "loss_conditioned_oracle": (
                    conditioned_oracle["test"].loss.cpu().numpy()
                ),
                "conditioned_oracle_frame_index": (
                    conditioned_oracle["test"].frame_index.cpu().numpy()
                ),
            }
        )
        execute_gate = bool(
            phase_b_local or (args.smoke and args.exercise_all_phases)
        )
        if execute_gate:
            features = {
                name: gate_features(
                    host,
                    conditioner,
                    split,
                    args.batch_size,
                )
                for name, split in splits.items()
            }
            utility = {
                name: (
                    conditioned_recent[name] - conditioned_robust[name]
                ).cpu().numpy()
                / np.maximum(
                    conditioned_recent[name].cpu().numpy(),
                    1e-8,
                )
                for name in splits
            }
            gate, gate_training, feature_mean, feature_std = train_gate(
                features["train"],
                features["validation"],
                utility["train"],
                utility["validation"],
                splits["train"].episode_ids,
                args,
                device,
            )
            if tensor_digest(conditioner) != conditioner_digest:
                raise RuntimeError(
                    "frozen conditioner changed during gate training"
                )
            validation_output = evaluate_gate_output(
                gate,
                features["validation"],
                feature_mean,
                feature_std,
                device,
            )
            validation_absolute = (
                conditioned_recent["validation"]
                - conditioned_robust["validation"]
            ).cpu().numpy()
            sweep, selected = delta_sweep(
                validation_output[0],
                validation_output[1],
                utility["validation"],
                validation_absolute,
                splits["validation"].episode_ids,
                args,
                args.seed + 81_019,
            )
            test_output = evaluate_gate_output(
                gate,
                features["test"],
                feature_mean,
                feature_std,
                device,
            )
            if selected is None:
                active = np.zeros(len(splits["test"]), dtype=bool)
                selected_delta = None
            else:
                selected_delta = float(selected["delta"])
                active = test_output[0] > selected_delta
            absolute_test = (
                conditioned_recent["test"]
                - conditioned_robust["test"]
            ).cpu().numpy()
            test_selective = selective_metrics(
                active,
                utility["test"],
                absolute_test,
                splits["test"].episode_ids,
                args.seed + 91_021,
                args.bootstrap_repetitions,
            )
            oracle_active = absolute_test > 0.0
            oracle_activation = selective_metrics(
                oracle_active,
                utility["test"],
                absolute_test,
                splits["test"].episode_ids,
                args.seed + 92_021,
                args.bootstrap_repetitions,
            )
            always = selective_metrics(
                np.ones(len(active), dtype=bool),
                utility["test"],
                absolute_test,
                splits["test"].episode_ids,
                args.seed + 93_021,
                args.bootstrap_repetitions,
            )
            rng = np.random.default_rng(args.seed + 94_021)
            random_active = np.zeros(len(active), dtype=bool)
            random_count = int(active.sum())
            if random_count:
                random_active[
                    rng.choice(len(active), size=random_count, replace=False)
                ] = True
            random_metrics = selective_metrics(
                random_active,
                utility["test"],
                absolute_test,
                splits["test"].episode_ids,
                args.seed + 95_021,
                args.bootstrap_repetitions,
            )
            test_ece = calibration_ece(
                test_output[1],
                (
                    utility["test"] > args.gate_help_margin
                ).astype(np.float32),
            )
            low_quantile = test_selective[
                "low_quantile_relative_utility"
            ]
            phase_c_local = bool(
                selected is not None
                and test_selective["coverage"]
                >= args.minimum_gate_coverage
                and test_selective["activated_gain"]["ci95"][0] is not None
                and test_selective["activated_gain"]["ci95"][0] > 0.0
                and test_selective["overall_policy_gain"]["mean"] is not None
                and test_selective["overall_policy_gain"]["mean"] >= 0.0
                and low_quantile is not None
                and low_quantile
                >= -args.maximum_low_quantile_degradation
            )
            recent_np = conditioned_recent["test"].cpu().numpy()
            memory_np = conditioned_robust["test"].cpu().numpy()
            gated_np = np.where(active, memory_np, recent_np)
            gate_metrics = {
                "passed_local": phase_c_local,
                "selected_delta": selected_delta,
                "validation_selection": selected,
                "validation_sweep": sweep,
                "test": {
                    "calibrated_gate": test_selective,
                    "always_memory": always,
                    "recent_only": {
                        "mean_loss": float(recent_np.mean()),
                        "coverage": 0.0,
                    },
                    "oracle_activation": oracle_activation,
                    "random_activation": random_metrics,
                    "gated_mean_loss": float(gated_np.mean()),
                    "activation_probability_ece": test_ece,
                    "conformal_lower_bound_coverage": float(
                        np.mean(utility["test"] >= test_output[0])
                    ),
                },
                "abstention_only_failure": selected is None,
                "conditioner_frozen_digest_before": conditioner_digest,
                "conditioner_frozen_digest_after": tensor_digest(conditioner),
            }
            evaluation.update(
                {
                    "gate_lcb": test_output[0],
                    "gate_probability": test_output[1],
                    "gate_mean": test_output[2],
                    "gate_std": test_output[3],
                    "activated": active,
                    "relative_memory_utility": utility["test"],
                    "realized_memory_gain": absolute_test,
                    "loss_gated": gated_np,
                }
            )
            robust_types = (
                splits["test"].robust.token_type[:, 0].cpu().numpy()
            )
            robust_sources = torch.gather(
                splits["test"].source_index,
                1,
                splits["test"].robust_slot[:, None],
            )[:, 0].cpu().numpy()
            for row in range(len(splits["test"])):
                decision_logs.append(
                    {
                        "episode_id": int(splits["test"].episode_ids[row]),
                        "query_t": int(splits["test"].query_t[row]),
                        "gap": int(splits["test"].gaps[row]),
                        "candidate_type": TOKEN_TYPE_NAMES[
                            int(robust_types[row])
                        ],
                        "candidate_source_index": int(
                            robust_sources[row]
                        ),
                        "activation_probability": float(
                            test_output[1][row]
                        ),
                        "predicted_utility_mean": float(
                            test_output[2][row]
                        ),
                        "predicted_utility_std": float(
                            test_output[3][row]
                        ),
                        "utility_lcb": float(test_output[0][row]),
                        "delta": selected_delta,
                        "activated": bool(active[row]),
                        "abstained": bool(not active[row]),
                        "realized_memory_vs_recent_delta": float(
                            absolute_test[row]
                        ),
                        "realized_relative_utility": float(
                            utility["test"][row]
                        ),
                        "test_future_used_for_activation": False,
                        "test_future_used_for_posthoc_delta": True,
                    }
                )
            torch.save(
                {
                    "schema": "cem_native_long_model_v1",
                    "host": host.state_dict(),
                    "raw_memory": raw_memory.state_dict(),
                    "conditioner": conditioner.state_dict(),
                    "gate": gate.state_dict(),
                    "host_config": {
                        "latent_dim": int(latents.shape[-1]),
                        "action_dim": int(actions.shape[-1]),
                        "context": CONTEXT,
                        "hidden": args.host_hidden,
                    },
                    "conditioner_config": {
                        "metadata_dim": METADATA_DIM,
                        "code_dim": args.conditioner_code_dim,
                        "hidden": args.conditioner_hidden,
                        "heads": args.conditioner_heads,
                        "max_residual": args.conditioner_max_residual,
                    },
                    "gate_config": {
                        "input_dim": int(features["train"].shape[1]),
                        "hidden": args.gate_hidden,
                        "members": GATE_FOLDS,
                        "normalizer_mean": feature_mean,
                        "normalizer_std": feature_std,
                        "selected_delta": selected_delta,
                    },
                },
                output_dir / "model.pt",
            )
    if not (output_dir / "model.pt").is_file():
        torch.save(
            {
                "schema": "cem_native_long_partial_model_v1",
                "host": host.state_dict(),
                "raw_memory": raw_memory.state_dict(),
                "conditioner": (
                    conditioner.state_dict()
                    if conditioner is not None
                    else None
                ),
            },
            output_dir / "model.pt",
        )
    if not decision_logs:
        robust_types = splits["test"].robust.token_type[:, 0].cpu().numpy()
        robust_sources = torch.gather(
            splits["test"].source_index,
            1,
            splits["test"].robust_slot[:, None],
        )[:, 0].cpu().numpy()
        if "loss_conditioned_recent" in evaluation:
            realized = (
                evaluation["loss_conditioned_recent"]
                - evaluation["loss_conditioned_robust"]
            )
            reason = "Gate B hard stop"
        else:
            realized = (
                evaluation["loss_raw_recent"]
                - evaluation["loss_raw_robust"]
            )
            reason = "Gate A hard stop"
        evaluation["activated"] = np.zeros(len(realized), dtype=bool)
        evaluation["loss_gated"] = (
            evaluation.get(
                "loss_conditioned_recent",
                evaluation["loss_raw_recent"],
            )
        )
        for row in range(len(splits["test"])):
            decision_logs.append(
                {
                    "episode_id": int(splits["test"].episode_ids[row]),
                    "query_t": int(splits["test"].query_t[row]),
                    "gap": int(splits["test"].gaps[row]),
                    "candidate_type": TOKEN_TYPE_NAMES[
                        int(robust_types[row])
                    ],
                    "candidate_source_index": int(robust_sources[row]),
                    "activation_probability": None,
                    "predicted_utility_mean": None,
                    "predicted_utility_std": None,
                    "utility_lcb": None,
                    "delta": None,
                    "activated": False,
                    "abstained": True,
                    "abstention_reason": reason,
                    "realized_memory_vs_recent_delta": float(realized[row]),
                    "test_future_used_for_activation": False,
                    "test_future_used_for_posthoc_delta": True,
                }
            )
    np.savez_compressed(output_dir / "evaluation.npz", **evaluation)
    decision = {
        "schema": "cem_native_long_decisions_v1",
        "environment": args.env_name,
        "seed": args.seed,
        "default_policy": "recent_only",
        "selected_delta": selected_delta,
        "queries": decision_logs,
    }
    (output_dir / "decision_log.json").write_text(
        stable_json(json_safe(decision))
    )
    audit = source_audit()
    if not audit["passed"]:
        raise RuntimeError(f"native source audit failed: {audit}")
    if tensor_digest(host) != host_digest:
        raise RuntimeError("frozen host changed in native-long campaign")
    result = {
        "schema": "cem_native_long_cell_v1",
        "status": "completed",
        "environment": args.env_name,
        "family": env_family(args.env_name),
        "seed": args.seed,
        "device": str(device),
        "protocol": {
            "native_chronology": True,
            "controlled_splicing": False,
            "trajectory_frames": int(latents.shape[1]),
            "gaps": list(GAPS),
            "prediction_horizon": HORIZON,
            "host_context": CONTEXT,
            "memory_tokens_per_arm": MEMORY_TOKENS,
            "serialized_memory_bytes_per_arm": (
                MEMORY_TOKENS * latents.shape[-1] * 4
            ),
            "equal_recent_memory_bytes": True,
            "equal_read_tokens": True,
            "equal_host_rollout_calls": True,
            "recent_only_default": True,
            "graph_components": False,
            "smoke_exercise_override": bool(
                args.smoke and args.exercise_all_phases
            ),
        },
        "source_contract": audit,
        "split_contract": {
            "trajectory_overlap": 0,
            "train_episodes": len(split_episode_sets["train"]),
            "validation_episodes": len(split_episode_sets["validation"]),
            "test_episodes": len(split_episode_sets["test"]),
        },
        "host": {
            "training": host_training,
            "digest": host_digest,
            "frozen": True,
        },
        "discovery": {
            "signals": [
                "frozen-host surprise",
                "frozen-DINO semantic change",
                "DINO revisit and reappearance",
                "action transition/contact proxy",
                "recent-region transition",
                "DINO k-center coverage",
            ],
            "train_only_scales": scales,
            "manual_event_labels": False,
        },
        "raw_memory_training": raw_training,
        "gates": {
            "A_opportunity": phase_a,
            "B_conditioner": conditioner_metrics,
            "C_activation": gate_metrics,
            "local_all_passed": bool(
                phase_a_local and phase_b_local and phase_c_local
            ),
        },
        "conditioner_training": conditioner_training,
        "gate_training": gate_training,
        "downstream": {
            "reached_local": False,
            "reason": (
                "aggregate A-C decision required"
                if phase_a_local and phase_b_local and phase_c_local
                else "hard-stopped because a prerequisite gate failed"
            ),
            "planning_claim": False,
        },
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "evaluation": str(
                (output_dir / "evaluation.npz").relative_to(ROOT)
            ),
            "decision_log": str(
                (output_dir / "decision_log.json").relative_to(ROOT)
            ),
            "model": str((output_dir / "model.pt").relative_to(ROOT)),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    result_path.write_text(stable_json(json_safe(result)))
    print(
        stable_json(
            json_safe(
                {
                    "status": "completed",
                    "environment": args.env_name,
                    "seed": args.seed,
                    "gate_A": phase_a_local,
                    "gate_B": phase_b_local,
                    "gate_C": phase_c_local,
                    "test_opportunity_coverage": opportunities["test"][
                        "retained_fraction"
                    ],
                    "result": str(result_path.relative_to(ROOT)),
                }
            )
        ),
        flush=True,
    )
    return result


def summary_stat(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if not len(array):
        return {"mean": None, "ci95": [None, None], "count": 0}
    if len(array) == 1:
        interval = [float(array[0]), float(array[0])]
    else:
        half = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
        interval = [float(array.mean() - half), float(array.mean() + half)]
    return {
        "mean": float(array.mean()),
        "ci95": interval,
        "count": int(len(array)),
        "values": array.astype(float).tolist(),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result.get("schema") == "cem_native_long_cell_v1":
            cells.append(result)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["environment"], []).append(cell)
    environments = []
    resolved_families = set()
    for environment, rows in sorted(grouped.items()):
        gaps = []
        high_resolved = True
        for gap in GAPS:
            gap_rows = [
                next(
                    item
                    for item in row["gates"]["A_opportunity"]["splits"][
                        "test"
                    ]["by_gap"]
                    if item["gap"] == gap
                )
                for row in rows
            ]
            gains = summary_stat(
                item["oracle_gain_vs_recent"]["mean"] for item in gap_rows
            )
            coverage = summary_stat(
                item["retained_fraction"] for item in gap_rows
            )
            gaps.append(
                {
                    "gap": gap,
                    "oracle_gain_vs_recent": gains,
                    "retained_fraction": coverage,
                }
            )
            if gap >= 32 and not (
                gains["ci95"][0] is not None and gains["ci95"][0] > 0.0
            ):
                high_resolved = False
        if high_resolved:
            resolved_families.add(rows[0]["family"])
        conditioner_rows = [
            row["gates"]["B_conditioner"]
            for row in rows
            if row["gates"]["B_conditioner"] is not None
        ]
        activation_rows = [
            row["gates"]["C_activation"]
            for row in rows
            if row["gates"]["C_activation"] is not None
        ]
        environments.append(
            {
                "environment": environment,
                "family": rows[0]["family"],
                "seeds": sorted(int(row["seed"]) for row in rows),
                "gaps": gaps,
                "opportunity_coverage": summary_stat(
                    row["gates"]["A_opportunity"]["splits"]["test"][
                        "retained_fraction"
                    ]
                    for row in rows
                ),
                "conditioner_recovery": summary_stat(
                    row["oracle_opportunity_recovery"]
                    for row in conditioner_rows
                ),
                "ordinary_degradation": summary_stat(
                    row["ordinary_recent_degradation"]
                    for row in conditioner_rows
                ),
                "gate_coverage": summary_stat(
                    row["test"]["calibrated_gate"]["coverage"]
                    for row in activation_rows
                ),
                "gate_precision": summary_stat(
                    row["test"]["calibrated_gate"][
                        "precision_positive_utility"
                    ]
                    for row in activation_rows
                ),
                "gate_overall_gain": summary_stat(
                    row["test"]["calibrated_gate"]["overall_policy_gain"][
                        "mean"
                    ]
                    for row in activation_rows
                ),
                "gate_activated_gain": summary_stat(
                    row["test"]["calibrated_gate"]["activated_gain"]["mean"]
                    for row in activation_rows
                ),
                "gate_ece": summary_stat(
                    row["test"]["activation_probability_ece"]
                    for row in activation_rows
                ),
            }
        )
    opportunity_coverage = summary_stat(
        row["gates"]["A_opportunity"]["splits"]["test"]["retained_fraction"]
        for row in cells
    )
    gate_a = {
        "passed": bool(
            opportunity_coverage["mean"] is not None
            and opportunity_coverage["mean"]
            >= args.minimum_opportunity_coverage
            and len(resolved_families) >= 2
        ),
        "retained_coverage": opportunity_coverage,
        "resolved_high_gap_families": sorted(resolved_families),
        "resolved_family_count": len(resolved_families),
        "rule": (
            "oracle gain lower CI >0 at gaps 32/64/128 in >=2 families "
            "and mean retained coverage >=20%"
        ),
    }
    conditioner_cells = [
        row["gates"]["B_conditioner"]
        for row in cells
        if row["gates"]["B_conditioner"] is not None
    ]
    recovery = summary_stat(
        row["oracle_opportunity_recovery"] for row in conditioner_cells
    )
    degradation = summary_stat(
        row["ordinary_recent_degradation"] for row in conditioner_cells
    )
    gate_b = {
        "passed": bool(
            conditioner_cells
            and recovery["mean"] is not None
            and recovery["mean"] >= 0.50
            and degradation["mean"] is not None
            and degradation["mean"] <= 0.05
        ),
        "oracle_opportunity_recovery": recovery,
        "ordinary_recent_degradation": degradation,
        "rule": "recovery >=50% and ordinary degradation <=5%",
    }
    activation_cells = [
        row["gates"]["C_activation"]
        for row in cells
        if row["gates"]["C_activation"] is not None
    ]
    coverage = summary_stat(
        row["test"]["calibrated_gate"]["coverage"]
        for row in activation_cells
    )
    precision = summary_stat(
        row["test"]["calibrated_gate"]["precision_positive_utility"]
        for row in activation_cells
    )
    activated_gain = summary_stat(
        row["test"]["calibrated_gate"]["activated_gain"]["mean"]
        for row in activation_cells
    )
    overall_gain = summary_stat(
        row["test"]["calibrated_gate"]["overall_policy_gain"]["mean"]
        for row in activation_cells
    )
    low_quantile = summary_stat(
        row["test"]["calibrated_gate"]["low_quantile_relative_utility"]
        for row in activation_cells
    )
    gate_c = {
        "passed": bool(
            activation_cells
            and coverage["mean"] is not None
            and coverage["mean"] >= args.minimum_gate_coverage
            and activated_gain["ci95"][0] is not None
            and activated_gain["ci95"][0] > 0.0
            and overall_gain["mean"] is not None
            and overall_gain["mean"] >= 0.0
            and low_quantile["mean"] is not None
            and low_quantile["mean"]
            >= -args.maximum_low_quantile_degradation
        ),
        "coverage": coverage,
        "precision": precision,
        "activated_gain": activated_gain,
        "overall_policy_gain": overall_gain,
        "low_quantile_relative_utility": low_quantile,
        "abstention_only_cell_count": (
            sum(bool(row["abstention_only_failure"]) for row in activation_cells)
            + sum(
                row["gates"]["C_activation"] is None
                for row in cells
            )
        ),
        "hard_stopped_before_gate_cell_count": sum(
            row["gates"]["C_activation"] is None for row in cells
        ),
        "rule": (
            "coverage >=10%, activated gain lower CI >0, overall gain >=0, "
            "and mean 5th-percentile degradation controlled at 5%"
        ),
    }
    all_passed = bool(gate_a["passed"] and gate_b["passed"] and gate_c["passed"])
    report = {
        "schema": "cem_native_long_report_v1",
        "status": "completed" if cells else "empty",
        "cell_count": len(cells),
        "environment_count": len(environments),
        "protocol": {
            "environments": list(ENVIRONMENTS),
            "gaps": list(GAPS),
            "seeds_per_environment": 3,
            "native_chronology": True,
            "synthetic_cue_injection": False,
            "graph_revived": False,
            "recent_only_default": True,
            "matched_memory_tokens": MEMORY_TOKENS,
        },
        "source_contract": source_audit(),
        "environments": environments,
        "gates": {
            "A_opportunity": gate_a,
            "B_conditioner": gate_b,
            "C_activation": gate_c,
            "all_passed": all_passed,
        },
        "claim": (
            "selective memory activation"
            if gate_c["passed"]
            else "no selective-memory claim"
        ),
        "downstream": {
            "reached": False,
            "reason": (
                "A-C passed; downstream execution requires explicit follow-up"
                if all_passed
                else "hard-stopped because A-C did not all pass"
            ),
            "latent_action_sequence_planning_claim": False,
        },
        "jobs_still_running": [],
        "artifacts": {
            "cells": "outputs/cem_native_long_v1/cells/<env>/s<seed>",
            "report": "outputs/cem_native_long_v1/report.json",
            "machine_decision": "outputs/cem_native_long_report.json",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        stable_json(json_safe(report))
    )
    ROOT_REPORT.write_text(stable_json(json_safe(report)))
    print(stable_json(json_safe(report)), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exercise-all-phases", action="store_true")
    parser.add_argument("--host-hidden", type=int, default=256)
    parser.add_argument("--host-epochs", type=int, default=45)
    parser.add_argument("--host-patience", type=int, default=7)
    parser.add_argument("--host-lr", type=float, default=3e-4)
    parser.add_argument("--raw-memory-hidden", type=int, default=160)
    parser.add_argument("--raw-memory-epochs", type=int, default=24)
    parser.add_argument("--raw-memory-patience", type=int, default=5)
    parser.add_argument("--raw-memory-lr", type=float, default=4e-4)
    parser.add_argument("--conditioner-code-dim", type=int, default=64)
    parser.add_argument("--conditioner-hidden", type=int, default=160)
    parser.add_argument("--conditioner-heads", type=int, default=4)
    parser.add_argument("--conditioner-max-residual", type=float, default=0.75)
    parser.add_argument("--conditioner-epochs", type=int, default=32)
    parser.add_argument("--conditioner-patience", type=int, default=7)
    parser.add_argument("--conditioner-lr", type=float, default=5e-4)
    parser.add_argument("--conditioner-oracle-weight", type=float, default=0.5)
    parser.add_argument("--conditioner-effect-weight", type=float, default=20.0)
    parser.add_argument(
        "--conditioner-recovery-target",
        type=float,
        default=0.50,
    )
    parser.add_argument("--geometry-weight", type=float, default=0.02)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--covariance-weight", type=float, default=0.001)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--dual-lr", type=float, default=2.0)
    parser.add_argument("--gate-hidden", type=int, default=128)
    parser.add_argument("--gate-epochs", type=int, default=30)
    parser.add_argument("--gate-patience", type=int, default=6)
    parser.add_argument("--gate-lr", type=float, default=4e-4)
    parser.add_argument("--gate-help-margin", type=float, default=0.0)
    parser.add_argument(
        "--gate-conformal-coverage",
        type=float,
        default=0.90,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--oracle-batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--latency-repetitions", type=int, default=50)
    parser.add_argument(
        "--minimum-opportunity-effect",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--minimum-opportunity-coverage",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--minimum-gate-coverage",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-gate-precision",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--maximum-low-quantile-degradation",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        parser.error("--gpu must be one of 0,1,2; GPU3 is prohibited")
    if args.exercise_all_phases and not args.smoke:
        parser.error("--exercise-all-phases is permitted only with --smoke")
    if args.smoke:
        args.host_epochs = min(args.host_epochs, 3)
        args.host_patience = min(args.host_patience, 2)
        args.raw_memory_epochs = min(args.raw_memory_epochs, 3)
        args.raw_memory_patience = min(args.raw_memory_patience, 2)
        args.conditioner_epochs = min(args.conditioner_epochs, 4)
        args.conditioner_patience = min(args.conditioner_patience, 2)
        args.gate_epochs = min(args.gate_epochs, 4)
        args.gate_patience = min(args.gate_patience, 2)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 100)
        args.latency_repetitions = min(args.latency_repetitions, 5)
        args.oracle_batch_size = min(args.oracle_batch_size, 1024)
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
        return
    if args.env_name not in ENVIRONMENTS:
        raise ValueError(
            f"--env-name must be one of {ENVIRONMENTS}; "
            f"received {args.env_name}"
        )
    run_cell(args)


if __name__ == "__main__":
    main()
