#!/usr/bin/env python3
"""Run the equal-budget suffix-collision oracle ladder for Graph-CEM gates.

The benchmark is intentionally controlled: raw frames are temporally spliced
without pixel modification and the branch-specific future begins at a
teleport boundary. It tests whether memory opportunity and automatic event
discovery exist; it is not evidence of native environment control.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_graph_cem_long_gap import (  # noqa: E402
    EVENT_TIMES,
    FILLER_TIMES,
    FUTURE_ACTION_TIMES,
    GAPS,
    TARGET_TIMES,
)
from scripts.run_cem_conditional_ce import (  # noqa: E402
    CEFeatures,
    ConditionalCEHead,
    ConditionalTargets,
    centered_rank_correlation,
    json_safe,
    per_query_loss,
    train_conditional_head,
)
from scripts.run_cem_raw_ogbench import (  # noqa: E402
    ActionConditionedHost,
    QueryTensors,
    RawMemoryConditioner,
    horizon_loss,
    rank_correlation,
    rollout,
    set_seed,
    stable_json,
    tensor_digest,
    train_memory,
)

DEFAULT_OUTPUT = ROOT / "outputs/graph_cem_long_gap_v1"
DEFAULT_BASE_OUTPUT = ROOT / "outputs/cem_raw_ogbench"
DEFAULT_CONDITIONAL_OUTPUT = ROOT / "outputs/graph_cem_conditional_v1"
ENVIRONMENTS = (
    "pointmaze-large-navigate-v0",
    "cube-single-play-v0",
    "puzzle-3x3-play-v0",
)
CONTEXT = 2
HORIZON = 4
MEMORY_TOKENS = 4
MAX_CANDIDATES = 12
METHODS = (
    "oracle_frame",
    "oracle_automatic_node",
    "oracle_event_set",
    "surprise",
    "singleton_ce",
    "conditional_ce",
    "random",
    "recent_only",
    "no_memory",
)


@dataclass
class RawGapSplit:
    gap: int
    split: str
    history: np.ndarray
    history_actions: np.ndarray
    context_z: np.ndarray
    action_history: np.ndarray
    future_actions: np.ndarray
    targets: np.ndarray
    pair_ids: np.ndarray
    branches: np.ndarray
    sources: np.ndarray
    donors: np.ndarray
    surprise: np.ndarray | None = None
    change: np.ndarray | None = None


@dataclass
class GapSplit:
    gap: int
    split: str
    batch: QueryTensors
    history: torch.Tensor
    history_metadata: torch.Tensor
    discovery_score: torch.Tensor
    candidate_indices: torch.Tensor
    pair_ids: np.ndarray
    branches: np.ndarray


def resolve_device(gpu: int) -> torch.device:
    if gpu == 3:
        raise ValueError("GPU3 is prohibited for this campaign")
    if gpu not in (0, 1, 2):
        raise ValueError(f"GPU must be one of 0,1,2; received {gpu}")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def feature_path(base_output: Path, env_name: str) -> Path:
    return base_output / "features" / env_name / "features.npz"


def recipe_path(output: Path, env_name: str) -> Path:
    return output / "build" / env_name / "pairs.npz"


def cell_dir(output: Path, env_name: str, seed: int) -> Path:
    return output / "cells" / env_name / f"s{seed}"


def load_host(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ActionConditionedHost, dict[str, Any]]:
    candidates = [
        args.base_output
        / "cells"
        / args.env_name
        / f"s{args.seed}"
        / "model.pt",
        args.conditional_output
        / "cells"
        / args.env_name
        / f"s{args.seed}"
        / "base_model.pt",
    ]
    checkpoint_path = next((path for path in candidates if path.is_file()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"no seed-{args.seed} host checkpoint in {candidates}"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    config = checkpoint["host_config"]
    host = ActionConditionedHost(
        config["latent_dim"],
        config["action_dim"],
        config["context"],
        config["hidden"],
    ).to(device)
    host.load_state_dict(checkpoint["host"])
    host.eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    return host, {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "digest": tensor_digest(host),
        "config": config,
        "frozen": True,
    }


def load_sources(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    with np.load(
        feature_path(args.base_output, args.env_name), allow_pickle=False
    ) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    with np.load(recipe_path(args.output, args.env_name), allow_pickle=False) as data:
        recipes = {
            split: (
                np.asarray(data[f"{split}_sources"], dtype=np.int64),
                np.asarray(data[f"{split}_donors"], dtype=np.int64),
            )
            for split in ("train", "validation", "test")
        }
    return latents, actions, recipes


def build_raw_gap(
    latents: np.ndarray,
    actions: np.ndarray,
    sources: np.ndarray,
    donors: np.ndarray,
    split: str,
    gap: int,
    limit_pairs: int | None = None,
) -> RawGapSplit:
    if limit_pairs is not None:
        sources = sources[:limit_pairs]
        donors = donors[:limit_pairs]
    rows = []
    for pair_id, ((first, second), donor) in enumerate(zip(sources, donors)):
        for branch, source in enumerate((first, second)):
            history = [
                latents[int(donor), 0],
                latents[int(donor), 1],
                latents[int(source), EVENT_TIMES[0]],
                latents[int(source), EVENT_TIMES[1]],
            ]
            history_actions = [
                actions[int(donor), 0],
                actions[int(donor), 1],
                actions[int(source), EVENT_TIMES[0]],
                actions[int(source), EVENT_TIMES[1]],
            ]
            for offset in range(gap):
                source_time = FILLER_TIMES[offset % len(FILLER_TIMES)]
                history.append(latents[int(donor), source_time])
                history_actions.append(actions[int(donor), source_time])
            history_np = np.asarray(history, dtype=np.float32)
            action_np = np.asarray(history_actions, dtype=np.float32)
            rows.append(
                {
                    "history": history_np,
                    "history_actions": action_np,
                    "context_z": history_np[-CONTEXT:],
                    "action_history": action_np[-CONTEXT:],
                    "future_actions": actions[
                        int(donor), FUTURE_ACTION_TIMES
                    ],
                    "targets": latents[int(source), TARGET_TIMES],
                    "pair_id": pair_id,
                    "branch": branch,
                    "source": int(source),
                    "donor": int(donor),
                }
            )
    return RawGapSplit(
        gap=gap,
        split=split,
        history=np.asarray([row["history"] for row in rows], np.float32),
        history_actions=np.asarray(
            [row["history_actions"] for row in rows], np.float32
        ),
        context_z=np.asarray([row["context_z"] for row in rows], np.float32),
        action_history=np.asarray(
            [row["action_history"] for row in rows], np.float32
        ),
        future_actions=np.asarray(
            [row["future_actions"] for row in rows], np.float32
        ),
        targets=np.asarray([row["targets"] for row in rows], np.float32),
        pair_ids=np.asarray([row["pair_id"] for row in rows], np.int64),
        branches=np.asarray([row["branch"] for row in rows], np.int64),
        sources=np.asarray([row["source"] for row in rows], np.int64),
        donors=np.asarray([row["donor"] for row in rows], np.int64),
    )


@torch.no_grad()
def annotate_surprise(
    host: ActionConditionedHost,
    raw: RawGapSplit,
    device: torch.device,
    batch_size: int,
) -> None:
    count, steps = raw.history.shape[:2]
    surprise = np.zeros((count, steps), dtype=np.float32)
    contexts = []
    context_actions = []
    targets = []
    locations = []
    for row in range(count):
        for time_index in range(CONTEXT - 1, steps - 1):
            contexts.append(
                raw.history[
                    row,
                    time_index - CONTEXT + 1 : time_index + 1,
                ]
            )
            context_actions.append(
                raw.history_actions[
                    row,
                    time_index - CONTEXT + 1 : time_index + 1,
                ]
            )
            targets.append(raw.history[row, time_index + 1])
            locations.append((row, time_index + 1))
    x = torch.from_numpy(np.asarray(contexts, np.float32)).to(device)
    action = torch.from_numpy(
        np.asarray(context_actions, np.float32)
    ).to(device)
    target = torch.from_numpy(np.asarray(targets, np.float32)).to(device)
    values = []
    for start in range(0, len(x), batch_size):
        prediction = host(
            x[start : start + batch_size],
            action[start : start + batch_size],
        )
        values.extend(
            (prediction - target[start : start + batch_size])
            .square()
            .mean(-1)
            .cpu()
            .numpy()
            .tolist()
        )
    for (row, time_index), value in zip(locations, values):
        surprise[row, time_index] = float(value)
    change = np.zeros_like(surprise)
    change[:, 1:] = np.mean(
        np.square(raw.history[:, 1:] - raw.history[:, :-1]), axis=-1
    )
    raw.surprise = surprise
    raw.change = change


def discovery_thresholds(raw: RawGapSplit) -> tuple[float, float]:
    if raw.surprise is None or raw.change is None:
        raise ValueError("raw gap has not been annotated")
    legal = raw.history.shape[1] - CONTEXT
    surprise = raw.surprise[:, :legal]
    change = raw.change[:, 1:legal]
    return (
        max(1e-8, float(np.quantile(surprise, 0.75))),
        max(1e-8, float(np.quantile(change, 0.75))),
    )


def tensorize_gap(
    raw: RawGapSplit,
    thresholds: tuple[float, float],
    device: torch.device,
) -> GapSplit:
    if raw.surprise is None or raw.change is None:
        raise ValueError("raw gap has not been annotated")
    count, steps, latent_dim = raw.history.shape
    legal = steps - CONTEXT
    score = np.maximum(
        raw.surprise / thresholds[0], raw.change / thresholds[1]
    )
    score[:, legal:] = -np.inf
    candidate_count = min(MAX_CANDIDATES, legal)
    candidate_indices = np.argsort(
        score[:, :legal], axis=1
    )[:, -candidate_count:][:, ::-1].copy()
    events = np.zeros(
        (count, MAX_CANDIDATES, latent_dim), dtype=np.float32
    )
    metadata = np.zeros((count, MAX_CANDIDATES, 3), dtype=np.float32)
    valid = np.zeros((count, MAX_CANDIDATES), dtype=bool)
    padded_indices = np.zeros(
        (count, MAX_CANDIDATES), dtype=np.int64
    )
    for row in range(count):
        for column, time_index in enumerate(candidate_indices[row]):
            events[row, column] = raw.history[row, time_index]
            metadata[row, column] = (
                (legal - int(time_index)) / max(1, legal),
                math.log1p(max(0.0, float(raw.surprise[row, time_index]))),
                math.log1p(max(0.0, float(raw.change[row, time_index]))),
            )
            valid[row, column] = True
            padded_indices[row, column] = int(time_index)
    batch = QueryTensors(
        context_z=torch.from_numpy(raw.context_z).to(device),
        action_history=torch.from_numpy(raw.action_history).to(device),
        future_actions=torch.from_numpy(raw.future_actions).to(device),
        targets=torch.from_numpy(raw.targets).to(device),
        events=torch.from_numpy(events).to(device),
        metadata=torch.from_numpy(metadata).to(device),
        valid=torch.from_numpy(valid).to(device),
        recent_event=torch.from_numpy(
            raw.history[:, legal - 1 : legal]
        ).to(device),
    )
    history_metadata = np.zeros((count, steps, 3), dtype=np.float32)
    for time_index in range(steps):
        history_metadata[:, time_index, 0] = (
            legal - time_index
        ) / max(1, legal)
        history_metadata[:, time_index, 1] = np.log1p(
            np.maximum(0.0, raw.surprise[:, time_index])
        )
        history_metadata[:, time_index, 2] = np.log1p(
            np.maximum(0.0, raw.change[:, time_index])
        )
    return GapSplit(
        gap=raw.gap,
        split=raw.split,
        batch=batch,
        history=torch.from_numpy(raw.history).to(device),
        history_metadata=torch.from_numpy(history_metadata).to(device),
        discovery_score=torch.from_numpy(score).to(device),
        candidate_indices=torch.from_numpy(padded_indices).to(device),
        pair_ids=raw.pair_ids,
        branches=raw.branches,
    )


def concatenate_batches(batches: list[QueryTensors]) -> QueryTensors:
    return QueryTensors(
        **{
            field: torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in QueryTensors.__dataclass_fields__
        }
    )


def memory_training_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        memory_hidden=args.memory_hidden,
        max_events=MAX_CANDIDATES,
        memory_lr=args.memory_lr,
        memory_epochs=args.memory_epochs,
        memory_patience=args.memory_patience,
        residual_cost=1e-3,
        batch_size=args.batch_size,
    )


@torch.no_grad()
def route_topk(
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    budget: int,
    excluded: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = batch.valid.clone()
    if excluded is not None:
        valid &= ~excluded
    score = memory.scores(
        batch.context_z,
        batch.future_actions[:, 0],
        batch.events,
        batch.metadata,
    ).masked_fill(~valid, -1e9)
    selected = torch.zeros_like(valid)
    k = min(budget, valid.shape[1])
    indices = torch.topk(score, k=k, dim=1).indices
    selected.scatter_(1, indices, True)
    selected &= valid
    return selected


@torch.no_grad()
def compute_selection_targets(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    batch_size: int,
) -> tuple[ConditionalTargets, ConditionalTargets]:
    rows, slots = batch.valid.shape
    empty_loss = per_query_loss(
        host, None, batch, batch_size=batch_size
    )
    full_mask = route_topk(memory, batch, MEMORY_TOKENS)
    full_loss = per_query_loss(
        host, memory, batch, full_mask, batch_size=batch_size
    )
    singleton = torch.zeros((rows, slots), device=batch.events.device)
    conditional = torch.zeros_like(singleton)
    for slot in range(slots):
        keep = torch.zeros_like(batch.valid)
        keep[:, slot] = batch.valid[:, slot]
        kept_loss = per_query_loss(
            host, memory, batch, keep, batch_size=batch_size
        )
        singleton[:, slot] = (empty_loss - kept_loss) / empty_loss.clamp_min(
            1e-8
        )
        excluded = torch.zeros_like(batch.valid)
        excluded[:, slot] = batch.valid[:, slot]
        rerouted = route_topk(
            memory, batch, MEMORY_TOKENS, excluded=excluded
        )
        deleted_loss = per_query_loss(
            host, memory, batch, rerouted, batch_size=batch_size
        )
        conditional[:, slot] = (
            deleted_loss - full_loss
        ) / empty_loss.clamp_min(1e-8)
    pair = torch.zeros(
        (rows, slots, slots), device=batch.events.device
    )
    pair_valid = torch.zeros(
        (rows, slots, slots), dtype=torch.bool, device=batch.events.device
    )
    common = {
        "pair": pair,
        "candidate": batch.valid,
        "pair_valid": pair_valid,
        "full_mask": full_mask,
        "full_loss": full_loss,
        "empty_loss": empty_loss,
    }
    singleton_targets = ConditionalTargets(
        singleton=singleton.masked_fill(~batch.valid, 0.0),
        telemetry={
            "definition": "[L(empty)-L(single candidate)]/L(empty)",
            "candidate_count": int(batch.valid.sum()),
        },
        **common,
    )
    conditional_targets = ConditionalTargets(
        singleton=conditional.masked_fill(~batch.valid, 0.0),
        telemetry={
            "definition": (
                "[L(top4 rerouted after candidate deletion)-"
                "L(full top4 route)]/L(empty)"
            ),
            "candidate_count": int(batch.valid.sum()),
            "selected_count": int(full_mask.sum()),
            "router_rerun_after_deletion": True,
        },
        **common,
    )
    return singleton_targets, conditional_targets


@torch.no_grad()
def selection_features(
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    selected: torch.Tensor,
) -> CEFeatures:
    query = memory.query_vector(
        batch.context_z, batch.future_actions[:, 0]
    )
    route = memory.scores(
        batch.context_z,
        batch.future_actions[:, 0],
        batch.events,
        batch.metadata,
    )
    count = batch.valid.sum(1, keepdim=True).clamp_min(1)
    store_mean = (
        batch.events * batch.valid.unsqueeze(-1)
    ).sum(1) / count
    query_expand = query[:, None].expand(-1, batch.events.shape[1], -1)
    mean_expand = store_mean[:, None].expand_as(batch.events)
    centered_route = route - (
        route.masked_fill(~batch.valid, 0.0).sum(1, keepdim=True) / count
    )
    item = torch.cat(
        [
            batch.events,
            query_expand,
            mean_expand,
            batch.events - mean_expand,
            batch.metadata,
            centered_route.unsqueeze(-1),
            selected.unsqueeze(-1).float(),
        ],
        dim=-1,
    ).detach()
    return CEFeatures(item=item, candidate=batch.valid)


def conditional_training_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        conditional_lr=args.ce_lr,
        conditional_epochs=args.ce_epochs,
        conditional_patience=args.ce_patience,
        batch_size=args.batch_size,
    )


@torch.no_grad()
def select_by_head(
    head: ConditionalCEHead,
    features: CEFeatures,
    valid: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    mean, _, _ = head(features.item)
    score = mean.masked_fill(~valid, -1e9)
    indices = torch.topk(
        score, k=min(budget, score.shape[1]), dim=1
    ).indices
    selected = torch.zeros_like(valid)
    selected.scatter_(1, indices, True)
    return selected & valid


def indices_to_events(
    gap: GapSplit,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_dim = gap.history.shape[-1]
    gather_event = indices.unsqueeze(-1).expand(-1, -1, latent_dim)
    events = torch.gather(gap.history, 1, gather_event)
    gather_meta = indices.unsqueeze(-1).expand(-1, -1, 3)
    metadata = torch.gather(gap.history_metadata, 1, gather_meta)
    return events, metadata


def candidate_positions_to_history(
    gap: GapSplit,
    positions: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(gap.candidate_indices, 1, positions)


@torch.no_grad()
def loss_for_history_indices(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    gap: GapSplit,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    events, metadata = indices_to_events(gap, indices)
    return loss_for_custom_events(
        host, memory, gap.batch, events, metadata, batch_size
    )


@torch.no_grad()
def loss_for_custom_events(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    events: torch.Tensor,
    metadata: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    values = []
    mask = torch.ones(
        events.shape[:2], dtype=torch.bool, device=events.device
    )
    for start in range(0, len(batch), batch_size):
        stop = min(len(batch), start + batch_size)
        index = torch.arange(start, stop, device=events.device)
        part = batch.index(index)
        prediction, _ = rollout(
            host,
            memory,
            part,
            events=events[start:stop],
            metadata=metadata[start:stop],
            mask=mask[start:stop],
        )
        values.append(horizon_loss(prediction, part.targets).mean(1))
    return torch.cat(values)


def repeat_batch_for_sets(
    batch: QueryTensors,
    repeats: int,
) -> QueryTensors:
    def repeated(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(
            -1, repeats, *value.shape[1:]
        ).reshape(-1, *value.shape[1:])

    return QueryTensors(
        **{
            field: repeated(getattr(batch, field))
            for field in QueryTensors.__dataclass_fields__
        }
    )


@torch.no_grad()
def evaluate_index_sets(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    gap: GapSplit,
    index_sets: torch.Tensor,
    batch_size: int,
    set_chunk: int = 32,
) -> torch.Tensor:
    """Evaluate N x K equal-token candidate sets and return N x K losses."""
    count, set_count, token_count = index_sets.shape
    output = []
    for start in range(0, set_count, set_chunk):
        stop = min(set_count, start + set_chunk)
        chunk_count = stop - start
        indices = index_sets[:, start:stop]
        flat_indices = indices.reshape(count * chunk_count, token_count)
        repeated_gap = GapSplit(
            gap=gap.gap,
            split=gap.split,
            batch=repeat_batch_for_sets(gap.batch, chunk_count),
            history=gap.history[:, None]
            .expand(-1, chunk_count, -1, -1)
            .reshape(count * chunk_count, *gap.history.shape[1:]),
            history_metadata=gap.history_metadata[:, None]
            .expand(-1, chunk_count, -1, -1)
            .reshape(
                count * chunk_count, *gap.history_metadata.shape[1:]
            ),
            discovery_score=gap.discovery_score,
            candidate_indices=gap.candidate_indices,
            pair_ids=gap.pair_ids,
            branches=gap.branches,
        )
        losses = loss_for_history_indices(
            host,
            memory,
            repeated_gap,
            flat_indices,
            batch_size,
        ).reshape(count, chunk_count)
        output.append(losses)
    return torch.cat(output, dim=1)


def recent_indices(gap: GapSplit) -> torch.Tensor:
    legal = gap.history.shape[1] - CONTEXT
    return torch.arange(
        legal - MEMORY_TOKENS,
        legal,
        device=gap.history.device,
    )[None].expand(len(gap.batch), -1)


def selector_indices_from_mask(
    gap: GapSplit,
    mask: torch.Tensor,
) -> torch.Tensor:
    positions = torch.topk(
        mask.float(), k=MEMORY_TOKENS, dim=1
    ).indices
    return candidate_positions_to_history(gap, positions)


def bootstrap_pair_mean(
    values: np.ndarray,
    pair_ids: np.ndarray,
    seed: int,
    repetitions: int = 2000,
) -> dict[str, Any]:
    unique = np.unique(pair_ids)
    pair_values = np.asarray(
        [np.mean(values[pair_ids == pair_id]) for pair_id in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(
                pair_values, size=len(pair_values), replace=True
            ).mean()
            for _ in range(repetitions)
        ],
        dtype=np.float64,
    )
    return {
        "mean": float(pair_values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).astype(float).tolist(),
        "pair_count": int(len(pair_values)),
        "bootstrap_unit": "suffix-collision pair",
    }


@torch.no_grad()
def evaluate_gap(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    singleton_head: ConditionalCEHead,
    conditional_head: ConditionalCEHead,
    gap: GapSplit,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = len(gap.batch)
    legal = gap.history.shape[1] - CONTEXT
    recent = recent_indices(gap)
    recent_loss = loss_for_history_indices(
        host, memory, gap, recent, args.batch_size
    )
    empty_loss = per_query_loss(
        host, None, gap.batch, batch_size=args.batch_size
    )
    auto_valid = gap.batch.valid
    route_selected = route_topk(memory, gap.batch, MEMORY_TOKENS)
    features = selection_features(memory, gap.batch, route_selected)
    singleton_mask = select_by_head(
        singleton_head, features, auto_valid, MEMORY_TOKENS
    )
    conditional_mask = select_by_head(
        conditional_head, features, auto_valid, MEMORY_TOKENS
    )
    surprise_positions = torch.arange(
        MEMORY_TOKENS, device=gap.history.device
    )[None].expand(count, -1)
    surprise_indices = candidate_positions_to_history(
        gap, surprise_positions
    )
    singleton_indices = selector_indices_from_mask(gap, singleton_mask)
    conditional_indices = selector_indices_from_mask(gap, conditional_mask)
    generator = torch.Generator(device=gap.history.device)
    generator.manual_seed(args.seed * 100_003 + gap.gap)
    random_score = torch.rand(
        auto_valid.shape,
        generator=generator,
        device=gap.history.device,
    ).masked_fill(~auto_valid, -1.0)
    random_positions = torch.topk(
        random_score, k=MEMORY_TOKENS, dim=1
    ).indices
    random_indices = candidate_positions_to_history(
        gap, random_positions
    )
    losses = {
        "recent_only": recent_loss,
        "no_memory": empty_loss,
        "surprise": loss_for_history_indices(
            host, memory, gap, surprise_indices, args.batch_size
        ),
        "singleton_ce": loss_for_history_indices(
            host, memory, gap, singleton_indices, args.batch_size
        ),
        "conditional_ce": loss_for_history_indices(
            host, memory, gap, conditional_indices, args.batch_size
        ),
        "random": loss_for_history_indices(
            host, memory, gap, random_indices, args.batch_size
        ),
    }
    recent_filler = recent[:, -3:]
    old_count = legal - MEMORY_TOKENS
    old = torch.arange(
        old_count, device=gap.history.device
    )[None, :, None].expand(count, -1, -1)
    frame_sets = torch.cat(
        [
            old,
            recent_filler[:, None].expand(-1, old_count, -1),
        ],
        dim=2,
    )
    frame_grid_loss = evaluate_index_sets(
        host,
        memory,
        gap,
        frame_sets,
        args.batch_size,
        set_chunk=args.oracle_set_chunk,
    )
    oracle_frame_loss, oracle_frame_choice = frame_grid_loss.min(1)
    auto_count = int(auto_valid[0].sum())
    auto_history = gap.candidate_indices[:, :auto_count]
    old_auto_mask = auto_history < (legal - MEMORY_TOKENS)
    if not bool(old_auto_mask.all()):
        sentinel = torch.zeros_like(auto_history)
        old_auto = torch.where(old_auto_mask, auto_history, sentinel)
    else:
        old_auto = auto_history
    auto_sets = torch.cat(
        [
            old_auto.unsqueeze(-1),
            recent_filler[:, None].expand(-1, auto_count, -1),
        ],
        dim=2,
    )
    auto_grid_loss = evaluate_index_sets(
        host,
        memory,
        gap,
        auto_sets,
        args.batch_size,
        set_chunk=args.oracle_set_chunk,
    ).masked_fill(~old_auto_mask, float("inf"))
    oracle_auto_loss, oracle_auto_choice = auto_grid_loss.min(1)
    combinations = torch.as_tensor(
        list(itertools.combinations(range(auto_count), MEMORY_TOKENS)),
        dtype=torch.long,
        device=gap.history.device,
    )
    event_sets = torch.gather(
        auto_history[:, None].expand(-1, len(combinations), -1),
        2,
        combinations[None].expand(count, -1, -1),
    )
    event_grid_loss = evaluate_index_sets(
        host,
        memory,
        gap,
        event_sets,
        args.batch_size,
        set_chunk=args.oracle_set_chunk,
    )
    oracle_event_loss, oracle_event_choice = event_grid_loss.min(1)
    losses.update(
        {
            "oracle_frame": oracle_frame_loss,
            "oracle_automatic_node": oracle_auto_loss,
            "oracle_event_set": oracle_event_loss,
        }
    )
    recent_np = recent_loss.cpu().numpy()
    method_metrics = {}
    arrays: dict[str, np.ndarray] = {
        "gap": np.full(count, gap.gap, dtype=np.int64),
        "pair_id": gap.pair_ids.astype(np.int64),
        "branch": gap.branches.astype(np.int64),
    }
    for method in METHODS:
        value = losses[method].detach().cpu().numpy()
        arrays[f"loss_{method}"] = value.astype(np.float32)
        gain = recent_np - value
        method_metrics[method] = {
            "mean_host_future_loss": float(value.mean()),
            "paired_gain_vs_recent": bootstrap_pair_mean(
                gain,
                gap.pair_ids,
                args.seed * 10_000 + gap.gap + METHODS.index(method),
                repetitions=args.bootstrap_repetitions,
            ),
        }
    frame_gain = recent_np - arrays["loss_oracle_frame"]
    auto_gain = recent_np - arrays["loss_oracle_automatic_node"]
    event_gain = recent_np - arrays["loss_oracle_event_set"]
    conditional_gain = recent_np - arrays["loss_conditional_ce"]
    recovery = float(auto_gain.mean() / max(frame_gain.mean(), 1e-12))
    selection_closure = float(
        conditional_gain.mean() / max(event_gain.mean(), 1e-12)
    )
    metrics = {
        "gap": gap.gap,
        "example_count": count,
        "pair_count": int(len(np.unique(gap.pair_ids))),
        "methods": method_metrics,
        "opportunity_gap_oracle_frame_vs_recent": float(frame_gain.mean()),
        "automatic_node_recovery_of_oracle_frame_gain": recovery,
        "conditional_selection_closure_of_oracle_event_set": selection_closure,
        "oracle_search": {
            "frame_candidates_per_example": int(old_count),
            "automatic_node_candidates_per_example": int(
                old_auto_mask.sum(1).float().mean()
            ),
            "event_sets_per_example": int(len(combinations)),
            "realized_future_used_for_selection": True,
            "primary_inference_host_calls_after_selection": 1,
            "primary_read_tokens": MEMORY_TOKENS,
        },
        "oracle_choices": {
            "frame_mean_index": float(oracle_frame_choice.float().mean()),
            "automatic_mean_candidate_position": float(
                oracle_auto_choice.float().mean()
            ),
            "event_set_mean_combination_index": float(
                oracle_event_choice.float().mean()
            ),
        },
    }
    return metrics, arrays


def suffix_diagnostics(
    train_raw: dict[int, RawGapSplit],
    test_raw: dict[int, RawGapSplit],
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    maximum_latent_difference = 0.0
    maximum_action_difference = 0.0
    exact_pairs = 0
    total_pairs = 0
    for gap in GAPS:
        for raw, xs, ys in (
            (train_raw[gap], train_x, train_y),
            (test_raw[gap], test_x, test_y),
        ):
            suffix = np.concatenate(
                [
                    raw.history[:, -6:].reshape(len(raw.history), -1),
                    raw.history_actions[:, -6:].reshape(
                        len(raw.history_actions), -1
                    ),
                ],
                axis=1,
            )
            xs.extend(suffix)
            ys.extend(raw.branches)
        raw = test_raw[gap]
        for pair_id in np.unique(raw.pair_ids):
            rows = np.flatnonzero(raw.pair_ids == pair_id)
            if len(rows) != 2:
                continue
            left, right = rows
            latent_delta = float(
                np.max(
                    np.abs(
                        raw.history[left, -6:]
                        - raw.history[right, -6:]
                    )
                )
            )
            action_delta = float(
                np.max(
                    np.abs(
                        raw.history_actions[left, -6:]
                        - raw.history_actions[right, -6:]
                    )
                )
            )
            maximum_latent_difference = max(
                maximum_latent_difference, latent_delta
            )
            maximum_action_difference = max(
                maximum_action_difference, action_delta
            )
            exact_pairs += int(latent_delta == 0.0 and action_delta == 0.0)
            total_pairs += 1
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )
    model.fit(np.asarray(train_x, np.float32), np.asarray(train_y, np.int64))
    prediction = model.predict(np.asarray(test_x, np.float32))
    truth = np.asarray(test_y, np.int64)
    balanced_accuracy = float(
        0.5
        * (
            np.mean(prediction[truth == 0] == 0)
            + np.mean(prediction[truth == 1] == 1)
        )
    )
    return {
        "suffix_frames": 6,
        "maximum_paired_latent_absolute_difference": (
            maximum_latent_difference
        ),
        "maximum_paired_action_absolute_difference": (
            maximum_action_difference
        ),
        "exact_pair_fraction": exact_pairs / max(1, total_pairs),
        "posthoc_linear_branch_balanced_accuracy": balanced_accuracy,
        "branch_labels_used_for_training_memory_or_selection": False,
        "interpretation": (
            "evaluator-only distinguishability audit; exact paired suffixes "
            "make branch identity unavailable to any recent-only function"
        ),
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = cell_dir(args.output, args.env_name, args.seed)
    result_path = output_dir / "result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    set_seed(args.seed)
    device = resolve_device(args.gpu)
    host, host_receipt = load_host(args, device)
    host_digest_before = tensor_digest(host)
    latents, actions, recipes = load_sources(args)
    raw: dict[str, dict[int, RawGapSplit]] = {
        split: {} for split in ("train", "validation", "test")
    }
    tensorized: dict[str, dict[int, GapSplit]] = {
        split: {} for split in raw
    }
    thresholds = {}
    for gap in GAPS:
        for split in raw:
            sources, donors = recipes[split]
            raw[split][gap] = build_raw_gap(
                latents,
                actions,
                sources,
                donors,
                split,
                gap,
                limit_pairs=args.limit_pairs,
            )
            annotate_surprise(
                host, raw[split][gap], device, args.discovery_batch_size
            )
        thresholds[gap] = discovery_thresholds(raw["train"][gap])
        for split in raw:
            tensorized[split][gap] = tensorize_gap(
                raw[split][gap], thresholds[gap], device
            )
    train_batch = concatenate_batches(
        [tensorized["train"][gap].batch for gap in GAPS]
    )
    validation_batch = concatenate_batches(
        [tensorized["validation"][gap].batch for gap in GAPS]
    )
    memory, memory_training = train_memory(
        host,
        train_batch,
        validation_batch,
        memory_training_args(args),
        device,
    )
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    memory.eval()
    singleton_train, conditional_train = compute_selection_targets(
        host, memory, train_batch, args.batch_size
    )
    singleton_validation, conditional_validation = compute_selection_targets(
        host, memory, validation_batch, args.batch_size
    )
    train_selected = route_topk(memory, train_batch, MEMORY_TOKENS)
    validation_selected = route_topk(
        memory, validation_batch, MEMORY_TOKENS
    )
    train_features = selection_features(
        memory, train_batch, train_selected
    )
    validation_features = selection_features(
        memory, validation_batch, validation_selected
    )
    head_args = conditional_training_args(args)
    singleton_head = ConditionalCEHead(
        train_features.item.shape[-1], hidden=args.ce_hidden
    ).to(device)
    singleton_training = train_conditional_head(
        singleton_head,
        train_features,
        singleton_train,
        validation_features,
        singleton_validation,
        head_args,
    )
    conditional_head = ConditionalCEHead(
        train_features.item.shape[-1], hidden=args.ce_hidden
    ).to(device)
    conditional_training = train_conditional_head(
        conditional_head,
        train_features,
        conditional_train,
        validation_features,
        conditional_validation,
        head_args,
    )
    gap_results = []
    evaluation_arrays = []
    for gap in GAPS:
        metrics, arrays = evaluate_gap(
            host,
            memory,
            singleton_head,
            conditional_head,
            tensorized["test"][gap],
            args,
        )
        gap_results.append(metrics)
        evaluation_arrays.append(arrays)
    suffix = suffix_diagnostics(raw["train"], raw["test"])
    evaluation_path = output_dir / "evaluation.npz"
    combined_arrays = {
        key: np.concatenate([row[key] for row in evaluation_arrays])
        for key in evaluation_arrays[0]
    }
    np.savez_compressed(evaluation_path, **combined_arrays)
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "schema": "graph_cem_long_gap_model_v1",
            "memory": memory.state_dict(),
            "singleton_head": singleton_head.state_dict(),
            "conditional_head": conditional_head.state_dict(),
            "memory_config": {
                "latent_dim": int(latents.shape[-1]),
                "action_dim": int(actions.shape[-1]),
                "hidden": args.memory_hidden,
                "budget": MAX_CANDIDATES,
            },
            "ce_config": {
                "input_dim": int(train_features.item.shape[-1]),
                "hidden": args.ce_hidden,
            },
        },
        checkpoint_path,
    )
    if tensor_digest(host) != host_digest_before:
        raise RuntimeError("frozen host changed during long-gap training")
    result = {
        "schema": "graph_cem_long_gap_cell_v1",
        "status": "completed",
        "environment": args.env_name,
        "seed": args.seed,
        "device": str(device),
        "protocol": {
            "controlled_splicing": True,
            "controlled_query_future_teleport": True,
            "raw_frames_modified": False,
            "native_rollout_claim": False,
            "training_cue_labels": False,
            "training_cue_times": False,
            "gaps": list(GAPS),
            "memory_tokens_per_arm": MEMORY_TOKENS,
            "serialized_memory_bytes_per_arm": (
                MEMORY_TOKENS * latents.shape[-1] * 4
            ),
            "read_tokens_per_arm": MEMORY_TOKENS,
            "online_host_calls_per_arm": 1,
            "recent_baseline_uses_newest_legal_tokens": True,
            "oracle_selection_uses_realized_future": True,
        },
        "host": host_receipt,
        "discovery": {
            "method": (
                "top frozen-host surprise or frozen-DINO temporal-change "
                "score with train-only 75th-percentile scales"
            ),
            "thresholds": {
                str(gap): {
                    "surprise": thresholds[gap][0],
                    "semantic_change": thresholds[gap][1],
                }
                for gap in GAPS
            },
            "maximum_candidates": MAX_CANDIDATES,
            "manual_event_times_used_by_discovery_or_selection": False,
            "task_builder_event_times_evaluator_only": True,
        },
        "memory_training": memory_training,
        "singleton_ce_training": singleton_training,
        "conditional_ce_training": conditional_training,
        "selection_targets": {
            "singleton": singleton_train.telemetry,
            "conditional": conditional_train.telemetry,
        },
        "recent_suffix_distinguishability": suffix,
        "gaps": gap_results,
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "evaluation": str(evaluation_path.relative_to(ROOT)),
            "model": str(checkpoint_path.relative_to(ROOT)),
            "build_receipt": str(
                (
                    args.output
                    / "build"
                    / args.env_name
                    / "receipt.json"
                ).relative_to(ROOT)
            ),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    result_path.write_text(stable_json(json_safe(result)))
    print(
        stable_json(
            {
                "status": "completed",
                "environment": args.env_name,
                "seed": args.seed,
                "gap_32_oracle_gain": next(
                    row["opportunity_gap_oracle_frame_vs_recent"]
                    for row in gap_results
                    if row["gap"] == 32
                ),
                "gap_32_automatic_recovery": next(
                    row["automatic_node_recovery_of_oracle_frame_gain"]
                    for row in gap_results
                    if row["gap"] == 32
                ),
                "result": str(result_path.relative_to(ROOT)),
            }
        ),
        flush=True,
    )
    return result


def bootstrap_env_gap(
    cells: list[dict[str, Any]],
    gap: int,
    method: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    loaded = []
    for cell in cells:
        with np.load(
            ROOT / cell["artifacts"]["evaluation"], allow_pickle=False
        ) as data:
            keep = np.asarray(data["gap"]) == gap
            loaded.append(
                {
                    "pair_id": np.asarray(data["pair_id"])[keep],
                    "recent": np.asarray(data["loss_recent_only"])[keep],
                    "method": np.asarray(data[f"loss_{method}"])[keep],
                }
            )
    seed_values = []
    for arrays in loaded:
        gain = arrays["recent"] - arrays["method"]
        pair_values = [
            gain[arrays["pair_id"] == pair_id].mean()
            for pair_id in np.unique(arrays["pair_id"])
        ]
        seed_values.append(float(np.mean(pair_values)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(len(loaded), size=len(loaded), replace=True)
        seed_draws = []
        for seed_index in chosen:
            arrays = loaded[int(seed_index)]
            pair_ids = np.unique(arrays["pair_id"])
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            gain = arrays["recent"] - arrays["method"]
            seed_draws.append(
                float(
                    np.mean(
                        [
                            gain[arrays["pair_id"] == pair_id].mean()
                            for pair_id in sampled
                        ]
                    )
                )
            )
        draws.append(float(np.mean(seed_draws)))
    draws_np = np.asarray(draws, dtype=np.float64)
    return {
        "paired_gain_vs_recent": float(np.mean(seed_values)),
        "ci95": np.quantile(
            draws_np, [0.025, 0.975]
        ).astype(float).tolist(),
        "seed_values": seed_values,
        "seed_count": len(loaded),
        "bootstrap_unit": "optimization seed, then suffix-collision pair",
        "_draws": draws_np,
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result.get("schema") == "graph_cem_long_gap_cell_v1":
            cells.append(result)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["environment"], []).append(cell)
    environments = []
    global_draws: dict[tuple[int, str], list[np.ndarray]] = {}
    for env_index, (environment, rows) in enumerate(sorted(grouped.items())):
        gaps = []
        for gap in GAPS:
            methods = {}
            for method_index, method in enumerate(METHODS):
                summary = bootstrap_env_gap(
                    rows,
                    gap,
                    method,
                    seed=99_001
                    + env_index * 1009
                    + gap * 11
                    + method_index,
                    repetitions=args.bootstrap_repetitions,
                )
                global_draws.setdefault((gap, method), []).append(
                    summary.pop("_draws")
                )
                methods[method] = summary
            frame_gain = methods["oracle_frame"][
                "paired_gain_vs_recent"
            ]
            auto_gain = methods["oracle_automatic_node"][
                "paired_gain_vs_recent"
            ]
            event_gain = methods["oracle_event_set"][
                "paired_gain_vs_recent"
            ]
            conditional_gain = methods["conditional_ce"][
                "paired_gain_vs_recent"
            ]
            gaps.append(
                {
                    "gap": gap,
                    "methods": methods,
                    "automatic_node_recovery": float(
                        auto_gain / max(frame_gain, 1e-12)
                    ),
                    "conditional_selection_closure": float(
                        conditional_gain / max(event_gain, 1e-12)
                    ),
                }
            )
        environments.append(
            {
                "environment": environment,
                "seeds": sorted(int(row["seed"]) for row in rows),
                "gaps": gaps,
                "suffix_distinguishability": rows[0][
                    "recent_suffix_distinguishability"
                ],
            }
        )
    aggregate_gaps = []
    for gap in GAPS:
        methods = {}
        for method in METHODS:
            env_rows = [
                next(row for row in environment["gaps"] if row["gap"] == gap)[
                    "methods"
                ][method]
                for environment in environments
            ]
            point = (
                float(
                    np.mean(
                        [row["paired_gain_vs_recent"] for row in env_rows]
                    )
                )
                if env_rows
                else None
            )
            draws = global_draws.get((gap, method), [])
            if draws:
                length = min(len(row) for row in draws)
                global_values = np.stack(
                    [row[:length] for row in draws]
                ).mean(0)
                ci = np.quantile(
                    global_values, [0.025, 0.975]
                ).astype(float).tolist()
            else:
                ci = [None, None]
            methods[method] = {
                "paired_gain_vs_recent": point,
                "ci95": ci,
                "hierarchy": (
                    "environment mean of seed/pair nested bootstrap"
                ),
            }
        frame_gain = methods["oracle_frame"]["paired_gain_vs_recent"]
        auto_gain = methods["oracle_automatic_node"][
            "paired_gain_vs_recent"
        ]
        event_gain = methods["oracle_event_set"][
            "paired_gain_vs_recent"
        ]
        conditional_gain = methods["conditional_ce"][
            "paired_gain_vs_recent"
        ]
        aggregate_gaps.append(
            {
                "gap": gap,
                "methods": methods,
                "automatic_node_recovery": (
                    float(auto_gain / max(frame_gain, 1e-12))
                    if frame_gain is not None
                    else None
                ),
                "conditional_selection_closure": (
                    float(conditional_gain / max(event_gain, 1e-12))
                    if event_gain is not None
                    else None
                ),
            }
        )
    high_gap_rows = [
        row for row in aggregate_gaps if row["gap"] >= 32
    ]
    oracle_resolved = bool(
        high_gap_rows
        and all(
            row["methods"]["oracle_frame"]["ci95"][0] is not None
            and row["methods"]["oracle_frame"]["ci95"][0] > 0
            for row in high_gap_rows
        )
    )
    total_frame_gain = (
        float(
            np.mean(
                [
                    row["methods"]["oracle_frame"][
                        "paired_gain_vs_recent"
                    ]
                    for row in high_gap_rows
                ]
            )
        )
        if high_gap_rows
        else None
    )
    total_auto_gain = (
        float(
            np.mean(
                [
                    row["methods"]["oracle_automatic_node"][
                        "paired_gain_vs_recent"
                    ]
                    for row in high_gap_rows
                ]
            )
        )
        if high_gap_rows
        else None
    )
    recovery = (
        float(total_auto_gain / max(total_frame_gain, 1e-12))
        if total_frame_gain is not None
        else None
    )
    recovery_pass = bool(recovery is not None and recovery >= 0.70)
    gate = {
        "name": "Gate 2: anti-recency opportunity and discovery",
        "passed": bool(oracle_resolved and recovery_pass),
        "oracle_frame_criterion": {
            "passed": oracle_resolved,
            "rule": (
                "oracle historical frame paired-gain lower CI > 0 at "
                "gaps 32,64,128"
            ),
        },
        "automatic_discovery_criterion": {
            "passed": recovery_pass,
            "high_gap_recovery": recovery,
            "rule": "automatic discovered node recovers >=70% of frame gain",
        },
    }
    report = {
        "schema": "graph_cem_long_gap_report_v1",
        "status": "completed" if cells else "empty",
        "phase": 2,
        "cell_count": len(cells),
        "environment_count": len(environments),
        "protocol": {
            "controlled_raw_frame_splicing": True,
            "controlled_query_future_teleport": True,
            "native_rollout_claim": False,
            "raw_frames_modified": False,
            "cue_labels_used_for_training": False,
            "gaps": list(GAPS),
            "matched_serialized_bytes": True,
            "memory_tokens_per_arm": MEMORY_TOKENS,
            "matched_read_tokens_and_online_host_calls": True,
        },
        "environments": environments,
        "aggregate_gaps": aggregate_gaps,
        "gate": gate,
        "jobs_still_running": [],
        "artifacts": {
            "build": "outputs/graph_cem_long_gap_v1/build",
            "cells": "outputs/graph_cem_long_gap_v1/cells/<env>/s<seed>",
            "report": "outputs/graph_cem_long_gap_v1/report.json",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        stable_json(json_safe(report))
    )
    print(stable_json(json_safe(report)), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-output", type=Path, default=DEFAULT_BASE_OUTPUT
    )
    parser.add_argument(
        "--conditional-output",
        type=Path,
        default=DEFAULT_CONDITIONAL_OUTPUT,
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--discovery-batch-size", type=int, default=4096)
    parser.add_argument("--memory-hidden", type=int, default=192)
    parser.add_argument("--memory-epochs", type=int, default=35)
    parser.add_argument("--memory-patience", type=int, default=7)
    parser.add_argument("--memory-lr", type=float, default=4e-4)
    parser.add_argument("--ce-hidden", type=int, default=192)
    parser.add_argument("--ce-epochs", type=int, default=25)
    parser.add_argument("--ce-patience", type=int, default=6)
    parser.add_argument("--ce-lr", type=float, default=4e-4)
    parser.add_argument("--oracle-set-chunk", type=int, default=24)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    for name in ("output", "base_output", "conditional_output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        parser.error("--gpu must be one of 0,1,2; GPU3 is prohibited")
    if args.smoke:
        args.limit_pairs = args.limit_pairs or 8
        args.memory_epochs = min(args.memory_epochs, 3)
        args.memory_patience = min(args.memory_patience, 2)
        args.ce_epochs = min(args.ce_epochs, 3)
        args.ce_patience = min(args.ce_patience, 2)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 100)
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
        return
    if args.env_name not in ENVIRONMENTS:
        raise ValueError(
            f"--env-name must be one of {ENVIRONMENTS}; received {args.env_name}"
        )
    if not recipe_path(args.output, args.env_name).is_file():
        raise FileNotFoundError(
            f"run build_graph_cem_long_gap.py first for {args.env_name}"
        )
    run_cell(args)


if __name__ == "__main__":
    main()
