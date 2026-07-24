#!/usr/bin/env python3
"""Cross-fitted flat frame+event fallback selector for long-gap CEM.

This is the required recovery experiment after Graph-CEM gates failed. It
combines label-free event proposals, sparse raw historical frames, and recent
tokens; trains bootstrap heteroscedastic heads on out-of-fold conditional
deletions; calibrates uncertainty; and selects by a lower confidence bound.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.fallback_selector import (  # noqa: E402
    CANDIDATE_TYPE_NAMES,
    EVENT,
    FRAME,
    RECENT,
    FallbackCandidateBatch,
    FallbackSelectorEnsemble,
    serialized_token_bytes,
)
from scripts.build_graph_cem_long_gap import GAPS  # noqa: E402
from scripts.run_cem_conditional_ce import (  # noqa: E402
    centered_rank_correlation,
    deletion_gap_by_query,
    heteroscedastic_loss,
    json_safe,
    pairwise_query_statistics,
    percentile_ci,
    per_query_loss,
    within_query_losses,
)
from scripts.run_cem_raw_ogbench import (  # noqa: E402
    QueryTensors,
    RawMemoryConditioner,
    batches,
    horizon_loss,
    rollout,
    set_seed,
    stable_json,
    tensor_digest,
    train_memory,
)
from scripts.run_graph_cem_long_gap import (  # noqa: E402
    CONTEXT,
    HORIZON,
    MEMORY_TOKENS,
    GapSplit,
    RawGapSplit,
    annotate_surprise,
    bootstrap_pair_mean,
    build_raw_gap,
    discovery_thresholds,
    evaluate_index_sets,
    feature_path,
    load_host,
    loss_for_custom_events,
    loss_for_history_indices,
    recent_indices,
    tensorize_gap,
)

DEFAULT_OUTPUT = ROOT / "outputs/cem_fallback_selector_v1"
DEFAULT_BASE_OUTPUT = ROOT / "outputs/cem_raw_ogbench"
DEFAULT_CONDITIONAL_OUTPUT = ROOT / "outputs/graph_cem_conditional_v1"
DEFAULT_GRAPH_OUTPUT = ROOT / "outputs/graph_cem_long_gap_v1"
ENVIRONMENTS = (
    "pointmaze-large-navigate-v0",
    "cube-single-play-v0",
    "puzzle-3x3-play-v0",
)
HIGH_GAPS = (32, 64, 128)
POOL_SLOTS = 20
EVENT_SLOTS = 8
FRAME_SLOTS = 8
RECENT_SLOTS = 4
STORE_OCCUPANCY = 8
OCCUPANCY_VARIANTS = 5
TYPE_METADATA_DIM = 3


@dataclass
class CandidatePool:
    gap: int
    split: str
    base: GapSplit
    batch: QueryTensors
    candidate_type: torch.Tensor
    uncertainty: torch.Tensor
    source_index: torch.Tensor
    proposal_score: torch.Tensor
    pair_ids: np.ndarray
    branches: np.ndarray

    def index(self, index: torch.Tensor) -> "CandidatePool":
        numpy_index = index.detach().cpu().numpy()
        return CandidatePool(
            gap=self.gap,
            split=self.split,
            base=GapSplit(
                gap=self.base.gap,
                split=self.base.split,
                batch=self.base.batch.index(index),
                history=self.base.history[index],
                history_metadata=self.base.history_metadata[index],
                discovery_score=self.base.discovery_score[index],
                candidate_indices=self.base.candidate_indices[index],
                pair_ids=self.base.pair_ids[numpy_index],
                branches=self.base.branches[numpy_index],
            ),
            batch=self.batch.index(index),
            candidate_type=self.candidate_type[index],
            uncertainty=self.uncertainty[index],
            source_index=self.source_index[index],
            proposal_score=self.proposal_score[index],
            pair_ids=self.pair_ids[numpy_index],
            branches=self.branches[numpy_index],
        )

    def __len__(self) -> int:
        return len(self.batch)


@dataclass
class SelectorTargets:
    features: FallbackCandidateBatch
    target: torch.Tensor
    train_valid: torch.Tensor
    query_ids: np.ndarray
    pair_ids: np.ndarray
    gaps: np.ndarray
    occupancy_variant: np.ndarray
    full_loss: torch.Tensor
    empty_loss: torch.Tensor
    full_read: torch.Tensor
    crossfit_fold: np.ndarray


@dataclass
class CombinedPool:
    batch: QueryTensors
    candidate_type: torch.Tensor
    uncertainty: torch.Tensor
    source_index: torch.Tensor
    proposal_score: torch.Tensor
    pair_ids: np.ndarray
    branches: np.ndarray
    gaps: np.ndarray

    def index(self, index: torch.Tensor) -> "CombinedPool":
        numpy_index = index.detach().cpu().numpy()
        return CombinedPool(
            batch=self.batch.index(index),
            candidate_type=self.candidate_type[index],
            uncertainty=self.uncertainty[index],
            source_index=self.source_index[index],
            proposal_score=self.proposal_score[index],
            pair_ids=self.pair_ids[numpy_index],
            branches=self.branches[numpy_index],
            gaps=self.gaps[numpy_index],
        )

    def __len__(self) -> int:
        return len(self.batch)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(gpu: int) -> torch.device:
    if gpu == 3:
        raise ValueError("GPU3 is prohibited for this campaign")
    if gpu not in (0, 1, 2):
        raise ValueError(f"GPU must be one of 0,1,2; received {gpu}")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def cell_dir(output: Path, env_name: str, seed: int) -> Path:
    return output / "cells" / env_name / f"s{seed}"


def load_sources(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    with np.load(
        feature_path(args.base_output, args.env_name), allow_pickle=False
    ) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    recipe = args.graph_output / "build" / args.env_name / "pairs.npz"
    with np.load(recipe, allow_pickle=False) as data:
        recipes = {
            split: (
                np.asarray(data[f"{split}_sources"], dtype=np.int64),
                np.asarray(data[f"{split}_donors"], dtype=np.int64),
            )
            for split in ("train", "validation", "test")
        }
    return latents, actions, recipes


def kcenter_frames(
    history: np.ndarray,
    excluded: set[int],
    limit: int,
) -> list[int]:
    available = [
        index for index in range(len(history)) if index not in excluded
    ]
    if not available:
        return []
    vectors = history[available]
    center = vectors.mean(0, keepdims=True)
    first = int(np.argmax(np.mean(np.square(vectors - center), axis=1)))
    selected = [available[first]]
    while len(selected) < min(limit, len(available)):
        selected_vectors = history[selected]
        distance = np.min(
            np.mean(
                np.square(
                    vectors[:, None, :] - selected_vectors[None, :, :]
                ),
                axis=-1,
            ),
            axis=1,
        )
        for index, value in enumerate(available):
            if value in selected:
                distance[index] = -np.inf
        selected.append(available[int(np.argmax(distance))])
    return selected


def build_candidate_pool(
    raw: RawGapSplit,
    threshold: tuple[float, float],
    device: torch.device,
) -> CandidatePool:
    base = tensorize_gap(raw, threshold, device)
    count, steps, latent_dim = raw.history.shape
    legal = steps - CONTEXT
    old_stop = legal - RECENT_SLOTS
    event_score = np.asarray(base.discovery_score.detach().cpu())
    events = np.zeros((count, POOL_SLOTS, latent_dim), dtype=np.float32)
    metadata = np.zeros((count, POOL_SLOTS, 3), dtype=np.float32)
    valid = np.zeros((count, POOL_SLOTS), dtype=bool)
    candidate_type = np.zeros((count, POOL_SLOTS), dtype=np.int64)
    uncertainty = np.ones((count, POOL_SLOTS), dtype=np.float32)
    source_index = np.zeros((count, POOL_SLOTS), dtype=np.int64)
    proposal_score = np.full(
        (count, POOL_SLOTS), -np.inf, dtype=np.float32
    )
    for row in range(count):
        old_indices = np.arange(max(0, old_stop), dtype=np.int64)
        ranked_event = old_indices[
            np.argsort(event_score[row, :old_stop])[::-1]
        ][:EVENT_SLOTS]
        used = set(int(value) for value in ranked_event)
        frame_indices = kcenter_frames(
            raw.history[row, :old_stop], used, FRAME_SLOTS
        )
        recent = list(range(legal - RECENT_SLOTS, legal))
        definitions = (
            (EVENT, list(map(int, ranked_event)), 0),
            (FRAME, frame_indices, EVENT_SLOTS),
            (
                RECENT,
                recent,
                EVENT_SLOTS + FRAME_SLOTS,
            ),
        )
        for type_id, indices, offset in definitions:
            for local, time_index in enumerate(indices):
                column = offset + local
                if column >= POOL_SLOTS:
                    continue
                events[row, column] = raw.history[row, time_index]
                metadata[row, column] = base.history_metadata[
                    row, time_index
                ].detach().cpu().numpy()
                valid[row, column] = True
                candidate_type[row, column] = type_id
                source_index[row, column] = time_index
                score = float(event_score[row, time_index])
                proposal_score[row, column] = score
                if type_id == EVENT:
                    surprise_scaled = (
                        raw.surprise[row, time_index] / threshold[0]
                    )
                    change_scaled = (
                        raw.change[row, time_index] / threshold[1]
                    )
                    disagreement = abs(surprise_scaled - change_scaled) / (
                        abs(surprise_scaled) + abs(change_scaled) + 1e-8
                    )
                    uncertainty[row, column] = float(
                        np.clip(
                            0.5 / (1.0 + max(score, 0.0))
                            + 0.5 * disagreement,
                            0.0,
                            1.0,
                        )
                    )
                elif type_id == FRAME:
                    uncertainty[row, column] = 0.25
                else:
                    uncertainty[row, column] = 0.0
    batch = QueryTensors(
        context_z=base.batch.context_z,
        action_history=base.batch.action_history,
        future_actions=base.batch.future_actions,
        targets=base.batch.targets,
        events=torch.from_numpy(events).to(device),
        metadata=torch.from_numpy(metadata).to(device),
        valid=torch.from_numpy(valid).to(device),
        recent_event=base.batch.recent_event,
    )
    return CandidatePool(
        gap=raw.gap,
        split=raw.split,
        base=base,
        batch=batch,
        candidate_type=torch.from_numpy(candidate_type).to(device),
        uncertainty=torch.from_numpy(uncertainty).to(device),
        source_index=torch.from_numpy(source_index).to(device),
        proposal_score=torch.from_numpy(proposal_score).to(device),
        pair_ids=raw.pair_ids,
        branches=raw.branches,
    )


def concatenate_query_tensors(values: list[QueryTensors]) -> QueryTensors:
    return QueryTensors(
        **{
            field: torch.cat(
                [getattr(value, field) for value in values], dim=0
            )
            for field in QueryTensors.__dataclass_fields__
        }
    )


def combine_pools(pools: list[CandidatePool]) -> CombinedPool:
    return CombinedPool(
        batch=concatenate_query_tensors([pool.batch for pool in pools]),
        candidate_type=torch.cat(
            [pool.candidate_type for pool in pools], dim=0
        ),
        uncertainty=torch.cat(
            [pool.uncertainty for pool in pools], dim=0
        ),
        source_index=torch.cat(
            [pool.source_index for pool in pools], dim=0
        ),
        proposal_score=torch.cat(
            [pool.proposal_score for pool in pools], dim=0
        ),
        pair_ids=np.concatenate([pool.pair_ids for pool in pools]),
        branches=np.concatenate([pool.branches for pool in pools]),
        gaps=np.concatenate(
            [
                np.full(len(pool), pool.gap, dtype=np.int64)
                for pool in pools
            ]
        ),
    )


def memory_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        memory_hidden=args.memory_hidden,
        max_events=POOL_SLOTS,
        memory_lr=args.memory_lr,
        memory_epochs=args.memory_epochs,
        memory_patience=args.memory_patience,
        residual_cost=1e-3,
        batch_size=args.batch_size,
    )


@torch.no_grad()
def router_scores(
    memory: RawMemoryConditioner,
    pool: CombinedPool,
) -> torch.Tensor:
    return memory.scores(
        pool.batch.context_z,
        pool.batch.future_actions[:, 0],
        pool.batch.events,
        pool.batch.metadata,
    ).masked_fill(~pool.batch.valid, -1e9)


def topk_mask(
    score: torch.Tensor,
    valid: torch.Tensor,
    count: int,
) -> torch.Tensor:
    selected = torch.zeros_like(valid)
    indices = torch.topk(
        score.masked_fill(~valid, -1e9),
        k=min(count, score.shape[1]),
        dim=1,
    ).indices
    selected.scatter_(1, indices, True)
    return selected & valid


@torch.no_grad()
def occupancy_masks(
    memory: RawMemoryConditioner,
    pool: CombinedPool,
    seed: int,
) -> list[torch.Tensor]:
    route = router_scores(memory, pool)
    valid = pool.batch.valid
    masks = [valid, topk_mask(route, valid, STORE_OCCUPANCY)]
    event_valid = valid & (pool.candidate_type == EVENT)
    frame_valid = valid & (pool.candidate_type == FRAME)
    event_mask = topk_mask(route, event_valid, STORE_OCCUPANCY // 2)
    frame_mask = topk_mask(route, frame_valid, STORE_OCCUPANCY // 2)
    balanced = event_mask | frame_mask
    missing = STORE_OCCUPANCY - balanced.sum(1)
    if bool((missing > 0).any()):
        fill = topk_mask(
            route.masked_fill(balanced, -1e9),
            valid & ~balanced,
            STORE_OCCUPANCY,
        )
        for row in torch.nonzero(missing > 0, as_tuple=False).flatten():
            columns = torch.nonzero(fill[row], as_tuple=False).flatten()
            balanced[row, columns[: int(missing[row])]] = True
    masks.append(balanced)
    masks.append(
        topk_mask(
            route - pool.uncertainty,
            valid,
            STORE_OCCUPANCY,
        )
    )
    generator = torch.Generator(device=route.device)
    generator.manual_seed(seed)
    random_score = torch.rand(
        route.shape, generator=generator, device=route.device
    )
    masks.append(topk_mask(random_score, valid, STORE_OCCUPANCY))
    return masks


@torch.no_grad()
def read_mask(
    route: torch.Tensor,
    occupancy: torch.Tensor,
    excluded: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = occupancy if excluded is None else occupancy & ~excluded
    return topk_mask(route, valid, MEMORY_TOKENS)


@torch.no_grad()
def make_selector_features(
    memory: RawMemoryConditioner,
    pool: CombinedPool,
    occupied: torch.Tensor,
) -> FallbackCandidateBatch:
    query = memory.query_vector(
        pool.batch.context_z, pool.batch.future_actions[:, 0]
    )
    route = router_scores(memory, pool)
    route_center = (
        route.masked_fill(~pool.batch.valid, 0.0).sum(1, keepdim=True)
        / pool.batch.valid.sum(1, keepdim=True).clamp_min(1)
    )
    return FallbackCandidateBatch(
        latent=pool.batch.events.detach(),
        query=query.detach(),
        metadata=pool.batch.metadata.detach(),
        candidate_type=pool.candidate_type,
        discovery_uncertainty=pool.uncertainty,
        router_score=(route - route_center).detach(),
        occupied=occupied,
        valid=pool.batch.valid,
    )


@torch.no_grad()
def generate_conditional_targets(
    host: nn.Module,
    memory: RawMemoryConditioner,
    pool: CombinedPool,
    seed: int,
    fold: int,
    batch_size: int,
) -> SelectorTargets:
    route = router_scores(memory, pool)
    empty_loss = per_query_loss(
        host, None, pool.batch, batch_size=batch_size
    )
    masks = occupancy_masks(memory, pool, seed)
    feature_rows = []
    target_rows = []
    valid_rows = []
    full_losses = []
    empty_losses = []
    full_reads = []
    query_ids = []
    pair_ids = []
    gaps = []
    variants = []
    for variant, occupied in enumerate(masks):
        full_read = read_mask(route, occupied)
        full_loss = per_query_loss(
            host,
            memory,
            pool.batch,
            full_read,
            batch_size=batch_size,
        )
        target = torch.zeros_like(route)
        for slot in range(POOL_SLOTS):
            excluded = torch.zeros_like(occupied)
            excluded[:, slot] = occupied[:, slot]
            rerouted = read_mask(route, occupied, excluded)
            deleted_loss = per_query_loss(
                host,
                memory,
                pool.batch,
                rerouted,
                batch_size=batch_size,
            )
            target[:, slot] = (
                deleted_loss - full_loss
            ) / empty_loss.clamp_min(1e-8)
        feature_rows.append(make_selector_features(memory, pool, occupied))
        target_rows.append(target.masked_fill(~occupied, 0.0))
        valid_rows.append(occupied)
        full_losses.append(full_loss)
        empty_losses.append(empty_loss)
        full_reads.append(full_read)
        query_ids.append(np.arange(len(pool), dtype=np.int64))
        pair_ids.append(pool.pair_ids)
        gaps.append(pool.gaps)
        variants.append(
            np.full(len(pool), variant, dtype=np.int64)
        )
    features = FallbackCandidateBatch(
        **{
            field: torch.cat(
                [getattr(value, field) for value in feature_rows], dim=0
            )
            for field in FallbackCandidateBatch.__dataclass_fields__
        }
    )
    return SelectorTargets(
        features=features,
        target=torch.cat(target_rows, dim=0),
        train_valid=torch.cat(valid_rows, dim=0),
        query_ids=np.concatenate(query_ids),
        pair_ids=np.concatenate(pair_ids),
        gaps=np.concatenate(gaps),
        occupancy_variant=np.concatenate(variants),
        full_loss=torch.cat(full_losses),
        empty_loss=torch.cat(empty_losses),
        full_read=torch.cat(full_reads),
        crossfit_fold=np.full(
            len(pool) * len(masks), fold, dtype=np.int64
        ),
    )


def concatenate_targets(values: list[SelectorTargets]) -> SelectorTargets:
    return SelectorTargets(
        features=FallbackCandidateBatch(
            **{
                field: torch.cat(
                    [getattr(value.features, field) for value in values],
                    dim=0,
                )
                for field in FallbackCandidateBatch.__dataclass_fields__
            }
        ),
        target=torch.cat([value.target for value in values], dim=0),
        train_valid=torch.cat(
            [value.train_valid for value in values], dim=0
        ),
        query_ids=np.concatenate([value.query_ids for value in values]),
        pair_ids=np.concatenate([value.pair_ids for value in values]),
        gaps=np.concatenate([value.gaps for value in values]),
        occupancy_variant=np.concatenate(
            [value.occupancy_variant for value in values]
        ),
        full_loss=torch.cat([value.full_loss for value in values]),
        empty_loss=torch.cat([value.empty_loss for value in values]),
        full_read=torch.cat([value.full_read for value in values]),
        crossfit_fold=np.concatenate(
            [value.crossfit_fold for value in values]
        ),
    )


def index_targets(
    value: SelectorTargets,
    index: torch.Tensor,
) -> tuple[FallbackCandidateBatch, torch.Tensor, torch.Tensor]:
    return (
        value.features.index(index),
        value.target[index],
        value.train_valid[index],
    )


def train_selector_ensemble(
    train: SelectorTargets,
    validation: SelectorTargets,
    args: argparse.Namespace,
    *,
    event_only: bool,
) -> tuple[FallbackSelectorEnsemble, dict[str, Any]]:
    ensemble = FallbackSelectorEnsemble(
        latent_dim=train.features.latent.shape[-1],
        metadata_dim=train.features.metadata.shape[-1],
        hidden=args.selector_hidden,
        members=args.ensemble_members,
    ).to(train.target.device)
    target_values = train.target[train.train_valid]
    temperature = max(
        1e-4, float(target_values.detach().float().std().cpu())
    )
    member_summaries = []
    for member_index, member in enumerate(ensemble.members):
        optimizer = torch.optim.AdamW(
            member.parameters(),
            lr=args.selector_lr,
            weight_decay=1e-4,
        )
        rng = np.random.default_rng(
            args.seed * 100_003 + member_index * 997 + int(event_only)
        )
        bootstrap = rng.choice(
            len(train.target), size=len(train.target), replace=True
        )
        best_state = None
        best_objective = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        for epoch in range(args.selector_epochs):
            member.train()
            epoch_losses = []
            order = rng.permutation(bootstrap)
            for start in range(0, len(order), args.batch_size):
                indices = torch.as_tensor(
                    order[start : start + args.batch_size],
                    device=train.target.device,
                )
                features, truth, valid = index_targets(train, indices)
                if event_only:
                    valid = valid & (
                        features.candidate_type == EVENT
                    )
                if not bool(valid.any()):
                    continue
                mean, log_variance = member(features)
                regression = heteroscedastic_loss(
                    mean[valid], log_variance[valid], truth[valid]
                ).mean()
                huber = F.smooth_l1_loss(
                    mean[valid],
                    truth[valid],
                    beta=max(temperature, 1e-4),
                )
                ranking, listwise = within_query_losses(
                    mean, truth, valid, temperature
                )
                loss = regression + 0.25 * huber + ranking + 0.25 * listwise
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(member.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            member.eval()
            with torch.no_grad():
                valid = validation.train_valid
                if event_only:
                    valid = valid & (
                        validation.features.candidate_type == EVENT
                    )
                mean, log_variance = member(validation.features)
                val_regression = heteroscedastic_loss(
                    mean[valid],
                    log_variance[valid],
                    validation.target[valid],
                ).mean()
                val_ranking, val_listwise = within_query_losses(
                    mean,
                    validation.target,
                    valid,
                    temperature,
                )
                objective = float(
                    val_regression
                    + val_ranking
                    + 0.25 * val_listwise
                )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_objective": float(np.mean(epoch_losses)),
                    "validation_objective": objective,
                }
            )
            if objective < best_objective - 1e-6:
                best_objective = objective
                best_epoch = epoch + 1
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in member.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= args.selector_patience:
                break
        if best_state is None:
            raise RuntimeError("selector member produced no finite state")
        member.load_state_dict(best_state)
        member.eval()
        member_summaries.append(
            {
                "member": member_index,
                "best_epoch": best_epoch,
                "best_validation_objective": best_objective,
                "bootstrap_unique_fraction": float(
                    len(np.unique(bootstrap)) / len(bootstrap)
                ),
                "history": history,
            }
        )
    valid = validation.train_valid
    if event_only:
        valid = valid & (
            validation.features.candidate_type == EVENT
        )
    scale = ensemble.calibrate_scale(
        validation.features,
        validation.target,
        valid,
        target_coverage=0.90,
    )
    return ensemble, {
        "members": member_summaries,
        "member_count": args.ensemble_members,
        "bootstrap_ensemble": True,
        "heteroscedastic_regression": "Gaussian NLL plus Huber",
        "within_store_pairwise_and_listwise": True,
        "event_only": event_only,
        "target_temperature": temperature,
        "uncertainty_scale": scale,
        "calibration_target_coverage": 0.90,
    }


@torch.no_grad()
def uncertainty_metrics(
    ensemble: FallbackSelectorEnsemble,
    targets: SelectorTargets,
    *,
    event_only: bool = False,
) -> dict[str, Any]:
    output = ensemble(targets.features)
    valid = targets.train_valid
    if event_only:
        valid = valid & (
            targets.features.candidate_type == EVENT
        )
    error = (targets.target - output["mean"]).abs()
    std = output["std"].clamp_min(1e-8)
    coverages = {}
    for nominal, z_value in ((0.50, 0.67449), (0.80, 1.28155), (0.90, 1.64485)):
        empirical = float((error[valid] <= z_value * std[valid]).float().mean())
        coverages[str(nominal)] = empirical
    ece = float(
        np.mean(
            [
                abs(float(nominal) - empirical)
                for nominal, empirical in coverages.items()
            ]
        )
    )
    return {
        "interval_coverage": coverages,
        "coverage_ece": ece,
        "mean_predictive_std": float(std[valid].mean()),
        "mean_absolute_error": float(error[valid].mean()),
        "sample_count": int(valid.sum()),
    }


@torch.no_grad()
def ensemble_scores(
    ensemble: FallbackSelectorEnsemble,
    memory: RawMemoryConditioner,
    pool: CombinedPool,
    *,
    event_only: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], FallbackCandidateBatch]:
    features = make_selector_features(
        memory, pool, pool.batch.valid
    )
    lcb, output = ensemble.lower_confidence_bound(features)
    valid = pool.batch.valid
    if event_only:
        valid = valid & (pool.candidate_type == EVENT)
    lcb = lcb.masked_fill(~valid, -1e9)
    output = {
        key: value.masked_fill(~valid, 0.0)
        if value.shape == valid.shape
        else value
        for key, value in output.items()
    }
    return lcb, output, features


def positions_from_score(
    score: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    return torch.topk(
        score.masked_fill(~valid, -1e9),
        k=MEMORY_TOKENS,
        dim=1,
    ).indices


def pool_positions_to_history(
    pool: CandidatePool,
    positions: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(pool.source_index, 1, positions)


def ranking_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from scipy.stats import spearmanr

    spearman = centered_rank_correlation(prediction, target, valid)
    pooled = (
        float(spearmanr(prediction[valid], target[valid]).statistic)
        if int(valid.sum()) >= 3
        else None
    )
    correct, pair_count = pairwise_query_statistics(
        prediction, target, valid
    )
    high, random_value, gap = deletion_gap_by_query(
        prediction, target, valid, seed
    )
    total_pairs = float(pair_count.sum())
    return {
        "within_query_spearman": spearman,
        "pooled_spearman": pooled,
        "pairwise_accuracy": (
            float(correct.sum() / total_pairs) if total_pairs else None
        ),
        "pair_count": int(total_pairs),
        "high_conditional_effect": (
            float(np.nanmean(high)) if np.isfinite(high).any() else None
        ),
        "random_conditional_effect": (
            float(np.nanmean(random_value))
            if np.isfinite(random_value).any()
            else None
        ),
        "high_minus_random": (
            float(np.nanmean(gap)) if np.isfinite(gap).any() else None
        ),
        "high_minus_random_query_ci95": percentile_ci(
            gap, seed + 7717
        ),
    }, {
        "pair_correct": correct.astype(np.float32),
        "pair_count": pair_count.astype(np.float32),
        "deletion_gap": gap.astype(np.float32),
    }


@torch.no_grad()
def evaluate_gap(
    host: nn.Module,
    memory: RawMemoryConditioner,
    fallback: FallbackSelectorEnsemble,
    event_only: FallbackSelectorEnsemble,
    pool: CandidatePool,
    test_targets: SelectorTargets,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    combined = combine_pools([pool])
    count = len(pool)
    legal = pool.base.history.shape[1] - CONTEXT
    recent = recent_indices(pool.base)
    recent_loss = loss_for_history_indices(
        host, memory, pool.base, recent, args.batch_size
    )
    no_memory_loss = per_query_loss(
        host, None, pool.batch, batch_size=args.batch_size
    )
    fallback_score, fallback_output, fallback_features = ensemble_scores(
        fallback, memory, combined, event_only=False
    )
    event_score, event_output, _ = ensemble_scores(
        event_only, memory, combined, event_only=True
    )
    fallback_positions = positions_from_score(
        fallback_score, pool.batch.valid
    )
    event_valid = pool.batch.valid & (
        pool.candidate_type == EVENT
    )
    event_positions = positions_from_score(event_score, event_valid)
    fallback_indices = pool_positions_to_history(
        pool, fallback_positions
    )
    event_indices = pool_positions_to_history(pool, event_positions)
    surprise_positions = torch.topk(
        pool.proposal_score.masked_fill(~event_valid, -1e9),
        k=MEMORY_TOKENS,
        dim=1,
    ).indices
    surprise_indices = pool_positions_to_history(
        pool, surprise_positions
    )
    generator = torch.Generator(device=pool.batch.events.device)
    generator.manual_seed(args.seed * 100_003 + pool.gap)
    random_score = torch.rand(
        pool.batch.valid.shape,
        generator=generator,
        device=pool.batch.events.device,
    )
    random_positions = positions_from_score(
        random_score, pool.batch.valid
    )
    random_indices = pool_positions_to_history(
        pool, random_positions
    )
    losses = {
        "fallback_selector": loss_for_history_indices(
            host,
            memory,
            pool.base,
            fallback_indices,
            args.batch_size,
        ),
        "event_only_conditional": loss_for_history_indices(
            host,
            memory,
            pool.base,
            event_indices,
            args.batch_size,
        ),
        "surprise": loss_for_history_indices(
            host,
            memory,
            pool.base,
            surprise_indices,
            args.batch_size,
        ),
        "random": loss_for_history_indices(
            host,
            memory,
            pool.base,
            random_indices,
            args.batch_size,
        ),
        "recent_only": recent_loss,
        "no_memory": no_memory_loss,
    }
    recent_filler = recent[:, -3:]
    old_count = legal - RECENT_SLOTS
    old = torch.arange(
        old_count, device=pool.batch.events.device
    )[None, :, None].expand(count, -1, -1)
    frame_sets = torch.cat(
        [
            old,
            recent_filler[:, None].expand(-1, old_count, -1),
        ],
        dim=2,
    )
    frame_grid = evaluate_index_sets(
        host,
        memory,
        pool.base,
        frame_sets,
        args.batch_size,
        set_chunk=args.oracle_set_chunk,
    )
    oracle_frame_loss, oracle_frame_index = frame_grid.min(1)
    event_columns = torch.nonzero(
        event_valid[0], as_tuple=False
    ).flatten()
    event_history = pool.source_index[:, event_columns]
    event_sets = torch.cat(
        [
            event_history.unsqueeze(-1),
            recent_filler[:, None].expand(
                -1, len(event_columns), -1
            ),
        ],
        dim=2,
    )
    event_grid = evaluate_index_sets(
        host,
        memory,
        pool.base,
        event_sets,
        args.batch_size,
        set_chunk=args.oracle_set_chunk,
    )
    oracle_event_loss, _ = event_grid.min(1)
    nonrecent_valid = pool.batch.valid & (
        pool.candidate_type != RECENT
    )
    union_columns = torch.nonzero(
        nonrecent_valid[0], as_tuple=False
    ).flatten()
    union_history = pool.source_index[:, union_columns]
    union_sets = torch.cat(
        [
            union_history.unsqueeze(-1),
            recent_filler[:, None].expand(
                -1, len(union_columns), -1
            ),
        ],
        dim=2,
    )
    union_grid = evaluate_index_sets(
        host,
        memory,
        pool.base,
        union_sets,
        args.batch_size,
        set_chunk=args.oracle_set_chunk,
    )
    oracle_union_loss, _ = union_grid.min(1)
    losses.update(
        {
            "oracle_frame": oracle_frame_loss,
            "oracle_event": oracle_event_loss,
            "oracle_union": oracle_union_loss,
        }
    )
    recent_np = recent_loss.detach().cpu().numpy()
    arrays: dict[str, np.ndarray] = {
        "gap": np.full(count, pool.gap, dtype=np.int64),
        "pair_id": pool.pair_ids.astype(np.int64),
        "branch": pool.branches.astype(np.int64),
    }
    methods = {}
    for method, loss in losses.items():
        value = loss.detach().cpu().numpy()
        arrays[f"loss_{method}"] = value.astype(np.float32)
        methods[method] = {
            "mean_host_future_loss": float(value.mean()),
            "paired_gain_vs_recent": bootstrap_pair_mean(
                recent_np - value,
                pool.pair_ids,
                args.seed * 1009 + pool.gap + len(method),
                repetitions=args.bootstrap_repetitions,
            ),
        }
    frame_gain = recent_np - arrays["loss_oracle_frame"]
    event_gain = recent_np - arrays["loss_oracle_event"]
    union_gain = recent_np - arrays["loss_oracle_union"]
    fallback_gain = recent_np - arrays["loss_fallback_selector"]
    event_method_gain = (
        recent_np - arrays["loss_event_only_conditional"]
    )
    union_sources = pool.source_index.detach().cpu().numpy()
    union_valid_np = nonrecent_valid.detach().cpu().numpy()
    oracle_index_np = oracle_frame_index.detach().cpu().numpy()
    proposal_hit = np.asarray(
        [
            oracle_index_np[row] in set(
                union_sources[row, union_valid_np[row]].tolist()
            )
            for row in range(count)
        ],
        dtype=np.float32,
    )
    selected_type = torch.gather(
        pool.candidate_type, 1, fallback_positions
    ).detach().cpu().numpy()
    frequencies = {
        CANDIDATE_TYPE_NAMES[type_id]: float(
            np.mean(selected_type == type_id)
        )
        for type_id in CANDIDATE_TYPE_NAMES
    }
    selected_uncertainty = torch.gather(
        fallback_output["std"], 1, fallback_positions
    ).detach().cpu().numpy()
    event_mean = fallback_output["mean"].masked_fill(
        ~event_valid, -1e9
    )
    frame_valid = pool.batch.valid & (
        pool.candidate_type == FRAME
    )
    frame_mean = fallback_output["mean"].masked_fill(
        ~frame_valid, -1e9
    )
    event_lcb = fallback_score.masked_fill(~event_valid, -1e9)
    frame_lcb = fallback_score.masked_fill(~frame_valid, -1e9)
    uncertainty_fallback = (
        (event_mean.max(1).values > frame_mean.max(1).values)
        & (event_lcb.max(1).values <= frame_lcb.max(1).values)
    ).detach().cpu().numpy()
    decision_logs = []
    mean_np = fallback_output["mean"].detach().cpu().numpy()
    std_np = fallback_output["std"].detach().cpu().numpy()
    lcb_np = fallback_score.detach().cpu().numpy()
    selected_np = np.zeros_like(mean_np, dtype=bool)
    np.put_along_axis(
        selected_np,
        fallback_positions.detach().cpu().numpy(),
        True,
        axis=1,
    )
    type_np = pool.candidate_type.detach().cpu().numpy()
    source_np = pool.source_index.detach().cpu().numpy()
    valid_np = pool.batch.valid.detach().cpu().numpy()
    true_np = test_targets.target[:count].detach().cpu().numpy()
    occupied_np = (
        test_targets.train_valid[:count].detach().cpu().numpy()
    )
    full_read_np = (
        test_targets.full_read[:count].detach().cpu().numpy()
    )
    for row in range(count):
        decision_logs.append(
            {
                "gap": pool.gap,
                "pair_id": int(pool.pair_ids[row]),
                "branch_posthoc": int(pool.branches[row]),
                "branch_used_for_selection": False,
                "fallback_reason": (
                    "event_uncertainty"
                    if bool(uncertainty_fallback[row])
                    else "highest_calibrated_lcb"
                ),
                "occupied_store_slots": np.flatnonzero(
                    occupied_np[row]
                ).astype(int).tolist(),
                "router_read_slots": np.flatnonzero(
                    full_read_np[row]
                ).astype(int).tolist(),
                "candidates": [
                    {
                        "slot": column,
                        "candidate_type": CANDIDATE_TYPE_NAMES[
                            int(type_np[row, column])
                        ],
                        "history_index": int(source_np[row, column]),
                        "selected": bool(selected_np[row, column]),
                        "ce_hat": float(mean_np[row, column]),
                        "ce_true": float(true_np[row, column]),
                        "uncertainty": float(std_np[row, column]),
                        "lcb": float(lcb_np[row, column]),
                        "discovery_uncertainty": float(
                            pool.uncertainty[row, column]
                        ),
                        "occupied_store": bool(
                            occupied_np[row, column]
                        ),
                    }
                    for column in range(POOL_SLOTS)
                    if valid_np[row, column]
                ],
            }
        )
    metrics = {
        "gap": pool.gap,
        "example_count": count,
        "pair_count": int(len(np.unique(pool.pair_ids))),
        "methods": methods,
        "oracle_frame_gain": float(frame_gain.mean()),
        "oracle_event_recovery": float(
            event_gain.mean() / max(frame_gain.mean(), 1e-12)
        ),
        "oracle_union_recovery": float(
            union_gain.mean() / max(frame_gain.mean(), 1e-12)
        ),
        "learned_fallback_recovery": float(
            fallback_gain.mean() / max(frame_gain.mean(), 1e-12)
        ),
        "selection_closure": float(
            fallback_gain.mean() / max(union_gain.mean(), 1e-12)
        ),
        "proposal_oracle_frame_hit_rate": float(proposal_hit.mean()),
        "selected_candidate_type_frequency": frequencies,
        "event_uncertainty_fallback_rate": float(
            uncertainty_fallback.mean()
        ),
        "mean_selected_uncertainty": float(
            selected_uncertainty.mean()
        ),
        "fallback_minus_event_gain": bootstrap_pair_mean(
            fallback_gain - event_method_gain,
            pool.pair_ids,
            args.seed * 701 + pool.gap,
            repetitions=args.bootstrap_repetitions,
        ),
        "resource_contract": {
            "serialized_bytes": serialized_token_bytes(
                pool.batch.events.shape[-1], MEMORY_TOKENS
            ),
            "read_tokens": MEMORY_TOKENS,
            "online_host_calls": 1,
        },
    }
    return metrics, arrays, decision_logs


@torch.no_grad()
def measure_recall_latency(
    ensemble: FallbackSelectorEnsemble,
    features: FallbackCandidateBatch,
    valid: torch.Tensor,
    repetitions: int,
) -> float:
    device = features.latent.device
    for _ in range(10):
        score, _ = ensemble.lower_confidence_bound(features)
        positions_from_score(score, valid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repetitions):
        score, _ = ensemble.lower_confidence_bound(features)
        positions_from_score(score, valid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / repetitions


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = cell_dir(args.output, args.env_name, args.seed)
    result_path = output_dir / "result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    set_seed(args.seed)
    device = resolve_device(args.gpu)
    host_args = argparse.Namespace(
        env_name=args.env_name,
        seed=args.seed,
        base_output=args.base_output,
        conditional_output=args.conditional_output,
    )
    host, host_receipt = load_host(host_args, device)
    host_digest_before = tensor_digest(host)
    latents, actions, recipes = load_sources(args)
    raw: dict[str, dict[int, RawGapSplit]] = {
        split: {} for split in ("train", "validation", "test")
    }
    pools: dict[str, dict[int, CandidatePool]] = {
        split: {} for split in raw
    }
    thresholds = {}
    pool_started = time.perf_counter()
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
                host,
                raw[split][gap],
                device,
                args.discovery_batch_size,
            )
        thresholds[gap] = discovery_thresholds(raw["train"][gap])
        for split in raw:
            pools[split][gap] = build_candidate_pool(
                raw[split][gap], thresholds[gap], device
            )
    pool_elapsed = time.perf_counter() - pool_started
    train_pool = combine_pools(
        [pools["train"][gap] for gap in HIGH_GAPS]
    )
    validation_pool = combine_pools(
        [pools["validation"][gap] for gap in HIGH_GAPS]
    )
    crossfit_targets = []
    crossfit_receipts = []
    fold_assignments = train_pool.pair_ids % args.crossfit_folds
    for fold in range(args.crossfit_folds):
        held_np = fold_assignments == fold
        train_np = ~held_np
        held_index = torch.as_tensor(
            np.flatnonzero(held_np), device=device
        )
        train_index = torch.as_tensor(
            np.flatnonzero(train_np), device=device
        )
        policy_train = train_pool.index(train_index)
        held_pool = train_pool.index(held_index)
        train_pairs = set(policy_train.pair_ids.tolist())
        held_pairs = set(held_pool.pair_ids.tolist())
        if train_pairs & held_pairs:
            raise RuntimeError("cross-fit policy/label pair leakage")
        fold_memory, fold_training = train_memory(
            host,
            policy_train.batch,
            validation_pool.batch,
            memory_args(args, args.seed + 1000 * (fold + 1)),
            device,
        )
        fold_memory.eval()
        for parameter in fold_memory.parameters():
            parameter.requires_grad_(False)
        fold_target = generate_conditional_targets(
            host,
            fold_memory,
            held_pool,
            seed=args.seed * 100_003 + fold,
            fold=fold,
            batch_size=args.batch_size,
        )
        crossfit_targets.append(fold_target)
        crossfit_receipts.append(
            {
                "fold": fold,
                "policy_train_pairs": len(train_pairs),
                "label_pairs": len(held_pairs),
                "pair_overlap": 0,
                "policy_train_rows": len(policy_train),
                "label_rows": len(held_pool),
                "memory_training": fold_training,
            }
        )
        del fold_memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
    oof_targets = concatenate_targets(crossfit_targets)
    final_memory, final_memory_training = train_memory(
        host,
        train_pool.batch,
        validation_pool.batch,
        memory_args(args, args.seed + 90_001),
        device,
    )
    final_memory.eval()
    for parameter in final_memory.parameters():
        parameter.requires_grad_(False)
    validation_targets = generate_conditional_targets(
        host,
        final_memory,
        validation_pool,
        seed=args.seed + 17_017,
        fold=-1,
        batch_size=args.batch_size,
    )
    fallback, fallback_training = train_selector_ensemble(
        oof_targets,
        validation_targets,
        args,
        event_only=False,
    )
    event_ensemble, event_training = train_selector_ensemble(
        oof_targets,
        validation_targets,
        args,
        event_only=True,
    )
    test_targets_by_gap = {
        gap: generate_conditional_targets(
            host,
            final_memory,
            combine_pools([pools["test"][gap]]),
            seed=args.seed * 101 + gap,
            fold=-2,
            batch_size=args.batch_size,
        )
        for gap in HIGH_GAPS
    }
    high_test_targets = concatenate_targets(
        [test_targets_by_gap[gap] for gap in HIGH_GAPS]
    )
    lcb, test_output = fallback.lower_confidence_bound(
        high_test_targets.features
    )
    valid = high_test_targets.train_valid
    ranking, ranking_arrays = ranking_metrics(
        lcb.detach().cpu().numpy(),
        high_test_targets.target.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
        seed=args.seed + 55_001,
    )
    uncertainty = uncertainty_metrics(
        fallback, high_test_targets, event_only=False
    )
    gap_results = []
    evaluation_arrays = []
    decision_logs = []
    for gap in GAPS:
        target = (
            test_targets_by_gap[gap]
            if gap in test_targets_by_gap
            else generate_conditional_targets(
                host,
                final_memory,
                combine_pools([pools["test"][gap]]),
                seed=args.seed * 101 + gap,
                fold=-2,
                batch_size=args.batch_size,
            )
        )
        metrics, arrays, logs = evaluate_gap(
            host,
            final_memory,
            fallback,
            event_ensemble,
            pools["test"][gap],
            target,
            args,
        )
        gap_results.append(metrics)
        evaluation_arrays.append(arrays)
        decision_logs.extend(logs)
    test_pool = combine_pools(
        [pools["test"][gap] for gap in HIGH_GAPS]
    )
    occupied = test_pool.batch.valid
    latency_features = make_selector_features(
        final_memory, test_pool, occupied
    )
    fallback_latency = measure_recall_latency(
        fallback,
        latency_features,
        occupied,
        args.latency_repetitions,
    )
    event_valid = occupied & (
        test_pool.candidate_type == EVENT
    )
    event_latency = measure_recall_latency(
        event_ensemble,
        latency_features,
        event_valid,
        args.latency_repetitions,
    )
    latency_ratio = fallback_latency / max(event_latency, 1e-12)
    samples_path = output_dir / "evaluation.npz"
    combined_arrays = {
        key: np.concatenate([row[key] for row in evaluation_arrays])
        for key in evaluation_arrays[0]
    }
    combined_arrays.update(
        {
            "ranking_pair_id": high_test_targets.pair_ids,
            "ranking_gap": high_test_targets.gaps,
            "ranking_query_id": high_test_targets.query_ids,
            "ranking_prediction": lcb.detach().cpu().numpy(),
            "ranking_target": high_test_targets.target.detach().cpu().numpy(),
            "ranking_valid": valid.detach().cpu().numpy(),
            "ranking_pair_correct": ranking_arrays["pair_correct"],
            "ranking_pair_count": ranking_arrays["pair_count"],
            "ranking_deletion_gap": ranking_arrays["deletion_gap"],
            "ranking_std": test_output["std"].detach().cpu().numpy(),
        }
    )
    np.savez_compressed(samples_path, **combined_arrays)
    decision_path = output_dir / "decision_log.json"
    decision_path.write_text(
        stable_json(
            json_safe(
                {
                    "schema": "cem_fallback_decisions_v1",
                    "environment": args.env_name,
                    "seed": args.seed,
                    "cue_labels_used": False,
                    "cue_times_used": False,
                    "queries": decision_logs,
                }
            )
        )
    )
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "schema": "cem_fallback_selector_model_v1",
            "memory": final_memory.state_dict(),
            "fallback_ensemble": fallback.state_dict(),
            "event_ensemble": event_ensemble.state_dict(),
            "config": {
                "latent_dim": int(latents.shape[-1]),
                "action_dim": int(actions.shape[-1]),
                "memory_hidden": args.memory_hidden,
                "selector_hidden": args.selector_hidden,
                "ensemble_members": args.ensemble_members,
                "pool_slots": POOL_SLOTS,
            },
        },
        model_path,
    )
    if tensor_digest(host) != host_digest_before:
        raise RuntimeError("frozen host changed in fallback campaign")
    contract = {
        "passed": True,
        "input_keys": ["frozen DINO latents", "actions", "timestamps"],
        "cue_labels_used_for_training_or_selection": False,
        "cue_times_used_for_training_or_selection": False,
        "realized_future_used_at_test": False,
        "realized_future_used_for_oracles_only": True,
        "crossfit_policy_label_pair_overlap": 0,
        "selected_memory_bytes": serialized_token_bytes(
            latents.shape[-1], MEMORY_TOKENS
        ),
        "read_tokens": MEMORY_TOKENS,
        "online_host_calls": 1,
    }
    result = {
        "schema": "cem_fallback_selector_cell_v1",
        "status": "completed",
        "environment": args.env_name,
        "seed": args.seed,
        "device": str(device),
        "protocol": {
            "gaps": list(GAPS),
            "crossfit_target_gaps": list(HIGH_GAPS),
            "crossfit_folds": args.crossfit_folds,
            "occupancy_variants": OCCUPANCY_VARIANTS,
            "pool_slots": POOL_SLOTS,
            "event_slots": EVENT_SLOTS,
            "frame_slots": FRAME_SLOTS,
            "recent_slots": RECENT_SLOTS,
            "store_occupancy": STORE_OCCUPANCY,
            "selected_tokens": MEMORY_TOKENS,
            "selected_serialized_bytes": serialized_token_bytes(
                latents.shape[-1], MEMORY_TOKENS
            ),
            "proposal_pool_serialized_bytes": serialized_token_bytes(
                latents.shape[-1], POOL_SLOTS
            ),
            "controlled_splicing": True,
            "native_rollout_claim": False,
        },
        "no_manual_cue_contract": contract,
        "host": host_receipt,
        "crossfit": {
            "folds": crossfit_receipts,
            "oof_label_rows": len(oof_targets.target),
            "leakage_assertion_passed": True,
        },
        "candidate_pool": {
            "method": (
                "8 automatic surprise/change events + 8 label-free DINO "
                "k-center historical frames + 4 recent legal frames"
            ),
            "construction_seconds": pool_elapsed,
            "discovery_thresholds": {
                str(gap): {
                    "surprise": thresholds[gap][0],
                    "change": thresholds[gap][1],
                }
                for gap in GAPS
            },
        },
        "final_memory_training": final_memory_training,
        "fallback_training": fallback_training,
        "event_only_training": event_training,
        "test_ranking": ranking,
        "test_uncertainty": uncertainty,
        "efficiency": {
            "fallback_recall_non_host_ms": fallback_latency,
            "event_only_recall_non_host_ms": event_latency,
            "ratio_vs_event_only": latency_ratio,
            "target_max_ratio": 1.5,
            "passed": latency_ratio <= 1.5,
            "same_selected_bytes": True,
            "same_host_forwards": True,
        },
        "gaps": gap_results,
        "host_digest": tensor_digest(host),
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "evaluation": str(samples_path.relative_to(ROOT)),
            "decision_log": str(decision_path.relative_to(ROOT)),
            "model": str(model_path.relative_to(ROOT)),
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
                    "spearman": ranking["within_query_spearman"],
                    "pairwise": ranking["pairwise_accuracy"],
                    "deletion_gap": ranking["high_minus_random"],
                    "recovery_32": next(
                        row["learned_fallback_recovery"]
                        for row in gap_results
                        if row["gap"] == 32
                    ),
                    "result": str(result_path.relative_to(ROOT)),
                }
            )
        ),
        flush=True,
    )
    return result


def ranking_metric_from_arrays(
    arrays: dict[str, np.ndarray],
    metric: str,
    indices: np.ndarray,
) -> float:
    if metric == "spearman":
        return float(
            centered_rank_correlation(
                arrays["ranking_prediction"][indices],
                arrays["ranking_target"][indices],
                arrays["ranking_valid"][indices],
            )
        )
    if metric == "pairwise":
        count = arrays["ranking_pair_count"][indices].sum()
        return float(
            arrays["ranking_pair_correct"][indices].sum() / count
        ) if count else float("nan")
    if metric == "deletion_gap":
        values = arrays["ranking_deletion_gap"][indices]
        return float(np.nanmean(values))
    raise KeyError(metric)


def bootstrap_ranking_environment(
    cells: list[dict[str, Any]],
    metric: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    loaded = []
    for cell in cells:
        with np.load(
            ROOT / cell["artifacts"]["evaluation"],
            allow_pickle=False,
        ) as data:
            loaded.append(
                {key: np.asarray(data[key]) for key in data.files}
            )
    pair_metrics = []
    for arrays in loaded:
        pair_metrics.append(
            np.asarray(
                [
                    ranking_metric_from_arrays(
                        arrays,
                        metric,
                        np.flatnonzero(
                            arrays["ranking_pair_id"] == pair_id
                        ),
                    )
                    for pair_id in np.unique(
                        arrays["ranking_pair_id"]
                    )
                ],
                dtype=np.float64,
            )
        )
    seed_values = [
        float(np.nanmean(values)) for values in pair_metrics
    ]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        chosen_seeds = rng.choice(
            len(loaded), size=len(loaded), replace=True
        )
        values = []
        for seed_index in chosen_seeds:
            available = pair_metrics[int(seed_index)]
            sampled = rng.choice(
                available,
                size=len(available),
                replace=True,
            )
            values.append(float(np.nanmean(sampled)))
        draws.append(float(np.nanmean(values)))
    draws_np = np.asarray(draws, dtype=np.float64)
    return {
        "mean": float(np.nanmean(seed_values)),
        "ci95": np.quantile(
            draws_np, [0.025, 0.975]
        ).astype(float).tolist(),
        "seed_values": seed_values,
        "seed_count": len(loaded),
        "_draws": draws_np,
    }


def bootstrap_method_environment(
    cells: list[dict[str, Any]],
    gap: int,
    left: str,
    right: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    loaded = []
    for cell in cells:
        with np.load(
            ROOT / cell["artifacts"]["evaluation"],
            allow_pickle=False,
        ) as data:
            keep = np.asarray(data["gap"]) == gap
            loaded.append(
                {
                    "pair_id": np.asarray(data["pair_id"])[keep],
                    "value": (
                        np.asarray(data[f"loss_{right}"])[keep]
                        - np.asarray(data[f"loss_{left}"])[keep]
                    ),
                }
            )
    pair_metrics = [
        np.asarray(
            [
                arrays["value"][
                    arrays["pair_id"] == pair_id
                ].mean()
                for pair_id in np.unique(arrays["pair_id"])
            ],
            dtype=np.float64,
        )
        for arrays in loaded
    ]
    seed_values = [
        float(np.mean(values)) for values in pair_metrics
    ]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        chosen = rng.choice(
            len(loaded), size=len(loaded), replace=True
        )
        values = []
        for seed_index in chosen:
            available = pair_metrics[int(seed_index)]
            values.append(
                float(
                    rng.choice(
                        available,
                        size=len(available),
                        replace=True,
                    ).mean()
                )
            )
        draws.append(float(np.mean(values)))
    draws_np = np.asarray(draws, dtype=np.float64)
    return {
        "mean": float(np.mean(seed_values)),
        "ci95": np.quantile(
            draws_np, [0.025, 0.975]
        ).astype(float).tolist(),
        "seed_values": seed_values,
        "_draws": draws_np,
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/result.json")):
        result = json.loads(path.read_text())
        if result.get("schema") == "cem_fallback_selector_cell_v1":
            cells.append(result)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["environment"], []).append(cell)
    environments = []
    global_rank_draws: dict[str, list[np.ndarray]] = {}
    global_method_draws: dict[tuple[int, str], list[np.ndarray]] = {}
    for env_index, (environment, rows) in enumerate(sorted(grouped.items())):
        ranking = {}
        for metric_index, metric in enumerate(
            ("spearman", "pairwise", "deletion_gap")
        ):
            summary = bootstrap_ranking_environment(
                rows,
                metric,
                seed=81_001 + env_index * 101 + metric_index,
                repetitions=args.bootstrap_repetitions,
            )
            global_rank_draws.setdefault(metric, []).append(
                summary.pop("_draws")
            )
            ranking[metric] = summary
        gaps = []
        for gap in GAPS:
            cell_gap_rows = [
                next(
                    value for value in row["gaps"]
                    if value["gap"] == gap
                )
                for row in rows
            ]
            comparisons = {}
            for index, (name, left, right) in enumerate(
                (
                    (
                        "fallback_vs_recent",
                        "fallback_selector",
                        "recent_only",
                    ),
                    (
                        "fallback_vs_event",
                        "fallback_selector",
                        "event_only_conditional",
                    ),
                    (
                        "event_vs_recent",
                        "event_only_conditional",
                        "recent_only",
                    ),
                    (
                        "surprise_vs_recent",
                        "surprise",
                        "recent_only",
                    ),
                    (
                        "random_vs_recent",
                        "random",
                        "recent_only",
                    ),
                    (
                        "oracle_frame_vs_recent",
                        "oracle_frame",
                        "recent_only",
                    ),
                    (
                        "oracle_union_vs_recent",
                        "oracle_union",
                        "recent_only",
                    ),
                )
            ):
                summary = bootstrap_method_environment(
                    rows,
                    gap,
                    left,
                    right,
                    seed=91_001
                    + env_index * 1009
                    + gap * 7
                    + index,
                    repetitions=args.bootstrap_repetitions,
                )
                global_method_draws.setdefault((gap, name), []).append(
                    summary.pop("_draws")
                )
                comparisons[name] = summary
            frame_gain = comparisons["oracle_frame_vs_recent"]["mean"]
            gaps.append(
                {
                    "gap": gap,
                    "comparisons": comparisons,
                    "learned_recovery": (
                        comparisons["fallback_vs_recent"]["mean"]
                        / max(frame_gain, 1e-12)
                    ),
                    "union_recovery": (
                        comparisons["oracle_union_vs_recent"]["mean"]
                        / max(frame_gain, 1e-12)
                    ),
                    "proposal_oracle_frame_hit_rate": float(
                        np.mean(
                            [
                                value[
                                    "proposal_oracle_frame_hit_rate"
                                ]
                                for value in cell_gap_rows
                            ]
                        )
                    ),
                    "selected_candidate_type_frequency": {
                        type_name: float(
                            np.mean(
                                [
                                    value[
                                        "selected_candidate_type_frequency"
                                    ][type_name]
                                    for value in cell_gap_rows
                                ]
                            )
                        )
                        for type_name in CANDIDATE_TYPE_NAMES.values()
                    },
                    "event_uncertainty_fallback_rate": float(
                        np.mean(
                            [
                                value[
                                    "event_uncertainty_fallback_rate"
                                ]
                                for value in cell_gap_rows
                            ]
                        )
                    ),
                }
            )
        environments.append(
            {
                "environment": environment,
                "seeds": sorted(int(row["seed"]) for row in rows),
                "ranking": ranking,
                "gaps": gaps,
                "uncertainty": {
                    "coverage_ece_mean": float(
                        np.mean(
                            [
                                row["test_uncertainty"]["coverage_ece"]
                                for row in rows
                            ]
                        )
                    ),
                    "coverage_90_mean": float(
                        np.mean(
                            [
                                row["test_uncertainty"][
                                    "interval_coverage"
                                ]["0.9"]
                                for row in rows
                            ]
                        )
                    ),
                },
                "efficiency_ratio_mean": float(
                    np.mean(
                        [
                            row["efficiency"]["ratio_vs_event_only"]
                            for row in rows
                        ]
                    )
                ),
            }
        )
    aggregate_ranking = {}
    for metric in ("spearman", "pairwise", "deletion_gap"):
        rows = [env["ranking"][metric] for env in environments]
        draws = global_rank_draws.get(metric, [])
        if draws:
            length = min(len(value) for value in draws)
            combined = np.stack(
                [value[:length] for value in draws]
            ).mean(0)
            ci = np.quantile(
                combined, [0.025, 0.975]
            ).astype(float).tolist()
        else:
            ci = [None, None]
        aggregate_ranking[metric] = {
            "mean": (
                float(np.mean([row["mean"] for row in rows]))
                if rows
                else None
            ),
            "ci95": ci,
            "hierarchy": (
                "environment mean of seed/pair nested bootstrap"
            ),
        }
    aggregate_gaps = []
    for gap in GAPS:
        env_gap_rows = [
            next(
                value for value in environment["gaps"]
                if value["gap"] == gap
            )
            for environment in environments
        ]
        comparisons = {}
        for name in (
            "fallback_vs_recent",
            "fallback_vs_event",
            "event_vs_recent",
            "surprise_vs_recent",
            "random_vs_recent",
            "oracle_frame_vs_recent",
            "oracle_union_vs_recent",
        ):
            env_rows = [
                next(
                    value
                    for value in environment["gaps"]
                    if value["gap"] == gap
                )["comparisons"][name]
                for environment in environments
            ]
            draws = global_method_draws.get((gap, name), [])
            if draws:
                length = min(len(value) for value in draws)
                combined = np.stack(
                    [value[:length] for value in draws]
                ).mean(0)
                ci = np.quantile(
                    combined, [0.025, 0.975]
                ).astype(float).tolist()
            else:
                ci = [None, None]
            comparisons[name] = {
                "mean": (
                    float(
                        np.mean([row["mean"] for row in env_rows])
                    )
                    if env_rows
                    else None
                ),
                "ci95": ci,
            }
        frame_gain = comparisons["oracle_frame_vs_recent"]["mean"]
        aggregate_gaps.append(
            {
                "gap": gap,
                "comparisons": comparisons,
                "learned_recovery": (
                    comparisons["fallback_vs_recent"]["mean"]
                    / max(frame_gain, 1e-12)
                    if frame_gain is not None
                    else None
                ),
                "union_recovery": (
                    comparisons["oracle_union_vs_recent"]["mean"]
                    / max(frame_gain, 1e-12)
                    if frame_gain is not None
                    else None
                ),
                "proposal_oracle_frame_hit_rate": (
                    float(
                        np.mean(
                            [
                                row["proposal_oracle_frame_hit_rate"]
                                for row in env_gap_rows
                            ]
                        )
                    )
                    if env_gap_rows
                    else None
                ),
                "selected_candidate_type_frequency": {
                    type_name: (
                        float(
                            np.mean(
                                [
                                    row[
                                        "selected_candidate_type_frequency"
                                    ][type_name]
                                    for row in env_gap_rows
                                ]
                            )
                        )
                        if env_gap_rows
                        else None
                    )
                    for type_name in CANDIDATE_TYPE_NAMES.values()
                },
                "event_uncertainty_fallback_rate": (
                    float(
                        np.mean(
                            [
                                row[
                                    "event_uncertainty_fallback_rate"
                                ]
                                for row in env_gap_rows
                            ]
                        )
                    )
                    if env_gap_rows
                    else None
                ),
            }
        )
    positive_deletion_envs = [
        environment["environment"]
        for environment in environments
        if environment["ranking"]["deletion_gap"]["ci95"][0] > 0
    ]
    rank_pass = bool(
        aggregate_ranking["spearman"]["ci95"][0] > 0.2
        or aggregate_ranking["pairwise"]["mean"] >= 0.65
    )
    deletion_pass = len(positive_deletion_envs) >= 2
    ranking_gate = {
        "passed": rank_pass and deletion_pass,
        "rank_pass": rank_pass,
        "deletion_pass": deletion_pass,
        "positive_deletion_environments": positive_deletion_envs,
        "rule": (
            "Spearman lower CI >0.2 OR pairwise >=0.65, and deletion "
            "lower CI >0 in >=2/3 environments"
        ),
    }
    high_rows = [
        row for row in aggregate_gaps if row["gap"] in HIGH_GAPS
    ]
    opportunity_pass = all(
        row["comparisons"]["oracle_frame_vs_recent"]["ci95"][0] > 0
        for row in high_rows
    )
    resolved_wins = [
        row["gap"]
        for row in high_rows
        if (
            row["comparisons"]["fallback_vs_recent"]["ci95"][0] > 0
            and row["comparisons"]["fallback_vs_event"]["ci95"][0] > 0
        )
    ]
    high_frame = float(
        np.mean(
            [
                row["comparisons"]["oracle_frame_vs_recent"]["mean"]
                for row in high_rows
            ]
        )
    )
    high_fallback = float(
        np.mean(
            [
                row["comparisons"]["fallback_vs_recent"]["mean"]
                for row in high_rows
            ]
        )
    )
    recovery = high_fallback / max(high_frame, 1e-12)
    recovery_gate = {
        "passed": bool(
            opportunity_pass
            and recovery >= 0.70
            and len(resolved_wins) >= 2
        ),
        "opportunity_pass": opportunity_pass,
        "high_gap_recovery": recovery,
        "resolved_fallback_wins_gaps": resolved_wins,
        "rule": (
            "oracle frame resolved at gaps>=32; fallback recovery >=70%; "
            "fallback beats event-only and recent at >=2 high gaps"
        ),
    }
    efficiency_values = [
        row["efficiency"]["ratio_vs_event_only"] for row in cells
    ]
    efficiency_gate = {
        "passed": bool(
            efficiency_values and max(efficiency_values) <= 1.5
        ),
        "mean_ratio": (
            float(np.mean(efficiency_values))
            if efficiency_values
            else None
        ),
        "maximum_ratio": (
            float(np.max(efficiency_values))
            if efficiency_values
            else None
        ),
        "rule": "<=1.5x event-only non-host recall latency",
    }
    all_recovery_gates = bool(
        ranking_gate["passed"]
        and recovery_gate["passed"]
        and efficiency_gate["passed"]
    )
    report = {
        "schema": "cem_fallback_selector_report_v1",
        "status": "completed" if cells else "empty",
        "cell_count": len(cells),
        "environment_count": len(environments),
        "protocol": {
            "environments": list(ENVIRONMENTS),
            "seeds_per_environment": 3,
            "gaps": list(GAPS),
            "high_target_gaps": list(HIGH_GAPS),
            "crossfit_folds": args.crossfit_folds,
            "pool_slots": POOL_SLOTS,
            "selected_tokens": MEMORY_TOKENS,
            "selected_serialized_bytes": 1536,
            "proposal_pool_serialized_bytes": 7680,
            "same_online_host_calls": True,
            "cue_labels_or_times": False,
        },
        "environments": environments,
        "aggregate_ranking": aggregate_ranking,
        "aggregate_gaps": aggregate_gaps,
        "uncertainty": {
            "coverage_ece_mean": (
                float(
                    np.mean(
                        [
                            row["uncertainty"]["coverage_ece_mean"]
                            for row in environments
                        ]
                    )
                )
                if environments
                else None
            ),
            "coverage_90_mean": (
                float(
                    np.mean(
                        [
                            row["uncertainty"]["coverage_90_mean"]
                            for row in environments
                        ]
                    )
                )
                if environments
                else None
            ),
        },
        "gates": {
            "conditional_ranking": ranking_gate,
            "opportunity_recovery": recovery_gate,
            "efficiency": efficiency_gate,
            "all_passed": all_recovery_gates,
        },
        "graph_reconsideration_authorized": all_recovery_gates,
        "jobs_still_running": [],
        "artifacts": {
            "cells": "outputs/cem_fallback_selector_v1/cells/<env>/s<seed>",
            "report": "outputs/cem_fallback_selector_v1/report.json",
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
    parser.add_argument(
        "--graph-output", type=Path, default=DEFAULT_GRAPH_OUTPUT
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--discovery-batch-size", type=int, default=4096)
    parser.add_argument("--memory-hidden", type=int, default=192)
    parser.add_argument("--memory-epochs", type=int, default=24)
    parser.add_argument("--memory-patience", type=int, default=6)
    parser.add_argument("--memory-lr", type=float, default=4e-4)
    parser.add_argument("--selector-hidden", type=int, default=192)
    parser.add_argument("--selector-epochs", type=int, default=28)
    parser.add_argument("--selector-patience", type=int, default=6)
    parser.add_argument("--selector-lr", type=float, default=4e-4)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--oracle-set-chunk", type=int, default=24)
    parser.add_argument("--latency-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    for name in (
        "output",
        "base_output",
        "conditional_output",
        "graph_output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        parser.error("--gpu must be one of 0,1,2; GPU3 is prohibited")
    if args.smoke:
        args.limit_pairs = args.limit_pairs or 6
        args.memory_epochs = min(args.memory_epochs, 3)
        args.memory_patience = min(args.memory_patience, 2)
        args.selector_epochs = min(args.selector_epochs, 3)
        args.selector_patience = min(args.selector_patience, 2)
        args.ensemble_members = min(args.ensemble_members, 2)
        args.latency_repetitions = min(args.latency_repetitions, 20)
        args.bootstrap_repetitions = min(
            args.bootstrap_repetitions, 100
        )
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
