#!/usr/bin/env python3
"""Decision-conditioned oracle ladder for the PointMaze anti-recency task."""
from __future__ import annotations

import argparse
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

from scripts.build_cem_decision_task import (  # noqa: E402
    BASE,
    ENV,
    OUTPUT,
    source_audit,
)
from scripts.build_graph_cem_long_gap import (  # noqa: E402
    FUTURE_ACTION_TIMES,
)
from scripts.run_cem_raw_ogbench import (  # noqa: E402
    QueryTensors,
    RawMemoryConditioner,
    batches,
    horizon_loss,
    json_safe,
    rollout,
    set_seed,
    stable_json,
    tensor_digest,
)
from scripts.run_graph_cem_long_gap import (  # noqa: E402
    CONTEXT,
    HORIZON,
    MEMORY_TOKENS,
    GapSplit,
    annotate_surprise,
    build_raw_gap,
    discovery_thresholds,
    feature_path,
    indices_to_events,
    load_host,
    recent_indices,
    tensorize_gap,
)

GAPS = (32, 64, 128)
CANDIDATES = 4
EVENTS = 8


class DecisionHead(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 192) -> None:
        super().__init__()
        query_dim = 4 * latent_dim + 1
        self.query = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.action = nn.Sequential(
            nn.LayerNorm(HORIZON * action_dim),
            nn.Linear(HORIZON * action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        goal: torch.Tensor,
        memory: torch.Tensor,
        age: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(
            torch.cat([context.flatten(1), goal, memory, age[:, None]], dim=-1)
        )
        action = self.action(actions.flatten(2))
        expanded = query[:, None].expand(-1, actions.shape[1], -1)
        score = self.output(
            torch.cat([expanded, action, expanded * action], dim=-1)
        ).squeeze(-1)
        return score


class UtilityRouter(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


class DecisionSplit:
    def __init__(
        self,
        context: np.ndarray,
        goal: np.ndarray,
        actions: np.ndarray,
        correct: np.ndarray,
        recent: np.ndarray,
        events: np.ndarray,
        ages: np.ndarray,
        proposal: np.ndarray,
        episode_ids: np.ndarray,
        pair_ids: np.ndarray,
        gaps: np.ndarray,
    ) -> None:
        self.context = context
        self.goal = goal
        self.actions = actions
        self.correct = correct
        self.recent = recent
        self.events = events
        self.ages = ages
        self.proposal = proposal
        self.episode_ids = episode_ids
        self.pair_ids = pair_ids
        self.gaps = gaps

    def __len__(self) -> int:
        return len(self.correct)


def cell_dir(output: Path, seed: int) -> Path:
    return output / "cells" / ENV / f"s{seed}"


def load_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, RawMemoryConditioner, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    host_args = argparse.Namespace(
        env_name=ENV,
        seed=args.seed,
        base_output=BASE,
        conditional_output=ROOT / "outputs/graph_cem_conditional_v1",
    )
    host, host_receipt = load_host(host_args, device)
    graph_model = (
        ROOT / "outputs/graph_cem_long_gap_v1/cells"
        / ENV / f"s{args.seed}/model.pt"
    )
    checkpoint = torch.load(
        graph_model, map_location=device, weights_only=True,
    )
    config = checkpoint["memory_config"]
    memory = RawMemoryConditioner(
        config["latent_dim"],
        config["action_dim"],
        config["hidden"],
        config["budget"],
    ).to(device)
    memory.load_state_dict(checkpoint["memory"])
    memory.eval()
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    with np.load(feature_path(BASE, ENV), allow_pickle=False) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    with np.load(
        args.output / "build" / ENV / "pairs.npz",
        allow_pickle=False,
    ) as data:
        sources = np.asarray(data["test_sources"], dtype=np.int64)
        donors = np.asarray(data["test_donors"], dtype=np.int64)
    return host, memory, latents, actions, sources, donors, host_receipt


def build_decision_split(
    args: argparse.Namespace,
    host: Any,
    latents: np.ndarray,
    actions: np.ndarray,
    split_name: str,
    device: torch.device,
) -> DecisionSplit:
    with np.load(
        args.output / "build" / ENV / "pairs.npz",
        allow_pickle=False,
    ) as data:
        pair_sources = np.asarray(
            data[f"{split_name}_sources"], dtype=np.int64,
        )
        donors = np.asarray(
            data[f"{split_name}_donors"], dtype=np.int64,
        )
    graph_result = json.loads(
        (
            ROOT / "outputs/graph_cem_long_gap_v1/cells"
            / ENV / f"s{args.seed}/result.json"
        ).read_text()
    )
    rows: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "context", "goal", "actions", "correct", "recent", "events",
            "ages", "proposal", "episode_ids", "pair_ids", "gaps",
        )
    }
    for gap_value in GAPS:
        raw = build_raw_gap(
            latents,
            actions,
            pair_sources,
            donors,
            split_name,
            gap_value,
            limit_pairs=args.limit_pairs if args.smoke else None,
        )
        annotate_surprise(
            host, raw, device, args.discovery_batch_size,
        )
        threshold_row = graph_result["discovery"]["thresholds"][
            str(gap_value)
        ]
        tensorized = tensorize_gap(
            raw,
            (
                float(threshold_row["surprise"]),
                float(threshold_row["semantic_change"]),
            ),
            device,
        )
        candidate, correct, _ = candidate_actions(
            raw,
            actions,
            pair_sources,
            args.seed + gap_value + len(split_name) * 101,
        )
        legal = raw.history.shape[1] - CONTEXT
        event_index = tensorized.candidate_indices[:, :EVENTS].cpu().numpy()
        event_index = np.minimum(event_index, legal - MEMORY_TOKENS - 1)
        event_index = np.maximum(event_index, 0)
        event = np.take_along_axis(
            raw.history,
            np.repeat(
                event_index[..., None],
                raw.history.shape[-1],
                axis=-1,
            ),
            axis=1,
        )
        age = (legal - event_index) / max(1, legal)
        proposal = np.take_along_axis(
            tensorized.discovery_score.cpu().numpy(),
            event_index,
            axis=1,
        )
        recent_index = np.arange(legal - MEMORY_TOKENS, legal)
        recent = raw.history[:, recent_index].mean(1)
        rows["context"].append(raw.context_z)
        rows["goal"].append(raw.targets.mean(1))
        rows["actions"].append(candidate.numpy())
        rows["correct"].append(correct)
        rows["recent"].append(recent)
        rows["events"].append(event)
        rows["ages"].append(age.astype(np.float32))
        rows["proposal"].append(proposal.astype(np.float32))
        rows["episode_ids"].append(raw.sources)
        rows["pair_ids"].append(
            raw.pair_ids + gap_value * 10_000
        )
        rows["gaps"].append(
            np.full(len(raw.history), gap_value, dtype=np.int64)
        )
    return DecisionSplit(
        **{
            key: np.concatenate(value)
            for key, value in rows.items()
        }
    )


def candidate_actions(
    raw: Any,
    actions: np.ndarray,
    pair_sources: np.ndarray,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, list[list[int]]]:
    count = len(raw.history)
    values = np.empty(
        (count, CANDIDATES, HORIZON, actions.shape[-1]),
        dtype=np.float32,
    )
    correct = np.empty(count, dtype=np.int64)
    source_ids: list[list[int]] = []
    for row in range(count):
        pair = int(raw.pair_ids[row])
        branch = int(raw.branches[row])
        own = int(raw.sources[row])
        other = int(pair_sources[pair, 1 - branch])
        donor = int(raw.donors[row])
        unrelated = int(pair_sources[(pair + 1) % len(pair_sources), branch])
        ids = [own, other, donor, unrelated]
        candidates = np.asarray(
            [actions[index, FUTURE_ACTION_TIMES] for index in ids],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed * 100_003 + row)
        order = rng.permutation(CANDIDATES)
        values[row] = candidates[order]
        correct[row] = int(np.flatnonzero(order == 0)[0])
        source_ids.append([ids[int(index)] for index in order])
    return torch.from_numpy(values), correct, source_ids


def repeat_batch(
    batch: QueryTensors,
    repeats: int,
    future_actions: torch.Tensor,
) -> QueryTensors:
    def repeated(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(
            -1, repeats, *value.shape[1:]
        ).reshape(-1, *value.shape[1:])

    return QueryTensors(
        context_z=repeated(batch.context_z),
        action_history=repeated(batch.action_history),
        future_actions=future_actions.reshape(
            -1, HORIZON, future_actions.shape[-1]
        ),
        targets=repeated(batch.targets),
        events=repeated(batch.events),
        metadata=repeated(batch.metadata),
        valid=repeated(batch.valid),
        recent_event=repeated(batch.recent_event),
    )


@torch.no_grad()
def action_losses_for_sets(
    host: Any,
    memory: RawMemoryConditioner | None,
    gap: GapSplit,
    action_candidates: torch.Tensor,
    index_sets: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    device = gap.batch.context_z.device
    actions = action_candidates.to(device)
    count, set_count = (
        (len(gap.batch), 1)
        if index_sets is None
        else index_sets.shape[:2]
    )
    output = []
    chunk_sets = 16
    for set_start in range(0, set_count, chunk_sets):
        set_stop = min(set_count, set_start + chunk_sets)
        local_sets = set_stop - set_start
        expanded_actions = actions[:, None].expand(
            -1, local_sets, -1, -1, -1
        ).reshape(count * local_sets, CANDIDATES, HORIZON, -1)
        base = repeat_batch(
            gap.batch,
            local_sets * CANDIDATES,
            expanded_actions.reshape(
                count, local_sets * CANDIDATES, HORIZON, -1
            ),
        )
        if memory is None:
            events = metadata = mask = None
        else:
            local_index = index_sets[:, set_start:set_stop]
            flat_index = local_index.reshape(count * local_sets, MEMORY_TOKENS)
            repeated_history = gap.history[:, None].expand(
                -1, local_sets, -1, -1
            ).reshape(count * local_sets, *gap.history.shape[1:])
            repeated_meta = gap.history_metadata[:, None].expand(
                -1, local_sets, -1, -1
            ).reshape(count * local_sets, *gap.history_metadata.shape[1:])
            latent_dim = repeated_history.shape[-1]
            events = torch.gather(
                repeated_history,
                1,
                flat_index[..., None].expand(-1, -1, latent_dim),
            )
            metadata = torch.gather(
                repeated_meta,
                1,
                flat_index[..., None].expand(-1, -1, 3),
            )
            events = events[:, None].expand(
                -1, CANDIDATES, -1, -1
            ).reshape(-1, MEMORY_TOKENS, latent_dim)
            metadata = metadata[:, None].expand(
                -1, CANDIDATES, -1, -1
            ).reshape(-1, MEMORY_TOKENS, 3)
            mask = torch.ones(
                events.shape[:2], dtype=torch.bool, device=device,
            )
        losses = []
        for start in range(0, len(base), batch_size):
            stop = min(len(base), start + batch_size)
            index = torch.arange(start, stop, device=device)
            part = base.index(index)
            prediction, _ = rollout(
                host,
                memory,
                part,
                events=None if events is None else events[start:stop],
                metadata=None if metadata is None else metadata[start:stop],
                mask=None if mask is None else mask[start:stop],
            )
            losses.append(
                horizon_loss(prediction, part.targets).mean(1)
            )
        value = torch.cat(losses).reshape(
            count, local_sets, CANDIDATES
        )
        output.append(value)
    return torch.cat(output, 1)


def margin_and_accuracy(
    losses: torch.Tensor,
    correct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = losses.detach().cpu().numpy()
    rows = np.arange(len(values))
    correct_loss = values[rows, correct]
    distractor = values.copy()
    distractor[rows, correct] = np.inf
    margin = distractor.min(1) - correct_loss
    rank = 1 + np.sum(values < correct_loss[:, None], axis=1)
    return margin, (rank == 1).astype(np.float32), rank.astype(np.int64)


def index_sets_for_method(
    gap: GapSplit,
    method: str,
    seed: int,
) -> torch.Tensor:
    recent = recent_indices(gap)
    legal = gap.history.shape[1] - CONTEXT
    recent_three = recent[:, -3:]
    if method == "recent_only":
        return recent[:, None]
    if method == "oracle_frame":
        choices = torch.tensor(
            [2, 3], device=gap.history.device,
        )[None].expand(len(gap.batch), -1)
    elif method == "oracle_discovered_event":
        choices = gap.candidate_indices[:, : min(8, gap.candidate_indices.shape[1])]
        choices = torch.where(
            choices < legal - MEMORY_TOKENS,
            choices,
            torch.zeros_like(choices),
        )
    elif method == "all_history_upper":
        choices = torch.arange(
            max(1, legal - MEMORY_TOKENS),
            device=gap.history.device,
        )[None].expand(len(gap.batch), -1)
    elif method == "surprise":
        choices = gap.candidate_indices[:, :1]
    elif method == "random_event":
        generator = torch.Generator(device=gap.history.device)
        generator.manual_seed(seed)
        choices = torch.randint(
            0,
            max(1, legal - MEMORY_TOKENS),
            (len(gap.batch), 1),
            generator=generator,
            device=gap.history.device,
        )
    else:
        raise KeyError(method)
    return torch.cat(
        [
            choices[..., None],
            recent_three[:, None].expand(-1, choices.shape[1], -1),
        ],
        dim=2,
    )


def pair_bootstrap(
    values: np.ndarray,
    pair_ids: np.ndarray,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    unique = np.unique(pair_ids)
    grouped = np.asarray(
        [values[pair_ids == pair].mean() for pair in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(grouped, len(grouped), replace=True).mean()
            for _ in range(repetitions)
        ]
    )
    return {
        "mean": float(grouped.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).astype(float).tolist(),
        "pair_count": int(len(grouped)),
    }


def tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value)).to(device)


def decision_scores(
    model: DecisionHead,
    data: DecisionSplit,
    memory: np.ndarray,
    age: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(len(data), start + batch_size)
            output.append(
                model(
                    tensor(data.context[start:stop], device).float(),
                    tensor(data.goal[start:stop], device).float(),
                    tensor(memory[start:stop], device).float(),
                    tensor(age[start:stop], device).float(),
                    tensor(data.actions[start:stop], device).float(),
                ).cpu().numpy()
            )
    return np.concatenate(output)


def train_decision_head(
    args: argparse.Namespace,
    train: DecisionSplit,
    validation: DecisionSplit,
    device: torch.device,
) -> tuple[DecisionHead, dict[str, Any]]:
    model = DecisionHead(
        train.context.shape[-1],
        train.actions.shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-4, weight_decay=1e-4,
    )
    rng = np.random.default_rng(args.seed + 19_001)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(args.decision_epochs):
        model.train()
        losses = []
        for indices in batches(len(train), args.batch_size, rng):
            event_slot = rng.integers(0, EVENTS, size=len(indices))
            event_memory = train.events[indices, event_slot]
            event_age = train.ages[indices, event_slot]
            use_recent = rng.random(len(indices)) < 0.25
            memory = np.where(
                use_recent[:, None],
                train.recent[indices],
                event_memory,
            )
            age = np.where(use_recent, 0.0, event_age)
            score = model(
                tensor(train.context[indices], device).float(),
                tensor(train.goal[indices], device).float(),
                tensor(memory, device).float(),
                tensor(age, device).float(),
                tensor(train.actions[indices], device).float(),
            )
            label = tensor(train.correct[indices], device).long()
            loss = F.cross_entropy(-score, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        recent_score = decision_scores(
            model,
            validation,
            validation.recent,
            np.zeros(len(validation), dtype=np.float32),
            device,
            args.batch_size,
        )
        val_loss = float(
            F.cross_entropy(
                -torch.from_numpy(recent_score),
                torch.from_numpy(validation.correct).long(),
            )
        )
        accuracy = float(
            np.mean(np.argmin(recent_score, axis=1) == validation.correct)
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_cross_entropy": float(np.mean(losses)),
                "validation_recent_cross_entropy": val_loss,
                "validation_recent_accuracy": accuracy,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("decision head produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "history": history,
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }


def utility_targets(
    model: DecisionHead,
    data: DecisionSplit,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recent_score = decision_scores(
        model,
        data,
        data.recent,
        np.zeros(len(data), dtype=np.float32),
        device,
        batch_size,
    )
    event_scores = []
    for slot in range(EVENTS):
        event_scores.append(
            decision_scores(
                model,
                data,
                data.events[:, slot],
                data.ages[:, slot],
                device,
                batch_size,
            )
        )
    event_score = np.stack(event_scores, axis=1)
    label = torch.from_numpy(data.correct).long()
    recent_ce = F.cross_entropy(
        -torch.from_numpy(recent_score),
        label,
        reduction="none",
    ).numpy()
    event_ce = np.stack(
        [
            F.cross_entropy(
                -torch.from_numpy(event_score[:, slot]),
                label,
                reduction="none",
            ).numpy()
            for slot in range(EVENTS)
        ],
        axis=1,
    )
    return recent_ce[:, None] - event_ce, recent_score, event_score


def router_features(data: DecisionSplit) -> np.ndarray:
    count = len(data)
    common = np.concatenate(
        [
            data.context.reshape(count, -1),
            data.goal,
            data.actions.reshape(count, -1),
        ],
        axis=1,
    )
    return np.concatenate(
        [
            np.repeat(common[:, None], EVENTS, axis=1),
            data.events,
            data.ages[..., None],
            data.proposal[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def train_router(
    args: argparse.Namespace,
    train: DecisionSplit,
    validation: DecisionSplit,
    train_target: np.ndarray,
    validation_target: np.ndarray,
    device: torch.device,
) -> tuple[list[UtilityRouter], dict[str, Any]]:
    train_feature = router_features(train)
    val_feature = router_features(validation)
    models = []
    receipts = []
    for member in range(3):
        model = UtilityRouter(train_feature.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=1e-4,
        )
        rng = np.random.default_rng(args.seed * 1009 + member)
        best_state = None
        best_loss = float("inf")
        for epoch in range(args.router_epochs):
            order = rng.permutation(len(train))
            for start in range(0, len(order), args.batch_size):
                index = order[start : start + args.batch_size]
                feature = tensor(
                    train_feature[index].reshape(-1, train_feature.shape[-1]),
                    device,
                ).float()
                target = tensor(
                    train_target[index].reshape(-1), device,
                ).float()
                prediction = model(feature)
                regression = F.smooth_l1_loss(
                    prediction, target, beta=0.02,
                )
                pred_rows = prediction.reshape(len(index), EVENTS)
                truth_rows = target.reshape(len(index), EVENTS)
                ranking = -(
                    torch.softmax(truth_rows / 0.05, dim=1).detach()
                    * torch.log_softmax(pred_rows / 0.05, dim=1)
                ).sum(1).mean()
                loss = regression + 0.25 * ranking
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            with torch.no_grad():
                val_prediction = model(
                    tensor(
                        val_feature.reshape(-1, val_feature.shape[-1]),
                        device,
                    ).float()
                )
                val_loss = float(
                    F.smooth_l1_loss(
                        val_prediction,
                        tensor(validation_target.reshape(-1), device).float(),
                        beta=0.02,
                    )
                )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if best_state is None:
            raise RuntimeError("router produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
        receipts.append(
            {"member": member, "best_validation_huber": best_loss}
        )
    return models, {"members": receipts}


def router_predict(
    models: list[UtilityRouter],
    data: DecisionSplit,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature = router_features(data)
    outputs = []
    with torch.no_grad():
        value = tensor(
            feature.reshape(-1, feature.shape[-1]), device,
        ).float()
        for model in models:
            outputs.append(
                model(value).reshape(len(data), EVENTS).cpu().numpy()
            )
    stack = np.stack(outputs)
    return stack.mean(0), stack.std(0)


def du_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    rho = float(
        spearmanr(prediction.reshape(-1), target.reshape(-1)).statistic
    )
    correct = 0
    count = 0
    for row in range(len(prediction)):
        for first in range(EVENTS):
            for second in range(first + 1, EVENTS):
                truth = target[row, first] - target[row, second]
                if abs(float(truth)) <= 1e-8:
                    continue
                correct += int(
                    (prediction[row, first] - prediction[row, second])
                    * truth > 0
                )
                count += 1
    high = target[np.arange(len(target)), np.argmax(prediction, axis=1)]
    rng = np.random.default_rng(311)
    random_value = target[
        np.arange(len(target)),
        rng.integers(0, EVENTS, size=len(target)),
    ]
    return {
        "spearman": rho if math.isfinite(rho) else None,
        "pairwise_accuracy": float(correct / count) if count else None,
        "pair_count": count,
        "high_utility": float(high.mean()),
        "random_utility": float(random_value.mean()),
        "high_minus_random": float((high - random_value).mean()),
    }


def action_accuracy(score: np.ndarray, correct: np.ndarray) -> np.ndarray:
    return (np.argmin(score, axis=1) == correct).astype(np.float32)


def run_phase2(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = cell_dir(args.output, args.seed)
    result_path = output_dir / "decision_result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    host, _, latents, actions, _, _, host_receipt = load_inputs(
        args, device,
    )
    host_digest = tensor_digest(host)
    data = {
        split: build_decision_split(
            args, host, latents, actions, split, device,
        )
        for split in ("train", "validation", "test")
    }
    split_sets = {
        split: set(value.episode_ids.tolist())
        for split, value in data.items()
    }
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("decision split leakage")
    head, head_training = train_decision_head(
        args, data["train"], data["validation"], device,
    )
    targets = {}
    recent_scores = {}
    event_scores = {}
    for split in data:
        targets[split], recent_scores[split], event_scores[split] = (
            utility_targets(
                head, data[split], device, args.batch_size,
            )
        )
    router, router_training = train_router(
        args,
        data["train"],
        data["validation"],
        targets["train"],
        targets["validation"],
        device,
    )
    val_mean, val_std = router_predict(
        router, data["validation"], device,
    )
    standardized = (
        val_mean - targets["validation"]
    ) / np.maximum(val_std, 1e-5)
    q = max(0.0, float(np.quantile(standardized, 0.90)))
    val_lcb = val_mean - q * np.maximum(val_std, 1e-5)
    deltas = sorted(
        set(
            [0.0]
            + np.quantile(val_lcb.max(1), np.linspace(0, 0.9, 20))
            .astype(float).tolist()
        )
    )
    recent_val_accuracy = action_accuracy(
        recent_scores["validation"], data["validation"].correct,
    )
    selected_delta = None
    best_coverage = -1.0
    for delta in deltas:
        slot = np.argmax(val_lcb, axis=1)
        active = val_lcb[np.arange(len(slot)), slot] > delta
        selected_score = event_scores["validation"][
            np.arange(len(slot)), slot
        ]
        policy_score = np.where(
            active[:, None],
            selected_score,
            recent_scores["validation"],
        )
        gain = float(
            action_accuracy(
                policy_score, data["validation"].correct,
            ).mean() - recent_val_accuracy.mean()
        )
        coverage = float(active.mean())
        if gain >= 0.0 and coverage > best_coverage:
            selected_delta = delta
            best_coverage = coverage
    if selected_delta is None:
        selected_delta = float("inf")
    test_mean, test_std = router_predict(
        router, data["test"], device,
    )
    test_lcb = test_mean - q * np.maximum(test_std, 1e-5)
    selected_slot = np.argmax(test_lcb, axis=1)
    active = (
        test_lcb[np.arange(len(selected_slot)), selected_slot]
        > selected_delta
    )
    learned_score = event_scores["test"][
        np.arange(len(selected_slot)), selected_slot
    ]
    policy_score = np.where(
        active[:, None], learned_score, recent_scores["test"],
    )
    oracle_slot = np.argmax(targets["test"], axis=1)
    oracle_score = event_scores["test"][
        np.arange(len(oracle_slot)), oracle_slot
    ]
    surprise_score = event_scores["test"][:, 0]
    rng = np.random.default_rng(args.seed + 51_003)
    random_slot = rng.integers(0, EVENTS, size=len(data["test"]))
    random_score = event_scores["test"][
        np.arange(len(random_slot)), random_slot
    ]
    conditions = {
        "recent_only": recent_scores["test"],
        "learned_lcb": policy_score,
        "always_learned_event": learned_score,
        "surprise": surprise_score,
        "random": random_score,
        "oracle_event": oracle_score,
    }
    accuracy = {
        name: action_accuracy(score, data["test"].correct)
        for name, score in conditions.items()
    }
    recent = accuracy["recent_only"]
    metrics = {
        name: {
            "accuracy": pair_bootstrap(
                values,
                data["test"].pair_ids,
                args.seed + 61_003 + len(name),
                args.bootstrap_repetitions,
            ),
            "gain_vs_recent": pair_bootstrap(
                values - recent,
                data["test"].pair_ids,
                args.seed + 62_003 + len(name),
                args.bootstrap_repetitions,
            ),
        }
        for name, values in accuracy.items()
    }
    du = du_metrics(test_mean, targets["test"])
    activated_positive = (
        targets["test"][
            np.arange(len(selected_slot)), selected_slot
        ][active]
    )
    activation = {
        "coverage": float(active.mean()),
        "precision_positive": (
            float(np.mean(activated_positive > 0.0))
            if len(activated_positive)
            else None
        ),
        "mean_true_utility": (
            float(activated_positive.mean())
            if len(activated_positive)
            else None
        ),
        "selected_delta": (
            None if not math.isfinite(selected_delta)
            else selected_delta
        ),
        "conformal_quantile": q,
    }
    if tensor_digest(host) != host_digest:
        raise RuntimeError("frozen host changed in decision phase")
    torch.save(
        {
            "schema": "cem_decision_head_v1",
            "head": head.state_dict(),
            "router": [model.state_dict() for model in router],
            "host_digest": host_digest,
        },
        output_dir / "decision_model.pt",
    )
    decision_logs = []
    for row in range(len(data["test"])):
        decision_logs.append(
            {
                "episode_id": int(data["test"].episode_ids[row]),
                "gap": int(data["test"].gaps[row]),
                "query_goal_norm": float(
                    np.linalg.norm(data["test"].goal[row])
                ),
                "candidate_actions": data["test"].actions[row].astype(float).tolist(),
                "selected_event_slot": int(selected_slot[row]),
                "du_hat": float(test_mean[row, selected_slot[row]]),
                "du_true": float(
                    targets["test"][row, selected_slot[row]]
                ),
                "lcb": float(test_lcb[row, selected_slot[row]]),
                "abstained": bool(not active[row]),
                "outcome": bool(accuracy["learned_lcb"][row]),
            }
        )
    (output_dir / "decision_phase_log.json").write_text(
        stable_json(json_safe({"queries": decision_logs}))
    )
    result = {
        "schema": "cem_decision_head_cell_v1",
        "status": "completed",
        "environment": ENV,
        "seed": args.seed,
        "host": host_receipt,
        "host_digest": host_digest,
        "head_training": head_training,
        "router_training": router_training,
        "test": {
            "conditions": metrics,
            "du": du,
            "activation": activation,
        },
        "ordinary_degradation": float(
            metrics["learned_lcb"]["gain_vs_recent"]["mean"] * -1.0
        ),
        "source_contract": source_audit(),
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "model": str(
                (output_dir / "decision_model.pt").relative_to(ROOT)
            ),
            "decision_log": str(
                (output_dir / "decision_phase_log.json").relative_to(ROOT)
            ),
        },
    }
    result_path.write_text(stable_json(json_safe(result)))
    print(
        stable_json(
            {
                "seed": args.seed,
                "recent_accuracy": metrics["recent_only"]["accuracy"]["mean"],
                "learned_accuracy": metrics["learned_lcb"]["accuracy"]["mean"],
                "learned_gain": metrics["learned_lcb"]["gain_vs_recent"]["mean"],
                "du": du,
                "activation": activation,
            }
        ),
        flush=True,
    )
    return result


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = cell_dir(args.output, args.seed)
    result_path = output_dir / "result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        raise ValueError("GPU3 prohibited; use GPU 0/1/2")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    (
        host,
        memory,
        latents,
        actions,
        pair_sources,
        donors,
        host_receipt,
    ) = load_inputs(args, device)
    host_digest = tensor_digest(host)
    memory_digest = tensor_digest(memory)
    graph_result = json.loads(
        (
            ROOT / "outputs/graph_cem_long_gap_v1/cells"
            / ENV / f"s{args.seed}/result.json"
        ).read_text()
    )
    gap_results = []
    evaluation_rows = []
    decisions = []
    for gap_value in GAPS:
        raw = build_raw_gap(
            latents,
            actions,
            pair_sources,
            donors,
            "test",
            gap_value,
            limit_pairs=args.limit_pairs,
        )
        annotate_surprise(
            host, raw, device, args.discovery_batch_size,
        )
        thresholds_row = graph_result["discovery"]["thresholds"][
            str(gap_value)
        ]
        threshold = (
            float(thresholds_row["surprise"]),
            float(thresholds_row["semantic_change"]),
        )
        tensorized = tensorize_gap(raw, threshold, device)
        actions_candidate, correct, candidate_sources = candidate_actions(
            raw,
            actions,
            pair_sources,
            args.seed + gap_value,
        )
        methods = {}
        arrays = {}
        no_loss = action_losses_for_sets(
            host,
            None,
            tensorized,
            actions_candidate,
            None,
            args.batch_size,
        )[:, 0]
        margin, accuracy, rank = margin_and_accuracy(no_loss, correct)
        methods["no_memory"] = {
            "accuracy": pair_bootstrap(
                accuracy, raw.pair_ids, args.seed + gap_value, args.bootstrap_repetitions,
            ),
            "correct_action_goal_error": pair_bootstrap(
                no_loss.cpu().numpy()[np.arange(len(raw.history)), correct],
                raw.pair_ids,
                args.seed + gap_value + 1,
                args.bootstrap_repetitions,
            ),
            "decision_margin": pair_bootstrap(
                margin, raw.pair_ids, args.seed + gap_value + 2, args.bootstrap_repetitions,
            ),
        }
        arrays["no_memory"] = (no_loss.cpu().numpy(), accuracy, rank)
        chosen_slots: dict[str, np.ndarray] = {}
        for method_index, method in enumerate(
            (
                "recent_only",
                "random_event",
                "surprise",
                "oracle_frame",
                "oracle_discovered_event",
                "all_history_upper",
            )
        ):
            sets = index_sets_for_method(
                tensorized,
                method,
                args.seed * 1009 + gap_value,
            )
            losses = action_losses_for_sets(
                host,
                memory,
                tensorized,
                actions_candidate,
                sets,
                args.batch_size,
            )
            set_margins = []
            for set_index in range(losses.shape[1]):
                set_margins.append(
                    margin_and_accuracy(losses[:, set_index], correct)[0]
                )
            set_margins_np = np.stack(set_margins, axis=1)
            if method.startswith("oracle_") or method == "all_history_upper":
                selected = np.argmax(set_margins_np, axis=1)
            else:
                selected = np.zeros(len(raw.history), dtype=np.int64)
            chosen_slots[method] = selected
            selected_loss = losses[
                torch.arange(len(raw.history), device=device),
                torch.from_numpy(selected).to(device),
            ]
            margin, accuracy, rank = margin_and_accuracy(
                selected_loss, correct,
            )
            methods[method] = {
                "accuracy": pair_bootstrap(
                    accuracy,
                    raw.pair_ids,
                    args.seed + gap_value + method_index * 11,
                    args.bootstrap_repetitions,
                ),
                "correct_action_goal_error": pair_bootstrap(
                    selected_loss.cpu().numpy()[
                        np.arange(len(raw.history)), correct
                    ],
                    raw.pair_ids,
                    args.seed + gap_value + method_index * 11 + 1,
                    args.bootstrap_repetitions,
                ),
                "decision_margin": pair_bootstrap(
                    margin,
                    raw.pair_ids,
                    args.seed + gap_value + method_index * 11 + 2,
                    args.bootstrap_repetitions,
                ),
            }
            arrays[method] = (
                selected_loss.cpu().numpy(),
                accuracy,
                rank,
            )
        recent_accuracy = arrays["recent_only"][1]
        for method, (_, accuracy, _) in arrays.items():
            methods[method]["accuracy_gain_vs_recent"] = pair_bootstrap(
                accuracy - recent_accuracy,
                raw.pair_ids,
                args.seed + gap_value + len(method) * 101,
                args.bootstrap_repetitions,
            )
        for row in range(len(raw.history)):
            decisions.append(
                {
                    "gap": gap_value,
                    "pair_id": int(raw.pair_ids[row]),
                    "branch_evaluator_only": int(raw.branches[row]),
                    "goal_query": "branch future DINO latent",
                    "action_candidate_source_episodes": candidate_sources[row],
                    "correct_candidate_index": int(correct[row]),
                    "recent_rank": int(arrays["recent_only"][2][row]),
                    "oracle_frame_rank": int(arrays["oracle_frame"][2][row]),
                    "oracle_event_rank": int(
                        arrays["oracle_discovered_event"][2][row]
                    ),
                    "all_history_rank": int(
                        arrays["all_history_upper"][2][row]
                    ),
                    "du_hat": None,
                    "du_true": None,
                    "abstained": True,
                    "outcome": (
                        "correct"
                        if arrays["oracle_discovered_event"][2][row] == 1
                        else "incorrect"
                    ),
                }
            )
        suffix_max = 0.0
        suffix_action_max = 0.0
        for pair in np.unique(raw.pair_ids):
            rows = np.flatnonzero(raw.pair_ids == pair)
            if len(rows) == 2:
                suffix_max = max(
                    suffix_max,
                    float(
                        np.max(
                            np.abs(
                                raw.history[rows[0], -6:]
                                - raw.history[rows[1], -6:]
                            )
                        )
                    ),
                )
                suffix_action_max = max(
                    suffix_action_max,
                    float(
                        np.max(
                            np.abs(
                                raw.history_actions[rows[0], -6:]
                                - raw.history_actions[rows[1], -6:]
                            )
                        )
                    ),
                )
        gap_results.append(
            {
                "gap": gap_value,
                "pair_count": int(len(np.unique(raw.pair_ids))),
                "example_count": len(raw.history),
                "methods": methods,
                "suffix_max_latent_difference": suffix_max,
                "suffix_max_action_difference": suffix_action_max,
            }
        )
        for row in range(len(raw.history)):
            item = {
                "gap": gap_value,
                "pair_id": int(raw.pair_ids[row]),
                "branch": int(raw.branches[row]),
                "correct_candidate": int(correct[row]),
            }
            for method, (loss, accuracy, rank) in arrays.items():
                item[f"accuracy_{method}"] = float(accuracy[row])
                item[f"rank_{method}"] = int(rank[row])
                item[f"correct_loss_{method}"] = float(
                    loss[row, correct[row]]
                )
            evaluation_rows.append(item)
    if tensor_digest(host) != host_digest or tensor_digest(memory) != memory_digest:
        raise RuntimeError("frozen host/memory changed")
    np.savez_compressed(
        output_dir / "evaluation.npz",
        **{
            key: np.asarray([row[key] for row in evaluation_rows])
            for key in evaluation_rows[0]
        },
    )
    (output_dir / "decision_log.json").write_text(
        stable_json(json_safe({"queries": decisions}))
    )
    result = {
        "schema": "cem_decision_memory_cell_v1",
        "status": "completed",
        "environment": ENV,
        "seed": args.seed,
        "device": str(device),
        "protocol": {
            "gaps": list(GAPS),
            "candidate_action_count": CANDIDATES,
            "decision_loss": (
                "goal-conditioned future latent rollout error over candidate actions"
            ),
            "matched_memory_tokens": MEMORY_TOKENS,
            "matched_host_calls": True,
            "raw_frames_modified": False,
            "native_chronology": False,
            "controlled_splicing": True,
            "planning_claim": False,
        },
        "host": host_receipt,
        "host_digest": host_digest,
        "memory_digest": memory_digest,
        "gap_results": gap_results,
        "source_contract": source_audit(),
        "executed_use": {
            "reached": False,
            "reason": (
                "controlled branch-future goal is not an admitted standard "
                "PointMaze task goal for the fixed controller"
            ),
        },
        "elapsed_seconds": float(time.time() - started),
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "evaluation": str(
                (output_dir / "evaluation.npz").relative_to(ROOT)
            ),
            "decision_log": str(
                (output_dir / "decision_log.json").relative_to(ROOT)
            ),
        },
    }
    result_path.write_text(stable_json(json_safe(result)))
    print(
        stable_json(
            {
                "seed": args.seed,
                "gap_accuracy": {
                    str(row["gap"]): {
                        method: values["accuracy"]["mean"]
                        for method, values in row["methods"].items()
                    }
                    for row in gap_results
                },
                "result": str(result_path.relative_to(ROOT)),
            }
        ),
        flush=True,
    )
    return result


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(101_003)
    draws = np.asarray(
        [
            rng.choice(array, len(array), replace=True).mean()
            for _ in range(4000)
        ]
    )
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).astype(float).tolist(),
        "values": array.astype(float).tolist(),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/result.json")):
        value = json.loads(path.read_text())
        if value.get("schema") == "cem_decision_memory_cell_v1":
            cells.append(value)
    if not cells:
        raise RuntimeError("no decision-memory cells")
    gap_rows = []
    methods = (
        "no_memory",
        "recent_only",
        "random_event",
        "surprise",
        "oracle_frame",
        "oracle_discovered_event",
        "all_history_upper",
    )
    global_values: dict[str, list[float]] = {method: [] for method in methods}
    for gap_value in GAPS:
        per_method = {}
        source_rows = [
            next(
                row for row in cell["gap_results"]
                if row["gap"] == gap_value
            )
            for cell in cells
        ]
        for method in methods:
            values = [
                row["methods"][method]["accuracy"]["mean"]
                for row in source_rows
            ]
            per_method[method] = {
                "accuracy": summary(values),
                "goal_error": summary(
                    [
                        row["methods"][method][
                            "correct_action_goal_error"
                        ]["mean"]
                        for row in source_rows
                    ]
                ),
            }
            global_values[method].extend(values)
        recent = np.asarray(
            per_method["recent_only"]["accuracy"]["values"]
        )
        comparisons = {}
        for method in (
            "oracle_frame",
            "oracle_discovered_event",
            "all_history_upper",
            "surprise",
            "random_event",
        ):
            value = np.asarray(per_method[method]["accuracy"]["values"])
            comparisons[f"{method}_vs_recent"] = summary(
                (value - recent).tolist()
            )
        denominator = (
            comparisons["all_history_upper_vs_recent"]["mean"]
        )
        closure = (
            comparisons["oracle_discovered_event_vs_recent"]["mean"]
            / max(denominator, 1e-12)
        )
        gap_rows.append(
            {
                "gap": gap_value,
                "methods": per_method,
                "comparisons": comparisons,
                "oracle_event_closure": closure,
                "suffix_exact": all(
                    row["suffix_max_latent_difference"] == 0.0
                    and row["suffix_max_action_difference"] == 0.0
                    for row in source_rows
                ),
            }
        )
    aggregate_methods = {
        method: summary(global_values[method]) for method in methods
    }
    recent = np.asarray(aggregate_methods["recent_only"]["values"])
    oracle_event = np.asarray(
        aggregate_methods["oracle_discovered_event"]["values"]
    )
    oracle_frame = np.asarray(
        aggregate_methods["oracle_frame"]["values"]
    )
    all_history = np.asarray(
        aggregate_methods["all_history_upper"]["values"]
    )
    event_gain = summary((oracle_event - recent).tolist())
    frame_gain = summary((oracle_frame - recent).tolist())
    all_gain = summary((all_history - recent).tolist())
    closure = event_gain["mean"] / max(all_gain["mean"], 1e-12)
    resolved = (
        (
            event_gain["ci95"][0] > 0.0
            and event_gain["mean"] >= 0.05
        )
        or (
            frame_gain["ci95"][0] > 0.0
            and frame_gain["mean"] >= 0.05
        )
        or (
            all_gain["ci95"][0] > 0.0 and closure >= 0.25
        )
    )
    gate1 = {
        "passed": bool(resolved),
        "oracle_event_accuracy_gain": event_gain,
        "oracle_frame_accuracy_gain": frame_gain,
        "all_history_accuracy_gain": all_gain,
        "oracle_event_closure": closure,
        "rule": (
            "resolved >=5pp oracle frame/event action-ranking gain OR "
            ">=25% closure of resolved all-history gap"
        ),
    }
    report = {
        "schema": "cem_decision_memory_report_v1",
        "status": "completed",
        "phase_reached": 1,
        "cell_count": len(cells),
        "environment": ENV,
        "gaps": gap_rows,
        "aggregate_action_accuracy": aggregate_methods,
        "gate1": gate1,
        "gate2": {
            "reached": bool(gate1["passed"]),
            "status": (
                "decision head required"
                if gate1["passed"]
                else "hard-stopped by oracle gate"
            ),
        },
        "executed_use": {
            "reached": False,
            "controller_available_for_standard_goals": True,
            "controller_valid_for_controlled_branch_goal": False,
        },
        "breadth": {
            "reached": False,
            "reason": "Gate 2 not passed",
        },
        "source_contract": source_audit(),
        "jobs_still_running": [],
        "artifacts": {
            "build": "outputs/cem_decision_memory_v1/build",
            "cells": "outputs/cem_decision_memory_v1/cells",
            "report": "outputs/cem_decision_memory_v1/report.json",
            "machine_report": "outputs/cem_decision_memory_report.json",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        stable_json(json_safe(report))
    )
    (ROOT / "outputs/cem_decision_memory_report.json").write_text(
        stable_json(json_safe(report))
    )
    print(stable_json(json_safe(report)), flush=True)
    return report


def aggregate_phase2(args: argparse.Namespace) -> dict[str, Any]:
    phase1 = json.loads((args.output / "report.json").read_text())
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/decision_result.json")):
        value = json.loads(path.read_text())
        if value.get("schema") == "cem_decision_head_cell_v1":
            cells.append(value)
    if not cells:
        raise RuntimeError("no decision-head cells")
    learned_gain = summary(
        [
            cell["test"]["conditions"]["learned_lcb"]["gain_vs_recent"]["mean"]
            for cell in cells
        ]
    )
    du_spearman = summary(
        [cell["test"]["du"]["spearman"] or 0.0 for cell in cells]
    )
    du_pairwise = summary(
        [cell["test"]["du"]["pairwise_accuracy"] or 0.0 for cell in cells]
    )
    deletion_gap = summary(
        [cell["test"]["du"]["high_minus_random"] for cell in cells]
    )
    coverage = summary(
        [cell["test"]["activation"]["coverage"] for cell in cells]
    )
    precision = summary(
        [
            cell["test"]["activation"]["precision_positive"] or 0.0
            for cell in cells
        ]
    )
    safety = summary([cell["ordinary_degradation"] for cell in cells])
    gate2_pass = bool(
        learned_gain["ci95"][0] > 0.0
        and (
            du_spearman["mean"] > 0.2
            or du_pairwise["mean"] >= 0.65
        )
        and safety["mean"] <= 0.05
    )
    phase1.update(
        {
            "phase_reached": 3,
            "gate2": {
                "passed": gate2_pass,
                "learned_accuracy_gain": learned_gain,
                "du_spearman": du_spearman,
                "du_pairwise_accuracy": du_pairwise,
                "high_minus_random_deletion": deletion_gap,
                "activation_coverage": coverage,
                "activation_precision": precision,
                "ordinary_degradation": safety,
                "rule": (
                    "learned gain lower CI >0; DU Spearman>0.2 or "
                    "pairwise>=0.65; degradation<=5%"
                ),
            },
            "executed_use": {
                **phase1["executed_use"],
                "reached": False,
                "reason": (
                    "controlled branch goal is not an admitted fixed-controller task"
                    if gate2_pass
                    else "Gate 2 failed"
                ),
            },
            "breadth": {
                "reached": False,
                "reason": (
                    "executed PointMaze use unavailable for controlled branch task"
                    if gate2_pass
                    else "Gate 2 failed"
                ),
            },
        }
    )
    (args.output / "report.json").write_text(
        stable_json(json_safe(phase1))
    )
    (ROOT / "outputs/cem_decision_memory_report.json").write_text(
        stable_json(json_safe(phase1))
    )
    print(stable_json(json_safe(phase1)), flush=True)
    return phase1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--aggregate-phase2", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--discovery-batch-size", type=int, default=4096)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--decision-epochs", type=int, default=20)
    parser.add_argument("--router-epochs", type=int, default=20)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    if args.gpu == 3 or args.gpu not in (0, 1, 2):
        parser.error("GPU3 prohibited; use GPU 0/1/2")
    if args.smoke:
        args.limit_pairs = args.limit_pairs or 8
        args.bootstrap_repetitions = min(
            args.bootstrap_repetitions, 200
        )
        args.decision_epochs = min(args.decision_epochs, 4)
        args.router_epochs = min(args.router_epochs, 4)
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate_phase2:
        aggregate_phase2(args)
    elif args.aggregate:
        aggregate(args)
    elif args.phase2:
        run_phase2(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
