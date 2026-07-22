#!/usr/bin/env python3
"""CEM architecture B: mixture-of-memory experts on frozen PushT LeWM.

Six disjoint three-frame causal events remain separate through retrieval and
expert proposal.  A context/event/causal-effect router sparsely fuses the
bounded retrieved proposals only at the frozen host interface.  Training is
label-free: strict host future-latent prediction plus same-base six-way
counterfactual separation.  Labels are read only by the final linear audit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm_pusht import load_official_pusht_checkpoint  # noqa: E402
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    resolve_pusht_path,
    validate_pusht_device,
)
from scripts.run_lewm_pusht_host_writer import (  # noqa: E402
    DEFAULT_COUNTERFACTUAL_CACHE,
    age_adjusted_spec,
    load_admitted,
    load_or_build_counterfactual_cache,
    state_digest,
)
from scripts.run_mem_jepa_stage_b import fit_classifier  # noqa: E402

LATENT_DIM = 192
ACTION_DIM = 10
EVENT_COUNT = 6
EVENT_WIDTH = 3
DEFAULT_OUTPUT = ROOT / "outputs/cem_lewm_memory_experts_v1"
DEFAULT_ASSET_PREFIX = ROOT / "docs/assets/cem_lewm_memory_experts"
BASELINE_ROOT = ROOT / "outputs/cem_lewm_v3/D/multi-item-visual-binding-recall"


def tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)


def host_predict(host: nn.Module, z: torch.Tensor,
                 actions: torch.Tensor) -> torch.Tensor:
    if z.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return host.predict(z, actions).float()
    return host.predict(z, actions).float()


def balanced_accuracy(prediction: np.ndarray, truth: np.ndarray,
                      classes: int) -> float:
    return float(np.mean([
        np.mean(prediction[truth == value] == value)
        for value in range(classes)
    ]))


def contrastive(query: torch.Tensor, candidates: torch.Tensor,
                temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.einsum(
        "bd,bkd->bk", F.normalize(query, dim=-1),
        F.normalize(candidates, dim=-1)) / temperature
    target = torch.zeros(len(query), dtype=torch.long, device=query.device)
    loss = F.cross_entropy(logits, target)
    positive = 1.0 - (
        F.normalize(query, dim=-1) * F.normalize(candidates[:, 0], dim=-1)
    ).sum(-1).mean()
    accuracy = (logits.argmax(-1) == 0).float().mean()
    return loss + positive, accuracy


def counterfactual_candidates(data: dict[str, Any], rows: np.ndarray,
                              device: torch.device
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive-first six-way absolute and delta cue candidates.

    The positive branch is discovered by latent distance to the observed cue;
    semantic labels are neither read nor used.
    """
    z_all = np.asarray(data["z_counterfactual"][rows], dtype=np.float32)
    z_cue = np.asarray(data["z_cue"][rows], dtype=np.float32)
    cue_start = 1
    base = np.asarray(
        data["z_base"][rows, cue_start:cue_start + EVENT_WIDTH],
        dtype=np.float32)
    positive = np.square(z_all - z_cue[:, None]).mean(axis=(2, 3)).argmin(1)
    order = np.asarray([
        [int(value)] + [j for j in range(EVENT_COUNT) if j != int(value)]
        for value in positive
    ])
    selected = z_all[np.arange(len(rows))[:, None], order]
    absolute = selected.mean(axis=2)
    delta = (selected - base[:, None]).mean(axis=2)
    return tensor(absolute, device), tensor(delta, device)


class MemoryExpertMixture(nn.Module):
    """One residual expert per causal event with sparse context-aware routing."""

    def __init__(self, dim: int = 128, retrieve_k: int = 4,
                 route_k: int = 2, residual_scale: float = 1.0) -> None:
        super().__init__()
        if not (1 <= route_k <= retrieve_k <= EVENT_COUNT):
            raise ValueError("require 1 <= route_k <= retrieve_k <= 6")
        self.dim = int(dim)
        self.retrieve_k = int(retrieve_k)
        self.route_k = int(route_k)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        input_dim = EVENT_WIDTH * LATENT_DIM + LATENT_DIM + ACTION_DIM + 1
        self.event_norm = nn.LayerNorm(input_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, dim), nn.GELU(),
                nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, EVENT_WIDTH * LATENT_DIM),
            )
            for _ in range(EVENT_COUNT)
        ])
        self.router = nn.Sequential(
            nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.route_bias = nn.Parameter(torch.zeros(EVENT_COUNT))
        # Small nonzero proposals permit gradients through the frozen host.
        for expert in self.experts:
            nn.init.normal_(expert[-1].weight, std=1e-3)
            nn.init.zeros_(expert[-1].bias)

    @staticmethod
    def event_indices(device: torch.device) -> torch.Tensor:
        return torch.arange(1, 19, device=device).reshape(
            EVENT_COUNT, EVENT_WIDTH)

    def _events(self, z: torch.Tensor, actions: torch.Tensor,
                surprise: torch.Tensor, reset: bool
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        indices = self.event_indices(z.device)
        event_z = z[:, indices].reshape(
            batch, EVENT_COUNT, EVENT_WIDTH * LATENT_DIM)
        event_a = actions[:, indices].mean(2)
        ce = surprise[:, indices].mean(2)
        if reset:
            # Reset memory contains only the legal final context event.
            event_z = torch.zeros_like(event_z)
            event_a = torch.zeros_like(event_a)
            ce = torch.full_like(ce, -1e4)
            event_z[:, -1] = z[:, indices[-1]].reshape(batch, -1)
            event_a[:, -1] = actions[:, indices[-1]].mean(1)
            ce[:, -1] = surprise[:, indices[-1]].mean(1)
        ce_scaled = (ce - ce.mean(1, keepdim=True)) / (
            ce.std(1, keepdim=True).clamp_min(1e-5))
        return event_z, event_a, ce_scaled

    def forward(self, z: torch.Tensor, actions: torch.Tensor,
                context_indices: torch.Tensor, surprise: torch.Tensor, *,
                reset: bool = False, delete_expert: int | None = None
                ) -> dict[str, torch.Tensor]:
        context_z = z[:, context_indices]
        context_a = actions[:, context_indices]
        event_z, event_a, ce = self._events(z, actions, surprise, reset)
        query = context_z.mean(1)[:, None].expand(-1, EVENT_COUNT, -1)
        inputs = self.event_norm(torch.cat(
            (event_z, query, event_a, ce[..., None]), dim=-1))
        proposals = torch.stack([
            expert(inputs[:, index]).reshape(-1, EVENT_WIDTH, LATENT_DIM)
            for index, expert in enumerate(self.experts)
        ], dim=1)

        # Bounded CE retrieval, followed by learned sparse top-k routing.
        retrieve = torch.zeros_like(ce, dtype=torch.bool)
        retrieve.scatter_(1, ce.topk(self.retrieve_k, dim=1).indices, True)
        logits = self.router(inputs).squeeze(-1) + self.route_bias
        logits = logits.masked_fill(~retrieve, -1e4)
        if delete_expert is not None:
            logits[:, int(delete_expert)] = -1e4
            proposals = proposals.clone()
            proposals[:, int(delete_expert)] = 0
        route = torch.zeros_like(retrieve)
        route.scatter_(1, logits.topk(self.route_k, dim=1).indices, True)
        sparse_logits = logits.masked_fill(~route, -1e4)
        weights = torch.softmax(sparse_logits, dim=1)
        mixture = torch.einsum("be,beth->bth", weights, proposals)
        fused = context_z + self.residual_scale * mixture
        return {
            "fused": fused,
            "mixture": mixture,
            "proposals": proposals,
            "weights": weights,
            "retrieved": retrieve,
            "ce": ce,
            "event_features": event_z,
        }


@torch.no_grad()
def frame_surprise(host: nn.Module, z: torch.Tensor,
                   actions: torch.Tensor, decision: int) -> torch.Tensor:
    result = torch.zeros(z.shape[0], decision, device=z.device)
    for frame in range(1, decision):
        low = max(0, frame - 3)
        prediction = host_predict(
            host, z[:, low:frame], actions[:, low:frame])[:, -1]
        result[:, frame] = torch.square(prediction - z[:, frame]).mean(-1)
    return result


def load_data(args: argparse.Namespace, host: nn.Module,
              device: torch.device) -> tuple[dict, dict, dict]:
    locked = load_locked_pusht_spec(str(DEFAULT_PUSHT_SPEC),
                                    str(DEFAULT_PUSHT_LOCK))
    spec = age_adjusted_spec(locked, args.age)
    train = load_admitted(locked, args.task, "train")
    validation = load_admitted(locked, args.task, "validation")
    cache = Path(args.counterfactual_cache)
    load_or_build_counterfactual_cache(
        train, spec, args.task, "train", host, device, cache, 128)
    load_or_build_counterfactual_cache(
        validation, spec, args.task, "validation", host, device, cache, 128)
    return spec, train, validation


def batch(data: dict[str, Any], rows: np.ndarray,
          device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return tensor(data["z"][rows], device), tensor(data["actions"][rows], device)


def train_seed(args: argparse.Namespace, seed: int, host: nn.Module,
               spec: dict, train: dict, validation: dict,
               device: torch.device, host_digest: str) -> dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sequence = spec["sequence"]
    decision = int(sequence["decision_index"])
    context_np = np.asarray(sequence["final_context_indices"], dtype=np.int64)
    context = torch.as_tensor(context_np, device=device)
    target_idx = torch.as_tensor(context_np + 1, device=device)
    model = MemoryExpertMixture(
        dim=args.dim, retrieve_k=args.retrieve_k, route_k=args.route_k,
        residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    rows_all = np.arange(len(train["z"]))
    rng = np.random.default_rng(730_000 + seed)
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(rows_all)
        accum: dict[str, list[float]] = {
            key: [] for key in ("loss", "future", "expert_sep", "host_sep",
                                "balance", "entropy", "expert_match",
                                "host_match")
        }
        model.train()
        for offset in range(0, len(rows_all), args.batch_size):
            rows = rows_all[offset:offset + args.batch_size]
            if len(rows) < 4:
                continue
            z, actions = batch(train, rows, device)
            surprise = frame_surprise(host, z, actions, decision)
            result = model(z, actions, context, surprise)
            predicted = host_predict(host, result["fused"],
                                     actions[:, context_np])
            absolute, delta = counterfactual_candidates(train, rows, device)
            expert_sep, expert_match = contrastive(
                result["mixture"].mean(1), delta, args.temperature)
            host_sep, host_match = contrastive(
                predicted[:, -1], absolute, args.temperature)
            future = F.mse_loss(predicted, z[:, target_idx])

            weights = result["weights"]
            load = weights.mean(0)
            balance = EVENT_COUNT * torch.square(
                load - (1.0 / EVENT_COUNT)).sum()
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(1).mean()
            residual = torch.square(result["mixture"]).mean()
            loss = (
                args.future_weight * future
                + args.expert_sep_weight * expert_sep
                + args.host_sep_weight * host_sep
                + args.balance_weight * balance
                + args.sparsity_weight * entropy
                + args.residual_weight * residual
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = {
                "loss": loss, "future": future, "expert_sep": expert_sep,
                "host_sep": host_sep, "balance": balance, "entropy": entropy,
                "expert_match": expert_match, "host_match": host_match,
            }
            for key, value in values.items():
                accum[key].append(float(value.detach()))
        scheduler.step()
        record = {"epoch": epoch}
        record.update({key: float(np.mean(value))
                       for key, value in accum.items()})
        history.append(record)
        print(
            f"[memory-experts] s{seed} epoch={epoch}/{args.epochs} "
            f"loss={record['loss']:.4f} expert={record['expert_match']:.3f} "
            f"host={record['host_match']:.3f} H={record['entropy']:.3f}",
            flush=True)

    if state_digest(host) != host_digest:
        raise RuntimeError("frozen official PushT LeWM digest changed")
    metrics = evaluate(
        args, seed, model, host, spec, train, validation, device)
    result = {
        "schema": "cem_lewm_memory_experts_seed_v1",
        "seed": seed,
        "age": args.age,
        "task": args.task,
        "labels_used_for_training_loss": False,
        "training_supervision": [
            "frozen_host_future_latent_mse",
            "six_way_same_base_counterfactual_expert_separability",
            "six_way_same_base_counterfactual_frozen_host_separability",
        ],
        "trainable_components": ["event_experts", "gate_router"],
        "host_digest": host_digest,
        "host_digest_unchanged": True,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": {
            "events": EVENT_COUNT, "event_width": EVENT_WIDTH,
            "retrieve_k": args.retrieve_k, "route_k": args.route_k,
            "dim": args.dim,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "history": history,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    seed_dir = Path(args.output) / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "result.json").write_text(json.dumps(result, indent=2))
    torch.save({
        "model_state_dict": model.state_dict(),
        "model": result["model"], "host_digest": host_digest,
    }, seed_dir / "memory_experts.pt")
    return result


@torch.no_grad()
def collect(model: MemoryExpertMixture, host: nn.Module, data: dict[str, Any],
            spec: dict, device: torch.device, condition: str,
            seed: int) -> dict[str, np.ndarray]:
    model.eval()
    context_np = np.asarray(spec["sequence"]["final_context_indices"],
                            dtype=np.int64)
    context = torch.as_tensor(context_np, device=device)
    decision = int(spec["sequence"]["decision_index"])
    packs: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "memory_only", "expert_mixture", "host_output", "weights",
            "retrieved", "ce")}
    rng = np.random.default_rng(810_000 + seed)
    permutation = rng.permutation(len(data["z"]))
    for offset in range(0, len(data["z"]), 128):
        rows = np.arange(offset, min(len(data["z"]), offset + 128))
        z, actions = batch(data, rows, device)
        surprise = frame_surprise(host, z, actions, decision)
        if condition in {"host_only", "no_state"}:
            prediction = host_predict(
                host, z[:, context_np], actions[:, context_np])[:, -1]
            zeros = torch.zeros(
                len(rows), EVENT_COUNT, device=device)
            result = {
                "event_features": torch.zeros(
                    len(rows), EVENT_COUNT, EVENT_WIDTH * LATENT_DIM,
                    device=device),
                "mixture": torch.zeros(
                    len(rows), EVENT_WIDTH, LATENT_DIM, device=device),
                "weights": zeros, "retrieved": zeros.bool(), "ce": zeros,
            }
        elif condition == "shuffled":
            source = permutation[rows]
            source_z, source_actions = batch(data, source, device)
            source_surprise = frame_surprise(
                host, source_z, source_actions, decision)
            result = model(
                source_z, source_actions, context, source_surprise)
            fused = z[:, context_np] + model.residual_scale * result["mixture"]
            prediction = host_predict(
                host, fused, actions[:, context_np])[:, -1]
        else:
            result = model(
                z, actions, context, surprise, reset=condition == "reset")
            fused = result["fused"]
            if condition == "random":
                generator = torch.Generator(device=device).manual_seed(
                    910_000 + seed + offset)
                noise = torch.randn(
                    result["mixture"].shape, generator=generator, device=device)
                norm = result["mixture"].norm(dim=-1, keepdim=True)
                noise = noise * (norm / noise.norm(
                    dim=-1, keepdim=True).clamp_min(1e-8))
                fused = z[:, context_np] + model.residual_scale * noise
            prediction = host_predict(
                host, fused, actions[:, context_np])[:, -1]
        packs["memory_only"].append(
            result["event_features"].reshape(len(rows), -1).cpu().numpy())
        packs["expert_mixture"].append(
            result["mixture"].reshape(len(rows), -1).cpu().numpy())
        packs["host_output"].append(prediction.cpu().numpy())
        for key in ("weights", "retrieved", "ce"):
            packs[key].append(result[key].float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in packs.items()}


@torch.no_grad()
def evaluate(args: argparse.Namespace, seed: int, model: MemoryExpertMixture,
             host: nn.Module, spec: dict, train: dict, validation: dict,
             device: torch.device) -> dict[str, Any]:
    labels_train = np.asarray(train["labels"])
    labels_validation = np.asarray(validation["labels"])
    classes = len(np.unique(labels_train))
    train_full = collect(
        model, host, train, spec, device, "full", seed)
    validation_packs = {
        condition: collect(
            model, host, validation, spec, device, condition, seed)
        for condition in (
            "full", "host_only", "reset", "no_state", "shuffled", "random")
    }
    ladder = {}
    for level in ("memory_only", "expert_mixture", "host_output"):
        prediction = fit_classifier(
            train_full[level], labels_train,
            validation_packs["full"][level])
        ladder[level] = balanced_accuracy(
            prediction, labels_validation, classes)
    controls = {}
    for condition, pack in validation_packs.items():
        prediction = fit_classifier(
            train_full["host_output"], labels_train, pack["host_output"])
        controls[condition] = balanced_accuracy(
            prediction, labels_validation, classes)

    full = validation_packs["full"]
    weights = full["weights"]
    load = weights.mean(0)
    selected = (weights > 0).mean(0)
    entropy = -np.sum(
        weights * np.log(np.maximum(weights, 1e-12)), axis=1)
    retrieved = full["retrieved"].mean(0)
    cue_expert = 0
    routing = {
        "mean_entropy": float(entropy.mean()),
        "normalized_entropy": float(
            entropy.mean() / math.log(args.route_k)
            if args.route_k > 1 else 0.0),
        "mean_load": load.tolist(),
        "selection_frequency": selected.tolist(),
        "retrieval_frequency": retrieved.tolist(),
        "cue_event_retrieval_frequency": float(retrieved[cue_expert]),
        "cue_event_selection_frequency": float(selected[cue_expert]),
        "event_frames": [
            list(range(1 + EVENT_WIDTH * index,
                       1 + EVENT_WIDTH * (index + 1)))
            for index in range(EVENT_COUNT)
        ],
    }

    # Strict host future-latent deletion effect for each event expert.
    context_np = np.asarray(spec["sequence"]["final_context_indices"],
                            dtype=np.int64)
    context = torch.as_tensor(context_np, device=device)
    target_idx = torch.as_tensor(context_np + 1, device=device)
    deletion_values: list[list[float]] = [[] for _ in range(EVENT_COUNT)]
    future_full: list[float] = []
    future_no_memory: list[float] = []
    future_reset: list[float] = []
    for offset in range(0, len(validation["z"]), 128):
        rows = np.arange(offset, min(len(validation["z"]), offset + 128))
        z, actions = batch(validation, rows, device)
        surprise = frame_surprise(
            host, z, actions, int(spec["sequence"]["decision_index"]))
        base = model(z, actions, context, surprise)
        base_loss = torch.square(
            host_predict(host, base["fused"], actions[:, context_np])
            - z[:, target_idx]).mean((1, 2))
        no_memory_loss = torch.square(
            host_predict(host, z[:, context_np], actions[:, context_np])
            - z[:, target_idx]).mean((1, 2))
        reset_result = model(z, actions, context, surprise, reset=True)
        reset_loss = torch.square(
            host_predict(
                host, reset_result["fused"], actions[:, context_np])
            - z[:, target_idx]).mean((1, 2))
        future_full.extend(base_loss.cpu().tolist())
        future_no_memory.extend(no_memory_loss.cpu().tolist())
        future_reset.extend(reset_loss.cpu().tolist())
        for expert in range(EVENT_COUNT):
            deleted = model(
                z, actions, context, surprise, delete_expert=expert)
            deleted_loss = torch.square(
                host_predict(host, deleted["fused"], actions[:, context_np])
                - z[:, target_idx]).mean((1, 2))
            deletion_values[expert].extend(
                (deleted_loss - base_loss).cpu().tolist())
    deletion = {
        "delta_future_loss_by_expert": [
            float(np.mean(value)) for value in deletion_values],
        "cue_expert_hurts_more_than_noncue_mean": bool(
            np.mean(deletion_values[0])
            > np.mean([np.mean(value) for value in deletion_values[1:]])),
    }
    control_max = 0.217
    gate = {
        "host_output_minimum": 0.75,
        "control_maximum": control_max,
        "host_output_passed": ladder["host_output"] >= 0.75,
        "controls_passed": all(
            controls[key] <= control_max
            for key in ("host_only", "reset", "no_state", "shuffled", "random")),
    }
    gate["passed"] = bool(
        gate["host_output_passed"] and gate["controls_passed"])
    return {
        "host_future_latent_loss": {
            "with_memory": float(np.mean(future_full)),
            "without_memory": float(np.mean(future_no_memory)),
            "reset_memory": float(np.mean(future_reset)),
            "improvement_vs_without_memory": float(
                np.mean(future_no_memory) - np.mean(future_full)),
            "cue_specific_improvement_vs_reset": float(
                np.mean(future_reset) - np.mean(future_full)),
        },
        "diagnostic_ladder": ladder,
        "controls": controls,
        "routing": routing,
        "causal_deletion": deletion,
        "success_gate": gate,
    }


def aggregate(args: argparse.Namespace, results: list[dict[str, Any]],
              host_digest: str) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        output = []
        for result in results:
            item: Any = result
            for key in path:
                item = item[key]
            output.append(float(item))
        return output

    ladder = {}
    for level in ("memory_only", "expert_mixture", "host_output"):
        current = values(("metrics", "diagnostic_ladder", level))
        ladder[level] = {
            "mean": float(np.mean(current)), "std": float(np.std(current)),
            "seed_values": current,
        }
    controls = {}
    for condition in (
            "full", "host_only", "reset", "no_state", "shuffled", "random"):
        current = values(("metrics", "controls", condition))
        controls[condition] = {
            "mean": float(np.mean(current)), "std": float(np.std(current)),
            "seed_values": current,
        }
    baseline_values = []
    for seed in args.seeds:
        path = BASELINE_ROOT / f"s{seed}" / "summary.json"
        if path.is_file():
            baseline_values.append(json.loads(path.read_text())[
                "diagnostic_ladder"]["host_output"])
    routing = {
        key: {
            "mean": float(np.mean(values(("metrics", "routing", key)))),
            "seed_values": values(("metrics", "routing", key)),
        }
        for key in (
            "mean_entropy", "normalized_entropy",
            "cue_event_retrieval_frequency", "cue_event_selection_frequency")
    }
    routing["mean_load_by_expert"] = np.mean([
        result["metrics"]["routing"]["mean_load"] for result in results
    ], axis=0).tolist()
    routing["selection_frequency_by_expert"] = np.mean([
        result["metrics"]["routing"]["selection_frequency"]
        for result in results
    ], axis=0).tolist()
    deletion = np.mean([
        result["metrics"]["causal_deletion"]["delta_future_loss_by_expert"]
        for result in results
    ], axis=0)
    control_pass = all(
        controls[key]["mean"] <= 0.217
        for key in ("host_only", "reset", "no_state", "shuffled", "random"))
    passed = ladder["host_output"]["mean"] >= 0.75 and control_pass
    geometry = (
        "preserved_at_host_output"
        if ladder["host_output"]["mean"] >= 0.75
        else "not_preserved_at_host_output")
    failures = []
    if ladder["host_output"]["mean"] < 0.75:
        failures.append(
            "Host-output six-way balanced accuracy did not meet 0.75.")
    if ladder["expert_mixture"]["mean"] < 0.75:
        failures.append(
            "Expert-mixture six-way geometry was not robust across seeds.")
    if routing["mean_load_by_expert"][0] > 0.50:
        failures.append(
            "Router load remained cue-expert dominated despite balancing.")
    if not bool(deletion[0] > deletion[1:].mean()):
        failures.append(
            "Deleting the cue expert improved strict future-latent loss, "
            "showing a separability/prediction objective conflict.")
    report = {
        "schema": "cem_lewm_memory_experts_report_v1",
        "status": "completed" if passed else "completed_with_failed_gate",
        "architecture": "B_MIXTURE_OF_MEMORY_EXPERTS",
        "task": args.task,
        "age": args.age,
        "seeds": args.seeds,
        "device": str(args.device),
        "gpu_constraint": "cuda:2 only; GPU3 never used",
        "host_digest": host_digest,
        "host_digest_unchanged_all_seeds": all(
            result["host_digest_unchanged"] for result in results),
        "labels_used_for_training_loss": False,
        "trainable_components": ["event_experts", "gate_router"],
        "retrieval": {
            "event_count": EVENT_COUNT, "event_width": EVENT_WIDTH,
            "bounded_top_k": args.retrieve_k,
            "sparse_route_top_k": args.route_k,
            "identity_preserved_until_final_fusion": True,
        },
        "diagnostic_ladder": ladder,
        "host_future_latent_loss": {
            key: {
                "mean": float(np.mean(values(
                    ("metrics", "host_future_latent_loss", key)))),
                "seed_values": values(
                    ("metrics", "host_future_latent_loss", key)),
            }
            for key in (
                "with_memory", "without_memory", "reset_memory",
                "improvement_vs_without_memory",
                "cue_specific_improvement_vs_reset")
        },
        "controls": controls,
        "v3_dense_residual_baseline_host_output": {
            "mean": float(np.mean(baseline_values)),
            "std": float(np.std(baseline_values)),
            "seed_values": baseline_values,
        },
        "routing": routing,
        "causal_deletion": {
            "mean_delta_future_loss_by_expert": deletion.tolist(),
            "cue_expert_hurts_more_than_noncue_mean": bool(
                deletion[0] > deletion[1:].mean()),
        },
        "success_gate": {
            "host_output_minimum": 0.75,
            "controls_maximum": 0.217,
            "host_output_passed": ladder["host_output"]["mean"] >= 0.75,
            "controls_passed": control_pass,
            "passed": passed,
        },
        "six_way_geometry": geometry,
        "failures": failures,
        "per_seed": results,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_figure(report, Path(args.asset_prefix))
    write_markdown(report, ROOT / "docs/CEM_LEWM_MEMORY_EXPERTS_REPORT.md")
    return report


def write_figure(report: dict[str, Any], prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    levels = ("memory_only", "expert_mixture", "host_output")
    means = [report["diagnostic_ladder"][key]["mean"] for key in levels]
    stds = [report["diagnostic_ladder"][key]["std"] for key in levels]
    axes[0].bar(
        range(3), means, yerr=stds, capsize=4,
        color=("#42a5f5", "#ff9800", "#5e35b1"),
        edgecolor="black", linewidth=0.5)
    axes[0].axhline(1 / 6, color="black", linestyle=":", label="chance")
    axes[0].axhline(0.75, color="#c62828", linestyle="--", label="gate")
    axes[0].set_xticks(
        range(3), ("Memory only", "Expert mixture", "Frozen host"))
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Six-way balanced accuracy")
    axes[0].set_title("Exposure ladder")
    axes[0].legend(frameon=False)
    load = report["routing"]["mean_load_by_expert"]
    selected = report["routing"]["selection_frequency_by_expert"]
    x = np.arange(EVENT_COUNT)
    axes[1].bar(
        x - 0.18, load, 0.36, label="Gate load", color="#26a69a")
    axes[1].bar(
        x + 0.18, selected, 0.36, label="Selected", color="#ef6c00")
    axes[1].set_xticks(x, [f"E{i}: {1+3*i}–{3+3*i}" for i in x])
    axes[1].set_ylabel("Mean fraction")
    axes[1].set_title("Routing by causal event (E0 is cue)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "CEM LeWM mixture-of-memory experts · age 15 · three seeds")
    figure.tight_layout()
    figure.savefig(prefix.with_name(prefix.name + "_routing_exposure.png"),
                   dpi=220, bbox_inches="tight")
    figure.savefig(prefix.with_name(prefix.name + "_routing_exposure.pdf"),
                   bbox_inches="tight")
    plt.close(figure)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    ladder = report["diagnostic_ladder"]
    controls = report["controls"]
    routing = report["routing"]
    deletion = report["causal_deletion"]["mean_delta_future_loss_by_expert"]
    lines = [
        "# CEM LeWM Mixture-of-Memory Experts",
        "",
        "Architecture B was evaluated on six-way PushT visual binding at age 15 "
        f"with seeds {report['seeds']}. The official LeWM host remained frozen "
        f"(digest `{report['host_digest']}`), and semantic labels were used only "
        "for the post-hoc audit.",
        "",
        "## Exact results",
        "",
        f"- Memory-only: {ladder['memory_only']['mean']:.6f} ± "
        f"{ladder['memory_only']['std']:.6f} "
        f"({ladder['memory_only']['seed_values']})",
        f"- Expert-mixture output: {ladder['expert_mixture']['mean']:.6f} ± "
        f"{ladder['expert_mixture']['std']:.6f} "
        f"({ladder['expert_mixture']['seed_values']})",
        f"- Frozen host output: {ladder['host_output']['mean']:.6f} ± "
        f"{ladder['host_output']['std']:.6f} "
        f"({ladder['host_output']['seed_values']})",
        f"- Dense-residual v3 host baseline: "
        f"{report['v3_dense_residual_baseline_host_output']['mean']:.6f} ± "
        f"{report['v3_dense_residual_baseline_host_output']['std']:.6f}",
        f"- Strict future-latent loss with memory: "
        f"{report['host_future_latent_loss']['with_memory']['mean']:.6f}; "
        f"without memory: "
        f"{report['host_future_latent_loss']['without_memory']['mean']:.6f}; "
        f"reset: {report['host_future_latent_loss']['reset_memory']['mean']:.6f}.",
        "",
        "Controls (mean balanced accuracy): " + ", ".join(
            f"{key}={value['mean']:.6f}" for key, value in controls.items()),
        "",
        "## Routing and causality",
        "",
        f"- Mean routing entropy: {routing['mean_entropy']['mean']:.6f}; "
        f"normalized entropy: {routing['normalized_entropy']['mean']:.6f}.",
        f"- Mean gate load by event expert: "
        f"{routing['mean_load_by_expert']}.",
        f"- Cue event retrieval frequency: "
        f"{routing['cue_event_retrieval_frequency']['mean']:.6f}; selection "
        f"frequency: {routing['cue_event_selection_frequency']['mean']:.6f}.",
        f"- Expert deletion Δ future loss: {deletion}.",
        "",
        "## Verdict and failures",
        "",
        f"Success gate: **{'PASS' if report['success_gate']['passed'] else 'FAIL'}** "
        "(host output ≥0.75 and every control ≤0.217).",
        f"Six-way geometry: **{report['six_way_geometry']}**.",
    ]
    if report["failures"]:
        lines.extend(["", *[f"- {failure}" for failure in report["failures"]]])
    lines.extend([
        "",
        "![Routing and exposure](assets/"
        "cem_lewm_memory_experts_routing_exposure.png)",
        "",
    ])
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="multi-item-visual-binding-recall",
        choices=["multi-item-visual-binding-recall"])
    parser.add_argument("--age", type=int, default=15, choices=[15])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--retrieve-k", type=int, default=4)
    parser.add_argument("--route-k", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--future-weight", type=float, default=0.25)
    parser.add_argument("--expert-sep-weight", type=float, default=1.0)
    parser.add_argument("--host-sep-weight", type=float, default=2.0)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--counterfactual-cache", default=str(DEFAULT_COUNTERFACTUAL_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--asset-prefix", default=str(DEFAULT_ASSET_PREFIX))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    validate_pusht_device(args.device)
    if args.device != "cuda:2":
        parser.error("this assigned experiment must run on cuda:2")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("assigned cuda:2 is unavailable")
    torch.cuda.set_device(torch.device(args.device))
    device = torch.device(args.device)
    locked = load_locked_pusht_spec(
        str(DEFAULT_PUSHT_SPEC), str(DEFAULT_PUSHT_LOCK))
    bundle = resolve_pusht_path(locked["official_host"]["bundle_path"])
    host = load_official_pusht_checkpoint(bundle, device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    host_digest = state_digest(host)
    spec, train, validation = load_data(args, host, device)
    results = []
    for seed in args.seeds:
        if args.evaluate_only:
            result_path = Path(args.output) / f"s{seed}" / "result.json"
            checkpoint_path = (
                Path(args.output) / f"s{seed}" / "memory_experts.pt")
            if not result_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"missing seed {seed} result/checkpoint")
            result = json.loads(result_path.read_text())
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
            model = MemoryExpertMixture(
                dim=args.dim, retrieve_k=args.retrieve_k,
                route_k=args.route_k,
                residual_scale=args.residual_scale).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            result["metrics"] = evaluate(
                args, seed, model, host, spec, train, validation, device)
            result_path.write_text(json.dumps(result, indent=2))
        else:
            result = train_seed(
                args, seed, host, spec, train, validation, device, host_digest)
        results.append(result)
    report = aggregate(args, results, host_digest)
    print(json.dumps({
        "report": str(Path(args.output) / "report.json"),
        "host_output": report["diagnostic_ladder"]["host_output"],
        "controls": {
            key: value["mean"] for key, value in report["controls"].items()},
        "gate": report["success_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
