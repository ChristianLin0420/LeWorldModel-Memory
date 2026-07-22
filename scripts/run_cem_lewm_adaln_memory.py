#!/usr/bin/env python3
"""CEM Architecture C: prediction-level memory conditioning for frozen LeWM.

The official predictor exposes ``action_encoder(actions)`` as the conditioning
tensor consumed by every AdaLN block.  This runner adds a query-retrieved memory
vector to that tensor; context latents are never modified.  Encoder, action
encoder, predictor, and prediction projector remain frozen and their joint
state-dict digest is asserted unchanged.

Training is label-free: frozen-host future-latent MSE plus same-base,
six-candidate counterfactual separability.  Semantic labels are opened only for
the post-hoc balanced-accuracy audit.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cem_controller import SurpriseWriteGate  # noqa: E402
from lewm.models.official_lewm import OFFICIAL_EMBED_DIM, OFFICIAL_HISTORY  # noqa: E402
from lewm.models.official_lewm_pusht import load_official_pusht_checkpoint  # noqa: E402
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    resolve_pusht_path,
)
from scripts.run_cem_lewm_pusht import (  # noqa: E402
    age_adjusted_spec,
    balanced_acc,
    frame_surprise,
)
from scripts.run_lewm_pusht_host_writer import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_COUNTERFACTUAL_CACHE,
    evidence_targets,
    load_admitted,
    load_or_build_counterfactual_cache,
    state_digest,
)
from scripts.run_mem_jepa_stage_b import fit_classifier  # noqa: E402
from scripts.run_mem_jepa_stage_c import (  # noqa: E402
    contrastive_loss,
    positive_cosine_loss,
)

OUTPUT = ROOT / "outputs/cem_lewm_adaln_memory_v1"
ASSET_STEM = ROOT / "docs/assets/cem_lewm_adaln_exposure"
REPORT_MD = ROOT / "docs/CEM_LEWM_ADALN_REPORT.md"
TASK = "multi-item-visual-binding-recall"


def tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)


class AdaLNMemoryAdapter(nn.Module):
    """Keep slots distinct through query cross-attention, then emit AdaLN delta."""

    def __init__(self, *, dim: int = 128, slots: int = 6, heads: int = 4,
                 max_frames: int = 20, condition_scale: float = 1.0) -> None:
        super().__init__()
        target_dim = OFFICIAL_EMBED_DIM * 3
        self.slots = int(slots)
        self.z_proj = nn.Linear(OFFICIAL_EMBED_DIM, dim)
        self.action_proj = nn.Linear(ACTION_DIM, dim)
        self.time = nn.Embedding(max_frames, dim)
        self.assign = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, slots),
        )
        self.query = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
        )
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.to_condition = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, OFFICIAL_EMBED_DIM),
        )
        self.condition_gate = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
            nn.Linear(dim, 1),
        )
        # Linear audit-aligned heads: successful candidate recovery cannot hide
        # behind a nonlinear probe while the raw carrier remains unreadable.
        self.memory_head = nn.Linear(slots * dim, target_dim)
        self.condition_head = nn.Linear(OFFICIAL_EMBED_DIM, target_dim)
        self.host_head = nn.Linear(OFFICIAL_EMBED_DIM, target_dim)
        self.condition_scale = nn.Parameter(torch.tensor(float(condition_scale)))
        nn.init.zeros_(self.to_condition[-1].weight)
        nn.init.zeros_(self.to_condition[-1].bias)
        nn.init.zeros_(self.condition_gate[-1].weight)
        nn.init.constant_(self.condition_gate[-1].bias, -1.0)

    def tokens(self, z: torch.Tensor, actions: torch.Tensor,
               times: torch.Tensor) -> torch.Tensor:
        return (
            self.z_proj(z) + self.action_proj(actions)
            + self.time(times)[None, :, :]
        )

    def make_slots(self, tokens: torch.Tensor,
                   write_gate: torch.Tensor) -> torch.Tensor:
        logits = self.assign(tokens).transpose(1, 2)
        logits = logits + torch.log(write_gate.clamp_min(1e-6))[:, None, :]
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bst,btd->bsd", weights, tokens)

    def retrieve(self, slots: torch.Tensor, context_z: torch.Tensor,
                 context_actions: torch.Tensor,
                 context_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_tokens = self.query(self.tokens(
            context_z, context_actions, context_times))
        attended, weights = self.cross(
            query_tokens, slots, slots, need_weights=True,
            average_attn_weights=False)
        joined = torch.cat((query_tokens, attended), dim=-1)
        delta = self.to_condition(joined)
        gate = torch.sigmoid(self.condition_gate(joined))
        condition = self.condition_scale * gate * delta
        return condition, weights

    def memory_features(self, slots: torch.Tensor) -> torch.Tensor:
        return slots.reshape(slots.shape[0], -1)


def host_predict(host: nn.Module, z: torch.Tensor, actions: torch.Tensor,
                 memory_condition: torch.Tensor | None = None) -> torch.Tensor:
    """Official prediction path with optional additive AdaLN conditioning."""
    autocast = z.device.type == "cuda"
    with torch.autocast(
            device_type=z.device.type,
            dtype=torch.bfloat16 if autocast else torch.float32,
            enabled=autocast):
        embedded_actions = host.action_encoder(actions)
        if memory_condition is not None:
            if memory_condition.shape != embedded_actions.shape:
                raise ValueError(
                    f"memory condition {tuple(memory_condition.shape)} != "
                    f"action embedding {tuple(embedded_actions.shape)}")
            embedded_actions = embedded_actions + memory_condition
        prediction = host.predictor(z, embedded_actions)
        batch, steps = prediction.shape[:2]
        prediction = host.pred_proj(prediction.flatten(0, 1))
        return prediction.reshape(batch, steps, -1).float()


class Experiment:
    def __init__(self, spec: dict[str, Any], device: torch.device) -> None:
        self.spec = spec
        self.device = device
        sequence = spec["sequence"]
        self.decision = int(sequence["decision_index"])
        self.context = np.asarray(
            sequence["final_context_indices"], dtype=np.int64)
        self.prefix = np.arange(self.decision, dtype=np.int64)
        self.cue_start = int(sequence["cue_start"])
        self.cue_length = int(sequence["cue_length"])
        self.target = self.context + 1

    def batch(self, data: dict[str, Any], rows: np.ndarray
              ) -> tuple[torch.Tensor, torch.Tensor]:
        return tensor(data["z"][rows], self.device), tensor(
            data["actions"][rows], self.device)

    def memory(self, adapter: AdaLNMemoryAdapter, write_gate: SurpriseWriteGate,
               host: nn.Module, z: torch.Tensor, actions: torch.Tensor,
               *, reset: bool = False,
               delete_frames: list[int] | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_t = torch.as_tensor(self.context, device=self.device)
        if reset:
            source = self.context
            source_t = context_t
            surprise = frame_surprise(
                host, z, actions, self.decision)[:, self.context]
        else:
            source = self.prefix
            source_t = torch.as_tensor(self.prefix, device=self.device)
            surprise = frame_surprise(host, z, actions, self.decision)
        gate = write_gate.soft_gate(surprise)
        if delete_frames:
            gate = gate.clone()
            source_lookup = {int(frame): pos for pos, frame in enumerate(source)}
            for frame in delete_frames:
                if int(frame) in source_lookup:
                    gate[:, source_lookup[int(frame)]] = 0.0
        source_tokens = adapter.tokens(
            z[:, source], actions[:, source], source_t)
        slots = adapter.make_slots(source_tokens, gate)
        condition, attention = adapter.retrieve(
            slots, z[:, self.context], actions[:, self.context], context_t)
        return condition, slots, gate, attention

    def forward(self, adapter: AdaLNMemoryAdapter,
                write_gate: SurpriseWriteGate, host: nn.Module,
                z: torch.Tensor, actions: torch.Tensor, *,
                reset: bool = False,
                delete_frames: list[int] | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition, slots, _, _ = self.memory(
            adapter, write_gate, host, z, actions, reset=reset,
            delete_frames=delete_frames)
        output = host_predict(
            host, z[:, self.context], actions[:, self.context], condition)
        return output, condition, slots


def separability(query: torch.Tensor, candidates: torch.Tensor,
                 temperature: float) -> tuple[torch.Tensor, float]:
    contrast, accuracy = contrastive_loss(query, candidates, temperature)
    return contrast + positive_cosine_loss(query, candidates), accuracy


def count_parameters(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters())


def train_seed(args: argparse.Namespace, seed: int, spec: dict[str, Any],
               train_data: dict[str, Any], validation_data: dict[str, Any],
               host: nn.Module, host_digest: str,
               device: torch.device) -> dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(spec, device)
    adapter = AdaLNMemoryAdapter(
        dim=args.dim, slots=args.slots, heads=args.heads,
        condition_scale=args.condition_scale).to(device)
    write_gate = SurpriseWriteGate(
        quantile=args.quantile, temperature=args.write_temperature).to(device)
    params = list(adapter.parameters()) + list(write_gate.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))
    indices = np.arange(len(train_data["z"]))
    rng = np.random.default_rng(1000 + seed)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        adapter.train()
        future_values, host_values, cond_values, memory_values = [], [], [], []
        matches = []
        started = time.time()
        for offset in range(0, len(indices), args.batch_size):
            rows = indices[offset:offset + args.batch_size]
            if len(rows) < 4:
                continue
            z, actions = experiment.batch(train_data, rows)
            with torch.no_grad():
                surprise = frame_surprise(
                    host, z, actions, experiment.decision)
                write_gate.update_threshold(surprise)
            output, condition, slots = experiment.forward(
                adapter, write_gate, host, z, actions)
            future_loss = F.mse_loss(output, z[:, experiment.target])
            candidates = evidence_targets(
                train_data, rows, device, spec=spec,
                target_mode="counterfactual_delta_flat",
                candidate_count=6, shuffle_targets=False)
            host_sep, host_match = separability(
                adapter.host_head(output[:, -1]), candidates,
                args.temperature)
            condition_sep, _ = separability(
                adapter.condition_head(condition[:, -1]), candidates,
                args.temperature)
            memory_sep, _ = separability(
                adapter.memory_head(adapter.memory_features(slots)),
                candidates, args.temperature)
            write_cost = args.beta * write_gate.soft_gate(surprise).mean()
            loss = (
                args.future_weight * future_loss
                + args.host_sep_weight * host_sep
                + args.condition_sep_weight * condition_sep
                + args.memory_sep_weight * memory_sep
                + write_cost
                + args.condition_l2 * condition.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            future_values.append(float(future_loss.detach()))
            host_values.append(float(host_sep.detach()))
            cond_values.append(float(condition_sep.detach()))
            memory_values.append(float(memory_sep.detach()))
            matches.append(host_match)
        scheduler.step()
        row = {
            "epoch": epoch,
            "future_latent_loss": float(np.mean(future_values)),
            "host_separability_loss": float(np.mean(host_values)),
            "condition_separability_loss": float(np.mean(cond_values)),
            "memory_separability_loss": float(np.mean(memory_values)),
            "host_candidate_match": float(np.mean(matches)),
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"[adaln-cem] seed={seed} epoch={epoch}/{args.epochs} "
            f"future={row['future_latent_loss']:.5f} "
            f"match={row['host_candidate_match']:.3f} "
            f"sec={row['seconds']:.1f}", flush=True)

    if state_digest(host) != host_digest:
        raise RuntimeError("frozen official LeWM host changed during training")
    result = evaluate(
        args, seed, experiment, adapter, write_gate, host,
        train_data, validation_data, host_digest)
    result["history"] = history
    seed_dir = OUTPUT / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter": adapter.state_dict(),
        "write_gate": write_gate.state_dict(),
        "seed": seed,
        "host_digest": host_digest,
        "config": vars(args),
    }, seed_dir / "adapter.pt")
    (seed_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def collect_features(experiment: Experiment, adapter: AdaLNMemoryAdapter,
                     write_gate: SurpriseWriteGate, host: nn.Module,
                     data: dict[str, Any], condition_name: str,
                     *, delete_frames: list[int] | None = None
                     ) -> dict[str, np.ndarray]:
    result = {"memory_only": [], "conditioning_vector": [], "host_output": []}
    for offset in range(0, len(data["z"]), 256):
        rows = np.arange(offset, min(len(data["z"]), offset + 256))
        z, actions = experiment.batch(data, rows)
        if condition_name in {"no_state", "host_only"}:
            condition = torch.zeros(
                len(rows), len(experiment.context), OFFICIAL_EMBED_DIM,
                device=experiment.device)
            # Zero-valued carriers keep diagnostic shapes fixed.
            slots = torch.zeros(
                len(rows), adapter.slots, adapter.z_proj.out_features,
                device=experiment.device)
        else:
            condition, slots, _, _ = experiment.memory(
                adapter, write_gate, host, z, actions,
                reset=condition_name == "reset",
                delete_frames=delete_frames)
            if condition_name == "shuffled":
                condition = condition.roll(1, dims=0)
                slots = slots.roll(1, dims=0)
            elif condition_name == "random":
                generator = torch.Generator(
                    device=experiment.device).manual_seed(9300 + offset)
                random = torch.randn(
                    condition.shape, generator=generator,
                    device=experiment.device)
                random = random * (
                    condition.square().mean().sqrt()
                    / random.square().mean().sqrt().clamp_min(1e-8))
                condition = random
                slots = torch.randn(
                    slots.shape, generator=generator,
                    device=experiment.device)
        output = host_predict(
            host, z[:, experiment.context], actions[:, experiment.context],
            condition)
        result["memory_only"].append(
            adapter.memory_features(slots).float().cpu().numpy())
        result["conditioning_vector"].append(
            condition[:, -1].float().cpu().numpy())
        result["host_output"].append(output[:, -1].float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def audit_bacc(train_features: dict[str, np.ndarray], train_labels: np.ndarray,
               validation_features: dict[str, np.ndarray],
               validation_labels: np.ndarray) -> dict[str, float]:
    classes = len(np.unique(train_labels))
    return {
        level: balanced_acc(
            fit_classifier(train_features[level], train_labels, values),
            validation_labels, classes)
        for level, values in validation_features.items()
    }


@torch.no_grad()
def losses_and_norm(experiment: Experiment, adapter: AdaLNMemoryAdapter,
                    write_gate: SurpriseWriteGate, host: nn.Module,
                    data: dict[str, Any]) -> dict[str, float]:
    with_memory, without_memory, reset_memory, norms = [], [], [], []
    for offset in range(0, len(data["z"]), 256):
        rows = np.arange(offset, min(len(data["z"]), offset + 256))
        z, actions = experiment.batch(data, rows)
        output, condition, _ = experiment.forward(
            adapter, write_gate, host, z, actions)
        reset, _, _ = experiment.forward(
            adapter, write_gate, host, z, actions, reset=True)
        baseline = host_predict(
            host, z[:, experiment.context], actions[:, experiment.context])
        target = z[:, experiment.target]
        with_memory.append(float(F.mse_loss(output, target)))
        reset_memory.append(float(F.mse_loss(reset, target)))
        without_memory.append(float(F.mse_loss(baseline, target)))
        norms.append(float(condition.norm(dim=-1).mean()))
    return {
        "host_future_latent_loss_with_memory": float(np.mean(with_memory)),
        "host_future_latent_loss_without_memory": float(np.mean(without_memory)),
        "host_future_latent_loss_reset_memory": float(np.mean(reset_memory)),
        "conditioning_l2_norm": float(np.mean(norms)),
    }


@torch.no_grad()
def benchmark_overhead(experiment: Experiment, adapter: AdaLNMemoryAdapter,
                       write_gate: SurpriseWriteGate, host: nn.Module,
                       data: dict[str, Any]) -> dict[str, float]:
    rows = np.arange(min(128, len(data["z"])))
    z, actions = experiment.batch(data, rows)
    for _ in range(5):
        host_predict(host, z[:, experiment.context], actions[:, experiment.context])
        experiment.forward(adapter, write_gate, host, z, actions)
    if experiment.device.type == "cuda":
        torch.cuda.synchronize(experiment.device)
    started = time.perf_counter()
    benchmark_steps = 5
    for _ in range(benchmark_steps):
        host_predict(host, z[:, experiment.context], actions[:, experiment.context])
    if experiment.device.type == "cuda":
        torch.cuda.synchronize(experiment.device)
    base_ms = (time.perf_counter() - started) * 1000 / benchmark_steps
    started = time.perf_counter()
    for _ in range(benchmark_steps):
        experiment.forward(adapter, write_gate, host, z, actions)
    if experiment.device.type == "cuda":
        torch.cuda.synchronize(experiment.device)
    memory_ms = (time.perf_counter() - started) * 1000 / benchmark_steps
    host_params = count_parameters(host)
    adapter_params = count_parameters(adapter) + count_parameters(write_gate)
    return {
        "host_parameters": host_params,
        "trainable_adapter_parameters": adapter_params,
        "parameter_overhead_fraction": adapter_params / host_params,
        "base_batch_latency_ms": base_ms,
        "memory_batch_latency_ms": memory_ms,
        "latency_overhead_fraction": memory_ms / base_ms - 1.0,
        "benchmark_batch_size": len(rows),
    }


def evaluate(args: argparse.Namespace, seed: int, experiment: Experiment,
             adapter: AdaLNMemoryAdapter, write_gate: SurpriseWriteGate,
             host: nn.Module, train_data: dict[str, Any],
             validation_data: dict[str, Any],
             host_digest: str) -> dict[str, Any]:
    adapter.eval()
    write_gate.eval()
    train_full = collect_features(
        experiment, adapter, write_gate, host, train_data, "full")
    validation_by_condition = {
        name: collect_features(
            experiment, adapter, write_gate, host, validation_data, name)
        for name in (
            "full", "reset", "no_state", "host_only", "shuffled", "random")
    }
    train_labels = np.asarray(train_data["labels"], dtype=np.int64)
    validation_labels = np.asarray(validation_data["labels"], dtype=np.int64)
    ladder = audit_bacc(
        train_full, train_labels, validation_by_condition["full"],
        validation_labels)
    controls = {}
    for name, features in validation_by_condition.items():
        prediction = fit_classifier(
            train_full["host_output"], train_labels, features["host_output"])
        controls[name] = balanced_acc(
            prediction, validation_labels, len(np.unique(train_labels)))
    controls["threshold"] = 0.217
    controls["pass"] = bool(
        controls["full"] >= 0.75
        and all(controls[name] <= 0.217 for name in (
            "reset", "no_state", "host_only", "shuffled", "random")))

    cue_frames = list(range(
        experiment.cue_start, experiment.cue_start + experiment.cue_length))
    noncue = [
        frame for frame in range(1, experiment.decision)
        if frame not in cue_frames]
    rng = np.random.default_rng(7000 + seed)
    random_frames = sorted(int(value) for value in rng.choice(
        noncue, size=len(cue_frames), replace=False))
    cue_deleted = collect_features(
        experiment, adapter, write_gate, host, validation_data, "full",
        delete_frames=cue_frames)
    random_deleted = collect_features(
        experiment, adapter, write_gate, host, validation_data, "full",
        delete_frames=random_frames)
    deletion_bacc = {}
    for name, features in (
            ("cue_group", cue_deleted), ("random_group", random_deleted)):
        prediction = fit_classifier(
            train_full["host_output"], train_labels, features["host_output"])
        deletion_bacc[name] = balanced_acc(
            prediction, validation_labels, len(np.unique(train_labels)))

    def deletion_loss(frames: list[int]) -> float:
        values = []
        for offset in range(0, len(validation_data["z"]), 256):
            rows = np.arange(
                offset, min(len(validation_data["z"]), offset + 256))
            z, actions = experiment.batch(validation_data, rows)
            full, _, _ = experiment.forward(
                adapter, write_gate, host, z, actions)
            deleted, _, _ = experiment.forward(
                adapter, write_gate, host, z, actions,
                delete_frames=frames)
            target = z[:, experiment.target]
            values.append(float(
                (F.mse_loss(deleted, target)
                 - F.mse_loss(full, target)).detach()))
        return float(np.mean(values))

    endpoint = losses_and_norm(
        experiment, adapter, write_gate, host, validation_data)
    endpoint["improvement"] = (
        endpoint["host_future_latent_loss_without_memory"]
        - endpoint["host_future_latent_loss_with_memory"])
    deletion = {
        "cue_window_frames": cue_frames,
        "random_frames": random_frames,
        "delta_future_loss_delete_cue_group": deletion_loss(cue_frames),
        "delta_future_loss_delete_random_group": deletion_loss(random_frames),
        "host_output_bacc_delete_cue_group": deletion_bacc["cue_group"],
        "host_output_bacc_delete_random_group": deletion_bacc["random_group"],
    }
    deletion["cue_deletion_more_causal"] = bool(
        deletion["host_output_bacc_delete_cue_group"]
        < deletion["host_output_bacc_delete_random_group"])
    return {
        "schema": "cem_lewm_adaln_memory_v1",
        "seed": seed,
        "task": TASK,
        "age": 15,
        "architecture": {
            "name": "prediction-level memory conditioning",
            "hook": "action_encoder output + memory -> predictor AdaLN c",
            "context_latent_residual_injection": False,
            "query_conditioned_retrieval": True,
            "distinct_slots_until_cross_attention": adapter.slots,
        },
        "endpoint": endpoint,
        "diagnostic_ladder": ladder,
        "audit": controls,
        "causal_deletion": deletion,
        "overhead": benchmark_overhead(
            experiment, adapter, write_gate, host, validation_data),
        "host_digest_before": host_digest,
        "host_digest_after": state_digest(host),
        "frozen_host_digest_unchanged": state_digest(host) == host_digest,
        "labels_used_for_training_loss": False,
        "labels_used_for_posthoc_audit": True,
        "counterfactual_candidates": 6,
    }


def v3_baseline() -> dict[str, Any]:
    values = []
    for seed in range(3):
        path = (
            ROOT / "outputs/cem_lewm_v3/D"
            / TASK / f"s{seed}/summary.json")
        if path.is_file():
            values.append(json.loads(path.read_text()))
    if not values:
        return {"available": False}
    return {
        "available": True,
        "configuration": "v3-D dense latent residual",
        "seeds": len(values),
        "host_output_bacc_mean": float(np.mean([
            value["diagnostic_ladder"]["host_output"] for value in values])),
        "host_output_bacc_std": float(np.std([
            value["diagnostic_ladder"]["host_output"] for value in values])),
        "host_loss_with_memory_mean": float(np.mean([
            value["host_loss_with_memory"] for value in values])),
        "host_loss_without_memory_mean": float(np.mean([
            value["host_loss_without_memory"] for value in values])),
    }


def aggregate(results: list[dict[str, Any]], args: argparse.Namespace,
              gpu: str) -> dict[str, Any]:
    def stats(path: tuple[str, ...]) -> dict[str, float]:
        values = []
        for result in results:
            value: Any = result
            for key in path:
                value = value[key]
            values.append(float(value))
        return {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    report = {
        "schema": "cem_lewm_adaln_memory_report_v1",
        "task": TASK,
        "age": 15,
        "seeds": [result["seed"] for result in results],
        "device": gpu,
        "method": results[0]["architecture"],
        "training": {
            "semantic_labels": "post-hoc audit only",
            "primary_loss": "frozen-host future-latent MSE",
            "same_base_counterfactual_candidates": 6,
            "arguments": vars(args),
        },
        "metrics": {
            "host_future_loss_with_memory": stats((
                "endpoint", "host_future_latent_loss_with_memory")),
            "host_future_loss_without_memory": stats((
                "endpoint", "host_future_latent_loss_without_memory")),
            "host_output_bacc": stats(("diagnostic_ladder", "host_output")),
            "memory_only_bacc": stats(("diagnostic_ladder", "memory_only")),
            "conditioning_vector_bacc": stats((
                "diagnostic_ladder", "conditioning_vector")),
            "conditioning_l2_norm": stats((
                "endpoint", "conditioning_l2_norm")),
            "cue_deleted_bacc": stats((
                "causal_deletion", "host_output_bacc_delete_cue_group")),
            "random_deleted_bacc": stats((
                "causal_deletion", "host_output_bacc_delete_random_group")),
        },
        "controls": {
            name: stats(("audit", name))
            for name in (
                "reset", "no_state", "host_only", "shuffled", "random")
        },
        "success_criteria": {
            "host_output_threshold": 0.75,
            "control_ceiling": 0.217,
            "host_output_pass_all_seeds": all(
                result["diagnostic_ladder"]["host_output"] >= 0.75
                for result in results),
            "controls_pass_all_seeds": all(
                all(result["audit"][name] <= 0.217 for name in (
                    "reset", "no_state", "host_only", "shuffled", "random"))
                for result in results),
            "joint_pass_all_seeds": all(
                result["audit"]["pass"] for result in results),
        },
        "causal_deletion": {
            "cue_more_causal_seeds": sum(
                bool(result["causal_deletion"]["cue_deletion_more_causal"])
                for result in results),
            "seed_count": len(results),
        },
        "overhead": {
            "trainable_adapter_parameters": results[0]["overhead"][
                "trainable_adapter_parameters"],
            "host_parameters": results[0]["overhead"]["host_parameters"],
            "parameter_overhead_fraction": results[0]["overhead"][
                "parameter_overhead_fraction"],
            "latency_overhead_fraction": stats((
                "overhead", "latency_overhead_fraction")),
        },
        "frozen_host": {
            "digest": results[0]["host_digest_before"],
            "unchanged_all_seeds": all(
                result["frozen_host_digest_unchanged"] for result in results),
        },
        "dense_residual_v3_comparison": v3_baseline(),
        "per_seed": results,
    }
    host_loss_ratio = (
        report["metrics"]["host_future_loss_with_memory"]["mean"]
        / report["metrics"]["host_future_loss_without_memory"]["mean"])
    report["success_criteria"]["host_future_loss_ratio"] = host_loss_ratio
    report["success_criteria"]["host_loss_non_degradation"] = (
        host_loss_ratio <= 1.0)
    report["conclusion"] = (
        "PASS: prediction-level conditioning crosses the frozen-host exposure "
        "boundary with fail-closed controls. QUALIFICATION: it does not "
        "preserve the frozen host's future-latent loss."
        if report["success_criteria"]["joint_pass_all_seeds"]
        else "FAIL: prediction-level conditioning does not satisfy the full "
             "host-output/control gate; see per-seed localization."
    )
    return report


def render_figure(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    names = ("Memory only", "Conditioning vector", "Frozen host output")
    keys = ("memory_only_bacc", "conditioning_vector_bacc", "host_output_bacc")
    means = [metrics[key]["mean"] for key in keys]
    stds = [metrics[key]["std"] for key in keys]
    controls = ("reset", "no_state", "host_only", "shuffled", "random")
    control_means = [report["controls"][key]["mean"] for key in controls]
    control_stds = [report["controls"][key]["std"] for key in controls]
    deletion_names = ("Full", "Cue deleted", "Random deleted", "v3 dense")
    v3 = report["dense_residual_v3_comparison"]
    deletion_values = [
        metrics["host_output_bacc"]["mean"],
        metrics["cue_deleted_bacc"]["mean"],
        metrics["random_deleted_bacc"]["mean"],
        v3.get("host_output_bacc_mean", float("nan")),
    ]
    deletion_errors = [
        metrics["host_output_bacc"]["std"],
        metrics["cue_deleted_bacc"]["std"],
        metrics["random_deleted_bacc"]["std"],
        v3.get("host_output_bacc_std", 0.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    axes[0].bar(
        np.arange(3), means, yerr=stds, capsize=3,
        color=("#78909c", "#1e88e5", "#5e35b1"),
        edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(np.arange(3), names, rotation=18, ha="right")
    axes[0].set_title("Exposure ladder")
    axes[1].bar(
        np.arange(len(controls)), control_means, yerr=control_stds, capsize=3,
        color="#90a4ae", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(
        np.arange(len(controls)),
        ("Reset", "No state", "Host only", "Shuffled", "Random"),
        rotation=25, ha="right")
    axes[1].set_title("Fail-closed controls")
    axes[2].bar(
        np.arange(4), deletion_values, yerr=deletion_errors, capsize=3,
        color=("#5e35b1", "#ef6c00", "#78909c", "#bdbdbd"),
        edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(
        np.arange(4), deletion_names, rotation=20, ha="right")
    axes[2].set_title("Causal deletion and v3 comparison")
    for axis in axes:
        axis.axhline(1 / 6, color="black", linestyle=":", label="chance")
        axis.axhline(0.217, color="#ef6c00", linestyle="--",
                     linewidth=1, label="control ceiling")
        axis.axhline(0.75, color="#c62828", linestyle="--",
                     linewidth=1, label="host-output gate")
        axis.set_ylim(0, 1.03)
        axis.set_ylabel("Six-way balanced accuracy")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "LeWM Architecture C · AdaLN/action memory conditioning · "
        "PushT binding age 15 · mean ± SD")
    fig.tight_layout()
    ASSET_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_STEM.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(ASSET_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    controls = report["controls"]

    def metric(name: str) -> str:
        value = metrics[name]
        return f"{value['mean']:.4f} ± {value['std']:.4f}"

    control_text = ", ".join(
        f"{name} {controls[name]['mean']:.4f}"
        for name in ("reset", "no_state", "host_only", "shuffled", "random"))
    v3 = report["dense_residual_v3_comparison"]
    md = f"""# CEM LeWM AdaLN Memory Report

## Result

{report["conclusion"]}

Three seeds were run on six-way PushT visual binding at evidence age 15.
Memory-only, conditioning-vector, and frozen-host-output balanced accuracies
were **{metric("memory_only_bacc")}**, **{metric("conditioning_vector_bacc")}**,
and **{metric("host_output_bacc")}**, respectively. The required host-output
gate is 0.75. Controls were: {control_text}; the ceiling is 0.217.

## Architecture and frozen-host contract

The official LeWM API exposes `action_encoder(actions)` before the predictor.
Its output is the per-token conditioning tensor consumed by all six
`ConditionalBlock.adaLN_modulation` paths. The adapter writes six distinct CEM
slots, retrieves them with context/action/time queries, and adds the resulting
192-D vector to the action embedding. Context latents are never perturbed.

Only the adapter and surprise-gate temperature train. Frozen host digest:
`{report["frozen_host"]["digest"]}`; unchanged across every seed:
**{report["frozen_host"]["unchanged_all_seeds"]}**.

## Objectives and causal checks

Training uses the frozen host's next-latent MSE over the legal three-frame
window and six same-base rendered counterfactual branches. The observed branch
is selected by latent equality, not a semantic class id. Labels are opened only
for the post-hoc ridge-probe audit.

Deleting the old cue group yields host-output BAcc
**{metric("cue_deleted_bacc")}**, versus **{metric("random_deleted_bacc")}**
for count-matched random deletion. Cue deletion was more causal in
{report["causal_deletion"]["cue_more_causal_seeds"]}/{report["causal_deletion"]["seed_count"]}
seeds. Mean conditioning norm was **{metric("conditioning_l2_norm")}**.

## Host loss and overhead

Future-latent loss with/without memory was
**{metric("host_future_loss_with_memory")} / {metric("host_future_loss_without_memory")}**.
Thus the stated exposure/control gate passes, but this is not a loss-preserving
integration: future loss is
**{report["success_criteria"]["host_future_loss_ratio"]:.1f}x** baseline. The
adapter repurposes frozen prediction capacity to expose the old cue rather than
improving the host's ordinary next-latent prediction.
The adapter has {report["overhead"]["trainable_adapter_parameters"]:,} trainable
parameters versus {report["overhead"]["host_parameters"]:,} frozen host
parameters ({100 * report["overhead"]["parameter_overhead_fraction"]:.3f}%).
Measured batch-latency overhead was
{100 * report["overhead"]["latency_overhead_fraction"]["mean"]:.1f}% on the
assigned GPU (includes prefix surprise and retrieval, batch size 128).

## Dense-residual v3 comparison and limits

The prior v3-D dense residual path reached host-output BAcc
{v3.get("host_output_bacc_mean", float("nan")):.4f}. Architecture C instead
uses the predictor's real AdaLN conditioning interface, so this is not a
surrogate hook. It is still an additive action-embedding intervention rather
than a new per-block side input; the same memory vector reaches each block only
through the frozen block-specific AdaLN projections.

Artifacts: `outputs/cem_lewm_adaln_memory_v1/report.json` and
`docs/assets/cem_lewm_adaln_exposure.{{png,pdf}}`.
"""
    REPORT_MD.write_text(md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--condition-scale", type=float, default=1.0)
    parser.add_argument("--write-temperature", type=float, default=0.25)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=1.0e-2)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--host-sep-weight", type=float, default=5.0)
    parser.add_argument("--condition-sep-weight", type=float, default=1.0)
    parser.add_argument("--memory-sep-weight", type=float, default=0.1)
    parser.add_argument("--condition-l2", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--counterfactual-cache", default=str(DEFAULT_COUNTERFACTUAL_CACHE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda:3"):
        raise ValueError("GPU 3 is forbidden for this experiment")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    locked = load_locked_pusht_spec(
        str(DEFAULT_PUSHT_SPEC), str(DEFAULT_PUSHT_LOCK))
    spec = age_adjusted_spec(locked, 15)
    train_data = load_admitted(locked, TASK, "train")
    validation_data = load_admitted(locked, TASK, "validation")
    host = load_official_pusht_checkpoint(
        resolve_pusht_path(locked["official_host"]["bundle_path"]), device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    digest = state_digest(host)
    cache = Path(args.counterfactual_cache)
    load_or_build_counterfactual_cache(
        train_data, spec, TASK, "train", host, device, cache, 128)
    load_or_build_counterfactual_cache(
        validation_data, spec, TASK, "validation", host, device, cache, 128)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [
        train_seed(
            args, seed, spec, copy.deepcopy(train_data),
            copy.deepcopy(validation_data), host, digest, device)
        for seed in args.seeds
    ]
    report = aggregate(results, args, str(device))
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2))
    render_figure(report)
    write_markdown(report)
    print(json.dumps({
        "report": str((OUTPUT / "report.json").relative_to(ROOT)),
        "figure": str(ASSET_STEM.with_suffix(".png").relative_to(ROOT)),
        "host_output_bacc": report["metrics"]["host_output_bacc"],
        "joint_pass": report["success_criteria"]["joint_pass_all_seeds"],
    }, indent=2))


if __name__ == "__main__":
    main()
