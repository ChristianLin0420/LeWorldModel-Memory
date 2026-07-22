#!/usr/bin/env python3
"""Final constrained CEM -> token -> semantic bottleneck -> frozen LeWM interface.

The CEM writer retains six distinct slots until a normalized 32/64-D semantic
bottleneck.  A bounded query-conditioned decoder exposes the code at the
complete frozen predictor output.  Training is label-free and uses a
primal-dual constraint on the official strict next-latent loss:

    L_host(memory) <= (1 + epsilon) L_host(no_memory).
"""
from __future__ import annotations

import argparse
import copy
import json
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
)
from scripts.run_cem_lewm_memory_tokens import (  # noqa: E402
    balanced_acc,
    frozen_boundary,
    frozen_project,
    make_data,
    sep_loss,
    tensor,
    write_gate,
)
from scripts.run_cem_lewm_semantic_adapter import (  # noqa: E402
    anti_collapse,
    code_contrastive,
    geometry_metrics,
    standardized_geometry_loss,
)
from scripts.run_lewm_pusht_host_writer import evidence_targets, state_digest  # noqa: E402
from scripts.run_mem_jepa_stage_b import fit_classifier  # noqa: E402

LATENT_DIM = 192
ACTION_DIM = 10
SLOTS = 6
TASK = "multi-item-visual-binding-recall"
OUTPUT = ROOT / "outputs/cem_lewm_hybrid_interface_v1"
DOCUMENT = ROOT / "docs/CEM_LEWM_HYBRID_INTERFACE_REPORT.md"
FIGURE = ROOT / "docs/assets/cem_lewm_hybrid"


class HybridExposure(nn.Module):
    """Identity-preserving CEM slots, normalized code, bounded host decoder."""

    def __init__(self, code_dim: int, max_residual: float) -> None:
        super().__init__()
        self.code_dim = int(code_dim)
        self.max_residual = float(max_residual)
        self.z_proj = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.action_proj = nn.Linear(ACTION_DIM, LATENT_DIM)
        self.time = nn.Embedding(20, LATENT_DIM)
        self.frame_type = nn.Parameter(torch.zeros(1, 1, LATENT_DIM))
        self.slot_position = nn.Parameter(
            torch.randn(1, SLOTS, LATENT_DIM) * 0.02)
        self.slot_norm = nn.LayerNorm(LATENT_DIM)
        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(LATENT_DIM), nn.Linear(LATENT_DIM, code_dim),
            nn.GELU(), nn.Linear(code_dim, code_dim),
        )
        self.read_query = nn.Sequential(
            nn.LayerNorm(LATENT_DIM), nn.Linear(LATENT_DIM, code_dim))
        self.read_score = nn.Linear(code_dim, 1, bias=False)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(3 * LATENT_DIM),
            nn.Linear(3 * LATENT_DIM, 128), nn.GELU(),
            nn.Linear(128, code_dim), nn.LayerNorm(code_dim),
        )
        self.decoder_query = nn.Sequential(
            nn.LayerNorm(LATENT_DIM), nn.Linear(LATENT_DIM, code_dim))
        self.decoder = nn.Sequential(
            nn.LayerNorm(2 * code_dim), nn.Linear(2 * code_dim, 128),
            nn.GELU(), nn.Linear(128, LATENT_DIM),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0)))
        # Begin close to the feasible no-memory point.
        self.residual_scale_logit = nn.Parameter(torch.tensor(-1.5))

    def slots_from_prefix(
        self, z: torch.Tensor, actions: torch.Tensor, times: torch.Tensor,
        gate: torch.Tensor, deletion: list[int] | None = None,
    ) -> torch.Tensor:
        tokens = (self.z_proj(z) + self.action_proj(actions)
                  + self.time(times)[None] + self.frame_type)
        if deletion:
            gate = gate.clone()
            gate[:, deletion] = 0.0
        # Hard top-k event identities: every selected frame appears exactly
        # once and remains a separate token until the semantic bottleneck.
        selected = torch.topk(gate, k=SLOTS, dim=1).indices
        selected = torch.sort(selected, dim=1).values
        gather = selected[..., None].expand(-1, -1, tokens.shape[-1])
        distinct_tokens = torch.gather(tokens, 1, gather)
        return self.slot_norm(distinct_tokens + self.slot_position)

    def bottleneck(self, slots: torch.Tensor,
                   host_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_code = self.slot_encoder(slots)
        query = self.read_query(host_query.mean(1))[:, None]
        score = self.read_score(torch.tanh(token_code + query)).squeeze(-1)
        code = F.normalize(
            torch.sum(torch.softmax(score, dim=1)[..., None] * token_code, dim=1),
            dim=-1)
        return code, token_code

    def candidate_codes(self, candidates: torch.Tensor) -> torch.Tensor:
        shape = candidates.shape
        code = self.candidate_encoder(candidates.reshape(-1, shape[-1]))
        return F.normalize(code.reshape(shape[0], shape[1], -1), dim=-1)

    def decode(self, code: torch.Tensor,
               base_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.decoder_query(base_output)
        expanded = code[:, None].expand(-1, base_output.shape[1], -1)
        raw = self.decoder(torch.cat((query, expanded), dim=-1))
        scale = self.max_residual * torch.sigmoid(self.residual_scale_logit)
        residual = scale * torch.tanh(raw)
        return base_output + residual, residual


def forward_hybrid(
    host: nn.Module, model: HybridExposure, z: torch.Tensor,
    actions: torch.Tensor, prefix_idx: np.ndarray, context_idx: np.ndarray,
    *, write_from_full: bool = True, deletion: list[int] | None = None,
    supplied_slots: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    boundary = frozen_boundary(host, z[:, context_idx], actions[:, context_idx])
    base_output = frozen_project(host, boundary)
    if supplied_slots is None:
        if write_from_full:
            idx = prefix_idx
            gate, _ = write_gate(host, z, actions, len(prefix_idx))
        else:
            # Reset is an empty causal store, not a smaller top-k retrieval.
            slots = torch.zeros(
                z.shape[0], SLOTS, LATENT_DIM,
                device=z.device, dtype=base_output.dtype)
            idx, gate = None, None
        local_delete = None
        if deletion and idx is not None:
            position = {int(value): i for i, value in enumerate(idx)}
            local_delete = [position[value] for value in deletion
                            if value in position]
        if idx is not None:
            slots = model.slots_from_prefix(
                z[:, idx], actions[:, idx],
                torch.as_tensor(idx, device=z.device), gate, local_delete)
    else:
        slots = supplied_slots
    code, token_code = model.bottleneck(slots, base_output)
    output, residual = model.decode(code, base_output)
    return {
        "slots": slots, "memory_tokens": token_code, "code": code,
        "decoded_signal": residual, "base_output": base_output, "output": output,
    }


def candidate_tensor(data: dict[str, Any], rows: np.ndarray,
                     device: torch.device, spec: dict[str, Any]) -> torch.Tensor:
    return evidence_targets(
        data, rows, device, spec=spec,
        target_mode="counterfactual_delta_flat", candidate_count=6,
        shuffle_targets=False)


def inferred_positive(data: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    """Infer observed branch by latent equality; semantic labels are untouched."""
    counterfactual = np.asarray(data["z_counterfactual"])[rows]
    observed = np.asarray(data["z_cue"])[rows]
    error = np.mean(
        (counterfactual - observed[:, None]) ** 2, axis=(2, 3))
    return np.argmin(error, axis=1).astype(np.int64)


@torch.no_grad()
def low_variance_carriers(
    host: nn.Module, data: dict[str, Any], context_idx: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Six fixed one-hot carriers in least-variable frozen-host coordinates."""
    outputs = []
    for off in range(0, len(data["labels"]), 256):
        rows = np.arange(off, min(off + 256, len(data["labels"])))
        z, actions = tensor(data["z"][rows], device), tensor(
            data["actions"][rows], device)
        base = frozen_project(
            host, frozen_boundary(
                host, z[:, context_idx], actions[:, context_idx]))
        outputs.append(base.flatten(1).float())
    variance = torch.cat(outputs).var(dim=0, unbiased=False)
    indices = torch.topk(variance, SLOTS, largest=False).indices
    carriers = torch.zeros(
        SLOTS, len(context_idx) * LATENT_DIM, device=device)
    carriers[torch.arange(SLOTS, device=device), indices] = 1.0
    return carriers, [int(value) for value in indices.cpu()]


def carrier_contrastive(
    residual: torch.Tensor, carriers: torch.Tensor, positive: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = F.normalize(residual.flatten(1), dim=-1) @ carriers.T
    return F.cross_entropy(logits / temperature, positive)


@torch.no_grad()
def audit_features(
    host: nn.Module, model: HybridExposure, data: dict[str, Any],
    prefix_idx: np.ndarray, context_idx: np.ndarray, device: torch.device,
    condition: str, seed: int,
) -> dict[str, np.ndarray]:
    levels = {key: [] for key in (
        "memory_token", "bottleneck", "decoded_signal", "host_output",
        "candidate_codes")}
    for off in range(0, len(data["labels"]), 256):
        rows = np.arange(off, min(off + 256, len(data["labels"])))
        z, actions = tensor(data["z"][rows], device), tensor(
            data["actions"][rows], device)
        full = forward_hybrid(
            host, model, z, actions, prefix_idx, context_idx)
        if condition == "full":
            used = full
        elif condition == "reset":
            used = forward_hybrid(
                host, model, z, actions, prefix_idx, context_idx,
                write_from_full=False)
        elif condition in ("host_only", "no_state"):
            used = {**full, "output": full["base_output"],
                    "decoded_signal": torch.zeros_like(full["decoded_signal"])}
        elif condition == "shuffled":
            used = forward_hybrid(
                host, model, z, actions, prefix_idx, context_idx,
                supplied_slots=full["slots"].roll(1, dims=0))
        elif condition == "random":
            generator = torch.Generator(device=device).manual_seed(
                19000 + seed + off)
            slots = torch.randn(
                full["slots"].shape, generator=generator, device=device)
            used = forward_hybrid(
                host, model, z, actions, prefix_idx, context_idx,
                supplied_slots=slots)
        else:
            raise ValueError(condition)
        candidates = candidate_tensor(data, rows, device, CURRENT_SPEC)
        candidate_codes = model.candidate_codes(candidates)
        values = {
            "memory_token": used["slots"].flatten(1),
            "bottleneck": used["code"],
            "decoded_signal": used["decoded_signal"].flatten(1),
            "host_output": used["output"].flatten(1),
            "candidate_codes": candidate_codes,
        }
        for key, value in values.items():
            levels[key].append(value.float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in levels.items()}


CURRENT_SPEC: dict[str, Any] = {}


@torch.no_grad()
def evaluate(
    host: nn.Module, model: HybridExposure, train: dict[str, Any],
    validation: dict[str, Any], spec: dict[str, Any], device: torch.device,
    epsilon: float, seed: int,
) -> dict[str, Any]:
    global CURRENT_SPEC
    CURRENT_SPEC = spec
    model.eval()
    seq = spec["sequence"]
    decision = int(seq["decision_index"])
    prefix = np.arange(decision, dtype=np.int64)
    context = np.asarray(seq["final_context_indices"], dtype=np.int64)
    target_idx = context + 1
    conditions = ("full", "reset", "host_only", "no_state", "shuffled", "random")
    train_full = audit_features(
        host, model, train, prefix, context, device, "full", seed)
    validation_features = {
        condition: audit_features(
            host, model, validation, prefix, context, device, condition, seed)
        for condition in conditions
    }
    ladder = {}
    for level in ("memory_token", "bottleneck", "decoded_signal", "host_output"):
        prediction = fit_classifier(
            train_full[level], train["labels"],
            validation_features["full"][level])
        ladder[level] = balanced_acc(prediction, validation["labels"], 6)
    controls = {}
    for condition in conditions:
        prediction = fit_classifier(
            train_full["host_output"], train["labels"],
            validation_features[condition]["host_output"])
        controls[condition] = balanced_acc(prediction, validation["labels"], 6)
    full_losses, base_losses = [], []
    for off in range(0, len(validation["labels"]), 256):
        rows = np.arange(off, min(off + 256, len(validation["labels"])))
        z, actions = tensor(validation["z"][rows], device), tensor(
            validation["actions"][rows], device)
        result = forward_hybrid(host, model, z, actions, prefix, context)
        target = z[:, target_idx]
        full_losses.append(float(F.mse_loss(result["output"], target)))
        base_losses.append(float(F.mse_loss(result["base_output"], target)))
    full_loss, base_loss = float(np.mean(full_losses)), float(np.mean(base_losses))
    ratio = full_loss / base_loss
    candidate_np = np.asarray([
        candidate_tensor(
            validation, np.asarray([i]), device, spec).cpu().numpy()[0]
        for i in range(len(validation["labels"]))
    ])
    geometry = geometry_metrics(
        candidate_np, validation_features["full"]["candidate_codes"])
    cue = list(range(
        int(seq["cue_start"]),
        int(seq["cue_start"]) + int(seq["cue_length"])))
    noncue = [value for value in prefix.tolist() if value not in cue]
    random_group = [int(x) for x in np.random.default_rng(707).choice(
        noncue, size=len(cue), replace=False)]
    deletion = {"cue": [], "random": []}
    for off in range(0, min(256, len(validation["labels"])), 128):
        rows = np.arange(off, min(off + 128, len(validation["labels"]), 256))
        z, actions = tensor(validation["z"][rows], device), tensor(
            validation["actions"][rows], device)
        target = z[:, target_idx]
        base = forward_hybrid(host, model, z, actions, prefix, context)
        base_value = F.mse_loss(base["output"], target)
        for name, group in (("cue", cue), ("random", random_group)):
            deleted = forward_hybrid(
                host, model, z, actions, prefix, context, deletion=group)
            deletion[name].append(float(
                F.mse_loss(deleted["output"], target) - base_value))
    cue_delta, random_delta = (
        float(np.mean(deletion["cue"])), float(np.mean(deletion["random"])))
    controls_pass = all(controls[key] <= 0.217 for key in
                        ("reset", "host_only", "no_state", "shuffled", "random"))
    feasible = ratio <= 1.0 + epsilon + 1e-6
    joint = controls["full"] >= 0.75 and controls_pass and feasible
    return {
        "diagnostic_ladder": ladder,
        "controls_host_output_bacc": controls,
        "host_future_latent_loss": {
            "with_memory": full_loss, "without_memory": base_loss,
            "ratio": ratio, "relative_degradation": ratio - 1.0,
            "epsilon": epsilon, "constraint_margin": 1.0 + epsilon - ratio,
            "feasible": feasible,
        },
        "six_way_geometry": geometry,
        "causal_group_deletion": {
            "cue_group": cue, "random_matched_group": random_group,
            "delta_host_loss_delete_cue": cue_delta,
            "delta_host_loss_delete_random": random_delta,
            "cue_hurts_more_than_random": cue_delta > random_delta,
        },
        "success_gate": {
            "host_output_at_least_0.75": controls["full"] >= 0.75,
            "all_controls_at_most_0.217": controls_pass,
            "host_constraint_feasible": feasible,
            "joint_passed": joint,
        },
    }


@torch.no_grad()
def benchmark(
    host: nn.Module, model: HybridExposure, data: dict[str, Any],
    spec: dict[str, Any], device: torch.device,
) -> dict[str, float]:
    rows = np.arange(min(128, len(data["labels"])))
    z, actions = tensor(data["z"][rows], device), tensor(
        data["actions"][rows], device)
    seq = spec["sequence"]
    prefix = np.arange(int(seq["decision_index"]), dtype=np.int64)
    context = np.asarray(seq["final_context_indices"], dtype=np.int64)

    def measure(adapted: bool) -> float:
        for _ in range(3):
            if adapted:
                forward_hybrid(host, model, z, actions, prefix, context)
            else:
                frozen_project(
                    host, frozen_boundary(
                        host, z[:, context], actions[:, context]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(20):
            if adapted:
                forward_hybrid(host, model, z, actions, prefix, context)
            else:
                frozen_project(
                    host, frozen_boundary(
                        host, z[:, context], actions[:, context]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return 1000 * (time.perf_counter() - start) / 20

    base, adapted = measure(False), measure(True)
    return {
        "batch_size": int(len(rows)), "base_batch_ms": base,
        "hybrid_batch_ms": adapted,
        "latency_overhead_fraction": adapted / base - 1.0,
    }


def train_one(
    args: argparse.Namespace, host: nn.Module, digest: str, spec: dict[str, Any],
    train: dict[str, Any], validation: dict[str, Any], *, code_dim: int,
    epsilon: float, seed: int, run_kind: str,
) -> dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = next(host.parameters()).device
    model = HybridExposure(code_dim, args.max_residual).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))
    seq = spec["sequence"]
    prefix = np.arange(int(seq["decision_index"]), dtype=np.int64)
    context = np.asarray(seq["final_context_indices"], dtype=np.int64)
    target_idx = context + 1
    carriers, carrier_indices = low_variance_carriers(
        host, train, context, device)
    rng, order = np.random.default_rng(5100 + seed), np.arange(
        len(train["labels"]))
    multiplier = float(args.dual_init)
    history = []
    best_state, best_train_ratio = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        model.train()
        metrics = {key: [] for key in (
            "total", "host", "baseline", "ratio", "violation", "multiplier",
            "budget", "host_sep", "carrier", "code_nce", "geometry")}
        start = time.perf_counter()
        for off in range(0, len(order), args.batch_size):
            rows = order[off:off + args.batch_size]
            if len(rows) < 4:
                continue
            z, actions = tensor(train["z"][rows], device), tensor(
                train["actions"][rows], device)
            result = forward_hybrid(
                host, model, z, actions, prefix, context)
            candidates = candidate_tensor(train, rows, device, spec)
            positive = torch.from_numpy(inferred_positive(train, rows)).to(device)
            candidate_codes = model.candidate_codes(candidates)
            host_loss = F.mse_loss(result["output"], z[:, target_idx])
            baseline = F.mse_loss(
                result["base_output"], z[:, target_idx]).detach()
            ratio = host_loss / baseline.clamp_min(1e-8)
            violation = ratio - (1.0 + epsilon)
            host_sep = sep_loss(
                result["decoded_signal"].flatten(1), candidates,
                args.temperature, 0.0)
            carrier = carrier_contrastive(
                result["decoded_signal"], carriers, positive, args.temperature)
            code_nce = code_contrastive(
                result["code"], candidate_codes, positive, model.logit_scale)
            geometry = standardized_geometry_loss(candidate_codes, candidates)
            variance, orthogonality = anti_collapse(torch.cat(
                (candidate_codes, result["code"][:, None]), dim=1))
            # Cosine separability is invariant to amplitude. Use the interior
            # of the allowed fidelity budget so the signal remains visible in
            # raw frozen-host features, while the dual term enforces the upper
            # boundary. This is not a soft replacement for the constraint.
            budget_target = 1.0 + args.budget_fraction * epsilon
            budget = (ratio - budget_target).square()
            objective = (
                args.host_sep_weight * host_sep
                + args.carrier_weight * carrier
                + args.code_nce_weight * code_nce
                + args.geometry_weight * geometry
                + args.variance_weight * variance
                + args.orthogonality_weight * orthogonality
                + args.budget_weight * budget)
            lagrangian = (
                objective + multiplier * violation
                + 0.5 * args.augmented_rho * F.relu(violation).square())
            optimizer.zero_grad(set_to_none=True)
            lagrangian.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            multiplier = float(np.clip(
                multiplier + args.dual_lr * float(violation.detach()),
                0.0, args.dual_max))
            for key, value in (
                ("total", lagrangian), ("host", host_loss),
                ("baseline", baseline), ("ratio", ratio),
                ("violation", violation), ("budget", budget),
                ("host_sep", host_sep), ("carrier", carrier),
                ("code_nce", code_nce), ("geometry", geometry)):
                metrics[key].append(float(value.detach()))
            metrics["multiplier"].append(multiplier)
        scheduler.step()
        record = {
            "epoch": epoch,
            **{key: float(np.mean(value)) for key, value in metrics.items()},
            "violation_rate": float(np.mean(np.asarray(
                metrics["violation"]) > 0)),
            "seconds": time.perf_counter() - start,
        }
        history.append(record)
        if record["ratio"] <= 1.0 + epsilon:
            best_train_ratio = record["ratio"]
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[hybrid] {run_kind} d{code_dim} eps={epsilon:.2f} s{seed} "
            f"ep{epoch}/{args.epochs} ratio={record['ratio']:.4f} "
            f"viol={record['violation_rate']:.2f} lambda={multiplier:.2f}",
            flush=True)
    if best_state is not None and args.restore_feasible:
        model.load_state_dict(best_state)
    if state_digest(host) != digest:
        raise RuntimeError("FROZEN official PushT LeWM digest changed")
    result = evaluate(
        host, model, train, validation, spec, device, epsilon, seed)
    result.update({
        "schema": "cem_lewm_hybrid_interface_cell_v1",
        "seed": seed, "age": 15,
        "config": {
            "code_dim": code_dim, "epsilon": epsilon,
            "decoder": "bounded_query_conditioned",
            "slots": SLOTS, "max_residual": args.max_residual,
            "constraint_method": "primal_dual_augmented_lagrangian",
            "dual_lr": args.dual_lr, "augmented_rho": args.augmented_rho,
            "budget_fraction": args.budget_fraction,
            "carrier": "fixed_low_variance_host_coordinates",
            "carrier_indices": carrier_indices,
            "retrieval": "top6 distinct surprise-ranked event identities",
        },
        "history": history,
        "final_dual_multiplier": multiplier,
        "frozen_host_digest_before": digest,
        "frozen_host_digest_after": state_digest(host),
        "frozen_host_digest_unchanged": True,
        "labels_used_for_training_loss": False,
        "candidate_identity_source": "argmin latent equality among six branches",
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()),
        "host_parameter_count": sum(
            parameter.numel() for parameter in host.parameters()),
        "overhead": benchmark(host, model, validation, spec, device),
    })
    run = OUTPUT / "runs" / run_kind / f"d{code_dim}_e{epsilon:.2f}" / f"s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), run / "adapter.pt")
    return result


def stats(results: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for result in results:
        value: Any = result
        for key in path:
            value = value[key]
        values.append(float(value))
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "by_seed": values}


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seeds": [result["seed"] for result in results],
        "host_output_bacc": stats(
            results, ("controls_host_output_bacc", "full")),
        "controls": {
            key: stats(results, ("controls_host_output_bacc", key))
            for key in ("reset", "host_only", "no_state", "shuffled", "random")
        },
        "max_control_bacc": {
            "mean": float(np.mean([
                max(result["controls_host_output_bacc"][key] for key in
                    ("reset", "host_only", "no_state", "shuffled", "random"))
                for result in results]))
        },
        "host_loss_with_memory": stats(
            results, ("host_future_latent_loss", "with_memory")),
        "host_loss_without_memory": stats(
            results, ("host_future_latent_loss", "without_memory")),
        "host_loss_ratio": stats(
            results, ("host_future_latent_loss", "ratio")),
        "geometry_pearson": stats(
            results, ("six_way_geometry", "pairwise_distance_correlation")),
        "geometry_rank": stats(
            results,
            ("six_way_geometry", "pairwise_distance_rank_correlation")),
        "cue_deletion_delta": stats(
            results,
            ("causal_group_deletion", "delta_host_loss_delete_cue")),
        "random_deletion_delta": stats(
            results,
            ("causal_group_deletion", "delta_host_loss_delete_random")),
        "ladder": {
            key: stats(results, ("diagnostic_ladder", key))
            for key in ("memory_token", "bottleneck", "decoded_signal",
                        "host_output")
        },
        "all_seeds_joint_passed": all(
            result["success_gate"]["joint_passed"] for result in results),
        "all_seeds_constraint_feasible": all(
            result["success_gate"]["host_constraint_feasible"]
            for result in results),
        "parameter_count": results[0]["trainable_parameter_count"],
        "parameter_overhead_fraction": (
            results[0]["trainable_parameter_count"]
            / results[0]["host_parameter_count"]),
        "latency_overhead_fraction": stats(
            results, ("overhead", "latency_overhead_fraction")),
    }


def previous_methods() -> list[dict[str, Any]]:
    return [
        {"name": "Dense residual", "bacc": 0.1556, "ratio": 0.2585 / 1.3400,
         "metric_note": "legacy cue-conditioned endpoint; not strict"},
        {"name": "MoE", "bacc": 0.3701, "ratio": 0.356193 / 0.009682},
        {"name": "Semantic bottleneck", "bacc": 0.7875,
         "ratio": 0.361792 / 1.339631,
         "metric_note": "legacy cue-conditioned target; not strict"},
        {"name": "AdaLN", "bacc": 0.8132, "ratio": 0.300911 / 0.009681},
        {"name": "Memory tokens", "bacc": 0.8194,
         "ratio": 0.131307 / 0.009664},
    ]


def plot_report(report: dict[str, Any]) -> list[str]:
    screen = report["screen"]
    best = report["best_three_seed"]
    methods = previous_methods()
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.axvspan(0.0, 1.10, color="#c8e6c9", alpha=0.55,
                 label="Feasible (ε≤10%)")
    for method in methods:
        marker = "x" if "metric_note" in method else "o"
        axis.scatter(method["ratio"], method["bacc"], marker=marker, s=65)
        axis.annotate(method["name"], (method["ratio"], method["bacc"]),
                      xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.scatter(best["host_loss_ratio"]["mean"],
                 best["host_output_bacc"]["mean"], marker="*", s=180,
                 color="#6a1b9a", label="Hybrid (strict)")
    axis.axhline(0.75, color="#c62828", linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("Host-loss ratio (memory / no memory; lower is better)")
    axis.set_ylabel("Host-output balanced accuracy (higher is better)")
    axis.set_title("Exposure–fidelity Pareto map (ideal upper-left)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    pareto = FIGURE.with_name(FIGURE.name + "_pareto")
    fig.tight_layout()
    fig.savefig(pareto.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(pareto.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for item in screen:
        history = item["history"]
        label = f"d{item['config']['code_dim']} ε={item['config']['epsilon']:.2f}"
        axes[0].plot([x["epoch"] for x in history],
                     [x["ratio"] for x in history], label=label)
        axes[1].plot([x["epoch"] for x in history],
                     [x["multiplier"] for x in history], label=label)
    axes[0].axhline(1.0, color="black", linestyle=":")
    axes[0].axhline(1.1, color="#c62828", linestyle="--")
    axes[0].set_title("Strict host-loss ratio by epoch")
    axes[0].set_ylabel("Train host-loss ratio")
    axes[1].set_title("Adaptive dual multiplier")
    axes[1].set_ylabel("Lagrange multiplier")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    constraint = FIGURE.with_name(FIGURE.name + "_constraint_training")
    fig.tight_layout()
    fig.savefig(constraint.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(constraint.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [
        str(pareto.with_suffix(".png").relative_to(ROOT)),
        str(constraint.with_suffix(".png").relative_to(ROOT)),
    ]


def write_markdown(report: dict[str, Any]) -> None:
    best = report["best_three_seed"]
    config = report["best_config"]
    controls = ", ".join(
        f"{key} {value['mean']:.3f}" for key, value in best["controls"].items())
    rows = []
    for method in report["comparison"]:
        note = method.get("metric_note", "")
        rows.append(
            f"| {method['name']} | {method['bacc']:.3f} | "
            f"{method['ratio']:.2f} | {note} |")
    rows.append(
        f"| **Hybrid** | **{best['host_output_bacc']['mean']:.3f}** | "
        f"**{best['host_loss_ratio']['mean']:.3f}** | strict next-latent |")
    text = f"""# Final Hybrid CEM–LeWM Exposure Interface

## Joint verdict

The selected configuration is **d={config['code_dim']}, ε={config['epsilon']:.2f}**
over seeds {best['seeds']}. The joint gate
**{'passed' if best['all_seeds_joint_passed'] else 'did not pass'}**. A run is
counted only when host-output BAcc ≥0.75, every control ≤0.217, and strict
next-latent loss respects its configured constraint.

- Host-output BAcc: {best['host_output_bacc']['mean']:.3f} ± {best['host_output_bacc']['std']:.3f}
- Controls: {controls}
- Strict host loss, memory / baseline:
  {best['host_loss_with_memory']['mean']:.6f} /
  {best['host_loss_without_memory']['mean']:.6f}
- Host-loss ratio: {best['host_loss_ratio']['mean']:.4f} ± {best['host_loss_ratio']['std']:.4f};
  all-seed feasible: {best['all_seeds_constraint_feasible']}
- Geometry Pearson / rank:
  {best['geometry_pearson']['mean']:.3f} /
  {best['geometry_rank']['mean']:.3f}
- Causal cue deletion / random deletion Δloss:
  {best['cue_deletion_delta']['mean']:.6f} /
  {best['random_deletion_delta']['mean']:.6f}

## Interface and training

The label-free path is CEM causal event store → six distinct retrieved memory
tokens → normalized semantic bottleneck → bounded query-conditioned decoder →
complete frozen official LeWM output. Same-base branch identity is inferred by
latent equality. A primal-dual augmented Lagrangian adapts its multiplier from
the relative constraint violation every minibatch; the JSON records per-epoch
violation magnitude/rate and final multipliers.

Diagnostic ladder (memory token → bottleneck → decoded signal → host):
{best['ladder']['memory_token']['mean']:.3f} →
{best['ladder']['bottleneck']['mean']:.3f} →
{best['ladder']['decoded_signal']['mean']:.3f} →
{best['ladder']['host_output']['mean']:.3f}.

## Overhead and integrity

- Trainable parameters: {best['parameter_count']:,}
  ({100 * best['parameter_overhead_fraction']:.3f}% of frozen host).
- Measured batch-latency overhead:
  {100 * best['latency_overhead_fraction']['mean']:.1f}%.
- Frozen digest: `{report['frozen_host_digest']}`; unchanged in all runs.
- Semantic labels used in training loss: false.

## Direct comparison

| Interface | Host BAcc | Host-loss ratio | Metric note |
|---|---:|---:|---|
{chr(10).join(rows)}

Legacy dense-residual and semantic-adapter ratios use their original
cue-conditioned target and are marked non-strict; they must not be interpreted
as strict feasibility evidence. The Pareto verdict is based on the strict
points for memory tokens, AdaLN, MoE, and this hybrid.

![Hybrid Pareto](assets/cem_lewm_hybrid_pareto.png)

![Constraint training](assets/cem_lewm_hybrid_constraint_training.png)
"""
    DOCUMENT.write_text(text)


def campaign(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    locked = load_locked_pusht_spec(
        str(DEFAULT_PUSHT_SPEC), str(DEFAULT_PUSHT_LOCK))
    host = load_official_pusht_checkpoint(
        resolve_pusht_path(locked["official_host"]["bundle_path"]), device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    digest = state_digest(host)
    spec, train, validation = make_data(args, host, device)
    if args.smoke:
        result = train_one(
            args, host, digest, spec, train, validation, code_dim=32,
            epsilon=0.10, seed=97, run_kind="smoke")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "smoke_report.json").write_text(json.dumps(result, indent=2))
        return {"smoke": result}
    screen, by_id = [], {}
    for code_dim in args.screen_dims:
        for epsilon in args.epsilons:
            result = train_one(
                args, host, digest, spec, train, validation, code_dim=code_dim,
                epsilon=epsilon, seed=0, run_kind="screen")
            screen.append(result)
            by_id[(code_dim, epsilon)] = result
    ranked = sorted(screen, key=lambda result: (
        result["success_gate"]["joint_passed"],
        result["success_gate"]["host_constraint_feasible"],
        result["controls_host_output_bacc"]["full"],
        result["host_future_latent_loss"]["constraint_margin"],
    ), reverse=True)
    chosen = ranked[0]
    code_dim, epsilon = (
        int(chosen["config"]["code_dim"]), float(chosen["config"]["epsilon"]))
    best_results = [by_id[(code_dim, epsilon)]]
    for seed in args.confirm_seeds:
        if seed == 0:
            continue
        best_results.append(train_one(
            args, host, digest, spec, train, validation, code_dim=code_dim,
            epsilon=epsilon, seed=seed, run_kind="confirm"))
    report = {
        "schema": "cem_lewm_hybrid_interface_report_v1",
        "task": TASK, "age": 15, "device": args.device,
        "screen": screen, "best_config": chosen["config"],
        "best_three_seed": aggregate(best_results),
        "selection_rule": (
            "joint pass, strict feasibility, host BAcc, constraint margin"),
        "comparison": previous_methods(),
        "frozen_host_digest": digest,
        "frozen_host_digest_unchanged_all_runs": all(
            result["frozen_host_digest_unchanged"]
            for result in screen + best_results),
        "labels_used_for_training_loss": False,
        "constraint_definition": (
            "strict next-latent MSE(memory) <= (1+epsilon) MSE(no_memory)"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report["figures"] = plot_report(report)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--task", default=TASK, choices=[TASK])
    parser.add_argument("--age", type=int, default=15, choices=[15])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-residual", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--host-sep-weight", type=float, default=2.0)
    parser.add_argument("--carrier-weight", type=float, default=3.0)
    parser.add_argument("--code-nce-weight", type=float, default=1.0)
    parser.add_argument("--geometry-weight", type=float, default=2.0)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--orthogonality-weight", type=float, default=0.01)
    parser.add_argument("--dual-init", type=float, default=1.0)
    parser.add_argument("--dual-lr", type=float, default=0.5)
    parser.add_argument("--dual-max", type=float, default=1000.0)
    parser.add_argument("--augmented-rho", type=float, default=10.0)
    parser.add_argument("--budget-weight", type=float, default=50.0)
    parser.add_argument("--budget-fraction", type=float, default=0.85)
    parser.add_argument("--restore-feasible", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10])
    parser.add_argument("--screen-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--confirm-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--counterfactual-cache",
                        default=str(ROOT / "outputs/lewm_pusht_counterfactual_cue_cache_v1"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda:1":
        raise ValueError("assigned device is cuda:1; GPU3 is forbidden")
    report = campaign(args)
    if args.smoke:
        print(json.dumps({
            "smoke": True,
            "joint": report["smoke"]["success_gate"]["joint_passed"],
            "ratio": report["smoke"]["host_future_latent_loss"]["ratio"],
        }, indent=2))
    else:
        print(json.dumps({
            "report": str((OUTPUT / "report.json").relative_to(ROOT)),
            "best_config": report["best_config"],
            "joint_passed": report["best_three_seed"]["all_seeds_joint_passed"],
            "host_output_bacc": report["best_three_seed"]["host_output_bacc"],
            "host_loss_ratio": report["best_three_seed"]["host_loss_ratio"],
        }, indent=2))


if __name__ == "__main__":
    main()
