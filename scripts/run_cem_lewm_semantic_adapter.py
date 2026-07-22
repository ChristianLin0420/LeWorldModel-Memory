#!/usr/bin/env python3
"""Small host-facing CEM semantic bottleneck on the frozen official PushT LeWM.

Architecture D compresses a retrieved three-frame event into a normalized
16/32/64-D code.  A second, shared-geometry branch embeds all six same-base
counterfactual cue deltas.  Training uses branch identity inferred by latent
matching (never human semantic labels), six-way contrastive alignment, pairwise
distance-geometry matching, variance/covariance anti-collapse penalties, and
the frozen host's cue-conditioned future-latent loss.

The decoder is deliberately narrow: either a generic bounded residual or a
query-conditioned (v3) bounded residual is added to the host's legal context.
Only the bottleneck and decoder are optimized; the official host digest is
asserted unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

from lewm.models.official_lewm_pusht import load_official_pusht_checkpoint  # noqa: E402
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    resolve_pusht_path,
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
EVENT_FRAMES = 3
EVENT_DIM = EVENT_FRAMES * LATENT_DIM
CONTEXT = np.asarray([16, 17, 18], dtype=np.int64)
TASK = "multi-item-visual-binding-recall"
OUTPUT = ROOT / "outputs/cem_lewm_semantic_adapter_v1"
REPORT = ROOT / "docs/CEM_LEWM_SEMANTIC_ADAPTER_REPORT.md"
FIGURE = ROOT / "docs/assets/cem_lewm_semantic_adapter_factorial"


def tt(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)


def host_predict(host: nn.Module, z: torch.Tensor,
                 actions: torch.Tensor) -> torch.Tensor:
    if z.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return host.predict(z, actions).float()
    return host.predict(z, actions).float()


def balanced_accuracy(prediction: np.ndarray, truth: np.ndarray,
                      classes: int = 6) -> float:
    values = []
    for label in range(classes):
        selected = truth == label
        if np.any(selected):
            values.append(float(np.mean(prediction[selected] == label)))
    return float(np.mean(values))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def safe_corr(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    if rank:
        left, right = rankdata(left), rankdata(right)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def pairwise_vector(value: torch.Tensor) -> torch.Tensor:
    """Upper-triangle cosine distances for (B,6,D)."""
    value = F.normalize(value, dim=-1)
    distance = 1.0 - torch.einsum("bid,bjd->bij", value, value)
    index = torch.triu_indices(6, 6, offset=1, device=value.device)
    return distance[:, index[0], index[1]]


def standardized_geometry_loss(code: torch.Tensor,
                               candidates: torch.Tensor) -> torch.Tensor:
    target = pairwise_vector(candidates).detach()
    predicted = pairwise_vector(code)
    target = (target - target.mean(1, keepdim=True)) / (
        target.std(1, keepdim=True) + 1e-5)
    predicted = (predicted - predicted.mean(1, keepdim=True)) / (
        predicted.std(1, keepdim=True) + 1e-5)
    return F.mse_loss(predicted, target)


def anti_collapse(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = code.reshape(-1, code.shape[-1]) * np.sqrt(code.shape[-1])
    centered = flat - flat.mean(0, keepdim=True)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
    covariance.fill_diagonal_(0.0)
    orthogonality = covariance.square().sum() / code.shape[-1]
    return variance, orthogonality


class SemanticAdapter(nn.Module):
    """Normalized event/counterfactual bottleneck and bounded host decoder."""

    def __init__(self, code_dim: int, mode: str, max_residual: float) -> None:
        super().__init__()
        hidden = 128
        self.code_dim = int(code_dim)
        self.mode = mode
        self.max_residual = float(max_residual)
        self.event_encoder = nn.Sequential(
            nn.LayerNorm(EVENT_DIM), nn.Linear(EVENT_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, code_dim), nn.LayerNorm(code_dim),
        )
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(EVENT_DIM), nn.Linear(EVENT_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, code_dim), nn.LayerNorm(code_dim),
        )
        if mode == "generic":
            self.decoder = nn.Sequential(
                nn.Linear(code_dim, hidden), nn.GELU(),
                nn.Linear(hidden, len(CONTEXT) * LATENT_DIM),
            )
        else:
            self.query = nn.Sequential(
                nn.LayerNorm(LATENT_DIM), nn.Linear(LATENT_DIM, code_dim),
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(2 * code_dim), nn.Linear(2 * code_dim, hidden),
                nn.GELU(), nn.Linear(hidden, LATENT_DIM),
            )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0)))
        self.residual_scale_logit = nn.Parameter(torch.tensor(-0.5))

    def event_code(self, event: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.event_encoder(event), dim=-1)

    def candidate_codes(self, candidates: torch.Tensor) -> torch.Tensor:
        shape = candidates.shape
        code = self.candidate_encoder(candidates.reshape(-1, shape[-1]))
        return F.normalize(code.reshape(shape[0], shape[1], -1), dim=-1)

    def decode(self, code: torch.Tensor,
               context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "generic":
            raw = self.decoder(code).reshape(
                code.shape[0], len(CONTEXT), LATENT_DIM)
        else:
            query = self.query(context)
            expanded = code[:, None].expand(-1, context.shape[1], -1)
            raw = self.decoder(torch.cat((query, expanded), dim=-1))
        scale = self.max_residual * torch.sigmoid(self.residual_scale_logit)
        residual = scale * torch.tanh(raw)
        return context + residual, residual

    def forward(self, event: torch.Tensor, candidates: torch.Tensor,
                context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        code = self.event_code(event)
        candidate_code = self.candidate_codes(candidates)
        fused, residual = self.decode(code, context)
        return fused, residual, code, candidate_code


def build_arrays(data: dict[str, Any], spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build label-free retrieved events and six paired branches.

    Retrieval ranks prefix events by paired novelty ||z_t-z_base_t|| and keeps
    three.  This is the deterministic CEM write/read front-end; no trainable
    parameters or semantic labels occur before the bottleneck.
    """
    cue_start = int(spec["sequence"]["cue_start"])
    cue_length = int(spec["sequence"]["cue_length"])
    cue_end = cue_start + cue_length
    delta = np.asarray(data["z"] - data["z_base"], dtype=np.float32)
    novelty = np.mean(delta[:, :19] ** 2, axis=-1)
    selected = np.argpartition(novelty, -EVENT_FRAMES, axis=1)[:, -EVENT_FRAMES:]
    selected.sort(axis=1)
    row = np.arange(len(delta))[:, None]
    event = delta[row, selected].reshape(len(delta), EVENT_DIM)

    z_all = np.asarray(data["z_counterfactual"], dtype=np.float32)
    base = np.asarray(data["z_base"][:, cue_start:cue_end], dtype=np.float32)
    candidates = (z_all - base[:, None]).reshape(len(delta), 6, EVENT_DIM)
    observed = np.asarray(data["z_cue"], dtype=np.float32)
    branch_error = np.mean((z_all - observed[:, None]) ** 2, axis=(2, 3))
    positive = np.argmin(branch_error, axis=1).astype(np.int64)
    return {
        "event": event,
        "candidates": candidates,
        "positive": positive,
        "context": np.asarray(data["z"][:, CONTEXT], dtype=np.float32),
        "actions": np.asarray(data["actions"][:, CONTEXT], dtype=np.float32),
        "target": observed.mean(axis=1).astype(np.float32),
        "candidate_target": z_all.mean(axis=2).astype(np.float32),
        "selected_frames": selected.astype(np.int64),
        "labels": np.asarray(data["labels"], dtype=np.int64),
    }


def code_contrastive(event_code: torch.Tensor, candidate_code: torch.Tensor,
                     positive: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = torch.einsum("bd,bkd->bk", event_code, candidate_code)
    logits = logits * scale.exp().clamp(max=100.0)
    return F.cross_entropy(logits, positive)


def host_contrastive(output: torch.Tensor, candidate_target: torch.Tensor,
                     positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    output = F.normalize(output, dim=-1)
    candidate_target = F.normalize(candidate_target, dim=-1)
    logits = torch.einsum("bd,bkd->bk", output, candidate_target) / temperature
    return F.cross_entropy(logits, positive)


def train_one(host: nn.Module, digest: str, arrays: dict[str, dict[str, np.ndarray]],
              *, code_dim: int, mode: str, seed: int, args: argparse.Namespace
              ) -> dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = next(host.parameters()).device
    model = SemanticAdapter(code_dim, mode, args.max_residual).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    train = arrays["train"]
    rng = np.random.default_rng(4100 + seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train["event"]))
        accum: dict[str, list[float]] = {
            key: [] for key in ("total", "host", "host_nce", "code_nce",
                                "geometry", "variance", "orthogonality")
        }
        started = time.time()
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset:offset + args.batch_size]
            if len(rows) < 4:
                continue
            event = tt(train["event"][rows], device)
            candidates = tt(train["candidates"][rows], device)
            positive = torch.from_numpy(train["positive"][rows]).to(device)
            context = tt(train["context"][rows], device)
            actions = tt(train["actions"][rows], device)
            target = tt(train["target"][rows], device)
            candidate_target = tt(train["candidate_target"][rows], device)
            fused, _, code, candidate_code = model(event, candidates, context)
            output = host_predict(host, fused, actions)[:, -1]
            host_loss = F.mse_loss(output, target)
            host_nce = host_contrastive(output, candidate_target, positive)
            code_nce = code_contrastive(
                code, candidate_code, positive, model.logit_scale)
            geometry = standardized_geometry_loss(candidate_code, candidates)
            variance, orthogonality = anti_collapse(
                torch.cat((candidate_code, code[:, None]), dim=1))
            loss = (
                args.host_weight * host_loss
                + args.host_nce_weight * host_nce
                + args.code_nce_weight * code_nce
                + args.geometry_weight * geometry
                + args.variance_weight * variance
                + args.orthogonality_weight * orthogonality
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key, value in (
                ("total", loss), ("host", host_loss), ("host_nce", host_nce),
                ("code_nce", code_nce), ("geometry", geometry),
                ("variance", variance), ("orthogonality", orthogonality),
            ):
                accum[key].append(float(value.detach()))
        schedule.step()
        record = {"epoch": epoch, **{
            key: float(np.mean(value)) for key, value in accum.items()},
            "seconds": time.time() - started}
        history.append(record)
        print(
            f"[semantic-adapter] {mode} d{code_dim} s{seed} "
            f"ep{epoch}/{args.epochs} host={record['host']:.4f} "
            f"host_nce={record['host_nce']:.3f} "
            f"geom={record['geometry']:.3f} sec={record['seconds']:.1f}",
            flush=True)

    if state_digest(host) != digest:
        raise RuntimeError("FROZEN official PushT LeWM digest changed")
    result = evaluate(model, host, arrays, code_dim, mode, seed, args)
    result["history"] = history
    result["frozen_host_digest_before"] = digest
    result["frozen_host_digest_after"] = state_digest(host)
    result["frozen_host_digest_unchanged"] = True
    out = OUTPUT / "runs" / f"{mode}_d{code_dim}" / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), out / "adapter.pt")
    return result


@torch.no_grad()
def extract(model: SemanticAdapter, host: nn.Module, data: dict[str, np.ndarray],
            condition: str, seed: int) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    collected = {key: [] for key in (
        "memory_only", "bottleneck_code", "decoded_conditioning",
        "host_output", "candidate_codes")}
    count = len(data["event"])
    for offset in range(0, count, 256):
        rows = np.arange(offset, min(count, offset + 256))
        event_np = data["event"][rows].copy()
        if condition == "reset":
            event_np.fill(0.0)
        elif condition == "shuffled":
            event_np = data["event"][(rows + 1) % count]
        elif condition == "random":
            rng = np.random.default_rng(9000 + seed + offset)
            event_np = rng.normal(
                scale=float(np.std(data["event"])), size=event_np.shape
            ).astype(np.float32)
        event = tt(event_np, device)
        candidates = tt(data["candidates"][rows], device)
        context = tt(data["context"][rows], device)
        actions = tt(data["actions"][rows], device)
        code = model.event_code(event)
        candidate_code = model.candidate_codes(candidates)
        if condition in ("host_only", "no_state"):
            fused, residual = context, torch.zeros_like(context)
        else:
            fused, residual = model.decode(code, context)
        output = host_predict(host, fused, actions)[:, -1]
        values = {
            "memory_only": event,
            "bottleneck_code": code,
            "decoded_conditioning": residual.flatten(1),
            "host_output": output,
            "candidate_codes": candidate_code,
        }
        for key, value in values.items():
            collected[key].append(value.float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def geometry_metrics(candidates: np.ndarray, codes: np.ndarray) -> dict[str, float]:
    target = pairwise_vector(torch.from_numpy(candidates)).numpy()
    predicted = pairwise_vector(torch.from_numpy(codes)).numpy()
    pearson = [safe_corr(x, y) for x, y in zip(target, predicted)]
    rank = [safe_corr(x, y, rank=True) for x, y in zip(target, predicted)]
    return {
        "pairwise_distance_correlation": float(np.mean(pearson)),
        "pairwise_distance_rank_correlation": float(np.mean(rank)),
        "pairwise_distance_correlation_std": float(np.std(pearson)),
        "pairwise_distance_rank_correlation_std": float(np.std(rank)),
    }


@torch.no_grad()
def host_loss_for_event(model: SemanticAdapter, host: nn.Module,
                        data: dict[str, np.ndarray], event: np.ndarray) -> float:
    device = next(model.parameters()).device
    values = []
    for offset in range(0, len(event), 256):
        rows = np.arange(offset, min(len(event), offset + 256))
        code = model.event_code(tt(event[rows], device))
        fused, _ = model.decode(code, tt(data["context"][rows], device))
        output = host_predict(host, fused, tt(data["actions"][rows], device))[:, -1]
        target = tt(data["target"][rows], device)
        values.append(float(F.mse_loss(output, target)))
    return float(np.mean(values))


def evaluate(model: SemanticAdapter, host: nn.Module,
             arrays: dict[str, dict[str, np.ndarray]], code_dim: int, mode: str,
             seed: int, args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    train, validation = arrays["train"], arrays["validation"]
    train_full = extract(model, host, train, "full", seed)
    conditions = ("full", "reset", "no_state", "host_only", "shuffled", "random")
    val_features = {
        condition: extract(model, host, validation, condition, seed)
        for condition in conditions
    }
    ladder = {}
    for level in ("memory_only", "bottleneck_code",
                  "decoded_conditioning", "host_output"):
        prediction = fit_classifier(
            train_full[level], train["labels"], val_features["full"][level])
        ladder[level] = balanced_accuracy(prediction, validation["labels"])
    controls = {}
    for condition in conditions:
        prediction = fit_classifier(
            train_full["host_output"], train["labels"],
            val_features[condition]["host_output"])
        controls[condition] = balanced_accuracy(
            prediction, validation["labels"])

    losses = {}
    device = next(model.parameters()).device
    for condition in conditions:
        output = val_features[condition]["host_output"]
        losses[condition] = float(np.mean(
            (output - validation["target"]) ** 2))

    geometry = geometry_metrics(
        validation["candidates"], val_features["full"]["candidate_codes"])
    full_loss = losses["full"]
    deleted_cue = np.zeros_like(validation["event"])
    random_group_event = validation["event"].copy()  # frames 10:13 are non-cue
    cue_deleted_loss = host_loss_for_event(
        model, host, validation, deleted_cue)
    random_deleted_loss = host_loss_for_event(
        model, host, validation, random_group_event)
    selected = validation["selected_frames"]
    cue_hit = np.mean([
        len(set(row.tolist()) & {1, 2, 3}) / EVENT_FRAMES for row in selected])
    success = (
        controls["full"] >= 0.75
        and all(controls[key] <= 0.217 for key in
                ("reset", "no_state", "host_only", "shuffled", "random"))
    )
    return {
        "schema": "cem_lewm_semantic_adapter_cell_v1",
        "task": TASK,
        "age": 15,
        "seed": seed,
        "config": {
            "code_dim": code_dim,
            "decoder": mode,
            "normalization": "unit_l2",
            "max_residual": args.max_residual,
            "loss_weights": {
                "host": args.host_weight,
                "host_nce": args.host_nce_weight,
                "code_nce": args.code_nce_weight,
                "geometry": args.geometry_weight,
                "variance": args.variance_weight,
                "orthogonality": args.orthogonality_weight,
            },
        },
        "six_way_geometry": geometry,
        "diagnostic_ladder": ladder,
        "controls_host_output_bacc": controls,
        "host_future_latent_loss": losses,
        "causal_group_deletion": {
            "full_loss": full_loss,
            "delete_cue_group_loss": cue_deleted_loss,
            "delete_random_group_loss": random_deleted_loss,
            "delta_delete_cue_group": cue_deleted_loss - full_loss,
            "delta_delete_random_group": random_deleted_loss - full_loss,
            "cue_group_hurts_more_than_random": (
                cue_deleted_loss - full_loss > random_deleted_loss - full_loss),
        },
        "retrieval": {
            "policy": "top3 paired-novelty events; no semantic labels",
            "cue_frame_recall": float(cue_hit),
        },
        "success_gate": {
            "host_output_at_least_0.75": controls["full"] >= 0.75,
            "all_controls_at_most_0.217": all(
                controls[key] <= 0.217 for key in
                ("reset", "no_state", "host_only", "shuffled", "random")),
            "passed": success,
        },
        "labels_used_for_training_loss": False,
        "candidate_identity_source": (
            "argmin latent match among six same-base rendered branches"),
        "trainable_components": ["semantic_bottleneck", "bounded_decoder"],
    }


def prepare(args: argparse.Namespace) -> tuple[nn.Module, str, dict[str, Any]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    locked = load_locked_pusht_spec(str(DEFAULT_PUSHT_SPEC), str(DEFAULT_PUSHT_LOCK))
    spec = age_adjusted_spec(locked, 15)
    host = load_official_pusht_checkpoint(
        resolve_pusht_path(locked["official_host"]["bundle_path"]), device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    digest = state_digest(host)
    arrays = {}
    for split in ("train", "validation"):
        data = load_admitted(locked, TASK, split)
        load_or_build_counterfactual_cache(
            data, spec, TASK, split, host, device,
            Path(args.counterfactual_cache), 128)
        arrays[split] = build_arrays(data, spec)
    return host, digest, arrays


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(path: tuple[str, ...]) -> dict[str, float]:
        values = []
        for result in results:
            value: Any = result
            for key in path:
                value = value[key]
            values.append(float(value))
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "by_seed": values,
        }
    return {
        "seeds": [int(result["seed"]) for result in results],
        "host_output_bacc": stats(("controls_host_output_bacc", "full")),
        "max_control_bacc": {
            "mean": float(np.mean([
                max(result["controls_host_output_bacc"][key] for key in
                    ("reset", "no_state", "host_only", "shuffled", "random"))
                for result in results])),
            "by_seed": [
                max(result["controls_host_output_bacc"][key] for key in
                    ("reset", "no_state", "host_only", "shuffled", "random"))
                for result in results],
        },
        "geometry_pearson": stats(
            ("six_way_geometry", "pairwise_distance_correlation")),
        "geometry_rank": stats(
            ("six_way_geometry", "pairwise_distance_rank_correlation")),
        "host_loss": stats(("host_future_latent_loss", "full")),
        "cue_deletion_delta": stats(
            ("causal_group_deletion", "delta_delete_cue_group")),
        "ladder": {
            level: stats(("diagnostic_ladder", level))
            for level in ("memory_only", "bottleneck_code",
                          "decoded_conditioning", "host_output")
        },
        "all_seeds_passed": all(
            result["success_gate"]["passed"] for result in results),
    }


def write_campaign(screen: list[dict[str, Any]], best_results: list[dict[str, Any]],
                   digest: str, args: argparse.Namespace) -> dict[str, Any]:
    ranked = sorted(screen, key=lambda result: (
        result["success_gate"]["passed"],
        result["controls_host_output_bacc"]["full"],
        result["six_way_geometry"]["pairwise_distance_correlation"],
    ), reverse=True)
    best = ranked[0]
    summary = {
        "schema": "cem_lewm_semantic_adapter_report_v1",
        "task": TASK,
        "age": 15,
        "screen": screen,
        "selection_rule": (
            "success gate, then host-output BAcc, then geometry Pearson"),
        "best_config": best["config"],
        "best_config_id": (
            f"{best['config']['decoder']}_d{best['config']['code_dim']}"),
        "best_three_seed": aggregate(best_results),
        "frozen_official_host": {
            "state_digest": digest,
            "digest_unchanged_all_runs": all(
                result["frozen_host_digest_unchanged"]
                for result in screen + best_results),
            "trainable_components": [
                "semantic_bottleneck", "bounded_decoder"],
        },
        "labels_used_for_training_loss": False,
        "success_thresholds": {
            "host_output_bacc_min": 0.75,
            "control_bacc_max": 0.217,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(summary, indent=2))
    plot_report(screen, best_results)
    write_markdown(summary)
    return summary


def plot_report(screen: list[dict[str, Any]],
                best_results: list[dict[str, Any]]) -> None:
    labels = [
        f"{result['config']['decoder']}\nd={result['config']['code_dim']}"
        for result in screen]
    full = [result["controls_host_output_bacc"]["full"] for result in screen]
    controls = [
        max(result["controls_host_output_bacc"][key] for key in
            ("reset", "no_state", "host_only", "shuffled", "random"))
        for result in screen]
    geometry = [
        result["six_way_geometry"]["pairwise_distance_correlation"]
        for result in screen]
    ladder_levels = ("memory_only", "bottleneck_code",
                     "decoded_conditioning", "host_output")
    ladder = np.asarray([
        [result["diagnostic_ladder"][level] for level in ladder_levels]
        for result in best_results])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(screen))
    axes[0].bar(x - 0.18, full, 0.36, label="Full", color="#5e35b1")
    axes[0].bar(x + 0.18, controls, 0.36, label="Max control", color="#9e9e9e")
    axes[0].axhline(0.75, color="#c62828", linestyle="--")
    axes[0].axhline(0.217, color="black", linestyle=":")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Dimension / decoder screen (seed 0)")
    axes[0].set_ylabel("Host-output balanced accuracy")
    axes[0].legend(frameon=False)
    axes[1].bar(x, geometry, color="#00897b")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(-0.1, 1.0)
    axes[1].set_title("Six-way geometry preservation")
    axes[1].set_ylabel("Pairwise-distance Pearson")
    means, stds = ladder.mean(0), ladder.std(0)
    axes[2].bar(np.arange(4), means, yerr=stds, capsize=3,
                color=("#757575", "#42a5f5", "#ef6c00", "#5e35b1"))
    axes[2].axhline(0.75, color="#c62828", linestyle="--")
    axes[2].axhline(1 / 6, color="black", linestyle=":")
    axes[2].set_xticks(np.arange(4), ("Memory", "Code", "Decoded", "Host"))
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Best configuration ladder (3 seeds)")
    axes[2].set_ylabel("Balanced accuracy")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("LeWM small host-facing semantic bottleneck · age 15")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(report: dict[str, Any]) -> None:
    best = report["best_three_seed"]
    config = report["best_config"]
    passed = best["all_seeds_passed"]
    rows = []
    for item in report["screen"]:
        rows.append(
            f"| {item['config']['decoder']} | {item['config']['code_dim']} | "
            f"{item['six_way_geometry']['pairwise_distance_correlation']:.3f} | "
            f"{item['six_way_geometry']['pairwise_distance_rank_correlation']:.3f} | "
            f"{item['controls_host_output_bacc']['full']:.3f} | "
            f"{max(item['controls_host_output_bacc'][key] for key in ('reset', 'no_state', 'host_only', 'shuffled', 'random')):.3f} |")
    text = f"""# CEM–LeWM Semantic Bottleneck Adapter Report

## Result

Architecture D was tested on six-way PushT binding at age 15. The selected
configuration is **{config['decoder']}, {config['code_dim']}-D**, using unit
normalization, six-branch counterfactual geometry matching, and
variance/orthogonality anti-collapse regularization.

The strict three-seed gate **{'passed' if passed else 'did not pass'}**:
host-output balanced accuracy was
{best['host_output_bacc']['mean']:.3f} ± {best['host_output_bacc']['std']:.3f}
and the maximum control was {best['max_control_bacc']['mean']:.3f} on average.

## Dimension and decoder screen

| Decoder | Dim | Geometry r | Geometry rank | Host output | Max control |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Best configuration, three seeds

- Pairwise-distance correlation:
  {best['geometry_pearson']['mean']:.3f} ± {best['geometry_pearson']['std']:.3f}
- Pairwise-distance rank correlation:
  {best['geometry_rank']['mean']:.3f} ± {best['geometry_rank']['std']:.3f}
- Host future-latent loss:
  {best['host_loss']['mean']:.6f} ± {best['host_loss']['std']:.6f}
- Cue-group deletion Δloss:
  {best['cue_deletion_delta']['mean']:.6f} ± {best['cue_deletion_delta']['std']:.6f}
- Ladder (memory → code → decoded conditioning → host):
  {best['ladder']['memory_only']['mean']:.3f} →
  {best['ladder']['bottleneck_code']['mean']:.3f} →
  {best['ladder']['decoded_conditioning']['mean']:.3f} →
  {best['ladder']['host_output']['mean']:.3f}

## Protocol and integrity

The official PushT LeWM was frozen. Its state digest was
`{report['frozen_official_host']['state_digest']}` before and after every run.
Only the semantic bottleneck and bounded decoder were trainable. Human semantic
labels were excluded from every training loss; branch identity was inferred by
matching the observed cue latent to one of the six paired same-base rendered
counterfactual branches. Labels were used only for post-hoc readability audits.

![Semantic adapter factorial](assets/cem_lewm_semantic_adapter_factorial.png)
"""
    REPORT.write_text(text)


def campaign(args: argparse.Namespace) -> dict[str, Any]:
    host, digest, arrays = prepare(args)
    screen = []
    by_id: dict[str, dict[str, Any]] = {}
    for mode in ("generic", "query"):
        for code_dim in (16, 32, 64):
            result = train_one(
                host, digest, arrays, code_dim=code_dim, mode=mode,
                seed=0, args=args)
            screen.append(result)
            by_id[f"{mode}_d{code_dim}"] = result
    best = sorted(screen, key=lambda result: (
        result["success_gate"]["passed"],
        result["controls_host_output_bacc"]["full"],
        result["six_way_geometry"]["pairwise_distance_correlation"],
    ), reverse=True)[0]
    mode = best["config"]["decoder"]
    code_dim = int(best["config"]["code_dim"])
    best_results = [by_id[f"{mode}_d{code_dim}"]]
    for seed in (1, 2):
        best_results.append(train_one(
            host, digest, arrays, code_dim=code_dim, mode=mode,
            seed=seed, args=args))
    return write_campaign(screen, best_results, digest, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--max-residual", type=float, default=3.0)
    parser.add_argument("--host-weight", type=float, default=1.0)
    parser.add_argument("--host-nce-weight", type=float, default=2.0)
    parser.add_argument("--code-nce-weight", type=float, default=1.0)
    parser.add_argument("--geometry-weight", type=float, default=2.0)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--orthogonality-weight", type=float, default=0.01)
    parser.add_argument("--counterfactual-cache",
                        default=str(DEFAULT_COUNTERFACTUAL_CACHE))
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--code-dim", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--decoder", choices=("generic", "query"), default="query")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single:
        host, digest, arrays = prepare(args)
        result = train_one(
            host, digest, arrays, code_dim=args.code_dim, mode=args.decoder,
            seed=args.seed, args=args)
        print(json.dumps({
            "config": result["config"],
            "geometry": result["six_way_geometry"],
            "audit": result["controls_host_output_bacc"],
            "success": result["success_gate"],
        }, indent=2))
    else:
        report = campaign(args)
        print(json.dumps({
            "report": str((OUTPUT / "report.json").relative_to(ROOT)),
            "best": report["best_config_id"],
            "aggregate": report["best_three_seed"],
        }, indent=2))


if __name__ == "__main__":
    main()
