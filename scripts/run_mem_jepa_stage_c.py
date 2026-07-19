#!/usr/bin/env python3
"""Stage-C label-free host-exposure test for Masked-Evidence JEPA memory.

Training uses no semantic class labels.  The model observes the legal stream,
predicts a compact old-evidence target, and writes a residual into the frozen
DINO-WM PointMaze host context.  Supervision is positive-first masked-evidence
matching: the positive target is the hidden cue evidence from the same stream;
the negatives are the other counterfactual cue variants from the same base
trajectory.  Class labels are used only for the final audit readout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.run_mem_jepa_stage_b import (  # noqa: E402
    DEFAULT_CONFIG,
    FeatureBank,
    FrozenPointMazeHost,
    atomic_json,
    batch_arrays,
    classification_record,
    compact_pyramid_pool,
    endpoint_frame,
    fit_classifier,
    load_config,
    no_state_arrays,
    predictor_context_for_endpoint,
    require,
    resolve,
    set_determinism,
    sha256_file,
)


DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_stage_c"
ROI_ROWS = tuple(range(0, 6))
ROI_COLS = tuple(range(0, 6))
TARGET_DIM = 5 * 384


def roi_pool_torch(patches: torch.Tensor) -> torch.Tensor:
    """Pool the cue-card ROI into 1x1 + 2x2 DINO feature cells."""

    require(patches.ndim >= 3 and tuple(patches.shape[-2:]) == (196, 384),
            "patch tensor must end in 196x384")
    grid = patches.reshape(*patches.shape[:-2], 14, 14, 384)
    roi = grid[..., ROI_ROWS, :, :][..., ROI_COLS, :]
    cells = [roi.mean(dim=(-3, -2))]
    splits = ((0, 3), (3, 6))
    for row_start, row_stop in splits:
        for col_start, col_stop in splits:
            cells.append(roi[..., row_start:row_stop, col_start:col_stop, :].mean(
                dim=(-3, -2)))
    return torch.cat(cells, dim=-1)


class MemJepaLabelFreeAdapter(nn.Module):
    """Compact slot memory plus label-free evidence decoder."""

    def __init__(self, *, dim: int, slots: int, heads: int,
                 max_frames: int = 20, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.slots = int(slots)
        self.visual_proj = nn.Linear(384, dim)
        self.action_proj = nn.Linear(10, dim)
        self.time = nn.Embedding(max_frames, dim)
        self.patch = nn.Parameter(torch.empty(196, dim))
        self.assign = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, slots),
        )
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.residual = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 384),
        )
        self.evidence_decoder = nn.Sequential(
            nn.LayerNorm(slots * dim),
            nn.Linear(slots * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, TARGET_DIM),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        nn.init.normal_(self.patch, std=0.02)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def _tokens(self, visual: torch.Tensor, actions: torch.Tensor,
                times: torch.Tensor) -> torch.Tensor:
        batch, steps, patches, _ = visual.shape
        require((patches, actions.shape[:2]) == (196, (batch, steps)),
                "visual/action token shapes are inconsistent")
        return (
            self.visual_proj(visual)
            + self.time(times.to(visual.device))[None, :, None, :]
            + self.patch[None, None, :, :]
            + self.action_proj(actions)[:, :, None, :]
        )

    def encode_memory(self, visual: torch.Tensor, actions: torch.Tensor,
                      times: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(visual, actions, times).reshape(
            visual.shape[0], -1, self.dim)
        weights = torch.softmax(self.assign(tokens).transpose(1, 2), dim=-1)
        return weights @ tokens

    def inject(self, prefix_visual: torch.Tensor, prefix_actions: torch.Tensor,
               prefix_times: torch.Tensor, context_visual: torch.Tensor,
               context_actions: torch.Tensor,
               context_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.encode_memory(prefix_visual, prefix_actions, prefix_times)
        queries = self._tokens(context_visual, context_actions, context_times).reshape(
            context_visual.shape[0], -1, self.dim)
        read, _ = self.cross(queries, slots, slots, need_weights=False)
        residual = self.residual(queries + read).reshape_as(context_visual)
        return context_visual + self.residual_scale * residual, slots

    def decode_evidence(self, slots: torch.Tensor) -> torch.Tensor:
        return self.evidence_decoder(slots.reshape(slots.shape[0], -1))


def cue_candidates(bank: FeatureBank, expanded: np.ndarray,
                   device: torch.device, *,
                   negative_mode: str = "within_base",
                   shuffle_targets: bool = False) -> torch.Tensor:
    """Return positive-first cue ROI targets with within-base negatives."""

    bases, variant = bank.decode_expanded(expanded)
    cue = torch.from_numpy(np.asarray(bank.cue_visual[bases], dtype=np.float32)).to(
        device)
    pooled = roi_pool_torch(cue.reshape(-1, 196, 384)).reshape(
        len(expanded), 4, 3, TARGET_DIM).mean(dim=2)
    if negative_mode == "within_base":
        order = []
        for value in variant:
            positives_first = [int(value)] + [
                idx for idx in range(4) if idx != int(value)]
            order.append(positives_first)
        order_tensor = torch.tensor(order, device=device, dtype=torch.long)
        batch_rows = torch.arange(len(expanded), device=device)[:, None]
        candidates = pooled[batch_rows, order_tensor]
    elif negative_mode == "batch":
        variant_tensor = torch.as_tensor(variant, device=device, dtype=torch.long)
        positive = pooled[torch.arange(len(expanded), device=device), variant_tensor]
        candidates = torch.stack(
            [positive] + [positive.roll(shift, dims=0) for shift in (1, 2, 3)],
            dim=1,
        )
    else:
        raise ValueError(f"unknown negative mode: {negative_mode}")
    if shuffle_targets:
        candidates = candidates[torch.randperm(len(candidates), device=device)]
    return candidates


def contrastive_loss(query: torch.Tensor, candidates: torch.Tensor,
                     temperature: float) -> tuple[torch.Tensor, float]:
    query = F.normalize(query, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    logits = torch.einsum("bd,bkd->bk", query, candidates) / float(temperature)
    target = torch.zeros(len(query), device=query.device, dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    accuracy = float((logits.argmax(dim=1) == target).float().mean().detach().cpu())
    return loss, accuracy


def positive_cosine_loss(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    positive = F.normalize(candidates[:, 0], dim=-1)
    return 1.0 - torch.sum(query * positive, dim=-1).mean()


def train_one_age(model: MemJepaLabelFreeAdapter, host: FrozenPointMazeHost,
                  bank: FeatureBank, *, age: int, seed: int, epochs: int,
                  batch_size: int, lr: float, weight_decay: float,
                  temperature: float, variant: str,
                  output_dir: Path) -> list[dict[str, Any]]:
    train_indices = bank.expanded_indices("train").copy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs))
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    csv_path = output_dir / f"age_{age}_history.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch", "loss", "host_loss", "context_loss", "memory_loss",
                "host_match", "context_match", "memory_match", "host_cos",
                "context_cos", "memory_cos", "residual_l2", "lr", "seconds",
            ],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            rng.shuffle(train_indices)
            losses, host_losses, context_losses, memory_losses, regs = [], [], [], [], []
            host_cos_values, context_cos_values, memory_cos_values = [], [], []
            host_matches, context_matches, memory_matches = [], [], []
            model.train()
            for offset in range(0, len(train_indices), batch_size):
                expanded = train_indices[offset:offset + batch_size]
                batch = batch_arrays(bank, expanded, age, "full", host.device)
                candidates = cue_candidates(
                    bank, expanded, host.device,
                    negative_mode="batch" if variant == "batch_negatives"
                    else "within_base",
                    shuffle_targets=variant == "shuffle_targets",
                )
                fused, slots = model.inject(
                    batch["prefix_visual"], batch["prefix_actions"],
                    batch["prefix_times"], batch["context_visual"],
                    batch["context_actions"], batch["context_times"])
                predicted = host.predict(
                    fused, batch["proprio_context"],
                    batch["context_actions"])[:, -1, :, :384]
                host_query = roi_pool_torch(predicted)
                context_query = roi_pool_torch(fused[:, -1])
                memory_query = model.decode_evidence(slots)
                host_loss, host_match = contrastive_loss(
                    host_query, candidates, temperature)
                context_loss, context_match = contrastive_loss(
                    context_query, candidates, temperature)
                memory_loss, memory_match = contrastive_loss(
                    memory_query, candidates, temperature)
                host_cos = positive_cosine_loss(host_query, candidates)
                context_cos = positive_cosine_loss(context_query, candidates)
                memory_cos = positive_cosine_loss(memory_query, candidates)
                residual_l2 = torch.mean(torch.square(fused - batch["context_visual"]))
                terms = [0.5 * memory_loss, 0.25 * memory_cos]
                if variant != "no_host":
                    terms.extend([host_loss, 0.5 * host_cos])
                if variant != "no_context":
                    terms.extend([context_loss, context_cos])
                terms.append(1.0e-4 * residual_l2)
                loss = sum(terms)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                host_losses.append(float(host_loss.detach().cpu()))
                context_losses.append(float(context_loss.detach().cpu()))
                memory_losses.append(float(memory_loss.detach().cpu()))
                host_cos_values.append(float(host_cos.detach().cpu()))
                context_cos_values.append(float(context_cos.detach().cpu()))
                memory_cos_values.append(float(memory_cos.detach().cpu()))
                regs.append(float(residual_l2.detach().cpu()))
                host_matches.append(host_match)
                context_matches.append(context_match)
                memory_matches.append(memory_match)
            scheduler.step()
            record = {
                "epoch": int(epoch),
                "loss": float(np.mean(losses)),
                "host_loss": float(np.mean(host_losses)),
                "context_loss": float(np.mean(context_losses)),
                "memory_loss": float(np.mean(memory_losses)),
                "host_match": float(np.mean(host_matches)),
                "context_match": float(np.mean(context_matches)),
                "memory_match": float(np.mean(memory_matches)),
                "host_cos": float(np.mean(host_cos_values)),
                "context_cos": float(np.mean(context_cos_values)),
                "memory_cos": float(np.mean(memory_cos_values)),
                "residual_l2": float(np.mean(regs)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": float(time.time() - started),
            }
            history.append(record)
            writer.writerow(record)
            stream.flush()
            print(
                f"[mem-jepa-stage-c] age={age} epoch={epoch}/{epochs} "
                f"loss={record['loss']:.4f} host_match={record['host_match']:.3f} "
                f"ctx_match={record['context_match']:.3f} "
                f"mem_match={record['memory_match']:.3f} sec={record['seconds']:.1f}",
                flush=True,
            )
    return history


@torch.no_grad()
def collect_features(model: MemJepaLabelFreeAdapter, host: FrozenPointMazeHost,
                     bank: FeatureBank, *, split: str, age: int,
                     condition: str, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    indices = bank.expanded_indices(split)
    features, retrieval, truth = [], [], []
    for offset in range(0, len(indices), batch_size):
        expanded = indices[offset:offset + batch_size]
        if condition == "no_state":
            batch = no_state_arrays(bank, expanded, age, host.device)
            predicted = host.predict(
                batch["context_visual"], batch["proprio_context"],
                batch["context_actions"])[:, -1, :, :384]
        else:
            batch = batch_arrays(bank, expanded, age, condition, host.device)
            fused, _ = model.inject(
                batch["prefix_visual"], batch["prefix_actions"],
                batch["prefix_times"], batch["context_visual"],
                batch["context_actions"], batch["context_times"])
            predicted = host.predict(
                fused, batch["proprio_context"],
                batch["context_actions"])[:, -1, :, :384]
        candidates = cue_candidates(bank, expanded, host.device)
        host_query = roi_pool_torch(predicted)
        scores = torch.einsum(
            "bd,bkd->bk",
            F.normalize(host_query, dim=-1),
            F.normalize(candidates, dim=-1),
        )
        retrieval.append(scores.argmax(dim=1).cpu().numpy())
        truth.append(np.zeros(len(expanded), dtype=np.int64))
        features.append(compact_pyramid_pool(predicted.float().cpu().numpy()))
    return {
        "features": np.concatenate(features),
        "retrieval_prediction": np.concatenate(retrieval).astype(np.int64),
        "retrieval_truth": np.concatenate(truth).astype(np.int64),
    }


def evaluate_one_age(model: MemJepaLabelFreeAdapter, host: FrozenPointMazeHost,
                     bank: FeatureBank, *, age: int,
                     batch_size: int) -> dict[str, Any]:
    train_full = collect_features(
        model, host, bank, split="train", age=age, condition="full",
        batch_size=batch_size)
    full = collect_features(
        model, host, bank, split="validation", age=age, condition="full",
        batch_size=batch_size)
    reset = collect_features(
        model, host, bank, split="validation", age=age, condition="reset",
        batch_size=batch_size)
    no_state = collect_features(
        model, host, bank, split="validation", age=age, condition="no_state",
        batch_size=batch_size)
    train_y = np.tile(np.arange(4, dtype=np.int64), len(bank.base_indices("train")))
    validation_y = np.tile(
        np.arange(4, dtype=np.int64), len(bank.base_indices("validation")))
    records = {}
    for name, pack in {"full": full, "reset": reset,
                       "no_state": no_state}.items():
        prediction = fit_classifier(
            train_full["features"], train_y, pack["features"])
        retrieval_acc = float(np.mean(
            pack["retrieval_prediction"] == pack["retrieval_truth"]))
        records[name] = classification_record(prediction, validation_y)
        records[name]["candidate_retrieval_accuracy"] = retrieval_acc
    gate = {
        "full_minimum": 0.75,
        "control_maximum": 0.40,
        "passed": bool(
            records["full"]["balanced_accuracy"] >= 0.75
            and records["reset"]["balanced_accuracy"] <= 0.40
            and records["no_state"]["balanced_accuracy"] <= 0.40
        ),
    }
    return {"records": records, "gate": gate}


def run_age(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(resolve(args.config))
    set_determinism(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bank = FeatureBank(cfg)
    host = FrozenPointMazeHost(cfg, device)
    host_before = host.digest()
    model = MemJepaLabelFreeAdapter(
        dim=args.dim, slots=args.slots, heads=args.heads,
        residual_scale=args.residual_scale).to(device)
    started = time.time()
    history = train_one_age(
        model, host, bank, age=args.age, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        temperature=args.temperature, variant=args.variant, output_dir=output)
    metrics = evaluate_one_age(
        model, host, bank, age=args.age, batch_size=args.eval_batch_size)
    host_after = host.digest()
    require(host_before == host_after, "frozen host digest changed")
    result = {
        "schema": "mem_jepa_stage_c_age_v1",
        "status": "completed",
        "claim_boundary": (
            "label-free masked-evidence host-exposure test; semantic labels are "
            "used only for final audit readout"),
        "labels_used_for_adapter_training": False,
        "counterfactual_candidate_matching": True,
        "variant": args.variant,
        "age": int(args.age),
        "endpoint_frame": endpoint_frame(3, int(args.age)),
        "predictor_context": list(predictor_context_for_endpoint(
            endpoint_frame(3, int(args.age)))),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": str(resolve(args.config).relative_to(ROOT)),
        "config_sha256": sha256_file(resolve(args.config)),
        "host_digest_unchanged": True,
        "host_digest": host_after,
        "model": {
            "dim": int(args.dim),
            "slots": int(args.slots),
            "heads": int(args.heads),
            "target_dim": TARGET_DIM,
            "roi_rows": list(ROI_ROWS),
            "roi_cols": list(ROI_COLS),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "temperature": float(args.temperature),
            "variant": args.variant,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "final": history[-1] if history else None,
        },
        "metrics": metrics,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(output / f"age_{args.age}.json", result)
    print(json.dumps({
        "age": args.age,
        "passed": metrics["gate"]["passed"],
        "full": metrics["records"]["full"]["balanced_accuracy"],
        "reset": metrics["records"]["reset"]["balanced_accuracy"],
        "no_state": metrics["records"]["no_state"]["balanced_accuracy"],
        "full_retrieval": metrics["records"]["full"][
            "candidate_retrieval_accuracy"],
    }, indent=2), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    cells = {}
    all_passed = True
    for age in [int(value) for value in args.ages]:
        path = output / f"age_{age}.json"
        require(path.is_file(), f"missing age result: {path}")
        value = json.loads(path.read_text())
        cells[str(age)] = value
        all_passed = all_passed and bool(value["metrics"]["gate"]["passed"])
    summary = {
        "schema": "mem_jepa_stage_c_summary_v1",
        "status": "completed" if all_passed else "completed_with_failed_gate",
        "claim_boundary": (
            "Stage C removes semantic class-label training. If gates pass, this "
            "is the first paper-candidate Mem-JEPA host-exposure result."),
        "labels_used_for_adapter_training": False,
        "counterfactual_candidate_matching": True,
        "all_gates_passed": bool(all_passed),
        "ages": [int(value) for value in args.ages],
        "cells": cells,
        "updated_unix": time.time(),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "all_gates_passed": summary["all_gates_passed"],
        "ages": summary["ages"],
    }, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--age", type=int, choices=[4, 8, 15])
    parser.add_argument("--ages", type=int, nargs="*", default=[4, 8, 15])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int, default=9600)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--variant", default="full", choices=[
        "full", "no_host", "no_context", "shuffle_targets",
        "batch_negatives",
    ])
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.aggregate and args.age is None:
        parser.error("--age is required unless --aggregate is used")
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_age(args)


if __name__ == "__main__":
    main()
