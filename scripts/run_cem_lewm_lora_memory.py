#!/usr/bin/env python3
"""Memory-native parameter-efficient adaptation of the official PushT LeWM.

The encoder and all released host tensors stay frozen.  External low-rank
deltas adapt predictor attention QKV and AdaLN modulation while a compact
label-free CEM interface turns the three most novel prefix events into distinct
64-D tokens and a query-conditioned semantic code.  LoRA tensors are saved in
a separate checkpoint and the original host digest is checked after every run.
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
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm import _modulate  # noqa: E402
from lewm.models.official_lewm_pusht import load_official_pusht_checkpoint  # noqa: E402
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK, DEFAULT_PUSHT_SPEC, load_locked_pusht_spec,
    resolve_pusht_path,
)
from scripts.run_lewm_pusht_host_writer import (  # noqa: E402
    DEFAULT_COUNTERFACTUAL_CACHE, age_adjusted_spec, load_admitted,
    load_or_build_counterfactual_cache, state_digest,
)
from scripts.run_mem_jepa_stage_b import fit_classifier  # noqa: E402

DIM = 192
CODE = 64
TOPK = 3
CONTEXT = np.asarray([16, 17, 18], dtype=np.int64)
TASK = "multi-item-visual-binding-recall"
DEFAULT_OUTPUT = ROOT / "outputs/cem_lewm_lora_memory_v1"
DEFAULT_REPORT = ROOT / "docs/CEM_LEWM_LORA_MEMORY_REPORT.md"
DEFAULT_PARETO = ROOT / "docs/assets/cem_lewm_lora_pareto"
DEFAULT_LADDER = ROOT / "docs/assets/cem_lewm_lora_ladder"


def tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def balanced_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([
        np.mean(pred[truth == label] == label) for label in range(6)
        if np.any(truth == label)
    ]))


def pairwise(value: torch.Tensor) -> torch.Tensor:
    value = F.normalize(value, dim=-1)
    distance = 1 - torch.einsum("...id,...jd->...ij", value, value)
    index = torch.triu_indices(6, 6, 1, device=value.device)
    return distance[..., index[0], index[1]]


def corr(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    if rank:
        left = np.argsort(np.argsort(left))
        right = np.argsort(np.argsort(right))
    if np.std(left) < 1e-10 or np.std(right) < 1e-10:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


class LoRADelta(nn.Module):
    """External BA delta; the referenced official Linear is never modified."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)
        if rank:
            self.a = nn.Parameter(torch.empty(rank, in_features))
            self.b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))
        else:
            self.register_parameter("a", None)
            self.register_parameter("b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.rank:
            return x.new_zeros((*x.shape[:-1], 0))
        return F.linear(F.linear(x, self.a), self.b) / self.rank

    def apply(self, base: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        value = base(x)
        return value if not self.rank else value + self.forward(x)

    def penalty(self) -> torch.Tensor:
        if not self.rank:
            return torch.tensor(0.0)
        return self.a.square().mean() + self.b.square().mean()


class MemoryInterface(nn.Module):
    """Distinct top-k tokens, query bottleneck, and predictor conditioning."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.event = nn.Sequential(
            nn.LayerNorm(DIM), nn.Linear(DIM, CODE), nn.GELU(),
            nn.LayerNorm(CODE),
        )
        self.query = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, CODE))
        self.event_bottleneck = nn.Sequential(
            nn.LayerNorm(TOPK * CODE), nn.Linear(TOPK * CODE, CODE))
        self.candidate_bottleneck = nn.Sequential(
            nn.LayerNorm(TOPK * CODE), nn.Linear(TOPK * CODE, CODE))
        self.token_to_condition = nn.Linear(CODE, DIM, bias=False)
        self.code_to_condition = nn.Linear(CODE, DIM, bias=False)
        self.code_to_output = nn.Linear(CODE, DIM, bias=False)
        self.scale_logit = nn.Parameter(torch.tensor(-3.0))
        self.output_scale_logit = nn.Parameter(torch.tensor(-1.0))

    def encode(self, events: torch.Tensor, context: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = F.normalize(self.event(events), dim=-1)
        query = F.normalize(self.query(context.mean(1)), dim=-1)
        # Flattening retains top-k identity/order; the query makes the 64-D
        # bottleneck retrieval-dependent without collapsing the token set.
        code = F.normalize(
            self.event_bottleneck(tokens.flatten(1)) + query, dim=-1)
        return tokens, code

    def candidate_codes(self, candidates: torch.Tensor,
                        context: torch.Tensor) -> torch.Tensor:
        batch = candidates.shape[0]
        tokens = F.normalize(self.event(candidates.reshape(-1, TOPK, DIM)), dim=-1)
        query = F.normalize(self.query(context.mean(1)), dim=-1)
        query = query[:, None].expand(-1, 6, -1).reshape(-1, CODE)
        code = F.normalize(
            self.candidate_bottleneck(tokens.flatten(1)) + query, dim=-1)
        return code.reshape(batch, 6, CODE)

    def condition(self, tokens: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        condition = self.token_to_condition(tokens)
        if self.mode == "bottleneck_tokens":
            condition = condition + self.code_to_condition(code)[:, None]
        return torch.sigmoid(self.scale_logit) * condition


class MemoryNativeLeWM(nn.Module):
    """Functional official predictor plus separately registered LoRA deltas."""

    def __init__(self, host: nn.Module, rank: int, interface: str,
                 location: str = "qkv_adaln") -> None:
        super().__init__()
        object.__setattr__(self, "_host", host)
        self.rank = int(rank)
        self.location = location
        self.memory = MemoryInterface(interface)
        layers = host.predictor.transformer.layers
        qkv_rank = rank if location in ("qkv", "qkv_adaln") else 0
        ada_rank = rank if location in ("adaln", "qkv_adaln") else 0
        self.qkv = nn.ModuleList([
            LoRADelta(DIM, block.attn.to_qkv.out_features, qkv_rank)
            for block in layers
        ])
        self.adaln = nn.ModuleList([
            LoRADelta(DIM, block.adaLN_modulation[1].out_features, ada_rank)
            for block in layers
        ])

    @property
    def host(self) -> nn.Module:
        return object.__getattribute__(self, "_host")

    def predictor(self, z: torch.Tensor, actions: torch.Tensor,
                  condition: torch.Tensor) -> torch.Tensor:
        host = self.host
        x = z + host.predictor.pos_embedding[:, :z.shape[1]]
        c = host.action_encoder(actions) + condition
        transformer = host.predictor.transformer
        for index, block in enumerate(transformer.layers):
            activated = block.adaLN_modulation[0](c)
            modulation = self.adaln[index].apply(
                block.adaLN_modulation[1], activated)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(
                6, dim=-1)
            attn = block.attn
            normalized = attn.norm(_modulate(block.norm1(x), shift_a, scale_a))
            qkv = self.qkv[index].apply(attn.to_qkv, normalized).chunk(3, -1)
            batch, steps = x.shape[:2]
            q, k, v = (
                item.reshape(batch, steps, attn.heads, -1).transpose(1, 2)
                for item in qkv
            )
            attended = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True)
            attended = attended.transpose(1, 2).reshape(batch, steps, -1)
            x = x + gate_a * attn.to_out(attended)
            x = x + gate_m * block.mlp(
                _modulate(block.norm2(x), shift_m, scale_m))
        hidden = transformer.output_proj(transformer.norm(x))
        batch, steps = hidden.shape[:2]
        return host.pred_proj(hidden.flatten(0, 1)).reshape(batch, steps, DIM)

    def forward(self, events: torch.Tensor, context: torch.Tensor,
                actions: torch.Tensor, *, memory: bool = True
                ) -> dict[str, torch.Tensor]:
        tokens, code = self.memory.encode(events, context)
        condition = self.memory.condition(tokens, code) if memory else \
            torch.zeros_like(context)
        output = self.predictor(context, actions, condition)
        if memory:
            # A bounded predictor-boundary route prevents the frozen pred_proj
            # from erasing a signal that the internal LoRA path has learned.
            residual = (torch.sigmoid(self.memory.output_scale_logit)
                        * self.memory.code_to_output(code))[:, None]
            output = output + residual
        return {"output": output, "tokens": tokens, "code": code,
                "condition": condition}

    def delta_penalty(self) -> torch.Tensor:
        values = [module.penalty().to(next(self.parameters()).device)
                  for module in [*self.qkv, *self.adaln] if module.rank]
        return sum(values, torch.tensor(0.0, device=next(self.parameters()).device))


def build_arrays(data: dict[str, Any], spec: dict[str, Any]) -> dict[str, np.ndarray]:
    cue_start = int(spec["sequence"]["cue_start"])
    cue_end = cue_start + int(spec["sequence"]["cue_length"])
    delta = np.asarray(data["z"] - data["z_base"], dtype=np.float32)
    novelty = np.mean(delta[:, :19].square() if hasattr(delta, "square")
                      else delta[:, :19] ** 2, axis=-1)
    selected = np.argpartition(novelty, -TOPK, axis=1)[:, -TOPK:]
    selected.sort(1)
    rows = np.arange(len(delta))[:, None]
    events = delta[rows, selected]
    z_all = np.asarray(data["z_counterfactual"], dtype=np.float32)
    base_cue = np.asarray(data["z_base"][:, cue_start:cue_end], dtype=np.float32)
    candidates = z_all - base_cue[:, None]
    observed = np.asarray(data["z_cue"], dtype=np.float32)
    positive = np.argmin(np.mean(
        (z_all - observed[:, None]) ** 2, axis=(2, 3)), axis=1)
    target_idx = CONTEXT + 1
    return {
        "events": events, "candidates": candidates,
        "positive": positive.astype(np.int64), "selected": selected,
        "context": np.asarray(data["z"][:, CONTEXT], dtype=np.float32),
        "actions": np.asarray(data["actions"][:, CONTEXT], dtype=np.float32),
        "target": np.asarray(data["z"][:, target_idx], dtype=np.float32),
        "ordinary_context": np.asarray(data["z_base"][:, CONTEXT], dtype=np.float32),
        "ordinary_target": np.asarray(data["z_base"][:, target_idx], dtype=np.float32),
        "labels": np.asarray(data["labels"], dtype=np.int64),
    }


@torch.no_grad()
def base_predict(host: nn.Module, context: torch.Tensor,
                 actions: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=context.device.type == "cuda"):
        return host.predict(context, actions).float()


def prepare(args: argparse.Namespace) -> tuple[nn.Module, str, dict[str, Any]]:
    device = torch.device(args.device)
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


def geometry_loss(codes: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    target = pairwise(candidates.flatten(-2)).detach()
    predicted = pairwise(codes)
    target = (target - target.mean(-1, keepdim=True)) / (
        target.std(-1, keepdim=True) + 1e-5)
    predicted = (predicted - predicted.mean(-1, keepdim=True)) / (
        predicted.std(-1, keepdim=True) + 1e-5)
    return F.mse_loss(predicted, target)


def config_id(rank: int, interface: str, distill: bool, location: str) -> str:
    return f"r{rank}_{interface}_{'distill' if distill else 'nodistill'}_{location}"


def train_one(host: nn.Module, digest: str, arrays: dict[str, Any], *,
              rank: int, interface: str, distill: bool, location: str,
              seed: int, epochs: int, args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = next(host.parameters()).device
    model = MemoryNativeLeWM(host, rank, interface, location).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))
    train = arrays["train"]
    rng = np.random.default_rng(5000 + seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train["events"]))
        accum = {key: [] for key in (
            "future", "cf", "geometry", "distill", "delta", "total")}
        started = time.perf_counter()
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset:offset + args.batch_size]
            if len(rows) < 4:
                continue
            events = tensor(train["events"][rows], device)
            context = tensor(train["context"][rows], device)
            actions = tensor(train["actions"][rows], device)
            target = tensor(train["target"][rows], device)
            candidates = tensor(train["candidates"][rows], device)
            positive = torch.as_tensor(train["positive"][rows], device=device)
            result = model(events, context, actions)
            candidate_codes = model.memory.candidate_codes(candidates, context)
            code_logits = torch.einsum(
                "bd,bkd->bk", result["code"], candidate_codes) / args.temperature
            with torch.no_grad():
                original = base_predict(host, context, actions)
            # The audited host output itself, without a trainable readout, must
            # align with the observed same-base branch.  A learned projection
            # can solve InfoNCE while leaving raw host features unreadable.
            host_code = F.normalize(
                result["output"][:, -1] - original[:, -1], dim=-1)
            candidate_host = F.normalize(candidates.mean(2), dim=-1)
            host_logits = torch.einsum(
                "bd,bkd->bk", host_code, candidate_host) / args.temperature
            row_index = torch.arange(len(rows), device=device)
            host_reconstruction = F.mse_loss(
                result["output"][:, -1] - original[:, -1],
                args.host_signal_amplitude * candidate_host[row_index, positive])
            future = F.mse_loss(result["output"], target)
            cf = (F.cross_entropy(code_logits, positive)
                  + args.host_cf_weight * F.cross_entropy(host_logits, positive)
                  + args.host_reconstruction_weight * host_reconstruction)
            geom = geometry_loss(candidate_codes, candidates)
            ordinary = tensor(train["ordinary_context"][rows], device)
            ordinary_target = tensor(train["ordinary_target"][rows], device)
            with torch.no_grad():
                original_ordinary = base_predict(host, ordinary, actions)
            adapted_ordinary = model(
                torch.zeros_like(events), ordinary, actions, memory=False)["output"]
            distill_loss = F.mse_loss(adapted_ordinary, original_ordinary)
            # Also retain ordinary next-latent quality, not merely output equality.
            distill_loss = distill_loss + 0.25 * F.mse_loss(
                adapted_ordinary, ordinary_target)
            delta = model.delta_penalty()
            loss = (future + args.cf_weight * cf + args.geometry_weight * geom
                    + (args.distill_weight if distill else 0.0) * distill_loss
                    + args.delta_weight * delta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            for key, value in (
                ("future", future), ("cf", cf), ("geometry", geom),
                ("distill", distill_loss), ("delta", delta), ("total", loss)):
                accum[key].append(float(value.detach()))
        scheduler.step()
        record = {key: float(np.mean(value)) for key, value in accum.items()}
        record.update(epoch=epoch, seconds=time.perf_counter() - started)
        history.append(record)
        print(f"[lora-memory] r{rank} {interface} distill={distill} "
              f"s{seed} ep{epoch}/{epochs} future={record['future']:.5f} "
              f"cf={record['cf']:.3f} sec={record['seconds']:.1f}", flush=True)
    result = evaluate(model, host, arrays, seed, args)
    host_params = sum(p.numel() for p in host.parameters())
    trainable = sum(p.numel() for p in params)
    cid = config_id(rank, interface, distill, location)
    result.update({
        "schema": "cem_lewm_lora_memory_run_v1", "seed": seed,
        "config": {"rank": rank, "interface": interface,
                   "distillation": distill, "location": location},
        "history": history, "trainable_parameters": trainable,
        "host_parameters": host_params,
        "trainable_fraction_percent": 100 * trainable / host_params,
        "frozen_base_digest_before": digest,
        "frozen_base_digest_after": state_digest(host),
        "frozen_base_digest_unchanged": state_digest(host) == digest,
    })
    if not result["frozen_base_digest_unchanged"]:
        raise RuntimeError("official base digest changed")
    out = Path(args.output) / "runs" / cid / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"memory_interface": model.memory.state_dict(),
                "qkv_lora": model.qkv.state_dict(),
                "adaln_lora": model.adaln.state_dict()}, out / "lora_adapter.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def collect(model: MemoryNativeLeWM, host: nn.Module, data: dict[str, np.ndarray],
            condition: str, seed: int) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    values = {key: [] for key in (
        "memory_tokens", "bottleneck_code", "conditioning", "host_output")}
    count = len(data["events"])
    for offset in range(0, count, 256):
        rows = np.arange(offset, min(count, offset + 256))
        events = data["events"][rows].copy()
        memory = True
        if condition in ("host_only", "no_state"):
            memory = False
        elif condition == "reset":
            events.fill(0)
        elif condition == "shuffled":
            events = data["events"][(rows + 1) % count]
        elif condition == "random":
            rng = np.random.default_rng(8000 + seed + offset)
            events = rng.normal(0, np.std(data["events"]), events.shape).astype(np.float32)
        result = model(
            tensor(events, device), tensor(data["context"][rows], device),
            tensor(data["actions"][rows], device), memory=memory)
        for key, item in (
            ("memory_tokens", result["tokens"].flatten(1)),
            ("bottleneck_code", result["code"]),
            ("conditioning", result["condition"].flatten(1)),
            ("host_output", result["output"][:, -1])):
            values[key].append(item.float().cpu().numpy())
    return {key: np.concatenate(item) for key, item in values.items()}


@torch.no_grad()
def evaluate(model: MemoryNativeLeWM, host: nn.Module,
             arrays: dict[str, Any], seed: int,
             args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    train, val = arrays["train"], arrays["validation"]
    train_full = collect(model, host, train, "full", seed)
    conditions = ("full", "reset", "no_state", "host_only", "shuffled", "random")
    features = {key: collect(model, host, val, key, seed) for key in conditions}
    ladder = {}
    for level in train_full:
        prediction = fit_classifier(
            train_full[level], train["labels"], features["full"][level])
        ladder[level] = balanced_accuracy(prediction, val["labels"])
    controls = {}
    for condition in conditions:
        prediction = fit_classifier(
            train_full["host_output"], train["labels"],
            features[condition]["host_output"])
        controls[condition] = balanced_accuracy(prediction, val["labels"])
    device = next(model.parameters()).device
    losses = {"with_memory": [], "frozen_baseline": [],
              "no_memory_adapted": [], "ordinary_frozen": [],
              "ordinary_adapted": []}
    candidate_codes, candidate_raw = [], []
    latency_base, latency_memory = [], []
    for offset in range(0, len(val["events"]), 256):
        rows = np.arange(offset, min(offset + 256, len(val["events"])))
        events = tensor(val["events"][rows], device)
        context = tensor(val["context"][rows], device)
        actions = tensor(val["actions"][rows], device)
        target = tensor(val["target"][rows], device)
        base = base_predict(host, context, actions)
        full = model(events, context, actions)
        no_memory = model(events, context, actions, memory=False)["output"]
        ordinary = tensor(val["ordinary_context"][rows], device)
        ordinary_target = tensor(val["ordinary_target"][rows], device)
        ordinary_base = base_predict(host, ordinary, actions)
        ordinary_adapted = model(
            torch.zeros_like(events), ordinary, actions, memory=False)["output"]
        for key, output, goal in (
            ("with_memory", full["output"], target),
            ("frozen_baseline", base, target),
            ("no_memory_adapted", no_memory, target),
            ("ordinary_frozen", ordinary_base, ordinary_target),
            ("ordinary_adapted", ordinary_adapted, ordinary_target)):
            losses[key].append(float(F.mse_loss(output, goal)))
        candidate_raw.append(val["candidates"][rows].reshape(len(rows), 6, -1))
        candidate_codes.append(model.memory.candidate_codes(
            tensor(val["candidates"][rows], device), context).cpu().numpy())
    # Warm both paths, then time repeated calls on the same resident batch.
    for _ in range(3):
        base_predict(host, context, actions)
        model(events, context, actions)
    torch.cuda.synchronize()
    for function, destination in (
            (lambda: base_predict(host, context, actions), latency_base),
            (lambda: model(events, context, actions), latency_memory)):
        started = time.perf_counter()
        for _ in range(20):
            function()
        torch.cuda.synchronize()
        destination.append((time.perf_counter() - started) / 20)
    losses = {key: float(np.mean(value)) for key, value in losses.items()}
    raw = np.concatenate(candidate_raw)
    codes = np.concatenate(candidate_codes)
    raw_pair, code_pair = pairwise(torch.from_numpy(raw)).numpy(), \
        pairwise(torch.from_numpy(codes)).numpy()
    geometry = {
        "pearson": float(np.mean([corr(a, b) for a, b in zip(raw_pair, code_pair)])),
        "spearman": float(np.mean([
            corr(a, b, True) for a, b in zip(raw_pair, code_pair)])),
    }
    # Direct event-set intervention: zero retrieved cue events versus a
    # count-matched cyclic replacement from another episode.
    deletion = {"cue": [], "random": []}
    for offset in range(0, min(256, len(val["events"])), 128):
        rows = np.arange(offset, min(offset + 128, len(val["events"]), 256))
        context = tensor(val["context"][rows], device)
        actions = tensor(val["actions"][rows], device)
        target = tensor(val["target"][rows], device)
        events = tensor(val["events"][rows], device)
        full = model(events, context, actions)["output"]
        cue_deleted = model(torch.zeros_like(events), context, actions)["output"]
        random_events = tensor(
            val["events"][(rows + 17) % len(val["events"])], device)
        random_deleted = model(random_events, context, actions)["output"]
        base_loss = F.mse_loss(full, target)
        deletion["cue"].append(float(F.mse_loss(cue_deleted, target) - base_loss))
        deletion["random"].append(float(F.mse_loss(random_deleted, target) - base_loss))
    deletion = {key: float(np.mean(value)) for key, value in deletion.items()}
    max_control = max(controls[key] for key in conditions if key != "full")
    memory_ratio = losses["with_memory"] / losses["frozen_baseline"]
    ordinary_ratio = losses["ordinary_adapted"] / losses["ordinary_frozen"]
    success = (controls["full"] >= .75 and max_control <= .217
               and memory_ratio <= 1.05 and ordinary_ratio <= 1.05
               and deletion["cue"] > deletion["random"])
    return {
        "diagnostic_ladder": ladder, "controls_host_output_bacc": controls,
        "max_control_bacc": max_control, "six_way_geometry": geometry,
        "host_future_loss": losses,
        "memory_loss_ratio": memory_ratio,
        "ordinary_no_memory_loss_ratio": ordinary_ratio,
        "causal_deletion_delta_loss": deletion,
        "latency": {
            "base_ms": 1000 * float(np.mean(latency_base)),
            "memory_ms": 1000 * float(np.mean(latency_memory)),
            "overhead_percent": 100 * (
                np.mean(latency_memory) / np.mean(latency_base) - 1),
        },
        "success_gate": {
            "host_bacc": controls["full"] >= .75,
            "controls": max_control <= .217,
            "memory_loss": memory_ratio <= 1.05,
            "ordinary_preservation": ordinary_ratio <= 1.05,
            "causal_deletion": deletion["cue"] > deletion["random"],
            "passed": success,
        },
        "labels_used_for_training_loss": False,
    }


def mean_std(results: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for result in results:
        value: Any = result
        for key in path:
            value = value[key]
        values.append(float(value))
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "by_seed": values}


def aggregate(screen: list[dict[str, Any]], confirmations: list[dict[str, Any]],
              digest: str, args: argparse.Namespace) -> dict[str, Any]:
    best = confirmations
    report = {
        "schema": "cem_lewm_lora_memory_report_v1",
        "screen": screen, "selected_config": best[0]["config"],
        "three_seed": {
            "seeds": [x["seed"] for x in best],
            "host_bacc": mean_std(best, ("controls_host_output_bacc", "full")),
            "max_control": mean_std(best, ("max_control_bacc",)),
            "memory_loss": mean_std(best, ("host_future_loss", "with_memory")),
            "frozen_memory_baseline": mean_std(
                best, ("host_future_loss", "frozen_baseline")),
            "memory_loss_ratio": mean_std(best, ("memory_loss_ratio",)),
            "ordinary_frozen_loss": mean_std(
                best, ("host_future_loss", "ordinary_frozen")),
            "ordinary_adapted_loss": mean_std(
                best, ("host_future_loss", "ordinary_adapted")),
            "ordinary_loss_ratio": mean_std(
                best, ("ordinary_no_memory_loss_ratio",)),
            "geometry_pearson": mean_std(best, ("six_way_geometry", "pearson")),
            "geometry_spearman": mean_std(best, ("six_way_geometry", "spearman")),
            "cue_deletion": mean_std(
                best, ("causal_deletion_delta_loss", "cue")),
            "random_deletion": mean_std(
                best, ("causal_deletion_delta_loss", "random")),
            "latency_overhead": mean_std(best, ("latency", "overhead_percent")),
            "ladder": {
                key: mean_std(best, ("diagnostic_ladder", key))
                for key in ("memory_tokens", "bottleneck_code",
                            "conditioning", "host_output")
            },
            "all_seeds_passed": all(x["success_gate"]["passed"] for x in best),
        },
        "frozen_base": {"digest": digest, "unchanged": all(
            x["frozen_base_digest_unchanged"] for x in screen + best)},
        "trainable_parameters": best[0]["trainable_parameters"],
        "trainable_fraction_percent": best[0]["trainable_fraction_percent"],
        "labels_used_for_training_loss": False,
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "report.json").write_text(json.dumps(report, indent=2))
    plot(report, args)
    write_report(report, args)
    return report


def plot(report: dict[str, Any], args: argparse.Namespace) -> None:
    screen = report["screen"]
    labels = [
        f"r{x['config']['rank']}\n{x['config']['interface'][:4]}\n"
        f"{'D' if x['config']['distillation'] else 'no D'}"
        for x in screen]
    exposure = [x["controls_host_output_bacc"]["full"] for x in screen]
    memory_ratio = [x["memory_loss_ratio"] for x in screen]
    ordinary_ratio = [x["ordinary_no_memory_loss_ratio"] for x in screen]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scatter = axes[0].scatter(
        memory_ratio, ordinary_ratio, c=exposure, s=100, cmap="viridis",
        vmin=1 / 6, vmax=1)
    for x, y, label in zip(memory_ratio, ordinary_ratio, labels):
        axes[0].annotate(label.replace("\n", "/"), (x, y), fontsize=7)
    axes[0].axvline(1.05, color="#c62828", linestyle="--")
    axes[0].axhline(1.05, color="#c62828", linestyle="--")
    axes[0].set(xlabel="Memory-conditioned / frozen loss",
                ylabel="No-memory adapted / frozen loss",
                title="Exposure–preservation Pareto screen")
    fig.colorbar(scatter, ax=axes[0], label="Host-output six-way BAcc")
    x = np.arange(len(screen))
    axes[1].bar(x - .2, exposure, .4, label="Memory", color="#5e35b1")
    axes[1].bar(x + .2, [x["max_control_bacc"] for x in screen], .4,
                label="Max control", color="#9e9e9e")
    axes[1].axhline(.75, color="#c62828", linestyle="--")
    axes[1].axhline(.217, color="black", linestyle=":")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1)
    axes[1].set(title="Factorial exposure screen", ylabel="Balanced accuracy")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    stem = Path(args.pareto)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    best = report["three_seed"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    levels = ("memory_tokens", "bottleneck_code", "conditioning", "host_output")
    confirmations = report.get("_confirmations", [])
    if confirmations:
        means = [np.mean([x["diagnostic_ladder"][k] for x in confirmations])
                 for k in levels]
        stds = [np.std([x["diagnostic_ladder"][k] for x in confirmations])
                for k in levels]
    else:
        selected = report["screen"][0]
        means = [selected["diagnostic_ladder"][k] for k in levels]
        stds = [0] * 4
    ax.bar(np.arange(4), means, yerr=stds, color=(
        "#607d8b", "#42a5f5", "#ef6c00", "#5e35b1"))
    ax.axhline(.75, color="#c62828", linestyle="--")
    ax.axhline(1 / 6, color="black", linestyle=":")
    ax.set_xticks(np.arange(4), ("Tokens", "64-D code", "Condition", "Host"))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Six-way balanced accuracy")
    ax.set_title("Memory-native diagnostic ladder")
    fig.tight_layout()
    ladder = Path(args.ladder)
    fig.savefig(ladder.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(ladder.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    best = report["three_seed"]
    config = report["selected_config"]
    rows = []
    for item in report["screen"]:
        cfg = item["config"]
        rows.append(
            f"| {cfg['rank']} | {cfg['interface']} | {cfg['distillation']} | "
            f"{cfg['location']} | {item['controls_host_output_bacc']['full']:.3f} | "
            f"{item['max_control_bacc']:.3f} | {item['memory_loss_ratio']:.3f} | "
            f"{item['ordinary_no_memory_loss_ratio']:.3f} | "
            f"{item['trainable_fraction_percent']:.3f}% |")
    text = f"""# CEM–LeWM Memory-Native LoRA Report

## Verdict

The joint gate **{'passed' if best['all_seeds_passed'] else 'did not pass'}**.
The selected configuration is rank **{config['rank']}**, interface
**{config['interface']}**, predictor location **{config['location']}**, with
no-memory distillation **{config['distillation']}**.

Across three seeds, memory-conditioned host-output BAcc was
**{best['host_bacc']['mean']:.3f} ± {best['host_bacc']['std']:.3f}** and the
maximum control was **{best['max_control']['mean']:.3f} ± {best['max_control']['std']:.3f}**.
Memory-conditioned future loss was **{best['memory_loss']['mean']:.6f}** versus
the frozen baseline **{best['frozen_memory_baseline']['mean']:.6f}**
(ratio **{best['memory_loss_ratio']['mean']:.3f}**). Ordinary no-memory PushT
loss changed from **{best['ordinary_frozen_loss']['mean']:.6f}** to
**{best['ordinary_adapted_loss']['mean']:.6f}**
(ratio **{best['ordinary_loss_ratio']['mean']:.3f}**).

## Factorial screen

| Rank | Interface | Distill | Location | Host BAcc | Max control | Memory loss ratio | No-memory ratio | Trainable |
|---:|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Geometry, causality, and efficiency

- Six-way geometry Pearson/Spearman:
  **{best['geometry_pearson']['mean']:.3f} / {best['geometry_spearman']['mean']:.3f}**.
- Cue-group / random deletion Δloss:
  **{best['cue_deletion']['mean']:.6f} / {best['random_deletion']['mean']:.6f}**.
- Trainable parameters: **{report['trainable_parameters']:,}**
  (**{report['trainable_fraction_percent']:.3f}%** of the official host).
- Predictor-plus-memory latency overhead:
  **{best['latency_overhead']['mean']:.1f}%**.
- Diagnostic ladder (tokens → 64-D code → conditioning → host):
  **{best['ladder']['memory_tokens']['mean']:.3f} →
  {best['ladder']['bottleneck_code']['mean']:.3f} →
  {best['ladder']['conditioning']['mean']:.3f} →
  {best['ladder']['host_output']['mean']:.3f}**.

## Integrity and protocol

The encoder and every original LeWM tensor remained frozen. Base digest
`{report['frozen_base']['digest']}` was unchanged: **{report['frozen_base']['unchanged']}**.
LoRA and memory-interface tensors were stored separately. Training used latent
matching among six same-base rendered counterfactuals; semantic labels were
used only by post-hoc linear probes. Ordinary no-memory examples were the
paired unmodified PushT trajectories (`z_base`) and were distilled against the
original host outputs.

![Pareto screen](assets/cem_lewm_lora_pareto.png)

![Diagnostic ladder](assets/cem_lewm_lora_ladder.png)
"""
    Path(args.report).write_text(text)


def campaign(args: argparse.Namespace) -> dict[str, Any]:
    host, digest, arrays = prepare(args)
    cells = [
        (0, "bottleneck_tokens", True, "qkv_adaln"),
        (2, "bottleneck_tokens", True, "qkv_adaln"),
        (4, "bottleneck_tokens", True, "qkv_adaln"),
        (8, "bottleneck_tokens", True, "qkv_adaln"),
        (4, "bottleneck_tokens", False, "qkv_adaln"),
        (4, "tokens", True, "qkv_adaln"),
        (4, "bottleneck_tokens", True, "qkv"),
        (4, "bottleneck_tokens", True, "adaln"),
    ]
    screen = [train_one(
        host, digest, arrays, rank=rank, interface=interface, distill=distill,
        location=location, seed=0, epochs=args.screen_epochs, args=args)
        for rank, interface, distill, location in cells]
    # Rank zero is the required frozen semantic baseline, not a selectable
    # memory-native parameter-efficient configuration.
    positive_rank = [x for x in screen if x["config"]["rank"] > 0]
    feasible = [x for x in positive_rank if x["success_gate"]["passed"]]
    pool = feasible or positive_rank
    selected = sorted(pool, key=lambda x: (
        x["success_gate"]["passed"],
        x["controls_host_output_bacc"]["full"],
        -max(x["memory_loss_ratio"], x["ordinary_no_memory_loss_ratio"]),
    ), reverse=True)[0]
    cfg = selected["config"]
    confirmations = []
    for seed in (0, 1, 2):
        confirmations.append(train_one(
            host, digest, arrays, rank=cfg["rank"],
            interface=cfg["interface"], distill=cfg["distillation"],
            location=cfg["location"], seed=seed,
            epochs=args.confirm_epochs, args=args))
    report = aggregate(screen, confirmations, digest, args)
    # Re-render ladder with all confirmation seeds.
    report["_confirmations"] = confirmations
    plot(report, args)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--pareto", default=str(DEFAULT_PARETO))
    parser.add_argument("--ladder", default=str(DEFAULT_LADDER))
    parser.add_argument("--counterfactual-cache", default=str(DEFAULT_COUNTERFACTUAL_CACHE))
    parser.add_argument("--screen-epochs", type=int, default=10)
    parser.add_argument("--confirm-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--temperature", type=float, default=.07)
    parser.add_argument("--cf-weight", type=float, default=.25)
    parser.add_argument("--host-cf-weight", type=float, default=1.0)
    parser.add_argument("--host-reconstruction-weight", type=float, default=1000.0)
    parser.add_argument("--host-signal-amplitude", type=float, default=.25)
    parser.add_argument("--geometry-weight", type=float, default=.5)
    parser.add_argument("--distill-weight", type=float, default=4.0)
    parser.add_argument("--delta-weight", type=float, default=1e-3)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 2, 4, 8), default=4)
    parser.add_argument("--interface", choices=("tokens", "bottleneck_tokens"),
                        default="bottleneck_tokens")
    parser.add_argument("--location", choices=("qkv", "adaln", "qkv_adaln"),
                        default="qkv_adaln")
    parser.add_argument("--no-distill", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device not in ("cuda:2", "cuda:0"):
        raise ValueError("assigned GPU is cuda:2 (cuda:0 fallback); GPU3 forbidden")
    if args.single:
        host, digest, arrays = prepare(args)
        result = train_one(
            host, digest, arrays, rank=args.rank, interface=args.interface,
            distill=not args.no_distill, location=args.location,
            seed=args.seed, epochs=args.epochs, args=args)
        print(json.dumps(result, indent=2))
    else:
        report = campaign(args)
        print(json.dumps({
            "selected": report["selected_config"],
            "three_seed": report["three_seed"],
            "report": str(Path(args.output) / "report.json"),
        }, indent=2))


if __name__ == "__main__":
    main()
