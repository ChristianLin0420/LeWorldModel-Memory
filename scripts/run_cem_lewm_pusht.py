#!/usr/bin/env python3
"""Causal-Effect Memory (CEM) on the FROZEN official PushT LeWM host.

Implements the CEM controller (``docs/EMERGENT_CUE_MEMORY_PARADIGM.md``) on the
LeWorldModel PushT binding task.  Everything below the host encoder/predictor
stays frozen (state-dict digest asserted unchanged); only the CEM controller
and the residual writer train.

The PRIMARY loss is the frozen host's OWN future-latent rollout loss with memory
injected over the post-window horizon (the cue frames are illegal to attend at
the readout, so memory is the only carrier).  Auxiliary terms: a free-energy
write cost ``beta*E[gate]`` and a periodic distillation loss that teaches the
amortized ``ce_head`` to match TRUE hard-deletion CE.

Cue labels are used ONLY in the post-hoc fail-closed audit.

Outputs (shared visualization schema):
  outputs/cem_lewm_v1/<env>/s<seed>/decision_log.json
  outputs/cem_lewm_v1/<env>/s<seed>/summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
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

from lewm.models.cem_controller import CEMController, spearman  # noqa: E402
from lewm.models.official_lewm_pusht import (  # noqa: E402
    load_official_pusht_checkpoint,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    resolve_pusht_path,
)
from scripts.run_lewm_pusht_host_writer import (  # noqa: E402
    DEFAULT_COUNTERFACTUAL_CACHE,
    LeWMHostAlignedEvidenceWriter,
    age_adjusted_spec,
    evidence_targets,
    load_admitted,
    load_or_build_counterfactual_cache,
    state_digest,
)
from scripts.run_mem_jepa_stage_c import (  # noqa: E402
    contrastive_loss,
    positive_cosine_loss,
)
from scripts.run_mem_jepa_stage_b import fit_classifier  # noqa: E402

LATENT_DIM = 192
DEFAULT_OUTPUT = ROOT / "outputs/cem_lewm_v1"


class CEMLeWMWriter(LeWMHostAlignedEvidenceWriter):
    """Residual writer with a surprise-driven WRITE gate over the prefix.

    Reuses the frozen-host residual-injection path
    (``fused = context_z + residual_scale * gate * delta``) from the host-writer
    verbatim, adding a per-prefix-frame write gate ``w_t`` that controls which
    latents are admitted into the slot memory.  ``w_t`` is the CEM surprise gate
    (host self-error over its legal window), so WRITE is label-free.
    """

    def _slots_weighted(self, tokens: torch.Tensor,
                        write_gate: torch.Tensor) -> torch.Tensor:
        logits = self.assign(tokens).transpose(1, 2)  # (B, S, T)
        logits = logits + torch.log(write_gate.clamp_min(1e-6))[:, None, :]
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bst,btd->bsd", weights, tokens)

    def set_injection_mode(self, mode: str) -> None:
        self.injection_mode = mode

    def inject_gated(self, prefix_z, prefix_actions, prefix_times, context_z,
                     context_actions, context_times, write_gate):
        prefix_tokens = self._tokens(prefix_z, prefix_actions, prefix_times)
        context_tokens = self._tokens(context_z, context_actions, context_times)
        slots = self._slots_weighted(prefix_tokens, write_gate)
        if getattr(self, "injection_mode", "query") == "generic":
            attended = slots.mean(dim=1, keepdim=True).expand(
                -1, context_tokens.shape[1], -1)
        else:
            attended, _ = self.cross(
                context_tokens, slots, slots, need_weights=False)
        joined = torch.cat((context_tokens, attended), dim=-1)
        delta = self.writer(joined)
        gate = torch.sigmoid(self.gate(joined))
        fused = context_z + self.residual_scale * gate * delta
        return fused, slots


def tt(a: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


def host_predict(host, z, a):
    if z.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return host.predict(z, a).float()
    return host.predict(z, a).float()


HOST_HISTORY = 3  # OFFICIAL_HISTORY: the frozen predictor's legal context window


@torch.no_grad()
def frame_surprise(host, z, a, decision: int) -> torch.Tensor:
    """Per-prefix-frame host self-surprise s_t = ||predict(z_{<t}) - z_t||^2.

    The frozen LeWM predictor only accepts a bounded ``HOST_HISTORY``-frame
    context, so surprise for frame ``t`` uses the legal sliding window
    ``z_{t-H:t}`` (the host's own one-step error over its legal window).
    Returns (B, decision) with frame 0 set to 0 (no predictor context).
    """
    bsz = z.shape[0]
    out = torch.zeros(bsz, decision, device=z.device)
    for t in range(1, decision):
        lo = max(0, t - HOST_HISTORY)
        pred = host_predict(host, z[:, lo:t], a[:, lo:t])[:, -1]
        out[:, t] = ((pred - z[:, t]) ** 2).mean(dim=-1)
    return out


def rollout_loss(host, cem, fused_context, ctx_actions, targets):
    pred = cem.readout(host_predict(host, fused_context, ctx_actions))
    return F.mse_loss(pred, targets)


def rollout_loss_per_sample(host, cem, fused_context, ctx_actions, targets):
    pred = cem.readout(host_predict(host, fused_context, ctx_actions))
    dims = tuple(range(1, pred.dim()))
    return ((pred - targets) ** 2).mean(dim=dims)


class CEMLeWM:
    def __init__(self, args, spec, device):
        self.args = args
        self.spec = spec
        self.device = device
        self.endpoint = getattr(args, "endpoint", "cue_conditioned")
        seq = spec["sequence"]
        self.decision = int(seq["decision_index"])
        self.context = np.asarray(seq["final_context_indices"], dtype=np.int64)
        self.cue_start = int(seq["cue_start"])
        self.cue_length = int(seq["cue_length"])
        # host next-latent targets over the post-window horizon
        self.target_idx = (self.context + 1).astype(np.int64)  # e.g. [17,18,19]
        self.prefix_idx = np.arange(self.decision, dtype=np.int64)  # 0..18

    def readout(self, pred_full):
        """Decision-step host output (both endpoints read the last step)."""
        return pred_full if self.endpoint == "own_rollout" else pred_full[:, -1]

    def target(self, data, rows, z):
        """Endpoint target: strict own next-latent, or the cue-recall latent.

        own_rollout    -> the host's OWN future latents z[17,18,19] (strict CEM
                          criterion; cue-free on this frozen host -> a null).
        cue_conditioned-> the observed cue latent (the RECALL readout target that
                          the repo's fail-closed audit / host-writer use): memory
                          must carry the cue for the host forward to reproduce it.
        """
        if self.endpoint == "own_rollout":
            return z[:, self.target_idx]
        return tt(data["z_cue"][rows].mean(axis=1), self.device)

    def batch(self, data, rows):
        z = tt(data["z"][rows], self.device)
        a = tt(data["actions"][rows], self.device)
        return z, a

    def forward(self, writer, ctrl, host, z, a, *, write_from_full: bool,
                deletion_frame: int | None = None):
        """Inject memory and return (fused_context, write_gate, surprise)."""
        ctx = torch.as_tensor(self.context, device=self.device)
        context_z = z[:, self.context]
        context_a = a[:, self.context]
        if write_from_full:
            pf = torch.as_tensor(self.prefix_idx, device=self.device)
            prefix_z = z[:, self.prefix_idx]
            prefix_a = a[:, self.prefix_idx]
            surprise = frame_surprise(host, z, a, self.decision)
        else:
            pf = ctx
            prefix_z = context_z
            prefix_a = context_a
            surprise = frame_surprise(
                host, z, a, self.decision)[:, self.context]
        w = ctrl.write_gate.soft_gate(surprise)
        if deletion_frame is not None:
            w = w.clone()
            if isinstance(deletion_frame, (list, tuple, np.ndarray)):
                for j in deletion_frame:
                    w[:, int(j)] = 0.0
            else:
                w[:, deletion_frame] = 0.0
        fused, slots = writer.inject_gated(
            prefix_z, prefix_a, pf, context_z, context_a, ctx, w)
        return fused, w, surprise


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    locked = load_locked_pusht_spec(str(DEFAULT_PUSHT_SPEC), str(DEFAULT_PUSHT_LOCK))
    spec = age_adjusted_spec(locked, int(args.age))
    train_data = load_admitted(locked, args.task, "train")
    val_data = load_admitted(locked, args.task, "validation")

    host = load_official_pusht_checkpoint(
        resolve_pusht_path(locked["official_host"]["bundle_path"]), device).eval()
    for p in host.parameters():
        p.requires_grad_(False)
    digest_before = state_digest(host)

    use_separability = (
        args.context_sep_weight > 0 or args.host_sep_weight > 0
        or args.memory_sep_weight > 0)
    if use_separability:
        cache_root = Path(args.counterfactual_cache)
        load_or_build_counterfactual_cache(
            train_data, spec, args.task, "train", host, device, cache_root, 128)
        load_or_build_counterfactual_cache(
            val_data, spec, args.task, "validation", host, device, cache_root, 128)

    cem = CEMLeWM(args, spec, device)
    writer = CEMLeWMWriter(
        target_dim=(LATENT_DIM * cem.cue_length
                    if use_separability else LATENT_DIM),
        dim=args.dim, slots=args.slots,
        heads=args.heads, residual_scale=args.residual_scale).to(device)
    writer.set_injection_mode(args.injection)
    ctrl = CEMController(
        latent_dim=LATENT_DIM, budget=args.budget, quantile=args.quantile,
        beta=args.beta).to(device)
    params = list(writer.parameters()) + list(ctrl.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    n = len(train_data["labels"])
    idx = np.arange(n)
    rng = np.random.default_rng(1000 + args.seed)
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(idx)
        writer.train(); ctrl.train()
        ep_roll, ep_base, ep_write, ep_distill, ep_sp = [], [], [], [], []
        ep_context_sep, ep_host_sep, ep_memory_sep = [], [], []
        t0 = time.time()
        for off in range(0, n, args.batch_size):
            rows = idx[off:off + args.batch_size]
            if len(rows) < 4:
                continue
            z, a = cem.batch(train_data, rows)
            targets = cem.target(train_data, rows, z)
            fused, w, surprise = cem.forward(
                writer, ctrl, host, z, a, write_from_full=True)
            ctrl.write_gate.update_threshold(surprise)
            roll = rollout_loss(host, cem, fused, a[:, cem.context], targets)
            with torch.no_grad():
                base = rollout_loss(
                    host, cem, z[:, cem.context], a[:, cem.context], targets)
            write_cost = ctrl.write_cost(surprise)
            loss = roll + write_cost
            context_sep = host_sep = memory_sep = torch.tensor(
                0.0, device=device)
            if use_separability:
                candidates = evidence_targets(
                    train_data, rows, device, spec=spec,
                    target_mode="counterfactual_delta_flat",
                    candidate_count=6, shuffle_targets=False)
                slots = writer._slots_weighted(
                    writer._tokens(
                        z[:, cem.prefix_idx], a[:, cem.prefix_idx],
                        torch.as_tensor(cem.prefix_idx, device=device)),
                    w)
                context_query = writer.query_context(fused[:, -1])
                host_query = writer.query_host(
                    host_predict(host, fused, a[:, cem.context])[:, -1])
                memory_query = writer.decode_evidence(slots)
                context_sep, _ = contrastive_loss(
                    context_query, candidates, args.temperature)
                host_sep, _ = contrastive_loss(
                    host_query, candidates, args.temperature)
                memory_sep, _ = contrastive_loss(
                    memory_query, candidates, args.temperature)
                context_sep = context_sep + positive_cosine_loss(
                    context_query, candidates)
                host_sep = host_sep + positive_cosine_loss(
                    host_query, candidates)
                memory_sep = memory_sep + positive_cosine_loss(
                    memory_query, candidates)
                loss = loss + (
                    args.context_sep_weight * context_sep
                    + args.host_sep_weight * host_sep
                    + args.memory_sep_weight * memory_sep)
            distill = torch.tensor(0.0, device=device)
            sp = float("nan")
            if step % args.distill_every == 0:
                # Normalized true CE by real deletion.  The no-memory
                # denominator makes calibration comparable across episodes.
                base_ps = rollout_loss_per_sample(
                    host, cem, fused, a[:, cem.context], targets).detach()
                with torch.no_grad():
                    no_mem_ps = rollout_loss_per_sample(
                        host, cem, z[:, cem.context],
                        a[:, cem.context], targets).clamp_min(1e-8)
                cand = list(range(1, cem.cue_start + cem.cue_length + 1))
                cand += [int(x) for x in cem.context[:1]]
                cand = sorted(set(c for c in cand if c < cem.decision))
                ce_true_cols, ce_hat_cols = [], []
                for j in cand:
                    fused_j, _, _ = cem.forward(
                        writer, ctrl, host, z, a, write_from_full=True,
                        deletion_frame=j)
                    with torch.no_grad():
                        del_ps = rollout_loss_per_sample(
                            host, cem, fused_j, a[:, cem.context], targets)
                    ce_true_j = ((del_ps - base_ps) / no_mem_ps).detach()
                    age_norm = torch.full((z.shape[0],),
                                          (cem.decision - j) / 20.0, device=device)
                    ce_hat_j = ctrl.ce_hat(z[:, j], age_norm, surprise[:, j])
                    ce_true_cols.append(ce_true_j)
                    ce_hat_cols.append(ce_hat_j)
                if args.group_ce:
                    cue_group = list(range(
                        cem.cue_start, cem.cue_start + cem.cue_length))
                    noncue = [
                        j for j in range(1, cem.decision)
                        if j not in cue_group]
                    width = len(cue_group)
                    distractor_groups = [
                        noncue[k:k + width]
                        for k in range(0, len(noncue) - width + 1, width)]
                    groups = [cue_group]
                    if distractor_groups:
                        groups.append(
                            distractor_groups[step % len(distractor_groups)])
                    for group in groups:
                        fused_g, _, _ = cem.forward(
                            writer, ctrl, host, z, a, write_from_full=True,
                            deletion_frame=group)
                        with torch.no_grad():
                            del_g = rollout_loss_per_sample(
                                host, cem, fused_g,
                                a[:, cem.context], targets)
                        ce_true_cols.append(
                            ((del_g - base_ps) / no_mem_ps).detach())
                        group_z = z[:, group].mean(dim=1)
                        group_surprise = surprise[:, group].mean(dim=1)
                        group_age = torch.full(
                            (z.shape[0],),
                            (cem.decision - float(np.mean(group))) / 20.0,
                            device=device)
                        ce_hat_cols.append(ctrl.ce_hat(
                            group_z, group_age, group_surprise))
                ce_true = torch.stack(ce_true_cols, 1)
                ce_hat = torch.stack(ce_hat_cols, 1)
                distill = ctrl.distillation_loss(ce_hat, ce_true)
                sp = spearman(ce_hat.mean(0), ce_true.mean(0))
                loss = loss + args.distill_weight * distill
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            ep_roll.append(float(roll.detach()))
            ep_base.append(float(base))
            ep_write.append(float(write_cost.detach()))
            ep_context_sep.append(float(context_sep.detach()))
            ep_host_sep.append(float(host_sep.detach()))
            ep_memory_sep.append(float(memory_sep.detach()))
            if step % args.distill_every == 0:
                ep_distill.append(float(distill.detach()))
                if not np.isnan(sp):
                    ep_sp.append(sp)
        sched.step()
        history.append({
            "epoch": epoch,
            "rollout_with_mem": float(np.mean(ep_roll)),
            "rollout_no_mem": float(np.mean(ep_base)),
            "write_cost": float(np.mean(ep_write)),
            "distill": float(np.mean(ep_distill)) if ep_distill else None,
            "spearman_ce": float(np.mean(ep_sp)) if ep_sp else None,
            "context_sep": float(np.mean(ep_context_sep)),
            "host_sep": float(np.mean(ep_host_sep)),
            "memory_sep": float(np.mean(ep_memory_sep)),
            "seconds": time.time() - t0,
        })
        print(f"[cem-lewm] {args.task} s{args.seed} age{args.age} ep{epoch}/"
              f"{args.epochs} roll_mem={history[-1]['rollout_with_mem']:.4f} "
              f"roll_nomem={history[-1]['rollout_no_mem']:.4f} "
              f"sp={history[-1]['spearman_ce']} sec={history[-1]['seconds']:.1f}",
              flush=True)

    digest_after = state_digest(host)
    if digest_before != digest_after:
        raise RuntimeError("FROZEN LeWM host changed during CEM training")

    result = evaluate(args, cem, writer, ctrl, host, train_data, val_data, spec)
    result["frozen_host_digest_unchanged"] = True
    result["history"] = history
    write_outputs(args, result)
    return result


@torch.no_grad()
def host_output_features(cem, writer, ctrl, host, data, condition):
    device = cem.device
    feats = []
    n = len(data["labels"])
    for off in range(0, n, 256):
        rows = np.arange(off, min(n, off + 256))
        z, a = cem.batch(data, rows)
        if condition in ("no_state", "host_only"):
            out = host_predict(host, z[:, cem.context], a[:, cem.context])[:, -1]
        else:
            fused, _, _ = cem.forward(
                writer, ctrl, host, z, a,
                write_from_full=(condition not in ("reset",)))
            if condition == "shuffled":
                fused = fused.roll(1, dims=0)
            elif condition == "random":
                gen = torch.Generator(device=device).manual_seed(9100 + off)
                fused = z[:, cem.context] + 0.05 * torch.randn(
                    fused.shape, generator=gen, device=device)
            out = host_predict(host, fused, a[:, cem.context])[:, -1]
        feats.append(out.float().cpu().numpy())
    return np.concatenate(feats)


def evaluate(args, cem, writer, ctrl, host, train_data, val_data, spec):
    writer.eval(); ctrl.eval()
    device = cem.device
    # ---- CEM endpoint: host future-latent loss with vs without memory ---- #
    with torch.no_grad():
        roll_mem, roll_nomem, sur_accum = [], [], []
        n = len(val_data["labels"])
        roll_reset = []
        for off in range(0, n, 256):
            rows = np.arange(off, min(n, off + 256))
            z, a = cem.batch(val_data, rows)
            targets = cem.target(val_data, rows, z)
            fused, w, surprise = cem.forward(
                writer, ctrl, host, z, a, write_from_full=True)
            fused_r, _, _ = cem.forward(
                writer, ctrl, host, z, a, write_from_full=False)
            roll_mem.append(float(rollout_loss(
                host, cem, fused, a[:, cem.context], targets)))
            roll_reset.append(float(rollout_loss(
                host, cem, fused_r, a[:, cem.context], targets)))
            roll_nomem.append(float(rollout_loss(
                host, cem, z[:, cem.context], a[:, cem.context], targets)))
            full_s = torch.zeros(z.shape[0], 20, device=device)
            full_s[:, :cem.decision] = surprise
            sur_accum.append(full_s.cpu().numpy())
    endpoint = {
        "endpoint_mode": cem.endpoint,
        "host_future_latent_loss_with_memory": float(np.mean(roll_mem)),
        "host_future_latent_loss_without_memory": float(np.mean(roll_nomem)),
        "host_future_latent_loss_reset_memory": float(np.mean(roll_reset)),
        "improvement": float(np.mean(roll_nomem) - np.mean(roll_mem)),
        "cue_specific_improvement": float(np.mean(roll_reset) - np.mean(roll_mem)),
    }
    frame_sur = np.mean(np.concatenate(sur_accum), axis=0).tolist()

    # ---- fail-closed audit (labels used ONLY here) ---- #
    ty = train_data["labels"]
    vy = val_data["labels"]
    classes = int(len(np.unique(ty)))
    train_full = host_output_features(cem, writer, ctrl, host, train_data, "full")
    audit = {}
    for cond in ("full", "reset", "no_state", "host_only",
                 "shuffled", "random"):
        vf = host_output_features(cem, writer, ctrl, host, val_data, cond)
        pred = fit_classifier(train_full, ty, vf)
        bacc = balanced_acc(pred, vy, classes)
        audit[cond] = float(bacc)
    ctrl_max = 1.0 / classes + 0.10
    audit["passed"] = bool(audit["full"] >= 0.75
                           and audit["reset"] <= ctrl_max
                           and audit["no_state"] <= ctrl_max)
    audit["classes"] = classes

    # ---- causal-deletion: high-CE-hat vs random ---- #
    deletion, events, spearman_val = causal_deletion(
        cem, writer, ctrl, host, val_data)

    # ---- host-sensitivity ceiling: is the cue even in the host's own latents?
    ceiling = host_sensitivity_ceiling(train_data, val_data, cem)
    ladder = diagnostic_ladder(
        cem, writer, ctrl, host, train_data, val_data)

    return {
        "schema": "cem_lewm_cell_v1",
        "host": "lewm",
        "env": args.task,
        "seed": int(args.seed),
        "age": int(args.age),
        "cue_window": [cem.cue_start, cem.cue_start + cem.cue_length],
        "readout_t": cem.decision,
        "endpoint": endpoint,
        "audit": audit,
        "causal_deletion": deletion,
        "surrogate_spearman": spearman_val,
        "host_sensitivity_ceiling": ceiling,
        "diagnostic_ladder": ladder,
        "frame_surprise": frame_sur,
        "events": events,
        "labels_used_for_training_loss": False,
    }


@torch.no_grad()
def diagnostic_ladder(cem, writer, ctrl, host, train_data, val_data):
    """Post-hoc readability across the exposure boundary."""
    def features(data):
        result = {"cue_latent": [], "memory_only": [],
                  "injected_context": [], "host_output": []}
        for off in range(0, len(data["labels"]), 256):
            rows = np.arange(off, min(len(data["labels"]), off + 256))
            z, a = cem.batch(data, rows)
            fused, w, _ = cem.forward(
                writer, ctrl, host, z, a, write_from_full=True)
            tokens = writer._tokens(
                z[:, cem.prefix_idx], a[:, cem.prefix_idx],
                torch.as_tensor(cem.prefix_idx, device=cem.device))
            slots = writer._slots_weighted(tokens, w)
            result["cue_latent"].append(
                z[:, cem.cue_start:cem.cue_start + cem.cue_length]
                .flatten(1).cpu().numpy())
            result["memory_only"].append(slots.flatten(1).cpu().numpy())
            result["injected_context"].append(
                fused.flatten(1).cpu().numpy())
            result["host_output"].append(
                host_predict(host, fused, a[:, cem.context])[:, -1]
                .cpu().numpy())
        return {k: np.concatenate(v) for k, v in result.items()}

    train_f = features(train_data)
    val_f = features(val_data)
    classes = int(len(np.unique(train_data["labels"])))
    ladder = {}
    for level in train_f:
        pred = fit_classifier(
            train_f[level], train_data["labels"], val_f[level])
        ladder[level] = balanced_acc(
            pred, val_data["labels"], classes)
    return ladder


def host_sensitivity_ceiling(train_data, val_data, cem):
    """Linear-probe the cue label from RAW host latents (labels post-hoc).

    Reports whether the cue survives into the host's OWN latents at the cue
    frame vs the post-window decision frame -- the interface ceiling any
    label-free host-loss policy is bounded by.
    """
    ty, vy = train_data["labels"], val_data["labels"]
    classes = int(len(np.unique(ty)))
    out = {}
    for name, fr in (("cue_frame", cem.cue_start + cem.cue_length - 1),
                     ("decision_frame", cem.decision)):
        pred = fit_classifier(train_data["z"][:, fr], ty, val_data["z"][:, fr])
        out[f"raw_latent_classBAcc_{name}"] = balanced_acc(pred, vy, classes)
    out["interpretation"] = (
        "cue present at cue frame but absent at decision frame => the frozen "
        "host discards the transient cue from its own rollout (interface null)"
        if out["raw_latent_classBAcc_decision_frame"] < 0.4
        else "cue survives into the host's own decision latent")
    return out


def balanced_acc(pred, truth, classes):
    vals = []
    for c in range(classes):
        m = truth == c
        if np.any(m):
            vals.append(float(np.mean(pred[m] == c)))
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def causal_deletion(cem, writer, ctrl, host, data):
    device = cem.device
    rows = np.arange(min(256, len(data["labels"])))
    z, a = cem.batch(data, rows)
    targets = cem.target(data, rows, z)
    fused, w, surprise = cem.forward(
        writer, ctrl, host, z, a, write_from_full=True)
    base_ps = rollout_loss_per_sample(host, cem, fused, a[:, cem.context], targets)
    no_mem_ps = rollout_loss_per_sample(
        host, cem, z[:, cem.context],
        a[:, cem.context], targets).clamp_min(1e-8)
    frames = list(range(1, cem.decision))
    ce_true_by_frame = {}
    ce_hat_by_frame = {}
    for j in frames:
        fused_j, _, _ = cem.forward(
            writer, ctrl, host, z, a, write_from_full=True, deletion_frame=j)
        del_ps = rollout_loss_per_sample(host, cem, fused_j, a[:, cem.context], targets)
        ce_true_by_frame[j] = float((del_ps - base_ps).mean())
        age_norm = torch.full((z.shape[0],), (cem.decision - j) / 20.0,
                              device=device)
        ce_hat_by_frame[j] = float(
            ctrl.ce_hat(z[:, j], age_norm, surprise[:, j]).mean())
    ce_hat_arr = torch.tensor([ce_hat_by_frame[j] for j in frames])
    ce_true_arr = torch.tensor([ce_true_by_frame[j] for j in frames])
    sp = spearman(ce_hat_arr, ce_true_arr)
    order = sorted(frames, key=lambda j: ce_hat_by_frame[j], reverse=True)
    k = ctrl.budget
    top = order[:k]
    rng = np.random.default_rng(7)
    rand = list(rng.choice(frames, size=k, replace=False))
    deletion = {
        "delta_loss_delete_high_ce_hat": float(np.mean([ce_true_by_frame[j] for j in top])),
        "delta_loss_delete_random": float(np.mean([ce_true_by_frame[j] for j in rand])),
        "high_ce_frames": [int(j) for j in top],
        "random_frames": [int(j) for j in rand],
    }
    deletion["high_ce_hurts_more_than_random"] = bool(
        deletion["delta_loss_delete_high_ce_hat"]
        > deletion["delta_loss_delete_random"])

    # ---- GROUP / set-level deletion (fixes per-item under-crediting of a
    #      redundant cue shown across several frames: the spec's count-matched
    #      design). Delete the whole cue window vs a matched random group. ----
    cue_frames = [f for f in range(cem.cue_start, cem.cue_start + cem.cue_length)
                  if f in frames]
    if cue_frames:
        fused_cue, _, _ = cem.forward(
            writer, ctrl, host, z, a, write_from_full=True,
            deletion_frame=cue_frames)
        cue_ps = rollout_loss_per_sample(
            host, cem, fused_cue, a[:, cem.context], targets)
        noncue = [f for f in frames if f not in cue_frames]
        rgrp = list(rng.choice(noncue, size=min(len(cue_frames), len(noncue)),
                               replace=False))
        fused_rg, _, _ = cem.forward(
            writer, ctrl, host, z, a, write_from_full=True, deletion_frame=rgrp)
        rg_ps = rollout_loss_per_sample(
            host, cem, fused_rg, a[:, cem.context], targets)
        deletion["group_delete_cue_window"] = float((cue_ps - base_ps).mean())
        deletion["group_delete_random"] = float((rg_ps - base_ps).mean())
        deletion["cue_group_hurts_more_than_random"] = bool(
            deletion["group_delete_cue_window"] > deletion["group_delete_random"])
        deletion["cue_window_frames"] = [int(f) for f in cue_frames]

        # Group-level calibrated KEEP: cue group plus disjoint matched-size
        # distractor groups, all scored with normalized joint deletion CE.
        groups = [cue_frames]
        for start in range(0, len(noncue), len(cue_frames)):
            group = noncue[start:start + len(cue_frames)]
            if len(group) == len(cue_frames):
                groups.append(group)
        group_true, group_hat = [], []
        for group in groups:
            fused_g, _, _ = cem.forward(
                writer, ctrl, host, z, a, write_from_full=True,
                deletion_frame=group)
            loss_g = rollout_loss_per_sample(
                host, cem, fused_g, a[:, cem.context], targets)
            group_true.append(float(
                ((loss_g - base_ps) / no_mem_ps).mean()))
            group_z = z[:, group].mean(dim=1)
            group_age = torch.full(
                (z.shape[0],),
                (cem.decision - float(np.mean(group))) / 20.0,
                device=device)
            group_hat.append(float(ctrl.ce_hat(
                group_z, group_age,
                surprise[:, group].mean(dim=1)).mean()))
        group_true_t = torch.tensor(group_true)
        group_hat_t = torch.tensor(group_hat)
        comparable = 0
        correct = 0
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                if abs(group_true[left] - group_true[right]) > 1e-8:
                    comparable += 1
                    correct += int(
                        (group_true[left] > group_true[right])
                        == (group_hat[left] > group_hat[right]))
        predicted = int(torch.argmax(group_hat_t))
        random_index = 1 if len(groups) > 1 else 0
        deletion["group_ce"] = {
            "groups": [[int(x) for x in group] for group in groups],
            "ce_true_norm": group_true,
            "ce_hat": group_hat,
            "spearman": spearman(group_hat_t, group_true_t),
            "pairwise_accuracy": (
                float(correct / comparable) if comparable else None),
            "predicted_high_group": groups[predicted],
            "predicted_high_delete_delta_norm": group_true[predicted],
            "random_group": groups[random_index],
            "random_delete_delta_norm": group_true[random_index],
            "predicted_hurts_more_than_random": bool(
                group_true[predicted] > group_true[random_index]),
        }

    # ---- build decision-log events ---- #
    # WRITE admits frames by host surprise (self-calibrating quantile); we log
    # the top-K most-surprising candidates so KEEP's evict decisions (down to the
    # budget, by amortized causal value ce_hat) are visible as abandoned slots.
    mean_sur = surprise.mean(0)
    thr = ctrl.write_gate.threshold
    by_sur = sorted(frames, key=lambda j: float(mean_sur[j]), reverse=True)
    topk = by_sur[:min(10, len(frames))]
    written = sorted(set([j for j in frames if float(mean_sur[j]) > float(thr)]
                         + topk))
    if not written:
        written = order[:k]
    kept = sorted(written, key=lambda j: ce_hat_by_frame[j], reverse=True)[:k]
    events = []
    accepted = kept[0] if kept else None
    for j in written:
        status = "kept" if j in kept else "evicted"
        ev = {
            "slot_id": int(j),
            "written_at": int(j),
            "cue_timestamp": float(j) / 19.0,
            "surprise_at_write": float(mean_sur[j]),
            "ce_hat": float(ce_hat_by_frame[j]),
            "ce_true": float(ce_true_by_frame[j]),
            "status": status,
            "evicted_at": int(cem.decision) if status == "evicted" else None,
            "retrieved_at": None,
            "verify_delta": None,
        }
        events.append(ev)
    # retrieve-then-verify at readout for the router's top-1 kept slot
    if accepted is not None:
        verify_delta = ce_true_by_frame[accepted]
        for ev in events:
            if ev["slot_id"] == accepted:
                if verify_delta > ctrl.verify_delta:
                    ev["status"] = "retrieved"
                    ev["retrieved_at"] = int(cem.decision)
                else:
                    ev["status"] = "rejected"
                ev["verify_delta"] = float(verify_delta)
    return deletion, events, sp


def write_outputs(args, result):
    out = Path(args.output) / result["env"] / f"s{result['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    decision_log = {
        "host": result["host"],
        "env": result["env"],
        "seed": result["seed"],
        "cue_window": result["cue_window"],
        "readout_t": result["readout_t"],
        "frame_surprise": result["frame_surprise"],
        "events": result["events"],
    }
    (out / "decision_log.json").write_text(json.dumps(decision_log, indent=2))
    summary = {
        "schema": "cem_summary_v1",
        "host": result["host"],
        "env": result["env"],
        "seed": result["seed"],
        "age": result["age"],
        "endpoint_mode": result["endpoint"].get("endpoint_mode"),
        "host_loss_with_memory": result["endpoint"]["host_future_latent_loss_with_memory"],
        "host_loss_without_memory": result["endpoint"]["host_future_latent_loss_without_memory"],
        "host_loss_reset_memory": result["endpoint"].get("host_future_latent_loss_reset_memory"),
        "endpoint_improvement": result["endpoint"]["improvement"],
        "cue_specific_improvement": result["endpoint"].get("cue_specific_improvement"),
        "audit": result["audit"],
        "causal_deletion": result["causal_deletion"],
        "surrogate_spearman": result["surrogate_spearman"],
        "host_sensitivity_ceiling": result.get("host_sensitivity_ceiling"),
        "diagnostic_ladder": result.get("diagnostic_ladder"),
        "v3_config": {
            "injection": args.injection,
            "context_sep_weight": args.context_sep_weight,
            "host_sep_weight": args.host_sep_weight,
            "memory_sep_weight": args.memory_sep_weight,
            "group_ce": args.group_ce,
        },
        "frozen_host_digest_unchanged": result["frozen_host_digest_unchanged"],
        "labels_used_for_training_loss": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    try:
        rel = (out / "summary.json").resolve().relative_to(ROOT)
    except ValueError:
        rel = out / "summary.json"
    print(json.dumps({"summary": str(rel),
                      "endpoint": summary["host_loss_with_memory"],
                      "no_mem": summary["host_loss_without_memory"],
                      "audit": result["audit"],
                      "causal_deletion": result["causal_deletion"][
                          "high_ce_hurts_more_than_random"]}, indent=2), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="transient-visual-token-recall",
                   choices=["transient-visual-token-recall",
                            "multi-item-visual-binding-recall"])
    p.add_argument("--age", type=int, default=15, choices=[4, 8, 15])
    p.add_argument("--endpoint", default="cue_conditioned",
                   choices=["cue_conditioned", "own_rollout"],
                   help="cue_conditioned = recall/audit readout target; "
                        "own_rollout = strict host next-latent (cue-free null)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--slots", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--quantile", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=1.0e-2)
    p.add_argument("--distill-every", type=int, default=8)
    p.add_argument("--distill-weight", type=float, default=1.0)
    p.add_argument("--group-ce", action="store_true")
    p.add_argument("--injection", choices=["generic", "query"],
                   default="query")
    p.add_argument("--context-sep-weight", type=float, default=0.0)
    p.add_argument("--host-sep-weight", type=float, default=0.0)
    p.add_argument("--memory-sep-weight", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--counterfactual-cache",
                   default=str(DEFAULT_COUNTERFACTUAL_CACHE))
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
