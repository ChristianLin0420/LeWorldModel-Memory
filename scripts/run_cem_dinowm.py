#!/usr/bin/env python3
"""Causal-Effect Memory (CEM) on the FROZEN official DINO-WM Wall host.

Reuses the Stage-H DINO feature cache (``outputs/dinowm_wall_audit_v1/
stage_h_carriers/cache``) and the frozen one-step Wall predictor.  Only the CEM
controller and a spatial residual writer train; the predictor / proprio / action
encoders stay frozen (a state-dict digest is asserted unchanged).

Endpoint: the frozen host's OWN forward-pass latent loss toward the cue-carrying
target latent, with vs without memory injected at a post-window readout (the
cue overlay lives only on frames 1..LAST_CUE_FRAME, so at the readout memory is
the sole carrier).  WRITE is the label-free host-surprise gate; KEEP is the
amortized ``ce_head`` distilled against true hard-deletion CE; RECALL is
retrieve-then-verify.

Outputs (shared visualization schema):
  outputs/cem_dinowm_v1/<env>/s<seed>/decision_log.json
  outputs/cem_dinowm_v1/<env>/s<seed>/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cem_controller import CEMController, spearman  # noqa: E402
from scripts.run_dinowm_wall_stage_g import (  # noqa: E402
    AGES, CLASSES, DEFAULT_CHECKPOINT, DEFAULT_VENDOR, LAST_CUE_FRAME, LENGTH,
)
from scripts.run_dinowm_wall_stage_h import (  # noqa: E402
    DEFAULT_OUTPUT as STAGE_H_OUTPUT,
    FrozenWallHost, WallFeatureBank,
)
from scripts.run_mem_jepa_stage_b import fit_classifier  # noqa: E402

TOKENS = 196
DIM = 384
DEFAULT_OUTPUT = ROOT / "outputs/cem_dinowm_v1"


def host_digest(host: FrozenWallHost) -> str:
    h = hashlib.sha256()
    for module in (host.predictor, host.action_encoder, host.proprio_encoder):
        for name, value in sorted(module.state_dict().items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SpatialCEMWriter(nn.Module):
    """Frozen-host residual writer with a surprise-gated WRITE over the prefix.

    Memory frames are pooled to per-frame vectors and assigned to ``slots``
    (WRITE gate multiplies the assignment logits, so only high-surprise frames
    are admitted).  The context frame's 196 tokens cross-attend to the slots to
    produce a spatial residual ``delta``; the frozen host consumes
    ``fused = context + residual_scale * gate * delta``.
    """

    def __init__(self, dim: int = 256, slots: int = 6, heads: int = 4,
                 max_frames: int = LENGTH) -> None:
        super().__init__()
        self.mem_proj = nn.Linear(DIM, dim)
        self.time = nn.Embedding(max_frames, dim)
        self.assign = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, slots))
        self.q_proj = nn.Linear(DIM, dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.writer = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, DIM))
        self.gate = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 1))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.writer[-1].weight)
        nn.init.zeros_(self.writer[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)

    def inject_gated(self, mem_vecs, mem_times, context_tokens, write_gate):
        m = self.mem_proj(mem_vecs) + self.time(mem_times)[None]
        logits = self.assign(m).transpose(1, 2)  # (B, S, P)
        logits = logits + torch.log(write_gate.clamp_min(1e-6))[:, None, :]
        w = torch.softmax(logits, dim=-1)
        slots = torch.einsum("bsp,bpd->bsd", w, m)
        q = self.q_proj(context_tokens)
        attended, _ = self.cross(q, slots, slots, need_weights=False)
        delta = self.writer(attended)
        gate = torch.sigmoid(self.gate(attended))
        fused = context_tokens + self.residual_scale * gate * delta
        return fused, slots


def tt(a, device):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


@torch.no_grad()
def frame_surprise(host, visual, proprio, actions, upto):
    """Host one-step self-surprise per frame over its legal one-step window."""
    bsz = visual.shape[0]
    out = torch.zeros(bsz, upto, device=visual.device)
    for t in range(1, upto):
        pred = host.predict(visual[:, t - 1:t], proprio[:, t - 1:t],
                            actions[:, t - 1:t])[:, :, :, :DIM]
        out[:, t] = ((pred[:, 0] - visual[:, t]) ** 2).mean(dim=(1, 2))
    return out


class CEMDino:
    def __init__(self, args, device):
        self.age = int(args.age)
        self.device = device
        self.patch_policy = args.patch_policy
        self.endpoint = LAST_CUE_FRAME + self.age
        self.context = self.endpoint - 1
        self.prefix = np.arange(0, self.context, dtype=np.int64)

    def memory_and_context(self, bank, expanded, condition):
        """Return (mem_visual_seq, ctx_tokens, proprio, actions, target)."""
        bases, labels = bank.decode(expanded)
        device = self.device
        cued = tt(bank.visual(expanded), device)          # (B,L,196,384) cue on 1..3
        base = tt(np.asarray(bank.base[bases], dtype=np.float32), device)
        proprio = tt(bank.proprio[bases], device)
        actions = tt(bank.actions[bases], device)
        # target: the cue-carrying last-cue-frame latent (depends on the cue)
        target = tt(bank.cue[bases, labels][:, LAST_CUE_FRAME - 1], device)
        if condition == "full":
            mem_seq = cued
        else:  # reset: memory built from cue-free base frames
            mem_seq = base
        ctx_tokens = base[:, self.context]                 # cue-free context frame
        return mem_seq, ctx_tokens, proprio, actions, target, cued

    def forward(self, writer, ctrl, host, bank, expanded, condition,
                deletion_frame=None):
        mem_seq, ctx_tokens, proprio, actions, target, cued = \
            self.memory_and_context(bank, expanded, condition)
        surprise = frame_surprise(host, cued, proprio, actions, self.context)
        semantic = mem_seq[:, self.prefix]  # frozen DINOv2 patch tokens
        if self.patch_policy == "random":
            gen = torch.Generator(device=self.device).manual_seed(4400)
            patch_idx = torch.randint(
                0, semantic.shape[2], (len(self.prefix),),
                generator=gen, device=self.device)
            mem_vecs = semantic[
                :, torch.arange(len(self.prefix), device=self.device),
                patch_idx]
        elif self.patch_policy == "surprise_semantic":
            # Select semantic patches with largest label-free temporal change;
            # frame WRITE timing remains the frozen host surprise gate below.
            previous = torch.cat(
                (semantic[:, :1], semantic[:, :-1]), dim=1)
            change = (semantic - previous).square().mean(dim=-1)
            top = torch.topk(
                change, k=min(8, semantic.shape[2]), dim=2).indices
            mem_vecs = torch.gather(
                semantic, 2,
                top[..., None].expand(-1, -1, -1, semantic.shape[-1])
            ).mean(dim=2)
        else:
            mem_vecs = semantic.mean(dim=2)
        mem_times = torch.as_tensor(self.prefix, device=self.device)
        w = ctrl.write_gate.soft_gate(surprise[:, self.prefix])
        if deletion_frame is not None:
            w = w.clone()
            if isinstance(deletion_frame, (list, tuple, np.ndarray)):
                for j in deletion_frame:
                    w[:, int(j)] = 0.0
            else:
                w[:, deletion_frame] = 0.0
        if condition == "no_state":
            fused = ctx_tokens
        else:
            fused, _ = writer.inject_gated(mem_vecs, mem_times, ctx_tokens, w)
        pred = host.predict(fused.unsqueeze(1), proprio[:, self.context:self.context + 1],
                            actions[:, self.context:self.context + 1])[:, 0, :, :DIM]
        return pred, target, surprise, w


def host_loss(pred, target):
    return F.mse_loss(pred, target)


def host_loss_per_sample(pred, target):
    return ((pred - target) ** 2).mean(dim=(1, 2))


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class _A:
        vendor = ROOT / "third_party/dino_wm"
        checkpoint = STAGE_H_OUTPUT.parent / "checkpoint"
    # Build the frozen host via stage-H helper args
    host = _load_host(args, device)
    digest_before = host_digest(host)
    bank = WallFeatureBank(Path(args.stage_h_output))

    cem = CEMDino(args, device)
    writer = SpatialCEMWriter(dim=args.dim, slots=args.slots,
                              heads=args.heads).to(device)
    ctrl = CEMController(latent_dim=DIM, budget=args.budget,
                         quantile=args.quantile, beta=args.beta).to(device)
    params = list(writer.parameters()) + list(ctrl.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    train_idx = bank.expanded_indices("train")
    rng = np.random.default_rng(2000 + args.seed)
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_idx))
        writer.train(); ctrl.train()
        ep_full, ep_reset, ep_write, ep_sp = [], [], [], []
        t0 = time.time()
        for off in range(0, len(order), args.batch_size):
            expanded = train_idx[order[off:off + args.batch_size]]
            if len(expanded) < 4:
                continue
            pred, target, surprise, w = cem.forward(
                writer, ctrl, host, bank, expanded, "full")
            ctrl.write_gate.update_threshold(surprise[:, cem.prefix])
            loss_full = host_loss(pred, target)
            with torch.no_grad():
                pred_r, _, _, _ = cem.forward(
                    writer, ctrl, host, bank, expanded, "reset")
                loss_reset = host_loss(pred_r, target)
            write_cost = ctrl.write_cost(surprise[:, cem.prefix])
            loss = loss_full + write_cost
            sp = float("nan")
            if step % args.distill_every == 0:
                base_ps = host_loss_per_sample(pred, target).detach()
                with torch.no_grad():
                    no_mem_ps = host_loss_per_sample(
                        pred_r, target).clamp_min(1e-8)
                cand = [int(j) for j in cem.prefix
                        if 1 <= j <= LAST_CUE_FRAME]
                cand += [int(cem.prefix[-1])]
                cand = sorted(set(cand))
                ce_true_cols, ce_hat_cols = [], []
                mem_vecs = None
                for j in cand:
                    pred_j, tgt_j, sur_j, _ = cem.forward(
                        writer, ctrl, host, bank, expanded, "full",
                        deletion_frame=list(cem.prefix).index(j))
                    with torch.no_grad():
                        del_ps = host_loss_per_sample(pred_j, tgt_j)
                    ce_true_cols.append(
                        ((del_ps - base_ps) / no_mem_ps).detach())
                    pooled = tt(bank.visual(expanded), device)[:, j].mean(dim=1)
                    age_norm = torch.full((len(expanded),),
                                          (cem.context - j) / float(LENGTH),
                                          device=device)
                    ce_hat_cols.append(ctrl.ce_hat(pooled, age_norm, sur_j[:, j]))
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
            ep_full.append(float(loss_full.detach()))
            ep_reset.append(float(loss_reset))
            ep_write.append(float(write_cost.detach()))
            if not np.isnan(sp):
                ep_sp.append(sp)
        sched.step()
        history.append({
            "epoch": epoch,
            "host_loss_with_memory": float(np.mean(ep_full)),
            "host_loss_reset": float(np.mean(ep_reset)),
            "write_cost": float(np.mean(ep_write)),
            "spearman_ce": float(np.mean(ep_sp)) if ep_sp else None,
            "seconds": time.time() - t0,
        })
        print(f"[cem-dino] wall s{args.seed} age{args.age} ep{epoch}/{args.epochs} "
              f"loss_mem={history[-1]['host_loss_with_memory']:.5f} "
              f"loss_reset={history[-1]['host_loss_reset']:.5f} "
              f"sp={history[-1]['spearman_ce']} sec={history[-1]['seconds']:.1f}",
              flush=True)

    digest_after = host_digest(host)
    if digest_before != digest_after:
        raise RuntimeError("FROZEN DINO-WM Wall host changed during CEM training")

    result = evaluate(args, cem, writer, ctrl, host, bank)
    result["frozen_host_digest_unchanged"] = True
    result["history"] = history
    write_outputs(args, result)
    return result


def _load_host(args, device):
    ns = argparse.Namespace(
        vendor=Path(args.vendor), checkpoint=Path(args.checkpoint))
    return FrozenWallHost(ns, device)


@torch.no_grad()
def host_output_features(cem, writer, ctrl, host, bank, split, condition):
    idx = bank.expanded_indices(split)
    feats = []
    for off in range(0, len(idx), 64):
        expanded = idx[off:off + 64]
        pred, _, _, _ = cem.forward(writer, ctrl, host, bank, expanded, condition)
        feats.append(pred.mean(dim=1).float().cpu().numpy())  # pool tokens -> 384
    return np.concatenate(feats)


def evaluate(args, cem, writer, ctrl, host, bank):
    writer.eval(); ctrl.eval()
    device = cem.device
    # ---- endpoint: host latent loss with vs without memory ---- #
    with torch.no_grad():
        idx = bank.expanded_indices("validation")
        lm, lr, ln, sur = [], [], [], []
        for off in range(0, len(idx), 64):
            expanded = idx[off:off + 64]
            pf, tf, s, _ = cem.forward(writer, ctrl, host, bank, expanded, "full")
            pr, _, _, _ = cem.forward(writer, ctrl, host, bank, expanded, "reset")
            pn, _, _, _ = cem.forward(writer, ctrl, host, bank, expanded, "no_state")
            lm.append(float(host_loss(pf, tf)))
            lr.append(float(host_loss(pr, tf)))
            ln.append(float(host_loss(pn, tf)))
            full_s = torch.zeros(len(expanded), LENGTH, device=device)
            full_s[:, :cem.context] = s[:, :cem.context]
            sur.append(full_s.cpu().numpy())
    endpoint = {
        "host_future_latent_loss_with_memory": float(np.mean(lm)),
        "host_future_latent_loss_without_memory": float(np.mean(lr)),
        "host_future_latent_loss_no_state": float(np.mean(ln)),
        "improvement": float(np.mean(lr) - np.mean(lm)),
    }
    frame_sur = np.mean(np.concatenate(sur), axis=0).tolist()

    # ---- fail-closed audit (labels ONLY here) ---- #
    ty = bank.labels("train")
    vy = bank.labels("validation")
    train_full = host_output_features(cem, writer, ctrl, host, bank, "train", "full")
    audit = {}
    for cond in ("full", "reset", "no_state"):
        vf = host_output_features(cem, writer, ctrl, host, bank, "validation", cond)
        pred = fit_classifier(train_full, ty, vf)
        audit[cond] = balanced_acc(pred, vy, CLASSES)
    ctrl_max = 1.0 / CLASSES + 0.10
    audit["passed"] = bool(audit["full"] >= 0.75
                           and audit["reset"] <= ctrl_max
                           and audit["no_state"] <= ctrl_max)
    audit["classes"] = CLASSES

    deletion, events, sp = causal_deletion(cem, writer, ctrl, host, bank)
    return {
        "schema": "cem_dinowm_cell_v1",
        "host": "dinowm",
        "env": "wall",
        "seed": int(args.seed),
        "age": int(args.age),
        "cue_window": [1, LAST_CUE_FRAME],
        "readout_t": cem.endpoint,
        "endpoint": endpoint,
        "audit": audit,
        "causal_deletion": deletion,
        "surrogate_spearman": sp,
        "frame_surprise": frame_sur,
        "events": events,
        "labels_used_for_training_loss": False,
    }


def balanced_acc(pred, truth, classes):
    vals = []
    for c in range(classes):
        m = truth == c
        if np.any(m):
            vals.append(float(np.mean(pred[m] == c)))
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def causal_deletion(cem, writer, ctrl, host, bank):
    device = cem.device
    expanded = bank.expanded_indices("validation")[:128]
    pred, target, surprise, w = cem.forward(
        writer, ctrl, host, bank, expanded, "full")
    base_ps = host_loss_per_sample(pred, target)
    frames = [int(j) for j in cem.prefix if j >= 1]
    ce_true, ce_hat = {}, {}
    for j in frames:
        pj = list(cem.prefix).index(j)
        pred_j, tgt_j, sur_j, _ = cem.forward(
            writer, ctrl, host, bank, expanded, "full", deletion_frame=pj)
        del_ps = host_loss_per_sample(pred_j, tgt_j)
        ce_true[j] = float((del_ps - base_ps).mean())
        pooled = tt(bank.visual(expanded), device)[:, j].mean(dim=1)
        age_norm = torch.full((len(expanded),), (cem.context - j) / float(LENGTH),
                              device=device)
        ce_hat[j] = float(ctrl.ce_hat(pooled, age_norm, surprise[:, j]).mean())
    sp = spearman(torch.tensor([ce_hat[j] for j in frames]),
                  torch.tensor([ce_true[j] for j in frames]))
    order = sorted(frames, key=lambda j: ce_hat[j], reverse=True)
    k = ctrl.budget
    top = order[:k]
    rng = np.random.default_rng(11)
    rand = list(rng.choice(frames, size=min(k, len(frames)), replace=False))
    deletion = {
        "delta_loss_delete_high_ce_hat": float(np.mean([ce_true[j] for j in top])),
        "delta_loss_delete_random": float(np.mean([ce_true[j] for j in rand])),
        "high_ce_frames": [int(j) for j in top],
        "random_frames": [int(j) for j in rand],
    }
    deletion["high_ce_hurts_more_than_random"] = bool(
        deletion["delta_loss_delete_high_ce_hat"]
        > deletion["delta_loss_delete_random"])

    # ---- GROUP / set-level deletion: whole cue window vs matched random group.
    cue_frames = [int(j) for j in range(1, LAST_CUE_FRAME + 1) if j in frames]
    if cue_frames:
        cue_pidx = [list(cem.prefix).index(j) for j in cue_frames]
        pred_cue, tgt_cue, _, _ = cem.forward(
            writer, ctrl, host, bank, expanded, "full", deletion_frame=cue_pidx)
        cue_ps = host_loss_per_sample(pred_cue, tgt_cue)
        noncue = [j for j in frames if j not in cue_frames]
        rgrp = list(rng.choice(noncue, size=min(len(cue_frames), len(noncue)),
                               replace=False))
        rgrp_pidx = [list(cem.prefix).index(int(j)) for j in rgrp]
        pred_rg, tgt_rg, _, _ = cem.forward(
            writer, ctrl, host, bank, expanded, "full", deletion_frame=rgrp_pidx)
        rg_ps = host_loss_per_sample(pred_rg, tgt_rg)
        deletion["group_delete_cue_window"] = float((cue_ps - base_ps).mean())
        deletion["group_delete_random"] = float((rg_ps - base_ps).mean())
        deletion["cue_group_hurts_more_than_random"] = bool(
            deletion["group_delete_cue_window"] > deletion["group_delete_random"])
        deletion["cue_window_frames"] = cue_frames

    # WRITE admits frames by host surprise (self-calibrating quantile); log the
    # top-K most-surprising candidates so KEEP's evictions (to budget, by ce_hat)
    # are visible as abandoned slots in the decision log / visualization.
    mean_sur = surprise.mean(0)
    thr = float(ctrl.write_gate.threshold)
    by_sur = sorted(frames, key=lambda j: float(mean_sur[j]), reverse=True)
    topk = by_sur[:min(10, len(frames))]
    written = sorted(set([j for j in frames if float(mean_sur[j]) > thr] + topk))
    if not written:
        written = top
    kept = sorted(written, key=lambda j: ce_hat[j], reverse=True)[:k]
    events = []
    accepted = kept[0] if kept else None
    for j in written:
        status = "kept" if j in kept else "evicted"
        events.append({
            "slot_id": int(j),
            "written_at": int(j),
            "cue_timestamp": float(j) / float(LENGTH - 1),
            "surprise_at_write": float(mean_sur[j]),
            "ce_hat": float(ce_hat[j]),
            "ce_true": float(ce_true[j]),
            "status": status,
            "evicted_at": int(cem.endpoint) if status == "evicted" else None,
            "retrieved_at": None,
            "verify_delta": None,
        })
    if accepted is not None:
        vd = ce_true[accepted]
        for ev in events:
            if ev["slot_id"] == accepted:
                if vd > ctrl.verify_delta:
                    ev["status"] = "retrieved"
                    ev["retrieved_at"] = int(cem.endpoint)
                else:
                    ev["status"] = "rejected"
                ev["verify_delta"] = float(vd)
    return deletion, events, sp


def write_outputs(args, result):
    out = Path(args.output) / result["env"] / f"s{result['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    decision_log = {
        "host": result["host"], "env": result["env"], "seed": result["seed"],
        "cue_window": result["cue_window"], "readout_t": result["readout_t"],
        "frame_surprise": result["frame_surprise"], "events": result["events"],
    }
    (out / "decision_log.json").write_text(json.dumps(decision_log, indent=2))
    summary = {
        "schema": "cem_summary_v1", "host": result["host"], "env": result["env"],
        "seed": result["seed"], "age": result["age"],
        "host_loss_with_memory": result["endpoint"]["host_future_latent_loss_with_memory"],
        "host_loss_without_memory": result["endpoint"]["host_future_latent_loss_without_memory"],
        "endpoint_improvement": result["endpoint"]["improvement"],
        "audit": result["audit"], "causal_deletion": result["causal_deletion"],
        "surrogate_spearman": result["surrogate_spearman"],
        "patch_policy": args.patch_policy,
        "semantic_target": "frozen_dinov2_x_norm_patchtokens_stop_gradient",
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
    p.add_argument("--age", type=int, default=15, choices=list(AGES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--slots", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--patch-policy",
                   choices=["pooled", "random", "surprise_semantic"],
                   default="surprise_semantic")
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--quantile", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=1.0e-2)
    p.add_argument("--distill-every", type=int, default=8)
    p.add_argument("--distill-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stage-h-output", default=str(STAGE_H_OUTPUT))
    p.add_argument("--vendor", default=str(DEFAULT_VENDOR))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
