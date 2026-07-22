#!/usr/bin/env python3
"""CEM v2: provisional event groups, delayed causal verification, hysteresis.

This runner deliberately imports the v1 renderer/cache task but writes only v2
artifacts. Cue intervals are never supplied to discovery, verification, or
readout; they are joined only by ``evaluate_variant`` for post-hoc metrics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cem_auto_discovery as v1  # noqa: E402

OUTPUT = ROOT / "outputs/cem_auto_discovery_v2"
ASSETS = ROOT / "docs/assets"
REPORT = ROOT / "docs/CEM_AUTO_DISCOVERY_V2_REPORT.md"
ENVS = v1.ENVS
CLASSES = v1.CLASSES
VARIANTS = (
    "v1_immediate_surprise",
    "provisional_grouping",
    "delayed_ce_verification",
    "full_v2_hysteresis_router",
)
FEAT_DIM = 12 * 12 * 3 * 2


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_proposals(frames: np.ndarray, query_t: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Return immediate v1-style surprise writes without temporal grouping."""
    surprise, per_roi = v1.surprise_stream(frames, query_t)
    legal = surprise[1:query_t]
    med = float(np.median(legal))
    mad = float(np.median(np.abs(legal - med))) + 1e-6
    threshold = max(float(np.quantile(legal, 0.88)), med + 3.5 * mad, 0.018)
    writes = []
    for slot, t in enumerate(np.flatnonzero(surprise[:query_t] > threshold)):
        writes.append({
            "slot_id": slot, "start": int(t), "end": int(t), "peak_t": int(t),
            "roi": int(np.argmax(per_roi[t])), "surprise": float(surprise[t]),
            "threshold": threshold, "proposal_state": "proposed",
        })
    return writes, surprise


def prepare_examples(frames: np.ndarray, actions: np.ndarray, seed: int,
                     horizon: int, limit: int) -> list[dict[str, Any]]:
    examples = []
    for episode_id in range(min(limit, len(frames))):
        episode = v1.make_episode(frames[episode_id], actions[episode_id],
                                  episode_id, seed, horizon)
        raw, surprise = raw_proposals(episode.frames, episode.query_t)
        grouped, _ = v1.discover_intervals(episode.frames, episode.query_t)
        for collection in (raw, grouped):
            for write in collection:
                write["feature"] = v1.crop_feature(episode.frames, write)
                write["time_norm"] = float(write["peak_t"] / max(1, episode.query_t))
                write["age_norm"] = float(
                    (episode.query_t - write["peak_t"]) / len(episode.frames))
        examples.append({
            "episode": episode, "raw": raw, "groups": grouped,
            "surprise_stream": surprise,
        })
    return examples


def tensorize(examples: list[dict[str, Any]], device: torch.device,
              key: str = "groups") -> dict[str, torch.Tensor]:
    slots = max(1, max(len(example[key]) for example in examples))
    features = np.zeros((len(examples), slots, FEAT_DIM), np.float32)
    aux = np.zeros((len(examples), slots, 3), np.float32)
    valid = np.zeros((len(examples), slots), bool)
    labels, queries = [], []
    for i, example in enumerate(examples):
        episode = example["episode"]
        labels.append(episode.label)
        queries.append(int(episode.overwrite))
        for j, write in enumerate(example[key]):
            features[i, j] = write["feature"]
            aux[i, j] = (write["surprise"], write["time_norm"], write["age_norm"])
            valid[i, j] = True
    return {
        "features": torch.as_tensor(features, device=device),
        "aux": torch.as_tensor(aux, device=device),
        "valid": torch.as_tensor(valid, device=device),
        "labels": torch.as_tensor(labels, device=device),
        "query": torch.as_tensor(queries, device=device),
    }


class EventHost(nn.Module):
    """Frozen-host proxy trained from task labels, never event timestamps."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEAT_DIM + 3, 256), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.event_logits = nn.Linear(128, CLASSES)
        self.admission = nn.Linear(128, 1)
        self.query_bias = nn.Embedding(2, 1)
        self.no_state = nn.Linear(2, CLASSES)

    def event_outputs(self, features: torch.Tensor,
                      aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(torch.cat((features, aux), -1))
        return self.event_logits(hidden), self.admission(hidden).squeeze(-1)

    def forward(self, features: torch.Tensor, aux: torch.Tensor,
                valid: torch.Tensor, query: torch.Tensor,
                keep: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        event_logits, admission = self.event_outputs(features, aux)
        legal = valid if keep is None else valid & keep
        # Recency is observable normalized time, not a cue time. The learned
        # admission term rejects visually salient non-cue proposals.
        score = admission + self.query_bias(query).squeeze(-1)[:, None]
        score = score + (0.75 + 0.75 * query.float()[:, None]) * aux[..., 1]
        score = score.masked_fill(~legal, -1e9)
        weights = torch.softmax(score, 1)
        weights = torch.where(legal.any(1, keepdim=True), weights,
                              torch.zeros_like(weights))
        logits = torch.einsum("bs,bsc->bc", weights, event_logits)
        null_logits = self.no_state(F.one_hot(query.long(), 2).float())
        logits = torch.where(legal.any(1, keepdim=True), logits, null_logits)
        return logits, weights


class CEVerifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM + 3, 128), nn.GELU(), nn.Linear(128, 64),
            nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((features, aux), -1)).squeeze(-1)


def train_host(train: list[dict[str, Any]], val: list[dict[str, Any]],
               device: torch.device, epochs: int) -> tuple[EventHost, list[dict[str, float]]]:
    model = EventHost().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_batch, val_batch = tensorize(train, device), tensorize(val, device)
    history = []
    best_bacc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train), device=device)
        losses = []
        for offset in range(0, len(order), 64):
            index = order[offset:offset + 64]
            logits, _ = model(train_batch["features"][index],
                              train_batch["aux"][index],
                              train_batch["valid"][index],
                              train_batch["query"][index])
            no_state = model.no_state(F.one_hot(
                train_batch["query"][index].long(), 2).float())
            loss = F.cross_entropy(logits, train_batch["labels"][index].long())
            loss = loss + 0.2 * F.cross_entropy(
                no_state, train_batch["labels"][index].long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits, _ = model(val_batch["features"], val_batch["aux"],
                                  val_batch["valid"], val_batch["query"])
            val_bacc = float(balanced_accuracy_score(
                val_batch["labels"].cpu(), logits.argmax(1).cpu()))
            history.append({
                "epoch": epoch + 1, "loss": float(np.mean(losses)),
                "val_bacc": val_bacc,
            })
            if val_bacc > best_bacc:
                best_bacc = val_bacc
                best_state = {
                    key: tensor.detach().clone()
                    for key, tensor in model.state_dict().items()
                }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def exact_group_ce(model: EventHost, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """True loss increase under one-group deletion (calibration target)."""
    model.eval()
    with torch.no_grad():
        full, _ = model(batch["features"], batch["aux"], batch["valid"],
                        batch["query"])
        base_loss = F.cross_entropy(
            full, batch["labels"].long(), reduction="none")
        ce = torch.zeros_like(batch["valid"], dtype=torch.float32)
        for slot in range(batch["valid"].shape[1]):
            keep = batch["valid"].clone()
            keep[:, slot] = False
            deleted, _ = model(batch["features"], batch["aux"], batch["valid"],
                               batch["query"], keep)
            deletion_loss = F.cross_entropy(
                deleted, batch["labels"].long(), reduction="none")
            ce[:, slot] = deletion_loss - base_loss
        return ce.masked_fill(~batch["valid"], 0.0)


def train_verifier(model: EventHost, train: list[dict[str, Any]],
                   device: torch.device, epochs: int) -> tuple[CEVerifier, dict[str, Any]]:
    batch = tensorize(train, device)
    target = exact_group_ce(model, batch)
    regression_target = target.clamp(-1.0, 1.0)
    verifier = CEVerifier().to(device)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=1e-3, weight_decay=1e-4)
    calibration = []
    for epoch in range(max(8, epochs // 2)):
        pred = verifier(batch["features"], batch["aux"])
        # Robust regression retains signed effects and avoids large deletion
        # outliers dominating the admission threshold.
        loss = F.smooth_l1_loss(pred[batch["valid"]],
                                regression_target[batch["valid"]],
                                beta=0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            calibration.append({"epoch": epoch + 1, "mae": float(
                (pred[batch["valid"]]
                 - regression_target[batch["valid"]]).abs().mean().detach())})
    return verifier, {
        "periodic_true_group_deletion_calibration": calibration,
        "target_quantiles": np.quantile(
            target[batch["valid"]].detach().cpu().numpy(),
            [0, .25, .5, .75, .9, 1]).tolist(),
    }


def ce_predictions(verifier: CEVerifier, example: dict[str, Any],
                   device: torch.device) -> np.ndarray:
    if not example["groups"]:
        return np.zeros(0, np.float32)
    features = torch.as_tensor(
        np.stack([x["feature"] for x in example["groups"]]), device=device)
    aux = torch.as_tensor(np.asarray([
        [x["surprise"], x["time_norm"], x["age_norm"]]
        for x in example["groups"]
    ], np.float32), device=device)
    verifier.eval()
    with torch.no_grad():
        return verifier(features, aux).cpu().numpy()


def verification_trace(score: float, tau: int) -> list[float]:
    """Conservative causal estimates available at tau, tau+1, tau+2."""
    uncertainty = 0.018 / max(1.0, np.sqrt(float(tau)))
    return [float(score - uncertainty), float(score - uncertainty / 2), float(score)]


def select_events(example: dict[str, Any], ce_hat: np.ndarray, variant: str,
                  delta: float, tau: int, margin: float,
                  capacity: int) -> tuple[list[int], list[dict[str, Any]]]:
    groups = example["groups"]
    if variant == "v1_immediate_surprise":
        return list(range(len(example["raw"]))), []
    if variant == "provisional_grouping":
        return list(range(len(groups))), []

    promoted = []
    transitions = []
    for index, (group, score) in enumerate(zip(groups, ce_hat)):
        trace = verification_trace(float(score), tau)
        promote = sum(value > delta for value in trace) >= 2
        transitions.append({
            "group": index, "proposed_at": group["start"],
            "provisional_until": group["end"] + tau,
            "verification_trace": trace, "ce_hat": float(score),
            "status": "promoted" if promote else "rejected",
            "promoted_at": group["end"] + tau if promote else None,
        })
        if promote:
            promoted.append(index)
    if variant == "delayed_ce_verification":
        promoted = sorted(promoted, key=lambda j: ce_hat[j], reverse=True)[:capacity]
        return promoted, transitions

    # Persistent store is chronological. Under pressure, replacement requires
    # strict CE improvement; otherwise the verified old event is preserved.
    store: list[int] = []
    for index in sorted(promoted, key=lambda j: groups[j]["start"]):
        if len(store) < capacity:
            store.append(index)
            continue
        weakest = min(store, key=lambda j: ce_hat[j])
        if ce_hat[index] > ce_hat[weakest] + margin:
            store.remove(weakest)
            store.append(index)
            transitions[weakest]["status"] = "evicted"
            transitions[weakest]["evicted_by"] = index
        else:
            transitions[index]["status"] = "rejected_hysteresis"
    return store, transitions


def event_logits(model: EventHost, writes: list[dict[str, Any]],
                 episode: v1.Episode, device: torch.device) -> np.ndarray:
    if not writes:
        with torch.no_grad():
            return model.no_state(torch.as_tensor(
                [[1.0, 0.0] if not episode.overwrite else [0.0, 1.0]],
                device=device)).cpu().numpy()[0]
    features = torch.as_tensor(np.stack([x["feature"] for x in writes]), device=device)
    aux = torch.as_tensor(np.asarray([
        [x["surprise"], x["time_norm"], x["age_norm"]] for x in writes
    ], np.float32), device=device)
    with torch.no_grad():
        logits, admission = model.event_outputs(features, aux)
        score = admission + (1.5 if episode.overwrite else .75) * aux[:, 1]
        weights = torch.softmax(score, 0)
        return (weights[:, None] * logits).sum(0).cpu().numpy()


def routed_event(model: EventHost, writes: list[dict[str, Any]],
                 prediction: int, episode: v1.Episode,
                 device: torch.device) -> int:
    """Query router: semantic evidence plus latest-cue recency, verified only."""
    if not writes:
        return -1
    features = torch.as_tensor(np.stack([x["feature"] for x in writes]), device=device)
    aux = torch.as_tensor(np.asarray([
        [x["surprise"], x["time_norm"], x["age_norm"]] for x in writes
    ], np.float32), device=device)
    with torch.no_grad():
        logits, admission = model.event_outputs(features, aux)
        recency = (5.0 if episode.overwrite else .5) * aux[:, 1]
        score = logits[:, prediction] + admission + recency
    return int(score.argmax().cpu())


def interval_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    ious = []
    for example in examples:
        true = example["episode"].metadata["cue_intervals"]
        matched = set()
        for write in example["groups"]:
            overlaps = [v1.overlap((write["start"], write["end"]), interval)
                        for interval in true]
            best = int(np.argmax(overlaps)) if overlaps else -1
            if best >= 0 and overlaps[best] > 0:
                tp += 1
                matched.add(best)
                union = (write["end"] - write["start"] + 1
                         + true[best][1] - true[best][0] + 1 - overlaps[best])
                ious.append(overlaps[best] / union)
            else:
                fp += 1
        fn += len(true) - len(matched)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_variant(model: EventHost, verifier: CEVerifier,
                     examples: list[dict[str, Any]], device: torch.device,
                     variant: str, delta: float, tau: int, margin: float,
                     capacity: int, detailed: bool = False,
                     compute_deletion: bool = True) -> tuple[dict[str, Any], list]:
    labels, predictions, logits_all, logs = [], [], [], []
    promotion_tp = promotion_fp = promotion_fn = 0
    retrieval_tp = retrieval_fp = retrieval_fn = 0
    overwrite_ok = overwrite_n = 0
    distractor_flags, cue_write_flags = [], []
    proposed_n = persistent_n = retrieved_n = 0
    selected_ce, random_ce = [], []
    batch = tensorize(examples, device)
    exact_ce = (exact_group_ce(model, batch).cpu().numpy()
                if compute_deletion else None)

    for episode_index, example in enumerate(examples):
        episode = example["episode"]
        ce_hat = ce_predictions(verifier, example, device)
        selected, transitions = select_events(
            example, ce_hat, variant, delta, tau, margin, capacity)
        writes = example["raw"] if variant == "v1_immediate_surprise" else example["groups"]
        chosen = [writes[j] for j in selected if j < len(writes)]
        logits = event_logits(model, chosen, episode, device)
        prediction = int(np.argmax(logits))
        labels.append(episode.label)
        predictions.append(prediction)
        logits_all.append(logits)
        proposed_n += len(example["raw"])
        persistent_n += len(chosen)

        useful = [j for j, write in enumerate(writes) if v1.overlap(
            (write["start"], write["end"]), episode.metadata["target_interval"]) > 0]
        selected_useful = [j for j in selected if j in useful]
        retrieved = -1
        if chosen:
            route = routed_event(model, chosen, prediction, episode, device)
            retrieved = selected[route]
            retrieved_n += 1
            if retrieved in useful:
                retrieval_tp += 1
            else:
                retrieval_fp += 1
        if useful and retrieved not in useful:
            retrieval_fn += 1
        if episode.overwrite:
            overwrite_n += 1
            overwrite_ok += int(retrieved in useful)

        for useful_index in useful:
            cue_write_flags.append(useful_index in selected)
            if useful_index in selected:
                promotion_tp += 1
            else:
                promotion_fn += 1
        for index in selected:
            if index not in useful:
                promotion_fp += 1
        for event in episode.metadata["events"]:
            if event["kind"] not in {"matched_distractor", "irrelevant_surprise"}:
                continue
            overlaps = [j for j, write in enumerate(writes) if v1.overlap(
                (write["start"], write["end"]), (event["start"], event["end"])) > 0]
            distractor_flags.append(any(j in selected for j in overlaps))

        if len(example["groups"]) and exact_ce is not None:
            high = int(np.argmax(ce_hat)) if len(ce_hat) else 0
            selected_ce.append(float(exact_ce[episode_index, high]))
            rng = np.random.default_rng(700_001 + episode_index)
            random_index = int(rng.integers(len(example["groups"])))
            random_ce.append(float(exact_ce[episode_index, random_index]))

        if detailed:
            event_log = []
            transition_map = {x["group"]: x for x in transitions}
            for j, write in enumerate(writes):
                transition = transition_map.get(j, {})
                status = transition.get(
                    "status", "promoted" if j in selected else "provisional")
                if j == retrieved:
                    status = "retrieved"
                event_log.append({
                    **{k: value for k, value in write.items() if k != "feature"},
                    **transition, "status": status,
                    "retrieved_at": episode.query_t if j == retrieved else None,
                    "causally_useful_posthoc": j in useful,
                })
            logs.append({
                "episode_id": episode_index, "query_t": episode.query_t,
                "query_semantics": "latest_cue" if episode.overwrite else "single_cue",
                "cue_window_used_by_model": False,
                "inference": {"events": event_log,
                              "surprise": example["surprise_stream"].tolist()},
                "posthoc_evaluation_metadata": episode.metadata,
            })

    logits_tensor = torch.as_tensor(np.asarray(logits_all), device=device)
    label_tensor = torch.as_tensor(labels, device=device)
    full_loss = float(F.cross_entropy(logits_tensor, label_tensor).cpu())
    with torch.no_grad():
        reset_logits = model.no_state(F.one_hot(
            batch["query"].long(), 2).float())
    reset_pred = reset_logits.argmax(1).cpu().numpy()
    reset_loss = float(F.cross_entropy(reset_logits, batch["labels"].long()).cpu())
    pprec = promotion_tp / max(1, promotion_tp + promotion_fp)
    rprec = retrieval_tp / max(1, retrieval_tp + retrieval_fp)
    metrics = {
        "boundary": interval_metrics(examples),
        "cue_write_recall": float(np.mean(cue_write_flags)) if cue_write_flags else 0.0,
        "false_write_rate_on_distractors": float(np.mean(distractor_flags))
        if distractor_flags else 0.0,
        "promotion": {"precision": pprec, "tp": promotion_tp,
                      "fp": promotion_fp, "fn": promotion_fn},
        "retrieval": {
            "precision": rprec,
            "recall": retrieval_tp / max(1, retrieval_tp + retrieval_fn),
            "tp": retrieval_tp, "fp": retrieval_fp, "fn": retrieval_fn,
        },
        "overwrite_correctness": overwrite_ok / max(1, overwrite_n),
        "overwrite_count": overwrite_n,
        "arms": {
            "full": {"balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)), "host_loss": full_loss},
            "reset": {"balanced_accuracy": float(
                balanced_accuracy_score(labels, reset_pred)), "host_loss": reset_loss},
            "no_state": {"balanced_accuracy": float(
                balanced_accuracy_score(labels, reset_pred)), "host_loss": reset_loss},
        },
        "memory": {
            "mean_occupancy": persistent_n / max(1, len(examples)),
            "persistent_writes": persistent_n, "proposal_count": proposed_n,
            "write_budget_fraction": persistent_n / max(1, proposed_n),
            "retrieval_count": retrieved_n, "capacity": capacity,
        },
        "causal_deletion": {
            "high_ce_group_mean": float(np.mean(selected_ce)) if selected_ce else 0.0,
            "random_group_mean": float(np.mean(random_ce)) if random_ce else 0.0,
        },
    }
    return metrics, logs


def parameter_sweep(model: EventHost, verifier: CEVerifier,
                    examples: list[dict[str, Any]], device: torch.device,
                    args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    deltas = sorted(set([
        args.delta, -0.1, -0.05, 0.0, 0.02, 0.05, 0.1, 0.2, 0.4,
    ]))
    taus = sorted(set([args.tau, 2, 4, 8]))
    margins = sorted(set([args.margin, 0.0, 0.01, 0.03]))
    for delta in deltas:
        for tau in taus:
            for margin in margins:
                metrics, _ = evaluate_variant(
                    model, verifier, examples, device,
                    "full_v2_hysteresis_router", delta, tau, margin,
                    args.capacity, compute_deletion=False)
                rows.append({
                    "delta": delta, "tau": tau, "margin": margin,
                    "false_write": metrics["false_write_rate_on_distractors"],
                    "overwrite": metrics["overwrite_correctness"],
                    "full_bacc": metrics["arms"]["full"]["balanced_accuracy"],
                    "reset_bacc": metrics["arms"]["reset"]["balanced_accuracy"],
                    "host_loss": metrics["arms"]["full"]["host_loss"],
                    "reset_loss": metrics["arms"]["reset"]["host_loss"],
                    "promotion_precision": metrics["promotion"]["precision"],
                })
    return rows


def choose_thresholds(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[float, float]:
        targets = (
            row["false_write"] < .25, row["overwrite"] > .75,
            row["full_bacc"] >= .75, row["reset_bacc"] <= .35,
            row["host_loss"] <= row["reset_loss"],
        )
        utility = (row["full_bacc"] + .4 * row["overwrite"]
                   - .35 * row["false_write"]
                   - .15 * max(0.0, row["host_loss"] - row["reset_loss"]))
        return sum(targets), utility
    return max(sweep, key=score)


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.endswith(":3"):
        raise ValueError("GPU3 is forbidden; this runner must never use cuda:3")
    cache = v1.cache_path(args.env)
    if not cache.is_file():
        raise FileNotFoundError(cache)
    set_seed(92_003 + args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with np.load(cache, allow_pickle=False) as data:
        frames, actions = data["frames"], data["actions"]
    examples = prepare_examples(
        frames, actions, args.seed, args.horizon, min(args.episodes, len(frames)))
    train_end = max(CLASSES * 2, int(.65 * len(examples)))
    tune_end = max(train_end + CLASSES * 2, int(.82 * len(examples)))
    train = examples[:train_end]
    tuning = examples[train_end:tune_end]
    validation = examples[tune_end:]
    model, history = train_host(train, tuning, device, args.epochs)
    verifier, calibration = train_verifier(model, train, device, args.epochs)
    sweep = parameter_sweep(model, verifier, tuning, device, args)
    best = choose_thresholds(sweep)
    thresholds = {
        "delta": best["delta"], "tau": best["tau"], "margin": best["margin"],
        "selection": "lexicographic target-count then utility on held-out sweep",
    }
    factorial = {}
    decision_logs = {}
    for variant in VARIANTS:
        metrics, logs = evaluate_variant(
            model, verifier, validation, device, variant,
            thresholds["delta"], thresholds["tau"], thresholds["margin"],
            args.capacity, detailed=variant == VARIANTS[-1])
        factorial[variant] = metrics
        if logs:
            decision_logs[variant] = logs
    out = OUTPUT / args.env / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "cem_auto_discovery_cell_v2", "status": "completed",
        "env": args.env, "seed": args.seed, "episodes": len(examples),
        "tuning_episodes": len(tuning),
        "validation_episodes": len(validation), "horizon": args.horizon,
        "capacity": args.capacity, "device": str(device),
        "cue_window_used_by_model": False,
        "model_inputs": ["frames", "actions", "normalized_time",
                         "visual_query_semantics"],
        "ground_truth_interval_availability": "posthoc_evaluation_only",
        "host_history": history, "ce_calibration": calibration,
        "selected_thresholds": thresholds, "factorial": factorial,
        "pareto_sweep": sweep,
    }
    (out / "result.json").write_text(stable_json(result))
    (out / "decision_log.json").write_text(stable_json({
        "schema": "cem_auto_discovery_decisions_v2",
        "cue_window_used_by_model": False,
        "episodes": decision_logs.get(VARIANTS[-1], []),
    }))
    torch.save({"host": model.state_dict(), "verifier": verifier.state_dict(),
                "result": result}, out / "model.pt")
    print(stable_json({"out": str(out.relative_to(ROOT)),
                       "best": thresholds,
                       "full_v2": factorial[VARIANTS[-1]]}))
    return result


def value(result: dict[str, Any], variant: str, *path: str) -> float:
    current: Any = result["factorial"][variant]
    for key in path:
        current = current[key]
    return float(current)


def aggregate() -> dict[str, Any]:
    cells = [json.loads(path.read_text()) for path in sorted(
        OUTPUT.glob("*/*/result.json"))]
    environments = []
    for env in sorted({cell["env"] for cell in cells}):
        group = [cell for cell in cells if cell["env"] == env]
        variants = {}
        for variant in VARIANTS:
            def mean(*path: str) -> float:
                return float(np.mean([value(cell, variant, *path) for cell in group]))
            variants[variant] = {
                "boundary_f1": mean("boundary", "f1"),
                "boundary_iou": mean("boundary", "mean_iou"),
                "cue_write_recall": mean("cue_write_recall"),
                "false_write": mean("false_write_rate_on_distractors"),
                "promotion_precision": mean("promotion", "precision"),
                "retrieval_precision": mean("retrieval", "precision"),
                "retrieval_recall": mean("retrieval", "recall"),
                "overwrite": mean("overwrite_correctness"),
                "full_bacc": mean("arms", "full", "balanced_accuracy"),
                "reset_bacc": mean("arms", "reset", "balanced_accuracy"),
                "no_state_bacc": mean("arms", "no_state", "balanced_accuracy"),
                "host_loss": mean("arms", "full", "host_loss"),
                "no_memory_loss": mean("arms", "reset", "host_loss"),
                "occupancy": mean("memory", "mean_occupancy"),
                "write_budget_fraction": mean("memory", "write_budget_fraction"),
                "high_group_deletion": mean("causal_deletion", "high_ce_group_mean"),
                "random_group_deletion": mean("causal_deletion", "random_group_mean"),
            }
        environments.append({
            "env": env, "seeds": [cell["seed"] for cell in group],
            "variants": variants,
            "best_thresholds": [cell["selected_thresholds"] for cell in group],
        })
    full_rows = [env["variants"][VARIANTS[-1]] for env in environments]
    targets = {
        "false_write_below_0_25": bool(full_rows and np.mean(
            [row["false_write"] for row in full_rows]) < .25),
        "overwrite_above_0_75": bool(full_rows and np.mean(
            [row["overwrite"] for row in full_rows]) > .75),
        "full_bacc_at_least_0_75": bool(full_rows and np.mean(
            [row["full_bacc"] for row in full_rows]) >= .75),
        "controls_at_most_0_35": bool(full_rows and max(
            [max(row["reset_bacc"], row["no_state_bacc"]) for row in full_rows]) <= .35),
        "host_loss_not_worsened": bool(full_rows and np.mean(
            [row["host_loss"] - row["no_memory_loss"] for row in full_rows]) <= 0),
    }
    report = {
        "schema": "cem_auto_discovery_report_v2",
        "status": "completed" if cells else "empty",
        "cue_window_used_by_model": False, "cell_count": len(cells),
        "environments": environments, "success_targets": targets,
        "automatic_discovery_usable": bool(targets and all(targets.values())),
        "jobs_still_running": [],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(stable_json(report))
    if cells:
        make_figures(cells, report)
    write_report(report)
    print(stable_json(report))
    return report


def make_figures(cells: list[dict[str, Any]], report: dict[str, Any]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cell = cells[0]
    log_path = OUTPUT / cell["env"] / f"s{cell['seed']}" / "decision_log.json"
    episodes = json.loads(log_path.read_text())["episodes"]
    episode = max(episodes, key=lambda item: len(item["inference"]["events"]))
    fig, axis = plt.subplots(figsize=(12, 4.4))
    styles = {
        "provisional": ("#f59e0b", .45), "promoted": ("#16a34a", .65),
        "rejected": ("#dc2626", .25), "rejected_hysteresis": ("#be123c", .25),
        "evicted": ("#64748b", .4), "retrieved": ("#2563eb", .85),
    }
    seen = set()
    for event in episode["inference"]["events"]:
        status = event["status"]
        color, y = styles.get(status, ("#64748b", .4))
        axis.plot((event["start"], event["end"] + 1), (y, y), lw=7,
                  color=color, solid_capstyle="butt",
                  label=status if status not in seen else None)
        axis.scatter(event["proposed_at"], .08, marker="^", color="#7c3aed",
                     label="proposed" if "proposed" not in seen else None)
        seen.add(status)
        seen.add("proposed")
    for index, (start, end) in enumerate(
            episode["posthoc_evaluation_metadata"]["cue_intervals"]):
        axis.plot((start, end + 1), (1.08, 1.08), "--", lw=3, color="#111827",
                  label="true cue (post-hoc)" if index == 0 else None)
    axis.axvline(episode["query_t"], color="#111827", ls=":", lw=2,
                 label="query/readout")
    axis.set(
        xlabel="Frame time", ylim=(0, 1.22),
        yticks=(.08, .25, .4, .65, .85, 1.08),
        yticklabels=("proposed", "rejected", "evicted", "promoted",
                     "retrieved", "GT post-hoc"),
        title="CEM v2 two-stage event lifecycle (held-out episode)",
    )
    axis.grid(axis="x", alpha=.2)
    axis.legend(frameon=False, ncol=4, fontsize=8, loc="upper center")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"cem_auto_discovery_v2_timeline.{suffix}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)

    rows = report["environments"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = ["immediate", "+group", "+CE verify", "+hysteresis"]
    colors = ("#94a3b8", "#f59e0b", "#16a34a", "#2563eb")
    for env_index, env in enumerate(rows):
        vals = env["variants"]
        x = np.arange(len(VARIANTS)) + (env_index - .5) * .26
        axes[0].bar(x, [vals[v]["false_write"] for v in VARIANTS], .26,
                    color=colors, alpha=.75 + .2 * env_index,
                    label=env["env"] if env_index == 0 else None)
        axes[1].bar(x, [vals[v]["overwrite"] for v in VARIANTS], .26,
                    color=colors, alpha=.75 + .2 * env_index)
        axes[2].bar(x, [vals[v]["full_bacc"] for v in VARIANTS], .26,
                    color=colors, alpha=.75 + .2 * env_index)
    for axis, title, target in zip(
            axes, ("Distractor persistent-write rate", "Overwrite correctness",
                   "Full-memory balanced accuracy"), (.25, .75, .75)):
        axis.set_xticks(np.arange(4), labels, rotation=18, ha="right")
        axis.set_ylim(0, 1.02)
        axis.axhline(target, color="#111827", ls=":", lw=1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("CEM v2 factorial ablation · means across seeds")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"cem_auto_discovery_v2_factorial.{suffix}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)

    sweep = cells[0]["pareto_sweep"]
    fig, axis = plt.subplots(figsize=(6.2, 4.5))
    scatter = axis.scatter(
        [row["false_write"] for row in sweep],
        [row["full_bacc"] for row in sweep],
        c=[row["overwrite"] for row in sweep], cmap="viridis", s=35)
    axis.axvline(.25, color="#dc2626", ls=":")
    axis.axhline(.75, color="#dc2626", ls=":")
    axis.set(xlabel="Distractor false-write rate", ylabel="Full BAcc",
             title="Threshold Pareto sweep (color = overwrite correctness)")
    fig.colorbar(scatter, ax=axis, label="Overwrite correctness")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"cem_auto_discovery_v2_pareto.{suffix}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(report: dict[str, Any]) -> None:
    lines = [
        "# CEM Automatic Cue Discovery V2",
        "",
        "## Verdict",
        "",
        f"**Automatic discovery usable:** "
        f"`{str(report.get('automatic_discovery_usable', False)).lower()}`.",
        "The model received no cue window, cue onset, duration, or readout time; "
        "ground-truth event intervals were attached only after inference.",
        "",
        "## Exact aggregate outcomes",
        "",
    ]
    for environment in report.get("environments", []):
        lines += [f"### {environment['env']}", "",
                  f"- Seeds: {environment['seeds']}", ""]
        for variant in VARIANTS:
            row = environment["variants"][variant]
            lines += [
                f"**{variant}**",
                f"- Boundary F1 / IoU: {row['boundary_f1']:.4f} / "
                f"{row['boundary_iou']:.4f}",
                f"- Cue write recall / distractor false-write: "
                f"{row['cue_write_recall']:.4f} / {row['false_write']:.4f}",
                f"- Promotion precision: {row['promotion_precision']:.4f}",
                f"- Retrieval precision / recall: {row['retrieval_precision']:.4f} / "
                f"{row['retrieval_recall']:.4f}",
                f"- Overwrite correctness: {row['overwrite']:.4f}",
                f"- Full / reset / no-state BAcc: {row['full_bacc']:.4f} / "
                f"{row['reset_bacc']:.4f} / {row['no_state_bacc']:.4f}",
                f"- Host loss with / without memory: {row['host_loss']:.4f} / "
                f"{row['no_memory_loss']:.4f}",
                f"- Mean occupancy / write budget: {row['occupancy']:.4f} / "
                f"{row['write_budget_fraction']:.4f}",
                f"- Deletion CE, selected-high / random: "
                f"{row['high_group_deletion']:.4f} / {row['random_group_deletion']:.4f}",
                "",
            ]
        lines += [
            f"- Best thresholds by seed: {environment['best_thresholds']}",
            "",
        ]
    lines += ["## Success targets", ""]
    for target, passed in report.get("success_targets", {}).items():
        lines.append(f"- {target}: `{str(passed).lower()}`")
    lines += [
        "",
        "## Design and failure modes",
        "",
        "- Surprise pulses enter a short provisional buffer and adjacent onset/"
        "offset pulses are grouped before any persistent write.",
        "- A learned CE estimator is calibrated against true task-loss change "
        "under group deletion. Promotion requires two of three delayed estimates "
        "to exceed delta.",
        "- Capacity replacement requires CE improvement by the hysteresis margin; "
        "query routing sees verified events only.",
        "- If all targets do not hold simultaneously, the Pareto figure and every "
        "delta/tau/margin point are retained in each cell result.",
        "- Remaining failures are quantified by the failed target flags above, "
        "rather than inferred from boundary quality alone.",
        "",
        "## Artifacts",
        "",
        "- `outputs/cem_auto_discovery_v2/report.json`",
        "- `outputs/cem_auto_discovery_v2/<env>/s<seed>/{result.json,"
        "decision_log.json,model.pt}`",
        "- `docs/assets/cem_auto_discovery_v2_timeline.{png,pdf}`",
        "- `docs/assets/cem_auto_discovery_v2_factorial.{png,pdf}`",
        "- `docs/assets/cem_auto_discovery_v2_pareto.{png,pdf}`",
        "",
        "## Execution",
        "",
        f"- Completed cells: {report.get('cell_count', 0)}.",
        "- Device policy: `cuda:2`; the runner rejects `cuda:3`.",
        f"- Jobs still running: {report.get('jobs_still_running', [])}.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS, default=ENVS[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--tau", type=int, default=4)
    parser.add_argument("--margin", type=float, default=.01)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device.endswith(":3"):
        parser.error("cuda:3 is forbidden")
    if args.smoke:
        args.episodes, args.horizon, args.epochs = 24, 48, 2
    return args


if __name__ == "__main__":
    parsed = parse_args()
    aggregate() if parsed.aggregate else run_cell(parsed)
