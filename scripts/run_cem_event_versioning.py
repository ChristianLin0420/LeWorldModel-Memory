#!/usr/bin/env python3
"""Event-versioned CEM discovery on randomized OGBench visual event streams.

The runner reuses v2's label-free surprise grouping and delayed group-CE
estimator, but replaces destructive overwrite with an explicit version store.
Cue intervals and event identities are unavailable to discovery and routing and
are joined only by the evaluator.  The trained host is frozen before all
factorial evaluations and its digest is asserted unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cem_auto_discovery_v2 as v2  # noqa: E402

OUTPUT = ROOT / "outputs/cem_event_versioning_v1"
REPORT = ROOT / "docs/CEM_EVENT_VERSIONING_REPORT.md"
ASSETS = ROOT / "docs/assets"
ENVS = v2.ENVS
VARIANTS = (
    "immediate_overwrite_v1",
    "hysteresis_only",
    "version_store_no_verification",
    "full_versioned_delayed_verification",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def event_outputs(model: v2.EventHost, groups: list[dict[str, Any]],
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if not groups:
        return np.zeros((0, v2.CLASSES), np.float32), np.zeros(0, np.float32)
    features = torch.as_tensor(
        np.stack([group["feature"] for group in groups]), device=device)
    aux = torch.as_tensor(np.asarray([
        [group["surprise"], group["time_norm"], group["age_norm"]]
        for group in groups
    ], np.float32), device=device)
    model.eval()
    with torch.no_grad():
        logits, admission = model.event_outputs(features, aux)
        return logits.cpu().numpy(), admission.cpu().numpy()


def persistent(trace: list[float], threshold: float, steps: int) -> bool:
    run = 0
    for value in trace:
        run = run + 1 if value > threshold else 0
        if run >= steps:
            return True
    return False


def build_versions(example: dict[str, Any], ce_hat: np.ndarray,
                   admission: np.ndarray, logits: np.ndarray, variant: str,
                   delta: float, margin: float, persistence: int,
                   verify_delay: int, capacity: int) -> tuple[list[dict[str, Any]], list[int]]:
    """Run the event store in timestamp order; return lifecycle and live slots."""
    groups = example["groups"]
    versions: list[dict[str, Any]] = []
    live: list[int] = []
    active_by_key: dict[int, int] = {}
    for index, group in enumerate(groups):
        confidence = float(torch.softmax(
            torch.as_tensor(logits[index]), 0).max()) if len(logits) else 0.0
        semantic_key = int(np.argmax(logits[index])) if len(logits) else -1
        trace = v2.verification_trace(float(ce_hat[index]), verify_delay)
        version = {
            "version_id": index,
            "object_key": semantic_key,
            "timestamp": int(group["peak_t"]),
            "start": int(group["start"]),
            "end": int(group["end"]),
            "value": semantic_key,
            "ce_hat": float(ce_hat[index]),
            "confidence": confidence,
            "admission": float(admission[index]),
            "verification_trace": trace,
            "status": "provisional",
            "promoted_at": None,
            "rejected_at": None,
            "evicted_at": None,
            "selected_at": None,
            "fallback_from": None,
        }
        versions.append(version)

        if variant == "immediate_overwrite_v1":
            old = active_by_key.get(semantic_key)
            if old is not None and old in live:
                live.remove(old)
                versions[old]["status"] = "overwritten"
            version["status"] = "verified"
            version["promoted_at"] = int(group["end"])
            active_by_key[semantic_key] = index
            live.append(index)
        elif variant == "hysteresis_only":
            old = active_by_key.get(semantic_key)
            improves = old is None or ce_hat[index] > ce_hat[old] + margin
            if improves:
                if old is not None and old in live:
                    live.remove(old)
                    versions[old]["status"] = "overwritten"
                version["status"] = "verified"
                version["promoted_at"] = int(group["end"])
                active_by_key[semantic_key] = index
                live.append(index)
            else:
                version["status"] = "rejected_hysteresis"
                version["rejected_at"] = int(group["end"])
        elif variant == "version_store_no_verification":
            version["status"] = "verified"
            version["promoted_at"] = int(group["end"])
            live.append(index)
            active_by_key[semantic_key] = index
        else:
            verified = persistent(trace, delta, persistence)
            old = active_by_key.get(semantic_key)
            improves = old is None or ce_hat[index] > ce_hat[old] + margin
            if verified and improves:
                version["status"] = "verified"
                version["promoted_at"] = int(group["end"] + verify_delay)
                live.append(index)
                active_by_key[semantic_key] = index
                if old is not None:
                    versions[old]["status"] = "superseded_verified"
            else:
                version["status"] = (
                    "rejected_verification" if not verified
                    else "rejected_hysteresis")
                version["rejected_at"] = int(group["end"] + verify_delay)

        # Versions survive supersession. Capacity pressure alone causes eviction.
        while len(live) > capacity:
            weakest = min(live, key=lambda slot: (
                versions[slot]["ce_hat"], versions[slot]["timestamp"]))
            live.remove(weakest)
            versions[weakest]["status"] = "capacity_evicted"
            versions[weakest]["evicted_at"] = int(group["end"])
            key = versions[weakest]["object_key"]
            if active_by_key.get(key) == weakest:
                same_key = [slot for slot in live
                            if versions[slot]["object_key"] == key]
                if same_key:
                    active_by_key[key] = max(
                        same_key, key=lambda slot: versions[slot]["timestamp"])
                else:
                    active_by_key.pop(key, None)
    return versions, live


def route_versions(versions: list[dict[str, Any]], live: list[int],
                   logits: np.ndarray, query_t: int, variant: str,
                   query_verify: float) -> tuple[int, int, list[dict[str, Any]]]:
    """Rank all live versions, then verify in order with older fallback."""
    if not live:
        return -1, -1, []
    candidates = []
    for slot in live:
        version = versions[slot]
        match = float(torch.softmax(
            torch.as_tensor(logits[slot]), 0).max())
        recency = float(np.exp(-(query_t - version["timestamp"]) /
                               max(6.0, .25 * query_t)))
        if variant == "immediate_overwrite_v1":
            score = version["admission"] + 0.65 * recency
        elif variant == "hysteresis_only":
            score = version["ce_hat"] + version["admission"] + 0.9 * recency
        else:
            # The contract is newest VERIFIED first.  CE and semantic match
            # break near-recency ties; a large bounded recency coefficient
            # prevents a high-CE stale version from shadowing a verified state
            # change at an unknown query delay.
            score = 1.5 * version["ce_hat"] + match + 32.0 * recency
        candidates.append({
            "version_id": slot, "rank_score": float(score),
            "query_match": match, "recency_kernel": recency,
            "query_verified": bool(
                version["confidence"] >= query_verify
                and version["admission"] > -1.0),
        })
    candidates.sort(key=lambda row: (
        row["rank_score"], versions[row["version_id"]]["timestamp"]),
                    reverse=True)
    first = candidates[0]["version_id"]
    selected = -1
    for row in candidates:
        if variant != "full_versioned_delayed_verification" or row["query_verified"]:
            selected = row["version_id"]
            break
    if selected < 0:
        # No unverified value is exposed; this is an explicit empty retrieval.
        return -1, first, candidates
    versions[selected]["selected_at"] = int(query_t)
    if selected != first:
        versions[selected]["fallback_from"] = first
    return selected, first, candidates


def evaluate(model: v2.EventHost, verifier: v2.CEVerifier,
             examples: list[dict[str, Any]], device: torch.device,
             variant: str, delta: float, margin: float, persistence_steps: int,
             verify_delay: int, capacity: int, query_verify: float,
             detailed: bool = False, deletion: bool = True) -> tuple[dict[str, Any], list]:
    labels, predictions, logits_all, logs = [], [], [], []
    overwrite_ok = overwrite_n = stale_errors = 0
    false_flags: list[bool] = []
    retrieval_tp = retrieval_fp = retrieval_fn = 0
    selection_ok = selection_n = fallback_n = retrieval_n = 0
    occupancy = evictions = 0
    high_ce, random_ce = [], []
    batch = v2.tensorize(examples, device)
    exact = v2.exact_group_ce(model, batch).cpu().numpy() if deletion else None

    for episode_index, example in enumerate(examples):
        episode = example["episode"]
        groups = example["groups"]
        ce_hat = v2.ce_predictions(verifier, example, device)
        event_logit, admission = event_outputs(model, groups, device)
        versions, live = build_versions(
            example, ce_hat, admission, event_logit, variant, delta, margin,
            persistence_steps, verify_delay, capacity)
        selected, first, ranking = route_versions(
            versions, live, event_logit, episode.query_t, variant, query_verify)
        if selected >= 0:
            logits = event_logit[selected]
            prediction = int(np.argmax(logits))
            retrieval_n += 1
            fallback_n += int(selected != first)
        else:
            with torch.no_grad():
                logits = model.no_state(torch.as_tensor(
                    [[1.0, 0.0] if not episode.overwrite else [0.0, 1.0]],
                    device=device)).cpu().numpy()[0]
            prediction = int(np.argmax(logits))
        labels.append(episode.label)
        predictions.append(prediction)
        logits_all.append(logits)
        occupancy += len(live)
        evictions += sum(v["status"] == "capacity_evicted" for v in versions)

        useful = [j for j, group in enumerate(groups) if v2.v1.overlap(
            (group["start"], group["end"]),
            episode.metadata["target_interval"]) > 0]
        selected_useful = selected in useful
        selection_n += int(bool(useful))
        selection_ok += int(selected_useful)
        retrieval_tp += int(selected_useful)
        retrieval_fp += int(selected >= 0 and not selected_useful)
        retrieval_fn += int(bool(useful) and not selected_useful)
        if episode.overwrite:
            overwrite_n += 1
            overwrite_ok += int(selected_useful)
            old_intervals = episode.metadata["cue_intervals"][:-1]
            stale_errors += int(selected >= 0 and any(v2.v1.overlap(
                (groups[selected]["start"], groups[selected]["end"]), interval) > 0
                for interval in old_intervals))

        for event in episode.metadata["events"]:
            if event["kind"] not in {"matched_distractor", "irrelevant_surprise"}:
                continue
            overlapping = [j for j, group in enumerate(groups) if v2.v1.overlap(
                (group["start"], group["end"]),
                (event["start"], event["end"])) > 0]
            false_flags.append(any(j in live for j in overlapping))

        if len(groups) and exact is not None:
            high = int(np.argmax(ce_hat))
            high_ce.append(float(exact[episode_index, high]))
            rng = np.random.default_rng(810_031 + episode_index)
            random_ce.append(float(exact[
                episode_index, int(rng.integers(len(groups)))]))
        if detailed:
            ranking_map = {row["version_id"]: row for row in ranking}
            logs.append({
                "episode_id": episode_index, "query_t": episode.query_t,
                "query_semantics": "latest_verified_event",
                "cue_window_used_by_model": False,
                "inference": {
                    "versions": [
                        {**version, **ranking_map.get(version["version_id"], {})}
                        for version in versions],
                    "selected_version": selected,
                    "first_ranked_version": first,
                    "live_version_ids": live,
                    "surprise": example["surprise_stream"].tolist(),
                },
                "posthoc_evaluation_metadata": episode.metadata,
            })

    logits_tensor = torch.as_tensor(np.asarray(logits_all), device=device)
    labels_tensor = torch.as_tensor(labels, device=device)
    full_loss = float(F.cross_entropy(logits_tensor, labels_tensor).cpu())
    with torch.no_grad():
        reset_logits = model.no_state(F.one_hot(
            batch["query"].long(), 2).float())
    reset_pred = reset_logits.argmax(1).cpu().numpy()
    reset_loss = float(F.cross_entropy(
        reset_logits, batch["labels"].long()).cpu())
    return {
        "false_write_rate": float(np.mean(false_flags)) if false_flags else 0.0,
        "overwrite_correctness": overwrite_ok / max(1, overwrite_n),
        "overwrite_count": overwrite_n,
        "version_selection_accuracy": selection_ok / max(1, selection_n),
        "fallback_rate": fallback_n / max(1, retrieval_n),
        "stale_version_error": stale_errors / max(1, overwrite_n),
        "retrieval": {
            "precision": retrieval_tp / max(1, retrieval_tp + retrieval_fp),
            "recall": retrieval_tp / max(1, retrieval_tp + retrieval_fn),
        },
        "arms": {
            "full": {"balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)),
                     "host_loss": full_loss},
            "reset": {"balanced_accuracy": float(
                balanced_accuracy_score(labels, reset_pred)),
                      "host_loss": reset_loss},
            "no_state": {"balanced_accuracy": float(
                balanced_accuracy_score(labels, reset_pred)),
                         "host_loss": reset_loss},
        },
        "memory": {
            "mean_occupancy": occupancy / max(1, len(examples)),
            "capacity": capacity, "capacity_evictions": evictions,
        },
        "causal_deletion": {
            "high_ce_group_mean": float(np.mean(high_ce)) if high_ce else 0.0,
            "random_group_mean": float(np.mean(random_ce)) if random_ce else 0.0,
        },
    }, logs


def parameter_sweep(model: v2.EventHost, verifier: v2.CEVerifier,
                    examples: list[dict[str, Any]], device: torch.device,
                    args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for margin in sorted(set((0.0, .01, .03, .06, args.margin))):
        for steps in sorted(set((1, 2, 3, args.persistence_steps))):
            for delta in sorted(set((-.05, 0.0, .02, .05, .1, args.delta))):
                metrics, _ = evaluate(
                    model, verifier, examples, device, VARIANTS[-1],
                    delta, margin, steps, args.verify_delay, args.capacity,
                    args.query_verify, deletion=False)
                rows.append({
                    "margin": margin, "persistence_steps": steps, "delta": delta,
                    "overwrite": metrics["overwrite_correctness"],
                    "false_write": metrics["false_write_rate"],
                    "full_bacc": metrics["arms"]["full"]["balanced_accuracy"],
                    "control_bacc": metrics["arms"]["reset"]["balanced_accuracy"],
                    "host_loss": metrics["arms"]["full"]["host_loss"],
                    "reset_loss": metrics["arms"]["reset"]["host_loss"],
                })
    return rows


def choose_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[int, float]:
        gates = (
            row["overwrite"] > .80, row["false_write"] < .20,
            row["full_bacc"] >= .75, row["control_bacc"] <= .35,
            row["host_loss"] <= row["reset_loss"],
        )
        utility = (1.4 * row["overwrite"] + row["full_bacc"]
                   - .45 * row["false_write"]
                   - .1 * max(0.0, row["host_loss"] - row["reset_loss"]))
        return sum(gates), utility
    best = max(rows, key=score)
    return {
        "delta": best["delta"], "margin": best["margin"],
        "persistence_steps": best["persistence_steps"],
        "selection": "held-out lexicographic gate-count then overwrite-weighted utility",
    }


def curves(model: v2.EventHost, verifier: v2.CEVerifier,
           examples: list[dict[str, Any]], device: torch.device,
           selected: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    capacity_curve = []
    for capacity in (1, 2, 3, 4):
        metrics, _ = evaluate(
            model, verifier, examples, device, VARIANTS[-1],
            selected["delta"], selected["margin"],
            selected["persistence_steps"], args.verify_delay, capacity,
            args.query_verify, deletion=False)
        capacity_curve.append({
            "capacity": capacity, "overwrite": metrics["overwrite_correctness"],
            "full_bacc": metrics["arms"]["full"]["balanced_accuracy"],
            "false_write": metrics["false_write_rate"],
        })
    delay_curve = []
    delays = np.asarray([example["episode"].metadata["delay"] for example in examples])
    for maximum in (8, 16, 32, 64, 10_000):
        subset = [example for example, delay in zip(examples, delays) if delay <= maximum]
        if len(subset) < 8:
            continue
        metrics, _ = evaluate(
            model, verifier, subset, device, VARIANTS[-1],
            selected["delta"], selected["margin"],
            selected["persistence_steps"], args.verify_delay, args.capacity,
            args.query_verify, deletion=False)
        delay_curve.append({
            "max_delay": int(maximum if maximum < 10_000 else delays.max()),
            "episodes": len(subset), "overwrite": metrics["overwrite_correctness"],
            "full_bacc": metrics["arms"]["full"]["balanced_accuracy"],
        })
    return {"capacity": capacity_curve, "max_delay": delay_curve}


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.endswith(":3"):
        raise ValueError("cuda:3 is forbidden")
    cache = v2.v1.cache_path(args.env)
    if not cache.is_file():
        raise FileNotFoundError(cache)
    v2.set_seed(103_019 + args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with np.load(cache, allow_pickle=False) as data:
        frames, actions = data["frames"], data["actions"]
    examples = v2.prepare_examples(
        frames, actions, args.seed, args.horizon,
        min(args.episodes, len(frames)))
    train_end = max(v2.CLASSES * 2, int(.65 * len(examples)))
    tune_end = max(train_end + v2.CLASSES * 2, int(.82 * len(examples)))
    train, tuning, validation = (
        examples[:train_end], examples[train_end:tune_end], examples[tune_end:])
    model, history = v2.train_host(train, tuning, device, args.epochs)
    verifier, calibration = v2.train_verifier(
        model, train, device, args.epochs)
    model.eval()
    verifier.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in verifier.parameters():
        parameter.requires_grad_(False)
    digest_before = model_digest(model)

    sweep = parameter_sweep(model, verifier, tuning, device, args)
    selected = choose_thresholds(sweep)
    factorial, decision_logs = {}, {}
    for variant in VARIANTS:
        metrics, logs = evaluate(
            model, verifier, validation, device, variant,
            selected["delta"], selected["margin"],
            selected["persistence_steps"], args.verify_delay,
            args.capacity, args.query_verify,
            detailed=True)
        factorial[variant] = metrics
        decision_logs[variant] = logs
    curve_data = curves(model, verifier, validation, device, selected, args)
    digest_after = model_digest(model)
    if digest_before != digest_after:
        raise RuntimeError("frozen host digest changed during version-store evaluation")
    result = {
        "schema": "cem_event_versioning_cell_v1", "status": "completed",
        "env": args.env, "seed": args.seed, "episodes": len(examples),
        "tuning_episodes": len(tuning), "validation_episodes": len(validation),
        "device": str(device), "capacity": args.capacity,
        "cue_window_used_by_model": False,
        "frozen_host": {
            "kind": "v2 OGBench event-host proxy over cached rendered frames",
            "digest_before": digest_before, "digest_after": digest_after,
            "unchanged": True,
        },
        "selected_thresholds": selected,
        "factorial": factorial, "pareto_sweep": sweep, "curves": curve_data,
        "host_history": history, "ce_calibration": calibration,
    }
    out = OUTPUT / args.env / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(stable_json(result))
    (out / "decision_log.json").write_text(stable_json({
        "schema": "cem_event_versioning_decisions_v1",
        "cue_window_used_by_model": False,
        "variants": decision_logs,
        "episodes": decision_logs[VARIANTS[-1]],
    }))
    torch.save({"host": model.state_dict(), "verifier": verifier.state_dict(),
                "result": result}, out / "model.pt")
    print(stable_json({
        "out": str(out.relative_to(ROOT)), "selected": selected,
        "full": factorial[VARIANTS[-1]],
    }))
    return result


def nested(result: dict[str, Any], variant: str, *path: str) -> float:
    value: Any = result["factorial"][variant]
    for key in path:
        value = value[key]
    return float(value)


def aggregate() -> dict[str, Any]:
    cells = [json.loads(path.read_text()) for path in sorted(
        OUTPUT.glob("*/*/result.json"))]
    environments = []
    for env in sorted({cell["env"] for cell in cells}):
        group = [cell for cell in cells if cell["env"] == env]
        variants = {}
        for variant in VARIANTS:
            def mean(*path: str) -> float:
                return float(np.mean([
                    nested(cell, variant, *path) for cell in group]))
            variants[variant] = {
                "overwrite": mean("overwrite_correctness"),
                "false_write": mean("false_write_rate"),
                "full_bacc": mean("arms", "full", "balanced_accuracy"),
                "control_bacc": mean("arms", "reset", "balanced_accuracy"),
                "host_loss": mean("arms", "full", "host_loss"),
                "reset_loss": mean("arms", "reset", "host_loss"),
                "version_selection_accuracy": mean("version_selection_accuracy"),
                "fallback_rate": mean("fallback_rate"),
                "stale_version_error": mean("stale_version_error"),
                "retrieval_precision": mean("retrieval", "precision"),
                "retrieval_recall": mean("retrieval", "recall"),
                "occupancy": mean("memory", "mean_occupancy"),
                "evictions": mean("memory", "capacity_evictions"),
                "high_group_deletion": mean(
                    "causal_deletion", "high_ce_group_mean"),
                "random_group_deletion": mean(
                    "causal_deletion", "random_group_mean"),
            }
        environments.append({
            "env": env, "seeds": [cell["seed"] for cell in group],
            "variants": variants,
            "selected_thresholds": [cell["selected_thresholds"] for cell in group],
        })
    full = [env["variants"][VARIANTS[-1]] for env in environments]
    targets = {
        "overwrite_above_0_80": bool(full and np.mean(
            [row["overwrite"] for row in full]) > .80),
        "false_write_below_0_20": bool(full and np.mean(
            [row["false_write"] for row in full]) < .20),
        "full_bacc_at_least_0_75": bool(full and np.mean(
            [row["full_bacc"] for row in full]) >= .75),
        "controls_at_most_0_35": bool(full and max(
            row["control_bacc"] for row in full) <= .35),
        "host_loss_nonworsening": bool(full and np.mean(
            [row["host_loss"] - row["reset_loss"] for row in full]) <= 0),
        "high_ce_deletion_above_random": bool(full and np.mean(
            [row["high_group_deletion"] - row["random_group_deletion"]
             for row in full]) > 0),
    }
    report = {
        "schema": "cem_event_versioning_report_v1",
        "status": "completed" if cells else "empty",
        "cell_count": len(cells), "environments": environments,
        "success_targets": targets,
        "overwrite_target_passed": targets["overwrite_above_0_80"],
        "all_targets_passed": bool(targets and all(targets.values())),
        "cue_window_used_by_model": False,
        "host_scope": (
            "Frozen v2 OGBench event-host proxy; official DINO-WM weights were "
            "not loaded by this runner."),
        "jobs_still_running": [],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(stable_json(report))
    if cells:
        make_figures(cells, report)
    write_report(report)
    print(stable_json(report))
    return report


def make_figures(cells: list[dict[str, Any]],
                 report: dict[str, Any]) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cell = cells[0]
    log = json.loads((OUTPUT / cell["env"] / f"s{cell['seed']}" /
                      "decision_log.json").read_text())
    episodes = log["variants"][VARIANTS[-1]]
    episode = max(episodes, key=lambda item: len(
        item["inference"]["versions"]))
    eviction_episodes = log["variants"]["version_store_no_verification"]
    eviction_episode = max(
        eviction_episodes,
        key=lambda item: sum(
            version["status"] == "capacity_evicted"
            for version in item["inference"]["versions"]))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    colors = {
        "provisional": "#f59e0b", "verified": "#16a34a",
        "superseded_verified": "#0f766e", "rejected_verification": "#dc2626",
        "rejected_hysteresis": "#be123c", "capacity_evicted": "#64748b",
    }
    for axis, item, title in (
        (axes[0], episode,
         "Full v3: old versions survive provisional verification"),
        (axes[1], eviction_episode,
         "No-verification ablation: lowest-CE capacity eviction"),
    ):
        for row, version in enumerate(item["inference"]["versions"]):
            y = row
            proposed = version["start"]
            end = (version["evicted_at"] or version["selected_at"]
                   or item["query_t"])
            axis.plot((proposed, end), (y, y), lw=5,
                      color=colors.get(version["status"], "#64748b"))
            axis.scatter(proposed, y, marker="^", color="#7c3aed", s=35)
            if version["promoted_at"] is not None:
                axis.scatter(version["promoted_at"], y, marker="o",
                             color="#16a34a", s=35)
            if version["selected_at"] is not None:
                axis.scatter(version["selected_at"], y, marker="*",
                             color="#2563eb", s=100)
            if version["evicted_at"] is not None:
                axis.scatter(version["evicted_at"], y, marker="x",
                             color="#111827", s=45)
        axis.axvline(item["query_t"], ls=":", color="#111827",
                     label="unknown-delay query")
        axis.set(xlabel="Frame time", ylabel="Version ID", title=title)
        axis.grid(axis="x", alpha=.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"cem_event_versioning_lifelines.{suffix}",
                    bbox_inches="tight", **options)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    labels = ("immediate", "hysteresis", "versions", "full")
    colors = ("#94a3b8", "#f59e0b", "#16a34a", "#2563eb")
    for env_index, environment in enumerate(report["environments"]):
        values = environment["variants"]
        x = np.arange(4) + (env_index - .5) * .25
        axes[0].bar(x, [values[v]["overwrite"] for v in VARIANTS], .25,
                    color=colors, alpha=.72 + .2 * env_index)
        axes[1].bar(x, [values[v]["false_write"] for v in VARIANTS], .25,
                    color=colors, alpha=.72 + .2 * env_index)
        axes[2].bar(x, [values[v]["full_bacc"] for v in VARIANTS], .25,
                    color=colors, alpha=.72 + .2 * env_index,
                    label=environment["env"])
    for axis, title, target in zip(
            axes, ("Overwrite correctness", "False-write rate", "Full BAcc"),
            (.80, .20, .75)):
        axis.axhline(target, color="#dc2626", ls=":")
        axis.set_xticks(np.arange(4), labels, rotation=18, ha="right")
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Event-versioned CEM factorial")
    fig.tight_layout()
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"cem_event_versioning_factorial.{suffix}",
                    bbox_inches="tight", **options)
    plt.close(fig)

    sweep = [row for cell in cells for row in cell["pareto_sweep"]]
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    scatter = axis.scatter(
        [row["false_write"] for row in sweep],
        [row["overwrite"] for row in sweep],
        c=[row["full_bacc"] for row in sweep], cmap="viridis", s=28, alpha=.8)
    axis.axvline(.20, color="#dc2626", ls=":")
    axis.axhline(.80, color="#dc2626", ls=":")
    axis.set(xlabel="False-write rate", ylabel="Overwrite correctness",
             title="Margin × persistence Pareto sweep")
    fig.colorbar(scatter, ax=axis, label="Full balanced accuracy")
    fig.tight_layout()
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"cem_event_versioning_pareto.{suffix}",
                    bbox_inches="tight", **options)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for cell in cells:
        label = f"{cell['env']} s{cell['seed']}"
        capacity = cell["curves"]["capacity"]
        delay = cell["curves"]["max_delay"]
        axes[0].plot(
            [row["capacity"] for row in capacity],
            [row["overwrite"] for row in capacity], marker="o", alpha=.55,
            label=label)
        axes[1].plot(
            [row["max_delay"] for row in delay],
            [row["overwrite"] for row in delay], marker="o", alpha=.55)
    axes[0].axhline(.80, color="#dc2626", ls=":")
    axes[1].axhline(.80, color="#dc2626", ls=":")
    axes[0].set(xlabel="Total version budget", ylabel="Overwrite correctness",
                title="Version-budget curve")
    axes[1].set(xlabel="Maximum query delay (frames)",
                ylabel="Overwrite correctness", title="Unknown-delay curve")
    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"cem_event_versioning_capacity_delay.{suffix}",
                    bbox_inches="tight", **options)
    plt.close(fig)


def write_report(report: dict[str, Any]) -> None:
    lines = [
        "# EVENT-VERSIONED CEM Discovery V3",
        "",
        "## Verdict",
        "",
        f"- Overwrite target >0.80 passed: "
        f"`{str(report['overwrite_target_passed']).lower()}`.",
        f"- All requested quantitative gates passed: "
        f"`{str(report['all_targets_passed']).lower()}`.",
        "- No cue window, onset, duration, or query delay was supplied to the "
        "store or router.",
        "",
        "## Host-scope qualification",
        "",
        f"{report['host_scope']} Each trained event host was frozen before the "
        "factorial/sweep, and its parameter digest was asserted unchanged. "
        "Consequently these runs are real cached OGBench visual-task evaluations, "
        "but they are not evidence about unchanged official DINO-WM weights.",
        "",
        "## Exact aggregate results",
        "",
    ]
    for environment in report["environments"]:
        lines += [
            f"### {environment['env']}", "",
            f"- Seeds: {environment['seeds']}",
            f"- Selected thresholds: {environment['selected_thresholds']}", "",
        ]
        for variant in VARIANTS:
            row = environment["variants"][variant]
            lines += [
                f"**{variant}**",
                f"- Overwrite / false-write: {row['overwrite']:.4f} / "
                f"{row['false_write']:.4f}",
                f"- Full / control BAcc: {row['full_bacc']:.4f} / "
                f"{row['control_bacc']:.4f}",
                f"- Host loss with / without memory: {row['host_loss']:.4f} / "
                f"{row['reset_loss']:.4f}",
                f"- Version selection / stale error / fallback: "
                f"{row['version_selection_accuracy']:.4f} / "
                f"{row['stale_version_error']:.4f} / "
                f"{row['fallback_rate']:.4f}",
                f"- Retrieval precision / recall: "
                f"{row['retrieval_precision']:.4f} / "
                f"{row['retrieval_recall']:.4f}",
                f"- Mean occupancy / capacity evictions: "
                f"{row['occupancy']:.4f} / {row['evictions']:.2f}",
                f"- High-CE / random deletion Δloss: "
                f"{row['high_group_deletion']:.4f} / "
                f"{row['random_group_deletion']:.4f}",
                "",
            ]
    lines += ["## Success gates", ""]
    for key, passed in report["success_targets"].items():
        lines.append(f"- {key}: `{str(passed).lower()}`")
    lines += [
        "",
        "## Architecture and failures",
        "",
        "- Every discovered semantic key owns timestamped versions with value, "
        "CE estimate, confidence, and lifecycle status.",
        "- Full v3 keeps the old verified version while a candidate remains "
        "provisional. Promotion requires persistent delayed CE improvement over "
        "the active same-key version by the selected hysteresis margin.",
        "- Retrieval ranks all live versions by verified CE, semantic confidence, "
        "and a recency kernel, then verifies candidates in order. Failed newest "
        "candidates fall back to an older verified version.",
        "- Eviction occurs only under total version-budget pressure and removes "
        "the lowest-CE live version.",
        "- Any failed gate above is retained as an explicit failure; the sweep "
        "and decision logs are sufficient to inspect rejected, superseded, "
        "selected, fallback, and capacity-evicted versions.",
        "",
        "## Artifacts",
        "",
        "- `outputs/cem_event_versioning_v1/report.json`",
        "- `outputs/cem_event_versioning_v1/<env>/s<seed>/{result.json,"
        "decision_log.json,model.pt}`",
        "- `docs/assets/cem_event_versioning_lifelines.{png,pdf}`",
        "- `docs/assets/cem_event_versioning_factorial.{png,pdf}`",
        "- `docs/assets/cem_event_versioning_pareto.{png,pdf}`",
        "- `docs/assets/cem_event_versioning_capacity_delay.{png,pdf}`",
        "",
        "## Execution",
        "",
        f"- Completed cells: {report['cell_count']}.",
        "- Requested device policy: `cuda:1`; the runner rejects `cuda:3`.",
        f"- Jobs still running: {report['jobs_still_running']}.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS, default=ENVS[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--delta", type=float, default=.02)
    parser.add_argument("--margin", type=float, default=.01)
    parser.add_argument("--persistence-steps", type=int, default=2)
    parser.add_argument("--verify-delay", type=int, default=4)
    parser.add_argument("--query-verify", type=float, default=.35)
    parser.add_argument("--device", default="cuda:1")
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
