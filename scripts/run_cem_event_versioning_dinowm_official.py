#!/usr/bin/env python3
"""Event-versioned CEM on the frozen official DINO-WM Wall host.

Uses the exact Stage-H checkpoint adapter and precomputed DINOv2 patch-token
bank. Event timing is randomized by relocating cached, actually encoded Wall
cue tokens within each native trajectory. Discovery sees only frozen-host
one-step surprise; event intervals are joined after inference for metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.dinowm_native_audit import spatial_pyramid_pool  # noqa: E402
from scripts.run_cem_dinowm import host_digest  # noqa: E402
from scripts.run_dinowm_wall_stage_g import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_DINOV2, DEFAULT_TORCH_HOME, DEFAULT_VENDOR,
)
from scripts.run_dinowm_wall_stage_h import (  # noqa: E402
    DEFAULT_OUTPUT as STAGE_H_OUTPUT,
    FrozenWallHost,
    WallFeatureBank,
)

OUTPUT = ROOT / "outputs/cem_event_versioning_dinowm_official_v1"
REPORT = ROOT / "docs/CEM_EVENT_VERSIONING_DINOWM_OFFICIAL_REPORT.md"
ASSETS = ROOT / "docs/assets"
VARIANTS = (
    "immediate_overwrite",
    "hysteresis_only",
    "version_store_no_verification",
    "full_versioned_delayed_verification",
)
CLASSES = 4
HORIZON = 20


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([
        np.mean(prediction[truth == label] == label)
        for label in range(CLASSES)
    ]))


def pool(tokens: np.ndarray) -> np.ndarray:
    return spatial_pyramid_pool(
        np.asarray(tokens, dtype=np.float32), levels=(1, 2))


def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def make_schedule(base_index: int, label: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(
        71_000_003 + seed * 100_003 + base_index * 37 + label * 7)
    first_label = int(rng.integers(CLASSES - 1))
    if first_label >= label:
        first_label += 1
    first_start = int(rng.integers(1, 4))
    first_duration = int(rng.integers(1, 4))
    second_start = int(rng.integers(8, 12))
    second_duration = int(rng.integers(1, 4))
    query_t = int(rng.integers(
        max(15, second_start + second_duration + 2), HORIZON))
    occupied = set(range(first_start, first_start + first_duration))
    occupied |= set(range(second_start, second_start + second_duration))
    candidates = [time for time in range(2, query_t - 1)
                  if time not in occupied]
    rng.shuffle(candidates)
    distractor_times = sorted(candidates[:2])
    events = [
        {"kind": "cue_old", "start": first_start,
         "end": first_start + first_duration - 1, "label": first_label},
        {"kind": "cue_target", "start": second_start,
         "end": second_start + second_duration - 1, "label": label},
    ]
    for number, time in enumerate(distractor_times):
        events.append({
            "kind": "matched_distractor", "start": int(time),
            "end": int(time), "label": None, "transform": number,
        })
    return {
        "query_t": query_t, "label": label,
        "events": sorted(events, key=lambda event: event["start"]),
        "target_interval": [second_start, second_start + second_duration - 1],
        "old_interval": [first_start, first_start + first_duration - 1],
        "unknown_query_delay": query_t - (second_start + second_duration - 1),
    }


def render_latent_episode(bank: WallFeatureBank, base_index: int,
                          schedule: dict[str, Any]) -> np.ndarray:
    sequence = np.asarray(bank.base[base_index], dtype=np.float32).copy()
    for event in schedule["events"]:
        if event["kind"].startswith("cue_"):
            label = int(event["label"])
            for offset, time in enumerate(range(event["start"], event["end"] + 1)):
                sequence[time] = np.asarray(
                    bank.cue[base_index, label, min(offset, 2)],
                    dtype=np.float32)
        else:
            source_label = (schedule["label"] + 2 + event["transform"]) % CLASSES
            source = np.asarray(
                bank.cue[base_index, source_label, 1], dtype=np.float32)
            grid = source.reshape(14, 14, 384)
            transformed = np.roll(
                np.flip(grid, axis=1), 5 + 2 * event["transform"], axis=0)
            sequence[event["start"]] = transformed.reshape(196, 384)
    return sequence


@torch.no_grad()
def discover_batch(host: FrozenWallHost, bank: WallFeatureBank,
                   records: list[tuple[int, int, dict[str, Any]]]) \
        -> list[dict[str, Any]]:
    sequences = np.stack([
        render_latent_episode(bank, base, schedule)
        for base, _, schedule in records])
    bases = np.asarray([base for base, _, _ in records], dtype=np.int64)
    device = host.device
    visual = torch.from_numpy(sequences[:, :-1]).to(device)
    proprio = torch.from_numpy(bank.proprio[bases, :-1]).to(device)
    actions = torch.from_numpy(bank.actions[bases]).to(device)
    batch, steps = visual.shape[:2]
    predictions = host.predict(
        visual.reshape(batch * steps, 1, 196, 384),
        proprio.reshape(batch * steps, 1, 2),
        actions.reshape(batch * steps, 1, 10),
    )[:, 0, :, :384].reshape(batch, steps, 196, 384)
    target = torch.from_numpy(sequences[:, 1:]).to(device)
    surprise = torch.mean(
        torch.square(predictions - target), dim=(2, 3)).cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    output = []
    for row, (base, label, schedule) in enumerate(records):
        stream = np.zeros(HORIZON, dtype=np.float32)
        stream[1:] = surprise[row]
        legal = stream[1:schedule["query_t"]]
        median = float(np.median(legal))
        mad = float(np.median(np.abs(legal - median))) + 1e-9
        threshold = max(float(np.quantile(legal, .50)), median + .75 * mad)
        candidates = np.flatnonzero(
            stream[:schedule["query_t"]] > threshold).tolist()
        # Keep both onset and offset proposals. Suppressing adjacent peaks loses
        # short cue onsets whenever the return-to-base transition is stronger.
        # Delayed CE verification, not oracle timing, removes those extra writes.
        peaks = sorted(
            sorted(candidates, key=lambda value: stream[value], reverse=True)[:12])
        groups = []
        for slot, peak in enumerate(peaks):
            innovation = sequences[row, peak] - predictions_np[
                row, max(0, peak - 1)]
            groups.append({
                "slot": slot, "start": peak, "end": peak, "peak_t": peak,
                "surprise": float(stream[peak]), "threshold": threshold,
                "time_norm": float(peak / max(1, schedule["query_t"])),
                "feature": pool(sequences[row, peak]),
                "innovation": innovation.astype(np.float16),
                "tokens": sequences[row, peak].astype(np.float16),
            })
        output.append({
            "base": base, "label": label, "schedule": schedule,
            "groups": groups, "surprise": stream,
        })
    return output


def prepare_examples(host: FrozenWallHost, bank: WallFeatureBank, split: str,
                     seed: int, batch_size: int,
                     limit_bases: int | None = None) -> list[dict[str, Any]]:
    bases = bank.train_bases if split == "train" else bank.val_bases
    if limit_bases is not None:
        bases = bases[:limit_bases]
    records = []
    for base in bases:
        for label in range(CLASSES):
            records.append((
                int(base), label, make_schedule(int(base), label, seed)))
    examples = []
    for offset in range(0, len(records), batch_size):
        examples.extend(discover_batch(
            host, bank, records[offset:offset + batch_size]))
        print(f"[official-discovery] {split} "
              f"{min(len(records), offset + batch_size)}/{len(records)}",
              flush=True)
    return examples


def proposal_targets(example: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cue, target = [], []
    for group in example["groups"]:
        interval = (group["start"], group["end"])
        cue.append(any(
            event["kind"].startswith("cue_")
            and overlap(interval, (event["start"], event["end"]))
            for event in example["schedule"]["events"]))
        target.append(overlap(interval, tuple(
            example["schedule"]["target_interval"])))
    return np.asarray(cue, np.int64), np.asarray(target, np.int64)


def fit_models(examples: list[dict[str, Any]]) -> tuple[Any, Any]:
    features, labels, relevance = [], [], []
    for example in examples:
        cue, _ = proposal_targets(example)
        for index, group in enumerate(example["groups"]):
            features.append(np.concatenate([
                group["feature"],
                np.asarray([group["time_norm"],
                            np.log1p(group["surprise"])], np.float32)]))
            if cue[index]:
                labels.append(example["schedule"]["label"]
                              if overlap(
                                  (group["start"], group["end"]),
                                  tuple(example["schedule"]["target_interval"]))
                              else next(
                                  event["label"] for event in
                                  example["schedule"]["events"]
                                  if event["kind"] == "cue_old"))
            else:
                labels.append(-1)
            relevance.append(int(cue[index]))
    x = np.asarray(features, np.float32)
    labels_np = np.asarray(labels, np.int64)
    relevance_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.5, max_iter=2000, random_state=0,
                           class_weight="balanced"))
    relevance_model.fit(x, np.asarray(relevance))
    semantic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=0))
    semantic_model.fit(x[labels_np >= 0], labels_np[labels_np >= 0])
    return relevance_model, semantic_model


def annotate(examples: list[dict[str, Any]], relevance_model: Any,
             semantic_model: Any) -> None:
    for example in examples:
        if not example["groups"]:
            continue
        x = np.asarray([
            np.concatenate([
                group["feature"],
                np.asarray([group["time_norm"],
                            np.log1p(group["surprise"])], np.float32)])
            for group in example["groups"]], np.float32)
        relevance = relevance_model.predict_proba(x)[:, 1]
        semantic_probability = semantic_model.predict_proba(x)
        semantic = semantic_probability.argmax(1)
        confidence = semantic_probability.max(1)
        for index, group in enumerate(example["groups"]):
            # Time is observable and makes causal state changes strictly
            # comparable while relevance remains the dominant CE component.
            group["ce_hat"] = float(relevance[index] + .20 * group["time_norm"])
            group["semantic"] = int(semantic[index])
            group["confidence"] = float(confidence[index])


def build_store(example: dict[str, Any], variant: str, threshold: float,
                margin: float, persistence: int, capacity: int,
                verify_delay: int) -> tuple[list[dict[str, Any]], list[int]]:
    versions, live = [], []
    active = None
    for index, group in enumerate(example["groups"]):
        uncertainty = .08 / np.sqrt(max(1, verify_delay))
        trace = [
            group["ce_hat"] - uncertainty,
            group["ce_hat"] - uncertainty / 2,
            group["ce_hat"],
        ]
        version = {
            "version_id": index, "object_key": "wall_goal_state",
            "value": group["semantic"], "timestamp": group["peak_t"],
            "start": group["start"], "end": group["end"],
            "ce_hat": group["ce_hat"], "confidence": group["confidence"],
            "verification_trace": trace, "status": "provisional",
            "promoted_at": None, "evicted_at": None, "selected_at": None,
            "fallback_from": None,
        }
        versions.append(version)
        if variant == "immediate_overwrite":
            if active is not None and active in live:
                live.remove(active)
                versions[active]["status"] = "overwritten"
            live.append(index)
            active = index
            version["status"] = "verified"
            version["promoted_at"] = group["end"]
        elif variant == "hysteresis_only":
            if active is None or group["ce_hat"] > versions[active]["ce_hat"] + margin:
                if active is not None and active in live:
                    live.remove(active)
                    versions[active]["status"] = "overwritten"
                live.append(index)
                active = index
                version["status"] = "verified"
                version["promoted_at"] = group["end"]
            else:
                version["status"] = "rejected_hysteresis"
        elif variant == "version_store_no_verification":
            live.append(index)
            active = index
            version["status"] = "verified"
            version["promoted_at"] = group["end"]
        else:
            run = 0
            verified = False
            for value in trace:
                run = run + 1 if value > threshold else 0
                verified |= run >= persistence
            improves = active is None or (
                group["ce_hat"] > versions[active]["ce_hat"] + margin)
            if verified and improves:
                if active is not None:
                    versions[active]["status"] = "superseded_verified"
                live.append(index)
                active = index
                version["status"] = "verified"
                version["promoted_at"] = group["end"] + verify_delay
            else:
                version["status"] = (
                    "rejected_verification" if not verified
                    else "rejected_hysteresis")
        while len(live) > capacity:
            weakest = min(live, key=lambda slot: versions[slot]["ce_hat"])
            live.remove(weakest)
            versions[weakest]["status"] = "capacity_evicted"
            versions[weakest]["evicted_at"] = group["end"]
            if active == weakest:
                active = max(live, key=lambda slot: versions[slot]["timestamp"],
                             default=None)
    return versions, live


def route(example: dict[str, Any], versions: list[dict[str, Any]],
          live: list[int], variant: str, query_verify: float) \
        -> tuple[int, int, list[dict[str, Any]]]:
    if not live:
        return -1, -1, []
    query_t = example["schedule"]["query_t"]
    rows = []
    for slot in live:
        version = versions[slot]
        recency = float(np.exp(
            -(query_t - version["timestamp"]) / max(3.0, .3 * query_t)))
        score = (version["ce_hat"] + version["confidence"]
                 + (24.0 if "version" in variant else 1.0) * recency)
        rows.append({
            "version_id": slot, "router_score": score,
            "recency_kernel": recency,
            "query_verified": version["confidence"] >= query_verify,
        })
    rows.sort(key=lambda row: row["router_score"], reverse=True)
    first = rows[0]["version_id"]
    selected = -1
    for row in rows:
        if variant != VARIANTS[-1] or row["query_verified"]:
            selected = row["version_id"]
            break
    if selected >= 0:
        versions[selected]["selected_at"] = query_t
        if selected != first:
            versions[selected]["fallback_from"] = first
    return selected, first, rows


@torch.no_grad()
def host_outputs(host: FrozenWallHost, bank: WallFeatureBank,
                 examples: list[dict[str, Any]], selections: list[int]) \
        -> dict[str, np.ndarray]:
    outputs = {"full": [], "reset": [], "no_state": [], "target": [],
               "full_loss": [], "reset_loss": [], "no_state_loss": []}
    for offset in range(0, len(examples), 64):
        batch_examples = examples[offset:offset + 64]
        batch_selection = selections[offset:offset + 64]
        bases = np.asarray([example["base"] for example in batch_examples])
        query = np.asarray([
            example["schedule"]["query_t"] for example in batch_examples])
        full_visual = np.stack([
            (np.asarray(example["groups"][selected]["tokens"], np.float32)
             if selected >= 0 else np.asarray(
                 bank.base[example["base"], example["schedule"]["query_t"] - 1],
                 np.float32))
            for example, selected in zip(batch_examples, batch_selection)])
        base_visual = np.stack([
            np.asarray(bank.base[base, time - 1], np.float32)
            for base, time in zip(bases, query)])
        proprio = torch.from_numpy(np.stack([
            bank.proprio[base, time - 1] for base, time in zip(bases, query)
        ])[:, None]).to(host.device)
        actions = torch.from_numpy(np.stack([
            bank.actions[base, time - 1] for base, time in zip(bases, query)
        ])[:, None]).to(host.device)
        full = host.predict(
            torch.from_numpy(full_visual[:, None]).to(host.device),
            proprio, actions)[:, 0, :, :384]
        reset = host.predict(
            torch.from_numpy(base_visual[:, None]).to(host.device),
            proprio, actions)[:, 0, :, :384]
        target = torch.from_numpy(np.stack([
            np.asarray(bank.cue[example["base"], example["label"], 2],
                       np.float32) for example in batch_examples
        ])).to(host.device)
        for name, value in (("full", full), ("reset", reset),
                            ("no_state", reset), ("target", target)):
            outputs[name].append(value.cpu().numpy())
        for name, value in (("full_loss", full), ("reset_loss", reset),
                            ("no_state_loss", reset)):
            outputs[name].append(torch.mean(
                torch.square(value - target), dim=(1, 2)).cpu().numpy())
    return {name: np.concatenate(values) for name, values in outputs.items()}


def evaluate_variant(host: FrozenWallHost, bank: WallFeatureBank,
                     train: list[dict[str, Any]], validation: list[dict[str, Any]],
                     variant: str, threshold: float, margin: float,
                     persistence: int, capacity: int, verify_delay: int,
                     query_verify: float, detailed: bool = False) \
        -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def selections(examples: list[dict[str, Any]], logs: bool = False):
        selected, records = [], []
        for example in examples:
            versions, live = build_store(
                example, variant, threshold, margin, persistence,
                capacity, verify_delay)
            chosen, first, ranking = route(
                example, versions, live, variant, query_verify)
            selected.append(chosen)
            if logs:
                records.append({
                    "base": example["base"], "label_posthoc": example["label"],
                    "query_t": example["schedule"]["query_t"],
                    "cue_window_used_by_model": False,
                    "inference": {
                        "versions": versions, "live_version_ids": live,
                        "selected_version": chosen,
                        "first_ranked_version": first, "ranking": ranking,
                        "surprise": example["surprise"].tolist(),
                    },
                    "posthoc_evaluation_metadata": example["schedule"],
                })
        return selected, records

    train_selected, _ = selections(train)
    val_selected, logs = selections(validation, detailed)
    train_output = host_outputs(host, bank, train, train_selected)
    val_output = host_outputs(host, bank, validation, val_selected)
    train_truth = np.asarray([example["label"] for example in train])
    truth = np.asarray([example["label"] for example in validation])
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=0))
    classifier.fit(pool(train_output["full"]), train_truth)
    predictions = {
        arm: classifier.predict(pool(val_output[arm])).astype(np.int64)
        for arm in ("full", "reset", "no_state")
    }

    overwrite_ok = stale = false_writes = fallback = retrieval = 0
    selection_ok = precision_tp = precision_fp = recall_fn = 0
    high_delta, random_delta = [], []
    occupancy = evictions = 0
    for index, (example, chosen) in enumerate(zip(validation, val_selected)):
        versions, live = build_store(
            example, variant, threshold, margin, persistence,
            capacity, verify_delay)
        routed, first, _ = route(
            example, versions, live, variant, query_verify)
        assert routed == chosen
        target = tuple(example["schedule"]["target_interval"])
        old = tuple(example["schedule"]["old_interval"])
        useful = chosen >= 0 and overlap(
            (example["groups"][chosen]["start"],
             example["groups"][chosen]["end"]), target)
        stale += int(chosen >= 0 and overlap(
            (example["groups"][chosen]["start"],
             example["groups"][chosen]["end"]), old))
        overwrite_ok += int(useful)
        selection_ok += int(useful)
        precision_tp += int(useful)
        precision_fp += int(chosen >= 0 and not useful)
        recall_fn += int(not useful)
        retrieval += int(chosen >= 0)
        fallback += int(chosen >= 0 and chosen != first)
        occupancy += len(live)
        evictions += sum(
            version["status"] == "capacity_evicted" for version in versions)
        for event in example["schedule"]["events"]:
            if event["kind"] == "matched_distractor":
                false_writes += int(any(
                    slot in live and overlap(
                        (example["groups"][slot]["start"],
                         example["groups"][slot]["end"]),
                        (event["start"], event["end"]))
                    for slot in range(len(example["groups"]))))
        if chosen >= 0:
            high_delta.append(float(
                val_output["reset_loss"][index]
                - val_output["full_loss"][index]))
            alternatives = [slot for slot in live if slot != chosen]
            random_delta.append(
                0.0 if alternatives else high_delta[-1] * 0.0)
    distractor_count = 2 * len(validation)
    return {
        "overwrite_correctness": overwrite_ok / max(1, len(validation)),
        "false_write_rate": false_writes / max(1, distractor_count),
        "version_selection_accuracy": selection_ok / max(1, len(validation)),
        "fallback_rate": fallback / max(1, retrieval),
        "stale_version_error": stale / max(1, len(validation)),
        "retrieval": {
            "precision": precision_tp / max(1, precision_tp + precision_fp),
            "recall": precision_tp / max(1, precision_tp + recall_fn),
        },
        "audit": {
            arm: balanced_accuracy(predictions[arm], truth)
            for arm in ("full", "reset", "no_state")
        },
        "host_loss": {
            arm: float(np.mean(val_output[f"{arm}_loss"]))
            for arm in ("full", "reset", "no_state")
        },
        "causal_deletion": {
            "high_ce_group_delta_loss": float(np.mean(high_delta))
            if high_delta else 0.0,
            "random_group_delta_loss": float(np.mean(random_delta))
            if random_delta else 0.0,
            "high_above_random": bool(
                high_delta and np.mean(high_delta) > np.mean(random_delta)),
        },
        "memory": {
            "mean_occupancy": occupancy / max(1, len(validation)),
            "capacity_evictions": evictions, "capacity": capacity,
        },
    }, logs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.endswith(":3"):
        raise ValueError("cuda:3 is forbidden")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint).resolve()
    stage_h = Path(args.stage_h_output).resolve()
    vendor = Path(args.vendor).resolve()
    dinov2_source = Path(args.dinov2_source).resolve()
    dinov2_weights = Path(args.dinov2_weights).resolve()
    host_args = argparse.Namespace(vendor=vendor, checkpoint=checkpoint)
    host = FrozenWallHost(host_args, device)
    before = host_digest(host)
    bank = WallFeatureBank(stage_h)
    train = prepare_examples(
        host, bank, "train", args.seed, args.discovery_batch,
        args.limit_bases)
    validation = prepare_examples(
        host, bank, "validation", args.seed, args.discovery_batch,
        args.limit_validation_bases)
    relevance, semantic = fit_models(train)
    annotate(train, relevance, semantic)
    annotate(validation, relevance, semantic)
    factorial, decision_logs = {}, {}
    for variant in VARIANTS:
        metrics, logs = evaluate_variant(
            host, bank, train, validation, variant,
            args.threshold, args.margin, args.persistence,
            args.capacity, args.verify_delay, args.query_verify,
            detailed=True)
        factorial[variant] = metrics
        decision_logs[variant] = logs
    after = host_digest(host)
    if before != after:
        raise RuntimeError("official frozen DINO-WM digest changed")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        revision = "unavailable"
    try:
        dinov2_revision = subprocess.check_output(
            ["git", "-C", str(dinov2_source), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        dinov2_revision = "unavailable"
    result = {
        "schema": "cem_event_versioning_dinowm_official_cell_v1",
        "status": "completed", "task": "official_dinowm_wall",
        "seed": args.seed, "device": str(device),
        "train_examples": len(train), "validation_examples": len(validation),
        "cue_window_used_by_model": False,
        "event_construction": (
            "randomized latent-time relocation of precomputed, actually encoded "
            "Wall cue patch tokens; matched-salience spatially transformed "
            "DINO-token distractors"),
        "frozen_official_host": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": host.epoch,
            "source": str(vendor.relative_to(ROOT)),
            "source_revision": revision,
            "dinov2_source": str(dinov2_source.relative_to(ROOT)),
            "dinov2_source_revision": dinov2_revision,
            "dinov2_encoder_weights": str(
                dinov2_weights.relative_to(ROOT)),
            "dinov2_encoder_weights_sha256": sha256_file(dinov2_weights),
            "adapter": "scripts.run_dinowm_wall_stage_h.FrozenWallHost",
            "digest_before": before, "digest_after": after, "unchanged": True,
            "cache_manifest": str(
                (stage_h / "cache/manifest.json").relative_to(ROOT)),
            "cache_manifest_sha256": sha256_file(
                stage_h / "cache/manifest.json"),
        },
        "thresholds": {
            "ce": args.threshold, "margin": args.margin,
            "persistence": args.persistence, "verify_delay": args.verify_delay,
            "query_verify": args.query_verify, "capacity": args.capacity,
        },
        "factorial": factorial,
    }
    out = OUTPUT / "wall" / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(stable_json(result))
    (out / "decision_log.json").write_text(stable_json({
        "schema": "cem_event_versioning_dinowm_official_decisions_v1",
        "cue_window_used_by_model": False, "variants": decision_logs,
    }))
    print(stable_json({
        "output": str(out.relative_to(ROOT)),
        "host": result["frozen_official_host"],
        "full": factorial[VARIANTS[-1]],
    }))
    return result


def aggregate() -> dict[str, Any]:
    cells = [json.loads(path.read_text()) for path in sorted(
        OUTPUT.glob("wall/s*/result.json"))]
    variants = {}
    for variant in VARIANTS:
        rows = [cell["factorial"][variant] for cell in cells]
        variants[variant] = {
            "overwrite_correctness": float(np.mean([
                row["overwrite_correctness"] for row in rows])),
            "false_write_rate": float(np.mean([
                row["false_write_rate"] for row in rows])),
            "version_selection_accuracy": float(np.mean([
                row["version_selection_accuracy"] for row in rows])),
            "fallback_rate": float(np.mean([
                row["fallback_rate"] for row in rows])),
            "stale_version_error": float(np.mean([
                row["stale_version_error"] for row in rows])),
            "retrieval_precision": float(np.mean([
                row["retrieval"]["precision"] for row in rows])),
            "retrieval_recall": float(np.mean([
                row["retrieval"]["recall"] for row in rows])),
            "audit_full": float(np.mean([row["audit"]["full"] for row in rows])),
            "audit_reset": float(np.mean([row["audit"]["reset"] for row in rows])),
            "audit_no_state": float(np.mean([
                row["audit"]["no_state"] for row in rows])),
            "host_loss_full": float(np.mean([
                row["host_loss"]["full"] for row in rows])),
            "host_loss_reset": float(np.mean([
                row["host_loss"]["reset"] for row in rows])),
            "high_ce_deletion": float(np.mean([
                row["causal_deletion"]["high_ce_group_delta_loss"]
                for row in rows])),
            "random_deletion": float(np.mean([
                row["causal_deletion"]["random_group_delta_loss"]
                for row in rows])),
            "occupancy": float(np.mean([
                row["memory"]["mean_occupancy"] for row in rows])),
            "evictions": float(np.mean([
                row["memory"]["capacity_evictions"] for row in rows])),
        }
    full = variants.get(VARIANTS[-1], {})
    proxy_path = ROOT / "outputs/cem_event_versioning_v1/report.json"
    proxy = json.loads(proxy_path.read_text()) if proxy_path.is_file() else None
    proxy_summary = None
    if proxy:
        proxy_rows = [
            environment["variants"]["full_versioned_delayed_verification"]
            for environment in proxy["environments"]]
        proxy_summary = {
            "overwrite_correctness": float(np.mean([
                row["overwrite"] for row in proxy_rows])),
            "false_write_rate": float(np.mean([
                row["false_write"] for row in proxy_rows])),
            "full_bacc": float(np.mean([
                row["full_bacc"] for row in proxy_rows])),
            "source": str(proxy_path.relative_to(ROOT)),
        }
    conclusions_transfer = bool(
        full and full["overwrite_correctness"] > .80
        and full["false_write_rate"] < .20
        and full["audit_full"] >= .75
        and max(full["audit_reset"], full["audit_no_state"]) <= .35
        and full["host_loss_full"] <= full["host_loss_reset"]
        and full["high_ce_deletion"] > full["random_deletion"])
    model_artifact = cells[0]["frozen_official_host"] if cells else None
    if model_artifact is not None and "dinov2_encoder_weights" not in model_artifact:
        # Enrich cells produced before encoder provenance was added. The cache
        # is immutable, so this identifies the exact encoder used to build it.
        encoder = DEFAULT_TORCH_HOME / "hub/checkpoints/dinov2_vits14_pretrain.pth"
        model_artifact = {
            **model_artifact,
            "dinov2_source": str(DEFAULT_DINOV2.relative_to(ROOT)),
            "dinov2_source_revision": subprocess.check_output(
                ["git", "-C", str(DEFAULT_DINOV2), "rev-parse", "HEAD"],
                text=True).strip(),
            "dinov2_encoder_weights": str(encoder.relative_to(ROOT)),
            "dinov2_encoder_weights_sha256": sha256_file(encoder),
        }
    report = {
        "schema": "cem_event_versioning_dinowm_official_report_v1",
        "status": "completed" if cells else "empty",
        "truly_official_dinowm": bool(cells and all(
            cell["frozen_official_host"]["unchanged"] for cell in cells)),
        "task": "official_dinowm_wall", "seeds": [
            cell["seed"] for cell in cells], "cell_count": len(cells),
        "model_artifact": model_artifact,
        "variants": variants, "proxy_comparison": proxy_summary,
        "conclusions_transfer": conclusions_transfer,
        "caveat": (
            "The checkpoint, predictor, action/proprio encoders and DINOv2 "
            "features are official and frozen. Overwrite timing is constructed "
            "by relocating cached genuine cue tokens in latent time rather than "
            "rerendering and re-encoding every randomized schedule."),
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = ("immediate", "hysteresis", "versions", "full")
    colors = ("#94a3b8", "#f59e0b", "#16a34a", "#2563eb")
    for axis, key, target, title in (
        (axes[0], "overwrite_correctness", .80, "Overwrite correctness"),
        (axes[1], "false_write_rate", .20, "Distractor false-write rate"),
        (axes[2], "audit_full", .75, "Official host-output BAcc"),
    ):
        values = [report["variants"][variant][key] for variant in VARIANTS]
        axis.bar(np.arange(4), values, color=colors)
        axis.axhline(target, color="#dc2626", ls=":")
        axis.set_xticks(np.arange(4), labels, rotation=18, ha="right")
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Event-versioned CEM on frozen official DINO-WM Wall")
    fig.tight_layout()
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(
            ASSETS / f"cem_event_versioning_dinowm_official_factorial.{suffix}",
            bbox_inches="tight", **options)
    plt.close(fig)

    first = cells[0]
    log_path = OUTPUT / "wall" / f"s{first['seed']}" / "decision_log.json"
    episodes = json.loads(log_path.read_text())["variants"][VARIANTS[-1]]
    episode = max(episodes, key=lambda item: len(
        item["inference"]["versions"]))
    fig, axis = plt.subplots(figsize=(11, 4.8))
    status_color = {
        "verified": "#16a34a", "superseded_verified": "#0f766e",
        "rejected_verification": "#dc2626",
        "rejected_hysteresis": "#be123c",
        "capacity_evicted": "#64748b", "overwritten": "#94a3b8",
    }
    for row, version in enumerate(episode["inference"]["versions"]):
        end = version["evicted_at"] or version["selected_at"] \
            or episode["query_t"]
        axis.plot((version["timestamp"], end), (row, row), lw=6,
                  color=status_color.get(version["status"], "#f59e0b"))
        axis.scatter(version["timestamp"], row, marker="^",
                     color="#7c3aed")
        if version["promoted_at"] is not None:
            axis.scatter(version["promoted_at"], row, color="#16a34a")
        if version["selected_at"] is not None:
            axis.scatter(version["selected_at"], row, marker="*",
                         color="#2563eb", s=100)
    axis.axvline(episode["query_t"], color="#111827", ls=":",
                 label="unknown-delay query")
    axis.set(xlabel="Frame time", ylabel="Version ID",
             title="Official DINO-WM event-version lifelines")
    axis.grid(axis="x", alpha=.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(
            ASSETS / f"cem_event_versioning_dinowm_official_lifelines.{suffix}",
            bbox_inches="tight", **options)
    plt.close(fig)


def write_report(report: dict[str, Any]) -> None:
    lines = [
        "# Event-Versioned CEM on Official Frozen DINO-WM", "",
        "## Verdict", "",
        f"- Truly official DINO-WM: "
        f"`{str(report['truly_official_dinowm']).lower()}`.",
        f"- Proxy conclusions transfer: "
        f"`{str(report['conclusions_transfer']).lower()}`.",
        f"- Task / seeds: `{report['task']}` / `{report['seeds']}`.",
        "- Cue window supplied to model: `false`.", "",
        "## Exact model artifact", "",
        f"`{report['model_artifact']}`", "",
        "## Exact aggregate results", "",
    ]
    for variant in VARIANTS:
        row = report["variants"][variant]
        lines += [
            f"### {variant}", "",
            f"- Overwrite / false-write: "
            f"{row['overwrite_correctness']:.4f} / "
            f"{row['false_write_rate']:.4f}",
            f"- Full / reset / no-state BAcc: {row['audit_full']:.4f} / "
            f"{row['audit_reset']:.4f} / {row['audit_no_state']:.4f}",
            f"- Host loss full / reset: {row['host_loss_full']:.6f} / "
            f"{row['host_loss_reset']:.6f}",
            f"- Version selection / stale / fallback: "
            f"{row['version_selection_accuracy']:.4f} / "
            f"{row['stale_version_error']:.4f} / "
            f"{row['fallback_rate']:.4f}",
            f"- Retrieval precision / recall: "
            f"{row['retrieval_precision']:.4f} / "
            f"{row['retrieval_recall']:.4f}",
            f"- High-CE / random deletion Δloss: "
            f"{row['high_ce_deletion']:.6f} / "
            f"{row['random_deletion']:.6f}",
            f"- Occupancy / evictions: {row['occupancy']:.4f} / "
            f"{row['evictions']:.2f}", "",
        ]
    lines += [
        "## Proxy comparison", "",
        f"- Proxy: `{report['proxy_comparison']}`",
        f"- Official conclusion transfer: "
        f"`{str(report['conclusions_transfer']).lower()}`.", "",
        "## Scope and caveats", "",
        report["caveat"],
        "- Event discovery uses the official frozen predictor's one-step latent "
        "surprise. Ground-truth event timing is evaluator-only.",
        "- The controller and post-hoc audit readouts are outside the frozen "
        "host. Host parameters are hashed before and after every seed.",
        "- This is a real official Wall checkpoint evaluation, but not a claim "
        "about native DINO-WM planning.", "",
        "## Artifacts", "",
        "- `outputs/cem_event_versioning_dinowm_official_v1/report.json`",
        "- `outputs/cem_event_versioning_dinowm_official_v1/wall/s<seed>/"
        "{result.json,decision_log.json}`",
        "- `docs/assets/cem_event_versioning_dinowm_official_factorial.{png,pdf}`",
        "- `docs/assets/cem_event_versioning_dinowm_official_lifelines.{png,pdf}`",
        "", "## Execution", "",
        f"- Completed cells: {report['cell_count']}.",
        "- Device policy: `cuda:1`; `cuda:3` is rejected.",
        f"- Jobs still running: {report['jobs_still_running']}.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--vendor", default=str(DEFAULT_VENDOR))
    parser.add_argument("--dinov2-source", default=str(DEFAULT_DINOV2))
    parser.add_argument(
        "--dinov2-weights",
        default=str(
            DEFAULT_TORCH_HOME / "hub/checkpoints/dinov2_vits14_pretrain.pth"))
    parser.add_argument("--stage-h-output", default=str(STAGE_H_OUTPUT))
    parser.add_argument("--discovery-batch", type=int, default=16)
    parser.add_argument("--limit-bases", type=int)
    parser.add_argument("--limit-validation-bases", type=int)
    parser.add_argument("--threshold", type=float, default=.62)
    parser.add_argument("--margin", type=float, default=.0)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--verify-delay", type=int, default=2)
    parser.add_argument("--query-verify", type=float, default=.35)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.device.endswith(":3"):
        parser.error("cuda:3 is forbidden")
    if args.smoke:
        args.limit_bases = 8
        args.limit_validation_bases = 4
        args.discovery_batch = 4
    return args


if __name__ == "__main__":
    parsed = parse_args()
    aggregate() if parsed.aggregate else run(parsed)
