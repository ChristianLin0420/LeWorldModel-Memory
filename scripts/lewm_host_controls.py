#!/usr/bin/env python3
"""Reviewer-requested control battery for the frozen LeWM PushT host-writer.

This script *reuses* the already-trained Host-Aligned Evidence Writer adapters
checkpointed under
``outputs/lewm_pusht_host_writer_counterfactual_checkpointed_v1`` and the frozen
official PushT LeWorldModel host.  It does **not** retrain any adapter.  For each
seed it measures host-output balanced accuracy (post-hoc RidgeClassifier readout
trained on the TRAIN split full condition, exactly as in the primary run) under a
battery of controls beyond full/reset/no-state:

* ``host_only``      -- frozen host on its legal 3-frame context, no memory
                        injected (identical computation to ``no_state``; the
                        native host capability baseline).
* ``correct``        -- the primary result: correct-episode slot memory injected
                        (``full`` condition).
* ``reset``          -- memory built from the reset prefix (final context only).
* ``no_state``       -- no memory injected (native host).
* ``shuffled_episode`` -- inject memory built from a DIFFERENT random episode's
                        prefix into this episode's context (episode specificity).
* ``random_memory``  -- inject a random residual of matching per-position L2 norm
                        (injection-path artifact control).
* ``memory_only``    -- probe the injected residual directly (bypassing the host
                        predictor) to test whether the host adds a useful
                        transformation.

The frozen host state dict digest is asserted unchanged before and after.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm_pusht import (  # noqa: E402
    load_official_pusht_checkpoint,
)
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    pusht_task_spec,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    resolve_pusht_path,
)
from scripts.run_mem_jepa_stage_b import (  # noqa: E402
    atomic_json,
    fit_classifier,
    resolve,
    set_determinism,
)
from scripts.run_lewm_pusht_host_writer import (  # noqa: E402
    LeWMHostAlignedEvidenceWriter,
    age_adjusted_spec,
    batch_arrays,
    classification_record,
    load_admitted,
    load_or_build_counterfactual_cache,
    no_state_arrays,
    predict_last,
    state_digest,
)


DEFAULT_CHECKPOINT_ROOT = (
    ROOT / "outputs/lewm_pusht_host_writer_counterfactual_checkpointed_v1"
)
DEFAULT_COUNTERFACTUAL_CACHE = (
    ROOT / "outputs/lewm_pusht_counterfactual_cue_cache_v1"
)
DEFAULT_OUTPUT = ROOT / "outputs/lewm_host_controls_v1"

# Bar order for the figure / paste-ready table.
CONDITION_ORDER = (
    "host_only",
    "correct",
    "reset",
    "no_state",
    "shuffled_episode",
    "random_memory",
    "memory_only",
)
CONDITION_LABELS = {
    "host_only": "host-only",
    "correct": "correct",
    "reset": "reset",
    "no_state": "no-state",
    "shuffled_episode": "shuffled",
    "random_memory": "random",
    "memory_only": "memory-only",
}


def derangement(count: int, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation with no fixed points (episode -> other episode)."""
    if count < 2:
        raise ValueError("derangement needs at least two episodes")
    while True:
        perm = rng.permutation(count)
        if not np.any(perm == np.arange(count)):
            return perm


@torch.no_grad()
def host_output_features(model: LeWMHostAlignedEvidenceWriter,
                         host: torch.nn.Module, data: dict[str, Any],
                         spec: dict[str, Any], *, condition: str,
                         batch_size: int, device: torch.device,
                         source_perm: np.ndarray | None = None,
                         rng: torch.Generator | None = None) -> np.ndarray:
    """Host prediction at the final legal context under a control condition."""
    model.eval()
    count = len(data["labels"])
    indices = np.arange(count, dtype=np.int64)
    features: list[np.ndarray] = []
    for offset in range(0, count, batch_size):
        rows = indices[offset:offset + batch_size]
        if condition in {"no_state", "host_only"}:
            batch = no_state_arrays(data, rows, spec, device)
            value = predict_last(host, batch["context_z"],
                                 batch["context_actions"])
        elif condition in {"full", "reset"}:
            batch = batch_arrays(data, rows, spec, condition, device)
            fused, _ = model.inject(
                batch["prefix_z"], batch["prefix_actions"],
                batch["prefix_times"], batch["context_z"],
                batch["context_actions"], batch["context_times"])
            value = predict_last(host, fused, batch["context_actions"])
        elif condition == "shuffled_episode":
            if source_perm is None:
                raise ValueError("shuffled_episode requires source_perm")
            src_rows = source_perm[rows]
            src = batch_arrays(data, src_rows, spec, "full", device)
            dst = batch_arrays(data, rows, spec, "full", device)
            fused, _ = model.inject(
                src["prefix_z"], src["prefix_actions"], src["prefix_times"],
                dst["context_z"], dst["context_actions"], dst["context_times"])
            value = predict_last(host, fused, dst["context_actions"])
        elif condition == "random_memory":
            if rng is None:
                raise ValueError("random_memory requires an rng")
            batch = batch_arrays(data, rows, spec, "full", device)
            fused, _ = model.inject(
                batch["prefix_z"], batch["prefix_actions"],
                batch["prefix_times"], batch["context_z"],
                batch["context_actions"], batch["context_times"])
            residual = fused - batch["context_z"]
            noise = torch.randn(residual.shape, generator=rng, device=device,
                                dtype=residual.dtype)
            res_norm = residual.norm(dim=-1, keepdim=True)
            noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            matched = noise * (res_norm / noise_norm)
            fused_random = batch["context_z"] + matched
            value = predict_last(host, fused_random, batch["context_actions"])
        else:
            raise ValueError(f"unknown condition: {condition}")
        features.append(value.float().cpu().numpy())
    return np.concatenate(features)


@torch.no_grad()
def injected_residual_features(model: LeWMHostAlignedEvidenceWriter,
                               data: dict[str, Any], spec: dict[str, Any], *,
                               batch_size: int,
                               device: torch.device) -> np.ndarray:
    """Flattened residual actually injected into the host input (memory-only)."""
    model.eval()
    count = len(data["labels"])
    indices = np.arange(count, dtype=np.int64)
    features: list[np.ndarray] = []
    for offset in range(0, count, batch_size):
        rows = indices[offset:offset + batch_size]
        batch = batch_arrays(data, rows, spec, "full", device)
        fused, _ = model.inject(
            batch["prefix_z"], batch["prefix_actions"], batch["prefix_times"],
            batch["context_z"], batch["context_actions"],
            batch["context_times"])
        residual = (fused - batch["context_z"]).reshape(len(rows), -1)
        features.append(residual.float().cpu().numpy())
    return np.concatenate(features)


def evaluate_seed(seed: int, *, host: torch.nn.Module, train: dict[str, Any],
                  validation: dict[str, Any], spec: dict[str, Any],
                  classes: int, checkpoint_root: Path, task: str, age: int,
                  batch_size: int, device: torch.device) -> dict[str, Any]:
    checkpoint_path = (checkpoint_root / task / f"s{seed}" / f"age_{age}"
                       / "adapter.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpointed adapter: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    cfg = checkpoint["model"]
    model = LeWMHostAlignedEvidenceWriter(
        target_dim=int(cfg["target_dim"]), dim=int(cfg["dim"]),
        slots=int(cfg["slots"]), heads=int(cfg["heads"]),
        residual_scale=float(cfg["residual_scale"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    set_determinism(seed)
    np_rng = np.random.default_rng(770_000 + seed)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(880_000 + seed)
    source_perm = derangement(len(validation["labels"]), np_rng)

    train_y = train["labels"]
    validation_y = validation["labels"]

    # Readout is trained on the TRAIN split full-condition host output, exactly
    # as in the primary run, then frozen and applied to every validation arm.
    train_full = host_output_features(
        model, host, train, spec, condition="full",
        batch_size=batch_size, device=device)

    host_output_conditions = {
        "correct": dict(condition="full"),
        "reset": dict(condition="reset"),
        "no_state": dict(condition="no_state"),
        "host_only": dict(condition="host_only"),
        "shuffled_episode": dict(condition="shuffled_episode",
                                 source_perm=source_perm),
        "random_memory": dict(condition="random_memory", rng=torch_rng),
    }
    records: dict[str, Any] = {}
    for name, kwargs in host_output_conditions.items():
        features = host_output_features(
            model, host, validation, spec, batch_size=batch_size,
            device=device, **kwargs)
        prediction = fit_classifier(train_full, train_y, features)
        records[name] = classification_record(prediction, validation_y, classes)

    # Memory-only probe: train + apply the readout on the injected residual,
    # bypassing the host predictor entirely.
    train_residual = injected_residual_features(
        model, train, spec, batch_size=batch_size, device=device)
    val_residual = injected_residual_features(
        model, validation, spec, batch_size=batch_size, device=device)
    mem_prediction = fit_classifier(train_residual, train_y, val_residual)
    records["memory_only"] = classification_record(
        mem_prediction, validation_y, classes)

    return {
        "seed": int(seed),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_host_digest": checkpoint.get("host_digest"),
        "records": {
            name: {
                "balanced_accuracy": records[name]["balanced_accuracy"],
                "per_class_recall": records[name]["per_class_recall"],
            }
            for name in CONDITION_ORDER
        },
    }


def aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for name in CONDITION_ORDER:
        values = [
            float(entry["records"][name]["balanced_accuracy"])
            for entry in per_seed
        ]
        conditions[name] = {
            "label": CONDITION_LABELS[name],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "seed_values": values,
        }
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(DEFAULT_PUSHT_SPEC))
    parser.add_argument("--lock", default=str(DEFAULT_PUSHT_LOCK))
    parser.add_argument("--task", default="multi-item-visual-binding-recall")
    parser.add_argument("--age", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--counterfactual-cache",
                        default=str(DEFAULT_COUNTERFACTUAL_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    locked_spec = load_locked_pusht_spec(resolve(args.spec), resolve(args.lock))
    spec = age_adjusted_spec(locked_spec, int(args.age))
    task_record = pusht_task_spec(locked_spec, args.task)
    classes = int(task_record["classes"])

    train = load_admitted(locked_spec, args.task, "train")
    validation = load_admitted(locked_spec, args.task, "validation")

    bundle = resolve_pusht_path(locked_spec["official_host"]["bundle_path"])
    host = load_official_pusht_checkpoint(bundle, device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    host_before = state_digest(host)

    cache_root = resolve(args.counterfactual_cache)
    cache_records = {
        "train": load_or_build_counterfactual_cache(
            train, spec, args.task, "train", host, device, cache_root,
            int(args.frame_batch_size)),
        "validation": load_or_build_counterfactual_cache(
            validation, spec, args.task, "validation", host, device,
            cache_root, int(args.frame_batch_size)),
    }

    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        entry = evaluate_seed(
            seed, host=host, train=train, validation=validation, spec=spec,
            classes=classes, checkpoint_root=resolve(args.checkpoint_root),
            task=args.task, age=int(args.age),
            batch_size=int(args.eval_batch_size), device=device)
        per_seed.append(entry)
        line = " ".join(
            f"{CONDITION_LABELS[name]}={entry['records'][name]['balanced_accuracy']:.3f}"
            for name in CONDITION_ORDER
        )
        print(f"[lewm-host-controls] seed={seed} {line}", flush=True)

    host_after = state_digest(host)
    if host_before != host_after:
        raise RuntimeError("frozen official PushT LeWM host changed during controls")

    conditions = aggregate(per_seed)
    chance = 1.0 / float(classes)
    summary = {
        "schema": "lewm_pusht_host_controls_v1",
        "status": "completed",
        "claim_boundary": (
            "Reviewer-requested control battery for the frozen official PushT "
            "LeWorldModel Host-Aligned Evidence Writer. Adapters are reused from "
            "the checkpointed primary run; no adapter is retrained. Labels are "
            "used only for the post-hoc readout."),
        "reused_adapters_from": str(resolve(args.checkpoint_root).relative_to(ROOT)),
        "counterfactual_cache": cache_records,
        "task": args.task,
        "semantic_name": task_record["display_name"],
        "classes": classes,
        "age": int(args.age),
        "seeds": [int(s) for s in args.seeds],
        "chance_level": chance,
        "gate": {"full_minimum": 0.75, "control_maximum": chance + 0.05},
        "host_digest_unchanged": True,
        "host_digest": host_after,
        "condition_order": list(CONDITION_ORDER),
        "conditions": conditions,
        "per_seed": per_seed,
        "readout": "RidgeClassifier trained on train full host output, applied to each validation arm; memory-only readout trained/applied on injected residual.",
        "device": str(device),
        "elapsed_seconds": float(time.time() - started),
    }
    output = resolve(args.output)
    atomic_json(output / "controls.json", summary)
    print(json.dumps({
        "output": str((output / "controls.json").relative_to(ROOT)),
        "means": {name: conditions[name]["mean"] for name in CONDITION_ORDER},
        "chance": chance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
