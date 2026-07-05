#!/usr/bin/env python3
"""V21 X2 provenance repair — persist the claim-5 mediation statistics.

The §10 mediation claim ("selector accuracy falls 0.825 -> 0.536 under
detune") was computed interactively during wave 3 and never written to an
artifact.  This script recomputes every wave-3 selector's eval-half
accuracy from the frozen W3 checkpoints with the byte-identical pipeline
(same bank, calibration split, features, probe fit) and persists them to
outputs/v21_x2/x2_selector_stats.json.  No planner involved — this is the
mediation layer only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import load_bank
import scripts.certify_v19_p1b as p1b
import scripts.eval_v20_w2 as w2
import scripts.make_v19_p0_data as p0_data
import scripts.train_v20_w1 as w1
from scripts.x2_planning_v21 import (SEEDS, W3, X2, detune_trust,
                                     fit_selector, selector_features)

PLAN_TIME = 24


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w3-root", default=str(W3))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    seeds = [int(value) for value in args.seeds.split(",")]

    task = make_task("t1")
    _ = task._rngs(p0_data.VAL_SEED)
    train_eps, val_eps = p0_data.episode_sizes()
    paths = p0_data.task_bank_paths(
        Path("outputs/v19_p0_a2/data"), "t1", train_eps, val_eps)
    bank = load_bank(paths["val"]["observed"])
    episodes = bank.num_episodes
    half = episodes // 2
    calibration_idx = np.arange(half)
    eval_idx = np.arange(half, episodes)
    cue_off = bank.events["cue_off"]
    actions_t = torch.from_numpy(bank.actions.astype(np.float32)).to(device)

    results: dict[str, Any] = {"schema_version": 1,
                               "study": "v21-x2-selector-mediation-stats",
                               "plan_time": PLAN_TIME,
                               "per_seed": {}}
    for seed in seeds:
        block: dict[str, float] = {}
        z_rfix = None
        for arm in ("lkc_rfix", "acgru", "none"):
            model, carrier, checkpoint = w2.load_checkpoint(
                Path(args.w3_root), "t1", arm, seed, device)
            z_bank = torch.from_numpy(p1b.encode_bank(
                w1.encode_host(checkpoint["host"]), model, bank, device)
                ).to(device)
            if arm == "lkc_rfix":
                z_rfix = z_bank
            variants = {arm: carrier}
            if arm == "lkc_rfix":
                variants["lkc_rfix_detuned"] = detune_trust(carrier)
            for label, cell in variants.items():
                prior = w2.plain_prior_read(cell, z_bank, actions_t)
                features = selector_features(prior, cue_off)
                probe = fit_selector(features, bank.xi, calibration_idx)
                proba = probe.predict_proba(features)
                block[label] = float(
                    (proba[eval_idx].argmax(axis=1)
                     == bank.xi[eval_idx]).mean())
            del model, carrier
            if device.type == "cuda":
                torch.cuda.empty_cache()
        floor_features = np.concatenate([
            z_rfix[:, 0].cpu().numpy(),
            bank.actions[:, :PLAN_TIME].reshape(episodes, -1)], axis=1)
        floor_probe = fit_selector(floor_features, bank.xi, calibration_idx)
        block["floor_integrator"] = float(
            (floor_probe.predict_proba(floor_features)[eval_idx].argmax(
                axis=1) == bank.xi[eval_idx]).mean())
        results["per_seed"][str(seed)] = block
        print(f"[v21-x2-sel] s{seed}: " + " ".join(
            f"{k}={v:.3f}" for k, v in block.items()), flush=True)

    arms = sorted(next(iter(results["per_seed"].values())))
    results["mean"] = {arm: float(np.mean(
        [results["per_seed"][str(s)][arm] for s in seeds])) for arm in arms}
    (X2 / "x2_selector_stats.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("[v21-x2-sel] means: " + " ".join(
        f"{k}={v:.3f}" for k, v in results["mean"].items()))
    print(f"[v21-x2-sel] wrote {X2 / 'x2_selector_stats.json'}")


if __name__ == "__main__":
    main()
