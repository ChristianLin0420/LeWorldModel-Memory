#!/usr/bin/env python3
"""V21 X2 wave-3 addendum — the envelope* selector arm (claim 4 vs gdelta).

Wave 3 ran before the X0b sweep selected envelope* = gdelta_l10, so claim 4
("rfix > envelope in certified return") was executed against acgru only.
This script runs the missing arm: gdelta_l10 selectors from the X1 t1
checkpoints (fresh seeds) through the byte-identical oracle-dynamics
planner at the tolerance_star recorded by the registered wave-3 run.  The
wave-3 script and its artifact are left untouched; results land in
outputs/v21_x2/x2_results_envelope.json.
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

from lewm.models.v21_carriers import make_carrier_v21
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import load_bank
import scripts.certify_v19_p1b as p1b
import scripts.eval_v20_w2 as w2
import scripts.make_v19_p0_data as p0_data
import scripts.train_v20_w1 as w1
from scripts.x2_dynplan_v21 import plan_states, run_variant
from scripts.x2_feasibility_v21 import build_env
from scripts.x2_planning_v21 import (X2, fit_selector, goal_weight_matrix,
                                     selector_features)

w2.make_carrier = make_carrier_v21


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="gdelta_l10")
    parser.add_argument("--root", default="outputs/v21_x1")
    parser.add_argument("--seeds", default="10,11,12")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    seeds = [int(value) for value in args.seeds.split(",")]

    v3 = json.loads((X2 / "x2_results_v3.json").read_text())
    tolerance_star = v3["tolerance_star"]

    task = make_task("t1")
    base_seed, _, _ = task._rngs(p0_data.VAL_SEED)
    train_eps, val_eps = p0_data.episode_sizes()
    paths = p0_data.task_bank_paths(
        Path("outputs/v19_p0_a2/data"), "t1", train_eps, val_eps)
    bank = load_bank(paths["val"]["observed"])
    episodes = bank.num_episodes
    half = episodes // 2
    calibration_idx = np.arange(half)
    eval_idx = np.arange(half, episodes)
    cue_off = bank.events["cue_off"]

    print("[v21-x2-env] computing plan-time states", flush=True)
    states = plan_states(bank, base_seed)
    env = build_env(base_seed)
    env.reset()

    results: dict[str, Any] = {"schema_version": 1,
                               "study": "v21-x2-consumer-envelope-arm",
                               "arm": args.arm,
                               "checkpoint_root": args.root,
                               "tolerance_star": tolerance_star,
                               "per_seed": {}}
    actions_t = torch.from_numpy(bank.actions.astype(np.float32)).to(device)
    for seed in seeds:
        model, carrier, checkpoint = w2.load_checkpoint(
            Path(args.root), "t1", args.arm, seed, device)
        z_bank = torch.from_numpy(p1b.encode_bank(
            w1.encode_host(checkpoint["host"]), model, bank, device)
            ).to(device)
        prior = w2.plain_prior_read(carrier, z_bank, actions_t)
        features = selector_features(prior, cue_off)
        probe = fit_selector(features, bank.xi, calibration_idx)
        proba = probe.predict_proba(features)
        selector_acc = float(
            (proba[eval_idx].argmax(axis=1) == bank.xi[eval_idx]).mean())

        seed_block: dict[str, Any] = {"selector_accuracy": selector_acc}
        for label, kind in ((f"{args.arm}_argmax", "argmax"),
                            (f"{args.arm}_hedged", "hedged")):
            seed_block[label] = run_variant(
                env, states, bank,
                lambda e, k=kind, p=proba: goal_weight_matrix(
                    k, p, bank.xi, np.array([e]))[0],
                eval_idx, base_seed, tolerance_star, f"{label}_s{seed}",
                verify_reroll=False)
            print(f"[v21-x2-env] s{seed} {label}: "
                  f"{seed_block[label]['success_rate']:.3f} "
                  f"(selector acc {selector_acc:.3f})", flush=True)
        results["per_seed"][str(seed)] = seed_block
        del model, carrier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for kind in ("argmax", "hedged"):
        rates = [results["per_seed"][str(s)][f"{args.arm}_{kind}"]
                 ["success_rate"] for s in seeds]
        results[f"mean_{kind}"] = float(np.mean(rates))
    (X2 / "x2_results_envelope.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[v21-x2-env] mean argmax {results['mean_argmax']:.3f} | "
          f"hedged {results['mean_hedged']:.3f}")
    print(f"[v21-x2-env] wrote {X2 / 'x2_results_envelope.json'}")


if __name__ == "__main__":
    main()
