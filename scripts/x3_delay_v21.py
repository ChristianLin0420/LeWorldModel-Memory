#!/usr/bin/env python3
"""V21 X3 — delay-scaling curves per carrier (docs/V21_PROPOSAL.md 4/X3).

The actual shape of a memory claim: how does the registered probe score
change as the cue-to-decision delay grows?  Frozen checkpoints are evaluated
(no training) on fresh t1-geometry banks with longer episodes — the task
instance is `dataclasses.replace(t1, length=L)` for L in {64, 96, 128}
(delays ~ L-1 - cue_off ~= 43-57 / 75-89 / 107-121 steps), corruption and
overlays per the registered recipe, a fresh X3 bank-seed namespace.

For each (arm, checkpoint seed, L): encode the bank with the arm's own
encoder, run the carrier, score the registered categorical probe (deep-gap
window mean ++ t_dec read, 3 probe seeds).  Writes
outputs/v21_x3/delay_scaling.{json,md}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import EpisodeBatch
import scripts.certify_v19_p1b as p1b
import scripts.eval_v19_p2 as p2eval
import scripts.eval_v20_w2 as w2
import scripts.make_v19_p0_data as p0_data
import scripts.train_v20_w1 as w1
from lewm.models.v21_carriers import make_carrier_v21

# X1 envelope checkpoints carry v21 sweep arms (e.g. gdelta_l10); the v21
# factory falls through to the v19 registry for the W3 arms.
w2.make_carrier = make_carrier_v21

X3 = ROOT / "outputs" / "v21_x3"
LENGTHS = (64, 96, 128)
X3_BANK_SEED = 270_903
EPISODES = 240


def build_delay_bank(length: int) -> EpisodeBatch:
    task = make_task("t1")
    # ``length`` is a plain class attribute on the (non-dataclass) V19Task
    # base, so dataclasses.replace cannot set it; shadow it on the frozen
    # instance instead (generate() reads self.length).
    object.__setattr__(task, "length", length)
    clean = task.generate(p0_data.STREAM, EPISODES, X3_BANK_SEED)
    return p0_data.corrupt_bank(clean, p0_data.CORRUPTION_SEED)


def probe_bank(model, carrier, host: str, bank: EpisodeBatch, device
               ) -> float:
    z = torch.from_numpy(p1b.encode_bank(
        w1.encode_host(host), model, bank, device)).to(device)
    actions = torch.from_numpy(bank.actions.astype(np.float32)).to(device)
    prior = w2.plain_prior_read(carrier, z, actions)
    export = {
        "prior_read": prior,
        "xi": bank.xi,
        **{f"event_{name}": value for name, value in bank.events.items()},
    }
    scores = []
    for probe_seed in p2eval.PROBE_SEEDS:
        features = p2eval.registered_cat_features(
            prior, p2eval.deep_window_start(p2eval.export_events(export)))
        order = np.random.default_rng(
            p2eval.PROBE_SPLIT_SALT + probe_seed).permutation(EPISODES)
        half = EPISODES // 2
        train_idx, eval_idx = order[:half], order[half:]
        from lewm.tasks_v19.certify import _cat_accuracy
        scores.append(_cat_accuracy(features[train_idx], bank.xi[train_idx],
                                    features[eval_idx], bank.xi[eval_idx]))
    return float(np.mean(scores))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="lkc_rfix:outputs/v20_w3,"
                                          "acgru:outputs/v20_w3",
                        help="comma-separated arm:checkpoint_root pairs")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    arm_roots = [pair.split(":") for pair in args.arms.split(",") if pair]
    seeds = [int(value) for value in args.seeds.split(",")]

    existing = {}
    path = X3 / "delay_scaling.json"
    if path.exists():
        existing = json.loads(path.read_text()).get("curves", {})

    banks = {}
    for length in LENGTHS:
        print(f"[v21-x3-delay] building L={length} bank", flush=True)
        banks[length] = build_delay_bank(length)

    curves: dict[str, Any] = dict(existing)
    for arm, root in arm_roots:
        for length in LENGTHS:
            key = f"{arm}@L{length}"
            if key in curves:
                continue
            scores = []
            for seed in seeds:
                model, carrier, checkpoint = w2.load_checkpoint(
                    Path(root), "t1", arm, seed, device)
                scores.append(probe_bank(model, carrier,
                                         checkpoint["host"], banks[length],
                                         device))
                del model, carrier
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            curves[key] = {"mean": float(np.mean(scores)),
                           "sd": float(np.std(scores)),
                           "scores": [round(s, 4) for s in scores]}
            print(f"[v21-x3-delay] {key}: {curves[key]['mean']:.3f} "
                  f"± {curves[key]['sd']:.3f}", flush=True)

    X3.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "study": "v21-x3-delay-scaling",
        "lengths": list(LENGTHS),
        "bank_seed": X3_BANK_SEED,
        "episodes": EPISODES,
        "chance": 0.25,
        "curves": curves,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    arms = sorted({key.split("@")[0] for key in curves})
    lines = ["# V21 X3 — delay scaling (registered probe vs episode length)",
             "", "| arm | " + " | ".join(f"L={l}" for l in LENGTHS) + " |",
             "|---|" + "---|" * len(LENGTHS)]
    for arm in arms:
        row = [f"{curves[f'{arm}@L{l}']['mean']:.3f}"
               if f"{arm}@L{l}" in curves else "—" for l in LENGTHS]
        lines.append(f"| {arm} | " + " | ".join(row) + " |")
    (X3 / "delay_scaling.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
