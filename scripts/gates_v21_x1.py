#!/usr/bin/env python3
"""V21 X1 — the preregistered confirmation gate (docs/V21_PROPOSAL.md 4/X1).

FROZEN BEFORE any X0b dev result is unblinded (registration.json is written
by --register with a timestamp; the gate refuses to run unless the
registration predates every X1 training artifact).

Registered endpoint (claim 2):

    d_pool = mean over tasks {t1, t3, t4} of the per-task standardized
             paired effect  d_task = mean(diff)/sd(diff),
             diff_s = score(lkc_rfix, s) - score(envelope*, s),
             fresh seeds s in {10..19}

    scores: t1/t3 -> the registered categorical probe
            (scripts/eval_v19_p2.py, probe_results.json 'registered');
            t4  -> the REPAIRED continuous family
            (scripts/x0_t4_probes_v21.py: StandardScaler + RidgeCV(1e-3..1e3),
            standardized targets, 3 probe seeds) — frozen by X0a's
            readout-fragility adjudication before this registration.

    envelope*: the single best baseline configuration by pooled dev mean
    (t1dev + t3dev, registered probe) from the X0b sweep, recipe frozen at
    selection; lkc_rfix keeps its registered V19 recipe (lr 3e-4) and gets
    no sweep.

    CONFIRMED if the seed-bootstrap p_pos(d_pool) < 0.05 (one-sided, 20k
    draws, seed 21021) AND d_task > 0 on >= 2/3 tasks.
    FALSIFIED otherwise — including if fair tuning closed the gap, which is
    reported as V20's baseline-effort asymmetry (a publishable correction).

Writes outputs/v21_x1/x1_gates.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.x0_t4_probes_v21 import (pooled_d, probe_cell,
                                      standardized_effect)

X1 = ROOT / "outputs" / "v21_x1"
TASKS = ("t1", "t3", "t4")
FRESH_SEEDS = tuple(range(10, 20))
ALPHA = 0.05
MIN_POSITIVE_TASKS = 2
REGISTRATION = X1 / "registration.json"


def register(envelope_note: str = "pending X0b selection") -> None:
    if REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION} already exists — refusing to "
                         f"re-register")
    X1.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "study": "v21-x1-confirmation-registration",
        "registered_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "endpoint": "pooled standardized paired d, lkc_rfix - envelope*, "
                    "tasks t1/t3/t4, fresh seeds 10-19",
        "t4_probe_family": "repaired: StandardScaler+RidgeCV(1e-3..1e3), "
                           "standardized targets (X0a adjudication: "
                           "readout_fragility)",
        "confirmation_rule": f"bootstrap p_pos < {ALPHA} AND d_task > 0 on "
                             f">= {MIN_POSITIVE_TASKS}/3 tasks",
        "bootstrap": {"draws": 20_000, "seed": 21_021},
        "envelope_selection_rule": "max pooled dev mean (t1dev+t3dev, "
                                   "registered probe) over the X0b sweep; "
                                   "recipe frozen at selection",
        "lkc_rfix_recipe": "V19 registered (lr 3e-4, matched widths), "
                           "no sweep",
        "envelope_star": envelope_note,
        "falsified_clause": "fair tuning closes the gap -> V20 result was "
                            "baseline-effort asymmetry; report and stop",
    }
    REGISTRATION.write_text(json.dumps(payload, indent=2, sort_keys=True)
                            + "\n")
    print(f"[v21-x1] registered {REGISTRATION}")


def _stationary_score(root: Path, task: str, arm: str, seed: int) -> float:
    return float(json.loads(
        (root / task / arm / f"s{seed}" / "probe_results.json").read_text()
    )["registered"]["mean"])


def _t4_repaired_score(root: Path, arm: str, seed: int) -> float:
    import scripts.x0_t4_probes_v21 as x0t4
    original = x0t4.W3
    try:
        x0t4.W3 = root
        return probe_cell(arm, seed)["repaired"]
    finally:
        x0t4.W3 = original


def evaluate(root: Path, envelope: str) -> dict[str, Any]:
    registration = json.loads(REGISTRATION.read_text())
    registered_at = datetime.datetime.fromisoformat(
        registration["registered_utc"])
    for task in TASKS:
        for arm in ("lkc_rfix", envelope):
            for seed in FRESH_SEEDS:
                artifact = root / task / arm / f"s{seed}" / "gates.json"
                if artifact.exists():
                    created = datetime.datetime.fromtimestamp(
                        artifact.stat().st_mtime, datetime.timezone.utc)
                    if created < registered_at:
                        raise SystemExit(
                            f"{artifact} predates the registration — the "
                            f"gate is void")
    per_task: dict[str, np.ndarray] = {}
    for task in ("t1", "t3"):
        per_task[task] = np.array(
            [_stationary_score(root, task, "lkc_rfix", seed)
             - _stationary_score(root, task, envelope, seed)
             for seed in FRESH_SEEDS])
    per_task["t4"] = np.array(
        [_t4_repaired_score(root, "lkc_rfix", seed)
         - _t4_repaired_score(root, envelope, seed)
         for seed in FRESH_SEEDS])
    pooled = pooled_d(per_task)
    positive_tasks = sum(d > 0 for d in pooled["per_task_d"].values())
    confirmed = bool(pooled["p_pos"] < ALPHA
                     and positive_tasks >= MIN_POSITIVE_TASKS)
    return {
        "schema_version": 1,
        "study": "v21-x1-confirmation-gate",
        "registration": registration,
        "envelope_star": envelope,
        "seeds": list(FRESH_SEEDS),
        "per_task_mean_diff": {task: float(diff.mean())
                               for task, diff in per_task.items()},
        "per_task_wins": {task: f"{int((diff > 0).sum())}/{len(diff)}"
                          for task, diff in per_task.items()},
        **pooled,
        "positive_tasks": positive_tasks,
        "alpha": ALPHA,
        "CONFIRMED": confirmed,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true",
                        help="write the frozen registration (once)")
    parser.add_argument("--root", default="outputs/v21_x1")
    parser.add_argument("--envelope", default=None,
                        help="the X0b-selected envelope* arm label")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.register:
        register()
        return
    if args.envelope is None:
        raise SystemExit("--envelope required for gate evaluation")
    gates = evaluate(Path(args.root), args.envelope)
    (X1 / "x1_gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True) + "\n")
    lines = ["# V21 X1 — preregistered confirmation gate", "",
             f"envelope\\* = `{gates['envelope_star']}` · fresh seeds "
             f"{gates['seeds'][0]}–{gates['seeds'][-1]}", "",
             f"| task | mean diff | wins | d |", "|---|---|---|---|"]
    for task in TASKS:
        lines.append(f"| {task} | {gates['per_task_mean_diff'][task]:+.4f} |"
                     f" {gates['per_task_wins'][task]} | "
                     f"{gates['per_task_d'][task]:+.3f} |")
    lines.append("")
    lines.append(f"**pooled d = {gates['pooled_d']:+.3f} "
                 f"CI95 [{gates['ci95'][0]:+.3f}, {gates['ci95'][1]:+.3f}], "
                 f"p_pos = {gates['p_pos']:.2e} → "
                 f"{'CONFIRMED' if gates['CONFIRMED'] else 'FALSIFIED'}**")
    (X1 / "x1_gates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
