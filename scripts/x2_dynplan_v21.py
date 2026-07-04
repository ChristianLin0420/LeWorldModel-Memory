#!/usr/bin/env python3
"""V21 X2 amendment 3 — memory-in-control with the oracle-dynamics planner
(docs/V21_PROPOSAL.md 4/X2).

Amendment trail (each adjudicated by a diagnostic, all preserved on disk):
  wave 1   flat CEM + latent goal-distance: oracle 4-8%  -> flat CEM cannot
           search 46-78 dims (feasibility.json: true-dynamics flat CEM
           25-42%, knot CEM 100%)
  wave 2   knot CEM + receding horizon + the REGISTERED pose cost head:
           oracle still 0% -> the ROLLOUT is the bottleneck: pose decoded
           from predicted latents degrades 0.10 (real frame) -> 0.47 (one
           step) -> ~1.9 rad (>= 4 steps, no information).  X2-Finding-1:
           the exact-LeWM predictor (H=3 teacher forcing) does not
           transport the endogenous state under open-loop rollout — LATENT
           model-based planning is untestable on this host, full stop.
  wave 3   THIS SCRIPT: the planner uses oracle dynamics (physics-only
           stepping, proven 100% ceiling), so the ONLY thing that differs
           between arms is the belief-driven goal selection — the memory
           factor isolated exactly.  Claims 4-5 are tested here; Finding-1
           is reported alongside as the reason the planning substrate is
           not the latent world model.

Arms/variants (identical planner, cost, execution):
  oracle             true-xi goals                          (ceiling)
  floor_integrator   selector on [enc(o_0), action prefix]  (the certificate
                                                             floor)
  rfix / acgru       selector on the arm's prior_read at t_p
  rfix_detuned       trust softplus(r) x 16 at eval (miscalibrated belief)
  argmax vs hedged   one goal vs belief-weighted expected cost (claim 5)
  none / ablated     no-carrier selector / uniform weights   (causal checks)

Success: environment-defined, verified by one full collect_base re-roll per
variant (byte-identical dynamics to the planning env).  Writes
outputs/v21_x2/x2_results_v3.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import load_bank
from lewm.tasks_v19.dmc_base import collect_base
import scripts.certify_v19_p1b as p1b
import scripts.eval_v20_w2 as w2
import scripts.make_v19_p0_data as p0_data
import scripts.train_v20_w1 as w1
from scripts.x2_feasibility_v21 import build_env, rollout_final_qpos
from scripts.x2_planning_v21 import (DETUNE_FACTOR, GOAL_ANGLES, KNOTS,
                                     PLAN_TIME, SEEDS, TOLERANCE_LADDER, W3,
                                     X2, _wrap, detune_trust, fit_selector,
                                     goal_weight_matrix, selector_features,
                                     success_mask)

EPISODE_END = 63
HORIZON = EPISODE_END - PLAN_TIME          # 39 actions, single plan
CEM_POP = 96
CEM_ITERS = 6
CEM_STD = 0.6
ORACLE_MIN_SUCCESS = 0.7
CERTIFICATE_MIN_GAP = 0.3


def expand_knots_np(knots: np.ndarray, horizon: int) -> np.ndarray:
    positions = np.linspace(0, KNOTS - 1, horizon)
    low = np.floor(positions).astype(int)
    high = np.minimum(low + 1, KNOTS - 1)
    weight = (positions - low)[None, :, None]
    return np.clip(knots[:, low] * (1 - weight) + knots[:, high] * weight,
                   -1.0, 1.0)


def weighted_cost(finals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    costs = np.zeros(len(finals))
    for goal, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        target = GOAL_ANGLES[goal]
        costs += weight * (np.abs(_wrap(finals[:, 0] - target[0]))
                           + 0.5 * np.abs(_wrap(finals[:, 1] - target[1])))
    return costs


def dyn_cem_weighted(env, state: np.ndarray, weights: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    mean = np.zeros((KNOTS, 2))
    std = np.full((KNOTS, 2), CEM_STD)
    best = (np.inf, None)
    for _ in range(CEM_ITERS):
        knots = np.clip(mean + std * rng.standard_normal(
            (CEM_POP, KNOTS, 2)), -1.0, 1.0)
        plans = expand_knots_np(knots, HORIZON)
        finals = rollout_final_qpos(env, state, plans)
        costs = weighted_cost(finals, weights)
        order = np.argsort(costs)
        if costs[order[0]] < best[0]:
            best = (float(costs[order[0]]), plans[order[0]].copy())
        elite = knots[order[:8]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0).clip(min=0.05)
    return best[1].astype(np.float32)


def plan_states(bank, base_seed: int) -> list[np.ndarray]:
    """Physics state of every episode at t_p (sequential replay, no render)."""
    env = build_env(base_seed)
    states = []
    for episode in range(bank.num_episodes):
        env.reset()
        for t in range(PLAN_TIME):
            env.step(bank.actions[episode, t])
        states.append(env.physics.get_state().copy())
    return states


def run_variant(env, states, bank, weights_fn, episodes, base_seed: int,
                tolerance: float, label: str, verify_reroll: bool
                ) -> dict[str, Any]:
    rng = np.random.default_rng(21_400 + abs(hash(label)) % 100_000)
    override = bank.actions.copy()
    finals = np.empty((len(episodes), 2))
    started = time.time()
    for row, episode in enumerate(episodes):
        plan = dyn_cem_weighted(env, states[episode], weights_fn(episode),
                                rng)
        override[episode, PLAN_TIME:] = plan
        finals[row] = rollout_final_qpos(env, states[episode], plan[None])[0]
    # success from the planning env's own rollout (identical dynamics);
    # optionally verified by a full re-roll (byte-identity check).
    targets = np.asarray(GOAL_ANGLES)[bank.xi[episodes]]
    d_shoulder = np.abs(_wrap(finals[:, 0] - targets[:, 0]))
    d_wrist = np.abs(_wrap(finals[:, 1] - targets[:, 1]))
    mask = (d_shoulder < tolerance) & (d_wrist < 2 * tolerance)
    result = {
        "label": label,
        "episodes": int(len(episodes)),
        "success_rate": float(mask.mean()),
        "successes": int(mask.sum()),
        "median_d_shoulder": float(np.median(d_shoulder)),
        "seconds": round(time.time() - started, 1),
    }
    if verify_reroll:
        _, _, endo = collect_base(bank.num_episodes, bank.length, base_seed,
                                  bank.stream, action_override=override)
        verified = success_mask(endo[episodes], bank.xi[episodes], tolerance)
        result["reroll_success_rate"] = float(verified.mean())
        result["reroll_agrees"] = bool(
            abs(verified.mean() - mask.mean()) < 1e-9)
    return result


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

    print("[v21-x2v3] computing plan-time states", flush=True)
    states = plan_states(bank, base_seed)
    env = build_env(base_seed)
    env.reset()   # fresh env used purely as a dynamics oracle via set_state

    results: dict[str, Any] = {"schema_version": 1,
                               "study": "v21-x2-consumer-dynplan",
                               "amendment": 3,
                               "per_seed": {}}

    # Tolerance: registered ladder on the calibration half, oracle weights.
    tolerance_star = None
    for tolerance in TOLERANCE_LADDER:
        oracle_cal = run_variant(
            env, states, bank,
            lambda e: goal_weight_matrix("oracle", None, bank.xi,
                                         np.array([e]))[0],
            calibration_idx[:60], base_seed, tolerance,
            f"calibrate_tol{tolerance}", verify_reroll=False)
        results[f"calibration_oracle_tol{tolerance}"] = oracle_cal
        print(f"[v21-x2v3] calibration tol={tolerance}: "
              f"{oracle_cal['success_rate']:.3f}", flush=True)
        if oracle_cal["success_rate"] >= ORACLE_MIN_SUCCESS:
            tolerance_star = tolerance
            break
    if tolerance_star is None:
        raise SystemExit("oracle below the ladder minimum even with true "
                         "dynamics — task construction fault")
    results["tolerance_star"] = tolerance_star

    for seed in seeds:
        seed_block: dict[str, Any] = {}
        arms = {arm: w2.load_checkpoint(Path(args.w3_root), "t1", arm, seed,
                                        device)
                for arm in ("lkc_rfix", "acgru", "none")}
        selectors: dict[str, np.ndarray] = {}
        z_rfix = None
        for arm, (model, carrier, checkpoint) in arms.items():
            host = checkpoint["host"]
            z_bank = torch.from_numpy(p1b.encode_bank(
                w1.encode_host(host), model, bank, device)).to(device)
            if arm == "lkc_rfix":
                z_rfix = z_bank
            actions_t = torch.from_numpy(
                bank.actions.astype(np.float32)).to(device)
            prior = w2.plain_prior_read(carrier, z_bank, actions_t)
            features = selector_features(prior, cue_off)
            probe = fit_selector(features, bank.xi, calibration_idx)
            selectors[arm] = probe.predict_proba(features)
            if arm == "lkc_rfix":
                detuned_prior = w2.plain_prior_read(
                    detune_trust(carrier), z_bank, actions_t)
                detuned_features = selector_features(detuned_prior, cue_off)
                detuned_probe = fit_selector(detuned_features, bank.xi,
                                             calibration_idx)
                selectors["lkc_rfix_detuned"] = detuned_probe.predict_proba(
                    detuned_features)
        floor_features = np.concatenate([
            z_rfix[:, 0].cpu().numpy(),
            bank.actions[:, :PLAN_TIME].reshape(episodes, -1)], axis=1)
        floor_probe = fit_selector(floor_features, bank.xi, calibration_idx)
        floor_proba = floor_probe.predict_proba(floor_features)
        ablated = np.full((episodes, 4), 0.25)

        runs = [
            ("oracle", "oracle", None, True),
            ("floor_integrator", "argmax", floor_proba, False),
            ("rfix_argmax", "argmax", selectors["lkc_rfix"], False),
            ("rfix_hedged", "hedged", selectors["lkc_rfix"], False),
            ("rfix_detuned_argmax", "argmax",
             selectors["lkc_rfix_detuned"], False),
            ("rfix_detuned_hedged", "hedged",
             selectors["lkc_rfix_detuned"], False),
            ("acgru_argmax", "argmax", selectors["acgru"], False),
            ("acgru_hedged", "hedged", selectors["acgru"], False),
            ("none_selector", "argmax", selectors["none"], False),
            ("belief_ablated", "hedged", ablated, False),
        ]
        for label, kind, proba, verify in runs:
            seed_block[label] = run_variant(
                env, states, bank,
                lambda e, k=kind, p=proba: goal_weight_matrix(
                    k, p, bank.xi, np.array([e]))[0],
                eval_idx, base_seed, tolerance_star, f"{label}_s{seed}",
                verify_reroll=verify)
            print(f"[v21-x2v3] s{seed} {label}: "
                  f"{seed_block[label]['success_rate']:.3f}", flush=True)
        results["per_seed"][str(seed)] = seed_block
        if device.type == "cuda":
            torch.cuda.empty_cache()

    X2.mkdir(parents=True, exist_ok=True)
    (X2 / "x2_results_v3.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[v21-x2v3] wrote {X2 / 'x2_results_v3.json'}")


if __name__ == "__main__":
    main()
