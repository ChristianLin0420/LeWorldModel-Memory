#!/usr/bin/env python3
"""V21 X2 feasibility diagnostic (docs/V21_PROPOSAL.md 4/X2; run after the
first X2 wave's oracle arm failed its calibration gate at 4-8% success).

Question: is T1-act goal reaching infeasible (horizon/actuation/tolerance),
or is the world-model planning stack the bottleneck (23-step open-loop
latent rollouts compounding error)?  Answer by replacing the latent rollout
with TRUE DYNAMICS: CEM whose cost is the simulator's final qpos distance to
the goal, physics-only stepping (no rendering), on a subset of episodes.

  dyn-CEM success >= 0.7  => task feasible; the latent planner is the
                             bottleneck => X2 amendment 1: receding-horizon
                             MPC (replan with re-observation) and/or plan
                             earlier (longer horizon)
  dyn-CEM success  < 0.3  => the task construction is at fault => amend
                             t_p / tolerance / goals before any arm claim

Writes outputs/v21_x2/feasibility.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from lewm.tasks_v19 import make_task
import scripts.make_v19_p0_data as p0_data
from scripts.x2_planning_v21 import (EPISODE_END, GOAL_ANGLES, PLAN_TIME,
                                     SUCCESS_LAST_FRAMES, TOLERANCE_LADDER,
                                     _wrap)

HORIZON = EPISODE_END - PLAN_TIME

EPISODES_TO_TEST = 12
CEM_POP = 64
CEM_ITERS = 5
CEM_STD = 0.6
LENGTH = 64


def build_env(seed: int):
    from dm_control import suite
    env = suite.load("reacher", "easy", task_kwargs={"random": seed % 2**32})
    env.physics.named.model.geom_rgba["target", 3] = 0.0
    return env


def state_at_plan_time(env, episode: int, actions: np.ndarray,
                       reset_counter: list[int]) -> np.ndarray:
    """Advance the env's reset stream to ``episode`` and replay its action
    prefix; returns the flattened physics state at t_p."""
    while reset_counter[0] <= episode:
        env.reset()
        reset_counter[0] += 1
    # reset_counter[0] == episode + 1: env is at episode's t=0 ... but any
    # extra reset would desync; callers must proceed in episode order.
    for t in range(PLAN_TIME):
        env.step(actions[episode, t])
    return env.physics.get_state().copy()


def rollout_final_qpos(env, state: np.ndarray, plans: np.ndarray
                       ) -> np.ndarray:
    """(P, HORIZON, 2) plans -> (P, 2) mean qpos over the last frames."""
    finals = np.empty((plans.shape[0], 2), dtype=np.float64)
    for index, plan in enumerate(plans):
        with env.physics.reset_context():
            env.physics.set_state(state)
        tail = []
        for h in range(plan.shape[0]):
            env.step(plan[h])
            if h >= plan.shape[0] - SUCCESS_LAST_FRAMES:
                tail.append(env.physics.data.qpos[:2].copy())
        finals[index] = np.mean(tail, axis=0)
    return finals


def dyn_cem(env, state: np.ndarray, target: np.ndarray,
            rng: np.random.Generator) -> tuple[np.ndarray, float]:
    mean = np.zeros((HORIZON, 2))
    std = np.full((HORIZON, 2), CEM_STD)
    best_cost = np.inf
    best_plan = mean.copy()
    for _ in range(CEM_ITERS):
        plans = np.clip(mean + std * rng.standard_normal(
            (CEM_POP, HORIZON, 2)), -1.0, 1.0)
        finals = rollout_final_qpos(env, state, plans)
        costs = (np.abs(_wrap(finals[:, 0] - target[0]))
                 + 0.5 * np.abs(_wrap(finals[:, 1] - target[1])))
        order = np.argsort(costs)
        if costs[order[0]] < best_cost:
            best_cost = float(costs[order[0]])
            best_plan = plans[order[0]].copy()
        elite = plans[order[:8]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0).clip(min=0.05)
    return best_plan, best_cost


def main() -> None:
    task = make_task("t1")
    base_seed, _, _ = task._rngs(p0_data.VAL_SEED)
    train_eps, val_eps = p0_data.episode_sizes()
    paths = p0_data.task_bank_paths(Path("outputs/v19_p0_a2/data"), "t1",
                                    train_eps, val_eps)
    from lewm.tasks_v19.base import load_bank
    bank = load_bank(paths["val"]["observed"])
    rng = np.random.default_rng(21_200)
    env = build_env(base_seed)
    reset_counter = [0]
    results = []
    episodes = list(range(120, 120 + EPISODES_TO_TEST))
    for episode in episodes:
        state = state_at_plan_time(env, episode, bank.actions, reset_counter)
        xi = int(bank.xi[episode])
        target = np.asarray(GOAL_ANGLES[xi])
        plan, cost = dyn_cem(env, state, target, rng)
        finals = rollout_final_qpos(env, state, plan[None])[0]
        d_shoulder = float(np.abs(_wrap(finals[0] - target[0])))
        d_wrist = float(np.abs(_wrap(finals[1] - target[1])))
        row = {"episode": episode, "xi": xi,
               "final_qpos": [round(float(q), 3) for q in finals],
               "target": list(target),
               "d_shoulder": round(d_shoulder, 3),
               "d_wrist": round(d_wrist, 3),
               "success_at": {str(tol): bool(d_shoulder < tol
                                             and d_wrist < 2 * tol)
                              for tol in TOLERANCE_LADDER}}
        results.append(row)
        print(f"[x2-feas] e{episode} xi={xi} d_sh={d_shoulder:.3f} "
              f"d_wr={d_wrist:.3f}", flush=True)
    summary = {
        "schema_version": 1,
        "study": "v21-x2-feasibility-diagnostic",
        "plan_time": PLAN_TIME,
        "horizon": HORIZON,
        "cem": {"pop": CEM_POP, "iters": CEM_ITERS},
        "episodes": results,
        "success_rate_at": {
            str(tol): float(np.mean([row["success_at"][str(tol)]
                                     for row in results]))
            for tol in TOLERANCE_LADDER},
    }
    out = ROOT / "outputs" / "v21_x2" / "feasibility.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["success_rate_at"], indent=2))


if __name__ == "__main__":
    main()
