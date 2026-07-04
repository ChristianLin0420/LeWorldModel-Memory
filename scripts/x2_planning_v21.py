#!/usr/bin/env python3
"""V21 X2 — the consumer: belief-conditioned MPC on a certified goal task
(docs/V21_PROPOSAL.md 4/X2; the V19 Tier-2 gate-6 registration, executed).

T1-act: each T1 episode's vanished cue xi in {0..3} indexes a REGISTERED
reacher goal configuration q*(xi).  At plan time t_p (after the cue is long
gone) a frozen post-hoc selector probe maps the carrier's belief
(prior_read features up to t_p) to goal weights; a CEM planner rolls the
frozen world model open-loop in latent space toward the weighted goal
embeddings; the plan is EXECUTED in the simulator (collect_base
action_override — the counterfactual machinery) and success is
environment-defined: final qpos within tolerance of q*(xi_true).

The certificate and the claims (registered in x2_registration.json before
any eval-half number is computed):

  return-floor certificate   oracle-selector success - integrator-selector
                             success >= 0.3 on the calibration half
                             (tolerance picked there by the registered
                             ladder rule: smallest of {0.25, 0.35, 0.5} rad
                             with oracle success >= 0.7)
  claim 4 (transfer)         success(lkc_rfix) > success(envelope arm) —
                             does the probe-space inversion survive the
                             consumer?
  claim 5 (uncertainty/      hedged (expected-cost-under-belief) planning
  calibration value)         vs argmax planning, under calibrated vs
                             detuned trust (softplus(r) x 16 at eval):
                             the hedging gain must GROW under detuning —
                             calibrated uncertainty finally priced in
                             return
  causal check               zeroed-belief ablation collapses toward floor

Every arm shares the identical planner, cost, and execution; only the
selector's input differs.  All numbers land in outputs/v21_x2/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v19_carriers import LatentKalmanCell
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import load_bank
from lewm.tasks_v19.dmc_base import CAMERA_ID, DOMAIN, TASK, collect_base
import scripts.certify_v19_p1b as p1b
import scripts.eval_v20_w2 as w2
import scripts.make_v19_p0_data as p0_data
import scripts.train_v20_w1 as w1

X2 = ROOT / "outputs" / "v21_x2"
W3 = ROOT / "outputs" / "v20_w3"

# ---------------------------------------------------------------- registered
TASK_NAME = "t1"
PLAN_TIME = 40                    # cue over by t=20 (max onset 14 + dur 6)
HORIZON = 23                      # actions t_p..62 -> final frame 63
GOAL_ANGLES = ((2.356, 0.0), (0.785, 0.0), (-2.356, 0.0), (-0.785, 0.0))
TOLERANCE_LADDER = (0.25, 0.35, 0.5)     # rad; wrist tolerance = 2x
ORACLE_MIN_SUCCESS = 0.7
CERTIFICATE_MIN_GAP = 0.3
SUCCESS_LAST_FRAMES = 3           # stability: mean qpos over last 3 frames
CEM_POPULATION = 128
CEM_ELITES = 16
CEM_ITERS = 3
CEM_INIT_STD = 0.6
DETUNE_FACTOR = 16.0
SEEDS = (0, 1, 2)                 # W3 checkpoint seeds
SELECTOR_C = 1.0
REGISTRATION = X2 / "x2_registration.json"


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------

def render_goal_frames() -> np.ndarray:
    """(4, 64, 64, 3) uint8 renders of the registered goal configurations."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite
    env = suite.load(DOMAIN, TASK, task_kwargs={"random": 0})
    env.physics.named.model.geom_rgba["target", 3] = 0.0
    frames = []
    for shoulder, wrist in GOAL_ANGLES:
        env.reset()
        with env.physics.reset_context():
            env.physics.data.qpos[:] = (shoulder, wrist)
            env.physics.data.qvel[:] = 0.0
        frames.append(env.physics.render(64, 64, camera_id=CAMERA_ID))
    return np.stack(frames).astype(np.uint8)


def success_mask(endo: np.ndarray, xi: np.ndarray, tolerance: float
                 ) -> np.ndarray:
    """Environment-defined success: mean qpos over the last frames within
    tolerance (shoulder) / 2x tolerance (wrist) of q*(xi_true)."""
    qpos = endo[:, -SUCCESS_LAST_FRAMES:, :2].mean(axis=1)      # (E, 2)
    targets = np.asarray(GOAL_ANGLES, dtype=np.float64)[xi]
    d_shoulder = np.abs(_wrap(qpos[:, 0] - targets[:, 0]))
    d_wrist = np.abs(_wrap(qpos[:, 1] - targets[:, 1]))
    return (d_shoulder < tolerance) & (d_wrist < 2 * tolerance)


# --------------------------------------------------------------------------
# Selector features (registered: prior_read up to t_p only)
# --------------------------------------------------------------------------

def selector_features(prior_read: np.ndarray, cue_off: np.ndarray
                      ) -> np.ndarray:
    """(E, 2D): mean prior_read over [cue_off+2 .. t_p] ++ prior_read[t_p]."""
    episodes, length, _ = prior_read.shape
    steps = np.arange(length)[None, :]
    start = np.asarray(cue_off, dtype=np.int64) + 2
    mask = (steps >= start[:, None]) & (steps <= PLAN_TIME)
    weights = mask.astype(np.float64) / mask.sum(axis=1, keepdims=True)
    window_mean = np.einsum("el,eld->ed", weights,
                            prior_read.astype(np.float64))
    return np.concatenate([window_mean,
                           prior_read[:, PLAN_TIME].astype(np.float64)],
                          axis=1)


def fit_selector(features: np.ndarray, xi: np.ndarray, train_idx: np.ndarray
                 ) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=SELECTOR_C, random_state=0))
    probe.fit(features[train_idx], xi[train_idx])
    return probe


# --------------------------------------------------------------------------
# CEM planning in latent space (identical for every arm)
# --------------------------------------------------------------------------

@torch.no_grad()
def rollout_final_latents(model, host: str, window_z: torch.Tensor,
                          window_a: torch.Tensor, plans: torch.Tensor
                          ) -> torch.Tensor:
    """Open-loop latent rollout of ``plans`` (P, HORIZON, A) from a shared
    3-frame context; returns the mean of the last SUCCESS_LAST_FRAMES
    predicted latents, (P, D)."""
    predictor = w1.p2.host_predictor(
        "sigreg" if host != "vicreg" else "vicreg", model)
    population = plans.shape[0]
    latents = window_z.unsqueeze(0).expand(population, -1, -1).clone()
    actions = window_a.unsqueeze(0).expand(population, -1, -1).clone()
    tail = []
    for h in range(plans.shape[1]):
        actions = torch.cat([actions[:, 1:], plans[:, h:h + 1]], dim=1)
        prediction = predictor(latents, actions)[:, -1]
        latents = torch.cat([latents[:, 1:], prediction.unsqueeze(1)], dim=1)
        if h >= plans.shape[1] - SUCCESS_LAST_FRAMES:
            tail.append(prediction)
    return torch.stack(tail, dim=1).mean(dim=1)


@torch.no_grad()
def cem_plan(model, host: str, window_z: torch.Tensor, window_a: torch.Tensor,
             goal_z: torch.Tensor, weights: np.ndarray, device,
             rng: np.random.Generator) -> np.ndarray:
    """CEM over action sequences toward the weighted goal embeddings."""
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    mean = torch.zeros(HORIZON, 2, device=device)
    std = torch.full((HORIZON, 2), CEM_INIT_STD, device=device)
    for _ in range(CEM_ITERS):
        noise = torch.tensor(
            rng.standard_normal((CEM_POPULATION, HORIZON, 2)),
            dtype=torch.float32, device=device)
        plans = (mean + std * noise).clamp(-1.0, 1.0)
        finals = rollout_final_latents(model, host, window_z, window_a, plans)
        distances = torch.cdist(finals, goal_z)                 # (P, 4)
        costs = (distances.square() * weight_t).sum(dim=-1)
        elite = plans[costs.argsort()[:CEM_ELITES]]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(0.05)
    return mean.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# Arm evaluation
# --------------------------------------------------------------------------

def detune_trust(carrier: LatentKalmanCell) -> LatentKalmanCell:
    """Deployment-time miscalibration: softplus(r) x DETUNE_FACTOR."""
    import copy
    import torch.nn.functional as F
    detuned = copy.deepcopy(carrier)
    with torch.no_grad():
        r = F.softplus(detuned.r_const) * DETUNE_FACTOR
        detuned.r_const.copy_(torch.log(torch.expm1(r)))
    return detuned


def goal_weight_matrix(kind: str, probabilities: np.ndarray | None,
                       xi: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """(len(episodes), 4) goal weights per planner variant."""
    if kind == "oracle":
        weights = np.zeros((len(episodes), 4))
        weights[np.arange(len(episodes)), xi[episodes]] = 1.0
        return weights
    if kind == "argmax":
        weights = np.zeros((len(episodes), 4))
        weights[np.arange(len(episodes)),
                probabilities[episodes].argmax(axis=1)] = 1.0
        return weights
    if kind == "hedged":
        return probabilities[episodes]
    raise ValueError(kind)


def plan_and_execute(model, host: str, z_bank: torch.Tensor,
                     bank, goal_z: torch.Tensor, weights_fn, episodes,
                     base_seed: int, tolerance: float, device,
                     label: str) -> dict[str, Any]:
    """Plan every episode in ``episodes``, execute one batched re-roll,
    return the environment-defined success summary."""
    rng = np.random.default_rng(21_100 + abs(hash(label)) % 100_000)
    override = bank.actions.copy()
    started = time.time()
    for episode in episodes:
        window_z = z_bank[episode, PLAN_TIME - 2:PLAN_TIME + 1]
        window_a = torch.from_numpy(
            bank.actions[episode, PLAN_TIME - 3:PLAN_TIME]).to(device)
        weights = weights_fn(episode)
        override[episode, PLAN_TIME:] = cem_plan(
            model, host, window_z, window_a, goal_z, weights, device, rng)
    _, _, endo = collect_base(bank.num_episodes, bank.length, base_seed,
                              bank.stream, action_override=override)
    mask = success_mask(endo[episodes], bank.xi[episodes], tolerance)
    return {
        "label": label,
        "episodes": int(len(episodes)),
        "success_rate": float(mask.mean()),
        "successes": int(mask.sum()),
        "seconds": round(time.time() - started, 1),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def register() -> None:
    if REGISTRATION.exists():
        return
    X2.mkdir(parents=True, exist_ok=True)
    import datetime
    REGISTRATION.write_text(json.dumps({
        "schema_version": 1,
        "study": "v21-x2-consumer-registration",
        "registered_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "task": TASK_NAME,
        "goals": GOAL_ANGLES,
        "plan_time": PLAN_TIME,
        "horizon": HORIZON,
        "tolerance_ladder": TOLERANCE_LADDER,
        "tolerance_rule": f"smallest with oracle success >= "
                          f"{ORACLE_MIN_SUCCESS} on the calibration half",
        "certificate": f"oracle - integrator_floor >= {CERTIFICATE_MIN_GAP}",
        "cem": {"population": CEM_POPULATION, "elites": CEM_ELITES,
                "iters": CEM_ITERS, "init_std": CEM_INIT_STD},
        "detune_factor": DETUNE_FACTOR,
        "claims": {
            "claim4_transfer": "success(lkc_rfix, argmax) > success(acgru, "
                               "argmax), paired over checkpoint seeds",
            "claim5_calibration": "hedging gain (hedged - argmax) grows "
                                  "under detuned trust vs calibrated",
            "causal": "zeroed-belief ablation collapses toward the floor",
        },
        "seeds": list(SEEDS),
    }, indent=2, sort_keys=True) + "\n")
    print(f"[v21-x2] registered {REGISTRATION}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w3-root", default=str(W3))
    parser.add_argument("--output", default=str(X2))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    register()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    output = Path(args.output)
    w3_root = Path(args.w3_root)
    seeds = [int(value) for value in args.seeds.split(",")]

    task = make_task(TASK_NAME)
    base_seed, _, _ = task._rngs(p0_data.VAL_SEED)
    train_eps, val_eps = p0_data.episode_sizes()
    paths = p0_data.task_bank_paths(
        Path("outputs/v19_p0_a2/data"), TASK_NAME, train_eps, val_eps)
    bank = load_bank(paths["val"]["observed"])
    episodes = bank.num_episodes
    half = episodes // 2
    calibration_idx = np.arange(half)
    eval_idx = np.arange(half, episodes)
    cue_off = bank.events["cue_off"]
    goal_frames = render_goal_frames()

    results: dict[str, Any] = {"schema_version": 1,
                               "study": "v21-x2-consumer",
                               "per_seed": {}}
    tolerance_star: float | None = None

    for seed in seeds:
        seed_block: dict[str, Any] = {}
        arms: dict[str, tuple] = {}
        for arm in ("lkc_rfix", "acgru", "none"):
            arms[arm] = w2.load_checkpoint(w3_root, TASK_NAME, arm, seed,
                                           device)
        # Shared per-arm assets: bank embeddings, goal embeddings,
        # prior_read features.
        assets: dict[str, dict[str, Any]] = {}
        for arm, (model, carrier, checkpoint) in arms.items():
            host = checkpoint["host"]
            z_bank = torch.from_numpy(p1b.encode_bank(
                w1.encode_host(host), model, bank, device)).to(device)
            goal_tensor = p0_data_frames_to_tensor(goal_frames).to(device)
            with torch.no_grad():
                goal_z = w1.p0.host_encode(
                    w1.encode_host(host), model,
                    goal_tensor.unsqueeze(0)).float()[0]
            prior = w2.plain_prior_read(
                carrier, z_bank,
                torch.from_numpy(bank.actions.astype(np.float32)).to(device))
            assets[arm] = {"model": model, "carrier": carrier, "host": host,
                           "z": z_bank, "goal_z": goal_z, "prior": prior}
        # Detuned-trust prior for the candidate.
        detuned = detune_trust(arms["lkc_rfix"][1])
        assets["lkc_rfix_detuned"] = {
            **assets["lkc_rfix"],
            "prior": w2.plain_prior_read(
                detuned, assets["lkc_rfix"]["z"],
                torch.from_numpy(bank.actions.astype(np.float32)).to(device)),
        }

        # Tolerance calibration (registered ladder; oracle on the
        # calibration half with the candidate's world model — planner and
        # model identical across arms, so any arm's model works; the
        # candidate's is registered).
        candidate = assets["lkc_rfix"]
        nonlocal_tolerance = tolerance_star
        if nonlocal_tolerance is None:
            for tolerance in TOLERANCE_LADDER:
                oracle_cal = plan_and_execute(
                    candidate["model"], candidate["host"], candidate["z"],
                    bank, candidate["goal_z"],
                    lambda e: goal_weight_matrix(
                        "oracle", None, bank.xi, np.array([e]))[0],
                    calibration_idx, base_seed, tolerance, device,
                    f"calibrate_tol{tolerance}_s{seed}")
                seed_block[f"calibration_oracle_tol{tolerance}"] = oracle_cal
                if oracle_cal["success_rate"] >= ORACLE_MIN_SUCCESS:
                    nonlocal_tolerance = tolerance
                    break
            if nonlocal_tolerance is None:
                nonlocal_tolerance = TOLERANCE_LADDER[-1]
                seed_block["tolerance_note"] = ("oracle never reached the "
                                                "ladder minimum; largest "
                                                "rung used, certificate "
                                                "will decide")
            tolerance_star = nonlocal_tolerance
            seed_block["tolerance_star"] = tolerance_star
        tolerance = tolerance_star

        # Selectors (trained on the calibration half).
        selectors: dict[str, np.ndarray] = {}
        for name, asset in assets.items():
            features = selector_features(asset["prior"], cue_off)
            probe = fit_selector(features, bank.xi, calibration_idx)
            selectors[name] = probe.predict_proba(features)
        floor_features = np.concatenate([
            assets["lkc_rfix"]["z"][:, 0].cpu().numpy(),
            bank.actions[:, :PLAN_TIME].reshape(episodes, -1)], axis=1)
        floor_probe = fit_selector(floor_features, bank.xi, calibration_idx)
        floor_proba = floor_probe.predict_proba(floor_features)
        ablated = np.full((episodes, 4), 0.25)

        runs = [
            ("oracle", "lkc_rfix", "oracle", None),
            ("floor_integrator", "lkc_rfix", "argmax", floor_proba),
            ("rfix_argmax", "lkc_rfix", "argmax", selectors["lkc_rfix"]),
            ("rfix_hedged", "lkc_rfix", "hedged", selectors["lkc_rfix"]),
            ("rfix_detuned_argmax", "lkc_rfix", "argmax",
             selectors["lkc_rfix_detuned"]),
            ("rfix_detuned_hedged", "lkc_rfix", "hedged",
             selectors["lkc_rfix_detuned"]),
            ("acgru_argmax", "acgru", "argmax", selectors["acgru"]),
            ("acgru_hedged", "acgru", "hedged", selectors["acgru"]),
            ("none_selector", "none", "argmax", selectors["none"]),
            ("rfix_ablated", "lkc_rfix", "hedged", ablated),
        ]
        for label, arm, kind, proba in runs:
            asset = assets[arm]
            seed_block[label] = plan_and_execute(
                asset["model"], asset["host"], asset["z"], bank,
                asset["goal_z"],
                lambda e, k=kind, p=proba: goal_weight_matrix(
                    k, p, bank.xi, np.array([e]))[0],
                eval_idx, base_seed, tolerance, device, f"{label}_s{seed}")
            print(f"[v21-x2] s{seed} {label}: "
                  f"{seed_block[label]['success_rate']:.3f}", flush=True)
        results["per_seed"][str(seed)] = seed_block
        for asset in assets.values():
            asset.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["tolerance_star"] = tolerance_star
    output.mkdir(parents=True, exist_ok=True)
    (output / "x2_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[v21-x2] wrote {output / 'x2_results.json'}")


def p0_data_frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(K, H, W, 3) uint8 -> (K, 3, H, W) float in [0, 1]."""
    return torch.from_numpy(
        frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)


if __name__ == "__main__":
    main()
