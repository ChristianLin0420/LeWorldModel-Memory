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
# AMENDMENT 1 (2026-07-05, after the wave-1 oracle failed its calibration
# gate at 4-8%): the true-dynamics diagnostic (scripts/x2_feasibility_v21.py
# + outputs/v21_x2/feasibility.json) adjudicated the failure — flat per-step
# CEM cannot search a 46/78-dim action space (25-42% even with the real
# simulator), while KNOT-parameterized CEM reaches 100% at every tolerance
# rung with ~0.01 rad residuals.  Amended, before any arm comparison:
# (i) knot-CEM (K=6 linear interpolation) for every planner identically;
# (ii) plan earlier (t_p 40 -> 24, horizon 23 -> 39);
# (iii) receding-horizon MPC — replan every 8 steps with re-observation
# (frames re-rolled, overlays + the episode's registered corruption
# re-applied so execution-branch observations match the training regime).
# Wave-1 numbers are preserved in x2_results.json as the open-loop
# descriptive reference.
TASK_NAME = "t1"
PLAN_TIME = 24                    # cue over by t=22 (max cue_off 20 + 2)
EPISODE_END = 63
REPLAN_EVERY = 8                  # receding-horizon rounds: t = 24/32/40/48
KNOTS = 6
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
REGISTRATION = X2 / "x2_registration_v2.json"


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


def expand_knots(knots: torch.Tensor, horizon: int) -> torch.Tensor:
    """(P, K, 2) knot plans -> (P, horizon, 2) by linear interpolation."""
    positions = torch.linspace(0, KNOTS - 1, horizon, device=knots.device)
    low = positions.floor().long()
    high = (low + 1).clamp(max=KNOTS - 1)
    weight = (positions - low).view(1, -1, 1)
    return (knots[:, low] * (1 - weight)
            + knots[:, high] * weight).clamp(-1.0, 1.0)


def fit_pose_head(model, host: str, bank, calibration_idx: np.ndarray,
                  device) -> Any:
    """The registered planner cost head (V19 gate-6 build list): a post-hoc
    ridge probe z -> (cos, sin) of each joint, trained on calibration-half
    clean-timestep frames with simulator qpos targets.  No gradient touches
    the host.  (Amendment 2: raw latent L2 distance to goal embeddings is
    pose-blind on this host — Spearman(z-dist, angle-dist) ~ 0.0-0.12 — so
    the cost head the registration named is now actually built.)"""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    frames, targets = [], []
    for episode in calibration_idx:
        for t in (4, 22, 30, 38, 46, 60):
            if (bank.events["corrupt_on"][episode] <= t
                    < bank.events["corrupt_off"][episode]):
                continue
            frames.append(bank.frames[episode, t])
            targets.append(bank.endo_state[episode, t, :2])
    frames = np.stack(frames)
    qpos = np.stack(targets)
    tensor = p0_data_frames_to_tensor(frames).to(device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), 256):
            chunks.append(w1.p0.host_encode(
                w1.encode_host(host), model,
                tensor[start:start + 256].unsqueeze(0)).float()[0].cpu())
    z = torch.cat(chunks).numpy()
    y = np.stack([np.cos(qpos[:, 0]), np.sin(qpos[:, 0]),
                  np.cos(qpos[:, 1]), np.sin(qpos[:, 1])], axis=1)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(z, y)


def pose_cost(pose_head, finals: torch.Tensor, weights: np.ndarray
              ) -> torch.Tensor:
    """Belief-weighted angular cost of predicted final latents, (P,)."""
    predicted = pose_head.predict(finals.cpu().numpy())
    shoulder = np.arctan2(predicted[:, 1], predicted[:, 0])
    wrist = np.arctan2(predicted[:, 3], predicted[:, 2])
    costs = np.zeros(len(predicted))
    for goal, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        target = GOAL_ANGLES[goal]
        costs += weight * (np.abs(_wrap(shoulder - target[0]))
                           + 0.25 * np.abs(_wrap(wrist - target[1])))
    return torch.tensor(costs, dtype=torch.float32, device=finals.device)


@torch.no_grad()
def cem_plan(model, host: str, window_z: torch.Tensor, window_a: torch.Tensor,
             pose_head, weights: np.ndarray, horizon: int, device,
             rng: np.random.Generator) -> np.ndarray:
    """Knot-parameterized CEM against the pose cost head (amendments 1-2)."""
    mean = torch.zeros(KNOTS, 2, device=device)
    std = torch.full((KNOTS, 2), CEM_INIT_STD, device=device)
    for _ in range(CEM_ITERS):
        noise = torch.tensor(
            rng.standard_normal((CEM_POPULATION, KNOTS, 2)),
            dtype=torch.float32, device=device)
        knots = (mean + std * noise).clamp(-1.0, 1.0)
        plans = expand_knots(knots, horizon)
        finals = rollout_final_latents(model, host, window_z, window_a, plans)
        costs = pose_cost(pose_head, finals, weights)
        elite = knots[costs.argsort()[:CEM_ELITES]]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(0.05)
    return expand_knots(mean.unsqueeze(0), horizon)[0].cpu().numpy(
        ).astype(np.float32)


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


def _reobserve(frames: np.ndarray, bank) -> np.ndarray:
    """Re-apply the observation process to re-rolled BASE frames: the task
    overlays (marker rings + the — already past — cue window) and the
    episode's registered corruption, so execution-branch observations match
    the training regime.  Returns the observed uint8 frames."""
    task = make_task(TASK_NAME)
    frames = frames.copy()
    script = {"xi": bank.xi, "cue_on": bank.events["cue_on"],
              "cue_off": bank.events["cue_off"]}
    task._render(frames, script)
    for episode in range(frames.shape[0]):
        frames[episode], _, _, _ = p0_data.corrupt_episode(
            frames[episode], episode)
    return frames


@torch.no_grad()
def _encode_context(model, host: str, observed: np.ndarray, t: int,
                    device) -> torch.Tensor:
    """(E, 3, D) embeddings of the observed frames [t-2 .. t]."""
    segment = observed[:, t - 2:t + 1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(segment).permute(0, 1, 4, 2, 3).to(device)
    chunks = []
    for start in range(0, tensor.shape[0], 32):
        chunks.append(w1.p0.host_encode(
            w1.encode_host(host), model,
            tensor[start:start + 32]).float())
    return torch.cat(chunks)


def plan_and_execute(model, host: str, z_bank: torch.Tensor,
                     bank, pose_head, weights_fn, episodes,
                     base_seed: int, tolerance: float, device,
                     label: str) -> dict[str, Any]:
    """Receding-horizon MPC (amendment 1): plan at t = 24/32/40/48/56,
    execute REPLAN_EVERY actions, re-roll for re-observation, repeat;
    success from the final re-roll's simulator state."""
    rng = np.random.default_rng(21_100 + abs(hash(label)) % 100_000)
    override = bank.actions.copy()
    observed = bank.frames                     # round-0 context: bank frames
    started = time.time()
    endo = None
    rounds = list(range(PLAN_TIME, EPISODE_END, REPLAN_EVERY))
    for round_index, t_plan in enumerate(rounds):
        horizon = EPISODE_END - t_plan
        contexts = _encode_context(model, host, observed, t_plan, device)
        for episode in episodes:
            window_a = torch.from_numpy(
                override[episode, t_plan - 3:t_plan]).to(device)
            plan = cem_plan(model, host, contexts[episode], window_a,
                            pose_head, weights_fn(episode), horizon, device,
                            rng)
            stop = (t_plan + REPLAN_EVERY if round_index < len(rounds) - 1
                    else EPISODE_END)
            override[episode, t_plan:stop] = plan[:stop - t_plan]
        frames, _, endo = collect_base(
            bank.num_episodes, bank.length, base_seed, bank.stream,
            action_override=override)
        if round_index < len(rounds) - 1:
            observed = _reobserve(frames, bank)
    mask = success_mask(endo[episodes], bank.xi[episodes], tolerance)
    return {
        "label": label,
        "episodes": int(len(episodes)),
        "success_rate": float(mask.mean()),
        "successes": int(mask.sum()),
        "rounds": [int(t) for t in rounds],
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
        "schema_version": 2,
        "study": "v21-x2-consumer-registration-amendment1",
        "registered_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "amendment1": "knot-CEM (K=6) + t_p 24 + receding-horizon MPC "
                      "(replan every 8 with re-observation), adjudicated by "
                      "the true-dynamics diagnostic "
                      "(outputs/v21_x2/feasibility.json: flat CEM 25-42%, "
                      "knot CEM 100%); wave-1 open-loop numbers preserved "
                      "as descriptive reference",
        "task": TASK_NAME,
        "goals": GOAL_ANGLES,
        "plan_time": PLAN_TIME,
        "replan_every": REPLAN_EVERY,
        "knots": KNOTS,
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
        # Shared per-arm assets: bank embeddings, the registered pose cost
        # head (amendment 2), prior_read features.
        assets: dict[str, dict[str, Any]] = {}
        for arm, (model, carrier, checkpoint) in arms.items():
            host = checkpoint["host"]
            z_bank = torch.from_numpy(p1b.encode_bank(
                w1.encode_host(host), model, bank, device)).to(device)
            pose_head = fit_pose_head(model, host, bank, calibration_idx,
                                      device)
            prior = w2.plain_prior_read(
                carrier, z_bank,
                torch.from_numpy(bank.actions.astype(np.float32)).to(device))
            assets[arm] = {"model": model, "carrier": carrier, "host": host,
                           "z": z_bank, "pose_head": pose_head,
                           "prior": prior}
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
                    bank, candidate["pose_head"],
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
                asset["pose_head"],
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
    (output / "x2_results_v2.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[v21-x2] wrote {output / 'x2_results_v2.json'}")


def p0_data_frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(K, H, W, 3) uint8 -> (K, 3, H, W) float in [0, 1]."""
    return torch.from_numpy(
        frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)


if __name__ == "__main__":
    main()
