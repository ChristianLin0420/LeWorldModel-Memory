#!/usr/bin/env python3
"""V20 W2 drift protocol: deployment adaptation where it can matter at all
(docs/V20_PROPOSAL.md 4.4/4.5, claims 3 and 5).

V19 never tested adaptivity: every evaluation stream was drawn from the
training corruption regime — stationary by construction.  W2 evaluates the
FROZEN W1 checkpoints (no training anywhere) on registered drifted streams:

  stationary    fresh banks, training corruption recipe end to end (control)
  drift_gap     from the shift episode onward the corruption window length
                doubles: gap ~ U[14, 20] instead of U[6, 12] (same rng
                namespace, parameterized window draw)
  drift_noise   from the shift episode onward every frame gains additive
                Gaussian pixel noise (sigma = 12/255) — the cleanest
                "observation-noise regime changed" manipulation

Streams: 480 fresh episodes per (task, regime) from the W2 bank-seed
namespace, shift at episode 240.  Fixed-trust is provably miscalibrated on
the post-shift half; the question is whether the slow filter converts that
into probe points without giving anything back before the shift.

Arms per (task, seed): lkc_rfix (rho = 0, the subsumption limit), dfc (rho*
from the W1 summary), dfc_etafix (eta* from the W1 summary), acgru, none.
Probes: the registered coordinate scored separately on the pre-shift and
post-shift halves (split-within-half, 3 probe seeds).  Everything lands in
<output>/<task>/<regime>/<arm>/s<seed>/{eval_export.npz,probe_results.json}.
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

from lewm.models.v19_carriers import make_carrier
from lewm.models.v20_dfc import SlowFilterConfig, dfc_stream_eval
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import EpisodeBatch, load_bank, save_bank
import scripts.certify_v19_p1b as p1b
import scripts.eval_v19_p2 as p2eval
import scripts.make_v19_p0_data as p0_data
import scripts.train_v19_p0 as p0
import scripts.train_v20_w1 as w1

REGIMES = ("stationary", "drift_gap", "drift_noise")
TASKS = ("t1dev", "t3dev")
SEEDS = (0, 1, 2)
EPISODES = 480
SHIFT_EPISODE = 240
W2_BANK_SEED = 270_902          # fresh W2 clean-bank namespace
DRIFT_GAP_RANGE = (14, 20)      # post-shift corruption window (train: 6-12)
NOISE_SIGMA = 12.0              # post-shift additive pixel noise (uint8 scale)
_NOISE_SALT = 909
PROBE_SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------
# Drift banks
# --------------------------------------------------------------------------

def _drift_window(length: int, episode: int) -> tuple[int, int]:
    """The training window draw with the drifted gap range, same rng
    namespace (mirrors p0_data.corruption_window with GAP_RANGE swapped)."""
    rng = p0_data._episode_rng(p0_data.CORRUPTION_SEED, episode,
                               p0_data._INTERVAL_SALT)
    gap = int(rng.integers(DRIFT_GAP_RANGE[0], DRIFT_GAP_RANGE[1] + 1))
    earliest = p0_data.HISTORY_LEN + 2
    latest = length - gap - 2
    start = int(rng.integers(earliest, max(latest, earliest) + 1))
    return start, min(start + gap, length - 1)


def _corrupt_drift_gap(clean: np.ndarray, episode: int
                       ) -> tuple[np.ndarray, int, int, str]:
    """Post-shift drift_gap corruption: the training modes with a doubled
    window (meanframe/cutout alternation and spatial rng kept verbatim)."""
    observed = clean.copy()
    start, end = _drift_window(clean.shape[0], episode)
    mode = p0_data.corruption_mode(episode)
    if mode == "meanframe":
        mean_frame = np.rint(clean.mean(axis=0)).clip(0, 255).astype(np.uint8)
        observed[start:end] = mean_frame
    else:
        rng = p0_data._episode_rng(p0_data.CORRUPTION_SEED, episode,
                                   p0_data._SPATIAL_SALT)
        height, width = clean.shape[1:3]
        cut_h = max(1, int(round(height * p0_data.CUTOUT_FRACTION)))
        cut_w = max(1, int(round(width * p0_data.CUTOUT_FRACTION)))
        top = int(rng.integers(0, height - cut_h + 1))
        left = int(rng.integers(0, width - cut_w + 1))
        fill = np.rint(clean.mean(axis=(0, 1, 2))).clip(0, 255).astype(np.uint8)
        observed[start:end, top:top + cut_h, left:left + cut_w] = fill
    return observed, start, end, mode


def _add_noise(frames: np.ndarray, episode: int) -> np.ndarray:
    rng = p0_data._episode_rng(p0_data.CORRUPTION_SEED, episode, _NOISE_SALT)
    noise = rng.normal(0.0, NOISE_SIGMA, size=frames.shape)
    return np.clip(frames.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def build_drift_bank(task_name: str, regime: str) -> EpisodeBatch:
    """480 fresh episodes; the corruption regime shifts at SHIFT_EPISODE."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    clean = make_task(task_name).generate(p0_data.STREAM, EPISODES,
                                          W2_BANK_SEED)
    frames = clean.frames.copy()
    corrupt_on = np.empty(EPISODES, dtype=np.int64)
    corrupt_off = np.empty(EPISODES, dtype=np.int64)
    corrupt_mode = np.empty(EPISODES, dtype=np.int64)
    for episode in range(EPISODES):
        drifted = regime != "stationary" and episode >= SHIFT_EPISODE
        if drifted and regime == "drift_gap":
            observed, start, end, mode = _corrupt_drift_gap(
                clean.frames[episode], episode)
        else:
            observed, start, end, mode = p0_data.corrupt_episode(
                clean.frames[episode], episode)
        if drifted and regime == "drift_noise":
            observed = _add_noise(observed, episode)
        frames[episode] = observed
        corrupt_on[episode] = start
        corrupt_off[episode] = end
        corrupt_mode[episode] = p0_data.MODES.index(mode)
    events = dict(clean.events)
    events.update({"corrupt_on": corrupt_on, "corrupt_off": corrupt_off,
                   "corrupt_mode": corrupt_mode,
                   "drifted": (np.arange(EPISODES) >= SHIFT_EPISODE
                               if regime != "stationary"
                               else np.zeros(EPISODES, dtype=bool)).astype(
                                   np.int64)})
    return EpisodeBatch(
        frames=frames, actions=clean.actions, xi=clean.xi,
        xi_kind=clean.xi_kind, n_classes=clean.n_classes,
        endo_state=clean.endo_state, exo_state=clean.exo_state,
        events=events, stream=clean.stream, task=clean.task,
        seed=W2_BANK_SEED)


def resolve_drift_bank(task_name: str, regime: str, data_root: Path
                       ) -> EpisodeBatch:
    path = data_root / f"{task_name}_{regime}_e{EPISODES}.npz"
    if p0_data._cache_valid(path):
        return load_bank(path)
    started = time.time()
    print(f"[v20-w2-data] building {task_name}/{regime} "
          f"({EPISODES} episodes)", flush=True)
    bank = build_drift_bank(task_name, regime)
    save_bank(bank, path)
    print(f"[v20-w2-data] wrote {path} ({time.time() - started:.1f}s)",
          flush=True)
    return bank


# --------------------------------------------------------------------------
# Arm evaluation on a drift bank
# --------------------------------------------------------------------------

def load_checkpoint(root: Path, task: str, arm: str, seed: int,
                    device: torch.device):
    path = root / task / arm / f"s{seed}" / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    host = checkpoint["host"]
    action_dim = int(checkpoint["action_dim"])
    model = w1.build_host(host, action_dim)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    embed_dim = int(p0.HOST_CONFIGS[
        "vicreg" if host == "vicreg" else "sigreg"]["embed_dim"])
    carrier = make_carrier(arm, embed_dim, action_dim)
    carrier.load_state_dict(checkpoint["carrier_state_dict"], strict=True)
    model.to(device).eval()
    carrier.to(device).eval()
    for module in (model, carrier):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, carrier, checkpoint


@torch.no_grad()
def plain_prior_read(carrier, z: torch.Tensor, actions: torch.Tensor,
                     chunk: int = 32) -> np.ndarray:
    outputs = []
    for start in range(0, z.shape[0], chunk):
        output = carrier(z[start:start + chunk], actions[start:start + chunk])
        outputs.append(output.prior_read.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def export_arm(out_path: Path, prior_read: np.ndarray, z0: np.ndarray,
               bank: EpisodeBatch, meta_extra: dict[str, Any],
               telemetry: dict[str, np.ndarray] | None = None) -> None:
    arrays: dict[str, np.ndarray] = {
        "prior_read": prior_read,
        "enc_o0": z0,
        "actions": bank.actions.astype(np.float32),
        "xi": bank.xi,
    }
    for name, value in bank.events.items():
        arrays[f"event_{name}"] = value
    for key, value in (telemetry or {}).items():
        arrays[key] = value
    meta = {
        "schema_version": 2,
        "study": "v20-w2-drift-protocol",
        "episodes": bank.num_episodes,
        "length": bank.length,
        "shift_episode": SHIFT_EPISODE,
        "xi_kind": bank.xi_kind,
        "n_classes": bank.n_classes,
        "stream": bank.stream,
        **meta_extra,
    }
    w1.p2.write_eval_export(out_path, arrays, meta)


# --------------------------------------------------------------------------
# Pre/post-shift probes
# --------------------------------------------------------------------------

def _probe_half(export: dict[str, Any], episodes: np.ndarray,
                probe_seed: int) -> float:
    """Registered categorical probe restricted to an episode subset."""
    prior_read = export["prior_read"][episodes]
    xi = export["xi"][episodes]
    events = {name: values[episodes]
              for name, values in p2eval.export_events(export).items()}
    features = p2eval.registered_cat_features(
        prior_read, p2eval.deep_window_start(events))
    order = np.random.default_rng(
        p2eval.PROBE_SPLIT_SALT + probe_seed).permutation(len(episodes))
    half = len(episodes) // 2
    train_idx, eval_idx = order[:half], order[half:]
    from lewm.tasks_v19.certify import _cat_accuracy
    return float(_cat_accuracy(features[train_idx], xi[train_idx],
                               features[eval_idx], xi[eval_idx]))


def probe_halves(export_path: Path) -> dict[str, Any]:
    export = p2eval.load_export(export_path)
    episodes = export["prior_read"].shape[0]
    pre = np.arange(0, SHIFT_EPISODE)
    post = np.arange(SHIFT_EPISODE, episodes)
    results: dict[str, Any] = {
        "schema_version": 1,
        "study": "v20-w2-drift-probes",
        **{key: export["meta"][key]
           for key in ("task", "arm", "seed", "regime", "n_classes")},
        "chance": 1.0 / export["meta"]["n_classes"],
        "shift_episode": SHIFT_EPISODE,
    }
    for label, subset in (("pre_shift", pre), ("post_shift", post)):
        scores = [_probe_half(export, subset, probe_seed)
                  for probe_seed in PROBE_SEEDS]
        results[label] = {"mean": float(np.mean(scores)),
                          "std": float(np.std(scores)),
                          "per_probe_seed": [round(s, 4) for s in scores]}
    results["drift_cost"] = (results["pre_shift"]["mean"]
                             - results["post_shift"]["mean"])
    # Deployment-adaptation telemetry (DFC arms only).
    if "tel_eta_mean" in export:
        eta = export["tel_eta_mean"][:, 1:]
        drift_trace = export["tel_phi_drift"][:, -1]
        results["telemetry"] = {
            "eta_pre_mean": float(eta[pre].mean()),
            "eta_post_mean": float(eta[post].mean()),
            "phi_drift_at_shift": float(drift_trace[SHIFT_EPISODE - 1]),
            "phi_drift_final": float(drift_trace[-1]),
            "post_shift_phi_velocity": float(
                (drift_trace[-1] - drift_trace[SHIFT_EPISODE - 1])
                / max(episodes - SHIFT_EPISODE, 1)),
            "pre_shift_phi_velocity": float(
                drift_trace[SHIFT_EPISODE - 1] / SHIFT_EPISODE),
        }
    # Calibration-certificate ratio (claim 3): mean squared innovation
    # z-score eps^2/S per half — 1.0 is perfectly calibrated trust.
    if "tel_calib_ratio" in export:
        ratio = export["tel_calib_ratio"][:, 1:]
        results["calibration"] = {
            "pre_shift_ratio": float(ratio[pre].mean()),
            "post_shift_ratio": float(ratio[post].mean()),
        }
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/v20_w2")
    parser.add_argument("--w1-root", default="outputs/v20_w1")
    parser.add_argument("--w1-summary", default="outputs/v20_w1/w1_summary.json")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    output = Path(args.output)
    w1_root = Path(args.w1_root)
    data_root = output / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    seeds = [int(value) for value in args.seeds.split(",")]

    summary = json.loads(Path(args.w1_summary).read_text())
    rho_star_variant = summary["rho_star"]           # e.g. "dfc_rho4"
    eta_star_variant = summary["eta_star"]           # e.g. "dfc_eta2"
    if rho_star_variant is None:
        raise SystemExit("W1 summary has no rho*; run W1 first")
    from scripts.eval_v20_w1 import DFC_VARIANTS
    configs = {
        "dfc": DFC_VARIANTS[rho_star_variant],
        "dfc_etafix": DFC_VARIANTS[eta_star_variant],
        "lkc_rfix": SlowFilterConfig(rho=0.0),       # exact subsumption limit
    }
    print(f"[v20-w2] dfc={rho_star_variant} etafix={eta_star_variant}",
          flush=True)

    for task in tasks:
        banks = {regime: resolve_drift_bank(task, regime, data_root)
                 for regime in REGIMES}
        if banks["stationary"].xi_kind != "cat":
            # The drift protocol's pre/post probe coordinate is categorical
            # by registration; continuous tasks keep their standard
            # stationary endpoint (docs/V20_PROPOSAL.md 4.4).
            print(f"[v20-w2] {task}: xi_kind != cat — drift protocol "
                  f"skipped by registration", flush=True)
            continue
        for seed in seeds:
            rfix = load_checkpoint(w1_root, task, "lkc_rfix", seed, device)
            others = {arm: load_checkpoint(w1_root, task, arm, seed, device)
                      for arm in ("acgru", "none")}
            for regime, bank in banks.items():
                actions = torch.from_numpy(
                    bank.actions.astype(np.float32)).to(device)
                # One encode per (regime, seed): all rfix-family arms share
                # the encoder; acgru/none have their own encoders.
                model, carrier, checkpoint = rfix
                z = torch.from_numpy(p1b.encode_bank(
                    w1.encode_host(checkpoint["host"]), model, bank,
                    device)).to(device)
                z0 = z[:, 0].cpu().numpy().astype(np.float32)
                for arm, config in configs.items():
                    out_path = (output / task / regime / arm / f"s{seed}"
                                / "eval_export.npz")
                    if out_path.exists():
                        continue
                    result = dfc_stream_eval(carrier, z, actions, config)
                    telemetry = {f"tel_{key}": value for key, value
                                 in result.telemetry.items()}
                    telemetry.update({f"phi_{key}": value for key, value
                                      in result.phi_trace.items()})
                    export_arm(out_path, result.prior_read, z0, bank,
                               {"task": task, "arm": arm, "seed": seed,
                                "regime": regime,
                                "host": checkpoint["host"],
                                "dfc_config": result.config,
                                "dfc_variant": (rho_star_variant
                                                if arm == "dfc" else
                                                eta_star_variant
                                                if arm == "dfc_etafix"
                                                else "rho0")},
                               telemetry)
                    print(f"[v20-w2] {task}/{regime}/{arm}/s{seed} exported",
                          flush=True)
                del z
                for arm, (model_o, carrier_o, checkpoint_o) in others.items():
                    out_path = (output / task / regime / arm / f"s{seed}"
                                / "eval_export.npz")
                    if out_path.exists():
                        continue
                    z_arm = torch.from_numpy(p1b.encode_bank(
                        w1.encode_host(checkpoint_o["host"]), model_o, bank,
                        device)).to(device)
                    prior = plain_prior_read(carrier_o, z_arm, actions)
                    export_arm(out_path, prior,
                               z_arm[:, 0].cpu().numpy().astype(np.float32),
                               bank,
                               {"task": task, "arm": arm, "seed": seed,
                                "regime": regime,
                                "host": checkpoint_o["host"]})
                    del z_arm
                    print(f"[v20-w2] {task}/{regime}/{arm}/s{seed} exported",
                          flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del rfix, others
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Pre/post-shift probes over every export.
    exports = sorted(output.glob("*/*/*/s*/eval_export.npz"))
    print(f"[v20-w2] probing {len(exports)} exports", flush=True)
    for export_path in exports:
        results_path = export_path.parent / "probe_results.json"
        if results_path.exists():
            continue
        results = probe_halves(export_path)
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"[v20-w2] {export_path.parent}: "
              f"pre={results['pre_shift']['mean']:.3f} "
              f"post={results['post_shift']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
