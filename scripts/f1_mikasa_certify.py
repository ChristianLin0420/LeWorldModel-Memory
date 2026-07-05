#!/usr/bin/env python3
"""V21 F1 stage 2 — certify RememberColor9 and run the inversion direction
(§12/F1 registration; runs in the program venv on the stage-1 NPZ banks).

Readouts (registered):
  (a) sighted certificate — registered P1b probe on cue-window DINOv2
      features, pass >= 0.75, 3 bank seeds, majority;
  (b) memory-demand — integrator floor (dino(o_0) + full action stream)
      within 0.05 of chance; no-leakage — decision-window frame probe
      within 0.05 of chance (chance = 1/9);
  (c) inversion direction — lkc_rfix vs gdelta_l10 carriers trained on
      frozen standardized DINOv2 features with the transfer recipe
      (next-feature prediction, the V19 residual-read objective's
      feature-space form), n = 5 seeds, registered probe family (delay-gap
      window mean ++ t_dec read), direction-only endpoint.

Writes outputs/v21_f1/certification.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v19_carriers import LatentKalmanCell
from lewm.models.v21_carriers import make_carrier_v21
import scripts.certify_v19_p1b as p1b
from scripts.x3_dino_sstar_v21 import DinoFeatures

F1 = ROOT / "outputs" / "v21_f1"
BANK_SEEDS = (0, 1, 2)
CHANCE = 1.0 / 9.0
GAP = (5, 10)          # all cubes hidden
T_DEC = 10             # decision onset
CARRIER_SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 40
BATCH = 64


def load_bank(path: Path) -> SimpleNamespace:
    data = np.load(path)
    frames = data["frames"]
    return SimpleNamespace(
        frames=frames, actions=data["actions"], xi=data["xi"],
        num_episodes=frames.shape[0], length=frames.shape[1],
        events={"cue_on": data["cue_on"], "cue_off": data["cue_off"]})


def frame_probe(emb_train, emb_eval, bank_train, bank_eval, window
                ) -> float:
    """Registered cat probe on 8 mean-pooled frames of ``window``."""
    lo, hi = window
    idx = np.linspace(lo, hi - 1, 8).round().astype(int)
    fit = p1b._probe_fn("cat")
    feat = lambda emb: emb[:, idx].mean(axis=1)
    return float(fit(feat(emb_train), bank_train.xi,
                     feat(emb_eval), bank_eval.xi))


def train_carrier(arm: str, z_train: torch.Tensor, a_train: torch.Tensor,
                  seed: int, device) -> torch.nn.Module:
    torch.manual_seed(seed)
    embed_dim, action_dim = z_train.shape[-1], a_train.shape[-1]
    if arm == "lkc_rfix":
        carrier, lr = LatentKalmanCell(embed_dim, action_dim,
                                       r_fixed=True), 3e-4
    else:
        carrier, lr = make_carrier_v21(arm, embed_dim, action_dim), 1e-3
    carrier.to(device).train()
    optim = torch.optim.Adam(carrier.parameters(), lr=lr)
    episodes = z_train.shape[0]
    for epoch in range(EPOCHS):
        order = torch.randperm(episodes)
        for start in range(0, episodes, BATCH):
            batch = order[start:start + BATCH]
            out = carrier(z_train[batch], a_train[batch])
            loss = torch.nn.functional.mse_loss(
                out.z_tilde[:, :-1], z_train[batch][:, 1:])
            optim.zero_grad()
            loss.backward()
            optim.step()
    carrier.eval()
    return carrier


@torch.no_grad()
def carrier_probe_score(carrier, z_train, a_train, xi_train,
                        z_eval, a_eval, xi_eval) -> float:
    def features(z, a):
        prior = carrier(z, a).prior_read.cpu().numpy()
        return np.concatenate([prior[:, GAP[0]:GAP[1]].mean(axis=1),
                               prior[:, T_DEC]], axis=1)
    fit = p1b._probe_fn("cat")
    return float(fit(features(z_train, a_train), xi_train,
                     features(z_eval, a_eval), xi_eval))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    encoder = DinoFeatures(device)

    report: dict[str, Any] = {"schema_version": 1,
                              "study": "v21-f1-mikasa-remembercolor9",
                              "chance": CHANCE, "banks": {}}
    embeddings = {}
    banks = {}
    for bank_seed in BANK_SEEDS:
        train = load_bank(F1 / "banks" / f"rc9_s{bank_seed}_train.npz")
        eval_ = load_bank(F1 / "banks" / f"rc9_s{bank_seed}_eval.npz")
        emb_train = encoder.encode_bank(train)
        emb_eval = encoder.encode_bank(eval_)
        banks[bank_seed] = (train, eval_)
        embeddings[bank_seed] = (emb_train, emb_eval)

        cue = (int(train.events["cue_on"][0]),
               int(train.events["cue_off"][0]) + 1)
        sighted = frame_probe(emb_train, emb_eval, train, eval_, cue)
        leakage = frame_probe(emb_train, emb_eval, train, eval_,
                              (T_DEC, train.length))
        fit = p1b._probe_fn("cat")
        floor_feat = lambda emb, bank: np.concatenate(
            [emb[:, 0], bank.actions.reshape(bank.num_episodes, -1)], axis=1)
        floor = float(fit(floor_feat(emb_train, train), train.xi,
                          floor_feat(emb_eval, eval_), eval_.xi))
        report["banks"][str(bank_seed)] = {
            "sighted": round(sighted, 4), "leakage": round(leakage, 4),
            "floor": round(floor, 4),
            "sighted_pass": bool(sighted >= p1b.SIGHTED_ACC_MIN),
            "leakage_pass": bool(leakage <= CHANCE + 0.05),
            "floor_pass": bool(floor <= CHANCE + 0.05)}
        print(f"[f1-cert] bank {bank_seed}: sighted {sighted:.3f} "
              f"leakage {leakage:.3f} floor {floor:.3f}", flush=True)

    for key in ("sighted_pass", "leakage_pass", "floor_pass"):
        report[key] = sum(report["banks"][str(s)][key]
                          for s in BANK_SEEDS) >= 2

    # (c) inversion direction on bank seed 0 (train split trains carrier
    # and probe; eval split scores), standardized features.
    train, eval_ = banks[0]
    emb_train, emb_eval = embeddings[0]
    mean = emb_train.reshape(-1, emb_train.shape[-1]).mean(axis=0)
    std = emb_train.reshape(-1, emb_train.shape[-1]).std(axis=0) + 1e-6
    z_train = torch.from_numpy((emb_train - mean) / std).float().to(device)
    z_eval = torch.from_numpy((emb_eval - mean) / std).float().to(device)
    a_train = torch.from_numpy(train.actions).float().to(device)
    a_eval = torch.from_numpy(eval_.actions).float().to(device)

    arms: dict[str, list[float]] = {"lkc_rfix": [], "gdelta_l10": []}
    for arm in arms:
        for seed in CARRIER_SEEDS:
            carrier = train_carrier(arm, z_train, a_train, seed, device)
            score = carrier_probe_score(carrier, z_train, a_train, train.xi,
                                        z_eval, a_eval, eval_.xi)
            arms[arm].append(round(score, 4))
            print(f"[f1-cert] {arm} s{seed}: probe {score:.3f}", flush=True)
            del carrier
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report["inversion"] = {
        arm: {"scores": scores, "mean": float(np.mean(scores))}
        for arm, scores in arms.items()}
    report["inversion"]["direction_rfix_minus_gdelta"] = float(
        np.mean(arms["lkc_rfix"]) - np.mean(arms["gdelta_l10"]))

    F1.mkdir(parents=True, exist_ok=True)
    (F1 / "certification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# V21 F1 — RememberColor9 certification (external family)", "",
             "| bank | sighted | leakage | floor |", "|---|---|---|---|"]
    for s in BANK_SEEDS:
        row = report["banks"][str(s)]
        lines += [f"| {s} | {row['sighted']:.3f} | {row['leakage']:.3f} | "
                  f"{row['floor']:.3f} |"]
    lines += ["", f"chance = {CHANCE:.3f} · sighted gate 0.75 · "
              f"demand = sighted PASS + leakage/floor at chance", "",
              f"inversion (n=5): rfix {report['inversion']['lkc_rfix']['mean']:.3f} "
              f"vs gdelta {report['inversion']['gdelta_l10']['mean']:.3f} "
              f"(direction {report['inversion']['direction_rfix_minus_gdelta']:+.3f})"]
    (F1 / "certification.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
