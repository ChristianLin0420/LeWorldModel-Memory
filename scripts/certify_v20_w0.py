#!/usr/bin/env python3
"""V20 W0 checkpoint certificates on the salience ladder (claim 2, the s*
instrument; docs/V20_PROPOSAL.md 4.1/4.5).

For every trained W0 encoder on a ladder level (t1s1/t1s2/t1s3/t1), this
script computes the P1b two-sided checkpoint certificate — the calibrated
V19 machinery verbatim (scripts/certify_v19_p1b.py feature builders, RidgeCV
probes, 200-permutation integrator nulls) — with one W0-specific extension:
the encoder loader understands the visreg arms (exact-LeWM architecture from
scripts/train_v20_w0.py) alongside the vicreg reference.

The per-host certified salience threshold s* is then the lowest ladder level
whose *sighted* certificate passes on a majority of seeds (>= 2/3), read out
by scripts/aggregate_v20_w0.py.  Registered prediction: s*(visreg) <
s*(vicreg).

Certificates land in <output>/certificates/<task>/<arm>/s<seed>.json.
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
import scripts.certify_v19_p1b as p1b
import scripts.train_v19_p0 as p0
import scripts.train_v20_w0 as w0

LADDER_TASKS = ("t1s1", "t1s2", "t1s3", "t1")


def load_w0_encoder(w0_root: str | Path, task: str, arm: str, seed: int,
                    device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild a W0 host and load its frozen encoder checkpoint."""
    path = Path(w0_root) / task / arm / f"s{seed}" / "encoder.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen W0 encoder: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("host", arm), ("task", task), ("seed", seed)):
        if checkpoint.get(key) != expected:
            raise ValueError(f"checkpoint {path} has {key}="
                             f"{checkpoint.get(key)!r}, expected {expected!r}")
    action_dim = int(checkpoint["action_dim"])
    kind = w0.host_kind(arm)
    model = (p0.build_vicreg_host(action_dim) if kind == "vicreg"
             else p0.build_sigreg_host(action_dim))
    encode_host = "vicreg" if kind == "vicreg" else "sigreg"
    p0.host_encoder(encode_host, model).load_state_dict(
        checkpoint["encoder_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def certify_cell(w0_root: str | Path, task_name: str, arm: str, seed: int,
                 args, device: torch.device) -> dict[str, Any]:
    """One (ladder level, arm, seed) certificate — P1b's certify_seed with
    the W0 encoder loader; probe configuration identical (calibrated
    machinery, proposal 4.5)."""
    started = time.time()
    model, checkpoint = load_w0_encoder(w0_root, task_name, arm, seed, device)
    encode_host = "vicreg" if w0.host_kind(arm) == "vicreg" else "sigreg"
    task = make_task(task_name)
    train_seed, eval_seed = p1b.bank_seeds(seed)
    train_bank = task.generate(p1b.STREAM, args.e_train, train_seed)
    eval_bank = task.generate(p1b.STREAM, args.e_eval, eval_seed)
    emb_train = p1b.encode_bank(encode_host, model, train_bank, device)
    emb_eval = p1b.encode_bank(encode_host, model, eval_bank, device)
    xi_kind = train_bank.xi_kind
    fit = p1b._probe_fn(xi_kind)
    rng = np.random.default_rng(p1b.BANK_SEED_BASE + 10 * seed + 3)

    integrator = p1b.probe_with_permutation_null(
        p1b.integrator_floor_features(emb_train[:, 0], train_bank.actions),
        train_bank.xi,
        p1b.integrator_floor_features(emb_eval[:, 0], eval_bank.actions),
        eval_bank.xi, xi_kind, rng, permutations=args.permutations)

    sighted_score = fit(
        p1b.sighted_features_cat(emb_train, train_bank), train_bank.xi,
        p1b.sighted_features_cat(emb_eval, eval_bank), eval_bank.xi)
    sighted = {
        "score": float(sighted_score),
        "gate": p1b.SIGHTED_ACC_MIN,
        "pass": bool(sighted_score >= p1b.SIGHTED_ACC_MIN),
        "feature": "8_frames_incl_full_cue_window_plus_meanpool",
    }

    truncation_curve = {
        str(window): float(fit(p1b.trailing_features(emb_train, window),
                               train_bank.xi,
                               p1b.trailing_features(emb_eval, window),
                               eval_bank.xi))
        for window in p1b.TRUNCATION_WINDOWS}

    return {
        "schema_version": 1,
        "study": "v20-w0-salience-certificates",
        "task": task_name,
        "arm": arm,
        "seed": seed,
        "encoder_epochs": checkpoint.get("epochs"),
        "visreg_lambda": checkpoint.get("visreg_lambda"),
        "bank_seeds": {"train": train_seed, "eval": eval_seed},
        "e_train": args.e_train,
        "e_eval": args.e_eval,
        "stream": p1b.STREAM,
        "xi_kind": xi_kind,
        "n_classes": train_bank.n_classes,
        "chance": 1.0 / train_bank.n_classes,
        "integrator": integrator,
        "sighted": sighted,
        "memory_demand": float(sighted["score"] - integrator["score"]),
        "truncation_curve": truncation_curve,
        "two_sided_pass": bool(integrator["pass"] and sighted["pass"]),
        "seconds": round(time.time() - started, 1),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v20_w0")
    parser.add_argument("--tasks", default=",".join(LADDER_TASKS))
    parser.add_argument("--arms", required=True,
                        help="comma-separated W0 arms (e.g. visreg75,vicreg)")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--e-train", type=int, default=p1b.DEFAULT_E_TRAIN)
    parser.add_argument("--e-eval", type=int, default=p1b.DEFAULT_E_EVAL)
    parser.add_argument("--permutations", type=int, default=p1b.PERMUTATIONS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    seeds = [int(value) for value in args.seeds.split(",")]
    for task in tasks:
        if task not in LADDER_TASKS:
            raise ValueError(f"{task!r} is not a ladder task {LADDER_TASKS}")

    for task, arm, seed in ((task, arm, seed) for task in tasks
                            for arm in arms for seed in seeds):
        out_path = (Path(args.root) / "certificates" / task / arm
                    / f"s{seed}.json")
        if out_path.exists():
            print(f"[v20-w0-cert] skip existing {out_path}", flush=True)
            continue
        encoder_path = (Path(args.root) / task / arm / f"s{seed}"
                        / "encoder.pt")
        if not encoder_path.exists():
            print(f"[v20-w0-cert] MISSING encoder {encoder_path} — skipped",
                  flush=True)
            continue
        certificate = certify_cell(args.root, task, arm, seed, args, device)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("x") as stream:
            json.dump(certificate, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"[v20-w0-cert] {task}/{arm}/s{seed}: "
              f"sighted={certificate['sighted']['score']:.3f} "
              f"(pass={certificate['sighted']['pass']}) "
              f"integrator={certificate['integrator']['score']:.3f} "
              f"(pass={certificate['integrator']['pass']}) "
              f"[{certificate['seconds']}s]", flush=True)


if __name__ == "__main__":
    main()
