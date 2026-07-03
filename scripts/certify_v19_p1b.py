#!/usr/bin/env python3
"""V19 P1b checkpoint-level certificates (docs/V19_PROPOSAL.md 4.4).

With each frozen P0 encoder (outputs/v19_p0/<task>/<host>/s<seed>/encoder.pt,
reconstructed through the scripts/train_v19_p0.py host builders), certify the
task-encoder pair two-sidedly on fresh banks (derived seeds, 512 train / 256
eval episodes, clean rendering — T4's freeze gap is part of the task itself):

- **Checkpoint integrator probe**: ``[enc(o_0), a_{t-3:t-1}, sum a,
  t/(L-1)] -> xi`` must be at chance.  Threshold: PERMUTATION NULL — 32
  label-shuffled refits (episode labels permuted jointly across both splits),
  pass iff the true score <= the null's 95th percentile.  This replaces the
  P1a fixed +0.05 margin (the registered P1a protocol improvement).
- **Sighted full-history probe**: encoder embeddings at 8 evenly spaced
  frames including the full cue window (4 frames spanning [cue_on, cue_off)
  plus 4 spanning the episode), concatenated with the mean-pooled embedding
  over all frames -> xi; gate >= 0.75 (categorical).  T4: embeddings of the
  last 4 pre-gap frames -> xi, posterior-bounded — gate at
  >= 0.8 * posterior-mean R^2.
- **Probe-level memory demand** := sighted - integrator (the world-model
  analogue of the MDP/POMDP twin gap), plus the probe-level temporal range:
  xi decodability from the trailing w frames, w in {1, 2, 4, 8, 16, 32,
  full} (the history-truncation curve, logged to W&B).

Certificates land in ``<output>/<task>/<host>/s<seed>/certificate.json``;
scripts/aggregate_v19_p1b.py builds the two-sided summary per task x host.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import EpisodeBatch
from lewm.tasks_v19.certify import _cat_accuracy, _ridge_r2
from sklearn.metrics import r2_score
import scripts.train_v19_p0 as p0
from scripts.eval_v19_p2 import integrator_floor_features

P1B_TASKS = ("t1", "t2", "t3", "t4", "t1dev", "t2dev")
DEFAULT_E_TRAIN = 512
DEFAULT_E_EVAL = 256
BANK_SEED_BASE = 190_000       # registered P1b bank-seed namespace
PERMUTATIONS = 32
NULL_PERCENTILE = 95.0
SIGHTED_ACC_MIN = 0.75
T4_POSTERIOR_FRACTION = 0.8
N_CUE_FRAMES = 4               # in-cue sighted frames (spanning the window)
N_EPISODE_FRAMES = 4           # episode-spanning sighted frames
T4_PREGAP_FRAMES = 4
TRUNCATION_WINDOWS = (1, 2, 4, 8, 16, 32, "full")
ENCODE_CHUNK_EPISODES = 8      # 512-frame batches (the P0 health convention)
STREAM = "iid"


def bank_seeds(seed: int) -> tuple[int, int]:
    """(train, eval) bank seeds derived from the encoder seed; disjoint from
    the P1a (seed*1000+k) and P0 (270_701/2) namespaces by construction."""
    return BANK_SEED_BASE + 10 * seed + 1, BANK_SEED_BASE + 10 * seed + 2


# --------------------------------------------------------------------------
# Frozen encoder
# --------------------------------------------------------------------------

def load_frozen_encoder(p0_root: str | Path, task: str, host: str, seed: int,
                        device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the P0 host and load its frozen encoder checkpoint."""
    path = Path(p0_root) / task / host / f"s{seed}" / "encoder.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen P0 encoder: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("host", host), ("task", task), ("seed", seed)):
        if checkpoint.get(key) != expected:
            raise ValueError(f"checkpoint {path} has {key}="
                             f"{checkpoint.get(key)!r}, expected {expected!r}")
    action_dim = int(checkpoint["action_dim"])
    model = (p0.build_sigreg_host(action_dim) if host == "sigreg"
             else p0.build_vicreg_host(action_dim))
    p0.host_encoder(host, model).load_state_dict(
        checkpoint["encoder_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def encode_bank(host: str, model: torch.nn.Module, bank: EpisodeBatch,
                device: torch.device,
                chunk: int = ENCODE_CHUNK_EPISODES) -> np.ndarray:
    """(E, L, D) frozen-encoder embeddings, fp32 eval mode, chunked so the
    sigreg projection head's batch statistics see P0-sized batches."""
    episodes, length = bank.num_episodes, bank.length
    outputs: list[np.ndarray] = []
    for start in range(0, episodes, chunk):
        stop = min(start + chunk, episodes)
        frames = p0.P0EpisodeDataset._frames_tensor(
            bank.frames[start:stop].reshape(-1, p0.IMG_SIZE, p0.IMG_SIZE, 3)
        ).reshape(stop - start, length, 3, p0.IMG_SIZE, p0.IMG_SIZE).to(device)
        outputs.append(
            p0.host_encode(host, model, frames).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


# --------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------

def _gather_frames(embeddings: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(E, L, D) + (E, K) -> (E, K*D) concatenated frame embeddings."""
    episodes = embeddings.shape[0]
    selected = embeddings[np.arange(episodes)[:, None], indices]
    return selected.reshape(episodes, -1)


def _spaced_indices(start: np.ndarray, stop: np.ndarray, count: int
                    ) -> np.ndarray:
    """(E, count) evenly spaced integer indices in [start, stop] inclusive."""
    start = np.asarray(start, dtype=np.float64)
    stop = np.broadcast_to(np.asarray(stop, dtype=np.float64), start.shape)
    return np.round(np.linspace(start, stop, count, axis=-1)).astype(np.int64)


def sighted_features_cat(embeddings: np.ndarray, bank: EpisodeBatch
                         ) -> np.ndarray:
    """Sighted full-history features: 8 frames (4 spanning the full cue
    window + 4 spanning the episode) concatenated, plus the mean-pooled
    embedding over all frames -> (E, 9D)."""
    episodes, length, _ = embeddings.shape
    cue_idx = _spaced_indices(bank.events["cue_on"],
                              bank.events["cue_off"] - 1, N_CUE_FRAMES)
    episode_idx = _spaced_indices(np.zeros(episodes, dtype=np.int64),
                                  np.full(episodes, length - 1), N_EPISODE_FRAMES)
    frames = np.concatenate([cue_idx, episode_idx], axis=1)
    return np.concatenate([_gather_frames(embeddings, frames),
                           embeddings.mean(axis=1)], axis=1)


def sighted_features_t4(embeddings: np.ndarray, bank: EpisodeBatch
                        ) -> np.ndarray:
    """T4 sighted features: embeddings of the last 4 pre-gap frames."""
    gap_on = np.asarray(bank.events["gap_on"], dtype=np.int64)
    indices = gap_on[:, None] + np.arange(-T4_PREGAP_FRAMES, 0)[None, :]
    return _gather_frames(embeddings, indices)


def trailing_features(embeddings: np.ndarray, window: int | str) -> np.ndarray:
    """Trailing-w-frame features for the history-truncation curve."""
    length = embeddings.shape[1]
    w = length if window == "full" else int(window)
    return embeddings[:, length - w:].reshape(embeddings.shape[0], -1)


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

def _probe_fn(xi_kind: str) -> Callable[..., float]:
    return _cat_accuracy if xi_kind == "cat" else _ridge_r2


def probe_with_permutation_null(
        features_train: np.ndarray, y_train: np.ndarray,
        features_eval: np.ndarray, y_eval: np.ndarray, xi_kind: str,
        rng: np.random.Generator,
        permutations: int = PERMUTATIONS) -> dict[str, Any]:
    """Probe score plus a label-permutation null (the registered threshold).

    Labels are permuted jointly across both splits (exchangeability under the
    null of no feature-label association), the probe refit, and the null
    score recorded; threshold = the null's 95th percentile.
    """
    fit = _probe_fn(xi_kind)
    score = fit(features_train, y_train, features_eval, y_eval)
    labels = np.concatenate([y_train, y_eval])
    n_train = len(y_train)
    nulls = []
    for _ in range(permutations):
        shuffled = labels[rng.permutation(len(labels))]
        nulls.append(fit(features_train, shuffled[:n_train],
                         features_eval, shuffled[n_train:]))
    threshold = float(np.percentile(nulls, NULL_PERCENTILE))
    return {
        "score": float(score),
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls)),
        "null_95pct": threshold,
        "permutations": permutations,
        "pass": bool(score <= threshold),
    }


def certify_seed(host: str, task_name: str, seed: int, args,
                 device: torch.device) -> dict[str, Any]:
    """One (task, host, seed) checkpoint certificate."""
    started = time.time()
    model, checkpoint = load_frozen_encoder(
        args.p0_root, task_name, host, seed, device)
    task = make_task(task_name)
    train_seed, eval_seed = bank_seeds(seed)
    train_bank = task.generate(STREAM, args.e_train, train_seed)
    eval_bank = task.generate(STREAM, args.e_eval, eval_seed)
    emb_train = encode_bank(host, model, train_bank, device)
    emb_eval = encode_bank(host, model, eval_bank, device)
    xi_kind = train_bank.xi_kind
    fit = _probe_fn(xi_kind)
    rng = np.random.default_rng(BANK_SEED_BASE + 10 * seed + 3)

    integrator = probe_with_permutation_null(
        integrator_floor_features(emb_train[:, 0], train_bank.actions),
        train_bank.xi,
        integrator_floor_features(emb_eval[:, 0], eval_bank.actions),
        eval_bank.xi, xi_kind, rng, permutations=args.permutations)

    if xi_kind == "cat":
        sighted_score = fit(sighted_features_cat(emb_train, train_bank),
                            train_bank.xi,
                            sighted_features_cat(emb_eval, eval_bank),
                            eval_bank.xi)
        sighted = {
            "score": float(sighted_score),
            "gate": SIGHTED_ACC_MIN,
            "pass": bool(sighted_score >= SIGHTED_ACC_MIN),
            "feature": "8_frames_incl_full_cue_window_plus_meanpool",
        }
    else:
        posterior_r2 = float(r2_score(
            eval_bank.xi, task.posterior_mean_prediction(eval_bank)))
        sighted_score = fit(sighted_features_t4(emb_train, train_bank),
                            train_bank.xi,
                            sighted_features_t4(emb_eval, eval_bank),
                            eval_bank.xi)
        gate = T4_POSTERIOR_FRACTION * posterior_r2
        sighted = {
            "score": float(sighted_score),
            "gate": float(gate),
            "pass": bool(sighted_score >= gate),
            "posterior_mean_r2": posterior_r2,
            "posterior_fraction": T4_POSTERIOR_FRACTION,
            "feature": f"last_{T4_PREGAP_FRAMES}_pregap_frames",
        }

    truncation_curve = {
        str(window): float(fit(trailing_features(emb_train, window),
                               train_bank.xi,
                               trailing_features(emb_eval, window),
                               eval_bank.xi))
        for window in TRUNCATION_WINDOWS}

    certificate: dict[str, Any] = {
        "schema_version": 1,
        "study": "v19-p1b-checkpoint-certificates",
        "task": task_name,
        "host": host,
        "seed": seed,
        "encoder_epochs": checkpoint.get("epochs"),
        "encoder_gates": checkpoint.get("gates"),
        "bank_seeds": {"train": train_seed, "eval": eval_seed},
        "e_train": args.e_train,
        "e_eval": args.e_eval,
        "stream": STREAM,
        "xi_kind": xi_kind,
        "n_classes": train_bank.n_classes,
        "chance": (1.0 / train_bank.n_classes if xi_kind == "cat" else 0.0),
        "integrator": integrator,
        "sighted": sighted,
        "memory_demand": float(sighted["score"] - integrator["score"]),
        "truncation_curve": truncation_curve,
        "two_sided_pass": bool(integrator["pass"] and sighted["pass"]),
        "seconds": round(time.time() - started, 1),
    }
    return certificate


# --------------------------------------------------------------------------
# Figures / W&B
# --------------------------------------------------------------------------

def truncation_figure(certificate: dict[str, Any]):
    from matplotlib.figure import Figure
    curve = certificate["truncation_curve"]
    labels = list(curve.keys())
    values = [curve[label] for label in labels]
    figure = Figure(figsize=(6.0, 3.5))
    axis = figure.subplots()
    axis.plot(range(len(labels)), values, "o-", lw=1.6)
    axis.set_xticks(range(len(labels)), labels)
    axis.axhline(certificate["chance"], color="crimson", ls="--", lw=1.2,
                 label="chance")
    axis.set_xlabel("trailing window w (frames)")
    axis.set_ylabel("accuracy" if certificate["xi_kind"] == "cat" else "R2")
    axis.set_title(f"xi decodability vs history truncation — "
                   f"{certificate['task']}/{certificate['host']}/"
                   f"s{certificate['seed']}")
    axis.legend(fontsize=8)
    return figure


def _log_wandb(args, certificate: dict[str, Any], figure) -> None:
    import wandb
    from lewm.tasks_v19.wandb_utils import _figure_to_image

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=(f"p1b-{certificate['host']}-{certificate['task']}-"
              f"s{certificate['seed']}"),
        group=f"p1b-{certificate['task']}", tags=["p1b", "v19"],
        config=certificate, settings=wandb.Settings(init_timeout=180))
    rows = [[clause, json.dumps(certificate[clause], default=str)]
            for clause in ("integrator", "sighted", "memory_demand",
                           "truncation_curve", "two_sided_pass")]
    run.log({
        "certificate/table": wandb.Table(columns=["clause", "value"], data=rows),
        "figures/truncation_curve": wandb.Image(_figure_to_image(figure)),
    })
    run.summary.update({
        "integrator_score": certificate["integrator"]["score"],
        "integrator_null_95pct": certificate["integrator"]["null_95pct"],
        "sighted_score": certificate["sighted"]["score"],
        "memory_demand": certificate["memory_demand"],
        "two_sided_pass": certificate["two_sided_pass"],
    })
    run.finish()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=P1B_TASKS)
    parser.add_argument("--host", required=True, choices=p0.HOSTS)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--p0-root", default="outputs/v19_p0")
    parser.add_argument("--output", default="outputs/v19_p1b")
    parser.add_argument("--e-train", type=int, default=DEFAULT_E_TRAIN)
    parser.add_argument("--e-eval", type=int, default=DEFAULT_E_EVAL)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", dest="wandb", action="store_true",
                        default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-v19")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for seed in seeds:
        torch.manual_seed(seed)
        certificate = certify_seed(args.host, args.task, seed, args, device)
        out_dir = Path(args.output) / args.task / args.host / f"s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "certificate.json"
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite {out_path}")
        out_path.write_text(json.dumps(certificate, indent=2, sort_keys=True))
        figure = truncation_figure(certificate)
        figure.savefig(out_dir / "truncation_curve.png", dpi=130,
                       bbox_inches="tight")
        if args.wandb:
            try:
                _log_wandb(args, certificate, figure)
            except Exception as error:  # noqa: BLE001 - reporting best-effort
                warnings.warn(f"wandb logging failed: {error!r}", stacklevel=1)
        verdict = "PASS" if certificate["two_sided_pass"] else "FAIL"
        print(f"=== v19-p1b {args.host}/{args.task}/s{seed}: {verdict} | "
              f"integrator={certificate['integrator']['score']:.4f} "
              f"(null95={certificate['integrator']['null_95pct']:.4f}) "
              f"sighted={certificate['sighted']['score']:.4f} "
              f"(gate={certificate['sighted']['gate']:.4f}) "
              f"demand={certificate['memory_demand']:.4f} -> {out_dir} ===",
              flush=True)


if __name__ == "__main__":
    main()
