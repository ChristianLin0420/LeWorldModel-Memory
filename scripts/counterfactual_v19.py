#!/usr/bin/env python3
"""Action-swap counterfactual gate for V19 P3 (docs/V19_PROPOSAL.md 4.5,
Tier 2 item 2, claims-ladder row 5: "transport is causal").

For a trained (task, arm, seed) cell the overlay design makes counterfactual
re-rendering exact: the exogenous script (cue timing, xi, OU noise) is drawn
from rngs that never touch actions, so re-running ``collect_base`` with the
same base seed and *permuted* actions replays every exogenous realization
verbatim while the endogenous (reacher) trajectory diverges.  Protocol per
run, on E fresh eval episodes (registered seed ``CF_SEED``, iid stream):

1. Factual branch: standard pipeline (actions ``a``, exogenous script ``S``).
2. Counterfactual branch: SAME script ``S`` and base seed, actions ``a'`` — a
   per-episode derangement (no fixed points) of the action sequence at
   indices ``t >= boundary``, with ``boundary = shuffle_off | cue_off`` for
   categorical tasks and ``boundary = gap_on`` for t4 (the proposal's
   "snapshot at gap onset, roll executed vs permuted actions").  Actions
   before the boundary are untouched, so cue-window frames are byte-identical
   across branches.  Branches are rendered CLEAN (no P0 training corruption):
   corruption is a training-regime nuisance, and the divergence measure is a
   *difference* between branches through the same model, so leaving it out
   keeps the endpoint purely action-attributable (registered resolution).
   T4's own observation freeze is task content and is kept.
3. Both branches run through the trained encoder+carrier; the per-episode
   predicted-latent divergence is the mean L2 distance between the branches'
   ``prior_read`` over the registered decision window
   (``[deep_window_start, t_dec]`` categorical / ``[gap_on+1, gap_off]`` t4);
   ground-truth divergence is the mean L2 distance between the branches'
   ``endo_state`` (qpos+qvel) over the same window.
4. Endpoints: Spearman correlation across episodes between predicted-latent
   and ground-truth divergence, with an episode-resampling bootstrap CI and
   the one-sided bootstrap p-value; plus the factorization check — the
   registered xi probe (fit exactly as in scripts/eval_v19_p2.py, on the
   run's own ``eval_export.npz``) applied to ``prior_read`` of both branches
   must be ~invariant (categorical: fraction of episodes with identical
   predicted class; t4: mean L2 between branch predictions).

Outputs per run: ``counterfactual_results.json`` (write-once) plus
``counterfactual_arrays.npz`` with the per-episode audit arrays, and guarded
W&B logging.  ``scripts/gates_v19_p3.py`` consumes the JSON for the pooled
Tier-2 gate (crossed-bootstrap CI over per-cell Spearman rhos > 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v19_carriers import make_carrier
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import EpisodeBatch
from lewm.tasks_v19.dmc_base import collect_base
from scripts.eval_v19_p2 import deep_window_start, load_export
import scripts.train_v19_p0 as p0

# Registered constants (frozen before any P3 checkpoint is opened).
CF_SEED = 270_731            # episode seed: disjoint from train/val/corruption
CF_EPISODES = 64
CF_STREAM = "iid"
CF_PERM_SALT = 0xCF19        # derangement rng domain separation
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 19_031
ENCODE_CHUNK = 16
RESULTS_NAME = "counterfactual_results.json"
ARRAYS_NAME = "counterfactual_arrays.npz"
SCHEMA_VERSION = 1
DEFAULT_ARMS = ("lkc", "lkc_b0")


# --------------------------------------------------------------------------
# Derangement (the counterfactual action swap)
# --------------------------------------------------------------------------

def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform fixed-point-free permutation of ``range(size)`` (rejection)."""
    if size < 2:
        raise ValueError(f"a derangement needs >= 2 elements, got {size}")
    while True:
        permutation = rng.permutation(size)
        if not np.any(permutation == np.arange(size)):
            return permutation


def derange_actions(actions: np.ndarray, boundary: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Per-episode derangement of the action rows at indices >= boundary.

    Rows before the boundary are copied verbatim (the cue window stays
    untouched); rows from the boundary on are permuted among themselves with
    no index fixed, so the executed multiset is preserved exactly.
    """
    if actions.ndim != 3:
        raise ValueError(f"actions must be (E, L-1, A), got {actions.shape}")
    episodes, num_actions, _ = actions.shape
    boundary = np.asarray(boundary, dtype=np.int64)
    if boundary.shape != (episodes,):
        raise ValueError(f"boundary must be ({episodes},), got {boundary.shape}")
    swapped = actions.copy()
    for episode in range(episodes):
        start = int(boundary[episode])
        if not 0 <= start <= num_actions - 2:
            raise ValueError(
                f"episode {episode}: boundary {start} leaves a segment of "
                f"length {num_actions - start} (< 2) in {num_actions} actions")
        segment = np.arange(start, num_actions)
        swapped[episode, segment] = actions[
            episode, segment[derangement(segment.size, rng)]]
    return swapped


def derangement_boundaries(events: Mapping[str, np.ndarray],
                           xi_kind: str) -> np.ndarray:
    """Registered per-episode permutation boundary (action index).

    Categorical tasks: the informative phase ends at ``shuffle_off`` (shell
    game) or ``cue_off``; t4 (continuous): ``gap_on`` — actions diverge
    exactly when observations freeze, the proposal's registered intervention
    point.
    """
    if xi_kind == "cont":
        return np.asarray(events["gap_on"], dtype=np.int64)
    key = "shuffle_off" if "shuffle_off" in events else "cue_off"
    return np.asarray(events[key], dtype=np.int64)


def decision_windows(events: Mapping[str, np.ndarray], length: int,
                     xi_kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Registered per-episode divergence window [start, end] (inclusive).

    Categorical: the deep-gap probe window ``[deep_window_start, t_dec]``
    (scripts/eval_v19_p2.py); t4: the unobserved gap ``[gap_on+1, gap_off]``
    ending at the decision time.
    """
    if xi_kind == "cont":
        start = np.asarray(events["gap_on"], dtype=np.int64) + 1
        end = np.asarray(events["gap_off"], dtype=np.int64)
    else:
        start = deep_window_start(events)
        end = np.full_like(start, length - 1)
    if not np.all((start >= 1) & (start <= end) & (end <= length - 1)):
        raise ValueError("invalid decision window bounds")
    return start, end


# --------------------------------------------------------------------------
# Branch generation (exogenous script shared verbatim)
# --------------------------------------------------------------------------

def generate_branches(task_name: str, episodes: int, seed: int,
                      stream: str = CF_STREAM
                      ) -> tuple[EpisodeBatch, EpisodeBatch, np.ndarray]:
    """(factual, counterfactual, boundary): same exogenous script, deranged
    actions after the per-episode boundary.

    Mirrors ``_OverlayTask._generate`` with the nuisance/xi rngs drawn ONCE
    and the script rendered onto both branches; the counterfactual branch
    re-rolls the reacher via ``collect_base(..., action_override=...)``.
    Hard sanity assertions (exogenous identity, action multiset, pre-boundary
    frame identity) fail the run rather than emit a wrong certificate.
    """
    task = make_task(task_name)
    base_seed, nuisance_rng, xi_rng = task._rngs(seed)
    script = task._sample_script(episodes, nuisance_rng, xi_rng, 0)
    events = {key: np.asarray(script[key]) for key in task.event_keys}
    boundary = derangement_boundaries(events, task.xi_kind)

    frames_f, actions, endo_f = collect_base(
        episodes, task.length, base_seed, stream)
    exo_f = task._render(frames_f, script)

    perm_rng = np.random.default_rng((CF_PERM_SALT, seed))
    actions_cf = derange_actions(actions, boundary, perm_rng)
    frames_c, actions_c, endo_c = collect_base(
        episodes, task.length, base_seed, stream, action_override=actions_cf)
    exo_c = task._render(frames_c, script)

    if not np.array_equal(exo_f, exo_c):
        raise RuntimeError("exogenous state differs across branches — the "
                           "script did not replay verbatim")
    if not np.array_equal(actions_c, actions_cf):
        raise RuntimeError("counterfactual branch did not execute the "
                           "deranged actions verbatim")
    for episode in range(episodes):
        stop = int(boundary[episode]) + 1
        if not np.array_equal(frames_f[episode, :stop], frames_c[episode, :stop]):
            raise RuntimeError(f"episode {episode}: pre-boundary frames "
                               "differ across branches")
        if not np.array_equal(
                np.sort(actions[episode], axis=0),
                np.sort(actions_cf[episode], axis=0)):
            raise RuntimeError(f"episode {episode}: action multiset changed")

    def _bank(frames: np.ndarray, bank_actions: np.ndarray,
              endo: np.ndarray) -> EpisodeBatch:
        return EpisodeBatch(
            frames=frames, actions=bank_actions, xi=script["xi"],
            xi_kind=task.xi_kind, n_classes=task.n_classes,
            endo_state=endo, exo_state=exo_f, events=dict(events),
            stream=stream, task=task_name, seed=seed)

    return _bank(frames_f, actions, endo_f), _bank(frames_c, actions_cf,
                                                   endo_c), boundary


# --------------------------------------------------------------------------
# Model plumbing (checkpoint format of scripts/train_v19_p2.py)
# --------------------------------------------------------------------------

def load_cell(checkpoint_path: str | Path, device: torch.device
              ) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    """Rebuild (host model, carrier) from a P2/P3 trainer checkpoint."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu",
                         weights_only=True)
    host, arm = str(payload["host"]), str(payload["arm"])
    action_dim = int(payload["action_dim"])
    model = (p0.build_sigreg_host(action_dim) if host == "sigreg"
             else p0.build_vicreg_host(action_dim))
    model.load_state_dict(payload["model_state_dict"])
    embed_dim = int(payload["host_config"]["embed_dim"])
    carrier = make_carrier(arm, embed_dim, action_dim)
    carrier.load_state_dict(payload["carrier_state_dict"])
    meta = {key: payload[key] for key in
            ("host", "arm", "task", "seed", "epochs", "action_dim",
             "host_config", "carrier_config")}
    return model.to(device).eval(), carrier.to(device).eval(), meta


def build_untrained_cell(task_name: str, arm: str, host: str, seed: int,
                         device: torch.device
                         ) -> tuple[torch.nn.Module, torch.nn.Module,
                                    dict[str, Any]]:
    """Freshly initialized cell for smoke runs when no checkpoint exists."""
    torch.manual_seed(seed)
    from lewm.tasks_v19.base import ACTION_DIM
    model = (p0.build_sigreg_host(ACTION_DIM) if host == "sigreg"
             else p0.build_vicreg_host(ACTION_DIM))
    embed_dim = int(p0.HOST_CONFIGS[host]["embed_dim"])
    carrier = make_carrier(arm, embed_dim, ACTION_DIM)
    meta = {"host": host, "arm": arm, "task": task_name, "seed": seed,
            "epochs": 0, "action_dim": ACTION_DIM,
            "host_config": p0.HOST_CONFIGS[host],
            "carrier_config": carrier.describe()}
    return model.to(device).eval(), carrier.to(device).eval(), meta


@torch.no_grad()
def branch_prior_read(host: str, model: torch.nn.Module,
                      carrier: torch.nn.Module, batch: EpisodeBatch,
                      device: torch.device, chunk: int = ENCODE_CHUNK
                      ) -> np.ndarray:
    """prior_read (E, L, D) of one branch, fp32 eval mode (no autocast)."""
    model.eval()
    carrier.eval()
    chunks: list[np.ndarray] = []
    episodes, length = batch.num_episodes, batch.length
    for start in range(0, episodes, chunk):
        stop = min(start + chunk, episodes)
        frames = p0.P0EpisodeDataset._frames_tensor(
            batch.frames[start:stop].reshape(-1, p0.IMG_SIZE, p0.IMG_SIZE, 3)
        ).reshape(stop - start, length, 3, p0.IMG_SIZE, p0.IMG_SIZE).to(device)
        actions = torch.from_numpy(batch.actions[start:stop]).to(device)
        z = p0.host_encode(host, model, frames).float()
        chunks.append(carrier(z, actions).prior_read.float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


# --------------------------------------------------------------------------
# Divergence endpoints
# --------------------------------------------------------------------------

def windowed_divergence(branch_a: np.ndarray, branch_b: np.ndarray,
                        start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """(E,) mean L2 distance between (E, L, D) branches over [start, end]."""
    if branch_a.shape != branch_b.shape or branch_a.ndim != 3:
        raise ValueError("branches must share an (E, L, D) shape")
    distance = np.linalg.norm(
        branch_a.astype(np.float64) - branch_b.astype(np.float64), axis=-1)
    steps = np.arange(branch_a.shape[1])[None, :]
    mask = (steps >= np.asarray(start)[:, None]) & \
           (steps <= np.asarray(end)[:, None])
    counts = mask.sum(axis=1)
    if not (counts > 0).all():
        raise ValueError("empty divergence window for at least one episode")
    return (distance * mask).sum(axis=1) / counts


def spearman_bootstrap(latent: np.ndarray, ground_truth: np.ndarray,
                       draws: int = BOOTSTRAP_DRAWS,
                       seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Spearman rho across episodes + episode-resampling bootstrap CI/p.

    ``p_pos`` is the one-sided bootstrap p-value for rho > 0 with the add-one
    convention ``(#draws <= 0 + 1) / (valid + 1)``; degenerate draws (a
    constant resample) are excluded and counted.  A constant divergence
    vector (e.g. a carrier whose read never left its zero init) has no
    defined rank correlation: the result is reported with ``rho = None`` and
    status ``degenerate`` rather than crashing — downstream gates read it as
    NA (fail-closed, reported).
    """
    from scipy.stats import rankdata, spearmanr

    latent = np.asarray(latent, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if latent.shape != ground_truth.shape or latent.ndim != 1:
        raise ValueError("divergence vectors must be matching 1-D arrays")
    episodes = latent.size
    if episodes < 4:
        raise ValueError(f"need >= 4 episodes for the Spearman gate, "
                         f"got {episodes}")
    if np.all(latent == latent[0]) or np.all(ground_truth == ground_truth[0]):
        return {"rho": None, "status": "degenerate_constant_divergence",
                "episodes": int(episodes), "draws": int(draws),
                "valid_draws": 0, "bootstrap_seed": int(seed)}
    rho = float(spearmanr(latent, ground_truth).statistic)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, episodes, size=(draws, episodes))
    ranks_x = rankdata(latent[indices], axis=1).astype(np.float64)
    ranks_y = rankdata(ground_truth[indices], axis=1).astype(np.float64)
    ranks_x -= ranks_x.mean(axis=1, keepdims=True)
    ranks_y -= ranks_y.mean(axis=1, keepdims=True)
    scale = np.sqrt((ranks_x ** 2).sum(axis=1) * (ranks_y ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        estimates = (ranks_x * ranks_y).sum(axis=1) / scale
    valid = estimates[np.isfinite(estimates)]
    if valid.size == 0:
        return {"rho": rho, "status": "degenerate_all_draws",
                "episodes": int(episodes), "draws": int(draws),
                "valid_draws": 0, "bootstrap_seed": int(seed)}
    return {
        "rho": rho,
        "status": "ok",
        "ci95_low": float(np.quantile(valid, 0.025, method="linear")),
        "ci95_high": float(np.quantile(valid, 0.975, method="linear")),
        "p_pos": float((int((valid <= 0.0).sum()) + 1) / (valid.size + 1)),
        "draws": int(draws),
        "valid_draws": int(valid.size),
        "bootstrap_seed": int(seed),
        "episodes": int(episodes),
    }


# --------------------------------------------------------------------------
# xi-probe factorization check
# --------------------------------------------------------------------------

def _xi_features(prior_read: np.ndarray, events: Mapping[str, np.ndarray],
                 xi_kind: str) -> np.ndarray:
    """The registered probe coordinate of scripts/eval_v19_p2.py."""
    if xi_kind == "cat":
        from scripts.eval_v19_p2 import registered_cat_features
        return registered_cat_features(prior_read, deep_window_start(events))
    gap_off = np.asarray(events["gap_off"], dtype=np.int64)
    return prior_read[np.arange(prior_read.shape[0]), gap_off]


def fit_xi_probe(prior_read: np.ndarray, events: Mapping[str, np.ndarray],
                 xi: np.ndarray, xi_kind: str):
    """Fit the eval_v19_p2 probe family on the registered coordinate."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = _xi_features(prior_read, events, xi_kind)
    if xi_kind == "cat":
        probe = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=0))
    else:
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
    probe.fit(features, xi)
    return probe


def xi_invariance(probe, probe_source: str, prior_f: np.ndarray,
                  prior_c: np.ndarray, events: Mapping[str, np.ndarray],
                  xi: np.ndarray, xi_kind: str) -> dict[str, Any]:
    """Factorization check: action swaps must leave the xi probe invariant."""
    features_f = _xi_features(prior_f, events, xi_kind)
    features_c = _xi_features(prior_c, events, xi_kind)
    prediction_f = probe.predict(features_f)
    prediction_c = probe.predict(features_c)
    result: dict[str, Any] = {"probe_source": probe_source, "xi_kind": xi_kind}
    if xi_kind == "cat":
        result["branch_agreement"] = float(
            np.mean(prediction_f == prediction_c))
        result["factual_accuracy"] = float(np.mean(prediction_f == xi))
    else:
        between = np.linalg.norm(
            prediction_f.astype(np.float64) - prediction_c.astype(np.float64),
            axis=-1)
        to_truth = np.linalg.norm(
            prediction_f.astype(np.float64) - xi.astype(np.float64), axis=-1)
        result["branch_mean_l2"] = float(between.mean())
        result["factual_mean_l2_to_truth"] = float(to_truth.mean())
    return result


# --------------------------------------------------------------------------
# Per-run protocol
# --------------------------------------------------------------------------

def _summary(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def run_counterfactual(model: torch.nn.Module, carrier: torch.nn.Module,
                       meta: Mapping[str, Any], task_name: str,
                       device: torch.device, episodes: int = CF_EPISODES,
                       cf_seed: int = CF_SEED, chunk: int = ENCODE_CHUNK,
                       export_path: Path | None = None,
                       untrained: bool = False) -> dict[str, Any]:
    """Execute the full protocol for one cell; returns the results payload."""
    factual, counterfactual, boundary = generate_branches(
        task_name, episodes, cf_seed)
    window_start, window_end = decision_windows(
        factual.events, factual.length, factual.xi_kind)

    host = str(meta["host"])
    prior_f = branch_prior_read(host, model, carrier, factual, device, chunk)
    prior_c = branch_prior_read(host, model, carrier, counterfactual, device,
                                chunk)
    latent_divergence = windowed_divergence(
        prior_f, prior_c, window_start, window_end)
    gt_divergence = windowed_divergence(
        factual.endo_state, counterfactual.endo_state,
        window_start, window_end)
    spearman = spearman_bootstrap(latent_divergence, gt_divergence)

    if export_path is not None and export_path.is_file():
        export = load_export(export_path)
        export_events = {name.removeprefix("event_"): value
                         for name, value in export.items()
                         if name.startswith("event_")}
        probe = fit_xi_probe(export["prior_read"], export_events,
                             export["xi"], factual.xi_kind)
        probe_source = "eval_export"
    else:
        probe = fit_xi_probe(prior_f, factual.events, factual.xi,
                             factual.xi_kind)
        probe_source = "factual_branch"
    invariance = xi_invariance(probe, probe_source, prior_f, prior_c,
                               factual.events, factual.xi, factual.xi_kind)

    return {
        "schema_version": SCHEMA_VERSION,
        "study": "v19-p3-counterfactual",
        "task": task_name,
        "arm": str(meta["arm"]),
        "seed": int(meta["seed"]),
        "host": host,
        "untrained": bool(untrained),
        "episodes": int(episodes),
        "cf_seed": int(cf_seed),
        "stream": CF_STREAM,
        "boundary_rule": ("gap_on" if factual.xi_kind == "cont"
                          else "shuffle_off|cue_off"),
        "window_rule": ("gap_on+1..gap_off" if factual.xi_kind == "cont"
                        else "deep_window_start..t_dec"),
        "latent_divergence": _summary(latent_divergence),
        "gt_divergence": _summary(gt_divergence),
        "spearman": spearman,
        "xi_invariance": invariance,
        "carrier_config": dict(meta["carrier_config"]),
        "checkpoint_epochs": int(meta["epochs"]),
        "arrays": {
            "latent_divergence": latent_divergence,
            "gt_divergence": gt_divergence,
            "boundary": boundary,
            "window_start": window_start,
            "window_end": window_end,
        },
    }


def write_results(run_dir: Path, results: dict[str, Any],
                  force: bool = False) -> None:
    """Write the JSON summary (write-once) and the per-episode audit NPZ."""
    arrays = results.pop("arrays")
    results_path = run_dir / RESULTS_NAME
    arrays_path = run_dir / ARRAYS_NAME
    if not force:
        for path in (results_path, arrays_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
    with results_path.open("w" if force else "x") as stream:
        json.dump(p0._sanitize(results), stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
    np.savez_compressed(arrays_path, **arrays)


# --------------------------------------------------------------------------
# W&B (guarded — an outage must never invalidate a gate input)
# --------------------------------------------------------------------------

def _scatter_figure(latent: np.ndarray, ground_truth: np.ndarray,
                    label: str, rho: float):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(5.0, 4.2))
    axis = figure.subplots()
    axis.scatter(ground_truth, latent, s=14, alpha=0.7, color="tab:blue")
    axis.set_xlabel("ground-truth endo divergence")
    axis.set_ylabel("prior_read divergence")
    axis.set_title(f"action-swap counterfactual — {label} "
                   f"(Spearman {rho:+.3f})")
    return figure


def log_wandb(args: argparse.Namespace, results: Mapping[str, Any],
              latent: np.ndarray, ground_truth: np.ndarray) -> None:
    def _log():
        import wandb
        from lewm.tasks_v19.wandb_utils import _figure_to_image
        label = (f"{results['arm']}/{results['task']}/s{results['seed']}")
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"p3cf-{results['arm']}-{results['task']}-s{results['seed']}",
            group=f"p3cf-{results['task']}",
            tags=["p3", "v19", "counterfactual", results["arm"]],
            config={key: value for key, value in results.items()
                    if key != "arrays"},
            settings=wandb.Settings(init_timeout=180))
        spearman = results["spearman"]
        payload = {
            "cf/latent_divergence_mean": results["latent_divergence"]["mean"],
            "cf/gt_divergence_mean": results["gt_divergence"]["mean"],
        }
        for key in ("rho", "ci95_low", "ci95_high", "p_pos"):
            if spearman.get(key) is not None:
                payload[f"cf/spearman_{key}"] = spearman[key]
        if spearman.get("rho") is not None:
            payload["figures/divergence_scatter"] = wandb.Image(
                _figure_to_image(_scatter_figure(
                    latent, ground_truth, label, spearman["rho"])))
        invariance = results["xi_invariance"]
        for key in ("branch_agreement", "factual_accuracy",
                    "branch_mean_l2", "factual_mean_l2_to_truth"):
            if key in invariance:
                payload[f"cf/xi_{key}"] = invariance[key]
        run.log(payload)
        run.summary.update({f"cf/{key}": value for key, value
                            in results["spearman"].items()})
        run.finish()
    p0._guarded("counterfactual_log", _log)


# --------------------------------------------------------------------------
# Discovery / main
# --------------------------------------------------------------------------

def discover_runs(root: Path, arms: Iterable[str],
                  tasks: Iterable[str] | None = None) -> list[Path]:
    """Run dirs ``root/<task>/<arm>/s*`` holding a checkpoint, arm-filtered."""
    arms = tuple(arms)
    runs = []
    for checkpoint in sorted(root.glob("*/*/s*/checkpoint.pt")):
        run_dir = checkpoint.parent
        task, arm = run_dir.parts[-3], run_dir.parts[-2]
        if arm in arms and (tasks is None or task in tuple(tasks)):
            runs.append(run_dir)
    return runs


def process_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    model, carrier, meta = load_cell(run_dir / "checkpoint.pt", device)
    results = run_counterfactual(
        model, carrier, meta, str(meta["task"]), device,
        episodes=args.episodes, cf_seed=args.cf_seed, chunk=args.chunk,
        export_path=run_dir / "eval_export.npz")
    arrays = results["arrays"]
    latent = arrays["latent_divergence"].copy()
    ground_truth = arrays["gt_divergence"].copy()
    write_results(run_dir, results, force=args.force)
    if args.wandb:
        log_wandb(args, results, latent, ground_truth)
    return results


def process_untrained(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    model, carrier, meta = build_untrained_cell(
        args.task, args.arm, args.host, args.seed, device)
    results = run_counterfactual(
        model, carrier, meta, args.task, device, episodes=args.episodes,
        cf_seed=args.cf_seed, chunk=args.chunk, export_path=None,
        untrained=True)
    arrays = results["arrays"]
    latent = arrays["latent_divergence"].copy()
    ground_truth = arrays["gt_divergence"].copy()
    output_dir = Path(args.output) / args.task / args.arm / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results(output_dir, results, force=args.force)
    if args.wandb:
        log_wandb(args, results, latent, ground_truth)
    print(f"[v19-cf] UNTRAINED {args.arm}/{args.task}/s{args.seed}: "
          f"{_spearman_line(results['spearman'])} "
          f"-> {output_dir / RESULTS_NAME}", flush=True)
    return results


def _spearman_line(spearman: Mapping[str, Any]) -> str:
    if spearman.get("rho") is None or spearman.get("status") != "ok":
        return f"spearman={spearman.get('status', 'NA')}"
    return (f"rho={spearman['rho']:+.4f} "
            f"ci=[{spearman['ci95_low']:+.4f}, {spearman['ci95_high']:+.4f}] "
            f"p_pos={spearman['p_pos']:.5f}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v19_p3",
                        help="grid root to sweep for trained cells")
    parser.add_argument("--run", action="append", default=None,
                        help="explicit run dir(s) instead of --root discovery")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="comma list of arms to sweep under --root")
    parser.add_argument("--tasks", default=None,
                        help="optional comma list of tasks to restrict to")
    parser.add_argument("--episodes", type=int, default=CF_EPISODES)
    parser.add_argument("--cf-seed", type=int, default=CF_SEED)
    parser.add_argument("--chunk", type=int, default=ENCODE_CHUNK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if results exist")
    parser.add_argument("--untrained", action="store_true",
                        help="smoke: run one freshly built (untrained) cell")
    parser.add_argument("--task", default="t1dev",
                        help="task for --untrained")
    parser.add_argument("--arm", default="lkc", help="arm for --untrained")
    parser.add_argument("--host", default="vicreg", choices=p0.HOSTS,
                        help="host for --untrained")
    parser.add_argument("--seed", type=int, default=0,
                        help="cell seed for --untrained")
    parser.add_argument("--output", default="outputs/v19_p3_cf_smoke",
                        help="output root for --untrained")
    parser.add_argument("--wandb", dest="wandb", action="store_true",
                        default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-v19")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.episodes < 4:
        raise ValueError("need >= 4 episodes")
    if args.untrained:
        process_untrained(args)
        return
    if args.run:
        run_dirs = [Path(run) for run in args.run]
    else:
        tasks = tuple(args.tasks.split(",")) if args.tasks else None
        run_dirs = discover_runs(Path(args.root), args.arms.split(","), tasks)
    if not run_dirs:
        raise FileNotFoundError("no trained cells found "
                                f"(root={args.root}, arms={args.arms})")
    for run_dir in run_dirs:
        if (run_dir / RESULTS_NAME).exists() and not args.force:
            print(f"[v19-cf] skip {run_dir} (results exist)", flush=True)
            continue
        results = process_run(run_dir, args)
        print(f"[v19-cf] {results['task']}/{results['arm']}/"
              f"s{results['seed']}: {_spearman_line(results['spearman'])}",
              flush=True)


if __name__ == "__main__":
    main()
