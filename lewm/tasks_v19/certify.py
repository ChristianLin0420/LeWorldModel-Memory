"""Construction-level certificates for the V19 P1a tasks.

Three clauses make a task admissible (V19_PROPOSAL.md 4.4, "certification,
two-level"):

(a) *Integrator at chance*: xi is not predictable from the true initial state
    (endogenous + exogenous), executed action features, and time — the probe
    that dominated every V1-V18 task must fail here by construction.
(b) *Non-re-observability*: a sighted pixel probe restricted to post-cue
    frames is at chance on xi — the cue really is transient.
(c) *Non-vacuousness*: the same pixel probe on cue-window frames reads xi with
    high accuracy — the cue really is there.

All probes are deliberately simple convex models (multinomial logistic
regression / ridge): the certificate asks whether the *information* is
present, not whether a clever model could find it.  No world model is trained
here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lewm.tasks_v19.base import STREAMS, EpisodeBatch, V19Task

# Registered thresholds (binding spec, frozen before any generation).
CHANCE_MARGIN = 0.05           # cat gates: acc <= 1/n + margin
CONT_INTEGRATOR_R2_MAX = 0.10  # t4 integrator gate
CUE_PROBE_ACC_MIN = 0.90       # non-vacuousness gate
SWAP_PROBE_ACC_MIN = 0.80      # t2 swap-visibility gate
MEMORY_DEMAND_MIN = 0.30       # t4: posterior-mean R2 - integrator R2
POSTCUE_OFFSET = 2             # probe frames start this many steps after cue end
N_PROBE_FRAMES = 4             # evenly spaced frames fed to pixel probes
DOWNSAMPLE_SIZE = 24           # pixel probe resolution
RIDGE_ALPHA = 1e-3
PROBE_RANDOM_STATE = 0


def _clause(value: float, threshold: float | None, direction: str) -> dict[str, Any]:
    """One certificate clause; ``threshold=None`` marks report-only telemetry."""
    passed: bool | None = None
    if threshold is not None:
        passed = bool(value <= threshold) if direction == "<=" else \
            bool(value >= threshold) if direction == ">=" else \
            bool(value == threshold)
    return {"value": float(value), "threshold": threshold,
            "direction": direction, "pass": passed}


def _cat_accuracy(train_x: np.ndarray, train_y: np.ndarray,
                  eval_x: np.ndarray, eval_y: np.ndarray) -> float:
    """Multinomial logistic probe accuracy (lbfgs default is multinomial).

    Features are standardized for optimizer conditioning only; a linear probe
    is scale-equivariant so this does not change what is decodable.
    """
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=PROBE_RANDOM_STATE))
    probe.fit(train_x, train_y)
    return float(probe.score(eval_x, eval_y))


def _ridge_r2(train_x: np.ndarray, train_y: np.ndarray,
              eval_x: np.ndarray, eval_y: np.ndarray) -> float:
    probe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    probe.fit(train_x, train_y)
    return float(r2_score(eval_y, probe.predict(eval_x)))


def downsample_frames(frames: np.ndarray, size: int = DOWNSAMPLE_SIZE) -> np.ndarray:
    """Area-mean downsample (..., H, W, 3) uint8 -> (..., size, size, 3) float32.

    Uses integer bin edges + reduceat (no cv2 dependency); area pooling keeps
    small sprites visible at 24x24 instead of aliasing them away.
    """
    height = frames.shape[-3]
    edges = (np.arange(size + 1) * height) // size
    counts = np.diff(edges).astype(np.float32)
    x = frames.astype(np.float32) / 255.0
    x = np.add.reduceat(x, edges[:-1], axis=-3) / counts[:, None, None]
    x = np.add.reduceat(x, edges[:-1], axis=-2) / counts[:, None]
    return x


def _frame_indices(start: np.ndarray, stop: int | np.ndarray,
                   count: int = N_PROBE_FRAMES) -> np.ndarray:
    """(E,) window starts -> (E, count) evenly spaced frame indices in [start, stop)."""
    start = np.asarray(start, dtype=np.float64)
    stop = np.broadcast_to(np.asarray(stop, dtype=np.float64), start.shape)
    return np.round(np.linspace(start, stop - 1, count, axis=-1)).astype(np.int64)


def _pixel_features(batch: EpisodeBatch, start: np.ndarray,
                    stop: int | np.ndarray) -> np.ndarray:
    """Downsampled pixels of N_PROBE_FRAMES frames per episode, flattened."""
    indices = _frame_indices(np.asarray(start), stop)
    selected = batch.frames[np.arange(batch.num_episodes)[:, None], indices]
    return downsample_frames(selected).reshape(batch.num_episodes, -1)


def integrator_features(batch: EpisodeBatch) -> np.ndarray:
    """The registered legal-integrator feature set at decision time.

    [true initial endogenous state, true initial exogenous state, the last 3
    executed actions, the summed action sequence, normalized decision time].
    This is strictly more information than any world model receives about the
    initial condition — which is why chance-level accuracy here certifies that
    xi is not action-integrable.
    """
    num_episodes, length = batch.num_episodes, batch.length
    t_dec = length - 1
    return np.concatenate([
        batch.endo_state[:, 0, :],
        batch.exo_state[:, 0, :],
        batch.actions[:, t_dec - 3:t_dec].reshape(num_episodes, -1),
        batch.actions.sum(axis=1),
        np.full((num_episodes, 1), t_dec / (length - 1), dtype=np.float32),
    ], axis=1)


def _postcue_start(batch: EpisodeBatch) -> np.ndarray:
    """First frame index admissible for the non-re-observability probe."""
    if "shuffle_off" in batch.events:      # shell game: after the last swap
        return batch.events["shuffle_off"] + POSTCUE_OFFSET
    return batch.events["cue_off"] + POSTCUE_OFFSET


def _trace_features(batch: EpisodeBatch) -> np.ndarray:
    return batch.exo_state.reshape(batch.num_episodes, -1)


def _swap_probe_accuracy(task: V19Task, train: EpisodeBatch,
                         evaluation: EpisodeBatch) -> float:
    """Predict which slot pair swapped from the mid-frame of each swap.

    One sample per (episode, swap); certifies the swap motion is visible in
    pixels — without it, "track the cued cup" would be unfalsifiable.
    """
    def samples(batch: EpisodeBatch) -> tuple[np.ndarray, np.ndarray]:
        mid_frames = np.asarray(task.swap_times, dtype=np.int64) + task.swap_frames // 2
        features = downsample_frames(batch.frames[:, mid_frames])
        labels = batch.events["swap_pairs"]
        return (features.reshape(batch.num_episodes * len(mid_frames), -1),
                labels.reshape(-1))

    return _cat_accuracy(*samples(train), *samples(evaluation))


def _certify_stream_cat(task: V19Task, train: EpisodeBatch,
                        evaluation: EpisodeBatch) -> dict[str, Any]:
    chance = 1.0 / task.n_classes
    clauses: dict[str, Any] = {}

    features = integrator_features(train), integrator_features(evaluation)
    clauses["integrator_probe"] = _clause(
        _cat_accuracy(features[0], train.xi, features[1], evaluation.xi),
        chance + CHANCE_MARGIN, "<=")

    clauses["postcue_pixel_probe"] = _clause(
        _cat_accuracy(_pixel_features(train, _postcue_start(train), train.length),
                      train.xi,
                      _pixel_features(evaluation, _postcue_start(evaluation),
                                      evaluation.length),
                      evaluation.xi),
        chance + CHANCE_MARGIN, "<=")

    # Non-vacuousness: for the shell game the cue frames show the *initial*
    # ball slot (the final slot is deliberately at chance given only the cue),
    # so that probe targets ball_slot0 and a separate clause certifies the
    # swap motion visibility.
    if "ball_slot0" in train.events:
        cue_target_train = train.events["ball_slot0"]
        cue_target_eval = evaluation.events["ball_slot0"]
    else:
        cue_target_train, cue_target_eval = train.xi, evaluation.xi
    clauses["cue_pixel_probe"] = _clause(
        _cat_accuracy(_pixel_features(train, train.events["cue_on"],
                                      train.events["cue_off"]),
                      cue_target_train,
                      _pixel_features(evaluation, evaluation.events["cue_on"],
                                      evaluation.events["cue_off"]),
                      cue_target_eval),
        CUE_PROBE_ACC_MIN, ">=")

    if "swap_pairs" in train.events:
        clauses["swap_visibility_probe"] = _clause(
            _swap_probe_accuracy(task, train, evaluation), SWAP_PROBE_ACC_MIN, ">=")

    clauses["trace_sanity_probe"] = _clause(
        _cat_accuracy(_trace_features(train), train.xi,
                      _trace_features(evaluation), evaluation.xi),
        None, "report")
    return clauses


def _certify_stream_cont(task: V19Task, train: EpisodeBatch,
                         evaluation: EpisodeBatch) -> dict[str, Any]:
    clauses: dict[str, Any] = {}
    integrator_r2 = _ridge_r2(integrator_features(train), train.xi,
                              integrator_features(evaluation), evaluation.xi)
    clauses["integrator_probe"] = _clause(integrator_r2,
                                          CONT_INTEGRATOR_R2_MAX, "<=")

    posterior_r2 = float(r2_score(evaluation.xi,
                                  task.posterior_mean_prediction(evaluation)))
    clauses["posterior_mean_r2"] = _clause(posterior_r2, None, "report")
    clauses["memory_demand"] = _clause(posterior_r2 - integrator_r2,
                                       MEMORY_DEMAND_MIN, ">=")

    def pregap_features(batch: EpisodeBatch) -> np.ndarray:
        """Last observable target states: (p_{b-4:b}, v_{b-1})."""
        index = np.arange(batch.num_episodes)[:, None]
        window = batch.events["gap_on"][:, None] + np.arange(-4, 0)[None, :]
        positions = batch.exo_state[index, window, 0:2]
        velocity = batch.exo_state[np.arange(batch.num_episodes),
                                   batch.events["gap_on"] - 1, 2:4]
        return np.concatenate([positions.reshape(batch.num_episodes, -1),
                               velocity], axis=1)

    clauses["pregap_probe_r2"] = _clause(
        _ridge_r2(pregap_features(train), train.xi,
                  pregap_features(evaluation), evaluation.xi),
        None, "report")

    clauses["trace_sanity_probe"] = _clause(
        _ridge_r2(_trace_features(train), train.xi,
                  _trace_features(evaluation), evaluation.xi),
        None, "report")
    return clauses


def _identical_rendering(task: V19Task, seed: int, out_dir: Path,
                         num_episodes: int) -> dict[str, Any]:
    """Exactness check: paired xi branches must render identically post-cue.

    Zero tolerance is achievable — and therefore demanded — because overlays
    are deterministic integer compositing on a shared base scene.
    """
    if task.xi_kind == "cont":
        return {"value": None, "threshold": None, "direction": "==", "pass": None,
                "skipped": True,
                "reason": ("t4: xi is a continuous readout of the shared "
                           "nuisance OU trajectory (no disjoint xi branch), and "
                           "post-gap frames re-show the target by design")}
    branch_a, branch_b = task.paired_branches(num_episodes, seed)
    if not (branch_a.xi != branch_b.xi).all():
        raise AssertionError("paired branches must differ in xi on every episode")
    start = _postcue_start(branch_a) - POSTCUE_OFFSET   # first post-cue frame
    length = branch_a.length
    diff_max = 0
    heatmap = np.zeros((branch_a.frames.shape[2], branch_a.frames.shape[3]),
                       dtype=np.float64)
    frame_count = 0
    for episode in range(num_episodes):
        window_a = branch_a.frames[episode, int(start[episode]):length].astype(np.int16)
        window_b = branch_b.frames[episode, int(start[episode]):length].astype(np.int16)
        diff = np.abs(window_a - window_b)
        diff_max = max(diff_max, int(diff.max()))
        heatmap += diff.mean(axis=-1).sum(axis=0)
        frame_count += window_a.shape[0]

    figure = Figure(figsize=(4.2, 3.6))
    axis = figure.subplots()
    image = axis.imshow(heatmap / max(frame_count, 1), cmap="magma")
    axis.set_title(f"{task.name}: paired post-cue |diff| (max={diff_max})")
    figure.colorbar(image, ax=axis)
    figure.savefig(out_dir / f"identical_rendering_{task.name}.png", dpi=120,
                   bbox_inches="tight")
    return {**_clause(float(diff_max), 0.0, "=="), "skipped": False,
            "episodes": num_episodes}


def run_certificates(task: V19Task, seed: int, out_dir: str | Path,
                     e_train: int = 512, e_eval: int = 256,
                     paired_episodes: int = 32) -> dict[str, Any]:
    """Run every construction-level certificate for one task and seed.

    Banks get derived seeds (seed*1000+k) so streams and splits never share
    randomness.  The certificate JSON is written to out_dir/certificate.json;
    ``overall_pass`` is the AND of every thresholded clause over both streams
    plus the identical-rendering check.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    certificate: dict[str, Any] = {
        "task": task.name, "seed": seed, "e_train": e_train, "e_eval": e_eval,
        "streams": {},
    }
    for offset, stream in enumerate(STREAMS):
        train = task.generate(stream, e_train, seed * 1000 + 2 * offset + 1)
        evaluation = task.generate(stream, e_eval, seed * 1000 + 2 * offset + 2)
        if task.xi_kind == "cat":
            certificate["streams"][stream] = _certify_stream_cat(task, train,
                                                                 evaluation)
        else:
            certificate["streams"][stream] = _certify_stream_cont(task, train,
                                                                  evaluation)
    certificate["identical_rendering"] = _identical_rendering(
        task, seed * 1000 + 5, out_dir, paired_episodes)

    gates = [clause["pass"]
             for stream_clauses in certificate["streams"].values()
             for clause in stream_clauses.values() if clause["pass"] is not None]
    if certificate["identical_rendering"]["pass"] is not None:
        gates.append(certificate["identical_rendering"]["pass"])
    certificate["overall_pass"] = bool(all(gates))

    (out_dir / "certificate.json").write_text(
        json.dumps(certificate, indent=2, sort_keys=True))
    return certificate
