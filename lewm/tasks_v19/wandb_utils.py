"""Weights & Biases reporting for the V19 P1a certificates.

Everything here is diagnostics: a wandb outage must never invalidate a
certificate run, so every public function is guarded and degrades to a
warning.  Video annotation happens on *copies* of the frames — the certified
banks never carry any xi-revealing decoration.
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from lewm.tasks_v19.base import EpisodeBatch
from lewm.tasks_v19.overlays import CUE_COLORS

_GAP_BORDER = (128, 128, 128)
_BORDER_PX = 2
_LEGEND_BOX = (2, 52, 12, 62)  # x0, y0, x1, y1 of the xi legend patch


def _guarded(fn: Callable) -> Callable:
    """Never let reporting failures propagate into the certificate flow."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - reporting is best-effort
            warnings.warn(f"wandb logging step {fn.__name__} failed: {error!r}",
                          stacklevel=2)
            return None
    return wrapper


def _figure_to_image(figure: Figure) -> np.ndarray:
    """Rasterize an OO-API figure without touching the pyplot global state."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _xi_color(batch: EpisodeBatch, episode: int) -> tuple[int, int, int]:
    if batch.xi_kind == "cat":
        return CUE_COLORS[int(batch.xi[episode]) % len(CUE_COLORS)]
    red, green = ((batch.xi[episode] + 1.0) * 0.5 * 255.0).astype(np.uint8)
    return int(red), int(green), 80


def _paint_border(frame: np.ndarray, color: tuple[int, int, int]) -> None:
    value = np.asarray(color, dtype=np.uint8)
    frame[:_BORDER_PX] = value
    frame[-_BORDER_PX:] = value
    frame[:, :_BORDER_PX] = value
    frame[:, -_BORDER_PX:] = value


def _annotate_episode(batch: EpisodeBatch, episode: int) -> np.ndarray:
    """Annotated copy: cue-window border, t4 gap border, xi legend patch."""
    frames = batch.frames[episode].copy()
    events = batch.events
    xi_color = _xi_color(batch, episode)
    for t, frame in enumerate(frames):
        if "cue_on" in events and events["cue_on"][episode] <= t < events["cue_off"][episode]:
            _paint_border(frame, xi_color)
        if "gap_on" in events and events["gap_on"][episode] <= t < events["gap_off"][episode]:
            _paint_border(frame, _GAP_BORDER)
        x0, y0, x1, y1 = _LEGEND_BOX
        frame[y0:y1, x0:x1] = np.asarray(xi_color, dtype=np.uint8)
    return frames


@_guarded
def log_certificates(run: Any, cert: dict) -> None:
    """Log the certificate as a clause table plus a value-vs-threshold figure."""
    import wandb

    rows: list[list[Any]] = []
    for stream, clauses in cert["streams"].items():
        for name, clause in clauses.items():
            rows.append([stream, name, clause["value"], clause["threshold"],
                         clause["direction"], clause["pass"]])
    rendering = cert["identical_rendering"]
    rows.append(["paired", "identical_rendering", rendering["value"],
                 rendering["threshold"], rendering["direction"], rendering["pass"]])
    columns = ["stream", "clause", "value", "threshold", "direction", "pass"]
    run.log({"certificates/table": wandb.Table(columns=columns, data=rows)})

    gated = [row for row in rows if row[5] is not None]
    figure = Figure(figsize=(7.0, 0.6 * len(gated) + 1.2))
    axis = figure.subplots()
    labels = [f"{stream}/{name}" for stream, name, *_ in gated]
    values = [row[2] for row in gated]
    colors = ["#2e9e4f" if row[5] else "#c8402e" for row in gated]
    positions = np.arange(len(gated))
    axis.barh(positions, values, color=colors)
    for position, row in zip(positions, gated):
        axis.plot([row[3], row[3]], [position - 0.4, position + 0.4], "k--", lw=1.2)
        axis.text(row[2], position, f" {row[2]:.3f} ({row[4]} {row[3]:.2f})",
                  va="center", fontsize=8)
    axis.set_yticks(positions, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_title(f"{cert['task']} certificates — overall "
                   f"{'PASS' if cert['overall_pass'] else 'FAIL'}")
    run.log({"certificates/summary": wandb.Image(_figure_to_image(figure))})


@_guarded
def log_rollout_video(run: Any, batch: EpisodeBatch, n: int = 3,
                      key: str = "rollouts/video") -> None:
    """Tile ``n`` annotated episodes side by side into one (T, C, H, W) video."""
    import wandb

    n = min(n, batch.num_episodes)
    tiled = np.concatenate([_annotate_episode(batch, episode)
                            for episode in range(n)], axis=2)   # (T, H, n*W, 3)
    run.log({key: wandb.Video(tiled.transpose(0, 3, 1, 2), fps=10, format="mp4")})


def _figure_transient(batch: EpisodeBatch) -> Figure:
    """Cue timing histogram + cue vs post-cue frame grid (t1/t1dev/t3)."""
    figure = Figure(figsize=(10, 6))
    grid = figure.add_gridspec(2, 4)
    axis = figure.add_subplot(grid[0, :2])
    axis.hist(batch.events["cue_on"], bins=np.arange(0, 25) - 0.5, alpha=0.7,
              label="cue_on")
    axis.hist(batch.events["cue_off"] - batch.events["cue_on"],
              bins=np.arange(0, 10) - 0.5, alpha=0.7, label="duration")
    axis.set_title("cue window draws (xi-independent)")
    axis.legend()
    for column, episode in enumerate(range(min(2, batch.num_episodes))):
        cue_mid = int((batch.events["cue_on"][episode]
                       + batch.events["cue_off"][episode]) // 2)
        post = int(batch.events["cue_off"][episode]) + 2
        for row, (t, label) in enumerate(((cue_mid, "cue"), (post, "post-cue"))):
            cell = figure.add_subplot(grid[row, 2 + column])
            cell.imshow(batch.frames[episode, t])
            cell.set_title(f"ep{episode} t={t} ({label}) xi={batch.xi[episode]}",
                           fontsize=8)
            cell.axis("off")
    return figure


def _figure_shell(batch: EpisodeBatch) -> Figure:
    """Cup slot trajectories with the cued cup highlighted (t2/t2dev)."""
    episodes = min(5, batch.num_episodes)
    figure = Figure(figsize=(10, 2.0 * episodes))
    axes = figure.subplots(episodes, 1, sharex=True, squeeze=False)[:, 0]
    time = np.arange(batch.length)
    for episode, axis in enumerate(axes):
        for entity in range(3):
            axis.plot(time, batch.exo_state[episode, :, 1 + entity], lw=1.0,
                      color="gray")
        axis.plot(time, batch.exo_state[episode, :, 0], lw=2.0, color="crimson",
                  label="cued cup")
        axis.axvspan(batch.events["cue_on"][episode],
                     batch.events["cue_off"][episode], color="gold", alpha=0.3)
        axis.set_ylabel(f"ep{episode} xi={batch.xi[episode]}", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("cup x-positions over time (cue phase shaded)")
    return figure


def _figure_freeze(task: Any, batch: EpisodeBatch) -> Figure:
    """OU fan + dispersion-vs-time + posterior-mean scatter (t4)."""
    figure = Figure(figsize=(13, 4))
    fan, dispersion, scatter = figure.subplots(1, 3)
    episodes = min(20, batch.num_episodes)
    for episode in range(episodes):
        fan.plot(batch.exo_state[episode, :, 0], batch.exo_state[episode, :, 1],
                 lw=0.8, alpha=0.5)
    fan.set_title(f"target trajectories ({episodes} eps)")
    fan.set_aspect("equal")

    dispersion.plot(np.arange(batch.length),
                    batch.exo_state[:, :, 0:2].std(axis=0).mean(axis=-1))
    dispersion.axvspan(batch.events["gap_on"].min(), batch.events["gap_off"].max(),
                       color="gray", alpha=0.3, label="gap window range")
    dispersion.set_title("cross-episode position dispersion vs time")
    dispersion.legend(fontsize=8)

    prediction = task.posterior_mean_prediction(batch)
    for dim, label in enumerate("xy"):
        scatter.scatter(batch.xi[:, dim], prediction[:, dim], s=12, alpha=0.6,
                        label=label)
    scatter.plot([-1, 1], [-1, 1], "k--", lw=1)
    scatter.set_xlabel("true xi (normalized)")
    scatter.set_ylabel("posterior-mean prediction")
    scatter.set_title("closed-form posterior mean vs truth")
    scatter.legend(fontsize=8)
    return figure


@_guarded
def log_task_figures(run: Any, task: Any, batch: EpisodeBatch,
                     key: str = "figures/task") -> None:
    """Log the per-task analytical figure for one generated batch."""
    import wandb

    if task.name in ("t1", "t1dev", "t3"):
        figure = _figure_transient(batch)
    elif task.name in ("t2", "t2dev"):
        figure = _figure_shell(batch)
    else:
        figure = _figure_freeze(task, batch)
    run.log({key: wandb.Image(_figure_to_image(figure))})
