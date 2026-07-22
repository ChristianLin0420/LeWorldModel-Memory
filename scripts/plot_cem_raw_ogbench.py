#!/usr/bin/env python3
"""Generate raw OGBench CEM figures and paper tables from exact results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/cem_raw_ogbench"
DOC_ASSETS = ROOT / "docs/assets"
PAPER_FIGURES = ROOT / "paper_d/figures"
PAPER_RESULTS = ROOT / "paper_d/generated_results"
CACHE_ROOT = (
    ROOT / "outputs/multiview_patchset_color_jepa_native_v1/cache"
)
INK = "#171717"
CREAM = "#f7f4ea"
YELLOW = "#d19e00"
GREEN = "#477a55"
RED = "#9a3f38"
MUTED = "#77736b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.facecolor": CREAM,
        "figure.facecolor": CREAM,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    for directory in (DOC_ASSETS, PAPER_FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            directory / f"{name}.pdf",
            bbox_inches="tight",
            facecolor=CREAM,
        )
        fig.savefig(
            directory / f"{name}.png",
            bbox_inches="tight",
            facecolor=CREAM,
            dpi=220,
        )
    plt.close(fig)


def load_cells(output: Path) -> list[dict[str, Any]]:
    cells = []
    for path in sorted((output / "cells").glob("*/*/result.json")):
        cell = json.loads(path.read_text())
        cell["_path"] = path
        cells.append(cell)
    if not cells:
        raise RuntimeError(f"no completed raw CEM cells under {output}")
    return cells


def select_timeline(
    cells: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select the retrieved event with largest model proposal score."""
    candidates = []
    for cell in cells:
        log_path = ROOT / cell["artifacts"]["decision_log"]
        decision = json.loads(log_path.read_text())
        stream_ids = {
            int(item["episode_id"])
            for item in decision["episode_streams"]
        }
        for query in decision["queries"]:
            if int(query["episode_id"]) not in stream_ids:
                continue
            for event in query["events"]:
                if event["retrieved"]:
                    candidates.append((
                        float(event["proposal_score"]),
                        cell,
                        decision,
                        query,
                        event,
                    ))
    if not candidates:
        raise RuntimeError("no retrieved raw event exists for timeline")
    _, cell, decision, query, event = max(
        candidates, key=lambda item: item[0]
    )
    return cell, decision, {"query": query, "event": event}


def event_timeline(
    cells: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    cell, decision, selection = select_timeline(cells)
    query = selection["query"]
    event = selection["event"]
    environment = cell["environment"]
    episode = int(query["episode_id"])
    query_t = int(query["query_t"])
    event_t = int(event["event_timestamp"])
    frame_indices = sorted(set([
        max(0, event_t - 1),
        event_t,
        min(query_t, event_t + 1),
        query_t,
    ]))
    cache_path = CACHE_ROOT / environment / "render_cache.npz"
    with np.load(cache_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"][episode, frame_indices])
    stream = next(
        item for item in decision["episode_streams"]
        if int(item["episode_id"]) == episode
    )
    surprise = np.asarray([
        np.nan if value is None else value
        for value in stream["host_surprise"]
    ], dtype=np.float64)
    fig = plt.figure(figsize=(7.1, 3.15))
    grid = fig.add_gridspec(2, len(frame_indices), height_ratios=[1.35, 1])
    for column, (time_index, frame) in enumerate(zip(frame_indices, frames)):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(frame)
        axis.set_title(
            (
                f"event discovered · t={time_index}"
                if time_index == event_t
                else (
                    f"prediction query · t={time_index}"
                    if time_index == query_t
                    else f"raw frame · t={time_index}"
                )
            ),
            color=YELLOW if time_index == event_t else INK,
            fontsize=7,
        )
        axis.axis("off")
    axis = fig.add_subplot(grid[1, :])
    time_axis = np.arange(len(surprise))
    axis.plot(time_axis, surprise, color=INK, lw=1.5, label="host surprise")
    status_by_id = {
        int(item["event_id"]): item["status"]
        for item in query["lifecycle"]
    }
    markers = {
        "active": ("o", GREEN, "promoted"),
        "rejected": ("x", RED, "rejected"),
        "superseded": ("s", MUTED, "superseded"),
        "evicted": ("v", MUTED, "evicted"),
        "provisional": ("^", YELLOW, "provisional"),
    }
    shown = set()
    for item in query["events"]:
        status = status_by_id.get(int(item["event_id"]), "provisional")
        marker, color, label = markers[status]
        time_index = int(item["event_timestamp"])
        axis.scatter(
            time_index,
            surprise[time_index],
            marker=marker,
            color=color,
            s=35,
            label=label if label not in shown else None,
            zorder=3,
        )
        shown.add(label)
        if item["retrieved"]:
            axis.scatter(
                time_index,
                surprise[time_index],
                facecolors="none",
                edgecolors=YELLOW,
                linewidths=2,
                s=90,
                label="retrieved" if "retrieved" not in shown else None,
            )
            shown.add("retrieved")
    axis.axvline(query_t, color=MUTED, ls="--", lw=1, label="query")
    axis.set_xlabel("trajectory timestamp")
    axis.set_ylabel("future-latent one-step MSE")
    axis.set_title(
        f"{environment} · automatic raw event lifecycle",
        loc="left",
        fontweight="bold",
    )
    axis.legend(ncol=4, fontsize=6.5)
    fig.tight_layout(pad=0.6)
    save(fig, "cem_raw_event_timeline")
    frame_hashes = [
        hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames
    ]
    receipt = {
        "schema": "cem_raw_ogbench_figure_receipt",
        "selection_rule": (
            "largest proposal score among events retrieved by completed "
            "test-time CEM logs; no frame or timestamp selected manually"
        ),
        "environment": environment,
        "seed": cell["seed"],
        "episode_id": episode,
        "event_timestamp": event_t,
        "query_timestamp": query_t,
        "frame_indices": frame_indices,
        "raw_cache": str(cache_path.relative_to(ROOT)),
        "raw_cache_sha256": sha256_file(cache_path),
        "frame_sha256": frame_hashes,
        "frames_modified": False,
        "cue_window": None,
        "cue_window_used_by_model": False,
        "decision_log": cell["artifacts"]["decision_log"],
        "figure": "docs/assets/cem_raw_event_timeline.pdf",
    }
    (output / "figure_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    (PAPER_RESULTS / "raw_architecture_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def architecture_figure(receipt: dict[str, Any]) -> None:
    cache_path = ROOT / receipt["raw_cache"]
    with np.load(cache_path, allow_pickle=False) as data:
        frames = np.asarray(data[
            "frames"
        ][receipt["episode_id"], receipt["frame_indices"]])
    figure = plt.figure(figsize=(7.1, 4.0))
    grid = figure.add_gridspec(
        2,
        len(frames),
        height_ratios=[1.45, 1.0],
        hspace=.38,
    )
    for column, (time_index, frame) in enumerate(zip(
        receipt["frame_indices"], frames
    )):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(frame)
        if time_index == receipt["event_timestamp"]:
            title = f"event discovered\nt={time_index}"
            color = YELLOW
        elif time_index == receipt["query_timestamp"]:
            title = f"prediction query\nt={time_index}"
            color = INK
        else:
            title = f"unmodified frame\nt={time_index}"
            color = INK
        axis.set_title(title, color=color, fontweight="bold", fontsize=7.2)
        axis.axis("off")
    axis = figure.add_subplot(grid[1, :])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (.01, .33, .14, .36, "Frozen DINOv2\npatch tokens"),
        (.18, .33, .14, .36, "Host surprise\nWRITE proposal"),
        (.35, .33, .14, .36, "Delayed group\nCE verification"),
        (.52, .33, .14, .36, "Versioned store\n+ hysteresis"),
        (.69, .33, .14, .36, "Query/content/\nage/need router"),
        (.86, .33, .13, .36, "Frozen host\nfuture prediction"),
    ]
    for index, (x, y, width, height, label) in enumerate(boxes):
        color = YELLOW if index in (1, 2) else (
            GREEN if index in (3, 4) else CREAM
        )
        axis.add_patch(plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor=INK,
            linewidth=1.0,
        ))
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold" if index in (1, 2, 3) else "normal",
        )
        if index < len(boxes) - 1:
            axis.annotate(
                "",
                xy=(boxes[index + 1][0], y + height / 2),
                xytext=(x + width, y + height / 2),
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2},
            )
    axis.text(
        .5,
        .12,
        r"$\mathrm{CE}(G)=L_{\mathrm{future}}(M\setminus G)"
        r"-L_{\mathrm{future}}(M)$"
        "  ·  event times and labels are not supplied",
        ha="center",
        va="center",
        fontsize=7.2,
    )
    figure.suptitle(
        "CEM on an unmodified OGBench trajectory",
        x=.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    figure.text(
        .99,
        .975,
        f"{receipt['environment']} · seed {receipt['seed']} · "
        "model-selected event",
        ha="right",
        va="top",
        fontsize=6.8,
        color=MUTED,
    )
    for directory in (
        PAPER_FIGURES,
        DOC_ASSETS,
        ROOT / "docs/figures",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png", "svg"):
            figure.savefig(
                directory / f"fig_d_architecture.{suffix}",
                bbox_inches="tight",
                facecolor=CREAM,
                dpi=220,
            )
    plt.close(figure)


def rollout_horizon(cells: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["environment"], []).append(cell)
    fig, axis = plt.subplots(figsize=(7.1, 2.6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
    for color, (environment, rows) in zip(colors, sorted(grouped.items())):
        memory = np.asarray([
            row["test"]["controls"]["memory"]["horizon_mse"]
            for row in rows
        ])
        baseline = np.asarray([
            row["test"]["controls"]["no_memory"]["horizon_mse"]
            for row in rows
        ])
        improvement = ((baseline - memory) / np.maximum(baseline, 1e-12))
        mean = improvement.mean(0) * 100
        std = improvement.std(0, ddof=1) * 100 if len(rows) > 1 else np.zeros_like(mean)
        x = np.arange(1, len(mean) + 1)
        axis.plot(x, mean, marker="o", ms=3, lw=1.5, color=color, label=environment)
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=.12)
    axis.axhline(0, color=INK, lw=.8)
    axis.set_xlabel("rollout horizon (steps)")
    axis.set_ylabel("MSE reduction vs no memory (%)")
    axis.set_title(
        "Raw OGBench future-latent prediction",
        loc="left",
        fontweight="bold",
    )
    axis.legend(fontsize=5.8, ncol=2)
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_rollout_horizon")


def causal_deletion(report: dict[str, Any]) -> None:
    rows = report["environments"]
    labels = [row["environment"].replace("-navigate-v0", "").replace("-play-v0", "") for row in rows]
    high = [row["high_ce_deletion"]["mean"] or 0.0 for row in rows]
    random = [row["random_deletion"]["mean"] or 0.0 for row in rows]
    x = np.arange(len(rows))
    width = .38
    fig, axis = plt.subplots(figsize=(7.1, 2.5))
    axis.bar(x - width / 2, high, width, color=GREEN, edgecolor=INK, lw=.5, label="delete predicted-high CE group")
    axis.bar(x + width / 2, random, width, color=MUTED, edgecolor=INK, lw=.5, label="delete matched random group")
    axis.axhline(0, color=INK, lw=.7)
    axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=6.5)
    axis.set_ylabel("change in future-latent MSE")
    axis.set_title("Causal deletion on automatic raw event groups", loc="left", fontweight="bold")
    axis.legend(fontsize=6.5)
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_causal_deletion")


def budget_pareto(cells: list[dict[str, Any]]) -> None:
    points: dict[int, list[tuple[float, float]]] = {}
    for cell in cells:
        baseline = cell["test"]["controls"]["no_memory"]["mse"]
        for row in cell["test"]["budget_curve"]:
            improvement = (baseline - row["mse"]) / max(baseline, 1e-12)
            points.setdefault(int(row["budget"]), []).append(
                (float(row["retrieval_rate"]), float(improvement))
            )
    fig, axis = plt.subplots(figsize=(4.5, 2.55))
    for budget, values in sorted(points.items()):
        array = np.asarray(values)
        axis.scatter(
            array[:, 0] * 100,
            array[:, 1] * 100,
            s=16,
            color=MUTED,
            alpha=.35,
        )
        mean = array.mean(0)
        axis.scatter(
            mean[0] * 100,
            mean[1] * 100,
            s=55,
            color=GREEN,
            edgecolor=INK,
            zorder=3,
        )
        axis.annotate(
            f"B={budget}",
            (mean[0] * 100, mean[1] * 100),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axhline(0, color=INK, lw=.7)
    axis.set_xlabel("queries with retrieved memory (%)")
    axis.set_ylabel("future-latent MSE reduction (%)")
    axis.set_title(
        "Write/retrieval budget Pareto",
        loc="left",
        fontweight="bold",
    )
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_budget_pareto")


def family_aggregate(report: dict[str, Any]) -> None:
    families = report["families"]
    labels = [row["family"] for row in families]
    improvement = [
        (row["memory_relative_improvement"]["mean"] or 0.0) * 100
        for row in families
    ]
    deletion = [
        row["deletion_gap"]["mean"] or 0.0 for row in families
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    x = np.arange(len(labels))
    axes[0].bar(
        x,
        improvement,
        color=[GREEN if value > 0 else RED for value in improvement],
        edgecolor=INK,
        lw=.5,
    )
    axes[0].axhline(0, color=INK, lw=.7)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("MSE reduction vs no memory (%)")
    axes[0].set_title(
        "DINO-feature breadth · raw renderings",
        loc="left",
        fontweight="bold",
    )
    axes[1].bar(
        x,
        deletion,
        color=YELLOW,
        edgecolor=INK,
        lw=.5,
    )
    axes[1].axhline(0, color=INK, lw=.7)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("high-minus-random deletion ΔMSE")
    axes[1].set_title(
        "Causal ordering",
        loc="left",
        fontweight="bold",
    )
    fig.text(
        .5,
        -.02,
        "Breadth host is DINO-WM-style, not official DINO-WM. "
        "The official Wall result remains a separate protocol.",
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_family_aggregate")


def latex_escape(value: str) -> str:
    return (
        value.replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def generated_tables(
    report: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    host_rows = []
    for row in report["environments"]:
        name = (
            row["environment"]
            .replace("-navigate-v0", "")
            .replace("-play-v0", "")
        )
        improvement = (row["memory_relative_improvement"]["mean"] or 0.0) * 100
        rows.append(
            f"{latex_escape(name)} & {row['seed_count']} & "
            f"{row['memory_mse']['mean']:.4f}/"
            f"{row['no_memory_mse']['mean']:.4f} & "
            f"{improvement:+.2f}\\% & "
            f"{(row['high_ce_deletion']['mean'] or 0.0):.4f}/"
            f"{(row['random_deletion']['mean'] or 0.0):.4f} & "
            f"{(row['ce_spearman']['mean'] or 0.0):.3f} \\\\"
        )
        host_rows.append(
            f"{latex_escape(name)} & {row['seed_count']} & "
            f"{row['host_test_mse']['mean']:.4f} & "
            f"{row['host_vs_persistence_ratio']['mean']:.3f} & "
            f"{row['reliable_host_count']}/{row['seed_count']} \\\\"
        )
    (PAPER_RESULTS / "raw_ogbench_main.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n"
    )
    (PAPER_RESULTS / "raw_ogbench_hosts.tex").write_text(
        "\n".join(host_rows) + "\n\\bottomrule\n"
    )
    snapshot = {
        "schema": "paper_d_raw_ogbench_snapshot",
        "source_report": "outputs/cem_raw_ogbench/report.json",
        "source_report_sha256": sha256_file(
            DEFAULT_OUTPUT / "report.json"
        ) if (DEFAULT_OUTPUT / "report.json").is_file() else None,
        "report": report,
        "architecture_receipt": receipt,
    }
    (PAPER_RESULTS / "raw_ogbench_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text())
    cells = load_cells(args.output)
    style()
    receipt = event_timeline(cells, args.output)
    architecture_figure(receipt)
    rollout_horizon(cells)
    causal_deletion(report)
    budget_pareto(cells)
    family_aggregate(report)
    generated_tables(report, receipt)
    print(json.dumps({
        "status": "completed",
        "figure_count": 5,
        "report": str(report_path.relative_to(ROOT)),
        "paper_snapshot": (
            "paper_d/generated_results/raw_ogbench_snapshot.json"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
