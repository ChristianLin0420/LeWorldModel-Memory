#!/usr/bin/env python3
"""Diagnose weak non-manual random-patch-set JEPA rows.

The analysis is post-hoc.  It may use labels and the known audit cue window to
explain failures, but it does not modify training or supply labels to models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402
from scripts import run_random_patchset_view_jepa_ogbench as view  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "random_patchset_view_color_jepa_diagnosis_v1"
DEFAULT_FIGURE = ROOT / "docs" / "assets" / "random_patchset_weak_diagnosis.svg"
SUMMARY_PATHS = (
    ROOT / "outputs" / "random_patchset_view_color_jepa_breadth_confirm_v1" / "summary.json",
    ROOT / "outputs" / "random_patchset_view_color_jepa_extra_v1" / "antmaze-large-navigate-v0" / "summary.json",
    ROOT / "outputs" / "random_patchset_view_color_jepa_extra_v1" / "antmaze-giant-navigate-v0" / "summary.json",
    ROOT / "outputs" / "random_patchset_view_color_jepa_extra_v1" / "humanoidmaze-large-navigate-v0" / "summary.json",
)
CACHE_PATHS = {
    "scene-play-v0": ROOT / "outputs" / "random_patchset_view_color_jepa_breadth_confirm_v1" / "cache" / "scene-play-v0" / "render_cache.npz",
    "puzzle-3x3-play-v0": ROOT / "outputs" / "random_patchset_view_color_jepa_breadth_confirm_v1" / "cache" / "puzzle-3x3-play-v0" / "render_cache.npz",
    "humanoidmaze-large-navigate-v0": ROOT / "outputs" / "random_patchset_view_color_jepa_extra_v1" / "humanoidmaze-large-navigate-v0" / "cache" / "humanoidmaze-large-navigate-v0" / "render_cache.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--episodes", type=int, default=160)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in SUMMARY_PATHS:
        if not path.exists():
            continue
        data = read_json(path)
        for row in data.get("rows", []):
            row = dict(row)
            row["source"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def selection_stats(cache_path: Path, *, age: int, episodes: int) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as data:
        frames = data["frames"]
        labels = data["cue_labels"]
        positions = data["cue_positions"]
    endpoint = int(base.LAST_CUE_FRAME + age)
    limit = min(int(episodes), int(len(frames)))
    target_times: list[int] = []
    first_selected: list[tuple[int, int, int]] = []
    cue_hits = 0
    for episode in range(limit):
        rng = np.random.default_rng(40_000_019 + int(episode) + 397 * int(age))
        full = base.inject_cue_sequence(frames[episode].copy(), int(labels[episode]), int(positions[episode]))
        target_time = view.choose_target_time(full, endpoint=endpoint, rng=rng, variant="full")
        _, selected = view.mine_single_view_patches(full, target_time=target_time, rng=rng)
        target_times.append(int(target_time))
        first_selected.extend(selected[:1])
        if 1 <= int(target_time) <= int(base.LAST_CUE_FRAME):
            cue_hits += 1
    return {
        "age": int(age),
        "episodes": int(limit),
        "cue_window_fraction": float(cue_hits / max(1, limit)),
        "mean_target_time": float(np.mean(target_times)) if target_times else 0.0,
        "std_target_time": float(np.std(target_times)) if target_times else 0.0,
        "target_time_histogram": {
            str(int(t)): int(c)
            for t, c in zip(*np.unique(np.asarray(target_times, dtype=np.int64), return_counts=True))
        },
    }


def overlay_example(ax: plt.Axes, cache_path: Path, *, env_name: str, age: int, episode: int = 0) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as data:
        frames = data["frames"]
        labels = data["cue_labels"]
        positions = data["cue_positions"]
    endpoint = int(base.LAST_CUE_FRAME + age)
    rng = np.random.default_rng(40_000_019 + int(episode) + 397 * int(age))
    full = base.inject_cue_sequence(frames[episode].copy(), int(labels[episode]), int(positions[episode]))
    target_time = view.choose_target_time(full, endpoint=endpoint, rng=rng, variant="full")
    _, selected = view.mine_single_view_patches(full, target_time=target_time, rng=rng)
    ax.imshow(full[int(target_time)])
    for _, y, x in selected:
        ax.add_patch(Rectangle((x, y), 16, 16, fill=False, linewidth=1.2, edgecolor="#fbd45b"))
    ax.set_title(f"{env_name.replace('-play-v0', '').replace('-navigate-v0', '')}\nage {age}, selected t={target_time}", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    return {
        "env_name": env_name,
        "age": int(age),
        "episode": int(episode),
        "label": int(labels[episode]),
        "position": int(positions[episode]),
        "target_time": int(target_time),
        "selected_patches": [{"time": int(t), "y": int(y), "x": int(x)} for t, y, x in selected],
        "cue_window_hit": bool(1 <= int(target_time) <= int(base.LAST_CUE_FRAME)),
    }


def make_figure(rows: list[dict[str, Any]], stats: dict[str, dict[str, Any]], examples: list[dict[str, Any]], figure: Path) -> None:
    plot_rows = [row for row in rows if row["env_name"] in {
        "scene-play-v0",
        "puzzle-3x3-play-v0",
        "humanoidmaze-large-navigate-v0",
        "antmaze-large-navigate-v0",
        "antmaze-giant-navigate-v0",
    }]
    plot_rows = sorted(plot_rows, key=lambda r: (r["env_name"], int(r["age"])))
    labels = [f"{r['env_name'].replace('-navigate-v0','').replace('-play-v0','')}\nage {r['age']}" for r in plot_rows]
    full = np.asarray([float(r["full_bacc_mean"]) for r in plot_rows], dtype=np.float64)
    controls = np.asarray([max(float(r["reset_bacc_mean"]), float(r["no_state_bacc_mean"])) for r in plot_rows], dtype=np.float64)
    top1 = np.asarray([float(r.get("retrieval_top1_mean", 0.0)) for r in plot_rows], dtype=np.float64)
    status = np.asarray([1.0 if r.get("all_pass") else 0.0 for r in plot_rows], dtype=np.float64)
    x = np.arange(len(plot_rows))

    fig = plt.figure(figsize=(12.8, 7.0), dpi=180)
    fig.patch.set_facecolor("#fbfbf9")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.1], hspace=0.55, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    overlay_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    for ax in [ax0, ax1, ax2, ax3, *overlay_axes]:
        ax.set_facecolor("#fbfbf9")
        for spine in ax.spines.values():
            spine.set_color("#333b49")
            spine.set_linewidth(0.7)
    ax0.bar(x - 0.18, full, width=0.36, color="#111827", label="full memory")
    ax0.bar(x + 0.18, controls, width=0.36, color="#d4d3cb", edgecolor="#333b49", linewidth=0.4, label="max control")
    for i, ok in enumerate(status):
        ax0.scatter(i, 1.03, s=26, marker="s", color="#fbd45b" if ok else "#7f1d1d", clip_on=False)
    ax0.axhline(0.75, color="#d8a900", linestyle="--", linewidth=0.8)
    ax0.axhline(0.35, color="#9ca3af", linestyle=":", linewidth=0.8)
    ax0.set_ylim(0, 1.08)
    ax0.set_ylabel("BAcc", fontsize=8)
    ax0.set_title("Weak-case diagnosis: full memory separates from controls, but some rows miss the full-memory gate", fontsize=10, loc="left", fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax0.legend(frameon=False, fontsize=7, ncols=2, loc="upper left")
    ax0.grid(axis="y", color="#d4d3cb", linewidth=0.6)

    ax1.scatter(top1, full, s=42, c=np.where(status > 0, "#111827", "#fbd45b"), edgecolors="#111827", linewidths=0.5)
    ax1.axhline(0.75, color="#d8a900", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("JEPA retrieval top-1", fontsize=8)
    ax1.set_ylabel("full-memory BAcc", fontsize=8)
    ax1.set_title("Retrieval/readout coupling", fontsize=9, loc="left", fontweight="bold")
    ax1.grid(color="#d4d3cb", linewidth=0.6)

    stat_rows = []
    for env_name, env_stats in stats.items():
        for age, item in env_stats.items():
            stat_rows.append((env_name, int(age), float(item["cue_window_fraction"])))
    stat_rows = sorted(stat_rows, key=lambda x: (x[0], x[1]))
    sx = np.arange(len(stat_rows))
    ax2.bar(sx, [r[2] for r in stat_rows], color="#fbd45b", edgecolor="#111827", linewidth=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("How often saliency picks cue-window views", fontsize=9, loc="left", fontweight="bold")
    ax2.set_ylabel("fraction", fontsize=8)
    ax2.set_xticks(sx)
    ax2.set_xticklabels([f"{r[0].split('-')[0]}\n{r[1]}" for r in stat_rows], fontsize=6.5)
    ax2.grid(axis="y", color="#d4d3cb", linewidth=0.6)

    gap = full - controls
    ax3.barh(np.arange(len(plot_rows)), gap, color=np.where(status > 0, "#111827", "#fbd45b"), edgecolor="#111827", linewidth=0.3)
    ax3.axvline(0.40, color="#d8a900", linestyle="--", linewidth=0.8)
    ax3.set_yticks(np.arange(len(plot_rows)))
    ax3.set_yticklabels(labels, fontsize=6.2)
    ax3.set_xlabel("full - max control", fontsize=8)
    ax3.set_title("Memory gap", fontsize=9, loc="left", fontweight="bold")
    ax3.grid(axis="x", color="#d4d3cb", linewidth=0.6)

    for ax, example in zip(overlay_axes, examples):
        overlay_example(ax, CACHE_PATHS[example["env_name"]], env_name=example["env_name"], age=example["age"])
    fig.text(
        0.01,
        0.01,
        "Diagnostics are post-hoc: labels/cue-window knowledge are used only to explain failure modes, not for training.",
        fontsize=7,
        color="#656760",
    )
    figure.parent.mkdir(parents=True, exist_ok=True)
    suffix = figure.suffix.lower().lstrip(".") or "svg"
    fig.savefig(figure, format=suffix, bbox_inches="tight")


def update_html(figure: Path, diagnosis_json: Path) -> None:
    html = ROOT / "docs" / "mesm_nvidia_plan.html"
    if not html.exists():
        return
    start = "<!-- NONMANUAL_WEAK_DIAGNOSIS_START -->"
    end = "<!-- NONMANUAL_WEAK_DIAGNOSIS_END -->"
    src = figure.relative_to(html.parent).as_posix()
    block = f"""
        {start}
        <section class=\"section-block\" id=\"weak-diagnosis\">
          <div class=\"section-kicker\">Post-hoc diagnosis</div>
          <h2>Why the weak non-manual rows failed</h2>
          <p>The weak rows are not shortcut leaks: reset and recent-only controls remain near chance. The diagnostic compares full-memory accuracy, JEPA target retrieval, and the fraction of automatically mined target views that fall inside the true cue-visible window. This cue-window statistic is post-hoc only; it is not used in training.</p>
          <div class=\"render-card\" style=\"margin-top:14px\">
            <img src=\"{src}\" alt=\"Weak-case diagnosis for random patch-set JEPA\">
            <div class=\"render-label\"><span>Weak-case diagnosis</span><span>{diagnosis_json.relative_to(ROOT).as_posix()}</span></div>
          </div>
        </section>
        {end}
"""
    text = html.read_text()
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        html.write_text(before + block + after)
    else:
        html.write_text(text.replace("</main>", block + "\n</main>"))


def main() -> None:
    args = parse_args()
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.figure = args.figure if args.figure.is_absolute() else ROOT / args.figure
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    stats: dict[str, dict[str, Any]] = {}
    for env_name, cache_path in CACHE_PATHS.items():
        stats[env_name] = {}
        for age in (4, 8, 15):
            stats[env_name][str(age)] = selection_stats(cache_path, age=age, episodes=args.episodes)
    examples = [
        {"env_name": "scene-play-v0", "age": 15},
        {"env_name": "puzzle-3x3-play-v0", "age": 15},
        {"env_name": "humanoidmaze-large-navigate-v0", "age": 15},
    ]
    overlaid = []
    for item in examples:
        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
        overlaid.append(overlay_example(ax, CACHE_PATHS[item["env_name"]], env_name=item["env_name"], age=item["age"]))
        plt.close(fig)
    payload = {
        "schema": "random_patchset_weak_diagnosis_v1",
        "rows": rows,
        "selection_stats": stats,
        "overlay_examples": overlaid,
        "interpretation": (
            "Low cue-window selection fraction indicates target-mining failure. "
            "High cue-window selection with low readout indicates target abstraction or memory-capacity failure."
        ),
    }
    diagnosis_json = args.output / "diagnosis.json"
    diagnosis_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    make_figure(rows, stats, examples, args.figure)
    update_html(args.figure, diagnosis_json)
    print(json.dumps({"diagnosis": str(diagnosis_json), "figure": str(args.figure)}, indent=2))


if __name__ == "__main__":
    main()
