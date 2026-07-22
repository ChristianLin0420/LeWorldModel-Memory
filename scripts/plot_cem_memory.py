#!/usr/bin/env python3
"""Visualize CEM (Causal-Effect Memory) write / keep / evict decisions.

Answers the question: for the frozen-host CEM controller, *which* past
observations get WRITTEN into memory and *which* get ABANDONED (evicted)
because they lack causal value (CE ~= 0)?

The core scalar is CE(m): how much the FROZEN host's own future prediction
loss rises if memory slot m were deleted. High CE = causally valuable (kept);
CE ~= 0 = abandoned. WRITE is gated by the host's one-step surprise; KEEP/EVICT
is decided by the amortized causal-value estimate ce_hat (calibrated against
ce_true when a periodic hard-deletion `do` is run).

Two figures are produced (see docs/CEM_MEMORY_VIZ_IDEAS.md):

  * cem_memory_timeline : Concept A + B. A surprise strip (writes fire on
    surprise, not on value) sitting on top of a per-slot lifespan Gantt chart
    (born -> kept/evicted), bar colour = CE value.
  * cem_value_scatter   : Concept E + D. Written-vs-abandoned 2D map
    (surprise-at-write x CE, filled=kept / hollow=abandoned) plus a
    ce_hat-vs-ce_true calibration scatter (does the cheap head predict the
    value we evict on?).

Input schema (per host,env,seed) at
  outputs/cem_<host>_v1/<env>/s<seed>/decision_log.json
matches the sibling worker's emitted logs. When no real log is passed and none
are auto-discovered, a synthetic sample is generated so the tool has a working
demo now and runs unchanged on the real logs later.

Usage:
  .venv/bin/python scripts/plot_cem_memory.py                # auto-glob or synth
  .venv/bin/python scripts/plot_cem_memory.py --log path/to/decision_log.json
  .venv/bin/python scripts/plot_cem_memory.py --synthetic    # force synthetic
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "docs" / "assets"
DEFAULT_GLOB = str(REPO / "outputs" / "cem_*_v1" / "**" / "decision_log.json")
SYNTH_PATH = ASSETS / "cem_decision_log_synthetic.json"
OUTPUT_PREFIX = "cem"

# --------------------------------------------------------------------------- #
# Physical Intelligence theme (black / cream / yellow)
# --------------------------------------------------------------------------- #
PI = {
    "ink": "#111827",
    "black": "#000000",
    "cream": "#f5f4ef",
    "paper": "#fbfbf9",
    "paper2": "#efeee8",
    "yellow": "#fbd45b",
    "yellow_deep": "#d8a900",
    "muted": "#656760",
    "line": "#d4d3cb",
    "line_dark": "#333b49",
    "good": "#315b2c",
    "bad": "#7f1d1d",
    "gray": "#9ca3af",
    "gray2": "#6b7280",
    "blue": "#2563eb",
    "white": "#ffffff",
}
# Causal value heat scale: low CE (cream) -> high CE (ink).
CE_CMAP = LinearSegmentedColormap.from_list(
    "cem_value", [PI["cream"], PI["yellow"], PI["yellow_deep"], PI["ink"]]
)
MONO = "DejaVu Sans Mono"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": PI["ink"],
            "axes.labelcolor": PI["ink"],
            "axes.titlecolor": PI["ink"],
            "xtick.color": PI["ink"],
            "ytick.color": PI["ink"],
            "text.color": PI["ink"],
            "figure.facecolor": PI["white"],
            "axes.facecolor": PI["paper"],
            "savefig.facecolor": PI["white"],
            "grid.color": PI["line"],
            "grid.linewidth": 0.6,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, stem: str) -> List[Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if stem.startswith("cem_"):
        stem = OUTPUT_PREFIX + stem[len("cem"):]
    out = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        p = ASSETS / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03, **kw)
        out.append(p)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Synthetic sample (matches sibling worker schema exactly)
# --------------------------------------------------------------------------- #
def make_synthetic(seed: int = 0, n_frames: int = 48, n_slots: int = 14) -> Dict[str, Any]:
    """Generate a decision_log.json-shaped dict for demo / development.

    The generative story mirrors CEM: surprise spikes drive writes; a subset of
    writes correspond to the true cue (high CE, kept and later retrieved), the
    rest are aleatoric-surprise / distractor writes (CE ~= 0, evicted early).
    """
    rng = random.Random(seed)
    cue_window = [8, 12]
    readout_t = 40

    # Baseline host surprise stream with a couple of structural spikes.
    frame_surprise = [0.10 + 0.05 * rng.random() for _ in range(n_frames)]
    for c in (10, 22, 31):  # spikes: one is the true cue, others are noise/distractor
        for d, amp in ((-1, 0.4), (0, 1.0), (1, 0.5)):
            t = c + d
            if 0 <= t < n_frames:
                frame_surprise[t] += amp * (0.8 + 0.4 * rng.random())

    events: List[Dict[str, Any]] = []
    write_frames = []
    # Writes fire when surprise exceeds a running quantile-ish threshold.
    thr = sorted(frame_surprise)[int(0.72 * n_frames)]
    for t, s in enumerate(frame_surprise):
        if s >= thr and rng.random() < 0.85:
            write_frames.append(t)
    # Ensure we have a decent number of slots to show.
    while len(write_frames) < n_slots:
        write_frames.append(rng.randint(2, n_frames - 6))
    write_frames = sorted(set(write_frames))[:n_slots]

    for i, t in enumerate(write_frames):
        surprise_at_write = frame_surprise[t]
        is_true_cue = (cue_window[0] <= t <= cue_window[1] + 4) and rng.random() < 0.85
        # True causal value: high only for genuine cue traces; noise ~= 0.
        if is_true_cue:
            ce_true = 0.55 + 0.35 * rng.random()
        elif surprise_at_write > thr + 0.4:  # high surprise but aleatoric -> low value
            ce_true = 0.02 + 0.10 * rng.random()
        else:
            ce_true = 0.0 + 0.08 * rng.random()
        # Amortized head: noisy estimate of ce_true.
        ce_hat = max(0.0, ce_true + rng.gauss(0.0, 0.07))

        kept = ce_hat >= 0.30  # eviction drops lowest-value slots
        status = "kept" if kept else "evicted"
        evicted_at: Optional[int] = None
        retrieved_at: Optional[int] = None
        verify_delta: Optional[float] = None
        if not kept:
            evicted_at = min(n_frames - 1, t + rng.randint(2, 7))
        else:
            # High-value kept cues get retrieved-then-verified near readout.
            if ce_hat > 0.45 and rng.random() < 0.8:
                status = "retrieved"
                retrieved_at = readout_t
                verify_delta = -(0.15 + 0.30 * rng.random())  # loss drops -> accept
            elif rng.random() < 0.15:
                status = "rejected"  # verify failed
                retrieved_at = readout_t
                verify_delta = 0.02 + 0.05 * rng.random()

        # ce_true is only known on a periodically-calibrated subset.
        ce_true_out: Optional[float] = ce_true if (i % 2 == 0) else None

        events.append(
            {
                "slot_id": i,
                "written_at": t,
                "cue_timestamp": round(t / n_frames, 4),
                "surprise_at_write": round(surprise_at_write, 4),
                "ce_hat": round(ce_hat, 4),
                "ce_true": (round(ce_true_out, 4) if ce_true_out is not None else None),
                "status": status,
                "evicted_at": evicted_at,
                "retrieved_at": retrieved_at,
                "verify_delta": (round(verify_delta, 4) if verify_delta is not None else None),
            }
        )

    return {
        "host": "lewm",
        "env": "pusht_synthetic",
        "seed": seed,
        "cue_window": cue_window,
        "readout_t": readout_t,
        "frame_surprise": [round(s, 4) for s in frame_surprise],
        "events": events,
    }


# --------------------------------------------------------------------------- #
# Loading / discovery
# --------------------------------------------------------------------------- #
def discover_logs() -> List[Path]:
    return sorted(Path(p) for p in glob.glob(DEFAULT_GLOB, recursive=True))


def load_log(explicit: Optional[str], force_synth: bool) -> Dict[str, Any]:
    if force_synth:
        log = make_synthetic()
        SYNTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYNTH_PATH.write_text(json.dumps(log, indent=2))
        print(f"[cem] using synthetic sample (forced) -> wrote {SYNTH_PATH}")
        return log
    if explicit:
        p = Path(explicit)
        print(f"[cem] loading explicit log: {p}")
        return json.loads(p.read_text())
    found = discover_logs()
    if found:
        print(f"[cem] auto-discovered {len(found)} real log(s); using {found[0]}")
        return json.loads(found[0].read_text())
    log = make_synthetic()
    SYNTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNTH_PATH.write_text(json.dumps(log, indent=2))
    print(f"[cem] no real logs at {DEFAULT_GLOB}; wrote synthetic sample -> {SYNTH_PATH}")
    return log


# --------------------------------------------------------------------------- #
# Status styling
# --------------------------------------------------------------------------- #
KEPT_STATES = {"kept", "retrieved"}


def _ce_of(ev: Dict[str, Any]) -> float:
    return float(ev.get("ce_hat", 0.0) or 0.0)


# --------------------------------------------------------------------------- #
# Figure 1 : surprise strip + lifespan timeline
# --------------------------------------------------------------------------- #
def plot_timeline(log: Dict[str, Any]) -> List[Path]:
    _style()
    events = log["events"]
    n_frames = len(log["frame_surprise"])
    cue_lo, cue_hi = log["cue_window"]
    readout_t = log["readout_t"]

    ce_vals = [_ce_of(e) for e in events] or [0.0, 1.0]
    norm = Normalize(vmin=0.0, vmax=max(0.6, max(ce_vals)))

    # sort slots so high-value (kept) sit at top, low-value (evicted) at bottom
    order = sorted(range(len(events)), key=lambda i: _ce_of(events[i]))
    ev_sorted = [events[i] for i in order]

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.7], hspace=0.08)
    ax_s = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1], sharex=ax_s)

    # ---- top: surprise strip with write markers ----
    xs = list(range(n_frames))
    surprise = log["frame_surprise"]
    ax_s.fill_between(xs, surprise, color=PI["yellow"], alpha=0.35, linewidth=0)
    ax_s.plot(xs, surprise, color=PI["yellow_deep"], lw=1.6)
    thr = sorted(surprise)[int(0.72 * n_frames)] if n_frames else 0.0
    ax_s.axhline(thr, color=PI["gray2"], lw=1.0, ls=(0, (4, 3)))
    ax_s.text(n_frames - 0.5, thr, " write gate  ", ha="right", va="bottom",
              fontsize=8, color=PI["gray2"])
    for e in events:
        t = e["written_at"]
        c = CE_CMAP(norm(_ce_of(e)))
        ax_s.scatter([t], [surprise[t]], s=46, color=c, edgecolor=PI["ink"],
                     linewidth=0.7, zorder=5)
    ax_s.axvspan(cue_lo, cue_hi, color=PI["blue"], alpha=0.10, linewidth=0)
    ax_s.axvline(readout_t, color=PI["ink"], lw=1.1, ls=":")
    ax_s.set_ylabel("host\nsurprise $s_t$", fontsize=9)
    ax_s.set_title(
        f"CEM memory lifecycle  ·  host={log['host']}  env={log['env']}  seed={log['seed']}"
        "\nwrites fire on surprise (top); slots kept/evicted by causal value CE (below)",
        fontsize=11, loc="left", pad=8,
    )
    ax_s.grid(True, axis="y", alpha=0.5)
    plt.setp(ax_s.get_xticklabels(), visible=False)

    # ---- bottom: per-slot lifespan Gantt ----
    for row, e in enumerate(ev_sorted):
        start = e["written_at"]
        kept = e["status"] in KEPT_STATES
        end = e["evicted_at"] if (e.get("evicted_at") is not None) else (n_frames - 1)
        c = CE_CMAP(norm(_ce_of(e)))
        if kept:
            ax_t.plot([start, end], [row, row], color=c, lw=7, solid_capstyle="round",
                      zorder=3)
            # arrow head showing it survives to the end of the episode
            ax_t.scatter([end], [row], marker=">", s=40, color=c,
                         edgecolor=PI["ink"], linewidth=0.5, zorder=4)
        else:
            # abandoned: greyed hollow bar terminating at eviction with an x
            ax_t.plot([start, end], [row, row], color=PI["gray"], lw=6,
                      alpha=0.55, solid_capstyle="butt", zorder=2)
            ax_t.scatter([end], [row], marker="x", s=42, color=PI["bad"],
                         linewidth=1.6, zorder=4)
        # write marker (filled dot at birth, coloured by CE)
        ax_t.scatter([start], [row], s=52, color=c, edgecolor=PI["ink"],
                     linewidth=0.8, zorder=5)
        # retrieval marker
        if e.get("retrieved_at") is not None:
            mk = "*" if e["status"] == "retrieved" else "P"
            mc = PI["good"] if e["status"] == "retrieved" else PI["bad"]
            ax_t.scatter([e["retrieved_at"]], [row], marker=mk, s=120, color=mc,
                         edgecolor=PI["ink"], linewidth=0.6, zorder=6)

    ax_t.axvspan(cue_lo, cue_hi, color=PI["blue"], alpha=0.10, linewidth=0,
                 label="_cue")
    ax_t.axvline(readout_t, color=PI["ink"], lw=1.1, ls=":")
    ax_t.text(readout_t, len(ev_sorted) - 0.3, " readout", fontsize=8,
              color=PI["ink"], va="top")
    ax_t.text((cue_lo + cue_hi) / 2, len(ev_sorted) - 0.3, "cue window",
              fontsize=8, color=PI["blue"], ha="center", va="top")
    ax_t.set_xlim(-0.5, n_frames - 0.5)
    ax_t.set_ylim(-0.8, len(ev_sorted) - 0.2)
    ax_t.set_yticks(range(len(ev_sorted)))
    ax_t.set_yticklabels([f"slot {e['slot_id']}" for e in ev_sorted], fontsize=7.5)
    ax_t.set_xlabel("frame time  t", fontsize=10)
    ax_t.set_ylabel("memory slots  (low value \u2192 high value)", fontsize=9)
    ax_t.grid(True, axis="x", alpha=0.4)

    # colourbar for CE
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_s, ax_t], fraction=0.03, pad=0.015)
    cbar.set_label("causal value  CE  ($\\uparrow$ host loss if deleted)", fontsize=9)

    legend_handles = [
        Line2D([0], [0], color=PI["yellow_deep"], lw=7, label="kept (high CE)"),
        Line2D([0], [0], color=PI["gray"], lw=6, alpha=0.55, label="abandoned / evicted (CE\u22480)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PI["yellow"],
               markeredgecolor=PI["ink"], markersize=9, label="write (born on surprise)"),
        Line2D([0], [0], marker="x", color=PI["bad"], lw=0, markersize=9,
               markeredgewidth=1.6, label="eviction"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=PI["good"],
               markeredgecolor=PI["ink"], markersize=13, label="retrieved + verified"),
    ]
    ax_t.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
                ncol=1, framealpha=0.0)

    return savefig(fig, "cem_memory_timeline")


# --------------------------------------------------------------------------- #
# Figure 2 : written-vs-abandoned 2D map + ce_hat/ce_true calibration
# --------------------------------------------------------------------------- #
def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for rank, idx in enumerate(order):
            r[idx] = rank
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def plot_value_scatter(log: Dict[str, Any]) -> List[Path]:
    _style()
    events = log["events"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.suptitle(
        f"Written vs abandoned by causal value  ·  host={log['host']}  env={log['env']}  seed={log['seed']}",
        fontsize=12.5, x=0.02, ha="left", y=1.0,
    )

    # ---------- panel 1: surprise-at-write x CE, kept=filled / evicted=hollow ----------
    evict_band = 0.30  # keep/evict decision boundary on ce_hat (demo)
    ax1.axhspan(0, evict_band, color=PI["gray"], alpha=0.14, linewidth=0)
    ax1.axhline(evict_band, color=PI["gray2"], lw=1.0, ls=(0, (4, 3)))
    ax1.text(ax1.get_xlim()[1], evict_band, "  evict below", fontsize=8,
             color=PI["gray2"], va="bottom", ha="left")

    for e in events:
        kept = e["status"] in KEPT_STATES
        x = float(e["surprise_at_write"])
        y = _ce_of(e)
        if kept:
            ax1.scatter([x], [y], s=95, color=PI["yellow"], edgecolor=PI["ink"],
                        linewidth=1.1, zorder=4)
            if e["status"] == "retrieved":
                ax1.scatter([x], [y], marker="*", s=70, color=PI["good"],
                            edgecolor="none", zorder=5)
        else:
            ax1.scatter([x], [y], s=85, facecolor="none", edgecolor=PI["gray2"],
                        linewidth=1.3, zorder=3)
    ax1.set_xlabel("surprise at write  $s_t$", fontsize=10)
    ax1.set_ylabel("causal value  CE  ($\\hat{CE}_\\psi$)", fontsize=10)
    ax1.set_title("high surprise \u2260 high value:\nabandoned writes sit in the low-CE band",
                  fontsize=10.5, loc="left")
    ax1.grid(True, alpha=0.45)
    ax1.set_ylim(bottom=min(-0.02, ax1.get_ylim()[0]))
    leg1 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PI["yellow"],
               markeredgecolor=PI["ink"], markersize=11, label="kept (high value)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=PI["gray2"], markersize=11, markeredgewidth=1.3,
               label="abandoned (low value)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=PI["good"],
               markeredgecolor="none", markersize=13, label="retrieved + verified"),
    ]
    ax1.legend(handles=leg1, loc="upper left", fontsize=8.5)

    # ---------- panel 2: ce_hat vs ce_true calibration ----------
    cal = [(float(e["ce_hat"]), float(e["ce_true"]), e["status"])
           for e in events if e.get("ce_true") is not None]
    if cal:
        hi = max(max(h, t) for h, t, _ in cal) * 1.1 + 0.02
        ax2.plot([0, hi], [0, hi], color=PI["gray2"], lw=1.0, ls="--",
                 label="ideal $\\hat{CE}=CE$")
        for h, t, st in cal:
            kept = st in KEPT_STATES
            if kept:
                ax2.scatter([h], [t], s=90, color=PI["yellow"], edgecolor=PI["ink"],
                            linewidth=1.0, zorder=4)
            else:
                ax2.scatter([h], [t], s=80, facecolor="none", edgecolor=PI["gray2"],
                            linewidth=1.3, zorder=3)
        rho = _spearman([h for h, _, _ in cal], [t for _, t, _ in cal])
        if rho is not None:
            ax2.text(0.04, 0.94, f"Spearman $\\rho$ = {rho:.2f}\n(n={len(cal)} calibrated slots)",
                     transform=ax2.transAxes, fontsize=9.5, va="top", family=MONO,
                     bbox=dict(boxstyle="round,pad=0.4", fc=PI["cream"], ec=PI["line"]))
        ax2.set_xlim(-0.02, hi)
        ax2.set_ylim(-0.02, hi)
    else:
        ax2.text(0.5, 0.5, "no ce_true calibration\navailable in this log",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=11, color=PI["muted"])
    ax2.set_xlabel("amortized estimate  $\\hat{CE}_\\psi$  (cheap KEEP head)", fontsize=10)
    ax2.set_ylabel("true hard-deletion  CE  (periodic $do$)", fontsize=10)
    ax2.set_title("does the cheap head predict\nthe value we evict on?", fontsize=10.5, loc="left")
    ax2.grid(True, alpha=0.45)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return savefig(fig, "cem_value_scatter")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    global OUTPUT_PREFIX
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=str, default=None,
                    help="path to a decision_log.json (real or synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="force regeneration of the synthetic sample")
    ap.add_argument("--list", action="store_true",
                    help="list auto-discovered real logs and exit")
    ap.add_argument("--prefix", default="cem",
                    help="output asset prefix (for example cem_v2)")
    args = ap.parse_args()
    OUTPUT_PREFIX = args.prefix

    if args.list:
        found = discover_logs()
        if found:
            print(f"[cem] {len(found)} real log(s) discovered:")
            for p in found:
                print("   ", p)
        else:
            print(f"[cem] no real logs matching {DEFAULT_GLOB}")
        return

    log = load_log(args.log, args.synthetic)
    n = len(log["events"])
    kept = sum(1 for e in log["events"] if e["status"] in KEPT_STATES)
    print(f"[cem] {n} memory events  |  kept={kept}  abandoned={n - kept}")

    paths = plot_timeline(log) + plot_value_scatter(log)
    print("[cem] wrote figures:")
    for p in paths:
        print("   ", p)


if __name__ == "__main__":
    main()
