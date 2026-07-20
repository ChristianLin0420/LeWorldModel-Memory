#!/usr/bin/env python3
"""Evidence-discovery / decision plot for paper_c.

Loads a trained one-shot slot-memory checkpoint and, on held-out episodes,
extracts the slot->frame cross-attention over time. This shows *which past
observations the label-free memory discovers as decision-relevant*: attention
concentrates on the cue-visible window even though the cue left the legal
context long before readout, and correct readout decisions place more attention
mass on that window than incorrect ones.

Outputs:
  paper_c/figures/fig_c_evidence_discovery.{pdf,png}
  outputs/evidence_discovery_v1/evidence_discovery.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402
from scripts import run_multiview_patchset_color_jepa_ogbench as multiview  # noqa: E402
from scripts import run_random_patchset_jepa_ogbench as patchset  # noqa: E402

NATIVE = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
OUT = ROOT / "outputs" / "evidence_discovery_v1"
FIGURES = ROOT / "paper_c" / "figures"
PI = {"ink": "#111827", "yellow": "#fbd45b", "yellow_deep": "#d8a900",
      "bad": "#7f1d1d", "gray": "#9ca3af", "good": "#315b2c", "white": "#ffffff", "paper": "#fbfbf9"}
ENVS = ["pointmaze-large-navigate-v0", "puzzle-3x3-play-v0"]
AGE = 15


class _Args:
    def __init__(self, env, age):
        self.env_name = env
        self.age = age
        self.seed = 0
        self.validation_fraction = 0.20
        self.output = NATIVE
        self.cache_root = NATIVE
        self.batch_size = 64


def _cache_path(env):
    return NATIVE / "cache" / base.env_key(env) / "render_cache.npz"


@torch.no_grad()
def _attention_over_time(env: str, device: torch.device) -> dict | None:
    cell = NATIVE / base.env_key(env) / f"age_{AGE}" / "s0"
    ckpt_path = cell / "model.pt"
    feats_path = cell / "features.npz"
    if not ckpt_path.is_file() or not feats_path.is_file():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    dim, slots, heads = int(a["dim"]), int(a["slots"]), int(a.get("heads", 4))
    with np.load(_cache_path(env), allow_pickle=False) as data:
        img_size = int(data["img_size"])
        action_dim = int(data["actions"].shape[-1])
    model = multiview.MultiViewPatchSetJEPA(
        img_size=img_size, action_dim=action_dim, dim=dim, slots=slots, heads=heads, chunk=0
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # readout from saved features
    feats = np.load(feats_path)
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(feats["train_memory"], feats["train_labels"].astype(np.int64))

    args = _Args(env, AGE)
    ds = multiview.TemporalCoveragePatchSetDataset(
        _cache_path(env), age=AGE, split="val", seed=0, validation_fraction=0.20, variant="full"
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    endpoint = base.LAST_CUE_FRAME + AGE
    time_mass_correct = np.zeros(endpoint + 1)
    time_mass_wrong = np.zeros(endpoint + 1)
    n_correct = n_wrong = 0
    cue_frac, labels_all, preds_all = [], [], []
    for batch in loader:
        batch = base.move_batch(batch, device)
        frames = batch["frames"]
        bsz, steps = frames.shape[:2]
        flat = frames.reshape(bsz * steps, *frames.shape[2:])
        tokens = model.frame(flat).reshape(bsz, steps, -1)
        tokens = tokens + model.action(batch["actions"]) + model.time(batch["times"])
        slots, weights = model.memory(tokens, batch["valid"])  # weights (B,S,L)
        mem = torch.nn.functional.normalize(slots.mean(1), dim=-1).cpu().numpy()
        pred = readout.predict(mem)
        valid = batch["valid"].cpu().numpy()
        attn = weights.mean(1).cpu().numpy()  # (B, L) attention mass per position
        y = batch["label"].cpu().numpy()
        for b in range(bsz):
            L = int(valid[b].sum())
            mass = attn[b, :L]
            mass = mass / max(1e-8, mass.sum())
            tt = min(L, endpoint + 1)
            correct = int(pred[b]) == int(y[b])
            if correct:
                time_mass_correct[:tt] += mass[:tt]; n_correct += 1
            else:
                time_mass_wrong[:tt] += mass[:tt]; n_wrong += 1
            cue_frac.append(float(mass[1:base.LAST_CUE_FRAME + 1].sum()))
            labels_all.append(int(y[b])); preds_all.append(int(pred[b]))
    if n_correct:
        time_mass_correct /= n_correct
    if n_wrong:
        time_mass_wrong /= n_wrong
    acc = float(np.mean(np.asarray(labels_all) == np.asarray(preds_all)))
    uniform = 1.0 / (endpoint + 1)
    return {
        "env": env,
        "endpoint": endpoint,
        "correct_mass": time_mass_correct.tolist(),
        "wrong_mass": time_mass_wrong.tolist(),
        "cue_window_frac_mean": float(np.mean(cue_frac)),
        "cue_window_frac_uniform": float(base.LAST_CUE_FRAME * uniform),
        "readout_accuracy": acc,
        "n_correct": n_correct, "n_wrong": n_wrong,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for env in ENVS:
        r = _attention_over_time(env, device)
        if r:
            results.append(r)
            print(json.dumps({k: r[k] for k in ["env", "cue_window_frac_mean", "cue_window_frac_uniform", "readout_accuracy"]}))
    if not results:
        print("no checkpoints found for evidence discovery")
        return
    (OUT / "evidence_discovery.json").write_text(json.dumps(results, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])
    # Panel (a): prefer an env with some incorrect decisions to contrast.
    r = max(results, key=lambda x: x["n_wrong"]) if any(x["n_wrong"] for x in results) else results[0]
    ax = axes[0]
    t = np.arange(r["endpoint"] + 1)
    ax.axvspan(0.5, base.LAST_CUE_FRAME + 0.5, color=PI["yellow"], alpha=0.35, label="cue-visible window")
    ax.axvspan(r["endpoint"] - 3.5, r["endpoint"] + 0.5, color=PI["gray"], alpha=0.25, label="legal readout window")
    ax.plot(t, r["correct_mass"], "-o", ms=3, lw=1.8, color=PI["ink"], label="correct decisions")
    ax.plot(t, r["wrong_mass"], "-o", ms=3, lw=1.5, color=PI["bad"], label="incorrect decisions")
    ax.set_xlabel("Frame index (time)", fontsize=9)
    ax.set_ylabel("Slot attention mass", fontsize=9)
    ax.set_title(f"(a) discovered evidence over time\n({r['env'].split('-')[0]}, age {AGE})", fontsize=8.5, weight="bold")
    ax.legend(fontsize=6.4, loc="upper right")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    envs = [r["env"] for r in results]
    x = np.arange(len(envs)); w = 0.38
    ax.bar(x - w / 2, [r["cue_window_frac_mean"] for r in results], w, color=PI["ink"], edgecolor=PI["ink"], label="discovered (attention in cue window)")
    ax.bar(x + w / 2, [r["cue_window_frac_uniform"] for r in results], w, color=PI["gray"], edgecolor=PI["ink"], label="uniform-attention baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([e.split("-")[0] for e in envs], fontsize=8)
    ax.set_ylabel("Attention mass in cue window", fontsize=9)
    ax.set_title("(b) label-free evidence discovery rate", fontsize=8.5, weight="bold")
    ax.legend(fontsize=6.4, loc="upper right")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    fig.savefig(FIGURES / "fig_c_evidence_discovery.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURES / "fig_c_evidence_discovery.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    print("wrote", FIGURES / "fig_c_evidence_discovery.pdf")


if __name__ == "__main__":
    main()
