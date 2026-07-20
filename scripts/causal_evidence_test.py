#!/usr/bin/env python3
"""Causal evidence-deletion test for paper_c (ICLR reviewer rebuttal).

The evidence-discovery figure shows that slot->frame cross-attention in a
trained *one-shot* slot-memory model (chunk=0, read-time attention over the full
prefix) concentrates on the transient cue window.  A reviewer correctly noted
that attention concentration is not causal attribution.  This script runs a
causal intervention: it deletes input frame tokens *before the writer reads
them* (valid=0 and the token zeroed, i.e. the frame is effectively removed from
the episode) and measures the drop in post-hoc readout balanced accuracy.

The cue is shown identically at times 1..LAST_CUE_FRAME (3 frames), so a
single-frame deletion is confounded by redundancy.  We therefore make the main
interventions *count-matched* at K = LAST_CUE_FRAME frames so that the reviewer's
hypothesised ranking is a fair comparison:

  main (K=3 frames each):
    * cue              : delete every cue frame (times 1..3).
    * top_attended_old : delete the K OLD frames with the highest slot-attention
                         mass (times strictly before the recent legal window,
                         time < endpoint-3).  Attention-guided deletion -- the
                         model itself picks the frames, no cue knowledge used.
    * random_old       : delete K random OLD non-cue frames (control).
    * recent_legal     : delete K random frames inside the recent legal window
                         [endpoint-3, endpoint] (sanity control).

  single-frame diagnostics (K=1, reported to expose cue redundancy):
    * cue_single, top_attended_old_1, random_old_1.

Hypothesised ranking:  Delta_cue ~= Delta_top_attended > Delta_random.

We also report the exact per-env "attention mass in cue window / uniform" ratio
with bootstrap error bars, fixing the imprecise "~2-3x uniform" claim.

Attention is one-shot READ-TIME attention from the full-prefix writer; a truly
streaming writer would only expose write-time attention.

Outputs:
  paper_c/figures/fig_c_causal_evidence.{pdf,png}
  outputs/evidence_discovery_v1/causal_evidence.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import RidgeClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402
from scripts import run_multiview_patchset_color_jepa_ogbench as multiview  # noqa: E402

NATIVE = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
OUT = ROOT / "outputs" / "evidence_discovery_v1"
FIGURES = ROOT / "paper_c" / "figures"
AGE = 15
SEED = 0
N_CLASSES = base.CLASSES
N_BOOT = 2000
BOOT_SEED = 20260720

PI = {
    "ink": "#111827", "yellow": "#fbd45b", "yellow_deep": "#d8a900",
    "bad": "#7f1d1d", "gray": "#9ca3af", "good": "#315b2c",
    "white": "#ffffff", "paper": "#fbfbf9", "blue": "#2b6cb0",
}

ENVS = [
    "pointmaze-large-navigate-v0",
    "puzzle-3x3-play-v0",
    "cube-single-play-v0",
    "scene-play-v0",
]

# Main (count-matched) interventions shown in the figure.
MAIN_INTERVENTIONS = [
    ("cue", "delete cue\n(3 frames)"),
    ("top_attended_old", "delete top-3\nattended old"),
    ("random_old", "delete 3\nrandom old"),
    ("recent_legal", "delete 3\nrecent legal"),
]
MAIN_COLORS = {
    "cue": PI["bad"],
    "top_attended_old": PI["yellow_deep"],
    "random_old": PI["gray"],
    "recent_legal": PI["blue"],
}


def _cache_path(env: str) -> Path:
    return NATIVE / "cache" / base.env_key(env) / "render_cache.npz"


def _cell(env: str) -> Path:
    return NATIVE / base.env_key(env) / f"age_{AGE}" / "s0"


@torch.no_grad()
def _forward_memory_and_attn(model, frames, actions, times, valid):
    """Return (memory (B,D) normalized numpy, per-position attention mass (B,L))."""
    bsz, steps = frames.shape[:2]
    flat = frames.reshape(bsz * steps, *frames.shape[2:])
    tokens = model.frame(flat).reshape(bsz, steps, -1)
    tokens = tokens + model.action(actions) + model.time(times)
    slots, weights = model.memory(tokens, valid)  # weights (B,S,L)
    mem = torch.nn.functional.normalize(slots.mean(dim=1), dim=-1)
    attn = weights.mean(dim=1)  # (B, L) averaged over slots
    return mem.cpu().numpy(), attn.cpu().numpy()


def _collect_val(env: str, device: torch.device):
    cell = _cell(env)
    ckpt = torch.load(cell / "model.pt", map_location="cpu", weights_only=False)
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

    feats = np.load(cell / "features.npz")
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(feats["train_memory"], feats["train_labels"].astype(np.int64))

    ds = multiview.TemporalCoveragePatchSetDataset(
        _cache_path(env), age=AGE, split="val", seed=SEED,
        validation_fraction=float(a.get("validation_fraction", 0.20)), variant="full",
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    frames, actions, times, valid, labels = [], [], [], [], []
    for batch in loader:
        frames.append(batch["frames"]); actions.append(batch["actions"])
        times.append(batch["times"]); valid.append(batch["valid"])
        labels.append(batch["label"])
    data = {
        "frames": torch.cat(frames).to(device),
        "actions": torch.cat(actions).to(device),
        "times": torch.cat(times).to(device),
        "valid": torch.cat(valid).to(device),
        "labels": torch.cat(labels).numpy().astype(np.int64),
    }
    return model, readout, data


def _delete_positions(frames, valid, pos_per_ep):
    frames = frames.clone()
    valid = valid.clone()
    for b, positions in enumerate(pos_per_ep):
        for p in positions:
            valid[b, int(p)] = 0.0
            frames[b, int(p)] = 0.0
    return frames, valid


def _ablate(model, readout, data, pos_per_ep):
    f2, v2 = _delete_positions(data["frames"], data["valid"], pos_per_ep)
    mem2, _ = _forward_memory_and_attn(model, f2, data["actions"], data["times"], v2)
    return readout.predict(mem2).astype(np.int64)


def _boot_bacc_matrix(labels, preds_by_key, base_pred, n_boot, seed):
    """Vectorised paired bootstrap. Returns dict key->delta stats + base stats."""
    n = len(labels)
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, n), dtype=np.float64)
    for i in range(n_boot):
        counts[i] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    onehot = (labels[:, None] == np.arange(N_CLASSES)[None, :]).astype(np.float64)  # (n,C)
    class_total = counts @ onehot  # (n_boot, C)

    def bacc(pred):
        corr = (pred == labels).astype(np.float64)
        class_corr = counts @ (onehot * corr[:, None])  # (n_boot, C)
        with np.errstate(invalid="ignore", divide="ignore"):
            recall = np.where(class_total > 0, class_corr / class_total, np.nan)
        return np.nanmean(recall, axis=1)

    base_boot = bacc(base_pred)
    out = {"_base": {"mean": float(np.mean(base_boot)), "std": float(np.std(base_boot))}}
    for key, pred in preds_by_key.items():
        d = base_boot - bacc(pred)
        out[key] = {
            "delta_boot_mean": float(np.mean(d)),
            "delta_std": float(np.std(d)),
            "delta_ci_lo": float(np.percentile(d, 2.5)),
            "delta_ci_hi": float(np.percentile(d, 97.5)),
        }
    return out


def _run_env(env: str, device: torch.device, rng: np.random.Generator) -> dict:
    model, readout, data = _collect_val(env, device)
    endpoint = base.LAST_CUE_FRAME + AGE  # 18
    recent_start = endpoint - 3  # 15 ; recent legal window = [15, 18]
    K = base.LAST_CUE_FRAME  # 3, count-matched to the cue window
    cue_positions = list(range(1, base.LAST_CUE_FRAME + 1))  # [1,2,3]
    old_positions = list(range(0, recent_start))  # time < endpoint-3 -> 0..14
    old_noncue = [p for p in old_positions if p not in cue_positions]  # 0,4..14
    recent_positions = list(range(recent_start, endpoint + 1))  # 15..18

    labels = data["labels"]
    n = len(labels)

    base_pred, base_attn = _forward_memory_and_attn(
        model, data["frames"], data["actions"], data["times"], data["valid"]
    )
    base_pred = readout.predict(base_pred).astype(np.int64)
    base_bacc = float(balanced_accuracy_score(labels, base_pred))

    valid_np = data["valid"].cpu().numpy()
    attn_norm = np.zeros_like(base_attn)
    for b in range(n):
        L = int(valid_np[b].sum())
        mass = base_attn[b, :L].copy()
        s = mass.sum()
        attn_norm[b, :L] = mass / s if s > 1e-12 else mass

    # ---- per-episode deletion positions ----
    pos = {"cue": [], "top_attended_old": [], "random_old": [], "recent_legal": [],
           "cue_single": [], "top_attended_old_1": [], "random_old_1": []}
    top1_is_cue = np.zeros(n, dtype=bool)
    topk_cue_overlap = np.zeros(n)  # how many of the top-K old frames are cue frames
    for b in range(n):
        a = attn_norm[b]
        old_arr = np.array(old_positions)
        order = old_arr[np.argsort(a[old_arr])[::-1]]
        topk_old = list(order[:K])
        top1_old = int(order[0])
        top1_is_cue[b] = top1_old in cue_positions
        topk_cue_overlap[b] = sum(int(p) in cue_positions for p in topk_old)
        cue_arr = np.array(cue_positions)
        top_cue = int(cue_arr[np.argmax(a[cue_arr])])

        pos["cue"].append(list(cue_positions))
        pos["top_attended_old"].append([int(p) for p in topk_old])
        pos["random_old"].append([int(x) for x in rng.choice(old_noncue, size=K, replace=False)])
        pos["recent_legal"].append([int(x) for x in rng.choice(recent_positions, size=K, replace=False)])
        pos["cue_single"].append([top_cue])
        pos["top_attended_old_1"].append([top1_old])
        pos["random_old_1"].append([int(rng.choice(old_noncue))])

    all_keys = list(pos.keys())
    preds_by_key = {k: _ablate(model, readout, data, pos[k]) for k in all_keys}
    ablated_bacc = {k: float(balanced_accuracy_score(labels, preds_by_key[k])) for k in all_keys}

    boot = _boot_bacc_matrix(labels, preds_by_key, base_pred, N_BOOT, BOOT_SEED)

    interv = {}
    for k in all_keys:
        interv[k] = {
            "ablated_bacc": ablated_bacc[k],
            "delta_point": float(base_bacc - ablated_bacc[k]),
            "delta_boot_mean": boot[k]["delta_boot_mean"],
            "delta_std": boot[k]["delta_std"],
            "delta_ci_lo": boot[k]["delta_ci_lo"],
            "delta_ci_hi": boot[k]["delta_ci_hi"],
        }

    # ---- attention-mass ratio in cue window vs uniform ----
    cue_frac = attn_norm[:, cue_positions].sum(axis=1)
    uniform_frac = base.LAST_CUE_FRAME / float(endpoint + 1)
    ratio = cue_frac / uniform_frac
    brng = np.random.default_rng(BOOT_SEED + 1)
    ratio_boot = np.array([float(np.mean(ratio[brng.integers(0, n, size=n)])) for _ in range(N_BOOT)])

    ranking_holds = bool(
        interv["cue"]["delta_point"] > interv["random_old"]["delta_point"]
        and interv["top_attended_old"]["delta_point"] > interv["random_old"]["delta_point"]
    )

    return {
        "env": env,
        "endpoint": endpoint,
        "K_matched": int(K),
        "n_val": int(n),
        "baseline_bacc": base_bacc,
        "baseline_bacc_boot_std": boot["_base"]["std"],
        "chance_bacc": float(1.0 / N_CLASSES),
        "interventions": interv,
        "top1_attended_old_is_cue_frac": float(np.mean(top1_is_cue)),
        "topk_attended_old_cue_overlap_mean": float(np.mean(topk_cue_overlap)),
        "attention_ratio": {
            "cue_window_frac_mean": float(np.mean(cue_frac)),
            "cue_window_frac_std": float(np.std(cue_frac)),
            "uniform_frac": float(uniform_frac),
            "ratio_mean": float(np.mean(ratio)),
            "ratio_std": float(np.std(ratio)),
            "ratio_boot_std": float(np.std(ratio_boot)),
            "ratio_ci_lo": float(np.percentile(ratio_boot, 2.5)),
            "ratio_ci_hi": float(np.percentile(ratio_boot, 97.5)),
        },
        "ranking_holds": ranking_holds,
    }


def _make_figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])

    ax = axes[0]
    keys = [k for k, _ in MAIN_INTERVENTIONS]
    n_int = len(keys)
    n_env = len(results)
    group_w = 0.82
    bar_w = group_w / n_int
    x = np.arange(n_env)
    for j, key in enumerate(keys):
        vals = [r["interventions"][key]["delta_point"] for r in results]
        lo = [r["interventions"][key]["delta_point"] - r["interventions"][key]["delta_ci_lo"] for r in results]
        hi = [r["interventions"][key]["delta_ci_hi"] - r["interventions"][key]["delta_point"] for r in results]
        offs = (j - (n_int - 1) / 2) * bar_w
        label = dict(MAIN_INTERVENTIONS)[key].replace("\n", " ")
        ax.bar(x + offs, vals, bar_w, yerr=[lo, hi], capsize=2,
               color=MAIN_COLORS[key], edgecolor=PI["ink"], linewidth=0.5,
               error_kw={"elinewidth": 0.8}, label=label)
    ax.axhline(0.0, color=PI["ink"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["env"].split("-")[0] for r in results], fontsize=8)
    ax.set_ylabel(r"$\Delta$ balanced accuracy" "\n(baseline $-$ ablated)", fontsize=9)
    ax.set_title("(a) causal evidence-deletion (3 frames deleted each)", fontsize=8.6, weight="bold")
    ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    envs = [r["env"].split("-")[0] for r in results]
    xr = np.arange(len(envs))
    ratios = [r["attention_ratio"]["ratio_mean"] for r in results]
    lo = [r["attention_ratio"]["ratio_mean"] - r["attention_ratio"]["ratio_ci_lo"] for r in results]
    hi = [r["attention_ratio"]["ratio_ci_hi"] - r["attention_ratio"]["ratio_mean"] for r in results]
    ax.bar(xr, ratios, 0.55, yerr=[lo, hi], capsize=3, color=PI["ink"],
           edgecolor=PI["ink"], error_kw={"elinewidth": 1.0})
    ax.axhline(1.0, color=PI["bad"], ls="--", lw=1.0, label="uniform-attention baseline")
    ax.set_xticks(xr)
    ax.set_xticklabels(envs, fontsize=8)
    ax.set_ylabel("cue-window attention mass\n/ uniform baseline", fontsize=9)
    ax.set_title("(b) read-time attention concentration", fontsize=8.6, weight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ytop = max(ratios) + max(hi) + 0.3
    for xi, rr, h in zip(xr, ratios, hi):
        ax.text(xi, rr + h + 0.08, f"{rr:.2f}x", ha="center", fontsize=7.5, weight="bold")
    ax.set_ylim(0, ytop)

    for ax in axes:
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "One-shot read-time slot attention: correlation (b) and causal necessity (a)",
        fontsize=9.5, weight="bold",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_c_causal_evidence.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIGURES / "fig_c_causal_evidence.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", FIGURES / "fig_c_causal_evidence.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(7)
    results = []
    for env in ENVS:
        if not (_cell(env) / "model.pt").is_file():
            print("skip (no checkpoint):", env)
            continue
        r = _run_env(env, device, rng)
        results.append(r)
        iv = r["interventions"]
        print(json.dumps({
            "env": env,
            "baseline_bacc": round(r["baseline_bacc"], 4),
            "delta_cue": round(iv["cue"]["delta_point"], 4),
            "delta_top3_old": round(iv["top_attended_old"]["delta_point"], 4),
            "delta_random3_old": round(iv["random_old"]["delta_point"], 4),
            "delta_recent3": round(iv["recent_legal"]["delta_point"], 4),
            "delta_cue1": round(iv["cue_single"]["delta_point"], 4),
            "delta_top1_old": round(iv["top_attended_old_1"]["delta_point"], 4),
            "attn_ratio": round(r["attention_ratio"]["ratio_mean"], 3),
            "topk_cue_overlap": round(r["topk_attended_old_cue_overlap_mean"], 2),
            "ranking_holds": r["ranking_holds"],
        }))

    if not results:
        print("no checkpoints found for causal evidence test")
        return

    payload = {
        "schema": "causal_evidence_test_v1",
        "age": AGE,
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "attention_type": "one-shot read-time slot->frame cross-attention over the full prefix (chunk=0)",
        "deletion": "frame token zeroed and valid=0 before the writer reads it",
        "hypothesised_ranking": "Delta_cue ~= Delta_top_attended > Delta_random",
        "note": (
            "Main interventions are count-matched at K=LAST_CUE_FRAME=3 deleted frames. "
            "Single-frame diagnostics (suffix _1 / _single) expose the 3x redundancy of the cue."
        ),
        "results": results,
    }
    (OUT / "causal_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT / "causal_evidence.json")
    _make_figure(results)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
