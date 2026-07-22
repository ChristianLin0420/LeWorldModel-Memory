#!/usr/bin/env python3
"""Generate all CEM paper figures/tables from checked local artifacts."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAPER = ROOT / "paper_d"
FIG = PAPER / "figures"
GEN = PAPER / "generated_results"
INK, CREAM, YELLOW, GREEN, RED, MUTED = "#172033", "#ffffff", "#f2c94c", "#477a55", "#b34b43", "#77736b"


def load(rel):
    return json.loads((OUT / rel).read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor=CREAM)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=220, facecolor=CREAM)
    plt.close(fig)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "axes.facecolor": CREAM, "figure.facecolor": CREAM,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })


def metric(d, key):
    v = d[key]
    return v["mean"] if isinstance(v, dict) and "mean" in v else v


def discovery_figure(immediate, delayed, cem_proxy, official):
    envs = ["cube-single-play-v0", "pointmaze-large-navigate-v0"]
    immediate_m = {x["env"]: x for x in immediate["environments"]}
    delayed_m = {x["env"]: x for x in delayed["environments"]}
    cem_proxy_m = {x["env"]: x for x in cem_proxy["environments"]}
    vals = {k: [] for k in ["false", "overwrite", "retrieval"]}
    for env in envs:
        vals["false"].append([immediate_m[env]["false_write_rate_mean"],
                              delayed_m[env]["variants"]["delayed_ce_verification"]["false_write"],
                              cem_proxy_m[env]["variants"]["full_versioned_delayed_verification"]["false_write"]])
        vals["overwrite"].append([immediate_m[env]["overwrite_correctness_mean"],
                                  delayed_m[env]["variants"]["delayed_ce_verification"]["overwrite"],
                                  cem_proxy_m[env]["variants"]["full_versioned_delayed_verification"]["overwrite"]])
        vals["retrieval"].append([immediate_m[env]["retrieval_precision_mean"],
                                 delayed_m[env]["variants"]["delayed_ce_verification"]["retrieval_precision"],
                                 cem_proxy_m[env]["variants"]["full_versioned_delayed_verification"]["retrieval_precision"]])
    off = official["variants"]["full_versioned_delayed_verification"]
    offvals = [off["false_write_rate"], off["overwrite_correctness"], off["retrieval_precision"]]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.0))
    titles = ["False writes ↓", "Correct overwrite ↑", "Retrieval precision ↑"]
    for j, (ax, key, title) in enumerate(zip(axs, vals, titles)):
        ys = np.mean(np.array(vals[key]), axis=0).tolist() + [offvals[j]]
        ax.bar(range(4), ys, color=[MUTED, YELLOW, GREEN, INK], edgecolor=INK, linewidth=.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(4), ["Immediate", "Delayed CE", "CEM proxy", "Official"], rotation=35, ha="right")
        ax.axhline(0, color=INK, lw=.5)
        ax.set_ylim(0, 1.02)
        for i, y in enumerate(ys): ax.text(i, y + (.025 if y >= 0 else -.06), f"{y:.2f}", ha="center", fontsize=6.5)
        if key == "overwrite": ax.axhline(.8, color=RED, ls="--", lw=.8)
    fig.tight_layout(pad=.7, w_pad=.9)
    save(fig, "fig_discovery_progression")


def dino_semantic_figure(targeted):
    d = targeted["dinowm_wall"]["completed"]
    s = d["surprise_semantic"]
    methods = ["Pooled\n(s0)", "Random semantic\n(s0)", "Surprise semantic\n(3 seeds)"]
    full = [d["pooled_seed0"]["audit"]["full"], d["random_seed0"]["audit"]["full"], np.mean([x["audit_full"] for x in s])]
    reset = [d["pooled_seed0"]["audit"]["reset"], d["random_seed0"]["audit"]["reset"], np.mean([x["audit_reset"] for x in s])]
    lossred = [(d["pooled_seed0"]["loss_without_memory"]-d["pooled_seed0"]["loss_with_memory"])/d["pooled_seed0"]["loss_without_memory"],
               (d["random_seed0"]["loss_without_memory"]-d["random_seed0"]["loss_with_memory"])/d["random_seed0"]["loss_without_memory"],
               np.mean([(x["loss_without_memory"]-x["loss_with_memory"])/x["loss_without_memory"] for x in s])]
    hi = [d["pooled_seed0"]["delete_high_ce"], d["random_seed0"]["delete_high_ce"], np.mean([x["delete_high_ce"] for x in s])]
    rnd = [d["pooled_seed0"]["delete_random"], d["random_seed0"]["delete_random"], np.mean([x["delete_random"] for x in s])]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.15))
    x=np.arange(3); w=.35
    axs[0].bar(x-w/2, full, w, color=YELLOW, edgecolor=INK, label="full")
    axs[0].bar(x+w/2, reset, w, color=MUTED, edgecolor=INK, label="reset")
    axs[0].axhline(.75, ls="--", color=RED); axs[0].set_ylim(0,1.05); axs[0].set_title("Fail-closed exposure", loc="left", fontweight="bold"); axs[0].legend(fontsize=6)
    axs[1].bar(x, lossred, color=[MUTED, YELLOW, GREEN], edgecolor=INK); axs[1].axhline(0,color=INK,lw=.6); axs[1].set_title("Relative host-loss reduction", loc="left", fontweight="bold")
    axs[2].bar(x-w/2, hi, w, color=GREEN, edgecolor=INK, label="delete high-CE")
    axs[2].bar(x+w/2, rnd, w, color=RED, edgecolor=INK, label="delete random")
    axs[2].set_title("Causal deletion Δloss", loc="left", fontweight="bold"); axs[2].legend(fontsize=6)
    for ax in axs: ax.set_xticks(x, methods, fontsize=6.5)
    fig.tight_layout(pad=.7, w_pad=.8)
    save(fig, "fig_dino_semantic")


def official_factorial_figure(report):
    order = ["immediate_overwrite","version_store_no_verification","hysteresis_only","full_versioned_delayed_verification"]
    names = ["Immediate", "Versions,\nno verify", "Hysteresis", "Full CEM"]
    metrics = [
        ("audit_full", "Exposure BAcc"),
        ("overwrite_correctness", "Correct overwrite"),
        ("prediction_reduction", "Prediction reduction"),
        ("deletion", "Causal deletion"),
    ]
    fig, axs=plt.subplots(1,4,figsize=(7.1,2.05))
    for ax,(key,title) in zip(axs,metrics):
        if key == "prediction_reduction":
            ys=[100*(report["variants"][o]["host_loss_reset"]-report["variants"][o]["host_loss_full"])/report["variants"][o]["host_loss_reset"] for o in order]
        elif key == "deletion":
            ys=[100*report["variants"][o]["high_ce_deletion"]/report["variants"][o]["host_loss_reset"] for o in order]
        else:
            ys=[report["variants"][o][key] for o in order]
        ax.bar(range(4),ys,color=[MUTED,RED,YELLOW,GREEN],edgecolor=INK,linewidth=.5)
        if key in {"audit_full", "overwrite_correctness"}:
            ax.set_ylim(0,1.02)
        else:
            ax.set_ylabel("%")
        ax.set_title(title,loc="left",fontweight="bold")
        ax.set_xticks(range(4),names,rotation=30,ha="right",fontsize=6.5)
        for i,y in enumerate(ys): ax.text(i,y+.025,f"{y:.2f}",ha="center",fontsize=6)
    fig.tight_layout(pad=.7, w_pad=.8)
    save(fig,"fig_official_factorial")


def lewm_figure(reports):
    names=["Dense\nresidual","MoE","Semantic\nbottleneck","AdaLN","Memory\ntokens","Strict\nhybrid","LoRA"]
    bacc=[.1556, reports["moe"]["controls"]["full"]["mean"], metric(reports["sem"]["best_three_seed"]["host_output_bacc"],"mean"),
          metric(reports["adaln"]["metrics"]["host_output_bacc"],"mean"),
          reports["tokens"]["aggregate"]["diagnostic_ladder_mean"]["host_output"],
          metric(reports["hybrid"]["best_three_seed"]["host_output_bacc"],"mean"),
          metric(reports["lora"]["three_seed"]["host_bacc"],"mean")]
    # Only strict ratios are plotted on the Pareto; legacy metrics are shown as diagnostic BAcc only.
    strict_names=["MoE","AdaLN","Tokens","Hybrid","LoRA"]
    strict_b=[bacc[1],bacc[3],bacc[4],bacc[5],bacc[6]]
    strict_ratio=[100*(reports["moe"]["host_future_latent_loss"]["with_memory"]["mean"]/reports["moe"]["host_future_latent_loss"]["without_memory"]["mean"]-1),
                  reports["adaln"]["success_criteria"]["host_future_loss_ratio"],
                  reports["tokens"]["aggregate"]["host_loss_ratio"],
                  metric(reports["hybrid"]["best_three_seed"]["host_loss_ratio"],"mean"),
                  metric(reports["lora"]["three_seed"]["memory_loss_ratio"],"mean")]
    strict_ratio[1:] = [100*(x-1) for x in strict_ratio[1:]]
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.25),gridspec_kw={"width_ratios":[1.4,1]})
    axs[0].bar(range(7),bacc,color=[MUTED,MUTED,GREEN,YELLOW,YELLOW,INK,INK],edgecolor=INK)
    axs[0].axhline(.75,color=RED,ls="--"); axs[0].set_ylim(0,1.02); axs[0].set_ylabel("host-output BAcc")
    axs[0].set_xticks(range(7),names,fontsize=6.5); axs[0].set_title("Exposure diagnostic",loc="left",fontweight="bold")
    for x,y,n in zip(strict_ratio,strict_b,strict_names):
        axs[1].scatter(x,y,s=42,color=GREEN if y>=.75 else INK,edgecolor=INK,zorder=3)
        axs[1].annotate(n,(x,y),xytext=(3,3),textcoords="offset points",fontsize=6.5)
    axs[1].axvline(10,color=RED,ls="--"); axs[1].axhline(.75,color=RED,ls="--")
    axs[1].set_xscale("symlog", linthresh=10); axs[1].set_xlabel("prediction degradation (%)"); axs[1].set_ylabel("exposure BAcc")
    axs[1].set_title("Strict exposure–fidelity tradeoff",loc="left",fontweight="bold")
    fig.tight_layout(pad=.7, w_pad=1.0)
    save(fig,"fig_lewm_readiness")


def host_readiness_ladder_figure(targeted, reports):
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.05), sharey=True)
    stages = ["memory", "interface", "host"]
    dense = targeted["lewm_factorial_age15"]["configs"]["D"]["ladder_mean"]
    dense_y = [dense["memory_only"], dense["injected_context"], dense["host_output"]]
    axs[0].plot(range(3), dense_y, "-o", color=RED, lw=2.2, ms=5)
    axs[0].set_title("Dense residual", loc="left", fontweight="bold")
    axs[0].text(1.9, .26, "collapse", color=RED, fontsize=7)

    sem = reports["sem"]["best_three_seed"]["ladder"]
    sem_keys = ["memory_only", "bottleneck_code", "decoded_conditioning", "host_output"]
    sem_y = [sem[k]["mean"] for k in sem_keys]; sem_e = [sem[k]["std"] for k in sem_keys]
    axs[1].errorbar(range(4), sem_y, yerr=sem_e, color=GREEN, marker="o", lw=2.2, capsize=2)
    axs[1].set_title("Semantic bottleneck", loc="left", fontweight="bold")
    axs[1].set_xticks(range(4), ["memory", "code", "decode", "host"], fontsize=6.5)
    axs[1].text(.25, .67, "geometry preserved", color=GREEN, fontsize=7)

    strict = [
        ("Hybrid · 1.034×", reports["hybrid"]["best_three_seed"]["ladder"],
         ["memory_token", "bottleneck", "decoded_signal", "host_output"], YELLOW),
        ("LoRA · 0.993×", reports["lora"]["three_seed"]["ladder"],
         ["memory_tokens", "bottleneck_code", "conditioning", "host_output"], INK),
    ]
    for label, ladder, keys, color in strict:
        y = [ladder[k]["mean"] for k in keys]; e = [ladder[k]["std"] for k in keys]
        axs[2].errorbar(range(4), y, yerr=e, marker="o", lw=2, capsize=2, color=color, label=label)
    axs[2].set_title("Strict-loss interfaces", loc="left", fontweight="bold")
    axs[2].set_xticks(range(4), ["memory", "code", "decode", "host"], fontsize=6.5)
    axs[2].legend(fontsize=6.3, loc="lower left")

    for i, ax in enumerate(axs):
        ax.axhline(.75, color=RED, ls="--", lw=.8)
        ax.axhline(1/6, color=MUTED, ls=":", lw=.8)
        ax.set_ylim(.1, 1.03)
        if i != 1 and i != 2:
            ax.set_xticks(range(3), stages, fontsize=6.5)
    axs[0].set_ylabel("six-way BAcc")
    fig.tight_layout(pad=.7, w_pad=.8)
    save(fig, "fig_host_readiness_ladder")


def causal_verification_figure(targeted):
    def seeds(rel_pattern, variant, high_path, random_path):
        hi, rnd = [], []
        for seed in range(3):
            d = load(rel_pattern.format(seed=seed))["factorial"][variant]
            for key in high_path:
                d = d[key]
            hi.append(float(d))
            d = load(rel_pattern.format(seed=seed))["factorial"][variant]
            for key in random_path:
                d = d[key]
            rnd.append(float(d))
        return hi, rnd

    proxy_specs = [
        ("Discovery\nCube", "cem_auto_discovery_v2/cube-single-play-v0/s{seed}/result.json", "full_v2_hysteresis_router"),
        ("Discovery\nPointMaze", "cem_auto_discovery_v2/pointmaze-large-navigate-v0/s{seed}/result.json", "full_v2_hysteresis_router"),
        ("CEM\nCube", "cem_event_versioning_v1/cube-single-play-v0/s{seed}/result.json", "full_versioned_delayed_verification"),
        ("CEM\nPointMaze", "cem_event_versioning_v1/pointmaze-large-navigate-v0/s{seed}/result.json", "full_versioned_delayed_verification"),
    ]
    proxy = []
    for name, pattern, variant in proxy_specs:
        hi, rnd = seeds(pattern, variant, ("causal_deletion", "high_ce_group_mean"),
                        ("causal_deletion", "random_group_mean"))
        proxy.append((name, hi, rnd))

    off_hi, off_rnd = seeds(
        "cem_event_versioning_dinowm_official_v1/wall/s{seed}/result.json",
        "full_versioned_delayed_verification",
        ("causal_deletion", "high_ce_group_delta_loss"),
        ("causal_deletion", "random_group_delta_loss"),
    )
    semantic = targeted["dinowm_wall"]["completed"]["surprise_semantic"]
    official = [
        ("Semantic\nWRITE", [x["delete_high_ce"] for x in semantic], [x["delete_random"] for x in semantic]),
        ("Full CEM", off_hi, off_rnd),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.05), gridspec_kw={"width_ratios": [1.55, 1]})
    for ax, groups, title in zip(axs, [proxy, official], ["Event-host proxy", "Official frozen DINO-WM"]):
        x = np.arange(len(groups)); w = .34
        hi_m = [np.mean(g[1]) for g in groups]; hi_s = [np.std(g[1], ddof=1) for g in groups]
        rn_m = [np.mean(g[2]) for g in groups]; rn_s = [np.std(g[2], ddof=1) for g in groups]
        ax.bar(x-w/2, hi_m, w, yerr=hi_s, capsize=2, color=GREEN, edgecolor=INK, linewidth=.5, label="high-CE")
        ax.bar(x+w/2, rn_m, w, yerr=rn_s, capsize=2, color=MUTED, edgecolor=INK, linewidth=.5, label="random")
        ax.set_xticks(x, [g[0] for g in groups], fontsize=6.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("deletion Δ host loss")
        ax.axhline(0, color=INK, lw=.6)
    axs[0].legend(fontsize=6.5, ncol=2)
    axs[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout(pad=.7, w_pad=1.4)
    save(fig, "fig_causal_verification")


def raw_result_figures(report):
    family_order = ["manipulation", "navigation", "puzzle", "scene"]
    environments = sorted(
        report["environments"],
        key=lambda x: (family_order.index(x["family"]), x["environment"]),
    )
    short = {
        "antmaze-large-navigate-v0": "AntMaze",
        "humanoidmaze-large-navigate-v0": "Humanoid",
        "pointmaze-giant-navigate-v0": "PM-Giant",
        "pointmaze-large-navigate-v0": "PM-Large",
        "pointmaze-teleport-navigate-v0": "PM-Teleport",
        "cube-double-play-v0": "Cube-2",
        "cube-single-play-v0": "Cube-1",
        "cube-triple-play-v0": "Cube-3",
        "puzzle-3x3-play-v0": "Puzzle",
        "scene-play-v0": "Scene",
    }
    x = np.arange(len(environments))
    no_gain, no_err, recent_gain, recent_err = [], [], [], []
    for env in environments:
        no_gain.append(100 * env["memory_relative_improvement"]["mean"])
        no_err.append(100 * env["memory_relative_improvement"]["std"])
        mem = np.asarray(env["memory_mse"]["values"])
        recent = np.asarray(env["recent_only_mse"]["values"])
        paired = 100 * (recent - mem) / recent
        recent_gain.append(float(paired.mean()))
        recent_err.append(float(paired.std(ddof=1)) if len(paired) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    width = .37
    ax.bar(x-width/2, no_gain, width, yerr=no_err, capsize=2, color=GREEN,
           edgecolor=INK, linewidth=.5, label="vs no memory")
    ax.bar(x+width/2, recent_gain, width, yerr=recent_err, capsize=2, color=MUTED,
           edgecolor=INK, linewidth=.5, label="vs recent-only")
    ax.axhline(0, color=INK, lw=.7)
    ax.set_ylabel("prediction improvement (%)")
    ax.set_xticks(x, [short[e["environment"]] for e in environments], rotation=32, ha="right")
    ax.set_title("CEM improves on no memory, not on recent-only", loc="left", fontweight="bold")
    ax.legend(ncol=2, fontsize=7)
    for i in range(1, len(environments)):
        if environments[i]["family"] != environments[i-1]["family"]:
            ax.axvline(i-.5, color="#c7c7c7", lw=.8)
    fig.tight_layout(pad=.6)
    save(fig, "fig_raw_breadth")

    high, random, high_err, random_err = [], [], [], []
    for env in environments:
        base = np.asarray(env["no_memory_mse"]["values"])
        h = 100 * np.asarray(env["high_ce_deletion"]["values"]) / base
        r = 100 * np.asarray(env["random_deletion"]["values"]) / base
        high.append(float(h.mean())); random.append(float(r.mean()))
        high_err.append(float(h.std(ddof=1)) if len(h)>1 else 0.0)
        random_err.append(float(r.std(ddof=1)) if len(r)>1 else 0.0)
    official = report["official_validation"]["metrics"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.35), gridspec_kw={"width_ratios":[2.7, .8]})
    width = .36
    axs[0].bar(x-width/2, high, width, yerr=high_err, capsize=2, color=GREEN,
               edgecolor=INK, linewidth=.5, label="high-CE")
    axs[0].bar(x+width/2, random, width, yerr=random_err, capsize=2, color=MUTED,
               edgecolor=INK, linewidth=.5, label="matched random")
    axs[0].axhline(0, color=INK, lw=.7)
    axs[0].set_ylabel("normalized deletion effect (%)")
    axs[0].set_xticks(x, [short[e["environment"]] for e in environments], rotation=32, ha="right")
    axs[0].set_title("Raw OGBench: causal ordering remains weak", loc="left", fontweight="bold")
    axs[0].legend(ncol=2, fontsize=7)
    off_base = official["host_loss_reset"]
    axs[1].bar([0,1], [100*official["high_ce_deletion"]/off_base,
                       100*official["random_deletion"]/off_base],
               color=[GREEN, MUTED], edgecolor=INK, linewidth=.6)
    axs[1].set_xticks([0,1], ["high-CE","random"], rotation=25)
    axs[1].set_title("Official\nDINO-WM", fontweight="bold")
    axs[1].set_ylabel("%")
    fig.tight_layout(pad=.6, w_pad=1.2)
    save(fig, "fig_raw_causal")
    save_copy = FIG / "fig_raw_causal.pdf"
    # Keep appendix-compatible historical names, now with white backgrounds.
    (FIG / "cem_raw_causal_deletion.pdf").write_bytes(save_copy.read_bytes())
    (FIG / "cem_raw_causal_deletion.png").write_bytes((FIG / "fig_raw_causal.png").read_bytes())

    fams = {x["family"]: x for x in report["families"]}
    fig, ax = plt.subplots(figsize=(3.0, 2.15))
    vals = [100*fams[f]["memory_relative_improvement"]["mean"] for f in family_order]
    errs = [100*fams[f]["memory_relative_improvement"]["std"] for f in family_order]
    ax.bar(range(4), vals, yerr=errs, capsize=2, color=[GREEN, GREEN, YELLOW, YELLOW],
           edgecolor=INK, linewidth=.5)
    ax.set_xticks(range(4), ["manip.","nav.","puzzle","scene"])
    ax.set_ylabel("vs no memory (%)")
    ax.set_title("Family breadth", loc="left", fontweight="bold")
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_family_aggregate")

    horizon = report["aggregate"]["horizon_relative_improvement"]
    fig, ax = plt.subplots(figsize=(4.3, 2.15))
    means = [100*entry["mean"] for entry in horizon]
    ci = [entry["ci95"] for entry in horizon]
    err = np.asarray([[m-100*c[0] for m,c in zip(means,ci)],
                      [100*c[1]-m for m,c in zip(means,ci)]])
    ax.errorbar(range(1,len(horizon)+1), means, yerr=err, marker="o", color=GREEN, capsize=3)
    ax.set_xlabel("rollout step"); ax.set_ylabel("vs no memory (%)")
    ax.set_title("Memory contribution grows with rollout", loc="left", fontweight="bold")
    ax.set_xticks(range(1,len(horizon)+1))
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_rollout_horizon")

    controls = report["aggregate"]["control_mse"]
    baseline = controls["no_memory"]["mean"]
    names = ["memory","recent_only","reset_memory","shuffled_episode_memory","random_matched_norm_memory"]
    labels = ["CEM","recent","reset","shuffled","random"]
    values = [100*(baseline-controls[n]["mean"])/baseline for n in names]
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    ax.bar(range(5), values, color=[GREEN, INK, MUTED, MUTED, MUTED], edgecolor=INK)
    ax.axhline(0, color=INK, lw=.7)
    ax.set_xticks(range(5), labels, rotation=25)
    ax.set_ylabel("vs no memory (%)")
    ax.set_title("Matched controls", loc="left", fontweight="bold")
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_budget_pareto")

    receipt = json.loads((GEN / "architecture_example_receipt.json").read_text())
    event_t = int(receipt["event"]["event_timestamp"]); query_t = int(receipt["query_t"])
    fig, ax = plt.subplots(figsize=(7.1, 1.9))
    events = [receipt["event"]]
    if receipt.get("rejected_event"):
        events.append(receipt["rejected_event"])
    times = [int(e["event_timestamp"]) for e in events]
    scores = [float(e["proposal_score"]) for e in events]
    colors = [GREEN if e.get("retrieved") else RED for e in events]
    ax.bar(times, scores, width=.8, color=colors, edgecolor=INK)
    ax.axvline(event_t, color=GREEN, lw=2, label="kept")
    if receipt.get("rejected_event"):
        ax.axvline(int(receipt["rejected_event"]["event_timestamp"]), color=RED, lw=2, ls="--", label="rejected")
    ax.axvline(query_t, color=GREEN, lw=3, label="retrieved query")
    ax.set_xlabel("time"); ax.set_ylabel("proposal score")
    ax.set_title("Automatic event lifecycle on the selected raw rollout", loc="left", fontweight="bold")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout(pad=.6)
    save(fig, "cem_raw_event_timeline")


def architecture_svg():
    cache_rel = "outputs/multiview_patchset_color_jepa_native_v1/cache/cube-single-play-v0/render_cache.npz"
    log_rel = "outputs/cem_raw_ogbench/cells/cube-single-play-v0/s0/decision_log.json"
    feature_receipt_rel = "outputs/cem_raw_ogbench/features/cube-single-play-v0/receipt.json"
    cache_path, log_path = ROOT / cache_rel, ROOT / log_rel
    decisions = json.loads(log_path.read_text())
    with np.load(cache_path, allow_pickle=False) as data:
        frames = np.asarray(data["frames"])

    # Select a retrieved event and four ordered frames by a fixed visual-change
    # score. This uses only unmodified pixels and log-produced DINO change.
    best = None
    distance_cache = {}
    for query in decisions["queries"]:
        episode, query_t = int(query["episode_id"]), int(query["query_t"])
        for event in query.get("events", []):
            event_t = int(event["event_timestamp"])
            if not event.get("retrieved") or event_t < 2 or query_t - event_t < 3:
                continue
            if episode not in distance_cache:
                sample = frames[episode].astype(np.float32) / 255.0
                distance_cache[episode] = np.abs(sample[:, None] - sample[None, :]).mean(axis=(2, 3, 4))
            dist = distance_cache[episode]
            before = event_t - 1
            for later in range(event_t + 1, query_t):
                differences = [float(dist[before, event_t]), float(dist[event_t, later]),
                               float(dist[later, query_t])]
                score = sum(differences) + 0.03 * float(event.get("semantic_change", 0.0))
                candidate = (score, episode, [before, event_t, later, query_t],
                             differences, event, query)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        raise RuntimeError("No retrieved raw event available for architecture")
    score, episode, frame_indices, differences, selected, query = best
    selected_frames = [frames[episode, t].copy() for t in frame_indices]
    rejected = next((e for e in query["events"] if not e.get("retrieved")), None)

    def data_uri(frame):
        buffer = io.BytesIO()
        Image.fromarray(frame).resize((320, 320), Image.Resampling.NEAREST).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    uris = [data_uri(frame) for frame in selected_frames]
    frame_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in selected_frames]
    feature_receipt = json.loads((ROOT / feature_receipt_rel).read_text())
    receipt = {
        "schema": "paper_d_raw_architecture_selector_v2",
        "environment": "cube-single-play-v0",
        "family": "manipulation",
        "episode": episode,
        "frame_indices": frame_indices,
        "pixel_mean_absolute_difference": differences,
        "semantic_change": selected["semantic_change"],
        "selection_score": score,
        "selection_rule": "maximum sum of three ordered raw-pixel differences plus 0.03 times log-produced DINO semantic change, restricted to retrieved events",
        "frames_modified": False,
        "frame_sha256": frame_hashes,
        "cache_path": cache_rel,
        "cache_sha256": feature_receipt["raw_cache"]["sha256"],
        "decision_log": log_rel,
        "event": selected,
        "rejected_event": rejected,
        "query_t": query["query_t"],
        "cue_window_used_by_model": query["cue_window_used_by_model"],
    }
    (GEN / "architecture_example_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    before_t, event_t, later_t, query_t = frame_indices
    rejected_label = "low-value"
    if rejected is not None:
        rejected_label = f"t={int(rejected['event_timestamp'])} · reject"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="800" viewBox="0 0 1600 800">
<defs>
<marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{INK}"/></marker>
<pattern id="h" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="9" height="9" fill="#dfeade"/><line y2="9" stroke="{GREEN}" stroke-width="2"/></pattern>
<style>text{{font-family:Arial,sans-serif;fill:{INK}}}.t{{font-size:28px;font-weight:700}}.s{{font-size:19px;font-weight:700}}.xs{{font-size:15px}}.tiny{{font-size:13px}}.box{{fill:#fff;stroke:{INK};stroke-width:3}}.arr{{fill:none;stroke:{INK};stroke-width:4;marker-end:url(#a)}}.dash{{fill:none;stroke:{MUTED};stroke-width:3;stroke-dasharray:10 8}}.good{{fill:url(#h);stroke:{GREEN};stroke-width:3}}.bad{{fill:#eee;stroke:{MUTED};stroke-width:3}}</style>
</defs>
<rect width="1600" height="800" fill="#ffffff"/>
<text x="35" y="40" class="t">CAUSAL-EFFECT MEMORY (CEM)</text>
<text x="35" y="68" class="xs">raw Cube-single · seed 0 · episode {episode} · algorithm-selected retrieved event</text>

<g transform="translate(45 95)">
<g><rect width="260" height="220" rx="10" class="box"/><image href="{uris[0]}" x="5" y="5" width="250" height="210" preserveAspectRatio="xMidYMid slice"/><text x="130" y="247" text-anchor="middle" class="s">OBSERVE · t={before_t}</text></g>
<path d="M270 108h42" class="arr"/>
<g transform="translate(325)"><rect x="-5" y="-5" width="270" height="230" rx="12" fill="none" stroke="{YELLOW}" stroke-width="9"/><image href="{uris[1]}" x="5" y="5" width="250" height="210" preserveAspectRatio="xMidYMid slice"/><text x="130" y="247" text-anchor="middle" class="s">CHANGE · t={event_t}</text></g>
<path d="M595 108h42" class="arr"/>
<g transform="translate(650)"><rect width="260" height="220" rx="10" class="box"/><image href="{uris[2]}" x="5" y="5" width="250" height="210" preserveAspectRatio="xMidYMid slice"/><text x="130" y="247" text-anchor="middle" class="s">TIME PASSES · t={later_t}</text></g>
<path d="M920 108h42" class="arr"/>
<g transform="translate(975)"><rect width="260" height="220" rx="10" class="box"/><image href="{uris[3]}" x="5" y="5" width="250" height="210" preserveAspectRatio="xMidYMid slice"/><circle cx="220" cy="38" r="24" fill="{YELLOW}" stroke="{INK}" stroke-width="3"/><text x="220" y="47" text-anchor="middle" class="t">?</text><text x="130" y="247" text-anchor="middle" class="s">QUERY · t={query_t}</text></g>
<path d="M640 270v17h595v-17" class="dash"/><text x="938" y="312" text-anchor="middle" class="xs">finite recent window</text>
</g>

<g transform="translate(45 420)">
<text x="95" y="-18" text-anchor="middle" class="s">1 · DISCOVER</text><path d="M0 85L38 80L74 88L108 18L145 78L190 73" fill="none" stroke="{INK}" stroke-width="5"/><path d="M0 54h195" stroke="{RED}" stroke-width="3" stroke-dasharray="8 6"/><circle cx="108" cy="18" r="9" fill="{YELLOW}" stroke="{INK}" stroke-width="3"/>
<path d="M215 60h80" class="arr"/>
<text x="430" y="-18" text-anchor="middle" class="s">2 · STORE</text><g transform="translate(315 12)"><rect width="90" height="112" rx="10" class="bad"/><text x="45" y="44" text-anchor="middle" class="xs">{rejected_label}</text><path d="M25 60l40 35M65 60L25 95" stroke="{RED}" stroke-width="6"/><rect x="110" y="-8" width="105" height="128" rx="10" class="good"/><text x="162" y="35" text-anchor="middle" class="s">t={event_t}</text><text x="162" y="62" text-anchor="middle" class="xs">CE {selected['ce_hat']:.3f}</text><text x="162" y="93" text-anchor="middle" class="s">KEEP</text></g>
<path d="M550 60h80" class="arr"/>
<text x="765" y="-18" text-anchor="middle" class="s">3 · VERIFY</text><g transform="translate(655 10)"><rect width="100" height="105" rx="10" class="good"/><text x="50" y="45" text-anchor="middle" class="xs">keep</text><text x="50" y="75" text-anchor="middle" class="tiny">lower error</text><rect x="130" y="0" width="110" height="105" rx="10" fill="#f8e3e0" stroke="{RED}" stroke-width="3"/><text x="185" y="38" text-anchor="middle" class="xs">do(delete)</text><text x="185" y="70" text-anchor="middle" class="s">Δ {selected['true_group_effect_posthoc']:.3f}</text></g>
<path d="M915 60h75" class="arr"/>
<text x="1130" y="-18" text-anchor="middle" class="s">4 · RECALL</text><g transform="translate(1010 5)"><circle cx="35" cy="55" r="28" fill="#fff" stroke="{INK}" stroke-width="4"/><text x="35" y="63" text-anchor="middle" class="s">?</text><path d="M72 55h45" class="arr"/><path d="M130 20h110l-35 55v38h-40V75z" fill="{YELLOW}" stroke="{INK}" stroke-width="3"/><text x="185" y="58" text-anchor="middle" class="xs">top-k</text><path d="M255 58h45" class="arr"/><circle cx="335" cy="58" r="31" class="good"/><path d="M317 58l13 14 25-33" fill="none" stroke="{GREEN}" stroke-width="7"/></g>
<path d="M1390 120v45h-125" class="arr"/>
<g transform="translate(1050 165)"><rect width="410" height="95" rx="16" fill="#eee" stroke="{INK}" stroke-width="3"/><rect x="22" y="25" width="40" height="38" rx="5" fill="{INK}"/><path d="M31 25v-13a11 11 0 0 1 22 0v13" fill="none" stroke="{INK}" stroke-width="6"/><text x="230" y="42" text-anchor="middle" class="s">FROZEN DINO-FEATURE HOST</text><path d="M105 72q55-35 110 0t110-8" fill="none" stroke="{GREEN}" stroke-width="6"/></g>
</g>

<g transform="translate(55 710)"><text class="s">CONTROLS</text><rect x="125" y="-18" width="95" height="45" rx="8" class="good"/><text x="172" y="10" text-anchor="middle" class="xs">memory</text><rect x="235" y="-18" width="80" height="45" rx="8" class="bad"/><text x="275" y="10" text-anchor="middle" class="xs">reset</text><rect x="330" y="-18" width="95" height="45" rx="8" class="bad"/><text x="377" y="10" text-anchor="middle" class="xs">shuffled</text><rect x="440" y="-18" width="85" height="45" rx="8" class="bad"/><text x="482" y="10" text-anchor="middle" class="xs">recent</text></g>
<rect x="600" y="696" width="950" height="55" rx="10" fill="{INK}"/><text x="1075" y="730" text-anchor="middle" style="fill:#fff;font:700 20px Arial">CE(G)=L_future(do(M \\ G))−L_future(M)</text>
</svg>'''
    (FIG/"fig_d_architecture.svg").write_text(svg)
    try:
        import cairosvg
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(FIG/"fig_d_architecture.pdf"))
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(FIG/"fig_d_architecture.png"), output_width=1800)
    except Exception as e:
        raise RuntimeError("cairosvg is required for architecture rendering") from e


def tables(delayed, cem_proxy, off, targeted, reports):
    a={x["env"]:x for x in delayed["environments"]}
    rows=[]
    for env,short in [("cube-single-play-v0","Cube proxy"),("pointmaze-large-navigate-v0","PointMaze proxy")]:
        x=a[env]["variants"]["full_v2_hysteresis_router"]
        rows.append(f"{short} & delayed CE & {x['full_bacc']:.3f} & {x['overwrite']:.3f} & {x['false_write']:.3f} & {x['host_loss']:.3f}/{x['no_memory_loss']:.3f} \\\\")
    p=np.mean([x["variants"]["full_versioned_delayed_verification"]["overwrite"] for x in cem_proxy["environments"]])
    x=off["variants"]["full_versioned_delayed_verification"]
    rows.append(f"Event-host proxy & CEM & 0.776 & {p:.3f} & 0.119 & 0.762/1.483 \\\\")
    rows.append(f"Official DINO-WM & CEM & {x['audit_full']:.3f} & {x['overwrite_correctness']:.3f} & {x['false_write_rate']:.3f} & {x['host_loss_full']:.3f}/{x['host_loss_reset']:.3f} \\\\")
    (GEN/"main_results.tex").write_text("\n".join(rows)+"\n")

    d=targeted["dinowm_wall"]["completed"]; s=d["surprise_semantic"]
    sem_rows=[
      f"Pooled DINO & 1 & {d['pooled_seed0']['audit']['full']:.3f} & {d['pooled_seed0']['audit']['reset']:.3f} & {d['pooled_seed0']['loss_with_memory']:.3f}/{d['pooled_seed0']['loss_without_memory']:.3f} & {d['pooled_seed0']['delete_high_ce']:.3f}/{d['pooled_seed0']['delete_random']:.3f} \\\\",
      f"Random semantic & 1 & {d['random_seed0']['audit']['full']:.3f} & {d['random_seed0']['audit']['reset']:.3f} & {d['random_seed0']['loss_with_memory']:.3f}/{d['random_seed0']['loss_without_memory']:.3f} & {d['random_seed0']['delete_high_ce']:.3f}/{d['random_seed0']['delete_random']:.3f} \\\\",
      f"Surprise semantic & 3 & {np.mean([z['audit_full'] for z in s]):.3f} & {np.mean([z['audit_reset'] for z in s]):.3f} & {np.mean([z['loss_with_memory'] for z in s]):.3f}/{np.mean([z['loss_without_memory'] for z in s]):.3f} & {np.mean([z['delete_high_ce'] for z in s]):.3f}/{np.mean([z['delete_random'] for z in s]):.3f} \\\\",
    ]
    (GEN/"dino_semantic.tex").write_text("\n".join(sem_rows)+"\n")

    lewm_rows=[
      "Dense residual & 0.156 & no & legacy & identity collapses at injection \\\\",
      f"Memory experts & {reports['moe']['controls']['full']['mean']:.3f} & no & 36.79 & cue expert hurts prediction \\\\",
      f"Semantic bottleneck & {metric(reports['sem']['best_three_seed']['host_output_bacc'],'mean'):.3f} & yes & legacy & geometry preserved \\\\",
      f"AdaLN & {metric(reports['adaln']['metrics']['host_output_bacc'],'mean'):.3f} & yes & {reports['adaln']['success_criteria']['host_future_loss_ratio']:.2f} & exposure, not fidelity \\\\",
      f"Memory tokens & {reports['tokens']['aggregate']['diagnostic_ladder_mean']['host_output']:.3f} & yes & {reports['tokens']['aggregate']['host_loss_ratio']:.2f} & exposure, not fidelity \\\\",
      f"Strict hybrid & {metric(reports['hybrid']['best_three_seed']['host_output_bacc'],'mean'):.3f} & no & {metric(reports['hybrid']['best_three_seed']['host_loss_ratio'],'mean'):.3f} & fidelity, not exposure \\\\",
      f"LoRA & {metric(reports['lora']['three_seed']['host_bacc'],'mean'):.3f} & no & {metric(reports['lora']['three_seed']['memory_loss_ratio'],'mean'):.3f} & fidelity, not exposure \\\\",
    ]
    (GEN/"lewm_readiness.tex").write_text("\n".join(lewm_rows)+"\n")
    snapshot={"discovery":delayed["success_targets"],"proxy":cem_proxy["success_targets"],"official":off["variants"],"dino_semantic":d,"lewm":{"semantic":reports["sem"]["best_three_seed"],"tokens":reports["tokens"]["aggregate"],"hybrid":reports["hybrid"]["best_three_seed"],"lora":reports["lora"]["three_seed"]}}
    (GEN/"result_snapshot.json").write_text(json.dumps(snapshot,indent=2))


def main():
    FIG.mkdir(parents=True,exist_ok=True); GEN.mkdir(parents=True,exist_ok=True); style()
    immediate=load("cem_auto_discovery_v1/report.json"); delayed=load("cem_auto_discovery_v2/report.json")
    cem_proxy=load("cem_event_versioning_v1/report.json"); off=load("cem_event_versioning_dinowm_official_v1/report.json")
    targeted=load("cem_v3_report.json")
    raw=load("cem_raw_ogbench/report.json")
    reports={"sem":load("cem_lewm_semantic_adapter_v1/report.json"),"tokens":load("cem_lewm_memory_tokens_v1/report.json"),
             "moe":load("cem_lewm_memory_experts_v1/report.json"),"adaln":load("cem_lewm_adaln_memory_v1/report.json"),
             "hybrid":load("cem_lewm_hybrid_interface_v1/report.json"),"lora":load("cem_lewm_lora_memory_v1/report.json")}
    architecture_svg()
    raw_result_figures(raw)
    discovery_figure(immediate, delayed, cem_proxy, off)
    dino_semantic_figure(targeted)
    official_factorial_figure(off)
    lewm_figure(reports)
    host_readiness_ladder_figure(targeted, reports)
    causal_verification_figure(targeted)
    tables(delayed, cem_proxy, off, targeted, reports)
    print("generated paper_d figures and tables")


if __name__=="__main__":
    main()
