#!/usr/bin/env python3
"""V21 X0a — t4 probe-family repair on the frozen W3 exports
(docs/V21_PROPOSAL.md 4/X0.2; panel objection I2).

The W3 t4 ridge probe (StandardScaler + Ridge(alpha=1e-3), unstandardized
targets) reads the rfix family at R^2 -3.2 to -4.2 while ac-GRU sits at
-0.37 — either the carriers destroy xi information on the continuous task or
the readout family is fragile to prior_read amplitude.  This script
re-probes the FROZEN t4 eval exports (no training) under three families:

  legacy    StandardScaler + Ridge(1e-3)                (the V19/V20 family)
  repaired  StandardScaler + RidgeCV(logspace(-3,3,7)) on STANDARDIZED
            targets, predictions unscaled before scoring (the registered
            scale-robust family)
  mlp       StandardScaler + MLPRegressor(64), early stopping (nonlinear
            control: is the information there at all?)

Registered adjudication: rfix within 0.1 R^2 of ac-GRU (or better) under
repaired/mlp => READOUT FRAGILITY (t4 rejoins pooled gates under the
repaired family); rfix still <= acgru - 0.5 under BOTH => INFORMATION LOSS
(t4 stays out, scoped claims only); otherwise MIXED (report both).

Also emits the X1 pooling preview: per-task standardized paired effects
(Cohen's d over seed pairs) for rfix - acgru on t1/t3 (frozen probes) and t4
(repaired family), pooled with a seed bootstrap — the scale-free pooling
rule X1 freezes.

Writes outputs/v21_x0/t4_probes.{json,md}.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.eval_v19_p2 as p2eval

W3 = ROOT / "outputs" / "v20_w3"
OUT = ROOT / "outputs" / "v21_x0"
ARMS = ("lkc_rfix", "dfc_rho6", "dfc_eta3", "acgru", "none")
SEEDS = tuple(range(10))
PROBE_SEEDS = (0, 1, 2)
RIDGECV_ALPHAS = np.logspace(-3, 3, 7)
FRAGILITY_MARGIN = 0.1
INFO_LOSS_MARGIN = 0.5


def registered_features(export: dict) -> tuple[np.ndarray, np.ndarray]:
    """The registered t4 coordinate: prior_read at gap_off -> xi (E, 2)."""
    events = p2eval.export_events(export)
    gap_off = np.asarray(events["gap_off"], dtype=np.int64)
    episodes = export["prior_read"].shape[0]
    features = export["prior_read"][np.arange(episodes), gap_off]
    return features.astype(np.float64), export["xi"].astype(np.float64)


def _fit_score(family: str, x_train, y_train, x_eval, y_eval) -> float:
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_eval = scaler.transform(x_eval)
    if family == "legacy":
        model = Ridge(alpha=1e-3).fit(x_train, y_train)
        prediction = model.predict(x_eval)
    elif family == "repaired":
        y_scaler = StandardScaler().fit(y_train)
        model = RidgeCV(alphas=RIDGECV_ALPHAS).fit(
            x_train, y_scaler.transform(y_train))
        prediction = y_scaler.inverse_transform(model.predict(x_eval))
    elif family == "mlp":
        y_scaler = StandardScaler().fit(y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = MLPRegressor(hidden_layer_sizes=(64,), max_iter=2000,
                                 early_stopping=True, random_state=0
                                 ).fit(x_train, y_scaler.transform(y_train))
        prediction = y_scaler.inverse_transform(model.predict(x_eval))
    else:
        raise ValueError(family)
    return float(r2_score(y_eval, prediction))


def probe_cell(arm: str, seed: int) -> dict[str, float]:
    export = p2eval.load_export(
        W3 / "t4" / arm / f"s{seed}" / "eval_export.npz")
    features, xi = registered_features(export)
    results = {}
    for family in ("legacy", "repaired", "mlp"):
        scores = []
        for probe_seed in PROBE_SEEDS:
            train_idx, eval_idx = p2eval._split(len(features), probe_seed)
            scores.append(_fit_score(family, features[train_idx],
                                     xi[train_idx], features[eval_idx],
                                     xi[eval_idx]))
        results[family] = float(np.mean(scores))
    return results


def standardized_effect(diffs: np.ndarray) -> float:
    return float(diffs.mean() / max(diffs.std(ddof=1), 1e-12))


def pooled_d(per_task_diffs: dict[str, np.ndarray], draws: int = 20_000
             ) -> dict[str, Any]:
    """Scale-free pooling: mean per-task Cohen's d, seed bootstrap CI."""
    rng = np.random.default_rng(21_021)
    point = float(np.mean([standardized_effect(d)
                           for d in per_task_diffs.values()]))
    estimates = []
    for _ in range(draws):
        ds = []
        for diffs in per_task_diffs.values():
            index = rng.integers(0, len(diffs), size=len(diffs))
            resample = diffs[index]
            if resample.std(ddof=1) > 1e-12:
                ds.append(standardized_effect(resample))
        if ds:
            estimates.append(np.mean(ds))
    estimates = np.asarray(estimates)
    return {"pooled_d": point,
            "ci95": [float(np.quantile(estimates, 0.025)),
                     float(np.quantile(estimates, 0.975))],
            "p_pos": float(((estimates <= 0).sum() + 1)
                           / (len(estimates) + 1)),
            "per_task_d": {task: standardized_effect(diffs)
                           for task, diffs in per_task_diffs.items()}}


def main() -> None:
    table: dict[str, Any] = {}
    for arm in ARMS:
        per_family = {family: [] for family in ("legacy", "repaired", "mlp")}
        for seed in SEEDS:
            cell = probe_cell(arm, seed)
            for family, value in cell.items():
                per_family[family].append(value)
        table[arm] = {family: {"mean": float(np.mean(values)),
                               "sd": float(np.std(values)),
                               "per_seed": [round(v, 3) for v in values]}
                      for family, values in per_family.items()}

    rfix, acgru = table["lkc_rfix"], table["acgru"]
    recovered = all(rfix[f]["mean"] >= acgru[f]["mean"] - FRAGILITY_MARGIN
                    for f in ("repaired", "mlp"))
    lost = all(rfix[f]["mean"] <= acgru[f]["mean"] - INFO_LOSS_MARGIN
               for f in ("repaired", "mlp"))
    verdict = ("readout_fragility" if recovered
               else "information_loss" if lost else "mixed")

    # X1 pooling preview: standardized paired rfix - acgru effects.
    def stationary(task, arm, seed):
        return float(json.loads(
            (W3 / task / arm / f"s{seed}" / "probe_results.json").read_text()
        )["registered"]["mean"])
    per_task = {
        task: np.array([stationary(task, "lkc_rfix", s)
                        - stationary(task, "acgru", s) for s in SEEDS])
        for task in ("t1", "t3")}
    per_task["t4_repaired"] = np.array(
        [table["lkc_rfix"]["repaired"]["per_seed"][s]
         - table["acgru"]["repaired"]["per_seed"][s] for s in SEEDS])
    pooling = pooled_d(per_task)

    report = {
        "schema_version": 1,
        "study": "v21-x0-t4-probe-repair",
        "families": {"legacy": "StandardScaler+Ridge(1e-3)",
                     "repaired": "StandardScaler+RidgeCV(1e-3..1e3), "
                                 "standardized targets",
                     "mlp": "StandardScaler+MLP(64), standardized targets"},
        "arms": table,
        "adjudication": {
            "rule": (f"rfix within {FRAGILITY_MARGIN} of acgru under repaired"
                     f" AND mlp => fragility; <= acgru-{INFO_LOSS_MARGIN} "
                     f"under both => information loss; else mixed"),
            "verdict": verdict,
        },
        "x1_pooling_preview_rfix_minus_acgru": pooling,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t4_probes.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = ["# V21 X0a — t4 probe-family repair", ""]
    lines.append("| arm | legacy R2 | repaired R2 | mlp R2 |")
    lines.append("|---|---|---|---|")
    for arm, row in table.items():
        lines.append(f"| {arm} | {row['legacy']['mean']:+.3f} | "
                     f"{row['repaired']['mean']:+.3f} | "
                     f"{row['mlp']['mean']:+.3f} |")
    lines.append("")
    lines.append(f"**Adjudication:** `{verdict}`")
    lines.append("")
    lines.append(f"**X1 pooling preview (standardized paired d, "
                 f"rfix − acgru, t1/t3/t4-repaired):** pooled d = "
                 f"{pooling['pooled_d']:+.3f} CI95 [{pooling['ci95'][0]:+.3f},"
                 f" {pooling['ci95'][1]:+.3f}], p_pos = {pooling['p_pos']:.2e}"
                 f"; per-task d = " + ", ".join(
                     f"{task} {d:+.2f}"
                     for task, d in pooling["per_task_d"].items()))
    (OUT / "t4_probes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
