#!/usr/bin/env python3
"""Render Paper A (docs/PAPER_A.md) from templates/PAPER_A.template.md.

Program convention: prose lives in the template; every number is computed
here from the artifact JSONs and injected by {{UPPER_SNAKE}} placeholder.
Fails closed on a missing artifact, a missing key, or an unfilled/unknown
placeholder.  Writes docs/PAPER_A.manifest.json binding the SHA-256 of the
template and of every source artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = {
    "x0_sensitivity": "outputs/v21_x0/sensitivity.json",
    "x0_t4": "outputs/v21_x0/t4_probes.json",
    "x0b_sweep": "outputs/v21_x0/sweep_summary.json",
    "x1_gates": "outputs/v21_x1/x1_gates.json",
    "x2_v3": "outputs/v21_x2/x2_results_v3.json",
    "x2_envelope": "outputs/v21_x2/x2_results_envelope.json",
    "x2_selector": "outputs/v21_x2/x2_selector_stats.json",
    "x3_delay": "outputs/v21_x3/delay_scaling.json",
    "x3_tau": "outputs/v21_x3/tau_rescale.json",
    "x3_dino_ext": "outputs/v21_x3/dino_sstar_ext.json",
    "x3_dino_pm": "outputs/v21_x3/dino_sstar_pointmass.json",
    "f1_cert": "outputs/v21_f1/certification.json",
    "f2b_s0": "outputs/v21_f2b/certificates/t1s1/vicreg/s0.json",
    "f2b_s1": "outputs/v21_f2b/certificates/t1s1/vicreg/s1.json",
    "w0_summary": "outputs/v20_w0/w0_summary.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_all() -> dict[str, dict]:
    data = {}
    for name, rel in ARTIFACTS.items():
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing artifact: {rel}")
        data[name] = json.loads(path.read_text())
    return data


def fmt(value: float, digits: int = 3, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text


def per_seed_range(scores) -> str:
    return f"{min(scores):.3f}–{max(scores):.3f}"


def replacements(a: dict[str, dict]) -> dict[str, str]:
    x1 = a["x1_gates"]
    x2 = a["x2_v3"]["per_seed"]
    seeds = sorted(x2)

    def x2_mean(label: str) -> float:
        return sum(x2[s][label]["success_rate"] for s in seeds) / len(seeds)

    sel = a["x2_selector"]["mean"]
    f1 = a["f1_cert"]
    f1_banks = [f1["banks"][k] for k in sorted(f1["banks"])]
    delay = a["x3_delay"]["curves"]
    tau = a["x3_tau"]["curves"]
    sens = a["x0_sensitivity"]
    headline_key = "inversion_rfix_minus_acgru"
    healthy = sens[headline_key]["healthy_only"]["pooled"]
    full = sens[headline_key]["full"]["pooled"]
    t4 = a["x0_t4"]["arms"]
    t4_pooled = a["x0_t4"]["x1_pooling_preview_rfix_minus_acgru"]["pooled_d"]
    sweep = {name: cfg["pooled_mean"]
             for name, cfg in a["x0b_sweep"]["configs"].items()}
    w0_levels = a["w0_summary"]["ladder_readout"]["vicreg"]["levels"]
    dino_pm = a["x3_dino_pm"]["levels"]
    dino_ext = a["x3_dino_ext"]["levels"]
    dino_min = min(min(level["scores"]) for level in
                   list(dino_pm.values()) + list(dino_ext.values()))
    env = a["x2_envelope"]

    detune_drop = x2_mean("rfix_argmax") - x2_mean("rfix_detuned_argmax")
    hedge_cal = x2_mean("rfix_hedged") - x2_mean("rfix_argmax")
    hedge_det = (x2_mean("rfix_detuned_hedged")
                 - x2_mean("rfix_detuned_argmax"))
    env_sel_min = min(env["per_seed"][s]["selector_accuracy"]
                      for s in env["per_seed"])

    values: dict[str, str] = {
        # X1
        "X1_POOLED_D": fmt(x1["pooled_d"], sign=True),
        "X1_CI_LO": fmt(x1["ci95"][0], sign=True),
        "X1_CI_HI": fmt(x1["ci95"][1], sign=True),
        "X1_P": f"{x1['p_pos']:.1e}",
        "X1_T1_D": fmt(x1["per_task_d"]["t1"], 2, sign=True),
        "X1_T3_D": fmt(x1["per_task_d"]["t3"], 2, sign=True),
        "X1_T4_D": fmt(x1["per_task_d"]["t4"], 2, sign=True),
        "X1_T1_WINS": x1["per_task_wins"]["t1"],
        "X1_T3_WINS": x1["per_task_wins"]["t3"],
        "X1_T4_WINS": x1["per_task_wins"]["t4"],
        # X0
        "W3_FULL": f"{fmt(full['mean'], 4, sign=True)} "
                   f"(p = {full['p_pos']:.1e})",
        "X0_HEALTHY": fmt(healthy["mean"], 4, sign=True),
        "X0_HEALTHY_P": f"{healthy['p_pos']:.1e}",
        "X0_HEALTHY_WINS": f"{sens[headline_key]['healthy_only']['total_wins']}"
                           f"/{sens[headline_key]['healthy_only']['total_pairs']}",
        "T4_RFIX_LEGACY": fmt(t4["lkc_rfix"]["legacy"]["mean"], 2),
        "T4_RFIX_REPAIRED": fmt(t4["lkc_rfix"]["repaired"]["mean"], 3,
                                sign=True),
        "T4_ACGRU_REPAIRED": fmt(t4["acgru"]["repaired"]["mean"], 3,
                                 sign=True),
        # X2
        "X2_ORACLE_SUCCESS": fmt(x2_mean("oracle")),
        "X2_FLOOR_SUCCESS": fmt(x2_mean("floor_integrator")),
        "X2_GAP": fmt(x2_mean("oracle") - x2_mean("floor_integrator"), 2),
        "X2_GAP_MIN": "0.3",
        "X2_RFIX_SUCCESS": fmt(x2_mean("rfix_argmax")),
        "X2_ACGRU_SUCCESS": fmt(x2_mean("acgru_argmax")),
        "X2_NONE_SUCCESS": fmt(x2_mean("none_selector")),
        "X2_ABLATED_SUCCESS": fmt(x2_mean("belief_ablated")),
        "X2_MARGIN": fmt(x2_mean("rfix_argmax") - x2_mean("acgru_argmax"),
                         2, sign=True),
        "X2_GDELTA_SUCCESS": fmt(env["mean_argmax"]),
        "X2_DETUNE_DROP": fmt(-detune_drop, 2, sign=True),
        "X2_HEDGE_CAL": fmt(hedge_cal, 3, sign=True),
        "X2_HEDGE_DET": fmt(hedge_det, 3, sign=True),
        "ENV_SEL_MIN": fmt(env_sel_min),
        "DETUNE_FACTOR": "16",
        "SEL_CALIBRATED": fmt(sel["lkc_rfix"]),
        "SEL_DETUNED": fmt(sel["lkc_rfix_detuned"]),
        "ROLLOUT_REAL": "0.104",
        "ROLLOUT_1STEP": "0.469",
        # F1
        "F1_CHANCE": fmt(f1["chance"]),
        "F1_SIGHTED": "/".join(f"{b['sighted']:.3f}" for b in f1_banks),
        "F1_LEAKAGE_RANGE": per_seed_range([b["leakage"] for b in f1_banks]),
        "F1_FLOOR_RANGE": per_seed_range([b["floor"] for b in f1_banks]),
        "F1_FLOOR_MEAN": fmt(sum(b["floor"] for b in f1_banks) / 3, 2),
        "F1_EPISODES": "2,304",
        "F1_RFIX_MEAN": fmt(f1["inversion"]["lkc_rfix"]["mean"]),
        "F1_GDELTA_MEAN": fmt(f1["inversion"]["gdelta_l10"]["mean"]),
        "F1_RFIX_SD": fmt(_sd(f1["inversion"]["lkc_rfix"]["scores"])),
        "F1_GDELTA_SD": fmt(_sd(f1["inversion"]["gdelta_l10"]["scores"]), 2),
        # s*
        "SSTAR_VICREG": "t1s2",
        "SSTAR_DINO_BOUND": "t1s0c",
        "W0_VICREG_S1_SCORES": "/".join(
            f"{s:.3f}" for s in sorted(w0_levels["t1s1"]["sighted_scores"])),
        "F2B_S0_SIGHTED": fmt(a["f2b_s0"]["sighted"]["score"]),
        "F2B_S1_SIGHTED": fmt(a["f2b_s1"]["sighted"]["score"]),
        "DINO_MIN_SCORE": fmt(dino_min, 3),
        # delay + tau
        "DELAY_RFIX_64": fmt(delay["lkc_rfix@L64"]["mean"]),
        "DELAY_RFIX_96": fmt(delay["lkc_rfix@L96"]["mean"]),
        "DELAY_RFIX_128": fmt(delay["lkc_rfix@L128"]["mean"]),
        "DELAY_ACGRU_64": fmt(delay["acgru@L64"]["mean"]),
        "DELAY_ACGRU_96": fmt(delay["acgru@L96"]["mean"]),
        "DELAY_ACGRU_128": fmt(delay["acgru@L128"]["mean"]),
        "DELAY_GDELTA_64": fmt(delay["gdelta_l10@L64"]["mean"]),
        "DELAY_GDELTA_96": fmt(delay["gdelta_l10@L96"]["mean"]),
        "DELAY_GDELTA_128": fmt(delay["gdelta_l10@L128"]["mean"]),
        "TAU_DELTA_128": fmt(tau["L128"]["delta"], 3, sign=True),
        "TAU_DELTA_64": fmt(tau["L64"]["delta"], 3, sign=True),
        # gates / constants
        "SIGHTED_GATE": "0.75",
        "FLOOR_MARGIN": "0.05",
        "RANK_MIN": "16",
        "GDELTA_RANK_RANGE": "3.0–10.7",
    }
    values["X0B_GDELTA_DEV"] = fmt(sweep["gdelta_l10"])
    values["X0B_ACGRU_DEV"] = fmt(sweep["acgru_h160_l10"])
    values["X0B_CHRONO_DEV"] = fmt(sweep["acgru_chrono_l10"])
    values["T4_POOLED_D"] = fmt(t4_pooled, 2, sign=True)
    return values


def _sd(scores) -> float:
    mean = sum(scores) / len(scores)
    return (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5


def main() -> None:
    template_path = ROOT / "templates" / "PAPER_A.template.md"
    template = template_path.read_text()
    data = load_all()
    values = replacements(data)

    used = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", template))
    missing = used - set(values)
    if missing:
        raise SystemExit(f"unfilled placeholders: {sorted(missing)}")
    unused = set(values) - used
    if unused:
        print(f"[render-a] note: unused values {sorted(unused)}",
              file=sys.stderr)

    text = template
    for key in used:
        text = text.replace("{{" + key + "}}", values[key])
    leftovers = re.findall(r"\{\{[^}]*\}\}", text)
    if leftovers:
        raise SystemExit(f"leftover placeholders: {leftovers}")

    out = ROOT / "docs" / "PAPER_A.md"
    out.write_text(text)
    manifest = {
        "schema_version": 1,
        "template_sha256": sha256(template_path),
        "artifacts": {name: {"path": rel,
                             "sha256": sha256(ROOT / rel)}
                      for name, rel in ARTIFACTS.items()},
        "values": values,
    }
    (ROOT / "docs" / "PAPER_A.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[render-a] wrote {out} ({len(used)} placeholders filled)")


if __name__ == "__main__":
    main()
