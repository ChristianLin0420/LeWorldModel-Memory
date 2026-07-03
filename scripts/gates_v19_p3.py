#!/usr/bin/env python3
"""Three-tier gate evaluator for the V19 P3 frozen confirmation
(docs/V19_PROPOSAL.md 4.5, section 5 claims ladder, section 10 freeze).

Reads, across ``--root`` (default outputs/v19_p3), per (task, arm, seed):
``probe_results.json`` (scripts/eval_v19_p2.py), ``gates.json`` (trainer
health), ``counterfactual_results.json`` (scripts/counterfactual_v19.py when
present) and the auxiliary ``p3_probes.json`` this script computes and caches
per run (endo-state probe + T4 per-episode advance counts, from
``eval_export.npz`` joined with the val bank's ``endo_state``).  Emits
``p3_gates.json`` and ``p3_gates.md``.

Registered structure (frozen here, before any P3 number is read):

- TIER 0 (preconditions, fail-closed REPORT): per-cell health gates from
  gates.json (rank / variance / convergence / overall).  Cells are reported
  — never silently dropped: a missing or failing cell appears in the pass
  matrix and as a study-level caveat, and analyses over incomplete grids
  return NA rather than pooling over fewer cells.
- TIER 1 (ONE primary endpoint): LKC-pure vs the action-conditioned
  recurrent envelope (per task-seed cell the better of Ac-GRU / Ac-SSM on
  the registered xi probe; both envelope arms required per cell).  Effects
  are plain paired differences (accuracy for t1/t3, R2 for t4 — the V18
  "paired relative improvement" mapping, positive favors LKC), pooled with
  a 100k-draw crossed bootstrap over tasks x seeds (the
  scripts/plot_v18_paper.py resampler generalized to a (T, S) grid).
  GATE: CI95 low > 0.  LKC-NLL runs the same contrast but is registered as
  secondary/exploratory: the Tier-1 verdict is decided by LKC-pure alone
  (no better-of selection); if LKC-NLL alone passes, that is reported
  honestly as exploratory.
- TIER 2 (mechanism gates; confirmatory only if Tier 1 passes, otherwise
  everything is computed and reported descriptively with the label
  ``tier1_failed``), Holm-corrected over the registered family:
    1. correction_useful:        lkc - lkc_k0 on xi          (one-sided >0)
    2. transport_endo:           lkc - lkc_b0 on the endo-state probe
       (ridge from prior_read[t_dec] to qpos[t_dec], the first half of
       endo_state; xi carries no action information by construction so this
       gate never uses the xi coordinate)                    (one-sided >0)
       transport_counterfactual: pooled crossed bootstrap over the per-cell
       action-swap Spearman rhos (lkc cells)                 (one-sided >0)
       xi-invariance: descriptive report from the counterfactual runs.
    3. gain_kfix / gain_rfix:    lkc - lkc_kfix, lkc - lkc_rfix on xi
                                                             (one-sided >0)
    4. spectrum_alearn / spectrum_a2: lkc - lkc_alearn, lkc - lkc_a2 on xi —
       re-admitted falsification retests, TWO-SIDED, direction + CI
       reported, no pass/fail language (verdict REPORT).
    5. unobserved_evolution (t4, lkc): per-episode indicator that the
       prior_read-probe prediction at gap_off is closer to the advanced
       (posterior-mean) position than to the frozen one, pooled over seeds
       (probe seed 0 eval half); gate: fraction > 0.5 with the Wilson CI95
       excluding 0.5; p-value from the exact one-sided binomial test (the
       one registered non-bootstrap p in the family — exact beats bootstrap
       for a binomial count).
  Bootstrap p-values use the add-one convention (#draws<=0 + 1)/(draws + 1);
  Holm runs over every Tier-2 member with an available p-value (NA members
  are excluded and listed); a gated member PASSes only if its registered CI
  condition holds AND Holm rejects at alpha = 0.05.
- Section 10 caveat carried verbatim: every T4 result inherits the frozen
  fidelity note (single-frame positional R2 ~= 0.47; velocity-limited
  extrapolation).

``--synthetic`` builds a deterministic fixture grid under an empty --root and
evaluates it — the smoke path for this evaluator and its tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.make_v19_p0_data as p0_data
from scripts.counterfactual_v19 import RESULTS_NAME as CF_RESULTS_NAME
from scripts.eval_v19_p2 import (RESULTS_NAME as PROBE_RESULTS_NAME,
                                 PROBE_SEEDS, _split, load_export)

# Registered grid + statistics constants.
CONFIRMATION_TASKS = ("t1", "t3", "t4")
ARM_DECK = ("none", "acgru", "acssm", "lkc", "lkc_nll", "lkc_k0", "lkc_b0",
            "lkc_kfix", "lkc_rfix", "lkc_alearn", "lkc_a2")
CANDIDATE_ARM = "lkc"
SECONDARY_ARM = "lkc_nll"
ENVELOPE_ARMS = ("acgru", "acssm")
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 19_019          # distinct from V18's 18018 by registration
HOLM_ALPHA = 0.05
DEFAULT_P0_DATA_ROOT = "outputs/v19_p0_a2/data"   # amendment-2 caches (§10)
PROBES_NAME = "p3_probes.json"
GATES_JSON_NAME = "p3_gates.json"
GATES_MD_NAME = "p3_gates.md"
SCHEMA_VERSION = 1
T4_CAVEAT = ("T4 fidelity caveat (proposal section 10, frozen): sighted "
             "single-frame positional R2 ~= 0.47, velocity-limited "
             "extrapolation — carried into every T4 result.")

TIER2_SIDEDNESS = {
    "correction_useful": "greater",
    "transport_endo": "greater",
    "transport_counterfactual": "greater",
    "gain_kfix": "greater",
    "gain_rfix": "greater",
    "spectrum_alearn": "two-sided",
    "spectrum_a2": "two-sided",
    "unobserved_evolution": "greater",
}
SPECTRUM_MEMBERS = ("spectrum_alearn", "spectrum_a2")


# --------------------------------------------------------------------------
# Statistics: crossed bootstrap, Holm, Wilson
# --------------------------------------------------------------------------

def crossed_bootstrap(values: np.ndarray, draws: int = BOOTSTRAP_DRAWS,
                      seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Crossed (tasks x seeds) bootstrap of the grand mean of a (T, S) grid.

    Generalizes scripts/plot_v18_paper.py ``_crossed_bootstrap`` to any grid
    shape: both axes are resampled with replacement per draw.  Returns the
    grand mean, the percentile CI95, and add-one bootstrap p-values (the
    one-sided ``p_pos`` for mean > 0 and the symmetric two-sided p).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"invalid crossed matrix with shape {values.shape}")
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = int(draws)
    while remaining:
        count = min(10_000, remaining)
        task_ids = rng.integers(0, values.shape[0],
                                size=(count, values.shape[0]))
        seed_ids = rng.integers(0, values.shape[1],
                                size=(count, values.shape[1]))
        sampled = values[task_ids[:, :, None], seed_ids[:, None, :]]
        chunks.append(sampled.mean(axis=(1, 2)))
        remaining -= count
    estimates = np.concatenate(chunks)
    at_most_zero = int((estimates <= 0.0).sum())
    at_least_zero = int((estimates >= 0.0).sum())
    p_pos = (at_most_zero + 1) / (estimates.size + 1)
    p_neg = (at_least_zero + 1) / (estimates.size + 1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025, method="linear")),
        "ci95_high": float(np.quantile(estimates, 0.975, method="linear")),
        "p_pos": float(p_pos),
        "p_two_sided": float(min(1.0, 2.0 * min(p_pos, p_neg))),
        "draws": int(estimates.size),
        "seed": int(seed),
    }


def holm(p_values: Mapping[str, float], alpha: float = HOLM_ALPHA
         ) -> dict[str, dict[str, Any]]:
    """Holm step-down correction: adjusted p-values + rejection decisions."""
    names = sorted(p_values, key=lambda name: p_values[name])
    count = len(names)
    adjusted: dict[str, dict[str, Any]] = {}
    running_max = 0.0
    rejecting = True
    for rank, name in enumerate(names):
        raw = float(p_values[name])
        if not 0.0 <= raw <= 1.0:
            raise ValueError(f"invalid p-value {raw} for {name!r}")
        adjusted_p = min(1.0, max(running_max, (count - rank) * raw))
        running_max = adjusted_p
        rejecting = rejecting and raw <= alpha / (count - rank)
        adjusted[name] = {"p": raw, "p_holm": adjusted_p,
                          "reject": bool(rejecting)}
    return adjusted


def wilson_ci(successes: int, trials: int, z: float = 1.959963984540054
              ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError(f"invalid binomial counts {successes}/{trials}")
    fraction = successes / trials
    denominator = 1.0 + z * z / trials
    center = (fraction + z * z / (2 * trials)) / denominator
    margin = (z * np.sqrt(fraction * (1 - fraction) / trials
                          + z * z / (4 * trials * trials)) / denominator)
    return float(center - margin), float(center + margin)


# --------------------------------------------------------------------------
# Per-run auxiliary probes (endo-state + T4 advance), cached as p3_probes.json
# --------------------------------------------------------------------------

def find_val_bank(task: str, data_roots: Iterable[Path]) -> Path | None:
    """Locate (never generate) the val observed bank at the current sizes."""
    train_episodes, val_episodes = p0_data.episode_sizes()
    for root in data_roots:
        paths = p0_data.task_bank_paths(root, task, train_episodes,
                                        val_episodes)
        candidate = paths["val"]["observed"]
        if p0_data._cache_valid(candidate):
            return candidate
    return None


def endo_qpos_r2(prior_read: np.ndarray, endo_state: np.ndarray,
                 probe_seeds: Iterable[int] = PROBE_SEEDS) -> dict[str, Any]:
    """Ridge R2 from prior_read[t_dec] to endo qpos[t_dec] (half/half splits).

    qpos is the first half of ``endo_state`` (concat(qpos, qvel) with equal
    dims for the reacher base).  Probe family mirrors eval_v19_p2._probe_cont
    (StandardScaler + Ridge alpha 1e-3), averaged over the probe seeds.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    episodes, length, _ = prior_read.shape
    if endo_state.shape[:2] != (episodes, length):
        raise ValueError("prior_read / endo_state episode grid mismatch")
    t_dec = length - 1
    state_dim = endo_state.shape[-1]
    if state_dim % 2:
        raise ValueError(f"endo_state dim {state_dim} is not qpos+qvel")
    features = prior_read[:, t_dec].astype(np.float64)
    target = endo_state[:, t_dec, :state_dim // 2].astype(np.float64)
    scores: list[float] = []
    for probe_seed in probe_seeds:
        train_idx, eval_idx = _split(episodes, probe_seed)
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
        probe.fit(features[train_idx], target[train_idx])
        scores.append(float(r2_score(target[eval_idx],
                                     probe.predict(features[eval_idx]))))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "per_probe_seed": scores, "coordinate": "endo_qpos_at_t_dec"}


def t4_advance_counts(export: Mapping[str, Any],
                      probe_seed: int = PROBE_SEEDS[0]) -> dict[str, int]:
    """Per-episode advanced-vs-frozen indicator counts on the eval half.

    Replicates eval_v19_p2._probe_cont for one probe seed but keeps the
    per-episode resolution: k = #(||pred - posterior_mean|| <
    ||pred - frozen_pos||), n = eval-half size.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prior_read = export["prior_read"]
    episodes = prior_read.shape[0]
    gap_off = np.asarray(export["event_gap_off"], dtype=np.int64)
    features = prior_read[np.arange(episodes), gap_off]
    train_idx, eval_idx = _split(episodes, probe_seed)
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
    probe.fit(features[train_idx], export["xi"][train_idx])
    prediction = probe.predict(features[eval_idx])
    to_posterior = np.linalg.norm(
        prediction - export["posterior_mean"][eval_idx], axis=1)
    to_frozen = np.linalg.norm(
        prediction - export["frozen_pos"][eval_idx], axis=1)
    return {"k": int((to_posterior < to_frozen).sum()),
            "n": int(eval_idx.size), "probe_seed": int(probe_seed)}


def compute_p3_probes(run_dir: Path, data_roots: Iterable[Path]
                      ) -> dict[str, Any] | None:
    """Load-or-compute the cached auxiliary probes for one run dir.

    Returns None when the export is missing (the cell then reads NA in every
    gate that needs it — fail-closed, reported).  A cached file (including
    synthetic fixtures) is trusted verbatim.
    """
    cache_path = run_dir / PROBES_NAME
    if cache_path.is_file():
        return json.loads(cache_path.read_text())
    export_path = run_dir / "eval_export.npz"
    if not export_path.is_file():
        return None
    export = load_export(export_path)
    meta = export["meta"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": meta["task"], "arm": meta["arm"], "seed": meta["seed"],
        "source": "computed",
    }
    result["t4_advance"] = (t4_advance_counts(export)
                            if meta["xi_kind"] == "cont" else None)
    bank_path = find_val_bank(str(meta["task"]), data_roots)
    if bank_path is None:
        result["endo_qpos_r2"] = None
        result["endo_note"] = "val bank not found under the data roots"
    else:
        from lewm.tasks_v19.base import load_bank
        bank = load_bank(bank_path)
        if bank.num_episodes != int(meta["episodes"]):
            raise RuntimeError(f"{run_dir}: export/bank episode mismatch "
                               f"({meta['episodes']} vs {bank.num_episodes})")
        xi_match = (np.array_equal(bank.xi, export["xi"])
                    if meta["xi_kind"] == "cat"
                    else np.allclose(bank.xi, export["xi"], atol=1e-5))
        if not xi_match:
            raise RuntimeError(f"{run_dir}: export xi differs from the val "
                               f"bank at {bank_path} — wrong bank")
        result["endo_qpos_r2"] = endo_qpos_r2(export["prior_read"],
                                              bank.endo_state)
        result["endo_bank"] = str(bank_path)
    cache_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def infer_seeds(root: Path, tasks: Iterable[str]) -> tuple[int, ...]:
    """Union of seeds present under the confirmation tasks (any arm)."""
    seeds: set[int] = set()
    for task in tasks:
        for run_dir in (root / task).glob("*/s*"):
            name = run_dir.name
            if name.startswith("s") and name[1:].isdigit():
                seeds.add(int(name[1:]))
    return tuple(sorted(seeds))


def collect_cells(root: Path, tasks: Iterable[str], arms: Iterable[str],
                  seeds: Iterable[int], data_roots: Iterable[Path]
                  ) -> dict[tuple[str, str, int], dict[str, Any]]:
    """{(task, arm, seed): {probe, gates, cf, probes}} with None for absent."""
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    data_roots = tuple(data_roots)
    for task in tasks:
        for arm in arms:
            for seed in seeds:
                run_dir = root / task / arm / f"s{seed}"
                probes = (compute_p3_probes(run_dir, data_roots)
                          if run_dir.is_dir() else None)
                cells[(task, arm, seed)] = {
                    "probe": _read_json(run_dir / PROBE_RESULTS_NAME),
                    "gates": _read_json(run_dir / "gates.json"),
                    "cf": _read_json(run_dir / CF_RESULTS_NAME),
                    "probes": probes,
                }
    return cells


# --------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------

Value = Callable[[str, int], float | None]


def paired_matrix(tasks: Iterable[str], seeds: Iterable[int],
                  candidate: Value, reference: Value | None
                  ) -> tuple[np.ndarray | None, list[str]]:
    """(T, S) effect matrix (candidate - reference, or candidate alone).

    Any missing entry makes the matrix None (fail-closed NA) and is listed.
    """
    tasks, seeds = tuple(tasks), tuple(seeds)
    matrix = np.full((len(tasks), len(seeds)), np.nan)
    missing: list[str] = []
    for row, task in enumerate(tasks):
        for column, seed in enumerate(seeds):
            value = candidate(task, seed)
            if value is None:
                missing.append(f"{task}/s{seed}:candidate")
                continue
            if reference is not None:
                ref = reference(task, seed)
                if ref is None:
                    missing.append(f"{task}/s{seed}:reference")
                    continue
                value = value - ref
            matrix[row, column] = value
    if missing or not np.isfinite(matrix).all():
        return None, missing
    return matrix, missing


def contrast(tasks: Iterable[str], seeds: Iterable[int], candidate: Value,
             reference: Value | None, sided: str) -> dict[str, Any]:
    """Pooled crossed-bootstrap contrast (or pooled statistic if reference
    is None), with per-task means and per-cell wins; NA when incomplete."""
    tasks, seeds = tuple(tasks), tuple(seeds)
    matrix, missing = paired_matrix(tasks, seeds, candidate, reference)
    if matrix is None:
        return {"status": "NA", "reason": "incomplete_cells",
                "missing": missing, "tasks": list(tasks),
                "seeds": list(seeds), "sided": sided}
    boot = crossed_bootstrap(matrix)
    return {
        "status": "ok",
        "sided": sided,
        "tasks": list(tasks),
        "seeds": list(seeds),
        "mean": boot["mean"],
        "ci95_low": boot["ci95_low"],
        "ci95_high": boot["ci95_high"],
        "p_pos": boot["p_pos"],
        "p_two_sided": boot["p_two_sided"],
        "draws": boot["draws"],
        "bootstrap_seed": boot["seed"],
        "per_task_mean": {task: float(matrix[row].mean())
                          for row, task in enumerate(tasks)},
        "cell_effects": [[float(value) for value in row] for row in matrix],
        "wins": int((matrix > 0).sum()),
        "cells": int(matrix.size),
    }


def _p_value(member: dict[str, Any], sided: str) -> float | None:
    if member.get("status") != "ok":
        return None
    return member["p_two_sided"] if sided == "two-sided" else member["p_pos"]


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------

def tier0_report(cells: Mapping[tuple[str, str, int], dict[str, Any]],
                 tasks: Iterable[str], arms: Iterable[str],
                 seeds: Iterable[int]) -> dict[str, Any]:
    """Fail-closed health matrix: every expected cell is reported."""
    rows: list[dict[str, Any]] = []
    failing: list[str] = []
    present = passing = 0
    for task in tasks:
        for arm in arms:
            for seed in seeds:
                gates = cells[(task, arm, seed)]["gates"]
                row: dict[str, Any] = {"task": task, "arm": arm, "seed": seed,
                                       "present": gates is not None}
                if gates is None:
                    failing.append(f"{task}/{arm}/s{seed}:missing")
                else:
                    present += 1
                    for key in ("rank_pass", "variance_pass",
                                "convergence_pass", "overall_pass"):
                        row[key] = bool(gates.get(key, False))
                    if row["overall_pass"]:
                        passing += 1
                    else:
                        failing.append(f"{task}/{arm}/s{seed}:health_fail")
                rows.append(row)
    expected = len(rows)
    return {
        "cells": rows,
        "n_expected": expected,
        "n_present": present,
        "n_overall_pass": passing,
        "all_pass": passing == expected,
        "failing_or_missing": failing,
        "policy": ("report-only, fail-closed: unhealthy cells are flagged "
                   "and carried as caveats, never silently dropped; "
                   "incomplete grids drive the affected contrasts to NA"),
    }


def _xi_value(cells: Mapping, arm: str) -> Value:
    def value(task: str, seed: int) -> float | None:
        probe = cells[(task, arm, seed)]["probe"]
        return None if probe is None else float(probe["registered"]["mean"])
    return value


def _envelope_value(cells: Mapping) -> Value:
    def value(task: str, seed: int) -> float | None:
        scores = []
        for arm in ENVELOPE_ARMS:
            probe = cells[(task, arm, seed)]["probe"]
            if probe is None:
                return None            # both envelope arms are required
            scores.append(float(probe["registered"]["mean"]))
        return max(scores)
    return value


def _endo_value(cells: Mapping, arm: str) -> Value:
    def value(task: str, seed: int) -> float | None:
        probes = cells[(task, arm, seed)]["probes"]
        if probes is None or probes.get("endo_qpos_r2") is None:
            return None
        return float(probes["endo_qpos_r2"]["mean"])
    return value


def _cf_rho_value(cells: Mapping, arm: str = CANDIDATE_ARM) -> Value:
    def value(task: str, seed: int) -> float | None:
        cf = cells[(task, arm, seed)]["cf"]
        if cf is None or cf.get("spearman", {}).get("rho") is None:
            return None
        return float(cf["spearman"]["rho"])
    return value


def tier1_report(cells: Mapping, tasks: Iterable[str], seeds: Iterable[int]
                 ) -> dict[str, Any]:
    envelope = _envelope_value(cells)
    primary = contrast(tasks, seeds, _xi_value(cells, CANDIDATE_ARM),
                       envelope, "greater")
    secondary = contrast(tasks, seeds, _xi_value(cells, SECONDARY_ARM),
                         envelope, "greater")
    if primary["status"] != "ok":
        status = "NA"
    else:
        status = "PASS" if primary["ci95_low"] > 0.0 else "FAIL"
    note = ("registered candidate is LKC-pure; LKC-NLL is secondary/"
            "exploratory and never substitutes for the primary")
    if (status == "FAIL" and secondary.get("status") == "ok"
            and secondary["ci95_low"] > 0.0):
        note += (" — EXPLORATORY: LKC-NLL alone clears the envelope; "
                 "reported honestly as exploratory, not confirmatory")
    return {
        "endpoint": "lkc vs action-conditioned recurrent envelope "
                    "(per-cell better of acgru/acssm) on the registered "
                    "xi probe",
        "gate_rule": "pooled crossed-bootstrap CI95 low > 0",
        "primary": primary,
        "secondary_nll": secondary,
        "status": status,
        "note": note,
    }


def tier2_report(cells: Mapping, tasks: Iterable[str], seeds: Iterable[int],
                 tier1_status: str) -> dict[str, Any]:
    tasks, seeds = tuple(tasks), tuple(seeds)
    lkc_xi = _xi_value(cells, CANDIDATE_ARM)
    members: dict[str, dict[str, Any]] = {
        "correction_useful": contrast(
            tasks, seeds, lkc_xi, _xi_value(cells, "lkc_k0"), "greater"),
        "transport_endo": contrast(
            tasks, seeds, _endo_value(cells, CANDIDATE_ARM),
            _endo_value(cells, "lkc_b0"), "greater"),
        "transport_counterfactual": contrast(
            tasks, seeds, _cf_rho_value(cells), None, "greater"),
        "gain_kfix": contrast(
            tasks, seeds, lkc_xi, _xi_value(cells, "lkc_kfix"), "greater"),
        "gain_rfix": contrast(
            tasks, seeds, lkc_xi, _xi_value(cells, "lkc_rfix"), "greater"),
        "spectrum_alearn": contrast(
            tasks, seeds, lkc_xi, _xi_value(cells, "lkc_alearn"),
            "two-sided"),
        "spectrum_a2": contrast(
            tasks, seeds, lkc_xi, _xi_value(cells, "lkc_a2"), "two-sided"),
        "unobserved_evolution": _unobserved_evolution(cells, seeds),
    }

    # Holm over every member with an available p-value.
    p_values: dict[str, float] = {}
    excluded: list[str] = []
    for name, member in members.items():
        p_value = _p_value(member, TIER2_SIDEDNESS[name])
        if p_value is None:
            excluded.append(name)
        else:
            p_values[name] = p_value
    corrected = holm(p_values) if p_values else {}

    confirmatory = tier1_status == "PASS"
    label = (None if confirmatory
             else ("tier1_failed" if tier1_status == "FAIL" else "tier1_na"))
    for name, member in members.items():
        member["gate_rule"] = _gate_rule(name)
        if name in corrected:
            member.update({key: corrected[name][key]
                           for key in ("p", "p_holm", "reject")})
        member["verdict"] = _verdict(name, member, confirmatory)

    return {
        "evaluated_confirmatory": confirmatory,
        "label": label,
        "members": members,
        "xi_invariance": _xi_invariance_report(cells, tasks, seeds),
        "holm": {"alpha": HOLM_ALPHA, "family": sorted(p_values),
                 "excluded_na": excluded},
    }


def _gate_rule(name: str) -> str:
    if name in SPECTRUM_MEMBERS:
        return ("two-sided evidence-gathering retest: direction + CI + "
                "Holm-adjusted p reported, no pass/fail")
    if name == "unobserved_evolution":
        return ("fraction > 0.5 with Wilson CI95 excluding 0.5, plus Holm "
                "rejection")
    return "CI95 low > 0, plus Holm rejection"


def _verdict(name: str, member: dict[str, Any], confirmatory: bool) -> str:
    if name in SPECTRUM_MEMBERS:
        return "REPORT"
    if member.get("status") != "ok":
        return "NA"
    if not confirmatory:
        return "DESCRIPTIVE"
    if name == "unobserved_evolution":
        ci_condition = member["ci95_low"] > 0.5 and member["fraction"] > 0.5
    else:
        ci_condition = member["ci95_low"] > 0.0
    return "PASS" if ci_condition and member.get("reject", False) else "FAIL"


def _unobserved_evolution(cells: Mapping, seeds: Iterable[int]
                          ) -> dict[str, Any]:
    """T4-only pooled advanced-vs-frozen binomial gate on the lkc arm."""
    from scipy.stats import binomtest

    successes = trials = 0
    missing: list[str] = []
    for seed in seeds:
        probes = cells[("t4", CANDIDATE_ARM, seed)]["probes"]
        if probes is None or probes.get("t4_advance") is None:
            missing.append(f"t4/s{seed}")
            continue
        successes += int(probes["t4_advance"]["k"])
        trials += int(probes["t4_advance"]["n"])
    if missing or trials == 0:
        return {"status": "NA", "reason": "incomplete_cells",
                "missing": missing, "sided": "greater"}
    low, high = wilson_ci(successes, trials)
    return {
        "status": "ok",
        "sided": "greater",
        "task": "t4",
        "arm": CANDIDATE_ARM,
        "k": successes,
        "n": trials,
        "fraction": float(successes / trials),
        "ci95_low": low,
        "ci95_high": high,
        "p_pos": float(binomtest(successes, trials, 0.5,
                                 alternative="greater").pvalue),
        "p_two_sided": float(binomtest(successes, trials, 0.5).pvalue),
        "p_kind": "exact_binomial",
        "note": T4_CAVEAT,
    }


def _xi_invariance_report(cells: Mapping, tasks: Iterable[str],
                          seeds: Iterable[int]) -> dict[str, Any]:
    """Descriptive factorization check pooled from counterfactual runs."""
    agreements: list[float] = []
    l2s: list[float] = []
    covered: list[str] = []
    missing: list[str] = []
    for task in tasks:
        for seed in seeds:
            cf = cells[(task, CANDIDATE_ARM, seed)]["cf"]
            if cf is None:
                missing.append(f"{task}/s{seed}")
                continue
            invariance = cf.get("xi_invariance", {})
            covered.append(f"{task}/s{seed}")
            if "branch_agreement" in invariance:
                agreements.append(float(invariance["branch_agreement"]))
            if "branch_mean_l2" in invariance:
                l2s.append(float(invariance["branch_mean_l2"]))
    return {
        "kind": "descriptive",
        "cells_covered": covered,
        "cells_missing": missing,
        "cat_branch_agreement_mean": (float(np.mean(agreements))
                                      if agreements else None),
        "cont_branch_l2_mean": float(np.mean(l2s)) if l2s else None,
    }


# --------------------------------------------------------------------------
# Claims ladder
# --------------------------------------------------------------------------

def _combine(*verdicts: str) -> str:
    if any(verdict == "NA" for verdict in verdicts):
        return "NA"
    if any(verdict == "DESCRIPTIVE" for verdict in verdicts):
        return "DESCRIPTIVE"
    if all(verdict == "PASS" for verdict in verdicts):
        return "PASS"
    return "FAIL"


def claims_ladder(tier1: Mapping[str, Any], tier2: Mapping[str, Any]
                  ) -> list[dict[str, Any]]:
    members = tier2["members"]
    endo = members["transport_endo"]["verdict"]
    cf = members["transport_counterfactual"]["verdict"]
    gain = _combine(members["gain_kfix"]["verdict"],
                    members["gain_rfix"]["verdict"])
    return [
        {"row": 3, "claim": "a canonical-structure carrier transports "
                            "exogenous state",
         "tier": 1, "coordinate": "xi", "outcome": tier1["status"],
         "basis": "tier1.primary (lkc vs recurrent envelope)"},
        {"row": 4, "claim": "correction is finally useful",
         "tier": 2, "coordinate": "xi",
         "outcome": members["correction_useful"]["verdict"],
         "basis": "tier2.correction_useful (lkc vs lkc_k0)"},
        {"row": 5, "claim": "action transport is the causal mechanism",
         "tier": 2, "coordinate": "endo / counterfactual",
         "outcome": _combine(endo, cf),
         "basis": "tier2.transport_endo AND tier2.transport_counterfactual; "
                  "xi-invariance reported descriptively"},
        {"row": 6, "claim": "gain adaptivity and calibration matter",
         "tier": 2, "coordinate": "xi",
         "outcome": gain,
         "basis": "tier2.gain_kfix AND tier2.gain_rfix; the LKC-NLL "
                  "calibration certificate is telemetry-based and not "
                  "evaluated by this script (reported NA here)"},
    ]


# --------------------------------------------------------------------------
# Evaluation + rendering
# --------------------------------------------------------------------------

def evaluate(root: Path, tasks: Iterable[str] = CONFIRMATION_TASKS,
             seeds: Iterable[int] | None = None,
             p0_data_root: str | Path = DEFAULT_P0_DATA_ROOT
             ) -> dict[str, Any]:
    tasks = tuple(tasks)
    if seeds is None:
        seeds = infer_seeds(root, tasks)
    seeds = tuple(seeds)
    if not seeds:
        raise FileNotFoundError(f"no run dirs under {root} for {tasks}")
    data_roots = (Path(p0_data_root), root / "data")
    cells = collect_cells(root, tasks, ARM_DECK, seeds, data_roots)
    tier0 = tier0_report(cells, tasks, ARM_DECK, seeds)
    tier1 = tier1_report(cells, tasks, seeds)
    tier2 = tier2_report(cells, tasks, seeds, tier1["status"])
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "v19-p3-three-tier-gates",
        "root": str(root),
        "grid": {"tasks": list(tasks), "arms": list(ARM_DECK),
                 "seeds": list(seeds)},
        "registration": {
            "candidate_arm": CANDIDATE_ARM,
            "secondary_arm": SECONDARY_ARM,
            "envelope_arms": list(ENVELOPE_ARMS),
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                          "p_convention": "(#draws<=0 + 1)/(draws + 1)"},
            "holm_alpha": HOLM_ALPHA,
            "tier2_sidedness": dict(TIER2_SIDEDNESS),
            "t4_caveat": T4_CAVEAT,
        },
        "tier0": tier0,
        "tier1": tier1,
        "tier2": tier2,
        "claims_ladder": claims_ladder(tier1, tier2),
    }


def _format_contrast_row(name: str, member: Mapping[str, Any]) -> str:
    if member.get("status") != "ok":
        reason = member.get("reason", "missing")
        return (f"| {name} | {member.get('sided', '-')} | NA ({reason}) | - "
                f"| - | {member['verdict']} |")
    if name == "unobserved_evolution":
        estimate = f"{member['fraction']:.4f} ({member['k']}/{member['n']})"
    else:
        estimate = f"{member['mean']:+.4f}"
    if member.get("p") is not None:
        p_column = f"{member['p']:.5f} / {member['p_holm']:.5f}"
    else:
        p_column = "-"
    return (f"| {name} | {member['sided']} | {estimate} "
            f"| [{member['ci95_low']:+.4f}, {member['ci95_high']:+.4f}] "
            f"| {p_column} | {member['verdict']} |")


def render_markdown(report: Mapping[str, Any]) -> str:
    grid = report["grid"]
    tier0 = report["tier0"]
    tier1 = report["tier1"]
    tier2 = report["tier2"]
    lines = [
        "# V19 P3 three-tier gate report",
        "",
        f"- root: `{report['root']}`",
        f"- grid: tasks {grid['tasks']} x {len(grid['arms'])} arms x seeds "
        f"{grid['seeds']}",
        f"- registration: candidate `{CANDIDATE_ARM}`, envelope "
        f"{list(ENVELOPE_ARMS)}, crossed bootstrap "
        f"{report['registration']['bootstrap']['draws']} draws "
        f"(seed {report['registration']['bootstrap']['seed']}), Holm alpha "
        f"{HOLM_ALPHA}",
        f"- {report['registration']['t4_caveat']}",
        "",
        "## Tier 0 — validity preconditions (fail-closed report)",
        "",
        f"{tier0['n_overall_pass']}/{tier0['n_expected']} cells pass all "
        f"health gates ({tier0['n_present']} present).  Policy: "
        f"{tier0['policy']}.",
        "",
        "| task | arm | seeds present | rank | variance | convergence "
        "| overall |",
        "|---|---|---|---|---|---|---|",
    ]
    by_task_arm: dict[tuple[str, str], list[dict]] = {}
    for row in tier0["cells"]:
        by_task_arm.setdefault((row["task"], row["arm"]), []).append(row)
    for (task, arm), rows in by_task_arm.items():
        present = [row for row in rows if row["present"]]
        counts = {key: sum(bool(row.get(key)) for row in present)
                  for key in ("rank_pass", "variance_pass",
                              "convergence_pass", "overall_pass")}
        lines.append(
            f"| {task} | {arm} | {len(present)}/{len(rows)} "
            f"| {counts['rank_pass']}/{len(rows)} "
            f"| {counts['variance_pass']}/{len(rows)} "
            f"| {counts['convergence_pass']}/{len(rows)} "
            f"| {counts['overall_pass']}/{len(rows)} |")
    if tier0["failing_or_missing"]:
        lines += ["", "Failing/missing cells: "
                  + ", ".join(tier0["failing_or_missing"])]
    lines += [
        "",
        "## Tier 1 — primary confirmatory endpoint",
        "",
        f"Endpoint: {tier1['endpoint']}.  Gate: {tier1['gate_rule']}.",
        "",
        "| candidate | mean | CI95 | per-task means | wins | status |",
        "|---|---|---|---|---|---|",
    ]
    for label, member, status in (
            ("lkc (registered primary)", tier1["primary"], tier1["status"]),
            ("lkc_nll (secondary/exploratory)", tier1["secondary_nll"],
             "exploratory")):
        if member.get("status") != "ok":
            lines.append(f"| {label} | NA ({member.get('reason')}) | - | - "
                         f"| - | NA |")
            continue
        per_task = ", ".join(f"{task} {mean:+.4f}" for task, mean
                             in member["per_task_mean"].items())
        lines.append(
            f"| {label} | {member['mean']:+.4f} "
            f"| [{member['ci95_low']:+.4f}, {member['ci95_high']:+.4f}] "
            f"| {per_task} | {member['wins']}/{member['cells']} "
            f"| {status} |")
    lines += ["", f"Note: {tier1['note']}.", ""]

    lines += [
        "## Tier 2 — mechanism gates (Holm-corrected family)",
        "",
    ]
    if not tier2["evaluated_confirmatory"]:
        lines += [f"Tier 1 did not pass — every Tier-2 result below is "
                  f"**descriptive** (label: `{tier2['label']}`).", ""]
    lines += [
        "| member | sided | estimate | CI95 | p / p_holm | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name in TIER2_SIDEDNESS:
        lines.append(_format_contrast_row(name, tier2["members"][name]))
    invariance = tier2["xi_invariance"]

    def _optional(value: float | None) -> str:
        return "NA" if value is None else f"{value:.4f}"

    lines += [
        "",
        "xi-invariance (descriptive factorization check on the "
        "counterfactual branches): cat branch agreement mean = "
        f"{_optional(invariance['cat_branch_agreement_mean'])}, "
        f"cont branch L2 mean = {_optional(invariance['cont_branch_l2_mean'])} "
        f"(covered: {len(invariance['cells_covered'])}, missing: "
        f"{len(invariance['cells_missing'])}).",
        "",
        f"Holm family: {tier2['holm']['family']} (alpha "
        f"{tier2['holm']['alpha']}); excluded as NA: "
        f"{tier2['holm']['excluded_na']}.",
        "",
        "## Claims ladder outcome (proposal section 5)",
        "",
        "| row | claim | tier | coordinate | outcome | basis |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["claims_ladder"]:
        lines.append(f"| {row['row']} | {row['claim']} | {row['tier']} "
                     f"| {row['coordinate']} | **{row['outcome']}** "
                     f"| {row['basis']} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Synthetic fixtures (smoke path; reused by tests/test_v19_p3.py)
# --------------------------------------------------------------------------

SYNTHETIC_XI = {
    "none": 0.30, "acgru": 0.72, "acssm": 0.70, "lkc": 0.86, "lkc_nll": 0.84,
    "lkc_k0": 0.55, "lkc_b0": 0.83, "lkc_kfix": 0.74, "lkc_rfix": 0.76,
    "lkc_alearn": 0.83, "lkc_a2": 0.79,
}
SYNTHETIC_ENDO = {arm: 0.70 for arm in ARM_DECK} | {"lkc": 0.82,
                                                    "lkc_b0": 0.35}


def synthetic_tree(root: Path, seeds: Iterable[int] = (0, 1, 2),
                   tier1_pass: bool = True, rng_seed: int = 7) -> None:
    """Write a deterministic full-grid fixture tree under an empty root."""
    root = Path(root)
    if any(root.rglob(PROBE_RESULTS_NAME)):
        raise FileExistsError(f"{root} already holds probe results — "
                              "refusing to mix synthetic fixtures in")
    rng = np.random.default_rng(rng_seed)
    seeds = tuple(seeds)
    for task in CONFIRMATION_TASKS:
        xi_kind = "cont" if task == "t4" else "cat"
        n_classes = 0 if task == "t4" else 4
        for arm in ARM_DECK:
            for seed in seeds:
                run_dir = root / task / arm / f"s{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                xi = SYNTHETIC_XI[arm] + float(rng.normal(0.0, 0.01))
                if arm == CANDIDATE_ARM and not tier1_pass:
                    xi -= 0.20
                floor = 0.28 + float(rng.normal(0.0, 0.01))
                (run_dir / "gates.json").write_text(json.dumps({
                    "task": task, "arm": arm, "seed": seed,
                    "rank_pass": True, "variance_pass": True,
                    "convergence_pass": True, "overall_pass": True,
                    "final_effective_rank": 60.0,
                    "convergence_relative_change": 0.01,
                }, indent=2, sort_keys=True))
                (run_dir / PROBE_RESULTS_NAME).write_text(json.dumps({
                    "schema_version": 1, "task": task, "arm": arm,
                    "seed": seed, "host": "vicreg", "xi_kind": xi_kind,
                    "n_classes": n_classes,
                    "chance": 0.25 if xi_kind == "cat" else 0.0,
                    "metric": "accuracy" if xi_kind == "cat" else "r2",
                    "registered": {"mean": xi, "std": 0.005,
                                   "per_probe_seed": [xi] * 3},
                    "floor": {"mean": floor, "std": 0.005,
                              "per_probe_seed": [floor] * 3},
                    "memory_advantage": xi - floor,
                }, indent=2, sort_keys=True))
                endo = SYNTHETIC_ENDO[arm] + float(rng.normal(0.0, 0.01))
                (run_dir / PROBES_NAME).write_text(json.dumps({
                    "schema_version": SCHEMA_VERSION, "task": task,
                    "arm": arm, "seed": seed, "source": "synthetic",
                    "endo_qpos_r2": {"mean": endo, "std": 0.005,
                                     "per_probe_seed": [endo] * 3,
                                     "coordinate": "endo_qpos_at_t_dec"},
                    "t4_advance": ({"k": 88 + int(rng.integers(0, 6)),
                                    "n": 120, "probe_seed": 0}
                                   if task == "t4" else None),
                }, indent=2, sort_keys=True))
                if arm in ("lkc", "lkc_b0"):
                    rho = ((0.65 if arm == "lkc" else 0.05)
                           + float(rng.normal(0.0, 0.02)))
                    (run_dir / CF_RESULTS_NAME).write_text(json.dumps({
                        "schema_version": SCHEMA_VERSION, "task": task,
                        "arm": arm, "seed": seed, "untrained": False,
                        "episodes": 64,
                        "spearman": {"rho": rho, "ci95_low": rho - 0.15,
                                     "ci95_high": rho + 0.12,
                                     "p_pos": 0.0004, "draws": 10_000},
                        "xi_invariance": (
                            {"probe_source": "eval_export",
                             "xi_kind": "cont", "branch_mean_l2": 0.05,
                             "factual_mean_l2_to_truth": 0.30}
                            if xi_kind == "cont" else
                            {"probe_source": "eval_export",
                             "xi_kind": "cat", "branch_agreement": 0.97,
                             "factual_accuracy": 0.90}),
                    }, indent=2, sort_keys=True))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/v19_p3")
    parser.add_argument("--seeds", default=None,
                        help="comma list of training seeds "
                             "(default: inferred from the tree)")
    parser.add_argument("--p0-data-root", default=DEFAULT_P0_DATA_ROOT,
                        help="amendment-2 data cache root for the endo probe")
    parser.add_argument("--synthetic", action="store_true",
                        help="build + evaluate a synthetic fixture grid "
                             "under --root (smoke)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    if args.synthetic:
        synthetic_tree(root)
        print(f"[v19-p3-gates] synthetic fixture grid written under {root}",
              flush=True)
    seeds = (tuple(int(seed) for seed in args.seeds.split(","))
             if args.seeds else None)
    report = evaluate(root, seeds=seeds, p0_data_root=args.p0_data_root)
    (root / GATES_JSON_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    (root / GATES_MD_NAME).write_text(render_markdown(report))
    ladder = {row["row"]: row["outcome"] for row in report["claims_ladder"]}
    print(f"[v19-p3-gates] tier0 {report['tier0']['n_overall_pass']}/"
          f"{report['tier0']['n_expected']} healthy | tier1 "
          f"{report['tier1']['status']} | ladder rows 3-6: "
          f"{ladder[3]}/{ladder[4]}/{ladder[5]}/{ladder[6]} | wrote "
          f"{root / GATES_JSON_NAME} and {root / GATES_MD_NAME}", flush=True)


if __name__ == "__main__":
    main()
