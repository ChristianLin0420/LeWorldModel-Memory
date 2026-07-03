"""Fast CPU tests for the V19 P3 confirmation machinery.

Covers the crossed-bootstrap resampler (mean preservation, CI coverage,
validation), Holm correction on synthetic p-values, the three-tier gate
logic (tier gating, NA propagation, fail-closed Tier-0 reporting) on the
deterministic synthetic fixture grid, the counterfactual divergence math on
synthetic trajectories, and the action-derangement construction (same
multiset, no fixed points, cue window untouched).  No GPU, no MuJoCo.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.counterfactual_v19 as cf
import scripts.gates_v19_p3 as gates


# --------------------------------------------------------------------------
# Crossed bootstrap
# --------------------------------------------------------------------------

def test_crossed_bootstrap_mean_preserved_and_ci_covers():
    rng = np.random.default_rng(0)
    matrix = 0.08 + 0.02 * rng.standard_normal((3, 5))
    result = gates.crossed_bootstrap(matrix, draws=20_000, seed=1)
    assert result["mean"] == pytest.approx(float(matrix.mean()))
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]
    # A strong all-positive effect: CI excludes zero, one-sided p is small.
    assert result["ci95_low"] > 0.0
    assert result["p_pos"] < 0.01
    assert result["draws"] == 20_000


def test_crossed_bootstrap_null_effect_covers_zero():
    rng = np.random.default_rng(3)
    matrix = 0.05 * rng.standard_normal((3, 8))
    matrix -= matrix.mean()                       # exactly centered
    result = gates.crossed_bootstrap(matrix, draws=20_000, seed=2)
    assert result["ci95_low"] < 0.0 < result["ci95_high"]
    assert result["p_two_sided"] > 0.05


def test_crossed_bootstrap_validates_input():
    with pytest.raises(ValueError):
        gates.crossed_bootstrap(np.zeros(5))              # not 2-D
    with pytest.raises(ValueError):
        gates.crossed_bootstrap(np.full((3, 3), np.nan))  # non-finite
    # Degenerate constant matrix: every resample equals the constant.
    result = gates.crossed_bootstrap(np.full((3, 4), 0.5), draws=1_000, seed=0)
    assert result["ci95_low"] == pytest.approx(0.5)
    assert result["ci95_high"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Holm correction
# --------------------------------------------------------------------------

def test_holm_worked_example():
    corrected = gates.holm({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert corrected["a"]["p_holm"] == pytest.approx(0.03)
    assert corrected["c"]["p_holm"] == pytest.approx(0.06)
    assert corrected["b"]["p_holm"] == pytest.approx(0.06)
    # Step-down: a rejects (0.01 <= 0.05/3); c fails (0.03 > 0.05/2), and the
    # chain stops so b cannot reject either.
    assert corrected["a"]["reject"] is True
    assert corrected["c"]["reject"] is False
    assert corrected["b"]["reject"] is False


def test_holm_monotone_clipped_and_edges():
    corrected = gates.holm({"x": 0.9, "y": 0.8})
    assert corrected["x"]["p_holm"] <= 1.0
    assert corrected["y"]["p_holm"] <= corrected["x"]["p_holm"]
    assert gates.holm({}) == {}
    single = gates.holm({"only": 0.02})
    assert single["only"]["p_holm"] == pytest.approx(0.02)
    assert single["only"]["reject"] is True
    with pytest.raises(ValueError):
        gates.holm({"bad": 1.5})


def test_wilson_ci_sanity():
    low, high = gates.wilson_ci(60, 120)
    assert low < 0.5 < high
    low, high = gates.wilson_ci(90, 120)          # 0.75: CI excludes 0.5
    assert low > 0.5
    with pytest.raises(ValueError):
        gates.wilson_ci(5, 0)


# --------------------------------------------------------------------------
# Action derangement
# --------------------------------------------------------------------------

def test_derangement_no_fixed_points_and_complete():
    rng = np.random.default_rng(0)
    for size in (2, 3, 7, 40):
        permutation = cf.derangement(size, rng)
        assert not np.any(permutation == np.arange(size))
        assert np.array_equal(np.sort(permutation), np.arange(size))
    with pytest.raises(ValueError):
        cf.derangement(1, rng)


def test_derange_actions_multiset_prefix_and_no_fixed_rows():
    rng = np.random.default_rng(1)
    episodes, num_actions, action_dim = 6, 30, 2
    # Unique rows per episode so "no fixed index" implies "row changed".
    actions = rng.standard_normal(
        (episodes, num_actions, action_dim)).astype(np.float32)
    boundary = np.array([8, 10, 12, 8, 9, 11], dtype=np.int64)
    swapped = cf.derange_actions(actions, boundary, np.random.default_rng(2))
    for episode in range(episodes):
        start = int(boundary[episode])
        # Cue window (prefix) untouched, byte-exact.
        assert np.array_equal(swapped[episode, :start], actions[episode, :start])
        # Same multiset over the permuted segment (hence the whole sequence).
        assert np.array_equal(np.sort(swapped[episode], axis=0),
                              np.sort(actions[episode], axis=0))
        # No fixed points: every post-boundary row moved.
        assert not np.any(np.all(swapped[episode, start:]
                                 == actions[episode, start:], axis=-1))
    # Deterministic under the same rng seed.
    again = cf.derange_actions(actions, boundary, np.random.default_rng(2))
    assert np.array_equal(swapped, again)


def test_derange_actions_rejects_short_segment():
    actions = np.zeros((1, 10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        cf.derange_actions(actions, np.array([9]), np.random.default_rng(0))
    with pytest.raises(ValueError):
        cf.derange_actions(actions, np.array([-1]), np.random.default_rng(0))


def test_boundaries_and_windows_registered_rules():
    cat_events = {"cue_on": np.array([5]), "cue_off": np.array([9])}
    assert cf.derangement_boundaries(cat_events, "cat").tolist() == [9]
    shell = {"cue_off": np.array([8]), "shuffle_off": np.array([20])}
    assert cf.derangement_boundaries(shell, "cat").tolist() == [20]
    cont_events = {"gap_on": np.array([24]), "gap_off": np.array([42])}
    assert cf.derangement_boundaries(cont_events, "cont").tolist() == [24]

    start, end = cf.decision_windows(cat_events, 64, "cat")
    assert start.tolist() == [11] and end.tolist() == [63]   # cue_off + 2
    start, end = cf.decision_windows(cont_events, 64, "cont")
    assert start.tolist() == [25] and end.tolist() == [42]
    with pytest.raises(ValueError):
        cf.decision_windows({"gap_on": np.array([40]),
                             "gap_off": np.array([30])}, 64, "cont")


# --------------------------------------------------------------------------
# Counterfactual divergence math
# --------------------------------------------------------------------------

def test_windowed_divergence_exact_values():
    episodes, length, dim = 2, 8, 3
    branch_a = np.zeros((episodes, length, dim))
    branch_b = np.zeros((episodes, length, dim))
    # Episode 0: constant offset of norm 2 inside the window [2, 5].
    branch_b[0, 2:6, 0] = 2.0
    # Episode 1: offset norm 3 only at t=4, window [4, 4].
    branch_b[1, 4, 1] = 3.0
    divergence = cf.windowed_divergence(
        branch_a, branch_b, np.array([2, 4]), np.array([5, 4]))
    assert divergence[0] == pytest.approx(2.0)
    assert divergence[1] == pytest.approx(3.0)
    # Outside-window differences do not leak in.
    branch_b[0, 0, 0] = 99.0
    divergence = cf.windowed_divergence(
        branch_a, branch_b, np.array([2, 4]), np.array([5, 4]))
    assert divergence[0] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        cf.windowed_divergence(branch_a, branch_b[:, :4], np.array([0, 0]),
                               np.array([3, 3]))


def test_spearman_bootstrap_monotone_and_null():
    rng = np.random.default_rng(0)
    ground_truth = rng.uniform(0.5, 3.0, size=64)
    latent = 2.0 * ground_truth + 0.01 * rng.standard_normal(64)
    result = cf.spearman_bootstrap(latent, ground_truth, draws=2_000, seed=1)
    assert result["rho"] > 0.95
    assert result["ci95_low"] > 0.0
    assert result["p_pos"] < 0.01
    assert result["valid_draws"] <= result["draws"]

    anti = cf.spearman_bootstrap(-latent, ground_truth, draws=2_000, seed=1)
    assert anti["rho"] < -0.95 and anti["p_pos"] > 0.99

    unrelated = cf.spearman_bootstrap(
        rng.standard_normal(64), ground_truth, draws=2_000, seed=2)
    assert unrelated["ci95_low"] < 0.0 < unrelated["ci95_high"]
    with pytest.raises(ValueError):
        cf.spearman_bootstrap(latent[:3], ground_truth[:3])


def test_spearman_bootstrap_degenerate_constant_is_na_not_crash():
    # A zero-init carrier read yields constant (zero) latent divergence: the
    # gate input must degrade to a reported NA, never a crash.
    ground_truth = np.linspace(0.1, 1.0, 16)
    result = cf.spearman_bootstrap(np.zeros(16), ground_truth, draws=200)
    assert result["rho"] is None
    assert result["status"] == "degenerate_constant_divergence"
    assert "ci95_low" not in result


def test_write_results_write_once(tmp_path):
    results = {"schema_version": 1, "task": "t1", "arm": "lkc", "seed": 0,
               "arrays": {"latent_divergence": np.arange(4.0),
                          "gt_divergence": np.arange(4.0),
                          "boundary": np.arange(4),
                          "window_start": np.arange(4),
                          "window_end": np.arange(4) + 5}}
    cf.write_results(tmp_path, dict(results), force=False)
    assert (tmp_path / cf.RESULTS_NAME).exists()
    assert (tmp_path / cf.ARRAYS_NAME).exists()
    with pytest.raises(FileExistsError):
        cf.write_results(tmp_path, dict(results), force=False)
    cf.write_results(tmp_path, dict(results), force=True)   # explicit only


# --------------------------------------------------------------------------
# Three-tier gate logic on the synthetic fixture grid
# --------------------------------------------------------------------------

def test_synthetic_full_ladder_pass(tmp_path):
    root = tmp_path / "p3"
    gates.synthetic_tree(root)
    gates.main(["--root", str(root)])

    report = json.loads((root / gates.GATES_JSON_NAME).read_text())
    assert report["tier0"]["all_pass"] is True
    assert report["tier0"]["n_expected"] == 3 * 11 * 3
    assert report["tier1"]["status"] == "PASS"
    assert report["tier1"]["primary"]["ci95_low"] > 0.0
    assert report["tier2"]["evaluated_confirmatory"] is True

    members = report["tier2"]["members"]
    for name in ("correction_useful", "transport_endo",
                 "transport_counterfactual", "gain_kfix", "gain_rfix",
                 "unobserved_evolution"):
        assert members[name]["verdict"] == "PASS", name
    for name in ("spectrum_alearn", "spectrum_a2"):
        assert members[name]["verdict"] == "REPORT", name
        assert members[name]["sided"] == "two-sided"
    assert members["unobserved_evolution"]["p_kind"] == "exact_binomial"

    outcomes = {row["row"]: row["outcome"] for row in report["claims_ladder"]}
    assert outcomes == {3: "PASS", 4: "PASS", 5: "PASS", 6: "PASS"}

    markdown = (root / gates.GATES_MD_NAME).read_text()
    for fragment in ("Tier 0", "Tier 1", "Tier 2", "Claims ladder",
                     "correction_useful", "spectrum_alearn", "REPORT",
                     "T4 fidelity caveat"):
        assert fragment in markdown, fragment
    # Refuses to overwrite real trees with fixtures.
    with pytest.raises(FileExistsError):
        gates.synthetic_tree(root)


def test_tier1_failure_gates_tier2_descriptive(tmp_path):
    root = tmp_path / "p3fail"
    gates.synthetic_tree(root, tier1_pass=False)
    report = gates.evaluate(root)
    assert report["tier1"]["status"] == "FAIL"
    assert report["tier2"]["evaluated_confirmatory"] is False
    assert report["tier2"]["label"] == "tier1_failed"
    members = report["tier2"]["members"]
    for name, member in members.items():
        if name in gates.SPECTRUM_MEMBERS:
            assert member["verdict"] == "REPORT"
        else:
            assert member["verdict"] == "DESCRIPTIVE", name
    # Numbers are still computed and reported (descriptively).
    assert members["transport_endo"]["status"] == "ok"
    outcomes = {row["row"]: row["outcome"] for row in report["claims_ladder"]}
    assert outcomes[3] == "FAIL"
    assert outcomes[4] == outcomes[5] == outcomes[6] == "DESCRIPTIVE"
    markdown = gates.render_markdown(report)
    assert "tier1_failed" in markdown


def test_na_propagation_missing_arm_and_cells(tmp_path):
    root = tmp_path / "p3na"
    gates.synthetic_tree(root)
    # Remove one intervention arm entirely and one candidate seed cell.
    shutil.rmtree(root / "t1" / "lkc_k0")
    shutil.rmtree(root / "t3" / "lkc" / "s1")
    report = gates.evaluate(root)

    # Tier 0 reports (never drops) the missing cells.
    assert report["tier0"]["all_pass"] is False
    missing = [entry for entry in report["tier0"]["failing_or_missing"]
               if entry.endswith(":missing")]
    assert "t1/lkc_k0/s0:missing" in missing
    assert "t3/lkc/s1:missing" in missing

    # The candidate cell gap drives Tier 1 (and every lkc contrast) to NA.
    assert report["tier1"]["status"] == "NA"
    members = report["tier2"]["members"]
    assert members["correction_useful"]["status"] == "NA"
    assert "t1/s0:reference" in members["correction_useful"]["missing"] or \
        any("candidate" in item or "reference" in item
            for item in members["correction_useful"]["missing"])
    # NA members leave the Holm family and are listed.
    assert "correction_useful" in report["tier2"]["holm"]["excluded_na"]
    assert report["tier2"]["label"] == "tier1_na"
    outcomes = {row["row"]: row["outcome"] for row in report["claims_ladder"]}
    assert outcomes[3] == "NA" and outcomes[4] == "NA"
    # Rendering an NA-laden report must not crash.
    markdown = gates.render_markdown(report)
    assert "NA" in markdown


def test_unobserved_evolution_binomial_gate(tmp_path):
    root = tmp_path / "p3t4"
    gates.synthetic_tree(root)
    # Force a near-chance advance fraction on every t4/lkc cell.
    for seed in (0, 1, 2):
        path = root / "t4" / "lkc" / f"s{seed}" / gates.PROBES_NAME
        payload = json.loads(path.read_text())
        payload["t4_advance"] = {"k": 61, "n": 120, "probe_seed": 0}
        path.write_text(json.dumps(payload))
    report = gates.evaluate(root)
    member = report["tier2"]["members"]["unobserved_evolution"]
    assert member["status"] == "ok"
    assert member["k"] == 183 and member["n"] == 360
    assert member["ci95_low"] < 0.5 < member["ci95_high"] or \
        member["ci95_low"] <= 0.5   # 183/360 does not clear 0.5
    assert member["verdict"] == "FAIL"


def test_envelope_requires_both_reference_arms(tmp_path):
    root = tmp_path / "p3env"
    gates.synthetic_tree(root)
    shutil.rmtree(root / "t1" / "acssm" / "s0")
    report = gates.evaluate(root)
    assert report["tier1"]["status"] == "NA"
    assert any("reference" in item
               for item in report["tier1"]["primary"]["missing"])


# --------------------------------------------------------------------------
# Auxiliary probe math (endo probe + t4 advance) on synthetic arrays
# --------------------------------------------------------------------------

def test_endo_qpos_r2_recovers_planted_signal():
    rng = np.random.default_rng(0)
    episodes, length, dim = 64, 16, 8
    qpos = rng.standard_normal((episodes, 2))
    endo_state = np.zeros((episodes, length, 4), dtype=np.float32)
    endo_state[:, -1, :2] = qpos
    prior_read = 0.05 * rng.standard_normal(
        (episodes, length, dim)).astype(np.float32)
    prior_read[:, -1, :2] += qpos                 # linearly decodable
    result = gates.endo_qpos_r2(prior_read, endo_state)
    assert result["mean"] > 0.9
    assert len(result["per_probe_seed"]) == 3
    # No signal -> R2 near zero or below.
    null = gates.endo_qpos_r2(
        0.05 * rng.standard_normal((episodes, length, dim)).astype(np.float32),
        endo_state)
    assert null["mean"] < 0.2
    with pytest.raises(ValueError):
        gates.endo_qpos_r2(prior_read, endo_state[:, :, :3])  # odd state dim


def test_t4_advance_counts_prefers_advanced_position():
    rng = np.random.default_rng(0)
    episodes, length, dim = 64, 32, 6
    xi = rng.uniform(-1, 1, size=(episodes, 2)).astype(np.float32)
    gap_off = rng.integers(20, 28, size=episodes).astype(np.int64)
    prior_read = 0.02 * rng.standard_normal(
        (episodes, length, dim)).astype(np.float32)
    prior_read[np.arange(episodes), gap_off, :2] += xi
    export = {
        "prior_read": prior_read,
        "xi": xi,
        "event_gap_off": gap_off,
        "posterior_mean": (xi + 0.01 * rng.standard_normal(
            (episodes, 2))).astype(np.float32),
        "frozen_pos": (xi + 1.0).astype(np.float32),
    }
    counts = gates.t4_advance_counts(export)
    assert counts["n"] == episodes // 2
    assert counts["k"] / counts["n"] > 0.9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
