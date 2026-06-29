#!/usr/bin/env python3
"""Unit tests for the frozen ORBIT-v10 decision analysis."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_hacssm_v10 as analysis


def synthetic_rows(seeds, *, candidate=0.80):
    values = {
        "none": 1.05,
        "gru": 1.02,
        "ssm": 1.00,
        "hacssmv8": 1.00,
        "orbitv10": candidate,
        "orbitv10_noaction": 1.00,
        "orbitv10_additive": 0.90,
        "orbitv10_scaled": 0.90,
        "orbitv10_static": 0.90,
    }
    rows = []
    for environment in analysis.ENVIRONMENTS:
        for design in analysis.DESIGNS:
            for seed in seeds:
                primary = values[design]
                row = {
                    "run": f"lewm-{environment}-{design}-s{seed}",
                    "env": environment,
                    "design": design,
                    "seed": seed,
                    "trainable_parameters": 100,
                    "heldout_state_nmse": primary,
                    "clean_state_nmse": 1.0,
                    "val_pred_loss": 1.0,
                    "orbit_orthogonality_error_max": 1e-7,
                    "orbit_streaming_max_abs": 1e-7,
                    "encoder_mean_channel_variance": 0.5,
                    "encoder_covariance_effective_rank": 80.0,
                    "encoder_singleton_max_abs": 1e-7,
                    "encoder_prefix_max_abs": 1e-7,
                }
                for condition in analysis.HELDOUT_CONDITIONS:
                    row[f"{condition}_state_nmse"] = primary
                rows.append(row)
    return rows


def synthetic_convergence(rows, value=0.001):
    return [
        {
            "run": row["run"],
            "env": row["env"],
            "design": row["design"],
            "seed": row["seed"],
            "relative_improvement": value,
        }
        for row in rows
    ]


class V10AnalysisTests(unittest.TestCase):
    def test_pairwise_summary_preserves_environment_seed_cells(self):
        rows = synthetic_rows(analysis.PILOT_SEEDS)
        summary = analysis.pairwise_summary(rows, "orbitv10", "ssm")
        self.assertEqual(summary["n_pairs"], 15)
        self.assertEqual(summary["paired_wins"], 15)
        self.assertEqual(summary["environment_mean_wins"], 5)
        self.assertAlmostEqual(summary["mean_paired_relative_reduction"], 0.20)

    def test_immutable_pilot_passes_all_frozen_gates(self):
        rows = synthetic_rows(analysis.PILOT_SEEDS)
        decision = analysis.pilot_decision(rows, synthetic_convergence(rows))
        self.assertTrue(decision["pilot_screen_passed"])
        self.assertEqual(decision["decision"], "PILOT_CONFIRMATION_PASS")
        self.assertTrue(all(decision["criteria"].values()))

    def test_noaction_magnitude_failure_is_no_go(self):
        rows = synthetic_rows(analysis.PILOT_SEEDS, candidate=0.97)
        decision = analysis.pilot_decision(rows, synthetic_convergence(rows))
        self.assertFalse(decision["pilot_screen_passed"])
        self.assertFalse(
            decision["criteria"]["vs_orbitv10_noaction_reduction_ge_5pct"]
        )
        self.assertEqual(decision["decision"], "NO_GO")

    def test_failed_pilot_cannot_be_reopened_by_completion(self):
        rows = synthetic_rows(analysis.FINAL_SEEDS)
        decision = analysis.final_decision(
            rows, synthetic_convergence(rows), pilot_screen_passed=False
        )
        self.assertTrue(decision["final_gates_passed"])
        self.assertFalse(decision["end_to_end_confirmation_passed"])
        self.assertEqual(decision["decision"], "PILOT_NO_GO_FINAL_DESCRIPTIVE")

    def test_final_pass_requires_bootstrap_and_exact_quality_receipts(self):
        rows = synthetic_rows(analysis.FINAL_SEEDS)
        decision = analysis.final_decision(
            rows, synthetic_convergence(rows), pilot_screen_passed=True
        )
        self.assertTrue(decision["end_to_end_confirmation_passed"])
        self.assertTrue(decision["scoped_component_confirmation_passed"])
        self.assertFalse(decision["iclr_submission_ready"])
        self.assertGreater(decision["observed"]["bootstrap"]["ssm"]["ci90"][0], 0.0)
        self.assertGreater(decision["observed"]["bootstrap"]["hacssmv8"]["ci90"][0], 0.0)

    def test_clean_harm_guard_uses_clean_state_metric(self):
        rows = synthetic_rows(analysis.PILOT_SEEDS)
        for row in rows:
            if row["design"] == "orbitv10":
                row["clean_state_nmse"] = 1.03
        decision = analysis.pilot_decision(rows, synthetic_convergence(rows))
        self.assertFalse(decision["criteria"]["clean_harm_vs_ssm_le_2pct"])
        self.assertFalse(decision["criteria"]["clean_harm_vs_hacssmv8_le_2pct"])

    def test_private_latent_mse_cannot_change_decision(self):
        rows = synthetic_rows(analysis.PILOT_SEEDS)
        baseline = analysis.pilot_decision(rows, synthetic_convergence(rows))
        for index, row in enumerate(rows):
            row["clean_mse_first_post"] = 1e-9 if index % 2 else 1e9
        changed = analysis.pilot_decision(rows, synthetic_convergence(rows))
        self.assertEqual(baseline, changed)

    def test_crossed_bootstrap_is_deterministic(self):
        matrix = np.full((5, 5), 0.1, dtype=np.float64)
        one = analysis.crossed_bootstrap(matrix, "constant")
        two = analysis.crossed_bootstrap(matrix, "constant")
        self.assertEqual(one, two)
        self.assertTrue(math.isclose(one["ci90"][0], 0.1, abs_tol=1e-12))
        self.assertEqual(one["contract_sha256"], analysis.BOOTSTRAP_CONTRACT_SHA256)


if __name__ == "__main__":
    unittest.main()
