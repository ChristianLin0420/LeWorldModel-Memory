#!/usr/bin/env python3
"""Unit tests for HACSSM-v5 post-hoc diagnostic arithmetic."""

from __future__ import annotations

import unittest

import analyze_hacssm_v5_diagnostics as diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_percentile_matches_linear_interpolation(self) -> None:
        self.assertEqual(diagnostics.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(diagnostics.percentile([0.0, 1.0], 0.95), 0.95)

    def test_paired_summary_preserves_pairing(self) -> None:
        lookup = {}
        for env, candidate, reference in (
            ("env-a", (1.0, 6.0), (2.0, 4.0)),
            ("env-b", (3.0, 2.0), (3.0, 4.0)),
        ):
            for seed in (0, 1):
                lookup[(env, "candidate", seed)] = {"metric": str(candidate[seed])}
                lookup[(env, "reference", seed)] = {"metric": str(reference[seed])}
        result = diagnostics.paired_summary(
            lookup, "candidate", "reference", metric="metric",
            envs=("env-a", "env-b"), seeds=(0, 1),
        )
        expected = ((2 - 1) / 2 + (4 - 6) / 4 + (3 - 3) / 3 + (4 - 2) / 4) / 4
        self.assertAlmostEqual(result["mean_paired_relative_reduction"], expected)
        self.assertEqual(result["paired_wins"], 2)
        self.assertEqual(result["paired_ties"], 1)
        self.assertEqual(result["n_pairs"], 4)


if __name__ == "__main__":
    unittest.main()
