#!/usr/bin/env python3
"""Unit tests for HACSSM-v6 post-hoc diagnostic arithmetic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import analyze_hacssm_v6_diagnostics as diagnostics


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

    def test_canonical_state_and_action_distance(self) -> None:
        reference = {
            "mem_hacsmv4.W_a.weight": torch.tensor([[1.0, 2.0], [2.0, 1.0]]),
            "mem_hacsmv4.W_x.weight": torch.tensor([[3.0]]),
            "predictor.weight": torch.tensor([[4.0]]),
        }
        canonical = diagnostics.canonical_state(reference)
        self.assertIn("mem_hacssmv6.W_a.weight", canonical)
        exact = diagnostics.state_distances(canonical, canonical)
        self.assertTrue(exact["state_exact"])
        self.assertEqual(exact["action_weight_delta_l2_vs_noaux"], 0.0)
        changed = {key: value.clone() for key, value in canonical.items()}
        changed["mem_hacssmv6.W_a.weight"][0, 0] += 3.0
        changed["predictor.weight"][0, 0] += 4.0
        observed = diagnostics.state_distances(changed, canonical)
        self.assertFalse(observed["state_exact"])
        self.assertEqual(observed["action_weight_delta_l2_vs_noaux"], 3.0)
        self.assertEqual(observed["nonaction_parameter_delta_l2_vs_noaux"], 4.0)

    def test_atomic_json_is_deterministic_and_does_not_touch_neighbor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locked = root / "locked.json"
            output = root / "output.json"
            locked.write_text('{"locked": true}\n')
            before = diagnostics.sha256(locked)
            value = {"b": [2, 1], "a": 3}
            diagnostics.atomic_json(output, value)
            first = output.read_bytes()
            diagnostics.atomic_json(output, value)
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(diagnostics.sha256(locked), before)


if __name__ == "__main__":
    unittest.main()
