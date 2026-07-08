from __future__ import annotations

import numpy as np
import pytest
import torch

from lewm.models.frozen_swap_carriers import (
    make_frozen_carrier,
    parameter_report,
)
from lewm.official_tasks.dinowm_spatial_carrier import (
    ACTION_DIM,
    PATCHES,
    VISUAL_DIM,
    balanced_accuracy_from_predictions,
    crossed_paired_bootstrap,
    endpoint_frame,
    predictor_context_for_endpoint,
    spatial_carrier_forward,
)


def test_native_parameter_match_ledger() -> None:
    report = parameter_report(VISUAL_DIM, ACTION_DIM)
    assert report["target_parameters"] == 299_520
    arms = report["arms"]
    assert arms["acgru"]["width"] == 148
    assert arms["acgru"]["parameters"] == 298_368
    assert arms["aclstm"]["width"] == 122
    assert arms["aclstm"]["parameters"] == 299_632
    assert arms["diag_ssm"]["width"] == 384
    assert arms["diag_ssm"]["parameters"] == 299_520
    assert arms["lkc_fixed_trust"]["parameters"] == 299_520


def test_none_is_exact_spatial_identity_without_permutation() -> None:
    values = torch.arange(
        2 * 4 * PATCHES * VISUAL_DIM, dtype=torch.float32).reshape(
            2, 4, PATCHES, VISUAL_DIM)
    actions = torch.randn(2, 3, ACTION_DIM)
    output = spatial_carrier_forward(
        make_frozen_carrier("none", VISUAL_DIM, ACTION_DIM), values, actions)
    assert torch.equal(output.fused_visual, values)
    assert torch.count_nonzero(output.prior_visual) == 0


@pytest.mark.parametrize("arm", ["gru", "lstm", "ssm", "fixed_trust"])
def test_future_inputs_do_not_change_earlier_spatial_carrier_outputs(
        arm: str) -> None:
    torch.manual_seed(3)
    visual = torch.randn(1, 5, PATCHES, VISUAL_DIM)
    actions = torch.randn(1, 4, ACTION_DIM)
    carrier = make_frozen_carrier(arm, VISUAL_DIM, ACTION_DIM)
    first = spatial_carrier_forward(carrier, visual, actions)
    changed_visual = visual.clone()
    changed_actions = actions.clone()
    changed_visual[:, 4] += 100
    changed_actions[:, 3] -= 100
    second = spatial_carrier_forward(carrier, changed_visual, changed_actions)
    torch.testing.assert_close(
        first.fused_visual[:, :4], second.fused_visual[:, :4])
    torch.testing.assert_close(
        first.prior_visual[:, :4], second.prior_visual[:, :4])


def test_endpoint_mapping_excludes_target_observation() -> None:
    assert endpoint_frame(3, 4) == 7
    assert endpoint_frame(3, 8) == 11
    assert endpoint_frame(3, 15) == 18
    assert predictor_context_for_endpoint(18) == (15, 16, 17)


def test_crossed_paired_bootstrap_is_paired_and_stratified() -> None:
    truth = np.tile(np.arange(4), 10)
    right = np.tile(truth, (5, 1))
    left = right.copy()
    left[:, ::4] = 1
    result = crossed_paired_bootstrap(
        left, right, truth, classes=4, draws=500, seed=7)
    assert result["mean"] == pytest.approx(-0.25)
    assert result["ci95"] == pytest.approx([-0.25, -0.25])
    assert result["ci_excludes_zero"] is True
    assert balanced_accuracy_from_predictions(right[0], truth, 4) == 1.0

