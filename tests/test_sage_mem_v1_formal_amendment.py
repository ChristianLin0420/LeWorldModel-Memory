from __future__ import annotations

import copy

import pytest
import yaml

from scripts.sage_mem_v1_formal_amendment import (
    DEFAULT_AMENDMENT, FormalAmendmentError, load_formal_amendment,
)


def test_formal_amendment_is_implementation_only_and_fail_closed() -> None:
    value = load_formal_amendment()
    assert value["phase_a"]["cells"] == 600
    assert value["phase_b"]["evidence_ages_reported_separately"] == [4, 8, 15]
    assert value["causal_endpoints"]["frozen_host_output"] == "primary"
    assert value["execution"][
        "program_level_use_claim_minimum_eligible_cohorts"] == 2
    assert value["execution"]["pre_reveal_artifact"] == \
        "row x selected-class x true-target-class success cube"
    assert value["execution"]["semantic_target_indexing"].startswith(
        "only after durable")
    assert value["phase_b"]["raw_context_reference"][
        "mse_endpoint_registered"] is False
    assert value["fairness_correction"] == {
        "trigger": (
            "deterministic parameter/FLOP ledger preflight before any "
            "complete development selection or formal run"),
        "invalid_partial_grid_archived": True,
        "outcome_dependent_choice": False,
        "thresholds_or_margins_changed": False,
        "candidate_revision": "two-dense-plus-diagonal-read-v1.1",
        "candidate_parameter_formula": "D(2D+A+2)",
        "gdelta_state_dim": {"SIGReg-LeWM": 95, "DINO-WM": 191},
        "rationale": (
            "replace a twice-applied dense read with a diagonal read plus "
            "one surprise projection; choose gDelta width nearest the "
            "unchanged target"),
    }


def test_formal_amendment_rejects_outcome_dependent_change(
        tmp_path) -> None:
    value = yaml.safe_load(DEFAULT_AMENDMENT.read_text())
    changed = copy.deepcopy(value)
    changed["outcome_independence"]["thresholds_changed"] = True
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(FormalAmendmentError, match="outcome-independence"):
        load_formal_amendment(path)
