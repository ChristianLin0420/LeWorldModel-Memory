from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_a_evidence_age_readtime import _bootstrap
from scripts.paper_a_evidence_age import (
    combine_age_mixture,
    endpoint_features,
    read_indices,
)
from scripts.paper_a_evidence_age_spec import (
    DEFAULT_SPEC,
    EvidenceAgeSpecError,
    validate_device,
    validate_spec,
)
from scripts.prepare_paper_a_evidence_age_strict import (
    _reacher_shifted_script,
    strict_cache_path,
)


def _spec() -> dict:
    return yaml.safe_load(DEFAULT_SPEC.read_text())


def test_protocol_contract_and_cuda_zero_only() -> None:
    spec = _spec()
    validate_spec(spec, verify_parents=False)
    assert validate_device(spec, "cuda:0") == "cuda:0"
    for forbidden in ("cuda", "cuda:1", "cuda:2", "cpu"):
        with pytest.raises(EvidenceAgeSpecError):
            validate_device(spec, forbidden)


def test_protocol_rejects_age_or_training_drift() -> None:
    spec = _spec()
    changed = copy.deepcopy(spec)
    changed["read_time"]["reacher_ages"][-1] = 43
    with pytest.raises(EvidenceAgeSpecError):
        validate_spec(changed, verify_parents=False)
    changed = copy.deepcopy(spec)
    changed["strict_fixed_endpoint"]["carrier_training"]["epochs"] = 99
    with pytest.raises(EvidenceAgeSpecError):
        validate_spec(changed, verify_parents=False)


def test_read_indices_exclude_cue_and_current_observation() -> None:
    cue_off = np.asarray([10, 20], dtype=np.int64)
    q = read_indices(cue_off, 4, length=64)
    assert q.tolist() == [14, 24]
    assert np.all(q - 3 >= cue_off)
    final = read_indices(cue_off, "final", length=64)
    assert final.tolist() == [63, 63]
    with pytest.raises(ValueError):
        read_indices(cue_off, 2, length=64)


def test_endpoint_feature_order_is_context_then_prior() -> None:
    z = np.arange(2 * 8 * 2, dtype=np.float32).reshape(2, 8, 2)
    prior = z + 1000
    q = np.asarray([5, 6])
    features = endpoint_features(z, prior, q)
    assert features.shape == (2, 8)
    np.testing.assert_array_equal(features[0, :6], z[0, 2:5].reshape(-1))
    np.testing.assert_array_equal(features[0, 6:], prior[0, 5])


def test_age_mixture_preserves_registered_order() -> None:
    values = {4: np.full((2, 3), 4), 8: np.full((2, 3), 8)}
    mixed = combine_age_mixture(values, (8, 4))
    assert mixed[:, 0].tolist() == [8, 8, 4, 4]


def test_paired_bootstrap_is_deterministic_and_stratified() -> None:
    labels = np.asarray([0, 0, 1, 1])
    values = np.arange(2 * 5 * 4, dtype=np.float64).reshape(2, 5, 4)
    one = _bootstrap(values, labels, draws=200, seed=17, stratified=True)
    two = _bootstrap(values, labels, draws=200, seed=17, stratified=True)
    np.testing.assert_allclose(one[0], two[0])
    np.testing.assert_allclose(one[1], two[1])


def test_strict_reacher_shift_holds_endpoint_and_duration() -> None:
    script = {
        "xi": np.asarray([0, 1]),
        "cue_on": np.asarray([6, 14]),
        "cue_off": np.asarray([10, 20]),
    }
    shifted = _reacher_shifted_script(script, age=15, decision=63)
    assert shifted["cue_off"].tolist() == [48, 48]
    assert shifted["cue_on"].tolist() == [44, 42]
    np.testing.assert_array_equal(
        shifted["cue_off"] - shifted["cue_on"], np.asarray([4, 6]))


def test_strict_cache_namespace_is_below_new_output_root() -> None:
    spec = _spec()
    path = strict_cache_path(spec, "pusht", "transient-visual-token-recall",
                             "validation", 15)
    assert "paper_a_evidence_age_v1/strict/cache/pusht" in str(path)
