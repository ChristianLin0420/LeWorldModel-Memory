from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import yaml

from scripts import aggregate_paper_a_matched_color_v1_1 as aggregate
from scripts import launch_paper_a_matched_color_v1_1 as launcher
from scripts import seal_paper_a_matched_color_v1_1 as sealer
from scripts.paper_a_matched_color_v1_1_spec import (
    AGES,
    ARMS,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    validate_spec,
)
from scripts.prepare_paper_a_matched_color_v1_1 import _fresh_hdf_selections
from scripts.train_paper_a_matched_color_v1_1 import _nuisance_breakdown


def _spec() -> dict:
    value = yaml.safe_load(DEFAULT_SPEC.read_text())
    assert isinstance(value, dict)
    return value


def test_wave1_1_contract_is_adaptive_two_host_color_only() -> None:
    spec = _spec()
    validate_spec(spec, verify_inputs=False)
    assert HOSTS == ("reacher", "pusht")
    assert tuple(spec["targets"]) == ("color",)
    assert spec["cue"]["location_role"] == "exact-balanced randomized nuisance"
    assert spec["adaptive_origin"]["prior_carrier_outcomes_observed"] is False
    assert spec["adaptive_origin"]["preserve_both_prior_failures"] is True
    assert spec["admission"]["all_hosts_ages_must_pass"] is True
    assert spec["admission"]["no_carrier_training_if_any_gate_fails"] is True
    assert spec["outputs"]["root"] == "outputs/paper_a_matched_color_v1_1"


def test_wave1_1_unchanged_core_contract() -> None:
    spec = _spec()
    assert AGES == (4, 8, 15)
    assert ARMS == ("none", "gru", "lstm", "ssm", "fixed_trust")
    assert SEEDS == (0, 1, 2, 3, 4)
    assert spec["sequence"]["endpoint_feature"] == (
        "concat(z[16],z[17],z[18],prior_read[19])")
    assert spec["admission"]["cue_balanced_accuracy_min"] == .75
    assert spec["admission"]["cue_min_class_recall_min"] == .70
    assert spec["admission"]["shortcut_ceiling"] == .30
    assert spec["carrier_training"]["training_rng_offset"] == 571000
    assert spec["carrier_training"]["epochs"] == 100
    assert spec["readout"]["model"] == "StandardScaler+LogisticRegression"
    assert spec["statistics"]["bootstrap_draws"] == 20000


def test_changed_gate_or_host_is_rejected() -> None:
    spec = _spec()
    changed = copy.deepcopy(spec)
    changed["admission"]["cue_balanced_accuracy_min"] = .74
    try:
        validate_spec(changed, verify_inputs=False)
    except ValueError:
        pass
    else:
        raise AssertionError("lowered admission gate was accepted")
    changed = copy.deepcopy(spec)
    changed["hosts"].append("tworoom")
    try:
        validate_spec(changed, verify_inputs=False)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered host was accepted")


def test_push_t_selector_excludes_both_prior_unions() -> None:
    spec = _spec()
    excluded = list(range(3360))
    spec["_lock"] = {"implementation": {
        "prior_hdf_exclusions": {"pusht": {"episode_indices": excluded}},
        "v1_hdf_exclusions": {"pusht": {"episode_indices": excluded}},
    }}
    dataset = SimpleNamespace(
        frame_skip=5, num_episodes=7000,
        episode_lengths=np.full(7000, 200, dtype=np.int64))
    first = _fresh_hdf_selections(dataset, spec, "pusht")
    second = _fresh_hdf_selections(dataset, spec, "pusht")
    assert first == second and len(first) == 1680
    selected = {item.episode_index for item in first}
    assert len(selected) == 1680 and selected.isdisjoint(excluded)
    assert sum(item.split == "train" for item in first) == 1200
    assert sum(item.split == "validation" for item in first) == 480


def test_sealer_authenticates_exact_3360_episode_union() -> None:
    exclusions = sealer._prior_exclusions(_spec())
    value = exclusions["pusht"]
    assert value["count"] == 3360
    assert value["cross_screen_overlap_count"] == 0
    assert sorted(item["count"] for item in value["per_study"].values()) \
        == [1680, 1680]
    assert len(value["cache_candidates"]) == 4


def test_reacher_rng_registry_is_disjoint_and_hashed() -> None:
    value = sealer._reacher_rng_exclusion(_spec())
    assert value["prior_seeds"] == [20260741, 20260742, 20260941, 20260942]
    assert value["new_seeds"] == [20261041, 20261042]
    assert value["all_seed_values_unique"] is True
    assert len(value["registry_sha256"]) == 64


def test_launcher_has_exact_50_gpu0_cells() -> None:
    cells = launcher._carrier_commands(DEFAULT_SPEC, DEFAULT_SPEC)
    assert len(cells) == 50
    assert len({(host, arm, seed) for host, arm, seed, _, _ in cells}) == 50
    assert all(command[command.index("--device") + 1] == "cuda:0"
               for _, _, _, command, _ in cells)
    assert all("paper_a_matched_color_v1_1" in str(log)
               for _, _, _, _, log in cells)


def test_nuisance_breakdown_and_two_host_bootstrap() -> None:
    location_one = np.repeat(np.arange(4), 4)
    correct_one = np.asarray(
        [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    nuisance = _nuisance_breakdown(correct_one, location_one)
    assert nuisance["per_location_accuracy"] == [1.0, .5, .25, 0.0]
    assert nuisance["worst_location_accuracy"] == 0.0

    correct = np.zeros((2, 5, 5, 3, 480), dtype=np.float32)
    joint = np.tile(np.repeat(np.arange(16), 30), (2, 1))
    location = joint % 4
    for host in range(2):
        for arm in range(5):
            for nuisance_index in range(4):
                rows = np.flatnonzero(location[host] == nuisance_index)[
                    :20 + arm + nuisance_index]
                correct[host, arm, :, :, rows] = 1.0
    first = aggregate._bootstrap(correct, joint, location, draws=16, seed=9)
    second = aggregate._bootstrap(correct, joint, location, draws=16, seed=9)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    point, samples, location_point, location_samples = first
    assert point.shape == (2, 5, 3)
    assert samples.shape == (16, 2, 5, 3)
    assert location_point.shape == (2, 5, 3, 4)
    assert location_samples.shape == (16, 2, 5, 3, 4)


def test_sealer_locks_every_formal_producer() -> None:
    required = {
        "scripts/paper_a_matched_color_v1_1_spec.py",
        "scripts/prepare_paper_a_matched_color_v1_1.py",
        "scripts/train_paper_a_matched_color_v1_1.py",
        "scripts/aggregate_paper_a_matched_color_v1_1.py",
        "scripts/launch_paper_a_matched_color_v1_1.py",
        "scripts/seal_paper_a_matched_color_v1_1.py",
        "scripts/prepare_paper_a_matched_host.py",
        "tests/test_paper_a_matched_color_v1_1.py",
    }
    assert required.issubset(sealer.PRODUCERS)
