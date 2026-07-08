"""CPU tests for V2 evidence, lock, seeds, and preview-only launcher."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.shell_game_spec import load_locked_spec  # noqa: E402
from lewm.official_tasks.shell_game_spec_v2 import (  # noqa: E402
    load_locked_spec_v2,
    validate_device_v2,
)
from scripts.launch_official_shell_game_capacity_v2 import (  # noqa: E402
    WAVES,
    build_plan_v2,
    parse_gpu_ids_v2,
    preview_lines_v2,
)


def test_v1_remains_valid_and_v2_records_only_its_cue_failure() -> None:
    v1 = load_locked_spec()
    v2 = load_locked_spec_v2()
    assert v1["_lock_record"]["sha256"] \
        == v2["amendment"]["parent_v1"]["lock_sha256"]
    amendment = v2["amendment"]
    assert amendment["kind"] == "pre-formal_salience_amendment"
    assert amendment["threshold_changed_from_v1"] is False
    assert amendment["semantic_capacity_contract_changed_from_v1"] is False
    assert amendment["carrier_definitions_changed_from_v1"] is False
    assert amendment["parent_v1"]["failed_gate"] \
        == "cue_initial_slot_availability"
    assert amendment["parent_v1"]["unchanged_threshold"] == 0.75
    assert amendment["parent_v1"]["all_other_gates_passed"] is True
    assert amendment["parent_v1"]["evidence"]["single-item"][
        "per_item_accuracy"] == [0.6375]
    assert amendment["parent_v1"]["evidence"]["two-item"][
        "per_item_accuracy"] == [0.6375, 0.6458333333333334]
    assert amendment["parent_v1"]["evidence"]["four-item"][
        "per_item_accuracy"] == [0.6375, 0.6458333333333334, 0.6,
                                 0.5958333333333333]


def test_v2_development_and_formal_seeds_are_all_new_and_distinct() -> None:
    v1 = load_locked_spec()
    v2 = load_locked_spec_v2()
    v1_seeds = {
        value
        for split in ("train", "validation")
        for value in (
            v1["data"][split]["base_seed"],
            v1["data"][split]["counterfactual_seed"])
    }
    v2_seeds = {
        value
        for split in ("development", "train", "validation")
        for value in (
            v2["data"][split]["base_seed"],
            v2["data"][split]["counterfactual_seed"])
    }
    assert len(v2_seeds) == 6
    assert not v1_seeds & v2_seeds
    selection = v2["development_selection"]
    assert selection["fit_episodes"] + selection["check_episodes"] \
        == v2["data"]["development"]["episodes"]
    assert selection["threshold"] == v2["admission"][
        "cue_initial_slot_accuracy_min"] == 0.75
    assert selection["threshold_changed_from_v1"] is False


@pytest.mark.parametrize("device", ("cuda:1", "cuda:2"))
def test_v2_device_allowlist(device) -> None:
    assert validate_device_v2(device) == device


@pytest.mark.parametrize("raw", ("0", "3", "0,1", "2,3", "1,1"))
def test_v2_launcher_rejects_forbidden_or_duplicate_gpus(raw) -> None:
    with pytest.raises(ValueError):
        parse_gpu_ids_v2(raw)
    assert parse_gpu_ids_v2("1,2") == (1, 2)


def test_v2_preview_plan_is_complete_semantic_and_read_only(tmp_path) -> None:
    spec = copy.deepcopy(load_locked_spec_v2())
    destination = tmp_path / "v2-formal"
    spec["artifacts"]["root"] = str(destination)
    plan = build_plan_v2(spec, "all", (1, 2))
    assert [wave for wave, _ in plan] == list(WAVES)
    counts = {wave: len(jobs) for wave, jobs in plan}
    assert counts == {
        "development-base": 1,
        "development-stages": 3,
        "development-salience": 3,
        "formal-base": 2,
        "formal-stages": 6,
        "formal-cache": 3,
        "carriers": 75,
    }
    jobs = [job for _, cells in plan for job in cells]
    assert len(jobs) == 93
    assert all(job.device in (None, "cuda:1", "cuda:2") for job in jobs)
    assert all(not re.search(r"(^|[-_])[tT]\d+($|[-_])", job.name)
               for job in jobs)
    lines = preview_lines_v2(plan)
    assert len(lines) == 93
    assert all("\tpending\t" in line for line in lines)
    assert not destination.exists()
