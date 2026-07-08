"""CPU tests for V3 evidence, lock, seeds, and preview-only launcher."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.shell_game_spec import load_locked_spec  # noqa: E402
from lewm.official_tasks.shell_game_spec_v2 import load_locked_spec_v2  # noqa: E402
from lewm.official_tasks.shell_game_spec_v3 import (  # noqa: E402
    load_locked_spec_v3,
    validate_device_v3,
)
from scripts.launch_official_shell_game_capacity_v3 import (  # noqa: E402
    WAVES,
    build_plan_v3,
    parse_gpu_ids_v3,
    preview_lines_v3,
)


def test_v1_v2_remain_valid_and_v3_records_only_v2_development_failure() -> None:
    v1 = load_locked_spec()
    v2 = load_locked_spec_v2()
    v3 = load_locked_spec_v3()
    assert v1["_lock_record"]["sha256"] \
        == v2["amendment"]["parent_v1"]["lock_sha256"]
    assert v2["_lock_record"]["sha256"] \
        == v3["amendment"]["parent_v2"]["lock_sha256"]
    amendment = v3["amendment"]
    assert amendment["kind"] == "pre-formal_salience_amendment_v3"
    assert amendment["threshold_changed_from_v1_or_v2"] is False
    assert amendment["semantic_capacity_contract_changed_from_v1_or_v2"] is False
    assert amendment["carrier_definitions_changed_from_v1_or_v2"] is False
    assert amendment["formal_protocol_changed_from_v2"] is False
    assert amendment["parent_v2"]["failed_gate"] \
        == "cue_initial_slot_availability"
    assert amendment["parent_v2"]["unchanged_threshold"] == 0.75
    assert amendment["parent_v2"]["formal_data_read"] is False
    assert amendment["parent_v2"]["evidence"]["single-item"][
        "per_item_accuracy"] == [0.6625]
    assert amendment["parent_v2"]["evidence"]["two-item"][
        "per_item_accuracy"] == [0.6625, 0.6875]
    assert amendment["parent_v2"]["evidence"]["four-item"][
        "per_item_accuracy"] == [0.6625, 0.6875, 0.6583333333333333,
                                 0.6541666666666667]


def test_v3_development_and_formal_seeds_are_all_new_and_distinct() -> None:
    v1 = load_locked_spec()
    v2 = load_locked_spec_v2()
    v3 = load_locked_spec_v3()
    v1_seeds = {
        value
        for split in ("train", "validation")
        for value in (
            v1["data"][split]["base_seed"],
            v1["data"][split]["counterfactual_seed"])
    }
    v3_seeds = {
        value
        for split in ("development", "train", "validation")
        for value in (
            v3["data"][split]["base_seed"],
            v3["data"][split]["counterfactual_seed"])
    }
    assert len(v3_seeds) == 6
    assert not v1_seeds & v3_seeds
    v2_seeds = {
        value
        for split in ("development", "train", "validation")
        for value in (
            v2["data"][split]["base_seed"],
            v2["data"][split]["counterfactual_seed"])
    }
    assert not v2_seeds & v3_seeds
    selection = v3["development_selection"]
    assert selection["fit_episodes"] + selection["check_episodes"] \
        == v3["data"]["development"]["episodes"]
    assert selection["threshold"] == v3["admission"][
        "cue_initial_slot_accuracy_min"] == 0.75
    assert selection["threshold_changed_from_v1_or_v2"] is False
    assert selection["fit_indices"] == [0, 240]
    assert selection["check_indices"] == [240, 480]
    assert selection["all_stages_must_pass"] is True


def test_v3_leaves_v2_formal_contract_untouched() -> None:
    v2 = load_locked_spec_v2()
    v3 = load_locked_spec_v3()
    for section in (
            "official_host", "semantic_stages", "task_contract",
            "admission", "carrier_training"):
        assert v3[section] == v2[section]
    for split in ("train", "validation"):
        assert v3["data"][split]["episodes"] == v2["data"][split]["episodes"]
    for key in ("frame_skip", "raw_action_dim", "source_stream",
                "compression_level"):
        assert v3["data"][key] == v2["data"][key]


@pytest.mark.parametrize("device", ("cuda:1", "cuda:2"))
def test_v3_device_allowlist(device) -> None:
    assert validate_device_v3(device) == device


@pytest.mark.parametrize("raw", ("0", "3", "0,1", "2,3", "1,1"))
def test_v3_launcher_rejects_forbidden_or_duplicate_gpus(raw) -> None:
    with pytest.raises(ValueError):
        parse_gpu_ids_v3(raw)
    assert parse_gpu_ids_v3("1,2") == (1, 2)


def test_v3_preview_plan_is_complete_semantic_and_read_only(tmp_path) -> None:
    spec = copy.deepcopy(load_locked_spec_v3())
    destination = tmp_path / "v3-formal"
    spec["artifacts"]["root"] = str(destination)
    plan = build_plan_v3(spec, "all", (1, 2))
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
    lines = preview_lines_v3(plan)
    assert len(lines) == 93
    assert all("\tpending\t" in line for line in lines)
    assert not destination.exists()
