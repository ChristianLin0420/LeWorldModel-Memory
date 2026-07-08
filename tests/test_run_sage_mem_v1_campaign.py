from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_sage_mem_v1 import FORMAL_CONFIRMATION
from scripts.run_sage_mem_v1_campaign import (
    SageMemCampaignError,
    campaign_commands,
    main,
    validate_development_audit,
)
from scripts.sage_mem_v1_spec import DEFAULT_SPEC, load_spec


def test_campaign_plan_is_exact_and_keeps_double_confirmation() -> None:
    commands = campaign_commands(DEFAULT_SPEC, resume=True)
    assert [command[command.index("--stage") + 1]
            for command in commands] == ["seal", "prepare", "full"]
    assert all("--resume" in command for command in commands)
    assert commands[-1][
        commands[-1].index("--formal-confirmation") + 1
    ] == FORMAL_CONFIRMATION


def test_campaign_preview_launches_nothing(capsys) -> None:
    assert main(["--spec", str(DEFAULT_SPEC), "--resume"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["preview"] is True
    assert value["outcomes_interpreted"] is False
    assert len(value["commands"]) == 3


def test_development_audit_validation_requires_exact_complete_grid(
        tmp_path: Path, monkeypatch) -> None:
    spec = load_spec(DEFAULT_SPEC, verify_parent_paths=False)
    monkeypatch.setattr(
        "scripts.run_sage_mem_v1_campaign.output_root",
        lambda unused: tmp_path)
    path = tmp_path / "development" / "audit_receipt.json"
    path.parent.mkdir(parents=True)
    value = {
        "study": "sage-mem-v1",
        "stage": "development-audit",
        "status": "complete",
        "registered_cells_verified": 180,
        "formal_execution_started": False,
        "selection_receipts": {
            cohort: {"sha256": "a" * 64} for cohort in spec["cohorts"]
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validate_development_audit(spec) == value
    value["registered_cells_verified"] = 179
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SageMemCampaignError, match="incomplete"):
        validate_development_audit(spec)

