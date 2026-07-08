from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_a_context_rollout_extension import (
    paired_seed_summary,
    render_markdown,
)
from scripts.launch_paper_a_context_rollout_extension import (
    build_plan,
    build_wave_jobs,
    parse_gpu_ids,
    preview_lines,
)
from scripts.paper_a_context_rollout_extension_spec import (
    COMBINED_SEEDS,
    DEFAULT_SPEC,
    EXTENSION_SEEDS,
    ExtensionSpecError,
    expected_cells,
    load_locked_spec,
    repo_path,
    validate_device,
)
from scripts.run_paper_a_context_rollout_extension import (
    stage_paths,
    underlying_command,
)


LOCKED_SPEC_SHA256 = (
    "b97dc9eb2460490119b28392909096083acff78597da3993ff2ccd4e739426de"
)


def _flag_value(command: tuple[str, ...], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_locked_spec_authenticates_complete_parent_and_isolated_extension() -> None:
    spec = load_locked_spec()
    assert spec["_spec_record"]["sha256"] == LOCKED_SPEC_SHA256
    parent = repo_path(spec["parent"]["root"], "parent.root")
    output = repo_path(spec["output"]["root"], "output.root")
    assert parent != output
    assert parent not in output.parents
    cells = expected_cells(spec)
    assert len(cells) == 60
    assert len([cell for cell in cells if cell.source == "parent"]) == 36
    assert len([cell for cell in cells if cell.source == "extension"]) == 24
    assert {cell.seed for cell in cells if cell.source == "extension"} == {3, 4}


def test_locked_spec_rejects_a_byte_change(tmp_path: Path) -> None:
    spec_path = tmp_path / DEFAULT_SPEC.name
    lock_path = spec_path.with_suffix(".sha256")
    shutil.copyfile(DEFAULT_SPEC, spec_path)
    shutil.copyfile(DEFAULT_SPEC.with_suffix(".sha256"), lock_path)
    spec_path.write_text(spec_path.read_text() + "\n# unauthorized amendment\n")
    with pytest.raises(ExtensionSpecError, match="hash mismatch"):
        load_locked_spec(spec_path, verify_parent=False, root=tmp_path)


@pytest.mark.parametrize("device", ["cuda:0", "cuda:3", "cpu", "cuda:7"])
def test_non_extension_devices_fail_closed(device: str) -> None:
    spec = load_locked_spec(verify_parent=False)
    with pytest.raises(ExtensionSpecError):
        validate_device(spec, device)


def test_gpu_parser_accepts_only_unique_gpu_one_and_two() -> None:
    spec = load_locked_spec(verify_parent=False)
    assert parse_gpu_ids("1,2", spec) == (1, 2)
    assert parse_gpu_ids("cuda:2,cuda:1", spec) == (2, 1)
    with pytest.raises(ExtensionSpecError, match="forbidden"):
        parse_gpu_ids("0,1", spec)
    with pytest.raises(ValueError, match="duplicate"):
        parse_gpu_ids("1,1", spec)


def test_launcher_grid_is_exact_isolated_and_balanced() -> None:
    spec = load_locked_spec(verify_parent=False)
    context = build_wave_jobs(spec, "context", (1, 2), DEFAULT_SPEC)
    rollout = build_wave_jobs(spec, "rollout", (1, 2), DEFAULT_SPEC)
    assert len(context) == 16
    assert len(rollout) == 8
    assert sum(job.device == "cuda:1" for job in context) == 8
    assert sum(job.device == "cuda:2" for job in context) == 8
    assert sum(job.device == "cuda:1" for job in rollout) == 4
    assert sum(job.device == "cuda:2" for job in rollout) == 4
    output = repo_path(spec["output"]["root"], "output.root")
    parent = repo_path(spec["parent"]["root"], "parent.root")
    for job in [*context, *rollout]:
        assert output in job.done_file.parents
        assert parent not in job.done_file.parents
        assert "--execute" in job.command
        assert job.device in job.command
        assert not any(device in job.command for device in ("cuda:0", "cuda:3"))


def test_context_command_reuses_exact_parent_training_deck() -> None:
    spec = load_locked_spec(verify_parent=False)
    _, produced, _ = stage_paths(spec, "long_context", "t1", "h56", 3)
    command = underlying_command(
        spec, "long_context", "t1", "h56", 3, "cuda:1", produced)
    assert command[1] == "scripts/train_official_long_context.py"
    assert _flag_value(command, "--history-len") == "56"
    assert _flag_value(command, "--epochs") == "60"
    assert _flag_value(command, "--batch-size") == "256"
    assert _flag_value(command, "--lr") == "0.0001"
    assert _flag_value(command, "--weight-decay") == "0.001"
    assert _flag_value(command, "--grad-clip") == "1.0"
    assert _flag_value(command, "--position-init") == "interpolate"
    assert _flag_value(command, "--task-family") == "transient-marker"
    assert _flag_value(command, "--seed") == "3"
    assert _flag_value(command, "--device") == "cuda:1"
    assert "--force" not in command


def test_rollout_command_reuses_exact_parent_training_deck_and_anchor_code() -> None:
    spec = load_locked_spec(verify_parent=False)
    job_stage, produced, _ = stage_paths(
        spec, "learned_rollout", "t3", "overshoot_8", 4)
    command = underlying_command(
        spec, "learned_rollout", "t3", "overshoot_8", 4,
        "cuda:2", produced)
    assert command[1] == "scripts/train_official_rollout.py"
    assert _flag_value(command, "--task") == "t3"
    assert _flag_value(command, "--objective") == "overshoot_8"
    assert _flag_value(command, "--epochs") == "60"
    assert _flag_value(command, "--batch-size") == "64"
    assert _flag_value(command, "--lr") == "0.0001"
    assert _flag_value(command, "--weight-decay") == "0.001"
    assert _flag_value(command, "--seed") == "4"
    assert _flag_value(command, "--device") == "cuda:2"
    assert Path(_flag_value(command, "--output")) == job_stage / "root"


def test_preview_is_semantic_and_contains_all_twenty_four_jobs() -> None:
    spec = load_locked_spec(verify_parent=False)
    plan = build_plan(spec, "all", (1, 2), DEFAULT_SPEC)
    lines = preview_lines(plan)
    assert len(lines) == 24
    assert sum("Transient-marker recall" in line for line in lines) == 12
    assert sum("Drifting-color recall" in line for line in lines) == 12
    assert all("cuda:0" not in line and "cuda:3" not in line for line in lines)


def test_paired_seed_bootstrap_differences_before_resampling() -> None:
    spec = load_locked_spec(verify_parent=False)
    candidate = {seed: float(seed + 2) for seed in COMBINED_SEEDS}
    reference = {seed: float(seed) for seed in COMBINED_SEEDS}
    first = paired_seed_summary(
        spec, candidate, reference, "unit-test-pair", "candidate", "reference")
    second = paired_seed_summary(
        spec, candidate, reference, "unit-test-pair", "candidate", "reference")
    assert first == second
    assert first["n"] == 5
    assert first["seeds"] == list(COMBINED_SEEDS)
    assert first["values"] == [2.0] * 5
    assert first["mean"] == 2.0
    assert first["ci95"] == [2.0, 2.0]
    assert first["bootstrap"]["paired"] is True


def test_paper_facing_markdown_uses_only_semantic_task_names() -> None:
    statistic = {"mean": 0.25, "ci95": [0.20, 0.30]}
    summary = {
        "long_context": {"tasks": {
            "transient-marker-recall": {
                "display_name": "Transient-marker recall",
                "histories": {"3": {
                    "raw_legal_context_readout": {"value": 0.25},
                    "trained_predictor_semantic_accuracy": statistic,
                    "validation_next_latent_mse": statistic,
                }},
                "paired_vs_three-latent_context": {},
            }}},
        "learned_rollout": {"tasks": {
            "drifting-color-recall": {
                "display_name": "Drifting-color recall",
                "objectives": {"one-step": {
                    "display_name": "One-step objective",
                    "competence_gate_through_horizon_8": {"pass_count": 3},
                    "horizons": {"1": {
                        "normalized_latent_mse": statistic,
                        "model_to_copy_ratio": statistic,
                        "true_action_advantage": statistic,
                    }},
                }},
            }}},
    }
    markdown = render_markdown(summary)
    assert "Transient-marker recall" in markdown
    assert "Drifting-color recall" in markdown
    assert " T1" not in markdown and " T3" not in markdown
    assert "`t1`" not in markdown and "`t3`" not in markdown
