from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from scripts import run_lewm_v8_v18 as run


@pytest.fixture(autouse=True)
def restore_inherited_runner_contract():
    yield
    run._ACTIVE_INTEGRITY_GUARD = None
    run._restore_contract()


def test_frozen_grid_and_gpu_queues() -> None:
    run._install_contract()
    assert len(run.TASKS) == 5
    assert len(run.DESIGNS) == 8
    assert len(run.SEEDS) == 5
    assert run.EPOCHS == 100
    assert len(run.base.cell_specs()) == 200
    queued = [task for queue in run.TASK_QUEUES for task in queue]
    assert len(queued) == len(set(queued)) == 5
    assert set(queued) == set(run.TASKS)
    assert run.TASK_QUEUES[0] == (
        "acrobot.swingup", "stacker.stack_4")


def test_protocol_records_every_cell_and_assignment(
        monkeypatch: pytest.MonkeyPatch) -> None:
    run._install_contract()
    monkeypatch.setattr(run.base, "git_receipt", lambda: {
        "git_branch": "test", "git_commit": "abc", "git_upstream_commit": "abc",
        "git_worktree_clean": True, "git_head_pushed": True,
        "git_status_sha256": "status", "git_clean_or_pushed_required": False,
    })
    payload = run.protocol_payload(
        python="/tmp/frozen-python",
        output_root=Path("/tmp/v18-output"),
        log_root=Path("/tmp/v18-log"),
        study=run.DEFAULT_STUDY,
        epochs=run.EPOCHS,
        gpu_ids=("0", "1", "2", "3"),
        wandb=True,
        data={task: {} for task in run.TASKS},
        source={"source": "sha256"},
    )
    assert payload["runs"] == 200
    assert len(payload["commands"]) == 200
    assert payload["task_pinned_gpu"] == {
        "acrobot.swingup": "0",
        "stacker.stack_4": "0",
        "manipulator.bring_ball": "1",
        "quadruped.run": "2",
        "swimmer.swimmer15": "3",
    }
    assert payload["gpu_task_queues"]["0"] == [
        "acrobot.swingup", "stacker.stack_4"]
    assert payload["candidate_ssl_selectable_hyperparameters"] == []
    assert payload["executed_return_claim_permitted"] is False
    assert all(
        command["argv"][1].endswith("train_lewm_v8_v18.py")
        for command in payload["commands"])


def test_train_command_is_frozen() -> None:
    run._install_contract()
    command = run.train_command(
        ".venv/bin/python", Path("outputs/test"), run.DEFAULT_STUDY,
        run.EPOCHS, run.TASKS[0], run.DESIGNS[0], run.SEEDS[0], wandb=True)
    assert command[command.index("--epochs") + 1] == "100"
    assert command[command.index("--history-len") + 1] == "3"
    assert command[command.index("--sigreg-lambda") + 1] == "0.0"
    assert command[command.index("--corruption-seed") + 1] == str(
        run.data.DEFAULT_CORRUPTION_SEED)
    assert "--wandb" in command
    assert "--no-wandb" not in command


def test_integrity_guard_detects_source_and_cache_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    manifest = tmp_path / "manifest.json"
    sidecar = tmp_path / "manifest.sha256"
    for path, payload in (
            (source, b"source"), (train, b"train"), (val, b"val"),
            (manifest, b"manifest"), (sidecar, b"sidecar")):
        path.write_bytes(payload)
    cohort = {
        task: {
            "train": str(train), "train_sha256": run.base.file_sha256(train),
            "train_content_sha256": "train-content",
            "val": str(val), "val_sha256": run.base.file_sha256(val),
            "val_content_sha256": "val-content",
        }
        for task in run.TASKS
    }
    cohort["__manifest__"] = {
        "path": str(manifest), "path_sha256": run.base.file_sha256(manifest),
        "sidecar": str(sidecar), "sidecar_sha256": run.base.file_sha256(sidecar),
    }
    guard = run.ProtocolIntegrityGuard(
        {str(source): run.base.file_sha256(source)}, cohort)
    guard.assert_all()
    source.write_bytes(b"tampered-source")
    with pytest.raises(run.base.ArtifactError, match="source changed"):
        guard.assert_before_cell(run.TASKS[0])


def test_failed_first_task_prevents_second_task_launch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    ledger = SimpleNamespace(records={}, lock=threading.Lock())

    class _Guard:
        def assert_task_data(self, task: str, *, full_hash: bool) -> None:
            assert full_hash is True

    def fake_queue(gpu, task, **kwargs):
        del gpu, kwargs
        calls.append(task)
        for seed in run.SEEDS:
            for design in run.DESIGNS:
                key = run.base.cell_key(task, design, seed)
                ledger.records[key] = {
                    "task": task, "design": design, "seed": seed,
                    "status": "complete"}
        ledger.records[run.base.cell_key(task, run.DESIGNS[0], run.SEEDS[0])][
            "status"] = "failed"

    run._ACTIVE_INTEGRITY_GUARD = _Guard()
    monkeypatch.setattr(run.base, "_run_task_queue", fake_queue)
    with pytest.raises(run.base.ArtifactError, match="task barrier failed"):
        run._run_gpu_queue(
            "0", run.TASK_QUEUES[0], python="python",
            output_root=Path("/tmp/out"), log_root=Path("/tmp/log"),
            study=run.DEFAULT_STUDY, epochs=run.EPOCHS, wandb=True,
            resume=False, ledger=ledger)
    assert calls == ["acrobot.swingup"]


def test_confirmation_rejects_disabled_wandb() -> None:
    with pytest.raises(ValueError, match="requires finished online W&B"):
        run.main(["--no-wandb"])


def test_protocol_rejects_dirty_uncommitted_source(
        monkeypatch: pytest.MonkeyPatch) -> None:
    run._install_contract()
    monkeypatch.setattr(run.base, "git_receipt", lambda: {
        "git_commit": "abc", "git_worktree_clean": False})
    with pytest.raises(RuntimeError, match="committed in a clean worktree"):
        run.protocol_payload(
            python="python", output_root=Path("/tmp/out"),
            log_root=Path("/tmp/log"), study=run.DEFAULT_STUDY,
            epochs=run.EPOCHS, gpu_ids=("0", "1", "2", "3"), wandb=True,
            data={}, source={})


def test_partial_or_stale_analysis_bundle_is_rejected(tmp_path: Path) -> None:
    run._install_contract()
    (tmp_path / "confirmation_analysis.json").write_text(
        '{"status":"INCOMPLETE_OR_INVALID"}\n', encoding="utf-8")
    with pytest.raises(run.base.ArtifactError, match="partial"):
        run.validate_analysis_bundle(tmp_path)
    (tmp_path / "confirmation_cells.csv").write_text("task,seed,design\n")
    (tmp_path / "confirmation_contrasts.csv").write_text("contrast,metric\n")
    with pytest.raises(run.base.ArtifactError, match="stale or invalid"):
        run.validate_analysis_bundle(tmp_path)
