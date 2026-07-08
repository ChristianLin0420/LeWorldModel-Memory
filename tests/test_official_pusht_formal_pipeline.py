"""CPU contract tests for the locked official-PushT formal pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.pusht_admission import (  # noqa: E402
    evaluate_pusht_admission,
)
from lewm.official_tasks.pusht_hdf5 import (  # noqa: E402
    OFFICIAL_PUSHT_EXTRACTED_HDF5,
)
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    aligned_pusht_latents,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    load_locked_pusht_spec,
    validate_pusht_device,
)
from scripts.cache_official_pusht_memory import (  # noqa: E402
    _encode_frame_stream,
)
from scripts.launch_official_pusht_memory import (  # noqa: E402
    build_pusht_plan,
    parse_pusht_gpu_ids,
    preview_pusht_plan,
)


def _admission_arrays(episodes: int, labels: np.ndarray,
                      *, episode_offset: int) -> tuple[dict, dict]:
    classes = int(labels.max()) + 1
    cue = np.eye(classes, dtype=np.float32)[labels]
    cue = np.repeat(cue[:, None, :], 3, axis=1)
    base = {
        "z_base": np.zeros((episodes, 20, 8), dtype=np.float32),
        "actions": np.zeros((episodes, 19, 10), dtype=np.float32),
        "state": np.zeros((episodes, 20, 7), dtype=np.float32),
        "episode_index": np.arange(
            episode_offset, episode_offset + episodes, dtype=np.int64),
        "local_start": np.zeros(episodes, dtype=np.int64),
    }
    task = {
        "z_cue": cue,
        "labels": labels,
        "episode_index": base["episode_index"].copy(),
        "local_start": base["local_start"].copy(),
    }
    return base, task


def test_frozen_admission_requires_cue_and_rejects_shortcuts() -> None:
    train_y = np.tile(np.arange(4, dtype=np.int64), 100)
    validation_y = np.tile(np.arange(4, dtype=np.int64), 50)
    train_base, train_task = _admission_arrays(
        len(train_y), train_y, episode_offset=0)
    validation_base, validation_task = _admission_arrays(
        len(validation_y), validation_y, episode_offset=10_000)
    report = evaluate_pusht_admission(
        task_key="transient-visual-token-recall",
        semantic_name="PushT transient visual-token recall", classes=4,
        train_base=train_base, train_task=train_task,
        validation_base=validation_base, validation_task=validation_task)
    assert report["admitted"] is True
    assert report["gates"]["cue_availability"]["value"][
        "balanced_accuracy"] == 1.0
    assert report["gates"]["final_context_action_shortcut"]["value"][
        "balanced_accuracy"] == 0.25

    train_base["actions"][:, 15, :4] = np.eye(4)[train_y]
    validation_base["actions"][:, 15, :4] = np.eye(4)[validation_y]
    leaked = evaluate_pusht_admission(
        task_key="transient-visual-token-recall",
        semantic_name="PushT transient visual-token recall", classes=4,
        train_base=train_base, train_task=train_task,
        validation_base=validation_base, validation_task=validation_task)
    assert leaked["admitted"] is False
    assert leaked["gates"]["final_context_action_shortcut"]["pass"] is False


def test_aligned_latents_replace_only_the_cue() -> None:
    base = {
        "z_base": np.zeros((2, 20, 3), dtype=np.float32),
        "episode_index": np.asarray([1, 2]),
        "local_start": np.asarray([4, 5]),
    }
    task = {
        "z_cue": np.ones((2, 3, 3), dtype=np.float32),
        "episode_index": np.asarray([1, 2]),
        "local_start": np.asarray([4, 5]),
    }
    combined = aligned_pusht_latents(base, task, 1, 3)
    np.testing.assert_array_equal(combined[:, 1:4], 1)
    np.testing.assert_array_equal(combined[:, :1], 0)
    np.testing.assert_array_equal(combined[:, 4:], 0)
    np.testing.assert_array_equal(base["z_base"], 0)


class _MeanEncoder(torch.nn.Module):
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        mean = pixels.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        return mean.repeat(1, 192)


def test_stream_encoder_is_single_pass_and_rejects_duplicate_positions() -> None:
    frames = [np.full((8, 8, 3), index, dtype=np.uint8)
              for index in range(5)]
    encoded = _encode_frame_stream(
        _MeanEncoder(), iter(enumerate(frames)), total_frames=5,
        frame_batch_size=2, image_size=8, device=torch.device("cpu"))
    assert encoded.shape == (5, 192)
    assert np.unique(encoded[:, 0]).size == 5
    with pytest.raises(ValueError, match="duplicate"):
        _encode_frame_stream(
            _MeanEncoder(), iter(((0, frames[0]), (0, frames[1]))),
            total_frames=2, frame_batch_size=2, image_size=8,
            device=torch.device("cpu"))


def test_locked_spec_pins_real_data_and_causal_predecision_endpoint() -> None:
    spec = load_locked_pusht_spec()
    assert spec["dataset"]["hdf5_path"] \
        == "outputs/paper_a_strengthening/data/pusht_expert_train.h5"
    assert spec["dataset"]["hdf5_sha256"] \
        == OFFICIAL_PUSHT_EXTRACTED_HDF5.sha256
    assert spec["dataset"]["hdf5_size"] \
        == OFFICIAL_PUSHT_EXTRACTED_HDF5.size
    assert spec["official_host"]["bundle_path"] \
        == "outputs/paper_a_strengthening/pretrained/lewm-pusht"
    sequence = spec["sequence"]
    assert sequence["decision_index"] == 19
    assert sequence["decision_observation_excluded"] is True
    assert sequence["final_context_indices"] == [16, 17, 18]
    assert max(sequence["final_context_indices"]) < sequence["decision_index"]
    assert sequence["context_cause_action_indices"] == [15, 16, 17]
    assert sequence["decision_prior_action_index"] == 18
    assert sequence["action_alignment"].startswith("action[t]")


def test_launcher_is_preview_only_and_rejects_gpus_zero_and_three(
        tmp_path: Path) -> None:
    spec = load_locked_pusht_spec()
    for forbidden in ("0", "3", "cuda:0", "cuda:3"):
        with pytest.raises(ValueError, match="permit only"):
            parse_pusht_gpu_ids(forbidden)
    assert parse_pusht_gpu_ids("1,2") == (1, 2)
    assert validate_pusht_device("cuda:2") == "cuda:2"
    plan = build_pusht_plan(spec, "all", (1, 2))
    assert [len(jobs) for _, jobs in plan] == [1, 2, 50]
    jobs = [job for _, wave_jobs in plan for job in wave_jobs]
    assert all(job.device in {"cuda:1", "cuda:2"} for job in jobs)
    assert all("--execute" in job.command for job in jobs)
    before = set(tmp_path.iterdir())
    lines = preview_pusht_plan(plan)
    assert len(lines) == 53
    assert set(tmp_path.iterdir()) == before

