"""Focused tests for the label-free formal raw-context producer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_sage_mem_v1_raw_context_reference as producer


class _LeWMSplit:
    spatial = False

    def __init__(self, offset: int, count: int = 4,
                 frames: int = 20) -> None:
        self.episode_ids = np.arange(offset, offset + count, dtype=np.int64)
        self.fit_indices = np.arange(count, dtype=np.int64)
        self.frames = frames

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(indices)
        frame = np.arange(self.frames, dtype=np.float32)[None, :, None]
        channel = np.arange(2, dtype=np.float32)[None, None, :]
        return np.repeat(frame + channel + age / 100, len(rows), axis=0)


class _DinoBank:
    spatial = True

    def __init__(self, frames: int = 20) -> None:
        self.frames = frames
        self._ids = {
            "consumer_train": np.arange(100, 104, dtype=np.int64),
            "formal_test": np.arange(200, 204, dtype=np.int64),
        }

    def indices(self, split: str) -> np.ndarray:
        return np.arange(len(self._ids[split]), dtype=np.int64)

    def identity(self, split: str) -> dict[str, np.ndarray]:
        ids = self._ids[split]
        return {"episode_id": ids, "native_cluster_id": ids + 10_000}

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(indices)
        frame = np.arange(self.frames, dtype=np.float32)[None, :, None, None]
        patch = np.arange(3, dtype=np.float32)[None, None, :, None]
        channel = np.arange(2, dtype=np.float32)[None, None, None, :]
        value = frame + patch + channel + age / 100
        return np.repeat(value, len(rows), axis=0)


def _view(cohort: str, *, spatial: bool | None = None,
          frames: int = 20) -> producer.PreparedBankView:
    if spatial is None:
        spatial = cohort not in producer.LEWM_COHORTS
    if spatial:
        bank = _DinoBank(frames=frames)
        splits = {split: bank for split in producer.SPLITS}
    else:
        splits = {
            "consumer_train": _LeWMSplit(100, frames=frames),
            "formal_test": _LeWMSplit(200, frames=frames),
        }
    return producer.PreparedBankView(
        cohort=cohort, spatial=spatial,
        bank_manifest_sha256="b" * 64, host_hash="a" * 64,
        split_banks=splits)


def _patch_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        producer, "_load_prepared_bank",
        lambda cohort, prepared_root, split_counts: _view(cohort))


def test_sixteen_slot_lewm_representation_is_right_aligned() -> None:
    sequence = _LeWMSplit(0, count=1).features(
        15, np.asarray([0], dtype=np.int64))
    short, long = producer._slot_representation(
        sequence, spatial=False, age=15)
    short = short.reshape(1, 16, 2)
    long = long.reshape(1, 16, 2)
    assert np.all(short[:, :13] == 0)
    assert np.allclose(short[:, 13:], sequence[:, 16:19])
    assert np.allclose(long, sequence[:, 3:19])


def test_dino_representation_mean_pools_patches_and_pads_early_age() -> None:
    sequence = _DinoBank().features(4, np.asarray([0], dtype=np.int64))
    short, long = producer._slot_representation(
        sequence, spatial=True, age=4)
    pooled = sequence.mean(axis=2)
    short = short.reshape(1, 16, 2)
    long = long.reshape(1, 16, 2)
    assert np.all(short[:, :13] == 0)
    assert np.allclose(short[:, 13:], pooled[:, 4:7])
    assert np.all(long[:, :9] == 0)
    assert np.allclose(long[:, 9:], pooled[:, :7])


def test_true_twenty_frame_sequence_is_required() -> None:
    sequence = _LeWMSplit(0, count=1, frames=19).features(
        4, np.asarray([0], dtype=np.int64))
    with pytest.raises(producer.RawContextReferenceError,
                       match="true 20-frame"):
        producer._slot_representation(sequence, spatial=False, age=4)


def test_feature_artifact_is_age_major_equal_dimensional_and_label_free() -> None:
    view = _view("dinowm_pusht_token", spatial=True)
    arrays = producer.build_feature_arrays(view)
    assert set(arrays) == producer.ARRAY_KEYS
    assert not any("label" in name or "pred" in name or "mse" in name
                   for name in arrays)
    assert arrays["formal_test_short_features"].shape == (3, 4, 32)
    assert arrays["formal_test_long_features"].shape == (3, 4, 32)
    assert arrays["consumer_train_short_features"].shape == (3, 4, 32)
    assert np.array_equal(
        arrays["formal_test_evidence_age"][:, 0], np.asarray([4, 8, 15]))


def test_atomic_exact_fifty_cell_build_and_resume(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loader(monkeypatch)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "raw"
    summary = producer.produce_all(
        prepared_root=prepared, output_root=output)
    assert summary["cells"] == 50
    assert summary["formal_labels_read"] is False
    assert summary["development_outcomes_read"] is False
    assert summary["mse_emitted"] is False
    manifests = sorted(output.glob("*/seed-*/manifest.json"))
    assert len(manifests) == 50
    manifest = json.loads(manifests[0].read_text())
    assert manifest["shared_consumer_sha256"] is None
    assert manifest["consumer_contract"] == \
        "post-reveal-shared-short-long-arm-blind"
    assert manifest["formal_test_labels_read"] is False
    assert not list(output.rglob("*.partial-*"))
    resumed = producer.produce_all(
        prepared_root=prepared, output_root=output, resume=True)
    assert resumed == summary


def test_resume_rejects_tampered_feature_artifact(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loader(monkeypatch)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "raw"
    producer.produce_all(prepared_root=prepared, output_root=output)
    artifact = next(output.glob("*/seed-0/raw_context_features.npz"))
    with artifact.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(producer.RawContextReferenceError,
                       match="artifact identity"):
        producer.produce_all(
            prepared_root=prepared, output_root=output, resume=True)


def test_preview_and_execute_cli_are_explicit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "raw"
    assert producer.main([
        "--prepared-root", str(prepared),
        "--output-root", str(output)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["planned_cells"] == 50
    assert preview["execute_required"] is True
    assert not output.exists()

    _patch_loader(monkeypatch)
    assert producer.main([
        "--prepared-root", str(prepared), "--output-root", str(output),
        "--execute"]) == 0
    complete = json.loads(capsys.readouterr().out)
    assert complete["status"] == "complete-label-free"
    assert complete["cells"] == 50

