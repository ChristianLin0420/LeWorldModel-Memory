#!/usr/bin/env python3
"""Host adapters for SAGE-Mem development and label-free formal Phase A.

This module deliberately has a narrower authority than the formal runner.  It
can smoke-test the model contract and train development cells on deterministic
subsets of existing parent *TRAIN* feature caches.  Its formal authority is
limited to fresh-bank preparation and label-free carrier Phase A.  Semantic
labels, consumers, correctness, and physical execution remain exclusively in
the complete-grid finalizer.

The adapters reuse the authenticated parent cache loaders and frozen host
loaders.  Candidate SAGE-Mem arms consume native ``(B,L,D)`` LeWM or
``(B,L,196,D)`` DINO tensors.  Existing baselines retain their registered
carrier implementations, including the tied-patch DINO wrapper.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

from lewm.models.frozen_swap_carriers import make_frozen_carrier
from lewm.models.v21_carriers import GatedDeltaCell, gdelta_parameter_count
from lewm.official_tasks.dinowm_spatial_carrier import (
    spatial_carrier_forward,
)
from scripts.prepare_sage_mem_v1_development import SOURCES
from scripts.sage_mem_v1_spec import (
    AGES, ARMS, COHORTS, canonical_json, output_root, spec_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
SAGE_MEM_HOST_ADAPTER_API_VERSION = "sage_mem_v1_host_adapter_v1"
FORMAL_PENDING_MESSAGE = (
    "formal SAGE-Mem execution is pending reviewed, parent-disjoint fresh-bank "
    "builders; development parent-TRAIN caches cannot be promoted to formal "
    "evidence"
)
# The registered effective batch remains 64.  Spatial hosts accumulate that
# batch in memory-only chunks; 32 is conservative on the assigned 98 GB
# Blackwell GPUs while avoiding the severe launch overhead of the original
# chunk size of four.  Evaluation is gradient-free and can use a larger chunk.
SPATIAL_TRAIN_MICRO_BATCH = 32
SPATIAL_EVAL_BATCH = 32
PHASE_A_SCHEMA = "sage_mem_v1_phase_a_cell_v1"
PHASE_A_HISTORY_SCHEMA = "sage_mem_v1_phase_a_history_v1"
PHASE_A_RESOURCE_SCHEMA = "sage_mem_v1_phase_a_resources_v1"

_COHORTS: dict[str, dict[str, Any]] = {
    "lewm_reacher_color": {
        "family": "lewm", "host": "reacher", "embed_dim": 192,
        "action_dim": 10, "tokens": 1, "task": "color", "classes": 4,
    },
    "lewm_pusht_color": {
        "family": "lewm", "host": "pusht", "embed_dim": 192,
        "action_dim": 10, "tokens": 1, "task": "color", "classes": 4,
    },
    "dinowm_pusht_token": {
        "family": "dinowm_pusht", "embed_dim": 384, "action_dim": 10,
        "tokens": 196, "task": "transient-visual-token-recall", "classes": 4,
    },
    "dinowm_pusht_binding": {
        "family": "dinowm_pusht", "embed_dim": 384, "action_dim": 10,
        "tokens": 196, "task": "multi-item-visual-binding-recall",
        "classes": 6,
    },
    "dinowm_pointmaze_goal": {
        "family": "dinowm_pointmaze", "embed_dim": 384, "action_dim": 10,
        "tokens": 196, "task": "delayed-goal-recall", "classes": 4,
    },
}


class FormalIntegrationPending(RuntimeError):
    """A formal-only operation was requested from a development adapter."""


class DevelopmentAdapterError(RuntimeError):
    """A development cache, model, or artifact violated its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _split_indices(count: int, seed: int, *, groups: int = 1
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Label-blind 75/25 split, optionally expanded within intact groups."""

    if count < 8 or groups < 1:
        raise DevelopmentAdapterError("development split is too small")
    order = np.random.default_rng(seed).permutation(count)
    fit_count = max(1, min(count - 1, int(math.floor(0.75 * count))))
    fit_base, readout_base = np.sort(order[:fit_count]), np.sort(order[fit_count:])
    if groups == 1:
        return fit_base, readout_base

    def expand(values: np.ndarray) -> np.ndarray:
        return (values[:, None] * groups
                + np.arange(groups, dtype=np.int64)[None]).reshape(-1)

    return expand(fit_base), expand(readout_base)


class _DevelopmentBank:
    """Small interface over selected parent TRAIN rows."""

    spatial: bool
    count: int
    fit_indices: np.ndarray
    readout_indices: np.ndarray
    labels: np.ndarray
    episode_ids: np.ndarray
    parent_config: Mapping[str, Any]
    parent_lock: Mapping[str, Any] | None

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def actions(self, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def proprio(self, indices: np.ndarray) -> np.ndarray | None:
        return None


class _LeWMBank(_DevelopmentBank):
    spatial = False

    def __init__(self, parent_config: Mapping[str, Any],
                 features: Mapping[int, np.ndarray], actions: np.ndarray,
                 labels: np.ndarray, episode_ids: np.ndarray,
                 split_seed: int) -> None:
        self.parent_config = parent_config
        self.parent_lock = None
        self._features = {int(key): np.asarray(value, dtype=np.float32)
                          for key, value in features.items()}
        self._actions = np.asarray(actions, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.count = len(self.labels)
        self.fit_indices, self.readout_indices = _split_indices(
            self.count, split_seed)

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self._features[int(age)][indices], dtype=np.float32)

    def actions(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self._actions[indices], dtype=np.float32)


class _DinoPushTBank(_DevelopmentBank):
    spatial = True

    def __init__(self, parent: Any, rows: np.ndarray, task: str,
                 parent_config: Mapping[str, Any],
                 parent_lock: Mapping[str, Any], split_seed: int) -> None:
        self.parent = parent
        self.rows = np.asarray(rows, dtype=np.int64)
        self.task = task
        self.parent_config = parent_config
        self.parent_lock = parent_lock
        if np.any(parent.split[self.rows] != 0):
            raise DevelopmentAdapterError("non-TRAIN DINO PushT row selected")
        self.labels = np.asarray(parent.labels[task][self.rows], dtype=np.int64)
        self.episode_ids = np.asarray(self.rows, dtype=np.int64)
        self.count = len(self.rows)
        self.fit_indices, self.readout_indices = _split_indices(
            self.count, split_seed)

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        del age
        return self.parent.visual(self.task, self.rows[indices])

    def actions(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.parent.actions[self.rows[indices]],
                          dtype=np.float32)

    def proprio(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.parent.proprio[self.rows[indices]],
                          dtype=np.float32)


class _PointMazeBank(_DevelopmentBank):
    spatial = True

    def __init__(self, parent: Any, base_rows: np.ndarray,
                 parent_config: Mapping[str, Any],
                 parent_lock: Mapping[str, Any], split_seed: int) -> None:
        self.parent = parent
        self.base_rows = np.asarray(base_rows, dtype=np.int64)
        self.parent_config = parent_config
        self.parent_lock = parent_lock
        if np.any(parent.split[self.base_rows] != 0):
            raise DevelopmentAdapterError("non-TRAIN PointMaze row selected")
        self.count = len(self.base_rows) * 4
        self.labels = np.tile(np.arange(4, dtype=np.int64), len(self.base_rows))
        self.episode_ids = (self.base_rows[:, None] * 4
                            + np.arange(4)[None]).reshape(-1)
        self.fit_indices, self.readout_indices = _split_indices(
            len(self.base_rows), split_seed, groups=4)

    def _global_expanded(self, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = np.asarray(local, dtype=np.int64)
        bases = self.base_rows[local // 4]
        labels = local % 4
        return bases * 4 + labels, bases

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        del age
        expanded, _ = self._global_expanded(indices)
        return self.parent.visual(expanded)

    def actions(self, indices: np.ndarray) -> np.ndarray:
        _, bases = self._global_expanded(indices)
        return np.asarray(self.parent.actions[bases], dtype=np.float32)

    def proprio(self, indices: np.ndarray) -> np.ndarray:
        _, bases = self._global_expanded(indices)
        return np.asarray(self.parent.proprio[bases], dtype=np.float32)


class _DinoFormalSplitBank:
    """Local-index view over one label-free DINO formal split."""

    spatial = True

    def __init__(self, parent: Any, split: str) -> None:
        self.parent = parent
        self.split = split
        self.rows = np.asarray(parent.indices(split), dtype=np.int64)
        self.count = len(self.rows)
        self.fit_indices = np.arange(self.count, dtype=np.int64)
        identity = parent.identity(split)
        self.episode_ids = np.asarray(
            identity["episode_id"], dtype=np.int64)
        self.native_cluster_ids = np.asarray(
            identity["native_cluster_id"], dtype=np.int64)
        if self.count < 1 or len(np.unique(self.episode_ids)) != self.count:
            raise DevelopmentAdapterError(
                f"DINO formal split identity is malformed: {split}")

    def _rows(self, indices: np.ndarray) -> np.ndarray:
        local = np.asarray(indices, dtype=np.int64)
        if local.ndim != 1 or np.any(local < 0) or np.any(local >= self.count):
            raise DevelopmentAdapterError(
                f"DINO formal local row leaves split: {self.split}")
        return self.rows[local]

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        return self.parent.features(int(age), self._rows(indices))

    def actions(self, indices: np.ndarray) -> np.ndarray:
        return self.parent.actions(self._rows(indices))

    def proprio(self, indices: np.ndarray) -> np.ndarray:
        return self.parent.proprio(self._rows(indices))


class SageMemV1HostAdapter:
    """One cohort-specific bridge over immutable parent TRAIN caches."""

    def __init__(self, *, cohort: str, spec: Mapping[str, Any]) -> None:
        if cohort not in COHORTS or cohort not in _COHORTS:
            raise DevelopmentAdapterError(f"unknown cohort {cohort!r}")
        if not isinstance(spec, Mapping) or spec.get("study") != "sage-mem-v1":
            raise DevelopmentAdapterError("adapter requires a SAGE-Mem v1 spec")
        if cohort not in spec.get("cohorts", {}):
            raise DevelopmentAdapterError("cohort is absent from the spec")
        self.cohort = cohort
        self.spec = spec
        self.info = _COHORTS[cohort]

    def describe(self) -> dict[str, Any]:
        source, _ = SOURCES[self.cohort]
        return {
            "api_version": SAGE_MEM_HOST_ADAPTER_API_VERSION,
            "cohort": self.cohort,
            "family": self.info["family"],
            "task": self.info["task"],
            "embed_dim": self.info["embed_dim"],
            "action_dim": self.info["action_dim"],
            "tokens": self.info["tokens"],
            "classes": self.info["classes"],
            "development_source": source,
            "development_source_policy": "manifest-selected parent TRAIN only",
            "semantic_labels_for_training": False,
            "candidate_spatial_path": "native_4d_no_patch_flatten",
            "formal_status": "pending_fresh_bank_builder",
        }

    def smoke(self, *, model_contract: Any) -> dict[str, Any]:
        """Label-free model/gradient/reset smoke; no parent metric is opened."""

        _configure_determinism(907_031)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        d, a = int(self.info["embed_dim"]), int(self.info["action_dim"])
        spatial = int(self.info["tokens"]) > 1
        model = _build_sage(model_contract, d, a, "full").to(device)
        with torch.no_grad():
            model.read_scale.fill_(0.01)
        shape = ((2, 6, int(self.info["tokens"]), d)
                 if spatial else (2, 6, d))
        features = torch.randn(shape, device=device)
        actions = torch.randn(2, 5, a, device=device)
        output = model.forward_sequence(features, actions)
        missing = set(model_contract.required_output_keys).difference(output)
        if missing:
            raise DevelopmentAdapterError(
                f"model smoke output missing {sorted(missing)}")
        loss = (output["fused"].float().square().mean()
                + output["prior"].float().square().mean()
                + output["posterior"].float().square().mean() * 0.01)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.requires_grad]
        gradient_finite = bool(
            gradients and all(value is not None and torch.isfinite(value).all()
                              for value in gradients))

        reset_mask = torch.zeros(2, 6, dtype=torch.bool, device=device)
        reset_mask[:, 3] = True
        changed = features.detach().clone()
        changed[:, :3] += 100.0
        with torch.no_grad():
            left = model.forward_sequence(
                features.detach(), actions, reset_mask=reset_mask)["fused"]
            right = model.forward_sequence(
                changed, actions, reset_mask=reset_mask)["fused"]
        reset_isolates = bool(torch.equal(left[:, 3:], right[:, 3:]))
        return {
            "status": "passed",
            "cohort": self.cohort,
            "device": str(device),
            "labels_used": False,
            "gradient_finite": gradient_finite,
            "reset_isolates_state": reset_isolates,
            "candidate_native_shape": list(shape),
            "candidate_native_spatial_path": spatial,
            "zero_semantic_readouts_fitted": True,
            "loss": float(loss.detach()),
        }

    def prepare_fresh_banks(
            self, *, split_counts: Mapping[str, int],
            seed_registry: Mapping[str, int],
            forbidden_parent_artifacts: Sequence[str],
            model_contract: Any) -> Mapping[str, Any]:
        """Materialize or revalidate one fresh, externally sealed formal bank."""

        self._validate_formal_prepare_inputs(
            split_counts, seed_registry, forbidden_parent_artifacts,
            model_contract)
        paths = self._formal_storage_paths()
        presence = {name: path.exists() for name, path in paths.items()
                    if name in {"bank", "vault", "custody"}}
        if any(presence.values()) and not all(presence.values()):
            raise DevelopmentAdapterError(
                f"partial formal bank/custody state: {presence}")
        if not any(presence.values()):
            self._formal_device()
            if self.info["family"] == "lewm":
                from scripts.sage_mem_v1_lewm_formal import (
                    prepare_lewm_formal_bank,
                )
                prepare_lewm_formal_bank(
                    cohort=self.cohort, spec=self.spec,
                    output_directory=paths["bank"],
                    label_vault_path=paths["vault"],
                    label_custody_receipt_path=paths["custody"],
                    device_name="cuda:0")
            else:
                from scripts.sage_mem_v1_dino_formal import (
                    materialize_dino_formal_bank,
                    plan_pointmaze_formal_from_spec,
                    plan_pusht_formal_pair_from_spec,
                )
                if self.info["family"] == "dinowm_pusht":
                    plan = plan_pusht_formal_pair_from_spec(
                        self.spec, root=ROOT)[self.cohort]
                else:
                    plan = plan_pointmaze_formal_from_spec(
                        self.spec, root=ROOT)
                materialize_dino_formal_bank(
                    plan, paths["bank"],
                    label_vault_destination=paths["vault"],
                    label_vault_receipt_destination=paths["custody"],
                    progress=lambda message: None)
            self._seal_formal_bank(paths["bank"])

        bank_handle, custody_handle = self._validated_formal_handles(paths)
        selection = self._locked_development_selection()
        development = self._development_split_handle()
        formal_splits = bank_handle["splits"]
        result_splits = {"development": development}
        for split in ("formal_train", "consumer_train", "formal_test"):
            result_splits[split] = {
                "count": int(formal_splits[split]["count"]),
                "selection_sha256": str(
                    formal_splits[split]["selection_sha256"]),
            }
        return {
            "status": "prepared",
            "cohort": self.cohort,
            "disjoint_with_every_parent_bank": True,
            "development_formal_disjoint": True,
            "formal_labels_hidden": True,
            "labels_used_for_carrier_training": False,
            "gdelta_development_healthy": selection[
                "gdelta_development_healthy"],
            "locked_comparators": selection["locked_comparators"],
            "splits": result_splits,
            "formal_bank": bank_handle,
            "label_custody": custody_handle,
            "phase_a_contract": {
                "schema": PHASE_A_SCHEMA,
                "representation": "feature_artifact",
                "consumer": "centralized-pooled-consumer-train-features",
                "formal_test_labels_available": False,
            },
        }

    def run_formal_cell(
            self, *, arm: str, seed: int, output_directory: Path,
            model_contract: Any, prepared: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Run label-free Phase A and emit finalizer-exact feature artifacts."""

        if arm not in ARMS:
            raise DevelopmentAdapterError(f"unknown formal arm {arm!r}")
        self._validate_phase_a_prepared(prepared)
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(
                f"formal Phase-A directory is not empty: {destination}")
        _configure_determinism(seed)
        device = self._formal_device()
        bank_handle, banks = self._open_formal_banks(prepared)
        host = self._open_formal_host(device)
        host_before = self._host_digest(host)
        if not isinstance(host_before, str) or len(host_before) != 64:
            raise DevelopmentAdapterError("formal frozen-host digest malformed")
        carrier, candidate_native = self._build_carrier(
            arm, model_contract, device)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        history, gradients_finite = self._train_carrier(
            host, banks["formal_train"], carrier, arm, seed,
            candidate_native, device)
        if not gradients_finite:
            raise DevelopmentAdapterError("formal carrier gradients are non-finite")
        arrays = self._phase_a_measurements(
            host, banks, carrier, candidate_native, device)
        elapsed = time.perf_counter() - started
        host_after = self._host_digest(host)
        if host_before != host_after:
            raise DevelopmentAdapterError("formal Phase A mutated frozen host")

        measurement_path = destination / "measurements.npz"
        measurement_sha = _atomic_npz(measurement_path, arrays)
        checkpoint_path = destination / "checkpoint.pt"
        checkpoint_sha = _atomic_torch(checkpoint_path, {
            "carrier_state_dict": carrier.state_dict(),
            "cohort": self.cohort, "arm": arm, "seed": int(seed),
            "bank_manifest_sha256": bank_handle["manifest_sha256"],
            "formal_test_labels_read": False,
        })
        history_path = destination / "history.json"
        history_sha = _atomic_json(history_path, {
            "schema": PHASE_A_HISTORY_SCHEMA,
            "study": "sage-mem-v1", "status": "complete",
            "formal_test_labels_read": False,
            "development_outcomes_read": False,
            "bank_manifest_sha256": bank_handle["manifest_sha256"],
            "epochs": [{
                "epoch": int(row["epoch"]),
                "train_label_free_loss": float(row["loss"]),
            } for row in history],
        })
        resources = self._resource_report(
            carrier, banks["formal_train"], elapsed, device)
        if not all(_finite_nonnegative(value) for value in resources.values()):
            raise DevelopmentAdapterError("formal resource report is non-finite")
        resource_path = destination / "resources.json"
        resource_sha = _atomic_json(resource_path, {
            "schema": PHASE_A_RESOURCE_SCHEMA,
            "study": "sage-mem-v1", "status": "complete",
            "metrics": resources,
        })

        def artifact(path: Path, digest: str) -> dict[str, Any]:
            return {"path": path.name, "sha256": digest,
                    "size": path.stat().st_size}

        physical_gpu = int(self.spec["cohorts"][self.cohort]["gpu"])
        return {
            "schema": PHASE_A_SCHEMA,
            "study": "sage-mem-v1",
            "stage": "formal-phase-a",
            "status": "complete-label-free",
            "cohort": self.cohort,
            "arm": arm,
            "seed": int(seed),
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": str(physical_gpu),
            "protocol_fingerprint": spec_fingerprint(self.spec),
            "completed_unix_ns": time.time_ns(),
            "ages": list(AGES),
            "formal_test_labels_read": False,
            "formal_test_labels_available": False,
            "development_outcomes_read": False,
            "labels_used_for_training": False,
            "bank_manifest_sha256": bank_handle["manifest_sha256"],
            "host_hash_before": host_before,
            "host_hash_after": host_after,
            "prediction_representation": "feature_artifact",
            "consumer_contract": (
                "centralized-pooled-consumer-train-features"),
            "shared_consumer_sha256": None,
            "artifacts": {
                "measurements": artifact(
                    measurement_path, measurement_sha),
                "checkpoint": artifact(checkpoint_path, checkpoint_sha),
                "history": artifact(history_path, history_sha),
                "resource_report": artifact(resource_path, resource_sha),
            },
        }

    def _validate_formal_prepare_inputs(
            self, split_counts: Mapping[str, int],
            seed_registry: Mapping[str, int],
            forbidden_parent_artifacts: Sequence[str],
            model_contract: Any) -> None:
        expected_counts = self.spec["cohorts"][self.cohort]["split_episodes"]
        if dict(split_counts) != dict(expected_counts):
            raise DevelopmentAdapterError("formal split counts changed")
        if dict(seed_registry) != dict(self.spec["_seed_registry"]):
            raise DevelopmentAdapterError("formal seed registry changed")
        expected_forbidden = self.spec["cohorts"][self.cohort][
            "forbidden_parent_artifacts"]
        if list(forbidden_parent_artifacts) != list(expected_forbidden):
            raise DevelopmentAdapterError("formal parent exclusions changed")
        if not callable(getattr(model_contract, "builder", None)):
            raise DevelopmentAdapterError("formal model contract is malformed")

    def _formal_storage_paths(self) -> dict[str, Path]:
        root = output_root(self.spec) / "formal_preparation"
        custody_root = root / "custody"
        return {
            "bank": root / "banks" / self.cohort,
            "vault": custody_root / "vaults" / f"{self.cohort}.npz",
            "custody": custody_root / "receipts" / f"{self.cohort}.json",
        }

    @staticmethod
    def _seal_formal_bank(root: Path) -> None:
        if not root.is_dir() or root.is_symlink():
            raise DevelopmentAdapterError("formal bank root is unsafe")
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_symlink():
                raise DevelopmentAdapterError("formal bank contains symlink")
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(root, 0o555)

    def _locked_development_selection(self) -> dict[str, Any]:
        path = (output_root(self.spec) / "development" / "selections"
                / self.cohort / "receipt.json")
        if not path.is_file() or path.is_symlink():
            raise DevelopmentAdapterError(
                "sealed development selection receipt is missing")
        value = json.loads(path.read_text())
        comparators = value.get("locked_comparators")
        if value.get("stage") != "development-selection" \
                or value.get("status") != "selected" \
                or value.get("cohort") != self.cohort \
                or value.get("protocol_fingerprint") != spec_fingerprint(
                    self.spec) \
                or not isinstance(value.get(
                    "gdelta_development_healthy"), bool) \
                or not isinstance(comparators, Mapping) \
                or set(comparators) != {
                    "retention", "next_feature", "execution"}:
            raise DevelopmentAdapterError(
                "sealed development selection receipt is malformed")
        return {
            "gdelta_development_healthy": bool(
                value["gdelta_development_healthy"]),
            "locked_comparators": dict(comparators),
            "receipt_sha256": _sha256_file(path),
        }

    def _development_split_handle(self) -> dict[str, Any]:
        path = (output_root(self.spec) / "development_banks" / self.cohort
                / "manifest.json")
        if not path.is_file() or path.is_symlink():
            raise DevelopmentAdapterError("development bank receipt is missing")
        value = json.loads(path.read_text())
        selection = value.get("selection")
        expected = int(self.spec["cohorts"][self.cohort][
            "split_episodes"]["development"])
        if value.get("status") != "prepared-parent-train-only" \
                or value.get("cohort") != self.cohort \
                or value.get("parent_train_only") is not True \
                or value.get("formal_evidence_permitted") is not False \
                or not isinstance(selection, Mapping) \
                or selection.get("count") != expected \
                or not isinstance(selection.get("sha256"), str) \
                or len(selection["sha256"]) != 64:
            raise DevelopmentAdapterError("development split receipt malformed")
        return {"count": expected,
                "selection_sha256": selection["sha256"]}

    def _validated_formal_handles(
            self, paths: Mapping[str, Path]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.info["family"] == "lewm":
            from scripts.sage_mem_v1_lewm_formal import (
                sealed_label_vault_handle, trajectory_bank_handle,
            )
            manifest_path = paths["bank"] / "manifest.json"
            bank = trajectory_bank_handle(manifest_path)
            custody = sealed_label_vault_handle(
                manifest_path, paths["custody"])
            vault_sha = custody["vault_sha256"]
            receipt_sha = custody["custody_receipt"]["sha256"]
            receipt_size = custody["custody_receipt"]["size"]
        else:
            from scripts.sage_mem_v1_dino_formal import (
                validate_materialized_bank_provenance,
            )
            bank = validate_materialized_bank_provenance(paths["bank"])
            if not paths["custody"].is_file() \
                    or paths["custody"].is_symlink() \
                    or oct(paths["custody"].stat().st_mode & 0o777) != "0o400":
                raise DevelopmentAdapterError(
                    "DINO formal custody receipt is not protected")
            receipt = json.loads(paths["custody"].read_text())
            vault = Path(receipt.get("path", ""))
            if receipt.get("status") != "sealed-for-post-grid-finalizer" \
                    or receipt.get("cohort") != self.cohort \
                    or receipt.get("per_cell_api_access") is not False \
                    or not vault.is_file() or vault.is_symlink() \
                    or oct(vault.stat().st_mode & 0o777) != "0o400" \
                    or vault.stat().st_size != receipt.get("size") \
                    or _sha256_file(vault) != receipt.get("sha256") \
                    or receipt.get("sha256") != bank.get(
                        "sealed_label_vault_sha256"):
                raise DevelopmentAdapterError(
                    "DINO formal label custody identity failed")
            vault_sha = receipt["sha256"]
            receipt_sha = _sha256_file(paths["custody"])
            receipt_size = paths["custody"].stat().st_size
        bank = dict(bank)
        bank["bank_root"] = str(paths["bank"].resolve())
        return bank, {
            "status": "sealed-for-post-grid-finalizer",
            "vault_sha256": vault_sha,
            "custody_receipt_sha256": receipt_sha,
            "custody_receipt_size": int(receipt_size),
            "per_cell_api_access": False,
            "path_exposed_to_phase_a": False,
        }

    def _validate_phase_a_prepared(self, value: Mapping[str, Any]) -> None:
        bank = value.get("formal_bank")
        custody = value.get("label_custody")
        custody_keys = {
            "status", "vault_sha256", "custody_receipt_sha256",
            "custody_receipt_size", "per_cell_api_access",
            "path_exposed_to_phase_a",
        }
        if value.get("status") != "prepared" \
                or value.get("cohort") != self.cohort \
                or value.get("formal_labels_hidden") is not True \
                or value.get("labels_used_for_carrier_training") is not False \
                or not isinstance(bank, Mapping) \
                or bank.get("cohort") != self.cohort \
                or not isinstance(bank.get("bank_root"), str) \
                or not isinstance(bank.get("manifest_sha256"), str) \
                or len(bank["manifest_sha256"]) != 64 \
                or not isinstance(custody, Mapping) \
                or set(custody) != custody_keys \
                or custody.get("status") != \
                    "sealed-for-post-grid-finalizer" \
                or custody.get("per_cell_api_access") is not False \
                or custody.get("path_exposed_to_phase_a") is not False \
                or not isinstance(custody.get("vault_sha256"), str) \
                or len(custody["vault_sha256"]) != 64 \
                or not isinstance(custody.get(
                    "custody_receipt_sha256"), str) \
                or len(custody["custody_receipt_sha256"]) != 64:
            raise DevelopmentAdapterError("formal Phase-A preparation malformed")
        root = Path(bank["bank_root"])
        if root.resolve() != self._formal_storage_paths()["bank"].resolve():
            raise DevelopmentAdapterError("formal bank root changed")
        manifest = root / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink() \
                or _sha256_file(manifest) != bank["manifest_sha256"]:
            raise DevelopmentAdapterError("formal bank manifest identity failed")

    def _open_formal_banks(
            self, prepared: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        expected = prepared["formal_bank"]
        root = Path(expected["bank_root"])
        if self.info["family"] == "lewm":
            from scripts.sage_mem_v1_lewm_formal import (
                load_lewm_trajectory_banks,
            )
            handle, banks = load_lewm_trajectory_banks(
                root / "manifest.json", expected_cohort=self.cohort,
                expected_counts={
                    split: int(self.spec["cohorts"][self.cohort][
                        "split_episodes"][split])
                    for split in ("formal_train", "consumer_train",
                                  "formal_test")})
        else:
            from scripts.sage_mem_v1_dino_formal import open_label_free_bank
            parent = open_label_free_bank(root)
            handle = parent.provenance_handle()
            handle["bank_root"] = str(root.resolve())
            banks = {split: _DinoFormalSplitBank(parent, split)
                     for split in ("formal_train", "consumer_train",
                                   "formal_test")}
        if handle["manifest_sha256"] != expected["manifest_sha256"]:
            raise DevelopmentAdapterError("formal bank changed after preparation")
        return handle, banks

    def _open_formal_host(self, device: torch.device) -> Any:
        parent_protocol = ROOT / self.spec["cohorts"][self.cohort][
            "parent_protocol"]
        if self.info["family"] == "lewm":
            from scripts.paper_a_matched_color_v1_1_spec import (
                DEFAULT_SHA, load_locked_spec,
            )
            from scripts.prepare_paper_a_matched_host import _load_host
            parent = load_locked_spec(
                parent_protocol, DEFAULT_SHA, verify_inputs=False)
            return _load_host(parent, str(self.info["host"]), device)
        import yaml
        cfg = yaml.safe_load(parent_protocol.read_text())
        if not isinstance(cfg, Mapping):
            raise DevelopmentAdapterError("DINO parent protocol malformed")
        from scripts.sage_mem_v1_dino_formal import (
            activate_parent_dependency_environment,
        )
        activate_parent_dependency_environment(cfg, root=ROOT)
        if self.info["family"] == "dinowm_pusht":
            from scripts.run_dinowm_wave2_spatial_carrier import FrozenNativeHost
            return FrozenNativeHost(cfg, load_encoder=False)
        from scripts.run_dinowm_pointmaze_wave3 import FrozenPointMazeHost
        return FrozenPointMazeHost(cfg, load_encoder=False)

    def _formal_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise DevelopmentAdapterError("formal execution requires CUDA")
        expected = str(self.spec["cohorts"][self.cohort]["gpu"])
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected \
                or torch.cuda.device_count() != 1:
            raise DevelopmentAdapterError(
                f"{self.cohort} formal execution requires physical GPU "
                f"{expected} as the only visible device")
        return torch.device("cuda:0")

    def _phase_a_measurements(
            self, host: Any, banks: Mapping[str, Any],
            carrier: torch.nn.Module, candidate_native: bool,
            device: torch.device) -> dict[str, np.ndarray]:
        carrier.eval()
        consumer = banks["consumer_train"]
        test = banks["formal_test"]
        result = _phase_a_identity_arrays(consumer, test)
        consumer_full, test_full, test_reset, test_prior = [], [], [], []
        full_mse, reset_mse, prior_mse = [], [], []
        for age in AGES:
            consumer_value = _collect_phase_a_features(
                host, consumer, carrier, int(age), consumer.fit_indices,
                candidate_native, device)
            test_value = _collect_phase_a_features(
                host, test, carrier, int(age), test.fit_indices,
                candidate_native, device)
            consumer_full.append(consumer_value["host"])
            test_full.append(test_value["host"])
            test_reset.append(test_value["reset"])
            test_prior.append(test_value["prior"])
            full_mse.append(test_value["mse"])
            reset_mse.append(test_value["reset_mse"])
            prior_mse.append(test_value["prior_mse"])
        result.update({
            "consumer_train_full_features": np.stack(consumer_full),
            "formal_test_full_features": np.stack(test_full),
            "formal_test_reset_features": np.stack(test_reset),
            "formal_test_prior_features": np.stack(test_prior),
            "formal_test_full_mse": np.stack(full_mse),
            "formal_test_reset_mse": np.stack(reset_mse),
            "formal_test_prior_mse": np.stack(prior_mse),
        })
        _validate_phase_a_arrays(result, consumer.count, test.count)
        return result

    def run_development_cell(
            self, *, arm: str, seed: int,
            development_manifest: Mapping[str, Any],
            output_directory: Path, model_contract: Any) -> dict[str, Any]:
        """Train/evaluate one development arm and atomically write receipts."""

        if arm not in ARMS:
            raise DevelopmentAdapterError(f"unknown arm {arm!r}")
        rows = self._validate_development_manifest(development_manifest)
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(
                f"development output directory is not empty: {destination}")

        _configure_determinism(seed)
        bank = self._open_bank(rows)
        device = self._development_device()
        host = self._open_host(bank, device)
        host_before = self._host_digest(host)
        if not isinstance(host_before, str) or len(host_before) != 64:
            raise DevelopmentAdapterError("frozen host digest is malformed")
        carrier, candidate_native = self._build_carrier(
            arm, model_contract, device)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        history, gradients_finite = self._train_carrier(
            host, bank, carrier, arm, seed, candidate_native, device)
        metrics, arrays, next_feature_mse = self._evaluate(
            host, bank, carrier, candidate_native, device)
        elapsed = time.perf_counter() - started
        host_after = self._host_digest(host)
        if host_before != host_after:
            raise DevelopmentAdapterError("development mutated the frozen host")
        _validate_episode_arrays(arrays, int(self.info["classes"]))

        artifact_path = destination / "development_results.npz"
        artifact_hash = _atomic_npz(artifact_path, arrays)
        checkpoint_path = destination / "carrier.pt"
        checkpoint_hash = _atomic_torch(checkpoint_path, {
            "carrier_state_dict": carrier.state_dict(),
            "cohort": self.cohort, "arm": arm, "seed": seed,
            "development_only": True,
        })
        resource_report = self._resource_report(
            carrier, bank, elapsed, device)
        if not all(_finite_nonnegative(value)
                   for value in resource_report.values()):
            raise DevelopmentAdapterError("non-finite resource report")
        return {
            "schema_version": 1,
            "study": "sage-mem-v1",
            "stage": "development-cell",
            "status": "complete",
            "cohort": self.cohort,
            "arm": arm,
            "seed": int(seed),
            "development_only": True,
            "formal_evidence_permitted": False,
            "parent_train_only": True,
            "labels_used_for_training": False,
            "labels_used_for_posthoc_readout": True,
            "host_hash_before": host_before,
            "host_hash_after": host_after,
            "gradient_finite": gradients_finite,
            "next_feature_mse": float(next_feature_mse),
            "ages": metrics,
            "objective": _objective_weights(arm),
            "resource_report": resource_report,
            "development_manifest_sha256": _sha256_json(
                dict(development_manifest)),
            "episode_results": {
                "path": artifact_path.name, "sha256": artifact_hash,
            },
            "checkpoint": {
                "path": checkpoint_path.name, "sha256": checkpoint_hash,
            },
            "history": history,
            "host_output_exposure_measured": True,
            "reset_intervention_measured": True,
            "external_consumer_gate_evaluated": False,
            "counterfactual_pairing_preserved": True,
        }

    def _validate_development_manifest(
            self, value: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(value, Mapping) \
                or value.get("status") != "prepared-parent-train-only" \
                or value.get("cohort") != self.cohort \
                or value.get("parent_train_only") is not True \
                or value.get("parent_validation_or_test_read") is not False \
                or value.get("semantic_labels_read_for_selection") is not False \
                or value.get("formal_evidence_permitted") is not False:
            raise DevelopmentAdapterError("invalid development-bank manifest")
        selection = value.get("selection")
        source = value.get("source")
        if not isinstance(selection, Mapping) or not isinstance(source, Mapping):
            raise DevelopmentAdapterError("development manifest lacks source/rows")
        expected_source, _ = SOURCES[self.cohort]
        source_path = ROOT / expected_source
        if source.get("path") != expected_source or not source_path.is_file() \
                or source.get("size") != source_path.stat().st_size \
                or source.get("sha256") != _sha256_file(source_path):
            raise DevelopmentAdapterError("parent TRAIN source identity changed")
        rows = np.asarray(selection.get("rows"), dtype=np.int64)
        expected_count = int(self.spec["cohorts"][self.cohort][
            "split_episodes"]["development"])
        if rows.shape != (expected_count,) or np.any(rows < 0) \
                or len(np.unique(rows)) != len(rows) \
                or not np.all(rows[:-1] < rows[1:]):
            raise DevelopmentAdapterError("development rows are malformed")
        return rows

    def _open_bank(self, rows: np.ndarray) -> _DevelopmentBank:
        split_seed = int(self.spec["_seed_registry"][
            f"{self.cohort}/development/loader"])
        family = self.info["family"]
        parent_protocol = ROOT / self.spec["cohorts"][self.cohort][
            "parent_protocol"]
        if family == "lewm":
            from scripts.paper_a_matched_color_v1_1_spec import (
                DEFAULT_SHA, load_locked_spec,
            )
            from scripts.train_paper_a_matched_color_v1_1 import (
                _aligned_latent, _load_base, _load_cue,
            )
            parent = load_locked_spec(
                parent_protocol, DEFAULT_SHA, verify_inputs=False)
            host = str(self.info["host"])
            base = _load_base(parent, host, "train")
            cues = {age: _load_cue(parent, host, "train", age)
                    for age in AGES}
            if np.any(rows >= len(base["z_base"])):
                raise DevelopmentAdapterError("LeWM row leaves TRAIN cache")
            features = {
                age: _aligned_latent(base, cues[age])[rows] for age in AGES}
            labels = np.asarray(cues[AGES[0]]["color_label"])[rows]
            for age in AGES[1:]:
                if not np.array_equal(
                        labels, np.asarray(cues[age]["color_label"])[rows]):
                    raise DevelopmentAdapterError("LeWM age labels are unpaired")
            episode = (np.asarray(base["episode_index"], dtype=np.int64)[rows]
                       * 1_000_000
                       + np.asarray(base["local_start"], dtype=np.int64)[rows])
            return _LeWMBank(
                parent, features, np.asarray(base["actions"])[rows], labels,
                episode, split_seed)
        if family == "dinowm_pusht":
            from scripts.run_dinowm_wave2_spatial_carrier import (
                FeatureBank, load_config,
            )
            cfg, lock = load_config(parent_protocol, locked=True)
            assert lock is not None
            parent = FeatureBank(cfg, lock)
            return _DinoPushTBank(
                parent, rows, str(self.info["task"]), cfg, lock, split_seed)
        from scripts.run_dinowm_pointmaze_wave3 import FeatureBank, load_config
        cfg, lock = load_config(parent_protocol, locked=True)
        assert lock is not None
        parent = FeatureBank(cfg, lock)
        return _PointMazeBank(parent, rows, cfg, lock, split_seed)

    def _development_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        expected = str(self.spec["cohorts"][self.cohort]["gpu"])
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected \
                or torch.cuda.device_count() != 1:
            raise DevelopmentAdapterError(
                f"{self.cohort} development requires exactly physical GPU "
                f"{expected} as its sole CUDA_VISIBLE_DEVICES entry")
        return torch.device("cuda:0")

    def _open_host(self, bank: _DevelopmentBank,
                   device: torch.device) -> Any:
        if self.info["family"] == "lewm":
            from scripts.prepare_paper_a_matched_host import _load_host
            return _load_host(
                dict(bank.parent_config), str(self.info["host"]), device)
        if device.type != "cuda":
            raise DevelopmentAdapterError(
                "released DINO-WM host development requires its assigned GPU")
        _activate_parent_environment(bank.parent_config)
        if self.info["family"] == "dinowm_pusht":
            from scripts.run_dinowm_wave2_spatial_carrier import FrozenNativeHost
            return FrozenNativeHost(bank.parent_config, load_encoder=False)
        from scripts.run_dinowm_pointmaze_wave3 import FrozenPointMazeHost
        return FrozenPointMazeHost(bank.parent_config, load_encoder=False)

    @staticmethod
    def _host_digest(host: Any) -> str:
        if callable(getattr(host, "digest", None)):
            return str(host.digest())
        from scripts.train_frozen_official_swap import state_digest
        return state_digest(host)

    def _build_carrier(self, arm: str, model_contract: Any,
                       device: torch.device) -> tuple[torch.nn.Module, bool]:
        d, a = int(self.info["embed_dim"]), int(self.info["action_dim"])
        if arm.startswith("sage_mem_"):
            variant = arm.removeprefix("sage_mem_")
            return _build_sage(model_contract, d, a, variant).to(device), True
        if arm == "gdelta":
            target = int(self.spec["cohorts"][self.cohort][
                "target_parameters"])
            state_dim = min(range(2, 4 * d + 1), key=lambda candidate: (
                abs(gdelta_parameter_count(candidate, d, a) - target),
                candidate))
            carrier = GatedDeltaCell(d, a, state_dim=state_dim).to(device)
            margin = float(self.spec["fairness_reporting"][
                "maximum_parameter_relative_gap"])
            if abs(carrier.parameter_count() - target) / target > margin:
                raise DevelopmentAdapterError(
                    "no registered parameter-matched gDelta width exists")
            return carrier, False
        base = {"fixed_trust_aux": "fixed_trust",
                "ssm_aux": "ssm"}.get(arm, arm)
        return make_frozen_carrier(base, d, a).to(device), False

    def _train_carrier(
            self, host: Any, bank: _DevelopmentBank,
            carrier: torch.nn.Module, arm: str, seed: int,
            candidate_native: bool, device: torch.device
    ) -> tuple[list[dict[str, Any]], bool]:
        if carrier.parameter_count() == 0:
            return [], True
        optimization = self.spec["optimization"]
        epochs = int(optimization["epochs"])
        effective_batch = int(optimization["batch_size"])
        micro_batch = (effective_batch if not bank.spatial else
                       SPATIAL_TRAIN_MICRO_BATCH)
        if effective_batch % micro_batch:
            raise DevelopmentAdapterError(
                "effective batch must be divisible by the spatial microbatch")
        optimizer = torch.optim.AdamW(
            carrier.parameters(), lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        rng = np.random.default_rng(seed + 4_731_000)
        training_ages = AGES if not bank.spatial else (AGES[0],)
        history: list[dict[str, Any]] = []
        gradients_finite = True
        for epoch in range(1, epochs + 1):
            epoch_losses: list[float] = []
            for age in training_ages:
                order = rng.permutation(bank.fit_indices)
                for offset in range(0, len(order), effective_batch):
                    group = order[offset:offset + effective_batch]
                    if len(group) < 2:
                        continue
                    starts = rng.choice(
                        17, size=2 if bank.spatial else 8,
                        replace=False).astype(np.int64)
                    optimizer.zero_grad(set_to_none=True)
                    group_loss = 0.0
                    for micro_start in range(0, len(group), micro_batch):
                        local = group[micro_start:micro_start + micro_batch]
                        features = torch.from_numpy(
                            bank.features(int(age), local)).to(device)
                        actions = torch.from_numpy(bank.actions(local)).to(device)
                        prop_np = bank.proprio(local)
                        proprio = (None if prop_np is None else
                                   torch.from_numpy(prop_np).to(device))
                        output = _carrier_forward(
                            carrier, features, actions,
                            candidate_native=candidate_native)
                        next_loss = _next_feature_loss(
                            host, output["fused"], features, actions,
                            proprio, starts, spatial=bank.spatial)
                        auxiliary = _auxiliary_loss(
                            arm, carrier, output, features, actions,
                            candidate_native=candidate_native)
                        loss = next_loss + auxiliary
                        if not torch.isfinite(loss).item():
                            raise DevelopmentAdapterError(
                                f"non-finite objective in {arm}/s{seed}")
                        scale = len(local) / len(group)
                        (scale * loss).backward()
                        group_loss += scale * float(loss.detach())
                    gradients = [parameter.grad for parameter in
                                 carrier.parameters() if parameter.requires_grad]
                    finite = bool(
                        gradients and any(value is not None for value in gradients)
                        and all(value is None or torch.isfinite(value).all()
                                for value in gradients))
                    gradients_finite = gradients_finite and finite
                    if not finite:
                        raise DevelopmentAdapterError(
                            f"non-finite gradient in {arm}/s{seed}")
                    torch.nn.utils.clip_grad_norm_(
                        carrier.parameters(),
                        float(optimization["gradient_clip_norm"]))
                    optimizer.step()
                    epoch_losses.append(group_loss)
            scheduler.step()
            history.append({
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })
        return history, gradients_finite

    def _evaluate(
            self, host: Any, bank: _DevelopmentBank,
            carrier: torch.nn.Module, candidate_native: bool,
            device: torch.device
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
        carrier.eval()
        classes = int(self.info["classes"])
        metrics: dict[str, Any] = {}
        artifact_rows: dict[str, list[np.ndarray]] = {
            key: [] for key in (
                "episode_id", "class_id", "evidence_age",
                "retention_correct", "reset_correct", "exposure_correct",
                "next_feature_mse", "reset_next_feature_mse",
                "oracle_success", "execution_success",
            )
        }
        all_mse: list[np.ndarray] = []
        for age in AGES:
            fit = _collect_features(
                host, bank, carrier, int(age), bank.fit_indices,
                candidate_native, device)
            heldout = _collect_features(
                host, bank, carrier, int(age), bank.readout_indices,
                candidate_native, device)
            train_y = bank.labels[bank.fit_indices]
            heldout_y = bank.labels[bank.readout_indices]
            exposure_prediction, reset_prediction = _fit_shared_readout(
                fit["host"], train_y, heldout["host"], heldout["reset"])
            prior_prediction, _ = _fit_shared_readout(
                fit["prior"], train_y, heldout["prior"], heldout["prior"])
            exposure_correct = (exposure_prediction == heldout_y).astype(np.int8)
            reset_correct = (reset_prediction == heldout_y).astype(np.int8)
            prior_correct = (prior_prediction == heldout_y).astype(np.int8)
            reset_ratio = float(
                np.mean(heldout["reset_mse"])
                / max(np.mean(heldout["mse"]), 1e-12))
            metrics[str(age)] = {
                "host_output_balanced_accuracy": _balanced_accuracy(
                    exposure_prediction, heldout_y, classes),
                "prior_balanced_accuracy": _balanced_accuracy(
                    prior_prediction, heldout_y, classes),
                "reset_with_full_readout_balanced_accuracy":
                    _balanced_accuracy(reset_prediction, heldout_y, classes),
                "full_next_feature_mse": float(np.mean(heldout["mse"])),
                "reset_next_feature_mse": float(
                    np.mean(heldout["reset_mse"])),
                "reset_to_full_mse_ratio": reset_ratio,
                "reset_health_ratio_maximum": 1.25,
                "reset_health_pass": bool(reset_ratio <= 1.25),
                "readout_fit_parent_train_rows": int(len(train_y)),
                "readout_eval_parent_train_rows": int(len(heldout_y)),
            }
            count = len(heldout_y)
            # Encode age in the identifier so the formal-like artifact remains
            # one-dimensional with globally unique rows.
            ids = (bank.episode_ids[bank.readout_indices].astype(np.int64)
                   * 100 + int(age))
            artifact_rows["episode_id"].append(ids)
            artifact_rows["class_id"].append(heldout_y.astype(np.int64))
            artifact_rows["evidence_age"].append(
                np.full(count, int(age), dtype=np.int64))
            artifact_rows["retention_correct"].append(prior_correct)
            artifact_rows["reset_correct"].append(reset_correct)
            artifact_rows["exposure_correct"].append(exposure_correct)
            artifact_rows["next_feature_mse"].append(heldout["mse"])
            artifact_rows["reset_next_feature_mse"].append(
                heldout["reset_mse"])
            artifact_rows["oracle_success"].append(
                np.zeros(count, dtype=np.int8))
            artifact_rows["execution_success"].append(
                np.zeros(count, dtype=np.int8))
            all_mse.append(heldout["mse"])
        arrays = {key: np.concatenate(values)
                  for key, values in artifact_rows.items()}
        return metrics, arrays, float(np.mean(np.concatenate(all_mse)))

    def _resource_report(
            self, carrier: torch.nn.Module, bank: _DevelopmentBank,
            elapsed: float, device: torch.device) -> dict[str, int | float]:
        parameters = int(carrier.parameter_count())
        tokens = int(self.info["tokens"])
        if callable(getattr(carrier, "estimate_flops", None)):
            flops = int(carrier.estimate_flops(
                batch_size=1, timesteps=20, tokens=tokens))
        else:
            flops = int(max(parameters, 1) * 2 * 20 * tokens)
        if callable(getattr(carrier, "persistent_state_floats", None)):
            persistent = int(carrier.persistent_state_floats()) * tokens
        elif hasattr(carrier, "hidden_dim"):
            multiplier = 2 if carrier.__class__.__name__.lower().find(
                "lstm") >= 0 else 1
            persistent = int(carrier.hidden_dim) * tokens * multiplier
        elif hasattr(carrier, "state_dim"):
            state_dim = int(carrier.state_dim)
            persistent = (state_dim * state_dim if isinstance(
                carrier, GatedDeltaCell) else state_dim) * tokens
        else:
            persistent = 0
        peak = (int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0)
        return {
            "trainable_parameters": parameters,
            "forward_flops_per_episode": flops,
            "persistent_state_floats": persistent,
            "peak_cuda_bytes": peak,
            "wall_clock_train_seconds": float(elapsed),
        }


def build_host_adapter(*, cohort: str,
                       spec: Mapping[str, Any]) -> SageMemV1HostAdapter:
    """Build one of the five registered development adapters."""

    return SageMemV1HostAdapter(cohort=cohort, spec=spec)


def _build_sage(model_contract: Any, embed_dim: int, action_dim: int,
                variant: str) -> torch.nn.Module:
    builder = getattr(model_contract, "builder", None)
    if not callable(builder):
        raise DevelopmentAdapterError("model contract has no callable builder")
    parameters = inspect.signature(builder).parameters
    kwargs: dict[str, Any] = {
        "embed_dim": embed_dim, "action_dim": action_dim,
        "variant": variant,
    }
    config = parameters.get("config")
    if config is not None and config.default is inspect.Parameter.empty:
        kwargs["config"] = {}
    model = builder(**kwargs)
    required = ("forward_sequence", "parameter_count", "describe")
    if any(not callable(getattr(model, name, None)) for name in required):
        raise DevelopmentAdapterError("candidate violates the model contract")
    return model


def _activate_parent_environment(config: Mapping[str, Any]) -> None:
    """Expose the parent's hash-pinned dependency/shim paths in-process."""

    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise DevelopmentAdapterError("parent execution registry is missing")
    python_value = execution.get("isolated_python")
    manifest_value = execution.get("dependency_manifest_path")
    identity = execution.get("dependency_manifest_identity")
    if not isinstance(python_value, str) or not isinstance(manifest_value, str) \
            or not isinstance(identity, Mapping):
        raise DevelopmentAdapterError("parent dependency registry is malformed")
    isolated = ROOT / python_value
    manifest = ROOT / manifest_value
    if not isolated.is_file() or not manifest.is_file() \
            or manifest.stat().st_size != int(identity.get("size", -1)) \
            or _sha256_file(manifest) != identity.get("sha256"):
        raise DevelopmentAdapterError("parent dependency identity changed")
    candidates = sorted(
        isolated.parent.parent.glob("lib/python*/site-packages/*.pth"))
    if len(candidates) != 1:
        raise DevelopmentAdapterError("parent isolated environment is ambiguous")
    paths = [Path(line.strip()) for line in candidates[0].read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not paths or any(not path.is_dir() for path in paths):
        raise DevelopmentAdapterError("parent dependency path is unavailable")
    # Reverse insertion preserves the order declared by the pinned .pth file.
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _carrier_forward(
        carrier: torch.nn.Module, features: torch.Tensor,
        actions: torch.Tensor, *, candidate_native: bool
) -> dict[str, Any]:
    if candidate_native:
        output = carrier.forward_sequence(features, actions)
        required = {"fused", "prior", "posterior", "exposure", "diagnostics"}
        if required.difference(output):
            raise DevelopmentAdapterError("candidate forward output is incomplete")
        return dict(output)
    if features.ndim == 4:
        spatial = spatial_carrier_forward(carrier, features, actions)
        fused, prior = spatial.fused_visual, spatial.prior_visual
        diagnostics: Mapping[str, torch.Tensor] = {}
    else:
        output = carrier(features, actions)
        fused, prior = output.z_tilde, output.prior_read
        diagnostics = output.telemetry
    return {
        "fused": fused,
        "prior": prior,
        # Registered baselines expose no unprojected posterior.  Their causal
        # projected prior is the common-coordinate storage diagnostic.
        "posterior": prior,
        "exposure": fused - features,
        "diagnostics": diagnostics,
    }


def _amp_context(device: torch.device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext())


def _next_feature_loss(
        host: Any, fused: torch.Tensor, features: torch.Tensor,
        actions: torch.Tensor, proprio: torch.Tensor | None,
        starts: Sequence[int], *, spatial: bool) -> torch.Tensor:
    latent_windows, action_windows, targets = [], [], []
    proprio_windows, target_proprio = [], []
    for raw_start in starts:
        start = int(raw_start)
        if not (0 <= start and start + 3 < features.shape[1]):
            raise DevelopmentAdapterError(f"illegal objective window {start}")
        latent_windows.append(fused[:, start:start + 3])
        action_windows.append(actions[:, start:start + 3])
        targets.append(features[:, start + 3])
        if spatial:
            if proprio is None:
                raise DevelopmentAdapterError("DINO host lacks proprioception")
            proprio_windows.append(proprio[:, start:start + 3])
            target_proprio.append(proprio[:, start + 1:start + 4])
    latent = torch.cat(latent_windows)
    action = torch.cat(action_windows)
    target = torch.cat(targets)
    with _amp_context(features.device):
        if spatial:
            assert proprio is not None
            prop = torch.cat(proprio_windows)
            target_visual = torch.cat([
                features[:, int(start) + 1:int(start) + 4]
                for start in starts])
            target_all = host.target_nonaction(
                target_visual, torch.cat(target_proprio))
            prediction = host.predict(latent, prop, action)[..., :394]
            return F.mse_loss(prediction.float(), target_all.float())
        prediction = host.predict(latent, action)[:, -1]
        return F.mse_loss(prediction.float(), target.float())


def _semantic_vector(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value.mean(dim=1)
    if value.ndim == 2:
        return value
    raise DevelopmentAdapterError(
        f"semantic vector expects (B,D) or (B,P,D), got {tuple(value.shape)}")


def _salient_past(features: torch.Tensor) -> torch.Tensor:
    """Select the largest frozen-feature change before the final context."""

    if features.shape[1] < 5:
        raise DevelopmentAdapterError("retrospective replay needs five frames")
    difference = features[:, 1:-3].float() - features[:, :-4].float()
    if difference.ndim == 4:
        score = difference.square().mean(dim=(2, 3))
    else:
        score = difference.square().mean(dim=2)
    event = score.argmax(dim=1) + 1
    rows = torch.arange(len(features), device=features.device)
    return _semantic_vector(features[rows, event].float()).detach()


def _objective_weights(arm: str) -> dict[str, float]:
    result = {"next": 1.0, "replay": 0.0,
              "exposure": 0.0, "reset": 0.0}
    if arm in ("sage_mem_full", "fixed_trust_aux", "ssm_aux"):
        result.update(replay=0.10, exposure=0.10, reset=0.10)
    elif arm == "sage_mem_no_exposure":
        result.update(replay=0.10, reset=0.10)
    elif arm == "sage_mem_exposure_only":
        result.update(exposure=0.10, reset=0.10)
    return result


def _auxiliary_loss(
        arm: str, carrier: torch.nn.Module, output: Mapping[str, Any],
        features: torch.Tensor, actions: torch.Tensor, *,
        candidate_native: bool) -> torch.Tensor:
    weights = _objective_weights(arm)
    if not any(weights[name] for name in ("replay", "exposure", "reset")):
        return output["fused"].float().sum() * 0.0
    past = _salient_past(features)
    current = _semantic_vector(features[:, -1].float()).detach()
    # The causal projected prior is taken before the final observation.  This
    # both avoids current-frame leakage and trains a diagnostic read for the
    # no-exposure control without injecting it into the frozen host.
    posterior = _semantic_vector(output["prior"][:, -1].float())
    exposure = _semantic_vector(output["exposure"][:, -1].float())
    replay = F.smooth_l1_loss(posterior, past)
    exposure_target = past - current
    exposure_alignment = (
        1.0 - F.cosine_similarity(
            exposure, exposure_target, dim=-1, eps=1e-8)).mean()
    reset = _carrier_forward(
        carrier, features[:, -3:], actions[:, -3:-1],
        candidate_native=candidate_native)
    reset_distillation = F.mse_loss(
        reset["fused"].float(), features[:, -3:].float())
    return (weights["replay"] * replay
            + weights["exposure"] * exposure_alignment
            + weights["reset"] * reset_distillation)


def _phase_a_identity_arrays(consumer: Any,
                             test: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for prefix, bank in (("consumer_train", consumer),
                         ("formal_test", test)):
        episode = np.asarray(bank.episode_ids, dtype=np.int64)
        cluster = np.asarray(
            getattr(bank, "native_cluster_ids", episode), dtype=np.int64)
        if episode.ndim != 1 or cluster.shape != episode.shape \
                or len(np.unique(episode)) != len(episode) \
                or np.any(episode < 0) or np.any(cluster < 0):
            raise DevelopmentAdapterError(
                f"formal Phase-A identity malformed: {prefix}")
        result[f"{prefix}_episode_id"] = np.repeat(
            episode[None], len(AGES), axis=0)
        result[f"{prefix}_native_cluster_id"] = np.repeat(
            cluster[None], len(AGES), axis=0)
        result[f"{prefix}_evidence_age"] = np.repeat(
            np.asarray(AGES, dtype=np.int64)[:, None], len(episode), axis=1)
    return result


def _validate_phase_a_arrays(arrays: Mapping[str, np.ndarray],
                             consumer_count: int, test_count: int) -> None:
    expected = {
        "formal_test_episode_id", "formal_test_native_cluster_id",
        "formal_test_evidence_age", "consumer_train_episode_id",
        "consumer_train_native_cluster_id", "consumer_train_evidence_age",
        "formal_test_full_mse", "formal_test_reset_mse",
        "formal_test_prior_mse", "formal_test_full_features",
        "formal_test_reset_features", "formal_test_prior_features",
        "consumer_train_full_features",
    }
    if set(arrays) != expected:
        raise DevelopmentAdapterError("formal Phase-A array schema changed")
    for prefix, count in (("consumer_train", consumer_count),
                          ("formal_test", test_count)):
        reference = np.asarray(arrays[f"{prefix}_episode_id"])
        cluster = np.asarray(arrays[f"{prefix}_native_cluster_id"])
        ages = np.asarray(arrays[f"{prefix}_evidence_age"])
        if reference.shape != (len(AGES), count) \
                or cluster.shape != reference.shape \
                or ages.shape != reference.shape \
                or not all(np.issubdtype(value.dtype, np.integer)
                           for value in (reference, cluster, ages)):
            raise DevelopmentAdapterError(
                f"formal Phase-A identity shape changed: {prefix}")
        for index, age in enumerate(AGES):
            if not np.array_equal(reference[index], reference[0]) \
                    or not np.array_equal(cluster[index], cluster[0]) \
                    or not np.all(ages[index] == age):
                raise DevelopmentAdapterError(
                    f"formal Phase-A identity drifts across ages: {prefix}")
    for name in ("formal_test_full_mse", "formal_test_reset_mse",
                 "formal_test_prior_mse"):
        value = np.asarray(arrays[name])
        if value.shape != (len(AGES), test_count) \
                or not np.isfinite(value).all() or np.any(value < 0):
            raise DevelopmentAdapterError(
                f"formal Phase-A MSE malformed: {name}")
    dimensions = set()
    for name in ("formal_test_full_features",
                 "formal_test_reset_features",
                 "formal_test_prior_features",
                 "consumer_train_full_features"):
        value = np.asarray(arrays[name])
        count = consumer_count if name.startswith("consumer_train") \
            else test_count
        if value.ndim != 3 or value.shape[:2] != (len(AGES), count) \
                or value.shape[2] < 1 or not np.isfinite(value).all():
            raise DevelopmentAdapterError(
                f"formal Phase-A feature malformed: {name}")
        dimensions.add(int(value.shape[2]))
    if len(dimensions) != 1:
        raise DevelopmentAdapterError(
            "formal Phase-A feature dimensions differ across interventions")


@torch.no_grad()
def _collect_phase_a_features(
        host: Any, bank: Any, carrier: torch.nn.Module,
        age: int, indices: np.ndarray, candidate_native: bool,
        device: torch.device) -> dict[str, np.ndarray]:
    """Collect label-free features plus full/reset/prior prediction health."""

    values: dict[str, list[np.ndarray]] = {
        "host": [], "reset": [], "prior": [], "mse": [],
        "reset_mse": [], "prior_mse": [],
    }
    batch_size = SPATIAL_EVAL_BATCH if bank.spatial else 64
    for offset in range(0, len(indices), batch_size):
        local = indices[offset:offset + batch_size]
        features = torch.from_numpy(bank.features(age, local)).to(device)
        actions = torch.from_numpy(bank.actions(local)).to(device)
        prop_np = bank.proprio(local)
        proprio = None if prop_np is None else torch.from_numpy(prop_np).to(
            device)
        full = _carrier_forward(
            carrier, features, actions, candidate_native=candidate_native)
        reset_at = _cue_offset_reset_index(
            spatial=bank.spatial, age=int(age),
            sequence_length=features.shape[1])
        reset_full = _carrier_forward(
            carrier, features[:, reset_at:], actions[:, reset_at:],
            candidate_native=candidate_native)
        if bank.spatial:
            endpoint = 3 + int(age)
            start, stop = endpoint - 3, endpoint
            assert proprio is not None
            with _amp_context(device):
                host_prediction = host.predict(
                    full["fused"][:, start:stop], proprio[:, start:stop],
                    actions[:, start:stop])[:, -1, :, :384]
                prior_prediction = host.predict(
                    full["prior"][:, start:stop], proprio[:, start:stop],
                    actions[:, start:stop])[:, -1, :, :384]
            reset_context = reset_full["fused"][
                :, start - reset_at:stop - reset_at]
            with _amp_context(device):
                reset_prediction = host.predict(
                    reset_context, proprio[:, start:stop],
                    actions[:, start:stop])[:, -1, :, :384]
            target = features[:, endpoint]
            mse_dimensions = (1, 2)
        else:
            endpoint, start, stop = 19, 16, 19
            with _amp_context(device):
                host_prediction = host.predict(
                    full["fused"][:, start:stop],
                    actions[:, start:stop])[:, -1]
                prior_prediction = host.predict(
                    full["prior"][:, start:stop],
                    actions[:, start:stop])[:, -1]
            reset_context = reset_full["fused"][
                :, start - reset_at:stop - reset_at]
            with _amp_context(device):
                reset_prediction = host.predict(
                    reset_context, actions[:, start:stop])[:, -1]
            target = features[:, endpoint]
            mse_dimensions = (1,)
        host_np = host_prediction.float().cpu().numpy()
        reset_np = reset_prediction.float().cpu().numpy()
        prior_np = full["prior"][:, endpoint].float().cpu().numpy()
        if bank.spatial:
            from lewm.official_tasks.dinowm_native_audit import (
                spatial_pyramid_pool,
            )
            host_np = spatial_pyramid_pool(host_np)
            reset_np = spatial_pyramid_pool(reset_np)
            prior_np = spatial_pyramid_pool(prior_np)
        values["host"].append(host_np)
        values["reset"].append(reset_np)
        values["prior"].append(prior_np)
        for name, prediction in (
                ("mse", host_prediction),
                ("reset_mse", reset_prediction),
                ("prior_mse", prior_prediction)):
            values[name].append(torch.mean(
                torch.square(prediction.float() - target.float()),
                dim=mse_dimensions).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in values.items()}


@torch.no_grad()
def _collect_features(
        host: Any, bank: _DevelopmentBank, carrier: torch.nn.Module,
        age: int, indices: np.ndarray, candidate_native: bool,
        device: torch.device) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {
        "host": [], "reset": [], "prior": [], "mse": [], "reset_mse": [],
    }
    batch_size = SPATIAL_EVAL_BATCH if bank.spatial else 64
    for offset in range(0, len(indices), batch_size):
        local = indices[offset:offset + batch_size]
        features = torch.from_numpy(bank.features(age, local)).to(device)
        actions = torch.from_numpy(bank.actions(local)).to(device)
        prop_np = bank.proprio(local)
        proprio = None if prop_np is None else torch.from_numpy(prop_np).to(device)
        full = _carrier_forward(
            carrier, features, actions, candidate_native=candidate_native)
        # The causal reset is applied immediately after the cue leaves the
        # observation stream, not at the beginning of the final predictor
        # context.  DINO cohorts keep the cue at frames 1:4 and move the
        # endpoint; matched LeWM cohorts keep endpoint 19 and move the cue.
        reset_at = _cue_offset_reset_index(
            spatial=bank.spatial, age=int(age), sequence_length=features.shape[1])
        reset_full = _carrier_forward(
            carrier, features[:, reset_at:], actions[:, reset_at:],
            candidate_native=candidate_native)
        if bank.spatial:
            endpoint = 3 + int(age)
            start, stop = endpoint - 3, endpoint
            assert proprio is not None
            with _amp_context(device):
                host_prediction = host.predict(
                    full["fused"][:, start:stop], proprio[:, start:stop],
                    actions[:, start:stop])[:, -1, :, :384]
            reset_context = reset_full["fused"][
                :, start - reset_at:stop - reset_at]
            with _amp_context(device):
                reset_prediction = host.predict(
                    reset_context, proprio[:, start:stop],
                    actions[:, start:stop])[:, -1, :, :384]
            target = features[:, endpoint]
            mse_dimensions = (1, 2)
        else:
            endpoint, start, stop = 19, 16, 19
            with _amp_context(device):
                host_prediction = host.predict(
                    full["fused"][:, start:stop],
                    actions[:, start:stop])[:, -1]
            reset_context = reset_full["fused"][
                :, start - reset_at:stop - reset_at]
            with _amp_context(device):
                reset_prediction = host.predict(
                    reset_context, actions[:, start:stop])[:, -1]
            target = features[:, endpoint]
            mse_dimensions = (1,)
        host_np = host_prediction.float().cpu().numpy()
        reset_np = reset_prediction.float().cpu().numpy()
        prior_np = full["prior"][:, endpoint].float().cpu().numpy()
        if bank.spatial:
            from lewm.official_tasks.dinowm_native_audit import (
                spatial_pyramid_pool,
            )
            host_np = spatial_pyramid_pool(host_np)
            reset_np = spatial_pyramid_pool(reset_np)
            prior_np = spatial_pyramid_pool(prior_np)
        values["host"].append(host_np)
        values["reset"].append(reset_np)
        values["prior"].append(prior_np)
        values["mse"].append(torch.mean(
            torch.square(host_prediction.float() - target.float()),
            dim=mse_dimensions).cpu().numpy())
        values["reset_mse"].append(torch.mean(
            torch.square(reset_prediction.float() - target.float()),
            dim=mse_dimensions).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in values.items()}


def _cue_offset_reset_index(*, spatial: bool, age: int,
                            sequence_length: int = 20) -> int:
    """Return the first post-cue observation for the registered timing design."""

    reset_at = 4 if spatial else 19 - int(age)
    if not 0 < reset_at < int(sequence_length):
        raise DevelopmentAdapterError(
            f"invalid cue-offset reset index {reset_at} for age {age}")
    return reset_at


def _fit_shared_readout(
        train_x: np.ndarray, train_y: np.ndarray,
        heldout_x: np.ndarray, reset_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(train_y, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise DevelopmentAdapterError("readout fit contains fewer than two classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=4000, random_state=0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(np.asarray(train_x), labels)
    return (model.predict(np.asarray(heldout_x)).astype(np.int64),
            model.predict(np.asarray(reset_x)).astype(np.int64))


def _balanced_accuracy(prediction: np.ndarray, truth: np.ndarray,
                       classes: int) -> float:
    prediction = np.asarray(prediction, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    matrix = confusion_matrix(truth, prediction, labels=np.arange(classes))
    denominator = matrix.sum(axis=1)
    if np.any(denominator == 0):
        raise DevelopmentAdapterError("held-out readout omits a semantic class")
    return float(np.mean(np.diag(matrix) / denominator))


def _validate_episode_arrays(arrays: Mapping[str, np.ndarray],
                             classes: int) -> None:
    expected = {
        "episode_id", "class_id", "evidence_age", "retention_correct",
        "reset_correct", "exposure_correct", "next_feature_mse",
        "reset_next_feature_mse", "oracle_success", "execution_success",
    }
    if set(arrays) != expected:
        raise DevelopmentAdapterError("development artifact schema changed")
    lengths = {np.asarray(value).shape for value in arrays.values()}
    if len(lengths) != 1 or not lengths or len(next(iter(lengths))) != 1 \
            or next(iter(lengths))[0] < 1:
        raise DevelopmentAdapterError("development artifact arrays are unaligned")
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    if len(np.unique(episode)) != len(episode):
        raise DevelopmentAdapterError("development episode IDs are not unique")
    class_id = np.asarray(arrays["class_id"], dtype=np.int64)
    if np.any(class_id < 0) or np.any(class_id >= classes):
        raise DevelopmentAdapterError("development class IDs leave range")
    if set(np.unique(arrays["evidence_age"]).tolist()) != set(AGES):
        raise DevelopmentAdapterError("development evidence ages changed")
    for name in ("retention_correct", "reset_correct", "exposure_correct",
                 "oracle_success", "execution_success"):
        if not set(np.unique(arrays[name]).tolist()).issubset({0, 1}):
            raise DevelopmentAdapterError(f"{name} is not binary")
    for name in ("next_feature_mse", "reset_next_feature_mse"):
        value = np.asarray(arrays[name], dtype=np.float64)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise DevelopmentAdapterError(f"{name} is invalid")


def _finite_nonnegative(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0.0)


__all__ = [
    "SAGE_MEM_HOST_ADAPTER_API_VERSION", "FormalIntegrationPending",
    "DevelopmentAdapterError", "SageMemV1HostAdapter",
    "build_host_adapter",
]
