#!/usr/bin/env python3
"""Label-free host-aligned evidence writer on frozen official PushT LeWM.

This is an exploratory model-agnostic extension of the Mem-JEPA / Host-Writer
idea to the released LeWorldModel PushT checkpoint.  It deliberately writes to
a separate output root and does not mutate the locked formal
``outputs/official_pusht_memory`` carrier grid.

Adapter training is label-free: cue latents define contrastive evidence
targets; semantic labels are used only after training for retention readouts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm_pusht import (  # noqa: E402
    load_official_pusht_checkpoint,
)
from lewm.models.official_lewm import preprocess_frames  # noqa: E402
from lewm.official_tasks.artifacts import sha256_file as artifact_sha256_file  # noqa: E402
from lewm.official_tasks.pusht_hdf5 import OfficialPushTHDF5  # noqa: E402
from lewm.official_tasks.pusht_memory import render_single_overlay  # noqa: E402
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    aligned_pusht_latents,
    load_pusht_base_cache,
    load_pusht_task_cache,
    pusht_admission_path,
    pusht_task_manifest_path,
    pusht_task_spec,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    pusht_lock_receipt,
    resolve_pusht_path,
)
from scripts.run_mem_jepa_stage_b import (  # noqa: E402
    atomic_json,
    fit_classifier,
    require,
    resolve,
    set_determinism,
)
from scripts.run_mem_jepa_stage_c import (  # noqa: E402
    contrastive_loss,
    positive_cosine_loss,
)


DEFAULT_OUTPUT = ROOT / "outputs/lewm_pusht_host_writer_v1"
DEFAULT_COUNTERFACTUAL_CACHE = ROOT / "outputs/lewm_pusht_counterfactual_cue_cache_v1"
SUPPORTED_AGES = (4, 8, 15)
LATENT_DIM = 192
ACTION_DIM = 10
COUNTERFACTUAL_TARGET_MODES = {"counterfactual_delta_flat"}
PROTOTYPE_TARGET_MODES = {"prototype_delta_flat"}


def cue_start_for_age(spec: dict[str, Any], age: int) -> int:
    sequence = spec["sequence"]
    start = (
        int(sequence["decision_index"])
        - int(age)
        - int(sequence["cue_length"])
    )
    if start < 0 or start + int(sequence["cue_length"]) >= int(sequence["decision_index"]):
        raise ValueError(f"unsupported PushT cue age {age}: cue_start={start}")
    return start


def age_adjusted_spec(locked_spec: dict[str, Any], age: int) -> dict[str, Any]:
    if int(age) not in SUPPORTED_AGES:
        raise ValueError(f"supported ages are {SUPPORTED_AGES}, got {age}")
    spec = copy.deepcopy(locked_spec)
    spec["sequence"]["cue_start"] = cue_start_for_age(spec, int(age))
    spec["sequence"]["cue_interval"] = [
        int(spec["sequence"]["cue_start"]),
        int(spec["sequence"]["cue_start"]) + int(spec["sequence"]["cue_length"]),
    ]
    spec["sequence"]["evidence_age"] = int(age)
    return spec


def classification_record(prediction: np.ndarray, truth: np.ndarray,
                          classes: int) -> dict[str, Any]:
    labels = np.arange(classes, dtype=np.int64)
    matrix = confusion_matrix(truth, prediction, labels=labels)
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    return {
        "balanced_accuracy": balanced_accuracy(prediction, truth, classes),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
        "count": int(len(truth)),
    }


def balanced_accuracy(prediction: np.ndarray, truth: np.ndarray,
                      classes: int) -> float:
    labels = np.arange(classes, dtype=np.int64)
    values = []
    for label in labels:
        mask = truth == label
        if np.any(mask):
            values.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(values)) if values else 0.0


def best_cluster_label_agreement(cluster: np.ndarray, labels: np.ndarray,
                                 classes: int) -> float:
    # Brute force is fine for 4/6 classes and avoids a scipy dependency.
    import itertools
    best = 0.0
    for perm in itertools.permutations(range(classes)):
        mapped = np.asarray([perm[int(value)] for value in cluster],
                            dtype=np.int64)
        best = max(best, float(np.mean(mapped == labels)))
    return best


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_admitted(spec: dict[str, Any], task_key: str, split: str) -> dict[str, Any]:
    task_manifest_path = pusht_task_manifest_path(spec, task_key)
    admission_path = pusht_admission_path(spec, task_key)
    if not task_manifest_path.is_file() or not admission_path.is_file():
        raise FileNotFoundError(f"{task_key} has no completed frozen admission")
    manifest = json.loads(task_manifest_path.read_text())
    admission = json.loads(admission_path.read_text())
    if manifest.get("formal_lock") != pusht_lock_receipt(spec) \
            or admission.get("formal_lock") != pusht_lock_receipt(spec):
        raise RuntimeError("task cache/admission uses a different formal lock")
    receipt = manifest.get("admission", {})
    if receipt.get("sha256") != artifact_sha256_file(admission_path) \
            or receipt.get("admitted") is not True \
            or admission.get("admitted") is not True:
        raise RuntimeError(f"{task_key} did not pass every frozen admission")
    base, base_meta = load_pusht_base_cache(spec, split)
    task, task_meta = load_pusht_task_cache(spec, task_key, split)
    z = aligned_pusht_latents(
        base, task, int(spec["sequence"]["cue_start"]),
        int(spec["sequence"]["cue_length"]))
    return {
        **base,
        **task,
        "z": z,
        "base_meta": base_meta,
        "task_meta": task_meta,
    }


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)


@torch.inference_mode()
def encode_frame_stream(model: torch.nn.Module,
                        indexed_frames: Iterable[tuple[int, np.ndarray]],
                        total_frames: int, frame_batch_size: int,
                        image_size: int,
                        device: torch.device) -> np.ndarray:
    output = np.empty((total_frames, LATENT_DIM), dtype=np.float32)
    seen = np.zeros(total_frames, dtype=np.bool_)
    indices: list[int] = []
    frames: list[np.ndarray] = []

    def flush() -> None:
        if not frames:
            return
        batch = np.stack(frames)
        pixels = torch.from_numpy(batch).permute(0, 3, 1, 2).to(
            device, non_blocking=True)
        pixels = preprocess_frames(pixels, image_size=image_size)
        encoded = model.encode_pixels(pixels).float().cpu().numpy()
        if encoded.shape != (len(indices), LATENT_DIM) \
                or not np.isfinite(encoded).all():
            raise RuntimeError("official LeWM encoder returned invalid latents")
        output[np.asarray(indices, dtype=np.int64)] = encoded
        indices.clear()
        frames.clear()

    for index, frame in indexed_frames:
        if not 0 <= index < total_frames or seen[index]:
            raise ValueError(f"duplicate or invalid frame index {index}")
        value = np.asarray(frame)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("streamed frames must be uint8 HWC RGB")
        seen[index] = True
        indices.append(index)
        frames.append(value)
        if len(frames) == frame_batch_size:
            flush()
    flush()
    if not seen.all():
        raise ValueError(
            f"frame stream omitted {int((~seen).sum())} positions")
    return output


def counterfactual_cache_path(cache_root: Path, task: str,
                              split: str, age: int) -> Path:
    return cache_root / task / f"age_{int(age)}" / f"{split}.npz"


def install_counterfactual_cue(data: dict[str, Any], spec: dict[str, Any],
                               z_counterfactual: np.ndarray,
                               z_observed: np.ndarray,
                               matched_label: np.ndarray) -> None:
    cue_start = int(spec["sequence"]["cue_start"])
    cue_length = int(spec["sequence"]["cue_length"])
    cue_end = cue_start + cue_length
    if z_observed.shape != data["z_base"][:, cue_start:cue_end].shape:
        raise RuntimeError("counterfactual observed cue shape mismatch")
    z = np.asarray(data["z_base"], dtype=np.float32).copy()
    z[:, cue_start:cue_end] = z_observed.astype(np.float32)
    data["z"] = z
    data["z_cue"] = z_observed.astype(np.float32)
    data["z_counterfactual"] = z_counterfactual.astype(np.float32)
    data["counterfactual_matched_label"] = matched_label.astype(np.int64)


def load_or_build_counterfactual_cache(
        data: dict[str, Any], spec: dict[str, Any], task_key: str,
        split: str, host: torch.nn.Module, device: torch.device,
        cache_root: Path, frame_batch_size: int) -> dict[str, Any]:
    """Load or build all cue-label alternatives for a PushT task split.

    The stored tensor is shaped (episodes, classes, cue_length, 192).  Training
    still uses the actually observed cue as the positive; alternatives are
    structured counterfactual negatives generated by the renderer.
    """

    age = int(spec["sequence"].get("evidence_age", 15))
    path = counterfactual_cache_path(cache_root, task_key, split, age)
    if path.is_file():
        with np.load(path, allow_pickle=False) as handle:
            z_counterfactual = np.asarray(handle["z_counterfactual"],
                                          dtype=np.float32)
            z_observed = np.asarray(handle["z_observed"], dtype=np.float32)
            episode_index = np.asarray(handle["episode_index"], dtype=np.int64)
            local_start = np.asarray(handle["local_start"], dtype=np.int64)
            matched_label = np.asarray(handle["matched_label"], dtype=np.int64)
        if not np.array_equal(episode_index, data["episode_index"]) \
                or not np.array_equal(local_start, data["local_start"]):
            raise RuntimeError(f"counterfactual cache selection mismatch: {path}")
        install_counterfactual_cue(
            data, spec, z_counterfactual, z_observed, matched_label)
        return {
            "path": str(path.relative_to(ROOT)),
            "age": age,
            "cue_start": int(spec["sequence"]["cue_start"]),
            "loaded": True,
            "matched_label_agreement": float(np.mean(matched_label == data["labels"])),
        }

    dataset_path = resolve_pusht_path(spec["dataset"]["hdf5_path"])
    dataset = OfficialPushTHDF5(
        dataset_path, expected_hdf5_sha256=spec["dataset"]["hdf5_sha256"])
    task = pusht_task_spec(spec, task_key)
    classes = int(task["classes"])
    count = len(data["labels"])
    cue_start = int(spec["sequence"]["cue_start"])
    cue_length = int(spec["sequence"]["cue_length"])
    num_frames = int(spec["sequence"]["num_frames"])
    image_size = int(spec["official_host"]["image_size"])
    total = count * classes * cue_length

    def stream() -> Iterable[tuple[int, np.ndarray]]:
        started = time.time()
        for row in range(count):
            if row and row % 10 == 0:
                elapsed = time.time() - started
                print(
                    f"[lewm-counterfactual-cache] task={task_key} "
                    f"split={split} rows={row}/{count} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            native = dataset.read_sequence(
                int(data["episode_index"][row]), int(data["local_start"][row]),
                num_frames)
            for label in range(classes):
                overlaid = render_single_overlay(
                    native.frames, task["display_name"], label,
                    cue_start, cue_length)
                for cue_offset in range(cue_length):
                    index = (row * classes + label) * cue_length + cue_offset
                    yield index, overlaid[cue_start + cue_offset]

    z_counterfactual = encode_frame_stream(
        host, stream(), total, frame_batch_size, image_size, device,
    ).reshape(count, classes, cue_length, LATENT_DIM)
    observed_label = np.asarray(data["labels"], dtype=np.int64)
    z_observed = z_counterfactual[np.arange(count), observed_label]
    deltas = np.mean(
        np.square(z_counterfactual - z_observed[:, None, :, :]),
        axis=(2, 3))
    matched_label = np.argmin(deltas, axis=1).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            z_counterfactual=z_counterfactual.astype(np.float32),
            z_observed=z_observed.astype(np.float32),
            episode_index=np.asarray(data["episode_index"], dtype=np.int64),
            local_start=np.asarray(data["local_start"], dtype=np.int64),
            labels=np.asarray(data["labels"], dtype=np.int64),
            matched_label=matched_label,
            cue_start=np.asarray([cue_start], dtype=np.int64),
            evidence_age=np.asarray([age], dtype=np.int64),
        )
    os.replace(tmp, path)
    install_counterfactual_cue(
        data, spec, z_counterfactual, z_observed, matched_label)
    return {
        "path": str(path.relative_to(ROOT)),
        "age": age,
        "cue_start": int(spec["sequence"]["cue_start"]),
        "loaded": False,
        "matched_label_agreement": float(np.mean(matched_label == data["labels"])),
    }


class LeWMHostAlignedEvidenceWriter(nn.Module):
    """Slot memory plus residual writer for a frozen LeWM predictor interface."""

    def __init__(self, *, target_dim: int, dim: int, slots: int, heads: int,
                 max_frames: int = 20, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.z_proj = nn.Linear(LATENT_DIM, dim)
        self.action_proj = nn.Linear(ACTION_DIM, dim)
        self.time = nn.Embedding(max_frames, dim)
        self.assign = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, slots),
        )
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.writer = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, LATENT_DIM),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.evidence_decoder = nn.Sequential(
            nn.LayerNorm(slots * dim),
            nn.Linear(slots * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, self.target_dim),
        )
        self.host_query = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, self.target_dim),
        )
        self.context_query = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, self.target_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        nn.init.zeros_(self.writer[-1].weight)
        nn.init.zeros_(self.writer[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)

    def _tokens(self, z: torch.Tensor, actions: torch.Tensor,
                times: torch.Tensor) -> torch.Tensor:
        if actions.shape[1] != z.shape[1]:
            raise ValueError("z/actions must have aligned time length")
        return (
            self.z_proj(z)
            + self.action_proj(actions)
            + self.time(times)[None, :, :]
        )

    def _slots(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.assign(tokens).transpose(1, 2)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bst,btd->bsd", weights, tokens)

    def inject(self, prefix_z: torch.Tensor, prefix_actions: torch.Tensor,
               prefix_times: torch.Tensor, context_z: torch.Tensor,
               context_actions: torch.Tensor,
               context_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_tokens = self._tokens(prefix_z, prefix_actions, prefix_times)
        context_tokens = self._tokens(context_z, context_actions, context_times)
        slots = self._slots(prefix_tokens)
        attended, _ = self.cross(context_tokens, slots, slots, need_weights=False)
        joined = torch.cat((context_tokens, attended), dim=-1)
        delta = self.writer(joined)
        gate = torch.sigmoid(self.gate(joined))
        fused = context_z + self.residual_scale * gate * delta
        return fused, slots

    def decode_evidence(self, slots: torch.Tensor) -> torch.Tensor:
        return self.evidence_decoder(slots.reshape(slots.shape[0], -1))

    def query_host(self, z: torch.Tensor) -> torch.Tensor:
        return self.host_query(z)

    def query_context(self, z: torch.Tensor) -> torch.Tensor:
        return self.context_query(z)


def evidence_targets(data: dict[str, Any], rows: np.ndarray, device: torch.device,
                     *, spec: dict[str, Any], target_mode: str, candidate_count: int,
                     shuffle_targets: bool) -> torch.Tensor:
    z_cue = data["z_cue"][rows]
    cue_start = int(spec["sequence"]["cue_start"])
    cue_end = cue_start + z_cue.shape[1]
    if target_mode == "cue_flat":
        target = tensor(z_cue.reshape(len(rows), -1), device)
    elif target_mode == "cue_mean":
        target = tensor(z_cue.mean(axis=1), device)
    elif target_mode == "delta_flat":
        base = data["z_base"][rows, cue_start:cue_end]
        target = tensor((z_cue - base).reshape(len(rows), -1), device)
    elif target_mode == "delta_mean":
        base = data["z_base"][rows, cue_start:cue_end]
        target = tensor((z_cue - base).mean(axis=1), device)
    elif target_mode == "counterfactual_delta_flat":
        if "z_counterfactual" not in data:
            raise RuntimeError("counterfactual target mode requires z_counterfactual")
        z_all = data["z_counterfactual"][rows]
        base = data["z_base"][rows, cue_start:cue_end]
        deltas = z_all - base[:, None, :, :]
        flat = deltas.reshape(len(rows), deltas.shape[1], -1)
        # Use the rendered branch that matches the observed cue as positive.
        scores = np.mean(
            np.square(z_all - z_cue[:, None, :, :]), axis=(2, 3))
        positive = np.argmin(scores, axis=1)
        order = []
        for value in positive:
            first = int(value)
            order.append([first] + [
                idx for idx in range(flat.shape[1]) if idx != first
            ])
        order_np = np.asarray(order, dtype=np.int64)
        selected = flat[np.arange(len(rows))[:, None], order_np]
        if candidate_count < selected.shape[1]:
            selected = selected[:, :candidate_count]
        target_candidates = tensor(selected, device)
        if shuffle_targets:
            target_candidates = target_candidates[
                torch.randperm(len(target_candidates), device=device)]
        return target_candidates
    elif target_mode == "prototype_delta_flat":
        if "prototype_centers" not in data or "prototype_index" not in data:
            raise RuntimeError("prototype target mode requires fitted prototypes")
        centers = np.asarray(data["prototype_centers"], dtype=np.float32)
        cluster = np.asarray(data["prototype_index"][rows], dtype=np.int64)
        order = []
        for value in cluster:
            first = int(value)
            order.append([first] + [
                idx for idx in range(centers.shape[0]) if idx != first
            ])
        selected = centers[np.asarray(order, dtype=np.int64)]
        if candidate_count < selected.shape[1]:
            selected = selected[:, :candidate_count]
        target_candidates = tensor(selected, device)
        if shuffle_targets:
            target_candidates = target_candidates[
                torch.randperm(len(target_candidates), device=device)]
        return target_candidates
    else:
        raise ValueError(f"unknown target mode: {target_mode}")
    if shuffle_targets:
        target = target[torch.randperm(len(target), device=device)]
    candidates = [target]
    for shift in range(1, max(2, int(candidate_count))):
        candidates.append(target.roll(shift, dims=0))
    return torch.stack(candidates, dim=1)


def target_dim_for_mode(mode: str, cue_length: int) -> int:
    if mode.endswith("_mean"):
        return LATENT_DIM
    return LATENT_DIM * int(cue_length)


def fit_prototype_targets(train: dict[str, Any], validation: dict[str, Any],
                          spec: dict[str, Any], *, classes: int,
                          seed: int) -> dict[str, Any]:
    cue_start = int(spec["sequence"]["cue_start"])
    cue_length = int(spec["sequence"]["cue_length"])
    cue_end = cue_start + cue_length

    def features(data: dict[str, Any]) -> np.ndarray:
        delta = data["z_cue"] - data["z_base"][:, cue_start:cue_end]
        return delta.reshape(len(delta), -1).astype(np.float32)

    train_x = features(train)
    validation_x = features(validation)
    model = KMeans(
        n_clusters=classes,
        n_init=20,
        random_state=int(seed),
        max_iter=1000,
    )
    train_idx = model.fit_predict(train_x).astype(np.int64)
    validation_idx = model.predict(validation_x).astype(np.int64)
    centers = model.cluster_centers_.astype(np.float32)
    for data, idx in ((train, train_idx), (validation, validation_idx)):
        data["prototype_centers"] = centers
        data["prototype_index"] = idx
    return {
        "schema": "lewm_self_discovered_cue_prototypes_v1",
        "source": "KMeans over train cue-delta evidence; labels not used",
        "clusters": int(classes),
        "target_dim": int(centers.shape[1]),
        "train_inertia": float(model.inertia_),
        "train_label_agreement_after_fit": best_cluster_label_agreement(
            train_idx, train["labels"], classes),
        "validation_label_agreement_after_fit": best_cluster_label_agreement(
            validation_idx, validation["labels"], classes),
    }


def batch_arrays(data: dict[str, Any], rows: np.ndarray, spec: dict[str, Any],
                 condition: str, device: torch.device) -> dict[str, torch.Tensor]:
    seq = spec["sequence"]
    decision = int(seq["decision_index"])
    context = np.asarray(seq["final_context_indices"], dtype=np.int64)
    z = data["z"][rows]
    actions = data["actions"][rows]
    if condition == "full":
        prefix_indices = np.arange(decision, dtype=np.int64)
    elif condition == "reset":
        prefix_indices = context
    else:
        raise ValueError(f"unknown recurrent condition: {condition}")
    return {
        "prefix_z": tensor(z[:, prefix_indices], device),
        "prefix_actions": tensor(actions[:, prefix_indices], device),
        "prefix_times": torch.as_tensor(prefix_indices, device=device,
                                        dtype=torch.long),
        "context_z": tensor(z[:, context], device),
        "context_actions": tensor(actions[:, context], device),
        "context_times": torch.as_tensor(context, device=device,
                                         dtype=torch.long),
    }


def no_state_arrays(data: dict[str, Any], rows: np.ndarray, spec: dict[str, Any],
                    device: torch.device) -> dict[str, torch.Tensor]:
    context = np.asarray(spec["sequence"]["final_context_indices"],
                         dtype=np.int64)
    return {
        "context_z": tensor(data["z"][rows][:, context], device),
        "context_actions": tensor(data["actions"][rows][:, context], device),
    }


def predict_last(host: torch.nn.Module, context_z: torch.Tensor,
                 context_actions: torch.Tensor) -> torch.Tensor:
    if context_z.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return host.predict(context_z, context_actions)[:, -1].float()
    return host.predict(context_z, context_actions)[:, -1].float()


def train_one_cell(model: LeWMHostAlignedEvidenceWriter, host: torch.nn.Module,
                   train: dict[str, Any], spec: dict[str, Any], *,
                   task: str, seed: int, epochs: int, batch_size: int,
                   lr: float, weight_decay: float, temperature: float,
                   candidate_count: int, target_mode: str, variant: str,
                   host_weight: float, context_weight: float,
                   memory_weight: float, residual_l2_weight: float,
                   output_dir: Path) -> list[dict[str, Any]]:
    indices = np.arange(len(train["labels"]), dtype=np.int64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs))
    rng = np.random.default_rng(941_000 + seed)
    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "loss", "host_loss", "context_loss", "memory_loss",
        "host_match", "context_match", "memory_match", "host_cos",
        "context_cos", "memory_cos", "residual_l2", "lr", "seconds",
    ]
    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            rng.shuffle(indices)
            model.train()
            losses: list[float] = []
            host_losses: list[float] = []
            context_losses: list[float] = []
            memory_losses: list[float] = []
            host_matches: list[float] = []
            context_matches: list[float] = []
            memory_matches: list[float] = []
            host_cos_values: list[float] = []
            context_cos_values: list[float] = []
            memory_cos_values: list[float] = []
            regs: list[float] = []
            for offset in range(0, len(indices), batch_size):
                rows = indices[offset:offset + batch_size]
                if len(rows) < 4:
                    continue
                batch = batch_arrays(train, rows, spec, "full",
                                     next(model.parameters()).device)
                candidates = evidence_targets(
                    train, rows, next(model.parameters()).device,
                    spec=spec,
                    target_mode=target_mode,
                    candidate_count=candidate_count,
                    shuffle_targets=variant == "shuffle_targets")
                fused, slots = model.inject(
                    batch["prefix_z"], batch["prefix_actions"],
                    batch["prefix_times"], batch["context_z"],
                    batch["context_actions"], batch["context_times"])
                predicted = predict_last(host, fused, batch["context_actions"])
                host_query = model.query_host(predicted)
                context_query = model.query_context(fused[:, -1])
                memory_query = model.decode_evidence(slots)
                host_loss, host_match = contrastive_loss(
                    host_query, candidates, temperature)
                context_loss, context_match = contrastive_loss(
                    context_query, candidates, temperature)
                memory_loss, memory_match = contrastive_loss(
                    memory_query, candidates, temperature)
                host_cos = positive_cosine_loss(host_query, candidates)
                context_cos = positive_cosine_loss(context_query, candidates)
                memory_cos = positive_cosine_loss(memory_query, candidates)
                residual_l2 = torch.mean(torch.square(fused - batch["context_z"]))
                terms = [
                    0.5 * float(memory_weight) * memory_loss,
                    0.25 * float(memory_weight) * memory_cos,
                ]
                if variant != "no_host":
                    terms.extend([
                        float(host_weight) * host_loss,
                        0.5 * float(host_weight) * host_cos,
                    ])
                if variant != "no_context":
                    terms.extend([
                        float(context_weight) * context_loss,
                        float(context_weight) * context_cos,
                    ])
                terms.append(float(residual_l2_weight) * residual_l2)
                loss = sum(terms)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                host_losses.append(float(host_loss.detach().cpu()))
                context_losses.append(float(context_loss.detach().cpu()))
                memory_losses.append(float(memory_loss.detach().cpu()))
                host_matches.append(host_match)
                context_matches.append(context_match)
                memory_matches.append(memory_match)
                host_cos_values.append(float(host_cos.detach().cpu()))
                context_cos_values.append(float(context_cos.detach().cpu()))
                memory_cos_values.append(float(memory_cos.detach().cpu()))
                regs.append(float(residual_l2.detach().cpu()))
            scheduler.step()
            record = {
                "epoch": int(epoch),
                "loss": float(np.mean(losses)),
                "host_loss": float(np.mean(host_losses)),
                "context_loss": float(np.mean(context_losses)),
                "memory_loss": float(np.mean(memory_losses)),
                "host_match": float(np.mean(host_matches)),
                "context_match": float(np.mean(context_matches)),
                "memory_match": float(np.mean(memory_matches)),
                "host_cos": float(np.mean(host_cos_values)),
                "context_cos": float(np.mean(context_cos_values)),
                "memory_cos": float(np.mean(memory_cos_values)),
                "residual_l2": float(np.mean(regs)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": float(time.time() - started),
            }
            history.append(record)
            writer.writerow(record)
            stream.flush()
            print(
                f"[lewm-host-writer] task={task} seed={seed} "
                f"epoch={epoch}/{epochs} loss={record['loss']:.4f} "
                f"host_match={record['host_match']:.3f} "
                f"ctx_match={record['context_match']:.3f} "
                f"mem_match={record['memory_match']:.3f} "
                f"sec={record['seconds']:.1f}",
                flush=True,
            )
    return history


@torch.no_grad()
def collect_level_features(model: LeWMHostAlignedEvidenceWriter,
                           host: torch.nn.Module, data: dict[str, Any],
                           spec: dict[str, Any], *, condition: str,
                           level: str, batch_size: int,
                           device: torch.device) -> np.ndarray:
    model.eval()
    indices = np.arange(len(data["labels"]), dtype=np.int64)
    features = []
    for offset in range(0, len(indices), batch_size):
        rows = indices[offset:offset + batch_size]
        if condition == "no_state":
            require(level in {"host_output", "injected_context"},
                    "no-state is meaningful only for host/context features")
            batch = no_state_arrays(data, rows, spec, device)
            if level == "host_output":
                value = predict_last(host, batch["context_z"],
                                     batch["context_actions"])
            else:
                value = batch["context_z"].reshape(len(rows), -1)
        else:
            batch = batch_arrays(data, rows, spec, condition, device)
            fused, slots = model.inject(
                batch["prefix_z"], batch["prefix_actions"],
                batch["prefix_times"], batch["context_z"],
                batch["context_actions"], batch["context_times"])
            if level == "memory_prior":
                value = model.decode_evidence(slots)
            elif level == "injected_context":
                value = fused.reshape(len(rows), -1)
            elif level == "host_output":
                value = predict_last(host, fused, batch["context_actions"])
            else:
                raise ValueError(f"unknown diagnostic level: {level}")
        features.append(value.float().cpu().numpy())
    return np.concatenate(features)


def evaluate_one_cell(model: LeWMHostAlignedEvidenceWriter,
                      host: torch.nn.Module, train: dict[str, Any],
                      validation: dict[str, Any], spec: dict[str, Any], *,
                      classes: int, batch_size: int,
                      device: torch.device) -> dict[str, Any]:
    train_y = train["labels"]
    validation_y = validation["labels"]
    train_full = collect_level_features(
        model, host, train, spec, condition="full", level="host_output",
        batch_size=batch_size, device=device)
    records = {}
    for condition in ("full", "reset", "no_state"):
        features = collect_level_features(
            model, host, validation, spec, condition=condition,
            level="host_output", batch_size=batch_size, device=device)
        prediction = fit_classifier(train_full, train_y, features)
        records[condition] = classification_record(
            prediction, validation_y, classes)
    control_maximum = 1.0 / float(classes) + 0.05
    return {
        "records": records,
        "gate": {
            "full_minimum": 0.75,
            "control_maximum": control_maximum,
            "passed": bool(
                records["full"]["balanced_accuracy"] >= 0.75
                and records["reset"]["balanced_accuracy"] <= control_maximum
                and records["no_state"]["balanced_accuracy"] <= control_maximum),
        },
    }


def evaluate_diagnostics(model: LeWMHostAlignedEvidenceWriter,
                         host: torch.nn.Module, train: dict[str, Any],
                         validation: dict[str, Any], spec: dict[str, Any], *,
                         classes: int, batch_size: int,
                         device: torch.device) -> dict[str, Any]:
    train_y = train["labels"]
    validation_y = validation["labels"]
    levels = {
        "memory_prior": ("full", "reset"),
        "injected_context": ("full", "reset", "no_state"),
        "host_output": ("full", "reset", "no_state"),
    }
    diagnostics: dict[str, Any] = {
        "schema": "lewm_pusht_host_writer_diagnostics_v1",
        "readout": "train_on_train_full_apply_to_validation_conditions",
        "levels": {},
    }
    for level, conditions in levels.items():
        train_full = collect_level_features(
            model, host, train, spec, condition="full", level=level,
            batch_size=batch_size, device=device)
        records = {}
        for condition in conditions:
            features = collect_level_features(
                model, host, validation, spec, condition=condition, level=level,
                batch_size=batch_size, device=device)
            prediction = fit_classifier(train_full, train_y, features)
            records[condition] = classification_record(
                prediction, validation_y, classes)
        full_bacc = float(records["full"]["balanced_accuracy"])
        reset_bacc = float(records["reset"]["balanced_accuracy"])
        no_state_bacc = records.get("no_state", {}).get("balanced_accuracy")
        diagnostics["levels"][level] = {
            "feature_dim": int(train_full.shape[1]),
            "records": records,
            "full_minus_reset_pp": float(100.0 * (full_bacc - reset_bacc)),
            "full_minus_no_state_pp": (
                None if no_state_bacc is None
                else float(100.0 * (full_bacc - float(no_state_bacc)))
            ),
        }
    return diagnostics


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve(args.output)
    output = output_root / args.task / f"s{args.seed}" / f"age_{args.age}"
    require(not (output / "result.json").exists() or args.overwrite,
            f"result already exists: {output / 'result.json'}")
    output.mkdir(parents=True, exist_ok=True)
    locked_spec = load_locked_pusht_spec(resolve(args.spec), resolve(args.lock))
    spec = age_adjusted_spec(locked_spec, int(args.age))
    if int(args.age) != 15 and args.target_mode not in COUNTERFACTUAL_TARGET_MODES:
        raise RuntimeError(
            "LeWM age4/age8 runs require counterfactual_delta_flat so the "
            "age-specific observed cue can be rendered and cached.")
    task_record = pusht_task_spec(locked_spec, args.task)
    classes = int(task_record["classes"])
    cue_length = int(spec["sequence"]["cue_length"])
    target_dim = target_dim_for_mode(args.target_mode, cue_length)
    set_determinism(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train = load_admitted(locked_spec, args.task, "train")
    validation = load_admitted(locked_spec, args.task, "validation")
    bundle = resolve_pusht_path(locked_spec["official_host"]["bundle_path"])
    host = load_official_pusht_checkpoint(bundle, device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    host_before = state_digest(host)
    counterfactual_cache_records = None
    if args.target_mode in COUNTERFACTUAL_TARGET_MODES:
        cache_root = resolve(args.counterfactual_cache)
        counterfactual_cache_records = {
            "train": load_or_build_counterfactual_cache(
                train, spec, args.task, "train", host, device, cache_root,
                int(args.frame_batch_size)),
            "validation": load_or_build_counterfactual_cache(
                validation, spec, args.task, "validation", host, device,
                cache_root, int(args.frame_batch_size)),
        }
    prototype_record = None
    if args.target_mode in PROTOTYPE_TARGET_MODES:
        prototype_record = fit_prototype_targets(
            train, validation, spec, classes=classes, seed=args.seed)
    model = LeWMHostAlignedEvidenceWriter(
        target_dim=target_dim, dim=args.dim, slots=args.slots,
        heads=args.heads, residual_scale=args.residual_scale).to(device)
    started = time.time()
    history = train_one_cell(
        model, host, train, spec, task=args.task, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, temperature=args.temperature,
        candidate_count=args.candidate_count, target_mode=args.target_mode,
        variant=args.variant, host_weight=args.host_weight,
        context_weight=args.context_weight, memory_weight=args.memory_weight,
        residual_l2_weight=args.residual_l2_weight, output_dir=output)
    metrics = evaluate_one_cell(
        model, host, train, validation, spec, classes=classes,
        batch_size=args.eval_batch_size, device=device)
    diagnostics = None
    if args.diagnostics:
        diagnostics = evaluate_diagnostics(
            model, host, train, validation, spec, classes=classes,
            batch_size=args.eval_batch_size, device=device)
    host_after = state_digest(host)
    require(host_before == host_after,
            "frozen official PushT LeWM host changed during training")
    checkpoint_path = output / "adapter.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}")
    torch.save({
        "schema": "lewm_pusht_host_writer_adapter_checkpoint_v1",
        "adapter": "host_aligned_evidence_writer",
        "model_state_dict": model.state_dict(),
        "model": {
            "latent_dim": LATENT_DIM,
            "target_dim": target_dim,
            "dim": int(args.dim),
            "slots": int(args.slots),
            "heads": int(args.heads),
            "residual_scale": float(args.residual_scale),
        },
        "task": args.task,
        "age": int(args.age),
        "seed": int(args.seed),
        "target_mode": args.target_mode,
        "candidate_count": int(args.candidate_count),
        "spec": str(resolve(args.spec).relative_to(ROOT)),
        "spec_sha256": artifact_sha256_file(resolve(args.spec)),
        "formal_lock": pusht_lock_receipt(locked_spec),
        "host_digest": host_after,
        "endpoint": {
            "decision_index": int(spec["sequence"]["decision_index"]),
            "raw_context_indices": list(spec["sequence"]["final_context_indices"]),
            "cue_start": int(spec["sequence"]["cue_start"]),
            "cue_length": cue_length,
            "evidence_age": int(args.age),
        },
        "labels_used_for_adapter_training": False,
    }, checkpoint_path)
    checkpoint_sha256 = artifact_sha256_file(checkpoint_path)
    result = {
        "schema": "lewm_pusht_host_writer_cell_v1",
        "status": "completed",
        "claim_boundary": (
            "Exploratory label-free Host-Aligned Evidence Writer on the "
            "frozen official PushT LeWorldModel host. Semantic labels are used "
            "only for readout after adapter training."),
        "labels_used_for_adapter_training": False,
        "task": args.task,
        "semantic_name": task_record["display_name"],
        "classes": classes,
        "age": int(args.age),
        "seed": int(args.seed),
        "target_mode": args.target_mode,
        "counterfactual_cache": counterfactual_cache_records,
        "self_discovered_prototypes": prototype_record,
        "candidate_count": int(args.candidate_count),
        "variant": args.variant,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "spec": str(resolve(args.spec).relative_to(ROOT)),
        "spec_sha256": artifact_sha256_file(resolve(args.spec)),
        "formal_lock": pusht_lock_receipt(locked_spec),
        "host_digest_unchanged": True,
        "host_digest": host_after,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": checkpoint_sha256,
            "schema": "lewm_pusht_host_writer_adapter_checkpoint_v1",
        },
        "endpoint": {
            "decision_index": int(spec["sequence"]["decision_index"]),
            "decision_observation_excluded": True,
            "raw_context_indices": list(spec["sequence"]["final_context_indices"]),
            "cue_start": int(spec["sequence"]["cue_start"]),
            "cue_length": cue_length,
            "evidence_age": int(args.age),
            "training_condition": "full prefix z[0:19], before consuming z[19]",
            "reset_condition": "prefix reduced to legal final context only",
        },
        "model": {
            "adapter": "host_aligned_evidence_writer",
            "latent_dim": LATENT_DIM,
            "target_dim": target_dim,
            "dim": int(args.dim),
            "slots": int(args.slots),
            "heads": int(args.heads),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "loss_weights": {
                "host": float(args.host_weight),
                "context": float(args.context_weight),
                "memory": float(args.memory_weight),
                "residual_l2": float(args.residual_l2_weight),
            },
            "final": history[-1] if history else None,
        },
        "metrics": metrics,
        "elapsed_seconds": float(time.time() - started),
    }
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    atomic_json(output / "result.json", result)
    print(json.dumps({
        "task": args.task,
        "age": args.age,
        "seed": args.seed,
        "passed": metrics["gate"]["passed"],
        "full": metrics["records"]["full"]["balanced_accuracy"],
        "reset": metrics["records"]["reset"]["balanced_accuracy"],
        "no_state": metrics["records"]["no_state"]["balanced_accuracy"],
    }, indent=2), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    summary: dict[str, Any] = {
        "schema": "lewm_pusht_host_writer_summary_v1",
        "status": "completed",
        "claim_boundary": (
            "Aggregates exploratory LeWM PushT Host-Aligned Evidence Writer "
            "runs. Labels remain readout-only."),
        "labels_used_for_adapter_training": False,
        "target_mode": args.target_mode,
        "tasks": {},
        "updated_unix": time.time(),
    }
    all_exist = True
    all_passed = True
    for task in args.tasks:
        summary["tasks"][task] = {}
        for age in args.ages:
            records = []
            for seed in args.seeds:
                path = output / task / f"s{seed}" / f"age_{age}" / "result.json"
                if not path.is_file():
                    all_exist = False
                    continue
                records.append(json.loads(path.read_text()))
            if not records:
                continue
            condition_values = {
                condition: [
                    r["metrics"]["records"][condition]["balanced_accuracy"]
                    for r in records
                ]
                for condition in ("full", "reset", "no_state")
            }
            passed = [bool(r["metrics"]["gate"]["passed"]) for r in records]
            all_passed = all_passed and all(passed)
            summary["tasks"][task][str(age)] = {
                "seeds": [int(r["seed"]) for r in records],
                "all_seed_gates_passed": bool(all(passed)),
                "gate_pass_seed_values": passed,
                "full_mean": float(np.mean(condition_values["full"])),
                "full_seed_values": condition_values["full"],
                "reset_mean": float(np.mean(condition_values["reset"])),
                "reset_seed_values": condition_values["reset"],
                "no_state_mean": float(np.mean(condition_values["no_state"])),
                "no_state_seed_values": condition_values["no_state"],
                "gate": records[0]["metrics"]["gate"],
            }
    summary["all_registered_cells_present"] = bool(all_exist)
    summary["all_gates_passed"] = bool(all_exist and all_passed)
    if not summary["all_gates_passed"]:
        summary["status"] = "completed_with_failed_or_missing_gate"
    atomic_json(output / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "all_registered_cells_present": summary["all_registered_cells_present"],
        "all_gates_passed": summary["all_gates_passed"],
    }, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(DEFAULT_PUSHT_SPEC))
    parser.add_argument("--lock", default=str(DEFAULT_PUSHT_LOCK))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task", choices=[
        "transient-visual-token-recall",
        "multi-item-visual-binding-recall",
    ])
    parser.add_argument("--tasks", nargs="*", default=[
        "transient-visual-token-recall",
        "multi-item-visual-binding-recall",
    ])
    parser.add_argument("--age", type=int, default=15,
                        choices=list(SUPPORTED_AGES))
    parser.add_argument("--ages", type=int, nargs="*", default=[15],
                        choices=list(SUPPORTED_AGES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--target-mode", default="delta_flat",
                        choices=["cue_flat", "cue_mean",
                                 "delta_flat", "delta_mean",
                                 "counterfactual_delta_flat",
                                 "prototype_delta_flat"])
    parser.add_argument("--counterfactual-cache",
                        default=str(DEFAULT_COUNTERFACTUAL_CACHE))
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--variant", default="full",
                        choices=["full", "no_host", "no_context",
                                 "shuffle_targets"])
    parser.add_argument("--host-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.aggregate and args.task is None:
        parser.error("--task is required unless --aggregate is used")
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
