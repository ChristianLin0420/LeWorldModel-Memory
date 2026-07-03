"""Shared containers, task interface, and disk format for the V19 P1a suite.

P1a builds pixel+action memory tasks whose exogenous factor xi is certified at
construction level (V19_PROPOSAL.md section 4.4).  Everything downstream (the
certificates, the wandb reports, later P0/P1b phases) consumes exactly one
container, ``EpisodeBatch``, so its invariants are validated eagerly here: a
shape or dtype drift would otherwise silently invalidate a certificate.
"""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMG_SIZE = 64
EPISODE_LENGTH = 64
ACTION_DIM = 2
XI_KINDS = ("cat", "cont")
STREAMS = ("iid", "script")

# Arrays hashed into the bank content digest, in a fixed order so the digest is
# stable across writers (same discipline as scripts/hacssm_v11_data.py).
_BANK_FIELDS = ("frames", "actions", "xi", "endo_state", "exo_state")


@dataclass
class EpisodeBatch:
    """One bank of generated episodes for a single (task, stream, seed) cell.

    Invariants are enforced at construction because every certificate clause
    indexes these arrays positionally; failing loudly here is what keeps the
    leakage accounting exact.

    Attributes:
        frames: uint8 (E, L, 64, 64, 3) rendered pixels, overlays composited.
        actions: float32 (E, L-1, A) executed controls (bounded, tanh-squashed).
        xi: int64 (E,) class ids for ``xi_kind == 'cat'``; float32 (E, 2)
            normalized positions for ``xi_kind == 'cont'``.
        xi_kind: 'cat' or 'cont'.
        n_classes: number of xi classes (0 for continuous xi).
        endo_state: float32 (E, L, S) simulator ground truth, concat(qpos, qvel).
        exo_state: float32 (E, L, X) ground-truth overlay state trace.
        events: per-episode integer event annotations (leading dim E), e.g.
            cue_on/cue_off, and gap_on/gap_off for t4.
        stream: 'iid' or 'script' action stream.
        task: task registry name.
        seed: generation seed of this bank.
    """

    frames: np.ndarray
    actions: np.ndarray
    xi: np.ndarray
    xi_kind: str
    n_classes: int
    endo_state: np.ndarray
    exo_state: np.ndarray
    events: dict[str, np.ndarray] = field(default_factory=dict)
    stream: str = "iid"
    task: str = ""
    seed: int = 0

    def __post_init__(self) -> None:
        if self.xi_kind not in XI_KINDS:
            raise ValueError(f"xi_kind must be one of {XI_KINDS}, got {self.xi_kind!r}")
        if self.stream not in STREAMS:
            raise ValueError(f"stream must be one of {STREAMS}, got {self.stream!r}")
        e, length = self.frames.shape[:2]
        if self.frames.dtype != np.uint8 or self.frames.shape[2:] != (IMG_SIZE, IMG_SIZE, 3):
            raise ValueError(f"frames must be uint8 (E,L,{IMG_SIZE},{IMG_SIZE},3), "
                             f"got {self.frames.dtype} {self.frames.shape}")
        if self.actions.dtype != np.float32 or self.actions.shape[:2] != (e, length - 1):
            raise ValueError(f"actions must be float32 (E,L-1,A), got "
                             f"{self.actions.dtype} {self.actions.shape}")
        if self.xi_kind == "cat":
            if self.xi.dtype != np.int64 or self.xi.shape != (e,):
                raise ValueError(f"categorical xi must be int64 (E,), got "
                                 f"{self.xi.dtype} {self.xi.shape}")
            if self.n_classes < 2 or self.xi.min() < 0 or self.xi.max() >= self.n_classes:
                raise ValueError("categorical xi out of range for n_classes")
        else:
            if self.xi.dtype != np.float32 or self.xi.shape != (e, 2):
                raise ValueError(f"continuous xi must be float32 (E,2), got "
                                 f"{self.xi.dtype} {self.xi.shape}")
            if self.n_classes != 0:
                raise ValueError("continuous xi requires n_classes == 0")
        for name, array in (("endo_state", self.endo_state), ("exo_state", self.exo_state)):
            if array.dtype != np.float32 or array.shape[:2] != (e, length):
                raise ValueError(f"{name} must be float32 (E,L,*), got "
                                 f"{array.dtype} {array.shape}")
        for name, array in self.events.items():
            if not np.issubdtype(array.dtype, np.integer) or array.shape[0] != e:
                raise ValueError(f"event {name!r} must be an integer array with "
                                 f"leading dim E, got {array.dtype} {array.shape}")

    @property
    def num_episodes(self) -> int:
        return int(self.frames.shape[0])

    @property
    def length(self) -> int:
        return int(self.frames.shape[1])


class V19Task(abc.ABC):
    """Interface each P1a task implements.

    ``paired_branches`` is the construction-level identical-rendering probe:
    two banks sharing every nuisance random draw (base scene, actions, cue
    timing, swap/OU patterns) and differing only in xi.  Any pixel difference
    outside the cue window is therefore a leakage bug by definition.
    """

    name: str
    xi_kind: str
    n_classes: int
    length: int = EPISODE_LENGTH

    def decision_time(self, length: int) -> int:
        """Frame index at which xi must be reported (last frame by default)."""
        return length - 1

    @abc.abstractmethod
    def generate(self, stream: str, num_episodes: int, seed: int) -> EpisodeBatch:
        """Generate one bank; deterministic in (stream, num_episodes, seed)."""

    @abc.abstractmethod
    def paired_branches(self, num_episodes: int, seed: int
                        ) -> tuple[EpisodeBatch, EpisodeBatch]:
        """Two banks with identical nuisance randomness and different xi."""

    @abc.abstractmethod
    def describe(self) -> dict:
        """All frozen task parameters (wandb run config / registration)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def save_bank(batch: EpisodeBatch, path: str | Path) -> dict:
    """Write a bank as .npz plus a JSON sidecar with metadata and sha256.

    The sidecar carries the scalar metadata (task/stream/seed/xi_kind) and the
    file digest so a re-run can verify it is reading the bytes it wrote —
    the same cache discipline as the frozen V11-V18 collectors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {name: getattr(batch, name) for name in _BANK_FIELDS}
    for name, value in batch.events.items():
        arrays[f"event_{name}"] = value
    np.savez_compressed(path, **arrays)
    metadata = {
        "format": "v19_p1a_bank_v1",
        "task": batch.task,
        "stream": batch.stream,
        "seed": batch.seed,
        "xi_kind": batch.xi_kind,
        "n_classes": batch.n_classes,
        "num_episodes": batch.num_episodes,
        "length": batch.length,
        "event_keys": sorted(batch.events),
        "npz_sha256": sha256_file(path),
    }
    _sidecar_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def load_bank(path: str | Path, verify: bool = True) -> EpisodeBatch:
    """Load a bank written by :func:`save_bank`, verifying the sha256 sidecar."""
    path = Path(path)
    metadata = json.loads(_sidecar_path(path).read_text())
    if metadata.get("format") != "v19_p1a_bank_v1":
        raise ValueError(f"unrecognized bank format in {_sidecar_path(path)}")
    if verify:
        actual = sha256_file(path)
        if actual != metadata["npz_sha256"]:
            raise ValueError(f"bank digest mismatch for {path}: {actual} != "
                             f"{metadata['npz_sha256']}")
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    events = {name.removeprefix("event_"): value
              for name, value in arrays.items() if name.startswith("event_")}
    return EpisodeBatch(
        frames=arrays["frames"], actions=arrays["actions"], xi=arrays["xi"],
        xi_kind=str(metadata["xi_kind"]), n_classes=int(metadata["n_classes"]),
        endo_state=arrays["endo_state"], exo_state=arrays["exo_state"],
        events=events, stream=str(metadata["stream"]), task=str(metadata["task"]),
        seed=int(metadata["seed"]))
