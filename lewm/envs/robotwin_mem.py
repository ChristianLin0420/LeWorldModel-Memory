"""RoboTwin-MeM dataset contract and fail-closed admission utilities.

The official simulator is imported only by the optional smoke path.  The
released LeRobot 2.1 trajectories are the canonical fallback when that
simulator cannot be installed.  Policy inputs are restricted to official RGB
views, proprioception, and the task instruction.  ``keyframe_steps`` and scene
metadata remain evaluator-only and are accepted solely by explicitly named
oracle selectors and leakage audits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OFFICIAL_PAPER = "https://arxiv.org/abs/2606.20092"
OFFICIAL_PROJECT_PAGE = "https://ganlin-yang.github.io/EventVLA.github.io/"
OFFICIAL_REPOSITORY = "https://github.com/InternRobotics/EventVLA"
OFFICIAL_COMMIT = "4b5b26030abddf83bc60e1a6b067de8f521fd0ec"
OFFICIAL_DATASET = "ganlinyang/RoboTwin-MeM"
OFFICIAL_DATASET_REVISION = "f67a4ee99a20c65c86897b85d3f5309b205cc897"
OFFICIAL_CODE_LICENSE = "MIT"
OFFICIAL_DATASET_LICENSE = "Apache-2.0"
OFFICIAL_CONFIG = "demo_clean"
PROTOCOL_VERSION = "robotwin-mem-admission-v1"

CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
CAMERA_SHAPE = (480, 640, 3)
FPS = 15
STATE_DIM = 14
ACTION_DIM = 14
DEFAULT_MEMORY_BUDGET = 4
FULL_HISTORY_FRAMES = 32

MATCHED_MEMORY_CONDITIONS = (
    "no_memory",
    "recent_only",
    "random_event",
    "oracle_best_event",
    "oracle_event_set",
)
ALL_MEMORY_CONDITIONS = MATCHED_MEMORY_CONDITIONS + ("full_history",)
EVALUATOR_ONLY_FIELDS = frozenset(
    {
        "keyframe_steps",
        "scene_info",
        "target_color",
        "target_visible_block_id",
        "covered_color_order",
        "visible_color_order",
    }
)


@dataclass(frozen=True)
class RoboTwinMemTaskSpec:
    task_id: str
    instruction: str
    paper_average_steps: int
    intermediate_keyframes: int
    query_prefix: str
    query_count: int
    action_candidates: int


TASK_SPECS: dict[str, RoboTwinMemTaskSpec] = {
    "pick_the_unhidden_block": RoboTwinMemTaskSpec(
        task_id="pick_the_unhidden_block",
        instruction=(
            "Open the covers one by one to identify the hidden colors, close it "
            "after inspection, then pick up the visible block whose color is not hidden."
        ),
        paper_average_steps=699,
        intermediate_keyframes=3,
        query_prefix="Pick up",
        query_count=1,
        action_candidates=4,
    ),
    "pick_objects_in_order": RoboTwinMemTaskSpec(
        task_id="pick_objects_in_order",
        instruction=(
            "Open the covers one by one to observe the objects inside, close it "
            "after inspection, then pick up the objects in the observed order."
        ),
        paper_average_steps=1124,
        intermediate_keyframes=3,
        query_prefix="Pick up",
        query_count=3,
        action_candidates=3,
    ),
    "cover_blocks_hard": RoboTwinMemTaskSpec(
        task_id="cover_blocks_hard",
        instruction=(
            "Open the covers one by one, close it after inspection, then reopen "
            "them in the order: red, green, blue, yellow."
        ),
        paper_average_steps=1544,
        intermediate_keyframes=4,
        query_prefix="Finally open",
        query_count=4,
        action_candidates=4,
    ),
}


@dataclass(frozen=True)
class EpisodeRecord:
    task_id: str
    episode_index: int
    episode_seed: int
    length: int
    instruction: str
    keyframe_steps: tuple[int, ...]
    query_steps: tuple[int, ...]

    @property
    def recall_step(self) -> int:
        return int(self.query_steps[0])


@dataclass(frozen=True)
class GateThresholds:
    minimum_oracle_gain: float = 0.10
    minimum_closed_gap: float = 0.25
    minimum_oracle_control: float = 0.75
    confidence: float = 0.95


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    clauses: dict[str, bool]
    metrics: dict[str, float]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def raw_memory_bytes(
    budget: int = DEFAULT_MEMORY_BUDGET,
    *,
    cameras: int = len(CAMERA_KEYS),
    shape: tuple[int, int, int] = CAMERA_SHAPE,
) -> int:
    return int(budget) * int(cameras) * int(np.prod(shape))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_query_steps(
    annotations: Sequence[Sequence[Any]],
    *,
    query_prefix: str,
    expected_queries: int,
    episode_length: int,
) -> tuple[int, ...]:
    """Locate delayed-query transitions from official dense language metadata.

    The text is used only to locate evaluator decision boundaries.  It is not
    returned as a policy input; policies receive the generic task instruction.
    """

    elapsed = 0
    previous: str | None = None
    query_steps: list[int] = []
    for row in annotations:
        if len(row) != 2:
            raise ValueError(f"invalid language annotation row: {row!r}")
        text, count = str(row[0]), int(row[1])
        if text != previous and text.startswith(query_prefix):
            query_steps.append(min(elapsed, int(episode_length) - 1))
        previous = text
        elapsed += count
    if len(query_steps) != int(expected_queries):
        raise AssertionError(
            f"expected {expected_queries} delayed queries, found {len(query_steps)}"
        )
    if any(right <= left for left, right in zip(query_steps, query_steps[1:])):
        raise AssertionError("query steps must be strictly increasing")
    return tuple(query_steps)


def load_episode_records(dataset_root: Path, task_id: str) -> list[EpisodeRecord]:
    if task_id not in TASK_SPECS:
        raise ValueError(f"unsupported RoboTwin-MeM task: {task_id}")
    spec = TASK_SPECS[task_id]
    task_root = Path(dataset_root) / "lerobot_2.1" / task_id
    hdf5_root = Path(dataset_root) / "hdf5" / task_id / OFFICIAL_CONFIG
    info = json.loads((task_root / "meta/info.json").read_text())
    episodes = _read_jsonl(task_root / "meta/episodes.jsonl")
    annotations = json.loads((hdf5_root / "language_annotation.json").read_text())
    seeds = [
        int(value)
        for value in (hdf5_root / "seed.txt").read_text().split()
        if value.strip()
    ]
    if int(info["total_episodes"]) != 50 or len(episodes) != 50 or len(seeds) != 50:
        raise AssertionError("official release must contain exactly 50 episodes per task")
    if info["codebase_version"] != "v2.1":
        raise AssertionError(f"unexpected LeRobot version: {info['codebase_version']}")

    records = []
    for episode in episodes:
        index = int(episode["episode_index"])
        instruction = str(episode["tasks"][0])
        if instruction != spec.instruction:
            raise AssertionError(f"{task_id}: task instruction differs from paper release")
        keyframes = tuple(int(value) for value in episode["keyframe_steps"])
        if len(keyframes) != spec.intermediate_keyframes:
            raise AssertionError(
                f"{task_id}: expected {spec.intermediate_keyframes} keyframes"
            )
        length = int(episode["length"])
        queries = parse_query_steps(
            annotations[f"episode_{index}"],
            query_prefix=spec.query_prefix,
            expected_queries=spec.query_count,
            episode_length=length,
        )
        if keyframes[-1] >= queries[0]:
            raise AssertionError(f"{task_id} episode {index}: event is not pre-query")
        records.append(
            EpisodeRecord(
                task_id=task_id,
                episode_index=index,
                episode_seed=seeds[index],
                length=length,
                instruction=instruction,
                keyframe_steps=keyframes,
                query_steps=queries,
            )
        )
    return records


def deterministic_episode_split(
    task_id: str,
    records: Sequence[EpisodeRecord],
    *,
    validation_episodes: int = 10,
    test_episodes: int = 10,
) -> dict[str, list[int]]:
    count = len(records)
    if validation_episodes <= 0 or test_episodes <= 0:
        raise ValueError("validation and test counts must be positive")
    if validation_episodes + test_episodes >= count:
        raise ValueError("split leaves no training episodes")
    ranked = sorted(
        range(count),
        key=lambda position: stable_digest(
            f"{PROTOCOL_VERSION}|{task_id}|"
            f"{records[position].episode_index}|{records[position].episode_seed}"
        ),
    )
    test = sorted(ranked[:test_episodes])
    validation = sorted(
        ranked[test_episodes : test_episodes + validation_episodes]
    )
    train = sorted(ranked[test_episodes + validation_episodes :])
    split = {"train": train, "validation": validation, "test": test}
    assert_split_disjoint(split, records)
    return split


def assert_split_disjoint(
    split: Mapping[str, Sequence[int]], records: Sequence[EpisodeRecord]
) -> None:
    names = tuple(split)
    sets = {name: {int(value) for value in split[name]} for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"episode leakage between {left} and {right}")
    flattened = [value for values in sets.values() for value in values]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("duplicate episode index across splits")
    if any(value < 0 or value >= len(records) for value in flattened):
        raise AssertionError("split position outside episode table")
    episode_ids = {
        name: {records[position].episode_index for position in positions}
        for name, positions in sets.items()
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if episode_ids[left] & episode_ids[right]:
                raise AssertionError("episode-id leakage across splits")


def _fixed_length(indices: Sequence[int], budget: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return np.full(int(budget), -1, dtype=np.int64)
    if values.size == int(budget):
        return values
    if values.size > int(budget):
        positions = np.rint(np.linspace(0, values.size - 1, int(budget))).astype(
            np.int64
        )
        return values[positions]
    output = np.full(int(budget), -1, dtype=np.int64)
    output[: values.size] = values
    return output


def select_memory_indices(
    condition: str,
    record: EpisodeRecord,
    *,
    budget: int = DEFAULT_MEMORY_BUDGET,
    random_seed: int = 0,
    oracle_event_position: int = 0,
    full_history_frames: int = FULL_HISTORY_FRAMES,
) -> np.ndarray:
    """Select policy memory while keeping oracle metadata evaluator-only."""

    if condition not in ALL_MEMORY_CONDITIONS:
        raise ValueError(f"unknown memory condition: {condition}")
    if budget <= 0:
        raise ValueError("memory budget must be positive")
    recall = int(record.recall_step)
    if recall <= budget:
        raise ValueError("recall occurs before a complete recent window")

    if condition == "no_memory":
        return np.full(int(budget), -1, dtype=np.int64)
    if condition == "recent_only":
        return np.arange(recall - int(budget), recall, dtype=np.int64)
    if condition == "random_event":
        latest_start = recall - 2 * int(budget)
        rng = np.random.default_rng(int(random_seed))
        start = int(rng.integers(0, max(1, latest_start + 1)))
        return np.arange(start, start + int(budget), dtype=np.int64)
    if condition == "full_history":
        return np.unique(
            np.rint(
                np.linspace(0, recall - 1, min(int(full_history_frames), recall))
            ).astype(np.int64)
        )

    keyframes = np.asarray(record.keyframe_steps, dtype=np.int64)
    if condition == "oracle_event_set":
        return _fixed_length(keyframes, int(budget))
    if not 0 <= int(oracle_event_position) < len(keyframes):
        raise ValueError("oracle event position outside keyframe set")
    return np.full(
        int(budget), keyframes[int(oracle_event_position)], dtype=np.int64
    )


def frame_union_for_encoding(
    record: EpisodeRecord,
    *,
    budget: int = DEFAULT_MEMORY_BUDGET,
    random_seed: int = 0,
    full_history_frames: int = FULL_HISTORY_FRAMES,
) -> np.ndarray:
    indices = {int(record.recall_step)}
    for condition in (
        "recent_only",
        "random_event",
        "oracle_event_set",
        "full_history",
    ):
        indices.update(
            int(value)
            for value in select_memory_indices(
                condition,
                record,
                budget=budget,
                random_seed=random_seed,
                full_history_frames=full_history_frames,
            )
            if int(value) >= 0
        )
    indices.update(int(value) for value in record.keyframe_steps)
    return np.asarray(sorted(indices), dtype=np.int64)


def assert_matched_budget(
    selections: Mapping[str, Sequence[int]],
    *,
    budget: int = DEFAULT_MEMORY_BUDGET,
) -> None:
    missing = set(MATCHED_MEMORY_CONDITIONS) - set(selections)
    if missing:
        raise AssertionError(f"missing matched conditions: {sorted(missing)}")
    lengths = {
        name: len(np.asarray(selections[name]).reshape(-1))
        for name in MATCHED_MEMORY_CONDITIONS
    }
    if set(lengths.values()) != {int(budget)}:
        raise AssertionError(f"memory budget mismatch: {lengths}")
    serialized = {
        name: raw_memory_bytes(int(budget)) for name in MATCHED_MEMORY_CONDITIONS
    }
    if len(set(serialized.values())) != 1:
        raise AssertionError("serialized raw memory byte budget differs")


def recent_suffix_audit(
    record: EpisodeRecord,
    *,
    budget: int = DEFAULT_MEMORY_BUDGET,
    event_radius: int = 8,
) -> dict[str, Any]:
    recent = select_memory_indices("recent_only", record, budget=budget)
    event_indices = np.asarray(record.keyframe_steps, dtype=np.int64)
    overlap = sum(
        int(np.any(np.abs(event_indices - int(index)) <= int(event_radius)))
        for index in recent
    )
    last_event = int(event_indices[-1])
    first_recent = int(recent[0])
    return {
        "passed": overlap == 0 and last_event + int(event_radius) < first_recent,
        "event_overlap_frames": int(overlap),
        "event_radius": int(event_radius),
        "last_keyframe_index": last_event,
        "first_recent_index": first_recent,
        "gap_frames": first_recent - last_event - 1,
        "recall_step": int(record.recall_step),
    }


def policy_view(
    *,
    rgb: np.ndarray,
    proprio: np.ndarray,
    instruction: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete policy surface and reject evaluator leakage."""

    if metadata:
        forbidden = set(metadata) & EVALUATOR_ONLY_FIELDS
        if forbidden:
            raise AssertionError(
                f"evaluator-only fields leaked to policy: {sorted(forbidden)}"
            )
    rgb_array = np.asarray(rgb)
    proprio_array = np.asarray(proprio)
    if rgb_array.shape[-4:] != (
        len(CAMERA_KEYS),
        *CAMERA_SHAPE,
    ):
        raise AssertionError(f"unexpected multiview RGB shape: {rgb_array.shape}")
    if proprio_array.shape[-1] != STATE_DIM:
        raise AssertionError(f"unexpected proprioception shape: {proprio_array.shape}")
    if not isinstance(instruction, str) or not instruction:
        raise AssertionError("missing official task instruction")
    return {
        "rgb": rgb_array,
        "proprio": proprio_array,
        "language_instruction": instruction,
    }


def paired_bootstrap_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 20_000,
    seed: int = 94117,
) -> tuple[float, float, float]:
    treatment_array = np.asarray(treatment, dtype=np.float64)
    control_array = np.asarray(control, dtype=np.float64)
    if treatment_array.shape != control_array.shape:
        raise ValueError("paired arrays must have identical shapes")
    if treatment_array.ndim == 1:
        paired = treatment_array - control_array
    elif treatment_array.ndim == 2:
        paired = (treatment_array - control_array).mean(axis=0)
    else:
        raise ValueError("paired arrays must be [episodes] or [seeds, episodes]")
    if paired.size < 2:
        raise ValueError("at least two paired episodes are required")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, paired.size, size=(int(samples), paired.size))
    estimates = paired[draws].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(paired.mean()), float(low), float(high)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 20_000,
    seed: int = 4759,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array.mean(axis=0)
    array = array.reshape(-1)
    if array.size < 2:
        raise ValueError("at least two episode values are required")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, array.size, size=(int(samples), array.size))
    estimates = array[draws].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(array.mean()), float(low), float(high)


def decide_admission_gate(
    *,
    recent_success: np.ndarray,
    oracle_success: np.ndarray,
    no_memory_success: np.ndarray,
    recent_probe_accuracy: np.ndarray,
    candidate_count: int,
    thresholds: GateThresholds = GateThresholds(),
) -> GateDecision:
    recent = np.asarray(recent_success, dtype=np.float64)
    oracle = np.asarray(oracle_success, dtype=np.float64)
    no_memory = np.asarray(no_memory_success, dtype=np.float64)
    probe = np.asarray(recent_probe_accuracy, dtype=np.float64)
    for name, value in (
        ("oracle", oracle),
        ("no_memory", no_memory),
        ("recent_probe", probe),
    ):
        if value.shape != recent.shape:
            raise ValueError(f"{name} shape differs from recent")
    gain, gain_low, gain_high = paired_bootstrap_ci(
        oracle, recent, confidence=thresholds.confidence
    )
    recent_mean, _, _ = bootstrap_mean_ci(
        recent, confidence=thresholds.confidence
    )
    oracle_mean, oracle_low, oracle_high = bootstrap_mean_ci(
        oracle, confidence=thresholds.confidence
    )
    probe_mean, probe_low, probe_high = bootstrap_mean_ci(
        probe, confidence=thresholds.confidence
    )
    no_memory_mean = float(no_memory.mean())
    recoverable_gap = max(1e-12, 1.0 - no_memory_mean)
    closed_gap = gain / recoverable_gap
    probe_ceiling = min(1.0, 1.0 / int(candidate_count) + 0.10)
    clauses = {
        "oracle_gain": (
            gain_low > 0.0
            and (
                gain >= thresholds.minimum_oracle_gain
                or closed_gap >= thresholds.minimum_closed_gap
            )
        ),
        "recent_suffix_probe": (
            probe_mean <= probe_ceiling and probe_high <= probe_ceiling + 0.05
        ),
        "oracle_control": oracle_mean > thresholds.minimum_oracle_control,
    }
    reasons = tuple(name for name, passed in clauses.items() if not passed)
    metrics = {
        "recent_success": recent_mean,
        "oracle_success": oracle_mean,
        "oracle_success_ci_low": oracle_low,
        "oracle_success_ci_high": oracle_high,
        "no_memory_success": no_memory_mean,
        "oracle_minus_recent": gain,
        "oracle_minus_recent_ci_low": gain_low,
        "oracle_minus_recent_ci_high": gain_high,
        "oracle_control_gap_closed": closed_gap,
        "recent_probe_accuracy": probe_mean,
        "recent_probe_ci_low": probe_low,
        "recent_probe_ci_high": probe_high,
        "recent_probe_ceiling": probe_ceiling,
    }
    return GateDecision(
        passed=all(clauses.values()),
        clauses=clauses,
        metrics=metrics,
        reasons=reasons,
    )


def source_receipt() -> dict[str, Any]:
    return {
        "paper": OFFICIAL_PAPER,
        "project_page": OFFICIAL_PROJECT_PAGE,
        "repository": OFFICIAL_REPOSITORY,
        "repository_commit": OFFICIAL_COMMIT,
        "code_license": OFFICIAL_CODE_LICENSE,
        "dataset": OFFICIAL_DATASET,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "dataset_license": OFFICIAL_DATASET_LICENSE,
        "dataset_format": "LeRobot 2.1 plus HDF5 trajectories",
        "tasks": {name: asdict(spec) for name, spec in TASK_SPECS.items()},
        "canonical_api": {
            "cameras": list(CAMERA_KEYS),
            "camera_shape_hwc": list(CAMERA_SHAPE),
            "fps": FPS,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_type": "absolute bimanual joint position plus grippers",
            "success_metric": "official binary task check_success",
            "dataset_fallback_metric": (
                "action-candidate ranking and exact delayed-query sequence accuracy"
            ),
        },
    }


__all__ = [
    "ACTION_DIM",
    "ALL_MEMORY_CONDITIONS",
    "CAMERA_KEYS",
    "CAMERA_SHAPE",
    "DEFAULT_MEMORY_BUDGET",
    "EVALUATOR_ONLY_FIELDS",
    "EpisodeRecord",
    "FULL_HISTORY_FRAMES",
    "GateDecision",
    "GateThresholds",
    "MATCHED_MEMORY_CONDITIONS",
    "OFFICIAL_COMMIT",
    "OFFICIAL_DATASET_REVISION",
    "PROTOCOL_VERSION",
    "RoboTwinMemTaskSpec",
    "STATE_DIM",
    "TASK_SPECS",
    "assert_matched_budget",
    "assert_split_disjoint",
    "bootstrap_mean_ci",
    "decide_admission_gate",
    "deterministic_episode_split",
    "frame_union_for_encoding",
    "load_episode_records",
    "paired_bootstrap_ci",
    "parse_query_steps",
    "policy_view",
    "raw_memory_bytes",
    "recent_suffix_audit",
    "select_memory_indices",
    "source_receipt",
    "stable_digest",
]
