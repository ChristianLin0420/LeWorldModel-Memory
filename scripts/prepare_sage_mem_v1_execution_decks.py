#!/usr/bin/env python3
"""Seal label-free physical execution cubes for SAGE-Mem v1.

The producer runs before formal semantic labels are available.  It therefore
executes every selected class against every possible physical target and
stores a ``row x selected-class x true-target-class`` success cube.  It never
opens a label vault and never collapses the target axis.  The post-grid
finalizer performs that indexing only after its durable label-reveal receipt.

Three fresh-bank cohorts have an exact public replay state and pinned
controller: LeWM Reacher, LeWM PushT, and DINO-WM PointMaze.  The two DINO-WM
PushT formal banks deliberately expose no native simulator state; they receive
explicit immutable unavailable receipts instead of being relinked to legacy
execution decks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sage_mem_v1_formal_finalizer import (  # noqa: E402
    COHORTS,
    EXECUTION_DECK_REGISTRY_SCHEMA,
    EXECUTION_REPLAY_RECEIPT_SCHEMA,
    EXECUTION_UNAVAILABLE_RECEIPT_SCHEMA,
    FORMAL_TEST_ROWS,
    VARIANTS_PER_NATIVE_CLUSTER,
)


PRODUCER_SCHEMA = "sage_mem_v1_execution_deck_producer_v1"
PREVIEW_SCHEMA = "sage_mem_v1_execution_deck_preview_v1"
RANDOM_POLICY_SEED = 2_026_070_829
SUPPORTED = (
    "lewm_reacher_color",
    "lewm_pusht_color",
    "dinowm_pointmaze_goal",
)
UNAVAILABLE = {
    "dinowm_pusht_token": "native-physics-state-not-exposed-by-formal-bank",
    "dinowm_pusht_binding": "native-physics-state-not-exposed-by-formal-bank",
}


class ExecutionDeckProducerError(RuntimeError):
    """A sealed boundary, exact replay, or publication invariant failed."""


class ReplayUnavailable(ExecutionDeckProducerError):
    """The cohort cannot be replayed exactly from its public formal state."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionDeckProducerError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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


def _handle(path: Path, root: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve())
    return {"path": str(relative), "sha256": _sha256_file(path),
            "size": path.stat().st_size}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionDeckProducerError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not a mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ExecutionDeckProducerError(f"cannot read YAML: {path}") from error
    _require(isinstance(value, dict), f"YAML root is not a mapping: {path}")
    return value


def _locked_yaml(path: Path, sha_path: Path) -> dict[str, Any]:
    _require(path.is_file() and sha_path.is_file(),
             f"locked controller specification is absent: {path}")
    fields = sha_path.read_text(encoding="utf-8").strip().split()
    _require(len(fields) >= 1 and fields[0] == _sha256_file(path),
             f"controller specification hash differs: {path}")
    return _load_yaml(path)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_archive_sha256(repo: Path) -> str:
    process = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    assert process.stdout is not None
    for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        digest.update(block)
    stderr = process.stderr.read().decode("utf-8", errors="replace") \
        if process.stderr is not None else ""
    _require(process.wait() == 0, f"git archive failed: {repo}: {stderr}")
    return digest.hexdigest()


def _implementation_sha256(paths: tuple[Path, ...]) -> str:
    values = []
    for path in paths:
        _require(path.is_file() and not path.is_symlink(),
                 f"controller implementation source is absent: {path}")
        values.append({"path": str(path.resolve()),
                       "sha256": _sha256_file(path)})
    return _sha256_json(values)


def _deterministic_random_class(
        cohort: str, cluster: np.ndarray, classes: int) -> np.ndarray:
    values: dict[int, int] = {}
    for raw in np.unique(np.asarray(cluster, dtype=np.int64)):
        digest = hashlib.sha256(
            f"sage-mem-v1:{RANDOM_POLICY_SEED}:{cohort}:{int(raw)}".encode(
                "ascii")).digest()
        values[int(raw)] = int.from_bytes(digest[:8], "little") % classes
    return np.asarray([values[int(raw)] for raw in cluster], dtype=np.int64)


@dataclass(frozen=True)
class ReplayProduct:
    cohort: str
    bank_manifest_sha256: str
    episode_id: np.ndarray
    native_cluster_id: np.ndarray
    success_cube: np.ndarray
    random_class: np.ndarray
    controller: Mapping[str, Any]
    executions: int
    replayed_executions: int
    replay_fidelity: float
    execution_endpoint: str


ReplayBuilder = Callable[[Path, Mapping[str, Any], Callable[[str], None]],
                         ReplayProduct]


def _validate_controller(value: Mapping[str, Any]) -> None:
    expected = {
        "controller_identity_sha256", "implementation_sha256",
        "physics_sha256", "pinned", "arm_identity_input", "input",
    }
    _require(set(value) == expected
             and all(_is_sha256(value[key]) for key in (
                 "controller_identity_sha256", "implementation_sha256",
                 "physics_sha256"))
             and value["pinned"] is True
             and value["arm_identity_input"] is False
             and value["input"] == "predicted_class_only",
             "controller identity is not pinned and arm-blind")


def _validate_product(product: ReplayProduct, *, classes: int,
                      expected_rows: int, expected_variants: int) -> None:
    episode = np.asarray(product.episode_id)
    cluster = np.asarray(product.native_cluster_id)
    cube = np.asarray(product.success_cube)
    random_class = np.asarray(product.random_class)
    _require(episode.shape == cluster.shape == random_class.shape ==
             (expected_rows,)
             and cube.shape == (expected_rows, classes, classes),
             f"{product.cohort} execution product shape differs")
    _require(all(np.issubdtype(value.dtype, np.integer)
                 or np.issubdtype(value.dtype, np.bool_)
                 for value in (episode, cluster, cube, random_class))
             and len(np.unique(episode)) == expected_rows
             and np.isin(cube, (0, 1)).all()
             and np.all((random_class >= 0) & (random_class < classes)),
             f"{product.cohort} execution product values differ")
    unique, counts = np.unique(cluster, return_counts=True)
    _require(len(unique) * expected_variants == expected_rows
             and np.all(counts == expected_variants),
             f"{product.cohort} native cluster multiplicity differs")
    for native in unique:
        rows = np.flatnonzero(cluster == native)
        _require(all(np.array_equal(cube[rows[0]], cube[row])
                     for row in rows[1:])
                 and np.all(random_class[rows] == random_class[rows[0]]),
                 f"{product.cohort} replay varies inside native cluster")
    _require(np.array_equal(
        random_class,
        _deterministic_random_class(product.cohort, cluster, classes)),
        f"{product.cohort} random policy is not deterministic")
    _validate_controller(product.controller)
    _require(product.executions > 0
             and product.replayed_executions == product.executions
             and product.replay_fidelity == 1.0
             and isinstance(product.execution_endpoint, str)
             and product.execution_endpoint,
             f"{product.cohort} deterministic replay gate failed")


def _bank_manifest(preparation_root: Path, cohort: str) -> Path:
    return preparation_root / "banks" / cohort / "manifest.json"


def _formal_rows(spec: Mapping[str, Any], cohort: str) -> int:
    raw = int(spec["cohorts"][cohort]["split_episodes"]["formal_test"])
    return raw * (4 if cohort == "dinowm_pointmaze_goal" else 1)


def _classes(spec: Mapping[str, Any], cohort: str) -> int:
    return int(spec["cohorts"][cohort]["classes"])


def _validate_spec(spec: Mapping[str, Any]) -> None:
    _require(spec.get("study") == "sage-mem-v1"
             and isinstance(spec.get("cohorts"), dict)
             and set(spec["cohorts"]) == set(COHORTS),
             "SAGE-Mem execution specification identity changed")
    for cohort in COHORTS:
        _require(_classes(spec, cohort) in (4, 6)
                 and _formal_rows(spec, cohort) == FORMAL_TEST_ROWS[cohort],
                 f"formal execution shape changed: {cohort}")
    gate = spec.get("confirmatory_gates", {}).get("execution", {})
    _require(gate.get("oracle_gate") == 0.90
             and gate.get("consumer_arm_blind") is True
             and gate.get("controller_and_physics_pinned") is True,
             "execution eligibility gate changed")


def _public_lewm_bank(bank_root: Path, cohort: str,
                      spec: Mapping[str, Any]) -> tuple[Any, str]:
    from scripts.sage_mem_v1_lewm_formal import load_lewm_trajectory_banks

    manifest = bank_root / "manifest.json"
    counts = {split: int(spec["cohorts"][cohort]["split_episodes"][split])
              for split in ("formal_train", "consumer_train", "formal_test")}
    _, banks = load_lewm_trajectory_banks(
        manifest, expected_cohort=cohort, expected_counts=counts)
    return banks["formal_test"], _sha256_file(manifest)


def _reacher_identity() -> tuple[dict[str, Any], dict[str, Any]]:
    from dm_control.suite import reacher

    protocol_path = ROOT / "configs/paper_a_delayed_goal_use.yaml"
    cfg = _locked_yaml(
        protocol_path, ROOT / "configs/paper_a_delayed_goal_use.sha256")
    choice = dict(cfg["executed_choice"])
    model, assets = reacher.get_model_and_assets()
    physics = {
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "assets": {name: hashlib.sha256(value).hexdigest()
                   for name, value in sorted(assets.items())},
        "dm_control_version": _package_version("dm-control"),
        "mujoco_version": _package_version("mujoco"),
    }
    implementation = _implementation_sha256((
        Path(__file__), ROOT / "scripts/paper_a_delayed_goal_use.py"))
    physics_sha = _sha256_json(physics)
    identity = _sha256_json({
        "cohort": "lewm_reacher_color",
        "protocol_sha256": _sha256_file(protocol_path),
        "choice": choice, "implementation_sha256": implementation,
        "physics_sha256": physics_sha,
    })
    return {
        "controller_identity_sha256": identity,
        "implementation_sha256": implementation,
        "physics_sha256": physics_sha,
        "pinned": True,
        "arm_identity_input": False,
        "input": "predicted_class_only",
    }, choice


def _run_reacher_choice(environment: Any, state: np.ndarray,
                        goal: np.ndarray, choice: Mapping[str, Any]) \
        -> tuple[np.ndarray, np.ndarray, int]:
    from scripts.paper_a_delayed_goal_use import pd_action

    environment.reset()
    with environment.physics.reset_context():
        environment.physics.set_state(np.asarray(state, dtype=np.float64))
    reset = np.asarray(environment.physics.get_state(), dtype=np.float64).copy()
    spec = environment.action_spec()
    steps = 0
    for _ in range(int(choice["executed_horizon"])):
        position = np.asarray(environment.physics.data.qpos, dtype=np.float64)
        velocity = np.asarray(environment.physics.data.qvel, dtype=np.float64)
        timestep = environment.step(pd_action(
            position, velocity, goal,
            float(choice["proportional_gain"]),
            float(choice["derivative_gain"]),
            np.asarray(spec.minimum), np.asarray(spec.maximum)))
        steps += 1
        if timestep.last():
            break
    return reset, np.asarray(
        environment.physics.data.qpos, dtype=np.float64).copy(), steps


def _build_reacher(bank_root: Path, spec: Mapping[str, Any],
                   progress: Callable[[str], None]) -> ReplayProduct:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite
    from scripts.paper_a_delayed_goal_use import wrap_angle

    cohort = "lewm_reacher_color"
    bank, bank_sha = _public_lewm_bank(bank_root, cohort, spec)
    rows = np.arange(bank.count, dtype=np.int64)
    states = np.asarray(bank.native_state(rows)[:, 19], dtype=np.float64)
    episode = np.asarray(bank.episode_ids, dtype=np.int64)
    cluster = episode.copy()
    controller, choice = _reacher_identity()
    goals = np.asarray(choice["joint_goals"], dtype=np.float64)
    classes = _classes(spec, cohort)
    if states.shape != (bank.count, 4) or goals.shape != (classes, 2):
        raise ReplayUnavailable("fresh Reacher native state/goal shape differs")
    environment = suite.load("reacher", "easy", task_kwargs={"random": 0})
    cube = np.empty((bank.count, classes, classes), dtype=np.uint8)
    replays = 0
    for row, state in enumerate(states):
        for selected, goal in enumerate(goals):
            first = _run_reacher_choice(environment, state, goal, choice)
            second = _run_reacher_choice(environment, state, goal, choice)
            if not (np.array_equal(first[0], second[0])
                    and np.array_equal(first[1], second[1])
                    and first[2] == second[2]):
                raise ReplayUnavailable(
                    f"Reacher replay is nondeterministic at {row}/{selected}")
            difference = wrap_angle(first[1][None, :] - goals)
            distance = np.sqrt(np.mean(np.square(difference), axis=1))
            cube[row, selected] = distance <= float(
                choice["success_tolerance_radians"])
            replays += 1
        if (row + 1) % 25 == 0:
            progress(f"[execution/reacher] {row + 1}/{bank.count}")
    return ReplayProduct(
        cohort, bank_sha, episode, cluster, cube,
        _deterministic_random_class(cohort, cluster, classes), controller,
        replays, replays, 1.0, "LeWM fixed endpoint native state[19]")


def _pusht_identity() -> tuple[dict[str, Any], dict[str, Any], Path]:
    from lewm.official_tasks.pusht_downstream import (
        STABLE_WORLDMODEL_COMMIT,
    )

    protocol_path = ROOT / "configs/paper_a_pusht_downstream_use_v1.yaml"
    cfg = _locked_yaml(
        protocol_path,
        ROOT / "configs/paper_a_pusht_downstream_use_v1.sha256")
    vendor = ROOT / cfg["upstream_simulator"]["checkout"]
    _require(vendor.is_dir(), "pinned PushT vendor checkout is absent")
    revision = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    _require(revision == STABLE_WORLDMODEL_COMMIT,
             "pinned PushT vendor revision changed")
    _require(not subprocess.run(
        ["git", "-C", str(vendor), "status", "--porcelain"], check=True,
        capture_output=True, text=True).stdout.strip(),
        "pinned PushT vendor checkout is dirty")
    archive_sha = _git_archive_sha256(vendor)
    implementation = _implementation_sha256((
        Path(__file__), ROOT / "lewm/official_tasks/pusht_downstream.py"))
    physics_sha = _sha256_json({
        "vendor_revision": revision,
        "vendor_archive_sha256": archive_sha,
        "pymunk_version": _package_version("pymunk"),
        "gymnasium_version": _package_version("gymnasium"),
    })
    parameters = {
        "physical_goal_set": cfg["physical_goal_set"],
        "controller": cfg["controller"],
    }
    identity = _sha256_json({
        "cohort": "lewm_pusht_color",
        "protocol_sha256": _sha256_file(protocol_path),
        "parameters": parameters,
        "implementation_sha256": implementation,
        "physics_sha256": physics_sha,
    })
    return {
        "controller_identity_sha256": identity,
        "implementation_sha256": implementation,
        "physics_sha256": physics_sha,
        "pinned": True,
        "arm_identity_input": False,
        "input": "predicted_class_only",
    }, cfg, vendor


def _build_pusht(bank_root: Path, spec: Mapping[str, Any],
                 progress: Callable[[str], None]) -> ReplayProduct:
    from lewm.official_tasks import pusht_downstream as pd

    cohort = "lewm_pusht_color"
    bank, bank_sha = _public_lewm_bank(bank_root, cohort, spec)
    rows = np.arange(bank.count, dtype=np.int64)
    states = np.asarray(bank.native_state(rows)[:, 19], dtype=np.float64)
    episode = np.asarray(bank.episode_ids, dtype=np.int64)
    cluster = episode.copy()
    controller_record, cfg, vendor = _pusht_identity()
    classes = _classes(spec, cohort)
    if states.shape != (bank.count, 7):
        raise ReplayUnavailable("fresh PushT native state shape differs")
    PushT, _ = pd.load_pinned_pusht(vendor)
    environment = pd.make_native_env(PushT)
    parameters = cfg["controller"]
    controller = pd.NativePushTController(
        environment,
        orbit_radius=float(parameters["orbit_radius_pixels"]),
        orbit_points=int(parameters["orbit_points"]),
        waypoint_steps=int(parameters["steps_per_waypoint"]),
        push_steps=int(parameters["push_steps"]),
        push_distance=float(parameters["push_distance_pixels"]),
    )
    directions = pd.goal_directions(classes)
    goals = np.empty((bank.count, classes, 3), dtype=np.float64)
    finals = np.empty_like(goals)
    replays = 0
    for row, state in enumerate(states):
        reference_state = state.copy()
        reference_state[5:] = 0.0
        for selected, direction in enumerate(directions):
            reference = controller.execute(reference_state, direction)
            reference_again = controller.execute(reference_state, direction)
            execution = controller.execute(state, direction)
            execution_again = controller.execute(state, direction)
            if not (np.array_equal(reference["reset_state"],
                                   reference_again["reset_state"])
                    and np.array_equal(reference["final_state"],
                                       reference_again["final_state"])
                    and np.array_equal(execution["reset_state"],
                                       execution_again["reset_state"])
                    and np.array_equal(execution["final_state"],
                                       execution_again["final_state"])):
                raise ReplayUnavailable(
                    f"PushT replay is nondeterministic at {row}/{selected}")
            goals[row, selected] = reference["final_block_pose"]
            finals[row, selected] = execution["final_block_pose"]
            replays += 2
        if (row + 1) % 20 == 0:
            progress(f"[execution/pusht] {row + 1}/{bank.count}")
    goal_cfg = cfg["physical_goal_set"]
    cube = pd.pose_success(
        finals[:, :, None, :], goals[:, None, :, :],
        float(goal_cfg["position_tolerance_pixels"]),
        float(goal_cfg["angle_tolerance_radians"])).astype(np.uint8)
    return ReplayProduct(
        cohort, bank_sha, episode, cluster, cube,
        _deterministic_random_class(cohort, cluster, classes),
        controller_record, replays, replays, 1.0,
        "LeWM fixed endpoint native state[19]")


def _pointmaze_identity() -> tuple[dict[str, Any], dict[str, Any], Path]:
    from lewm.official_tasks.dinowm_pointmaze import CurrentMujocoPointMaze
    from scripts.run_dinowm_pointmaze_wave3 import load_config, resolve

    protocol = ROOT / "configs/dinowm_pointmaze_wave3.yaml"
    cfg, lock = load_config(protocol, locked=True)
    _require(lock is not None, "PointMaze protocol lock is absent")
    vendor = resolve(cfg["source"]["dino_wm"]["repo_path"])
    simulator = CurrentMujocoPointMaze(vendor)
    implementation = _implementation_sha256((
        Path(__file__), ROOT / "lewm/official_tasks/dinowm_pointmaze.py"))
    physics_sha = _sha256_json({
        "released_xml_sha256": simulator.xml_sha256,
        "mujoco_version": simulator.mujoco.__version__,
        "frame_skip": cfg["external_use"]["frame_skip"],
        "physics_timestep": cfg["external_use"]["physics_timestep"],
    })
    identity = _sha256_json({
        "cohort": "dinowm_pointmaze_goal",
        "protocol_sha256": lock["protocol_sha256"],
        "external_use": cfg["external_use"],
        "implementation_sha256": implementation,
        "physics_sha256": physics_sha,
    })
    return {
        "controller_identity_sha256": identity,
        "implementation_sha256": implementation,
        "physics_sha256": physics_sha,
        "pinned": True,
        "arm_identity_input": False,
        "input": "predicted_class_only",
    }, cfg, vendor


def _build_pointmaze(bank_root: Path, spec: Mapping[str, Any],
                     progress: Callable[[str], None]) -> ReplayProduct:
    from lewm.official_tasks.dinowm_pointmaze import (
        GOAL_WAYPOINTS, CurrentMujocoPointMaze, execute_released_waypoint,
    )
    from scripts.sage_mem_v1_dino_formal import open_label_free_bank

    cohort = "dinowm_pointmaze_goal"
    bank = open_label_free_bank(bank_root)
    _require(bank.cohort == cohort, "PointMaze formal bank cohort changed")
    bank_sha = _sha256_file(bank_root / "manifest.json")
    rows = bank.indices("formal_test")
    identity = bank.identity("formal_test")
    episode = np.asarray(identity["episode_id"], dtype=np.int64)
    cluster = np.asarray(identity["native_cluster_id"], dtype=np.int64)
    states = np.asarray(bank.states(rows), dtype=np.float64)
    controller, cfg, vendor = _pointmaze_identity()
    classes = _classes(spec, cohort)
    endpoint = 18
    simulator = CurrentMujocoPointMaze(vendor)
    use = cfg["external_use"]
    unique = np.unique(cluster)
    base_cube: dict[int, np.ndarray] = {}
    executions = 0
    for base_number, native in enumerate(unique):
        selected_rows = np.flatnonzero(cluster == native)
        base_states = states[selected_rows, endpoint]
        if not all(np.array_equal(base_states[0], value)
                   for value in base_states[1:]):
            raise ReplayUnavailable(
                f"PointMaze public state varies in cluster {int(native)}")
        state = base_states[0]
        cube = np.empty((classes, classes), dtype=np.uint8)
        for selected in range(classes):
            seed = 9_170_000 + base_number * classes + selected
            kwargs = {
                "initial_state": state,
                "target": GOAL_WAYPOINTS[selected],
                "horizon": int(use["execution_horizon"]),
                "controller_seed": seed,
                "success_radius": float(use["success_radius"]),
            }
            first = execute_released_waypoint(simulator, vendor, **kwargs)
            second = execute_released_waypoint(simulator, vendor, **kwargs)
            if not (np.array_equal(first["reset_state"], second["reset_state"])
                    and np.array_equal(first["final_state"],
                                       second["final_state"])
                    and first["steps"] == second["steps"]):
                raise ReplayUnavailable(
                    f"PointMaze replay is nondeterministic at "
                    f"{base_number}/{selected}")
            distance = np.linalg.norm(
                GOAL_WAYPOINTS - np.asarray(first["final_state"][:2]), axis=1)
            cube[selected] = distance < float(use["success_radius"])
            executions += 1
        base_cube[int(native)] = cube
        if (base_number + 1) % 10 == 0:
            progress(f"[execution/pointmaze] {base_number + 1}/{len(unique)}")
    expanded = np.stack([base_cube[int(native)] for native in cluster])
    return ReplayProduct(
        cohort, bank_sha, episode, cluster, expanded,
        _deterministic_random_class(cohort, cluster, classes), controller,
        executions, executions, 1.0,
        "DINO-WM PointMaze fixed age-15 endpoint native state[18]")


DEFAULT_BUILDERS: Mapping[str, ReplayBuilder] = {
    "lewm_reacher_color": _build_reacher,
    "lewm_pusht_color": _build_pusht,
    "dinowm_pointmaze_goal": _build_pointmaze,
}


def preview_execution_decks(spec_path: Path, preparation_root: Path,
                            output_root: Path) -> dict[str, Any]:
    spec = _load_yaml(spec_path)
    _validate_spec(spec)
    cohorts: dict[str, Any] = {}
    for cohort in COHORTS:
        manifest = _bank_manifest(preparation_root, cohort)
        if not manifest.is_file() or manifest.is_symlink():
            status, reason, digest = "blocked", "formal-bank-not-prepared", None
        elif cohort in UNAVAILABLE:
            status, reason, digest = (
                "unavailable", UNAVAILABLE[cohort], _sha256_file(manifest))
        else:
            status, reason, digest = (
                "replayable", None, _sha256_file(manifest))
        cohorts[cohort] = {
            "status": status, "reason_code": reason,
            "bank_manifest_sha256": digest,
            "formal_labels_required": False,
        }
    return {
        "schema": PREVIEW_SCHEMA,
        "study": "sage-mem-v1",
        "mode": "preview-no-write",
        "spec_sha256": _sha256_file(spec_path),
        "preparation_root": str(preparation_root.resolve()),
        "output_root": str(output_root.resolve()),
        "formal_labels_read": False,
        "development_outcomes_read": False,
        "cohorts": cohorts,
    }


def _write_unavailable(staging: Path, cohort: str, bank_sha: str,
                       reason: str) -> dict[str, Any]:
    receipt = staging / "unavailable" / f"{cohort}.json"
    _atomic_json(receipt, {
        "schema": EXECUTION_UNAVAILABLE_RECEIPT_SCHEMA,
        "study": "sage-mem-v1",
        "status": "unavailable",
        "cohort": cohort,
        "reason_code": reason,
        "bank_manifest_sha256": bank_sha,
        "formal_labels_read": False,
        "development_outcomes_read": False,
    })
    return {
        "status": "unavailable",
        "bank_manifest_sha256": bank_sha,
        "reason_code": reason,
        "receipt": _handle(receipt, staging),
    }


def _write_product(staging: Path, product: ReplayProduct,
                   classes: int) -> dict[str, Any]:
    directory = staging / "cohorts" / product.cohort
    artifact = directory / "execution_cube.npz"
    _atomic_npz(artifact, {
        "formal_test_episode_id": np.asarray(
            product.episode_id, dtype=np.int64),
        "formal_test_native_cluster_id": np.asarray(
            product.native_cluster_id, dtype=np.int64),
        "selected_class_by_true_target_success": np.asarray(
            product.success_cube, dtype=np.uint8),
        "deterministic_random_class": np.asarray(
            product.random_class, dtype=np.int64),
    })
    receipt = directory / "replay_receipt.json"
    _atomic_json(receipt, {
        "schema": EXECUTION_REPLAY_RECEIPT_SCHEMA,
        "study": "sage-mem-v1",
        "status": "sealed-label-free",
        "cohort": product.cohort,
        "bank_manifest_sha256": product.bank_manifest_sha256,
        "formal_labels_read": False,
        "development_outcomes_read": False,
        "controller_identity_sha256": product.controller[
            "controller_identity_sha256"],
        "rows": int(len(product.episode_id)),
        "classes": classes,
        "native_clusters": int(len(np.unique(product.native_cluster_id))),
        "executions": int(product.executions),
        "replayed_executions": int(product.replayed_executions),
        "deterministic_replay_fidelity": float(product.replay_fidelity),
        "execution_endpoint": product.execution_endpoint,
    })
    return {
        "bank_manifest_sha256": product.bank_manifest_sha256,
        "classes": classes,
        "controller": dict(product.controller),
        "eligibility_gate": {
            "metric": "mean_oracle_success",
            "operator": ">=",
            "threshold": 0.90,
            "preregistered": True,
        },
        "artifact": _handle(artifact, staging),
        "replay_receipt": _handle(receipt, staging),
    }


def _safe_registry_path(root: Path, handle: Mapping[str, Any]) -> Path:
    _require(set(handle) == {"path", "sha256", "size"},
             "registry artifact handle schema differs")
    relative = Path(str(handle["path"]))
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "registry artifact path is unsafe")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ExecutionDeckProducerError(
            "registry artifact leaves publication root") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        _require(not cursor.is_symlink(),
                 "registry artifact path contains a symlink")
    _require(path.is_file() and not path.is_symlink()
             and path.stat().st_size == handle["size"]
             and _sha256_file(path) == handle["sha256"],
             f"registry artifact identity differs: {path}")
    return path


def validate_published_registry(
        output_root: Path, spec_path: Path, preparation_root: Path, *,
        require_immutable: bool = True,
        verify_runtime_identities: bool = False) -> dict[str, Any]:
    """Revalidate an existing publication without opening formal labels."""

    spec = _load_yaml(spec_path)
    _validate_spec(spec)
    _require(output_root.is_dir() and not output_root.is_symlink(),
             "published execution root is absent or unsafe")
    if require_immutable:
        for path in (output_root, *output_root.rglob("*")):
            _require(not path.is_symlink()
                     and (path.stat().st_mode & 0o222) == 0,
                     f"published execution tree is mutable: {path}")
    registry_path = output_root / "registry.json"
    value = _read_json(registry_path)
    expected_top = {
        "schema", "study", "status",
        "available_only_after_complete_phase_a_grid",
        "development_outcomes_read", "cohorts", "unavailable_cohorts",
    }
    _require(set(value) == expected_top
             and value["schema"] == EXECUTION_DECK_REGISTRY_SCHEMA
             and value["study"] == "sage-mem-v1"
             and value["status"] == "sealed"
             and value["available_only_after_complete_phase_a_grid"] is True
             and value["development_outcomes_read"] is False
             and isinstance(value["cohorts"], dict)
             and isinstance(value["unavailable_cohorts"], dict),
             "published execution registry identity differs")
    supplied = set(value["cohorts"])
    unavailable = set(value["unavailable_cohorts"])
    _require(not supplied.intersection(unavailable)
             and supplied.union(unavailable) == set(COHORTS),
             "published execution registry cohort partition differs")
    for cohort in COHORTS:
        manifest = _bank_manifest(preparation_root, cohort)
        _require(manifest.is_file() and not manifest.is_symlink(),
                 f"formal bank manifest is absent: {cohort}")
        bank_sha = _sha256_file(manifest)
        if cohort in supplied:
            record = value["cohorts"][cohort]
            _require(isinstance(record, dict)
                     and set(record) == {
                         "bank_manifest_sha256", "classes", "controller",
                         "eligibility_gate", "artifact", "replay_receipt",
                     }
                     and record["bank_manifest_sha256"] == bank_sha
                     and record["classes"] == _classes(spec, cohort),
                     f"published execution deck bank differs: {cohort}")
            _validate_controller(record["controller"])
            gate = record["eligibility_gate"]
            _require(isinstance(gate, dict) and set(gate) == {
                "metric", "operator", "threshold", "preregistered"}
                and gate == {
                    "metric": "mean_oracle_success",
                    "operator": ">=", "threshold": 0.90,
                    "preregistered": True,
                }, f"published execution gate differs: {cohort}")
            if verify_runtime_identities:
                identity = {
                    "lewm_reacher_color": lambda: _reacher_identity()[0],
                    "lewm_pusht_color": lambda: _pusht_identity()[0],
                    "dinowm_pointmaze_goal":
                        lambda: _pointmaze_identity()[0],
                }[cohort]()
                _require(record["controller"] == identity,
                         f"runtime controller/physics identity drifted: "
                         f"{cohort}")
            artifact = _safe_registry_path(output_root, record["artifact"])
            replay_path = _safe_registry_path(
                output_root, record["replay_receipt"])
            with np.load(artifact, allow_pickle=False) as archive:
                expected = {
                    "formal_test_episode_id",
                    "formal_test_native_cluster_id",
                    "selected_class_by_true_target_success",
                    "deterministic_random_class",
                }
                _require(set(archive.files) == expected,
                         f"execution cube schema differs: {cohort}")
                arrays = {name: np.asarray(archive[name]).copy()
                          for name in expected}
            replay = _read_json(replay_path)
            expected_replay_keys = {
                "schema", "study", "status", "cohort",
                "bank_manifest_sha256", "formal_labels_read",
                "development_outcomes_read", "controller_identity_sha256",
                "rows", "classes", "native_clusters", "executions",
                "replayed_executions", "deterministic_replay_fidelity",
                "execution_endpoint",
            }
            _require(set(replay) == expected_replay_keys,
                     f"published replay receipt schema differs: {cohort}")
            product = ReplayProduct(
                cohort=cohort,
                bank_manifest_sha256=bank_sha,
                episode_id=arrays["formal_test_episode_id"],
                native_cluster_id=arrays["formal_test_native_cluster_id"],
                success_cube=arrays[
                    "selected_class_by_true_target_success"],
                random_class=arrays["deterministic_random_class"],
                controller=record["controller"],
                executions=int(replay["executions"]),
                replayed_executions=int(replay["replayed_executions"]),
                replay_fidelity=float(replay[
                    "deterministic_replay_fidelity"]),
                execution_endpoint=str(replay["execution_endpoint"]),
            )
            _validate_product(
                product, classes=_classes(spec, cohort),
                expected_rows=_formal_rows(spec, cohort),
                expected_variants=VARIANTS_PER_NATIVE_CLUSTER[cohort])
            _require(replay["schema"] == EXECUTION_REPLAY_RECEIPT_SCHEMA
                     and replay["study"] == "sage-mem-v1"
                     and replay["status"] == "sealed-label-free"
                     and replay["cohort"] == cohort
                     and replay["bank_manifest_sha256"] == bank_sha
                     and replay["formal_labels_read"] is False
                     and replay["development_outcomes_read"] is False
                     and replay["rows"] == _formal_rows(spec, cohort)
                     and replay["classes"] == _classes(spec, cohort)
                     and replay["native_clusters"] == len(np.unique(
                         arrays["formal_test_native_cluster_id"]))
                     and replay["controller_identity_sha256"] == record[
                         "controller"]["controller_identity_sha256"],
                     f"published replay receipt differs: {cohort}")
        else:
            record = value["unavailable_cohorts"][cohort]
            _require(isinstance(record, dict)
                     and set(record) == {
                         "status", "bank_manifest_sha256", "reason_code",
                         "receipt"}
                     and record["status"] == "unavailable"
                     and record["bank_manifest_sha256"] == bank_sha,
                     f"unavailable execution record differs: {cohort}")
            receipt = _read_json(_safe_registry_path(
                output_root, record["receipt"]))
            _require(receipt == {
                "schema": EXECUTION_UNAVAILABLE_RECEIPT_SCHEMA,
                "study": "sage-mem-v1",
                "status": "unavailable",
                "cohort": cohort,
                "reason_code": record["reason_code"],
                "bank_manifest_sha256": bank_sha,
                "formal_labels_read": False,
                "development_outcomes_read": False,
            }, f"unavailable execution receipt differs: {cohort}")
    return value


def _seal_tree(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda value: len(value.parts),
                   reverse=True)
    for path in paths:
        _require(not path.is_symlink(), f"publication contains symlink: {path}")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def produce_execution_decks(
        spec_path: Path, preparation_root: Path, output_root: Path, *,
        resume: bool = False,
        builders: Mapping[str, ReplayBuilder] | None = None,
        progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Build or exactly revalidate the pre-reveal execution registry."""

    spec_path = spec_path.resolve()
    preparation_root = preparation_root.resolve()
    output_root = output_root.resolve()
    spec = _load_yaml(spec_path)
    _validate_spec(spec)
    using_default_builders = builders is None
    if output_root.exists():
        _require(resume, f"execution output already exists: {output_root}")
        registry = validate_published_registry(
            output_root, spec_path, preparation_root,
            verify_runtime_identities=using_default_builders)
        return {
            "schema": PRODUCER_SCHEMA,
            "study": "sage-mem-v1",
            "status": "validated-existing",
            "registry": _handle(output_root / "registry.json", output_root),
            "supported_cohorts": sorted(registry["cohorts"]),
            "unavailable_cohorts": sorted(registry["unavailable_cohorts"]),
            "formal_labels_read": False,
            "development_outcomes_read": False,
        }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(
        f".{output_root.name}.{os.getpid()}.staging")
    _require(not staging.exists(), f"stale execution staging exists: {staging}")
    staging.mkdir()
    builders = dict(DEFAULT_BUILDERS if builders is None else builders)
    try:
        available: dict[str, Any] = {}
        unavailable: dict[str, Any] = {}
        for cohort in COHORTS:
            manifest = _bank_manifest(preparation_root, cohort)
            _require(manifest.is_file() and not manifest.is_symlink(),
                     f"formal bank manifest is absent: {cohort}")
            bank_sha = _sha256_file(manifest)
            builder = builders.get(cohort)
            if builder is None:
                reason = UNAVAILABLE.get(
                    cohort, "no-registered-exact-label-free-replay-backend")
                unavailable[cohort] = _write_unavailable(
                    staging, cohort, bank_sha, reason)
                continue
            try:
                product = builder(manifest.parent, spec, progress)
                _require(product.cohort == cohort
                         and product.bank_manifest_sha256 == bank_sha,
                         f"replay product bank identity differs: {cohort}")
                _validate_product(
                    product, classes=_classes(spec, cohort),
                    expected_rows=_formal_rows(spec, cohort),
                    expected_variants=VARIANTS_PER_NATIVE_CLUSTER[cohort])
                available[cohort] = _write_product(
                    staging, product, _classes(spec, cohort))
            except ReplayUnavailable as error:
                reason = "exact-label-free-replay-unavailable-" + \
                    hashlib.sha256(str(error).encode("utf-8")).hexdigest()[:16]
                unavailable[cohort] = _write_unavailable(
                    staging, cohort, bank_sha, reason)
        _require(len(available) >= 2,
                 "fewer than two cohorts have exact label-free replay; "
                 "refusing to seal a program-level execution registry")
        registry_path = staging / "registry.json"
        _atomic_json(registry_path, {
            "schema": EXECUTION_DECK_REGISTRY_SCHEMA,
            "study": "sage-mem-v1",
            "status": "sealed",
            "available_only_after_complete_phase_a_grid": True,
            "development_outcomes_read": False,
            "cohorts": available,
            "unavailable_cohorts": unavailable,
        })
        validate_published_registry(
            staging, spec_path, preparation_root, require_immutable=False)
        _seal_tree(staging)
        os.rename(staging, output_root)
        registry = validate_published_registry(
            output_root, spec_path, preparation_root,
            verify_runtime_identities=using_default_builders)
        return {
            "schema": PRODUCER_SCHEMA,
            "study": "sage-mem-v1",
            "status": "sealed-label-free",
            "registry": _handle(output_root / "registry.json", output_root),
            "supported_cohorts": sorted(registry["cohorts"]),
            "unavailable_cohorts": sorted(registry["unavailable_cohorts"]),
            "formal_labels_read": False,
            "development_outcomes_read": False,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path,
                        default=ROOT / "configs/sage_mem_v1.yaml")
    parser.add_argument("--preparation-root", type=Path,
                        default=ROOT / "outputs/sage_mem_v1/formal_preparation")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / (
                            "outputs/sage_mem_v1/formal_preparation/"
                            "execution_decks"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true",
                      help="validate paths and print the no-write plan")
    mode.add_argument("--execute", action="store_true",
                      help="run exact physics and atomically seal the registry")
    parser.add_argument("--resume", action="store_true",
                        help="revalidate an existing sealed registry exactly")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume and not args.execute:
        raise ExecutionDeckProducerError("--resume requires --execute")
    if not args.execute:
        value = preview_execution_decks(
            args.spec, args.preparation_root, args.output_root)
    else:
        value = produce_execution_decks(
            args.spec, args.preparation_root, args.output_root,
            resume=args.resume)
    print(_canonical_json(value))
    return 0


__all__ = [
    "ExecutionDeckProducerError", "ReplayProduct", "ReplayUnavailable",
    "preview_execution_decks", "produce_execution_decks",
    "validate_published_registry",
]


if __name__ == "__main__":
    raise SystemExit(main())
