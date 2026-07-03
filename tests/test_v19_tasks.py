"""Unit tests for the V19 P1a task suite (lewm/tasks_v19).

The leakage invariants tested here are the *construction-level* guarantees the
certificates rely on: exact determinism, exact paired-branch rendering
equality outside the cue window, and independence of every nuisance draw from
xi.  Tests that need MuJoCo rendering are marked and skipped when dm_control
is unavailable; the independence checks run on the pure-numpy script sampler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.tasks_v19 import TASKS, load_bank, make_task, save_bank
from lewm.tasks_v19.certify import run_certificates
from lewm.tasks_v19.overlays import CUE_COLORS, OUProcess2D

try:
    import dm_control  # noqa: F401
    _HAS_DMC = True
except ImportError:
    _HAS_DMC = False

needs_dmc = pytest.mark.skipif(not _HAS_DMC, reason="dm_control unavailable")

E_TINY = 4


@needs_dmc
@pytest.mark.parametrize("name", ["t1", "t2", "t4"])
def test_determinism(name):
    task = make_task(name)
    first = task.generate("iid", E_TINY, seed=3)
    second = task.generate("iid", E_TINY, seed=3)
    assert np.array_equal(first.frames, second.frames)
    assert np.array_equal(first.actions, second.actions)
    assert np.array_equal(first.xi, second.xi)
    assert np.array_equal(first.exo_state, second.exo_state)


@needs_dmc
@pytest.mark.parametrize("name", TASKS)
def test_shapes_and_dtypes(name):
    task = make_task(name)
    batch = task.generate("script", 2, seed=1)
    length = task.length
    assert batch.frames.shape == (2, length, 64, 64, 3)
    assert batch.frames.dtype == np.uint8
    assert batch.actions.shape == (2, length - 1, 2)
    assert batch.actions.dtype == np.float32
    if task.xi_kind == "cat":
        assert batch.xi.shape == (2,) and batch.xi.dtype == np.int64
        assert batch.n_classes == task.n_classes >= 2
    else:
        assert batch.xi.shape == (2, 2) and batch.xi.dtype == np.float32
        assert np.abs(batch.xi).max() <= 1.0
    assert batch.endo_state.shape[:2] == (2, length)
    assert batch.exo_state.shape[:2] == (2, length)
    for value in batch.events.values():
        assert np.issubdtype(value.dtype, np.integer) and value.shape[0] == 2
    assert task.decision_time(length) == length - 1


def test_amendment2_salience_geometry():
    """Registered amendment-2 (cue salience) parameters, per task."""
    for name in TASKS:
        assert make_task(name).describe()["amendment"] == 2
    t1, t1dev = make_task("t1"), make_task("t1dev")
    assert (t1.marker_radius, t1.cue_half, t1.cue_border_px) == (6, 5, 3)
    assert (t1dev.marker_radius, t1dev.cue_half, t1dev.cue_border_px) == (6, 5, 3)
    t2, t2dev = make_task("t2"), make_task("t2dev")
    assert t2.cup_size == (12, 14) and t2dev.cup_size == (11, 13)
    assert t2.ball_radius == t2dev.ball_radius == 4
    assert t2.cue_border_px == 3 and t2.cue_border_color == (200, 200, 200)
    assert make_task("t3").sprite_size == 12
    assert make_task("t3").cue_border_px == 3
    assert make_task("t4").target_radius == 6
    assert make_task("t4").halo_color == (255, 255, 255)
    # T4 amendment 1 (exogenous respawn) must remain untouched by amendment 2.
    assert make_task("t4").respawn_range == (12, 20)


def test_t2_amended_cup_geometry_no_resting_or_long_swap_overlap():
    """Documented spacing check for the amendment-2 cup width (12/11 px).

    Cups at rest never overlap and stay in-frame, and the two movers of the
    long (0, 2) swap clear each other at every sampled frame.  (Exchanging
    cups on adjacent pairs necessarily cross mid-swap — inherent to linear
    swap motion and already true in amendment 1.)
    """
    for name in ("t2", "t2dev"):
        task = make_task(name)
        width = task.cup_size[0]
        half = width // 2
        # Resting spans disjoint and inside the 64px frame.
        spans = [(x - half, x - half + width) for x in task.slot_x]
        assert spans[0][0] >= 0 and spans[-1][1] <= 64
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0
        # Long-swap movers stay disjoint at the sampled frames.
        x0, _, x2 = task.slot_x
        for u in range(task.swap_frames):
            progress = u / (task.swap_frames - 1)
            mover_a = round(x0 + progress * (x2 - x0))
            mover_b = round(x2 + progress * (x0 - x2))
            if mover_a < mover_b:  # before they pass each other
                assert mover_b - mover_a >= width


@needs_dmc
@pytest.mark.parametrize("name", ["t1", "t1dev", "t3"])
def test_cue_border_xi_colored_during_cue_only(name):
    """Amendment 2: xi-colored frame border exactly on the cue window."""
    task = make_task(name)
    batch = task.generate("iid", E_TINY, seed=3)
    corners = [(0, 0), (0, 63), (63, 0), (63, 63)]
    for episode in range(E_TINY):
        xi = int(batch.xi[episode])
        on = int(batch.events["cue_on"][episode])
        off = int(batch.events["cue_off"][episode])
        for y, x in corners:
            assert (batch.frames[episode, on:off, y, x]
                    == np.asarray(CUE_COLORS[xi], dtype=np.uint8)).all()
        # Border strictly confined to the cue window: corner pixels change at
        # both edges of the window (scene corners never match the cue color
        # exactly, and the paired-branch tests prove post-cue xi-independence).
        assert not np.array_equal(batch.frames[episode, on - 1, 0, 0],
                                  batch.frames[episode, on, 0, 0])
        assert not np.array_equal(batch.frames[episode, off, 0, 0],
                                  batch.frames[episode, off - 1, 0, 0])


@needs_dmc
def test_t2_cue_border_fixed_color():
    """Amendment 2: t2 border is a fixed color — identical across xi0."""
    task = make_task("t2")
    branch_a, branch_b = task.paired_branches(E_TINY, seed=9)
    on, off = task.cue_window
    color = np.asarray(task.cue_border_color, dtype=np.uint8)
    table = np.asarray(task.table_color, dtype=np.uint8)
    for batch in (branch_a, branch_b):
        assert (batch.frames[:, on:off, 0, 0] == color).all()
        assert (batch.frames[:, on:off, 63, 63] == color).all()
        # Outside the cue phase the corner reverts to the table strip.
        assert (batch.frames[:, off, 0, 0] == table).all()
        assert (batch.frames[:, on - 1, 0, 0] == table).all()
    # The bottom/left/right border bands are byte-identical across xi0
    # branches (no cup reaches them).  The top band is *not* asserted: the
    # lifted cup — the cue itself, xi0-dependent by design — reaches row 0
    # during the cue phase and occludes the band at the cued slot's columns.
    assert np.array_equal(branch_a.frames[:, on:off, -3:, :],
                          branch_b.frames[:, on:off, -3:, :])
    assert np.array_equal(branch_a.frames[:, on:off, :, :3],
                          branch_b.frames[:, on:off, :, :3])
    assert np.array_equal(branch_a.frames[:, on:off, :, -3:],
                          branch_b.frames[:, on:off, :, -3:])


@needs_dmc
def test_t4_target_halo_drawn():
    """Amendment 2: 1px white halo ring around the enlarged target."""
    task = make_task("t4")
    batch = task.generate("iid", E_TINY, seed=2)
    ring_radius = task.target_radius + 1
    halo = np.asarray(task.halo_color, dtype=np.uint8)
    target = np.asarray(task.target_color, dtype=np.uint8)
    checked = 0
    for episode in range(E_TINY):
        gap_on = int(batch.events["gap_on"][episode])
        for t in range(gap_on):  # pre-gap frames show the live target
            cx, cy = np.round(batch.exo_state[episode, t, 0:2]).astype(int)
            if not (ring_radius <= cx < 64 - ring_radius
                    and ring_radius <= cy < 64 - ring_radius):
                continue
            assert (batch.frames[episode, t, cy, cx] == target).all()
            assert (batch.frames[episode, t, cy - ring_radius, cx] == halo).all()
            assert (batch.frames[episode, t, cy, cx + ring_radius] == halo).all()
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("name", ["t1", "t1dev", "t3"])
def test_cue_window_independent_of_xi(name):
    """Onset/duration come from the nuisance rng: correlation with xi ~ 0."""
    script = make_task(name).sample_script(200, seed=7)
    duration = script["cue_off"] - script["cue_on"]
    assert abs(np.corrcoef(script["xi"], script["cue_on"])[0, 1]) < 0.2
    if duration.std() > 0:  # t1dev has a fixed duration
        assert abs(np.corrcoef(script["xi"], duration)[0, 1]) < 0.2


@pytest.mark.parametrize("name", ["t2", "t2dev"])
def test_swap_pattern_independent_of_ball_slot(name):
    task = make_task(name)
    branch_a = task.sample_script(200, seed=11, xi_shift=0)
    branch_b = task.sample_script(200, seed=11, xi_shift=1)
    # Same nuisance stream -> identical swap patterns; xi stream shifted.
    assert np.array_equal(branch_a["swap_pairs"], branch_b["swap_pairs"])
    assert (branch_a["ball_slot0"] != branch_b["ball_slot0"]).all()
    assert (branch_a["xi"] != branch_b["xi"]).all()
    flat_pairs = branch_a["swap_pairs"][:, 0]
    assert abs(np.corrcoef(branch_a["ball_slot0"], flat_pairs)[0, 1]) < 0.2


@needs_dmc
@pytest.mark.parametrize("name", ["t1", "t3"])
def test_paired_branches_postcue_identical(name):
    task = make_task(name)
    branch_a, branch_b = task.paired_branches(E_TINY, seed=5)
    assert (branch_a.xi != branch_b.xi).all()
    for episode in range(E_TINY):
        on = int(branch_a.events["cue_on"][episode])
        off = int(branch_a.events["cue_off"][episode])
        assert np.array_equal(branch_a.frames[episode, off:],
                              branch_b.frames[episode, off:])
        assert np.array_equal(branch_a.frames[episode, :on],
                              branch_b.frames[episode, :on])
        assert not np.array_equal(branch_a.frames[episode, on:off],
                                  branch_b.frames[episode, on:off])


@needs_dmc
def test_t2_frames_identical_outside_cue_phase():
    """Identical cups: only the cue phase [4, 8) may differ across xi."""
    task = make_task("t2")
    branch_a, branch_b = task.paired_branches(E_TINY, seed=9)
    on, off = int(branch_a.events["cue_on"][0]), int(branch_a.events["cue_off"][0])
    assert np.array_equal(branch_a.frames[:, :on], branch_b.frames[:, :on])
    assert np.array_equal(branch_a.frames[:, off:], branch_b.frames[:, off:])
    assert not np.array_equal(branch_a.frames[:, on:off], branch_b.frames[:, on:off])


@needs_dmc
def test_t4_truth_advances_during_freeze():
    task = make_task("t4")
    batch = task.generate("iid", E_TINY, seed=2)
    for episode in range(E_TINY):
        gap_on = int(batch.events["gap_on"][episode])
        gap_off = int(batch.events["gap_off"][episode])
        frozen = batch.frames[episode, gap_on - 1]
        assert (batch.frames[episode, gap_on:gap_off] == frozen).all()
        moved = np.ptp(batch.exo_state[episode, gap_on:gap_off, 0:2], axis=0)
        assert moved.max() > 0.5
        assert not np.array_equal(batch.frames[episode, gap_off], frozen)


def test_t4_paired_branches_skipped_with_reason():
    with pytest.raises(NotImplementedError, match="nuisance OU trajectory"):
        make_task("t4").paired_branches(2, seed=0)


def test_ou_conditional_mean_closed_form():
    ou = OUProcess2D(theta=0.15, sigma=0.55, x_bounds=(6.0, 58.0),
                     y_bounds=(6.0, 58.0))
    pos = np.array([[10.0, 10.0]])
    vel = np.array([[2.0, -0.5]])
    one_step = ou.conditional_mean(pos, vel, np.array([1]))
    assert np.allclose(one_step, [[12.0, 9.5]])
    long_run = ou.conditional_mean(pos, vel, np.array([10_000]))
    assert np.allclose(long_run, pos + vel / ou.theta)  # geometric limit, in-bounds
    rng = np.random.default_rng(0)
    trajectories, velocities = ou.rollout(64, 32, rng)
    assert trajectories.min() >= 6.0 and trajectories.max() <= 58.0
    folded = ou.conditional_mean(trajectories[:, -1], 50.0 * velocities[:, -1],
                                 np.full(64, 40))
    assert folded.min() >= 6.0 and folded.max() <= 58.0


@needs_dmc
def test_bank_roundtrip(tmp_path):
    task = make_task("t1")
    batch = task.generate("iid", 2, seed=4)
    path = tmp_path / "bank.npz"
    metadata = save_bank(batch, path)
    assert metadata["npz_sha256"]
    loaded = load_bank(path)
    assert np.array_equal(loaded.frames, batch.frames)
    assert np.array_equal(loaded.actions, batch.actions)
    assert np.array_equal(loaded.xi, batch.xi)
    assert loaded.events.keys() == batch.events.keys()
    assert loaded.task == "t1" and loaded.stream == "iid" and loaded.seed == 4


@needs_dmc
def test_certificate_smoke(tmp_path):
    task = make_task("t1")
    cert = run_certificates(task, seed=0, out_dir=tmp_path, e_train=64,
                            e_eval=32, paired_episodes=8)
    for stream in ("iid", "script"):
        clauses = cert["streams"][stream]
        for name in ("integrator_probe", "postcue_pixel_probe",
                     "cue_pixel_probe", "trace_sanity_probe"):
            assert name in clauses
            assert np.isfinite(clauses[name]["value"])
    assert cert["identical_rendering"]["pass"] is True
    assert cert["identical_rendering"]["value"] == 0.0
    assert isinstance(cert["overall_pass"], bool)
    assert (tmp_path / "certificate.json").exists()
    assert (tmp_path / "identical_rendering_t1.png").exists()
