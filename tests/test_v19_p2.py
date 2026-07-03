"""Fast CPU tests for the V19 P2 evaluation stack.

Covers the eval-export round trip (writer in scripts/train_v19_p2.py, reader
in scripts/eval_v19_p2.py), the registered probe coordinates on synthetic
exports where the ground truth is known by construction (categorical and T4
continuous), the checkpoint integrator floor layout, and the aggregation +
power-analysis pipeline on a fabricated multi-arm results tree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.aggregate_v19_p2 as p2agg
import scripts.eval_v19_p2 as p2eval
import scripts.train_v19_p2 as p2train

E, L, D, A, K = 64, 32, 8, 2, 3


def _synthetic_cat_export(seed: int = 0, signal: float = 1.0) -> dict:
    """Cat export whose prior_read one-hot-encodes xi from t=10 onward."""
    rng = np.random.default_rng(seed)
    xi = rng.integers(0, K, size=E).astype(np.int64)
    prior_read = 0.05 * rng.standard_normal((E, L, D)).astype(np.float32)
    for episode in range(E):
        prior_read[episode, 10:, xi[episode]] += signal
    return {
        "prior_read": prior_read,
        "enc_o0": rng.standard_normal((E, D)).astype(np.float32),
        "actions": rng.standard_normal((E, L - 1, A)).astype(np.float32),
        "xi": xi,
        "event_cue_on": np.full(E, 4, dtype=np.int64),
        "event_cue_off": np.full(E, 8, dtype=np.int64),
        "tel_k_mean": rng.random((E, L)).astype(np.float32),
        "meta": {
            "schema_version": 1, "task": "t1dev", "arm": "lkc", "seed": 0,
            "host": "vicreg", "xi_kind": "cat", "n_classes": K,
            "episodes": E, "length": L, "embed_dim": D,
        },
    }


def _synthetic_cont_export(seed: int = 0) -> dict:
    """T4-style export: prior_read at gap_off linearly encodes xi."""
    rng = np.random.default_rng(seed)
    xi = rng.uniform(-1.0, 1.0, size=(E, 2)).astype(np.float32)
    gap_off = rng.integers(20, 28, size=E).astype(np.int64)
    prior_read = 0.02 * rng.standard_normal((E, L, D)).astype(np.float32)
    prior_read[np.arange(E), gap_off, 0:2] += xi
    return {
        "prior_read": prior_read,
        "enc_o0": rng.standard_normal((E, D)).astype(np.float32),
        "actions": rng.standard_normal((E, L - 1, A)).astype(np.float32),
        "xi": xi,
        "event_gap_on": (gap_off - 8).astype(np.int64),
        "event_gap_off": gap_off,
        "posterior_mean": (xi + 0.01).astype(np.float32),
        "frozen_pos": (xi + 1.0).astype(np.float32),
        "meta": {
            "schema_version": 1, "task": "t4", "arm": "lkc", "seed": 0,
            "host": "vicreg", "xi_kind": "cont", "n_classes": 0,
            "episodes": E, "length": L, "embed_dim": D,
        },
    }


def _write_export(export: dict, path: Path) -> None:
    arrays = {name: value for name, value in export.items() if name != "meta"}
    p2train.write_eval_export(path, arrays, export["meta"])


# --------------------------------------------------------------------------
# Export round trip
# --------------------------------------------------------------------------

def test_eval_export_round_trip(tmp_path):
    export = _synthetic_cat_export()
    path = tmp_path / "t1dev" / "lkc" / "s0" / "eval_export.npz"
    _write_export(export, path)
    loaded = p2eval.load_export(path)
    assert loaded["meta"] == export["meta"]
    for name, value in export.items():
        if name == "meta":
            continue
        assert np.array_equal(loaded[name], value), name
    events = p2eval.export_events(loaded)
    assert set(events) == {"cue_on", "cue_off"}
    with pytest.raises(FileExistsError):
        _write_export(export, path)          # write-once discipline


# --------------------------------------------------------------------------
# Registered feature coordinates
# --------------------------------------------------------------------------

def test_registered_cat_features_window_math():
    prior_read = np.zeros((2, 6, 2), dtype=np.float32)
    prior_read[0, 3:, 0] = 2.0               # window [3..5] on episode 0
    prior_read[1, 4:, 1] = 4.0               # window [4..5] on episode 1
    features = p2eval.registered_cat_features(
        prior_read, np.array([3, 4], dtype=np.int64))
    assert features.shape == (2, 4)          # mean(D) ++ prior_read[t_dec](D)
    assert features[0].tolist() == [2.0, 0.0, 2.0, 0.0]
    assert features[1].tolist() == [0.0, 4.0, 0.0, 4.0]


def test_deep_window_start_prefers_shuffle_off():
    events = {"cue_off": np.array([8]), "shuffle_off": np.array([20])}
    assert p2eval.deep_window_start(events).tolist() == [22]
    assert p2eval.deep_window_start({"cue_off": np.array([8])}).tolist() == [10]


def test_integrator_floor_feature_layout():
    enc_o0 = np.zeros((4, D), dtype=np.float32)
    actions = np.ones((4, L - 1, A), dtype=np.float32)
    features = p2eval.integrator_floor_features(enc_o0, actions)
    # [enc(o_0) (D), a_{t-3:t-1} (3A), sum a (A), t/(L-1) (1)]
    assert features.shape == (4, D + 3 * A + A + 1)
    assert np.allclose(features[:, D:D + 3 * A], 1.0)
    assert np.allclose(features[:, D + 3 * A:D + 4 * A], L - 1)
    assert np.allclose(features[:, -1], 1.0)


# --------------------------------------------------------------------------
# Probe pipeline on synthetic exports
# --------------------------------------------------------------------------

def test_probe_pipeline_cat(tmp_path):
    export = _synthetic_cat_export()
    results = p2eval.run_probes(export, probe_seeds=(0, 1, 2))
    assert results["metric"] == "accuracy"
    assert results["chance"] == pytest.approx(1.0 / K)
    # The signal is planted in the registered window -> near-perfect probe.
    assert results["registered"]["mean"] > 0.9
    assert results["t_dec"]["mean"] > 0.9
    assert results["last8"]["mean"] > 0.9
    # Random enc_o0/actions -> integrator floor near chance.
    assert results["floor"]["mean"] < 0.7
    assert results["memory_advantage"] > 0.2
    assert len(results["registered"]["per_probe_seed"]) == 3

    path = tmp_path / "t1dev" / "lkc" / "s0" / "eval_export.npz"
    _write_export(export, path)
    written = p2eval.process_run(path)
    assert (path.parent / p2eval.RESULTS_NAME).exists()
    assert written["registered"]["mean"] == pytest.approx(
        results["registered"]["mean"])
    # Cached on the second call (no force).
    assert p2eval.process_run(path)["registered"]["mean"] == pytest.approx(
        results["registered"]["mean"])
    assert p2eval.discover_exports(tmp_path) == [path]


def test_probe_pipeline_cont():
    export = _synthetic_cont_export()
    results = p2eval.run_probes(export, probe_seeds=(0, 1, 2))
    assert results["metric"] == "r2"
    assert results["registered"]["mean"] > 0.8
    assert results["posterior_mean_r2"]["mean"] > 0.99
    assert results["closer_to_posterior"]["mean"] == pytest.approx(1.0)
    assert (results["dist_to_posterior_mean"]["mean"]
            < results["dist_to_frozen"]["mean"])
    assert results["floor"]["mean"] < 0.3


# --------------------------------------------------------------------------
# Aggregation + power analysis
# --------------------------------------------------------------------------

def _fake_results(task: str, arm: str, seed: int, xi: float,
                  floor: float = 0.33) -> dict:
    return {
        "schema_version": 1, "task": task, "arm": arm, "seed": seed,
        "host": "vicreg", "xi_kind": "cat", "n_classes": 3, "chance": 1 / 3,
        "metric": "accuracy",
        "registered": {"mean": xi, "std": 0.0, "per_probe_seed": [xi] * 3},
        "floor": {"mean": floor, "std": 0.0, "per_probe_seed": [floor] * 3},
        "memory_advantage": xi - floor,
    }


def test_aggregate_and_power_analysis(tmp_path):
    scores = {
        "lkc": {0: 0.82, 1: 0.85, 2: 0.80},
        "acgru": {0: 0.70, 1: 0.66, 2: 0.72},
        "acssm": {0: 0.72, 1: 0.64, 2: 0.69},
        "lkc_k0": {0: 0.55, 1: 0.52, 2: 0.58},
    }
    for arm, cells in scores.items():
        for seed, xi in cells.items():
            run_dir = tmp_path / "t1dev" / arm / f"s{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / p2eval.RESULTS_NAME).write_text(
                json.dumps(_fake_results("t1dev", arm, seed, xi)))

    summary = p2agg.aggregate(tmp_path)
    task = summary["tasks"]["t1dev"]
    assert set(task["arms"]) == set(scores)
    assert task["arms"]["lkc"]["xi_probe"]["n"] == 3

    paired = task["paired_vs_envelope"]
    assert set(paired) == {"lkc", "lkc_k0"}          # references excluded
    # Envelope = per-seed max(acgru, acssm) = {0: .72, 1: .66, 2: .72}.
    assert paired["lkc"]["differences"] == pytest.approx([0.10, 0.19, 0.08])
    assert paired["lkc"]["wins"] == 3
    assert paired["lkc_k0"]["mean"] < 0

    power = task["power_analysis"]
    assert power["status"] == "ok"
    assert power["observed_effect"] == pytest.approx(np.mean([0.10, 0.19, 0.08]))
    observed = power["effects"]["observed"]
    assert set(observed["power_by_n_seeds"]) == {3, 5, 8, 10}
    # A ~12-point effect with ~5-point sd is detectable at tiny n.
    assert observed["smallest_n_with_80pct_power"] in (3, 5)
    assert "registered_plus_5pct" in power["effects"]
    assert summary["pooled_power_analysis"]["status"] == "ok"

    # main() writes both summary artifacts.
    p2agg.main(["--root", str(tmp_path)])
    assert (tmp_path / "p2_summary.json").exists()
    markdown = (tmp_path / "p2_summary.md").read_text()
    assert "t1dev" in markdown and "power" in markdown


def test_power_analysis_edge_cases():
    strong = p2agg.power_analysis([0.10, 0.12, 0.11])
    assert strong["status"] == "ok"
    assert strong["effects"]["observed"]["smallest_n_with_80pct_power"] == 3
    assert p2agg.power_analysis([0.1])["status"] == "insufficient_seed_pairs"
    null = p2agg.power_analysis([0.001, -0.002, 0.0005, -0.001])
    weak_power = null["effects"]["observed"]["power_by_n_seeds"][3]
    assert weak_power < 0.8


# --------------------------------------------------------------------------
# Full export path on the real vicreg host (CPU; exercises the t4 branch)
# --------------------------------------------------------------------------

def test_export_eval_cont_branch_end_to_end(tmp_path):
    import torch

    from lewm.models.v19_carriers import make_carrier
    from lewm.tasks_v19.base import EpisodeBatch
    import scripts.train_v19_p0 as p0

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    episodes, length = 6, 64
    gap_on = rng.integers(28, 32, size=episodes).astype(np.int64)
    gap_off = gap_on + rng.integers(16, 20, size=episodes).astype(np.int64)
    bank = EpisodeBatch(
        frames=rng.integers(0, 256, size=(episodes, length, 64, 64, 3),
                            dtype=np.uint8),
        actions=rng.standard_normal(
            (episodes, length - 1, 2)).astype(np.float32),
        xi=rng.uniform(-1, 1, size=(episodes, 2)).astype(np.float32),
        xi_kind="cont", n_classes=0,
        endo_state=np.zeros((episodes, length, 3), dtype=np.float32),
        exo_state=rng.uniform(10, 50, size=(episodes, length, 4)).astype(
            np.float32),
        events={"gap_on": gap_on, "gap_off": gap_off,
                "respawn": np.full(episodes, 15, dtype=np.int64)},
        stream="iid", task="t4", seed=0)

    model = p0.build_vicreg_host(action_dim=2)
    carrier = make_carrier("lkc", 128, 2)
    path = tmp_path / "t4" / "lkc" / "s0" / "eval_export.npz"
    arrays = p2train.export_eval(
        "vicreg", model, carrier, bank, "t4", "lkc", 0, path,
        torch.device("cpu"), chunk=4)
    assert arrays["prior_read"].shape == (episodes, length, 128)
    assert arrays["posterior_mean"].shape == (episodes, 2)
    assert arrays["frozen_pos"].shape == (episodes, 2)
    assert "tel_k" in arrays and arrays["tel_k"].shape == (episodes, length, 128)

    loaded = p2eval.load_export(path)
    assert loaded["meta"]["xi_kind"] == "cont"
    results = p2eval.run_probes(loaded, probe_seeds=(0,))
    for key in ("registered", "floor", "posterior_mean_r2",
                "dist_to_posterior_mean", "dist_to_frozen"):
        assert np.isfinite(results[key]["mean"]), key


# --------------------------------------------------------------------------
# Trainer helpers that need no GPU
# --------------------------------------------------------------------------

def test_resolve_banks_rejects_unknown_task(tmp_path):
    with pytest.raises(ValueError):
        p2train.resolve_banks("t9", tmp_path / "p0", tmp_path / "p2")


def test_resolve_banks_refuses_p0_root_generation(tmp_path):
    root = tmp_path / "shared"
    with pytest.raises(ValueError):
        p2train.resolve_banks("t1dev", root, root)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
