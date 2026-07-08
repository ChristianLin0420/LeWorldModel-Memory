#!/usr/bin/env python3
"""Freeze features, pre-test gates, and arm-blind PushT consumers on CUDA:1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
from joblib import Parallel, delayed
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier  # noqa: E402
from lewm.official_tasks.artifacts import sha256_file  # noqa: E402
from lewm.official_tasks.pusht_downstream import (  # noqa: E402
    deterministic_partition,
    stable_json,
)
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    aligned_pusht_latents,
    load_pusht_base_cache,
    load_pusht_task_cache,
    pusht_carrier_directory,
)
from lewm.official_tasks.pusht_spec import load_locked_pusht_spec  # noqa: E402


DEFAULT_SPEC = ROOT / "configs/paper_a_pusht_downstream_use_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="verify and reuse completed immutable feature/consumer artifacts")
    return parser.parse_args()


def _load_locked_spec(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    sidecar = path.with_suffix(".sha256")
    fields = sidecar.read_text().strip().split()
    actual = sha256_file(path)
    if len(fields) != 2 or fields[0] != actual or fields[1] != path.name:
        raise RuntimeError("PushT downstream protocol SHA sidecar does not verify")
    return yaml.safe_load(path.read_text()), actual


def _classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000,
                           random_state=0),
    )


def _score(train_x: np.ndarray, train_y: np.ndarray,
           eval_x: np.ndarray, eval_y: np.ndarray) -> dict:
    model = _classifier()
    model.fit(train_x, train_y)
    pred = model.predict(eval_x)
    labels = np.arange(int(np.max(train_y)) + 1)
    recalls = recall_score(eval_y, pred, labels=labels,
                           average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(eval_y, pred)),
        "accuracy": float(np.mean(eval_y == pred)),
        "per_class_recall": [float(value) for value in recalls],
    }


@torch.no_grad()
def _prior(carrier: torch.nn.Module, z: np.ndarray, actions: np.ndarray,
           device: torch.device, batch_size: int) -> np.ndarray:
    carrier.eval()
    rows = []
    for start in range(0, len(z), batch_size):
        output = carrier(
            torch.from_numpy(z[start:start + batch_size]).to(device),
            torch.from_numpy(actions[start:start + batch_size]).to(device),
        )
        rows.append(output.prior_read[:, 19].float().cpu().numpy())
    return np.concatenate(rows)


def _load_carrier(spec: dict, task: str, arm: str, seed: int,
                  device: torch.device) -> tuple[torch.nn.Module, dict]:
    directory = pusht_carrier_directory(spec, task, arm, seed)
    manifest = json.loads((directory / "manifest.json").read_text())
    checkpoint = directory / "carrier.pt"
    expected = manifest["artifacts"]["checkpoint"]["sha256"]
    if sha256_file(checkpoint) != expected:
        raise RuntimeError(f"carrier checkpoint hash differs: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    carrier = make_frozen_carrier(arm, 192, 10)
    carrier.load_state_dict(payload["carrier_state_dict"], strict=True)
    return carrier.to(device), {
        "path": str(checkpoint.relative_to(ROOT)), "sha256": expected,
    }


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez_compressed(path, **arrays)


def _feature_path(root: Path, task: str, arm: str, seed: int) -> Path:
    return root / "features" / task / arm / f"seed-{seed}.npz"


def main() -> None:
    args = parse_args()
    spec, protocol_sha = _load_locked_spec(args.spec)
    if not args.execute:
        raise RuntimeError("preparation writes formal artifacts; pass --execute")
    if args.device != "cuda:1":
        raise ValueError("locked PushT downstream preparation only permits cuda:1")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        raise RuntimeError("physical CUDA device 1 is unavailable")
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")
    torch.manual_seed(791201)
    np.random.seed(791201)

    parent = load_locked_pusht_spec()
    root = ROOT / spec["artifacts"]["root"]
    receipt_path = ROOT / spec["artifacts"]["protocol_receipt"]
    gates_path = ROOT / spec["artifacts"]["gates"]
    if receipt_path.exists() or gates_path.exists():
        raise FileExistsError("formal downstream preparation already exists")

    base_train, _ = load_pusht_base_cache(parent, "train")
    base_test, _ = load_pusht_base_cache(parent, "validation")
    train_rows, dev_rows = deterministic_partition(
        base_train["episode_index"],
        seed=int(spec["splits"]["consumer_train"]["seed"]),
        train_count=int(spec["splits"]["consumer_train"]["episodes"]),
    )
    if set(base_train["episode_index"]).intersection(
            set(base_test["episode_index"])):
        raise RuntimeError("source train and held-out episode ids overlap")

    root.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict] = []
    tasks_data: dict[str, dict] = {}
    task_gates: dict[str, dict] = {}
    arms = list(spec["source_audit"]["carriers"]["arms"])
    seeds = [int(value) for value in spec["source_audit"]["carriers"]["seeds"]]
    batch_size = int(spec["compute"]["feature_batch_size"])

    for task_record in spec["tasks"]:
        task = task_record["key"]
        classes = int(task_record["classes"])
        task_train, _ = load_pusht_task_cache(parent, task, "train")
        task_test, _ = load_pusht_task_cache(parent, task, "validation")
        for base, cue in ((base_train, task_train), (base_test, task_test)):
            if (not np.array_equal(base["episode_index"], cue["episode_index"])
                    or not np.array_equal(base["local_start"], cue["local_start"])):
                raise RuntimeError(f"base/task row alignment differs for {task}")
        z_train = aligned_pusht_latents(base_train, task_train, 1, 3)
        z_test = aligned_pusht_latents(base_test, task_test, 1, 3)
        y_train = task_train["labels"].astype(np.int64)

        cue = _score(
            task_train["z_cue"][train_rows].reshape(len(train_rows), -1),
            y_train[train_rows],
            task_train["z_cue"][dev_rows].reshape(len(dev_rows), -1),
            y_train[dev_rows],
        )
        shortcut_features = {
            "final_context_latent": z_train[:, 16:19].reshape(len(z_train), -1),
            "final_action": base_train["actions"][:, 18],
            "final_state": base_train["state"][:, 19],
            "time": base_train["local_start"][:, None].astype(np.float32),
        }
        shortcuts = {
            name: _score(values[train_rows], y_train[train_rows],
                         values[dev_rows], y_train[dev_rows])
            for name, values in shortcut_features.items()
        }
        shuffle_rng = np.random.default_rng(
            int(spec["pre_test_gates"]["label_shuffle"]["seed"]))
        shuffle_scores = []
        cue_train_x = task_train["z_cue"][train_rows].reshape(len(train_rows), -1)
        cue_dev_x = task_train["z_cue"][dev_rows].reshape(len(dev_rows), -1)
        for _ in range(int(spec["pre_test_gates"]["label_shuffle"]["permutations"])):
            shuffled = shuffle_rng.permutation(y_train[train_rows])
            shuffle_scores.append(_score(
                cue_train_x, shuffled, cue_dev_x, y_train[dev_rows]
            )["balanced_accuracy"])

        chance = 1.0 / classes
        gate_spec = spec["pre_test_gates"]
        cue_pass = (cue["balanced_accuracy"]
                    >= float(gate_spec["cue_window_upper_bound"]["development_minimum"])
                    and min(cue["per_class_recall"])
                    >= float(gate_spec["cue_window_upper_bound"]["per_class_recall_minimum"]))
        max_shortcut = chance + float(
            gate_spec["no_shortcut_receipts"]["maximum_above_chance"])
        shortcut_pass = all(value["balanced_accuracy"] <= max_shortcut
                            for value in shortcuts.values())
        shuffle_mean = float(np.mean(shuffle_scores))
        shuffle_pass = shuffle_mean <= chance + float(
            gate_spec["label_shuffle"]["maximum_mean_above_chance"])
        task_gates[task] = {
            "classes": classes,
            "chance": chance,
            "cue_window_upper_bound": cue,
            "cue_window_pass": cue_pass,
            "shortcuts": shortcuts,
            "shortcut_maximum": max_shortcut,
            "shortcuts_pass": shortcut_pass,
            "label_shuffle_balanced_accuracy": shuffle_scores,
            "label_shuffle_mean": shuffle_mean,
            "label_shuffle_pass": shuffle_pass,
            "semantic_pass": bool(cue_pass and shortcut_pass and shuffle_pass),
            "physical_oracle_pass": None,
            "formal_test_released": False,
        }

        raw_train = z_train[:, 16:19].reshape(len(z_train), -1).astype(np.float32)
        raw_test = z_test[:, 16:19].reshape(len(z_test), -1).astype(np.float32)
        tasks_data[task] = {
            "z_train": z_train, "z_test": z_test,
            "raw_train": raw_train, "raw_test": raw_test,
            "y_train": y_train,
        }
        for arm in arms:
            for seed in seeds:
                carrier, checkpoint = _load_carrier(
                    parent, task, arm, seed, device)
                checkpoints.append({"task": task, "arm": arm, "seed": seed,
                                    **checkpoint})
                path = _feature_path(root, task, arm, seed)
                if path.exists():
                    if not args.resume:
                        raise FileExistsError(f"refusing to overwrite {path}")
                    completed = np.load(path)
                    if (completed["prior_train"].shape != (len(z_train), 192)
                            or completed["prior_test"].shape != (len(z_test), 192)
                            or not np.array_equal(
                                completed["episode_train"],
                                base_train["episode_index"])
                            or not np.array_equal(
                                completed["episode_test"],
                                base_test["episode_index"])):
                        raise RuntimeError(f"resume feature contract differs: {path}")
                    del carrier
                    continue
                prior_train = _prior(
                    carrier, z_train, base_train["actions"], device, batch_size)
                prior_test = _prior(
                    carrier, z_test, base_test["actions"], device, batch_size)
                _write_npz(
                    path,
                    prior_train=prior_train.astype(np.float32),
                    prior_test=prior_test.astype(np.float32),
                    episode_train=base_train["episode_index"],
                    episode_test=base_test["episode_index"],
                )
                del carrier, prior_train, prior_test

    # Fit one arm-blind consumer per task and carrier-training seed.  Held-out
    # labels are deliberately not loaded in this preparation process.
    def fit_consumer(task_record: dict, seed: int) -> dict:
        task = task_record["key"]
        data = tasks_data[task]
        x_parts, y_parts = [], []
        for arm in arms:
            feature = np.load(_feature_path(root, task, arm, seed))
            x_parts.append(np.concatenate(
                (data["raw_train"][train_rows],
                 feature["prior_train"][train_rows]), axis=1))
            y_parts.append(data["y_train"][train_rows])
        output = root / "consumers" / task / f"seed-{seed}.joblib"
        if output.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite {output}")
            model = joblib.load(output)
        else:
            model = _classifier()
            model.fit(np.concatenate(x_parts), np.concatenate(y_parts))
            output.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, output)
        dev_by_arm = {}
        for arm in arms:
            feature = np.load(_feature_path(root, task, arm, seed))
            x_dev = np.concatenate(
                (data["raw_train"][dev_rows],
                 feature["prior_train"][dev_rows]), axis=1)
            pred = model.predict(x_dev)
            dev_by_arm[arm] = float(balanced_accuracy_score(
                data["y_train"][dev_rows], pred))
        return {
            "task": task, "seed": seed,
            "path": str(output.relative_to(ROOT)),
            "sha256": sha256_file(output),
            "development_balanced_accuracy_by_arm": dev_by_arm,
        }

    consumer_jobs = [
        (task_record, seed)
        for task_record in spec["tasks"] for seed in seeds
    ]
    consumers = Parallel(n_jobs=5, prefer="threads")(
        delayed(fit_consumer)(task_record, seed)
        for task_record, seed in consumer_jobs
    )

    receipt = {
        "schema": "paper_a_pusht_downstream_protocol_receipt_v1",
        "protocol": {
            "path": str(args.spec.resolve().relative_to(ROOT)),
            "sha256": protocol_sha,
        },
        "physical_cuda_device": 1,
        "split": {
            "consumer_train_rows": train_rows.tolist(),
            "development_rows": dev_rows.tolist(),
            "consumer_train_episode_ids": base_train["episode_index"][train_rows].tolist(),
            "development_episode_ids": base_train["episode_index"][dev_rows].tolist(),
            "heldout_test_episode_ids": base_test["episode_index"].tolist(),
        },
        "simulator_contract": {
            "commit": spec["upstream_simulator"]["commit"],
            "checkout": spec["upstream_simulator"]["checkout"],
            "eligibility_interval": spec["physical_goal_set"]["eligibility"]["block_x_y_closed_interval"],
            "position_tolerance_pixels": spec["physical_goal_set"]["position_tolerance_pixels"],
            "angle_tolerance_radians": spec["physical_goal_set"]["angle_tolerance_radians"],
            "controller": spec["controller"],
            "physical_gate": spec["pre_test_gates"]["physical_oracle"],
        },
        "source_checkpoints": checkpoints,
        "consumers": consumers,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(stable_json(receipt))
    gates = {
        "schema": "paper_a_pusht_downstream_gates_v1",
        "protocol_sha256": protocol_sha,
        "protocol_receipt": {
            "path": str(receipt_path.relative_to(ROOT)),
            "sha256": sha256_file(receipt_path),
        },
        "tasks": task_gates,
        "physical_development_pending": True,
    }
    gates_path.write_text(stable_json(gates))
    print(stable_json({
        "status": "prepared_without_heldout_metrics",
        "protocol_sha256": protocol_sha,
        "gates": task_gates,
        "receipt": str(receipt_path),
    }))


if __name__ == "__main__":
    main()
