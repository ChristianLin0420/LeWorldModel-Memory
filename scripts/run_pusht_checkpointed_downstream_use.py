#!/usr/bin/env python3
"""Shared PushT downstream-use gate for checkpointed host-writer adapters.

This script is intentionally conservative.  It tests an external goal-selection
consumer on checkpointed DINO-WM and LeWM host-writer features.  Physical
execution is reported only for hosts whose validation rows have a matching
PushT simulator deck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_mem_jepa_pusht_stage_f import (  # noqa: E402
    FrozenPushTHost,
    HostAlignedEvidenceWriterAdapter,
    MemJepaLabelFreeAdapter,
    PushTFeatureBank,
    collect_features as collect_dino_features,
    load_config as load_dino_config,
)
from scripts.run_lewm_pusht_host_writer import (  # noqa: E402
    LeWMHostAlignedEvidenceWriter,
    age_adjusted_spec,
    collect_level_features as collect_lewm_features,
    load_admitted as load_lewm_admitted,
)
from lewm.models.official_lewm_pusht import load_official_pusht_checkpoint  # noqa: E402
from lewm.official_tasks.artifacts import sha256_file  # noqa: E402
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    resolve_pusht_path,
)


DEFAULT_OUTPUT = ROOT / "outputs/pusht_checkpointed_downstream_use_v1"
DEFAULT_DINO_ROOT = ROOT / "outputs/mem_jepa_pusht_stage_f_host_writer_checkpointed_v1"
DEFAULT_LEWM_ROOT = ROOT / "outputs/lewm_pusht_host_writer_counterfactual_checkpointed_v1"
DEFAULT_TASK = "multi-item-visual-binding-recall"
DEFAULT_DINO_DECK_DIR = ROOT / "outputs/pusht_checkpointed_downstream_use_v1/dinowm_simulator/test"


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=4000,
            random_state=0,
        ),
    )


def score_predictions(prediction: np.ndarray, truth: np.ndarray,
                      classes: int) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "count": int(len(truth)),
        "classes": int(classes),
    }


def result_path(root: Path, task: str, seed: int, age: int) -> Path:
    return root / task / f"s{int(seed)}" / f"age_{int(age)}" / "result.json"


def display_path(path: Path) -> str:
    value = path if path.is_absolute() else ROOT / path
    return str(value.resolve().relative_to(ROOT))


def load_checkpoint_from_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(path.read_text())
    checkpoint = result.get("checkpoint")
    if not checkpoint:
        raise RuntimeError(f"result has no checkpoint record: {path}")
    checkpoint_path = ROOT / checkpoint["path"]
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return result, payload


def load_dino_model(payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = payload["model"]
    if payload["adapter"] == "host_writer":
        model = HostAlignedEvidenceWriterAdapter(
            dim=int(model_cfg["dim"]),
            slots=int(model_cfg["slots"]),
            heads=int(model_cfg["heads"]),
            residual_scale=float(model_cfg.get("residual_scale", 1.0)),
        )
    else:
        model = MemJepaLabelFreeAdapter(
            dim=int(model_cfg["dim"]),
            slots=int(model_cfg["slots"]),
            heads=int(model_cfg["heads"]),
            residual_scale=float(model_cfg.get("residual_scale", 1.0)),
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def dino_host_result(root: Path, task: str, age: int, seeds: list[int],
                     device: torch.device, batch_size: int) -> dict[str, Any]:
    first_result, first_payload = load_checkpoint_from_result(
        result_path(root, task, seeds[0], age))
    cfg = load_dino_config(ROOT / first_payload["config"])
    bank = PushTFeatureBank(cfg)
    host = FrozenPushTHost(cfg, device)
    task_record = bank.task_record(task)
    classes = int(task_record["classes"])
    train_idx = bank.indices("train")
    val_idx = bank.indices("validation")
    y_train = np.asarray(bank.labels[task][train_idx], dtype=np.int64)
    y_val = np.asarray(bank.labels[task][val_idx], dtype=np.int64)
    per_seed = []
    execution_records = []
    physical = load_dino_physical_deck(task, y_val, classes)
    conditions = ("full", "reset", "no_state")
    for seed in seeds:
        result, payload = load_checkpoint_from_result(result_path(root, task, seed, age))
        model = load_dino_model(payload, device)
        feature_mode = str(payload.get("feature_mode", result.get("feature_mode", "binding_slots")))
        train_full = collect_dino_features(
            model, host, bank, task=task, split="train", age=age,
            condition="full", batch_size=batch_size,
            feature_mode=feature_mode)["features"]
        consumer = classifier()
        consumer.fit(train_full, y_train)
        seed_record = {"seed": int(seed), "conditions": {}}
        for condition in conditions:
            x_val = collect_dino_features(
                model, host, bank, task=task, split="validation", age=age,
                condition=condition, batch_size=batch_size,
                feature_mode=feature_mode)["features"]
            pred = consumer.predict(x_val).astype(np.int64)
            condition_record = {
                **score_predictions(pred, y_val, classes),
                "prediction_histogram": np.bincount(pred, minlength=classes).tolist(),
            }
            if physical is not None:
                condition_record["execution"] = execute_with_physical_deck(
                    physical, pred, y_val)
            seed_record["conditions"][condition] = condition_record
        per_seed.append(seed_record)
        if physical is not None:
            execution_records.append(seed_record)
    return summarize_host(
        host_key="dinowm",
        host_label="DINO-WM",
        task=task,
        age=age,
        classes=classes,
        per_seed=per_seed,
        execution=execution_records if execution_records else None,
        extra={
            "bank": "DINO-WM native PushT feature bank",
            "physical_execution": (
                "reported with DINO-WM native PushT simulator deck"
                if physical is not None else "not reported"
            ),
            "source_result_example": display_path(result_path(root, task, seeds[0], age)),
            "checkpoint_schema": first_payload["schema"],
        },
    )


def load_lewm_model(payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = payload["model"]
    model = LeWMHostAlignedEvidenceWriter(
        target_dim=int(model_cfg["target_dim"]),
        dim=int(model_cfg["dim"]),
        slots=int(model_cfg["slots"]),
        heads=int(model_cfg["heads"]),
        residual_scale=float(model_cfg.get("residual_scale", 1.0)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def lewm_host_result(root: Path, task: str, age: int, seeds: list[int],
                     device: torch.device, batch_size: int) -> dict[str, Any]:
    locked_spec = load_locked_pusht_spec(DEFAULT_PUSHT_SPEC, DEFAULT_PUSHT_LOCK)
    spec = age_adjusted_spec(locked_spec, int(age))
    train = load_lewm_admitted(locked_spec, task, "train")
    validation = load_lewm_admitted(locked_spec, task, "validation")
    if int(age) != 15:
        raise RuntimeError("LeWM downstream extractor currently supports age15 only")
    bundle = resolve_pusht_path(locked_spec["official_host"]["bundle_path"])
    host = load_official_pusht_checkpoint(bundle, device).eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    classes = int(next(item["classes"] for item in locked_spec["semantic_tasks"]
                      if item["key"] == task))
    y_train = np.asarray(train["labels"], dtype=np.int64)
    y_val = np.asarray(validation["labels"], dtype=np.int64)
    per_seed = []
    execution_records = []
    physical = load_lewm_physical_deck(task, y_val)
    for seed in seeds:
        result, payload = load_checkpoint_from_result(result_path(root, task, seed, age))
        model = load_lewm_model(payload, device)
        train_full = collect_lewm_features(
            model, host, train, spec, condition="full", level="host_output",
            batch_size=batch_size, device=device)
        consumer = classifier()
        consumer.fit(train_full, y_train)
        seed_record = {"seed": int(seed), "conditions": {}}
        for condition in ("full", "reset", "no_state"):
            x_val = collect_lewm_features(
                model, host, validation, spec, condition=condition,
                level="host_output", batch_size=batch_size, device=device)
            pred = consumer.predict(x_val).astype(np.int64)
            condition_record = {
                **score_predictions(pred, y_val, classes),
                "prediction_histogram": np.bincount(pred, minlength=classes).tolist(),
            }
            if physical is not None:
                condition_record["execution"] = execute_with_physical_deck(
                    physical, pred, y_val)
            seed_record["conditions"][condition] = condition_record
        per_seed.append(seed_record)
        if physical is not None:
            execution_records.append(seed_record)
    return summarize_host(
        host_key="lewm",
        host_label="LeWM",
        task=task,
        age=age,
        classes=classes,
        per_seed=per_seed,
        execution=execution_records if execution_records else None,
        extra={
            "bank": "Official LeWM PushT memory bank",
            "physical_execution": (
                "reported with existing PushT simulator deck"
                if physical is not None else "not reported"
            ),
            "source_result_example": display_path(result_path(root, task, seeds[0], age)),
        },
    )


def load_lewm_physical_deck(task: str, y_val: np.ndarray) -> dict[str, np.ndarray] | None:
    path = ROOT / "outputs/pusht_downstream_use_v1/simulator/test" / f"{task}.npz"
    base_path = ROOT / "outputs/official_pusht_memory/cache/base/validation.npz"
    if not path.is_file() or not base_path.is_file():
        return None
    deck = np.load(path, allow_pickle=False)
    base = np.load(base_path, allow_pickle=False)
    rows = np.asarray(deck["source_rows"], dtype=np.int64)
    if not np.array_equal(deck["episode_ids"], base["episode_index"][rows]):
        return None
    if rows.max(initial=-1) >= len(y_val):
        return None
    return {key: np.asarray(deck[key]) for key in deck.files}


def load_dino_physical_deck(task: str, y_val: np.ndarray,
                            classes: int) -> dict[str, np.ndarray] | None:
    path = DEFAULT_DINO_DECK_DIR / f"{task}.npz"
    receipt_path = DEFAULT_DINO_DECK_DIR / f"{task}.receipt.json"
    if not path.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if int(receipt.get("classes", -1)) != int(classes):
        return None
    if receipt.get("task") != task:
        return None
    if sha256_file(path) != receipt.get("artifact", {}).get("sha256"):
        raise RuntimeError(f"DINO physical deck hash changed: {path}")
    deck = np.load(path, allow_pickle=False)
    rows = np.asarray(deck["source_rows"], dtype=np.int64)
    if rows.max(initial=-1) >= len(y_val):
        return None
    expected = {
        "source_rows", "success_matrix", "regret_matrix",
        "goal_pose", "final_pose", "cost_matrix",
    }
    if not expected.issubset(set(deck.files)):
        return None
    return {key: np.asarray(deck[key]) for key in deck.files}


def execute_with_physical_deck(deck: dict[str, np.ndarray], prediction: np.ndarray,
                               labels: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(deck["source_rows"], dtype=np.int64)
    pred = np.asarray(prediction[rows], dtype=np.int64)
    truth = np.asarray(labels[rows], dtype=np.int64)
    index = np.arange(len(rows))
    success = deck["success_matrix"][index, pred, truth].astype(np.float64)
    regret = deck["regret_matrix"][index, pred, truth].astype(np.float64)
    oracle = deck["success_matrix"][index, truth, truth].astype(np.float64)
    return {
        "eligible_episodes": int(len(rows)),
        "executed_success": float(success.mean()),
        "oracle_success": float(oracle.mean()),
        "mean_pose_regret": float(regret.mean()),
    }


def summarize_host(host_key: str, host_label: str, task: str, age: int,
                   classes: int, per_seed: list[dict[str, Any]],
                   execution: list[dict[str, Any]] | None,
                   extra: dict[str, Any]) -> dict[str, Any]:
    conditions = ("full", "reset", "no_state")
    summary: dict[str, Any] = {
        "host": host_label,
        "task": task,
        "age": int(age),
        "classes": int(classes),
        "seeds": [int(item["seed"]) for item in per_seed],
        "conditions": {},
        "extra": extra,
    }
    for condition in conditions:
        bacc = [item["conditions"][condition]["balanced_accuracy"]
                for item in per_seed]
        acc = [item["conditions"][condition]["accuracy"]
               for item in per_seed]
        condition_summary = {
            "balanced_accuracy_mean": float(np.mean(bacc)),
            "balanced_accuracy_values": [float(value) for value in bacc],
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_values": [float(value) for value in acc],
        }
        if execution:
            success = [
                item["conditions"][condition]["execution"]["executed_success"]
                for item in per_seed
                if "execution" in item["conditions"][condition]
            ]
            regret = [
                item["conditions"][condition]["execution"]["mean_pose_regret"]
                for item in per_seed
                if "execution" in item["conditions"][condition]
            ]
            if success:
                condition_summary["execution"] = {
                    "executed_success_mean": float(np.mean(success)),
                    "executed_success_values": [float(value) for value in success],
                    "mean_pose_regret_mean": float(np.mean(regret)),
                    "mean_pose_regret_values": [float(value) for value in regret],
                }
        summary["conditions"][condition] = condition_summary
    full = summary["conditions"]["full"]["balanced_accuracy_mean"]
    reset = summary["conditions"]["reset"]["balanced_accuracy_mean"]
    none = summary["conditions"]["no_state"]["balanced_accuracy_mean"]
    summary["contrasts"] = {
        "full_minus_reset_bacc": float(full - reset),
        "full_minus_no_state_bacc": float(full - none),
    }
    summary["gate"] = {
        "full_minimum": 0.75,
        "control_maximum": 1.0 / float(classes) + 0.05,
        "passed": bool(
            full >= 0.75
            and reset <= 1.0 / float(classes) + 0.05
            and none <= 1.0 / float(classes) + 0.05
        ),
    }
    return summary


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Checkpointed PushT downstream-use gate",
        "",
        f"Status: `{summary['status']}`",
        f"Task: `{summary['task']}`, age `{summary['age']}`.",
        "",
        "| Host | Full BAcc | Reset BAcc | No-state BAcc | Gate | Execution |",
        "|---|---:|---:|---:|---|---|",
    ]
    for key, host in summary["hosts"].items():
        full = host["conditions"]["full"]["balanced_accuracy_mean"]
        reset = host["conditions"]["reset"]["balanced_accuracy_mean"]
        none = host["conditions"]["no_state"]["balanced_accuracy_mean"]
        execution = host["conditions"]["full"].get("execution")
        exec_text = "not reported"
        if execution is not None:
            exec_text = f"{execution['executed_success_mean']:.3f} success"
        lines.append(
            f"| {host['host']} | {full:.3f} | {reset:.3f} | {none:.3f} | "
            f"{'pass' if host['gate']['passed'] else 'fail'} | {exec_text} |"
        )
    lines.extend(["", "Notes:"])
    lines.append("- DINO-WM and LeWM use the same consumer protocol, but their PushT banks are not row-matched.")
    lines.append("- Physical execution is reported only when the validation rows match a host-specific simulator deck.")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dino-root", default=str(DEFAULT_DINO_ROOT))
    parser.add_argument("--lewm-root", default=str(DEFAULT_LEWM_ROOT))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--age", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hosts", nargs="*", default=["dinowm", "lewm"],
                        choices=["dinowm", "lewm"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    started = time.time()
    hosts: dict[str, Any] = {}
    if "dinowm" in args.hosts:
        hosts["dinowm"] = dino_host_result(
            Path(args.dino_root), args.task, args.age, args.seeds,
            device, int(args.batch_size))
    if "lewm" in args.hosts:
        hosts["lewm"] = lewm_host_result(
            Path(args.lewm_root), args.task, args.age, args.seeds,
            device, int(args.batch_size))
    summary = {
        "schema": "pusht_checkpointed_downstream_use_v1",
        "status": "completed",
        "task": args.task,
        "age": int(args.age),
        "seeds": [int(seed) for seed in args.seeds],
        "hosts": hosts,
        "all_goal_selection_gates_passed": bool(
            hosts and all(host["gate"]["passed"] for host in hosts.values())
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    (output / "summary.json").write_text(stable_json(summary))
    write_markdown(summary, output / "summary.md")
    print(stable_json({
        "status": summary["status"],
        "all_goal_selection_gates_passed": summary["all_goal_selection_gates_passed"],
        "summary": display_path(output / "summary.json"),
    }))


if __name__ == "__main__":
    main()
