#!/usr/bin/env python3
"""Mixed-age temporal-coverage slot-memory JEPA for OGBench renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402
from scripts import run_multiview_patchset_color_jepa_ogbench as multiview  # noqa: E402
from scripts import run_multiview_patchset_memory_baseline as memory_base  # noqa: E402
from scripts import run_random_patchset_jepa_ogbench as patchset  # noqa: E402
from scripts import run_random_patchset_view_jepa_ogbench as view  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "multiview_patchset_auto_age_v1"
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
DEFAULT_TRAIN_AGES = (4, 8, 15)
DEFAULT_EVAL_AGES = (4, 6, 8, 10, 12, 15, 18)


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--env-name", default="pointmaze-large-navigate-v0")
    parser.add_argument("--train-ages", type=int, nargs="*", default=list(DEFAULT_TRAIN_AGES))
    parser.add_argument("--eval-ages", type=int, nargs="*", default=list(DEFAULT_EVAL_AGES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=160)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--cos-weight", type=float, default=0.35)
    parser.add_argument("--std-weight", type=float, default=0.05)
    parser.add_argument("--temporal-drop", type=float, default=0.12)
    parser.add_argument("--patch-drop", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def cache_path(args: argparse.Namespace) -> Path:
    root = Path(args.cache_root)
    if not root.is_absolute():
        root = ROOT / root
    return root / "cache" / base.env_key(args.env_name) / "render_cache.npz"


def result_dir(args: argparse.Namespace) -> Path:
    return Path(args.output) / base.env_key(args.env_name) / "auto_age" / f"s{int(args.seed)}"


def validate_ages(args: argparse.Namespace) -> None:
    max_age = max([int(value) for value in args.train_ages + args.eval_ages])
    if base.LAST_CUE_FRAME + max_age >= base.LENGTH:
        raise ValueError(
            f"max age {max_age} exceeds cache length {base.LENGTH}; "
            f"largest supported age is {base.LENGTH - base.LAST_CUE_FRAME - 1}"
        )


class AutoAgePatchSetDataset(Dataset):
    """Balanced mixture over several evidence ages."""

    def __init__(
        self,
        archive: Path,
        *,
        ages: Iterable[int],
        split: str,
        seed: int,
        validation_fraction: float,
        variant: str = "full",
        augment: bool = False,
        temporal_drop: float = 0.0,
        patch_drop: float = 0.0,
        max_context: int,
    ) -> None:
        with np.load(archive, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.actions = data["actions"]
            self.labels = data["cue_labels"]
            self.positions = data["cue_positions"]
        self.ages = [int(value) for value in ages]
        self.variant = str(variant)
        self.augment = bool(augment)
        self.temporal_drop = float(temporal_drop)
        self.patch_drop = float(patch_drop)
        self.max_context = int(max_context)
        rng = np.random.default_rng(97_101 + int(seed))
        order = rng.permutation(len(self.frames))
        val_count = max(base.CLASSES, int(round(len(order) * float(validation_fraction))))
        indices = order[:-val_count] if split == "train" else order[-val_count:]
        self.items = [(int(index), int(age)) for age in self.ages for index in indices]

    def __len__(self) -> int:
        return len(self.items)

    def _valid_times(self, endpoint: int, rng: np.random.Generator) -> list[int]:
        if self.variant == "no_state":
            return list(range(max(0, endpoint - 3), endpoint + 1))
        times = list(range(0, endpoint + 1))
        if self.augment and self.temporal_drop > 0:
            times = [time for time in times if time == endpoint or rng.random() > self.temporal_drop]
            if not times:
                times = [endpoint]
        return sorted(set(times))

    def _maybe_mask_random_patch(self, frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.augment or rng.random() >= self.patch_drop:
            return frame
        out = frame.copy()
        size = max(6, frame.shape[0] // 5)
        x = int(rng.integers(0, max(1, frame.shape[1] - size)))
        y = int(rng.integers(0, max(1, frame.shape[0] - size)))
        patchset._mask_patch(out, y, x, size)
        return out

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode, age = self.items[item]
        endpoint = base.LAST_CUE_FRAME + int(age)
        label = int(self.labels[episode])
        position = int(self.positions[episode])
        rng = np.random.default_rng(70_000_031 + episode + 541 * int(age))
        clean = self.frames[episode].copy()
        full = base.inject_cue_sequence(clean, label, position)
        source = clean if self.variant in {"reset", "no_state"} else full
        target_times = multiview.choose_temporal_coverage_times(
            source,
            endpoint=endpoint,
            rng=rng,
            variant=self.variant,
            target_views=multiview.TARGET_VIEWS,
        )
        target_patches = []
        selected_all = []
        for target_time in target_times:
            patches, selected = view.mine_single_view_patches(source, target_time=target_time, rng=rng)
            target_patches.append(patches)
            selected_all.extend(selected)
        patches = np.concatenate(target_patches, axis=0).astype(np.uint8)

        visible = source.copy()
        if self.variant != "no_state":
            for time, y, x in selected_all:
                patchset._mask_patch(visible[int(time)], int(y), int(x), patchset.PATCH_SIZE)

        times = self._valid_times(endpoint, rng)
        frame_tokens = np.zeros(
            (self.max_context, clean.shape[-3], clean.shape[-2], clean.shape[-1]),
            dtype=np.uint8,
        )
        action_tokens = np.zeros((self.max_context, self.actions.shape[-1]), dtype=np.float32)
        time_tokens = np.zeros((self.max_context, 1), dtype=np.float32)
        valid = np.zeros((self.max_context,), dtype=np.float32)
        for slot, time in enumerate(times[: self.max_context]):
            frame_tokens[slot] = self._maybe_mask_random_patch(visible[time], rng)
            if time > 0:
                action_tokens[slot] = self.actions[episode, time - 1]
            time_tokens[slot, 0] = float(time) / float(base.LENGTH - 1)
            valid[slot] = 1.0
        return {
            "frames": torch.from_numpy(frame_tokens.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(action_tokens),
            "times": torch.from_numpy(time_tokens),
            "valid": torch.from_numpy(valid),
            "target_patches": torch.from_numpy(patches.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "label": torch.tensor(label, dtype=torch.long),
            "age": torch.tensor(int(age), dtype=torch.long),
        }


def make_loader(dataset: Dataset, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_dataset(args: argparse.Namespace, *, ages: list[int], split: str, variant: str, augment: bool) -> AutoAgePatchSetDataset:
    max_age = max([int(value) for value in args.train_ages + args.eval_ages])
    return AutoAgePatchSetDataset(
        cache_path(args),
        ages=ages,
        split=split,
        seed=int(args.seed),
        validation_fraction=float(args.validation_fraction),
        variant=variant,
        augment=augment,
        temporal_drop=float(args.temporal_drop),
        patch_drop=float(args.patch_drop),
        max_context=base.LAST_CUE_FRAME + int(max_age) + 1,
    )


def evaluate_age(model, readout, args: argparse.Namespace, device: torch.device, age: int) -> dict[str, Any]:
    payloads = {}
    for variant in ["full", "reset", "no_state"]:
        loader = make_loader(
            build_dataset(args, ages=[int(age)], split="val", variant=variant, augment=False),
            args,
            shuffle=False,
        )
        payloads[variant] = memory_base.extract(model, loader, device)
    readout_metrics = {
        name: base.readout_metric(readout, payload["memory"], payload["labels"])
        for name, payload in payloads.items()
    }
    retrieval_metrics = {
        name: patchset.set_retrieval(
            torch.from_numpy(payload["pred_slots"]).to(device),
            torch.from_numpy(payload["target_set"]).to(device),
            args.temperature,
        )
        for name, payload in payloads.items()
    }
    full = readout_metrics["full"]["balanced_accuracy"]
    reset = readout_metrics["reset"]["balanced_accuracy"]
    no_state = readout_metrics["no_state"]["balanced_accuracy"]
    return {
        "age": int(age),
        "trained_age": int(age) in {int(value) for value in args.train_ages},
        "readout": readout_metrics,
        "retrieval": retrieval_metrics,
        "gate": {
            "full_minimum": 0.75,
            "control_maximum": 0.35,
            "pass": bool(full >= 0.75 and reset <= 0.35 and no_state <= 0.35),
        },
    }


def train_cell(args: argparse.Namespace) -> dict[str, Any]:
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    validate_ages(args)
    if not cache_path(args).is_file():
        raise FileNotFoundError(cache_path(args))
    out_dir = result_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(911_203 + int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_aug = build_dataset(args, ages=list(args.train_ages), split="train", variant="full", augment=True)
    train_eval = build_dataset(args, ages=list(args.train_ages), split="train", variant="full", augment=False)
    train_loader = make_loader(train_aug, args, shuffle=True)
    train_eval_loader = make_loader(train_eval, args, shuffle=False)

    with np.load(cache_path(args), allow_pickle=False) as data:
        img_size = int(data["img_size"])
        action_dim = int(data["actions"].shape[-1])
    model = multiview.MultiViewPatchSetJEPA(
        img_size=img_size,
        action_dim=action_dim,
        dim=int(args.dim),
        slots=int(args.slots),
        heads=int(args.heads),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = memory_base.run_epoch(model, train_loader, optimizer, device, args)
        history.append({"epoch": int(epoch), "train": train_metrics})
        if epoch == 1 or epoch % max(1, int(args.epochs) // 4) == 0:
            print(
                stable_json(
                    {
                        "env": args.env_name,
                        "seed": int(args.seed),
                        "epoch": int(epoch),
                        "train_loss": train_metrics["loss"],
                        "train_top1": train_metrics["retrieval_top1"],
                    }
                ).strip(),
                flush=True,
            )

    train_features = memory_base.extract(model, train_eval_loader, device)
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(train_features["memory"], train_features["labels"])
    eval_rows = [evaluate_age(model, readout, args, device, int(age)) for age in args.eval_ages]
    result = {
        "schema": "multiview_patchset_auto_age_cell_v1",
        "status": "completed",
        "env_name": str(args.env_name),
        "seed": int(args.seed),
        "train_ages": [int(value) for value in args.train_ages],
        "eval_ages": [int(value) for value in args.eval_ages],
        "epochs": int(args.epochs),
        "dim": int(args.dim),
        "slots": int(args.slots),
        "heads": int(args.heads),
        "target_views": int(multiview.TARGET_VIEWS),
        "target_patches": int(multiview.TARGET_VIEWS * patchset.TARGET_PATCHES),
        "training_loss_uses_cue_labels": False,
        "posthoc_readout_uses_cue_labels": True,
        "history": history,
        "eval": eval_rows,
        "registered_all_pass": bool(
            all(row["gate"]["pass"] for row in eval_rows if row["age"] in set(DEFAULT_TRAIN_AGES))
        ),
        "unseen_pass_count": int(
            sum(row["gate"]["pass"] for row in eval_rows if row["age"] not in set(args.train_ages))
        ),
        "claim_boundary": (
            "First mixed-age training run.  It evaluates age generalization, but does not yet "
            "implement a learned explicit age router or automatic maximum-age search."
        ),
    }
    (out_dir / "result.json").write_text(stable_json(result))
    np.savez_compressed(
        out_dir / "features.npz",
        train_memory=train_features["memory"],
        train_labels=train_features["labels"],
    )
    torch.save({"model": model.state_dict(), "args": vars(args), "result": result}, out_dir / "model.pt")
    print(stable_json({"env": args.env_name, "seed": int(args.seed), "status": "completed"}), flush=True)
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train_cell(args)


if __name__ == "__main__":
    main()
