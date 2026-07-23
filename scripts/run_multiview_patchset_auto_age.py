#!/usr/bin/env python3
"""Mixed-age temporal-coverage slot-memory JEPA for OGBench renders."""

from __future__ import annotations

import argparse
import json
import math
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
CONTINUOUS_OUTPUT = ROOT / "outputs" / "paper_c_continuous_v1"


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def max_legal_age() -> int:
    """Largest cue delay whose readout endpoint still fits inside the cache.

    The readout endpoint is ``LAST_CUE_FRAME + age`` and must be a valid frame
    index (``< LENGTH``), so the largest legal delay is ``LENGTH - LAST_CUE_FRAME
    - 1``.  With the 140-frame long-horizon caches this is 136.
    """
    return int(base.LENGTH - base.LAST_CUE_FRAME - 1)


def split_indices(n: int, seed: int, validation_fraction: float, split: str) -> np.ndarray:
    """Deterministic train/val episode split shared by every dataset variant."""
    rng = np.random.default_rng(97_101 + int(seed))
    order = rng.permutation(int(n))
    val_count = max(base.CLASSES, int(round(int(n) * float(validation_fraction))))
    return order[:-val_count] if split == "train" else order[-val_count:]


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
    # Continuous-delay mode: instead of a fixed grid of ages, sample a cue delay
    # per episode from a log-uniform distribution over [min-age, max-age].
    parser.add_argument(
        "--continuous-eval",
        action="store_true",
        help="evaluate held-out episodes at per-episode delays drawn log-uniformly "
        "over [continuous-min-age, continuous-max-age] and write continuous_eval.json",
    )
    parser.add_argument(
        "--continuous-train",
        action="store_true",
        help="sample the training cue delay per episode log-uniformly over the same "
        "range instead of the fixed --train-ages grid",
    )
    parser.add_argument("--continuous-min-age", type=float, default=4.0)
    parser.add_argument(
        "--continuous-max-age",
        type=float,
        default=0.0,
        help="upper delay bound (0 = auto = LENGTH - LAST_CUE_FRAME - 1)",
    )
    parser.add_argument(
        "--continuous-draws",
        type=int,
        default=500,
        help="number of (episode, delay) evaluation draws",
    )
    parser.add_argument(
        "--continuous-train-reps",
        type=int,
        default=3,
        help="log-uniform delay draws per training episode when --continuous-train",
    )
    parser.add_argument("--continuous-output", type=Path, default=CONTINUOUS_OUTPUT)
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
        ages: Iterable[int] = (),
        split: str = "val",
        seed: int = 0,
        validation_fraction: float = 0.20,
        variant: str = "full",
        augment: bool = False,
        temporal_drop: float = 0.0,
        patch_drop: float = 0.0,
        max_context: int,
        explicit_items: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        with np.load(archive, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.actions = data["actions"]
            self.labels = data["cue_labels"]
            self.positions = data["cue_positions"]
        self.variant = str(variant)
        self.augment = bool(augment)
        self.temporal_drop = float(temporal_drop)
        self.patch_drop = float(patch_drop)
        self.max_context = int(max_context)
        if explicit_items is not None:
            # Per-episode delays supplied directly (continuous-delay mode).
            self.items = [(int(index), int(age)) for index, age in explicit_items]
            self.ages = sorted({age for _, age in self.items})
        else:
            self.ages = [int(value) for value in ages]
            indices = split_indices(len(self.frames), seed, validation_fraction, split)
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


def continuous_range(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve the log-uniform delay range, clamped to what the cache supports."""
    hi = float(args.continuous_max_age) if float(args.continuous_max_age) > 0 else float(max_legal_age())
    hi = min(hi, float(max_legal_age()))
    lo = max(1.0, float(args.continuous_min_age))
    if lo >= hi:
        raise ValueError(f"continuous-min-age {lo} must be < max age {hi}")
    return lo, hi


def _sample_log_uniform_age(rng: np.random.Generator, lo: float, hi: float) -> int:
    age = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    return max(int(round(lo)), min(int(round(hi)), age))


def build_continuous_items(
    n_episodes: int,
    args: argparse.Namespace,
    *,
    split: str,
    draws: int,
    lo: float,
    hi: float,
    seed_offset: int,
) -> list[tuple[int, int]]:
    """List of (episode, delay) pairs with delays drawn log-uniformly per pair.

    Episodes cycle through the requested split so coverage is balanced; each draw
    receives an independent continuous delay, producing a dense (non-grid) sweep.
    """
    idx = split_indices(n_episodes, int(args.seed), float(args.validation_fraction), split)
    idx = np.asarray(idx)
    rng = np.random.default_rng(20_260_720 + int(args.seed) + int(seed_offset))
    items: list[tuple[int, int]] = []
    for k in range(int(draws)):
        episode = int(idx[k % len(idx)])
        items.append((episode, _sample_log_uniform_age(rng, lo, hi)))
    return items


def evaluate_continuous(
    model,
    readout,
    args: argparse.Namespace,
    device: torch.device,
    items: list[tuple[int, int]],
    max_context: int,
) -> list[dict[str, Any]]:
    """Per-episode correctness at each drawn delay under matched conditions."""
    payloads = {}
    for variant in ["full", "reset", "no_state"]:
        dataset = AutoAgePatchSetDataset(
            cache_path(args),
            split="val",
            seed=int(args.seed),
            validation_fraction=float(args.validation_fraction),
            variant=variant,
            augment=False,
            max_context=int(max_context),
            explicit_items=items,
        )
        payloads[variant] = memory_base.extract(model, make_loader(dataset, args, shuffle=False), device)
    preds = {name: readout.predict(payload["memory"]).astype(np.int64) for name, payload in payloads.items()}
    labels = payloads["full"]["labels"].astype(np.int64)
    rows: list[dict[str, Any]] = []
    for i, (episode, age) in enumerate(items):
        rows.append(
            {
                "alpha": float(age),
                "episode": int(episode),
                "label": int(labels[i]),
                "full_correct": int(preds["full"][i] == labels[i]),
                "reset_correct": int(preds["reset"][i] == labels[i]),
                "no_state_correct": int(preds["no_state"][i] == labels[i]),
            }
        )
    return rows


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
    args.continuous_output = (
        args.continuous_output if args.continuous_output.is_absolute() else ROOT / args.continuous_output
    )
    continuous = bool(args.continuous_eval or args.continuous_train)
    if continuous:
        lo, hi = continuous_range(args)
    else:
        validate_ages(args)
    if not cache_path(args).is_file():
        raise FileNotFoundError(cache_path(args))
    out_dir = result_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(911_203 + int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with np.load(cache_path(args), allow_pickle=False) as data:
        img_size = int(data["img_size"])
        action_dim = int(data["actions"].shape[-1])
        n_episodes = int(data["frames"].shape[0])

    if args.continuous_train:
        cont_context = base.LAST_CUE_FRAME + int(round(hi)) + 1
        train_items = build_continuous_items(
            n_episodes, args, split="train", draws=len(
                split_indices(n_episodes, int(args.seed), float(args.validation_fraction), "train")
            ) * max(1, int(args.continuous_train_reps)), lo=lo, hi=hi, seed_offset=1,
        )
        train_aug = AutoAgePatchSetDataset(
            cache_path(args), variant="full", augment=True,
            temporal_drop=float(args.temporal_drop), patch_drop=float(args.patch_drop),
            max_context=cont_context, explicit_items=train_items,
        )
        train_eval = AutoAgePatchSetDataset(
            cache_path(args), variant="full", augment=False,
            max_context=cont_context, explicit_items=train_items,
        )
    else:
        train_aug = build_dataset(args, ages=list(args.train_ages), split="train", variant="full", augment=True)
        train_eval = build_dataset(args, ages=list(args.train_ages), split="train", variant="full", augment=False)
    train_loader = make_loader(train_aug, args, shuffle=True)
    train_eval_loader = make_loader(train_eval, args, shuffle=False)

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

    if args.continuous_eval:
        eval_context = base.LAST_CUE_FRAME + int(round(hi)) + 1
        eval_items = build_continuous_items(
            n_episodes, args, split="val", draws=int(args.continuous_draws), lo=lo, hi=hi, seed_offset=2,
        )
        rows = evaluate_continuous(model, readout, args, device, eval_items, eval_context)
        cont_result = {
            "schema": "multiview_patchset_continuous_eval_v1",
            "status": "completed",
            "env_name": str(args.env_name),
            "seed": int(args.seed),
            "continuous_train": bool(args.continuous_train),
            "train_ages": None if args.continuous_train else [int(v) for v in args.train_ages],
            "delay_distribution": "log-uniform",
            "min_alpha": float(lo),
            "max_alpha": float(hi),
            "max_legal_alpha": int(max_legal_age()),
            "length": int(base.LENGTH),
            "last_cue_frame": int(base.LAST_CUE_FRAME),
            "draws": len(rows),
            "epochs": int(args.epochs),
            "dim": int(args.dim),
            "slots": int(args.slots),
            "heads": int(args.heads),
            "training_loss_uses_cue_labels": False,
            "posthoc_readout_uses_cue_labels": True,
            "n_full_correct": int(sum(r["full_correct"] for r in rows)),
            "n_reset_correct": int(sum(r["reset_correct"] for r in rows)),
            "n_no_state_correct": int(sum(r["no_state_correct"] for r in rows)),
            "rows": rows,
        }
        cont_dir = args.continuous_output / base.env_key(args.env_name)
        cont_dir.mkdir(parents=True, exist_ok=True)
        (cont_dir / "continuous_eval.json").write_text(stable_json(cont_result))
        torch.save(
            {"model": model.state_dict(), "args": vars(args)}, cont_dir / "model.pt"
        )
        print(
            stable_json(
                {
                    "env": args.env_name,
                    "seed": int(args.seed),
                    "status": "completed",
                    "draws": len(rows),
                    "alpha_range": [float(lo), float(hi)],
                    "acc_full": round(cont_result["n_full_correct"] / max(1, len(rows)), 4),
                    "acc_reset": round(cont_result["n_reset_correct"] / max(1, len(rows)), 4),
                }
            ),
            flush=True,
        )
        return cont_result

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
