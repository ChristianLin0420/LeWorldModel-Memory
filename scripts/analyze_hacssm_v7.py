#!/usr/bin/env python3
"""Fail-closed aggregation for the HACSSM/HCRD-v7 all-environment study."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_hacssm_v5 as shared


OCC_TO_CLEAN = shared.OCC_TO_CLEAN
DESIGNS = (
    'ssm',
    'hacsmv4_two_noaux',
    'hacssmv6',
    'hacssmv6_static',
    'hacssmv7_noaux',
    'hacssmv7_sharedaction',
    'hacssmv7_noshrink',
    'hacssmv7_actiononly',
    'hacssmv7_uniform',
    'hacssmv7_norecovery',
    'hacssmv7_noaction',
    'hacssmv7_single',
    'hacssmv7',
)
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
V7_DESIGNS = frozenset(d for d in DESIGNS if d.startswith('hacssmv7'))
HIER_DESIGNS = frozenset(
    d for d in DESIGNS if d.startswith(('hacsmv4', 'hacssmv6', 'hacssmv7')))
PRIMARY = 'clean_mse_first_post'
EPOCHS = 200
WANDB_ENTITY = 'crlc112358'
WANDB_PROJECT = 'lewm-memory-popgym'
WANDB_MODE = 'online'
WANDB_STUDY = 'hacssm-v7'
CANDIDATE = 'hacssmv7'


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    if design in V7_DESIGNS:
        return 0.02, 'v6_bootstrap', design != 'hacssmv7_noaux'
    if design in {'hacssmv6', 'hacssmv6_static'}:
        return 0.02, 'v6_bootstrap', True
    if design == 'hacsmv4_two_noaux':
        return 0.1, 'fixed', False
    if design == 'ssm':
        return 0.0, 'fixed', False
    raise ValueError(f'unknown V7 design {design!r}')


def scheduled_weight(base: float, schedule: str, epoch: int) -> float:
    if epoch < 1:
        raise ValueError(f'epoch must be positive, got {epoch}')
    if schedule == 'fixed':
        return float(base)
    if schedule == 'v6_bootstrap':
        if epoch <= 40:
            return float(base)
        if epoch <= 100:
            return float(base) * 0.5 * (1.0 + math.cos(math.pi * (epoch - 40) / 60.0))
        return 0.0
    raise ValueError(f'unknown hierarchy schedule {schedule!r}')


def validate_history(history: Any, design: str, run_dir: Path) -> None:
    """Validate the epoch record independently of the execution harness.

    The generic V5 loader delegates history validation through a module-level hook.  V7
    strengthens that hook because zero overlap with the original hidden interval is part of
    the scientific leakage contract, not merely a runner diagnostic.  Re-analysis therefore
    fails closed even when it is invoked directly rather than through ``run_hacssm_v7.py``.
    """
    if not isinstance(history, list) or len(history) != EPOCHS:
        length = len(history) if isinstance(history, list) else None
        raise ValueError(f'{run_dir}: history length {length}, expected {EPOCHS}')
    base, schedule, active = design_aux_contract(design)
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get('epoch') != epoch:
            raise ValueError(f'{run_dir}: malformed epoch {epoch}')
        if set(record) != {'epoch', 'train', 'val'}:
            raise ValueError(f'{run_dir}: unexpected history fields at epoch {epoch}')
        for split in ('train', 'val'):
            values = record.get(split)
            if not isinstance(values, dict):
                raise ValueError(f'{run_dir}: missing {split} at epoch {epoch}')
            for metric in ('loss', 'pred_loss', 'sigreg_loss'):
                shared.finite(values.get(metric), f'{run_dir}:{epoch}:{split}.{metric}')
            shared.finite_tree(values, f'{run_dir}:{epoch}:{split}')
            if design in HIER_DESIGNS:
                for metric in ('hier_loss', 'hier_loss_fast', 'hier_loss_medium'):
                    shared.finite(values.get(metric), f'{run_dir}:{epoch}:{split}.{metric}')
                observed = shared.finite(
                    values.get('hier_loss_weight'),
                    f'{run_dir}:{epoch}:{split}.hier_loss_weight')
                wanted = scheduled_weight(base, schedule, epoch) if active else 0.0
                if not math.isclose(observed, wanted, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(
                        f'{run_dir}:{epoch}:{split} auxiliary weight {observed}, '
                        f'expected {wanted}')
            if design in V7_DESIGNS:
                for metric in ('hier_loss_bridge', 'hier_loss_recovery', 'hier_overlap'):
                    shared.finite(values.get(metric), f'{run_dir}:{epoch}:{split}.{metric}')
                if float(values['hier_overlap']) != 0.0:
                    raise ValueError(
                        f'{run_dir}:{epoch}:{split} V7 counterfactual windows overlap '
                        f'original hidden targets: {values["hier_overlap"]}')


def configure_shared() -> None:
    shared.DESIGNS = DESIGNS
    shared.PILOT_SEEDS = PILOT_SEEDS
    shared.FINAL_SEEDS = FINAL_SEEDS
    shared.V5_DESIGNS = V7_DESIGNS
    shared.HIER_DESIGNS = HIER_DESIGNS
    shared.NO_AUX_DESIGNS = frozenset({'hacsmv4_two_noaux', 'hacssmv7_noaux'})
    shared.PRIMARY = PRIMARY
    shared.EPOCHS = EPOCHS
    shared.WANDB_ENTITY = WANDB_ENTITY
    shared.WANDB_PROJECT = WANDB_PROJECT
    shared.WANDB_MODE = WANDB_MODE
    shared.WANDB_STUDY = WANDB_STUDY
    shared.design_aux_contract = design_aux_contract
    shared.scheduled_weight = scheduled_weight
    shared.validate_history = validate_history
    extras = (
        'val_hier_loss_bridge', 'val_hier_loss_recovery', 'val_hier_overlap',
        'rho_fast', 'rho_medium', 'action_head_fast_norm',
        'action_head_medium_norm', 'action_head_cosine',
    )
    shared.METRIC_FIELDS = tuple(dict.fromkeys((*shared.METRIC_FIELDS, *extras)))


def environment_wins(rows: Sequence[Mapping[str, Any]], reference: str) -> int:
    candidate = shared.environment_means(rows, CANDIDATE)
    baseline = shared.environment_means(rows, reference)
    return sum(candidate[env] < baseline[env] for env in OCC_TO_CLEAN)


def contrast_map(contrasts) -> dict[str, Mapping[str, Any]]:
    return {
        design: shared.overall_contrast(contrasts, design)
        for design in DESIGNS if design != CANDIDATE
    }


def pilot_decision(rows, convergence, contrasts) -> dict[str, Any]:
    compared = contrast_map(contrasts)
    env_wins = {d: environment_wins(rows, d) for d in compared}
    absolute = np.abs(np.asarray(
        [float(row['relative_improvement']) for row in convergence], dtype=np.float64))
    criteria = {
        'vs_ssm_reduction_ge_5pct':
            float(compared['ssm']['mean_paired_relative_reduction']) >= 0.05,
        'vs_ssm_wins_ge_10_of_15': int(compared['ssm']['paired_wins']) >= 10,
        'vs_ssm_env_wins_ge_3_of_5': env_wins['ssm'] >= 3,
        'vs_v4_two_reduction_ge_1pct':
            float(compared['hacsmv4_two_noaux']['mean_paired_relative_reduction']) >= 0.01,
        'vs_v4_two_wins_ge_9_of_15':
            int(compared['hacsmv4_two_noaux']['paired_wins']) >= 9,
        'vs_v4_two_env_wins_ge_3_of_5': env_wins['hacsmv4_two_noaux'] >= 3,
        'vs_v6_reduction_ge_0_5pct':
            float(compared['hacssmv6']['mean_paired_relative_reduction']) >= 0.005,
        'vs_v6_wins_ge_9_of_15': int(compared['hacssmv6']['paired_wins']) >= 9,
        'vs_v6_env_wins_ge_3_of_5': env_wins['hacssmv6'] >= 3,
        'vs_v6_static_positive':
            float(compared['hacssmv6_static']['mean_paired_relative_reduction']) > 0.0,
        'vs_v6_static_wins_ge_8_of_15':
            int(compared['hacssmv6_static']['paired_wins']) >= 8,
        'vs_noaux_reduction_ge_1pct':
            float(compared['hacssmv7_noaux']['mean_paired_relative_reduction']) >= 0.01,
        'vs_noaux_wins_ge_9_of_15': int(compared['hacssmv7_noaux']['paired_wins']) >= 9,
        'vs_noaux_env_wins_ge_3_of_5': env_wins['hacssmv7_noaux'] >= 3,
    }
    for reference in (
        'hacssmv7_sharedaction', 'hacssmv7_noshrink', 'hacssmv7_actiononly',
        'hacssmv7_uniform', 'hacssmv7_norecovery', 'hacssmv7_noaction',
        'hacssmv7_single',
    ):
        label = reference.removeprefix('hacssmv7_')
        criteria[f'vs_{label}_positive'] = (
            float(compared[reference]['mean_paired_relative_reduction']) > 0.0)
        criteria[f'vs_{label}_wins_ge_8_of_15'] = int(compared[reference]['paired_wins']) >= 8
    criteria.update({
        'convergence_absolute_median_lt_1pct': float(np.median(absolute)) < 0.01,
        'convergence_absolute_p95_lt_3pct': float(np.quantile(absolute, 0.95)) < 0.03,
        'convergence_absolute_max_lt_5pct': float(absolute.max()) < 0.05,
    })
    passed = all(criteria.values())
    return {
        'schema_version': 1,
        'phase': 'pilot',
        'decision': 'PILOT_PASS' if passed else 'NO_GO',
        'pilot_screen_passed': passed,
        'criteria': criteria,
        'observed': {
            'overall_contrasts': compared,
            'environment_mean_wins': env_wins,
            'convergence_absolute_median': float(np.median(absolute)),
            'convergence_absolute_p95': float(np.quantile(absolute, 0.95)),
            'convergence_absolute_max': float(absolute.max()),
        },
        'note': 'Immutable V7 pilot; all five seeds run regardless of this screen.',
    }


def final_summary(rows, convergence, contrasts, *, pilot_screen_passed: bool) -> dict[str, Any]:
    compared = contrast_map(contrasts)
    env_wins = {d: environment_wins(rows, d) for d in compared}
    candidate_means = shared.environment_means(rows, CANDIDATE)
    design_means = {d: shared.environment_means(rows, d) for d in DESIGNS}
    hold = shared.environment_means(rows, CANDIDATE, 'last_visible_mse_first_post')
    envelope = sum(
        candidate_means[env] < min(
            design_means[d][env] for d in DESIGNS if d != CANDIDATE)
        for env in OCC_TO_CLEAN)
    hold_wins = sum(candidate_means[env] < hold[env] for env in OCC_TO_CLEAN)
    absolute = np.abs(np.asarray(
        [float(row['relative_improvement']) for row in convergence], dtype=np.float64))
    criteria = {
        'vs_ssm_reduction_ge_6pct':
            float(compared['ssm']['mean_paired_relative_reduction']) >= 0.06,
        'vs_ssm_wins_ge_20_of_25': int(compared['ssm']['paired_wins']) >= 20,
        'vs_ssm_env_wins_ge_4_of_5': env_wins['ssm'] >= 4,
        'vs_v4_two_reduction_ge_1_5pct':
            float(compared['hacsmv4_two_noaux']['mean_paired_relative_reduction']) >= 0.015,
        'vs_v4_two_wins_ge_17_of_25':
            int(compared['hacsmv4_two_noaux']['paired_wins']) >= 17,
        'vs_v4_two_env_wins_ge_4_of_5': env_wins['hacsmv4_two_noaux'] >= 4,
        'vs_v6_reduction_ge_1pct':
            float(compared['hacssmv6']['mean_paired_relative_reduction']) >= 0.01,
        'vs_v6_wins_ge_15_of_25': int(compared['hacssmv6']['paired_wins']) >= 15,
        'vs_v6_env_wins_ge_3_of_5': env_wins['hacssmv6'] >= 3,
        'vs_v6_static_positive':
            float(compared['hacssmv6_static']['mean_paired_relative_reduction']) > 0.0,
        'vs_v6_static_wins_ge_13_of_25':
            int(compared['hacssmv6_static']['paired_wins']) >= 13,
        'vs_v6_static_env_wins_ge_3_of_5': env_wins['hacssmv6_static'] >= 3,
        'vs_noaux_reduction_ge_1pct':
            float(compared['hacssmv7_noaux']['mean_paired_relative_reduction']) >= 0.01,
        'vs_noaux_wins_ge_15_of_25': int(compared['hacssmv7_noaux']['paired_wins']) >= 15,
        'vs_noaux_env_wins_ge_3_of_5': env_wins['hacssmv7_noaux'] >= 3,
        'locked_grid_envelope_wins_ge_3_of_5': envelope >= 3,
        'beats_hold_ge_4_of_5': hold_wins >= 4,
        'convergence_absolute_median_lt_1pct': float(np.median(absolute)) < 0.01,
        'convergence_absolute_p95_lt_3pct': float(np.quantile(absolute, 0.95)) < 0.03,
        'convergence_absolute_max_lt_5pct': float(absolute.max()) < 0.05,
    }
    for reference in (
        'hacssmv7_sharedaction', 'hacssmv7_noshrink', 'hacssmv7_actiononly',
        'hacssmv7_uniform', 'hacssmv7_norecovery',
    ):
        label = reference.removeprefix('hacssmv7_')
        criteria[f'vs_{label}_positive'] = (
            float(compared[reference]['mean_paired_relative_reduction']) > 0.0)
        criteria[f'vs_{label}_wins_ge_13_of_25'] = int(compared[reference]['paired_wins']) >= 13
        criteria[f'vs_{label}_env_wins_ge_3_of_5'] = env_wins[reference] >= 3
    for reference in ('hacssmv7_noaction', 'hacssmv7_single'):
        label = reference.removeprefix('hacssmv7_')
        criteria[f'vs_{label}_reduction_ge_3pct'] = (
            float(compared[reference]['mean_paired_relative_reduction']) >= 0.03)
        criteria[f'vs_{label}_wins_ge_17_of_25'] = int(compared[reference]['paired_wins']) >= 17
        criteria[f'vs_{label}_env_wins_ge_3_of_5'] = env_wins[reference] >= 3

    good_enough = bool(pilot_screen_passed and all(criteria.values()))
    primary_positive = all(
        float(compared[d]['mean_paired_relative_reduction']) > 0.0
        for d in ('ssm', 'hacsmv4_two_noaux', 'hacssmv6', 'hacssmv6_static',
                  'hacssmv7_noaux'))
    if not pilot_screen_passed:
        decision = 'PILOT_NO_GO_FINAL_DESCRIPTIVE'
    elif good_enough:
        decision = 'OVERALL_BEST_IN_LOCKED_GRID'
    elif primary_positive:
        decision = 'PROMISING_NOT_OVERALL_BEST'
    else:
        decision = 'NO_GO'
    return {
        'schema_version': 1,
        'phase': 'final',
        'decision': decision,
        'pilot_screen_passed': pilot_screen_passed,
        'good_enough_for_overall_best_claim': good_enough,
        'criteria': criteria,
        'completed_runs': len(rows),
        'observed': {
            'overall_contrasts': compared,
            'environment_mean_wins': env_wins,
            'locked_grid_envelope_env_wins': envelope,
            'hold_environment_wins': hold_wins,
            'convergence_absolute_median': float(np.median(absolute)),
            'convergence_absolute_p95': float(np.quantile(absolute, 0.95)),
            'convergence_absolute_max': float(absolute.max()),
        },
        'limitations': [
            'The exact black corruption and seed-7777 trajectories remain adaptive development data.',
            'Counterfactual masks use only originally visible frames but the same black token family.',
            'No simulator-state outcome or executed-control return is measured.',
        ],
        'note': (
            'Even OVERALL_BEST_IN_LOCKED_GRID is not an untouched-test or ICLR claim; '
            'new corruptions, rollout seeds, state/return outcomes, and tuned baselines remain required.'),
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/hacssm_v7_shared'))
    parser.add_argument('--phase', choices=('pilot', 'final'), required=True)
    return parser.parse_args(argv)


def strict_validate_cells(root: Path, seeds: Sequence[int]) -> None:
    """Apply the execution harness's complete local-artifact contract before aggregation.

    The inherited table loader intentionally focuses on statistical aggregation.  It is not
    sufficient as a provenance boundary by itself, so direct analyzer invocations first reuse
    the V7 runner's exact checkpoint-key, argument, feature/hash, rollout, W&B-receipt, model
    state, metric, and history checks.  ``root`` remains configurable for an independently run
    namespace; the checkpoint's own ``output_dir`` must match it.
    """
    import scripts.run_hacssm_v7 as runner

    original_root = runner.OUTPUT_ROOT
    try:
        runner.OUTPUT_ROOT = root.resolve()
        runner.configure_shared()
        jobs = tuple(
            runner.shared.Job(
                'pilot' if seed in PILOT_SEEDS else 'completion',
                seed, env, OCC_TO_CLEAN[env], design)
            for seed in seeds
            for env in OCC_TO_CLEAN
            for design in DESIGNS
        )
        for job in jobs:
            runner.shared.validate_job(job, allow_missing=False)
    finally:
        runner.OUTPUT_ROOT = original_root


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = PILOT_SEEDS if args.phase == 'pilot' else FINAL_SEEDS
    strict_validate_cells(args.root, seeds)
    configure_shared()
    expected = len(OCC_TO_CLEAN) * len(DESIGNS) * len(seeds)
    rows, convergence = shared.load_cells(args.root, seeds)
    if len(rows) != expected:
        raise ValueError(f'{args.phase} grid has {len(rows)} rows, expected {expected}')
    grouped = shared.grouped_rows(rows)
    contrasts = shared.contrast_rows(rows, candidate=CANDIDATE)
    prefix = 'pilot_' if args.phase == 'pilot' else ''
    if args.phase == 'pilot':
        decision = pilot_decision(rows, convergence, contrasts)
    else:
        pilot_path = args.root / 'pilot_decision.json'
        pilot = shared.read_json(pilot_path)
        pilot_rows = [row for row in rows if int(row['seed']) in PILOT_SEEDS]
        pilot_convergence = [
            row for row in convergence if int(row['seed']) in PILOT_SEEDS]
        recomputed_pilot = pilot_decision(
            pilot_rows, pilot_convergence,
            shared.contrast_rows(pilot_rows, candidate=CANDIDATE))
        if pilot != recomputed_pilot:
            raise ValueError(f'invalid immutable pilot decision: {pilot_path}')
        decision = final_summary(
            rows, convergence, contrasts,
            pilot_screen_passed=recomputed_pilot['pilot_screen_passed'])
    shared.atomic_csv(args.root / f'{prefix}per_run.csv', rows)
    shared.atomic_csv(args.root / f'{prefix}grouped.csv', grouped)
    shared.atomic_csv(args.root / f'{prefix}paired_contrasts.csv', contrasts)
    shared.atomic_csv(args.root / f'{prefix}convergence.csv', convergence)
    shared.atomic_json(
        args.root / ('pilot_decision.json' if args.phase == 'pilot' else 'decision.json'),
        decision)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
