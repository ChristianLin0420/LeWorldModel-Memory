#!/usr/bin/env python3
"""Dependency-free tests for the locked V5 runner/analyzer contracts."""

from pathlib import Path
import json
import numpy as np
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.analyze_hacssm_v5 as analysis
import scripts.run_hacssm_v5 as runner


def _rows(seeds, *, full_mse=0.80):
    reference_mse = {
        'none': 1.20,
        'ssm': 1.00,
        'hacsmv4': 1.02,
        'hacsmv4_noaux': 1.00,
        'hacsmv4_two_noaux': 0.98,
        'hacssmv5_ssmcontrol': 0.98,
        'hacssmv5_fixedbeta_noaux': 1.00,
        'hacssmv5_noaux': 0.90,
        'hacssmv5_noaction': 0.96,
        'hacssmv5_static': 0.96,
        'hacssmv5_single': 0.96,
        'hacssmv5': full_mse,
    }
    rows = []
    for env in analysis.OCC_TO_CLEAN:
        for design in analysis.DESIGNS:
            for seed in seeds:
                mse = reference_mse[design]
                rows.append({
                    'env': env,
                    'design': design,
                    'seed': seed,
                    analysis.PRIMARY: mse,
                    'last_visible_mse_first_post': 1.10,
                    'clean_input_mse_first_post': 0.90 if design == 'hacssmv5' else 1.00,
                })
    return rows


def _convergence(rows):
    return [
        {'run': f"{row['env']}:{row['design']}:{row['seed']}",
         'relative_improvement': 0.001}
        for row in rows
    ]


def test_memory_and_schedule_contracts() -> None:
    contract = runner.memory_contract()
    assert contract['memory_parameters']['hacssmv5_all_modes'] == 34_820
    assert contract['streaming_recurrent_floats']['hacssmv5_all_modes'] == 256
    assert runner.stable_equal(contract, json.loads(json.dumps(contract)))
    assert runner.scheduled_weight(0.05, 'v5_frontload', 20) == 0.05
    assert abs(runner.scheduled_weight(0.05, 'v5_frontload', 70) - 0.025) < 1e-12
    assert runner.scheduled_weight(0.05, 'v5_frontload', 120) == 0.0
    assert runner.scheduled_weight(0.05, 'v5_frontload', 200) == 0.0
    preflight = {
        'authenticated': True,
        'base_url': runner.WANDB_BASE_URL,
        'entity': runner.WANDB_ENTITY,
        'mode': runner.WANDB_MODE,
        'project': runner.WANDB_PROJECT,
        'sdk_version': 'test',
        'study': runner.WANDB_STUDY,
    }
    protocol = runner.build_protocol('0' * 40, True, preflight)
    assert runner.stable_equal(protocol, json.loads(json.dumps(protocol)))


def test_stopped_stage_cannot_launch_or_create_a_log() -> None:
    stop = runner.threading.Event()
    stop.set()
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / 'must_not_exist.log'
        result = runner.run_logged_process(
            [sys.executable, '-c', 'raise SystemExit(99)'],
            log_path,
            runner.os.environ.copy(),
            stop,
        )
        assert result is None
        assert not log_path.exists()


def test_online_wandb_and_rollout_contracts() -> None:
    assert runner.COMMON['wandb'] is True
    assert runner.COMMON['wandb_entity'] == 'crlc112358'
    assert runner.COMMON['wandb_project'] == 'lewm-memory-popgym'
    assert runner.COMMON['wandb_mode'] == 'online'
    assert runner.COMMON['wandb_study'] == 'hacssm-v5'
    snapshot = runner.eval_rollout_snapshot()
    assert len(snapshot) == 5
    assert all(record['bytes'] > 0 and len(record['sha256']) == 64
               for record in snapshot.values())

    job = runner.PILOT_JOBS[0]
    expected = runner.expected_args(job)
    assert expected['wandb'] is True
    assert expected['eval_rollout_episode'] == 0
    assert expected['eval_rollout_cache'] == runner.rel(
        runner.eval_rollout_cache(job.clean_env))
    command = runner.train_command(sys.executable, job)
    for flag in ('--wandb', '--wandb-project', '--wandb-entity', '--wandb-mode',
                 '--wandb-study', '--eval-rollout-cache', '--eval-rollout-episode'):
        assert flag in command
    assert '--no-wandb' not in command


def test_tracking_receipt_binds_rollout_and_local_transaction() -> None:
    old_output = runner.OUTPUT_ROOT
    try:
        with tempfile.TemporaryDirectory() as directory:
            runner.OUTPUT_ROOT = Path(directory)
            job = runner.Job(
                'pilot', 0, 'dmc:reacher.hard.occ', 'dmc:reacher.hard', 'hacssmv5')
            job.run_dir.mkdir(parents=True)
            h, length, dim = 3, 32, 128
            target_times = np.arange(h, length, dtype=np.int64)
            target = np.zeros((length - h, dim), dtype=np.float32)
            prediction = np.full_like(target, 0.1)
            prediction_no_memory = np.full_like(target, 0.2)
            prediction_last_visible = np.full_like(target, 0.3)
            visible = np.ones(length, dtype=np.bool_)
            visible[length // 3:length // 3 + max(4, length // 5)] = False
            with np.load(runner.eval_rollout_cache(job.clean_env), allow_pickle=False) as cache:
                actions = np.asarray(cache['actions'][0], dtype=np.int64)[h - 1:length - 1]
            np.savez_compressed(
                job.eval_rollout_path,
                schema_version=np.asarray(1, dtype=np.int64),
                episode_index=np.asarray(0, dtype=np.int64),
                history_len=np.asarray(h, dtype=np.int64),
                target_times=target_times,
                target_visible=visible[h:],
                actions_to_target=actions,
                prediction=prediction,
                prediction_no_memory=prediction_no_memory,
                prediction_last_visible=prediction_last_visible,
                target=target,
                mse=np.square(prediction - target).mean(-1),
                mse_no_memory=np.square(prediction_no_memory - target).mean(-1),
                mse_last_visible=np.square(prediction_last_visible - target).mean(-1),
            )
            rollout_hash = runner.sha256_file(job.eval_rollout_path)
            run_id = 'test1234'
            artifact = 'hacssm-v5-eval-test1234'
            receipt = {
                'schema_version': 1,
                'run_id': run_id,
                'run_name': f'hacssm-v5-{job.run_name}',
                'url': f'https://wandb.ai/crlc112358/lewm-memory-popgym/runs/{run_id}',
                'entity': 'crlc112358',
                'project': 'lewm-memory-popgym',
                'mode': 'online',
                'study': 'hacssm-v5',
                'state': 'finished',
                'eval_rollout_artifact_name': artifact,
                'eval_rollout_sha256': rollout_hash,
                'eval_rollout_episode': 0,
            }
            job.wandb_run_path.write_text(json.dumps(receipt))
            transaction = job.run_dir / 'wandb' / 'run-test' / f'run-{run_id}.wandb'
            transaction.parent.mkdir(parents=True)
            transaction.write_bytes(b'nonempty transaction')
            metrics = {
                'wandb_run_id': run_id,
                'wandb_run_url': receipt['url'],
                'synced_online': True,
                'eval_rollout_sha256': rollout_hash,
                'eval_rollout_artifact_name': artifact,
            }
            runner.validate_tracking_artifacts(job, metrics)
    finally:
        runner.OUTPUT_ROOT = old_output


def test_pilot_screen_passes_only_when_every_criterion_passes() -> None:
    rows = _rows(analysis.PILOT_SEEDS)
    decision = analysis.pilot_decision(
        rows, _convergence(rows), analysis.contrast_rows(rows))
    assert decision['decision'] == 'PILOT_PASS'
    assert decision['pilot_screen_passed'] is True
    assert all(decision['criteria'].values())

    failed_rows = _rows(analysis.PILOT_SEEDS, full_mse=1.05)
    failed = analysis.pilot_decision(
        failed_rows, _convergence(failed_rows), analysis.contrast_rows(failed_rows))
    assert failed['decision'] == 'NO_GO'
    assert failed['pilot_screen_passed'] is False
    assert not all(failed['criteria'].values())


def test_final_label_is_locked_grid_not_publication_claim() -> None:
    rows = _rows(analysis.FINAL_SEEDS)
    result = analysis.final_summary(
        rows, _convergence(rows), analysis.contrast_rows(rows), pilot_screen_passed=True)
    assert result['decision'] == 'OVERALL_BEST_IN_LOCKED_GRID'
    assert result['completed_runs'] == 300
    assert all(result['criteria'].values())
    assert 'not an untouched-test' in result['note']

    negative_rows = _rows(analysis.FINAL_SEEDS, full_mse=1.05)
    negative = analysis.final_summary(
        negative_rows, _convergence(negative_rows), analysis.contrast_rows(negative_rows),
        pilot_screen_passed=True)
    assert negative['decision'] == 'NO_GO'

    descriptive = analysis.final_summary(
        rows, _convergence(rows), analysis.contrast_rows(rows), pilot_screen_passed=False)
    assert descriptive['decision'] == 'PILOT_NO_GO_FINAL_DESCRIPTIVE'

    none_best = _rows(analysis.FINAL_SEEDS)
    for row in none_best:
        if row['design'] == 'none':
            row[analysis.PRIMARY] = 0.70
    no_false_best = analysis.final_summary(
        none_best, _convergence(none_best), analysis.contrast_rows(none_best),
        pilot_screen_passed=True)
    assert no_false_best['decision'] != 'OVERALL_BEST_IN_LOCKED_GRID'
    assert not no_false_best['criteria']['locked_grid_envelope_wins_ge_4_of_5']


if __name__ == '__main__':
    tests = (
        test_memory_and_schedule_contracts,
        test_stopped_stage_cannot_launch_or_create_a_log,
        test_online_wandb_and_rollout_contracts,
        test_tracking_receipt_binds_rollout_and_local_transaction,
        test_pilot_screen_passes_only_when_every_criterion_passes,
        test_final_label_is_locked_grid_not_publication_claim,
    )
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} HACSSM-v5 protocol tests passed.')
