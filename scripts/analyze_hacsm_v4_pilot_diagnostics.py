#!/usr/bin/env python3
"""Post-decision causal diagnostics for a stopped 135-cell HACSM-v4 pilot.

The prospective pilot decision is never recomputed or altered here.  This script reuses the
locked evaluator to measure shifted masks, gates, memory-only action interventions, and level
ablations on the completed three-seed cohort, then hashes its own descriptive outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.analyze_hacsm_v4 as analysis


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def verify_primary_manifest(root: Path) -> dict:
    """Fail closed if any producer source, feature, checkpoint, metric, or log drifted."""
    manifest_path = root / 'hacsm_v4_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('completed_runs') != 135 or manifest.get('expected_runs') != 135:
        raise ValueError('primary manifest does not attest the complete 135-cell pilot')
    if manifest.get('expanded') is not False:
        raise ValueError('post-decision pilot diagnostics require expanded=false')

    sidecar_path = root / 'hacsm_v4_manifest.sha256'
    sidecar_fields = sidecar_path.read_text().strip().split()
    if not sidecar_fields or sidecar_fields[0] != sha256(manifest_path):
        raise ValueError('primary manifest SHA-256 sidecar mismatch')

    for section in ('source_artifacts', 'feature_artifacts',
                    'output_artifacts', 'log_artifacts'):
        artifacts = manifest.get(section)
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(f'primary manifest has no {section}')
        for recorded_path, expected in artifacts.items():
            path = Path(recorded_path)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f'manifest artifact is missing: {recorded_path}')
            if path.stat().st_size != int(expected['bytes']):
                raise ValueError(f'manifest byte-count mismatch: {recorded_path}')
            if sha256(path) != expected['sha256']:
                raise ValueError(f'manifest SHA-256 mismatch: {recorded_path}')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/hacsm_v4_shared'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    verify_primary_manifest(args.root)
    decision_path = args.root / 'pilot_decision.json'
    decision = json.loads(decision_path.read_text())
    if decision.get('expand') is not False or decision.get('decision') != 'NO_GO':
        raise ValueError('pilot diagnostics are only defined for a terminal pilot NO_GO')

    rows, checkpoints, _convergence = analysis.load_cells(
        args.root, analysis.PILOT_DESIGNS, analysis.PILOT_SEEDS)
    # The shared evaluator is written for the expanded grid; bind it to the already locked pilot.
    analysis.FINAL_DESIGNS = analysis.PILOT_DESIGNS
    analysis.FINAL_SEEDS = analysis.PILOT_SEEDS
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    mask_rows, gate_rows, action_rows, level_rows = analysis.final_interventions(
        args.root, checkpoints, device)

    output_paths = {
        'mask': args.root / 'pilot_mask_generalization_per_run.csv',
        'gate': args.root / 'pilot_v4_gate_per_run.csv',
        'action': args.root / 'pilot_v4_action_interventions.csv',
        'level': args.root / 'pilot_v4_level_ablations.csv',
    }
    analysis.write_csv(output_paths['mask'], mask_rows)
    analysis.write_csv(output_paths['gate'], gate_rows)
    analysis.write_csv(output_paths['action'], action_rows)
    analysis.write_csv(output_paths['level'], level_rows)

    mask_lookup = {(row['env'], row['design'], int(row['seed']), row['condition']):
                   float(row['first_post_mse']) for row in mask_rows}
    mask_effects = {}
    for condition in analysis.MASKS:
        values = []
        for env in analysis.OCC_TO_CLEAN:
            for seed in analysis.PILOT_SEEDS:
                candidate = mask_lookup[(env, 'hacsmv4', seed, condition)]
                baseline = mask_lookup[(env, 'ssm', seed, condition)]
                values.append((baseline - candidate) / baseline)
        mask_effects[condition] = mean(values)

    action_summary = defaultdict(list)
    changed_fraction = defaultdict(list)
    for row in action_rows:
        key = f"{row['design']}:{row['condition']}"
        action_summary[key].append(float(row['relative_change']))
        changed_fraction[key].append(float(row['changed_action_fraction']))
    action_summary = {key: mean(value) for key, value in sorted(action_summary.items())}
    changed_fraction = {key: mean(value) for key, value in sorted(changed_fraction.items())}

    gate_summary = {}
    for level, tau in enumerate((2, 8, 32)):
        selected = [row for row in gate_rows
                    if row['design'] == 'hacsmv4' and int(row['level']) == level]
        gate_summary[f'tau_{tau}'] = {
            'visible_mean': mean(float(row['visible_mean']) for row in selected),
            'black_mean': mean(float(row['black_mean']) for row in selected),
            'visible_minus_black': mean(float(row['visible_minus_black']) for row in selected),
            'visible_auroc': mean(float(row['visible_auroc']) for row in selected),
            'mean_gate_relative_change': mean(
                float(row['mean_gate_relative_change']) for row in selected),
        }

    level_summary = defaultdict(list)
    for row in level_rows:
        level_summary[row['condition']].append(float(row['relative_change']))
    level_summary = {key: mean(value) for key, value in sorted(level_summary.items())}

    contrasts = {
        row['reference']: row for row in analysis.contrast_rows(rows)
        if row['env'] == '__overall__'
    }
    architecture = {
        reference: {
            'mean_paired_relative_reduction': float(
                contrasts[reference]['mean_paired_relative_reduction']),
            'paired_wins': int(contrasts[reference]['paired_wins']),
            'n_pairs': int(contrasts[reference]['n_pairs']),
        }
        for reference in ('ssm', 'smtv3', 'hacsmv4_static', 'hacsmv4_noaction',
                          'hacsmv4_noaux', 'hacsmv4_single')
    }
    summary = {
        'schema_version': 1,
        'label': 'descriptive_post_decision_pilot_diagnostics',
        'changes_prospective_decision': False,
        'pilot_decision_sha256': sha256(decision_path),
        'mask_v4_vs_ssm_mean_paired_relative_reduction': mask_effects,
        'architecture_contrasts': architecture,
        'full_v4_gate_summary': gate_summary,
        'memory_only_action_interventions': action_summary,
        'changed_action_fraction': changed_fraction,
        'full_v4_level_ablations': level_summary,
    }
    summary_path = args.root / 'pilot_diagnostics.json'
    atomic_json(summary_path, summary)
    manifest = {
        'schema_version': 1,
        'analysis_script': {
            str(Path(__file__).resolve().relative_to(Path.cwd().resolve())): sha256(Path(__file__)),
            'scripts/analyze_hacsm_v4.py': sha256(Path('scripts/analyze_hacsm_v4.py')),
        },
        'inputs': {
            'pilot_decision.json': sha256(decision_path),
            'protocol.json': sha256(args.root / 'protocol.json'),
            'hacsm_v4_manifest.json': sha256(args.root / 'hacsm_v4_manifest.json'),
        },
        'outputs': {path.name: sha256(path) for path in (*output_paths.values(), summary_path)},
    }
    atomic_json(args.root / 'pilot_diagnostics_manifest.json', manifest)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
