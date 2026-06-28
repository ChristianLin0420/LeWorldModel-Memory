#!/usr/bin/env bash
# SMT-v3 shared-target experiment.
#
# Fixed factorial: five simulator tasks x nine designs x five optimizer seeds = 225 runs.
# All designs use the same immutable DINOv2-PCA feature artifact for an environment.  The
# runner validates every existing checkpoint before it skips it, refuses partial/stale grids,
# propagates every background-worker failure, and only invokes analysis after all 225 cells
# have passed an exact checkpoint/config/metrics audit.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-.venv/bin/python}
OUT=${OUT:-outputs/smt_v3_shared}
LOG_DIR=${LOG_DIR:-logs/smt_v3_shared}
DATA_DIR=${DATA_DIR:-outputs/popgym_data}
TRAIN_EPS=${TRAIN_EPS:-600}
VAL_EPS=${VAL_EPS:-150}
LEN=${LEN:-32}
EPOCHS=${EPOCHS:-200}
FEATURE_DIM=${FEATURE_DIM:-128}
FIRST_POST_LOSS_WEIGHT=${FIRST_POST_LOSS_WEIGHT:-0.5}
BATCH_SIZE=${BATCH_SIZE:-64}
read -r -a GPU_LIST <<< "${GPU_IDS:-0 0 1 1 2 2 3 3}"
WORKERS=${WORKERS:-${#GPU_LIST[@]}}

if [[ ! -x "$PY" ]]; then
  echo "Python executable is missing or not executable: $PY" >&2
  exit 2
fi
if (( WORKERS < 1 || WORKERS > ${#GPU_LIST[@]} )); then
  echo "WORKERS must be in [1, ${#GPU_LIST[@]}], got $WORKERS" >&2
  exit 2
fi
if [[ "$EPOCHS" != 200 ]]; then
  echo "SMT-v3 is a fixed 200-epoch protocol after the 100-epoch convergence gate failed; got EPOCHS=$EPOCHS" >&2
  exit 2
fi
if [[ "$TRAIN_EPS" != 600 || "$VAL_EPS" != 150 || "$LEN" != 32 ||
      "$FEATURE_DIM" != 128 || "$BATCH_SIZE" != 64 ||
      "$FIRST_POST_LOSS_WEIGHT" != 0.5 ]]; then
  echo "SMT-v3 protocol override rejected: expected train/val/length/dim/batch/first-post-weight = 600/150/32/128/64/0.5" >&2
  exit 2
fi
GPU_LIST=("${GPU_LIST[@]:0:$WORKERS}")

OCC_ENVS=(
  dmc:reacher.hard.occ
  dmc:ball_in_cup.catch.occ
  dmc:finger.spin.occ
  dmc:cheetah.run.occ
  ogbench:cube-single.occ
)
CLEAN_ENVS=(
  dmc:reacher.hard
  dmc:ball_in_cup.catch
  dmc:finger.spin
  dmc:cheetah.run
  ogbench:cube-single
)
DESIGNS=(none multi gru ssm smt smtv3_static smtv3 smtv3_old smtv3_oracle)
SEEDS=(0 1 2 3 4)

# Fail before collection/precompute when this runner is installed ahead of the corresponding
# model/training CLI.  This is expected during development and must not leave a partial run grid.
TRAIN_HELP=$("$PY" scripts/train_popgym.py --help)
if [[ "$TRAIN_HELP" != *"--first-post-loss-weight"* ]]; then
  echo "train_popgym.py does not yet provide --first-post-loss-weight; SMT-v3 was not started" >&2
  exit 2
fi
for design in "${DESIGNS[@]}"; do
  if [[ "$TRAIN_HELP" != *"$design"* ]]; then
    echo "train_popgym.py does not yet provide memory mode '$design'; SMT-v3 was not started" >&2
    exit 2
  fi
done

mkdir -p "$OUT" "$DATA_DIR" "$LOG_DIR"
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent concurrent SMT-v3 writers" >&2
  exit 2
fi
exec 9>"$OUT/.run_smt_v3.lock"
if ! flock -n 9; then
  echo "another SMT-v3 runner already holds $OUT/.run_smt_v3.lock" >&2
  exit 2
fi

wait_all() {
  local label=$1
  shift
  local pid status=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      echo "$label worker failed (pid=$pid)" >&2
      status=1
    fi
  done
  return "$status"
}

# Collect schema-v3 clean data first.  The .occ cache is then an exact masked copy of the
# clean pixels/actions, with prototype_seed=0 shared between train and validation.
collect_worker() {
  local slot=$1 gpu=${GPU_LIST[$1]}
  local idx env tag
  for idx in "${!COLLECT_ENVS[@]}"; do
    (( idx % WORKERS == slot )) || continue
    env=${COLLECT_ENVS[$idx]}
    tag=${env//[:.]/_}
    echo "$(date +%T) [gpu $gpu] collect $env"
    if ! MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" "$PY" - \
      "$env" "$TRAIN_EPS" "$VAL_EPS" "$LEN" "$DATA_DIR" \
      > "$LOG_DIR/collect_${tag}.log" 2>&1 <<'PY'
import sys
from lewm.envs.popgym_arcade import get_or_collect

env = sys.argv[1]
ntr, nva, length = map(int, sys.argv[2:5])
data_dir = sys.argv[5]
train = get_or_collect(
    env, ntr, length, img_size=64, seed=0, data_dir=data_dir, prototype_seed=0)
val = get_or_collect(
    env, nva, length, img_size=64, seed=7777, data_dir=data_dir, prototype_seed=0)
print(f"collected {env}: train={train[0].shape}, val={val[0].shape}, actions={train[2]}")
PY
    then
      echo "collection failed for $env (see $LOG_DIR/collect_${tag}.log)" >&2
      return 2
    fi
  done
}

collect_phase() {
  local label=$1
  local pids=() slot
  for ((slot=0; slot<WORKERS; slot++)); do
    collect_worker "$slot" &
    pids+=("$!")
  done
  if ! wait_all "$label" "${pids[@]}"; then
    return 2
  fi
}

COLLECT_ENVS=("${CLEAN_ENVS[@]}")
echo "$(date +%T) collecting ${#COLLECT_ENVS[@]} clean trajectory caches"
collect_phase "clean collection"
COLLECT_ENVS=("${OCC_ENVS[@]}")
echo "$(date +%T) deriving ${#COLLECT_ENVS[@]} paired occlusion caches"
collect_phase "occlusion collection"

# Independently fail before feature generation if a paired cache is not synchronized.
for idx in "${!OCC_ENVS[@]}"; do
  "$PY" - "${OCC_ENVS[$idx]}" "${CLEAN_ENVS[$idx]}" \
    "$TRAIN_EPS" "$VAL_EPS" "$LEN" "$DATA_DIR" <<'PY'
import sys
import numpy as np
from lewm.data import PopgymDataset

occ, clean = sys.argv[1:3]
ntr, nva, length = map(int, sys.argv[3:6])
data_dir = sys.argv[6]
train = PopgymDataset(
    occ, ntr, length, 64, seed=0, data_dir=data_dir,
    prototype_seed=0, target_env_id=clean)
val = PopgymDataset(
    occ, nva, length, 64, seed=7777, data_dir=data_dir,
    prototype_seed=0, target_env_id=clean)
start = length // 3
end = min(length, start + max(4, length // 5))
for split, pair in [('train', train), ('val', val)]:
    if not np.array_equal(pair.obs[:, :start], pair.target_obs[:, :start]):
        raise SystemExit(f'{split}: pre-blackout frames differ for {occ} -> {clean}')
    if not np.array_equal(pair.obs[:, end:], pair.target_obs[:, end:]):
        raise SystemExit(f'{split}: post-blackout frames differ for {occ} -> {clean}')
    if np.any(pair.obs[:, start:end]):
        raise SystemExit(f'{split}: .occ input is not black in [{start},{end})')
    if not np.any(pair.target_obs[:, start:end]):
        raise SystemExit(f'{split}: clean target is entirely black in [{start},{end})')
print(f'paired cache validation passed: {occ} -> {clean}')
PY
done

# Deterministically encode clean pixels once, fit visible-train-only non-whitened PCA, and
# derive the paired occluded feature stream.  The precompute script itself performs strict,
# content-hash-checked resume validation when these files already exist.
FEATURE_DIR="$OUT/dino_features_d${FEATURE_DIM}"
mkdir -p "$FEATURE_DIR"
precompute_worker() {
  local slot=$1 gpu=${GPU_LIST[$1]}
  local idx
  for idx in "${!CLEAN_ENVS[@]}"; do
    (( idx % WORKERS == slot )) || continue
    echo "$(date +%T) [gpu $gpu] DINO-PCA ${CLEAN_ENVS[$idx]}"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/precompute_dino_clean_features.py \
      --clean-env "${CLEAN_ENVS[$idx]}" --data-dir "$DATA_DIR" \
      --output-dir "$FEATURE_DIR" --train-episodes "$TRAIN_EPS" \
      --val-episodes "$VAL_EPS" --length "$LEN" --device cuda \
      --batch-size 64 --dim "$FEATURE_DIM" \
      > "$LOG_DIR/precompute_${idx}.log" 2>&1; then
      echo "DINO-PCA precompute failed for ${CLEAN_ENVS[$idx]} (see $LOG_DIR/precompute_${idx}.log)" >&2
      return 2
    fi
  done
}

PIDS=()
for ((slot=0; slot<WORKERS; slot++)); do
  precompute_worker "$slot" &
  PIDS+=("$!")
done
if ! wait_all "DINO-PCA precompute" "${PIDS[@]}"; then
  exit 2
fi
for clean in "${CLEAN_ENVS[@]}"; do
  safe=${clean//[^[:alnum:]]/_}
  for artifact in "$FEATURE_DIR/${safe}_train.npz" "$FEATURE_DIR/${safe}_val.npz" \
                  "$FEATURE_DIR/${safe}_manifest.json"; do
    if [[ ! -s "$artifact" ]]; then
      echo "missing DINO-PCA artifact: $artifact" >&2
      exit 2
    fi
  done
done

JOB_OCC=()
JOB_CLEAN=()
JOB_DESIGN=()
JOB_SEED=()
RUN_DIRS=()
LOGS=()
for seed in "${SEEDS[@]}"; do
  for idx in "${!OCC_ENVS[@]}"; do
    occ=${OCC_ENVS[$idx]}
    for design in "${DESIGNS[@]}"; do
      run="lewm-${occ}-${design}-s${seed}"
      logtag=${run//[:.]/_}
      JOB_OCC+=("$occ")
      JOB_CLEAN+=("${CLEAN_ENVS[$idx]}")
      JOB_DESIGN+=("$design")
      JOB_SEED+=("$seed")
      RUN_DIRS+=("$OUT/$run")
      LOGS+=("$LOG_DIR/${logtag}.log")
    done
  done
done
if (( ${#RUN_DIRS[@]} != 225 )); then
  echo "internal error: constructed ${#RUN_DIRS[@]} jobs, expected 225" >&2
  exit 2
fi

# Validate every complete checkpoint before allowing it to be skipped.  In preflight mode,
# cells with neither artifact are allowed; in final mode, all 225 cells are mandatory.
validate_grid() {
  local phase=$1
  "$PY" - "$phase" "$OUT" "$FEATURE_DIR" "$DATA_DIR" \
    "$TRAIN_EPS" "$VAL_EPS" "$LEN" "$EPOCHS" "$FEATURE_DIM" \
    "$BATCH_SIZE" "$FIRST_POST_LOSS_WEIGHT" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

(
    phase, output_root, feature_root, data_root, train_eps, val_eps, length,
    epochs, feature_dim, batch_size, first_post_weight,
) = sys.argv[1:]
if phase not in {'preflight', 'final'}:
    raise SystemExit(f'unknown validation phase: {phase}')
root = Path(output_root).resolve()
features = Path(feature_root).resolve()
data_root = Path(data_root).resolve()
train_eps, val_eps, length, epochs, feature_dim, batch_size = map(
    int, (train_eps, val_eps, length, epochs, feature_dim, batch_size))
first_post_weight = float(first_post_weight)

occ_to_clean = {
    'dmc:reacher.hard.occ': 'dmc:reacher.hard',
    'dmc:ball_in_cup.catch.occ': 'dmc:ball_in_cup.catch',
    'dmc:finger.spin.occ': 'dmc:finger.spin',
    'dmc:cheetah.run.occ': 'dmc:cheetah.run',
    'ogbench:cube-single.occ': 'ogbench:cube-single',
}
designs = ('none', 'multi', 'gru', 'ssm', 'smt',
           'smtv3_static', 'smtv3', 'smtv3_old', 'smtv3_oracle')
seeds = range(5)

def safe_env(value):
    return ''.join(char if char.isalnum() else '_' for char in value).strip('_')

def resolve_repo_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()

def reject_non_rfc_json(value):
    raise ValueError(f'non-RFC JSON constant {value}')

def semantically_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            semantically_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            semantically_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
    return left == right

expected = {
    f'lewm-{occ}-{design}-s{seed}': (occ, clean, design, seed)
    for occ, clean in occ_to_clean.items()
    for design in designs
    for seed in seeds
}
actual_run_dirs = {path.name for path in root.glob('lewm-*') if path.is_dir()}
unexpected_dirs = actual_run_dirs - set(expected)
if unexpected_dirs:
    raise SystemExit(f'unexpected SMT-v3 run directories: {sorted(unexpected_dirs)[:8]}')
expected_models = {root / name / 'model.pt' for name in expected}
expected_metrics = {root / name / 'metrics.json' for name in expected}
unexpected_models = set(root.rglob('model.pt')) - expected_models
unexpected_metrics = set(root.rglob('metrics.json')) - expected_metrics
if unexpected_models or unexpected_metrics:
    raise SystemExit(
        'unexpected SMT-v3 artifacts: '
        f'models={sorted(map(str, unexpected_models))[:4]}, '
        f'metrics={sorted(map(str, unexpected_metrics))[:4]}')

complete = 0
for run_name, (occ, clean, design, seed) in sorted(expected.items()):
    run_dir = root / run_name
    model_path = run_dir / 'model.pt'
    metrics_path = run_dir / 'metrics.json'
    model_exists = model_path.is_file() and model_path.stat().st_size > 0
    metrics_exists = metrics_path.is_file() and metrics_path.stat().st_size > 0
    if model_exists != metrics_exists:
        raise SystemExit(f'partial run artifacts: {run_dir}')
    if not model_exists:
        if model_path.exists() or metrics_path.exists():
            raise SystemExit(f'empty/non-file run artifact: {run_dir}')
        if phase == 'final':
            raise SystemExit(f'missing final run artifacts: {run_dir}')
        continue

    try:
        metrics = json.loads(metrics_path.read_text(), parse_constant=reject_non_rfc_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f'{metrics_path}: invalid metrics JSON: {exc}') from exc
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model_state_dict'), dict):
        raise SystemExit(f'{model_path}: invalid checkpoint structure')
    if not checkpoint['model_state_dict']:
        raise SystemExit(f'{model_path}: empty model state')
    history = checkpoint.get('history')
    if not isinstance(history, list) or len(history) != epochs:
        raise SystemExit(
            f'{run_name}: training history has {len(history) if isinstance(history, list) else None} '
            f'epochs, expected {epochs}')
    for expected_epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get('epoch') != expected_epoch:
            raise SystemExit(f'{run_name}: malformed training history at epoch {expected_epoch}')
        for split in ('train', 'val'):
            values = record.get(split)
            if not isinstance(values, dict):
                raise SystemExit(f'{run_name}: missing {split} history at epoch {expected_epoch}')
            for key in ('loss', 'pred_loss', 'sigreg_loss'):
                value = values.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise SystemExit(
                        f'{run_name}: non-finite {split}.{key} at epoch {expected_epoch}')
    if not semantically_equal(metrics, checkpoint.get('final_metrics')):
        raise SystemExit(f'{run_name}: metrics.json differs from checkpoint final_metrics')
    cfg = checkpoint.get('args')
    if not isinstance(cfg, dict):
        raise SystemExit(f'{run_name}: checkpoint has no args dictionary')

    exact = {
        'env_id': occ,
        'target_env_id': clean,
        'memory_mode': design,
        'smt_router': 'sigmoid',
        'seed': seed,
        'num_episodes': train_eps,
        'val_episodes': val_eps,
        'prototype_seed': 0,
        'mask_occluded_target_loss': True,
        'freeze_encoder': False,
        'encoder_type': 'precomputed',
        'length': length,
        'img_size': 64,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': 3e-4,
        'weight_decay': 1e-5,
        'num_workers': 2,
        'no_amp': False,
        'patch_size': 8,
        'embed_dim': feature_dim,
        'encoder_layers': 6,
        'encoder_heads': 4,
        'predictor_layers': 4,
        'predictor_heads': 8,
        'history_len': 3,
        'dropout': 0.1,
        'sigreg_lambda': 0.1,
        'sigreg_projections': 512,
        'tau_fast': 3.0,
        'tau_slow': 25.0,
        'fixed_alpha': True,
        'wandb': False,
        'device': 'cuda',
        'first_post_loss_weight': first_post_weight,
    }
    for key, wanted in exact.items():
        if cfg.get(key) != wanted:
            raise SystemExit(
                f'{run_name}: {key}={cfg.get(key)!r}, expected {wanted!r}')
    if cfg.get('encoder_checkpoint') is not None or cfg.get('encoder_stats') is not None:
        raise SystemExit(f'{run_name}: unexpected learned/frozen encoder source')
    if resolve_repo_path(cfg.get('output_dir', '')) != root:
        raise SystemExit(f'{run_name}: output_dir does not resolve to {root}')
    if resolve_repo_path(cfg.get('data_dir', '')) != data_root:
        raise SystemExit(f'{run_name}: data_dir does not resolve to {data_root}')

    safe = safe_env(clean)
    manifest = features / f'{safe}_manifest.json'
    train_features = features / f'{safe}_train.npz'
    val_features = features / f'{safe}_val.npz'
    for artifact in (manifest, train_features, val_features):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise SystemExit(f'{run_name}: missing feature artifact {artifact}')
    path_checks = {
        'feature_manifest': manifest,
        'train_feature_cache': train_features,
        'val_feature_cache': val_features,
    }
    for key, wanted in path_checks.items():
        if resolve_repo_path(cfg.get(key, '')) != wanted:
            raise SystemExit(f'{run_name}: unexpected {key}={cfg.get(key)!r}')
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if cfg.get('feature_manifest_sha256') != manifest_sha:
        raise SystemExit(f'{run_name}: checkpoint feature-manifest hash mismatch')

    metric_checks = {
        'env': occ,
        'design': design,
        'target_env': clean,
        'masked_clean_blackout_loss': True,
        'first_post_loss_weight': first_post_weight,
        'dataset_schema_version': 3,
        'feature_schema_version': 1,
        'val_pred_loss_target_kind': 'observed_pre_post_only',
        'deep_blackout_target_kind': 'evaluation_only_hidden_clean',
        'primary_common_target_metric': 'clean_mse_first_post',
        'encoder_type': 'precomputed',
        'external_features_fixed': True,
        'encoder_frozen': False,
        'feature_manifest_sha256': manifest_sha,
    }
    for key, wanted in metric_checks.items():
        if metrics.get(key) != wanted:
            raise SystemExit(
                f'{run_name}: metric {key}={metrics.get(key)!r}, expected {wanted!r}')
    if not isinstance(metrics.get('trainable_parameters'), int) or metrics['trainable_parameters'] <= 0:
        raise SystemExit(f'{run_name}: invalid trainable parameter count')
    required_finite_metrics = (
        'val_pred_loss', 'infl_fast', 'infl_slow',
        'clean_mse_deep_blackout', 'clean_mse_deep_blackout_ablated',
        'clean_mse_first_post', 'clean_mse_first_post_ablated',
        'constant_mse_first_post', 'last_visible_mse_first_post',
    )
    for key in required_finite_metrics:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f'{run_name}: invalid metric {key}={value!r}')
    complete += 1

if phase == 'final' and complete != len(expected):
    raise SystemExit(f'incomplete SMT-v3 grid: {complete}/{len(expected)}')
print(f'{phase} validation passed: {complete}/{len(expected)} complete runs')
PY
}

validate_grid preflight

echo "$(date +%T) training ${#RUN_DIRS[@]} SMT-v3 runs on GPUs [${GPU_LIST[*]}]"
train_worker() {
  local slot=$1 gpu=${GPU_LIST[$1]}
  local idx run_dir occ clean design seed safe train_features val_features manifest
  for idx in "${!RUN_DIRS[@]}"; do
    (( idx % WORKERS == slot )) || continue
    run_dir=${RUN_DIRS[$idx]}
    if [[ -s "$run_dir/model.pt" && -s "$run_dir/metrics.json" ]]; then
      echo "$(date +%T) [gpu $gpu] skip validated $run_dir"
      continue
    fi
    if [[ -e "$run_dir/model.pt" || -e "$run_dir/metrics.json" ]]; then
      echo "partial run exists; refusing to overwrite: $run_dir" >&2
      return 2
    fi
    occ=${JOB_OCC[$idx]}
    clean=${JOB_CLEAN[$idx]}
    design=${JOB_DESIGN[$idx]}
    seed=${JOB_SEED[$idx]}
    safe=${clean//[^[:alnum:]]/_}
    train_features="$FEATURE_DIR/${safe}_train.npz"
    val_features="$FEATURE_DIR/${safe}_val.npz"
    manifest="$FEATURE_DIR/${safe}_manifest.json"
    echo "$(date +%T) [gpu $gpu] >>> $run_dir"
    if ! MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_popgym.py \
      --env-id "$occ" --target-env-id "$clean" --mask-occluded-target-loss \
      --memory-mode "$design" --smt-router sigmoid --seed "$seed" --fixed-alpha \
      --encoder-type precomputed --train-feature-cache "$train_features" \
      --val-feature-cache "$val_features" --feature-manifest "$manifest" \
      --prototype-seed 0 --data-dir "$DATA_DIR" --output-dir "$OUT" \
      --num-episodes "$TRAIN_EPS" --val-episodes "$VAL_EPS" --length "$LEN" \
      --img-size 64 --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr 3e-4 \
      --weight-decay 1e-5 --num-workers 2 --patch-size 8 --embed-dim "$FEATURE_DIM" \
      --encoder-layers 6 --encoder-heads 4 --predictor-layers 4 --predictor-heads 8 \
      --history-len 3 --dropout 0.1 --sigreg-lambda 0.1 --sigreg-projections 512 \
      --tau-fast 3.0 --tau-slow 25.0 --first-post-loss-weight "$FIRST_POST_LOSS_WEIGHT" \
      --device cuda --no-wandb > "${LOGS[$idx]}" 2>&1; then
      echo "training failed: $run_dir (see ${LOGS[$idx]})" >&2
      return 2
    fi
    if [[ ! -s "$run_dir/model.pt" || ! -s "$run_dir/metrics.json" ]]; then
      echo "training completed without both required artifacts: $run_dir" >&2
      return 2
    fi
    echo "$(date +%T) [gpu $gpu] <<< $run_dir"
  done
}

PIDS=()
for ((slot=0; slot<WORKERS; slot++)); do
  train_worker "$slot" &
  PIDS+=("$!")
done
if ! wait_all "SMT-v3 training" "${PIDS[@]}"; then
  exit 2
fi

validate_grid final

if [[ -f scripts/analyze_smt_v3.py ]]; then
  "$PY" scripts/analyze_smt_v3.py --root "$OUT"
else
  echo "scripts/analyze_smt_v3.py is not present; exact grid validation completed, analysis deferred"
fi
echo "=== SMT-v3 SHARED-TARGET STUDY COMPLETE: 225/225 VALIDATED ==="
