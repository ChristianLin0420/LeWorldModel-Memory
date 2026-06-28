#!/usr/bin/env bash
# Shared-encoder, paired-clean-target occlusion study.
# Five simulator tasks x five designs x five seeds = 125 runs in one fixed DINOv2-PCA space.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-.venv/bin/python}
OUT=${OUT:-outputs/shared_clean_occlusion}
LOG_DIR=${LOG_DIR:-logs/shared_clean_occlusion}
DATA_DIR=${DATA_DIR:-outputs/popgym_data}
TRAIN_EPS=${TRAIN_EPS:-600}
VAL_EPS=${VAL_EPS:-150}
LEN=${LEN:-32}
EPOCHS=${EPOCHS:-30}
FEATURE_DIM=${FEATURE_DIM:-128}
SEEDS_STR=${SEEDS:-"0 1 2 3 4"}
DESIGNS_STR=${DESIGNS:-"none multi gru ssm smt"}
GPU_LIST=(${GPU_IDS:-0 0 1 1 2 2 3 3})
WORKERS=${WORKERS:-${#GPU_LIST[@]}}
if (( WORKERS < 1 || WORKERS > ${#GPU_LIST[@]} )); then
  echo "WORKERS must be in [1, ${#GPU_LIST[@]}], got $WORKERS" >&2
  exit 2
fi
GPU_LIST=("${GPU_LIST[@]:0:$WORKERS}")

mkdir -p "$OUT" "$DATA_DIR" "$LOG_DIR"

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

# Collect schema-v3 data with prototype_seed=0. Clean caches must complete first:
# each .occ cache is then derived from the exact clean pixels/actions by masking.
collect_worker() {
  local slot=$1 gpu=${GPU_LIST[$1]}
  for idx in "${!COLLECT_ENVS[@]}"; do
    (( idx % WORKERS == slot )) || continue
    local env=${COLLECT_ENVS[$idx]}
    local tag=${env//[:.]/_}
    echo "$(date +%T) [gpu $gpu] collect $env"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" "$PY" - \
      "$env" "$TRAIN_EPS" "$VAL_EPS" "$LEN" "$DATA_DIR" \
      > "$LOG_DIR/collect_${tag}.log" 2>&1 <<'PY'
import sys
from lewm.envs.popgym_arcade import get_or_collect
env = sys.argv[1]
ntr, nva, length = map(int, sys.argv[2:5])
data_dir = sys.argv[5]
tr = get_or_collect(env, ntr, length, img_size=64, seed=0, data_dir=data_dir, prototype_seed=0)
va = get_or_collect(env, nva, length, img_size=64, seed=7777, data_dir=data_dir, prototype_seed=0)
print(f"collected {env}: train={tr[0].shape}, val={va[0].shape}, actions={tr[2]}")
PY
  done
}
collect_phase() {
  local pids=()
  for ((slot=0; slot<WORKERS; slot++)); do collect_worker "$slot" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid"; done
}
COLLECT_ENVS=("${CLEAN_ENVS[@]}")
echo "$(date +%T) collecting ${#COLLECT_ENVS[@]} clean trajectory caches"
collect_phase
COLLECT_ENVS=("${OCC_ENVS[@]}")
echo "$(date +%T) deriving ${#COLLECT_ENVS[@]} paired occlusion caches"
collect_phase

# Fail before training if any paired action stream differs.
for idx in "${!OCC_ENVS[@]}"; do
  "$PY" - "${OCC_ENVS[$idx]}" "${CLEAN_ENVS[$idx]}" "$TRAIN_EPS" "$VAL_EPS" "$LEN" "$DATA_DIR" <<'PY'
import sys
from lewm.data import PopgymDataset
occ, clean = sys.argv[1:3]
ntr, nva, length = map(int, sys.argv[3:6])
data_dir = sys.argv[6]
tr = PopgymDataset(occ, ntr, length, 64, seed=0, data_dir=data_dir,
                   prototype_seed=0, target_env_id=clean)
va = PopgymDataset(occ, nva, length, 64, seed=7777, data_dir=data_dir,
                   prototype_seed=0, target_env_id=clean)
import numpy as np
for name, pair in [('train', tr), ('val', va)]:
    start = length // 3
    end = min(length, start + max(4, length // 5))
    if not np.array_equal(pair.obs[:, :start], pair.target_obs[:, :start]):
        raise SystemExit(f'{name}: pre-blackout frames differ for {occ} -> {clean}')
    if not np.array_equal(pair.obs[:, end:], pair.target_obs[:, end:]):
        raise SystemExit(f'{name}: post-blackout frames differ for {occ} -> {clean}')
    if np.any(pair.obs[:, start:end]):
        raise SystemExit(f'{name}: .occ input is not black in [{start},{end})')
print(f"paired action validation passed: {occ} -> {clean}")
PY
done

# Encode clean pixels once with fixed pretrained DINOv2, fit a visible-train-only
# non-whitened PCA, and derive exact black-frame feature streams.
FEATURE_DIR="$OUT/dino_features_d${FEATURE_DIM}"
mkdir -p "$FEATURE_DIR"
precompute_worker() {
  local slot=$1 gpu=${GPU_LIST[$1]}
  for idx in "${!CLEAN_ENVS[@]}"; do
    (( idx % WORKERS == slot )) || continue
    echo "$(date +%T) [gpu $gpu] DINO-PCA ${CLEAN_ENVS[$idx]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/precompute_dino_clean_features.py \
      --clean-env "${CLEAN_ENVS[$idx]}" --data-dir "$DATA_DIR" \
      --output-dir "$FEATURE_DIR" --train-episodes "$TRAIN_EPS" \
      --val-episodes "$VAL_EPS" --length "$LEN" --device cuda \
      --batch-size 64 --dim "$FEATURE_DIM" \
      > "$LOG_DIR/precompute_${idx}.log" 2>&1
  done
}
PIDS=()
for ((slot=0; slot<WORKERS; slot++)); do precompute_worker "$slot" & PIDS+=("$!"); done
for pid in "${PIDS[@]}"; do wait "$pid"; done
for clean in "${CLEAN_ENVS[@]}"; do
  safe=${clean//[^[:alnum:]]/_}
  for artifact in "$FEATURE_DIR/${safe}_train.npz" "$FEATURE_DIR/${safe}_val.npz" \
                  "$FEATURE_DIR/${safe}_manifest.json"; do
    [[ -s "$artifact" ]] || { echo "missing DINO-PCA artifact: $artifact" >&2; exit 2; }
  done
done

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a DESIGN_LIST <<< "$DESIGNS_STR"
JOBS=(); RUN_DIRS=(); LOGS=()
for seed in "${SEED_LIST[@]}"; do
  for idx in "${!OCC_ENVS[@]}"; do
    occ=${OCC_ENVS[$idx]}; clean=${CLEAN_ENVS[$idx]}; safe=${clean//[^[:alnum:]]/_}
    train_features="$FEATURE_DIR/${safe}_train.npz"
    val_features="$FEATURE_DIR/${safe}_val.npz"
    manifest="$FEATURE_DIR/${safe}_manifest.json"
    for design in "${DESIGN_LIST[@]}"; do
      run="lewm-${occ}-${design}-s${seed}"
      logtag=${run//[:.]/_}
      JOBS+=("scripts/train_popgym.py --env-id $occ --target-env-id $clean --mask-occluded-target-loss --memory-mode $design --smt-router sigmoid --seed $seed --fixed-alpha --encoder-type precomputed --train-feature-cache $train_features --val-feature-cache $val_features --feature-manifest $manifest --embed-dim $FEATURE_DIM --prototype-seed 0 --data-dir $DATA_DIR --output-dir $OUT --epochs $EPOCHS --num-episodes $TRAIN_EPS --val-episodes $VAL_EPS --length $LEN --batch-size 64 --num-workers 2 --no-wandb")
      RUN_DIRS+=("$OUT/$run")
      LOGS+=("$LOG_DIR/${logtag}.log")
    done
  done
done

echo "$(date +%T) training ${#JOBS[@]} runs on GPUs [${GPU_LIST[*]}]"
train_worker() {
  local slot=$1 gpu=${GPU_LIST[$1]}
  for idx in "${!JOBS[@]}"; do
    (( idx % WORKERS == slot )) || continue
    local run_dir=${RUN_DIRS[$idx]}
    if [[ -s "$run_dir/model.pt" && -s "$run_dir/metrics.json" ]]; then
      echo "$(date +%T) [gpu $gpu] skip complete $run_dir"
      continue
    fi
    if [[ -e "$run_dir/model.pt" || -e "$run_dir/metrics.json" ]]; then
      echo "partial run exists; refusing to overwrite: $run_dir" >&2
      return 2
    fi
    echo "$(date +%T) [gpu $gpu] >>> $run_dir"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" "$PY" ${JOBS[$idx]} > "${LOGS[$idx]}" 2>&1
    [[ -s "$run_dir/model.pt" && -s "$run_dir/metrics.json" ]] || {
      echo "run completed without required artifacts: $run_dir" >&2
      return 2
    }
    echo "$(date +%T) [gpu $gpu] <<< $run_dir"
  done
}
PIDS=()
for ((slot=0; slot<WORKERS; slot++)); do train_worker "$slot" & PIDS+=("$!"); done
for pid in "${PIDS[@]}"; do wait "$pid"; done

"$PY" scripts/analyze_shared_clean_occlusion.py \
  --root "$OUT" --seeds "${SEED_LIST[@]}" --designs "${DESIGN_LIST[@]}" \
  --num-episodes "$TRAIN_EPS" --val-episodes "$VAL_EPS" --length "$LEN" \
  --epochs "$EPOCHS" --feature-dim "$FEATURE_DIM"
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" scripts/evaluate_shared_mask_generalization.py \
  --root "$OUT" --device cuda --batch-size 64 --num-workers 2
echo "=== SHARED-ENCODER CLEAN-TARGET OCCLUSION STUDY COMPLETE ==="
