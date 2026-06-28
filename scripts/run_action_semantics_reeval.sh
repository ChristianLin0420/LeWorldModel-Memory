#!/usr/bin/env bash
# Re-evaluate (do not retrain) every canonical DMC/OGBench robotic checkpoint with
# validation rollout seed 7777 and the training-compatible shared prototype seed 0.
#
# Overrides:
#   GPU_IDS="0 1 2 3"  evaluator processes / visible GPUs (one shard per entry)
#   WORKERS=2           DataLoader workers inside each evaluator process
#   PY=.venv/bin/python Python interpreter
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
WORKERS=${WORKERS:-2}
read -r -a GPU_SLOTS <<< "${GPU_IDS:-0 1 2 3}"

OUT=outputs/action_semantics_fix
DATA_DIR=outputs/popgym_data
LOG_DIR=logs/action_semantics_fix
VAL_EPISODES=150
LENGTH=32
PROTOTYPE_SEED=0
ROLLOUT_SEED=7777

if [[ ! "$WORKERS" =~ ^[0-9]+$ ]]; then
  echo "WORKERS must be a non-negative integer, got: $WORKERS" >&2
  exit 2
fi
if (( ${#GPU_SLOTS[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi
for gpu in "${GPU_SLOTS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS entries must be non-negative integers, got: $gpu" >&2
    exit 2
  fi
done

mkdir -p "$OUT" "$DATA_DIR" "$LOG_DIR"

DMC_TASKS=(reacher.hard ball_in_cup.catch finger.spin cheetah.run)
FULL_ENVS=()
OCC_ENVS=()
for task in "${DMC_TASKS[@]}"; do
  FULL_ENVS+=("dmc:$task")
  OCC_ENVS+=("dmc:$task.occ")
done
FULL_ENVS+=("ogbench:cube-single")
OCC_ENVS+=("ogbench:cube-single.occ")

echo "=== validating the exact 60-DMC + 15-OGBench checkpoint factorial ==="
"$PY" scripts/analyze_action_semantics_fix.py --validate-only

# Generate only corrected schema-v3 validation data. Clean rollouts complete first;
# each occluded cache is then derived by masking the exact clean cache.
collect_worker() {
  local slot=$1
  local gpu=${GPU_SLOTS[$slot]}
  local idx env safe log
  for idx in "${!ENVS[@]}"; do
    if (( idx % ${#GPU_SLOTS[@]} != slot )); then
      continue
    fi
    env=${ENVS[$idx]}
    safe=${env//[:.]/_}
    log="$LOG_DIR/collect_${safe}.log"
    echo "$(date +%T) [gpu $gpu] collect $env"
    if ! MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" "$PY" - \
      "$env" "$VAL_EPISODES" "$LENGTH" "$DATA_DIR" \
      "$PROTOTYPE_SEED" "$ROLLOUT_SEED" >"$log" 2>&1 <<'PY'
import sys
from lewm.envs.popgym_arcade import get_or_collect

env = sys.argv[1]
episodes = int(sys.argv[2])
length = int(sys.argv[3])
data_dir = sys.argv[4]
prototype_seed = int(sys.argv[5])
rollout_seed = int(sys.argv[6])
obs, _actions, n_actions = get_or_collect(
    env,
    episodes,
    length,
    img_size=64,
    seed=rollout_seed,
    data_dir=data_dir,
    prototype_seed=prototype_seed,
)
print(
    f"collected {env}: obs={obs.shape}, n_actions={n_actions}, "
    f"rollout_seed={rollout_seed}, prototype_seed={prototype_seed}"
)
PY
    then
      echo "collection failed for $env; inspect $log" >&2
      return 1
    fi
  done
}

collect_phase() {
  local collection_pids=() collection_status=0
  for slot in "${!GPU_SLOTS[@]}"; do
    collect_worker "$slot" &
    collection_pids+=("$!")
  done
  for pid in "${collection_pids[@]}"; do
    if ! wait "$pid"; then collection_status=1; fi
  done
  if (( collection_status != 0 )); then
    echo "one or more corrected validation datasets failed to collect" >&2
    exit 1
  fi
}
ENVS=("${FULL_ENVS[@]}")
echo "=== collecting ${#ENVS[@]} clean schema-v3 validation datasets ==="
collect_phase
ENVS=("${OCC_ENVS[@]}")
echo "=== deriving ${#ENVS[@]} exact paired-occlusion validation datasets ==="
collect_phase

echo "=== re-evaluating 75 saved checkpoints on ${#GPU_SLOTS[@]} GPU shard(s) ==="
evaluation_pids=()
for slot in "${!GPU_SLOTS[@]}"; do
  gpu=${GPU_SLOTS[$slot]}
  log="$LOG_DIR/eval_shard_${slot}_gpu_${gpu}.log"
  echo "$(date +%T) [gpu $gpu] evaluator shard $slot/${#GPU_SLOTS[@]}"
  (
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" "$PY" \
      scripts/analyze_action_semantics_fix.py \
      --evaluate-shard \
      --shard-index "$slot" \
      --num-shards "${#GPU_SLOTS[@]}" \
      --device cuda \
      --batch-size 64 \
      --data-workers "$WORKERS" \
      --data-dir "$DATA_DIR" \
      --output-root "$OUT"
  ) >"$log" 2>&1 &
  evaluation_pids+=("$!")
done
evaluation_status=0
for pid in "${evaluation_pids[@]}"; do
  if ! wait "$pid"; then
    evaluation_status=1
  fi
done
if (( evaluation_status != 0 )); then
  echo "one or more re-evaluation shards failed; inspect $LOG_DIR/eval_shard_*" >&2
  exit 1
fi

echo "=== validating and aggregating fresh re-evaluation results ==="
"$PY" scripts/analyze_action_semantics_fix.py \
  --aggregate-only \
  --data-dir "$DATA_DIR" \
  --output-root "$OUT"

echo "=== corrected action-semantics re-evaluation complete: $OUT ==="
