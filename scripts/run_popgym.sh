#!/usr/bin/env bash
# POPGym Arcade benchmark: vanilla LeWM (none) vs two-timescale memory (short/long/both)
# on memory-centric POMDPs. Pre-collects data once (JAX/CPU) so training procs never import
# JAX, then runs the matrix round-robin over a GPU list (default 1 2 3 -> avoid a shared GPU0).
# wandb project: lewm-memory-popgym.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs/popgym
ENVS=${ENVS:-"CountRecallEasy AutoEncodeEasy"}
DESIGNS=${DESIGNS:-"none short long both"}
SEEDS=${SEEDS:-"0 1 2"}
GPU_LIST=(${GPUS:-1 2 3}); NG=${#GPU_LIST[@]}
EPOCHS=${EPOCHS:-30}; NUM_EPISODES=${NUM_EPISODES:-4000}; BATCH=${BATCH:-64}; NW=${NW:-4}
COMMON="--epochs $EPOCHS --num-episodes $NUM_EPISODES --batch-size $BATCH --num-workers $NW --fixed-alpha --tau-fast 3 --tau-slow 25"

# 1) pre-collect train (seed 0) + val (seed 7777) data once per env (JAX on CPU)
echo "$(date +%T) pre-collecting POPGym data for: $ENVS"
for env in $ENVS; do
  JAX_PLATFORMS=cpu $PY - "$env" "$NUM_EPISODES" <<'PY'
import sys
from lewm.envs.popgym_arcade import get_or_collect
env, n = sys.argv[1], int(sys.argv[2])
o,_,na = get_or_collect(env, n, 32, img_size=64, seed=0)
get_or_collect(env, 512, 32, img_size=64, seed=7777)
print(f"  collected {env}: train{o.shape} n_actions={na}")
PY
done
echo "$(date +%T) data ready"

# 2) job list
JOBS=()
for seed in $SEEDS; do for env in $ENVS; do for d in $DESIGNS; do
  JOBS+=("--env-id $env --memory-mode $d --seed $seed")
done; done; done
echo "=== popgym jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="

run_worker () {
  local slot=$1
  local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % NG )) -eq "$slot" ]; then
      local args="${JOBS[$idx]}" name
      name=$(sed -E 's/.*--env-id ([A-Za-z0-9]+) --memory-mode ([a-z]+) --seed ([0-9]+).*/\1-\2-s\3/' <<<"$args")
      if [ -f "outputs/popgym/lewm-${name}/model.pt" ]; then echo "skip lewm-$name"; continue; fi
      echo "$(date +%T) [gpu $gpu] >>> lewm-$name"
      JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_popgym.py $args $COMMON > "logs/popgym_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] <<< lewm-$name (exit $?)"
    fi
  done
}
for s in $(seq 0 $((NG - 1))); do run_worker "$s" & done
wait
echo "=== POPGYM RUNS COMPLETE ==="
$PY scripts/aggregate_popgym.py
echo "=== POPGYM ANALYSIS COMPLETE ==="
