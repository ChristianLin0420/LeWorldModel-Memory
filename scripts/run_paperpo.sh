#!/usr/bin/env bash
# Partially-observable variants of the LeWorldModel paper's envs (Two-Room, Reacher, Push-T,
# OGBench-Cube), each goal-cued: vanilla LeWM (none) vs two-timescale memory (short/long/both).
# Reuses the full pipeline incl. availability/usage cue probes. wandb: lewm-memory-paperpo;
# output dir outputs/paperpo (no collision with outputs/mem or outputs/4ens).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUTDIR=outputs/paperpo
mkdir -p logs "$OUTDIR"
EPOCHS=${EPOCHS:-30}; NUM_EPISODES=${NUM_EPISODES:-5000}; BATCH=${BATCH:-64}
NW=${NW:-4}; EVAL=${EVAL:-10}; SEEDS=${SEEDS:-"0 1 2"}
PROJECT=${PROJECT:-lewm-memory-paperpo}
ENVS=${ENVS:-"tworoom_po reacher_po pusht_po cube_po"}
DESIGNS=${DESIGNS:-"none short long both"}
GPU_LIST=(${GPUS:-0 1 2 3}); NG=${#GPU_LIST[@]}
COMMON="--epochs $EPOCHS --num-episodes $NUM_EPISODES --batch-size $BATCH --num-workers $NW \
--eval-interval $EVAL --fixed-alpha --tau-fast 3 --tau-slow 25 --wandb-project $PROJECT --output-dir $OUTDIR"

JOBS=()
for seed in $SEEDS; do for env in $ENVS; do for d in $DESIGNS; do
  JOBS+=("--env $env --memory-mode $d --seed $seed --extra-tag exp:paperpo")
done; done; done
echo "=== paperpo jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="

run_worker () {
  local slot=$1
  local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % NG )) -eq "$slot" ]; then
      local args="${JOBS[$idx]}" env d seed name
      env=$(sed -E 's/.*--env ([a-z_]+).*/\1/' <<<"$args")
      d=$(sed -E 's/.*--memory-mode ([a-z]+).*/\1/' <<<"$args")
      seed=$(sed -E 's/.*--seed ([0-9]+).*/\1/' <<<"$args")
      name="lewm-${env}-${d}-s${seed}"
      if [ -f "$OUTDIR/${name}/model.pt" ]; then echo "skip $name"; continue; fi
      echo "$(date +%T) [gpu $gpu] >>> $name"
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_memory.py $args $COMMON > "logs/paperpo_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] <<< $name (exit $?)"
    fi
  done
}
for s in $(seq 0 $((NG - 1))); do run_worker "$s" & done
wait
echo "=== PAPERPO RUNS COMPLETE ==="
$PY scripts/analyze_runs.py "$OUTDIR"
echo "=== PAPERPO ANALYSIS COMPLETE ($OUTDIR/master_metrics.csv) ==="
