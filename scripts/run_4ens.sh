#!/usr/bin/env bash
# Clean vanilla-vs-memory comparison on the 4 memory envs, logged to a dedicated
# wandb project (lewm-memory-4ens), 3 seeds, with probe_cue_over_time figures recorded.
#
#   vanilla LeWorldModel = memory_mode "none" (memoryless, short 3-frame window)
#   ours                 = memory_mode short / long / both (two-timescale EMA)
#
# Outputs go to outputs/4ens (NOT outputs/mem) to avoid colliding with the sweep runs.
# Run-name convention is unchanged: lewm-<env>-<design>-s<seed>.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUTDIR=outputs/4ens
mkdir -p logs "$OUTDIR"
EPOCHS=${EPOCHS:-30}; NUM_EPISODES=${NUM_EPISODES:-5000}; BATCH=${BATCH:-64}
NW=${NW:-4}; EVAL=${EVAL:-10}; SEEDS=${SEEDS:-"0 1 2"}
PROJECT=${PROJECT:-lewm-memory-4ens}
COMMON="--epochs $EPOCHS --num-episodes $NUM_EPISODES --batch-size $BATCH --num-workers $NW \
--eval-interval $EVAL --fixed-alpha --tau-fast 3 --tau-slow 25 \
--wandb-project $PROJECT --output-dir $OUTDIR"

JOBS=()
for seed in $SEEDS; do
  for env in tmaze occlusion recall distractor; do
    for d in none short long both; do
      JOBS+=("--env $env --memory-mode $d --seed $seed --extra-tag exp:4ens")
    done
  done
done
echo "=== lewm-memory-4ens: ${#JOBS[@]} jobs (4 envs x {none,short,long,both} x seeds [$SEEDS]) ==="

run_worker () {
  local gpu=$1
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % 4 )) -eq "$gpu" ]; then
      local name; name=$(echo "${JOBS[$idx]}" | sed -E 's/.*--env ([a-z]+) --memory-mode ([a-z]+) --seed ([0-9]+).*/\1-\2-s\3/')
      echo "$(date +%T) [gpu $gpu] >>> $name"
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_memory.py ${JOBS[$idx]} $COMMON > "logs/4ens_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] <<< $name (exit $?)"
    fi
  done
}
for g in 0 1 2 3; do run_worker "$g" & done
wait
echo "=== 4ens RUNS COMPLETE ==="
$PY scripts/analyze_runs.py "$OUTDIR"
echo "=== 4ens ANALYSIS COMPLETE ($OUTDIR/master_metrics.csv) ==="
