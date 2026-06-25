#!/usr/bin/env bash
# Bump sweeps (2) tau_slow, (3) learnable tau, (4) gap -- to 3 seeds (runs seeds 1 & 2).
# Distributes round-robin over a configurable GPU list (GPUS env, default "0 1 2 3");
# set GPUS="1 2 3" to avoid a GPU another user is on. Skips runs whose model.pt already
# exists (so it resumes cleanly after an interruption). Re-analyzes + re-plots at the end.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

# wait for the 4ens batch if it is still running (returns instantly otherwise)
while pgrep -f run_4ens.sh >/dev/null 2>&1; do sleep 30; done

EPOCHS=${EPOCHS:-30}; NUM_EPISODES=${NUM_EPISODES:-5000}; BATCH=${BATCH:-64}; NW=${NW:-4}; EVAL=${EVAL:-30}
GPU_LIST=(${GPUS:-0 1 2 3}); NG=${#GPU_LIST[@]}
COMMON="--epochs $EPOCHS --num-episodes $NUM_EPISODES --batch-size $BATCH --num-workers $NW --eval-interval $EVAL"

JOBS=()
for seed in 1 2; do
  for ts in 3 6 12 21 30 45; do
    JOBS+=("--env tmaze --memory-mode both --seed $seed --fixed-alpha --tau-fast 3 --tau-slow $ts --run-suffix tslow$ts --extra-tag exp:tau_slow_sweep")
  done
  for env in tmaze occlusion recall distractor; do
    JOBS+=("--env $env --memory-mode both --seed $seed --tau-fast 3 --tau-slow 25 --run-suffix learnable --extra-tag exp:learnable_tau")
  done
  for rev in 6 12 18 24 30 36 42; do
    for d in none both; do
      JOBS+=("--env tmaze --memory-mode $d --seed $seed --fixed-alpha --tau-fast 3 --tau-slow 25 --reveal $rev --cue-len 3 --length 44 --run-suffix gap$rev --extra-tag exp:gap_sweep")
    done
  done
done
echo "=== sweep-seed jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="

run_worker () {
  local slot=$1
  local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % NG )) -eq "$slot" ]; then
      local args="${JOBS[$idx]}" env mode seed suf name
      env=$(sed -E 's/.*--env ([a-z]+).*/\1/' <<<"$args")
      mode=$(sed -E 's/.*--memory-mode ([a-z]+).*/\1/' <<<"$args")
      seed=$(sed -E 's/.*--seed ([0-9]+).*/\1/' <<<"$args")
      suf=$(sed -nE 's/.*--run-suffix ([^ ]+).*/\1/p' <<<"$args")
      name="lewm-${env}-${mode}-s${seed}${suf:+-$suf}"
      if [ -f "outputs/mem/${name}/model.pt" ]; then
        echo "$(date +%T) [gpu $gpu] skip $name (already done)"; continue
      fi
      echo "$(date +%T) [gpu $gpu] >>> $name"
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_memory.py $args $COMMON > "logs/sweepseed_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] <<< $name (exit $?)"
    fi
  done
}
for s in $(seq 0 $((NG - 1))); do run_worker "$s" & done
wait
echo "=== SWEEP-SEED RUNS COMPLETE ==="
$PY scripts/analyze_runs.py
$PY scripts/plot_experiments.py
echo "=== SWEEP-SEED ANALYSIS COMPLETE (3-seed figures regenerated) ==="
