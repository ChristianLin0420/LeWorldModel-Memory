#!/usr/bin/env bash
# Experiments (1)-(4) for the ICLR study, distributed round-robin across 4 GPUs.
#   (1) >=3 seeds (runs seeds 1,2; seed 0 already exists) -> error bars
#   (2) tau_slow sweep on tmaze -> usage tracks the gap Delta
#   (3) learnable tau on the 4 envs -> discovered horizons
#   (4) tmaze gap sweep (none vs both) -> finite-window cliff vs graceful memory
# After all runs: analyze_runs.py (master CSV) + plot_experiments.py (figures).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs
EPOCHS=${EPOCHS:-30}; NUM_EPISODES=${NUM_EPISODES:-5000}; BATCH=${BATCH:-64}
NW=${NW:-4}; EVAL=${EVAL:-30}
COMMON="--epochs $EPOCHS --num-episodes $NUM_EPISODES --batch-size $BATCH --num-workers $NW --eval-interval $EVAL"

JOBS=()
# (1) seeds 1,2 over the full matrix + tworoom control
for seed in 1 2; do
  for env in tmaze occlusion recall distractor; do
    for d in none short long both; do
      JOBS+=("--env $env --memory-mode $d --seed $seed --fixed-alpha --tau-fast 3 --tau-slow 25 --extra-tag exp:seeds")
    done
  done
  for d in none both; do
    JOBS+=("--env tworoom --memory-mode $d --seed $seed --fixed-alpha --tau-fast 3 --tau-slow 25 --extra-tag exp:seeds")
  done
done
# (2) tau_slow sweep (tmaze, both, gap Delta~21)
for ts in 3 6 12 21 30 45; do
  JOBS+=("--env tmaze --memory-mode both --seed 0 --fixed-alpha --tau-fast 3 --tau-slow $ts --run-suffix tslow$ts --extra-tag exp:tau_slow_sweep")
done
# (3) learnable tau (omit --fixed-alpha) on the 4 envs
for env in tmaze occlusion recall distractor; do
  JOBS+=("--env $env --memory-mode both --seed 0 --tau-fast 3 --tau-slow 25 --run-suffix learnable --extra-tag exp:learnable_tau")
done
# (4) gap sweep on tmaze (none vs both), length 44 to reach large Delta
for rev in 6 12 18 24 30 36 42; do
  for d in none both; do
    JOBS+=("--env tmaze --memory-mode $d --seed 0 --fixed-alpha --tau-fast 3 --tau-slow 25 --reveal $rev --cue-len 3 --length 44 --run-suffix gap$rev --extra-tag exp:gap_sweep")
  done
done

echo "=== total jobs: ${#JOBS[@]} (across 4 GPUs) ==="

run_worker () {
  local gpu=$1
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % 4 )) -eq "$gpu" ]; then
      echo "$(date +%T) [gpu $gpu] job $idx >>> ${JOBS[$idx]}"
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_memory.py ${JOBS[$idx]} $COMMON > "logs/exp_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] job $idx <<< done (exit $?)"
    fi
  done
}

for g in 0 1 2 3; do run_worker "$g" & done
wait
echo "=== ALL EXPERIMENT RUNS COMPLETE ==="
$PY scripts/analyze_runs.py
$PY scripts/plot_experiments.py
echo "=== ANALYSIS COMPLETE ==="
