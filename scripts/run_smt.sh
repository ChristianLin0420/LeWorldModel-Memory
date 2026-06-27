#!/usr/bin/env bash
# Validation sweep for the Selective Multi-Timescale (SMT) memory (docs/LEARNABLE_MEMORY.md):
#   vanilla 'none' vs fixed 'multi' (K-bank) vs learnable 'smt'  on the 4 memory envs x 3 seeds.
# Hypothesis: smt matches multi on clean long-gap tasks and helps most on Distractor (write gate
# suppresses distractors) and Recall (router switches horizons). -> outputs/smt, wandb lewm-memory-smt.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs/smt
GPU_LIST=(${GPUS:-0 0 1 1 2 2 3 3}); NG=${#GPU_LIST[@]}

JOBS=(); SK=()
for s in 0 1 2; do for env in tmaze distractor recall occlusion; do for d in none multi smt; do
  JOBS+=("scripts/train_memory.py --env $env --memory-mode $d --seed $s --output-dir outputs/smt --wandb-project lewm-memory-smt --extra-tag exp:smt --epochs 30 --num-episodes 4000 --batch-size 64")
  SK+=("outputs/smt/lewm-$env-$d-s$s/model.pt")
done; done; done
echo "=== SMT jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${SK[$idx]}" ] && { echo "skip ${SK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] >>> ${SK[$idx]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/smt_${idx}.log" 2>&1
    echo "$(date +%T) [gpu $gpu] <<< $idx (exit $?)"
  fi; done; }
for s in $(seq 0 $((NG-1))); do worker "$s" & done
wait
echo "=== SMT RUNS COMPLETE ==="
$PY scripts/analyze_runs.py outputs/smt
echo "=== SMT ANALYSIS COMPLETE ==="
