#!/usr/bin/env bash
# SMT v2: independent additive sigmoid gates (every fixed-horizon bank can contribute fully,
# input-conditioned) vs the v1 softmax mixture. 4 envs x 3 seeds -> outputs/smt_v2.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; mkdir -p logs outputs/smt_v2
GPU_LIST=(${GPUS:-0 0 1 1 2 2 3 3}); NG=${#GPU_LIST[@]}
JOBS=(); SK=()
for s in 0 1 2; do for env in tmaze distractor recall occlusion; do
  JOBS+=("scripts/train_memory.py --env $env --memory-mode smt --smt-router sigmoid --seed $s --output-dir outputs/smt_v2 --wandb-project lewm-memory-smt --extra-tag exp:smt_v2 --epochs 30 --num-episodes 4000 --batch-size 64")
  SK+=("outputs/smt_v2/lewm-$env-smt-s$s/model.pt")
done; done
echo "=== SMT-v2 jobs: ${#JOBS[@]} ==="
worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${SK[$idx]}" ] && { echo "skip ${SK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] >>> ${SK[$idx]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/smtv2_${idx}.log" 2>&1
  fi; done; }
for s in $(seq 0 $((NG-1))); do worker "$s" & done
wait
echo "=== SMT-v2 COMPLETE ==="; $PY scripts/analyze_runs.py outputs/smt_v2
echo "=== SMT-v2 ANALYSIS COMPLETE ==="
