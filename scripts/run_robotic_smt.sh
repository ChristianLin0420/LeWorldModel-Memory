#!/usr/bin/env bash
# Real-robot SMT (exp #5): learnable smt(sigmoid) vs the existing none/multi on the cached
# occlusion data (dm_control 4 robots + OGBench cube). Does learned write-gating beat the
# fixed K-bank at bridging the blackout? Reuses cached npz (no collection). GPUs avoid 0.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; mkdir -p logs
GPU_LIST=(${GPUS:-1 1 2 2 3 3}); NG=${#GPU_LIST[@]}
JOBS=(); SK=()
add(){ JOBS+=("$1"); SK+=("$2"); }
for s in 0 1 2; do
  for env in dmc:reacher.hard.occ dmc:ball_in_cup.catch.occ dmc:finger.spin.occ dmc:cheetah.run.occ; do
    add "scripts/train_popgym.py --env-id $env --memory-mode smt --smt-router sigmoid --seed $s --fixed-alpha --output-dir outputs/robotic --wandb-project lewm-memory-robotic --extra-tag exp:robotic_smt --epochs 30 --num-episodes 600 --val-episodes 150 --length 32 --batch-size 64 --num-workers 2" \
        "outputs/robotic/lewm-$env-smt-s$s/model.pt"
  done
  add "scripts/train_popgym.py --env-id ogbench:cube-single.occ --memory-mode smt --smt-router sigmoid --seed $s --fixed-alpha --output-dir outputs/ogbench --wandb-project lewm-memory-ogbench --extra-tag exp:ogbench_smt --epochs 30 --num-episodes 600 --val-episodes 150 --length 32 --batch-size 64 --num-workers 2" \
      "outputs/ogbench/lewm-ogbench:cube-single.occ-smt-s$s/model.pt"
done
echo "=== robotic-SMT jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${SK[$idx]}" ] && { echo "skip ${SK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] >>> ${SK[$idx]}"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/robsmt_${idx}.log" 2>&1
  fi; done; }
for s in $(seq 0 $((NG-1))); do worker "$s" & done
wait
echo "=== ROBOTIC-SMT COMPLETE ==="
