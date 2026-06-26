#!/usr/bin/env bash
# Reviewer follow-ups (a),(b),(c) run concurrently in one packed 12-way queue (3 jobs/GPU):
#   (a) stronger baselines: learned SSM/RetNet-lite (`ssm`) + episodic retrieval (`retrieval`)
#       on the 4 memory envs x 3 seeds  -> outputs/4ens, lewm-memory-4ens
#   (b) broader standard benchmark: 5 POPGym Arcade tasks x {none, multi} x 5 seeds
#       -> outputs/popgym, lewm-memory-popgym
#   (c) frozen-backbone: memory on a frozen pretrained (vanilla `none`) encoder,
#       4 envs x {none, both, multi} x 3 seeds -> outputs/frozen, lewm-memory-frozen
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs/frozen outputs/popgym
GPU_LIST=(${GPUS:-0 0 0 1 1 1 2 2 2 3 3 3}); NG=${#GPU_LIST[@]}
COMMON="--epochs 30 --num-episodes 5000 --batch-size 64 --num-workers 2 --eval-interval 30 --fixed-alpha"
PG="--epochs 30 --num-episodes 4000 --batch-size 64 --num-workers 2 --fixed-alpha"

# pre-collect POPGym data once (cached ones skip instantly)
echo "$(date +%T) pre-collecting POPGym data..."
for env in CountRecallEasy AutoEncodeEasy BattleShipEasy MineSweeperEasy NavigatorEasy; do
  JAX_PLATFORMS=cpu $PY - "$env" <<'PY'
import sys; from lewm.envs.popgym_arcade import get_or_collect
e=sys.argv[1]; get_or_collect(e,4000,32,img_size=64,seed=0); get_or_collect(e,512,32,img_size=64,seed=7777); print("ok",e)
PY
done
echo "$(date +%T) data ready"

JOBS=(); SKIPS=()
add(){ JOBS+=("$1"); SKIPS+=("$2"); }
# (a) ssm + retrieval
for s in 0 1 2; do for env in tmaze occlusion recall distractor; do for d in ssm retrieval; do
  add "scripts/train_memory.py --env $env --memory-mode $d --seed $s --output-dir outputs/4ens --wandb-project lewm-memory-4ens --extra-tag exp:$d $COMMON" "outputs/4ens/lewm-$env-$d-s$s/model.pt"
done; done; done
# (c) frozen encoder (from vanilla none-s0 of each env)
for s in 0 1 2; do for env in tmaze occlusion recall distractor; do for d in none both multi; do
  add "scripts/train_memory.py --env $env --memory-mode $d --seed $s --freeze-encoder --init-from outputs/4ens/lewm-$env-none-s0/model.pt --output-dir outputs/frozen --wandb-project lewm-memory-frozen --extra-tag exp:frozen $COMMON" "outputs/frozen/lewm-$env-$d-s$s/model.pt"
done; done; done
# (b) broader POPGym: 5 tasks x {none, multi} x 5 seeds
for s in 0 1 2 3 4; do for env in CountRecallEasy AutoEncodeEasy BattleShipEasy MineSweeperEasy NavigatorEasy; do for d in none multi; do
  add "scripts/train_popgym.py --env-id $env --memory-mode $d --seed $s --output-dir outputs/popgym --wandb-project lewm-memory-popgym --extra-tag exp:popgym_broad $PG" "outputs/popgym/lewm-$env-$d-s$s/model.pt"
done; done; done

echo "=== a/b/c jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
run_worker(){
  local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % NG )) -eq "$slot" ]; then
      if [ -f "${SKIPS[$idx]}" ]; then echo "skip ${SKIPS[$idx]}"; continue; fi
      echo "$(date +%T) [gpu $gpu] >>> ${SKIPS[$idx]}"
      JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/abc_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] <<< idx $idx (exit $?)"
    fi
  done
}
for s in $(seq 0 $((NG-1))); do run_worker "$s" & done
wait
echo "=== A/B/C RUNS COMPLETE ==="
$PY scripts/analyze_runs.py outputs/4ens
$PY scripts/analyze_runs.py outputs/frozen
$PY scripts/aggregate_popgym.py
echo "=== A/B/C ANALYSIS COMPLETE ==="
