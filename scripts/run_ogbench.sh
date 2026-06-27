#!/usr/bin/env bash
# OGBench-Cube manipulation (the goal-conditioned benchmark used by DINO-WM / LeWorldModel).
# Same protocol as the dm_control robotic sweep (§5.15):
#   cube-single  : full-obs Markovian control (is memory inert?)
#   cube-single.occ : mid-episode blackout the arm manipulates through (does memory bridge it?)
# {none, multi} x 3 seeds -> outputs/ogbench, wandb lewm-memory-ogbench. npz -> train_popgym.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs/ogbench
GPU_LIST=(${GPUS:-0 0 1 1 2 2 3 3}); NG=${#GPU_LIST[@]}
TRAIN_EPS=600; VAL_EPS=150; LEN=32
ENVS=(ogbench:cube-single ogbench:cube-single.occ)

# ---- phase 0: pre-collect (distinct files -> safe to parallelize) ----
echo "$(date +%T) pre-collecting ${#ENVS[@]} OGBench datasets (train $TRAIN_EPS / val $VAL_EPS)"
collect_one(){ local env=$1
  MUJOCO_GL=egl $PY - "$env" "$TRAIN_EPS" "$VAL_EPS" "$LEN" <<'PY' > "logs/ogbcollect_$(echo $1|tr ':.' '__').log" 2>&1
import sys
from lewm.envs.popgym_arcade import get_or_collect
env, tr, va, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
o,_,n = get_or_collect(env, tr, L, img_size=64, seed=0)
get_or_collect(env, va, L, img_size=64, seed=7777)
print(f"collected {env} train{o.shape} n_actions={n}")
PY
}
for env in "${ENVS[@]}"; do collect_one "$env" & done
wait
echo "$(date +%T) collection done: $(grep -l collected logs/ogbcollect_*.log 2>/dev/null | wc -l)/${#ENVS[@]} ok"

# ---- phase 1: training ----
JOBS=(); SK=()
for s in 0 1 2; do for env in "${ENVS[@]}"; do for d in none multi; do
  JOBS+=("scripts/train_popgym.py --env-id $env --memory-mode $d --seed $s --fixed-alpha --output-dir outputs/ogbench --wandb-project lewm-memory-ogbench --extra-tag exp:ogbench --epochs 30 --num-episodes $TRAIN_EPS --val-episodes $VAL_EPS --length $LEN --batch-size 64 --num-workers 2")
  SK+=("outputs/ogbench/lewm-$env-$d-s$s/model.pt")
done; done; done
echo "=== ogbench jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${SK[$idx]}" ] && { echo "skip ${SK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] >>> ${SK[$idx]}"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/ogbench_${idx}.log" 2>&1
    echo "$(date +%T) [gpu $gpu] <<< $idx (exit $?)"
  fi; done; }
for s in $(seq 0 $((NG-1))); do worker "$s" & done
wait
echo "=== OGBENCH RUNS COMPLETE ==="

# ---- phase 2: aggregate ----
$PY - <<'PY'
import json, glob, statistics as st
from collections import defaultdict
agg=defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob('outputs/ogbench/*/metrics.json')):
    d=json.load(open(f)); name=f.split('/')[-2]
    design='multi' if '-multi-' in name else 'none'
    env=name.replace('lewm-','').rsplit('-'+design,1)[0]
    for k in ('val_pred_loss','infl_slow'):
        if k in d: agg[(env,design)][k].append(d[k])
print(f"\n{'env':<28}{'design':<7}{'val_mse':>9}{'infl':>7}{'n':>3}")
for (env,d),m in sorted(agg.items()):
    vm=st.mean(m['val_pred_loss']) if m['val_pred_loss'] else float('nan')
    inf=st.mean(m['infl_slow']) if m['infl_slow'] else float('nan')
    print(f"{env:<28}{d:<7}{vm:>9.4f}{inf:>7.3f}{len(m['val_pred_loss']):>3}")
PY
echo "=== OGBENCH AGGREGATION COMPLETE ==="
