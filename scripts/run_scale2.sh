#!/usr/bin/env bash
# Scale experiments, restructured: Memory-Maze collection (slow single-env MuJoCo) runs in the
# BACKGROUND while the DINOv2 jobs train; Memory-Maze training runs after DINO frees the GPUs.
#   (DINO)  frozen DINOv2 ViT-S backbone: 4 envs x {none,multi} x 2 seeds -> outputs/dino
#   (MMAZE) Memory-Maze 9x9 (3D): {none,multi} x 3 seeds (400 collected episodes) -> outputs/mmaze
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs/dino outputs/mmaze
GPU_LIST=(${GPUS:-0 0 1 1 2 2 3 3}); NG=${#GPU_LIST[@]}

echo "$(date +%T) pre-warm DINOv2 + start Memory-Maze collection in background (400 eps)"
$PY -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=True,num_classes=0)" >/dev/null 2>&1
( MUJOCO_GL=egl $PY - <<'PY' > logs/mmaze_collect.log 2>&1
from lewm.envs.popgym_arcade import get_or_collect
get_or_collect('mmaze:9x9',400,32,img_size=64,seed=0); get_or_collect('mmaze:9x9',150,32,img_size=64,seed=7777)
print("mmaze collected")
PY
) &
COLLECT_PID=$!

# ---- phase 1: DINO jobs (run while mmaze collects) ----
DJOBS=(); DSK=()
for s in 0 1; do for env in tmaze occlusion recall distractor; do for d in none multi; do
  DJOBS+=("scripts/train_memory.py --env $env --memory-mode $d --seed $s --encoder dino --fixed-alpha --output-dir outputs/dino --wandb-project lewm-memory-dino --extra-tag exp:dino --epochs 20 --num-episodes 2000 --batch-size 24 --num-workers 2 --eval-interval 20")
  DSK+=("outputs/dino/lewm-$env-$d-s$s/model.pt")
done; done; done
echo "=== DINO jobs: ${#DJOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
dino_worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!DJOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${DSK[$idx]}" ] && { echo "skip ${DSK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] DINO>>> ${DSK[$idx]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY ${DJOBS[$idx]} > "logs/dino_${idx}.log" 2>&1
    echo "$(date +%T) [gpu $gpu] DINO<<< $idx (exit $?)"
  fi; done; }
for s in $(seq 0 $((NG-1))); do dino_worker "$s" & done
wait
echo "=== DINO COMPLETE; waiting for mmaze collection ==="
wait $COLLECT_PID; cat logs/mmaze_collect.log | tail -1

# ---- phase 2: Memory-Maze jobs ----
MJOBS=(); MSK=()
for s in 0 1 2; do for d in none multi; do
  MJOBS+=("scripts/train_popgym.py --env-id mmaze:9x9 --memory-mode $d --seed $s --fixed-alpha --output-dir outputs/mmaze --wandb-project lewm-memory-mmaze --extra-tag exp:mmaze --epochs 30 --num-episodes 400 --val-episodes 150 --batch-size 64 --num-workers 2")
  MSK+=("outputs/mmaze/lewm-mmaze:9x9-$d-s$s/model.pt")
done; done
echo "=== MMAZE jobs: ${#MJOBS[@]} ==="
mmaze_worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!MJOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${MSK[$idx]}" ] && { echo "skip ${MSK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] MMAZE>>> ${MSK[$idx]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY ${MJOBS[$idx]} > "logs/mmaze_${idx}.log" 2>&1
    echo "$(date +%T) [gpu $gpu] MMAZE<<< $idx (exit $?)"
  fi; done; }
for s in $(seq 0 $((NG-1))); do mmaze_worker "$s" & done
wait
echo "=== SCALE2 RUNS COMPLETE ==="
$PY scripts/analyze_runs.py outputs/dino
$PY scripts/analyze_runs.py outputs/mmaze
echo "=== SCALE2 ANALYSIS COMPLETE ==="
