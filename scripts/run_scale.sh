#!/usr/bin/env bash
# Scale / external-backbone experiments:
#   (DINO) frozen pretrained DINOv2 ViT-S backbone (the DINO-WM backbone) + trained memory:
#          4 memory envs x {none, multi} x 2 seeds  -> outputs/dino, lewm-memory-dino
#   (MMAZE) Memory-Maze 9x9 (3D MuJoCo POMDP): {none, multi} x 3 seeds -> outputs/mmaze, lewm-memory-mmaze
# Packed 2 jobs/GPU (DINOv2 @224 is VRAM-heavier). Pre-warms DINOv2 weights + pre-collects MMAZE.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs/dino outputs/mmaze
GPU_LIST=(${GPUS:-0 0 1 1 2 2 3 3}); NG=${#GPU_LIST[@]}

echo "$(date +%T) pre-warming DINOv2 weights..."
$PY -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=True,num_classes=0); print('dino cached')" 2>&1 | tail -1
echo "$(date +%T) pre-collecting Memory-Maze 9x9..."
MUJOCO_GL=egl $PY - <<'PY' 2>&1 | grep -iE "ok|shape" | tail -2
from lewm.envs.popgym_arcade import get_or_collect
o,_,na=get_or_collect('mmaze:9x9',2000,32,img_size=64,seed=0); get_or_collect('mmaze:9x9',512,32,img_size=64,seed=7777)
print(f"ok mmaze train{o.shape} n_actions={na}")
PY
echo "$(date +%T) data ready"

JOBS=(); SKIPS=()
add(){ JOBS+=("$1"); SKIPS+=("$2"); }
# (DINO) frozen DINOv2 backbone
for s in 0 1; do for env in tmaze occlusion recall distractor; do for d in none multi; do
  add "scripts/train_memory.py --env $env --memory-mode $d --seed $s --encoder dino --fixed-alpha --output-dir outputs/dino --wandb-project lewm-memory-dino --extra-tag exp:dino --epochs 20 --num-episodes 2000 --batch-size 24 --num-workers 2 --eval-interval 20" "outputs/dino/lewm-$env-$d-s$s/model.pt"
done; done; done
# (MMAZE) Memory-Maze 9x9
for s in 0 1 2; do for d in none multi; do
  add "scripts/train_popgym.py --env-id mmaze:9x9 --memory-mode $d --seed $s --fixed-alpha --output-dir outputs/mmaze --wandb-project lewm-memory-mmaze --extra-tag exp:mmaze --epochs 30 --num-episodes 2000 --batch-size 64 --num-workers 2" "outputs/mmaze/lewm-mmaze:9x9-$d-s$s/model.pt"
done; done

echo "=== scale jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
run_worker(){
  local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % NG )) -eq "$slot" ]; then
      if [ -f "${SKIPS[$idx]}" ]; then echo "skip ${SKIPS[$idx]}"; continue; fi
      echo "$(date +%T) [gpu $gpu] >>> ${SKIPS[$idx]}"
      CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/scale_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] <<< idx $idx (exit $?)"
    fi
  done
}
for s in $(seq 0 $((NG-1))); do run_worker "$s" & done
wait
echo "=== SCALE RUNS COMPLETE ==="
$PY scripts/analyze_runs.py outputs/dino
$PY scripts/analyze_runs.py outputs/mmaze
echo "=== SCALE ANALYSIS COMPLETE ==="
