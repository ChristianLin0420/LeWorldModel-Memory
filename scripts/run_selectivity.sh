#!/usr/bin/env bash
# Selectivity experiments (docs/LEARNABLE_MEMORY.md §6): does SMT's LEARNABLE input-conditioned
# gating buy something the fixed K-bank cannot, as the task gets harder?
#   Distractor-hard: more interference flashes (n_distract 10, 16) -> the WRITE GATE should learn
#                    to not store distractors -> smt should beat fixed multi at high interference.
#   Recall-hard:     longer colour sequence (seq_len 5, 7) -> more interfering symbols to hold.
# none vs multi(fixed) vs smt(sigmoid, v2) x 3 seeds. -> outputs/sel, wandb lewm-memory-smt.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; mkdir -p logs outputs/sel
GPU_LIST=(${GPUS:-0 0 1 1 2 2 3 3}); NG=${#GPU_LIST[@]}
COMMON="--output-dir outputs/sel --wandb-project lewm-memory-smt --extra-tag exp:selectivity --epochs 30 --num-episodes 4000 --batch-size 64"

JOBS=(); SK=()
add(){ JOBS+=("$1"); SK+=("$2"); }
for s in 0 1 2; do
  for nd in 10 16; do for d in none multi smt; do
    R="--smt-router sigmoid"; [ "$d" = smt ] || R=""
    add "scripts/train_memory.py --env distractor --n-distract $nd --run-suffix nd$nd --memory-mode $d $R --seed $s $COMMON" \
        "outputs/sel/lewm-distractor-$d-s$s-nd$nd/model.pt"
  done; done
  for sl in 5 7; do for d in none multi smt; do
    R="--smt-router sigmoid"; [ "$d" = smt ] || R=""
    add "scripts/train_memory.py --env recall --seq-len $sl --run-suffix sl$sl --memory-mode $d $R --seed $s $COMMON" \
        "outputs/sel/lewm-recall-$d-s$s-sl$sl/model.pt"
  done; done
done
echo "=== selectivity jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${SK[$idx]}" ] && { echo "skip ${SK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] >>> ${SK[$idx]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/sel_${idx}.log" 2>&1
  fi; done; }
for s in $(seq 0 $((NG-1))); do worker "$s" & done
wait
echo "=== SELECTIVITY RUNS COMPLETE ==="; $PY scripts/analyze_runs.py outputs/sel
echo "=== SELECTIVITY ANALYSIS COMPLETE ==="
