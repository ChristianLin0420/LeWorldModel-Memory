#!/usr/bin/env bash
# Launch the full {env} x {design} ablation matrix for LeWM-Memory across 4 GPUs.
#
#   one memory-stressing env per GPU (run in parallel):
#       GPU0: tmaze        (long-term recall)        + tworoom control afterwards
#       GPU1: occlusion    (short-term / permanence)
#       GPU2: recall       (sequential / mixed)
#       GPU3: distractor   (long-term + interference)
#   each env trains 4 designs sequentially: none / short / long / both
#
# wandb: project=lewm-memory, group=<env>, job_type=<design>,
#        tags=[env:<env>, design:<design>, kind:<memory-kind>, lewm-memory].
#
# Override anything via env vars, e.g.:  EPOCHS=50 NUM_EPISODES=8000 bash scripts/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs

EPOCHS=${EPOCHS:-30}
NUM_EPISODES=${NUM_EPISODES:-5000}
BATCH=${BATCH:-64}
SEED=${SEED:-0}
TAU_FAST=${TAU_FAST:-3}
TAU_SLOW=${TAU_SLOW:-25}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
NUM_WORKERS=${NUM_WORKERS:-4}
DESIGNS=${DESIGNS:-"none short long both"}

run_one () {
  local env=$1 gpu=$2 d=$3
  local name="lewm-${env}-${d}-s${SEED}"
  echo "$(date +%T) [gpu $gpu] >>> $name"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_memory.py \
    --env "$env" --memory-mode "$d" --seed "$SEED" \
    --epochs "$EPOCHS" --num-episodes "$NUM_EPISODES" --batch-size "$BATCH" \
    --num-workers "$NUM_WORKERS" --eval-interval "$EVAL_INTERVAL" \
    --fixed-alpha --tau-fast "$TAU_FAST" --tau-slow "$TAU_SLOW" \
    > "logs/${name}.log" 2>&1
  echo "$(date +%T) [gpu $gpu] <<< $name done (exit $?)"
}

run_seq () {  # env gpu design1 design2 ...
  local env=$1 gpu=$2; shift 2
  for d in "$@"; do run_one "$env" "$gpu" "$d"; done
}

echo "=== LeWM-Memory matrix | epochs=$EPOCHS episodes=$NUM_EPISODES tau=($TAU_FAST,$TAU_SLOW) ==="

# one background group per GPU
( run_seq tmaze      0 $DESIGNS; run_seq tworoom 0 none both ) &  # +Markovian control
( run_seq occlusion  1 $DESIGNS ) &
( run_seq recall     2 $DESIGNS ) &
( run_seq distractor 3 $DESIGNS ) &

wait
echo "=== ALL RUNS COMPLETE ==="
$PY scripts/aggregate_results.py || true
