#!/usr/bin/env bash
# Refine the OC-SMT L0 phase transition at weights below the coarse
# {0, 0.05, 0.2} sweep. This is a one-seed pilot; promote useful operating
# points to the full four-environment, three-seed sweep only after they show
# non-trivial cardinality (strictly between 0 and M).
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
ENV=${ENV:-tmaze}
SEED=${SEED:-0}
OUT=${OUT:-outputs/ocsmt_refine}
LAMBDAS=(${LAMBDAS:-0.0003 0.0004 0.0005 0.0006 0.0008 0.001 0.003 0.01})
GPUS=(${GPUS:-0 1 2 3})
WORKERS=${WORKERS:-6}

mkdir -p "$OUT" logs

pids=()
for idx in "${!LAMBDAS[@]}"; do
  lam=${LAMBDAS[$idx]}
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
  run="lewm-${ENV}-ocsmt-s${SEED}-l0${lam}"
  checkpoint="${OUT}/${run}/model.pt"
  log="logs/ocsmt_refine_${ENV}_s${SEED}_l0${lam}.log"

  if [[ -f "$checkpoint" ]]; then
    echo "skip $checkpoint"
    continue
  fi

  echo "$(date +%T) [gpu $gpu] >>> $run"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_memory.py \
    --env "$ENV" --memory-mode ocsmt \
    --l0-lambda "$lam" --gate-lr-mult 8 --oc-num 28 \
    --run-suffix "l0${lam}" --seed "$SEED" \
    --output-dir "$OUT" --epochs 30 --num-episodes 4000 --batch-size 64 \
    --num-workers "$WORKERS" \
    --no-wandb > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "one or more OC-SMT refinement runs failed; inspect logs/ocsmt_refine_*" >&2
  exit "$status"
fi

"$PY" scripts/analyze_runs.py "$OUT"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" scripts/analyze_ocsmt.py \
  "$OUT" --device cuda --eval-n 128
echo "=== OC-SMT refinement complete: $OUT ==="
