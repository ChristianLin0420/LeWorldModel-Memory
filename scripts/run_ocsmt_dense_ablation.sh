#!/usr/bin/env bash
# Explain dense OC-SMT's Occlusion gain by crossing basis size, horizon range,
# and hard-concrete training noise. All runs use lambda0=0; this is a capacity /
# optimization control, not a sparsity sweep.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/ocsmt_dense_ablation}
SEEDS=(${SEEDS:-0 1 2})
MS=(${MS:-6 12 28})
TAU_MAXES=(${TAU_MAXES:-64 256})
GATE_MODES=(${GATE_MODES:-stochastic deterministic})
GPU_SLOTS=(${GPUS:-0 0 1 1 2 2 3 3})
WORKERS=${WORKERS:-2}

mkdir -p "$OUT" logs

jobs=()
checkpoints=()
for seed in "${SEEDS[@]}"; do
  for m in "${MS[@]}"; do
    for tau_max in "${TAU_MAXES[@]}"; do
      for gate_mode in "${GATE_MODES[@]}"; do
        suffix="dense-M${m}-tmax${tau_max}-${gate_mode}"
        jobs+=("$seed $m $tau_max $gate_mode $suffix")
        checkpoints+=("${OUT}/lewm-occlusion-ocsmt-s${seed}-${suffix}/model.pt")
      done
    done
  done
done

worker() {
  local slot=$1
  local gpu=${GPU_SLOTS[$slot]}
  local idx seed m tau_max gate_mode suffix
  for idx in "${!jobs[@]}"; do
    if (( idx % ${#GPU_SLOTS[@]} != slot )); then
      continue
    fi
    read -r seed m tau_max gate_mode suffix <<< "${jobs[$idx]}"
    if [[ -f "${checkpoints[$idx]}" ]]; then
      echo "skip ${checkpoints[$idx]}"
      continue
    fi
    echo "$(date +%T) [gpu $gpu] >>> occlusion seed=$seed M=$m tau_max=$tau_max gate=$gate_mode"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_memory.py \
      --env occlusion --memory-mode ocsmt --l0-lambda 0 \
      --gate-lr-mult 8 --oc-num "$m" --oc-tau-min 1.5 --oc-tau-max "$tau_max" \
      --oc-gate-mode "$gate_mode" --run-suffix "$suffix" --seed "$seed" \
      --output-dir "$OUT" --epochs 30 --num-episodes 4000 --batch-size 64 \
      --num-workers "$WORKERS" --no-wandb \
      > "logs/ocsmt_dense_s${seed}_M${m}_tmax${tau_max}_${gate_mode}.log" 2>&1
  done
}

pids=()
for slot in "${!GPU_SLOTS[@]}"; do
  worker "$slot" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "one or more dense OC-SMT ablations failed; inspect logs/ocsmt_dense_*" >&2
  exit "$status"
fi

"$PY" scripts/analyze_runs.py "$OUT"
CUDA_VISIBLE_DEVICES=${GPU_SLOTS[0]} "$PY" scripts/analyze_ocsmt.py \
  "$OUT" --device cuda --eval-n 128
"$PY" scripts/analyze_ocsmt_dense_ablation.py "$OUT"
echo "=== dense OC-SMT ablation complete: $OUT ==="
