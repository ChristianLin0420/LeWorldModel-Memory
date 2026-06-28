#!/usr/bin/env bash
# Mass-matched control for the SMT v1 -> v2 router change. A standard softmax
# has total read mass 1, whereas six sigmoid gates initialize with total mass
# about 3. scaled_softmax preserves softmax's relative routing while multiplying
# it by K/2, isolating amplitude from independent additive gating.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/smt_scaled_softmax}
SEEDS=(${SEEDS:-0 1 2})
ENVS=(${ENVS:-tmaze distractor recall occlusion})
GPU_SLOTS=(${GPUS:-0 0 1 1 2 2 3 3})
WORKERS=${WORKERS:-6}

mkdir -p "$OUT" logs

jobs=()
checkpoints=()
for seed in "${SEEDS[@]}"; do
  for env in "${ENVS[@]}"; do
    jobs+=("$env $seed")
    checkpoints+=("${OUT}/lewm-${env}-smt-s${seed}/model.pt")
  done
done

validate_checkpoint() {
  local checkpoint=$1 env=$2 seed=$3
  "$PY" - "$checkpoint" "$env" "$seed" <<'PY'
import sys, torch
path, env, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = torch.load(path, map_location='cpu', weights_only=False)['args']
checks = {
    'env': env, 'seed': seed, 'memory_mode': 'smt',
    'smt_router': 'scaled_softmax', 'epochs': 30,
    'num_episodes': 4000, 'batch_size': 64,
}
for name, want in checks.items():
    if cfg.get(name) != want:
        raise SystemExit(f'{path}: {name}={cfg.get(name)!r}, expected {want!r}')
PY
}

worker() {
  local slot=$1
  local gpu=${GPU_SLOTS[$slot]}
  local idx env seed
  for idx in "${!jobs[@]}"; do
    if (( idx % ${#GPU_SLOTS[@]} != slot )); then
      continue
    fi
    read -r env seed <<< "${jobs[$idx]}"
    if [[ -f "${checkpoints[$idx]}" ]]; then
      validate_checkpoint "${checkpoints[$idx]}" "$env" "$seed"
      echo "skip ${checkpoints[$idx]}"
      continue
    fi
    echo "$(date +%T) [gpu $gpu] >>> lewm-${env}-smt-s${seed}"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_memory.py \
      --env "$env" --memory-mode smt --smt-router scaled_softmax \
      --seed "$seed" --output-dir "$OUT" \
      --epochs 30 --num-episodes 4000 --batch-size 64 \
      --num-workers "$WORKERS" --no-wandb \
      > "logs/smt_scaled_${env}_s${seed}.log" 2>&1
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
  echo "one or more scaled-softmax runs failed; inspect logs/smt_scaled_*" >&2
  exit "$status"
fi

for idx in "${!jobs[@]}"; do
  read -r env seed <<< "${jobs[$idx]}"
  [[ -s "${checkpoints[$idx]}" ]] || {
    echo "missing scaled-softmax checkpoint: ${checkpoints[$idx]}" >&2
    exit 2
  }
  validate_checkpoint "${checkpoints[$idx]}" "$env" "$seed"
done
"$PY" - "$OUT" "${checkpoints[@]}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
expected = {str(Path(path).resolve()) for path in sys.argv[2:]}
actual = {str(path.resolve()) for path in root.glob('*/model.pt')}
if actual != expected:
    raise SystemExit(
        f'scaled-softmax factorial mismatch: missing={sorted(expected-actual)}, '
        f'extra={sorted(actual-expected)}')
print(f'validated exact scaled-softmax factorial: {len(actual)} checkpoints')
PY

"$PY" scripts/analyze_runs.py "$OUT"
echo "=== SMT scaled-softmax control complete: $OUT ==="
