#!/usr/bin/env bash
# OC-SMT (§8/§9): over-complete fixed basis + L0 sparse gates. Coarse test of usage versus
# deterministic active count: l0=0 is the dense upper bound; positive l0 tests whether annealed
# sparsification finds a useful subset. 4 envs x {0, 0.05, 0.2} x 3 seeds -> outputs/ocsmt.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; mkdir -p logs outputs/ocsmt
GPU_LIST=(${GPUS:-1 1 2 2 3 3}); NG=${#GPU_LIST[@]}
JOBS=(); SK=()
for s in 0 1 2; do for env in tmaze distractor recall occlusion; do for lam in 0 0.05 0.2; do
  JOBS+=("scripts/train_memory.py --env $env --memory-mode ocsmt --l0-lambda $lam --gate-lr-mult 8 --oc-num 28 --run-suffix l0$lam --seed $s --output-dir outputs/ocsmt --wandb-project lewm-memory-smt --extra-tag exp:ocsmt --epochs 30 --num-episodes 4000 --batch-size 64")
  SK+=("outputs/ocsmt/lewm-$env-ocsmt-s$s-l0$lam/model.pt")
done; done; done
echo "=== OC-SMT jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="
worker(){ local slot=$1; local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do if [ $((idx%NG)) -eq "$slot" ]; then
    [ -f "${SK[$idx]}" ] && { echo "skip ${SK[$idx]}"; continue; }
    echo "$(date +%T) [gpu $gpu] >>> ${SK[$idx]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY ${JOBS[$idx]} > "logs/ocsmt_${idx}.log" 2>&1
  fi; done; }
PIDS=()
for s in $(seq 0 $((NG-1))); do worker "$s" & PIDS+=("$!"); done
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then
  echo "one or more OC-SMT workers failed; inspect logs/ocsmt_*" >&2
  exit "$status"
fi
echo "=== OC-SMT RUNS COMPLETE ==="
$PY - "$PWD/outputs/ocsmt" <<'PY'
import itertools, sys, torch
from pathlib import Path
root = Path(sys.argv[1])
expected = set(itertools.product(('tmaze','distractor','recall','occlusion'), (0,1,2), (0.0,0.05,0.2)))
seen = set()
for path in sorted(root.glob('*/model.pt')):
    cfg = torch.load(path, map_location='cpu', weights_only=False)['args']
    key = (cfg.get('env'), int(cfg.get('seed', -1)), float(cfg.get('l0_lambda', float('nan'))))
    if key not in expected or key in seen:
        raise SystemExit(f'unexpected/duplicate OC-SMT checkpoint {path}: {key}')
    checks = {'memory_mode':'ocsmt', 'oc_num':28, 'gate_lr_mult':8.0,
              'epochs':30, 'num_episodes':4000, 'batch_size':64}
    for name, want in checks.items():
        if cfg.get(name) != want:
            raise SystemExit(f'{path}: {name}={cfg.get(name)!r}, expected {want!r}')
    seen.add(key)
if seen != expected:
    raise SystemExit(f'incomplete OC-SMT factorial: {len(seen)}/{len(expected)}')
print(f'validated exact OC-SMT factorial: {len(seen)} checkpoints')
PY
$PY scripts/analyze_runs.py outputs/ocsmt
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} $PY scripts/analyze_ocsmt.py \
  outputs/ocsmt --device cuda --eval-n 128
echo "=== OC-SMT AGG COMPLETE ==="
