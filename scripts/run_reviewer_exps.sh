#!/usr/bin/env bash
# Autonomous reviewer-response program: E5 (more seeds) -> E3 (single-tau & log-spaced K-bank)
# -> E4 (horizon-law grid) -> E2 (GRU + long-context baselines). Waits for run_paperpo.sh to
# release the GPUs, distributes round-robin over a GPU list, skips finished runs, then analyzes.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs outputs/4ens outputs/rev

echo "$(date +%T) waiting for run_paperpo.sh to finish..."
while pgrep -f run_paperpo.sh >/dev/null 2>&1; do sleep 30; done
echo "$(date +%T) GPUs free; starting reviewer experiments"

GPU_LIST=(${GPUS:-0 1 2 3}); NG=${#GPU_LIST[@]}
COMMON="--epochs ${EPOCHS:-30} --num-episodes ${NUM_EPISODES:-5000} --batch-size ${BATCH:-64} --num-workers ${NW:-4} --eval-interval ${EVAL:-10}"

JOBS=()
# E5 — more seeds (3,4) on the headline 4-env matrix -> 5 seeds total
for s in 3 4; do for env in tmaze occlusion recall distractor; do for d in none short long both; do
  JOBS+=("--env $env --memory-mode $d --seed $s --fixed-alpha --tau-fast 3 --tau-slow 25 --output-dir outputs/4ens --wandb-project lewm-memory-4ens --extra-tag exp:E5")
done; done; done
# E3 (log-spaced K-bank) + E2 (GRU) as new designs on the 4-env matrix (seeds 0-2)
for s in 0 1 2; do for env in tmaze occlusion recall distractor; do for d in multi gru; do
  JOBS+=("--env $env --memory-mode $d --seed $s --fixed-alpha --output-dir outputs/4ens --wandb-project lewm-memory-4ens --extra-tag exp:$d")
done; done; done
# E3 — single-timescale sweep (design long, one slow bank) on tmaze
for s in 0 1 2; do for t in 2 4 8 16 32 64; do
  JOBS+=("--env tmaze --memory-mode long --seed $s --fixed-alpha --tau-fast 3 --tau-slow $t --run-suffix single$t --output-dir outputs/rev --wandb-project lewm-memory-rev --extra-tag exp:E3single")
done; done
# E4 — horizon-law grid (gap Δ via reveal × τ_slow), design long, tmaze, length 48
for s in 0 1; do for rev in 6 12 24 42; do for t in 4 16 64; do
  JOBS+=("--env tmaze --memory-mode long --seed $s --fixed-alpha --tau-fast 3 --tau-slow $t --reveal $rev --cue-len 3 --length 48 --run-suffix grid_r${rev}_t${t} --output-dir outputs/rev --wandb-project lewm-memory-rev --extra-tag exp:E4grid")
done; done; done
# E2 — long-context predictor baseline (design none, larger window h), tmaze, length 48
for s in 0 1 2; do for hh in 9 21 39; do
  JOBS+=("--env tmaze --memory-mode none --seed $s --fixed-alpha --history-len $hh --length 48 --run-suffix h$hh --output-dir outputs/rev --wandb-project lewm-memory-rev --extra-tag exp:E2longctx")
done; done

echo "=== reviewer-exp jobs: ${#JOBS[@]} on GPUs [${GPU_LIST[*]}] ==="

run_worker () {
  local slot=$1
  local gpu=${GPU_LIST[$slot]}
  for idx in "${!JOBS[@]}"; do
    if [ $(( idx % NG )) -eq "$slot" ]; then
      local args="${JOBS[$idx]}" env mode seed suf od name
      env=$(sed -E 's/.*--env ([a-z_]+).*/\1/' <<<"$args")
      mode=$(sed -E 's/.*--memory-mode ([a-z]+).*/\1/' <<<"$args")
      seed=$(sed -E 's/.*--seed ([0-9]+).*/\1/' <<<"$args")
      suf=$(sed -nE 's/.*--run-suffix ([^ ]+).*/\1/p' <<<"$args")
      od=$(sed -E 's#.*--output-dir ([^ ]+).*#\1#' <<<"$args")
      name="lewm-${env}-${mode}-s${seed}${suf:+-$suf}"
      if [ -f "${od}/${name}/model.pt" ]; then echo "skip $name"; continue; fi
      echo "$(date +%T) [gpu $gpu] >>> $name"
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/train_memory.py $args $COMMON > "logs/rev_${idx}.log" 2>&1
      echo "$(date +%T) [gpu $gpu] <<< $name (exit $?)"
    fi
  done
}
for s in $(seq 0 $((NG - 1))); do run_worker "$s" & done
wait
echo "=== REVIEWER-EXP RUNS COMPLETE ==="
$PY scripts/analyze_runs.py outputs/4ens
$PY scripts/analyze_runs.py outputs/rev
echo "=== REVIEWER-EXP ANALYSIS COMPLETE ==="
