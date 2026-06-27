#!/usr/bin/env bash
# OC-SMT (§8/§9): over-complete fixed basis + L0 sparse gates. Sweep lambda0 to trace the
# auto-sizing curve (usage vs mean active-bank count): l0=0 is the dense over-complete bank;
# l0>0 anneals in the L0 prune. 4 envs x {0, 0.05, 0.2} x 3 seeds -> outputs/ocsmt.
set -u
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
for s in $(seq 0 $((NG-1))); do worker "$s" & done
wait
echo "=== OC-SMT RUNS COMPLETE ==="
$PY scripts/analyze_runs.py outputs/ocsmt
# active-bank count aggregation (the learned effective size)
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} $PY - <<'PYEOF'
import torch, csv, glob, statistics as st, sys
from collections import defaultdict
sys.path.insert(0,'.')
from scripts.analyze_runs import build_model
from lewm.data import generate_eval_batch
usage={}
for r in csv.DictReader(open('outputs/ocsmt/master_metrics.csv')):
    usage[r['run']]=float(r['usage_matched'])
agg=defaultdict(lambda: defaultdict(list))
for d in sorted(glob.glob('outputs/ocsmt/*/')):
    mp=d+'model.pt'
    try: ck=torch.load(mp,map_location='cuda',weights_only=False)
    except: continue
    a=ck['args']; name=d.split('/')[-2]
    m=build_model(a).cuda(); m.load_state_dict(ck['model_state_dict']); m.eval()
    b=generate_eval_batch(a['env'],128,img_size=a['img_size'],length=a['length'],seed=4242)
    z=m.encode(b['obs'].cuda()); ac=float(m.mem_ocsmt.active_count(z))
    agg[(a['env'],a.get('run_suffix',''))]['act'].append(ac)
    agg[(a['env'],a.get('run_suffix',''))]['use'].append(usage.get(name,float('nan')))
print("\n=== OC-SMT auto-sizing: usage vs mean active banks (/28), 3 seeds ===")
print(f"{'env':<11}{'l0':<7}{'usage':>10}{'active/28':>12}")
for (env,suf),m in sorted(agg.items()):
    print(f"{env:<11}{suf:<7}{st.mean(m['use']):>10.2f}{st.mean(m['act']):>12.1f}")
PYEOF
echo "=== OC-SMT AGG COMPLETE ==="
