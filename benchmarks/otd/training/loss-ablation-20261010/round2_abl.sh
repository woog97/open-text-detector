#!/bin/bash
# Loss ablation, round 2 (Oct 10), run detached on the H200 from the final-20261007 work dir; W&B key on stdin.
# As round-1 runs finish and their GPUs empty: 1st free GPU trains abl-tok-only, 2nd trains abl-sent1 (same single-GPU settings
# as round 1), 3rd becomes an eval worker. The 4th is released. When a round-2 run finishes, its GPU becomes an eval worker too.
# Eval workers share jobs through eval_final.py's claim files (WORKERS > 1), so up to 3 GPUs split the remaining evals of all 6 runs.
set -u; cd /workspace/woog/pangram/final-20261007; read -r K
declare -A GPU=([abl-nostage1]=0 [abl-noseg-nomix]=1 [abl-tok-sent]=2 [abl-uniform-tok]=3)
R1="abl-nostage1 abl-noseg-nomix abl-tok-sent abl-uniform-tok"; R2="abl-tok-only abl-sent1"; ALL="$R1 $R2"
declare -A FLAGS=([abl-tok-only]="--skip-stage1 --segment-weight 0 --mixed-weight 0 --document-weight 0 --drop-gradtex --sentence-weight 0" [abl-sent1]="--sentence-weight 1.0")
declare -A PORT=([abl-tok-only]=29905 [abl-sent1]=29906)
declare -A R2GPU=()
L=sweeps/abl-round2.log; log() { echo "$(date -u +%T) $*" >> $L; }
free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; }
finished() { [ -f sweeps/$1/train.done ]; }
ok() { grep -q '^rc=0' sweeps/$1/train.done 2>/dev/null && python3 -c "import json,sys;sys.exit(json.load(open('sweeps/$1/status.json'))['state']!='trained')"; }
pending() { python3 mixgrid/abl_eval_pending.py $1 2>/dev/null || echo 0; }
eval_worker() {  # one GPU; repeatedly joins any finished run that still has unclaimed eval jobs; exits when all 6 are settled
  local g=$1 t busy
  export CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PWD/ccdshim:$PWD/vendor HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 FAST_EVAL=1
  log "eval worker on gpu $g"
  while true; do
    busy=0
    for t in $ALL; do
      finished $t || { busy=1; continue; }
      ok $t || continue
      [ -f sweeps/$t/eval/summary.json ] && continue
      busy=1; [ "$(pending $t)" -gt 0 ] || continue
      mkdir -p sweeps/$t/eval; log "gpu $g eval $t"
      venv/bin/python -u eval_final.py $t 3 >> sweeps/$t/eval/gpu$g.out 2>&1
      log "gpu $g eval $t rc=$? summary=$([ -f sweeps/$t/eval/summary.json ] && echo yes || echo no)"
      continue 2
    done
    [ $busy -eq 0 ] && break
    sleep 60
  done
  log "eval worker on gpu $g finished"
}
log "round2 scheduler (v2) start pid $$"
jobs_left="abl-tok-only abl-sent1 EVAL"; used=""
while true; do
  for t in $R1; do
    [ -z "$jobs_left" ] && break
    g=${GPU[$t]}; case " $used " in *" $g "*) continue;; esac
    finished $t && free $g || continue
    used="$used $g"; job=${jobs_left%% *}; jobs_left=$(echo ${jobs_left#"$job"})
    if [ "$job" = EVAL ]; then
      log "gpu $g ($t done) -> eval worker"; eval_worker $g &
    else
      [ -e sweeps/$job ] && { log "$job exists, not relaunched"; continue; }
      mkdir -p sweeps/$job; R2GPU[$job]=$g
      printf '%s\n' "$K" | nohup setsid bash mixgrid/run_abl_h200.sh $job $g ${PORT[$job]} ${FLAGS[$job]} > sweeps/$job.wrapper.out 2>&1 &
      echo $! > sweeps/$job.wrapper.pid; log "gpu $g ($t done) -> train $job wrapper pid $!"
    fi
  done
  for job in "${!R2GPU[@]}"; do  # a finished round-2 run hands its GPU to the evals
    g=${R2GPU[$job]}
    if finished $job && free $g; then unset "R2GPU[$job]"; log "gpu $g ($job done) -> eval worker"; eval_worker $g & fi
  done
  [ -z "$jobs_left" ] && [ ${#R2GPU[@]} -eq 0 ] && break
  sleep 60
done
K=""; log "all GPUs handed out; waiting for eval workers"; wait; log "scheduler exit"
