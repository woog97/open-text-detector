#!/bin/bash
# Loss ablation scheduler v4 (Oct 10, replaces round2_abl.sh mid-run). W&B key on stdin. Run from the final-20261007 work dir.
# GPU 1: finish round 1's evaluations, then is left alone (released). abl-control trains on GPU 3 (started separately).
# GPU 2 (abl-tok-only), GPU 0 (abl-sent1), GPU 3 (abl-control): each becomes an eval worker when its run finishes.
# Eval workers share eval_final.py jobs through claim files (WORKERS=3) and exit when all 7 runs are evaluated or failed.
set -u; cd /workspace/woog/pangram/final-20261007; read -r K
R1="abl-nostage1 abl-noseg-nomix abl-tok-sent abl-uniform-tok"; ALL="$R1 abl-tok-only abl-sent1 abl-control"
declare -A HAND=([abl-tok-only]=2 [abl-sent1]=0 [abl-control]=3)
L=sweeps/abl-round2.log; log() { echo "$(date -u +%T) $*" >> $L; }
free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; }
wait_free() { until free $1; do sleep 30; done; }
finished() { [ -f sweeps/$1/train.done ]; }
ok() { grep -q '^rc=0' sweeps/$1/train.done 2>/dev/null && python3 -c "import json,sys;sys.exit(json.load(open('sweeps/$1/status.json'))['state']!='trained')"; }
pending() { python3 mixgrid/abl_eval_pending.py $1 2>/dev/null || echo 0; }
evenv() { export CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PWD/ccdshim:$PWD/vendor HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 FAST_EVAL=1; }
run_eval() { mkdir -p sweeps/$2/eval; log "gpu $1 eval $2"; venv/bin/python -u eval_final.py $2 3 >> sweeps/$2/eval/gpu$1.out 2>&1
  log "gpu $1 eval $2 rc=$? summary=$([ -f sweeps/$2/eval/summary.json ] && echo yes || echo no)"; }
eval_worker() {  # tags = which runs this worker may take; exits when all of them are settled
  local g=$1 tags=$2 t busy; evenv $g; log "eval worker on gpu $g for: $tags"
  while true; do
    busy=0
    for t in $tags; do
      finished $t || { busy=1; continue; }
      ok $t || continue
      [ -f sweeps/$t/eval/summary.json ] && continue
      busy=1; [ "$(pending $t)" -gt 0 ] || continue
      run_eval $g $t; continue 2
    done
    [ $busy -eq 0 ] && break; sleep 60
  done
  log "eval worker on gpu $g finished"
}
gpu1() {
  log "gpu 1: waiting for its current eval, then round-1 evals, then released"; wait_free 1
  ( eval_worker 1 "$R1" ); log "gpu 1 released"
}
log "scheduler v4 start pid $$"
gpu1 &
sleep 5
declare -A DONE=()
while [ ${#DONE[@]} -lt 3 ]; do
  for job in "${!HAND[@]}"; do
    [ -n "${DONE[$job]:-}" ] && continue; g=${HAND[$job]}
    [ $job = abl-control ] && [ ! -f sweeps/abl-control/run.json ] && continue  # not started yet
    if finished $job && free $g; then DONE[$job]=1; log "gpu $g ($job done) -> eval worker"; ( eval_worker $g "$ALL" ) & fi
  done
  [ ${#DONE[@]} -lt 3 ] && sleep 60
done
K=""; log "all GPUs handed to evals; waiting"; wait; log "scheduler v4 exit"
