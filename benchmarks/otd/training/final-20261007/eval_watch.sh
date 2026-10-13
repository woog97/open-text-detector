#!/bin/bash
# H200 scoring watcher for the final runs. Usage (from the final-20261007 work dir): nohup bash eval_watch.sh TAG:GPU [TAG:GPU ...] &
# For each TAG, waits for sweeps/TAG/train.done with rc=0 and stage2-selection.json, then runs eval_final.py TAG on its
# preferred GPU when that GPU is free (no compute process, <100 MiB used); otherwise on GPU 2 or 3 if one of those is free.
# Never takes a busy GPU (the H200 is shared). Writes eval-watch.log; each run's log goes to sweeps/TAG/eval/eval-final.out.
cd "$(dirname "$0")"
export PYTHONPATH=$PWD/vendor:$PYTHONPATH HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
log() { echo "$(date -u +%H:%M:%S) $*" >> eval-watch.log; }
free_gpus() {
  local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r i u m; do
    if [ "$m" -lt 100 ] && ! grep -q "$u" <<< "$busy"; then echo "$i"; fi; done
}
declare -A DONE
log "start $*"
while true; do
  left=0
  for job in "$@"; do
    tag=${job%%:*}; pref=${job##*:}
    [ -n "${DONE[$tag]}" ] && continue; left=1
    d=sweeps/$tag
    if [ -f $d/eval/summary.json ]; then DONE[$tag]=1; log "$tag already scored"; continue; fi
    [ -f $d/train.done ] || continue
    if ! grep -q '^rc=0' $d/train.done || [ ! -f $d/stage2-selection.json ]; then DONE[$tag]=1; log "$tag training failed: $(cat $d/train.done)"; continue; fi
    f=$(free_gpus); g=""
    if grep -qx "$pref" <<< "$f"; then g=$pref; else for c in 2 3; do grep -qx "$c" <<< "$f" && { g=$c; break; }; done; fi
    [ -z "$g" ] && continue
    mkdir -p $d/eval; log "$tag scoring on GPU $g"
    CUDA_VISIBLE_DEVICES=$g venv/bin/python -u eval_final.py $tag >> $d/eval/eval-final.out 2>&1; rc=$?
    log "$tag eval rc=$rc"; DONE[$tag]=1
  done
  [ $left -eq 0 ] && { log "all done"; break; }
  sleep 120
done
