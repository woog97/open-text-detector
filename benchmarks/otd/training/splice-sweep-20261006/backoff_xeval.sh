#!/bin/bash
# Backoff checks (1,2,4,8,16,32 min after start, then hourly) for an H200 cross-model eval; exits on finish, error or stall.
LOG=$1; start=$(date +%s); prev=""
for t in 60 120 240 480 960 1920 5520 9120 12720; do
  now=$(date +%s); [ $((start + t)) -gt $now ] && sleep $((start + t - now))
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 otd-h200 "cd /workspace/woog/pangram/backbones-20261003; pgrep -f 'cross_model_eval.py' >/dev/null && echo alive || echo exited; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n 4p; grep -c . $LOG; grep -a -E 'Traceback|Error|groups|threshold' $LOG | tail -3 | cut -c1-300" 2>&1)
  echo "$(TZ=America/Los_Angeles date +%H:%M) $(echo $out | tr '\n' ' ')"
  case "$out" in *Traceback*|*exited*) exit 0;; esac
done
