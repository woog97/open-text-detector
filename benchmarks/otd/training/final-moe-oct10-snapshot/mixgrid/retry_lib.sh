# Shared retry logic for the full-size mix runs (sourced by run_fsdp_full.sh / run_ddp_h200.sh). Needs: TR, ARGS, TAG, GPUS, FIRST.
# Runs the trainer in its own process group; on failure kills that group, waits until the run's GPUs are below 1 GB, sleeps 60 s,
# and retries with --resume (no state yet means a fresh start). Stops early when the same OOM / nonfinite-loss error repeats.
errclass() { local l; l=$(grep -E "Error|error|Traceback|Nonfinite|out of memory" sweeps/$TAG/train.log | tail -3 | tr '\n' ' ')
  case "$l" in *"out of memory"*|*OutOfMemory*) echo oom;; *Nonfinite*|*nan*) echo nonfinite;; *"Disk quota"*|*"No space"*) echo disk;; *) echo other;; esac; }
# Stop the previous attempt: SIGTERM to torchrun (it forwards to its workers, which run in their own process groups, so a
# group SIGKILL leaves them orphaned holding GPU memory), then SIGKILL anything still on this run's GPUs.
stop_run() { local pid u
  kill -TERM $PG 2>/dev/null; kill -TERM -- -$PG 2>/dev/null; for i in $(seq 1 12); do kill -0 $PG 2>/dev/null || break; sleep 5; done
  kill -KILL -- -$PG 2>/dev/null; kill -KILL $PG 2>/dev/null
  for u in $(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v G=",$GPUS," 'index(G, ","$1",")>0 {print $2}'); do
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$u '$1==u{print $2}'); do
      [ "$(stat -c %U /proc/$pid 2>/dev/null)" = "$(id -un)" ] && kill -KILL $pid 2>/dev/null; done; done; }
gpus_quiet() { local g used; for g in ${GPUS//,/ }; do used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1==g{print $2}'); [ "${used:-0}" -lt 1000 ] || return 1; done; }
run_once() { setsid $TR train_sweep.py $ARGS $1 >> sweeps/$TAG/train.log 2>&1 & PG=$!; echo $PG > sweeps/$TAG/train.pgid; wait $PG; rc=$?; }
REASON=ok; prev=""
echo "start $(date -u +%T) gpus=$GPUS $FIRST" >> sweeps/$TAG/train.log
run_once "$FIRST"
for attempt in 1 2 3; do
  [ $rc -eq 0 ] && break
  cls=$(errclass); REASON="$cls"
  if [ -n "$prev" ] && [ "$cls" = "$prev" ] && { [ "$cls" = oom ] || [ "$cls" = nonfinite ] || [ "$cls" = disk ]; }; then REASON="repeated-$cls"; break; fi
  prev=$cls
  stop_run
  for i in $(seq 1 30); do gpus_quiet && break; sleep 10; done
  gpus_quiet || { REASON="gpus-not-released"; break; }
  sleep 60
  echo "retry $attempt after $cls $(date -u +%T)" >> sweeps/$TAG/train.log
  run_once --resume
done
[ $rc -eq 0 ] && REASON=ok
