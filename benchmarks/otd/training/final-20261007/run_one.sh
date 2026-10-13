#!/bin/bash
# Usage: run_one.sh ROOT PYTHON MODEL TAG GPUS [extra train_sweep args...]
# One training process (no DDP) on the listed GPUs; --model-parallel is added by the launcher when GPUS has two entries.
# Resumes up to 3 times from resume state after a crash, then mirrors outputs (Space: persist_daemon --once) and writes train.done.
R=$1; PY=$2; MODEL=$3; TAG=$4; GPUS=$5; shift 5
cd $R; mkdir -p sweeps/$TAG
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$R/vendor:/tmp/pangram-wandb-vendor:$PYTHONPATH HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  HF_HUB_DISABLE_PROGRESS_BARS=1 WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
echo "start $(date -u +%H:%M:%S) host=$(hostname) gpus=$GPUS" >> sweeps/$TAG/train.log
$PY -u train_sweep.py $MODEL $TAG "$@" >> sweeps/$TAG/train.log 2>&1; rc=$?
STATE=$(python3 -c "import sys;a=sys.argv[1:];print(a[a.index('--state-dir')+1] if '--state-dir' in a else '')" "$@")
RS=${STATE:+$STATE/$TAG}; RS=${RS:-sweeps/$TAG}
# Resume up to 3 times; FSDP runs save per-rank, step-tagged files (resume-state-rank*-s*.pt), so match any resume-state*.pt.
for attempt in 1 2 3; do
  [ $rc -ne 0 ] && ls $RS/resume-state*.pt >/dev/null 2>&1 || break
  echo "resuming (attempt $attempt) $(date -u +%H:%M:%S)" >> sweeps/$TAG/train.log
  $PY -u train_sweep.py $MODEL $TAG "$@" --resume >> sweeps/$TAG/train.log 2>&1; rc=$?
done
[ -f persist_daemon.py ] && [ -d /data/workspace ] && $PY persist_daemon.py $R --once >> sweeps/$TAG/persist.log 2>&1
echo "rc=$rc $(date -u +%H:%M:%S)" > sweeps/$TAG/train.done
