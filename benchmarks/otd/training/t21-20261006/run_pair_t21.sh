#!/bin/bash
# Usage: run_pair.sh SEED GPUS PORT — 2-GPU data-parallel T2.1 run (effective batch 32), then persist to /data.
R=/tmp/pangram-t2-20261006; cd $R; SEED=$1; T=q4b-T21-s$SEED; mkdir -p sweeps/$T
export CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=$R/vendor:/tmp/pangram-wandb-vendor HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 \
  WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
ARGS="qwen35-4b-t21 $T --fraction 0.2 --micro-batch 8 --lr 2e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed $SEED"
TR="python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:$3"
$TR train_sweep.py $ARGS >> sweeps/$T/train.log 2>&1; rc=$?
if [ $rc -ne 0 ] && [ -f sweeps/$T/resume-state.pt ]; then echo resuming >> sweeps/$T/train.log; $TR train_sweep.py $ARGS --resume >> sweeps/$T/train.log 2>&1; rc=$?; fi
python persist_daemon.py $R --once >> sweeps/$T/persist.log 2>&1
echo "rc=$rc $(date -u +%H:%M:%S)" > sweeps/$T/train.done
