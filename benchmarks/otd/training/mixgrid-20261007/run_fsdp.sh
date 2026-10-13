#!/bin/bash
# Usage: run_fsdp.sh MODEL TAG GPUS PORT — one full-length MoE run with FSDP2 on 2 GPUs; resumes up to 3 times after a crash; persists to /data.
R=/tmp/mixgrid-20261007; cd $R; MODEL=$1; TAG=$2; mkdir -p sweeps/$TAG
export CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=/tmp/final-20261007/vendor:/tmp/pangram-wandb-vendor HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
ARGS="$MODEL $TAG --fraction 1.0 --stage2-epochs 3 --micro-batch 16 --lr 1e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed 1 --fsdp --fsdp-prefetch 2 --on-diverge stop --snapshot-every 100 --resume-every 20"
TR="python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:$4"
$TR train_sweep.py $ARGS >> sweeps/$TAG/train.log 2>&1; rc=$?
# Resume up to 3 times. train_sweep.py keeps two step-tagged files per rank and resumes from the newest step every rank has.
for attempt in 1 2 3; do
  [ $rc -ne 0 ] && ls sweeps/$TAG/resume-state-rank*.pt >/dev/null 2>&1 || break
  echo "resuming (attempt $attempt) $(date -u +%T)" >> sweeps/$TAG/train.log; $TR train_sweep.py $ARGS --resume >> sweeps/$TAG/train.log 2>&1; rc=$?
done
python persist_daemon.py $R --once >> sweeps/$TAG/persist.log 2>&1
echo "rc=$rc $(date -u +%T)" > sweeps/$TAG/train.done
