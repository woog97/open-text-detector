#!/bin/bash
# Usage (on the H200, from the final-20261007 work dir): run_4b_h200.sh MODEL TAG GPU PORT
#   e.g. run_4b_h200.sh qwen35-4b-t21-1ep q4b-t21-1ep 0 29831
# Qwen3.5-4B on one H200, one stage-2 epoch, the final 4B recipe (micro-batch 8, lr 2e-4, head lr 2e-5, cosine, warmup 0.06)
# with the Oct 8 speedups that keep the recipe: --pack, ccdshim, all 32 layers without activation checkpointing
# (bench4b: 7.85 -> 17.3 stage-2 rows/s, 61 GB; micro-batch 32 ran out of memory).
# The W&B key arrives on stdin (first line) and is exported only to the training process. Resumes up to 3 times.
set -u; MODEL=$1; TAG=$2; GPUS=$3; PORT=$4; R=/workspace/woog/pangram/final-20261007; cd $R; mkdir -p sweeps/$TAG
read -r WANDB_API_KEY; export WANDB_API_KEY
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$R:$R/ccdshim HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
STATE=/workspace/woog/pangram/state-4b
ARGS="$MODEL $TAG --fraction 1.0 --stage2-epochs 1 --ckpt-fracs 0.3333,0.6667 --micro-batch 8 --lr 2e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed 1 \
  --pack --fsdp-no-ckpt-layers 32 --on-diverge rollback --snapshot-every 100 --snapshot-keep 3 --resume-every 20 --state-dir $STATE"
TR="venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT"
FIRST=""; [ -f sweeps/$TAG/run.json ] && FIRST="--resume"  # relaunch after a restart: resume, never start over
. $R/mixgrid/retry_lib.sh
echo "rc=$rc reason=$REASON $(date -u +%T)" > sweeps/$TAG/train.done
