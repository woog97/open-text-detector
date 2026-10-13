#!/bin/bash
# Usage (on the H200, from the final-20261007 work dir): run_ddp_h200.sh MODEL TAG GPUS PORT
#   e.g. run_ddp_h200.sh moe-mixB-full moe-mixB-full 0,1 29811
# Plain torchrun data parallel (no FSDP) with the Oct 7 speedups: --fast-moe --pack, ccdshim, 16 layers without activation
# checkpointing. Micro-batch follows the GPU count (effective batch 32): 1-2 GPUs mb16, 4 GPUs mb8; 3 GPUs are not supported.
# The W&B key arrives on stdin (first line) and is exported only to the training processes. Resumes up to 3 times.
set -u; MODEL=$1; TAG=$2; GPUS=$3; PORT=$4; R=/workspace/woog/pangram/final-20261007; cd $R; mkdir -p sweeps/$TAG
read -r WANDB_API_KEY; export WANDB_API_KEY
NP=$(echo $GPUS | tr ',' '\n' | wc -l)
case $NP in 1|2) MB=16; NOCK=0;; 4) MB=8; NOCK=0;; *) echo "unsupported GPU count $NP" >> sweeps/$TAG/train.log; exit 2;; esac
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$R:$R/ccdshim HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 TORCH_NCCL_DUMP_ON_TIMEOUT=1 WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
STATE=/workspace/woog/pangram/state-mixfull
ARGS="$MODEL $TAG --fraction 1.0 --stage2-epochs 1 --ckpt-fracs 0.3333,0.6667 --micro-batch $MB --lr 1e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed 1 \
  --fast-moe --pack --fsdp-no-ckpt-layers $NOCK --on-diverge rollback --snapshot-every 300 --resume-every 20 --state-dir $STATE"
TR="venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NP --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT"
FIRST=""; [ -f sweeps/$TAG/run.json ] && FIRST="--resume"  # relaunch after a restart: resume, never start over
. $R/mixgrid/retry_lib.sh
echo "rc=$rc reason=$REASON $(date -u +%T)" > sweeps/$TAG/train.done
