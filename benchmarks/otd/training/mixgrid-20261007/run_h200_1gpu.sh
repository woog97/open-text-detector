#!/bin/bash
# Usage (on the H200, from the final-20261007 work dir): run_h200_1gpu.sh MODEL TAG GPU PORT — e.g. run_h200_1gpu.sh moe-mixA-full moe-mixA-full 0 29812
# One H200, micro-batch 32 (no accumulation; benchmarked 8.95 stage-2 rows/s on Oct 7), otherwise the run_ddp_h200.sh settings.
# Used Oct 8 to continue mixA-full after the Space A100s were withdrawn (resume state merged by merge_fsdp_resume.py).
# The W&B key arrives on stdin (first line) and is exported only to the training process. Resumes up to 3 times.
set -u; MODEL=$1; TAG=$2; GPUS=$3; PORT=$4; R=/workspace/woog/pangram/final-20261007; cd $R; mkdir -p sweeps/$TAG
read -r WANDB_API_KEY; export WANDB_API_KEY
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$R:$R/ccdshim HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 TORCH_NCCL_DUMP_ON_TIMEOUT=1 WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
STATE=/workspace/woog/pangram/state-mixfull
ARGS="$MODEL $TAG --fraction 1.0 --stage2-epochs 1 --ckpt-fracs 0.3333,0.6667 --micro-batch 32 --lr 1e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed 1 \
  --fast-moe --pack --fsdp-no-ckpt-layers 0 --on-diverge rollback --snapshot-every 300 --resume-every 20 --state-dir $STATE"
TR="venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT"
FIRST=""; [ -f sweeps/$TAG/run.json ] && FIRST="--resume"
. $R/mixgrid/retry_lib.sh
echo "rc=$rc reason=$REASON $(date -u +%T)" > sweeps/$TAG/train.done
