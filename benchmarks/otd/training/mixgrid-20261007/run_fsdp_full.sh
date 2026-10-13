#!/bin/bash
# Usage (on the Space): run_fsdp_full.sh MODEL TAG GPUS PORT — e.g. run_fsdp_full.sh moe-mixA-full moe-mixA-full 2,3,4,5 29701
# MoE FSDP2 over the listed GPUs with the Oct 7 speedups (--fast-moe --pack; gmm_triton is used automatically on A100s;
# ccdshim on PYTHONPATH). Micro-batch follows the GPU count (effective batch 32): 2 GPUs mb16, 4 GPUs mb8; others unsupported.
# Expects WANDB_API_KEY in the environment. Retries up to 3 times (retry_lib.sh); --on-diverge stop (FSDP has no rollback); persists to /data after the run.
set -u; MODEL=$1; TAG=$2; GPUS=$3; PORT=$4; R=/tmp/mixfull-20261008; cd $R; mkdir -p sweeps/$TAG
NP=$(echo $GPUS | tr ',' '\n' | wc -l)
case $NP in 2) MB=16; NOCK=0;; 4) MB=8; NOCK=16;; *) echo "unsupported GPU count $NP" >> sweeps/$TAG/train.log; exit 2;; esac
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=/tmp/final-20261007/vendor:/tmp/pangram-wandb-vendor:$R/ccdshim HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 TORCH_NCCL_DUMP_ON_TIMEOUT=1 WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
STATE=/data/workspace/mixgrid-20261007/state  # survives a Space restart (/tmp does not; persist_daemon skips resume state)
ARGS="$MODEL $TAG --fraction 1.0 --stage2-epochs 1 --ckpt-fracs 0.3333,0.6667 --micro-batch $MB --lr 1e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed 1 \
  --fsdp --fsdp-prefetch 2 --fast-moe --pack --fsdp-no-ckpt-layers $NOCK --on-diverge stop --snapshot-every 100 --resume-every 20 --state-dir $STATE"
TR="python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NP --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT"
FIRST=""; [ -f sweeps/$TAG/run.json ] && FIRST="--resume"  # relaunch after a restart: resume, never start over
. $R/retry_lib.sh
python persist_daemon.py $R --once >> sweeps/$TAG/persist.log 2>&1
echo "rc=$rc reason=$REASON $(date -u +%T)" > sweeps/$TAG/train.done
