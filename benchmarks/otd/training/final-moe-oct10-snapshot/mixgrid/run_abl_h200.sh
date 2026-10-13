#!/bin/bash
# Loss-term ablation (Oct 10): one MoE arm on one H200 with the fast trainer, the Oct 7 final mix as one 72k stage-2 pass.
# Usage (from the final-20261007 work dir, W&B key on stdin): run_abl_h200.sh TAG GPUS PORT [ARM FLAGS...]
# GPUS: one id, or a comma list for plain torchrun data parallel (micro-batch = 32 / GPU count; 1, 2 or 4 GPUs).
# Env: MODEL (default moe-final-1ep), NOCK = decoder layers without activation checkpointing (default 0).
# Same settings as mixgrid/run_h200_1gpu.sh (micro-batch 32, lr 1e-4, head lr 2e-5, cosine, warmup 0.06, seed 1, --fast-moe --pack),
# except: model moe-final-1ep (runs/moe-final-1ep = the final mix, stage 2 merged), no intermediate checkpoints (the volume is
# nearly full), and resume state + rollback snapshots in /dev/shm (outside the /workspace quota; lost on reboot).
set -u; TAG=$1; GPUS=$2; PORT=$3; shift 3; EXTRA="$*"; MODEL=${MODEL:-moe-final-1ep}  # MODEL=moe-final-1020 for the long-context mix
R=/workspace/woog/pangram/final-20261007; cd $R; mkdir -p sweeps/$TAG
read -r WANDB_API_KEY; export WANDB_API_KEY
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$R:$R/ccdshim HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 TORCH_NCCL_DUMP_ON_TIMEOUT=1 WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
STATE=/dev/shm/woog-pangram/state-abl
NP=$(echo $GPUS | tr ',' '\n' | wc -l); case $NP in 1) MB=32;; 2) MB=16;; 4) MB=8;; *) echo "unsupported GPU count $NP" >> sweeps/$TAG/train.log; exit 2;; esac
ARGS="$MODEL $TAG --fraction 1.0 --stage2-epochs 1 --micro-batch $MB --lr 1e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed 1 \
  --fast-moe --pack --fsdp-no-ckpt-layers ${NOCK:-0} --on-diverge rollback --snapshot-every 300 --snapshot-keep 2 --resume-every 20 --state-dir $STATE $EXTRA"
TR="venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NP --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT"
FIRST=""; [ -f sweeps/$TAG/run.json ] && FIRST="--resume"
echo "args: $ARGS" >> sweeps/$TAG/train.log
. $R/mixgrid/retry_lib.sh
echo "rc=$rc reason=$REASON $(date -u +%T)" > sweeps/$TAG/train.done
