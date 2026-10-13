#!/bin/bash
# Usage: start_when_free.sh MODEL TAG GPUS PORT  (W&B key on stdin). Waits until every listed GPU is free, then runs run_ddp_h200.sh.
set -u; read -r K; MODEL=$1; TAG=$2; G=$3; PORT=$4
free() { local g u; for g in ${G//,/ }; do u=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1==g{print $2}'); [ "${u:-1}" -lt 100 ] || return 1; done; }
until free; do echo "waiting for GPUs $G $(date -u +%T)"; sleep 60; done
printf '%s\n' "$K" | bash mixgrid/run_ddp_h200.sh $MODEL $TAG $G $PORT
