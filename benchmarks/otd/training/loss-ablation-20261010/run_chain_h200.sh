#!/bin/bash
# One run end to end on one H200: train with run_abl_h200.sh, then FAST_EVAL=1 eval_final.py on the same GPU.
# Usage (from the final-20261007 work dir, W&B key on stdin): run_chain_h200.sh TAG GPU PORT [ARM FLAGS...]
set -u; TAG=$1; GPU=${2%%,*}; R=/workspace/woog/pangram/final-20261007; cd $R; read -r K  # eval runs on the first GPU of the list
printf '%s\n' "$K" | bash mixgrid/run_abl_h200.sh "$@"; K=""
if grep -q '^rc=0' sweeps/$TAG/train.done 2>/dev/null; then
  until [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 20; done
  mkdir -p sweeps/$TAG/eval; echo "eval start $(date -u +%T)" >> sweeps/$TAG/chain.log
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$R/ccdshim:$R/vendor HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 FAST_EVAL=1 \
    venv/bin/python -u eval_final.py $TAG >> sweeps/$TAG/eval/gpu$GPU.out 2>&1
  echo "eval rc=$? summary=$([ -f sweeps/$TAG/eval/summary.json ] && echo yes || echo no) $(date -u +%T)" >> sweeps/$TAG/chain.log
else
  echo "training failed: $(cat sweeps/$TAG/train.done 2>/dev/null); no eval $(date -u +%T)" >> sweeps/$TAG/chain.log
fi
