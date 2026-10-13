#!/bin/bash
# Space side of the Atlas rescore, one worker per GPU given (Alice allowed A100s 0-3, and 5 and 7 from 11:40 AM Oct 7; never 4 or 6).
# Each worker: 1) human calibration papers (2,299 Oct 6 + 215 Oct 5), 2) year diagnostic (7,102 papers, 2023-2026),
# 3) the Atlas papers from shard 0 upward, skipping shards the H200 already synced to out-h200.
cd /tmp/final-20261007; D=/data/workspace/atlas-rescore-moe-20261007; L=/tmp/atlas-rescore
export PYTHONPATH=/tmp/final-20261007/vendor OTD_EXTRA_PATH=/tmp/final-20261007/vendor-peft021 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
C="python -u score_moe.py --run /tmp/final-20261007/runs-moe --assets /tmp/final-20261007/assets/qwen36-35b-a3b --batch-size 32 --merge"
for g in "$@"; do
  case $g in 0|1|2|3|5|7) ;; *) echo "refusing GPU $g"; continue ;; esac
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "
    $C --source $L/human-cal-clean --source /data/workspace/classifications/calibration-inputs/human-pre2023-batch001-v1 --output $D/out-cal --claims $L/claims-cal --worker space-gpu$g --rows-per-shard 50 &&
    $C --source /data/workspace/classifications/calibration-inputs/year-diagnostic-v1 --output $D/out-yd --claims $L/claims-yd --worker space-gpu$g &&
    $C --source /data/workspace/datasets/iclr2027-clean-text-v2-20261005 --source /data/workspace/datasets/iclr2027-backfill-clean-text-v2-20261005 --output $D/out-space --claims $L/claims-atlas --done-dir $D/out-h200 --worker space-gpu$g
  " >> $L/worker-gpu$g.log 2>&1 < /dev/null &
done
