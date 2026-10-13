#!/bin/bash
# Start (or restart) the H200 Atlas rescore workers: one per GPU in $@ (default 0 1 2). Never touches GPU 3.
cd /workspace/woog/pangram/final-20261007; L=/workspace/woog/pangram/atlas-rescore; T=$L/text
GPUS=${@:-0 1 2}
for g in $GPUS; do
  [ "$g" = 3 ] && continue
  pkill -f "worker h200-gpu$g\$" ; sleep 2
  CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OTD_EXTRA_PATH=$L/vendor-ccd setsid nohup \
    venv/bin/python -u score_moe.py --source $T/iclr2027-clean-text-v2-20261005 --source $T/iclr2027-backfill-clean-text-v2-20261005 \
    --run sweeps/final-moe --assets assets/qwen36-35b-a3b --batch-size 32 --merge --order desc --output $L/out --claims $L/claims \
    --done-dir $L/space-done --worker h200-gpu$g >> $L/worker-gpu$g.log 2>&1 < /dev/null &
done
sleep 1; pgrep -af "score_moe" | grep -c "worker h200"
