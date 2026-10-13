#!/bin/bash
# Mix-grid orchestrator (Oct 7): wide pool -> mixA/mixB builds -> tokenizer fit -> two MoE FSDP runs on A100 pairs. Log: /data/workspace/mixgrid-20261007/orchestrate.log
set -u
R=/tmp/mixgrid-20261007; D=/data/workspace/mixgrid-20261007; SMP=/data/workspace/corpus-sampling-20261007; F=/tmp/final-20261007
L=$D/orchestrate.log; say(){ echo "{\"t\":\"$(date -u +%T)\",\"event\":\"$*\"}" >> $L; }
cd $R; export PYTHONPATH=$F/vendor HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
say start
if [ ! -f $SMP/pool-v1/stats.json ]; then
  say pool_start
  (cd $SMP && python -u build_pool.py --calibration /data/workspace/human-calibration-20261006/selection.json --out $SMP/pool-v1 --workers 64 > $D/pool-v1.log 2>&1) || { say pool_failed; exit 1; }
  say pool_done
fi
for M in mixA mixB; do
  if [ ! -f $D/$M/mix-report.json ]; then
    say build_$M
    python -u mix_build.py --inputs /data/workspace/final-20261007/mix/mix-inputs.tgz --out $D/$M --plan $R/plan-$M.json --pool $SMP/pool-v1 --workers 24 > $D/$M-build.out 2>&1 || { say build_${M}_failed; exit 1; }
  fi
  if [ ! -f $R/runs/moe-$M/prepared-v2/manifest.json ]; then
    say fit_$M
    rm -rf $R/runs/moe-$M/prepared-v2; mkdir -p $R/runs/moe-$M
    python -u fit_rows.py $D/$M/prepared-v2 $R/runs/moe-$M/prepared-v2 $F/assets > $D/$M-fit.log 2>&1 || { say fit_${M}_failed; exit 1; }
    rm -rf $D/$M/fitted-prepared-v2; cp -r $R/runs/moe-$M/prepared-v2 $D/$M/fitted-prepared-v2
  fi
done
say mixes_ready
free(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$1" '$1==g{print ($2<1000)}'; }
for G in 0 1 4 5; do while [ "$(free $G)" != "1" ]; do say waiting_gpu_$G; sleep 60; done; done
say launching
bash run_fsdp.sh moe-mixA moe-mixA 0,1 29701 &
bash run_fsdp.sh moe-mixB moe-mixB 4,5 29702 &
wait
say all_done
