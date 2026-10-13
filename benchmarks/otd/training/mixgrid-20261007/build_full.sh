#!/bin/bash
# Build and fit the full-size mixes (Oct 8): mixA-full (~144k stage-2 rows) and mixB-full (~717k). Log: $D/build-full.log
set -u
R=/tmp/mixgrid-20261007; D=/data/workspace/mixgrid-20261007; SMP=/data/workspace/corpus-sampling-20261007; F=/tmp/final-20261007
L=$D/build-full.log; say(){ echo "{\"t\":\"$(date -u +%T)\",\"event\":\"$*\"}" >> $L; }
cd $R; export PYTHONPATH=$F/vendor HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for M in mixA mixB; do
  HW=hosts; [ $M = mixB ] && HW=full
  if [ ! -f $D/$M-full/mix-report.json ]; then
    say build_$M-full
    python -u mix_build.py --inputs /data/workspace/final-20261007/mix/mix-inputs.tgz --out $D/$M-full --plan $R/plan-$M-full.json \
      --pool $SMP/pool-v1 --human-windows $HW --doc-window-cap 0 --workers 64 > $D/$M-full-build.out 2>&1 || { say build_${M}_full_failed; continue; }
  fi
  if [ ! -f $R/runs/moe-$M-full/prepared-v2/manifest.json ]; then
    say fit_$M-full
    rm -rf $R/runs/moe-$M-full/prepared-v2; mkdir -p $R/runs/moe-$M-full
    python -u fit_rows.py $D/$M-full/prepared-v2 $R/runs/moe-$M-full/prepared-v2 $F/assets > $D/$M-full-fit.log 2>&1 || { say fit_${M}_full_failed; continue; }
    rm -rf $D/$M-full/fitted-prepared-v2; cp -r $R/runs/moe-$M-full/prepared-v2 $D/$M-full/fitted-prepared-v2
    say fitted_$M-full
  fi
done
say all_done
