#!/bin/bash
# Usage: run_soft_t21.sh PYTHON — wait for GPUs 0,1,4-7 to be free of T2.1 scoring, build the soft-label prepared dir, then
# 3 seeds on GPU pairs 0-1, 4-5, 6-7 (2 and 3 are reserved) and one scoring worker per GPU once its pair has trained.
R=/tmp/pangram-t2-20261006; cd $R; PY=$1; S=$R/sweeps
$PY -u soft_gate_build.py >> $S/soft-gate.out 2>&1 || { echo "soft build failed $(date -u +%H:%M:%S)" > $S/soft-t21.failed; exit 1; }
bash run_pair_soft.sh 1 0,1 29531 & bash run_pair_soft.sh 2 4,5 29532 & bash run_pair_soft.sh 3 6,7 29533 &
unset WANDB_API_KEY
for g in 0 1; do (while [ ! -f $S/q4b-T21S-s1/train.done ]; do sleep 30; done; $PY -u score_worker_t21s.py $g >> $S/score3-gpu$g.out 2>&1) & done
for g in 4 5; do (while [ ! -f $S/q4b-T21S-s2/train.done ]; do sleep 30; done; $PY -u score_worker_t21s.py $g >> $S/score3-gpu$g.out 2>&1) & done
for g in 6 7; do (while [ ! -f $S/q4b-T21S-s3/train.done ]; do sleep 30; done; $PY -u score_worker_t21s.py $g >> $S/score3-gpu$g.out 2>&1) & done
wait
