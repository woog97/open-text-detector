#!/bin/bash
# Usage: run_t21.sh PYTHON — build the T2.1 mix, persist it to /data, then 3 seeds on GPU pairs 0-1, 4-5, 6-7 (2 and 3 are reserved)
# and one scoring worker per GPU (each waits for its pair's train.done).
R=/tmp/pangram-t2-20261006; cd $R; PY=$1; S=$R/sweeps
PYTHONPATH=$R/vendor $PY -u build_t21.py >> build-t21.out 2>&1
if ! grep -q '"event": "done"' build-t21.log; then echo "build failed $(date -u +%H:%M:%S)" > $S/t21-build.failed; exit 1; fi
$PY persist_t21_build.py >> build-t21.out 2>&1
bash run_pair_t21.sh 1 0,1 29521 & bash run_pair_t21.sh 2 4,5 29522 & bash run_pair_t21.sh 3 6,7 29523 &
unset WANDB_API_KEY
for g in 0 1; do $PY -u score_worker_t21.py $g $S/q4b-T21-s1/train.done >> $S/score-t21-gpu$g.out 2>&1 & done
for g in 4 5; do $PY -u score_worker_t21.py $g $S/q4b-T21-s2/train.done >> $S/score-t21-gpu$g.out 2>&1 & done
for g in 6 7; do $PY -u score_worker_t21.py $g $S/q4b-T21-s3/train.done >> $S/score-t21-gpu$g.out 2>&1 & done
wait
