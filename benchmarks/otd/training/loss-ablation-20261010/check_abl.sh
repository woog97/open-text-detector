#!/bin/bash
# Status of the loss-ablation runs on the H200 (read-only). Usage: check_abl.sh [TAGS...]
TAGS="${*:-abl-nostage1 abl-noseg-nomix abl-tok-sent abl-uniform-tok abl-tok-only abl-sent1 abl-control}"
ssh -o ConnectTimeout=20 otd-h200 "cd /workspace/woog/pangram/final-20261007; date -u +%T; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
for t in $TAGS; do
  w=\$(cat sweeps/\$t.chain.pid 2>/dev/null || cat sweeps/\$t.wrapper.pid 2>/dev/null); alive=dead; kill -0 \$w 2>/dev/null && alive=alive
  s=\$(python3 -c \"import json;d=json.load(open('sweeps/\$t/status.json'));print(d.get('state'),'stage',d.get('stage'),'step',d.get('optimizer_steps'),'rows',d.get('logical_rows'),'/',d.get('total_rows'),'loss %.3f'%d['loss'] if 'loss' in d else '','ema',d.get('loss_ema'),'min',d.get('loss_ema_min'),'gn',d.get('grad_norm'),'rb',d.get('rollbacks'),'mem',d.get('max_memory_gb'),'el %.0fm'%(d.get('elapsed_seconds',0)/60))\" 2>/dev/null || echo no-status)
  e=\$(grep -E 'Error|Traceback|out of memory|retry|Nonfinite|rollback' sweeps/\$t/train.log 2>/dev/null | grep -v -e 'kernels-community' -e '^args:' -e '^{' | tail -2 | cut -c1-200 | tr '\n' ' ')
  d=\$(cat sweeps/\$t/train.done 2>/dev/null)
  echo \"\$t [\$alive] \$s \${d:+DONE: \$d} \${e:+ERR: \$e}\"
done
echo \"scheduler: \$(kill -0 \$(cat sweeps/abl-round2.pid) 2>/dev/null && echo alive || echo dead); \$(tail -3 sweeps/abl-round2.log | tr '\n' ' ')\"
echo \"archiver: \$(kill -0 \$(cat sweeps/abl-archive.pid) 2>/dev/null && echo alive || echo dead); \$(tail -2 sweeps/abl-archive.log | tr '\n' ' ')\""
