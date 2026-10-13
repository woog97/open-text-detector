"""Isolated full-paper confirmation of the second optimization round."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/benchmarks/meld-v5-max'
choice=min((r for r in json.loads((OUT/'combined-sweep.json').read_text()) if 'seconds'in r and r['tag']!='paired-control' and not r['doc_changes']),key=lambda r:r['seconds'])
(OUT/'selection.json').write_text(json.dumps(choice,indent=2))
steps=[('control-full',False,2),('max-full',True,2),('control-repeat',False,1),('max-repeat',True,1)]
for tag,selected,repeats in steps:
 env=choice['env'] if selected else {}
 args=[sys.executable,str(ROOT/'scripts/benchmark_meld_v5_max.py'),'--runtime','mlx','--precision','float16','--attention','tiled','--tag',tag,'--repeats',str(repeats),'--pipeline',str(choice['depth'] if selected else 0),'--streams',str(choice['streams'] if selected else 1),'--advanced',json.dumps(choice['advanced'] if selected else {})]
 with (OUT/(tag+'.log')).open('w') as log:
  subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,**env})
 print('completed',tag,flush=True)
