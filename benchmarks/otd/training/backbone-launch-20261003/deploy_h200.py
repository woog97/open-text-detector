"""Launch the two enabled H200 runs; credentials travel only through stdin/env."""
import netrc,subprocess
from pathlib import Path
key=netrc.netrc().authenticators('api.wandb.ai')[2]
code='''
import os,subprocess,json
from pathlib import Path
r=Path('/workspace/woog/pangram/backbones-20261003')
models=json.loads((r/'models.json').read_text())
assert [m['name'] for m in models if m['host']=='h200' and m.get('enabled',True)]==['gemma4-12b','qwen36-27b']
e=os.environ.copy();e['WANDB_API_KEY']=KEY
subprocess.run([str(r/'venv/bin/python'),'-u',str(r/'launch_workers.py'),'h200'],env=e,check=True)
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','otd-h200','python3 -'],input='KEY='+repr(key)+'\n'+code,text=True)
raise SystemExit(p.returncode)
