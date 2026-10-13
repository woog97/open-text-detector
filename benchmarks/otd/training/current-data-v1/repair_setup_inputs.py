from pathlib import Path
import sys,base64
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
root=Path(__file__).resolve().parent
payload={p.name:base64.b64encode(p.read_bytes()).decode() for p in [root/'prepare.py',root/'runtime.py']}
for name in ['selection','calibration','selection-windows','calibration-windows']:
 p=root.parent/'backbone-comparison/prepared'/f'{name}.jsonl.gz';payload['sources/frozen-paper-evaluation/'+p.name]=base64.b64encode(p.read_bytes()).decode()
code='PAYLOAD='+repr(payload)+'\n'+'''
from pathlib import Path
import base64,json,subprocess,sys,os
r=Path('/data/workspace/current-data-v1');d=json.loads((r/'prepare-process.json').read_text());p=Path('/proc',str(d['pid']),'stat')
assert not p.exists() or p.read_text().split()[2]=='Z','Preparation still running'
for name,blob in PAYLOAD.items():
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(blob))
with (r/'dependencies.log').open('a') as f:
 p=subprocess.Popen([sys.executable,'-m','pip','install','--target',str(r/'vendor'),'--no-deps','peft==0.18.1','bitsandbytes==0.49.2'],stdout=f,stderr=f,start_new_session=True)
(r/'dependencies-process.json').write_text(json.dumps({'pid':p.pid}))
with (r/'prepare-v2.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'prepare.py')],cwd=r,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]};(r/'prepare-process-v2.json').write_text(json.dumps(d));print(d)
'''
run(code,timeout=120)
