from pathlib import Path
import sys,base64
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
root=Path(__file__).resolve().parent
payload={str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in [*root.glob('*.py'),*root.glob('*.json'),*root.glob('configs/*.json')] if not p.name.startswith(('deploy','inspect','download'))}
code='PAYLOAD='+repr(payload)+'\n'+'''
from pathlib import Path
import base64,subprocess,sys,os,json
r=Path('/data/workspace/current-data-v1');assert not (r/'prepare-process.json').exists(),'Reconcile existing process first'
for name,blob in PAYLOAD.items():
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(blob))
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',HF_HOME=str(r/'hf-home'),TOKENIZERS_PARALLELISM='false')
with (r/'prepare.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'prepare.py')],cwd=r,env=env,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]};(r/'prepare-process.json').write_text(json.dumps(d));print(d)
'''
run(code)
