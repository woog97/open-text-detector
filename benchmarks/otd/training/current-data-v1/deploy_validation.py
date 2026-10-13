from pathlib import Path
import base64,sys
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
r=Path(__file__).resolve().parent
payload={n:base64.b64encode((r/n).read_bytes()).decode() for n in ['preflight.py','validate_when_ready.py']}
run('PAYLOAD='+repr(payload)+'\n'+'''
from pathlib import Path
import base64,subprocess,sys,json
r=Path('/data/workspace/current-data-v1');assert not (r/'validation-process.json').exists()
for n,v in PAYLOAD.items():(r/n).write_bytes(base64.b64decode(v))
with (r/'validation.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'validate_when_ready.py')],cwd=r,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]};(r/'validation-process.json').write_text(json.dumps(d));print(d)
''')
