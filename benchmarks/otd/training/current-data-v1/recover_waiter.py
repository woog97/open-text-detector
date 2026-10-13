from pathlib import Path
import base64,sys
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
p=Path(__file__).with_name('validate_when_ready.py')
run('PAYLOAD='+repr(base64.b64encode(p.read_bytes()).decode())+'\n'+'''
from pathlib import Path
import base64,subprocess,sys,json
r=Path('/data/workspace/current-data-v1');p=Path('/proc/64878/stat');assert not p.exists() or p.read_text().split()[2]=='Z'
(r/'validate_when_ready.py').write_bytes(base64.b64decode(PAYLOAD))
with (r/'validation-v3.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'validate_when_ready.py')],cwd=r,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]};(r/'validation-process-v3.json').write_text(json.dumps(d));print(d)
''')
