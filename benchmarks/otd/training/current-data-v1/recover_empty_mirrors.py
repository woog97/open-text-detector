from pathlib import Path
import base64,sys
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
p=Path(__file__).with_name('prepare.py')
run('PAYLOAD='+repr(base64.b64encode(p.read_bytes()).decode())+'\n'+'''
from pathlib import Path
import json,base64,subprocess,sys
r=Path('/data/workspace/current-data-v1')
assert json.loads((r/'prepare-status.json').read_text())['state']=='failed'
for pid in [63479,63762]:
 p=Path('/proc',str(pid),'stat');assert not p.exists() or p.read_text().split()[2]=='Z'
assert not (r/'prepared-v2').exists()
(r/'prepare.py').write_bytes(base64.b64decode(PAYLOAD))
# Preserve the failed attempt. Point training/verification at the new version.
for name in ['train.py','preflight.py','calibrate.py']:
 p=r/name;s=p.read_text().replace("/'prepared'","/'prepared-v2'");p.write_text(s)
with (r/'prepare-v3.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'prepare.py')],cwd=r,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]};(r/'prepare-process-v3.json').write_text(json.dumps(d));print(d)
''')
