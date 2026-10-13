from pathlib import Path
import sys
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
src=Path(__file__).with_name('download_models.py').read_text()
run('SOURCE='+repr(src)+'\n'+'''
from pathlib import Path
import subprocess,sys,os,json,time
r=Path('/data/workspace/backbone-launch-20261003');r.mkdir(exist_ok=True)
assert not (r/'download-process.json').exists(),'Existing download identity: inspect first'
(r/'download_models.py').write_text(SOURCE)
env=os.environ.copy();env['HF_HUB_OFFLINE']='0';env['HF_HOME']='/data/workspace/hf-home'
with (r/'downloads.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'download_models.py')],env=env,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'time':time.time()};(r/'download-process.json').write_text(json.dumps(d));print(d)
''')
