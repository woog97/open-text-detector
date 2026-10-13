from pathlib import Path
import sys
from huggingface_hub import get_token
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
source=Path(__file__).with_name('supervisor.py').read_text()
code='''from pathlib import Path
import os,subprocess,sys,json,time
r=Path('/tmp/pangram-eval-20261003');r.mkdir(exist_ok=True)
assert not (r/'supervisor-process.json').exists(),'Evaluation already launched'
(r/'supervisor.py').write_text(SOURCE)
env=os.environ.copy();env.update(HF_TOKEN=TOKEN,HF_HUB_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
with (r/'supervisor.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'supervisor.py')],env=env,cwd=r,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'time':time.time(),'root':str(r)};(r/'supervisor-process.json').write_text(json.dumps(d));print(json.dumps(d))
'''
run('SOURCE='+repr(source)+'\nTOKEN='+repr(get_token())+'\n'+code)
