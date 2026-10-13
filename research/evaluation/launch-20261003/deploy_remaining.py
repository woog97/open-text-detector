from pathlib import Path
import sys,subprocess
from huggingface_hub import get_token
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
source=Path(__file__).with_name('remaining_worker.py').read_text();compile(source,'remaining_worker.py','exec')
launcher='''from pathlib import Path
import os,sys,subprocess,json
root=Path(ROOT);root.mkdir(parents=True,exist_ok=True);script=root/'remaining_worker.py';script.write_text(SOURCE)
env=os.environ.copy();env.update(HF_TOKEN=TOKEN,HF_HUB_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
for name,gpu in NAMES:
 dest=root/name;dest.mkdir(exist_ok=True)
 assert not (dest/'launch.json').exists(),'Already launched '+name
 with (dest/'worker.log').open('a') as f:p=subprocess.Popen([PYTHON,'-u',str(script),name,str(gpu)],env=env,cwd=root,stdout=f,stderr=f,start_new_session=True)
 d={'name':name,'pid':p.pid,'gpu':gpu};(dest/'launch.json').write_text(json.dumps(d));print(json.dumps(d),flush=True)
'''
shared='SOURCE='+repr(source)+'\nTOKEN='+repr(get_token())+'\n'
if sys.argv[1]=='space':run(shared+"ROOT='/tmp/pangram-eval-20261003/remaining'\nNAMES=[('qwen35-4b',2),('qwen35-9b',3)]\nimport sys\nPYTHON=sys.executable\n"+launcher)
else:
 code=shared+"ROOT='/workspace/woog/pangram/backbones-20261003/evaluations-20261003'\nNAMES=[('gemma4-12b',0),('qwen36-35b-a3b',1)]\nPYTHON='/workspace/woog/pangram/backbones-20261003/venv/bin/python'\n"+launcher
 subprocess.run(['ssh','-o','BatchMode=yes','otd-h200','python3 -'],input=code,text=True,check=True)
