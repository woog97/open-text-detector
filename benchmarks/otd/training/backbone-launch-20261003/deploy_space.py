from pathlib import Path
import sys,netrc
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
r=Path(__file__).resolve().parent
files={p.name:p.read_text() for p in r.iterdir() if p.suffix in ('.py','.json') and not p.name.startswith('deploy')}
key=netrc.netrc().authenticators('api.wandb.ai')[2]
code='''
from pathlib import Path
import subprocess,os,sys
r=Path('/data/workspace/backbone-launch-20261003')
for n,s in FILES.items():(r/n).write_text(s)
env=os.environ.copy();env['WANDB_API_KEY']=KEY
subprocess.run([sys.executable,'-u',str(r/'launch_workers.py'),'space'],env=env,check=True)
'''
run('FILES='+repr(files)+'\nKEY='+repr(key)+'\n'+code)
