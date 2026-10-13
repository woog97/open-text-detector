from pathlib import Path
import netrc,sys
from huggingface_hub import get_token
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
r=Path(__file__).resolve().parent
files={n:(r/n).read_text() for n in ['train_short.py','adapters_short.py','data.py','modeling.py','models.json','fast_space_supervisor.py']}
files['runtime.py']="from pathlib import Path\nimport sys,os\nROOT=Path(__file__).resolve().parent\nsys.path.insert(0,str(ROOT/'vendor'))\nsys.path.insert(0,'/tmp/pangram-wandb-vendor')\nSPACE_CACHE=str(ROOT/'assets')\ndef require_space():\n assert str(ROOT)=='/tmp/pangram-space-fast10'\nos.environ.setdefault('HF_HUB_OFFLINE','1')\nos.environ.setdefault('TOKENIZERS_PARALLELISM','false')\n"
s=files['train_short.py'];s=s.replace("tmp.replace(path)\n", "tmp.replace(path)\n from huggingface_hub import HfApi\n api=HfApi();dest='workspace/backbone-fast10-20261003/'+str(path.relative_to(ROOT))\n for attempt in range(4):\n  try:\n   api.batch_bucket_files('open-text-detector/training-storage',add=[(path,dest)])\n   assert api.get_bucket_file_metadata('open-text-detector/training-storage',dest).size==path.stat().st_size\n   break\n  except Exception:\n   if attempt==3:raise\n   time.sleep(5*(attempt+1))\n",1)
files['train_short.py']=s
code=r'''from pathlib import Path
import os,json,signal,subprocess,time,sys
r=Path('/tmp/pangram-space-fast10');assert not r.exists(),'Inspect existing fast working directory before retry';r.mkdir()
for pid,name in [(574,'ettin-1b'),(575,'qwen35-4b'),(576,'qwen35-9b')]:
 p=Path('/proc',str(pid))
 if p.exists():
  cmd=(p/'cmdline').read_bytes();assert b'train_short.py' in cmd and name.encode() in cmd
  os.kill(pid,signal.SIGTERM)
for n,s in FILES.items():(r/n).write_text(s)
env=os.environ.copy();env.update(HF_TOKEN=HFKEY,WANDB_API_KEY=WKEY)
with (r/'supervisor.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'fast_space_supervisor.py')],env=env,cwd=r,stdout=f,stderr=f,start_new_session=True)
print(json.dumps({'pid':p.pid,'root':str(r)}))
'''
run('FILES='+repr(files)+'\nHFKEY='+repr(get_token())+'\nWKEY='+repr(netrc.netrc().authenticators('api.wandb.ai')[2])+'\n'+code)
