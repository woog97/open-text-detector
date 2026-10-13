"""Retry only confirmed pre-training failures, preserving the failed attempt."""
from pathlib import Path
import netrc,sys
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
key=netrc.netrc().authenticators('api.wandb.ai')[2]
code='''
from pathlib import Path
import json,subprocess,sys,time,os
r=Path('/data/workspace/backbone-launch-20261003')
for name,gpu in [('ettin-1b',1),('qwen35-4b',2)]:
 root=r/'runs'/name
 for f in ['worker-process.json','training-process.json']:
  d=json.loads((root/f).read_text());p=Path('/proc',str(d['pid']),'stat')
  if p.exists():
   s=p.read_text().split();assert s[2]=='Z' or s[21]!=str(d['start_ticks']),name+' process still running'
 assert json.loads((root/'run/status.json').read_text())['state']=='failed'
 assert not (root/'run/run.json').exists(),'Model had reached preflight; inspect separately'
 assert not (root/'wandb-tracking.json').exists()
 assert int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
 if name=='ettin-1b':assert (r/'assets'/name/'pytorch_model.bin').stat().st_size>1000000000
 archive=root/'attempts'/('before-training-'+str(int(time.time())));archive.mkdir(parents=True)
 for f in ['run','worker-process.json','training-process.json','worker-status.json','worker.log','training.log']:
  p=root/f
  if p.exists():p.rename(archive/f)
 env=os.environ.copy();env.update(WANDB_API_KEY=KEY,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
 with (root/'worker.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'worker.py'),name],env=env,cwd=r,stdout=f,stderr=f,start_new_session=True)
 d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'gpu':gpu,'time':time.time()};(root/'worker-process.json').write_text(json.dumps(d));print(json.dumps({'name':name,**d}),flush=True)
'''
run('KEY='+repr(key)+'\n'+code)
