from pathlib import Path
import sys,netrc
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
r=Path(__file__).resolve().parent
files={n:(r/n).read_text() for n in ['train_short.py','adapters_short.py','worker_short.py']}
key=netrc.netrc().authenticators('api.wandb.ai')[2]
code=r'''from pathlib import Path
import os,subprocess,json,time,sys
r=Path('/data/workspace/backbone-launch-20261003')
for name,gpu in [('ettin-1b',1),('qwen35-4b',2),('qwen35-9b',3)]:
 root=r/'runs'/name
 assert not (root/'short-launch.json').exists(),'Already launched'
 for f in ['worker-process.json','training-process.json']:
  d=json.loads((root/f).read_text());p=Path('/proc',str(d['pid']),'stat')
  if p.exists():assert p.read_text().split()[21]!=str(d['start_ticks']) or p.read_text().split()[2]=='Z'
 assert int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
 assert (root/'prepared-v2/manifest.json').exists()
for n,s in FILES.items():(r/n).write_text(s)
if not Path('/tmp/pangram-wandb-vendor/wandb').exists():
 subprocess.run([sys.executable,'-m','pip','install','--target','/tmp/pangram-wandb-vendor','wandb==0.30.0'],check=True,stdout=subprocess.DEVNULL)
for name,gpu in [('ettin-1b',1),('qwen35-4b',2),('qwen35-9b',3)]:
 root=r/'runs'/name;archive=root/'attempts'/('storage-recovery-10pct-'+str(int(time.time())));archive.mkdir(parents=True)
 for f in ['run','preflight.json','wandb','wandb-tracking.json','training-process.json','worker-process.json','worker-status.json','training.log','worker.log']:
  p=root/f
  if p.exists():p.rename(archive/f)
 env=os.environ.copy();env.update(WANDB_API_KEY=KEY,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
 with (root/'worker.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'worker_short.py'),name],env=env,cwd=r,stdout=f,stderr=f,start_new_session=True)
 d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'gpu':gpu,'time':time.time(),'training_examples':8400,'archive':str(archive)}
 (root/'worker-process.json').write_text(json.dumps(d));(root/'short-launch.json').write_text(json.dumps(d));print(json.dumps({'name':name,**d}),flush=True)
'''
run('FILES='+repr(files)+'\nKEY='+repr(key)+'\n'+code)
