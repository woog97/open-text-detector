"""Remote launcher. Secrets are inherited from environment, never saved."""
from pathlib import Path
import os,sys,json,subprocess,time
R=Path(__file__).resolve().parent
host=sys.argv[1];models=[m for m in json.loads((R/'models.json').read_text()) if m['host']==host and m.get('enabled',True)]
assert os.environ.get('WANDB_API_KEY'),'Metrics tracking credential missing'
for m in models:
 r=R/'runs'/m['name'];r.mkdir(parents=True,exist_ok=True)
 assert not (r/'worker-process.json').exists(), 'Existing worker must be inspected before any restart'
 assert json.loads((R/'downloads'/m['name']/'status.json').read_text())['state']=='complete'
 for gpu in m.get('gpus',[m['gpu']]):
  mem=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  assert mem<100, f'GPU {gpu} busy: {mem}MB' 
for m in models:
 r=R/'runs'/m['name'];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,m.get('gpus',[m['gpu']]))),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off',TOKENIZERS_PARALLELISM='false')
 with (r/'worker.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(R/'worker.py'),m['name']],cwd=R,env=env,stdout=f,stderr=f,start_new_session=True)
 d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'gpu':m['gpu'],'time':time.time()};(r/'worker-process.json').write_text(json.dumps(d));print(json.dumps({'name':m['name'],**d}),flush=True)
