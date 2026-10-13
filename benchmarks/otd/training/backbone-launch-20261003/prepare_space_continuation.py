from pathlib import Path
import sys,netrc
from huggingface_hub import get_token
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
code=r'''from pathlib import Path
import os,json,subprocess,sys,time
r=Path('/tmp/pangram-space-fast10');s=(r/'train_short.py').read_text()
s=s.replace("assert not (out/'run.json').exists(),'Existing training run: do not restart'", "assert (out/'stage1-epoch0-adapters.safetensors').exists(),'Completed stage1 checkpoint required'")
s=s.replace("save(out/'run.json',", "save(out/'continuation.json',")
s=s.replace(" # Same metrics-only payload", " restore(model,out/'stage1-epoch0-adapters.safetensors')\n # Same metrics-only payload")
s=s.replace("wr=wandb.init(entity=", "prior=json.loads((root/'wandb-tracking.json').read_text());wr=wandb.init(id=prior['run_id'],resume='must',entity=")
s=s.replace("history=[];started=time.time();global_step=0;processed=0;done_rows=0", "history=json.loads((out/'history.json').read_text());started=time.time();global_step=38;processed=0;done_rows=1200")
s=s.replace('for stage in [1,2]:','for stage in [2]:')
compile(s,'continue_short.py','exec');(r/'continue_short.py').write_text(s)
(r/'continue_watch.py').write_text(WATCH)
e=os.environ.copy();e.update(HF_TOKEN=HFKEY,WANDB_API_KEY=WKEY,HF_HUB_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
with (r/'continue-watch.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(r/'continue_watch.py')],env=e,cwd=r,stdout=f,stderr=f,start_new_session=True)
print(json.dumps({'pid':p.pid,'action':'Wait for natural exit; preserve stage1 checkpoint; resume stage2 without discarding updates'}))
'''
watch=r'''from pathlib import Path
import os,sys,json,time,subprocess
from huggingface_hub import HfApi
R=Path('/tmp/pangram-space-fast10');B='open-text-detector/training-storage';PREFIX='workspace/backbone-fast10-20261003'
api=HfApi();names=['ettin-1b','qwen35-4b','qwen35-9b'];last={};continued=set()
def publish(p):api.batch_bucket_files(B,add=[(p,PREFIX+'/'+str(p.relative_to(R)))])
for p in list(R.glob('*.py'))+[R/'models.json']:publish(p)
while True:
 states={}
 for name,gpu in zip(names,[1,2,3]):
  root=R/'runs'/name;out=root/'run';sp=out/'status.json';state=json.loads(sp.read_text()) if sp.exists() else {};states[name]=state
  if state.get('state')=='failed' and 'offline' in state.get('error','').lower() and name not in continued:
   ident=json.loads((root/'training-process.json').read_text());proc=Path('/proc',str(ident['pid']),'stat')
   alive=proc.exists() and proc.read_text().split()[21]==str(ident['start_ticks']) and proc.read_text().split()[2]!='Z'
   if not alive:
    cp=out/'stage1-epoch0-adapters.safetensors';assert cp.exists()
    publish(cp);assert api.get_bucket_file_metadata(B,PREFIX+'/'+str(cp.relative_to(R))).size==cp.stat().st_size
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
    with (root/'continuation.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(R/'continue_short.py'),name],env=env,cwd=R,stdout=f,stderr=f,start_new_session=True)
    ident={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'gpu':gpu,'time':time.time(),'resumed_after_stage1':True};(root/'training-process.json').write_text(json.dumps(ident));continued.add(name);print(json.dumps({'continued':name,**ident}),flush=True)
  for p in list(root.glob('*.json'))+list(out.glob('*.json')):
   stamp=(p.stat().st_mtime_ns,p.stat().st_size)
   if last.get(str(p))!=stamp:
    try:publish(p);last[str(p)]=stamp
    except Exception as e:print('backup retry',type(e).__name__,flush=True)
 if all(s.get('state')=='trained_calibration_pending' for s in states.values()):break
 time.sleep(20)
'''
run('WATCH='+repr(watch)+'\nHFKEY='+repr(get_token())+'\nWKEY='+repr(netrc.netrc().authenticators('api.wandb.ai')[2])+'\n'+code)
