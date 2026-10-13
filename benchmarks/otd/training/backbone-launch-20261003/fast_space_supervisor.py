"""Space-local working copies, direct persistent-bucket backups, no mount in hot path."""
from pathlib import Path
import os,sys,json,time,subprocess,concurrent.futures
from huggingface_hub import HfApi
R=Path('/tmp/pangram-space-fast10');BUCKET='open-text-detector/training-storage';SOURCE='workspace/backbone-launch-20261003';DEST='workspace/backbone-fast10-20261003'
api=HfApi();names=['ettin-1b','qwen35-4b','qwen35-9b']
def publish(p):
 api.batch_bucket_files(BUCKET,add=[(p,DEST+'/'+str(p.relative_to(R)))])
def status(d):
 p=R/'supervisor-status.json';p.write_text(json.dumps({'time':time.time(),**d}));publish(p);print(p.read_text(),flush=True)
def copy_prefix(prefix,dest):
 files=[x for x in api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True) if hasattr(x,'size') and '/.cache/' not in x.path]
 assert files,prefix
 pairs=[]
 for x in files:
  p=dest/x.path[len(prefix.rstrip('/'))+1:];p.parent.mkdir(parents=True,exist_ok=True);pairs.append((x,p))
 api.download_bucket_files(BUCKET,pairs,raise_on_missing_files=True)
 assert all(p.stat().st_size==x.size for x,p in pairs)
 return len(pairs)
def main():
 status({'state':'installing_local_runtime'})
 with (R/'install.log').open('w') as f:
  subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--no-deps','--target',str(R/'vendor'),'peft==0.18.1','bitsandbytes==0.49.2','fla-core==0.5.2','flash-linear-attention==0.5.2','einops==0.8.2'],stdout=f,stderr=f,check=True)
 status({'state':'copying_existing_bucket_files'})
 def prepare(name):
  a=copy_prefix(SOURCE+'/assets/'+name,R/'assets'/name)
  b=copy_prefix(SOURCE+'/runs/'+name+'/prepared-v2',R/'runs'/name/'prepared-v2')
  print(json.dumps({'copied':name,'asset_files':a,'schedule_files':b}),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(prepare,names))
 processes={}
 for name,gpu in zip(names,[1,2,3]):
  root=R/'runs'/name;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
  assert int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
  with (root/'training.log').open('a') as f:p=subprocess.Popen([sys.executable,'-u',str(R/'train_short.py'),name],env=env,cwd=R,stdout=f,stderr=f,start_new_session=True)
  d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'gpu':gpu,'time':time.time()};(root/'training-process.json').write_text(json.dumps(d));publish(root/'training-process.json');processes[name]=p
 status({'state':'training','names':names})
 last={}
 while True:
  for name,p in processes.items():
   root=R/'runs'/name
   for f in list(root.glob('*.json'))+list((root/'run').glob('*.json'))+[root/'training.log']:
    if f.exists() and f.stat().st_size<2_000_000:
     stamp=(f.stat().st_mtime_ns,f.stat().st_size)
     if last.get(str(f))!=stamp:
      try:publish(f);last[str(f)]=stamp
      except Exception as e:print('backup retry',type(e).__name__,flush=True)
  if all(p.poll() is not None for p in processes.values()):
   status({'state':'workers_exited','exit_codes':{n:p.returncode for n,p in processes.items()}});return
  time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception as e:
  status({'state':'failed','error':repr(e)});raise
