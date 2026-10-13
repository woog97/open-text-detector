"""Launch the frozen baseline on the Space. --check validates without starting work."""
import argparse,json,netrc,re,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--gpu',type=int);p.add_argument('--check',action='store_true');a=p.parse_args()
if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,70}',a.name):p.error('Use 3–71 lowercase letters, numbers, or hyphens')
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'));from remote import run
key=None if a.check else netrc.netrc().authenticators('api.wandb.ai')[2]
code=r'''
from pathlib import Path
import os,json,hashlib,subprocess,sys,shutil,time
recipe=Path('/data/workspace/baseline-mix-v1');meta=json.loads((recipe/'recipe.json').read_text())
for name,want in meta['sha256'].items():assert hashlib.sha256((recipe/name).read_bytes()).hexdigest()==want, 'Recipe changed: '+name
report=json.loads((recipe/'split-report.json').read_text());assert all(x['training_documents']<=x['cap'] for x in report['sources'].values())
root=Path('/data/workspace/baseline-mix-runs')/NAME
assert not root.exists(),'Run name already exists; never overwrite or restart implicitly'
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
gpus=[x.split(',') for x in lines];free=[x for x in gpus if int(x[2])<100 and (GPU is None or int(x[0])==GPU)]
if CHECK:
 print(json.dumps({'recipe':'verified','new_run':str(root),'available_gpus':[int(x[0]) for x in free],'sources':report['sources'],'training_started':False}));
else:
 assert free,'Requested GPU is not idle; existing work is untouched'
 gpu=free[0];lockdir=Path('/tmp/pangram-baseline-launch-locks');lockdir.mkdir(exist_ok=True)
 import fcntl
 lock=(lockdir/('gpu-'+gpu[0].strip()+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Recheck after acquiring the launch lock.
 used=int(subprocess.check_output(['nvidia-smi','-i',gpu[1].strip(),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True));assert used<100
 assert KEY,'W&B credentials required before launch'
 if not Path('/tmp/pangram-wandb-vendor/wandb').exists():subprocess.run([sys.executable,'-m','pip','install','--target','/tmp/pangram-wandb-vendor','wandb==0.30.0'],check=True,stdout=subprocess.DEVNULL)
 root.mkdir(parents=True,exist_ok=False)
 for src in (recipe/'trainer').glob('*.py'):shutil.copy2(src,root/src.name)
 for name in ['worker.py','track.py','models.lock.json']:shutil.copy2(recipe/name,root/name)
 shutil.copytree(recipe/'configs',root/'configs')
 (root/'prepared-v2').symlink_to(recipe/'prepared-v2');(root/'vendor').symlink_to('/data/workspace/current-data-v1/vendor');(root/'assets').symlink_to('/data/workspace/current-data-v1/assets')
 (root/'recipe-identity.json').write_text(json.dumps({'recipe':str(recipe),'sha256':hashlib.sha256((recipe/'recipe.json').read_bytes()).hexdigest()}))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu[1].strip(),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',WANDB_API_KEY=KEY)
 with (root/'launch.log').open('a') as log:
  child=subprocess.Popen([sys.executable,'-u',str(root/'worker.py')],cwd=root,env=env,stdout=log,stderr=log,start_new_session=True,pass_fds=(lock.fileno(),))
 receipt={'state':'preflight_started','root':str(root),'pid':child.pid,'start_ticks':Path('/proc',str(child.pid),'stat').read_text().split()[21],'gpu':int(gpu[0]),'tracking':'automatic after training initialization; URL in wandb-tracking.json'}
 (root/'launcher-process.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
run('NAME='+repr(a.name)+'\nGPU='+repr(a.gpu)+'\nCHECK='+repr(a.check)+'\nKEY='+repr(key)+'\n'+code)
