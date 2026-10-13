"""Retry Qwen only after its guarded Hopper compiler preflight failure."""
import netrc,subprocess
key=netrc.netrc().authenticators('api.wandb.ai')[2]
code=r'''from pathlib import Path
import os,subprocess,json,time
r=Path('/workspace/woog/pangram/backbones-20261003');root=r/'runs/qwen36-27b'
for f in ['worker-process.json','training-process.json']:
 d=json.loads((root/f).read_text());p=Path('/proc',str(d['pid']),'stat')
 if p.exists():
  s=p.read_text().split();assert s[2]=='Z' or s[21]!=str(d['start_ticks'])
assert not (root/'wandb-tracking.json').exists()
assert 'Please upgrade Triton' in (root/'training.log').read_text()
assert int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<100
archive=root/'attempts'/('hopper-preflight-'+str(int(time.time())));archive.mkdir(parents=True)
for f in ['run','worker-process.json','training-process.json','worker-status.json','worker.log','training.log']:
 p=root/f
 if p.exists():p.rename(archive/f)
e=os.environ.copy();e.update(WANDB_API_KEY=KEY,CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
with (root/'worker.log').open('a') as f:p=subprocess.Popen([str(r/'venv/bin/python'),'-u',str(r/'worker.py'),'qwen36-27b'],env=e,cwd=r,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'gpu':1,'time':time.time()};(root/'worker-process.json').write_text(json.dumps(d));print(json.dumps(d))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','otd-h200','python3 -'],input='KEY='+repr(key)+'\n'+code,text=True)
raise SystemExit(p.returncode)
