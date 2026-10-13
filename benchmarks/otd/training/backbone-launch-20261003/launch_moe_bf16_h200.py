"""Launch only the user-selected BF16 MoE replacement; secrets stay in memory."""
from pathlib import Path
import netrc,subprocess
key=netrc.netrc().authenticators('api.wandb.ai')[2]
code='KEY='+repr(key)+'\n'+'''
from pathlib import Path
import subprocess,json,os,time
r=Path('/workspace/woog/pangram/backbones-20261003');name='qwen36-35b-a3b';out=r/'runs'/name;out.mkdir(parents=True,exist_ok=True)
assert not (out/'worker-process.json').exists(),'Inspect existing worker before restarting'
assert json.loads((r/'downloads'/name/'status.json').read_text())['state']=='complete'
assert int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<100
spec=next(m for m in json.loads((r/'models.json').read_text()) if m['name']==name)
assert spec['enabled'] and spec['lora_rank']==128 and spec['expert_rank']==16
assert not (r/'assets/qwen36-27b').exists()
env=os.environ.copy();env.update(WANDB_API_KEY=KEY,CUDA_VISIBLE_DEVICES='1',PYTHONPATH=str(r/'moe-vendor'),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
with (out/'worker.log').open('a') as f:p=subprocess.Popen([str(r/'venv/bin/python'),'-u',str(r/'worker_moe_bf16.py'),name],cwd=r,env=env,stdout=f,stderr=f,start_new_session=True)
d={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'time':time.time(),'gpu':1,'trainer':'train_moe_bf16.py','lora_rank':128,'expert_rank':16,'trainable_dtype':'bfloat16','checkpoint_dtype':'bfloat16'};(out/'worker-process.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
'''
subprocess.run(['ssh','-o','BatchMode=yes','otd-h200','python3 -'],input=code,text=True,check=True)
