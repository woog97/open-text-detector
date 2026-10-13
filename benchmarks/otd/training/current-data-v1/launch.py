"""Explicit single-run launcher. Preparation and validation never invoke this."""
from pathlib import Path
import os,json,subprocess,sys,time
R=Path(__file__).resolve().parent
assert str(R).startswith('/data/workspace/'),'Launch only on training Space'
assert (R/'READY.json').exists(),'Setup validation has not passed'
assert not (R/'training-process.json').exists(),'Existing training identity; inspect before retry'
assert not (R/'run').exists(),'Preserve existing output'
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
free=[line.split(',') for line in gpus.splitlines() if int(line.split(',')[2])<100]
assert free,'No idle GPU; no existing jobs changed'
gpu=free[0];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu[1].strip(),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
with (R/'training.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-u',str(R/'train.py'),'--flow','encoder','--output',str(R/'run')],cwd=R,env=env,stdout=log,stderr=log,start_new_session=True)
info={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'gpu_index':int(gpu[0]),'gpu_uuid':gpu[1].strip(),'started_at':time.time()}
(R/'training-process.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
