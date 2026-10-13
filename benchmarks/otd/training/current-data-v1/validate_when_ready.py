"""Finish setup only; never start a training run."""
from pathlib import Path
import json,time,subprocess,sys,os
R=Path(__file__).resolve().parent
for _ in range(240):
 p=R/'prepare-status.json'
 try:s=json.loads(p.read_text()) if p.exists() else {}
 except (json.JSONDecodeError,OSError):time.sleep(2);continue
 if s.get('state')=='failed':raise RuntimeError('Preparation failed; inspect prepare-status.json')
 if s.get('state')=='complete':break
 time.sleep(10)
else:raise RuntimeError('Preparation not complete within 40 minutes')
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
free=[int(line.split(',')[0]) for line in gpus.splitlines() if int(line.split(',')[1])<100]
if not free:raise RuntimeError('No idle GPU for setup verification; no jobs interrupted')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(free[0]),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
result=subprocess.run([sys.executable,'-u',str(R/'preflight.py')],cwd=R,env=env)
(R/'setup-status.json').write_text(json.dumps({'state':'ready_not_started' if result.returncode==0 else 'verification_failed','returncode':result.returncode,'gpu':free[0],'time':time.time()}))
sys.exit(result.returncode)
