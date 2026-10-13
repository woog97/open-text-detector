"""Train/evaluate the two smaller LoRA backbones, then stop for presentation before Qwen3.5."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/data/workspace/paper-backbone-comparison-v1');Q=ROOT/'priority-queue';Q.mkdir(exist_ok=True)
SUITE='/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1';REFERENCE='/data/workspace/paper-v3-modernbert-20260930/run-01/best_model'
def status(state,**kw):
 record={'state':state,'time':time.time(),**kw};tmp=Q/'status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(Q/'status.json');print(json.dumps(record),flush=True)
def job(root,model,phase,args):
 with (Q/(root.name+'-'+model+'-'+phase+'.log')).open('w') as log:
  p=subprocess.Popen([sys.executable,'-u',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  status('running',root=str(root),model=model,phase=phase,child_pid=p.pid);code=p.wait()
 if code:raise RuntimeError(f'{root.name} {model} {phase} failed with exit {code}')
 with (Q/'completed.jsonl').open('a') as f:f.write(json.dumps({'root':str(root),'model':model,'phase':phase,'time':time.time()})+'\n')
try:
 job(ROOT,'small-models','preflight',['preflight.py','--models','encoder,causal'])
 for model in ['encoder','causal']:
  job(ROOT,model,'training',['train.py','--flow',model,'--output',str(ROOT/'runs'/model)])
 for root in [BASE,ROOT]:
  for model in ['encoder','causal']:
   run=root/'runs'/model
   if not (run/'thresholds.json').exists():job(root,model,'calibration',['calibrate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE])
   for profile in ['workflow','comparison']:
    dest=root/'results'/(model+'-'+profile)
    if not (dest/'results.json').exists():job(root,model,'evaluation-'+profile,['evaluate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE,'--profile',profile,'--output',str(dest)])
 status('comparison_ready',models=['encoder','causal'],qwen35='held_until_results_presented',result_roots=[str(BASE/'results'),str(ROOT/'results')])
except Exception as e:
 status('failed',error=str(e));traceback.print_exc();sys.exit(1)
