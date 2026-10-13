"""Durable dependent queue: never overlap the baseline GPU workload."""
import os,json,subprocess,sys,time,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;Q=ROOT/'training-queue';Q.mkdir(exist_ok=True)
BASE=Path('/data/workspace/paper-backbone-comparison-v1')
SUITE=Path('/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1');REFERENCE=Path('/data/workspace/paper-v3-modernbert-20260930/run-01/best_model')
def status(state,**kw):
 record={'state':state,'time':time.time(),**kw};tmp=Q/'status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(Q/'status.json');print(json.dumps(record),flush=True)
def run_job(flow,phase,args):
 with (Q/(flow+'-'+phase+'.log')).open('w') as f:
  child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  status('running',model=flow,phase=phase,child_pid=child.pid)
  code=child.wait()
 if code:raise RuntimeError(flow+' '+phase+' failed, exit '+str(code))
 with (Q/'completed.jsonl').open('a') as f:f.write(json.dumps({'model':flow,'phase':phase,'completed_at':time.time()})+'\n')
try:
 status('waiting_for_baseline',dependency=str(BASE/'training-queue/status.json'))
 while True:
  baseline=json.loads((BASE/'training-queue/status.json').read_text())
  if baseline['state']=='complete':break
  if baseline['state']=='failed':raise RuntimeError('Baseline failed; LoRA will not start automatically')
  pid=json.loads((BASE/'training-queue/process.json').read_text())['pid']
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Baseline queue stopped without completion')
  time.sleep(30)
 completed=[json.loads(x) for x in (BASE/'training-queue/completed.jsonl').read_text().splitlines()]
 expected={(m,p) for m in ['encoder','causal','qwen35'] for p in ['training','calibration','evaluation-workflow','evaluation-comparison']}
 assert expected<={(x['model'],x['phase']) for x in completed}
 plan=json.loads((ROOT/'PLAN.json').read_text())
 for name,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 assert hashlib.sha256((ROOT/'prepared/manifest.json').read_bytes()).hexdigest()==plan['data_manifest_sha256']
 for flow in ['encoder','causal','qwen35']:
  assert not (ROOT/'runs'/flow).exists(),'Refusing to overwrite previous training'
 run_job('all','preflight',['preflight.py'])
 for flow in ['encoder','causal','qwen35']:
  run_job(flow,'training',['train.py','--flow',flow,'--output',str(ROOT/'runs'/flow)])
 for flow in ['encoder','causal','qwen35']:
  run=ROOT/'runs'/flow
  run_job(flow,'calibration',['calibrate.py','--run',str(run),'--suite',str(SUITE),'--reference',str(REFERENCE)])
  for profile in ['workflow','comparison']:
   run_job(flow,'evaluation-'+profile,['evaluate.py','--run',str(run),'--suite',str(SUITE),'--reference',str(REFERENCE),'--profile',profile,'--output',str(ROOT/'results'/(flow+'-'+profile))])
 status('complete',models=['encoder','causal','qwen35'])
except Exception as e:
 status('failed',error=str(e));traceback.print_exc();sys.exit(1)
