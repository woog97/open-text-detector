"""Automatic Qwen3.5 continuation after matched small-model evaluations; no approval gate."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;BASE=Path('/data/workspace/paper-backbone-comparison-v1');PRIORITY=ROOT/'priority-queue'
SUITE='/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1';REFERENCE='/data/workspace/paper-v3-modernbert-20260930/run-01/best_model'
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def status(state,**kw):
 rec={'state':state,'time':time.time(),**kw};write(HERE/'status.json',rec);print(json.dumps(rec),flush=True)
def job(root,phase,args):
 with (HERE/(root.name+'-'+phase+'.log')).open('w') as log:
  p=subprocess.Popen([sys.executable,'-u',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  status('running',model='qwen35',root=str(root),phase=phase,child_pid=p.pid);code=p.wait()
 if code:raise RuntimeError(phase+' failed: '+str(code))
def process_live(pid):
 path=Path(f'/proc/{pid}/status')
 if not path.exists():return False
 return not any(x.startswith('State:') and ('Z (zombie)' in x or 'X (dead)' in x) for x in path.read_text().splitlines())
def compare():
 records=[];violations=[];primary_seen=set()
 groups={'document':'document_native_label_metrics','tokens':'span_or_historical_human_token_metrics','sentences':'span_or_historical_human_sentence_metrics'}
 for model in ['encoder','causal']:
  for profile in ['workflow','comparison']:
   a=json.loads((BASE/'results'/f'{model}-{profile}'/'results.json').read_text());b=json.loads((ROOT/'results'/f'{model}-{profile}'/'results.json').read_text())
   assert a['profile_sha256']==b['profile_sha256']
   assert a['training']['data_manifest_sha256']==b['training']['data_manifest_sha256']
   assert set(a['datasets'])==set(b['datasets'])
   for ds,entry in a['datasets'].items():
    fa=entry['overall'];lb=b['datasets'][ds]['overall'];assert fa['rows']==lb['rows']
    primary=ds in ['paper_v3_target','paper_workflow_reconstruction']
    if primary:primary_seen.add((model,ds))
    for unit,key in groups.items():
     am=fa.get(key) or {};bm=lb.get(key) or {}
     for metric in ['f1','recall','human_fpr']:
      av=am.get(metric);bv=bm.get(metric)
      if av is None and bv is None:continue
      records.append({'model':model,'profile':profile,'dataset':ds,'unit':unit,'metric':metric,'full':av,'lora':bv,'delta_pp':None if av is None or bv is None else 100*(bv-av)})
      if av is None or bv is None:
       if primary and unit in ['tokens','sentences'] and metric in ['f1','recall']:violations.append({'reason':'missing_primary_metric','model':model,'dataset':ds,'unit':unit,'metric':metric})
       continue
      if metric=='human_fpr':limit=.0025;degradation=bv-av
      else:limit=.01 if primary and unit in ['tokens','sentences'] else .02;degradation=av-bv
      relevant=(primary and unit in ['tokens','sentences']) or fa['unique_texts']>=100
      if relevant and degradation>limit+1e-12:violations.append({'model':model,'dataset':ds,'unit':unit,'metric':metric,'degradation_pp':100*degradation,'limit_pp':100*limit})
 assert len(primary_seen)==4,'Primary evaluation groups missing'
 decision={'small_gap':not violations,'rule':'No >1 pp paper token/sentence F1 or recall loss, no >0.25 pp human FPR rise, no >2 pp F1/recall loss in groups with >=100 unique texts; both backbones must pass. One seed, descriptive decision rather than proof of equivalence.','violations':violations,'qwen35_path':'lora_only' if not violations else 'full_then_lora','records':records}
 write(HERE/'comparison.json',decision)
 lines=['# Full tuning versus LoRA','',decision['rule'],'','Qwen3.5 path: **'+decision['qwen35_path']+'**. Metrics below are percentages; bold marks the better score within each pair.','']
 for model in ['encoder','causal']:
  lines.extend(['## '+model,'','| Suite / dataset | Unit / metric | Full | LoRA | LoRA − full (pp) |','|---|---|---:|---:|---:|'])
  for r in records:
   if r['model']!=model:continue
   av,bv=r['full'],r['lora'];aa='-' if av is None else f'{100*av:.2f}';bb='-' if bv is None else f'{100*bv:.2f}'
   if av is not None and bv is not None:
    if av==bv:aa='**'+aa+'**';bb='**'+bb+'**'
    elif (av<bv if r['metric']=='human_fpr' else av>bv):aa='**'+aa+'**'
    else:bb='**'+bb+'**'
   dd='-' if r['delta_pp'] is None else f'{r["delta_pp"]:+.2f}'
   lines.append(f'| {r["profile"]} / {r["dataset"]} | {r["unit"]} / {r["metric"]} | {aa} | {bb} | {dd} |')
 (HERE/'comparison.md').write_text('\n'.join(lines)+'\n')
 return decision

def evaluate(root):
 run=root/'runs/qwen35'
 if not (run/'thresholds.json').exists():job(root,'calibration',['calibrate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE])
 for profile in ['workflow','comparison']:
  dest=root/'results'/('qwen35-'+profile)
  if not (dest/'results.json').exists():job(root,'evaluation-'+profile,['evaluate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE,'--profile',profile,'--output',str(dest)])
try:
 status('waiting_for_smaller_model_evaluations')
 while True:
  state=json.loads((PRIORITY/'status.json').read_text())
  if state['state']=='comparison_ready':break
  if state['state']=='failed':raise RuntimeError('Priority comparison failed: '+state.get('error',''))
  pid=json.loads((PRIORITY/'process.json').read_text())['pid']
  if not process_live(pid):raise RuntimeError('Priority queue stopped before comparison completed')
  time.sleep(30)
 decision=compare();status('comparison_saved',qwen35_path=decision['qwen35_path'],report=str(HERE/'comparison.md'))
 hold=json.loads((PRIORITY/'qwen35-hold.json').read_text());pid=hold['pid']
 full_status=BASE/'runs/qwen35/status.json'
 full_finished=full_status.exists() and json.loads(full_status.read_text()).get('state')=='trained_calibration_pending'
 if process_live(pid) and not full_finished:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
  assert 'train.py' in cmd and 'qwen35' in cmd,'Held process identity changed'
 if decision['small_gap']:
  if process_live(pid):
   os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
   for _ in range(60):
    if not process_live(pid):break
    time.sleep(1)
   assert not process_live(pid),'Held process did not terminate'
  write(PRIORITY/'qwen35-hold.json',{**hold,'state':'terminated_for_lora_only','decision':str(HERE/'comparison.json')})
 else:
  runstatus=BASE/'runs/qwen35/status.json'
  if json.loads(runstatus.read_text())['state']!='trained_calibration_pending':
   assert process_live(pid),'Paused full-tuning state was lost; inspect before restarting'
   os.kill(pid,signal.SIGCONT);write(PRIORITY/'qwen35-hold.json',{**hold,'state':'resumed_automatically'})
   status('running',model='qwen35',phase='full_training_resumed',child_pid=pid)
   while process_live(pid):time.sleep(30)
   assert json.loads(runstatus.read_text())['state']=='trained_calibration_pending','Resumed full training failed'
  evaluate(BASE)
 # Automated optimization gate, fulfilled by the supervising agent, never user approval.
 status('profiling_optimization_pending',model='qwen35',plan=str(HERE/'profiling-plan.json'))
 ready=HERE/'optimization-ready.json'
 while not ready.exists():time.sleep(30)
 optimization=json.loads(ready.read_text())
 assert optimization['bf16_verified'] and optimization['gradient_checks_passed']
 assert optimization['training_recipe_preserved'] and optimization['profiling_completed']
 import hashlib
 for name,digest in optimization['verified_files_sha256'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Optimization changed after validation: '+name
 # All smaller jobs are complete before the preflight regenerates READY.
 p=ROOT/'preflight-results.json'
 if p.exists():(HERE/'small-model-preflight-results.json').write_bytes(p.read_bytes())
 job(ROOT,'preflight',['preflight.py','--models','qwen35'])
 job(ROOT,'lora-training',['train.py','--flow','qwen35','--output',str(ROOT/'runs/qwen35')])
 evaluate(ROOT)
 status('complete',qwen35_path=decision['qwen35_path'],comparison_report=str(HERE/'comparison.md'))
except Exception as e:
 status('failed',error=str(e));traceback.print_exc();sys.exit(1)
