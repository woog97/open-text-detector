"""Single durable dispatcher for prepared backlog jobs; no speculative job creation."""
from pathlib import Path
import os,json,time,subprocess,fcntl,traceback
ROOT=Path('/data/workspace/paper-diversity-v1/auto-dispatch');ROOT.mkdir(exist_ok=True)
lock=(ROOT/'dispatcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def alive(d):
 p=Path('/proc')/str(d.get('pid'))
 return p.exists() and (p/'stat').read_text().split()[2] not in ['Z','X'] and (p/'stat').read_text().split()[21]==d.get('start_ticks')
while True:
 try:
  jobs=[]
  for p in (ROOT/'jobs').glob('*.json'):
   d=json.loads(p.read_text());state=ROOT/'states'/p.name
   rec=json.loads(state.read_text()) if state.exists() else {}
   if rec.get('state')=='running' and not alive(rec):
    done=Path(d['completion_file']);ok=done.exists() and json.loads(done.read_text()).get('state')=='complete'
    rec.update(state='complete' if ok else 'needs_attention',checked=time.time());save(state,rec)
   if not rec:jobs.append((d,p,state))
  query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  busy=set()
  for p in (ROOT/'states').glob('*.json'):
   d=json.loads(p.read_text())
   if d.get('state')=='running' and alive(d):busy.add(d['uuid'])
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
  busy.update(x.strip() for x in active.splitlines() if x.strip().startswith('GPU-'))
  free=[(int(i),u.strip()) for i,u,m,v in (line.split(',') for line in query.splitlines()) if int(m)<100 and int(v)<5 and u.strip() not in busy]
  for d,p,state in sorted(jobs,key=lambda x:(x[0]['rank'],x[0]['id'])):
   if not free:break
   if d.get('enabled') is not True or not Path(d['ready_file']).exists():continue
   if any(not ((ROOT/'states'/f'{dep}.json').exists() and json.loads((ROOT/'states'/f'{dep}.json').read_text()).get('state')=='complete') for dep in d.get('dependencies',[])):continue
   done=Path(d['completion_file'])
   if done.exists():
    # A pre-existing record belongs to another owner or previous run; never duplicate.
    save(state,{'state':'complete' if json.loads(done.read_text()).get('state')=='complete' else 'needs_attention','reason':'pre-existing job state; reconcile ownership'});continue
   gpu,uuid=free.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,HF_HOME='/data/workspace/hf-home',HF_HUB_CACHE='/data/workspace/model-cache',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
   log=(ROOT/'logs'/f"{d['id']}.log").open('a');child=subprocess.Popen(d['command'],cwd=d['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save(state,{'state':'running','pid':child.pid,'start_ticks':(Path('/proc')/str(child.pid)/'stat').read_text().split()[21],'gpu':gpu,'uuid':uuid,'started':time.time()})
  save(ROOT/'status.json',{'state':'watching','updated':time.time(),'registered_jobs':len(list((ROOT/'jobs').glob('*.json'))),'unstarted_jobs':len(jobs),'free_gpus':[i for i,u in free]})
 except Exception:traceback.print_exc()
 time.sleep(15)
