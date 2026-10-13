from pathlib import Path
import os,sys,json,gzip,hashlib,shutil,subprocess,time
R=Path('/data/workspace/paper-diversity-v1')
def save(p,x):p.write_text(json.dumps(x,indent=2))
def read(p):return [json.loads(l) for l in gzip.decompress(p.read_bytes()).splitlines()]
def write(p,rows):
 blob=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows).encode();p.write_bytes(gzip.compress(blob,mtime=0));return {'rows':len(rows),'sha256':hashlib.sha256(blob).hexdigest(),'papers':len({x['paper_id'] for x in rows})}
trials=[('raid-curriculum','raid',42),('mage-curriculum','mage',42),('control-seed73','control',73)]
for name,source,seed in trials:assert not (R/name).exists(),name
for name,source,seed in trials:
 src=R/source;dst=R/name;dst.mkdir();(dst/'prepared').mkdir();(dst/'configs').mkdir()
 for p in src.glob('*.py'):shutil.copy2(p,dst/p.name)
 shutil.copy2(src/'models.lock.json',dst/'models.lock.json');(dst/'vendor').symlink_to(src/'vendor')
 cfg=json.loads((src/'configs/encoder.json').read_text());cfg['seed']=seed;save(dst/'configs/encoder.json',cfg)
 m=json.loads((src/'prepared/manifest.json').read_text())
 for p in (src/'prepared').glob('*.gz'):
  if not p.name.startswith('stage') or seed!=42:(dst/'prepared'/p.name).symlink_to(p)
 if seed==42:
  check={}
  for stage,epochs in [(1,1),(2,3)]:
   keys=[f'stage{stage}-epoch{e}' for e in range(epochs)];chunks=[read(src/'prepared'/(k+'.jsonl.gz')) for k in keys];rows=[x for chunk in chunks for x in chunk]
   # Stable partition preserves within-stratum order and exact example multiset.
   reordered=[x for x in rows if x.get('dataset')==source]+[x for x in rows if x.get('dataset')!=source]
   counter=lambda rs:sorted(hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest() for x in rs)
   assert counter(rows)==counter(reordered)
   offset=0
   for key,chunk in zip(keys,chunks):
    part=reordered[offset:offset+len(chunk)];offset+=len(chunk);m['files'][key]=write(dst/'prepared'/(key+'.jsonl.gz'),part)
   check[str(stage)]={'rows':len(rows),'external_draws':sum(x.get('dataset')==source for x in rows),'exact_multiset_preserved':True,'tokens_preserved_by_identical_texts':True}
  m['experiment']={'arm':name,'parent':source,'order':'external first then paper within each objective stage; stage2 partition spans all3epochs','seed':seed,'checks':check,'optimizer_steps_and_LR_schedule':'unchanged; exposure-to-LR association intentionally differs as part of curriculum','selection':'same minimum validation loss protocol; report chosen epoch and final-epoch selection loss'}
 else:m['experiment']={'arm':name,'parent':source,'seed':seed,'data_order':'identical to original control; changes training RNG only','purpose':'measure seed sensitivity independently of candidate selection'}
 save(dst/'prepared/manifest.json',m);save(dst/'spec.json',m['experiment'])
queue_src=r'''
from pathlib import Path
import subprocess,sys,json,time,traceback
r=Path(__file__).resolve().parent
names=sys.argv[1:];key='followup-'+names[0]
try:
 for name in names:
  if (r/(name+'-status.json')).exists() and json.loads((r/(name+'-status.json')).read_text()).get('state')=='complete':continue
  child=subprocess.Popen([sys.executable,'-u',str(r/'worker.py'),name],cwd=r)
  (r/(key+'-status.json')).write_text(json.dumps({'time':time.time(),'state':'running','arm':name,'child_pid':child.pid,'pending':names[names.index(name)+1:]}))
  if child.wait():raise RuntimeError(name+' failed; inspect trial logs')
 (r/(key+'-status.json')).write_text(json.dumps({'time':time.time(),'state':'complete','trials':names}))
except Exception as e:
 (r/(key+'-status.json')).write_text(json.dumps({'time':time.time(),'state':'failed','error':str(e)}));traceback.print_exc();sys.exit(1)
'''
(R/'followup_queue.py').write_text(queue_src)
gpus={int(a.strip()):(b.strip(),int(c.strip())) for a,b,c in [x.split(',') for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]}
for gpu,names in [(1,['raid-curriculum','control-seed73']),(3,['mage-curriculum'])]:
 assert gpus[gpu][1]<100,(gpu,gpus[gpu]);key='followup-'+names[0];assert not (R/(key+'-process.json')).exists()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpus[gpu][0],HF_HUB_CACHE='/data/workspace/model-cache',HF_HOME='/data/workspace/hf-home',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
 p=subprocess.Popen([sys.executable,'-u',str(R/'followup_queue.py'),*names],cwd=R,env=env,stdout=(R/(key+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 save(R/(key+'-process.json'),{'pid':p.pid,'gpu':gpu,'uuid':gpus[gpu][0],'trials':names,'time':time.time()});print('queued',gpu,names,p.pid)
save(R/'followup-plan.json',{'trials':trials,'reason':'Test whether paper-specialization recovers paper metrics after diversity; estimate control seed variation','new_generation':False,'no_change_to_active_jobs':True})
