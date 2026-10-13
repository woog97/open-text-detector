"""Independent single-GPU, matched-exposure experiments. No test-based selection."""
import os,sys,json,time,hashlib,gzip,random,collections,subprocess,shutil,traceback
from pathlib import Path
BASE=Path('/data/workspace/paper-lora-comparison-v1')
ROOT=Path(__file__).resolve().parent
SUITE='/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1'
REFERENCE='/data/workspace/paper-v3-modernbert-20260930/run-01/best_model'
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**kw):save(ROOT/'status.json',{'time':time.time(),**kw});print(json.dumps(kw),flush=True)
def read(p):return [json.loads(l) for l in gzip.decompress(p.read_bytes()).splitlines()]
def command(root,label,args):
 with (root/(label+'.log')).open('a') as log:
  child=subprocess.Popen([sys.executable,'-u',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  status(state='running',trial=root.name,phase=label,child_pid=child.pid)
  if child.wait():raise RuntimeError(root.name+' '+label+' failed; inspect log')
def prepare(spec):
 root=ROOT/spec['name']
 if root.exists():
  assert (root/'prepared/manifest.json').exists(),'Incomplete preparation; inspect before resuming'
  return root
 root.mkdir();(root/'configs').mkdir();(root/'prepared').mkdir()
 for p in BASE.glob('*.py'):shutil.copy2(p,root/p.name)
 for name in ['models.lock.json']:shutil.copy2(BASE/name,root/name)
 (root/'vendor').symlink_to(BASE/'vendor',target_is_directory=True)
 cfg=json.loads((BASE/'configs'/(spec['flow']+'.json')).read_text())
 cfg['effective_batch']=spec['effective_batch'];cfg['learning_rate']=spec['lr'];cfg['head_learning_rate']=spec['lr']/10
 # Identical microbatch and checkpointing within each sweep isolates effective-batch/mix effects.
 (root/'configs'/(spec['flow']+'.json')).write_text(json.dumps(cfg,indent=2))
 manifest=json.loads((BASE/'prepared/manifest.json').read_text())
 for p in (BASE/'prepared').glob('*.jsonl.gz'):(root/'prepared'/p.name).symlink_to(p)
 if spec.get('human_share',.25)>.25:
  sys.path.insert(0,str(root));from transformers import AutoTokenizer
  from data import shared_crop
  toks=[AutoTokenizer.from_pretrained(v['repo'],revision=v['revision'],local_files_only=True) for v in manifest['models'].values()]
  pool=read(BASE/'prepared/train.jsonl.gz');groups=collections.defaultdict(list)
  for r in pool:
   if r['kind']=='novel_human':groups[r['paper_id']].append(r)
  assert groups
  for stage,epochs in [(1,1),(2,3)]:
   for epoch in range(epochs):
    key=f'stage{stage}-epoch{epoch}';rows=read(BASE/'prepared'/(key+'.jsonl.gz'));rng=random.Random(87421+stage*100+epoch)
    replaced=0
    for i,row in enumerate(rows):
     if i%4 in ([1] if spec['human_share']==.5 else [1,2]):
      paper=rng.choice(sorted(groups));source=rng.choice(sorted(groups[paper],key=lambda r:r['id']))
      row=shared_crop(source,rng,toks,['target','context','random'][i%3]);row['draw_id']=f's{stage}-e{epoch}-{i}';rows[i]=row;replaced+=1
    blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
    # Replace only this trial's symlink atomically; the frozen source remains untouched.
    dest=root/'prepared'/(key+'.jsonl.gz');tmp=dest.with_suffix('.new');tmp.write_bytes(gzip.compress(blob,mtime=0));tmp.replace(dest)
    manifest['files'][key]={'sha256':hashlib.sha256(blob).hexdigest(),'rows':len(rows),'papers':len({r['paper_id'] for r in rows}),'replaced_with_novel_human':replaced}
 manifest['experiment']={'spec':spec,'baseline_manifest_sha256':digest(BASE/'prepared/manifest.json'),'selection':'Original unchanged selection windows; lowest stage2 selection loss; frozen tests only for control and winner','general_domain_arm':'Deferred: available additional datasets are held-out evaluation sets, not audited general-domain training data'}
 save(root/'prepared/manifest.json',manifest);save(root/'spec.json',spec)
 return root

def main():
 plan=json.loads((ROOT/'plan.json').read_text());scores=[]
 for spec in plan['trials']:
  # Other workers own delegated trials; wait for their completion without duplication.
  external=plan.get('delegated',{}).get(spec['name'])
  adopted=plan.get('adopted',{}).get(spec['name'])
  while external or adopted:
   run=ROOT/spec['name']/'run'
   if (run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('state')=='trained_calibration_pending':break
   pid=(json.loads((ROOT/external).read_text())['pid'] if external else adopted['pid'])
   proc=Path('/proc')/str(pid)
   if not proc.exists() or (proc/'stat').read_text().split()[2] in ['Z','X']:raise RuntimeError('Assigned worker stopped before completion: '+spec['name'])
   args=(proc/'cmdline').read_bytes().decode(errors='replace')
   expected='delegated_worker.py' if external else adopted['expected']
   if expected not in args:raise RuntimeError('Assigned process identity mismatch')
   status(state='running',trial=spec['name'],phase='delegated-training' if external else 'training',child_pid=pid)
   time.sleep(15)
  root=prepare(spec);run=root/'run'
  if not (root/'READY.json').exists():command(root,'preflight',['preflight.py','--models',spec['flow']])
  complete=(run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('state')=='trained_calibration_pending'
  if not complete:command(root,'training',['train.py','--flow',spec['flow'],'--output',str(run)])
  hist=json.loads((run/'history.json').read_text());score=min(r['selection_loss'] for r in hist if r['stage']==2)
  scores.append({'trial':spec['name'],'selection_loss':score,'training':json.loads((run/'status.json').read_text()),'epochs':hist})
  save(ROOT/'validation-results.json',scores)
 winner=min(scores,key=lambda s:s['selection_loss'])['trial'];control=plan['trials'][0]['name']
 save(ROOT/'selection.json',{'winner':winner,'control':control,'criterion':'Minimum stage2 selection loss; fixed before frozen evaluation','one_seed_only':True,'scores':scores})
 for name in dict.fromkeys([control,winner]):
  root=ROOT/name;run=root/'run'
  if not (run/'thresholds.json').exists():command(root,'calibration',['calibrate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE])
  for profile in ['workflow','comparison']:
   out=root/'results'/profile
   if not (out/'results.json').exists():command(root,'evaluation-'+profile,['evaluate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE,'--profile',profile,'--output',str(out)])
 status(state='complete',winner=winner,control=control)
if __name__=='__main__':
 try:main()
 except Exception as e:status(state='failed',error=str(e));traceback.print_exc();sys.exit(1)
