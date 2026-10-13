"""BF16 selection-only operating-point audit, then training-only human mining."""
from pathlib import Path
import sys,json,gzip,hashlib,time,gc,collections,shutil,subprocess
import numpy as np
R=Path('/data/workspace/paper-diversity-v1');OUT=R/'backlog-next';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'control'))
from inference import load_checkpoint,predict
from train import require_gpu
require_gpu()
SUITE=Path('/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1');sys.path.insert(0,str(SUITE));import compare_models as c
from transformers import AutoTokenizer
ref=AutoTokenizer.from_pretrained('/data/workspace/paper-v3-modernbert-20260930/run-01/best_model',local_files_only=True)
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def rows(p):return [json.loads(x) for x in gzip.decompress(p.read_bytes()).splitlines()]
def status(**x):save(OUT/'analysis-status.json',{'time':time.time(),**x})
summary={}
for name in ['control','boundary','raid','mage','raid-curriculum','mage-curriculum']:
 dest=OUT/(name+'-selection.json')
 if dest.exists():summary[name]=json.loads(dest.read_text());continue
 model,tok,contract=load_checkpoint(R/name/'run');values={}
 for split in ['calibration','selection']:
  rs=rows(R/'control/prepared'/(split+'-windows.jsonl.gz'));buckets={}
  for start in range(0,len(rs),32):
   chunk=rs[start:start+32];pr=predict(chunk,model,tok);offs=ref([r['text'] for r in chunk],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
   for r,off,(native,prob) in zip(chunk,offs,pr):
    p=c.project(r['text'],off,native,prob);ts,ty,ss,sy=c.units(r,off,p)
    for unit,scores,labels in [('tokens',ts,ty),('sentences',ss,sy)]:
     k=(r['kind'],unit);v=buckets.setdefault(k,([],[]));v[0].extend(map(float,scores));v[1].extend(map(int,labels))
   status(state='scoring',model=name,split=split,rows=min(start+32,len(rs)),total=len(rs))
  values[split]=buckets
 result={'model':name,'policy':'Calibration-only thresholds, selection-only comparison. No test fitting; fixed row chunks32 / model microbatch8. No checkpoint overwritten. Only retained stage2-best checkpoint assessed, not unavailable epoch checkpoints.','points':{}}
 for fpr in [.005,.01,.02]:
  point={}
  for unit in ['tokens','sentences']:
   threshold=max(c.choose_threshold(*v,fpr=fpr) for (kind,u),v in values['calibration'].items() if u==unit)
   point[unit]={'threshold':float(threshold),'selection':{kind:c.metrics(c.counts(*v,threshold)) for (kind,u),v in values['selection'].items() if u==unit}}
  result['points'][str(fpr)]=point
 save(dest,result);summary[name]=result
 del model;gc.collect();__import__('torch').cuda.empty_cache()
save(OUT/'selection-summary.json',summary)
# Bounded reweighting of already approved, human-only training windows, not unseen provenance claims.
source=R/'control';files=sorted((source/'prepared').glob('stage*-epoch*.jsonl.gz'));chunks={p.name:rows(p) for p in files}
humans={hashlib.sha256(r['text'].encode()).hexdigest():r for rs in chunks.values() for r in rs if r.get('kind')=='novel_human' and {q['label'] for q in r['regions']}=={0}}
assert humans
heldout=set()
for split in ['selection','calibration']:
 for r in rows(source/'prepared'/(split+'-windows.jsonl.gz')):heldout.add(r['paper_id'])
assert not heldout.intersection(r['paper_id'] for r in humans.values())
model,tok,contract=load_checkpoint(source/'run');scores={};items=sorted(humans.items())
for start in range(0,len(items),32):
 part=items[start:start+32]
 for (h,r),(off,prob) in zip(part,predict([r for h,r in part],model,tok)):
  valid=np.array([bool(r['text'][a:b].strip()) for a,b in off]);scores[h]={'score':float(np.quantile(prob[valid],.9)),'length':len(off),'paper_id':r['paper_id']}
 status(state='mining',rows=min(start+32,len(items)),total=len(items))
save(OUT/'training-human-scores.json',scores)
del model;gc.collect();__import__('torch').cuda.empty_cache()
name='hard-human-v1';dst=R/name
if dst.exists():raise RuntimeError('Existing hard-human directory; inspect before resuming')
# Same-length replacements preserve token exposure; limit repetitions and source concentration.
bins=collections.defaultdict(list)
for h in humans:bins[scores[h]['length']].append(h)
for v in bins.values():v.sort(key=lambda h:(-scores[h]['score'],h))
changes={};planned={}
for filename,rs in chunks.items():
 use=collections.Counter();papers=collections.Counter();out=[];replaced=0;eligible=0
 original_counts=collections.Counter(r['paper_id'] for r in rs if r.get('kind')=='novel_human')
 for i,r in enumerate(rs):
  if r.get('kind')=='novel_human':
   eligible+=1
   if eligible%2==0:
    h=hashlib.sha256(r['text'].encode()).hexdigest(); candidates=bins[scores[h]['length']]
    chosen=next((x for x in candidates if scores[x]['score']>scores[h]['score'] and use[x]<2 and papers[scores[x]['paper_id']]<max(2,original_counts[scores[x]['paper_id']])),None)
    if chosen:
     r=humans[chosen];use[chosen]+=1;papers[r['paper_id']]+=1;replaced+=1
  out.append(r)
 planned[filename]=out;changes[filename]={'eligible_human':eligible,'replacements':replaced,'rows':len(out)}
assert sum(x['replacements'] for x in changes.values())>0
# Reuses frozen train-only pool. Rank on model scores does not certify human provenance beyond original corpus audit.
dst.mkdir();(dst/'prepared').mkdir();(dst/'configs').mkdir()
for p in source.glob('*.py'):shutil.copy2(p,dst/p.name)
shutil.copy2(source/'models.lock.json',dst/'models.lock.json');shutil.copy2(source/'configs/encoder.json',dst/'configs/encoder.json');(dst/'vendor').symlink_to(source/'vendor')
m=json.loads((source/'prepared/manifest.json').read_text())
for p in (source/'prepared').glob('*.gz'):
 if p.name not in planned:(dst/'prepared'/p.name).symlink_to(p);continue
 rs=planned[p.name];blob=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs).encode();(dst/'prepared'/p.name).write_bytes(gzip.compress(blob,mtime=0));key=p.name.removesuffix('.jsonl.gz');m['files'][key]={'rows':len(rs),'sha256':hashlib.sha256(blob).hexdigest(),'papers':len({r['paper_id'] for r in rs})}
m['experiment']={'arm':name,'mining_model':'control','pool':'existing approved novel-human training windows only','split_paper_disjoint':True,'rank':'90th percentile nonwhitespace native token AI score','same_native_token_lengths':True,'replacement_limit':'at most half novel-human draws; higher-scoring same-length windows only','replacement_repeat_cap':2,'replacement_paper_cap':'max(2, original novel-human paper draws), applies to added replacement draws','seed':42,'changes':changes,'caveat':'reweights already-seen human negatives; not an unseen-human mining experiment; original labels retained'}
save(dst/'prepared/manifest.json',m);save(dst/'spec.json',m['experiment']);save(OUT/'mining-prepared.json',m['experiment'])
status(state='training_hard_human',arm=name)
rc=subprocess.call([sys.executable,'-u',str(R/'worker.py'),name],cwd=R)
if rc:raise RuntimeError('hard-human worker failed')
status(state='complete',arm=name)
