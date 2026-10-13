"""BF16 selection-only operating-point audit, then training-only human mining."""
from pathlib import Path
import sys,json,gzip,hashlib,time,gc,collections,shutil,subprocess
import numpy as np
R=Path('/data/workspace/paper-diversity-v1');OUT=R/'auto-dispatch'/'audits'/sys.argv[1];OUT.mkdir(parents=True,exist_ok=True)
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
for name in [sys.argv[1]]:
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

status(state="complete",model=sys.argv[1])
