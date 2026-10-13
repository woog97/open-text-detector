"""Space-only BF16 evaluation; supports existing LoRA run adapters and HF binary token classifiers."""
import argparse,hashlib,json,gzip,os,sys,time,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME','/data/workspace/hf-home');os.environ.setdefault('HF_HUB_CACHE','/data/workspace/model-cache');os.environ.setdefault('HF_HUB_OFFLINE','1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def validate():
 m=json.loads((HERE/'manifest.json').read_text())
 for name,h in m['files'].items():
  if sha(HERE/name)!=h:raise ValueError('Package changed: '+name)
 for name,s in m['profiles'].items():
  blob=gzip.decompress((HERE/(name+'.jsonl.gz')).read_bytes());rs=[json.loads(l) for l in blob.splitlines()]
  assert len(rs)==s['rows'] and hashlib.sha256(blob).hexdigest()==s['sha256']
  assert all(hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256'] for r in rs)
 return m

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validate-only',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--adapter-dir',type=Path,help='Existing training code with inference.load_checkpoint and predict; omit for HF AutoModelForTokenClassification');p.add_argument('--reference',type=Path,help='Existing common reference tokenizer directory, required when scoring');p.add_argument('--output',type=Path);p.add_argument('--profiles',default='workflow,comparison,assistance,manuscripts');p.add_argument('--thresholds',type=Path);p.add_argument('--threshold',type=float,help='Explicit uncalibrated threshold for all units');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--rows-per-chunk',type=int,default=16);p.add_argument('--limit',type=int,default=0,help='Smoke test rows per profile; results explicitly marked partial');a=p.parse_args()
 m=validate()
 if a.validate_only:print(json.dumps(m['profiles'],indent=2));return
 if not str(HERE).startswith('/data/workspace/'):raise RuntimeError('Run inference on the training Space under /data/workspace')
 if not a.run or not a.output or not a.reference:p.error('--run, --output and --reference required')
 if a.batch_size<1 or a.rows_per_chunk<1 or a.limit<0:p.error('Invalid batch/chunk/limit')
 names=a.profiles.split(',')
 if any(n not in m['profiles'] for n in names):p.error('Unknown profile')
 if a.threshold is not None and a.thresholds:p.error('Choose --threshold OR --thresholds')
 threshold_path=a.thresholds or a.run/'thresholds.json'
 if a.threshold is not None:th={k:a.threshold for k in ['tokens','sentences','document']};origin='explicit uncalibrated'
 else:
  d=json.loads(threshold_path.read_text());th=d.get('thresholds',d);origin=str(threshold_path)
 if any(k not in th or not math.isfinite(th[k]) or not 0<=th[k]<=1 for k in ['tokens','sentences','document']):raise ValueError('Invalid thresholds')
 import torch,numpy as np
 from transformers import AutoTokenizer,AutoModelForTokenClassification
 from score_wide_eval import summarize_row,aggregate,score_batch
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU required')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 files=[p for p in a.run.rglob('*') if p.is_file() and (p.suffix in ['.safetensors','.json','.bin'] or p.name.endswith('.index.json'))]
 contract={'suite':sha(HERE/'manifest.json'),'run':str(a.run),'model_files':{str(p.relative_to(a.run)):sha(p) for p in files},'adapter_files':{p.name:sha(p) for p in a.adapter_dir.glob('*.py')} if a.adapter_dir else None,'reference_files':{p.name:sha(p) for p in a.reference.iterdir() if p.is_file()},'thresholds':th,'threshold_origin':origin,'profiles':names,'limit':a.limit,'batch_size':a.batch_size,'rows_per_chunk':a.rows_per_chunk,'precision':'BF16','torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
 a.output.mkdir(parents=True,exist_ok=True);lock=a.output/'run.json'
 if lock.exists() and json.loads(lock.read_text())!=contract:raise ValueError('Resume settings changed; use a new output directory')
 save(lock,contract)
 ref=AutoTokenizer.from_pretrained(a.reference,local_files_only=True)
 if a.adapter_dir:
  sys.path.insert(0,str(a.adapter_dir));from inference import load_checkpoint,predict
  model,tok,_=load_checkpoint(a.run)
  def predict_rows(rs):return predict(rs,model,tok,batch_size=a.batch_size)
 else:
  model=AutoModelForTokenClassification.from_pretrained(a.run,local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval();tok=AutoTokenizer.from_pretrained(a.run,local_files_only=True)
  if model.config.num_labels!=2:raise ValueError('Binary head required; expected label 0 human, 1 AI')
  def predict_rows(rs):return score_batch(rs,model,tok,a.batch_size)
 def project(text,reference,offsets,scores):
  out=[];j=0
  for x,y in reference:
   while j<len(offsets) and offsets[j][1]<=x:j+=1
   k=j;v=w=0
   while k<len(offsets) and offsets[k][0]<y:
    c,d=offsets[k];n=max(0,min(y,d)-max(x,c));v+=n*float(scores[k]);w+=n;k+=1
   if not w and text[x:y].strip():raise ValueError('Uncovered reference text')
   out.append(v/w if w else 0)
  return np.asarray(out)
 for name in names:
  dest=a.output/name;dest.mkdir(exist_ok=True);rows=[json.loads(l) for l in gzip.open(HERE/(name+'.jsonl.gz'),'rt')];rows=rows[:a.limit] if a.limit else rows;started=time.time();records=[]
  for start in range(0,len(rows),a.rows_per_chunk):
   chunk=rows[start:start+a.rows_per_chunk];cache=dest/f'chunk-{start:06d}.json'
   if cache.exists():records.extend(json.loads(cache.read_text()));continue
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=predict_rows(chunk)
   if len(pred)!=len(chunk):raise ValueError('Prediction count mismatch')
   refs=ref([r['text'] for r in chunk],add_special_tokens=False,return_offsets_mapping=True,truncation=False)['offset_mapping'];part=[]
   raw={}
   for i,(r,off,(native,probs)) in enumerate(zip(chunk,refs,pred)):
    probs=np.asarray(probs)
    if len(native)!=len(probs) or not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any():raise ValueError('Invalid probabilities')
    scores=project(r['text'],off,native,probs);z=summarize_row(r,off,scores,th);z['document_flag']=z['mean_ai_probability']>=th['document'] if z['mean_ai_probability'] is not None else None
    for k in ['condition','view','split']:z[k]=r.get(k)
    z['span_annotation_conflict']=r.get('span_annotation_conflict',False)
    if name=='assistance':
     mask=np.array([b>r['target_start'] and c<r['target_end'] and bool(r['text'][c:b].strip()) for c,b in off]);z['target_flagged_fraction']=float((scores[mask]>=th['tokens']).mean()) if mask.any() else None
    raw[f'offsets_{i}']=np.asarray(off);raw[f'probabilities_{i}']=scores;raw[f'id_{i}']=np.array(r['id']);part.append(z)
   np.savez_compressed(dest/f'scores-{start:06d}.npz',**raw);save(cache,part);records.extend(part);save(a.output/'status.json',{'profile':name,'done':len(records),'total':len(rows),'state':'running'})
  reports={}
  for dataset in sorted({r['dataset'] for r in records}):
   rs=[r for r in records if r['dataset']==dataset];entry={'overall':aggregate(rs),'breakdowns':{}}
   for key in ['generator','condition','view']:
    entry['breakdowns'][key]={str(v):aggregate([r for r in rs if r.get(key)==v],False) for v in sorted({r[key] for r in rs if r.get(key) is not None})}
   if name=='assistance':entry['target_flagged_fraction_by_condition']={str(v):float(np.mean([r['target_flagged_fraction'] for r in rs if r['condition']==v and r['target_flagged_fraction'] is not None])) for v in sorted({r['condition'] for r in rs})}
   reports[dataset]=entry
  save(dest/'results.json',{'profile':name,'partial':bool(a.limit),'diagnostic_only':name=='assistance','rows':len(records),'elapsed_this_session_seconds':time.time()-started,'datasets':reports})
  with gzip.open(dest/'predictions.jsonl.gz','wt') as f:
   for r in records:f.write(json.dumps(r)+'\n')
 save(a.output/'status.json',{'state':'complete','profiles':names,'partial':bool(a.limit)})
if __name__=='__main__':main()
