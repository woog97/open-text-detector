"""Frozen multi-model comparison with validation-only operating points."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,gzip,time,hashlib,collections,math,gc,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
from common import labels_from_regions,choose_threshold,counts,metrics
from score_wide_eval import score_batch,sentences,summarize_row,aggregate,write
from meld_model import MeldModel
ROOT=Path(__file__).parent;OUT=ROOT/'comparison-v1'
def status(event,**kw):write(OUT/'status.json',{'event':event,'time':time.time(),**kw});print(json.dumps({'event':event,**kw}),flush=True)
def project(text,reference,offsets,values):
 if list(map(tuple,reference))==list(map(tuple,offsets)):return np.asarray(values,np.float64)
 out=[];j=0
 for a,b in reference:
  while j<len(offsets) and offsets[j][1]<=a:j+=1
  k=j;total=0.;weight=0
  while k<len(offsets) and offsets[k][0]<b:
   c,d=offsets[k];n=max(0,min(b,d)-max(a,c));total+=n*float(values[k]);weight+=n;k+=1
  if not weight and text[a:b].strip():raise ValueError(f'Uncovered nonspace reference token {a}:{b}')
  out.append(total/weight if weight else 0.)
 return np.asarray(out,np.float64)
@torch.inference_mode()
def meld_score(rows,model,tok,batch=8):
 enc=tok([r['text'] for r in rows],add_special_tokens=False,return_offsets_mapping=True,truncation=False);ids=enc['input_ids'];values=[np.zeros(len(x),np.float32) for x in ids];jobs=[]
 for i,x in enumerate(ids):
  for s in range(0,len(x),2046):jobs.append((i,s,min(s+2046,len(x))))
 jobs.sort(key=lambda q:q[2]-q[1])
 for pos in range(0,len(jobs),batch):
  part=jobs[pos:pos+batch];n=max(e-s+2 for i,s,e in part);ii=[];am=[]
  for i,s,e in part:
   seq=[tok.cls_token_id]+ids[i][s:e]+[tok.sep_token_id];ii.append(seq+[tok.pad_token_id]*(n-len(seq)));am.append([1]*len(seq)+[0]*(n-len(seq)))
  with torch.autocast('cuda',dtype=torch.bfloat16):
   scores=model.token_scores(torch.tensor(ii,device='cuda'),torch.tensor(am,device='cuda')).float().cpu().numpy()
  for (i,s,e),v in zip(part,scores):values[i][s:e]=v[1:1+e-s]
 assert all(np.isfinite(v).all() for v in values)
 return list(zip(enc['offset_mapping'],values))
def units(row,off,p):
 off=np.asarray(off);ys=np.asarray(labels_from_regions(row['text'],off,row['regions']));valid=np.asarray([bool(row['text'][a:b].strip()) for a,b in off]);s=[];y=[]
 for a,b in sentences(row['text']):
  use=valid&(off[:,1]>a)&(off[:,0]<b);labs=set(ys[use])
  if use.any() and len(labs)==1 and next(iter(labs)) in (0,1):s.append(float(p[use].mean()));y.append(next(iter(labs)))
 mask=valid&(ys>=0)
 return p[mask],ys[mask],s,y

def compute(rows,model,tok,ref,name,folder):
 folder.mkdir(exist_ok=True);result=[];unique=list({r['text_sha256']:r for r in rows}.values())
 for start in range(0,len(unique),64):
  chunk=unique[start:start+64];file=folder/f'{start:06d}.npz'
  if file.exists():
   with np.load(file,allow_pickle=False) as d:
    assert d['hashes'].tolist()==[r['text_sha256'] for r in chunk]
    for i,r in enumerate(chunk):a,b=d['indptr'][i:i+2];result.append((r,d['offsets'][a:b].copy(),d['scores'][a:b].copy()))
   continue
  pred=score_batch(chunk,model,tok,32) if name=='ours' else meld_score(chunk,model,tok)
  ref_off=ref([r['text'] for r in chunk],add_special_tokens=False,return_offsets_mapping=True,truncation=False)['offset_mapping'];part=[]
  for r,off,(native,p) in zip(chunk,ref_off,pred):part.append((r,np.asarray(off,np.int32),project(r['text'],off,native,p)))
  ind=np.cumsum([0]+[len(p) for _,_,p in part]);tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,hashes=[r['text_sha256'] for r in chunk],indptr=ind,offsets=np.concatenate([o for _,o,_ in part]),scores=np.concatenate([p for _,_,p in part]));tmp.replace(file);result+=part
  status('scoring',model=name,phase=folder.name,completed=min(start+64,len(unique)),total=len(unique))
 by={r['text_sha256']:(o,p) for r,o,p in result}
 return [(r,*by[r['text_sha256']]) for r in rows]

def report(pred):
 out={}
 for ds in sorted({r['dataset'] for r in pred}):
  rows=[r for r in pred if r['dataset']==ds];out[ds]={'overall':aggregate(rows),'breakdowns':{}}
  for field in ['cohort','generator','domain','attack','clean_prose','year','conference']:
   values={str(r[field]) for r in rows if r.get(field) is not None}
   if len(values)>1:out[ds]['breakdowns'][field]={v:aggregate([r for r in rows if str(r.get(field))==v],False) for v in sorted(values)}
  if ds.startswith('human_paper'):
   out[ds]['novel_clean']=aggregate([r for r in rows if r.get('cohort')=='novel_body' and r.get('clean_prose')])
 return out

def main():
 started=time.time();manifest=json.loads((OUT/'manifest.json').read_text());data={}
 for split in ['suite','validation']:
  blob=gzip.decompress((OUT/(split+'.jsonl.gz')).read_bytes());assert hashlib.sha256(blob).hexdigest()==manifest['files'][split];data[split]=[json.loads(l) for l in blob.splitlines()]
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ref=AutoTokenizer.from_pretrained(ROOT/'run-01/best_model');ref.model_max_length=10**9
 for name in os.environ.get('COMPARISON_MODELS','ours,meld-v5,meld-v8').split(','):
  dest=OUT/name;dest.mkdir(exist_ok=True)
  if (dest/'results.json').exists():continue
  begin=time.time();status('loading',model=name)
  if name=='ours':
   model=AutoModelForTokenClassification.from_pretrained(ROOT/'run-01/best_model',attn_implementation='sdpa').cuda().eval();tok=ref
  else:
   directory=ROOT/'baseline-models'/name;model=MeldModel(directory).cuda().eval();tok=AutoTokenizer.from_pretrained(directory,trust_remote_code=False);tok.model_max_length=10**9
   # Compare BF16 batched inference against BF16 single-example inference.
   probes=data['validation'][:3];bat=meld_score(probes,model,tok);errors=[]
   for r,(off,p) in zip(probes,bat):
    oo,pp=meld_score([r],model,tok,1)[0];assert off==oo;errors.append(float(np.max(np.abs(p-pp))))
   assert all(np.isfinite(errors)),errors
   write(dest/'batch_parity.json',{'max_absolute_token_error':max(errors),'examples':len(probes),'precision':'bfloat16','note':'Numerical padding sensitivity recorded; both paths use BF16. No FP32 reference inference.'})
  validation=compute(data['validation'],model,tok,ref,name,dest/'validation-scores');ts=[];ty=[];ss=[];sy=[];doc=[]
  for r,o,p in validation:
   a,b,c,d=units(r,o,p);ts.extend(a);ty.extend(b);ss.extend(c);sy.extend(d)
   if r['target_label']==0:doc.append(float(p[[bool(r['text'][a:b].strip()) for a,b in o]].mean()))
  th={'tokens':choose_threshold(ts,ty),'sentences':choose_threshold(ss,sy),'document':choose_threshold(doc,[0]*len(doc))}
  write(dest/'thresholds.json',{'thresholds':th,'validation_rows':len(validation),'token_validation':metrics(counts(ts,ty,th['tokens'])),'sentence_validation':metrics(counts(ss,sy,th['sentences'])),'document_human_validation_fpr':float(np.mean(np.asarray(doc)>=th['document'])),'document_human_rows':len(doc),'policy':manifest['calibration']})
  del ts,ty,ss,sy,validation;gc.collect();status('calibrated',model=name,thresholds=th)
  scored=compute(data['suite'],model,tok,ref,name,dest/'test-scores');del model;torch.cuda.empty_cache();pred=[];frozen=[]
  old=json.loads((ROOT/'run-01/thresholds.json').read_text())
  for r,o,p in scored:
   rec=summarize_row(r,o,p,th);rec['span_annotation_conflict']=r.get('span_annotation_conflict',False);rec['document_flag']=rec['mean_ai_probability']>=th['document'];pred.append(rec)
   if name=='ours':
    z=summarize_row(r,o,p,old);z['span_annotation_conflict']=rec['span_annotation_conflict'];frozen.append(z)
  with gzip.open(dest/'predictions.jsonl.gz','wt') as f:
   for r in pred:f.write(json.dumps(r)+'\n')
  elapsed=time.time()-begin;status('aggregating',model=name,inference_and_calibration_seconds=elapsed)
  write(dest/'results.json',{'model':name,'thresholds':th,'elapsed_seconds':elapsed,'datasets':report(pred),'protocol':manifest,'document_score_name':'mean reference-token AI probability (ours) or raw evidence (MELD); historical field name mean_ai_probability retained in machine records'})
  if frozen:write(dest/'original-threshold-results.json',{'model':name,'thresholds':old,'datasets':report(frozen)})
  status('model_complete',model=name,elapsed_seconds=time.time()-begin)
  del scored,pred,frozen;gc.collect()
 status('complete',elapsed_seconds=time.time()-started)
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=type(e).__name__,message=str(e));traceback.print_exc();raise
