"""Score a frozen broad suite; no fitting, generation, or threshold selection."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,json,gzip,collections,hashlib,time,re,math,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
from common import labels_from_regions,window_starts,counts,metrics

def write(path,value):
 path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def progress(out,event,**kw):
 r={'event':event,'time':time.time(),**kw};write(out/'status.json',r);print(json.dumps(r),flush=True)
def sentences(text):
 return [(m.start(),m.end()) for m in re.finditer(r'\S.*?(?:[.!?](?=\s|$)|$)',text,re.S) if m.group().strip()]
def auroc(scores,labels):
 scores=np.asarray(scores);labels=np.asarray(labels);n1=int(labels.sum());n0=len(labels)-n1
 if not n0 or not n1:return None
 order=np.argsort(scores,kind='stable');ss=scores[order];ys=labels[order];total=0.;lo=0
 while lo<len(ss):
  hi=lo+1
  while hi<len(ss) and ss[hi]==ss[lo]:hi+=1
  total+=float(ys[lo:hi].sum())*((lo+1+hi)/2);lo=hi
 return (total-n1*(n1+1)/2)/(n1*n0)

@torch.inference_mode()
def score_batch(rows,model,tok,batch_size=32):
 encoded=tok([r['text'] for r in rows],add_special_tokens=False,return_offsets_mapping=True,truncation=False)
 ids=encoded['input_ids'];offsets=encoded['offset_mapping'];summed=[np.zeros(len(x),np.float64) for x in ids];denom=[np.zeros(len(x),np.int32) for x in ids];jobs=[]
 for i,x in enumerate(ids):
  for s in window_starts(len(x)):jobs.append((i,s,min(s+510,len(x))))
 jobs.sort(key=lambda x:x[2]-x[1])
 for b in range(0,len(jobs),batch_size):
  part=jobs[b:b+batch_size];n=math.ceil(max(e-s+2 for i,s,e in part)/8)*8;ii=[];am=[]
  for i,s,e in part:
   x=[tok.cls_token_id]+ids[i][s:e]+[tok.sep_token_id];ii.append(x+[tok.pad_token_id]*(n-len(x)));am.append([1]*len(x)+[0]*(n-len(x)))
  with torch.autocast('cuda',dtype=torch.bfloat16):p=model(input_ids=torch.tensor(ii,device='cuda'),attention_mask=torch.tensor(am,device='cuda')).logits.float().softmax(-1)[...,1].cpu().numpy()
  for (i,s,e),prob in zip(part,p):summed[i][s:e]+=prob[1:1+e-s];denom[i][s:e]+=1
 assert all(np.all(d>0) for d in denom)
 return [(off,(a/d).astype(np.float32)) for off,a,d in zip(offsets,summed,denom)]

def summarize_row(r,off,p,thresholds):
 off=np.asarray(off);text=r['text'];valid=np.array([bool(text[a:b].strip()) for a,b in off]);scores=p.astype(np.float64);flag=scores>=thresholds['tokens']
 strong='regions' in r or r['granularity']=='observed_generated_response'
 reg=r.get('regions')
 if reg is None and strong:reg=[{'start':0,'end':len(text),'label':1}]
 ys=np.asarray(labels_from_regions(text,off,reg)) if strong else np.full(len(p),-100)
 tokmask=valid & (ys>=0)
 tc=counts(scores[tokmask],ys[tokmask],thresholds['tokens']).tolist() if strong else None
 s_scores=[];s_labels=[];mixed=0
 for a,b in sentences(text):
  use=valid & (off[:,1]>a) & (off[:,0]<b)
  if not use.any():continue
  s_scores.append(float(scores[use].mean()))
  labs=set(ys[use]);label=next(iter(labs)) if len(labs)==1 and next(iter(labs)) in (0,1) else -100
  s_labels.append(label)
  if strong and label==-100:mixed+=1
 s_scores=np.asarray(s_scores);s_labels=np.asarray(s_labels);sm=s_labels>=0
 sc=counts(s_scores[sm],s_labels[sm],thresholds['sentences']).tolist() if strong else None
 n=int(valid.sum());tf=float(flag[valid].mean()) if n else None
 return {'id':r['id'],'dataset':r['dataset'],'cohort':r.get('cohort'),'group_id':str(r.get('group_id',r['id'])),'text_sha256':r['text_sha256'],'native_label':r['label'],'granularity':r['granularity'],'generator':r.get('generator'),'domain':r.get('domain'),'attack':r.get('attack'),'operation':r.get('operation'),'conference':r.get('conference'),'year':r.get('year'),'quality_flags':r.get('quality_flags',[]),'clean_prose':r.get('clean_prose'),'length_bucket':r.get('length_bucket'),'words':r.get('words',len(text.split())),'tokens':n,'sentences':len(s_scores),'mean_ai_probability':float(scores[valid].mean()) if n else None,'flagged_token_fraction':tf,'flagged_sentence_fraction':float((s_scores>=thresholds['sentences']).mean()) if len(s_scores) else None,'document_flag':bool(tf>=.5) if tf is not None else None,'token_counts':tc,'sentence_counts':sc,'mixed_or_ignored_sentences':mixed}

def bootstrap_counts(rows,key,reps=400):
 bygroup=collections.defaultdict(lambda:np.zeros(4,np.int64))
 for r in rows:
  if r[key] is not None:bygroup[r['group_id']]+=r[key]
 if not bygroup:return None
 a=np.asarray(list(bygroup.values()));out={**metrics(a.sum(0)),'units':int(a.sum()),'groups':len(a)};samples=collections.defaultdict(list);rng=np.random.default_rng(42)
 for _ in range(reps):
  m=metrics(a[rng.integers(0,len(a),len(a))].sum(0))
  for k in ['precision','recall','human_fpr']:
   if m[k] is not None:samples[k].append(m[k])
 out['cluster_bootstrap_95ci']={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items() if v}
 return out

def aggregate(rows,bootstrap=True):
 # Each text has one vote within the displayed condition. Conflicting native labels
 # remain in the audit but are removed from binary document classification metrics.
 groups=collections.defaultdict(list)
 for r in rows:groups[r['text_sha256']].append(r)
 unique=[v[0] for v in groups.values()];conflict={k for k,v in groups.items() if len({r['native_label'] for r in v})>1}
 native=[r for r in unique if r['native_label'] in ['human','ai'] and r['granularity']!='assisted_or_ambiguous_native_label' and r['document_flag'] is not None and r['text_sha256'] not in conflict]
 labels=[int(r['native_label']=='ai') for r in native];scores=[r['mean_ai_probability'] for r in native]
 c=counts([int(r['document_flag']) for r in native],labels,.5) if native else np.zeros(4,int)
 strong=[r for r in unique if r['token_counts'] is not None and r['text_sha256'] not in conflict and not r.get('span_annotation_conflict',False)]
 out={'rows':len(rows),'unique_texts':len(unique),'duplicate_rows':len(rows)-len(unique),'conflicting_native_label_texts':len(conflict),'span_annotation_conflict_rows':sum(bool(r.get('span_annotation_conflict')) for r in rows),'native_labels':dict(collections.Counter(r['native_label'] for r in unique)),'document_native_label_metrics':{**metrics(c),'evaluated_documents':len(native),'auroc_mean_probability':auroc(scores,labels) if native else None},'span_or_historical_human_token_metrics':bootstrap_counts(strong,'token_counts',400 if bootstrap else 0),'span_or_historical_human_sentence_metrics':bootstrap_counts(strong,'sentence_counts',400 if bootstrap else 0),'by_native_label':{}}
 for label in ['human','ai','mixed']:
  rs=[r for r in unique if r['native_label']==label]
  if rs:out['by_native_label'][label]={'documents':len(rs),'mean_flagged_token_fraction':float(np.mean([r['flagged_token_fraction'] for r in rs if r['flagged_token_fraction'] is not None])),'mean_flagged_sentence_fraction':float(np.mean([r['flagged_sentence_fraction'] for r in rs if r['flagged_sentence_fraction'] is not None])),'document_flag_rate':float(np.mean([r['document_flag'] for r in rs if r['document_flag'] is not None]))}
 return out

def main(args):
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);started=time.time();manifest=json.loads((out/'manifest.json').read_text())
 run=Path(args.run);thresholds=json.loads((run/'thresholds.json').read_text());assert thresholds==manifest['thresholds']
 h=hashlib.sha256((run/'best_model/model.safetensors').read_bytes()).hexdigest();assert h==manifest['checkpoint_sha256']
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
 progress(out,'loading',checkpoint_sha256=h)
 tok=AutoTokenizer.from_pretrained(run/'best_model');model=AutoModelForTokenClassification.from_pretrained(run/'best_model',attn_implementation='sdpa').cuda().eval()
 path=out/'suite.jsonl.gz'
 with gzip.open(path,'rt') as f:rows=[json.loads(l) for l in f]
 assert len(rows)==manifest['evaluation_rows']
 assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()==manifest['files']['suite.jsonl']
 grouped=collections.defaultdict(list)
 for r in rows:grouped[r['text_sha256']].append(r)
 unique=[v[0] for v in grouped.values()];dest=out/'predictions.jsonl'
 done={}
 if dest.exists():
  for line in dest.open():
   r=json.loads(line);done[r['id']]=r
 pending=[r for r in unique if not all(x['id'] in done for x in grouped[r['text_sha256']])]
 total_tokens=0
 with dest.open('a') as f:
  for start in range(0,len(pending),128):
   chunk=pending[start:start+128];pred=score_batch(chunk,model,tok)
   for r,(off,p) in zip(chunk,pred):
    total_tokens+=len(p)
    for variant in grouped[r['text_sha256']]:
     if variant['id'] in done:continue
     record=summarize_row(variant,off,p,thresholds);f.write(json.dumps(record,allow_nan=False)+'\n');done[record['id']]=record
   f.flush()
   progress(out,'scoring',completed_rows=len(done),total_rows=len(rows),unique_texts_remaining=max(0,len(pending)-start-len(chunk)),tokens_scored_this_process=total_tokens,elapsed_seconds=time.time()-started)
 assert set(done)=={r['id'] for r in rows}
 del model;torch.cuda.empty_cache();predictions=list(done.values());result={'manifest':manifest,'completed_at':time.time(),'elapsed_seconds':time.time()-started,'datasets':{},'breakdowns':{},'counts':{'rows':len(predictions),'unique_texts':len(unique)}}
 progress(out,'aggregating',completed_rows=len(done))
 for dataset in sorted({r['dataset'] for r in predictions}):
  rr=[r for r in predictions if r['dataset']==dataset];result['datasets'][dataset]=aggregate(rr)
  fields=['cohort','generator','domain','attack','length_bucket']
  if dataset=='human_paper_remaining':fields+=['conference','year','clean_prose']
  result['breakdowns'][dataset]={}
  for field in fields:
   parts=collections.defaultdict(list)
   for r in rr:
    if r.get(field) is not None:parts[str(r[field])].append(r)
   if len(parts)>1:result['breakdowns'][dataset][field]={k:aggregate(v,False) for k,v in sorted(parts.items())}
  if dataset=='human_paper_remaining':
   for name,rs in [('novel_clean',[r for r in rr if r['cohort']=='novel_body' and r['clean_prose']]),('novel_all',[r for r in rr if r['cohort']=='novel_body']),('previous_context',[r for r in rr if r['cohort']=='previous_context'])]:result['breakdowns'][dataset][name]=aggregate(rs)
 write(out/'results.json',result);progress(out,'complete',rows=len(done),unique_texts=len(unique),elapsed_seconds=time.time()-started)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 try:main(a)
 except Exception as e:
  progress(Path(a.output),'failed',error=type(e).__name__,message=str(e));traceback.print_exc();raise
