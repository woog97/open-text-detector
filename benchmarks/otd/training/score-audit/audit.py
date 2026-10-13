import os,sys,json,gzip,time,collections
from pathlib import Path
import numpy as np
def roc_curve(y,p,drop_intermediate=False):
 order=np.argsort(-p,kind='stable');p=p[order];y=y[order]
 ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1]
 tp=np.cumsum(y)[ends];fp=ends+1-tp
 return np.r_[0,fp/(y==0).sum()],np.r_[0,tp/(y==1).sum()],np.r_[np.inf,p[ends]]
def roc_auc_score(y,p):
 f,t,_=roc_curve(y,p);return np.trapezoid(t,f)
ROOT=Path(__file__).resolve().parent
W=Path('/data/workspace')
def save(name,x):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def metrics(scores,labels,threshold):
 p=np.asarray(scores);y=np.asarray(labels);valid=np.isfinite(p)&np.isin(y,[0,1]);p=p[valid];y=y[valid]
 out={'n_human':int((y==0).sum()),'n_ai':int((y==1).sum()),'threshold':threshold}
 for label,name in [(0,'human'),(1,'ai')]:
  vals=p[y==label];out[name+'_quantiles']=np.quantile(vals,[0,.1,.5,.9,.99,1]).tolist() if len(vals) else None
 out['current_recall']=float((p[y==1]>=threshold).mean()) if out['n_ai'] else None
 out['current_fpr']=float((p[y==0]>=threshold).mean()) if out['n_human'] else None
 if out['n_ai'] and out['n_human']:
  out['auroc']=float(roc_auc_score(y,p));fpr,tpr,ths=roc_curve(y,p,drop_intermediate=False)
  i=np.flatnonzero(fpr<=.01)[-1];out['diagnostic_test_oracle_recall_at_1pct_fpr']=float(tpr[i]);out['diagnostic_threshold']=float(ths[i]) if np.isfinite(ths[i]) else None
 return out

def documents():
 outputs={}
 paths=list((W/'paper-backbone-comparison-v1/results').glob('*/predictions.jsonl.gz'))+list((W/'paper-lora-comparison-v1/results').glob('*/predictions.jsonl.gz'))+list((W/'paper-mix-sweep-v1').glob('*/results/*/predictions.jsonl.gz'))
 for p in paths:
  if 'paper-mix-sweep-v1' in str(p):run=p.parents[2]/'run'
  else:run=p.parents[2]/'runs'/p.parent.name.rsplit('-',1)[0]
  if not (run/'thresholds.json').exists():continue
  threshold=json.loads((run/'thresholds.json').read_text())['thresholds']['document'];groups=collections.defaultdict(list)
  with gzip.open(p,'rt') as f:
   for line in f:
    row=json.loads(line)
    if row.get('native_label') in ['human','ai']:groups[row['dataset']].append(row)
  report={}
  for ds,rows in groups.items():
   byhash=collections.defaultdict(list)
   for row in rows:byhash[row['text_sha256']].append(row)
   unique=[rs[0] for rs in byhash.values() if len({r['native_label'] for r in rs})==1 and rs[0]['granularity']!='assisted_or_ambiguous_native_label']
   report[ds]=metrics([r['mean_ai_probability'] for r in unique],[int(r['native_label']=='ai') for r in unique],threshold)
   report[ds]['deduplication']='Unique text within dataset; conflicting labels excluded'
  outputs[str(p.relative_to(W))]=report
 save('document-diagnostic.json',{'policy':'Test-oracle thresholds diagnose separability ONLY; not deployable or unbiased performance estimates. Native document labels never used as token gold. Small human samples cannot establish reliable 1% FPR.','results':outputs})
 print('Saved document diagnostic for',len(outputs),'model/profile combinations',flush=True)

def token_audit():
 modelroot=W/'paper-mix-sweep-v1/human50';suite=W/'paper-v3-modernbert-20260930/eval-suite-v1'
 sys.path.insert(0,str(modelroot));from train import require_gpu
 from inference import load_checkpoint,predict
 from transformers import AutoTokenizer
 sys.path.insert(0,str(suite));import compare_models as c
 require_gpu();model,tok,contract=load_checkpoint(modelroot/'run');ref=AutoTokenizer.from_pretrained(W/'paper-v3-modernbert-20260930/run-01/best_model',local_files_only=True)
 thresholds=json.loads((modelroot/'run/thresholds.json').read_text())['thresholds'];reports={};max_mean_delta=0.
 for profile in ['workflow','comparison']:
  rows=[json.loads(l) for l in gzip.decompress((suite/(profile+'.jsonl.gz')).read_bytes()).splitlines()]
  # Token gold comes only from recorded spans or observed generated responses.
  # Preserve the original inference batch composition, including document-only rows.
  old={r['id']:r for r in (json.loads(l) for l in gzip.open(modelroot/'results'/profile/'predictions.jsonl.gz','rt'))}
  groups=collections.defaultdict(lambda:{'tokens':([],[]),'sentences':([],[])})
  labels_by_text=collections.defaultdict(set)
  for x in old.values():labels_by_text[(x['dataset'],x['text_sha256'])].add(x['native_label'])
  seen=set();filtered=[]
  for r in rows:
   key=(r['dataset'],r['text_sha256'])
   if key in seen or len(labels_by_text[key])>1 or r.get('span_annotation_conflict',False):continue
   seen.add(key);filtered.append(r)
  allowed={r['id'] for r in filtered if 'regions' in r or r.get('granularity')=='observed_generated_response'}
  for start in range(0,len(rows),32):
   chunk=rows[start:start+32];pred=predict(chunk,model,tok);offs=ref([r['text'] for r in chunk],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
   for r,off,(native,prob) in zip(chunk,offs,pred):
    p=c.project(r['text'],off,native,prob);valid=np.array([bool(r['text'][a:b].strip()) for a,b in off]);max_mean_delta=max(max_mean_delta,abs(float(p[valid].mean())-old[r['id']]['mean_ai_probability']))
    if r['id'] not in allowed:continue
    if 'regions' not in r:r={**r,'regions':[{'start':0,'end':len(r['text']),'label':1}]}
    ts,ty,ss,sy=c.units(r,off,p)
    for unit,sc,la in [('tokens',ts,ty),('sentences',ss,sy)]:groups[r['dataset']][unit][0].extend(map(float,sc));groups[r['dataset']][unit][1].extend(map(int,la))
   save('status.json',{'state':'token_sentence_diagnostic','profile':profile,'rows':min(start+32,len(rows)),'total':len(rows)})
  reports[profile]={ds:{u:metrics(*v,thresholds[u]) for u,v in vals.items()} for ds,vals in groups.items()}
 save('token-sentence-diagnostic.json',{'model':'human50 ModernBERT LoRA','results':reports,'max_document_mean_delta_vs_saved':max_mean_delta,'policy':'Diagnostic test-oracle only; existing calibration thresholds unchanged. Token/sentence counts are correlated within papers, not independent samples.'})
 save('status.json',{'state':'complete','max_document_mean_delta_vs_saved':max_mean_delta})
if __name__=='__main__':
 try:documents();token_audit()
 except Exception as e:
  save('status.json',{'state':'failed','error':str(e)});raise
