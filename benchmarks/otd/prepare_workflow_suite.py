"""Package numeric provenance labels for the existing reduced-precision scorers."""
import gzip,json,hashlib
from collections import Counter
from pathlib import Path
import paper_workflow_eval as w
B=Path(__file__).resolve().parent

def main():
 out=w.OUT/'scoring';out.mkdir(exist_ok=True)
 papers={p['paper_id']:p for p in w.rows('papers.jsonl')}
 parent={pid:pid for pid in papers}
 def find(pid):
  while parent[pid]!=pid:parent[pid]=parent[parent[pid]];pid=parent[pid]
  return pid
 author_owner={}
 for pid,p in papers.items():
  for author in p['authors']:
   key=w.src.norm(author)
   if key in author_owner:
    a,b=find(pid),find(author_owner[key]);parent[max(a,b)]=min(a,b)
   else:author_owner[key]=pid
 components={pid:'authors-'+hashlib.sha256(find(pid).encode()).hexdigest()[:16] for pid in papers}
 component_splits={}
 for pid,c in components.items():
  assert c not in component_splits or component_splits[c]==papers[pid]['split'];component_splits[c]=papers[pid]['split']
 controls=w.rows('matched-human-controls.jsonl');generated=w.rows('dataset.jsonl');body=w.rows('human-body-audited.jsonl')
 suite=[];validation=[];assistance=[]
 def finish(r,dataset):
  r=dict(r);r.update(dataset=dataset,group_id=r.get('paper_id',r.get('group_id')),text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),domain=r.get('domain','research_paper'),cohort=r.get('condition',r.get('cohort','novel_body'))+':'+r.get('view','isolated'))
  if 'regions' not in r:r['regions']=[{'start':0,'end':len(r['text']),'label':0}]
  else:r['regions']=[{**x,'label':{'human':0,'ai':1,'assisted_unknown':-100}.get(x['label'],x['label'])} for x in r['regions']]
  labs={x['label'] for x in r['regions']};r['label']='human' if labs=={0} else 'ai' if labs=={1} else 'mixed'
  r['target_label']=1 if 1 in labs else -100 if -100 in labs else 0
  if r.get('paper_id') in papers:
   p=papers[r['paper_id']];r.update(conference=p['conference'],year=p['year'],author_component_id=components[r['paper_id']])
  assert all(x['label'] in [-100,0,1] for x in r['regions'])
  assert r['regions'][0]['start']==0 and r['regions'][-1]['end']==len(r['text'])
  assert all(a['end']==b['start'] for a,b in zip(r['regions'],r['regions'][1:]))
  return r
 for r in generated:
  if r['split']=='pilot':continue
  assisted=r['label_policy']=='human_origin_ai_assisted_no_binary_gold'
  row=finish(r,'paper_workflow_assistance' if assisted else 'paper_workflow_reconstruction')
  if assisted:assistance.append(row)
  else:(suite if r['split']=='test' else validation).append(row)
 for r in controls:
  if r['split']=='pilot':continue
  (suite if r['split']=='test' else validation).append(finish(r,'human_paper_workflow_matched'))
 for r in body:
  if r['split']=='pilot' or r['target_or_context']:continue
  row=finish(r,'human_paper_workflow_remaining');row['cohort']='novel_body'
  if r['split']=='test':suite.append(row)
  elif r['eligible_clean_novel_control']:validation.append(row)
 ellipse=B/'data/prepared/ellipse.jsonl'
 if ellipse.exists():
  for line in ellipse.open():suite.append(finish(json.loads(line),'ellipse'))
 def write(name,rs):
  assert len({r['id'] for r in rs})==len(rs)
  blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs).encode()
  (out/(name+'.jsonl.gz')).write_bytes(gzip.compress(blob,mtime=0))
  return hashlib.sha256(blob).hexdigest()
 hashes={'suite':write('suite',suite),'validation':write('validation',validation),'assistance':write('assistance',assistance)}
 previous=B/'training/paper-v3-modernbert/comparison-v1'
 thresholds={name:json.loads((previous/name/'thresholds.json').read_text())['thresholds'] for name in ['ours','meld-v5','meld-v8']}
 manifest={'created_utc':w.b.now(),'generation_protocol_sha256':w.b.sha((w.OUT/'protocol.json').read_bytes()),'source_freeze':json.loads((w.OUT/'main-freeze.json').read_text()),'files':hashes,'suite_rows':len(suite),'unique_suite_texts':len({r['text_sha256'] for r in suite}),'validation_rows':len(validation),'assistance_diagnostic_rows':len(assistance),'datasets':dict(Counter(r['dataset'] for r in suite)),'primary_thresholds':thresholds,'primary_threshold_policy':'Reuse comparison-v1 thresholds for every model; no new tuning. The original training sentence threshold for ours is an optional separately named sensitivity analysis.','secondary_calibration':'Use only this validation file for any explicitly reported recalibration; human remaining pool restricted to clean, novel paragraphs. Never fit a threshold on suite or ELLIPSE.','author_components_by_split':dict(Counter(component_splits.values())),'bootstrap_policy':'Primary intervals resample papers, keeping conditions and views together. Also report a sensitivity analysis resampling connected author components; ELLIPSE has no author ID, so essay-level uncertainty is qualified.','precision':'BF16 on A100; no FP32 reference inference','metric_units':'Common reference-token grid and sentence aggregation, with mixed-boundary/unknown labels masked. Report per condition and view, paired matched-human controls, and paper-clustered uncertainty.','label_contract':'0=historical human, 1=known reconstruction replacement, -100=unknown/assisted; interval endpoints are exclusive. Assistance is outside the binary suite.','controls':'Match reconstruction family_id to the shared original in human_paper_workflow_matched; do not count views or conditions as independent papers. Report clean-novel remaining controls separately from extraction/overlap diagnostics.','sealed_policy':'No detector scores used in data selection. Keep test data out of training and prompt development.','ellipse':'Official test essays; independent public supplement, not Pangram exact subset. See data/raw/ellipse/preparation-summary.json.','generation_complete':json.loads((w.OUT/'summary.json').read_text())['complete'],'model_scoring_started':False}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'suite_rows':len(suite),'datasets':manifest['datasets'],'assistance_rows':len(assistance)}))
if __name__=='__main__':main()
