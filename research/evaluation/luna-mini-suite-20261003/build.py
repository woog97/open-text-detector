import collections,gzip,hashlib,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OLD=HERE.parent/'luna-flex-hillclimb-20261003';BASE=HERE.parent/'luna-flex-pilot-20261003'
def load(p):return [json.loads(l) for l in gzip.open(p,'rt')]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def rank(r):return sha('mini-suite-v1/'+r['id'])
prior=json.loads((BASE/'manifest.json').read_text())['rows']; excluded={r['paper_id'] for r in prior};texts={r['text_sha256'] for r in prior}
for name in ['fewshot-examples.json','fewshot-diverse-examples.json']:
 for r in json.loads((OLD/name).read_text()):excluded.add(r['paper_id']);texts.update(r.get(k+'_sha256') or sha(r[k]) for k in ['HUMAN','AI'])  # files now hold text hashes only (third-party text removed Oct 13 2026)
def eligible(r):return not ({str(r.get(k)) for k in ['paper_id','group_id','family_id']} & excluded) and sha(r['text']) not in texts
bundle=ROOT/'benchmarks/otd/eval_suite/bundle';package=HERE.parent/'space-suite-v1/package';selected=[]
def add(r,profile):
 selected.append({'id':r['id'],'paper_id':r.get('paper_id') or r.get('group_id') or r['id'],'label':'HUMAN' if r['label']=='human' else 'AI','original_label':r['label'],'text':r['text'],'text_sha256':sha(r['text']),'split':r.get('split'),'profile':profile,'dataset':r.get('dataset'),'condition':r.get('condition'),'generator':r.get('generator'),'source_group':r.get('group_id'),'source_development_exposed':r.get('development_exposed')})
comparison=[r for r in load(package/'comparison.jsonl.gz') if eligible(r)]
usedgroups=set();usedtexts=set()
for label in ['human','ai','mixed']:
 buckets=collections.defaultdict(list)
 for r in sorted(comparison,key=rank):
  if r['label']==label:buckets[r['dataset']].append(r)
 count=0
 while count<40:
  progress=False
  for dataset in sorted(buckets):
   while buckets[dataset]:
    r=buckets[dataset].pop(0);g=r.get('group_id') or r['id'];h=sha(r['text'])
    if g in usedgroups or h in usedtexts:continue
    add(r,'comparison');usedgroups.add(g);usedtexts.add(h);count+=1;progress=True;break
   if count==40:break
  assert progress
workflow=[r for r in load(bundle/'workflow_validation.jsonl.gz') if eligible(r) and r.get('view')=='paragraph']
humans=sorted([r for r in workflow if r.get('condition')=='untouched'],key=rank)
families={r['paper_id'] for r in humans[:12]};assert len(families)==12
for r in workflow:
 if r['paper_id'] in families:add(r,'reconstruction')
for r in load(bundle/'assistance.jsonl.gz'):
 if r.get('paper_id') in families and r.get('view')=='paragraph' and r.get('condition') in ['proofread','light_polish']:
  assert eligible(r);add(r,'assistance')
ms=[r for r in load(package/'manuscripts.jsonl.gz') if eligible(r)]
for generator in sorted({r['generator'] for r in ms}):
 for r in sorted([r for r in ms if r['generator']==generator],key=rank)[:4]:add(r,'manuscripts')
assert collections.Counter(r['profile'] for r in selected)=={'comparison':120,'reconstruction':60,'assistance':24,'manuscripts':12}
assert len({r['id'] for r in selected})==216
random.Random(20261003).shuffle(selected)
(HERE/'dataset.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
(HERE/'sampling.json').write_text(json.dumps({'seed':20261003,'excluded_papers':sorted(excluded),'profiles':dict(collections.Counter(r['profile'] for r in selected)),'comparison_datasets':dict(collections.Counter(r['dataset'] for r in selected if r['profile']=='comparison')),'notes':['Comparison uses 40 human,40 AI,40 mixed, round-robin dataset sampling; one example per source group within comparison.','Workflow uses 12 unexposed calibration papers with all four reconstruction conditions and human controls.','Assistance reuses these 12 families; report flagging rates, not classification accuracy.','Manuscripts: four per final revision generator; positive-only detection, no specificity.','Known prompt tuning/demo papers and exact texts excluded; this is not an exhaustive cross-corpus near-duplicate or pretraining contamination audit.']},indent=2)+'\n')
print((HERE/'sampling.json').read_text())
