"""Freeze a broad, score-independent evaluation suite and split-safe human pools."""
from pathlib import Path
import sys,json,re,hashlib,collections,concurrent.futures,time,unicodedata
ROOT=Path(__file__).resolve().parents[4];B=ROOT/'benchmarks/otd'
sys.path.insert(0,str(B));import prepare as published
import paper_gap50 as gap
import paper_gap250_sources as source
import pyarrow.parquet as pq
SOURCE=ROOT/'research/data/paper-gap10000-v3-luna-20260930'
OUT=Path(__file__).parent/'wide-eval-v1'

def readl(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def writel(p,rows):
 with Path(p).open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).lower()))
def shingle(s):
 w=s.split()
 return (hashlib.blake2b(' '.join(w[i:i+32]).encode(),digest_size=12).digest() for i in range(len(w)-31))

def extract(p):
 gap.OUT=SOURCE;gap.SOURCE=SOURCE
 try:return p['paper_id'],source.paragraphs(p),None
 except Exception as e:return p['paper_id'],[],type(e).__name__+': '+str(e)[:250]

def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'manifest.json').exists():raise RuntimeError('Frozen suite already exists; use a new version.')
 papers=list(readl(SOURCE/'papers.jsonl'));byid={p['paper_id']:p for p in papers};passages=list(readl(SOURCE/'passages.jsonl'))
 targets=collections.defaultdict(set);positions=collections.defaultdict(set);contexts=collections.defaultdict(set)
 for r in passages:
  pid=r['paper_id'];targets[pid].add(norm(r['held_out']))
  sp=r.get('source_paragraphs',[])
  if len(sp)==3:positions[pid].add(tuple(sp[1].get(k) for k in ['page','block','chunk']))
  contexts[pid].update([norm(r['before']),norm(r['after'])])
 train_manifest=json.loads((OUT.parent/'run-01/data_manifest.json').read_text());used=set(train_manifest['train_ids']+train_manifest['validation_ids'])
 exported=B/'exports/ai-paper-provenance-v3-10000/data/fresh_passages'
 forbidden_hashes=set();forbidden_shingles=set();forbidden_papers=set();forums={}
 for split in ['train','validation','test']:
  rows=pq.read_table(exported/f'{split}-00000-of-00001.parquet',columns=['id','paper_id','forum_id','text','sentences']).to_pylist()
  for r in rows:
   forums[r['paper_id']]=r['forum_id']
   if r['id'] not in used:continue
   forbidden_papers.add(r['paper_id']);t=norm(r['text']);forbidden_hashes.add(sha(t));forbidden_shingles.update(shingle(t))
   for text in [*r['text'].split('\n\n'),*[s['text'] for s in r['sentences']]]:forbidden_hashes.add(sha(norm(text)))
 def leakage(text):
  t=norm(text)
  return sha(t) in forbidden_hashes or any(x in forbidden_shingles for x in shingle(t))
 print('Frozen training/validation overlap index:',len(forbidden_shingles),flush=True)
 human=[];extraction_errors=[];target_exclusions=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
  for i,(pid,paras,error) in enumerate(pool.map(extract,papers)):
   if error:extraction_errors.append({'paper_id':pid,'error':error});continue
   p=byid[pid];seen=set()
   for j,q in enumerate(paras):
    t=q['text'];n=norm(t)
    if n in seen:continue
    seen.add(n);pos=tuple(q.get(k) for k in ['page','block','chunk'])
    if n in targets[pid] or pos in positions[pid] or any(len(x)>100 and (x in n or n in x) for x in targets[pid]):target_exclusions+=1;continue
    prior_context=n in contexts[pid];flags=[]
    if not re.match(r'[A-Z“]',t):flags.append('nonstandard_start')
    if not re.search(r'[.!?][\]”\"]?$',t):flags.append('incomplete_ending')
    if len(t.split())>600:flags.append('long_extraction')
    if re.match(r'(?:Theorem|Lemma|Proof|Proposition|Corollary|Definition|Remark|Assumption)\b',t):flags.append('formal_statement')
    if sum(c.isalpha() or c.isspace() for c in t)/len(t)<.85:flags.append('symbol_heavy')
    human.append({'id':f'human_remaining:{pid}:{j:04d}','dataset':'human_paper_remaining','text':t,'text_sha256':sha(t),'label':'human','paper_id':pid,'forum_id':forums.get(pid),'group_id':pid,'split':p['split'],'development_exposed':p.get('development_exposed',False),'conference':p['conference'],'year':p['year'],'domain':'research_paper','cohort':'previous_context' if prior_context else 'novel_body','quality_flags':flags,'clean_prose':not flags,'page':q['page'],'section':q['section'],'bbox':q['bbox'],'pdf_url':p['pdf_url'],'pdf_sha256':p['pdf_sha256'],'source_path':str((SOURCE/p['pdf_path']).relative_to(ROOT)),'label_basis':p['human_label_basis'],'granularity':'historical_human_paragraph','regions':[{'start':0,'end':len(t),'label':0}],'overlaps_training_or_validation':leakage(t) if p['split']=='test' else None})
   if (i+1)%100==0:print('Extracted',i+1,'papers,',len(human),'remaining paragraphs',flush=True)
 if extraction_errors:
  save('extraction_errors.json',extraction_errors);raise RuntimeError('Resolve extraction errors before freezing suite')
 # Mark exact text duplicated across paper splits without discarding the available human pool.
 hs=collections.defaultdict(set)
 for r in human:hs[sha(norm(r['text']))].add(r['split'])
 for r in human:r['text_appears_in_multiple_splits']=len(hs[sha(norm(r['text']))])>1
 for split in ['train','validation','test']:writel(OUT/f'human_remaining_{split}.jsonl',[r for r in human if r['split']==split])
 print('Human extraction complete',len(human),flush=True)
 eval_rows=[];excluded=[];inventory={}
 for r in human:
  if r['split']!='test':continue
  reason='development_exposed' if r['development_exposed'] else 'training_validation_overlap' if r['overlaps_training_or_validation'] else 'cross_split_exact_duplicate' if r['text_appears_in_multiple_splits'] else None
  if reason:excluded.append({'id':r['id'],'reason':reason});continue
  assert r['paper_id'] not in forbidden_papers
  eval_rows.append(r)
 # Expand the prior four-per-stratum pilot to 32 per stratum. Full small benchmarks;
 # all OpAI trajectories, peer reviews and Epoch samples; 2,000 PELIC responses.
 limits={'meld_eval':32,'detectrl':32,'gede':32,'pelic':2000}
 for name,fn in published.ADAPTERS.items():
  rows,info=published.sample(fn(),limits.get(name,0));published.validate(rows);inventory[name]=info
  for r in rows:
   r={**r,'granularity':'document_native_label','source_adapter':name,'split':'external_test'}
   # Editing conventions disagree across benchmarks; retain their native label,
   # but don't claim these are known human-only or AI-only token labels.
   ambiguous=(r.get('task') in ['polish','edited'] or r['label']=='mixed' or (name=='detectrl' and 'human_test' in r.get('cohort','')))
   if r.get('ai_spans') is not None:
    spans=r['ai_spans'];regions=[];last=0
    for a,z in spans:
     if a>last:regions.append({'start':last,'end':a,'label':0})
     regions.append({'start':a,'end':z,'label':1});last=z
    if last<len(r['text']):regions.append({'start':last,'end':len(r['text']),'label':0})
    r['regions']=regions;r['granularity']='publisher_character_spans'
   elif ambiguous:r['granularity']='assisted_or_ambiguous_native_label'
   if leakage(r['text']):excluded.append({'id':r['id'],'reason':'training_validation_text_overlap'});continue
   eval_rows.append(r)
  print('Imported',name,len(rows),flush=True)
 # Existing local book/MDTA/interleaving proxies retained as diagnostics.
 for r in published.validate(published.local(32)):
  r={**r,'granularity':'document_native_label','split':'external_test'}
  if r.get('ai_spans') is not None:
   reg=[];last=0
   for a,z in r['ai_spans']:
    if a>last:reg.append({'start':last,'end':a,'label':0})
    reg.append({'start':a,'end':z,'label':1});last=z
   if last<len(r['text']):reg.append({'start':last,'end':len(r['text']),'label':0})
   r['regions']=reg;r['granularity']='synthetic_proxy_character_spans'
  if not leakage(r['text']):eval_rows.append(r)
  else:excluded.append({'id':r['id'],'reason':'training_validation_text_overlap'})
 arena=B/'runs/arena100-expanded-kimi/dataset.jsonl'
 for r in readl(arena):
  if not r['success'] or not r['mechanically_eligible']:
   excluded.append({'id':'arena:'+r['model_id']+':'+r['prompt_id'],'reason':'mechanical_generation_screen'});continue
  t=r['response_text'];row={'id':'arena:'+r['model_id']+':'+r['prompt_id'],'dataset':'arena50','cohort':'mechanically_eligible','text':t,'text_sha256':sha(t),'label':'ai','generator':r['model_id'],'group_id':r['prompt_id'],'domain':'arena_prompts','split':'external_test','granularity':'observed_generated_response','provenance':'Existing generation log; no new generation calls; whole emitted response labeled AI, including any copied wording.'}
  if leakage(t):excluded.append({'id':row['id'],'reason':'training_validation_text_overlap'});continue
  eval_rows.append(row)
 # Earlier paper pilots are explicitly exploratory, never pooled with clean holdout.
 dirs=['paper-pilot10-gpt61-sol-20260929','paper-luna50-20260929','paper-luna50-quality-v2-20260929','paper-gap50-luna-20260929','paper-gap50-luna-v2-20260929','paper-gap250-luna-v2-20260929','paper-gap250-luna-v4-20260930']
 for directory in dirs:
  path=ROOT/'research/data'/directory/'dataset.jsonl'
  if not path.exists():continue
  for r in readl(path):
   if r['paper_id'] not in byid or byid[r['paper_id']]['split']!='test':continue
   if r.get('operation')=='human_original':continue
   regions=[]
   for x in r['regions']:
    label=x['label'];label=label if isinstance(label,int) else (0 if label.startswith('human') else 1 if label.startswith('ai') else -100)
    regions.append({'start':x['start'],'end':x['end'],'label':label})
   row={'id':directory+':'+r['id'],'dataset':'paper_pilots_exploratory','cohort':directory,'text':r['text'],'text_sha256':sha(r['text']),'label':'mixed','paper_id':r['paper_id'],'group_id':r['paper_id'],'split':'test','generator':r.get('model'),'operation':r.get('operation'),'regions':regions,'granularity':'observed_edit_spans','development_exposed':True,'quality_flags':r.get('quality_flags',[])}
   if leakage(row['text']):excluded.append({'id':row['id'],'reason':'training_validation_text_overlap'});continue
   eval_rows.append(row)
 ids=[r['id'] for r in eval_rows]
 assert len(ids)==len(set(ids)), 'Duplicate IDs'
 for r in eval_rows:
  published.validate([r])
  r['text_sha256']=sha(r['text'])
  if 'regions' in r:
   assert all(0<=x['start']<x['end']<=len(r['text']) and x['label'] in [0,1,-100] for x in r['regions'])
 writel(OUT/'suite.jsonl',eval_rows);writel(OUT/'exclusions.jsonl',excluded)
 counts=collections.Counter(r['dataset'] for r in eval_rows)
 manifest={'version':'wide-eval-v1','created_at':time.time(),'checkpoint_sha256':'707a6f1f5ed491a351abee473c8e3f7beebe639a2534953926a4e0e6b576ff14','thresholds':json.loads((OUT.parent/'run-01/thresholds.json').read_text()),'frozen_model_and_thresholds':True,'new_generation_calls':0,'evaluation_rows':len(eval_rows),'unique_evaluation_texts':len({r['text_sha256'] for r in eval_rows}),'by_dataset':dict(counts),'source_sampling':inventory,'human_remaining_total':len(human),'human_remaining_by_split':dict(collections.Counter(r['split'] for r in human)),'human_by_split_cohort':dict(collections.Counter(r['split']+'/'+r['cohort'] for r in human)),'source_papers':len(papers),'target_paragraphs_excluded':target_exclusions,'extraction_errors':extraction_errors,'exclusions':dict(collections.Counter(r['reason'] for r in excluded)),'leakage_policy':'Paper splits preserved; fresh test paper IDs exclude all training/validation papers. All scored texts screened against actual training+validation examples for normalized exact paragraph/sentence matches and contiguous 32-word overlap. External author/source-paper identity generally unavailable; text screening cannot prove author-disjointness. Cross-split exact human duplicates excluded from scoring but retained with flags in human pools.','scope':'All extractable remaining body prose >=20 words from the established PDF extraction, excluding target paragraphs, headers, captions, small-font footnotes and material after reference/acknowledgment/appendix boundaries. Does not claim every physical paragraph is recovered. Modern OpenReview papers without human-authorship evidence are not labeled human.','metric_policy':'No threshold tuning or retraining. Report known span/historical-human metrics separately from document-native labels and ambiguous/polished cases. No pooled cross-benchmark headline score. Exact-text inference cached; counts and conflicts disclosed. Document classification uses >=50% of scored tokens above the frozen token threshold, an uncalibrated diagnostic rule.','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.jsonl')}}
 save('manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['evaluation_rows','unique_evaluation_texts','by_dataset','human_remaining_by_split','human_by_split_cohort','exclusions']},indent=2),flush=True)
if __name__=='__main__':main()
