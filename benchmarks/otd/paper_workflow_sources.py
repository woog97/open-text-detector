"""Fresh author-separated source acquisition for the evaluation-only workflow suite."""
import concurrent.futures, json, shutil, re, fcntl
from collections import Counter
import paper_scale10000_acquire as src
b,g=src.b,src.g
BASE=src.OUT
OUT=b.ROOT/'research/data/paper-eval-workflows-luna-20260930'
SEED='paper-workflow-eval-20260930-v1'
SLOTS=['pilot','calibration','calibration','test','test','test','test']
original_get=src.get

def configure():
 src.OUT=OUT;src.SEED=SEED;g.OUT=OUT;g.SOURCE=OUT
 (OUT/'sources').mkdir(parents=True,exist_ok=True)
 def cached_get(url,path):
  old=BASE/'sources'/path.name
  if not path.exists() and old.exists():shutil.copyfile(old,path)
  return original_get(url,path)
 src.get=cached_get

def sentence_spans(text):
 # Also recognize starts with parentheses, quotation marks and numbered citations.
 breaks=[0]
 for match in re.finditer(r'[.!?][\]\)"”’]*\s+(?=(?:[A-Z]|[\(“"‘][A-Z]|\[\d))',text):
  preceding=text[max(0,match.start()-12):match.start()+1]
  if re.search(r'(?:\bet al|\bFig|\bEq|\bSec|\bDr|\bMr|\bvs|\be\.g|\bi\.e)\.$',preceding):continue
  breaks.append(match.end())
 breaks.append(len(text));result=[]
 for left,right in zip(breaks,breaks[1:]):
  while left<right and text[left].isspace():left+=1
  while right>left and text[right-1].isspace():right-=1
  if right>left:result.append({'start':left,'end':right,'text':text[left:right]})
 return result

def source_flags(text):
 flags=[]
 if re.search(r'\b[a-z]{2,}-\s',text):flags.append('broken_hyphenated_word')
 if re.search(r'\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b',text):flags.append('flattened_mathematics')
 if re.search(r'\b(?:\d+(?:\.\d+)?\s+){3,}\d+(?:\.\d+)?\b',text):flags.append('embedded_numeric_table')
 return flags

def valid_target(p):
 if source_flags(p['text']):return False
 spans=sentence_spans(p['text'])
 return len(spans)>=4 and any(len(s['text'].split())>=15 for s in spans[1:-1])

def acquire():
 configure()
 lock=(OUT/'source-worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=src.rows(BASE/'papers.jsonl');accepted=src.rows(OUT/'papers.jsonl')
 roster=src.rows(BASE/'candidate-roster.jsonl')
 oldauthors={src.norm(a) for p in old for a in p['authors']}
 usedids={p['paper_id'] for p in old+accepted};titles={src.norm(p['title']) for p in old+accepted};pdfs={p['pdf_sha256'] for p in old+accepted}
 authors={src.norm(a):p['split'] for p in accepted for a in p['authors']}
 counts=Counter((p['conference'],p['year']) for p in accepted)
 splitcounts=Counter((p['conference'],p['year'],p['split']) for p in accepted)
 pilot_exposed={src.norm(a) for p in src.rows(OUT/'pilot-v1'/'papers.jsonl') if p['split']=='pilot' for a in p['authors']}
 def assignment(cell):return next(s for s,quota in [('pilot',1),('calibration',2),('test',4)] if splitcounts[cell[0],cell[1],s]<quota)
 excluded=src.rows(OUT/'source-exclusions.jsonl');attempted={p['paper_id'] for p in accepted+excluded}
 cells=sorted({(p['conference'],p['year']) for p in roster})
 queues={cell:sorted([p for p in roster if (p['conference'],p['year'])==cell and p['paper_id'] not in attempted and p['paper_id'] not in usedids],key=lambda p:b.sha(SEED+p['paper_id'])) for cell in cells}
 def reject(p,reason):b.transport.append(OUT/'source-exclusions.jsonl',{'paper_id':p['paper_id'],'conference':p['conference'],'year':p['year'],'reason':reason})
 def linked(p,split):return any(src.norm(a) in oldauthors or (split!='pilot' and src.norm(a) in pilot_exposed) or (src.norm(a) in authors and authors[src.norm(a)]!=split) for a in p['authors'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  while True:
   active=[cell for cell in cells if counts[cell]<len(SLOTS) and queues[cell]]
   if not active:break
   batch=[(cell,queues[cell].pop(0)) for cell in active]
   futures=[pool.submit(src.metadata,p) for cell,p in batch]
   eligible=[]
   for (cell,p),future in zip(batch,futures):
    try:
     p=future.result();split=assignment(cell)
     if linked(p,split):reject(p,'Previously exposed author or cross-split author');continue
     if src.norm(p['title']) in titles:reject(p,'Duplicate title');continue
     eligible.append((cell,p,pool.submit(src.extract_candidate,p,1,valid_target,lambda q:not source_flags(q['text']))))
    except Exception as e:reject(p,type(e).__name__+': '+str(e)[:180])
   for cell,p,future in eligible:
    paper,passages,error=future.result();split=assignment(cell)
    if error:reject(p,error);continue
    if linked(paper,split) or paper['pdf_sha256'] in pdfs:reject(p,'Author conflict or duplicate PDF');continue
    candidates=[q for q in passages if len(b.sentence_spans(q['held_out']))>=4]
    if not candidates:reject(p,'No selected target with four sentence spans');continue
    q=min(candidates,key=lambda q:b.sha(SEED+q['passage_id']))
    paper.update(split=split,forum_id=paper.get('forum_id'),development_exposed=split=='pilot')
    q.update(split=split,forum_id=paper['forum_id'],development_exposed=split=='pilot')
    # Body inventory is frozen before any model outputs are available.
    for i,para in enumerate(src.v.prior.sources.paragraphs(paper)):
     clean=bool(src.v.prior.sources.target(para)) and not source_flags(para['text']);overlap=para['text'] in [q['held_out'],q['before'],q['after']]
     b.transport.append(OUT/'human-body.jsonl',{'id':paper['paper_id']+f'/body{i:04d}','paper_id':paper['paper_id'],'forum_id':paper['forum_id'],'split':split,'text':para['text'],'page':para['page'],'section':para['section'],'clean_prose':clean,'source_quality_flags':source_flags(para['text']),'target_or_context':overlap,'human_label_basis':paper['human_label_basis'],'pdf_sha256':paper['pdf_sha256'],'label':'human_historical'})
    b.transport.append(OUT/'passages.jsonl',q);b.transport.append(OUT/'papers.jsonl',paper)
    accepted.append(paper);counts[cell]+=1;splitcounts[cell[0],cell[1],split]+=1;titles.add(src.norm(paper['title']));pdfs.add(paper['pdf_sha256'])
    for a in paper['authors']:authors[src.norm(a)]=split
    print(json.dumps({'selected':len(accepted),'cell':cell,'split':split,'paper_id':paper['paper_id']}),flush=True)
   b.save(OUT/'acquisition-status.json',{'at_utc':b.now(),'papers':len(accepted),'planned':189,'strata':{str(c):counts[c] for c in cells},'splits':dict(Counter(p['split'] for p in accepted))})
 b.save(OUT/'source-selection-summary.json',{'papers':len(accepted),'complete':len(accepted)==189,'shortfalls':{str(c):7-counts[c] for c in cells if counts[c]<7},'old_papers_excluded':len(old),'old_author_names_excluded':len(oldauthors),'author_disjoint_from_old':True,'author_disjoint_across_new_splits':True,'eligibility':'One clean extractable triple per paper; target has at least four heuristic sentences and an internal sentence of at least 15 words','seed':SEED})
 if len(accepted)!=189:raise RuntimeError('Source stratum exhausted; see shortfalls, no independence criteria relaxed')
if __name__=='__main__':acquire()
