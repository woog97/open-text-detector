"""Replace three unwritten targets whose original sentences leak through source context."""
import json,re,unicodedata,shutil
import paper_scale2500_run as run
s,b,g,v=run.src,run.b,run.g,run.v
OUT=s.OUT
ps=run.rows(OUT/'passages.jsonl');papers={p['paper_id']:p for p in run.rows(OUT/'papers.jsonl')}
writers={w['passage_id'] for w in run.rows(OUT/'writer-responses.jsonl')}
repairs=[];replace_papers=set()
for p in ps:
 if p['passage_id'] in writers:continue
 context=json.dumps([p['abstract'],p['before'],p['after']],ensure_ascii=False)
 if not any(x['text'] in context for x in b.sentence_spans(p['held_out']) if len(x['text'].split())>=15):continue
 candidates=v.prior.sources.paragraphs(papers[p['paper_id']]);others=[q for q in ps if q['paper_id']==p['paper_id'] and q is not p];eligible=[]
 for i,t in enumerate(candidates):
  if i==0 or i==len(candidates)-1 or not v.prior.sources.target(t):continue
  a,z=candidates[i-1],candidates[i+1]
  if not all(20<=len(q['text'].split())<=600 for q in [a,z]) or z['page']-a['page']>2:continue
  if not all(re.match(r'[A-Z“]',q['text']) and re.search(r'[.!?][\]”\"]?$',q['text']) for q in [a,z]):continue
  if any(re.match(r'(Figure|Table|Algorithm|Theorem|Lemma|Proof|Proposition|Corollary)\b',q['text']) for q in [a,z]):continue
  if any(t['text']==q['held_out'] for q in others) or t['text']==p['held_out']:continue
  context=json.dumps([p['abstract'],a['text'],z['text']],ensure_ascii=False)
  if t['text'] in context or any(x['text'] in context for x in b.sentence_spans(t['text']) if len(x['text'].split())>=15):continue
  eligible.append((a,t,z))
 def score(triple):
  a,t,z=triple;symbols=sum(unicodedata.category(ch) in {'Sm','So'} or 'GREEK' in unicodedata.name(ch,'') for ch in t['text'])
  return (symbols,sum(not g.clean(q) for q in [a,z]),any(t['section']==q['extracted_section_hint'] for q in others),-min([min(abs(t['page']-q['page']),3) for q in others] or [0]),b.sha(s.SEED+p['paper_id']+t['text']))
 if not eligible:
  replace_papers.add(p['paper_id']);continue
 a,t,z=min(eligible,key=score);before=dict(p);text=a['text']+'\n\n'+t['text']+'\n\n'+z['text'];start=len(a['text'])+2
 p.update(page=t['page'],extracted_section_hint=t['section'],before=a['text'],held_out=t['text'],after=z['text'],source_paragraphs=[a,t,z],text=text,text_sha256=b.sha(text),held_out_start=start,held_out_end=start+len(t['text']),source_revision=2)
 repairs.append({'passage_id':p['passage_id'],'reason':'Original sentence of at least 15 words repeats verbatim in abstract or immediate neighbor. Replaced before any writer call; no generation outcome used.','original_source':before,'replacement_source':dict(p)})
assert len(repairs)==2 and len(replace_papers)==1
old_paper=papers[next(iter(replace_papers))]
used=list(papers.values())
for path in (b.ROOT/'research/data').glob('paper-*/papers.jsonl'):
 if path.parent!=OUT:used+=run.rows(path)
seen_ids={q['paper_id'] for q in used};seen_authors={s.norm(a) for q in used for a in q['authors']};seen_titles={s.norm(q['title']) for q in used};seen_pdfs={q['pdf_sha256'] for q in used}
roster=[c for c in run.rows(OUT/'candidate-roster.jsonl') if c['conference']==old_paper['conference'] and c['year']==old_paper['year']]
for c in sorted(roster,key=lambda c:b.sha(s.SEED+c['abstract_url'])):
 if c['paper_id'] in seen_ids:continue
 c=s.metadata(c)
 if s.norm(c['title']) in seen_titles or {s.norm(a) for a in c['authors']} & seen_authors:continue
 paper,selected,error=s.extract_candidate(c)
 if error or paper['pdf_sha256'] in seen_pdfs:continue
 paper.update(split=old_paper['split'],eligible_prose_blocks=5,sampling_cell=old_paper['sampling_cell']);paper.pop('abstract',None)
 for q in selected:q.update(split=old_paper['split'],development_exposed=False)
 break
else:raise RuntimeError('No eligible replacement paper')
retired_ps=[q for q in ps if q['paper_id']==old_paper['paper_id']]
repairs.append({'reason':'Only four eligible targets remain after preventing original-sentence leakage; replace paper before evaluating outputs, using next seeded eligible paper in same venue/year and same split. No generation outcome filtering.','original_paper':old_paper,'replacement_paper':paper,'original_sources':retired_ps,'replacement_sources':selected})
ps=[q for q in ps if q['paper_id']!=old_paper['paper_id']]+selected
old_papers=list(papers.values());new_papers=[q for q in old_papers if q['paper_id']!=old_paper['paper_id']]+[paper]
assert not (OUT/'source-corrections.json').exists()
for name in ['papers.jsonl','new-papers.jsonl','passages.jsonl','new-passages.jsonl','outline-responses.jsonl','writer-responses.jsonl','manifest.json']:
 shutil.copyfile(OUT/name,OUT/('pre-source-correction-'+name))
ids={r['passage_id'] for r in repairs if 'passage_id' in r};retired_ids={q['passage_id'] for q in retired_ps}
retired_writers=[w for w in run.rows(OUT/'writer-responses.jsonl') if w['passage_id'] in retired_ids]
b.writel(OUT/'retired-source-writers.jsonl',retired_writers)
b.writel(OUT/'writer-responses.jsonl',[w for w in run.rows(OUT/'writer-responses.jsonl') if w['passage_id'] not in retired_ids])
notes=run.rows(OUT/'outline-responses.jsonl')
b.writel(OUT/'retired-source-outlines.jsonl',[r for r in notes if r['passage_id'] in ids|retired_ids])
b.writel(OUT/'outline-responses.jsonl',[r for r in notes if r['passage_id'] not in ids|retired_ids])
attempts=run.rows(OUT/'attempts.jsonl')
for a in attempts:
 if a['passage_id'] in ids:
  assert a['stage']=='outline'
  a['retired_source_revision']=1
b.writel(OUT/'retired-source-attempts.jsonl',[a for a in attempts if a['passage_id'] in retired_ids])
b.writel(OUT/'attempts.jsonl',[a for a in attempts if a['passage_id'] not in retired_ids])
b.writel(OUT/'papers.jsonl',new_papers);b.writel(OUT/'new-papers.jsonl',[q for q in new_papers if not q.get('development_exposed')])
b.writel(OUT/'passages.jsonl',ps);b.writel(OUT/'new-passages.jsonl',[p for p in ps if not p['development_exposed']])
b.save(OUT/'source-corrections.json',repairs)
m=run.v.read(OUT/'manifest.json');m['papers_sha256']=b.sha((OUT/'papers.jsonl').read_bytes());m['passages_sha256']=b.sha((OUT/'passages.jsonl').read_bytes());m['source_corrections']={'source_leak_cases':3,'replacement_papers':1,'replacement_targets_within_paper':2,'record':'source-corrections.json','reason':'Pre-writer source-context leakage; retain all retired note costs and requests. Same venue/year quotas, splits, source eligibility and prompts.'};b.save(OUT/'manifest.json',m)
print('Replaced two unwritten leaking targets, plus one paper lacking five non-leaking targets. All original sources, outputs and costs retained; quotas and splits unchanged.')
