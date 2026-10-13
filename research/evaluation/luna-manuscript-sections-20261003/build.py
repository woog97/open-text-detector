import collections,hashlib,json,re,random
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def read(p):return [json.loads(l) for l in p.open()]
old= P.parent/'luna-flex-hillclimb-20261003'; mini=json.loads((P.parent/'luna-mini-suite-20261003/dataset.json').read_text())
excluded=set(json.loads((old/'development-exposure.json').read_text())['rows'][i]['paper_id'] for i in range(216))
for f in ['fewshot-examples.json','fewshot-diverse-examples.json']:excluded.update(r['paper_id'] for r in json.loads((old/f).read_text()))
excluded.update(r['paper_id'] for r in mini if r['profile']!='manuscripts')
source=ROOT/'research/data/paper-eval-workflows-luna-20260930';papers=read(source/'papers.jsonl');body=read(source/'human-body.jsonl');groups=collections.defaultdict(list)
for r in body:groups[r['paper_id']].append(r)
selected=[]
for venue in ['ACL','ICML','NeurIPS']:
 candidates=[r for r in papers if r['conference'].lower()==venue.lower() and r['paper_id'] not in excluded and r['split']=='calibration' and sum(len(x['text'].split()) for x in groups[r['paper_id']])>=1500]
 selected.extend(sorted(candidates,key=lambda r:sha('manuscript-human-v1/'+r['paper_id']))[:4])
assert len(selected)==12
manuscripts=[]
for r in selected:
 blocks=sorted(groups[r['paper_id']],key=lambda x:x['id']);kept=[];excluded_blocks=[]
 for i,b in enumerate(blocks):
  if (i==0 and str(b['section']).lower()=='abstract') or re.match(r'^(references|bibliography|acknowledg)',str(b['section']),re.I):excluded_blocks.append(b['id']);continue
  kept.append(b['text'])
 manuscripts.append({'paper_id':'human/'+r['paper_id'],'label':'HUMAN','generator':None,'source_id':r['paper_id'],'conference':r['conference'],'year':r['year'],'source_url':r['pdf_url'],'blocks':kept,'excluded_blocks':excluded_blocks,'provenance':'Historical public proceedings paper; existing extracted body-prose blocks, not a full PDF transcription.'})
for r in mini:
 if r['profile']!='manuscripts':continue
 lines=[];inbody=False
 for line in r['text'].splitlines():
  h=re.match(r'^#{1,6}\s+(.+)',line)
  if h:
   title=h.group(1).strip();plain=re.sub(r'^\d+(?:\.\d+)*[.)]?\s*','',title).lower()
   if plain.startswith(('references','bibliography','acknowledg')):break
   if plain=='abstract':inbody=False;continue
   if line.startswith('## ') and plain!='abstract':inbody=True
   if inbody:lines.append('')
   continue
  if inbody:lines.append(line)
 blocks=[b.strip() for b in re.split(r'\n\s*\n','\n'.join(lines)) if b.strip()]
 assert len(' '.join(blocks).split())>1000,r['id']
 manuscripts.append({'paper_id':r['id'],'label':'AI','generator':r['generator'],'source_id':r['paper_id'],'blocks':blocks,'provenance':'Generated body only; title/abstract and reference tail excluded; prior mini-suite full-document exposure retained.'})
assert len(manuscripts)==24
rows=[]
for paper in manuscripts:
 parts=[];current=[];count=0
 for block in paper['blocks']:
  words=block.split()
  while len(words)>600:
   if current:parts.append(' '.join(current));current=[];count=0
   parts.append(' '.join(words[:500]));words=words[500:]
  if count+len(words)>600 and current:parts.append(' '.join(current));current=[];count=0
  current.append(' '.join(words));count+=len(words)
  if count>=400:parts.append(' '.join(current));current=[];count=0
 if current:
  tail=' '.join(current)
  if parts and len(tail.split())<100 and len(parts[-1].split())+len(tail.split())<=700:parts[-1]+=' '+tail
  else:parts.append(tail)
 assert ' '.join(parts).split()==' '.join(paper['blocks']).split()
 paper['section_count']=len(parts);paper['body_words']=sum(len(x.split()) for x in parts)
 for i,text in enumerate(parts):rows.append({'id':paper['paper_id']+'/chunk%03d'%i,'paper_id':paper['paper_id'],'source_id':paper['source_id'],'label':paper['label'],'generator':paper['generator'],'text':text,'text_sha256':sha(text),'words':len(text.split()),'chunk_index':i,'split':'manuscript_section_evaluation'})
random.Random(20261003).shuffle(rows)
(P/'dataset.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(P/'manuscripts.json').write_text(json.dumps(manuscripts,ensure_ascii=False,indent=2)+'\n')
meta={'papers':24,'human_papers':12,'ai_papers':12,'sections':len(rows),'section_labels':dict(collections.Counter(r['label'] for r in rows)),'words_by_label':dict(collections.Counter({label:sum(r['words'] for r in rows if r['label']==label) for label in ['HUMAN','AI']})),'protocol':'Paragraph-aligned section-sized chunks, target400 words, maximum600 except merged short tail up to700; oversized blocks split into500-word pieces. Whitespace normalized identically for both classes. No chunk overlap or whole-paper context. Human extracted body prose excludes first abstract block and reference-tagged blocks; AI markdown body excludes title/abstract/reference tail and heading markers. These are chunks, not verified semantic sections. All kept body words covered exactly once.','limitations':'Historical human PDFs versus generated Markdown differ in extraction and formatting; not generator- or topic-matched. AI full papers were previously scored in the mini-suite; this is a new section-level evaluation, not entirely unseen manuscripts.'}
(P/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
