"""Frozen, stratified historical-paper sampling for 7,500 new v3 reconstructions."""
import argparse, concurrent.futures, json, re, subprocess, unicodedata, time
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
import paper_gap250_v3 as v
b,g=v.b,v.gap
OUT=b.ROOT/'research/data/paper-gap10000-v3-luna-20260930'
BASE=b.ROOT/'research/data/paper-gap2500-v3-luna-20260930'
SEED='gap10000-v3-historical-balanced-20260930'
VOLUMES={2013:28,2014:32,2015:37,2016:48,2017:70,2018:80,2019:97,2020:119,2021:139}
g.OUT=OUT;g.SOURCE=OUT

def norm(s):return re.sub(r'\W+','',unicodedata.normalize('NFKC',s).casefold())
def rows(p):return b.readl(p) if p.exists() else []
def get(url,path):
 if path.exists():return path.read_bytes()
 with httpx.Client(timeout=90,follow_redirects=True) as client:
  for attempt in range(3):
   r=client.get(url)
   if r.status_code==429:time.sleep(30*(attempt+1));continue
   r.raise_for_status();path.write_bytes(r.content);return r.content
 raise RuntimeError('Source rate limit: '+url)

def catalog():
 (OUT/'sources').mkdir(parents=True,exist_ok=True);allrows=[]
 for year in range(2013,2022):
  for conf in ['NeurIPS','ICML','ACL']:
   url=f'https://proceedings.neurips.cc/paper/{year}' if conf=='NeurIPS' else f'https://proceedings.mlr.press/v{VOLUMES[year]}/' if conf=='ICML' else f'https://aclanthology.org/events/acl-{year}/'
   soup=BeautifulSoup(get(url,OUT/'sources'/f'index-{conf}-{year}.html'),'html.parser');found={}
   for a in soup.select('a[href]'):
    href=urljoin(url,a['href']);title=a.get_text(' ',strip=True)
    ok=('-Abstract.html' in href) if conf=='NeurIPS' else (href.endswith('.html') and f'/v{VOLUMES[year]}/' in href) if conf=='ICML' else bool(re.search(r'/(?:P(?:13|14|15|16|17|18|19)-[12]\d{3}|202[01]\.acl-(?:main|long)\.\d+)/$',href))
    if not ok:continue
    if conf=='ACL' and (re.search(r'-1000/$',href) or re.search(r'\.0/$',href)):continue
    pid=href.split('/')[-1].split('-')[0] if conf=='NeurIPS' else conf.lower()+'-'+str(year)+'-'+b.sha(href)[:16]
    found[href]={'paper_id':pid,'conference':conf,'year':year,'abstract_url':href,'index_url':url,'index_label':title}
   assert len(found)>=100,(conf,year,len(found))
   allrows.extend(found.values());print('Catalog',conf,year,len(found),flush=True)
 b.writel(OUT/'candidate-roster.jsonl',sorted(allrows,key=lambda x:(x['year'],x['conference'],b.sha(SEED+x['abstract_url']))))
 b.save(OUT/'sampling-plan.json',{'seed':SEED,'new_papers':1500,'new_paragraphs':7500,'final_papers':2000,'final_paragraphs':10000,'years':list(range(2013,2022)),'conferences':['NeurIPS','ICML','ACL'],'papers_per_conference':500,'split_per_conference':{'train':300,'validation':100,'test':100},'selection':'Seeded candidate order; interleave years and prefer underrepresented years within each conference. Equal conference totals; year allocation may vary with eligible source availability. Five clean body paragraphs per paper, no generation outcomes used. Existing papers and duplicate titles/PDFs excluded. Authors may recur within one split but cannot cross splits; papers linking existing different splits are excluded. Historical pilot author splits remain fixed.','human_provenance':'Official 2013–2021 proceedings and PDF creation/modification metadata before 2022, not direct proof of human-only authorship.','fresh_test_policy':'Preserve all existing split and development-exposure flags; new papers development_exposed=false.'})

def metadata(c):
 pid=c['paper_id'];soup=BeautifulSoup(get(c['abstract_url'],OUT/'sources'/f'{pid}.html'),'html.parser')
 meta=lambda n:[x.get('content','').strip() for x in soup.select(f'meta[name="{n}"]')]
 titles=meta('citation_title');title=titles[0] if titles else (soup.select_one('h1') or soup.select_one('h4')).get_text(' ',strip=True)
 authors=meta('citation_author')
 if not authors:authors=[a.get_text(' ',strip=True) for a in soup.select('a[href*="/author/"]')]
 if not authors:raise ValueError('Missing author metadata')
 links=meta('citation_pdf_url')
 if not links:links=[urljoin(c['abstract_url'],a['href']) for a in soup.select('a[href]') if a['href'].endswith('-Paper.pdf')]
 if not links:raise ValueError('No main-paper PDF')
 pdf=links[0].replace('http://','https://')
 abstract=soup.select_one('#abstract, .paper-abstract, .acl-abstract')
 if not abstract:
  head=next((h for h in soup.find_all(['h2','h3','h4','h5']) if h.get_text(' ',strip=True).lower()=='abstract'),None)
  abstract=head.find_next('p') if head else None
 if not abstract:
  # Older ACL landing pages omit abstracts; read the paper's labeled Abstract section.
  pdfpath=OUT/'sources'/f'{pid}.pdf';get(pdf,pdfpath)
  blocks=g.extract({**c,'pdf_path':str(pdfpath.relative_to(OUT))})
  parts=[];abstract_x=None
  for block in blocks:
   if block['section'].strip().lower()!='abstract' or block['page']!=1:continue
   if re.match(r'(Figure|Table|Algorithm)\b',block['text']):break
   if abstract_x is None:abstract_x=block['bbox'][0]
   if abs(block['bbox'][0]-abstract_x)>30:break
   parts.append(block['text'])
  abstract=' '.join(parts)
  if not abstract or len(abstract.split())>500:raise ValueError('No reliable labeled PDF abstract')
 else:abstract=abstract.get_text(' ',strip=True)
 abstract=re.sub(r'^Abstract\s*','',abstract)
 if len(abstract.split())<35:raise ValueError('Short/invalid abstract')
 return {**c,'title':title,'authors':authors,'pdf_url':pdf,'abstract':abstract}

def extract_candidate(c, minimum_targets=5, target_predicate=None, context_predicate=None):
 try:
  pid=c['paper_id'];pdfpath=OUT/'sources'/f'{pid}.pdf';get(c['pdf_url'],pdfpath)
  reader=PdfReader(pdfpath);md={str(k):str(val.get_object() if hasattr(val,'get_object') else val) for k,val in (reader.metadata or {}).items()}
  years=[int(md[k][2:6]) for k in ['/CreationDate','/ModDate'] if re.match(r'D:\d{4}',md.get(k,''))]
  if not years or max(years)>=2022:raise ValueError('PDF metadata does not establish pre-2022 version')
  c={**c,'pdf_path':str(pdfpath.relative_to(OUT)),'pdf_metadata':md,'pdf_sha256':b.sha(pdfpath.read_bytes()),'page_count':len(reader.pages),'downloaded_utc':b.now(),'human_label_basis':f'Official {c["conference"]} {c["year"]} proceedings PDF; creation/modification metadata before 2022; historical provenance, not observed writing logs'}
  paragraphs=v.prior.sources.paragraphs(c);triples=[]
  for i,p in enumerate(paragraphs):
   if i==0 or i==len(paragraphs)-1 or not v.prior.sources.target(p):continue
   if target_predicate is not None and not target_predicate(p):continue
   a,z=paragraphs[i-1],paragraphs[i+1]
   if context_predicate is not None and not all(context_predicate(q) for q in [a,z]):continue
   if not all(20<=len(q['text'].split())<=600 for q in [a,z]):continue
   if z['page']-a['page']>2:continue
   # Prevent two-column extraction joins, captions, and visibly incomplete neighbors.
   if not all(re.match(r'[A-Z“]',q['text']) and re.search(r'[.!?][\]”\"]?$',q['text']) for q in [a,z]):continue
   if any(re.match(r'(Figure|Table|Algorithm|Theorem|Lemma|Proof|Proposition|Corollary)\b',q['text']) for q in [a,z]):continue
   context=json.dumps([c['abstract'],a['text'],z['text']],ensure_ascii=False)
   if p['text'] in context or any(x['text'] in context for x in b.sentence_spans(p['text']) if len(x['text'].split())>=15):continue
   triples.append((a,p,z))
  if len(triples)<minimum_targets:raise ValueError(f'Only {len(triples)} eligible targets with intact prose neighbors')
  chosen=[];selected=[]
  for ordinal in range(1,minimum_targets+1):
   def score(triple):
    a,t,z=triple;txt=t['text'];symbols=sum(unicodedata.category(ch) in {'Sm','So'} or 'GREEK' in unicodedata.name(ch,'') for ch in txt)
    return (symbols,sum(not g.clean(q) for q in [a,z]),any(t['section']==x['section'] for x in chosen),-min([min(abs(t['page']-x['page']),3) for x in chosen] or [0]),b.sha(SEED+pid+txt))
   available=[x for x in triples if all(x[1]['text']!=p['text'] for p in chosen)]
   if not available:raise ValueError('Insufficient distinct targets')
   a,t,z=min(available,key=score);chosen.append(t);text=a['text']+'\n\n'+t['text']+'\n\n'+z['text'];start=len(a['text'])+2
   selected.append({'passage_id':pid+f'/g{ordinal:02d}','paper_id':pid,'title':c['title'],'page':t['page'],'section':'Body paragraph','extracted_section_hint':t['section'],'abstract':c['abstract'],'before':a['text'],'held_out':t['text'],'after':z['text'],'source_paragraphs':[a,t,z],'text':text,'text_sha256':b.sha(text),'held_out_start':start,'held_out_end':start+len(t['text']),'extraction':'Same Poppler bbox normalization and prose filtering as v3; intact sentence boundaries required in immediate neighbors','selection_note':'40–320-word prose target; no more than one infinity/Pi symbol; deterministic seeded source selection; no model outcome filtering'})
  return c,selected,None
 except Exception as e:return c,[],type(e).__name__+': '+str(e)[:300]

def select():
 roster=rows(OUT/'candidate-roster.jsonl');accepted=rows(OUT/'new-papers.jsonl');passages=rows(OUT/'new-passages.jsonl');excluded=rows(OUT/'source-exclusions.jsonl')
 old=rows(BASE/'papers.jsonl')
 seen_ids={p['paper_id'] for p in old+accepted};seen_titles={norm(p['title']) for p in old+accepted};seen_pdfs={p['pdf_sha256'] for p in old+accepted}
 authors={}
 for p in old+accepted:
  for author in p['authors']:
   key=norm(author);assert key not in authors or authors[key]==p['split'];authors[key]=p['split']
 attempted={x['abstract_url'] for x in excluded+accepted};counts=Counter((p['conference'],p['year']) for p in accepted);splits=Counter((p['conference'],p['split']) for p in accepted)
 cells=sorted({(c['conference'],c['year']) for c in roster},key=lambda x:b.sha(SEED+str(x)))
 queues={cell:[c for c in roster if (c['conference'],c['year'])==cell and c['abstract_url'] not in attempted] for cell in cells}
 quotas={'train':300,'validation':100,'test':100}
 def reject(c,reason):
  row={**c,'reason':reason};excluded.append(row);b.transport.append(OUT/'source-exclusions.jsonl',row)
 def assignment(c):
  linked={authors[norm(a)] for a in c['authors'] if norm(a) in authors}
  if len(linked)>1:return None
  available=[s for s in quotas if splits[c['conference'],s]<quotas[s] and (not linked or s in linked)]
  return min(available,key=lambda s:(splits[c['conference'],s]/quotas[s],b.sha(SEED+c['paper_id']+s))) if available else None
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as network, concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  while len(accepted)<1500:
   active=[cell for cell in cells if queues[cell] and sum(splits[cell[0],s] for s in quotas)<500]
   if not active:raise RuntimeError('Candidate roster exhausted before requested totals: '+str(splits))
   # One candidate per venue/year per round; do not let faster network responses affect selection.
   active.sort(key=lambda cell:(counts[cell],b.sha(SEED+str(cell))))
   batch=[(cell,queues[cell].pop(0)) for cell in active]
   metas=[network.submit(metadata,c) if c['paper_id'] not in seen_ids else None for cell,c in batch]
   pending=[]
   for (cell,c),future in zip(batch,metas):
    if future is None:reject(c,'Previously used paper');continue
    try:c=future.result()
    except Exception as e:reject(c,type(e).__name__+': '+str(e)[:200]);continue
    if norm(c['title']) in seen_titles:reject(c,'Duplicate normalized title');continue
    if assignment(c) is None:reject(c,'Author links multiple splits or compatible split quota full');continue
    pending.append((cell,c,pool.submit(extract_candidate,c)))
   for cell,c,future in pending:
    paper,selected,error=future.result()
    if error:reject(c,error);continue
    if paper['pdf_sha256'] in seen_pdfs:reject(c,'Duplicate PDF');continue
    split=assignment(c)
    if split is None:reject(c,'Author links multiple splits or compatible split quota full');continue
    paper.update(split=split,eligible_prose_blocks=len(selected),sampling_cell=str(cell),development_exposed=False);paper.pop('abstract',None)
    for p in selected:p.update(split=split,development_exposed=False)
    accepted.append(paper);passages+=selected;counts[cell]+=1;splits[cell[0],split]+=1
    seen_ids.add(paper['paper_id']);seen_titles.add(norm(paper['title']));seen_pdfs.add(paper['pdf_sha256'])
    for author in paper['authors']:authors[norm(author)]=split
    b.transport.append(OUT/'new-papers.jsonl',paper)
    for p in selected:b.transport.append(OUT/'new-passages.jsonl',p)
    print(f'Selected {len(accepted)}/1500 | {cell} {counts[cell]} | {split} | {paper["title"]}',flush=True)
   b.save(OUT/'acquisition-status.json',{'at_utc':b.now(),'papers':len(accepted),'paragraphs':len(passages),'strata':{str(k):v for k,v in counts.items()},'splits':{str(k):v for k,v in splits.items()},'excluded':len(excluded)})
 assert len(accepted)==1500 and len(passages)==7500
 assert Counter(p['split'] for p in accepted)=={'train':900,'validation':300,'test':300}
 b.save(OUT/'source-selection-summary.json',{'selected_papers':1500,'selected_paragraphs':7500,'strata':[{'conference':c,'year':y,'papers':counts[c,y]} for c,y in sorted(cells)],'exclusions':dict(Counter(x['reason'] for x in excluded)),'split':dict(Counter(p['split'] for p in accepted)),'validated_author_disjoint_across_splits':True})
 print('Source selection complete.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['catalog','select']);a=p.parse_args();globals()[a.action]()
