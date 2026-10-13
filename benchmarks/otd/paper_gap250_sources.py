"""Body-prose candidate inventory with PDF footnotes and captions excluded."""
import sys,re,unicodedata,json,statistics
import paper_gap50 as g
from bs4 import BeautifulSoup

def paragraphs(paper):
 raw=g.extract(paper)
 xml=BeautifulSoup((g.OUT/'sources'/f"{paper['paper_id']}.bbox.html").read_text(),'xml')
 pages=xml.find_all('page'); out=[]
 for idx,p in enumerate(raw):
  t=p['text'];box=p['bbox'];ws=pages[p['page']-1].find_all('word');hs=[float(w['yMax'])-float(w['yMin']) for w in ws if float(w['yMin'])>=box[1]-.1 and float(w['yMax'])<=box[3]+.1 and float(w['xMin'])>=box[0]-.1 and float(w['xMax'])<=box[2]+.1]
  height=statistics.median(hs) if hs else 9
  if height<8.8 or re.match(r'(?:Figure|Table|Algorithm)\s*\d',t) or re.match(r'^\d+\s',t):continue
  if len(t.split())<20 or p['section']=='Front matter':continue
  if sum(c.isalpha() or c.isspace() for c in t)/len(t)<.65:continue
  out.append({**p,'raw_index':idx,'median_word_height':height})
 merged=[]
 for p in out:
  if merged and p['page']==merged[-1]['page']+1 and re.match(r'[a-z]',p['text']) and not re.match(r'(?:[ivx]+\)|[a-z]\))\s',p['text']) and (not re.search(r'[.!?]$',merged[-1]['text']) or re.search(r'(?:e\.g\.|i\.e\.)$',merged[-1]['text'])):
   merged[-1]['text']+=' '+p['text'];merged[-1]['continued_on_page']=p['page'];merged[-1]['continued_raw_index']=p['raw_index']
  else:merged.append(p)
 return merged

def target(p):
 t=p['text'];symbols=[c for c in t if unicodedata.category(c) in {'Sm','So'} or 'GREEK' in unicodedata.name(c,'')]
 return (p['section']!='Abstract' and 40<=len(t.split())<=320 and re.match(r'[A-Z“]',t) and re.search(r'[.!?][\]”\"]?$',t) and not re.search(r'(?:e\.g\.|i\.e\.|et al\.)$',t) and not re.match(r'(Figure|Table|Algorithm|Theorem|Lemma|Proof|Proposition|Corollary|Definition|Remark|Assumption|Left:|Right:)\b',t) and all(s in '∞Π' for s in symbols) and len(symbols)<=1 and not any(s in t for s in [' = ','@','http','Conference on Neural Information','`','⇢']) and sum(c.isalpha() or c.isspace() for c in t)/len(t)>.85)
