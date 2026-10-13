import json,sqlite3,collections,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
papers=json.loads((OUT/'live-papers.json').read_text())['items']
notes={};rawfields=collections.Counter()
for f in (ROOT/'research/data/openreview_catalogue_all/raw').glob('*.json'):
 for n in json.loads(f.read_text()).get('notes',[]):
  if not isinstance(n,dict):continue
  rawfields.update(n.get('content',{}).keys())
  old=notes.get(n.get('id'))
  if old is None or len(n.get('content',{}))>len(old.get('content',{})):notes[n.get('id')]=n
c=sqlite3.connect(f'file:{ROOT}/research/data/reviewbench/catalogue.sqlite3?mode=ro',uri=True)
abstracts={id:a for id,a in c.execute('select id,abstract from papers')};c.close()
stats=collections.defaultdict(collections.Counter);unmatched=[]
def field(note,k):
 if 'everyone' not in note.get('readers',[]):return None
 v=note.get('content',{}).get(k)
 if isinstance(v,dict):return v.get('value') if ('readers' not in v or 'everyone' in v['readers']) else None
 return v
for p in papers:
 d=stats[p['collection']];d['papers']+=1
 for k in ['tldr','abstract_preview','keywords','primary_area']:d['live_'+k]+=bool(p.get(k))
 d['live_summary']+=bool(p.get('tldr') or p.get('abstract_preview'))
 fid=p.get('forum_id') or p['filename'].removesuffix('.pdf');conf=p['collection'].split('/')[0]
 a=abstracts.get(conf+':'+fid);note=notes.get(fid,{})
 d['saved_abstract']+=bool(a);d['raw_note_match']+=bool(note)
 for k in ['abstract','TLDR','keywords','primary_area']:d['raw_public_'+k]+=bool(field(note,k))
 d['available_abstract']+=bool(a or field(note,'abstract') or p.get('abstract_preview'))
 if not a and p['collection']!='iclr/2027':unmatched.append({'id':p['id'],'forum_id':fid,'collection':p['collection'],'title':p['title']})
rows=[dict(collection=k,**v) for k,v in sorted(stats.items())]
(OUT/'coverage.json').write_text(json.dumps({'rows':rows,'raw_fields':rawfields,'unmatched':unmatched},indent=2))
with (OUT/'coverage.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('raw fields',rawfields)
for r in rows:print(r['collection'],r['papers'],'savedAbstract',r['saved_abstract'],'publicRawAbstract',r['raw_public_abstract'],'tldr',r['raw_public_TLDR'],'tags',r['raw_public_keywords'])
