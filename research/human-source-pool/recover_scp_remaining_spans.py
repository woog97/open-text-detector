"""Remote append-only SCP recovery from unused spans of verified cached originals."""
import sys,json,gzip,sqlite3,hashlib,re
from pathlib import Path
from collections import Counter
BASE=Path('/tmp/pangram-scp-cccc-staging-20261002')
OUT=Path('/tmp/pangram-scp-spans-v2-20261002')
sys.path.insert(0,str(BASE/'pipeline'))
import ingest_scp_cccc as scp
from ingest_scp import MARKUP
from collect_pool import prose_spans,BINS,digest,apportion,LENGTH_WEIGHTS,atomic_json
from expand_pool import connect,add

def main():
 OUT.mkdir(exist_ok=False);db=connect(OUT/'collection.sqlite3');old=sqlite3.connect('file:'+str(BASE/'collection.sqlite3')+'?mode=ro',uri=True);old.execute('BEGIN');old.execute('SELECT count(*) FROM passages').fetchone();old.backup(db);old.close()
 baseline={x[0] for x in db.execute('SELECT id FROM passages')};index=json.loads((BASE/'tales-index.json').read_text());targets=apportion(4444,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['creative'])});stats=Counter()
 def selector(text,doc,remaining,capacity=3):
  prior=[json.loads(x[0]) for x in db.execute('SELECT row FROM passages WHERE source=? AND doc=?',('scp',doc))]
  if len(prior)>=3:return []
  if prior and any(x['raw_text_sha256']!=digest(current_text[0]) for x in prior):return []
  used=[(x['raw_start'],x['raw_end']) for x in prior];chosen=[];left=list(remaining)
  spans=[(a,b,w) for a,b,w in prose_spans(text) if not MARKUP.search(text[a:b])]
  for k in sorted(range(len(spans)),key=lambda i:digest(doc+':recovery:'+str(i))):
   if len(chosen)>=3-len(prior):break
   a,b,w=spans[k]
   for binid in sorted(range(4),key=lambda i:(-left[i],i)):
    if left[binid]<=0:continue
    lo,hi=BINS[binid];end=b;j=k;words=w
    while words<lo and j+1<len(spans):
     na,nb,nw=spans[j+1]
     if na-end>8 or len(text[a:nb].split())>hi:break
     end=nb;words=len(text[a:end].split());j+=1
    if lo<=words<=hi and not any(a<y and end>x for x,y in used):
     chosen.append((a,end,words,binid));used.append((a,end));left[binid]-=1;break
  return chosen
 scp.select_spans=selector;current_text=[''];scp.VERSION='scp-unused-contiguous-spans-v2'
 for f in sorted((BASE/'scan').glob('*.tales.jsonl.gz')):
  for line in gzip.open(f,'rt'):
   wrapper=json.loads(line)
   try:scp.eligible(wrapper,index)
   except scp.Excluded:continue
   current_text[0]=wrapper['record']['text'];bins=dict(db.execute('SELECT bin,count(*) FROM passages WHERE source="scp" GROUP BY bin'));remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
   candidates=list(scp.pairs(wrapper,index,remaining))
   with db:
    for row,raw in candidates:
     if add(db,row,raw,4444):stats['added']+=1
 package=OUT/'accepted-pairs-new.jsonl.gz';count=0
 with gzip.open(package,'wt') as stream:
  for id,saved in db.execute('SELECT id,row FROM passages'):
   if id in baseline:continue
   row=json.loads(saved);raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]));text=raw['record']['text'];assert digest(text)==row['raw_text_sha256'] and text[row['raw_start']:row['raw_end']]==row['text'] and digest(row['text'])==row['passage_sha256'];stream.write(json.dumps({'row':row,'raw':raw})+'\n');count+=1
 for doc,n in db.execute('SELECT doc,count(*) FROM passages GROUP BY doc'):assert n<=3
 for doc, in db.execute('SELECT DISTINCT doc FROM passages'):
  rows=[json.loads(x[0]) for x in db.execute('SELECT row FROM passages WHERE doc=?',(doc,))];ranges=sorted((x['raw_start'],x['raw_end']) for x in rows);assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
 receipt={'new_candidates':count,'baseline':len(baseline),'total':len(baseline)+count,'bins':dict(db.execute('SELECT bin,count(*) FROM passages GROUP BY bin')),'package':str(package),'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'prior_rows_preserved':True,'same_raw_original_only':True,'max_parent_passages':3,'nonoverlap_verified':True,'state':'bounded_unused_cached_spans_processed'};atomic_json(OUT/'package-manifest.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
